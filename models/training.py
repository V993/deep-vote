import os
import copy
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from models import (
    create_model
)
from data import (
    load_pregenerated_data,
    get_dataset_name
)
from utils import (
    get_loss_fn,
    load_job_parameters,
    save_experiment,
    get_concatenated_data,
    CONCAT_TRANSFORM_ORDER
)

AXIOMATIC_THRESHOLD = 1e-5  # Approximate considering the 11c44v dist. We never achieve "0" loss, just tiny numbers which are "nonzero"
TIE_DISTRIBUTION_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
TIE_DISTRIBUTION_EPS = 0.0

def _resolve_transform_name(transform_name, data):
    if transform_name == "rankfreq" and "rank_matrix-normalized" in data:
        return "rank_matrix-normalized"
    if transform_name == "weighted_tournament" and "weighted_tournament-normalized" in data:
        return "weighted_tournament-normalized"
    return transform_name

# ---------------------------------------
# ----- Rule training (with labels) -----
# ---------------------------------------


def train_model(model, rule_name, transform_name, train_generator, valid_generator, device, params=None):
    """ Fits a model using provided training data over a set number of epochs

    Returns:
        (list, list, nn.Module): training losses, validation losses, fitted model
    """

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=params.get("lr_optimizer", 0.001),
        # weight_decay=params.get("weight_decay", 0.01)
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=params.get("patience", 100),
        min_lr=params.get("min_lr", 1e-6),
        factor=params.get("factor", 0.1),
        threshold=params.get("threshold", 1e-4),
    )

    training_losses = []
    valid_losses = []

    training_batch_losses = []
    valid_batch_losses = []

    n_epochs = params["n_epochs"]
    loss_fn = get_loss_fn(params["loss_fn"])
    save_interval = params.get("save_interval", False)

    # Warmup configuration
    warmup_epochs = params.get("warmup_epochs", 5)
    base_lr = params.get("lr_optimizer", 0.001)

    best_model_state = None
    best_valid_loss = float("inf")

    # Stall detection and intervention configuration
    stall_check_interval = params.get("stall_check_interval", 10)  # Check every N epochs
    stall_threshold = params.get("stall_threshold", 0.01)  # Relative improvement needed
    max_restarts = params.get("max_restarts", 3)  # Maximum number of restarts
    restart_counter = 0
    initial_model_state = copy.deepcopy(model.state_dict())  # Save initial weights

    # Learning rate restarts: once the LR has sat at min_lr for lr_restart_patience_after_min_lr epochs without the
    # validation loss improving (relative to the best loss since the last restart) by more than lr_restart_min_change,
    # reset the LR to its initial value and start a fresh scheduler. Adam's state is kept. Set the patience to 0 or
    # null to disable.
    lr_restart_patience = params.get("lr_restart_patience_after_min_lr", 100)
    lr_restart_min_change = params.get("lr_restart_min_change", 0.0005)
    scheduler_min_lr = params.get("min_lr", 1e-6)
    lr_restart_epochs = []
    cycle_best_valid_loss = None
    epochs_at_min_lr_without_improvement = 0

    # Track initial loss for stall detection
    initial_loss = None
    last_check_loss = None

    # Initialize trange:
    t = trange(n_epochs, desc="Losses: Train: 1.0000 | Valid: 1.0000", unit="epoch")
    grad_norm_history = []

    for epoch_id in t:

        # Learning rate warmup
        if epoch_id < warmup_epochs:
            lr = base_lr * (epoch_id + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Perform full pass of training data
        avg_train_loss, train_single_batch_losses, grad_norms = train_one_epoch(train_generator, rule_name,
                                                                                transform_name, device, optimizer,
                                                                                model, loss_fn, epoch_id, params)

        # Validate over full pass of validation data
        avg_valid_loss, valid_single_batch_losses, _ = validate_one_epoch(valid_generator, rule_name, transform_name,
                                                                          device, model, loss_fn, params)

        # Step the scheduler to reduce LR as needed
        scheduler.step(avg_valid_loss)

        # Append running losses:
        training_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        grad_norm_history.append(grad_norms)

        # Concatenate batch losses for loss tracking:
        training_batch_losses += train_single_batch_losses
        valid_batch_losses += valid_single_batch_losses

        # Initialize baseline loss for stall detection
        if initial_loss is None:
            initial_loss = avg_train_loss
            last_check_loss = avg_train_loss

        # Check for stalled learning after warmup
        if epoch_id > 0 and epoch_id % stall_check_interval == 0 and epoch_id >= warmup_epochs:
            relative_improvement = (last_check_loss - avg_train_loss) / last_check_loss
            print("CHECKING RELATIVE IMPROVEMENT")
            print(f"   Last check loss: {last_check_loss:.4f}, Current loss: {avg_train_loss:.4f}")
            print(f"   Loss improvement: {relative_improvement:.4f}. Required threshold: {stall_threshold}")

            if relative_improvement < stall_threshold:
                print(f"\n⚠️  Learning stalled detected at epoch {epoch_id}", file=sys.stderr)

                if restart_counter < max_restarts:
                    restart_counter += 1
                    intervention_type = params.get("intervention_type", "perturb")  # "restart" or "perturb"

                    if intervention_type == "restart":
                        print(f"   🔄 Performing restart #{restart_counter}/{max_restarts}")
                        # Reinitialize model weights
                        model.load_state_dict(initial_model_state)
                        model.apply(lambda m: reinitialize_weights(m))

                    elif intervention_type == "perturb":
                        print(f"   🎲 Performing weight perturbation #{restart_counter}/{max_restarts}")
                        # Add noise to current weights
                        perturb_weights(model, scale=params.get("perturb_scale", 0.1))

                    elif intervention_type == "lr_boost":
                        print(f"   ⚡ Boosting learning rate #{restart_counter}/{max_restarts}")
                        # Temporarily increase learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= params.get("lr_boost_factor", 10.0)

                    # Reset optimizer state (momentum, adaptive learning rates, etc.)
                    optimizer = torch.optim.Adam(
                        params=model.parameters(),
                        lr=base_lr,
                        # weight_decay=params.get("weight_decay", 0.01)
                    )

                    scheduler = ReduceLROnPlateau(
                        optimizer=optimizer,
                        mode='min',
                        patience=params.get("patience", 100),
                        min_lr=params.get("min_lr", 1e-6),
                        factor=params.get("factor", 0.1),
                        threshold=params.get("threshold", 1e-4),
                    )

                    # Reset tracking
                    last_check_loss = avg_train_loss
                    # best_valid_loss = float('inf')
                    # patience_counter = 0

                else:
                    print(f"   ❌ Max restarts ({max_restarts}) reached. Continuing without intervention.")
            else:
                last_check_loss = avg_train_loss

        # Track best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # Learning rate restart once training has stalled at min_lr
        if lr_restart_patience:
            improved = (cycle_best_valid_loss is None or
                        avg_valid_loss < cycle_best_valid_loss * (1 - lr_restart_min_change))
            if improved:
                cycle_best_valid_loss = avg_valid_loss

            at_min_lr = optimizer.param_groups[0]['lr'] <= scheduler_min_lr * (1 + 1e-6)
            if at_min_lr and not improved:
                epochs_at_min_lr_without_improvement += 1
            else:
                epochs_at_min_lr_without_improvement = 0

            if epochs_at_min_lr_without_improvement >= lr_restart_patience:
                print(f"\n🔁 LR restart at epoch {epoch_id}: {lr_restart_patience} epochs at min_lr without improvement. "
                      f"Resetting LR to {base_lr} (Adam state kept).")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode='min',
                    patience=params.get("patience", 100),
                    min_lr=params.get("min_lr", 1e-6),
                    factor=params.get("factor", 0.1),
                    threshold=params.get("threshold", 1e-4),
                )
                lr_restart_epochs.append(epoch_id)
                params["lr_restart_epochs"] = lr_restart_epochs  # recorded in metadata / model.info
                cycle_best_valid_loss = None
                epochs_at_min_lr_without_improvement = 0

        # Set running description dynamically
        current_lr = optimizer.param_groups[0]['lr']
        t.set_description("Epoch {} Losses: Train: {:.5f} | Valid: {:.5f} | LR: {:.5f}".format(epoch_id, avg_train_loss,
                                                                                               avg_valid_loss,
                                                                                               current_lr))

        print(f"Best Validation Loss: {best_valid_loss}")

        if (save_interval) and (epoch_id % save_interval == 0) and (epoch_id >= save_interval):
            # Save the best weights so far, then restore the live weights so saving never affects training.
            live_model_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_state)
            save_experiment(model, training_losses, params, train_batches=training_batch_losses,
                            valid_losses=valid_losses, valid_batches=valid_batch_losses)
            model.load_state_dict(live_model_state)

    if len(grad_norm_history) > 0:
        print("Grad Norms per Epoch")
        print("Avg; Max")
        for gn in grad_norm_history:
            print(f"{round(gn[0], 3)}; {round(gn[1], 3)}")

    # Load model which had lowest validation loss
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, training_losses, valid_losses, training_batch_losses, valid_batch_losses


def reinitialize_weights(module):
    """Reinitialize weights for a given module"""
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


def perturb_weights(model, scale=0.1):
    """Add random noise to model weights to escape local minima"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * scale * param.std()
            param.add_(noise)


def validate_one_epoch(valid_generator, rule_name, transform_name, device, model, loss_fn, params, instance_records=None):
    """ Perform one full pass of the validation set

    Args:
        valid_generator (DataGenerator): batching is not required
        device (str): device on which to perform computations
    
    Returns:
        float: Average batch loss
        list: Batch losses
        dict: Rule evaluation metrics
    """
    num_batches, running_vloss = 0, 0.

    # Track batch losses separate from epoch-wide loss for batch loss tracking:
    batch_losses = []
    instance_idx = len(instance_records) if instance_records is not None else 0
    tie_thresholds = params.get("tie_distribution_thresholds", TIE_DISTRIBUTION_THRESHOLDS)
    tie_violation_counts = {thr: 0 for thr in tie_thresholds}
    tie_total = 0
    single_winner_total = 0
    single_winner_correct = 0
    single_winner_violations = 0
    rule_correct_total = 0
    rule_pred_total = 0
    any_winner_correct_total = 0
    winner_sum_correct_total = 0
    track_tie_metrics = instance_records is not None or params.get("track_rule_tie_metrics", False)

    # model.eval()
    with torch.set_grad_enabled(False):
        for data, labels in valid_generator:

            # Unpack data from dataloader
            if params.get("use_whalrus_rule_module", False):
                # Used when the model is simply a whalrus rule
                X_train = data["Profile"]
            elif params.get("concatenate", False) == True:
                X_train = get_concatenated_data(data, transforms=CONCAT_TRANSFORM_ORDER)
            else:
                X_train = data[_resolve_transform_name(transform_name, data)]
            Y_train = labels[rule_name]

            # Transfer to GPU
            X_train, Y_train = X_train.float().to(device), Y_train.float().to(device)

            # Get model predictions and loss
            logits = model(X_train)
            vloss = loss_fn(logits, Y_train)

            # Add running loss for epoch-wide loss tracking:
            running_vloss += vloss.item()
            num_batches += 1

            # Append batch loss to our list of batch losses for in-epoch loss tracking
            batch_losses.append(vloss.item())

            preds, targets, batch_correct, batch_total = _compute_rule_argmax_batch(logits, Y_train)
            rule_correct_total += batch_correct
            rule_pred_total += batch_total
            any_winner_correct_total += int(_compute_rule_any_winner_batch(Y_train, preds).sum())
            winner_sum_correct_total += int(_compute_rule_winner_sum_batch(logits, Y_train).sum())
            if track_tie_metrics or instance_records is not None:
                tie_mass, is_tied, tie_violations = _compute_rule_tie_mass_batch(
                    logits, Y_train, tie_thresholds, preds, targets
                )
                tie_total += int(is_tied.sum())
                for threshold in tie_thresholds:
                    tie_violation_counts[threshold] += int(tie_violations[threshold].sum())

            single_winner_flags, batch_single_winner_total, _, batch_single_winner_violations = _compute_rule_single_winner_batch(
                data.get("is_single_winner_across_rules"),
                preds=preds,
                targets=targets
            )
            single_winner_total += batch_single_winner_total
            single_winner_violations += batch_single_winner_violations
            single_winner_correct += (batch_single_winner_total - batch_single_winner_violations)

            if instance_records is not None:
                rule_l1 = F.l1_loss(logits, Y_train, reduction="none").mean(dim=1).detach().cpu().numpy()
                for i in range(len(preds)):
                    record = {
                        "instance_id": instance_idx,
                        "pred_argmax": int(preds[i]),
                        "true_argmax": int(targets[i]),
                        "correct": bool(preds[i] == targets[i]),
                        "rule_l1_loss": float(rule_l1[i]),
                        "logits": logits[i].detach().cpu().numpy().tolist(),
                        "label": Y_train[i].detach().cpu().numpy().tolist(),
                    }
                    if single_winner_flags is not None:
                        record["is_single_winner_across_rules"] = bool(single_winner_flags[i])
                    if track_tie_metrics:
                        record["tied_winner_mass"] = float(tie_mass[i]) if is_tied[i] else None
                        for threshold in tie_thresholds:
                            key = _format_threshold_key(threshold)
                            record[f"tie_mass_violation_{key}"] = bool(tie_violations[threshold][i])
                    instance_records.append(record)
                    instance_idx += 1

    # Return the average loss over the entire epoch (average of all batches)
    metrics = {}
    violation_summary = {
        "argmax": {
            "correct": rule_correct_total,
            "incorrect": max(rule_pred_total - rule_correct_total, 0),
            "total": rule_pred_total,
            "accuracy": float(rule_correct_total / rule_pred_total) if rule_pred_total else 0.0,
        },
        # Correct when the predicted argmax is any of the (possibly tied) winners; threshold-free.
        # Correct when the summed probability on the winners exceeds every individual non-winner output.
        "winner_sum": {
            "correct": winner_sum_correct_total,
            "total": rule_pred_total,
            "accuracy": float(winner_sum_correct_total / rule_pred_total) if rule_pred_total else 0.0,
        },
        "argmax_any_winner": {
            "correct": any_winner_correct_total,
            "total": rule_pred_total,
            "accuracy": float(any_winner_correct_total / rule_pred_total) if rule_pred_total else 0.0,
        },
        "tie_mass": {
            "total_tied_instances": tie_total if track_tie_metrics else None,
            "thresholds": None,
        },
        "single_winner_across_rules": {
            "total": single_winner_total if single_winner_total else None,
            "correct": single_winner_correct if single_winner_total else None,
            "violations": single_winner_violations if single_winner_total else None,
            "accuracy": float(single_winner_correct / single_winner_total) if single_winner_total else None,
        },
    }
    if track_tie_metrics:
        violation_summary["tie_mass"]["thresholds"] = {
            threshold: {
                "count": tie_violation_counts[threshold],
                "rate": float(tie_violation_counts[threshold] / rule_pred_total) if rule_pred_total else 0.0,
            }
            for threshold in tie_thresholds
        }
    metrics["rule_violations"] = violation_summary

    return (running_vloss / num_batches), batch_losses, metrics


def _compute_rule_correct_flags(pred_probs: np.ndarray, target_probs: np.ndarray):
    """Return per-instance correctness by comparing argmax of preds vs labels."""
    pred_idx = pred_probs.argmax(axis=1)
    true_idx = target_probs.argmax(axis=1)

    # print(f"{pred_probs=}")
    # print(f"{pred_idx=}")
    # print(f"{true_idx=}")

    return list(pred_idx == true_idx)


def _format_threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _compute_tie_distribution_batch(pred_scores: np.ndarray, target_probs: np.ndarray, thresholds):
    # Targets are exact, so ties are defined by equality with the max value.
    max_vals = target_probs.max(axis=1, keepdims=True)
    tie_mask = target_probs == max_vals

    # All profiles in the batch that are ties according to the target: 
    is_tied = tie_mask.sum(axis=1) > 1

    # Get all of the mass on the tied candidates:
    tie_mass = np.where(is_tied, (pred_scores * tie_mask).sum(axis=1), np.nan)

    # Compute violation count across all profiles in the batch for each specified threshold (violation if under threshold)
    violations = {thr: (tie_mass < thr) & is_tied for thr in thresholds}
    return tie_mass, is_tied, violations


def _compute_rule_argmax_batch(logits: torch.Tensor, labels: torch.Tensor):
    # Argmax-based correctness uses the predicted max index vs target max index.
    preds = logits.detach().cpu().numpy().argmax(axis=1)
    targets = labels.detach().cpu().numpy().argmax(axis=1)
    correct = int((preds == targets).sum())
    total = int(len(targets))
    return preds, targets, correct, total


def _compute_rule_any_winner_batch(labels: torch.Tensor, preds):
    # Correct iff the predicted argmax lands on a candidate whose target value equals the max (any tied winner).
    target_probs = labels.detach().cpu().numpy()
    return target_probs[np.arange(len(preds)), preds] == target_probs.max(axis=1)


def _compute_rule_winner_sum_batch(logits: torch.Tensor, labels: torch.Tensor):
    # Correct iff the total predicted mass on the (possibly tied) winners is strictly larger than the largest
    # single non-winner output. With one winner this is plain argmax; if every candidate wins it is always correct.
    pred_scores = logits.detach().cpu().numpy()
    target_probs = labels.detach().cpu().numpy()
    winner_mask = target_probs == target_probs.max(axis=1, keepdims=True)
    winner_mass = (pred_scores * winner_mask).sum(axis=1)
    best_loser = np.where(winner_mask, -np.inf, pred_scores).max(axis=1)
    return winner_mass > best_loser


def _compute_rule_tie_mass_batch(logits: torch.Tensor, labels: torch.Tensor, thresholds, preds, targets):
    # Tie-mass uses raw model scores over tied target indices; non-ties fall back to argmax correctness.
    pred_scores = logits.detach().cpu().numpy()
    target_probs = labels.detach().cpu().numpy()
    tie_mass, is_tied, violations = _compute_tie_distribution_batch(pred_scores, target_probs, thresholds)
    for thr in thresholds:
        violations[thr] = np.where(is_tied, violations[thr], preds != targets)
    return tie_mass, is_tied, violations


def _compute_rule_single_winner_batch(single_winner_flags, preds=None, targets=None):
    # Single-winner accuracy only evaluates rows flagged as single-winner instances.
    if single_winner_flags is None:
        return None, 0, 0, 0
    if isinstance(single_winner_flags, torch.Tensor):
        single_winner_flags = single_winner_flags.detach().cpu().numpy()
    mask = single_winner_flags.astype(bool)
    total = int(np.sum(mask))
    if preds is None or targets is None:
        return single_winner_flags, total, total, 0
    violations = int(np.sum((preds != targets) & mask))
    return single_winner_flags, total, total, violations


def _compute_rule_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Return accuracy and raw counts for a batch of rule predictions using argmax comparison.
    """
    pred_probs = logits.detach().cpu().numpy()
    target_probs = labels.detach().cpu().numpy()
    correct_flags = _compute_rule_correct_flags(pred_probs, target_probs)
    correct = int(sum(correct_flags))
    total = int(len(target_probs))
    acc = float(correct / total) if total else 0.0
    return acc, correct, total, correct_flags


def _compute_axiom_violation_stats(violation_instances: int, total_instances: int):
    """Return violation rate and counts for axioms at instance granularity."""
    if total_instances == 0:
        return 0.0, 0, 0
    rate = violation_instances / total_instances
    non_violations = max(total_instances - violation_instances, 0)
    return rate, violation_instances, non_violations


def validate_one_epoch_no_labels(valid_generator, rule_name, transform_name, device, model, loss_fn, params, instance_records=None):
    """Full pass over validation data for axiom objectives, mirroring train_one_epoch_no_labels logic.

    Returns:
        avg_axiom_loss (float)
        avg_rule_loss (float)
        metrics (dict): violation/accuracy details
    """
    num_batches = 0

    # Track the rule/axiom losses:
    rule_losses, axiom_losses = [], []
    rule_accuracies = []
    rule_correct_total = 0
    rule_pred_total = 0
    axiom_violation_instances = 0
    axiom_instance_total = 0
    instance_idx = len(instance_records) if instance_records is not None else 0
    tie_thresholds = params.get("tie_distribution_thresholds", TIE_DISTRIBUTION_THRESHOLDS)
    tie_violation_counts = {thr: 0 for thr in tie_thresholds}
    tie_total = 0
    single_winner_total = 0
    single_winner_correct = 0
    single_winner_violations = 0
    track_tie_metrics = instance_records is not None or params.get("track_rule_tie_metrics", False)

    # Instantiate the L1 loss metric for rule validation while we train if we are doing transfer learning:
    L1Loss = torch.nn.L1Loss(reduction='mean')

    axiom_loss_weight = params.get("axiom_loss_weight", 0.5)

    with torch.set_grad_enabled(False):
        for data, labels in valid_generator:

            if data.get("profitable_abstentions") is not None or data.get("topk_swaps") is not None:
                transform_labels = ['tournament', 'weighted_tournament-normalized', 'rank_matrix-normalized']
            else:
                raise NotImplementedError(
                    "We now require 'profitable_abstensions' or 'topk_swaps' for axiomatic learning, please check your data.")

            # Unpack data from dataloader (this is only used for rule training)
            if params.get("use_whalrus_rule_module", False):
                # Used when the model is simply a whalrus rule
                X_train = data["Profile"]
            elif params.get("concatenate", False) == True:
                X_train = get_concatenated_data(data, transforms=transform_labels)
            else:
                X_train = data[_resolve_transform_name(transform_name, data)]

            # Transfer to GPU
            X_train = X_train.float().to(device)

            # Take the axiom loss over the entire batch:
            axiom_loss = loss_fn(data, model, params.get("transform_name"), device)
            batch_size = len(data.get("Profile", []))

            # Get the rule loss and accuracy:
            if len(labels) > 0 and axiom_loss_weight < 1 and rule_name in labels:
                logits = model(X_train)  # Use the correct data
                Y_train = labels[rule_name]
                Y_train = Y_train.float().to(device)

                rule_loss = L1Loss(logits, Y_train)
                rule_losses.append(rule_loss.item())
                acc, correct, total, correct_flags = _compute_rule_accuracy(logits, Y_train)
                rule_accuracies.append(acc)
                rule_correct_total += correct
                rule_pred_total += total
                if track_tie_metrics:
                    preds, targets, _, _ = _compute_rule_argmax_batch(logits, Y_train)
                    tie_mass, is_tied, tie_violations = _compute_rule_tie_mass_batch(
                        logits, Y_train, tie_thresholds, preds, targets
                    )
                    tie_total += int(is_tied.sum())
                    for threshold in tie_thresholds:
                        tie_violation_counts[threshold] += int(tie_violations[threshold].sum())
            else:
                rule_losses.append(0)
                correct_flags = [None] * batch_size
                logits = None
                Y_train = None
                tie_mass = [None] * batch_size
                is_tied = [False] * batch_size
                tie_violations = {thr: [False] * batch_size for thr in tie_thresholds}

            axiom_losses.append(axiom_loss.item())
            # Compute per-instance axiom violations by evaluating each profile independently
            per_instance_axiom_losses = []
            for i in range(batch_size):
                single_data = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        single_data[k] = v[i].unsqueeze(0)
                    elif isinstance(v, list):
                        single_data[k] = [v[i]]
                    elif isinstance(v, dict):
                        # Dictionaries are indexed per batch already
                        single_data[k] = v[i]
                    else:
                        single_data[k] = v
                try:
                    inst_loss = loss_fn(single_data, model, params.get("transform_name"), device)
                    inst_loss_val = float(inst_loss.item())
                except Exception:
                    inst_loss_val = float(axiom_loss.item())  # fallback to batch loss
                per_instance_axiom_losses.append(inst_loss_val)
                if inst_loss_val > AXIOMATIC_THRESHOLD:
                    axiom_violation_instances += 1
            axiom_instance_total += batch_size
            num_batches += 1

            if instance_records is not None and batch_size > 0:
                axiom_violation_flag = axiom_loss.item() > 0
                if len(labels) > 0 and axiom_loss_weight < 1 and rule_name in labels:
                    preds, targets, _, _ = _compute_rule_argmax_batch(logits, Y_train)
                    rule_l1 = F.l1_loss(logits, Y_train, reduction="none").mean(dim=1).detach().cpu().numpy()
                else:
                    preds = [None] * batch_size
                    targets = [None] * batch_size
                    rule_l1 = [None] * batch_size
                    correct_flags = [None] * batch_size
                    tie_mass = [None] * batch_size
                    is_tied = [False] * batch_size
                    tie_violations = {thr: [False] * batch_size for thr in tie_thresholds}

                if len(labels) > 0 and axiom_loss_weight < 1 and rule_name in labels:
                    preds, targets, _, _ = _compute_rule_argmax_batch(logits, Y_train)
                else:
                    preds = [None] * batch_size
                    targets = [None] * batch_size
                single_winner_flags, batch_single_winner_total, _, batch_single_winner_violations = _compute_rule_single_winner_batch(
                    data.get("is_single_winner_across_rules"),
                    preds=preds,
                    targets=targets
                )
                single_winner_total += batch_single_winner_total
                single_winner_violations += batch_single_winner_violations
                single_winner_correct += (batch_single_winner_total - batch_single_winner_violations)

                for i in range(batch_size):
                    record = {
                        "instance_id": instance_idx,
                        "axiom_loss": per_instance_axiom_losses[i] if i < len(per_instance_axiom_losses) else float(axiom_loss.item()),
                        "axiom_violation": per_instance_axiom_losses[i] > 0 if i < len(per_instance_axiom_losses) else axiom_violation_flag,
                        "profile": data.get("Profile")[i].detach().cpu().numpy().tolist() if data.get("Profile") is not None else None,
                    }
                    if preds[i] is not None:
                        record.update({
                            "pred_argmax": int(preds[i]),
                            "true_argmax": int(targets[i]),
                            "correct": correct_flags[i],
                            "rule_l1_loss": float(rule_l1[i]) if rule_l1[i] is not None else None,
                            "logits": logits[i].detach().cpu().numpy().tolist(),
                            "label": Y_train[i].detach().cpu().numpy().tolist(),
                        })
                        if track_tie_metrics:
                            record["tied_winner_mass"] = float(tie_mass[i]) if is_tied[i] else None
                            for threshold in tie_thresholds:
                                key = _format_threshold_key(threshold)
                                record[f"tie_mass_violation_{key}"] = bool(tie_violations[threshold][i])
                    if single_winner_flags is not None:
                        record["is_single_winner_across_rules"] = bool(single_winner_flags[i])
                    instance_records.append(record)
                    instance_idx += 1

    # Aggregate metrics:
    avg_rule_loss = np.mean(rule_losses) if len(rule_losses) > 0 else 0.0
    avg_axiom_loss = np.mean(axiom_losses) if len(axiom_losses) > 0 else 0.0

    violation_rate, violation_count, non_violation_count = _compute_axiom_violation_stats(
        axiom_violation_instances, axiom_instance_total
    )

    metrics = {
        "axiom_violation_rate": violation_rate,
        "axiom_violations": violation_count,
        "axiom_nonviolations": non_violation_count,
        "axiom_accuracy": 1 - violation_rate,
        "batch_axiom_losses": axiom_losses,
        "batch_rule_losses": rule_losses,
        "num_batches": num_batches
    }

    violation_summary = {
        "argmax": {
            "correct": rule_correct_total,
            "incorrect": max(rule_pred_total - rule_correct_total, 0),
            "total": rule_pred_total,
            "accuracy": float(np.mean(rule_accuracies)) if len(rule_accuracies) > 0 else 0.0,
        },
        "tie_mass": {
            "total_tied_instances": tie_total if track_tie_metrics else None,
            "thresholds": None,
        },
        "single_winner_across_rules": {
            "total": single_winner_total if single_winner_total else None,
            "correct": single_winner_correct if single_winner_total else None,
            "violations": single_winner_violations if single_winner_total else None,
            "accuracy": float(single_winner_correct / single_winner_total) if single_winner_total else None,
        },
    }
    if track_tie_metrics:
        violation_summary["tie_mass"]["thresholds"] = {
            threshold: {
                "count": tie_violation_counts[threshold],
                "rate": float(tie_violation_counts[threshold] / rule_pred_total) if rule_pred_total else 0.0,
            }
            for threshold in tie_thresholds
        }
    metrics["rule_violations"] = violation_summary

    return avg_axiom_loss, avg_rule_loss, metrics


def train_one_epoch(train_generator, rule_name, transform_name, device, optimizer, model, loss_fn, epoch_id, params):
    """ Perform one full pass of the training set

    Args:
        train_generator (DataGenerator): batching is not required
        device (str): device on which to perform computations
        epoch_id (int): epoch number for logging

    Returns:
        float: Average batch loss
        list: Individual batch losses
    """

    num_batches, running_loss = 0, 0.
    batch_losses = []

    # Gradient clipping parameter
    max_grad_norm = params.get("max_grad_norm", 1.0)

    # For monitoring gradient health
    grad_norms = []

    model.train()  # Set to training mode

    for data, labels in train_generator:

        # Unpack data from dataloader
        if params.get("use_whalrus_rule_module", False):
            # Used when the model is simply a whalrus rule
            X_train = data["Profile"]
        elif params.get("concatenate", False) == True:
            X_train = get_concatenated_data(data, transforms=CONCAT_TRANSFORM_ORDER)
        else:
            X_train = data[_resolve_transform_name(transform_name, data)]
        Y_train = labels[rule_name]

        # Transfer to GPU
        X_train, Y_train = X_train.float().to(device), Y_train.float().to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make model predictions
        logits = model(X_train)

        # Compute loss
        loss = loss_fn(logits, Y_train)
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(grad_norm.item())

        # Update weights
        optimizer.step()

        # Add running loss for epoch-wide loss tracking:
        running_loss += loss.item()
        num_batches += 1
        batch_losses.append(loss.item())

    # After a full epoch, output diagnostics
    # print(f"\nEpoch {epoch_id} diagnostics:")
    # print(f"Sample logits vs true label:")
    logits_np = logits[0].detach().cpu().numpy()
    y_true_np = Y_train[0].detach().cpu().numpy()
    logits_str = [f"{x:.4f}" if abs(x) >= 0.0001 else "0.0000" for x in logits_np]
    y_true_str = [f"{x:.4f}" if abs(x) >= 0.0001 else "0.0000" for x in y_true_np]
    # print(f"Pred: [{', '.join(logits_str)}]")
    # print(f"True: [{', '.join(y_true_str)}]")

    grad_norm_history = -1, -1
    if grad_norms:
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm_val = max(grad_norms)
        # print(f"Gradient norms - Avg: {avg_grad_norm:.4f}, Max: {max_grad_norm_val:.4f}")
        grad_norm_history = avg_grad_norm, max_grad_norm_val

    return (running_loss / num_batches), batch_losses, grad_norm_history


# --------------------------------------
# ----- Axiom training (no labels) -----
# --------------------------------------


def train_model_axioms(model, rule_name, transform_name, train_generator, device, params=None):
    """ Fits a model using provided training data over a set number of epochs

    Returns:
        (list, list, nn.Module): training losses, validation losses, fitted model
    """

    # For CUDA, spawn instead of fork for resource allocation: TODO!! uncomment this! It doesn't play nice with jupyternotebooks
    # multiprocessing.set_start_method('spawn')

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=params.get("lr_optimizer", 0.001)
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        patience=params.get("patience", 100),
        min_lr=params.get("min_lr", 1e-6),
        factor=params.get("factor", 0.1),
        threshold=params.get("threshold", 1e-4),
    )

    training_losses = []
    rule_losses = []
    training_batch_losses = []

    n_epochs = params["n_epochs"]
    loss_fn = get_loss_fn(params["loss_fn"])
    if isinstance(loss_fn, dict):
        raise NotImplementedError(f"No valid loss function found in get_loss_fn() for '{params['loss_fn']}'")
    save_interval = params.get("save_interval", False)

    # Initialize trange:
    t = trange(n_epochs, desc="Losses: Axiom: 1.0000 | Rule: 1.0000", unit="epoch")

    for epoch_id in t:

        # Perform full pass of data
        # This returns the average rule and axiom loss separately, we can aggregate or backprop however we'd like:
        av_axiom_loss, avg_rule_loss = train_one_epoch_no_labels(train_generator, transform_name, rule_name, device,
                                                                 optimizer, model, loss_fn, epoch_id, params)

        # Step the scheduler to reduce LR as needed
        scheduler.step(av_axiom_loss)

        # Append running losses and rule losses:
        training_losses.append(av_axiom_loss)
        rule_losses.append(avg_rule_loss)

        # Set running description dynamically
        epoch_log = "Epoch {} Losses: Axiom: {:.4f} | Rule: {:.4f}".format(epoch_id, av_axiom_loss, avg_rule_loss)
        t.set_description(epoch_log)
        # print(epoch_log)

        if (save_interval) and (epoch_id % save_interval == 0) and (epoch_id >= save_interval):
            save_experiment(model, training_losses, params, valid_losses=rule_losses,
                            train_batches=training_batch_losses)

    return model, training_losses, rule_losses, training_batch_losses


def train_one_epoch_no_labels(train_generator, transform_name, rule_name, device, optimizer, model, loss_fn, epoch_id,
                              params):
    """ Perform one full pass of the training set

    Args:
        train_generator (DataGenerator): batching is not required
        device (str): device on which to perform computations
        epoch_id (int): epoch number for logging
    
    Returns:
        int: Average batch loss 
    """

    # if device == "cpu":
    #     print("WARNING! USING CPU! ARE YOU SURE?")

    num_batches = 0

    # Track the rule/axiom losses:
    rule_losses, axiom_losses = [], []

    # Instantiate the L1 loss metric for rule validation while we train if we are doing transfer learning:
    L1Loss = torch.nn.L1Loss(reduction='mean')

    axiom_loss_weight = params.get("axiom_loss_weight", 0.5)

    # Set to training mode
    model.train()

    # Each data element is a dictionary of shape data_key -> [list of len(batch_size) data objects]
    # If we want to pull the profiles in the batch, we'd simply use data["profiles"] which would return the list of all profiles for the batch.
    # Similarly, alternatives can be retrieved with data["topk_swaps"] or data["profitable_abstentions"]
    # These alternatives are dictionaries, and should be callable as such... but I'm not sure
    for data, labels in train_generator:

        if data.get("profitable_abstentions") is not None or data.get("topk_swaps") is not None:
            transform_labels = ['tournament', 'weighted_tournament-normalized', 'rank_matrix-normalized']
            # transform_labels = ["tournament", "weighted_tournament", "rankfreq"]
        else:
            raise NotImplementedError(
                "We now require 'profitable_abstensions' or 'topk_swaps' for axiomatic learning, please check your data.")

        # Unpack data from dataloader (this is only used for rule training)
        if params.get("concatenate", False) == True:
            X_train = get_concatenated_data(data, transforms=transform_labels)
        else:
            X_train = data["Profile"]  # axioms require further proessing of the profile as a tensor

        # Transfer to GPU
        X_train = X_train.float().to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Take the axiom loss over the entire batch:
        axiom_loss = loss_fn(data, model, params.get("transform_name"), device)

        # Get the rule loss:
        # Produce model logits
        if len(labels) > 0 and axiom_loss_weight < 1 and rule_name in labels:
            logits = model(X_train)  # Use the correct data
            Y_train = labels[rule_name]
            Y_train = Y_train.float().to(device)

            # Compute rule loss
            rule_loss = L1Loss(logits, Y_train)
            rule_losses.append(rule_loss.item())
        else:
            rule_loss = 0
            rule_losses.append(0)

        # Update weights from both loss functions:
        combined_loss = axiom_loss*axiom_loss_weight + rule_loss*(1-axiom_loss_weight)
        combined_loss.backward()

        # Update weights
        optimizer.step()

        axiom_losses.append(axiom_loss.item())
        num_batches += 1

        # print('Batch: {} | Avg batch loss: {}'.format(num_batches, running_loss/num_batches))

    # Return the average rule and axiom loss over the entire epoch (average of all batches)
    avg_rule_loss = np.mean(rule_losses)
    avg_axiom_loss = np.mean(axiom_losses)

    return avg_axiom_loss, avg_rule_loss


def train_one_batch_no_labels(X_train, model, loss_fn, device):
    """DEPRECATED! Iterate over the batch and apply the loss_fn on each instance. 
    This computes the gradient updates in-place for each instance. 
    Notably, if you do not perform teh optimizer step, the gradients are
    not updated (https://discuss.pytorch.org/t/pytorch-gradients/884/2)

    Theoretically, this should not be much slower.

    Args:
        X_train (Tensor): Tensor containing a list of profiles
        loss_fn (func): loss function (specifcally one that does not take multiple instances as input)

    Returns:
        float: averaged loss over the batch
    """

    batch_loss = torch.tensor([.0], device=device)

    X_train = X_train["profile"]

    for instance in X_train:
        # Loss functions require atomic entities and should not be responsible for handling data:
        # TODO: we will have to pass the weighted_tournament for condorect here to avoid recomputing it needlessly
        profile = instance.float().to(device)
        loss = loss_fn(profile, model)
        batch_loss += loss

    return batch_loss / len(X_train)
