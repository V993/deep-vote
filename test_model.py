"""
Lightweight runner to evaluate a saved model (rule network or axiom learner)
against the pregenerated dataset with alternatives.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import csv
import time

from data import load_pregenerated_data
from models import validate_one_epoch, validate_one_epoch_no_labels
from utils import get_loss_fn

# Default dataset used for all evaluations
DEFAULT_DATA_PATH = "data/pregenerated/with_alternatives/n_profiles=10000-num_voters=44-m=11-committee_size=1-pref_dist=IC-has_tied_profiles-TRAIN.h5"


def _infer_rule_and_transform(model_path):
    """Infer rule/transform from a filename like 'plurality_pscf_concatenate.pt'."""
    stem = model_path.stem
    known_transforms = (
        "weighted_tournament",
        "weighted-tournament",
        "rankfreq",
        "tournament",
        "concatenate",
    )
    for transform in known_transforms:
        suffix = f"_{transform}"
        if stem.endswith(suffix):
            rule_name = stem[: -len(suffix)]
            if transform == "weighted-tournament":
                transform = "weighted_tournament"
            return rule_name, transform
    stem_parts = stem.split("_")
    if len(stem_parts) < 2:
        return None, None
    return "_".join(stem_parts[:-1]), stem_parts[-1]


def _load_params(params_path):
    """Load evaluation parameters from a JSON file."""
    with params_path.open() as handle:
        return json.load(handle)


def _prepare_params(model_path, params_path, data_path, overrides=None):
    """
    Merge metadata with runtime overrides and fill the fields needed for dataloading/evaluation.
    """
    model_path = Path(model_path)
    if not params_path:
        raise ValueError("params_path is required for evaluation.")
    params = dict(_load_params(Path(params_path)))
    overrides = overrides or {}

    for key, value in overrides.items():
        if value is None:
            continue
        params[key] = value

    if params.get("concatenate"):
        params["transform_name"] = "concatenate"
    if overrides.get("transform_name") is not None:
        params["transform_name"] = overrides["transform_name"]

    if data_path is not None:
        params["data_filename"] = data_path
    params.setdefault("data_filename", DEFAULT_DATA_PATH)
    params.setdefault("pref_dist", "impartial_culture")
    params.setdefault("n_profiles", params.get("dataset_profile_count"))
    params.setdefault("dataset_profile_count", params.get("n_profiles"))
    params.setdefault("batch_size", 32)
    params["shuffle"] = False
    params.setdefault("num_workers", 0)
    params.setdefault("concatenate", False)
    params.setdefault("load_additional_data", True)

    inferred_rule, inferred_transform = _infer_rule_and_transform(model_path)
    params.setdefault("rule_name", inferred_rule)
    params.setdefault("transform_name", inferred_transform)
    if params.get("concatenate"):
        params["transform_name"] = "concatenate"

    if not params.get("rule_name") or not params.get("transform_name"):
        raise ValueError("rule_name and transform_name are required in params or must be inferrable from the model name.")
    if not params.get("loss_fn"):
        raise ValueError("loss_fn must be provided in params to pick the evaluation flow.")
    if not params.get("n_profiles"):
        raise ValueError("n_profiles (or dataset_profile_count) is required to size the dataloaders.")

    base_losses = get_loss_fn()
    if params["loss_fn"] not in base_losses:
        params.setdefault("extra_data_datasets", [])
        for key in ["tournament", "weighted_tournament-normalized", "rank_matrix-normalized"]:
            if key not in params["extra_data_datasets"]:
                params["extra_data_datasets"].append(key)

    return params


def evaluate_saved_model(model_path, params_path, data_path=None, device=None, instance_csv=None, overrides=None):
    """
    Load a saved model and compute evaluation losses on the pregenerated dataset.

    Returns a dict with axiom/rule losses depending on the configured loss_fn.
    """
    params = _prepare_params(model_path, params_path, data_path, overrides=overrides)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    loss_fn = get_loss_fn(params["loss_fn"])
    base_losses = get_loss_fn()

    # Use all available records as a test set
    train_loader, valid_loader, test_loader = load_pregenerated_data(
        params,
        proportions=(0.0, 0.0),
        load_profile=params.get("load_additional_data", True),
    )
    eval_loader = test_loader or valid_loader or train_loader
    if eval_loader is None:
        raise RuntimeError("No evaluation data was loaded from the pregenerated file.")

    instance_records = []

    if params["loss_fn"] not in base_losses:
        axiom_loss, rule_loss, metrics = validate_one_epoch_no_labels(
            eval_loader,
            params["rule_name"],
            params["transform_name"],
            device,
            model,
            loss_fn,
            params,
            instance_records=instance_records
        )
        results = {"axiom_loss": axiom_loss, "rule_loss": rule_loss, **metrics}
    else:
        avg_loss, batch_losses, metrics = validate_one_epoch(
            eval_loader,
            params["rule_name"],
            params["transform_name"],
            device,
            model,
            loss_fn,
            params,
            instance_records=instance_records
        )
        results = {"loss": avg_loss, "batch_losses": batch_losses, **metrics}

    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(instance_csv) if instance_csv is not None else Path("experiments/final_validations")
    model_parts = Path(model_path).parts
    if "finetuned_models" in model_parts:
        group = "finetuned"
    elif "pretrained_models" in model_parts:
        group = "pretrained"
    else:
        group = params.get("batch_label", "pretrained")
    target_dir = base_dir / group / params["loss_fn"]
    target_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"{group}-{params['loss_fn']}-{params['rule_name']}-{params['transform_name']}"
    results_path = target_dir / f"{job_name}_{run_id}_metrics.json"
    results_json = {k: v for k, v in results.items() if not k.startswith("batch_")}
    with results_path.open("w") as handle:
        json.dump(results_json, handle, indent=2)
    print(f"[save] Results JSON: {results_path}")
    print(json.dumps(results_json, indent=2))

    if instance_records:
        instance_path = target_dir / f"{job_name}_{run_id}_instances.csv"
        fieldnames = sorted({k for row in instance_records for k in row.keys()})
        with instance_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance_records)
        print(f"[save] Instance-level metrics: {instance_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on pregenerated election data.")
    parser.add_argument("model_path", help="Path to the saved model (.pt).")
    parser.add_argument(
        "--params",
        help="Path to a JSON params file (e.g., utils/parameters/...).",
        required=False,
    )
    parser.add_argument(
        "--metadata",
        help="Deprecated: use --params instead.",
        required=False,
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Override the pregenerated data path used for evaluation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device (e.g., cpu, cuda:0). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--instances-csv",
        default=None,
        help="Optional path to save per-instance outputs/metrics as CSV.",
    )
    parser.add_argument(
        "--rule-name",
        default=None,
        help="Override rule_name in params for evaluation.",
    )
    parser.add_argument(
        "--transform-name",
        default=None,
        help="Override transform_name in params for evaluation.",
    )
    parser.add_argument(
        "--loss-fn",
        default=None,
        help="Override loss_fn in params for evaluation.",
    )
    parser.add_argument(
        "--axiom-loss-weight",
        type=float,
        default=None,
        help="Override axiom_loss_weight in params for evaluation.",
    )
    parser.add_argument(
        "--batch-label",
        default=None,
        help="Override batch_label in params for output grouping.",
    )
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="Force concatenate transform for evaluation.",
    )
    args = parser.parse_args()

    overrides = {
        "rule_name": args.rule_name,
        "transform_name": args.transform_name,
        "loss_fn": args.loss_fn,
        "axiom_loss_weight": args.axiom_loss_weight,
        "batch_label": args.batch_label,
        "concatenate": True if args.concatenate else None,
    }

    results = evaluate_saved_model(
        model_path=args.model_path,
        params_path=args.params or args.metadata,
        data_path=args.data,
        device=args.device,
        instance_csv=args.instances_csv,
        overrides=overrides,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
