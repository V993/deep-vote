import sys
import time
import json
import torch
import pandas as pd
from sys import argv
from pathlib import Path
import csv
import time

from data import (
    load_pregenerated_data,
    get_dataset_name
)
from utils import (
    get_loss_fn,
    load_job_parameters,
    save_experiment,
)
from models import (
    train_model_axioms,
    train_model,
    create_model,
    validate_one_epoch, validate_one_epoch_no_labels
)
from rules import WhalrusModule


def usage():
    print("Usage: python learn_rule.py <rule_name> <transform_name> <params_path> ")
    print("  <rule_name>     : Name of the rule")
    print("  <transform_name>: Name of the transformation")
    print("  <params_path>: Path to .json with job parameters")


def eval_dispatcher(rule_name, transform_name, valid_generator, device, params):
    """Training handler that dispatches to the correct training function based params
    """

    print(f"[dispatch] Using device: {device}")
    print(f"[dispatch] Evaluating rule: '{rule_name}' with transform: '{transform_name}'")

    if params is None:
        raise NotImplementedError("Missing parameters, no defaults set yet")

    model = WhalrusModule(rule=params["rule"])
    loss_fn = get_loss_fn(params["loss_fn"])

    # # Set model location and train mode:
    model.eval()
    instance_records = []

    if params.get("loss_fn") not in get_loss_fn():
        # Loss function is an axiom
        print(f"[dispatch] Evaluating an axiom. Ensure that 'axiom_loss_weight' is set to 0; it is currently: {params['axiom_loss_weight']}")
        loss, rule_loss, metrics = validate_one_epoch_no_labels(valid_generator, rule_name, transform_name, device, model, loss_fn,
                                                params, instance_records=instance_records)
        return model, {"axiom_loss": loss, "rule_loss": rule_loss, **metrics}, instance_records

    print(f"[dispatch] Evaluating a whalrus rule on itself. This is not really the point of the function :)")
    loss, batch_losses, metrics = validate_one_epoch(valid_generator, rule_name, transform_name, device, model, loss_fn,
                                                     params, instance_records=instance_records)
    return model, {"rule_loss": loss, "batch_losses": batch_losses, **metrics}, instance_records


# learning driver:
def main():
    """Begin eval loop for a single batch job

    Args:
        [1] rule_name (str)
        [2] transform_name (str)
        [3] path to parameter file (str)
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[setup] Running on device: '{device}'")


    if len(argv) < 4:
        usage()
        print("Proceeding with hardcoded default parameters.")

        # for some debugging
        rule_name, transform_name, param_path = "schulze_pscf", "concatenate", "utils/parameters/11c44v_120kp_evaluate_whalrus.json"
    else:
        rule_name, transform_name, param_path = argv[1:4]


    params = load_job_parameters(rule_name, transform_name, param_path)
    if "pref_dist" not in params:
        # This could also be added to the list of required parameters
        # Not required in order to maintain backwards compatibility
        params["pref_dist"] = "impartial_culture"

    if bool(params.get("concatenate")) == True:
        params["transform"] = None
        params["transform_name"] = "concatenate"
        print("[setup] Received 'concatenate'=True, will concatenate transforms at runtime")

    print(f"[setup] Received params: \n{params['rule_name']=}, {params['transform_name']=}, {param_path=}")

    # Set data split proportions; all data is used for validation
    # Not the typical proportion usage; using (0, 1) breaks as non-empty 'training' data is required
    proportions = (0.99, 0.01)
    # if params.get("loss_fn") not in get_loss_fn():
    #     print("[setup] Training axiom loss; using all data for training, no validation set.")
    #     proportions = (1.0, 0.0)

    # Load data generators from hdf5
    try:
        load_additional_data = params.get("load_additional_data", False)
        valid_generator, _, _ = load_pregenerated_data(params, proportions=proportions,
                                                                                  load_profile=load_additional_data)
    except Exception as e:
        print(f"[setup] Failed to load data generators with error: {e}")
        print(f"\tValidate that '{get_dataset_name(params)}' exists\n")
        exit(1)

    # Train model:
    start = time.perf_counter()
    model, eval_results, instance_records = eval_dispatcher(
        rule_name,
        transform_name,
        # train_generator,
        valid_generator,
        device,
        params
    )
    end = time.perf_counter()

    duration_minutes = f"{((end - start) / 60):0.4f}"
    print()
    print(f"!! --- Finished model training, took {duration_minutes} minutes --- !!")

    # Save per-instance outputs/metrics
    run_id = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path("experiments/final_validations") / "whalrus" / params["loss_fn"]
    base_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"whalrus-{params['loss_fn']}-{params['rule_name']}-{params['transform_name']}"
    instance_csv = base_dir / f"{job_name}_{run_id}_instances.csv"
    results_json = base_dir / f"{job_name}_{run_id}_metrics.json"
    results_payload = {k: v for k, v in eval_results.items() if not k.startswith("batch_")}
    with results_json.open("w") as handle:
        json.dump(results_payload, handle, indent=2)
    print(f"[save] Results JSON: {results_json}")
    print(json.dumps(results_payload, indent=2))
    if instance_records:
        fieldnames = sorted({k for row in instance_records for k in row.keys()})
        with instance_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(instance_records)
        print(f"[save] Instance-level metrics: {instance_csv}")

    # Skip save_experiment during evaluations to avoid writing model/loss artifacts.


if __name__ == "__main__":
    main()
