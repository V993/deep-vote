
import sys
import time
import json
import torch
import pandas as pd
from sys import argv

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
    create_model
)

def usage():
    print("Usage: python learn_rule.py <rule_name> <transform_name> <params_path> [replicate]")
    print("  <rule_name>     : Name of the rule")
    print("  <transform_name>: Name of the transformation")
    print("  <params_path>: Path to .json with job parameters")
    print("  [replicate]  : Optional replicate id; appended to the experiment directory as '_rep<id>'")


def training_dispatcher(rule_name, transform_name, train_generator, valid_generator, device, params):
    """Training handler that dispatches to the correct training function based params
    """

    print(f"[dispatch] Using device: {device}")
    print(f"[dispatch] Learning rule: '{rule_name}' with transform: '{transform_name}'")
    
    if params is None:
        raise NotImplementedError("Missing parameters, no defaults set yet")

    # Load a pretrained mode if provided by parameters:
    if params.get("models_directory") is not None:
        # Get the specific model from the model directory and update in the job params for future reference:
        params["models_directory"] = f"{params.get('models_directory')}/{rule_name}_{transform_name}.pt"
        
        print("[dispatch] Loading pretrained model from:", params["models_directory"])
        print("[dispatch] Model will be loaded on device:", device)
        model = torch.load(params["models_directory"], map_location=device, weights_only=False)
    else:
        print("[dispatch] Training model from scratch, no models_directory provided.")
        model = create_model(params)

    # Set model location and train mode:
    model.to(device)
    model.train()

    if params.get("loss_fn") not in get_loss_fn(): # If the loss function is not one of the default loss functions (it's probably an axiom!)
        print("[dispatch] Training axioms, no labels in data!\n")
        model, train_losses, valid_losses, training_batch_losses = train_model_axioms(
            model, rule_name, transform_name, train_generator, device, params=params # the "rule" here is actually an axiom
        )
        # valid_losses = []
        valid_batch_losses = []
    else:
        print("[dispatch] Training rule with labels from precomputed rule outcomes\n")
        model, train_losses, valid_losses, training_batch_losses, valid_batch_losses = train_model(
            model, rule_name, transform_name, train_generator, valid_generator, device, params
        )

    return model, train_losses, valid_losses, training_batch_losses, valid_batch_losses

# learning driver:
def main():
    """Begin training loop for a single batch job

    Args:
        [1] rule_name (str)
        [2] transform_name (str)
        [3] path to parameter file (str)
        [4] replicate id (int, optional) -- appended to the experiment directory so repeated runs of the
            same config do not overwrite each other. Omit for a single one-off run.
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[setup] Running on device: '{device}'")

    replicate = argv[4] if len(argv) > 4 else None

    if len(argv) < 4:
        usage()
        print("Proceeding with hardcoded default parameters.")

        # # for some debugging
        # # rule_name, transform_name, param_path = "plurality_pscf", "concatenate", "utils/parameters/11c44v_120kp_finetune_assp.json"
        # rule_name, transform_name, param_path = "maximin_pscf", "concatenate", "utils/parameters/11c44v_120kp_finetune_assp.json"
        rule_name, transform_name, param_path = "copeland_pscf", "concatenate", "utils/parameters/11c44v_120kp_finetune_participation_efficient.json"
        # rule_name, transform_name, param_path = "copeland_pscf", "concatenate", "utils/parameters/11c44v_120kp_learn_rules.json"
        # rule_name, transform_name, param_path = "maximin_pscf", "weighted_tournament", "utils/parameters/11c44v_120kp_learn_rules.json"
    else:
        rule_name, transform_name, param_path = argv[1:4]


    params = load_job_parameters(rule_name, transform_name, param_path)
    if replicate is not None:
        # Recorded in params so it lands in the saved model's .info and in metadata.json, and so
        # get_experiment_directory() gives this replicate its own output directory.
        params["replicate"] = replicate
        print(f"[setup] Replicate id: {replicate}")
    if "pref_dist" not in params:
        # This could also be added to the list of required parameters
        # Not required in order to maintain backwards compatibility
        params["pref_dist"] = "impartial_culture"

    if bool(params.get("concatenate")) == True:
        params["transform"] = None
        params["transform_name"] = "concatenate"
        print("[setup] Received 'concatenate'=True, will concatenate transforms at runtime")

    print(f"[setup] Received params: \n{params['rule_name']=}, {params['transform_name']=}, {param_path=}")

    # Set data split proportions
    proportions = (0.80, 0.10)  # default train/valid split
    if params.get("loss_fn") not in get_loss_fn():
        print("[setup] Training axiom loss; using all data for training, no validation set.")
        proportions = (1.0, 0.0)
    
    # Load data generators from hdf5
    try:
        load_additional_data = params.get("load_additional_data", False)
        train_generator, valid_generator, test_generator = load_pregenerated_data(params, proportions=proportions, load_profile=load_additional_data)
    except Exception as e:
        print(f"[setup] Failed to load data generators with error: {e}")
        print(f"\tValidate that '{get_dataset_name(params)}' exists\n")
        exit(1)
    
    # Train model: 
    start = time.perf_counter()
    model, train_losses, valid_losses, training_batch_losses, valid_batch_losses = training_dispatcher(
        rule_name, 
        transform_name, 
        train_generator, 
        valid_generator, 
        device, 
        params
    )
    end = time.perf_counter()

    duration_minutes = f"{((end - start) / 60):0.4f}"
    print()
    print(f"!! --- Finished model training, took {duration_minutes} minutes --- !!")

    pd.DataFrame({"train_loss":train_losses, "valid_loss":valid_losses}).plot()

    # Final model save: (do not save model, we will )
    save_experiment(
        model, 
        train_losses, 
        params, 
        train_batches=training_batch_losses, # Separate loss tracking for in-epoch batch loss (more granular)
        valid_batches=valid_batch_losses, # Separate loss tracking for in-epoch batch loss (more granular)
        duration=duration_minutes,
        valid_losses=valid_losses
    )


if __name__ == "__main__":
    main()

