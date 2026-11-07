
import sys
import time
import json
import torch
import pandas as pd
from sys import argv
from tqdm import trange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rules import borda_pscf
from transforms import tournament
import numpy as np

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
    get_concatenated_data,
    load_model,
    CONCAT_TRANSFORM_ORDER
)


def usage():
    print("Usage: python evaluate_rule.py <rule_name> <transform_name> <params_path> ")
    print("  <rule_name>     : Name of the rule")
    print("  <transform_name>: Name of the transformation")
    print("  <params_path>: Path to .json with job parameters")


def test_one_epoch(test_generator, rule_name, transform_name, device, model, loss_fn, params):
    """ Perform one full pass of the test set

    Args:
        valid_generator (DataGenerator): batching is not required
        device (str): device on which to perform computations
    
    Returns:
        int: Average batch loss 
    """
    num_batches, running_vloss = 0, 0.

    # Track batch losses separate from epoch-wide loss for batch loss tracking:
    batch_losses = []
    batch_winner_accs = []
    batch_tie_rates = []

    with torch.set_grad_enabled(False):
        for idx, (data, labels) in enumerate(test_generator):
            
            # Unpack data from dataloader
            if params.get("concatenate", False) == True:
                X = get_concatenated_data(data, transforms=CONCAT_TRANSFORM_ORDER)
            else:
                X = data[transform_name]
            Y = labels[rule_name]
            
            # Transfer to GPU
            X, Y = X.float().to(device), Y.float().to(device)

            # Get model predictions and loss
            logits = model(X)
            vloss = loss_fn(logits, Y)

            # Add running loss for epoch-wide loss tracking:
            running_vloss += vloss.item()
            num_batches += 1

            # Track how often single winner is correct; only counting when there is a clear predicted winner
            winner_pred = [i.numpy().argmax() for i in logits]
            winner_conf = [i.numpy().max() for i in logits]
            winner_real = [i.numpy().argmax() for i in Y]

            single_winner_errors = 0
            single_winner_ties = 0
            for idx in range(len(Y)):
                # if winner_conf[idx] > 0.9:
                #     if winner_pred[idx] != winner_real[idx]:
                #         single_winner_errors += 1
                # else:
                #     single_winner_ties += 1
                if winner_pred[idx] != winner_real[idx]:
                    single_winner_errors += 1
            batch_winner_accs.append(1 - single_winner_errors / len(Y)) # value of 1 means no errors; 0 all errors
            batch_tie_rates.append(single_winner_ties/len(Y))

            # Append batch loss to our list of batch losses for in-epoch loss tracking
            batch_losses.append(vloss.item())
    
    # Return the average loss over the entire epoch (average of all batches)
    return (running_vloss / num_batches), batch_losses, batch_winner_accs, batch_tie_rates

def eval_model(rule_name, transform_name, test_generator, device, params=None):
    """ Tests a given model over the given data and reports the accuracy.

    Returns:
        (list, list, nn.Module): training losses, validation losses, fitted model
    """
    
    print(f"Using device: {device}")
    print(f"Learning rule: '{rule_name}' with transform: '{transform_name}'")
    
    if params is None:
        raise NotImplementedError("Missing parameters, no defaults set yet")

    # model = create_model(params)
    model = load_model(params, device)
    model.to(device)

    # n_epochs = params["n_epochs"]
    loss_fn = get_loss_fn(params["loss_fn"])
    # save_interval = params.get("save_interval", False)

    eval_result = test_one_epoch(test_generator, rule_name, transform_name, device, model, loss_fn, params)

    return eval_result


def main():
    """Begin evaluation of a single trained network.

    Args:
        [1] rule_name (str)
        [2] transform_name (str)
        [3] path to parameter file (str)
    """

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: '{device}'")

    if len(argv) < 4:
        usage()
        exit(1)

    rule_name, transform_name, param_path = argv[1:4]

    params = load_job_parameters(rule_name, transform_name, param_path)

    if bool(params.get("concatenate")) == True:
        params["transform"] = None
        params["transform_name"] = "concatenate"
        print("Concatenating!")

    print(f"Recieved params: \n{params['rule_name']=}, {params['transform_name']=}, {param_path=}")
    
    # Load data generators from hdf5
    try:
        load_profile = not ("pref_dist" in params and params["pref_dist"] == "preflib")
        train_generator, valid_generator, test_generator = load_pregenerated_data(params, proportions=(0, 0), load_profile=load_profile)
    except Exception as e:
        print(f"Failed to load data generators with error: {e}")
        print(f"\tValidate that '{get_dataset_name(params)}' exists\n")
        exit(1)

    # Train model: 
    start = time.perf_counter()
    eval_result = eval_model(
        params['rule_name'], params['transform_name'], 
        test_generator, device, params=params
    )
    end = time.perf_counter()

    validation_loss, single_batch_losses, winner_batch_accuracies, batch_tie_rates = eval_result

    # print(validation_loss)
    print(f"Validation Loss: {validation_loss}")
    print(f"Single Batch Loss: {np.mean(single_batch_losses)}")
    print(f"Winner Batch Accuracy: {np.mean(winner_batch_accuracies)}")
    print(f"Batch Tie Rate: {np.mean(batch_tie_rates)}")

    duration_minutes = f"{((end - start) / 60):0.4f}"
    print(f"Finished model evaluation, took {duration_minutes} minutes.")


if __name__ == "__main__":
    main()




