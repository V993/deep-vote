
import argparse
import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import tqdm

import evaluate_rule as ev
from data import generate_data_wrapper

# param_filename = "utils/parameters/7c29v_pref_testing.json"
param_filename = "utils/parameters/local_job_parameters.json"

# All distributions to generate data from:
dists_to_test = [
    "preflib",
    "impartial_culture",
    "impartial_anonymous_culture",
    "urn",
    "mallows",
    "sp_walsh",
]
dist_dataset_files = {
    "impartial_anonymous_culture": "data/pregenerated/11c_44v_10000p_impartial_anonymous_culture.h5",
    "impartial_culture": "data/pregenerated/11c_44v_10000p_impartial_culture.h5",
    "mallows": "data/pregenerated/11c_44v_10000p_mallows.h5",
    "preflib": "data/pregenerated/11c_44v_10000p_preflib.h5",
    "sp_walsh": "data/pregenerated/11c_44v_10000p_sp_walsh.h5",
    "urn": "data/pregenerated/11c_44v_10000p_urn.h5",
}


def update_test_dist_in_json(
    test_dist,
    param_filename="utils/parameters/7c29v_pref_testing.json",
    dataset_files=None,
):
    """Update the pref_dist entry in a job parameter JSON file."""
    try:
        # Load the JSON file
        with open(param_filename, 'r') as f:
            params = json.load(f)
        
        # Update the test_dist value
        params['pref_dist'] = test_dist
        if dataset_files and test_dist in dataset_files:
            params["data_filename"] = dataset_files[test_dist]
            dataset_count = get_dataset_profile_count(dataset_files[test_dist])
            params["dataset_profile_count"] = dataset_count
            params["n_profiles"] = dataset_count
        
        # Save the file back
        with open(param_filename, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✓ Updated pref_dist to {test_dist} in {param_filename}")
        if dataset_files and test_dist in dataset_files:
            print(
                f"✓ Updated data_filename to {dataset_files[test_dist]} "
                f"(profiles={params['dataset_profile_count']})"
            )
        
    except FileNotFoundError:
        print(f"❌ Error: {param_filename} file not found")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {param_filename}")
    except Exception as e:
        print(f"❌ Error: {e}")


def get_dataset_profile_count(dataset_path):
    """Return the number of profiles in an HDF5 dataset file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with h5py.File(dataset_path, "r") as handle:
        keys = list(handle.keys())
        if not keys:
            raise ValueError(f"No datasets found in {dataset_path}")
        return len(handle[keys[0]])


def generate_data_for_dists(dists, param_filename, skip_generation=True):
    """Generate datasets for each distribution unless skipping is requested."""
    if skip_generation:
        print("Skipping data generation.")
        return
    for test_dist in dists:
        update_test_dist_in_json(
            test_dist=test_dist,
            param_filename=param_filename,
            dataset_files=dist_dataset_files,
        )
        generate_data_wrapper(param_filename)


def evaluate_distributions(dists, rule_names, transform_names, param_filename, out_folder):
    """Evaluate models for each distribution and persist per-dist CSV results."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(out_folder, exist_ok=True)

    for test_dist in tqdm.tqdm(dists):
        results_rows = []
        print(f"Testing on {test_dist}")

        update_test_dist_in_json(
            test_dist=test_dist,
            param_filename=param_filename,
            dataset_files=dist_dataset_files,
        )

        for rule_name in rule_names:
            row = {"rule_names": rule_name}
            for transform_name in transform_names:
                params = ev.load_job_parameters(rule_name, transform_name, param_filename)
                if "train_dist" not in params:
                    params["train_dist"] = "impartial_culture"
                if "test_dist" not in params:
                    params["test_dist"] = "impartial_culture"

                if bool(params.get("concatenate")) is True:
                    params["transform"] = None
                    params["transform_name"] = "concatenate"
                    print("Concatenating!")

                if f"{params['transform_name']}-loss" in row:
                    continue

                print(
                    "Received params:\n"
                    f"{params['rule_name']=}, {params['transform_name']=}, {param_filename=}"
                )

                # Load data generators from hdf5
                try:
                    load_profile = not (
                        "pref_dist" in params and params["pref_dist"] == "preflib"
                    )
                    _train_generator, _valid_generator, test_generator = (
                        ev.load_pregenerated_data(
                            params, proportions=(0, 0), load_profile=load_profile
                        )
                    )
                except Exception as e:
                    print(f"Failed to load data generators with error: {e}")
                    print(f"\tValidate that '{ev.get_dataset_name(params)}' exists\n")
                    raise

                # Evaluate model
                start = time.perf_counter()
                eval_result = ev.eval_model(
                    params["rule_name"],
                    params["transform_name"],
                    test_generator,
                    device,
                    params=params,
                )
                end = time.perf_counter()

                (
                    validation_loss,
                    _single_batch_losses,
                    winner_batch_accuracies,
                    batch_tie_rates,
                ) = eval_result

                print(validation_loss)
                row[f"{params['transform_name']}-loss"] = validation_loss
                row[f"{params['transform_name']}-winner_acc"] = (
                    np.mean(winner_batch_accuracies)
                )
                row[f"{params['transform_name']}-tie_rate"] = (
                    np.mean(batch_tie_rates)
                )

                duration_minutes = f"{((end - start) / 60):0.4f}"
                print(f"Finished model evaluation, took {duration_minutes} minutes.")

                if params["transform_name"] == "concatenate":
                    break

            results_rows.append(row)

        df = pd.DataFrame(results_rows)
        cols = [
            "rule_names",
            # "rankfreq-loss",
            # "tournament-loss",
            # "weighted_tournament-loss",
            # "rankfreq-winner_acc",
            # "tournament-winner_acc",
            # "weighted_tournament-winner_acc",
            # "rankfreq-tie_rate",
            # "tournament-tie_rate",
            # "weighted_tournament-tie_rate",
        ]
        extra_cols = [col for col in df.columns if col not in cols]
        df = df.reindex(columns=cols + extra_cols)

        df.to_csv(
            f"{out_folder}/losses_train=impartial_culture-test={test_dist}-concatenate.csv"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate preference-distribution model performance."
    )
    parser.add_argument(
        "--param-file",
        default=param_filename,
        help="Path to the job parameter JSON file.",
    )
    parser.add_argument(
        "--out-folder",
        default="pref_dist_testing/11c44v",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--dists",
        nargs="+",
        default=dists_to_test,
        help="Preference distributions to evaluate.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate data before evaluation (skipped by default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generate_data_for_dists(
        args.dists, args.param_file, skip_generation=not args.generate_data
    )

    rule_names = [
        "black_pscf",
        "borda_pscf",
        # "condorcet_pscf",
        "copeland_pscf",
        "irv_pscf",
        "maximin_pscf",
        "plurality_pscf",
        "rankedpairs_pscf",
        "schulze_pscf",
    ]
    transform_names = [
        # "rankfreq",
        # "tournament",
        # "weighted_tournament",
        "concatenate"
    ]

    evaluate_distributions(
        args.dists, rule_names, transform_names, args.param_file, args.out_folder
    )
    print(f"Saved results to {args.out_folder}")


if __name__ == "__main__":
    main()
