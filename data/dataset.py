""" 
    Funcs for pregenerating generating data
"""

import os
import time
import h5py
import pandas as pd
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from p_tqdm import p_map
from itertools import repeat
from tqdm import trange, tqdm
from pprint import PrettyPrinter
import prefsampling.ordinal as ps
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from preflibtools.instances import OrdinalInstance
import random

printer = PrettyPrinter(indent=2, sort_dicts=False)

from transforms import (
    rankfreq,
    weighted_tournament,
    tournament,
    lex,
    # full
)

from rules import (
    black_pscf,
    borda_pscf,
    condorcet_pscf,
    copeland_pscf,
    irv_pscf,
    plurality_pscf,
    rankedpairs_pscf,
    schulze_pscf,
    maximin_pscf
)

transform_funcs = {
    "rankfreq": rankfreq,
    "weighted_tournament": weighted_tournament,
    "tournament": tournament,
    # "lex": lex,
    # "full": full,
    # "none": None, # obselete, full implements this well enough
}

rules_funcs = {
    "black_pscf": black_pscf,
    "borda_pscf": borda_pscf,
    # "condorcet_pscf": condorcet_pscf,
    "copeland_pscf": copeland_pscf,
    "irv_pscf": irv_pscf,
    "plurality_pscf": plurality_pscf,
    "rankedpairs_pscf": rankedpairs_pscf,
    "schulze_pscf": schulze_pscf,
    "maximin_pscf": maximin_pscf
}


# --------------------------
# ----- Generate data: -----
# --------------------------

def generate_profile(distribution: str, n_voters: int, n_cands: int, rng, **kwargs) -> np.ndarray:
    prof_seed = rng.integers(0, 1000000)
    if distribution == "impartial_culture" or distribution is None:
        # prof = rand_rankings_profile(n_voters=n_voters, n_cands=n_cands, rng=rng)
        prof = ps.impartial(num_voters=n_voters, num_candidates=n_cands, seed=prof_seed)
    elif distribution == "impartial_anonymous_culture" or distribution is None:
        prof = ps.impartial_anonymous(num_voters=n_voters, num_candidates=n_cands, seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "mallows":
        if "mallows-phi" in kwargs:
            phi = kwargs["mallows-phi"]
        else:
            phi = rng.uniform(0, 1)
        prof = ps.mallows(num_voters=n_voters,
                          num_candidates=n_cands,
                          phi=phi,
                          normalise_phi=True,
                          central_vote=None,
                          impartial_central_vote=True,  # central vote is sampled from IC. Better for anonymity.
                          seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "urn":
        if "urn-alpha" in kwargs:
            alpha = kwargs["urn-alpha"]
        else:
            alpha = rng.uniform(0, 1)
        prof = ps.urn(num_voters=n_voters,
                      num_candidates=n_cands,
                      alpha=alpha,
                      seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "sp_walsh":
        prof = ps.single_peaked_walsh(num_voters=n_voters,
                                      num_candidates=n_cands,
                                      seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "sp_conitzer":
        prof = ps.single_peaked_conitzer(num_voters=n_voters,
                                         num_candidates=n_cands,
                                         seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "sp_circle":
        prof = ps.single_peaked_circle(num_voters=n_voters,
                                       num_candidates=n_cands,
                                       seed=prof_seed)
        # prof = np.asarray(prof)
    elif distribution == "preflib":
        raise ValueError(f"Should not have reached this code! 'preflib' should branch out before this method.")
    else:
        raise ValueError(f"Unsupported preference distribution. Was given: {distribution}")

    if isinstance(prof, list):
        prof = np.asarray(prof)
    return prof


def load_preflib_profiles(n_cands, n_profiles, rng, strict_n_cands=None, min_num_voters=None, max_num_voters=None,
                          **params):
    """
    Load all preflib instances in the specified folder and return the instances with the matching number of alternatives.
    :param folder:
    :param all_num_alternatives:
    :param max_num_voters: If set, limits the number of voters to reduce time needed for axiom evaluation.
    :param n_profiles: Load all matching instances if None or load at most the value given (set lower for quick testing)
    :return:
    """

    print(f"Loading preflib profiles with {n_cands} candidates")

    # folder = Path(folder)
    if "preflib_dir" not in params:
        raise KeyError("Told to load preflib data but no preflib_dir key in params.")
    folder = Path(params["preflib_dir"])

    instances = []
    profiles = []
    for file_path in folder.glob("*.soc"):
        # if max_profile_count and len(instances) >= max_profile_count:
        #     break
        try:
            # Create new election instance for each file
            instance = OrdinalInstance()
            instance.parse_file(str(file_path))

            if max_num_voters is not None and max_num_voters < instance.num_voters:
                continue
            if min_num_voters is not None and min_num_voters > instance.num_voters:
                continue

            # Keep only the elections that have the right number of alternatives
            if strict_n_cands and instance.num_alternatives == n_cands:
                # instances.append(instance)
                # orders = instance.flatten_strict()
                # profile = []
                # for order, mult in orders:
                #     order = [o-min(order) for o in order]   # ensure alternative labels start at 0
                #     profile.extend([list(order)] * mult)
                # profiles.append(np.asarray(profile))
                profile = _sample_from_preflib_instance(instance, n_cands=n_cands, rng=rng)
                profiles.append(np.asarray(profile))
            elif instance.num_alternatives >= n_cands:
                # Keep all elections with enough candidates, downsampling those with too many
                # orders = instance.flatten_strict()
                # profile = []
                # for order, mult in orders:
                #     order = [o-min(order) for o in order]   # ensure alternative labels start at 0

                #     profile.extend([list(order)] * mult)
                # profiles.append(np.asarray(profile))
                profile = _sample_from_preflib_instance(instance, n_cands=n_cands, rng=rng)
                profiles.append(np.asarray(profile))

            if len(profiles) >= n_profiles:
                # unlikely that we will encounter more profiles than we want in most cases
                break

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return profiles


def _sample_from_preflib_instance(instance, n_cands, rng):
    orders = instance.flatten_strict()
    cands_to_keep = rng.choice(orders[0][0], size=n_cands, replace=False)
    # get mapping of candidate to their renamed selves
    unique_sorted = sorted(set(cands_to_keep))
    mapping = {val: i for i, val in enumerate(unique_sorted)}

    profile = []
    for order, mult in orders:
        order = [mapping[o] for o in order if o in cands_to_keep]
        profile.extend([list(order)] * mult)

    return profile


def rand_rankings_profile(n_voters: int, n_cands: int, rng) -> np.ndarray:
    permutations = np.zeros((n_voters, n_cands), dtype=int)
    for i in range(n_voters):
        permutations[i] = rng.permutation(n_cands)
    return permutations


def get_upper_trianglular(transformed_profile):
    """Helper to compute the upper triangular (not order sensitive) as a 1d tensor"""
    size = ((len(transformed_profile) * len(transformed_profile)) - len(transformed_profile)) // 2
    upper_t = torch.empty(size, dtype=float)

    t_index = 0
    for i in range(len(transformed_profile)):
        for j in range(0, i):
            upper_t[t_index] = transformed_profile[i, j]
            t_index += 1

    return upper_t


def compute_upper_triangulars(upper_triangle_eligible):
    result = {}
    for transform_name, transformed_profiles in upper_triangle_eligible.items():
        result.update({
            transform_name + "_upper_triangular": [
                get_upper_trianglular(transformed_profile)
                for transformed_profile in
                tqdm(transformed_profiles, desc=f"Computing upper triangular for {transform_name}")
            ]
        })
    return result


def apply_function(profiles, function_name, fn, **kwargs):
    """Apply a function over a list of profiles

    Args:
        profiles (list): list of profiles to compute over
        function_name (str): name of the function to be applied (arbitrary, for tracking)
        fn (callable): function to apply (transform or rule)

    Returns:
        tuple: (str, list) returns original function name for tracking and the list of function results
    """
    return function_name, [fn(profile) if fn != None else profile for profile in profiles]


def apply_padding(transformed_profile, orig_n_cands):
    """Obv does not work with lex transform, must be square matrix"""
    n_cands, _ = transformed_profile.shape
    padding = orig_n_cands - n_cands
    transformed_profile = np.pad(transformed_profile, [(0, padding), (0, padding)], mode="constant", constant_values=-1)
    return transformed_profile


def compute_over_profiles(profiles, functions, desc="", orig_n_cands=None, output_len=None):
    """Apply a list of functions over a profile in parallel with individual job monitoring using p_tqdm"""
    results = {}

    # Determine the number of processes as one less than the number of CPU cores
    num_processes = max(1, os.cpu_count() - 1)

    # p_map ensures ordering is maintained for cross-dataset indexing
    results_list = p_map(lambda func: apply_function(profiles, func[0], func[1]), functions, desc=desc,
                         num_cpus=num_processes)

    # Unpack results into dictionary
    results = {function_name: result for function_name, result in results_list}

    # If we are padding the transformed profiles, apply -1 padding to precomputed transforms:
    if orig_n_cands is not None:
        results = {function_name: list(map(apply_padding, result, repeat(orig_n_cands))) for function_name, result in
                   results.items()}

    # If we are padding the outputs, apply 0 padding to precomputed rule outcomes:
    if output_len is not None:
        results = {function_name: list(
            map(lambda x: np.pad(x, (0, output_len - len(x)), mode="constant", constant_values=0), result)) for
                   function_name, result in results.items()}

    return results


def get_dataset_name(params, dist=None):
    if params.get("data_filename"):  # Include override for specific data filename
        # if "data/pregenerated/" not in params.get("data_filename"):
        #     raise ValueError(
        #         "data_filename must be a full path to the pregenerated data file, starting with 'data/pregenerated/'")
        return params.get("data_filename")

    if dist is None:
        dist = ""
    else:
        dist = f"_{dist}"
        # underscore included here rather than below to maximize compatibilty with old files
    return f"data/pregenerated/{params.get('n_cands')}c_{params.get('n_voters')}v_{params.get('dataset_profile_count')}p{dist}.h5"


def generate_data(n_voters, n_cands, dataset_profile_count, rng, dist=None, **kwargs):
    """Generate profiles, transforms, and rules according to parameters

    Args:
        n_voters (int): number of voters
        n_cands (int): number of candidates
        n_profiles (int): number of profiles to generate
        rng (callable): random number generator for profile generation

    Returns:
        dict: dictionary with profiles, outcomes, transformed profiles, and upper triangular of tournaments
    """

    if dist is None:
        # For backwards compatibility -- if no dist is specified, then use the older IC method
        profiles = [rand_rankings_profile(n_voters, n_cands, rng) for _ in
                    trange(dataset_profile_count, desc="Generating profiles")]
    elif dist == "preflib":
        profiles = load_preflib_profiles(n_cands=n_cands, rng=rng, **kwargs)
        print(f"Loaded {len(profiles)} preflib profiles.")
    else:
        # profiles = [generate_profile(dist, n_voters, n_cands, rng, **kwargs) for _ in trange(dataset_profile_count, desc="Generating profiles")]

        print("Generating profiles with unique winners.")
        profiles = []
        while len(profiles) < dataset_profile_count:
            if len(profiles) % 100 == 0:
                print(f"# profiles is {len(profiles)}")
            profile = generate_profile(dist, n_voters, n_cands, rng, **kwargs)

            all_unique_winners = True
            for name, rf in rules_funcs.items():
                winner = rf(profile)
                if max(winner) < 1:
                    all_unique_winners = False
                    # print(f"{name} had multiple winners: {winner}")
            if all_unique_winners:
                profiles.append(profile)

    # Parallelizing outcomes and transformations
    orig_n_cands = 11 # Hardcoding for testing  
    outcomes = compute_over_profiles(profiles, rules_funcs.items(), "Applying rules", output_len=orig_n_cands)
    transformed_profiles = compute_over_profiles(profiles, transform_funcs.items(), "Applying transforms",
                                                 orig_n_cands=orig_n_cands)

    # Compute the upper triangle for tournament and weighted_tournament transforms:
    # upper_triangle_eligible = {k: v for k, v in transformed_profiles.items() if 'tournament' in k}
    # upper_triangle_profiles = compute_upper_triangulars(upper_triangle_eligible)

    # Aggregate the dataset:
    data = {"profile": profiles}
    data.update(outcomes)
    data.update(transformed_profiles)
    # data.update(upper_triangle_profiles)

    return data


def generate_data_wrapper(parameter_filename):
    """Generate data based off of provided parameters, return full pregenerated data and save to hdf5

    Args:
        parameter_filename (dict): parameters for data generation

    Returns:
        dict: pregenerated data in its entirety, with all transforms/rules as a dictionary of lists.
    """

    with open(parameter_filename, "r") as file:
        params = json.load(file)

    # Optional parameter to hardcode data generation name/path
    data_filename = params.get("data_filename")

    # Get preference distribution if specified, otherwise use IC
    pref_dist = params["pref_dist"] if "pref_dist" in params else None

    # Default data name:
    if not data_filename:
        data_filename = get_dataset_name(params, dist=pref_dist)

    print("Saving data to:", data_filename)

    profile_rng = np.random.default_rng(params['seed'])
    generated_data = generate_data(rng=profile_rng, dist=pref_dist, **params)

    first_record = {key: value[0] for key, value in generated_data.items()}
    print("Example record in the dataset (all rules/transforms):")
    printer.pprint(first_record)
    # printer.pprint(generated_data)

    # Save data to h5 format:
    # save_to_h5py_simple(generated_data, data_filename)    # NEED TO USE THIS FOR PREFLIB DATA! (profiles are various sizes)
    save_to_h5py(generated_data, data_filename, pref_dist=pref_dist)

    # Check that all of the datasets are present in the saved file:
    check_h5py(generated_data, data_filename)

    return generated_data

    # test_data = None
    # if train_dist != test_dist and test_dist != None:
    #     # Repeat the above, but with data corresponding to the test distribution

    #     profile_rng = np.random.default_rng(params['seed'])
    #     test_data = generate_data(rng=profile_rng, pref_dist=test_dist, **params)

    #     first_record = {key: value[0] for key, value in test_data.items()}
    #     print("Example record in the test dataset (all rules/transforms):")
    #     printer.pprint(first_record)
    #     # printer.pprint(pregenerated_data)

    #     # Save data to h5 format:
    #     save_to_h5py(test_data, test_filename)

    #     # Check that all of the datasets are present in the saved file:
    #     check_h5py(test_data, test_filename)

    # return train_data, test_data


# ---------------------------
# ----- Save/load data: -----
# ---------------------------

def tensor_to_np(tensor):
    # h5py does not allow for tensor storage
    return tensor.detach().cpu().numpy()


def save_to_h5py_simple(data, filename):
    """Save datasets to H5 format - simple approach that always uses groups
    Args:
        data (dict): dictionary of datasets where each value is a list of ndarrays
        filename (str): filename for H5 data
    """
    with h5py.File(filename, 'w') as f:
        for key, values in tqdm(data.items(), desc="Saving datasets to h5"):
            # Check if values is empty
            if not values:
                continue

            # Create a group for each key
            group = f.create_group(key)
            group.attrs['count'] = len(values)

            # Store each array individually
            for i, value in enumerate(values):
                # Handle tensors
                if isinstance(value, torch.Tensor):
                    np_val = tensor_to_np(value)
                else:
                    np_val = value

                # Create dataset for this array
                dataset_name = f'item_{i}'
                group.create_dataset(dataset_name, data=np_val)

        # Verification
        for key, original_values in data.items():
            if not original_values:
                continue

            group = f[key]
            count = group.attrs['count']
            for i in range(count):
                stored_value = group[f'item_{i}'][:]
                original_value = tensor_to_np(original_values[i]) if isinstance(original_values[i], torch.Tensor) else \
                original_values[i]
                np.testing.assert_array_equal(original_value, stored_value,
                                              err_msg=f"Mismatch in dataset '{key}' at index {i}")


def save_to_h5py(data, filename, pref_dist=None):
    """Save datasets to H5 format

    Args:
        data (dict): dictionary of datasets (profiles, transforms, rule outcomes)
        filename (str): filename for H5 data

    Notes:
        h5py does not support jagged arrays in a single dataset, so we must flatten profiles if they are different shapes.
        This specifically concerns preflib data, where profiles can have different shapes for single candidate sizes.
        In this case, we flatten the profiles and store the original shapes in a new dataset.
    """
    # Create a new HDF5 file:
    with h5py.File(filename, 'w') as f:
        for key, values in tqdm(data.items(), desc="Saving datasets to h5"):
            # Check type of values:
            is_tensor = isinstance(values[0], torch.Tensor)

            # Convert first element to get dtype
            first_element = tensor_to_np(values[0]) if is_tensor else values[0]
            dtype = first_element.dtype

            # Handle keys that require flattening due to jagged 2D shapes
            flattened_keys = {"profile", "lex"}
            if key in flattened_keys:
                flattened_values = [tensor_to_np(value.flatten()) if is_tensor else value.flatten() for value in values]
                vlen_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
                dataset = f.create_dataset(key, (len(values),), dtype=vlen_dtype)
                for i, profile in enumerate(flattened_values):
                    dataset[i] = profile

                # store the original shapes of each entry to be unpacked later
                shapes = [value.shape for value in values]
                f.create_dataset(f'{key}_shapes', data=np.array(shapes, dtype='int32'))

            else:
                # Check if all shapes match
                shapes = [tensor_to_np(v).shape if is_tensor else v.shape for v in values]
                uniform = all(s == shapes[0] for s in shapes)

                if not uniform:
                    raise ValueError(f"Inconsistent shapes found in key '{key}': {set(shapes)}")

                shape = (len(values),) + shapes[0]  # Use actual common shape
                dataset = f.create_dataset(key, shape, dtype=dtype)
                for i, value in enumerate(values):
                    dataset[i] = tensor_to_np(value) if is_tensor else value


def check_h5py(data, filename):
    """Validate that all datasets from original data are present in saved data"""
    with h5py.File(filename, 'r') as f:
        file_keys = set(f.keys())
        for key in data:
            if key in {"profile", "lex"}:
                expected_keys = {key, f"{key}_shapes"}
            else:
                expected_keys = {key}

            missing = expected_keys - file_keys
            assert not missing, f"Missing dataset(s): {missing}"

        print("DATA VALIDATED")


def load_h5py(filename, num_samples, replace=False):
    """Load sample of data from h5 dataset (for testing, better to use dataset class for training!!)

    Args:
        filename (str): filename of h5 data
        num_samples (int): number of randomly selected samples to pull from dataset
        replace (bool, optional): if True, sample with replacement. Defaults to False.

    Returns:
        dict: dictionary of datasets, same formatting as the return from load_all_data()
    """

    with h5py.File(filename, 'r') as f:
        # Assume that all datasets are of the same length
        total_records = len(f[list(f.keys())[0]])

        # Generate random indices
        indices = np.random.choice(total_records, size=num_samples, replace=replace)

        # Load the selected indices from each dataset
        data_subset = {}
        for name in f.keys():
            data_subset[name] = f[name][sorted(indices)]

        # Validate cross-dataset indexing
        dataset1 = data_subset[list(data_subset.keys())[0]]
        dataset2 = data_subset[list(data_subset.keys())[1]]
        for i in range(num_samples):
            assert np.all(dataset1[i] == dataset2[i]), "Mismatch found at index {}".format(i)

    return data_subset


def load_pregenerated_data(params, proportions=(0.80, 0.10), load_profile=True):
    """ Loads train, test, and valid dataloaders from a large dataset in hdf5 format

    Args:
        params (dict): full job params. Required: rule_name, transform_name, n_profiles, shuffle, num_workers, batch_size
        proportions (tuple, optional): Proportion of n_profiles for train and test/valid. Defaults to (0.80, 0.10).
        load_profile (bool, optional): Include the full profile when loading data. If profiles within a dataset have
        differing numbers of voters (as with Preflib) this should be set to False. Default is True.

    Returns:
        tuple: train, test, and valid dataloaders (or just train if proportions is (1.0, 0.0)).
        If both proportions are 0, train/valid loaders are None and stats are computed over all data.
    """

    rule_name = params.get("rule_name", None)
    transform_name = params.get("transform_name")
    train_proportion, valid_proportion = proportions

    label_datasets = [rule_name] if rule_name else list(rules_funcs.keys())

    if load_profile:
        data_datasets = ["Profile", "topk_swaps", "profitable_abstentions", "is_single_winner_across_rules"]
    else:
        data_datasets = []

    if params.get("concatenate", False) == True:
        if "committee_size" in params.get("data_filename"):
            data_datasets += ["tournament", "weighted_tournament-normalized", "rank_matrix-normalized"]
        else:
            data_datasets += ["tournament", "weighted_tournament", "rankfreq"]
    else:
        if "committee_size" in params.get("data_filename"):
            if transform_name == "rankfreq":
                data_datasets += ["rank_matrix-normalized"]
            elif transform_name == "weighted_tournament":
                data_datasets += ["weighted_tournament-normalized"]
            else:
                data_datasets += [transform_name]
        else:
            data_datasets += [transform_name]
    extra_datasets = params.get("extra_data_datasets", [])
    for key in extra_datasets:
        if key not in data_datasets:
            data_datasets.append(key)

    if "pref_dist" not in params:
        # This could also be added to the list of required parameters
        # Not required in order to maintain backwards compatibility
        params["pref_dist"] = "impartial_culture"
    dist_type = params["pref_dist"]

    dist = get_dataset_name(params, dist=dist_type)
    dataset_name = get_dataset_name(params, dist=dist_type)
    assert os.path.exists(dist)

    n_profiles_for_job = int(params.get("n_profiles"))
    indices = np.arange(n_profiles_for_job)

    train_end = int(train_proportion * n_profiles_for_job)
    valid_end = train_end + int(valid_proportion * n_profiles_for_job)

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    print(f"[setup] Loading data from {dataset_name}")
    stats_indices = train_indices if len(train_indices) > 0 else indices
    print(
        f"[setup] Loading stats dataset from {dataset_name} "
        f"(indices={len(stats_indices)})"
    )
    stats_dataset = LazyProfilesDataset(
        dataset_name,
        data_datasets=data_datasets,
        label_datasets=label_datasets,
        allowed_indices=stats_indices,
        allow_objects=True,
    )

    use_zstandardization = params.get("z-standardize_data", False)
    mean_dict, std_dict = compute_standardization_stats(
        stats_dataset,
        data_keys=data_datasets,
    )

    def build_loader(dataset_indices):
        if len(dataset_indices) == 0:
            print("[setup] Skipping empty dataset split.")
            return None
        print(f"[setup] Loading dataset split from {dist} (indices={len(dataset_indices)})")
        dataset = LazyProfilesDataset(
            dist,
            data_datasets=data_datasets,
            label_datasets=[rule_name] if rule_name else label_datasets,
            allowed_indices=dataset_indices,
            allow_objects=True,
        )
        dataset = ZStandardizedDataset(
            dataset,
            mean_dict,  # Use SAME statistics from training
            std_dict,  # Use SAME statistics from training
            standardize_keys=["tournament", "weighted_tournament", "rankfreq"],
            do_standardization=use_zstandardization,
        )
        return DataLoader(
            dataset=dataset,
            shuffle=params.get("shuffle"),
            num_workers=params.get("num_workers"),
            batch_size=params.get("batch_size"),
            collate_fn=safe_collate,
        )

    train_loader = build_loader(train_indices)

    # If 100% of the data is used for training, return only the train loader:
    if proportions == (1.0, 0.0):
        return train_loader, None, None

    # The totally normal and reasonable way of getting preflib data to work for evaluation
    # Otherwise there's an issue with indices which can be very easily fixed by just restructuring the entire data model
    # test_indices = test_indices[np.where(test_indices < 2000)]

    dataloaders = [
        train_loader,
        build_loader(valid_indices),
        build_loader(test_indices),
    ]

    return (*dataloaders,)


class PreloadedProfilesDataset(Dataset):
    ## TODO: Evaluate if loading all of the dataset into memory is more effective than
    ##       lazy loading hdf5, which should be faster but is certainly more complex.
    ...


class LazyProfilesDataset(Dataset):
    def __init__(self, file_path, data_datasets, label_datasets, allowed_indices, allow_objects: bool = False):
        """Initializes the dataset class with paths to the datasets within the HDF5 file.
        
        Args:
            file_path (str): Path to the HDF5 file.
            data_datasets (list): List of dataset names that contain the training data.
            label_datasets (list): List of dataset names that contain the labels.
            allowed_indices (list): List of indices allowed for dataset to partition between train/test/valid
            allow_objects (bool): If True, return object datasets (NOT PICKLED!).
                                  Object shapes:
                                  - topk_swaps: dict[(voter_idx, swap_start_pos)] -> swapped profile (list of num_voters tuples of length m)
                                  - profitable_abstentions: dict[abstainer_idx] -> features/profile with that voter removed
        """
        self.file_path = file_path

        self.dataset_names = data_datasets
        self.label_names = label_datasets
        self.allow_objects = allow_objects

        # To manage train/valid/test data, datasets are allowed specific indices to draw from:
        self.allowed_indices = allowed_indices
        self.dataset_len = len(allowed_indices)

        self.datasets = {}
        self.labels = {}
        self.object_datasets = {}
        self.file = None

    def _resolve_dataset_key(self, name):
        """Return the key in the h5 file that matches the requested name (handles a few aliases)."""
        if name in self.file:
            return name

        # Support legacy names that drop underscores (e.g., rank_freq -> rankfreq)
        alias = name.replace("_", "")
        if alias in self.file:
            return alias

        return None

    def _resolve_object_key(self, name):
        """Resolve object dataset keys under the objects/ namespace."""
        candidates = [f"objects/{name}", f"objects/{name.replace('_', '')}"]
        for key in candidates:
            if key in self.file:
                return key
        return None

    def _open_file(self):
        """(Re)open the backing h5 file and refresh cached dataset handles."""
        self.file = h5py.File(self.file_path, 'r')

        self.datasets = {}
        for name in self.dataset_names:
            resolved = self._resolve_dataset_key(name)
            if resolved:
                self.datasets[name] = self.file[resolved]

        self.labels = {}
        for name in self.label_names:
            resolved = self._resolve_dataset_key(name)
            if resolved:
                self.labels[name] = self.file[resolved]

        self.object_datasets = {}
        # Load pickled object datasets (stored under objects/<name>)
        for name in self.dataset_names + self.label_names:
            obj_key = self._resolve_object_key(name)
            if obj_key:
                self.object_datasets[name] = self.file[obj_key]

    def reset_file(self):
        """Close any open h5 handles so workers reopen cleanly after forking."""
        try:
            if self.file is not None:
                self.file.close()
        finally:
            self.file = None
            self.datasets = {}
            self.labels = {}
            self.object_datasets = {}

    def __len__(self):
        # Use the length of the first data dataset as reference for the length of the dataset
        return self.dataset_len

    def __getitem__(self, idx):
        # While it is much more intuitive to load the file in the .init(), there is a bug with num_workers > 0:
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/5
        # Hence, we must open the file once per getitem... :

        # Get valid index from all required datasets
        valid_idx = self.allowed_indices[idx]

        if self.file is None or (hasattr(self.file, "id") and not getattr(self.file.id, "valid", False)):
            self._open_file()

        # Get the data for the valid ids:
        data = {}
        for name in self.dataset_names:
            if name in self.datasets:
                data[name] = torch.tensor(self.datasets[name][valid_idx])
            elif name in self.object_datasets:
                raw_bytes = bytes(self.object_datasets[name][valid_idx])
                obj = pickle.loads(raw_bytes)
                # skip non-tensorable, non-uniform objects unless explicitly allowed
                if isinstance(obj, (dict, list, tuple)) and not hasattr(obj, "shape") and not self.allow_objects:
                    continue
                data[name] = obj

        labels = {}
        for name in self.label_names:
            if name in self.labels:
                labels[name] = torch.tensor(self.labels[name][valid_idx])
            elif name in self.object_datasets:
                raw_bytes = bytes(self.object_datasets[name][valid_idx])
                labels[name] = pickle.loads(raw_bytes)

        if len(data) == 0:
            available = list(self.file.keys())
            raise ValueError(
                f"No data found for index {valid_idx} in datasets {self.dataset_names}. "
                f"Available datasets: {available}"
            )

        return data, labels

    # def close(self):
    #     # Close the h5 file when the dataset object is no longer in use
    #     self.file.close()


def safe_collate(batch):
    """
    Collate that stacks tensorable entries and leaves non-tensorable entries as lists (for object columns).
    """
    data_batch, label_batch = zip(*batch)

    def collate_dict(items):
        out = {}
        # gather keys across items
        keys = set()
        for d in items:
            keys.update(d.keys())
        for k in keys:
            vals = [d[k] for d in items if k in d]
            # If this is a non-tensorable object (e.g., dict of swaps), keep it as a list of the same shape- do not apply default_collate as this will merge keys and result in indexing errors across batches, especially evident at smaller batch sizes. 
            first = vals[0]
            if isinstance(first, (dict, list, tuple)) and not isinstance(first, torch.Tensor) and not hasattr(first,
                                                                                                               "shape"):
                out[k] = vals
                continue
            try:
                out[k] = default_collate(vals)
            except Exception:
                out[k] = vals  # leave as list
        return out

    return collate_dict(data_batch), collate_dict(label_batch)


def compute_standardization_stats(dataset, data_keys=None):
    """
    Compute mean and std for z-standardization from a LazyProfilesDataset.

    This function iterates through the dataset ONCE to compute statistics,
    then closes everything cleanly so the result can be used with multi-worker DataLoaders.

    Args:
        dataset: A LazyProfilesDataset instance
        data_keys: List of keys in the data dict to standardize. If None, uses all keys.

    Returns:
        mean_dict: Dictionary mapping data keys to mean tensors
        std_dict: Dictionary mapping data keys to std tensors
    """
    # print("Computing z-standardization statistics...")

    # Accumulate all data
    accumulated_data = {}

    for idx in range(len(dataset)):
        data, labels = dataset[idx]

        # Determine which keys to process
        keys_to_process = data_keys if data_keys is not None else data.keys()

        for key in keys_to_process:
            if key not in data:
                continue

            # Only standardize tensor data
            if not isinstance(data[key], torch.Tensor):
                continue

            if key not in accumulated_data:
                accumulated_data[key] = []

            accumulated_data[key].append(data[key].flatten().numpy())

    # Compute statistics
    mean_dict = {}
    std_dict = {}

    for key, data_list in accumulated_data.items():
        all_data = np.array(data_list)
        mean_dict[key] = torch.tensor(all_data.mean(axis=0), dtype=torch.float32)
        std_dict[key] = torch.tensor(all_data.std(axis=0), dtype=torch.float32)

        # Prevent division by zero
        std_dict[key][std_dict[key] == 0] = 1.0

        print(f"  {key}: mean shape={mean_dict[key].shape}, std shape={std_dict[key].shape}")

    print("Standardization statistics computed.")

    # Ensure no open h5 handles are inherited by DataLoader workers
    if hasattr(dataset, "reset_file"):
        dataset.reset_file()

    return mean_dict, std_dict


class ZStandardizedDataset(Dataset):
    """
    Wrapper dataset that applies z-standardization to specified keys in the data dictionary.

    This works with LazyProfilesDataset which returns (data_dict, labels_dict) tuples.
    """

    def __init__(self, base_dataset, mean_dict, std_dict, standardize_keys=None, do_standardization=False):
        """
        Args:
            base_dataset: The underlying dataset (e.g., LazyProfilesDataset)
            mean_dict: Dictionary mapping data keys to mean tensors
            std_dict: Dictionary mapping data keys to std tensors
            standardize_keys: List of keys to standardize. If None, standardizes all keys in mean_dict.
        """
        self.base_dataset = base_dataset
        self.mean_dict = mean_dict
        self.std_dict = std_dict
        self.standardize_keys = standardize_keys if standardize_keys is not None else list(mean_dict.keys())
        self.do_standardization = do_standardization

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, labels = self.base_dataset[idx]

        # Create a copy of data dict to avoid modifying the original
        standardized_data = {}

        for key, value in data.items():
            if key in self.standardize_keys and key in self.mean_dict:
                # Apply standardization
                if isinstance(value, torch.Tensor):
                    original_shape = value.shape
                    if self.do_standardization:
                        value_flat = value.flatten()
                        standardized = (value_flat - self.mean_dict[key]) / self.std_dict[key]
                        standardized_data[key] = standardized.reshape(original_shape)
                    else:
                        standardized_data[key] = value
                else:
                    # Keep non-tensor data as-is
                    standardized_data[key] = value
            else:
                # Keep data that shouldn't be standardized
                standardized_data[key] = value

        return standardized_data, labels
