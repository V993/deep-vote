
import os
import json
import glob
import torch
import numpy as np 
import pandas as pd
import torch.nn as nn
from os import cpu_count
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import KLDivLoss, L1Loss, NLLLoss, CrossEntropyLoss, MSELoss
from scipy.spatial.distance import jensenshannon

from transforms import (
    rankfreq,
    weighted_tournament,
    tournament,
    lex,
    full
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
from loss_fns import (
    AsspLoss,
    MonotonictyLoss,
    ParticipationLoss,
    TopCycleLoss,
    JensonShannonLoss,
    SymKLDivLoss,
    MajorityWinnerLoss,
    CondorcetLoss,
    JointLoss
)


## Rules/Transforms/Losses:

def get_rule(rule=None):
    rules_funcs = {
        "black_pscf": black_pscf,
        "borda_pscf": borda_pscf,
        "condorcet_pscf": condorcet_pscf,
        "copeland_pscf": copeland_pscf,
        "irv_pscf": irv_pscf,
        "plurality_pscf": plurality_pscf,
        "rankedpairs_pscf": rankedpairs_pscf,
        "schulze_pscf": schulze_pscf,
        "maximin_pscf" : maximin_pscf
    }

    if rule is None:
        return rules_funcs
    if isinstance(rule, str) and rule.lower() in ("none", "null"):
        return None
    if rule in rules_funcs:
        return rules_funcs[rule]
    
    return rules_funcs

def get_transform(transform=None):
    transform_funcs = {
        "rankfreq": rankfreq,
        "weighted_tournament": weighted_tournament,
        "tournament": tournament,
        "lex": lex, # ignore 
        "full": full, # ignore
        "tournament_upper_triangular": None, # if we want to learn axioms, this will need a function
        "weighted_tournament_upper_triangular": None, # if we want to learn axioms, this will need a function
    }

    if transform and transform in transform_funcs:
        return transform_funcs[transform]

    # For concatenation jobs, we should return None as the transform
    return None


def get_axiom(axiom=None):
    axioms = {
        "AsspLoss": AsspLoss(),
        "MontonicityLoss": MonotonictyLoss(),
        "MonotonicityLoss": MonotonictyLoss(),
        "ParticipationLoss": ParticipationLoss(),
        "TopCycleLoss": TopCycleLoss(),
        "MajorityWinnerLoss": MajorityWinnerLoss(),
        "CondorcetLoss": CondorcetLoss(),
        "Participation+TopCycle": JointLoss(loss_fns=[ParticipationLoss(), TopCycleLoss()]),
        "AsspLoss+TopCycle": JointLoss(loss_fns=[AsspLoss(), TopCycleLoss()]),
        "AllAxiomsLoss": JointLoss(loss_fns=[ParticipationLoss(), TopCycleLoss(), AsspLoss(), MonotonictyLoss()]),
    }

    if axiom and axiom in axioms:
        return axioms[axiom]
    
    return None

def get_loss_fn(loss=None):
    loss_fns = {
        "L1Loss": L1Loss(reduction="mean"),
        "L2Loss": MSELoss(reduction="mean"),
        "KLDivLoss": KLDivLoss(reduction="batchmean"),
        "CrossEntropyLoss": CrossEntropyLoss(),
        "JensonShannonLoss": JensonShannonLoss(reduction="batchmean"),
        "SymKLDivLoss": SymKLDivLoss(reduction="batchmean")
    }

    if loss and loss in loss_fns:
        return loss_fns[loss]
    else:
        # print("[utils] Non-default loss, checking for axiom loss")
        axiom = get_axiom(loss)
        if axiom is not None:
            return axiom

    return loss_fns

def get_experiments(experiment_name=None):
    if experiment_name==None:
        raise NotImplementedError("Need to add a default experiment dir")
    
    csv_mask = "*.csv"

    directory = str(os.getcwd()) + experiment_name
    csv_files = glob.glob(directory + csv_mask)

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found, check your formatting (probably need a slash somewhere)")

    return csv_files

## Experiment helpers:

def get_duration(metadata_filename, job_name):
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as f:
            metadata = json.loads(f.read())
            if metadata.get(f"duration_minutes_{job_name}") is not None:
                return float(metadata.get(f"duration_minutes_{job_name}"))
    
    return 0.0

# Order in which transforms are stacked to build the concatenated ("CO") embedding.
# Feeding a model any permutation other than the one it was trained on degrades accuracy
# silently rather than erroring (e.g. plurality 1.00 -> 0.09), so every training and
# evaluation path must share this constant.
#
# This is the order used by the Dec 2025 11c44v run in
# experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v/. Note the older Jul 2025
# checkpoints in experiments/11c_44v_120000p_L1Loss_11c44v/ were trained on
# ["tournament", "rankfreq", "weighted_tournament"] and must not be evaluated with this.
CONCAT_TRANSFORM_ORDER = ["tournament", "weighted_tournament", "rankfreq"]

def get_concatenated_data(data, transforms=CONCAT_TRANSFORM_ORDER):

    concatenated = torch.Tensor([])
    for transform in transforms:
        key = transform
        if key not in data:
            if key == "weighted_tournament" and "weighted_tournament-normalized" in data:
                key = "weighted_tournament-normalized"
            elif key == "rankfreq" and "rank_matrix-normalized" in data:
                key = "rank_matrix-normalized"
        # print(transform, data[key].shape)
        concatenated = torch.cat((concatenated, data[key]), dim=1) # Assumes shape = (batch_size, n_cands, n_cands)
    
    return concatenated

def save_experiment(model, train_losses, params, train_batches=None, valid_batches=None, valid_losses=None, duration=None):
    """ Saved Experiment Architecture:
    
    experiments/
        3c_5v_5000p_rules/ # contains data for all jobs
            losses/ # contains train/valid losses for all models
                batch_losses/ # contains train/valid lossses with in-epoch loss tracking (more granular)
                    train_borda_pscf_tournament.csv 
                    valid_borda_pscf_tournament.csv
                    ...
                    train_copeland_pscf_rank_freq.csv
                    valid_copeland_pscf_rank_freq.csv
                borda_pscf_tournament.csv 
                ...
                copeland_pscf_rank_freq.csv
            models/ # contains pickled models for all jobs
                borda_pscf_tournament.pt 
                ...
                copeland_pscf_tournament.pt
            metadata.json # contains the parameters.json used to run this experiment
        7c29v_100000p_axioms/
            losses/
            models/
            metadata.json
        ...
    """

    # Get the directory for the experiment (and create it and its subdirectories if it does not exist)
    experiment_dir = get_experiment_directory(params)

    print(f"Saving model/losses to '{experiment_dir}'")

    # Get the name for the job based on the job parameters (for model/losses)
    job_name = get_job_name(params)

    metadata_filename = experiment_dir+"/metadata.json"
    model_filename = experiment_dir + "/models/" + job_name + '.pt'
    losses_filename = experiment_dir + "/losses/" + job_name + '.csv'
    batch_losses_filename = experiment_dir + "/losses/batch_losses/{kind}_" + job_name + '.csv'

    # Save model:
    torch.save(model, model_filename)

    # Save losses:
    if valid_losses is not None:
        if params.get("models_directory", None) is not None:
            pd.DataFrame({"axiom_losses":train_losses, "rule_losses":valid_losses}).to_csv(losses_filename, index=False)
        else:
            pd.DataFrame({"train_losses":train_losses, "valid_losses":valid_losses}).to_csv(losses_filename, index=False)
    else:
        pd.DataFrame({"train_losses":train_losses}).to_csv(losses_filename, index=False)

    if train_batches and valid_batches:
        pd.DataFrame({
            "train_batch_losses":train_batches,
        }).to_csv(batch_losses_filename.format(kind="train"), index=False)
        pd.DataFrame({
            "valid_batch_losses":valid_batches,
        }).to_csv(batch_losses_filename.format(kind="valid"), index=False)

    # Save metadata:
    serializable_params = {k: v for k, v in params.items() if is_json_serializable(v)}
    if duration is not None:
        current_duration = get_duration(metadata_filename, job_name)
        serializable_params.update({f"duration_minutes_{job_name}":float(duration)+float(current_duration)})

    with open(metadata_filename, 'w') as f: f.write(json.dumps(serializable_params, indent=2))

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, OverflowError):
        return False

def load_model(params, device):
    """
    Load and return a saved model.
    """

    if "saved_experiment_dir" in params:
        experiment_dir = params["saved_experiment_dir"]
    else:
        # Get the directory for the experiment (and create it and its subdirectories if it does not exist)
        # Expected usage is to include a saved_experiment_dir folder rather than relying on this.
        experiment_dir = get_experiment_directory(params)

    # Get the name for the job based on the job parameters (for model/losses)
    job_name = get_job_name(params)

    # TODO: the commented out line is the correct production line, I shouldn't have hardcoded this
    # model_filename = experiment_dir + "/models/" + job_name + '.pt'

    # Temporary solution to allow for out of dist testing
    model_filename = experiment_dir + "/" + job_name + '.pt'

    print(f"Loading model named: '{model_filename}'")

    # load model:
    model = torch.load(model_filename, map_location=device, weights_only=False)

    return model

def get_job_name(params):
    # return "{rule_name}_{transform_name}_{train_dist}".format(**params)
    return "{rule_name}_{transform_name}".format(**params)

def get_experiment_directory(params):
    if bool(params.get("concatenate")) == True and not params.get("experiment_type"): # hardcode experiment type if not provided for concatenate experiments
        params["experiment_type"] = "concatenate"
    if "pref_dist" in params:
        experiment_dir = "experiments/{n_cands}c_{n_voters}v_{n_profiles}p_{pref_dist}_{loss_fn}_{experiment_type}".format(**params)
    else:
        experiment_dir = "experiments/{n_cands}c_{n_voters}v_{n_profiles}p_{loss_fn}_{experiment_type}".format(**params)

    # When the same config is run several times to estimate variance, every replicate must get its own
    # directory. Without this suffix all replicates resolve to the same path and silently overwrite each
    # other's models, losses and metadata -- and would also clobber the existing single-run results.
    replicate = params.get("replicate")
    if replicate is not None:
        experiment_dir = f"{experiment_dir}_rep{replicate}"


    # If the experiment directory doesn't exist, create the required directories:
    required_dirs = ['', '/models', '/losses', '/losses/batch_losses']

    for dir in required_dirs:
        os.makedirs(experiment_dir+dir, exist_ok=True)

    return experiment_dir

def load_job_parameters(rule_name, transform_name, param_path):
    rule = get_rule(rule_name)
    transform = get_transform(transform_name)
    
    params = {}

    with open(param_path, 'r') as f:
        params = json.load(f)

    # print(f"{json.dumps(params,indent=2)}")
    
    params["rule"] = rule
    params["transform"] = transform
    params["rule_name"] = rule_name
    params["transform_name"] = transform_name

    required = [
        'n_voters', 
        'n_cands', 
        'batch_size', 
        'n_profiles', 
        'n_epochs', 
        'loss_fn', 
        'lr', 
        'patience', 
        'device', 
        'shuffle', 
        'num_workers', 
        'output_layer', 
        'experiment_type' # "rules" or "axioms" or "concatenate"
    ]
    missing = [key for key in required if key not in params.keys()]
    if missing != []:
        raise NotImplementedError(f"Missing required learning parameter(s): {missing}")
        
    return params


## Plotting:

def close_factors(number):
    ''' 
    find the closest pair of factors for a given number
    '''
    factor1 = 0
    factor2 = number
    while factor1 +1 <= factor2:
        factor1 += 1
        if number % factor1 == 0:
            factor2 = number // factor1
        
    return factor1, factor2

def almost_factors(number):
    '''
    find a pair of factors that are close enough for a number that is close enough
    '''
    while True:
        factor1, factor2 = close_factors(number)
        if 1/2 * factor1 <= factor2: # the fraction in this line can be adjusted to change the threshold aspect ratio
            break
        number += 1
    return factor1, factor2

def generate_sinle_plot(csv_files, title="", key="valid_losses", loss="L1 Loss", scalar_multiplier=1):
    TRANSFORMS = {
        "tournament": "Tournament",
        "weighted_tournament": "Weighted Tournament",
        "rankfreq": "Rank Frequency",
        "concatenate": "Concatenated"
    }

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    fig.suptitle(title, fontsize=16)

    for filename in csv_files:
        data = pd.read_csv(filename)
        name = filename.split("/")[-1].replace(".csv", "")
        # Try to extract rule and transform
        parts = name.split("_")
        if len(parts) >= 2:
            transform = "_".join(parts[1:])
        else:
            transform = name
        label = TRANSFORMS.get(transform, transform)
        ax.plot(data[key]*scalar_multiplier, label=label, linewidth=1.2)

    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.set_title("Combined Plot", fontsize=10)
    ax.set_xlabel("Epochs", fontsize=8)
    ax.set_ylabel(loss, fontsize=8)
    ax.set_ylim(-0.01, 0.3)
    ax.tick_params(axis='both', labelsize=7)

    return fig

def generate_plots(csv_files, title="", key="valid_losses", key2="", loss="L1 Loss"):
    RULES = {
        "black_pscf": "Black's Rule",
        "maximin_pscf": "Simpson-Kramer",
        "borda_pscf": "Borda",
        # "condorcet_pscf": "Condorcet",
        "copeland_pscf": "Copeland",
        "irv_pscf": "IRV",
        "plurality_pscf": "Plurality",
        "rankedpairs_pscf": "Ranked Pairs",
        "schulze_pscf": "Schulze",
        "AXIOM" : "Participation"
    }
    TRANSFORMS = {
        "tournament": "Tournament",
        "weighted_tournament": "Weighted Tournament",
        "rankfreq": "Rank Frequency",
        "concatenate": "Concatenated"
    }
    
    csv_files = sorted(csv_files)

    num_rules = len(RULES)
    ncols = 3
    nrows = (num_rules + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4.5, nrows * 4), tight_layout=True)
    fig.suptitle(title, fontsize=16)
    axs = axs.flatten()

    for i, (rule, title) in enumerate(RULES.items()):
        ax = axs[i]
        for filename in csv_files:
            if rule not in filename:
                continue
            data = pd.read_csv(filename)
            transform = filename.split("/")[-1].replace(".csv", "").replace(rule + "_", "")
            ax.plot(data[key], label=TRANSFORMS.get(transform, transform), linewidth=1.2)

        ax.legend(frameon=False, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epochs", fontsize=8)
        ax.set_ylabel(loss, fontsize=8)
        ax.set_ylim(-0.01, 0.3)
        ax.tick_params(axis='both', labelsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    return fig

def get_plots(csv_files, refiner=None, max_n=None, figsize=(20,10), title=None):
    # Generate an assortment of many different plots for multiple experiments using an array of csv files

    if len(csv_files) == 0:
        print("no csv files in recieved list")
        return None,None

    # Get factors
    s1, s2 = almost_factors(len(csv_files))

    # Establish figure with factors
    fig, axs = plt.subplots(s1,s2, figsize=figsize, tight_layout=True, squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Unpack all data
    all_data = []
    for filename in csv_files:
        if refiner:
            data = pd.read_csv(filename)[refiner]
        else:
            data = pd.read_csv(filename)

        if max_n is not None:
            data = data.iloc[0:int(max_n)]
        all_data.append((filename, data))
    
    # Compute shared maxs/mins
    max_x, min_x, max_y, min_y = get_bounds(all_data)

    # Plot on axes
    for (filename, data), ax in zip(all_data, axs.ravel()):
        data.plot(ax=ax)
        ax.axhline(y=0, color='black')

        ax.grid(True)
        title = filename.split("/")[-1].replace(".csv", "")
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

    return fig
    # return fig, axs

def get_bounds(all_data):
    max_x = max_y = float('-inf')
    min_x = min_y = float('inf')
    for _, data in all_data:
        if not data.empty:
            max_x = max(max_x, data.index.max())
            min_x = min(min_x, data.index.min())
            max_y = max(max_y, data.max(numeric_only=True).max())
            min_y = min(min_y, data.min(numeric_only=True).min()) -0.01
    return max_x, min_x, max_y, min_y

def get_plots_by_tournament(csv_files, refiner=None, max_n=None, figsize=(20, 10), title=None):
    if not csv_files:
        print("No CSV files in received list")
        return None

    # Filter files into categories
    file_categories = {
        "tournament": [f for f in csv_files if "tournament" in f and "weighted" not in f],
        "weighted": [f for f in csv_files if "weighted" in f],
        "rankfreq": [f for f in csv_files if "rankfreq" in f],
        "concatenate": [f for f in csv_files if "concatenate" in f],
    }

    # Remove empty categories
    non_empty_categories = {k: v for k, v in file_categories.items() if v}
    if not non_empty_categories:
        print("No matching CSV categories found.")
        return None

    file_groups = list(non_empty_categories.values())
    group_names = list(non_empty_categories.keys())

    cols = len(file_groups)
    rows = max(1, max(len(g) for g in file_groups))
    total_files = sum(len(g) for g in file_groups)

    # Special layout for exactly 2 files total
    if total_files == 2:
        rows, cols = 1, 2

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.patch.set_facecolor('white')

    # Load and prepare data
    all_data = []
    grouped_data = []
    for group in file_groups:
        group_data = []
        for filename in group:
            try:
                data = pd.read_csv(filename)
                if refiner:
                    data = data[refiner]
                if max_n is not None:
                    data = data.iloc[:int(max_n)]
                all_data.append((filename, data))
                group_data.append((filename, data))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        grouped_data.append(group_data)

    if not all_data:
        print("No valid data loaded from CSV files.")
        return None

    # Compute shared bounds
    max_x, min_x, max_y, min_y = get_bounds(all_data)

    # Plotting
    plot_index = 0
    for col, group_data in enumerate(grouped_data):
        for row, (filename, data) in enumerate(group_data):
            if total_files == 2:
                ax = axs[0][plot_index]
            else:
                ax = axs[row][col]
            data.plot(ax=ax)
            ax.set_title(filename.split("/")[-1].replace(".csv", ""))
            ax.set_xlabel("Epochs")
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.grid(True)
            plot_index += 1

    # Hide unused axes
    for r in range(rows):
        for c in range(cols):
            if total_files == 2 and (r > 0 or c >= total_files):
                axs[r][c].axis("off")
            elif c >= len(grouped_data) or r >= len(grouped_data[c]):
                axs[r][c].axis("off")

    plt.tight_layout(pad=3)
    return fig

def make_scalar_multiplier(scalar):
    def transform(df):
        return df.apply(lambda col: col * scalar if pd.api.types.is_numeric_dtype(col) else col)
    return transform

def get_combined_plots(named_file_groups, refiner=None, max_n=None, figsize=(20, 10), title=None, transform_first_group=None):
    def get_bounds(all_data):
        max_x = max_y = float('-inf')
        min_x = min_y = float('inf')
        for _, data in all_data:
            if not data.empty:
                max_x = max(max_x, data.index.max()) + 0.01
                min_x = min(min_x, data.index.min()) - 0.01
                max_y = max(max_y, data.max(numeric_only=True).max()) + 0.01
                min_y = min(min_y, data.min(numeric_only=True).min()) - 0.01
        return max_x, min_x, max_y, min_y

    # Flatten and filter
    grouped_data = []
    all_data = []
    valid_groups = []

    for i, (label, file_list) in enumerate(named_file_groups):
        group = []
        for f in file_list:
            try:
                df = pd.read_csv(f)
                if refiner:
                    df = df[refiner]
                if transform_first_group and i == 0:
                    df = transform_first_group(df)
                if max_n is not None:
                    df = df.iloc[:max_n]
                group.append((f, df))
                all_data.append((f, df))
            except Exception as e:
                print(f"Failed loading {f}: {e}")
        if group:
            grouped_data.append(group)
            valid_groups.append(label)

    if not all_data:
        print("No data to plot.")
        return None

    rows = max(len(g) for g in grouped_data)
    cols = len(grouped_data)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle(title or "Combined Plots", fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("white")

    max_x, min_x, max_y, min_y = get_bounds(all_data)

    for col, group in enumerate(grouped_data):
        for row, (fname, df) in enumerate(group):
            ax = axs[row][col]
            df.plot(ax=ax)
            ax.set_title(fname.split("/")[-1].replace(".csv", ""))
            ax.set_xlabel("Epochs")
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.grid(True)

    # Hide empty subplots
    for r in range(rows):
        for c in range(cols):
            if r >= len(grouped_data[c]):
                axs[r][c].axis("off")

    # Optionally label each column with group name
    for col, label in enumerate(valid_groups):
        axs[0][col].set_title(f"{label}\n{axs[0][col].get_title()}")

    plt.tight_layout(pad=3)
    return fig

def plot_grouped_training_loss_grid(crossentropy_paths, l1_paths, refiner=None, max_epochs=None, figsize=(20, 10), title=None):
    def extract_suffix(path):
        return "_".join(os.path.basename(path).split("_")[1:])

    # Pair files by suffix
    ce_dict = {extract_suffix(p): p for p in crossentropy_paths}
    l1_dict = {extract_suffix(p): p for p in l1_paths}
    common_keys = sorted(set(ce_dict) & set(l1_dict))

    # Categorize keys by transform type
    tournament_keys = [k for k in common_keys if "tournament" in k and "weighted" not in k]
    weighted_keys = [k for k in common_keys if "weighted" in k]
    rankfreq_keys = [k for k in common_keys if "rankfreq" in k]
    file_groups = [tournament_keys, weighted_keys, rankfreq_keys]

    max_len = max(len(g) for g in file_groups)
    total_files = sum(len(g) for g in file_groups)

    # Layout
    cols = 3 if total_files > 2 else total_files
    rows = max_len

    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.patch.set_facecolor('white')

    # Preload all data and compute Y bounds
    all_train_losses = []
    grouped_data = []
    for group in file_groups:
        group_data = []
        for key in group:
            ce_df = pd.read_csv(ce_dict[key])
            l1_df = pd.read_csv(l1_dict[key])
            if refiner:
                ce_df = ce_df[refiner]
                l1_df = l1_df[refiner]
            ce_losses = ce_df['train_losses'].values
            l1_losses = l1_df['train_losses'].values
            min_len = min(len(ce_losses), len(l1_losses))
            if max_epochs:
                min_len = min(min_len, max_epochs)
            ce_losses = ce_losses[:min_len]
            l1_losses = l1_losses[:min_len]
            all_train_losses.extend(ce_losses)
            all_train_losses.extend(l1_losses)
            group_data.append((key, ce_losses, l1_losses))
        grouped_data.append(group_data)

    min_y, max_y = min(all_train_losses), max(all_train_losses)

    # Plotting
    for col, group in enumerate(grouped_data):
        for row, (key, ce_losses, l1_losses) in enumerate(group):
            if row >= rows or col >= cols:
                continue
            ax = axs[row][col]
            x = list(range(len(ce_losses)))
            ax.plot(x, ce_losses, label="CrossEntropy", linewidth=2)
            ax.plot(x, l1_losses, label="L1Loss", linewidth=2, linestyle='--')
            ax.set_title(key.replace(".csv", ""), fontsize=10)
            ax.set_ylim(min_y, max_y)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend()

    # Hide unused subplots
    for r in range(rows):
        for c in range(cols):
            if r >= len(grouped_data[c]) if c < len(grouped_data) else True:
                axs[r][c].axis("off")

    plt.tight_layout(pad=3, rect=[0, 0, 1, 0.95])
    return fig

def get_comparisons(list1, list2, list3=None, refiner=None, max_n=None, title=None, figsize=(30, 20), rescale=1):
    """Generates comparison plots for experiment results with a distinct line for concatenated_valid_losses."""
    
    if len(list1) == 0:
        print("No CSV files found in the received list.")
        return None, None

    # Determine grid layout
    s1, s2 = almost_factors(len(list1))

    # Set up the figure with an improved aesthetic
    fig, axs = plt.subplots(s1, s2, figsize=figsize, squeeze=False)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.patch.set_facecolor('white')
    plt.style.use("ggplot")  # Use a more visually appealing style
        
    def get_column(filename):
        """Extracts the base name of the job from a file path."""
        job_name = filename.split("/")[-1]
        return job_name.split('.')[0]

    def scaler(x):
        """Rescales the data if a rescale function is provided."""
        return rescale*x

    # Load and prepare data
    dataframe_list = []
    for filename in list1:
        list2_matches = [item for item in list2 if get_column(filename).split("_")[0] in item]
        list3_matches = [item for item in list3 if get_column(filename).split("_")[0] in item] if list3 else []
        
        completed_dataframe = pd.DataFrame()
        base_data = pd.read_csv(filename)

        if refiner:
            completed_dataframe = base_data[refiner]
            for filename2 in list2_matches:
                data2 = pd.read_csv(filename2)[refiner]
                data2 = data2.map(scaler) # Rescale by our given rescale value
                data2 = data2.rename(columns={
                    "train_losses": f"{get_column(filename2)}_train_losses", 
                    "valid_losses": f"{get_column(filename2)}_valid_losses"
                })
                completed_dataframe = pd.concat([completed_dataframe, data2], axis=1)
        else:
            completed_dataframe = base_data
            for filename2 in list2_matches:
                data2 = pd.read_csv(filename2)
                data2 = data2.map(scaler) # Rescale by our given rescale value
                data2 = data2.rename(columns={
                    "train_losses": f"{get_column(filename2)}_train_losses", 
                    "valid_losses": f"{get_column(filename2)}_valid_losses"
                })
                completed_dataframe = pd.concat([completed_dataframe, data2], axis=1)

        if max_n is not None:
            completed_dataframe = completed_dataframe.iloc[:max_n]

        # Select only validation losses
        completed_dataframe = completed_dataframe[[col for col in completed_dataframe.columns if "valid" in col]]
        completed_dataframe = completed_dataframe.rename(columns={"valid_losses": "concatenated_valid_losses"})
        
        # If list3 is provided, add its concatenated_valid_losses
        if list3_matches:
            for filename3 in list3_matches:
                data3 = pd.read_csv(filename3)
                if "valid_losses" in data3.columns:
                    completed_dataframe[f"{get_column(filename3)}_concatenated_valid_losses"] = data3["valid_losses"]
        
        dataframe_list.append((filename, completed_dataframe))
    
    # Compute shared bounds
    max_x, min_x, max_y, min_y = get_bounds(dataframe_list)

    # Plot while making "concatenated_valid_losses" stand out
    for (filename, data), ax in zip(dataframe_list, axs.ravel()):
        # Assign distinct colors excluding red for valid_loss_cols
        valid_loss_cols = [col for col in data.columns if col != "concatenated_valid_losses" and "_concatenated_valid_losses" not in col]
        colors = cm.autumn(np.linspace(0, 1, len(valid_loss_cols)))
        for col, color in zip(valid_loss_cols, colors):
            data[col].plot(ax=ax, alpha=0.5, linewidth=1, color=color, label=col)

        # Highlight the "concatenated_valid_losses" line separately
        if "concatenated_valid_losses" in sorted(data.columns):
            data["concatenated_valid_losses"].plot(
                ax=ax, linewidth=2, color='navy', linestyle='dashed', zorder=1, label="150W Concatenated Valid Losses"
            )
        
        # Plot additional concatenated_valid_losses from list3
        for col in sorted(data.columns):
            if "_concatenated_valid_losses" in col:
                data[col].plot(
                    ax=ax, linewidth=2, color='purple', linestyle='dotted', zorder=1, label="120W Concatenated Valid Losses"
                )
        
        title = filename.split("/")[-1].replace(".csv", "")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.8)
        
        # Customize legend
        legend = ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, borderpad=1)
        for line in legend.get_lines():
            line.set_linewidth(2.5)

    return fig

def get_csvs(dir):
    directory = str(os.getcwd())+dir
    csv_mask = "*.csv"
    return glob.glob(directory + csv_mask)
