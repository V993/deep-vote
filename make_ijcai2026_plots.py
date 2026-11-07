import itertools
import os
from idlelib.format import find_paragraph

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from matplotlib.lines import Line2D
import re
import json
from pathlib import Path

rule_plot_names = {
    "black_pscf": "Black's Rule",
    "borda_pscf": "Borda",
    "copeland_pscf": "Copeland",
    "irv_pscf": "IRV",
    "maximin_pscf": "Simpson-Kramer",
    "plurality_pscf": "Plurality",
    "rankedpairs_pscf": "Ranked Pairs",
    "schulze_pscf": "Schulze",
}
axiom_plot_shortnames = {
    "MonotonicityLoss": "Mono.",
    "ParticipationLoss": "Part.",
    "TopCycleLoss": "TC",
    "AsspLoss": "ASSP",
}
axiom_plot_names = {
    "MonotonicityLoss": "Monotonicity",
    "ParticipationLoss": "Participation",
    "TopCycleLoss": "Top Cycle",
    "AsspLoss": "ASSP",
}
rule_order = [
    "plurality_pscf",
    "borda_pscf",
    "copeland_pscf",
    "schulze_pscf",
    "maximin_pscf",
    "irv_pscf",
    "black_pscf",
    "rankedpairs_pscf",
]
rule_names = [
    "plurality_pscf",
    "borda_pscf",
    "copeland_pscf",
    "schulze_pscf",
    "maximin_pscf",
    "irv_pscf",
    "black_pscf",
    "rankedpairs_pscf",
]
features = [
    "rankfreq",
    "tournament",
    "weighted_tournament",
    "concatenate"
]
transform_names = {
    "rankfreq": "Rank Frequency",
    "tournament": "Tournament",
    "weighted_tournament": "Weighted Tournament",
    "concatenate": "Concatenation"
}
transform_table_names = {
    "rule_names": "Target Rule",
    "rankfreq-winner_acc": "$T_{RF}$",
    "tournament-winner_acc": "$T_{T}$",
    "weighted_tournament-winner_acc": "$T_{WT}$",
    "concat-winner_acc": "$T_{CO}$",
    "rankfreq-loss": "$T_{RF}$",
    "tournament-loss": "$T_{T}$",
    "weighted_tournament-loss": "$T_{WT}$",
    "concat-loss": "$T_{CO}$",
    "rankfreq": "$T_{RF}$",
    "tournament": "$T_{T}$",
    "weighted_tournament": "$T_{WT}$",
    "concat": "$T_{CO}$",
    "concatenate": "$T_{CO}$",
}
transform_colours = {
    "rankfreq": "tab:green",
    "tournament": "tab:orange",
    "weighted_tournament": "tab:red",
    "concatenate": "tab:blue"
}
participation_plot_colours = {
    "Participation Loss": "tab:orange",
    "Rule Loss": "tab:green",
    "Combined Loss": "tab:blue",
}

dist_names = {
    "rule_names": "Target Rule",
    "preflib": "PrefLib",
    "impartial_culture": "IC",
    "impartial_anonymous_culture": "IAC",
    "urn": "Urn",
    "mallows": "Mallows",
    "sp_walsh": "SP",
}


def augment_final_validations():
    """
    Function to add a column to each csv file in the final validations subfolder.
    Should also edit the corresponding json file with a new key computed from the added column.
    """

    def _add_tied_winner_accuracy(df_to_edit, json_to_edit, new_col, new_key):
        df_to_edit[new_col] = df_to_edit.apply(
            lambda row: eval(row["label"])[row["pred_argmax"]] > 0,
            axis=1
        )
        acc = df_to_edit[new_col].mean()
        json_to_edit[new_key] = acc
        return df_to_edit, json_to_edit

    base_folder = "IJCAI-2026/final_validations"
    # Data folder names
    whalrus_folder = "whalrus"
    pretrained_folder = "pretrained_rules_eval"
    finetuned_folder = "finetuned_rules_eval"
    data_folders = [
        pretrained_folder,
        whalrus_folder,
        finetuned_folder
    ]

    # Axiom folder names; all axiom folders appears within each data folder
    axiom_folders = [
        "MonotonicityLoss",
        "ParticipationLoss",
        "TopCycleLoss",
        "AsspLoss",
    ]
    feature_types = [
        "rankfreq",
        "tournament",
        "weighted_tournament",
        "concatenate"
    ]

    rule_names = ["plurality_pscf", "borda_pscf", "copeland_pscf", "schulze_pscf", "maximin_pscf", "irv_pscf",
                  "black_pscf", "rankedpairs_pscf"]

    for rule_name, feature, axiom_folder, data_folder in itertools.product(rule_names, feature_types, axiom_folders,
                                                                           data_folders):
        # Extract violation rates for each model type
        # for data_folder, model_type in [
        #     (pretrained_folder, "base"),
        #     (whalrus_folder, "whalrus"),
        #     (finetuned_folder, "finetune")
        # ]:
        folder_path = os.path.join(base_folder, data_folder, axiom_folder)

        df = None
        df_filename = None
        json_file = None
        json_filename = None
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                # load json data
                pattern = rf".*-{axiom_folder}-{re.escape(rule_name)}-{feature}.*_metrics\.json"
                if re.match(pattern, filename):
                    json_filename = os.path.join(folder_path, filename)
                    try:
                        with open(json_filename, 'r') as f:
                            json_file = json.load(f)

                        # load csv data; construct by replacing 'metrics' with 'instances'
                        df_filename = json_filename.replace("metrics.json", "instances.csv")
                        df = pd.read_csv(df_filename)
                        break
                    except Exception as e:
                        print(f"Error reading {json_filename}: {e}")
                # pattern = rf".*-{axiom_folder}-{re.escape(rule_name)}-{feature}.*_instances\.csv"
                # if re.match(pattern, filename):
                #     filepath = os.path.join(folder_path, filename)
                #     df_filename = filepath
                #     df = pd.read_csv(filepath)
                #     break
        if df is None and json_file is None:
            print(f"Skipping ({rule_name}, {feature}, {axiom_folder}) in {data_folder} BOTH")
            continue
        if df is None:
            print(f"Skipping ({rule_name}, {feature}, {axiom_folder}) in {data_folder} PANDAS")
            continue
        if json_file is None:
            print(f"Skipping ({rule_name}, {feature}, {axiom_folder}) in {data_folder} JSON")
            continue

        print(f"Successfully loaded ({rule_name}, {feature}, {axiom_folder}) in {data_folder}")

        #####
        # add new column here
        # Define column names and then call a function created to
        #####

        # add column for loss being generous about ties
        # a prediction is correct if the predicted winner is ANY one of the tied winners
        new_df_col = "any_tied_winner_correct"
        new_json_key = "rule_accuracy_any_tied_winner"

        if new_df_col in df.columns or new_json_key in json_file.keys():
            # continue
            print(
                "About to fail but if you know that some data should already have this key replace the error with 'continue'")
            raise ValueError("A desired key already exists in the df or json. Make sure you know what you're doing!")

        df, json_file = _add_tied_winner_accuracy(df_to_edit=df,
                                                  json_to_edit=json_file,
                                                  new_col=new_df_col,
                                                  new_key=new_json_key)

        # Save new files
        # Only uncomment once you're sure!
        df.to_csv(df_filename, index=False)
        with open(json_filename, 'w') as f:
            json.dump(json_file, f, indent=2)


def construct_df_from_final_validations(json_key):
    """
    Construct a giant dataframe from all the final evaluation data across each type of rule/model.
    Pass in the json key of the value that should be tracked, e.g., violation rate, accuracy, etc.
    json_key can be a string or a list of string which are successively keys of the previous value.
    """

    base_folder = "IJCAI-2026/final_validations"
    # Data folder names
    whalrus_folder = "whalrus"
    pretrained_folder = "pretrained_rules_eval"
    finetuned_folder = "finetuned_rules_eval"
    data_folders = [
        pretrained_folder,
        whalrus_folder,
        finetuned_folder
    ]

    # Axiom folder names; all axiom folders appears within each data folder
    axiom_folders = [
        "AsspLoss",
        "MonotonicityLoss",
        "ParticipationLoss",
        "TopCycleLoss",
    ]

    feature_types = [
        "rankfreq",
        "tournament",
        "weighted_tournament",
        "concatenate"
    ]

    rule_names = ["plurality_pscf", "borda_pscf", "copeland_pscf", "schulze_pscf", "maximin_pscf", "irv_pscf",
                  "black_pscf", "rankedpairs_pscf"]

    # Create list to store all data rows
    data_rows = []

    for rule_name, feature, axiom_folder in itertools.product(rule_names, feature_types, axiom_folders):
        # for rule_name in rule_names:
        #     for axiom_folder in axiom_folders:

        row = {"rule_name": rule_name, "axiom": axiom_folder, "feature": feature}
        # Extract violation rates for each model type
        for data_folder in data_folders:
            folder_path = os.path.join(base_folder, data_folder, axiom_folder)

            # Find the matching JSON file
            tracked_value = None
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    # Match pattern: {prefix}-{axiom}-{rule_name}-concatenate_{timestamp}_metrics.json
                    pattern = rf".*-.*-{re.escape(rule_name)}-{feature}.*_metrics\.json"
                    if re.match(pattern, filename):
                        filepath = os.path.join(folder_path, filename)
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                if isinstance(json_key, str):
                                    tracked_value = data.get(json_key)
                                elif isinstance(json_key, list):
                                    tracked_value = data.get(json_key[0])
                                    for idx in range(1, len(json_key)):
                                        tracked_value = tracked_value.get(json_key[idx])
                                break
                        except Exception as e:
                            print(f"Error reading {filepath}: {e}")

            if tracked_value is None:
                continue
            row[data_folder] = tracked_value if tracked_value is not None else 0.0
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


def plot_validation_losses(path, rules, features, concat_path=None):
    """
    Plot validation losses for different combinations of rules and features.
    Code is largely AI slop, sorry.
    Parameters:
    path (str): Path to directory containing csv files for individual transforms
    concat_path (str): Path to directory containing csv files for results of concatenation
    rules (list): List of rule names
    features (list): List of feature values
    """
    y_col = "valid_losses"

    def create_subplot(ax, path, rule, features, concat_path=None):
        """
        Create a subplot for a given rule with one series for each feature value.

        Parameters:
        ax: matplotlib axis object
        path (str): Directory path containing the CSV files
        rule (str): Rule name
        features (list): List of feature values
        """

        linewidth = 2

        rule_label = rule_plot_names[rule]
        ax.set_title(f'{rule_label}', fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=12)

        # Plot series for each feature
        for feature in features:
            filename = f"{rule}_{feature}.csv"
            filepath = os.path.join(path, filename)

            try:
                # Load CSV file
                df = pd.read_csv(filepath)
                x_col = df.index

                label = transform_names[feature]
                colour = transform_colours[feature]

                # Plot the series
                ax.plot(x_col, df[y_col], label=label, markersize=2, linewidth=linewidth, color=colour)

            except FileNotFoundError:
                print(f"Warning: File {filepath} not found, skipping...")
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")
        if concat_path is not None:
            filename = f"{rule}_concatenate.csv"
            filepath = os.path.join(concat_path, filename)

            try:
                # Load CSV file
                df = pd.read_csv(filepath)
                x_col = df.index

                label = transform_names["concatenate"]
                colour = transform_colours["concatenate"]

                # Plot the series
                ax.plot(x_col, df[y_col], label=label, markersize=2, linewidth=linewidth, color=colour)

            except FileNotFoundError:
                print(f"Warning: File {filepath} not found, skipping...")
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")

        # ax.legend()
        ax.grid(True, alpha=0.3)

    # Calculate rows and columns for subplots (2 rows)
    n_rules = len(rules)
    n_cols = (n_rules + 1) // 2  # Ceiling division to get columns needed
    n_rows = 2

    # Create figure with subplots (2 rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.3 * n_rows), sharex=True, sharey=True)

    # Handle different cases for axes shape
    if n_rules == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Create subplot for each rule
    for i, rule in enumerate(rules):
        create_subplot(axes[i], path, rule, features, concat_path)

    # Hide any unused subplots
    for i in range(n_rules, len(axes)):
        axes[i].set_visible(False)

    # Create a single legend outside the plots based on the first subplot
    if n_rules > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.014), ncol=4, fontsize=12)

    plt.tight_layout()
    supy_label = fig.supylabel('L1 Loss', fontsize=14, x=0.004, y=0.5)
    supx_label = fig.supxlabel('Epoch', fontsize=14, x=0.5, y=0.08)

    # Adjust layout and show
    plt.subplots_adjust(bottom=0.19, left=0.05)
    # plt.show()
    plt.savefig(f"IJCAI-2026/plots_and_figures/{y_col}-all_rules.png", dpi=300)


def plot_participation_losses(path, rules, transform):
    """
    Plot rule and participation losses for different combinations of rules and features.
    Code is largely AI slop, sorry.
    """
    linewidth = 2

    def create_subplot(ax, rule):

        rule_label = rule_plot_names[rule]
        ax.set_title(f'{rule_label}', fontsize=16)

        # Construct filename and load dataframe
        filename = f"{path}/{rule}_{transform}.csv"
        df = pd.read_csv(filename)
        df = df.round(3)

        # Trim to first 40 rows
        if len(df) > 40:
            df = df.head(40)

        ax_series = df['axiom_losses']
        rule_series = df['rule_losses']
        combined_series = ax_series + rule_series
        series = {
            "Participation Loss": ax_series,
            "Rule Loss": rule_series,
            "Combined Loss": combined_series,
        }

        for series_name, data in series.items():
            colour = participation_plot_colours[series_name]
            ax.plot(data, label=series_name, markersize=2, linewidth=linewidth, color=colour)
        ax.grid(True, alpha=0.3)

    # Calculate rows and columns for subplots (2 rows)
    n_rules = len(rules)
    n_cols = (n_rules + 1) // 2  # Ceiling division to get columns needed
    n_rows = 2

    # Create figure with subplots (2 rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.3 * n_rows), sharex=True, sharey=True)

    # Handle different cases for axes shape
    if n_rules == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Create subplot for each rule
    for i, rule in enumerate(rules):
        create_subplot(axes[i], rule)

    # Hide any unused subplots
    for i in range(n_rules, len(axes)):
        axes[i].set_visible(False)

    # Create a single legend outside the plots based on the first subplot
    if n_rules > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.014), ncol=4, fontsize=12)

    plt.tight_layout()
    supy_label = fig.supylabel('Loss', fontsize=14, x=0.004, y=0.5)
    supx_label = fig.supxlabel('Epoch', fontsize=14, x=0.5, y=0.08)

    # Adjust layout and show
    plt.subplots_adjust(bottom=0.19, left=0.05)
    # plt.show()

    plt.savefig(f"aaai_2026_plotting/participation_loss-transform={transform}.png", dpi=300)


def print_table_from_df(df, axioms, rule_type="pretrained_rules_eval"):
    """
    Filter dataframe by rule_name, axiom, and feature values, then sort and display.
    Used, for example, to print csv data that shows the accuracy of each rule.
    Args:
        df: DataFrame to filter
        rule_names: List of rule_name values to keep
        axioms: List of axiom values to keep
        features: List of feature values to keep
        sort_order: List defining the order for rule_name sorting
    """
    features = ["tournament", "weighted_tournament", "rankfreq", "concatenate"]

    # Filter the dataframe
    df = df[
        df["rule_name"].isin(rule_order) &
        df["axiom"].isin(axioms) &
        df["feature"].isin(features)
        ]

    # Pivot: create a column for each unique feature value
    df = df.pivot_table(
        index=["rule_name", "axiom"],
        columns="feature",
        values=rule_type,
        aggfunc='first'  # Use first value if there are duplicates
    ).reset_index()

    # Sort by custom order
    # Create a categorical type with the specified order
    df["rule_name"] = pd.Categorical(
        df["rule_name"],
        categories=rule_order,
        ordered=True
    )
    df = df.sort_values("rule_name")

    # Drop the axiom column
    df = df.drop(columns=["axiom"])

    # Sort the feature columns by their order in features
    display_cols = ["rule_name"] + [f for f in features if f in df.columns]
    df = df[display_cols]

    # Replace each rule_name value with the corresponding key in rule_plot_names
    df["rule_name"] = df["rule_name"].map(rule_plot_names)

    # Rename columns using transform_table_names for feature columns
    column_renames = {f: transform_table_names.get(f, f) for f in features if f in df.columns}
    column_renames["rule_name"] = ""  # Empty cell for header
    df = df.rename(columns=column_renames)

    # Display as CSV
    print(df.to_csv(index=False))


def format_losses_main_results(folder, features, rule_dict):
    """
    Print csv formatted data representing a table showing loss given a folder containing loss data for all given
    rule and feature combos.
    """
    results = {}

    # Load data for each rule and feature
    for rule in rule_dict.keys():
        results[rule] = {}
        for feature in features:
            filename = f"{rule}_{feature}.csv"
            filepath = os.path.join(folder, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(filepath)

                # Extract the final value from the 'valid_losses' column
                if 'valid_losses' in df.columns and len(df) > 0:
                    final_loss = df['valid_losses'].iloc[-1]
                    results[rule][feature] = float(final_loss)
                else:
                    results[rule][feature] = 'N/A'
            except FileNotFoundError:
                results[rule][feature] = 'N/A'
            except Exception as e:
                results[rule][feature] = f'Error: {e}'

    # Print CSV formatted output
    # Header row
    feature_names = [transform_table_names[fn] for fn in features]
    header = ',' + ','.join(feature_names)
    print(header)

    # print final data
    for rule in rule_order:
        if rule in rule_dict.keys():
            row_values = [rule_dict[rule]]  # Start with the rule's display name
            row_values.extend([str(round(results[rule][feature], 3)) for feature in features])
            print(','.join(row_values))


def bar_plot_axiom_violations():
    """
    Make figure with one subplot for each rule.
    Each subplot shows one group of bars for each axiom.
    Each group of bars shows violation rate of the rule on the axiom for each of {whalrus, pretrained model, finetuned model}
    """

    # Axiom folder names; all axiom folders appears within each data folder
    axiom_folders = [
        "AsspLoss",
        "MonotonicityLoss",
        "ParticipationLoss",
        "TopCycleLoss",
    ]
    df = construct_df_from_final_validations(json_key="axiom_violation_rate")

    # Create figure with 4 columns and 2 rows
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    # Bar width and positions
    bar_width = 0.25
    axiom_positions = np.arange(len(axiom_folders))

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    labels = ['Whalrus', 'Pretrained', 'Finetuned']

    # Create one subplot for each rule
    for idx, rule_name in enumerate(rule_names):
        ax = axes[idx]
        rule_data = df[df['rule_name'] == rule_name]

        # Create bars for each model type
        for i, (col, color, label) in enumerate(zip(
                ['whalrus_violation_rate', 'base_violation_rate', 'ft_violation_rate'],
                colors,
                labels
        )):
            values = [rule_data[rule_data['axiom'] == axiom][col].values[0]
                      if len(rule_data[rule_data['axiom'] == axiom]) > 0 else 0.0
                      for axiom in axiom_folders]

            positions = axiom_positions + i * bar_width
            ax.bar(positions, values, bar_width, label=label, color=color, alpha=0.8)

        # Customize subplot
        ax.set_xlabel('Axiom', fontsize=10)
        ax.set_ylabel('Violation Rate', fontsize=10)
        ax.set_title(rule_name.replace('_pscf', '').capitalize(), fontsize=12, fontweight='bold')
        ax.set_xticks(axiom_positions + bar_width)
        ax.set_xticklabels([axiom.replace('Loss', '') for axiom in axiom_folders], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

    # Overall title
    fig.suptitle('Axiom Violation Rates by Rule and Model Type', fontsize=16, fontweight='bold', y=0.995)

    # Adjust layout
    plt.tight_layout()

    # Show the figure
    plt.show()


def finetune_changes_plot(df_loss, df_acc, out_path="IJCAI-2026/plots_and_figures"):
    """
    Make big plot showing all changes between actual rule, base model, and after finetuning.
    One subplot for each rule, showing changes across all axioms.
    Use two y-axes, one for rule loss and one for rule accuracy.
    """
    df_loss = df_loss[df_loss["feature"] == "concatenate"]
    df_loss = df_loss.drop(columns=["feature"])

    df_acc = df_acc[df_acc["feature"] == "concatenate"]
    df_acc = df_acc.drop(columns=["feature"])

    axioms = [
        "AsspLoss",
        "MonotonicityLoss",
        "ParticipationLoss",
        "TopCycleLoss",
    ]

    def _plot_single_axis(ax, rule_name):
        df_loss_filtered = df_loss[df_loss["rule_name"] == rule_name]
        df_acc_filtered = df_acc[df_acc["rule_name"] == rule_name]

        ax2 = ax.twinx()

        indices = list(range(len(axioms)))
        offset = 0.15
        for axiom_idx in indices:
            # left axis gets losses

            # circle for base model and finetuned, triangle for whalrus
            df_ax_loss = df_loss_filtered[df_loss_filtered["axiom"] == axioms[axiom_idx]]
            base_model_loss = df_ax_loss["pretrained_rules_eval"].iloc[0]
            finetuned_model_loss = df_ax_loss["finetuned_rules_eval"].iloc[0]
            whalrus_loss = df_ax_loss["whalrus"].iloc[0]

            markersize = 12

            plt.plot([axiom_idx - offset, axiom_idx - offset],
                     [base_model_loss, finetuned_model_loss],
                     linewidth=2,
                     alpha=0.7,
                     linestyle="--",
                     color='tab:blue',
                     zorder=1)

            ax.plot(axiom_idx - offset, base_model_loss,
                    marker='o',
                    markersize=markersize,
                    color='tab:blue',
                    zorder=10,
                    alpha=0.7,
                    markeredgecolor='black', markeredgewidth=1)
            ax.plot(axiom_idx - offset, finetuned_model_loss,
                    marker='H',
                    markersize=markersize,
                    color='tab:blue',
                    zorder=10,
                    alpha=0.8,
                    markeredgecolor='black', markeredgewidth=1)
            ax.plot(axiom_idx - offset, whalrus_loss,
                    marker='X',
                    markersize=markersize,
                    color='black',
                    zorder=1,
                    alpha=0.6,
                    markeredgecolor='black', markeredgewidth=1)

            # right axis gets accuracies
            df_ax_acc = df_acc_filtered[df_acc_filtered["axiom"] == axioms[axiom_idx]]
            base_model_acc = df_ax_acc["pretrained_rules_eval"].iloc[0]
            finetuned_model_acc = df_ax_acc["finetuned_rules_eval"].iloc[0]
            whalrus_acc = df_ax_acc["whalrus"].iloc[0]

            plt.plot([axiom_idx + offset, axiom_idx + offset],
                     [base_model_acc, finetuned_model_acc],
                     linewidth=2,
                     alpha=0.7,
                     linestyle="--",
                     color='tab:green',
                     zorder=1
                     )
            ax2.plot(axiom_idx + offset, base_model_acc,
                     marker='^',
                     markersize=markersize,
                     color='tab:green',
                     zorder=10,
                     alpha=0.7,
                     markeredgecolor='black',
                     markeredgewidth=0.5
                     )
            ax2.plot(axiom_idx + offset, finetuned_model_acc,
                     marker='v',
                     markersize=markersize,
                     color='tab:green',
                     zorder=10,
                     alpha=0.7,
                     markeredgecolor='black',
                     markeredgewidth=0.5
                     )
            # ax2.plot(axiom_idx + offset, whalrus_acc,
            #          marker='P',
            #          markersize=10,
            #          color='black',
            #          zorder=10,
            #          alpha=0.7,
            #          markeredgecolor='black',
            #          markeredgewidth=0.5
            #          )

        return ax, ax2

    fig, axs = plt.subplots(2, 4, figsize=(12, 4.2), sharey=True)
    axs = axs.flatten()

    rules = rule_order

    for idx, rule in enumerate(rules):
        ax_idx = idx % len(rules)
        ax = axs[ax_idx]

        # add data to plot
        ax, ax2 = _plot_single_axis(ax, rule)

        ax.set_ylim((-0.04, 1.04))
        ax2.set_ylim((-0.04, 1.04))
        ax.grid(True, alpha=0.3)

        ax.set_title(rule_plot_names[rule], fontsize=13)

        # rightmost subplots
        if (idx + 1) % 4 == 0:
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
        else:
            ax2.yaxis.set_visible(False)

        if idx % 4 == 0:
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])

        # top row
        if idx < 4:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels([])

        # bottom row
        if idx >= 4:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels([axiom_plot_shortnames[axiom_name] for axiom_name in axioms], rotation=45, ha='right')

    fig.supylabel("Axiom Loss / Rule Accuracy", fontsize=16, x=0.005)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markeredgecolor='black',
               markeredgewidth=0.5, markersize=11, label='Axiom Loss Before FT'),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='tab:blue', markeredgecolor='black',
               markeredgewidth=0.5, markersize=11, label='Axiom Loss After FT'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:green', markeredgecolor='black',
               markeredgewidth=0.5, markersize=11, label='Rule Accuracy Before FT'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='tab:green', markeredgecolor='black',
               markeredgewidth=0.5, markersize=11, label='Rule Accuracy After FT'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markeredgecolor='black',
               markeredgewidth=0.5, markersize=11, label='Whalrus Axiom Loss'),
    ]

    # Add legend to figure
    fig.legend(handles=legend_elements, loc='lower center', fontsize=11,
               bbox_to_anchor=(0.5, -0.012), ncol=5, frameon=True)

    plt.tight_layout(rect=[0.01, 0.08, 0.999, 1], pad=0, h_pad=0.3, w_pad=0)
    # plt.show()
    plt.savefig(f"{out_path}/finetune_summary.png", dpi=300)


def format_loss_tables_single_dist(path, rule_names, pref_dists=None, concat_path=None, use_accuracy=False):
    """
    Print loss or accuracy values for the given rules and features.
    Format values as a table with one row per rule and one column for each feature.
    """

    def format_single_dist(df):
        if use_accuracy:
            loss_columns = ['rule_names'] + [col for col in df.columns if col.endswith('-winner_acc')]
        else:
            loss_columns = ['rule_names'] + [col for col in df.columns if col.endswith('-loss')]

        df = df.round(3)

        first_col = df.columns[1]  # not actually first but close enough...

        # Filter first
        filtered_df = df[df[first_col].isin(rule_names)]

        # Set first column as index, reindex with values_list order, then reset index
        df = (filtered_df.set_index(first_col)
              .reindex(rule_names)
              .dropna()  # Remove any NaN rows if value not found
              .reset_index())

        # Filter to only include columns that actually exist in the dataframe
        available_loss_columns = [col for col in loss_columns if col in df.columns]
        loss_df = df[available_loss_columns].copy()

        # Rename values in the first column (rule_names) using rule_names dictionary
        # if rule_names is not None and 'rule_names' in loss_df.columns:
        loss_df['rule_names'] = loss_df['rule_names'].map(rule_plot_names).fillna(loss_df['rule_names'])

        # Rename column names with the embedding name
        column_mapping = {col: transform_table_names[col] for col in loss_df.columns if col in transform_table_names}
        if column_mapping:
            loss_df = loss_df.rename(columns=column_mapping)

        print(loss_df.to_csv(index=False))

    df = None
    for dist in pref_dists:
        filename = f"losses_train=impartial_culture-test={dist}-13v.csv"
        filepath = os.path.join(path, filename)

        try:
            # Load CSV file
            df = pd.read_csv(filepath)

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found, skipping...")
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")

        if df is None:
            print("Warning: Did not load data file.")
            return

        concat_df = None
        if concat_path is not None:
            filename = f"losses_train=impartial_culture-test={dist}-concatenate-13v.csv"
            filepath = os.path.join(path, filename)

            try:
                # Load CSV file
                concat_df = pd.read_csv(filepath)

                # rename columns
                column_mapping = {}
                for col in concat_df.columns:
                    if 'rankfreq' in col:
                        new_col = col.replace('rankfreq', 'concat')
                        column_mapping[col] = new_col
                if column_mapping:
                    concat_df = concat_df.rename(columns=column_mapping)

                # add the new columns into df
                for old_col, new_col in column_mapping.items():
                    if new_col in concat_df.columns:
                        df[new_col] = concat_df[new_col]

            except FileNotFoundError:
                print(f"Warning: File {filepath} not found, skipping...")
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")

        print(f"About to print {'accuracy' if use_accuracy else 'loss'} on preferences sampled from {dist}:")
        format_single_dist(df)
        print()


def format_concat_loss_table(path, feature, rule_names, pref_dists=None, use_accuracy=False, concatenate=False):
    """
    Print CSV data for networks trained on concatenated data. One column per distribution.
    """

    def format_single_dist(df):

        df = df.round(3)

        first_col = df.columns[0]  # not actually first but close enough...

        # Filter first
        df = df[df[first_col].isin(rule_names)]

        # Sort rows in order given in rule name list
        df = (df.set_index(first_col)
              .reindex(rule_names)
              .dropna()  # Remove any NaN rows if value not found
              .reset_index())

        # Rename values in the first column (rule_names) using rule_names dictionary
        # if rule_names is not None and 'rule_names' in loss_df.columns:
        df['rule_names'] = df['rule_names'].map(rule_plot_names).fillna(df['rule_names'])

        # Rename column names with the embedding name
        column_mapping = {col: dist_names[col] for col in df.columns if col in dist_names}
        if column_mapping:
            df = df.rename(columns=column_mapping)

        print(df.to_csv(index=False))

    all_dfs = []
    for dist in pref_dists:
        filename = f"losses_train=impartial_culture-test={dist}{'-concatenate' if concatenate else ''}.csv"
        filepath = os.path.join(path, filename)

        try:
            # Load CSV file
            df = pd.read_csv(filepath)

            result_col_name = dist
            if use_accuracy:
                df[result_col_name] = df[f"{feature}-winner_acc"]
            else:
                df[result_col_name] = df[f"{feature}-loss"]
            cols_to_keep = ["rule_names", result_col_name]
            df = df[cols_to_keep]

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found, skipping...")
        except Exception as e:
            print(f"Warning: Error reading {filepath}: {e}")

        if df is None:
            print("Warning: Did not load data file.")
            return
        all_dfs.append(df)

    df = all_dfs[0]
    for df1 in all_dfs[1:]:
        col_name = df1.columns[1]
        df[col_name] = df1[col_name]

    print(f"About to print {'accuracy' if use_accuracy else 'loss'} on networks trained with {feature}:")
    print(f"{'Not' if not concatenate else ''} using concatenated features")
    format_single_dist(df)
    print()


def format_participation_loss(path, transform):
    """
    Process CSV files and print participation loss data in CSV format.

    Args:
        path (str): Folder containing the data files
        transform (str): Transform string used in filename
    """

    # Print CSV header
    # print("Target Rule,$L_0^R$,$L_0^A$,$L_{50}^R$,$L_{50}^R$")
    print("Target Rule,Rule,Part.,Rule,Part.")

    for rule in rule_names:
        rule_plot_name = rule_plot_names[rule]
        try:
            # Construct filename and load dataframe
            filename = f"{path}/{rule}_{transform}.csv"
            df = pd.read_csv(filename)

            df = df.round(3)

            # Extract first and last values from axiom_losses column
            first_axiom = df['axiom_losses'].iloc[0]
            last_axiom = df['axiom_losses'].iloc[-1]

            # Extract first and last values from rule_losses column
            first_rule = df['rule_losses'].iloc[0]
            last_rule = df['rule_losses'].iloc[-1]

            # Print CSV row
            print(f"{rule_plot_name},{first_rule},{first_axiom},{last_rule},{last_axiom}")

        except FileNotFoundError:
            print(f"Warning: File {filename} not found")
        except KeyError as e:
            print(f"Warning: Column {e} not found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def format_participation_loss_v2(csv_file_path):
    rule_order = [
        "plurality_pscf",
        "borda_pscf",
        "copeland_pscf",
        "schulze_pscf",
        "maximin_pscf",
        "irv_pscf",
        "black_pscf",
        "rankedpairs_pscf",
    ]

    # Read the CSV file
    # Assuming the CSV has columns: rule, transform, rule_average, axiom_average
    # If the CSV doesn't have headers, specify them
    df = pd.read_csv(csv_file_path)

    # For each unique value 'rn' in rule_order list, find rows with the value {rn}-FT
    # Move the FT results to new columns in the row corresponding to the unique pair of rule_name and transform
    # The end result should have one rule column, one transform column, and four result columns: rule_average, axiom_average, rule_average-FT, axiom_average-FT

    result_rows = []
    transforms = df['transform'].unique()

    # For each combination of rule and transform
    for rule in rule_order:
        for transform in transforms:
            # Find the base rule row (without FT)
            base_row = df[(df['rule'] == rule) & (df['transform'] == transform)]

            # Find the FT rule row
            ft_rule_name = f"{rule} FT"
            ft_row = df[(df['rule'] == ft_rule_name) & (df['transform'] == transform)]

            # Create the result row
            result_row = {
                'rule': rule,
                'transform': transform,
                'rule_average': base_row['rule_average'].iloc[0] if not base_row.empty else None,
                'axiom_average': base_row['axiom_average'].iloc[0] if not base_row.empty else None,
                'rule_average_FT': ft_row['rule_average'].iloc[0] if not ft_row.empty else None,
                'axiom_average_FT': ft_row['axiom_average'].iloc[0] if not ft_row.empty else None
            }

            # Only add the row if we have at least some data (base or FT)
            if not base_row.empty or not ft_row.empty:
                result_rows.append(result_row)

    # Create the final dataframe
    result_df = pd.DataFrame(result_rows)

    result_df = result_df.round(3)

    # Sort by rule order and then by transform
    result_df['rule'] = pd.Categorical(result_df['rule'], categories=rule_order, ordered=True)
    result_df = result_df.sort_values(['rule', 'transform']).reset_index(drop=True)

    transform_dfs = {}

    for transform in transforms:
        transform_data = result_df[result_df['transform'] == transform].copy()

        transform_data['rule'] = pd.Categorical(transform_data['rule'], categories=rule_order, ordered=True)
        transform_data = transform_data.sort_values('rule').reset_index(drop=True)

        # Drop the transform column since it's the same for all rows in this df
        transform_data = transform_data.drop('transform', axis=1)

        # Store in dictionary with transform name as key
        transform_dfs[transform] = transform_data

        transform_data['rule'] = transform_data['rule'].map(rule_plot_names)
        transform_data.columns = ["Target Rule", "Rule", "Part.", "Rule ", "Part. "]

        print(f"Printing data for {transform}")

        # print(transform_data.head(10))
        print(transform_data.to_csv(index=False))

        print("\n")

    # print(df.head(10))


def plot_axiom_finetune_results_single_rule(in_folder, rule_name, out_folder, out_filename, axiom_name, mean_window=5,
                                            xlim=None, ylim=None):
    """
    Make plot with axiom loss on one axis and rule loss on the other axis, show how it changes during finetuning.
    """
    in_filename = f"{rule_name}_concatenate.csv"
    df = pd.read_csv(os.path.join(in_folder, in_filename))

    use_mean = True
    if use_mean:
        x = df['axiom_losses'].rolling(window=mean_window, center=True).mean().fillna(df['axiom_losses']).tolist()
        y = df['rule_losses'].rolling(window=mean_window, center=True).mean().fillna(df['rule_losses']).tolist()

        df = df.rolling(window=mean_window, center=False).mean().dropna(how="all")

        # # average over each mean_window set of points
        x = [np.mean(x[0:i + 1]) for i in range(mean_window // 2)] + df["axiom_losses"].tolist() + [
            np.mean(x[len(x) - i - 1:]) for i in range(mean_window // 2)]
        y = [np.mean(y[0:i + 1]) for i in range(mean_window // 2)] + df["rule_losses"].tolist() + [
            np.mean(y[len(y) - i - 1:]) for i in range(mean_window // 2)]
    else:
        x = df["axiom_losses"].tolist()
        y = df["rule_losses"].tolist()

    print(f"{axiom_name} -- min(x): {round(min(x), 5)} -- max(x): {round(max(x), 5)}")
    print(f"{axiom_name} -- min(y): {round(min(y), 5)} -- max(y): {round(max(y), 5)}")
    # Create figure with specified size
    plt.figure(figsize=(10, 8))

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Plot the main line
    plt.plot(x, y, linewidth=2, alpha=0.7, linestyle="--")

    plt.text(x[0], y[0], 'Start', fontsize=12, fontweight='bold', ha='right', va='bottom', zorder=10)
    plt.text(x[-1], y[-1], 'End', fontsize=12, fontweight='bold', ha='left', va='top', zorder=10)

    # Set labels and title
    plt.xlabel("Axiom Loss", fontsize=12)
    plt.ylabel("Rule Loss", fontsize=12)
    t = rule_plot_names[rule_name]
    plt.title(f"{t} {axiom_name} Fine-Tuning")

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, out_filename), dpi=300, bbox_inches='tight')

    # plt.show()


def plot_axiom_finetune_results_all_rules_all_axioms(in_folders, rule_train_folder, rule_names, out_folder, out_filename, axiom_names,
                                                     mean_window=5, xlim=None, ylim=None):
    """
    Make 2x2 plot with axiom loss on one axis and rule loss on the other axis, show how it changes during finetuning.
    in_folders: list of 4 folder paths
    axiom_names: list of 4 axiom names (one for each subplot)
    """

    def single_rule_data(in_folder, rule_name):
        in_filename = f"{rule_name}_concatenate.csv"
        df = pd.read_csv(os.path.join(in_folder, in_filename))

        # saved finetune data starts after the first epoch of training
        # provides an inconsistent baseline for rule loss in plots
        rule_loss_folder = rule_train_folder + f"/{rule_name}_concatenate.csv"
        rule_data = pd.read_csv(rule_loss_folder)

        rule_loss_initial = rule_data["valid_losses"].iloc[-1]
        axiom_loss0 = df["axiom_losses"].iloc[0]
        new_row = (axiom_loss0, rule_loss_initial)
        # unused experiments
        # axiom_loss1 = df["axiom_losses"].iloc[1]
        # y = axiom_loss1 / axiom_loss0
        # ax_orig = axiom_loss0 / y if (axiom_loss0 / y < 1.3*axiom_loss0) else axiom_loss0
        # print(f"scaling to {ax_orig} using {axiom_loss0} and {axiom_loss1}")
        # new_row = (ax_orig, rule_loss_initial)

        # Create a new dataframe with the new row and concatenate
        df = pd.concat([pd.DataFrame([new_row], columns=df.columns), df], ignore_index=True)

        use_mean = True
        if use_mean:
            x = df['axiom_losses'].rolling(window=mean_window, center=True).mean().fillna(
                df['axiom_losses']).tolist()
            y = df['rule_losses'].rolling(window=mean_window, center=True).mean().fillna(df['rule_losses']).tolist()

            df = df.rolling(window=mean_window, center=False).mean().dropna(how="all")

            # # average over each mean_window set of points
            x = [np.mean(x[0:i + 1]) for i in range(mean_window // 2)] + df["axiom_losses"].tolist() + [
                np.mean(x[len(x) - i - 1:]) for i in range(mean_window // 2)]
            y = [np.mean(y[0:i + 1]) for i in range(mean_window // 2)] + df["rule_losses"].tolist() + [
                np.mean(y[len(y) - i - 1:]) for i in range(mean_window // 2)]
        else:
            x = df["axiom_losses"].tolist()
            y = df["rule_losses"].tolist()
        return x, y

    # Create 2x2 subplots with shared y-axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 4.6), sharey=True)
    axes = axes.flatten()  # Flatten to make indexing easier

    for idx, (in_folder, axiom_name) in enumerate(zip(in_folders, axiom_names)):
        ax = axes[idx]

        for rn in rule_names:
            x, y = single_rule_data(in_folder, rn)
            # Plot the main line
            t = rule_plot_names[rn]

            line, = ax.plot(x, y, linewidth=3, alpha=0.7, label=t, linestyle="--")
            color = line.get_color()

            # Add star markers with same color
            ax.plot(x[0], y[0], marker='X', markersize=13, color=color, zorder=10, alpha=0.7,
                    markeredgecolor='black', markeredgewidth=1)
            ax.plot(x[-1], y[-1], marker='*', markersize=14, color=color, zorder=10, alpha=0.7,
                    markeredgecolor='black', markeredgewidth=1)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Set labels and title
        # ax.set_xlabel("Axiom Loss", fontsize=16)
        # if idx % 2 == 0:  # Only leftmost plots get y-label
        #     ax.set_ylabel("Rule Loss", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title(f"{axiom_name}", fontsize=13)

        # Update legend with special markers
        if idx == 2:
            legend_elements = [
                Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                       markersize=11, label='Fine-Tuning Start'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                       markersize=16, label='Fine-Tuning End'),
            ]
            handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles + legend_elements, labels + ['Fine-Tuning Start', 'Fine-Tuning End'],
            #           ncol=2,
            #           fontsize=10.5,
            #           columnspacing=0.7,
            #           labelspacing=0.5
            #           )
            # bbox_to_anchor=(0.5, -0.012)
            fig.legend(handles + legend_elements, labels + ['Fine-Tuning Start', 'Fine-Tuning End'],
                       ncol=1,
                       fontsize=11,
                       columnspacing=0.7,
                       labelspacing=0.5,
                       bbox_to_anchor=(0.99, 0.95)
                       )

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    fig.supxlabel("Axiom Loss", fontsize=13, x=0.45)
    fig.supylabel("Rule Loss", fontsize=13, x=0.01)

    plt.tight_layout(pad=0.8, rect=[0, 0, 0.83, 1])

    # Save figure
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, out_filename), dpi=300, bbox_inches='tight', pad_inches=0.05)
    # plt.show()


def plot_axiom_finetune_results_all_rules(in_folder, rule_names, out_folder, out_filename, axiom_name, mean_window=5,
                                          xlim=None, ylim=None):
    """
    Make plot with axiom loss on one axis and rule loss on the other axis, show how it changes during finetuning.
    """

    def single_rule_data(rule_name):
        in_filename = f"{rule_name}_concatenate.csv"
        df = pd.read_csv(os.path.join(in_folder, in_filename))

        use_mean = False
        if use_mean:
            x = df['axiom_losses'].rolling(window=mean_window, center=True).mean().fillna(df['axiom_losses']).tolist()
            y = df['rule_losses'].rolling(window=mean_window, center=True).mean().fillna(df['rule_losses']).tolist()

            df = df.rolling(window=mean_window, center=False).mean().dropna(how="all")

            # # average over each mean_window set of points
            x = [np.mean(x[0:i + 1]) for i in range(mean_window // 2)] + df["axiom_losses"].tolist() + [
                np.mean(x[len(x) - i - 1:]) for i in range(mean_window // 2)]
            y = [np.mean(y[0:i + 1]) for i in range(mean_window // 2)] + df["rule_losses"].tolist() + [
                np.mean(y[len(y) - i - 1:]) for i in range(mean_window // 2)]
        else:
            x = df["axiom_losses"].tolist()
            y = df["rule_losses"].tolist()
        return x, y

    plt.figure(figsize=(8, 6))

    for rn in rule_names:
        x, y = single_rule_data(rn)
        # Plot the main line
        t = rule_plot_names[rn]

        line, = plt.plot(x, y, linewidth=2, alpha=0.7, label=t, linestyle="--")
        color = line.get_color()

        # Add star markers with same color
        plt.plot(x[0], y[0], marker='X', markersize=10, color=color, zorder=10, alpha=0.5, markeredgecolor='black',
                 markeredgewidth=1)
        plt.plot(x[-1], y[-1], marker='*', markersize=10, color=color, zorder=10, alpha=0.9, markeredgecolor='black',
                 markeredgewidth=1)

        # plt.text(x[0], y[0], 'Start', fontsize=12, fontweight='bold', ha='right', va='bottom', zorder=10)
        # plt.text(x[-1], y[-1], 'End', fontsize=12, fontweight='bold', ha='left', va='top', zorder=10)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Set labels and title
    plt.xlabel("Axiom Loss", fontsize=16)
    plt.ylabel("Rule Loss", fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.title(f"{axiom_name} Fine-Tuning", fontsize=18)

    # Update legend with special markers
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
               markersize=10, label='Fine-Tuning Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=14, label='Fine-Tuning End'),
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + legend_elements, labels + ['Fine-Tuning Start', 'Fine-Tuning End'])
    # plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(os.path.join(out_folder, out_filename), dpi=300, bbox_inches='tight')

    # plt.show()


def print_complete_table_all_model_types(axiom, json_key, caption=""):
    df = construct_df_from_final_validations(json_key)

    # filter to only the rows with concatenate features and correct axiom
    df = df[(df["feature"] == "concatenate") & (df["axiom"] == axiom)]
    df = df.drop(columns=["feature", "axiom"])

    df = df.round(3)

    # update rule names
    df["rule_name"] = df["rule_name"].map(rule_plot_names)

    # rename columns according to map
    col_name_map = {
        "rule_name": "",
        "pretrained_rules_eval": "Base Model",
        "finetuned_rules_eval": "Finetuned Model",
        "whalrus": "Rule"
    }
    df.columns = [col_name_map[col] for col in df.columns]

    # invert df
    df_transposed = df.T

    df_transposed.columns = df_transposed.iloc[0]  # Set first row as columns
    df_transposed = df_transposed[1:]  # Remove the first row
    df_transposed = df_transposed.sort_index()

    if json_key == "rule_loss":
        df_transposed.loc['Rule', 'IRV'] = 0.0  # correct library issue with IRV

    latex_output = df_transposed.to_latex(index=True,
                                          float_format="%.3f",
                                          position="htbp",
                                          column_format="lcccccccc",
                                          caption=caption,
                                          label=f"tab:{axiom}-{json_key}")
    lines = latex_output.split('\n')
    lines.insert(-2, lines.pop(1))  # move caption to bottom
    lines.insert(-2, lines.pop(1))  # move label to bottom
    latex_output = '\n'.join(lines)
    latex_output= latex_output.replace("table", "table*")
    print(latex_output)


def plot_axiom_scratch_train_losses(directory):
    """Plot training losses from subfolders matching the pattern."""

    series_names = {
        "Participation+TopCycle": "Part. + TC",
        "AsspLoss+TopCycle": "ASSP + TC",
        "AllAxiomsLoss": "All Axioms"
    }

    directory = Path(directory)

    plt.figure(figsize=(5,3))

    for subfolder in directory.iterdir():
        if not subfolder.is_dir():
            continue

        # Extract series name from subfolder
        match = re.search(r'11c_44v_10000p_impartial_culture_(.+?)_all_axioms', subfolder.name)
        if not match:
            continue

        series_name = match.group(1)
        print(series_name)

        # Load the data
        csv_path = f'{subfolder}/losses/none_concatenate.csv'
        # if not csv_path.exists():
        #     continue

        sname = series_names[series_name]

        df = pd.read_csv(csv_path)
        losses = df['train_losses'][:100]
        plt.plot(losses, label=sname, linewidth=3)

    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().grid(axis='y', alpha=0.3)

    plt.xlabel('Epoch', size=14)
    plt.ylabel('Train Loss', size=14)
    plt.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig("IJCAI-2026/plots_and_figures/axioms_from_scratch-loss_curves", dpi=300)


if __name__ == "__main__":

    ##########################
    # IJCAI - Figure 2
    plot_validation_losses(
        path="IJCAI-2026/training_final_results/base_models/losses",
        rules=rule_names,
        features=features,
    )
    # # End of Figure 2
    # #####################

    #####################
    # IJCAI - Table 2
    # augment_final_validations()     # Need only be run once. Not needed for provided data.
    df_acc = construct_df_from_final_validations(json_key="rule_accuracy_any_tied_winner")
    print_table_from_df(df_acc, axioms=["MonotonicityLoss"])
    # End of Table 2
    #####################

    ##########################
    # IJCAI - Figure 3
    plot_axiom_finetune_results_all_rules_all_axioms(
        in_folders=[
            "IJCAI-2026/training_final_results/fine_tune/11c_44v_10000p_impartial_culture_ParticipationLoss_participation_efficient/losses",
            "IJCAI-2026/training_final_results/fine_tune/11c_44v_10000p_impartial_culture_MontonicityLoss_monotonicity_efficient/losses",
            "IJCAI-2026/training_final_results/fine_tune/11c_44v_10000p_impartial_culture_AsspLoss_assp_efficient/losses",
            "IJCAI-2026/training_final_results/fine_tune/11c_44v_10000p_impartial_culture_TopCycleLoss_topcycle/losses",
        ],
        rule_train_folder="IJCAI-2026/training_final_results/base_models/losses",
        rule_names=list(rule_plot_names.keys()),
        out_folder="IJCAI-2026/plots_and_figures/axiom_finetuning",
        out_filename=f"all_axioms-finetuning.png",
        axiom_names=[
            "Participation",
            "Monotonicity",
            "Adjacent Swap Strategy-Proofness",
            "Top Cycle",
        ],
        mean_window=3,
        # xlim=(-0.001, 5.1),
        # ylim=(0.01, 0.17)
    )
    # End Figure 3
    ##########################

    ##########################
    # IJCAI - Figure 4
    df_acc = construct_df_from_final_validations(json_key="rule_accuracy_any_tied_winner")
    df_loss = construct_df_from_final_validations(json_key="axiom_loss")

    finetune_changes_plot(df_loss=df_loss, df_acc=df_acc)
    # End Figure 5
    ##########################


    ####################
    # IJCAI - Appendix Finetune Results
    # print tables for appendix for each axiom and evaluation metric
    for axiom, metric in itertools.product(axiom_plot_shortnames.keys(), ["axiom_loss", "axiom_violation_rate", "rule_loss"]):
        print_metrics = {
            "axiom_loss": "Axiom Loss",
            "axiom_violation_rate": "Axiom Violation Rate",
            "rule_loss": "Rule Loss",
        }
        print_metric = print_metrics[metric]
        caption = f"\\underline{{{print_metric}}} of all trained models and actual rules on \\underline{{{axiom_plot_names[axiom]}}} axiom."
        print_complete_table_all_model_types(axiom=axiom,
                                             json_key=metric,
                                             caption=caption)
        print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    ####################

    ######################
    # IJCAI - Appendix Train from scratch results
    plot_axiom_scratch_train_losses(directory="IJCAI-2026/training_final_results/axioms_scratch")
    ######################
