""" PAPER TABLES 6-10 AND FIGURE 2 FROM THE 5 RULE REPLICATES

CPU-only. Everything comes from checkpoints and loss CSVs already on disk plus the local test files in
data/pregenerated/; nothing here needs the cluster or the tied 10k file.

    Table 6    best-epoch (minimum) valid_losses from each replicate's losses/{rule}_{transform}.csv; this is the
               epoch whose weights are saved. --table6-epoch final reproduces the published (final-epoch) method.
    Table 7    L1 loss on 512 IC profiles with 199 voters (networks trained on 44 voters), all 4 embeddings
    Table 8    same with 13 voters
    Table 9    L1 loss of the CO models on the 44-voter, 10k-profile file of each preference distribution
    Table 10   argmax accuracy from the same evaluation as Table 9
    Figure 2   validation loss per epoch, mean across replicates (formatting of the published figure);
               --fig2-replicate N plots one replicate only, --fig2-variance adds a +/- std band

Metrics reproduce how the published numbers were computed (see paper_variance_reruns.md): batch size 32,
the mean of per-batch L1 losses, and for Table 10 the mean of per-batch argmax accuracies. The 13v, 199v and
pref-dist files carry only the raw rankfreq / weighted_tournament keys, which are also what the replicates
were trained on (the 120k training file never loads the '-normalized' keys).

    Usage:
        python paper_tables_replicates.py                   # everything
        python paper_tables_replicates.py --only 6 fig2     # no model evaluation
        python paper_tables_replicates.py --only 9 10       # OOD tables only
        python paper_tables_replicates.py --only 6 --table6-epoch final   # old final-epoch Table 6
        python paper_tables_replicates.py --only fig2 --fig2-replicate 3  # Figure 2 from replicate 3 alone
        python paper_tables_replicates.py --only fig2 --fig2-variance     # mean with +/- std band

Outputs (default --out-dir experiments/paper_tables/):
    per_model_metrics.csv   one row per (table, rule, embedding or distribution, replicate)
    summary.csv             mean / std / min / max / count per cell, with the published value and delta
    table{6,7,8,9,10}.tex   LaTeX tables, mean +/- std
    plots_and_figures/valid_losses-all_rules-replicates.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import load_pregenerated_data
from evaluate_replicates import CONCAT_CONFIG, DEFAULT_BASE, NONCAT_CONFIG, RULES, TRANSFORMS, profile_count
# Copied from plots_and_figures/make_aaai_plots.py, which was removed from the repo in a7e4b78.
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
transform_names = {
    "rankfreq": "Rank Frequency",
    "tournament": "Tournament",
    "weighted_tournament": "Weighted Tournament",
    "concatenate": "Concatenation",
}
transform_colours = {
    "rankfreq": "tab:green",
    "tournament": "tab:orange",
    "weighted_tournament": "tab:red",
    "concatenate": "tab:blue",
}
from utils import CONCAT_TRANSFORM_ORDER, get_concatenated_data, get_loss_fn, load_job_parameters


VOTER_FILES = {
    "7": (199, "data/pregenerated/11c_199v_512p_impartial_culture.h5"),
    "8": (13, "data/pregenerated/11c_13v_512p_impartial_culture.h5"),
}

# Column order of published Tables 9/10
DISTS = [
    ("impartial_culture", "IC"),
    ("impartial_anonymous_culture", "IAC"),
    ("urn", "Urn"),
    ("mallows", "Mall."),
    ("sp_walsh", "SP"),
    ("preflib", "PrefL"),
]
OOD_FILE = "data/pregenerated/11c_44v_10000p_{dist}.h5"

# Published values, row order of RULES. Tables 6-8 columns follow TRANSFORMS; Tables 9/10 follow DISTS.
PUBLISHED = {
    "6": [(0.0, 0.132, 0.131, 0.0), (0.051, 0.048, 0.004, 0.01), (0.07, 0.0, 0.059, 0.0),
          (0.067, 0.042, 0.064, 0.041), (0.074, 0.043, 0.07, 0.042), (0.11, 0.094, 0.106, 0.098),
          (0.025, 0.043, 0.009, 0.021), (0.088, 0.049, 0.081, 0.061)],
    "7": [(0.04, 0.133, 0.142, 0.058), (0.165, 0.059, 0.039, 0.057), (0.095, 0.0, 0.159, 0.001),
          (0.08, 0.057, 0.082, 0.057), (0.083, 0.057, 0.078, 0.057), (0.166, 0.112, 0.166, 0.165),
          (0.061, 0.054, 0.166, 0.045), (0.164, 0.059, 0.164, 0.078)],
    "8": [(0.023, 0.119, 0.131, 0.019), (0.166, 0.051, 0.024, 0.047), (0.095, 0.001, 0.159, 0.0),
          (0.083, 0.05, 0.082, 0.049), (0.079, 0.049, 0.076, 0.05), (0.158, 0.108, 0.158, 0.16),
          (0.055, 0.042, 0.166, 0.037), (0.165, 0.055, 0.165, 0.072)],
    "9": [(0.0, 0.0, 0.006, 0.003, 0.006, 0.004), (0.005, 0.005, 0.078, 0.095, 0.177, 0.110),
          (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.017, 0.017, 0.026, 0.025, 0.027, 0.036),
          (0.018, 0.017, 0.026, 0.025, 0.024, 0.033), (0.090, 0.089, 0.076, 0.069, 0.076, 0.069),
          (0.014, 0.015, 0.086, 0.096, 0.034, 0.111), (0.047, 0.047, 0.042, 0.038, 0.169, 0.043)],
    "10": [(1.0, 1.0, 0.990, 0.997, 1.0, 0.978), (0.974, 0.972, 0.574, 0.477, 0.028, 0.396),
           (1.0, 1.0, 1.0, 1.0, 1.0, 0.988), (0.910, 0.915, 0.871, 0.884, 0.852, 0.818),
           (0.908, 0.912, 0.876, 0.890, 0.903, 0.832), (0.505, 0.508, 0.579, 0.622, 0.580, 0.621),
           (0.925, 0.918, 0.528, 0.472, 0.820, 0.391), (0.742, 0.739, 0.769, 0.793, 0.069, 0.765)],
}

CAPTIONS = {
    "6": ("L1 loss on the validation split of the training data (Impartial Culture, $(m=11, n=44)$) at the "
          "epoch with the lowest validation loss, for models targeting each rule using each embedding."),
    "7": ("L1 loss on 512 Impartial Culture profiles with 11 candidates and 199 voters, for networks trained on "
          "each embedding with 11 candidates and 44 voters."),
    "8": ("L1 loss on 512 Impartial Culture profiles with 11 candidates and 13 voters, for networks trained on "
          "each embedding with 11 candidates and 44 voters."),
    "9": ("Test loss on 10,000 profiles $(m=11, n=44)$ from each preference distribution, for networks trained "
          "using Impartial Culture preferences and the $T_{CO}$ embedding."),
    "10": ("Test accuracy on 10,000 profiles $(m=11, n=44)$ from each preference distribution, for networks "
           "trained using Impartial Culture preferences and the $T_{CO}$ embedding. A prediction is correct when "
           "the highest-scoring candidate is the winner; these test sets contain no tied winners."),
}
LABELS = {
    "6": "tab:test_loss_all_rules_IC",
    "7": "tab:rule_loss_199v11c",
    "8": "tab:rule_loss_13v11c",
    "9": "tab:out_of_dist_test_loss",
    "10": "tab:out_of_dist_test_acc",
}

# Line styles back up the published colours so embeddings are not told apart by colour alone
# Panel order of the published Figure 2
FIG2_RULES = ["plurality_pscf", "borda_pscf", "copeland_pscf", "schulze_pscf",
              "maximin_pscf", "irv_pscf", "black_pscf", "rankedpairs_pscf"]

L1 = get_loss_fn("L1Loss")


def run_dir(base, replicate):
    return f"{base}_rep{replicate}"


def model_path(base, replicate, rule, transform):
    return f"{run_dir(base, replicate)}/models/{rule}_{transform}.pt"


def build_loader(rule, transform, data_path, n_voters):
    """Batch-32, unshuffled loader over the whole file, keyed the way the training config keys it."""
    config = CONCAT_CONFIG if transform == "concatenate" else NONCAT_CONFIG
    params = load_job_parameters(rule, transform, config)
    n_profiles = profile_count(data_path)
    params.update({
        "data_filename": data_path,
        "n_voters": n_voters,
        "n_profiles": n_profiles,
        "dataset_profile_count": n_profiles,
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 0,
        "pref_dist": "impartial_culture",
    })
    # Profiles are not needed (and preflib profiles vary in length), only the transforms and labels.
    _, _, loader = load_pregenerated_data(params, proportions=(0, 0), load_profile=False)
    return loader


def evaluate(model, loader, rule, transform):
    """Mean of per-batch L1 loss and mean of per-batch argmax accuracy, as the published tables computed them."""
    losses, accuracies = [], []
    with torch.no_grad():
        for data, labels in loader:
            if transform == "concatenate":
                X = get_concatenated_data(data, transforms=CONCAT_TRANSFORM_ORDER)
            else:
                X = data[transform]
            Y = labels[rule].float()
            logits = model(X.float())
            losses.append(L1(logits, Y).item())
            accuracies.append((logits.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accuracies))


def load_model(path, device):
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def eval_cell(base, replicates, rule, transform, data_path, n_voters, device):
    """Evaluate every replicate of one (rule, transform) on one file, building the loader once."""
    paths = {rep: model_path(base, rep, rule, transform) for rep in replicates}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    for p in missing:
        print(f"[warn] missing model, skipped: {p}")
    if len(missing) == len(paths):
        return []

    loader = build_loader(rule, transform, data_path, n_voters)
    rows = []
    for rep, path in paths.items():
        if path in missing:
            continue
        loss, accuracy = evaluate(load_model(path, device), loader, rule, transform)
        rows.append({"rep": rep, "loss": loss, "argmax_accuracy": accuracy})
    return rows


def table6_rows(base, replicates, epoch="best"):
    """Validation L1 loss per model. 'best' is the minimum over epochs, which is the epoch whose weights
    training saves (models/training.py keeps the best-validation state). 'final' is the last epoch, as in the
    published table; with LR restarts the last epoch can sit inside a post-restart loss spike."""
    rows = []
    for rule, _ in RULES:
        for transform, _ in TRANSFORMS:
            for rep in replicates:
                path = f"{run_dir(base, rep)}/losses/{rule}_{transform}.csv"
                if not os.path.exists(path):
                    print(f"[warn] missing loss CSV, skipped: {path}")
                    continue
                valid = pd.read_csv(path)["valid_losses"]
                value = valid.min() if epoch == "best" else valid.iloc[-1]
                rows.append({"table": "6", "rule": rule, "column": transform, "rep": rep, "value": value})
    return rows


def voter_table_rows(table, base, replicates, device):
    n_voters, data_path = VOTER_FILES[table]
    rows = []
    for rule, _ in RULES:
        for transform, _ in TRANSFORMS:
            print(f"[table {table}] {rule} {transform} on {n_voters} voters")
            for r in eval_cell(base, replicates, rule, transform, data_path, n_voters, device):
                rows.append({"table": table, "rule": rule, "column": transform, "rep": r["rep"], "value": r["loss"],
                             "test_file": data_path})
    return rows


def ood_table_rows(base, replicates, device):
    """Tables 9 (loss) and 10 (accuracy) come from one pass over the CO models."""
    rows = []
    for rule, _ in RULES:
        for dist, _ in DISTS:
            data_path = OOD_FILE.format(dist=dist)
            print(f"[tables 9/10] {rule} on {dist}")
            for r in eval_cell(base, replicates, rule, "concatenate", data_path, 44, device):
                rows.append({"table": "9", "rule": rule, "column": dist, "rep": r["rep"], "value": r["loss"],
                             "test_file": data_path})
                rows.append({"table": "10", "rule": rule, "column": dist, "rep": r["rep"],
                             "value": r["argmax_accuracy"], "test_file": data_path})
    return rows


def columns_for(table):
    return DISTS if table in ("9", "10") else TRANSFORMS


def summarize(per_model):
    # Sample std (ddof=1): 5 draws from the training procedure, not a population. Matches evaluate_replicates.py.
    summary = per_model.groupby(["table", "rule", "column"])["value"].agg(
        ["mean", "std", "min", "max", "count"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    # Standard error of the mean: sample std / sqrt(n). Zero when there is a single replicate.
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    published = []
    for _, row in summary.iterrows():
        r = [name for name, _ in RULES].index(row["rule"])
        c = [name for name, _ in columns_for(row["table"])].index(row["column"])
        published.append(PUBLISHED[row["table"]][r][c])
    summary["published"] = published
    summary["delta_vs_published"] = summary["mean"] - summary["published"]
    return summary


ERROR_NAMES = {"std": "standard deviation", "sem": "standard error of the mean"}


def format_number(value, table):
    text = f"{value:.3f}"
    # Tables 9/10 drop the leading zero, as published
    return text.replace("0.", ".", 1) if table in ("9", "10") and text.startswith("0.") else text


def render_latex(summary, table, error="std"):
    columns = columns_for(table)
    cells = {(r["rule"], r["column"]): r for _, r in summary[summary["table"] == table].iterrows()}
    counts = sorted({int(r["count"]) for r in cells.values()})
    header = "Target Rule" if table != "6" else "\\textbf{}"

    lines = [
        "\\begin{table}[tbp]",
        "\\centering",
        "\\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}",
        "\\toprule",
        f"{header} & " + " & ".join(display for _, display in columns) + " \\\\ \\midrule",
    ]
    for rule, display in RULES:
        row = []
        for column, _ in columns:
            cell = cells.get((rule, column))
            if cell is None:
                row.append("--")
            elif cell["count"] < 2:
                row.append(format_number(cell["mean"], table))
            else:
                row.append(f"{format_number(cell['mean'], table)} $\\pm$ {format_number(cell[error], table)}")
        lines.append(f"\\textbf{{{display}}} & " + " & ".join(row) + " \\\\")
    lines[-1] += " \\bottomrule"

    error_name = ERROR_NAMES[error]
    if counts and counts[0] == counts[-1] and counts[0] > 1:
        spread = f" Cells show mean $\\pm$ {error_name} over {counts[0]} independently trained models."
    elif counts and counts[-1] > 1:
        spread = (f" Cells show mean $\\pm$ {error_name} over {counts[0]}--{counts[-1]} independently trained "
                  f"models; per-cell counts are in summary.csv.")
    else:
        spread = ""
    lines += [
        "\\end{tabular}",
        "\\caption{",
        CAPTIONS[table] + spread,
        "}",
        f"\\label{{{LABELS[table]}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def plot_figure2(base, replicates, out_path, variance=False):
    """Replicate version of make_ijcai2026_plots.plot_validation_losses: identical formatting, mean over replicates.

    variance=True shades +/- one sample std across replicates (needs at least 2).
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 4.6), sharex=True, sharey=True)
    for ax, rule in zip(axes.flatten(), FIG2_RULES):
        ax.set_title(rule_plot_names[rule], fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        for transform, _ in TRANSFORMS:
            curves = []
            for rep in replicates:
                path = f"{run_dir(base, rep)}/losses/{rule}_{transform}.csv"
                if os.path.exists(path):
                    curves.append(pd.read_csv(path)["valid_losses"].to_numpy())
            if not curves:
                print(f"[warn] no loss CSVs for {rule} {transform}")
                continue
            # Replicates normally all run n_epochs; truncate to the shortest so every point averages all of them
            length = min(len(c) for c in curves)
            stacked = np.stack([c[:length] for c in curves])
            mean = stacked.mean(axis=0)
            epochs = np.arange(length)
            if variance and len(curves) > 1:
                std = stacked.std(axis=0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color=transform_colours[transform], alpha=0.2,
                                linewidth=0)
            ax.plot(epochs, mean, label=transform_names[transform], markersize=2, linewidth=2,
                    color=transform_colours[transform])
        ax.grid(True, alpha=0.3)
    # Averaged curves peak lower than the published single run; pin the published y-range (0.00-0.15 ticks)
    axes[0, 0].set_ylim(-0.0075, 0.1575)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.014), ncol=4, fontsize=12)
    plt.tight_layout()
    fig.supylabel("L1 Loss", fontsize=14, x=0.004, y=0.5)
    fig.supxlabel("Epoch", fontsize=14, x=0.5, y=0.08)
    plt.subplots_adjust(bottom=0.19, left=0.05)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[fig2] wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build paper Tables 6-10 and Figure 2 from the rule replicates.")
    parser.add_argument("--experiment-base", default=DEFAULT_BASE,
                        help="Replicate run prefix; '_rep{n}' is appended.")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--only", nargs="+", choices=["6", "7", "8", "9", "10", "fig2"], default=None,
                        help="Restrict to these outputs (default: all). 9 and 10 share one evaluation.")
    parser.add_argument("--out-dir", default="experiments/paper_tables")
    parser.add_argument("--table6-epoch", choices=["best", "final"], default="best",
                        help="Table 6 value per model: best-epoch (saved model, default) or final-epoch validation loss.")
    parser.add_argument("--figure-path", default="plots_and_figures/valid_losses-all_rules-replicates.png")
    parser.add_argument("--fig2-replicate", type=int, default=None,
                        help="Plot Figure 2 from this single replicate only (1-based) instead of the mean of all.")
    parser.add_argument("--fig2-variance", action="store_true",
                        help="Shade +/- one std across replicates in Figure 2 (ignored with --fig2-replicate).")
    parser.add_argument("--error", choices=["std", "sem"], default="std",
                        help="Spread printed after +/- in the tables: sample std (default) or standard error of the "
                             "mean (std / sqrt(n)). 'sem' writes table{N}_sem.tex so the std tables are kept.")
    parser.add_argument("--rerender", action="store_true",
                        help="Skip evaluation: re-aggregate --out-dir/per_model_metrics.csv and rewrite the tables.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    wanted = set(args.only) if args.only else {"6", "7", "8", "9", "10", "fig2"}
    replicates = list(range(1, args.replicates + 1))
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    if args.rerender:
        saved = pd.read_csv(os.path.join(args.out_dir, "per_model_metrics.csv"), dtype={"table": str})
        rows = saved[saved["table"].isin(wanted)].to_dict("records")
        wanted.discard("fig2")
    else:
        if "6" in wanted:
            rows += table6_rows(args.experiment_base, replicates, args.table6_epoch)
        for table in ("7", "8"):
            if table in wanted:
                rows += voter_table_rows(table, args.experiment_base, replicates, args.device)
        if wanted & {"9", "10"}:
            rows += [r for r in ood_table_rows(args.experiment_base, replicates, args.device)
                     if r["table"] in wanted]

    if rows:
        per_model = pd.DataFrame(rows)
        if not args.rerender:
            per_model.to_csv(os.path.join(args.out_dir, "per_model_metrics.csv"), index=False)
        summary = summarize(per_model)
        summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        suffix = "_sem" if args.error == "sem" else ""
        for table in sorted(set(summary["table"]), key=int):
            path = os.path.join(args.out_dir, f"table{table}{suffix}.tex")
            with open(path, "w") as f:
                f.write(render_latex(summary, table, args.error) + "\n")
            deltas = summary[summary["table"] == table]["delta_vs_published"].abs()
            print(f"[table {table}] wrote {path}   mean |replicate mean - published| = {deltas.mean():.4f}")

    if "fig2" in wanted:
        fig2_replicates, figure_path = replicates, args.figure_path
        if args.fig2_replicate is not None:
            if args.fig2_replicate not in replicates:
                raise SystemExit(f"--fig2-replicate must be in 1..{args.replicates}, got {args.fig2_replicate}")
            fig2_replicates = [args.fig2_replicate]
            if args.fig2_variance:
                print("[fig2] --fig2-variance ignored: a single replicate has no spread")
        stem, ext = os.path.splitext(figure_path)
        if args.fig2_replicate is not None:
            stem += f"-rep{args.fig2_replicate}"
        elif args.fig2_variance:
            stem += "-std"
        plot_figure2(args.experiment_base, fig2_replicates, stem + ext,
                     variance=args.fig2_variance and args.fig2_replicate is None)


if __name__ == "__main__":
    main()
