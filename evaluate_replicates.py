""" TABLE 3 EVALUATION / AGGREGATION

Evaluates every trained rule model against the tied-profile test set and aggregates mean +/- std across
replicates, reproducing Table 3.

Metric is the tie-mass accuracy that produced the published table: for a profile whose true label is a
tie, the prediction counts as correct when at least `--threshold` of the model's probability mass falls on
the tied winner set; for an untied profile it falls back to plain argmax. See
_compute_rule_tie_mass_batch in models/training.py. Plain argmax and single-winner accuracy are reported
alongside so the choice of metric stays visible.

This calls validate_one_epoch directly rather than shelling out to test_model.py. Same metric code, but it
skips the per-instance CSVs (160 evaluations x 10k profiles would be well over a gigabyte of logits) and
sidesteps test_model.py's second-resolution run_id, which collides when evaluations run in parallel.

    Usage:
        # aggregate the 5 replicates
        python evaluate_replicates.py --replicates 5

        # reproduce the published table from the original single run
        python evaluate_replicates.py --reference-only
"""

import argparse
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch

from data import load_pregenerated_data
from models import validate_one_epoch
from utils import get_loss_fn



DEFAULT_BASE = "experiments/reruns/experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v"
DEFAULT_DATA = ("data/with_alternatives/"
                "n_profiles=10000-num_voters=44-m=11-committee_size=1-pref_dist=IC-has_tied_profiles-TRAIN.h5")
NONCAT_CONFIG = "utils/parameters/11c44v_120kp_learn_rules.json"
CONCAT_CONFIG = "utils/parameters/11c44v_120kp_learn_rules_concatenate.json"

# Row order and display names follow the published table.
RULES = [
    ("plurality_pscf", "Plurality"),
    ("borda_pscf", "Borda"),
    ("copeland_pscf", "Copeland"),
    ("schulze_pscf", "Schulze"),
    ("maximin_pscf", "Simpson-Kramer"),
    ("irv_pscf", "IRV"),
    ("black_pscf", "Black's Rule"),
    ("rankedpairs_pscf", "Ranked Pairs"),
]
TRANSFORMS = [
    ("rankfreq", "$T_{RF}$"),
    ("tournament", "$T_{T}$"),
    ("weighted_tournament", "$T_{WT}$"),
    ("concatenate", "$T_{CO}$"),
]

# The published Table 3, for the sanity check against the reference run.
PUBLISHED = {
    "plurality_pscf": (1.0, 0.3101, 0.3132, 1.0),
    "borda_pscf": (0.7321, 0.6921, 0.993, 0.9492),
    "copeland_pscf": (0.6512, 1.0, 0.7061, 1.0),
    "schulze_pscf": (0.6863, 0.8025, 0.7001, 0.8075),
    "maximin_pscf": (0.6531, 0.7926, 0.6753, 0.7979),
    "irv_pscf": (0.3627, 0.3978, 0.3548, 0.3992),
    "black_pscf": (0.8804, 0.7076, 0.972, 0.8805),
    "rankedpairs_pscf": (0.4703, 0.6761, 0.5104, 0.6237),
}


def profile_count(data_path):
    """Number of profiles in the test file. Sizing the loaders off this avoids silently evaluating a slice."""
    with h5py.File(data_path, "r") as handle:
        for key in handle:
            item = handle[key]
            if isinstance(item, h5py.Dataset):
                return len(item)
    raise ValueError(f"No datasets found in {data_path}")


def build_eval_params(config_path, rule, transform, data_path, n_profiles, thresholds):
    """Eval params from the config that trained this model, so the data keys line up automatically."""
    with open(config_path) as handle:
        params = json.load(handle)

    params.update({
        "rule_name": rule,
        "transform_name": transform,
        "concatenate": transform == "concatenate",
        "data_filename": data_path,
        "n_profiles": n_profiles,
        "dataset_profile_count": n_profiles,
        "batch_size": 500,
        "shuffle": False,
        "num_workers": 0,
        "pref_dist": "impartial_culture",
        # Turns on tie-mass tracking without materialising per-instance records.
        "track_rule_tie_metrics": True,
        "tie_distribution_thresholds": tuple(thresholds),
        # Pulls in the single-winner flag without loading Profile/topk_swaps/profitable_abstentions,
        # which are pickled per-row objects and would dominate runtime.
        "extra_data_datasets": ["is_single_winner_across_rules"],
    })
    return params


def evaluate_model(model_path, rule, transform, data_path, n_profiles, thresholds, device, configs):
    """Run one model over the whole test file. Returns the metric row."""
    config_path = configs["concat"] if transform == "concatenate" else configs["noncat"]
    params = build_eval_params(config_path, rule, transform, data_path, n_profiles, thresholds)

    train_loader, valid_loader, test_loader = load_pregenerated_data(
        params, proportions=(0.0, 0.0), load_profile=False
    )
    loader = test_loader or valid_loader or train_loader
    if loader is None:
        raise RuntimeError(f"No evaluation data loaded from {data_path}")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    loss_fn = get_loss_fn(params["loss_fn"])
    avg_loss, _batch_losses, metrics = validate_one_epoch(
        loader, rule, transform, device, model, loss_fn, params, instance_records=None
    )

    violations = metrics["rule_violations"]
    row = {
        "rule": rule,
        "transform": transform,
        "loss": avg_loss,
        "argmax_accuracy": violations["argmax"]["accuracy"],
        "any_winner_accuracy": violations["argmax_any_winner"]["accuracy"],
        "winner_sum_accuracy": violations["winner_sum"]["accuracy"],
        "single_winner_accuracy": violations["single_winner_across_rules"]["accuracy"],
        # Recorded per checkpoint, unlike metadata.json which all 32 jobs of a replicate overwrite.
        "model_replicate_tag": getattr(model, "info", {}).get("replicate"),
    }
    for threshold, stats in violations["tie_mass"]["thresholds"].items():
        row[f"tie_mass_acc@{threshold:.2f}"] = 1.0 - stats["rate"]
    return row


def discover_runs(base, replicates, reference_only):
    """Return [(label, directory)]. Numeric labels are replicates; 'reference' is the original single run."""
    runs = []
    if not reference_only:
        for n in range(1, replicates + 1):
            directory = f"{base}_rep{n}"
            if os.path.isdir(os.path.join(directory, "models")):
                runs.append((str(n), directory))
            else:
                print(f"[warn] missing replicate directory, skipping: {directory}")
    if reference_only or not runs:
        if os.path.isdir(os.path.join(base, "models")):
            runs.append(("reference", base))
    return runs


def epochs_completed(run_dir, rule, transform):
    """Epochs finished, from the row count of the losses CSV (one row per completed epoch).

    A .pt file existing does NOT mean training finished: save_experiment is called every save_interval
    epochs (models/training.py), so a checkpoint appears from epoch 100 onward and is overwritten as
    training proceeds. The losses CSV is the only per-model progress signal -- metadata.json is shared and
    clobbered by all 32 jobs in a replicate.
    """
    path = os.path.join(run_dir, "losses", f"{rule}_{transform}.csv")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return max(sum(1 for _ in handle) - 1, 0)  # discount the header
    except OSError:
        return None


def completeness_report(base, expected_replicates, expected_epochs):
    """Inventory what exists against what a full sweep should contain.

    Returns (incomplete, missing_models, missing_dirs) where incomplete is the set of
    (run_label, rule, transform) whose training has not reached expected_epochs.
    """
    per_run = len(RULES) * len(TRANSFORMS)
    total = expected_replicates * per_run
    print(f"\n=== Completeness: expecting {expected_replicates} replicates x {per_run} models = {total} ===")

    missing_dirs, missing_models, incomplete = [], [], []
    for n in range(1, expected_replicates + 1):
        directory = f"{base}_rep{n}"
        if not os.path.isdir(os.path.join(directory, "models")):
            missing_dirs.append(n)
            print(f"  rep{n}: MISSING directory")
            continue
        present = finished = 0
        for rule, _display in RULES:
            for transform, _col in TRANSFORMS:
                if not os.path.exists(os.path.join(directory, "models", f"{rule}_{transform}.pt")):
                    missing_models.append((n, rule, transform))
                    continue
                present += 1
                done = epochs_completed(directory, rule, transform)
                if expected_epochs and done is not None and done < expected_epochs:
                    incomplete.append((n, rule, transform, done))
                else:
                    finished += 1
        flag = "" if present == per_run and finished == per_run else "   <-- incomplete"
        print(f"  rep{n}: {present}/{per_run} models present, {finished} finished training{flag}")

    def listing(rows, header, formatter, limit=15):
        if not rows:
            return
        print(f"\n{header} ({len(rows)}):")
        for row in rows[:limit]:
            print(f"    {formatter(row)}")
        if len(rows) > limit:
            print(f"    ... and {len(rows) - limit} more")

    listing(missing_models, "Missing models", lambda r: f"rep{r[0]}  {r[1]}_{r[2]}")
    listing(incomplete, f"Still training (below {expected_epochs} epochs)",
            lambda r: f"rep{r[0]}  {r[1] + '_' + r[2]:<38} {r[3]}/{expected_epochs} epochs")

    accounted = total - len(missing_dirs) * per_run - len(missing_models)
    print(f"\n  {accounted}/{total} models present, {accounted - len(incomplete)} of those fully trained.")
    return {(str(r[0]), r[1], r[2]) for r in incomplete}, missing_models, missing_dirs


def collect(runs, data_path, thresholds, device, configs, skip=frozenset()):
    n_profiles = profile_count(data_path)
    print(f"[setup] Test set: {data_path} ({n_profiles} profiles)")
    print(f"[setup] Device: {device}\n")

    records = []
    for label, directory in runs:
        for rule, _display in RULES:
            for transform, _col in TRANSFORMS:
                model_path = os.path.join(directory, "models", f"{rule}_{transform}.pt")
                if not os.path.exists(model_path):
                    print(f"[warn] missing model, skipping: {model_path}")
                    continue
                if (label, rule, transform) in skip:
                    print(f"[skip] run={label:<9} {rule:<18} {transform:<20} still training")
                    continue
                row = evaluate_model(model_path, rule, transform, data_path,
                                     n_profiles, thresholds, device, configs)
                row["run"] = label
                row["epochs_completed"] = epochs_completed(directory, rule, transform)
                records.append(row)
                print(f"[eval] run={label:<9} {rule:<18} {transform:<20} "
                      f"tie_mass@{thresholds[0]:.2f}={row[f'tie_mass_acc@{thresholds[0]:.2f}']:.4f}")
    return pd.DataFrame(records)


def aggregate(per_model, metric_column):
    """Mean/std per (rule, transform) over replicate runs only; the reference run is kept separate."""
    replicate_rows = per_model[per_model["run"] != "reference"]
    if replicate_rows.empty:
        replicate_rows = per_model

    grouped = replicate_rows.groupby(["rule", "transform"])[metric_column]
    # Sample std/variance (ddof=1) -- these are 5 draws from the training procedure, not a population.
    summary = grouped.agg(["mean", "std", "var", "min", "max", "count"]).reset_index()
    # std/var are NaN for a single observation; report 0 spread rather than a blank cell.
    summary["std"] = summary["std"].fillna(0.0)
    summary["var"] = summary["var"].fillna(0.0)
    summary["range"] = summary["max"] - summary["min"]
    # Standard error of the mean: sample std / sqrt(n). Zero when there is a single run.
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    published = []
    for _, row in summary.iterrows():
        idx = [t for t, _ in TRANSFORMS].index(row["transform"])
        published.append(PUBLISHED[row["rule"]][idx])
    summary["published"] = published
    summary["delta_vs_published"] = summary["mean"] - summary["published"]
    return summary


def format_cell(mean, std, n):
    return f"{mean:.4f}" if n < 2 else f"{mean:.4f} $\\pm$ {std:.4f}"


# Cells shaded in the published Table 3: the embedding carries enough information to learn the rule perfectly.
# This is a property of the embedding, not of the measured accuracy, so it is fixed rather than derived.
SHADED = {
    ("plurality_pscf", "rankfreq"), ("plurality_pscf", "concatenate"),
    ("borda_pscf", "rankfreq"), ("borda_pscf", "weighted_tournament"), ("borda_pscf", "concatenate"),
    ("copeland_pscf", "tournament"), ("copeland_pscf", "weighted_tournament"), ("copeland_pscf", "concatenate"),
    ("schulze_pscf", "weighted_tournament"), ("schulze_pscf", "concatenate"),
    ("maximin_pscf", "weighted_tournament"), ("maximin_pscf", "concatenate"),
    ("black_pscf", "weighted_tournament"), ("black_pscf", "concatenate"),
    ("rankedpairs_pscf", "weighted_tournament"), ("rankedpairs_pscf", "concatenate"),
}


ERROR_NAMES = {"std": "standard deviation", "sem": "standard error of the mean"}


def render_latex(summary, metric_column, threshold, n_runs, error="std", metric="tie_mass"):
    lookup = {(r["rule"], r["transform"]): r for _, r in summary.iterrows()}
    lo, hi = int(summary["count"].min()), int(summary["count"].max())

    lines = [
        "\\begin{table}[tbp]",
        "\\centering",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "\\textbf{}               & " + " & ".join(col for _, col in TRANSFORMS) + " \\\\ \\midrule",
    ]

    for rule, display in RULES:
        cells, means = [], []
        for transform, _col in TRANSFORMS:
            row = lookup.get((rule, transform))
            if row is None:
                cells.append("--")
                means.append(-1.0)
                continue
            cells.append(format_cell(row["mean"], row[error], row["count"]))
            means.append(row["mean"])
        # Bold every joint maximum, not just the first -- rules like Plurality tie at 1.0 across embeddings.
        if max(means) >= 0:
            best = max(means)
            cells = [f"\\textbf{{{c}}}" if m >= best - 1e-9 else c for c, m in zip(cells, means)]
        cells = [f"\\cellcolor{{green!25}}{c}" if (rule, transform) in SHADED else c
                 for c, (transform, _col) in zip(cells, TRANSFORMS)]
        lines.append(f"\\textbf{{{display}}} & " + " & ".join(cells) + " \\\\")

    lines[-1] += " \\bottomrule"
    # Never claim a uniform n when cells were aggregated over different numbers of replicates.
    if hi < 2:
        spread = ""
    elif lo == hi:
        spread = f" Cells show mean $\\pm$ {ERROR_NAMES[error]} over {hi} independently trained models."
    else:
        spread = (f" Cells show mean $\\pm$ {ERROR_NAMES[error]} over {lo}--{hi} independently trained "
                  f"models; per-cell counts differ and are listed in table3\\_summary.csv.")
    if metric == "winner_sum":
        definition = [
            "$(m=11, n=44)$, on profiles that may contain ties. A prediction counts as correct when the total",
            "predicted probability on the rule's winners exceeds the predicted probability of every other",
            f"individual candidate; for profiles with a single winner this reduces to picking that winner.{spread}",
        ]
    elif metric == "any_winner":
        definition = [
            "$(m=11, n=44)$, on profiles that may contain ties. A prediction counts as correct when the",
            "candidate with the highest predicted probability is one of the rule's winners (any member of the",
            f"tied winner set when the rule produces a tie).{spread}",
        ]
    else:
        definition = [
            "$(m=11, n=44)$, on profiles that may contain ties. A prediction counts as correct when at least",
            # '%' must be escaped or LaTeX treats the rest of the line as a comment.
            f"{threshold * 100:.0f}\\% of the predicted probability mass falls on the tied winner set; for profiles with a",
            f"single winner this reduces to picking that winner.{spread}",
        ]
    lines += [
        "\\end{tabular}",
        "\\caption{",
        "Accuracy of our models on each embedding using test data sampled from the Impartial Culture using",
        *definition,
        "Most accurate embeddings are shown in bold. Shaded cells indicate the embedding contains sufficient",
        "information to learn the rule perfectly.",
        "}",
        "\\label{tab:test_acc_all_rules_IC}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate replicate models and rebuild Table 3.")
    parser.add_argument("--experiment-base", default=DEFAULT_BASE,
                        help="Experiment directory without the _rep<n> suffix.")
    parser.add_argument("--replicates", type=int, default=5, help="How many replicate directories to look for.")
    parser.add_argument("--reference-only", action="store_true",
                        help="Evaluate the unsuffixed reference run instead of the replicates.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Test set to evaluate against.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Tie-mass threshold for the headline metric.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.80, 0.90, 0.95, 0.99],
                        help="All tie-mass thresholds to record.")
    parser.add_argument("--noncat-config", default=NONCAT_CONFIG)
    parser.add_argument("--concat-config", default=CONCAT_CONFIG)
    parser.add_argument("--out-dir", default="experiments/table3", help="Where to write CSVs and the LaTeX table.")
    parser.add_argument("--device", default=None, help="Force device. Defaults to auto-detection.")
    parser.add_argument("--error", choices=["std", "sem"], default="std",
                        help="Spread printed after +/- in the table: sample std (default) or standard error of the "
                             "mean (std / sqrt(n)). 'sem' writes table3_sem.tex so the std table is kept.")
    parser.add_argument("--metric", choices=["tie_mass", "any_winner", "winner_sum"], default="tie_mass",
                        help="Headline metric: tie-mass at --threshold (published), or argmax counted correct when it "
                             "is any of the tied winners, or winner_sum: summed winner mass beats every single "
                             "non-winner output. Both are threshold-free.")
    parser.add_argument("--rerender", action="store_true",
                        help="Skip evaluation: re-aggregate --out-dir/per_model_metrics.csv and rewrite the table.")
    parser.add_argument("--expected-epochs", type=int, default=None,
                        help="Epochs a finished model should have. Defaults to n_epochs from the config.")
    parser.add_argument("--skip-incomplete", action="store_true",
                        help="Exclude models that have not reached expected-epochs from the aggregate.")
    return parser.parse_args()


def evaluate_runs(args, thresholds, device):
    """Evaluate every replicate model on the test set; one row per model."""
    expected_epochs = args.expected_epochs
    if expected_epochs is None:
        with open(args.noncat_config) as handle:
            expected_epochs = json.load(handle).get("n_epochs")

    incomplete = set()
    if not args.reference_only:
        incomplete, _missing_models, _missing_dirs = completeness_report(
            args.experiment_base, args.replicates, expected_epochs
        )
        if incomplete and not args.skip_incomplete:
            print(f"\n  !! {len(incomplete)} model(s) are still training but will be INCLUDED in the "
                  f"aggregate. Pass --skip-incomplete to exclude them.")

    runs = discover_runs(args.experiment_base, args.replicates, args.reference_only)
    if not runs:
        print(f"No model directories found under '{args.experiment_base}'.")
        raise SystemExit(1)
    print(f"\n[setup] Evaluating {len(runs)} run(s): {', '.join(label for label, _ in runs)}")

    configs = {"noncat": args.noncat_config, "concat": args.concat_config}
    per_model = collect(runs, args.data, thresholds, device, configs,
                        skip=incomplete if args.skip_incomplete else frozenset())
    if per_model.empty:
        print("No models were evaluated.")
        raise SystemExit(1)
    return per_model


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    thresholds = list(args.thresholds)
    if args.threshold not in thresholds:
        thresholds.append(args.threshold)
    thresholds = sorted(thresholds)
    metric_column = {"any_winner": "any_winner_accuracy", "winner_sum": "winner_sum_accuracy"}.get(
        args.metric, f"tie_mass_acc@{args.threshold:.2f}")

    if args.rerender:
        per_model = pd.read_csv(os.path.join(args.out_dir, "per_model_metrics.csv"))
        print(f"[rerender] Re-aggregating {len(per_model)} saved per-model rows; no models evaluated.")
    else:
        per_model = evaluate_runs(args, thresholds, device)

    summary = aggregate(per_model, metric_column)
    n_runs = int(summary["count"].max())

    os.makedirs(args.out_dir, exist_ok=True)
    per_model_path = os.path.join(args.out_dir, "per_model_metrics.csv")
    summary_path = os.path.join(args.out_dir, "table3_summary.csv")
    latex_path = os.path.join(args.out_dir, "table3_sem.tex" if args.error == "sem" else "table3.tex")

    if not args.rerender:
        per_model.to_csv(per_model_path, index=False)
    summary.to_csv(summary_path, index=False)
    latex = render_latex(summary, metric_column, args.threshold, n_runs, args.error, args.metric)
    with open(latex_path, "w") as handle:
        handle.write(latex + "\n")

    print(f"\n[save] Per-model metrics : {per_model_path}")
    print(f"[save] Aggregated summary: {summary_path}")
    print(f"[save] LaTeX table       : {latex_path}")

    worst = summary.reindex(summary["delta_vs_published"].abs().sort_values(ascending=False).index).head(5)
    print(f"\nLargest deviations from the published table (metric: {metric_column}):")
    for _, row in worst.iterrows():
        print(f"    {row['rule']:<18} {row['transform']:<20} "
              f"got {row['mean']:.4f}  published {row['published']:.4f}  delta {row['delta_vs_published']:+.4f}")
    print(f"\nMean |deviation| vs published: {summary['delta_vs_published'].abs().mean():.4f}")

    if n_runs < 2:
        print("Single run: no spread to report. Cells are point estimates.")
    else:
        noisiest = summary.reindex(summary["var"].sort_values(ascending=False).index).head(5)
        print(f"\nVariance across replicates (sample, ddof=1) — noisiest cells:")
        for _, row in noisiest.iterrows():
            print(f"    {row['rule']:<18} {row['transform']:<20} mean {row['mean']:.4f}  "
                  f"std {row['std']:.4f}  var {row['var']:.2e}  range {row['range']:.4f}  n={int(row['count'])}")
        print(f"\n  Mean variance across cells : {summary['var'].mean():.2e}")
        print(f"  Mean std across cells      : {summary['std'].mean():.4f}")
        print(f"  Largest single-cell range  : {summary['range'].max():.4f}")

    under = summary[summary["count"] < args.replicates]
    if not under.empty and not args.reference_only:
        print(f"\n  !! {len(under)}/{len(summary)} cells aggregate over fewer than {args.replicates} "
              f"replicates (counts present: {sorted(summary['count'].unique().tolist())}).")
        print("     Cells with differing n are not directly comparable; see the count column.")

    print("\n" + latex)


if __name__ == "__main__":
    main()
