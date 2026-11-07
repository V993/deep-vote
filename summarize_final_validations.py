"""Summarize final validation metrics into tables and CSV."""

import csv
import json
import sys
from pathlib import Path

# Computed thresholds from validation: [0.8, 0.9, 0.95, 0.99]
TIE_MASS_THRESHOLD = 0.95


def _infer_base_dir():
    for candidate in (Path("final_validations"), Path("experiments/final_validations"), Path("IJCAI-2026/final_validations")):
        if candidate.exists():
            return candidate
    return Path("final_validations")


def _parse_job_name(path: Path):
    name = path.name
    if not name.endswith("_metrics.json"):
        return None
    job_name = name[:-len("_metrics.json")].rsplit("_", 1)[0]
    parts = job_name.split("-")
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2]


def _format_cell(axiom_rate, tie_mass_rate):
    if axiom_rate is None and tie_mass_rate is None:
        return "n/a"
    axiom_str = "n/a" if axiom_rate is None else f"{axiom_rate:.4f}"
    tie_str = "n/a" if tie_mass_rate is None else f"{tie_mass_rate:.4f}"
    return f"{axiom_str}|{tie_str}"


def _render_table(group, losses, rules, grid):
    header = ["axiom|rule violation rate"] + list(rules)
    rows = [[loss] + [grid.get((loss, rule), "n/a") for rule in rules] for loss in losses]

    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def fmt_row(items):
        return " | ".join(str(item).ljust(width) for item, width in zip(items, widths))

    print(f"\n{group}")
    print(fmt_row(header))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt_row(row))


def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _infer_base_dir()
    group_map = {
        "pretrained_rules_eval": "pretrained",
        "finetuned_rules_eval": "finetuned",
        "whalrus": "whalrus",
    }
    groups = list(group_map)
    print("Assessing", groups, "in directory:", f"'{base_dir}/'")
    rendered = False

    rows = []
    for group in groups:
        group_dir = base_dir / group
        if not group_dir.exists():
            continue

        metrics_by_loss_rule = {}
        losses = set()
        rules = set()

        for metrics_path in sorted(group_dir.rglob("*_metrics.json")):
            parsed = _parse_job_name(metrics_path)
            if not parsed:
                continue
            file_group, loss_fn, rule_name = parsed
            expected_file_group = group_map.get(group, group)
            if file_group != expected_file_group:
                continue

            with metrics_path.open() as handle:
                data = json.load(handle)
            axiom_rate = data.get("axiom_violation_rate")
            thresholds = (
                data.get("rule_violations", {})
                .get("tie_mass", {})
                .get("thresholds", {})
            )
            threshold_key = f"{TIE_MASS_THRESHOLD}"
            threshold_data = thresholds.get(threshold_key) or thresholds.get(TIE_MASS_THRESHOLD, {})
            tie_mass_rate = threshold_data.get("rate") if isinstance(threshold_data, dict) else None

            metrics_by_loss_rule[(loss_fn, rule_name)] = _format_cell(axiom_rate, tie_mass_rate)
            rows.append(
                {
                    "group": group_map.get(group, group),
                    "loss_fn": loss_fn,
                    "rule": rule_name,
                    "axiom_violation_rate": axiom_rate,
                    f"tie_mass_{TIE_MASS_THRESHOLD}_rate": tie_mass_rate,
                }
            )
            losses.add(loss_fn)
            rules.add(rule_name)

        if not losses or not rules:
            continue

        _render_table(group_map.get(group, group), sorted(losses), sorted(rules), metrics_by_loss_rule)
        rendered = True

    if not rendered:
        print(f"No metrics found under {base_dir}")
        return

    csv_path = Path("final_validations_summary.csv")
    fieldnames = [
        "group",
        "loss_fn",
        "rule",
        "axiom_violation_rate",
        f"tie_mass_{TIE_MASS_THRESHOLD}_rate",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] CSV summary: {csv_path}")



if __name__ == "__main__":
    main()
