"""Figure 4: axiom loss and rule accuracy per rule, base model vs finetuned model.

The original plotting code for this figure is not in the repo. This script rebuilds it from:

- Axiom loss (base, finetuned, actual rule via whalrus): Appendix I, Tables 12, 15, 18 and 21 of the paper.
  The raw metrics JSONs these came from are on the cluster, not here.
- Rule accuracy: 1 - tie_mass_0.95_rate from final_validations_summary.csv.
  The CSV has no finetuned MonotonicityLoss rows. Those 8 values are read off the published figure
  (base accuracy plus the offset of the "after" marker), see FT_MONO_DELTA.

Run from the repo root:  python plots_and_figures/make_fig4_axiom_summary.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

CSV_PATH = "final_validations_summary.csv"
OUT_PATH = "plots_and_figures/axiom_finetuning/finetune_summary.png"
FIGSIZE = (12, 5.1)  # published version (finetune_summary_old.png) is 12 x 4.2; saved at exactly this size

# Subplot order, as in the published figure.
RULES = ["plurality", "borda", "copeland", "schulze", "maximin", "irv", "black", "rankedpairs"]
RULE_TITLES = {
    "plurality": "Plurality", "borda": "Borda", "copeland": "Copeland", "schulze": "Schulze",
    "maximin": "Simpson-Kramer", "irv": "IRV", "black": "Black's Rule", "rankedpairs": "Ranked Pairs",
}
AXIOMS = ["AsspLoss", "MonotonicityLoss", "ParticipationLoss", "TopCycleLoss"]
AXIOM_LABELS = ["ASSP", "Mono.", "Part.", "TC"]

# Axiom loss from the paper's appendix tables. Column order:
# Plurality, Borda, Copeland, Schulze, Simpson-Kramer, IRV, Black's Rule, Ranked Pairs (same as RULES).
AXIOM_LOSS = {
    "MonotonicityLoss": {  # Table 12
        "base": [0.083, 0.096, 0.076, 0.072, 0.073, 0.077, 0.094, 0.071],
        "finetuned": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "rule": [0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.001],
    },
    "ParticipationLoss": {  # Table 15
        "base": [0.446, 0.422, 0.291, 0.291, 0.288, 0.293, 0.396, 0.280],
        "finetuned": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "rule": [0.000, 0.000, 0.060, 0.019, 0.019, 0.090, 0.047, 0.059],
    },
    "TopCycleLoss": {  # Table 18
        "base": [0.155, 0.015, 0.000, 0.001, 0.001, 0.069, 0.009, 0.032],
        "finetuned": [0.017, 0.012, 0.000, 0.001, 0.001, 0.070, 0.007, 0.035],
        "rule": [0.155, 0.017, 0.000, 0.000, 0.000, 0.063, 0.001, 0.000],
    },
    "AsspLoss": {  # Table 21
        "base": [0.446, 0.440, 0.348, 0.363, 0.377, 0.343, 0.409, 0.369],
        "finetuned": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "rule": [0.036, 0.015, 0.016, 0.015, 0.016, 0.064, 0.027, 0.025],
    },
}

# Finetuned Monotonicity rule accuracy is missing from the CSV. Offsets from base accuracy,
# measured from marker positions in the published figure (resolution about 0.01).
FT_MONO_DELTA = {"borda": -0.056, "black": -0.010}


def load_rule_accuracy(csv_path):
    df = pd.read_csv(csv_path)
    df["rule"] = df["rule"].str.replace("_pscf", "", regex=False)
    df["acc"] = 1.0 - df["tie_mass_0.95_rate"]

    # The base model is the same for every axiom, so its rule accuracy is too. Take it from the
    # AsspLoss rows; the TopCycleLoss rows have duplicates for black and borda.
    base = df[(df.group == "pretrained") & (df.loss_fn == "AsspLoss")].set_index("rule")["acc"].to_dict()

    ft = {}
    for axiom in AXIOMS:
        rows = df[(df.group == "finetuned") & (df.loss_fn == axiom)].set_index("rule")["acc"].to_dict()
        if not rows and axiom == "MonotonicityLoss":
            rows = {r: base[r] + FT_MONO_DELTA.get(r, 0.0) for r in RULES}
        ft[axiom] = rows
    return base, ft


def main():
    base_acc, ft_acc = load_rule_accuracy(CSV_PATH)

    blue, green = "tab:blue", "tab:green"
    off = 0.3  # horizontal offset of the rule-accuracy markers from the axiom-loss markers
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE, sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)

    for i, rule in enumerate(RULES):
        ax = axes.flat[i]
        for x, axiom in enumerate(AXIOMS):
            loss = AXIOM_LOSS[axiom]
            a0, a1, aw = loss["base"][i], loss["finetuned"][i], loss["rule"][i]
            r0, r1 = base_acc[rule], ft_acc[axiom][rule]

            ax.plot([x, x], [a0, a1], ls="--", lw=2.5, color=blue, alpha=0.7, zorder=2)
            ax.plot([x + off, x + off], [r0, r1], ls="--", lw=2.5, color=green, alpha=0.7, zorder=2)

            ax.plot(x, aw, "X", ms=12, color="black", alpha=0.6, mec="black", zorder=3)
            ax.plot(x, a0, "o", ms=13, color=blue, alpha=0.75, mec="black", zorder=4)
            ax.plot(x, a1, "h", ms=12, color=blue, alpha=0.9, mec="black", zorder=5)
            ax.plot(x + off, r0, "^", ms=13, color=green, alpha=0.75, mec="black", zorder=4)
            ax.plot(x + off, r1, "v", ms=13, color=green, alpha=0.75, mec="black", zorder=5)

        ax.set_title(RULE_TITLES[rule], fontsize=16, pad=4)
        ax.set_xticks(range(len(AXIOMS)))
        ax.set_xticklabels(AXIOM_LABELS, rotation=45, ha="right", fontsize=12)
        ax.set_xlim(-0.35, len(AXIOMS) - 1 + off + 0.35)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0.0", "0.25", "0.5", "0.75", "1.0"])
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, alpha=0.3)
        if i % 4 == 3:  # right column carries its tick labels on the right
            ax.tick_params(axis="y", labelleft=False, labelright=True, right=True)

    fig.supylabel("Axiom Loss / Rule Accuracy", fontsize=15)

    legend = [
        Line2D([0], [0], marker="o", ls="", ms=12, mfc=blue, mec="black", label="Axiom Loss Before FT"),
        Line2D([0], [0], marker="h", ls="", ms=12, mfc=blue, mec="black", label="Axiom Loss After FT"),
        Line2D([0], [0], marker="^", ls="", ms=12, mfc=green, mec="black", label="Rule Accuracy Before FT"),
        Line2D([0], [0], marker="v", ls="", ms=12, mfc=green, mec="black", label="Rule Accuracy After FT"),
        Line2D([0], [0], marker="X", ls="", ms=12, mfc="black", mec="black", label="Whalrus Axiom Loss"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=5, fontsize=12, mode="expand",
               handletextpad=0.4, borderaxespad=0.2)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)  # no bbox trimming: canvas is exactly FIGSIZE
    print(f"[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
