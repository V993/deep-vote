# Deep Voting: Learning the Possibilities Frontier of Social Choice

This repository contains the code for training neural networks that (1) learn existing voting rules from compact profile representations and (2) learn or fine-tune voting rules that are highly likely to satisfy axiomatic criteria such as Participation, Monotonicity, Strategyproofness (ASSP), and Top-Cycle consistency.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | Profile generation (impartial culture, IAC, Mallows, urn, single-peaked (Walsh), PrefLib) and HDF5 dataset loading. |
| `rules/` | Target voting rules (Borda, Plurality, Copeland, Maximin, Schulze, Ranked Pairs, IRV, Black, Condorcet), implemented on top of [whalrus](https://francois-durand.github.io/whalrus/). Rules suffixed `_pscf` are probabilistic social choice functions that output a distribution over candidates. |
| `transforms/` | Profile embeddings (compressions): rank frequency, tournament, weighted tournament, lexicographic, and full profile. |
| `models/` | Network architectures (`models/architectures/`), model construction, and the training/validation loops. |
| `loss_fns/` | Rule-learning losses and differentiable axiom losses, plus the generators for counterfactual profiles (abstentions, adjacent top-k swaps) used by the axiom losses. |
| `utils/` | Parameter loading, experiment naming/saving, seeding and device helpers. |
| `utils/parameters/` | JSON job configurations used for training and evaluation. |
| `utils/job_scripts/` | SLURM submission scripts and the job template used on our HPC cluster. |

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 is recommended. PrefLib experiments additionally require the Strict Order Complete (`.soc`) collection from [preflib.org](https://preflib.org), unpacked into `data/preflib/`.

## Usage

All scripts are run from the repository root and are driven by a JSON parameter file (see `utils/parameters/`). The key fields are `n_voters`, `n_cands`, `n_profiles`, `pref_dist`, `loss_fn`, `lr`, `batch_size`, `n_epochs`, and `concatenate` (train on all embeddings concatenated).

### 1. Generate data

```bash
python generate_data.py <params_path>
```

This samples profiles from the configured `pref_dist` (defaults to `impartial_culture`), computes rule outcomes with whalrus, and writes an HDF5 dataset to `data/pregenerated/`.

### 2. Learn a voting rule

```bash
python learn_rule.py <rule_name> <transform_name> <params_path> [replicate]

# example
python learn_rule.py borda_pscf concatenate utils/parameters/local_job_parameters.json
```

Models, loss curves, and the parameters used are saved under `experiments/`. Passing a replicate id appends `_rep<id>` to the experiment directory so repeated runs do not overwrite each other.

### 3. Axiom fine-tuning and training from scratch

Axiom training uses the same entry point with an axiom loss (`ParticipationLoss`, `MonotonicityLoss`, `AsspLoss`, `TopCycleLoss`) set in the parameter file; see the `11c44v_120kp_finetune_*` and `11c44v_120kp_learn_axioms_*` configurations.

### 4. Evaluation

| Script | Purpose |
| --- | --- |
| `test_model.py` | Evaluate a saved model on a held-out dataset for rule accuracy and axiom satisfaction. |
| `evaluate_rule.py` | Evaluate a trained rule network on a given preference distribution. |
| `evaluate_whalrus_rule.py` | Evaluate the exact whalrus rules against the axiom losses as a baseline. |
| `pref_distribution_testing.py` | Test models trained on impartial culture against other preference distributions. |
| `evaluate_replicates.py` | Evaluate all replicate models and aggregate mean ± std. |
| `paper_tables_replicates.py` | Produce the replicate-based tables and figures. |
| `summarize_final_validations.py` | Summarize final validation metrics into tables and CSV. |
| `make_ijcai2026_plots.py`, `plots_and_figures/make_fig4_axiom_summary.py` | Plotting scripts for the paper figures. |

### Running on a SLURM cluster

`rules_driver.py` and `replicate_driver.py` sweep every (rule, transform) pair in a parameter file and submit one SLURM job per pair using `utils/job_scripts/job_template.txt`. Pass `--local` (`replicate_driver.py`) or `false` as the second argument (`rules_driver.py`) to run locally instead. The `allocation`, `partition`, and output paths in the parameter files and job scripts are specific to our cluster and should be changed for your environment.

## License

TBD.
