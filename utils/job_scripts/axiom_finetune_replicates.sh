#!/bin/bash
#SBATCH -A loni_tudai26
#SBATCH -p gpu2
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -J axiom_finetune_replicates
#SBATCH -o /work/lmatone/deep-voting/loni_output/axiom_finetune_replicates/%x-%j.o
#SBATCH -e /work/lmatone/deep-voting/loni_output/axiom_finetune_replicates/%x-%j.e
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

# AXIOM FINETUNING + EVALUATION FOR THE 5 RULE REPLICATES, SEQUENTIALLY ON ONE GPU NODE
#
# The full sweep normally goes through the arrays (axiom_replicates_submit.sh). This single-node version is
# for timing replicate 1 and for mopping up a handful of leftover units. It shares done-markers, configs and
# output paths with the arrays (axiom_replicates_common.sh), so either picks up where the other stopped.
#
# Runs, in order:
#   1. base evals       each replicate's CO rule model, scored with each axiom loss       (test_model.py)
#   2. finetunes        each CO model finetuned on each axiom loss                        (learn_rule.py)
#   3. finetuned evals  each finetuned model scored with its axiom loss, right after its finetune
#
# Resumable: units with a done-marker are skipped, so resubmitting after a walltime kill continues where it
# stopped. A failed unit is logged and the job moves on.
#
# Submit from the repo root. sbatch needs the log directory to exist before the job starts:
#
#   mkdir -p loni_output/axiom_finetune_replicates
#   REPS="1" sbatch --export=ALL utils/job_scripts/axiom_finetune_replicates.sh    # time replicate 1
#   sbatch --dependency=singleton utils/job_scripts/axiom_finetune_replicates.sh   # queue a continuation
#
# Overrides (pass with --export=ALL): REPS, AXIOMS, RULES, DATA, BASE_RUN (see axiom_replicates_common.sh),
# and STAGES = "base", "finetune" or both (default "base finetune"; finetune includes its eval).
#
# Outputs:
#   finetuned models/losses  experiments/11c_44v_10000p_impartial_culture_{loss}_{experiment_type}_rep{n}/
#   eval metrics             experiments/final_validations_reps/rep{n}/{pretrained,finetuned}/{loss}/
#   derived configs          utils/parameters/replicates/fig3/{axiom}_rep{n}.json
#   per-unit logs            loni_output/axiom_finetune_replicates/logs/

set -uo pipefail

module load conda
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source activate testing
source utils/job_scripts/axiom_replicates_common.sh

STAGES="${STAGES:-base finetune}"

# ---------- preflight: fail fast before burning node time ----------

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T')"
echo "REPS=[$REPS] AXIOMS=[$AXIOMS] STAGES=[$STAGES]"
echo "DATA=$DATA"
echo "BASE_RUN=${BASE_RUN}_rep{n}"

preflight_inputs || exit 1
require_cuda || exit 1
mkdir -p "$STATE_DIR/done" "$STATE_DIR/logs" "$CONFIG_DIR"

# ---------- stage 1: base evals (Figure 4, base rows) ----------

if [[ " $STAGES " == *" base "* ]]; then
    echo; echo "=== Stage 1: base evals ==="
    for rep in $REPS; do
        for axiom in $AXIOMS; do
            for rule in $RULES; do
                run_unit "$(unit_key base "$axiom" "$rule" "$rep")" run_base_eval "$axiom" "$rule" "$rep"
            done
        done
    done
fi

# ---------- stage 2: finetunes, each followed by its eval (Figure 3, Figure 4 finetuned rows) ----------

if [[ " $STAGES " == *" finetune "* ]]; then
    echo; echo "=== Stage 2: finetunes + finetuned evals ==="
    for rep in $REPS; do
        for axiom in $AXIOMS; do
            write_finetune_config "$axiom" "$rep" || { echo "Could not write $(derived_config "$axiom" "$rep")"; exit 1; }
            for rule in $RULES; do
                # The eval only runs when the finetune succeeded (or completed in an earlier job).
                run_unit "$(unit_key finetune "$axiom" "$rule" "$rep")" run_finetune "$axiom" "$rule" "$rep" \
                    && run_unit "$(unit_key ft_eval "$axiom" "$rule" "$rep")" run_finetuned_eval "$axiom" "$rule" "$rep"
            done
        done
    done
fi

# ---------- summary ----------

echo
echo "=== Summary $(date '+%F %T') ==="
echo "completed this job: $COMPLETED   already done (skipped): $SKIPPED   failed: ${#FAILED[@]}"
for key in ${FAILED[@]+"${FAILED[@]}"}; do
    echo "  FAILED: $key   ($STATE_DIR/logs/$key.log)"
done
remaining=0
for stage in base finetune ft_eval; do
    for rep in $REPS; do for axiom in $AXIOMS; do for rule in $RULES; do
        is_done "$(unit_key "$stage" "$axiom" "$rule" "$rep")" || remaining=$((remaining + 1))
    done; done; done
done
echo "units in this selection still without a done-marker: $remaining"

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
exit 0
