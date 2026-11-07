#!/bin/bash
#SBATCH -A loni_tudai26
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 72:00:00
#SBATCH -J axiom_finetune
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

# GPU ARRAY TASK: one axiom finetune of one replicate's CO model, then the eval of the finetuned model.
#
# Do not sbatch this directly; axiom_replicates_submit.sh writes the configs and the manifest it reads, and
# sets --array, logs and throttle. Usage (as the submit script calls it):
#
#   sbatch --array=1-N%12 utils/job_scripts/axiom_finetune_array.sh <manifest.tsv>
#
# --gres=gpu:1 takes one of the node's two GPUs, so SLURM can place two tasks on one gpu2 node. SLURM sets
# CUDA_VISIBLE_DEVICES, so learn_rule.py's 'cuda:0' is this task's GPU.

set -uo pipefail

MANIFEST="${1:?usage: sbatch --array=... axiom_finetune_array.sh <manifest.tsv>}"

module load conda
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source activate testing
source utils/job_scripts/axiom_replicates_common.sh

read_manifest_row "$MANIFEST" "${SLURM_ARRAY_TASK_ID:?not running as an array task}" || exit 1
echo "Task ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}: $AXIOM $RULE rep$REP"
require_cuda || exit 1

# The eval only runs once the finetune has succeeded, in this task or an earlier one.
run_unit "$(unit_key finetune "$AXIOM" "$RULE" "$REP")" run_finetune "$AXIOM" "$RULE" "$REP" \
    && run_unit "$(unit_key ft_eval "$AXIOM" "$RULE" "$REP")" run_finetuned_eval "$AXIOM" "$RULE" "$REP"
