#!/bin/bash
#SBATCH -A loni_tudai26
#SBATCH -p workq
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -J axiom_base_eval
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --mail-user lmatone@tulane.edu

# CPU ARRAY TASK: one replicate CO rule model scored with one axiom loss (Figure 4, base rows).
#
# Same node shape and walltime as the January pretrained evals (submit_pretrained_batch.sh). Do not sbatch
# this directly; axiom_replicates_submit.sh writes the manifest it reads and sets --array and logs:
#
#   sbatch --array=1-N%20 utils/job_scripts/axiom_base_eval_array.sh <manifest.tsv>

set -uo pipefail

MANIFEST="${1:?usage: sbatch --array=... axiom_base_eval_array.sh <manifest.tsv>}"

module load conda
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source activate testing
source utils/job_scripts/axiom_replicates_common.sh

read_manifest_row "$MANIFEST" "${SLURM_ARRAY_TASK_ID:?not running as an array task}" || exit 1
echo "Task ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} on $(hostname): $AXIOM $RULE rep$REP"

run_unit "$(unit_key base "$AXIOM" "$RULE" "$REP")" run_base_eval "$AXIOM" "$RULE" "$REP"
