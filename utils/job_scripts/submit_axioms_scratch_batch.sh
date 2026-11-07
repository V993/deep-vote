#!/bin/bash
# Submission helper for axioms_scratch rule/axiom evaluations (run from a login node).
# Usage: bash utils/job_scripts/submit_axioms_scratch_batch.sh

set -euo pipefail

AXIOM_DIRS=(
  "experiments/11c_44v_10000p_impartial_culture_AllAxiomsLoss_all_axioms"
  "experiments/11c_44v_10000p_impartial_culture_AsspLoss+TopCycle_all_axioms"
  "experiments/11c_44v_10000p_impartial_culture_Participation+TopCycle_all_axioms"
)

RULES=(
  "black_pscf"
  "borda_pscf"
  "copeland_pscf"
  "irv_pscf"
  "plurality_pscf"
  "rankedpairs_pscf"
  "schulze_pscf"
  "maximin_pscf"
)

AXIOM_PARAM_FILES=(
  "utils/parameters/11c44v_pretrained_rules_eval_AsspLoss.json"
  "utils/parameters/11c44v_pretrained_rules_eval_MonotonicityLoss.json"
  "utils/parameters/11c44v_pretrained_rules_eval_ParticipationLoss.json"
  "utils/parameters/11c44v_pretrained_rules_eval_TopCycleLoss.json"
)

BASE_RULE_PARAMS="utils/parameters/11c44v_pretrained_rules_eval_MonotonicityLoss.json"
LOG_ROOT="loni_output/axioms_scratch"
INSTANCES_ROOT="experiments/final_validations/axioms_scratch"

mkdir -p "$LOG_ROOT"

submit_job () {
  local model_path="$1"
  local params_path="$2"
  local job_name="$3"
  local log_dir="$4"
  local extra_args="$5"

  local out_file="${log_dir}/${job_name}.out"
  local err_file="${log_dir}/${job_name}.err"

  sbatch \
    -A loni_tudai \
    -p workq \
    -N 1 \
    -c 64 \
    -t 24:00:00 \
    -J "${job_name}" \
    -o "${out_file}" \
    -e "${err_file}" \
    --mail-type=END,FAIL \
    --mail-user=lmatone@tulane.edu \
    --wrap "module load conda; cd ${SLURM_SUBMIT_DIR:-$PWD}; source activate testing; python test_model.py ${model_path} --params ${params_path} ${extra_args} --instances-csv ${INSTANCES_ROOT}"
}

for axiom_dir in "${AXIOM_DIRS[@]}"; do
  batch_label="$(basename "$axiom_dir")"
  model_path="${axiom_dir}/models/none_concatenate.pt"
  if [[ ! -f "$model_path" ]]; then
    echo "Skipping ${axiom_dir}: none_concatenate.pt not found"
    continue
  fi

  log_dir="${LOG_ROOT}/${batch_label}"
  mkdir -p "$log_dir"

  for rule in "${RULES[@]}"; do
    job_name="axioms_scratch-${batch_label}-${rule}-L1Loss"
    extra_args="--loss-fn L1Loss --rule-name ${rule} --concatenate --axiom-loss-weight 0 --batch-label ${batch_label}"
    submit_job "$model_path" "$BASE_RULE_PARAMS" "$job_name" "$log_dir" "$extra_args"
  done

  for axiom_param in "${AXIOM_PARAM_FILES[@]}"; do
    axiom_loss="$(python3 - "$axiom_param" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    params = json.load(handle)
print(params["loss_fn"])
PY
)"
    job_name="axioms_scratch-${batch_label}-none-${axiom_loss}"
    extra_args="--rule-name none --concatenate --axiom-loss-weight 1 --batch-label ${batch_label}"
    submit_job "$model_path" "$axiom_param" "$job_name" "$log_dir" "$extra_args"
  done
done
