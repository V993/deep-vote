#!/bin/bash
# Submission helper for pretrained rule evaluations (run from a login node).
# Usage: bash utils/job_scripts/submit_pretrained_batch.sh

set -euo pipefail

PARAM_FILES=(utils/parameters/11c44v_pretrained_rules_eval_AsspLoss.json \
             utils/parameters/11c44v_pretrained_rules_eval_MonotonicityLoss.json \
             utils/parameters/11c44v_pretrained_rules_eval_ParticipationLoss.json \
             utils/parameters/11c44v_pretrained_rules_eval_TopCycleLoss.json)

LOG_DIR="loni_output/transform_tests"
mkdir -p "$LOG_DIR"

submit_job () {
  local model_path="$1"
  local param_file="$2"

  # Extract loss_fn and data_filename from param file
  local loss_fn data_filename
  loss_fn=$(python3 - "$param_file" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    p=json.load(f)
print(p["loss_fn"])
PY
)
  data_filename=$(python3 - "$param_file" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    p=json.load(f)
print(p.get("data_filename",""))
PY
)

  local model_base
  model_base=$(basename "$model_path" .pt)
  local transform_name=""
  for candidate in weighted_tournament; do
    if [[ "$model_base" == *"_${candidate}" ]]; then
      transform_name="$candidate"
      break
    fi
  done
  if [[ -z "$transform_name" ]]; then
    echo "Skipping $model_base: unknown transform suffix"
    return
  fi
  local rule_name="${model_base%_${transform_name}}"
  if [[ "$transform_name" != "weighted_tournament" ]]; then
    echo "Skipping $model_base: transform ${transform_name} already handled"
    return
  fi

  local job_name="pretrained-${loss_fn}-${rule_name}-${transform_name}"
  local out_file="${LOG_DIR}/${job_name}.out"
  local err_file="${LOG_DIR}/${job_name}.err"

  echo "Submitting pretrained eval: $model_path ($loss_fn)"
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
    --wrap "module load conda; cd ${SLURM_SUBMIT_DIR:-$PWD}; source activate testing; python test_model.py ${model_path} --params ${param_file} ${data_filename:+--data ${data_filename}} --instances-csv experiments/final_validations/transform_tests"
}

while IFS= read -r model_path; do
  [[ -z "$model_path" ]] && continue
  for param_file in "${PARAM_FILES[@]}"; do
    submit_job "$model_path" "$param_file"
  done
done < <(find models/pretrained_models -maxdepth 1 -type f -name '*.pt' ! -name '*_concatenate.pt' | sort)
