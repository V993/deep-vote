#!/bin/bash
# Submission helper for finetuned rule evaluations (run from a login node).
# Usage: bash utils/job_scripts/submit_finetuned_batch.sh

set -euo pipefail

PARAM_FILES=(utils/parameters/11c44v_finetuned_rules_eval_MonotonicityLoss.json)

LOG_DIR="loni_output/final_tests"
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
  local rule_name="${model_base%_*}"
  local transform_name="${model_base##*_}"
  local job_name="finetuned-${loss_fn}-${rule_name}-${transform_name}"
  local out_file="${LOG_DIR}/${job_name}.out"
  local err_file="${LOG_DIR}/${job_name}.err"

  echo "Submitting finetuned eval: $model_path ($loss_fn)"
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
    --wrap "module load conda; cd ${SLURM_SUBMIT_DIR:-$PWD}; source activate testing; python test_model.py ${model_path} --params ${param_file} ${data_filename:+--data ${data_filename}} --instances-csv experiments/final_validations/finetuned_rules_eval"
}

for param_file in "${PARAM_FILES[@]}"; do
  loss_fn=$(python3 - "$param_file" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    p=json.load(f)
print(p["loss_fn"])
PY
)

  search_loss="${loss_fn}"
  if [[ "${loss_fn}" == "MonotonicityLoss" ]]; then
    search_loss="MonotonicityLoss|MontonicityLoss"
  fi

  while IFS= read -r model_path; do
    [[ -z "$model_path" ]] && continue
    submit_job "$model_path" "$param_file"
  done < <(find models/finetuned_models -type f -path "*/models/*_concatenate.pt" | grep -E "${search_loss}" | sort)
done
