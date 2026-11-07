#!/bin/bash

PARAMS_DIR="utils/parameters"
RULES=(black_pscf borda_pscf copeland_pscf irv_pscf maximin_pscf plurality_pscf rankedpairs_pscf schulze_pscf)
LOSSES=(AsspLoss MonotonicityLoss ParticipationLoss TopCycleLoss)
TRANSFORM="concatenate"

LOG_DIR="loni_output/final_tests"
mkdir -p "$LOG_DIR"

job_idx=0
for loss in "${LOSSES[@]}"; do
  param_file="${PARAMS_DIR}/11c44v_whalrus_rules_full_eval_${loss}.json"
  if [[ ! -f "$param_file" ]]; then
    echo "Missing param file: $param_file" >&2
    exit 1
  fi
  for rule in "${RULES[@]}"; do
    job_name="whalrus-${loss}-${rule}-${TRANSFORM}"
    out_file="${LOG_DIR}/${job_name}.out"
    err_file="${LOG_DIR}/${job_name}.err"
    echo "${job_idx} : Submitting whalrus eval for $rule ($loss)"
    sbatch \
      -A loni_tudai \
      -p workq \
      -N 1 \
      -c 64 \
      -t 10:00:00 \
      -J "${job_name}" \
      -o "${out_file}" \
      -e "${err_file}" \
      --mail-type=END,FAIL \
      --mail-user=lmatone@tulane.edu \
      --wrap "module load conda; cd ${SLURM_SUBMIT_DIR:-$PWD}; source activate testing; python evaluate_whalrus_rule.py ${rule} ${TRANSFORM} ${param_file}"
    echo ""
    job_idx=$((job_idx + 1))
  done
done
