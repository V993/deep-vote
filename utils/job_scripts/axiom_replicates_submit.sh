#!/bin/bash
# SUBMIT THE AXIOM FINETUNE + EVAL ARRAYS FOR THE 5 RULE REPLICATES (paper Figures 3 and 4)
#
# Run on the login node, from the repo root:
#
#   bash utils/job_scripts/axiom_replicates_submit.sh --dry-run     # checks, configs, manifests; no sbatch
#   bash utils/job_scripts/axiom_replicates_submit.sh               # submit both arrays
#   bash utils/job_scripts/axiom_replicates_submit.sh --only base   # or --only finetune
#
# What it does:
#   1. Checks every input the selection needs: tied data file, finetune/eval configs, all base models.
#   2. Writes one derived finetune config per (axiom, replicate) to utils/parameters/replicates/fig3/, with
#      models_directory pointing at that replicate's CO models.
#   3. Writes two manifests (one row per task, replicate 1 first) to loni_output/axiom_finetune_replicates/
#      manifests/, timestamped so a resubmission never rewrites a manifest a running array is reading.
#      Units whose done-markers exist are left out, so rerunning this script submits only unfinished work.
#   4. Submits:
#        GPU array on gpu2, one finetune + its eval per task      --array=1-N%GPU_THROTTLE  (default 12)
#        CPU array on workq, one base eval per task               --array=1-N%CPU_THROTTLE  (default 20)
#
# Full sweep: 4 axioms x 8 rules x 5 replicates = 160 tasks in each array.
#
# Environment overrides: REPS, AXIOMS, RULES, DATA, BASE_RUN (see axiom_replicates_common.sh),
# GPU_THROTTLE, CPU_THROTTLE. Example, replicate 1 only to time it:
#
#   REPS="1" bash utils/job_scripts/axiom_replicates_submit.sh

set -uo pipefail

DRY_RUN=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only)    ONLY="${2:-}"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
if [[ -n "$ONLY" && "$ONLY" != "base" && "$ONLY" != "finetune" ]]; then
    echo "--only takes 'base' or 'finetune'"
    exit 1
fi

[[ -f utils/job_scripts/axiom_replicates_common.sh ]] || { echo "Run from the repo root."; exit 1; }
source utils/job_scripts/axiom_replicates_common.sh

GPU_THROTTLE="${GPU_THROTTLE:-12}"
CPU_THROTTLE="${CPU_THROTTLE:-20}"

echo "REPS=[$REPS] AXIOMS=[$AXIOMS]"
echo "RULES=[$RULES]"
echo "DATA=$DATA"
echo "BASE_RUN=${BASE_RUN}_rep{n}"
echo

# ---------- 1. checks ----------

preflight_inputs || { echo "Preflight failed; nothing written or submitted."; exit 1; }
if [[ $DRY_RUN -eq 0 ]] && ! command -v sbatch > /dev/null; then
    echo "sbatch not found; run this on the cluster login node (or use --dry-run)."
    exit 1
fi
echo "Preflight ok."

# ---------- 2. derived configs ----------

mkdir -p "$CONFIG_DIR" "$STATE_DIR/manifests" "$STATE_DIR/array_logs" "$STATE_DIR/done" "$STATE_DIR/logs"
for rep in $REPS; do
    for axiom in $AXIOMS; do
        write_finetune_config "$axiom" "$rep" || { echo "Could not write $(derived_config "$axiom" "$rep")"; exit 1; }
        echo "config $(derived_config "$axiom" "$rep")  ->  $(finetune_out_dir "$axiom" "$rep")"
    done
done

# ---------- 3. manifests ----------

stamp=$(date +%Y%m%d-%H%M%S)
ft_manifest="$STATE_DIR/manifests/finetune_${stamp}.tsv"
base_manifest="$STATE_DIR/manifests/base_${stamp}.tsv"
: > "$ft_manifest"
: > "$base_manifest"

for rep in $REPS; do
    for axiom in $AXIOMS; do
        for rule in $RULES; do
            row=$(printf '%s\t%s\t%s\t%s\t%s' "$axiom" "$rule" "$rep" "$DATA" "$BASE_RUN")
            if ! is_done "$(unit_key finetune "$axiom" "$rule" "$rep")" \
               || ! is_done "$(unit_key ft_eval "$axiom" "$rule" "$rep")"; then
                echo "$row" >> "$ft_manifest"
            fi
            if ! is_done "$(unit_key base "$axiom" "$rule" "$rep")"; then
                echo "$row" >> "$base_manifest"
            fi
        done
    done
done

n_ft=$(wc -l < "$ft_manifest" | tr -d ' ')
n_base=$(wc -l < "$base_manifest" | tr -d ' ')
echo
echo "finetune tasks to run: $n_ft   ($ft_manifest)"
echo "base eval tasks to run: $n_base   ($base_manifest)"

# ---------- 4. submit ----------

submit_array() {  # submit_array <script> <manifest> <n> <throttle>
    local script="$1" manifest="$2" n="$3" throttle="$4"
    local name
    name=$(basename "$script" .sh)
    if [[ "$n" -eq 0 ]]; then
        echo "[$name] nothing left to run; not submitted."
        return 0
    fi
    local cmd=(sbatch
        --array="1-${n}%${throttle}"
        -o "$PWD/$STATE_DIR/array_logs/${name}-%A_%a.o"
        -e "$PWD/$STATE_DIR/array_logs/${name}-%A_%a.e"
        "$script" "$manifest")
    echo "[$name] ${cmd[*]}"
    if [[ $DRY_RUN -eq 0 ]]; then
        "${cmd[@]}" || { echo "[$name] sbatch failed"; return 1; }
    fi
}

echo
status=0
if [[ "$ONLY" != "base" ]]; then
    submit_array utils/job_scripts/axiom_finetune_array.sh "$ft_manifest" "$n_ft" "$GPU_THROTTLE" || status=1
fi
if [[ "$ONLY" != "finetune" ]]; then
    submit_array utils/job_scripts/axiom_base_eval_array.sh "$base_manifest" "$n_base" "$CPU_THROTTLE" || status=1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo
    echo "Dry run: configs and manifests written, nothing submitted. First manifest rows:"
    head -3 "$ft_manifest"
fi
exit $status
