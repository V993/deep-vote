# Shared by the axiom finetune/eval scripts for the 5 rule replicates (paper Figures 3 and 4):
#
#   axiom_replicates_submit.sh      login node: writes configs + manifests, runs checks, submits both arrays
#   axiom_finetune_array.sh         GPU array task: one finetune + its eval
#   axiom_base_eval_array.sh        CPU array task: one base-model eval
#   axiom_finetune_replicates.sh    single GPU node, everything sequentially (timing runs, mop-up)
#
# All four use the same done-markers, configs and output paths, so work finished by one is skipped by the
# others. Source this file from the repo root.
#
# Selection and inputs, overridable from the environment:
#   REPS      replicate ids                  default "1 2 3 4 5"
#   AXIOMS    subset of axioms               default "participation monotonicity assp topcycle"
#   RULES     subset of rules                default all 8
#   DATA      tied 10k IC file               default the path in the committed finetune configs
#   BASE_RUN  replicate rule-run prefix      default experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v

REPS="${REPS:-1 2 3 4 5}"
AXIOMS="${AXIOMS:-participation monotonicity assp topcycle}"
RULES="${RULES:-plurality_pscf borda_pscf copeland_pscf schulze_pscf maximin_pscf irv_pscf black_pscf rankedpairs_pscf}"
DATA="${DATA:-data/pregenerated/with_alternatives/n_profiles=10000-num_voters=44-m=11-committee_size=1-pref_dist=IC-has_tied_profiles-TRAIN.h5}"
BASE_RUN="${BASE_RUN:-experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v}"

STATE_DIR="loni_output/axiom_finetune_replicates"
CONFIG_DIR="utils/parameters/replicates/fig3"
EVAL_ROOT="experiments/final_validations_reps"

finetune_config() {
    case "$1" in
        participation) echo "utils/parameters/11c44v_120kp_finetune_participation_efficient.json" ;;
        monotonicity)  echo "utils/parameters/11c44v_120kp_finetune_monotonicity_efficient.json" ;;
        assp)          echo "utils/parameters/11c44v_120kp_finetune_assp.json" ;;
        topcycle)      echo "utils/parameters/11c44v_120kp_finetune_topcycle.json" ;;
        *) echo "unknown axiom '$1'" >&2; return 1 ;;
    esac
}

# Loss names as the eval configs spell them (the monotonicity finetune config uses 'MontonicityLoss')
eval_loss() {
    case "$1" in
        participation) echo "ParticipationLoss" ;;
        monotonicity)  echo "MonotonicityLoss" ;;
        assp)          echo "AsspLoss" ;;
        topcycle)      echo "TopCycleLoss" ;;
        *) echo "unknown axiom '$1'" >&2; return 1 ;;
    esac
}

base_model() {  # base_model <rep> <rule>
    echo "${BASE_RUN}_rep$1/models/$2_concatenate.pt"
}

derived_config() {  # derived_config <axiom> <rep>
    echo "$CONFIG_DIR/$1_rep$2.json"
}

# Writes the derived config for one (axiom, replicate). The committed configs point models_directory at the
# December run; every replicate must instead finetune its own CO models. Written via a temp file and rename,
# so an array task reading it while the submit script rewrites it never sees a half-written file.
write_finetune_config() {  # write_finetune_config <axiom> <rep>
    python3 - "$(finetune_config "$1")" "$(derived_config "$1" "$2")" "${BASE_RUN}_rep$2/models" "$DATA" <<'PY'
import json, os, sys
src, dst, models_dir, data = sys.argv[1:5]
with open(src) as f:
    params = json.load(f)
params["models_directory"] = models_dir
params["data_filename"] = data
tmp = f"{dst}.tmp{os.getpid()}"
with open(tmp, "w") as f:
    json.dump(params, f, indent=4)
os.replace(tmp, dst)
PY
}

# Directory the finetune saves to. Mirrors utils.get_experiment_directory().
finetune_out_dir() {  # finetune_out_dir <axiom> <rep>
    python3 - "$(derived_config "$1" "$2")" "$2" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    params = json.load(f)
params.setdefault("pref_dist", "impartial_culture")  # learn_rule.py applies the same default
print("experiments/{n_cands}c_{n_voters}v_{n_profiles}p_{pref_dist}_{loss_fn}_{experiment_type}".format(**params)
      + f"_rep{sys.argv[2]}")
PY
}

# ---------- the three kinds of unit ----------

run_base_eval() {  # run_base_eval <axiom> <rule> <rep>
    # The pretrained eval config sets concatenate=false; these are CO models, so force it on.
    python3 test_model.py "$(base_model "$3" "$2")" \
        --params "utils/parameters/11c44v_pretrained_rules_eval_$(eval_loss "$1").json" \
        --data "$DATA" \
        --concatenate \
        --batch-label pretrained \
        --instances-csv "$EVAL_ROOT/rep$3"
}

run_finetune() {  # run_finetune <axiom> <rule> <rep>
    python3 learn_rule.py "$2" concatenate "$(derived_config "$1" "$3")" "$3"
}

run_finetuned_eval() {  # run_finetuned_eval <axiom> <rule> <rep>
    local out_dir
    out_dir=$(finetune_out_dir "$1" "$3") || return 1
    python3 test_model.py "$out_dir/models/$2_concatenate.pt" \
        --params "utils/parameters/11c44v_finetuned_rules_eval_$(eval_loss "$1").json" \
        --data "$DATA" \
        --batch-label finetuned \
        --instances-csv "$EVAL_ROOT/rep$3"
}

# ---------- done-markers ----------

unit_key() {  # unit_key <stage> <axiom> <rule> <rep>; stage is base, finetune or ft_eval
    echo "$1-$2-$3-rep$4"
}

is_done() {  # is_done <key>
    [[ -f "$STATE_DIR/done/$1" ]]
}

COMPLETED=0
SKIPPED=0
FAILED=()

# Runs one unit unless its done-marker exists. Writes the marker only on exit 0; logs to $STATE_DIR/logs/.
run_unit() {  # run_unit <key> <command...>
    local key="$1"; shift
    local log="$STATE_DIR/logs/$key.log"

    if is_done "$key"; then
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    mkdir -p "$STATE_DIR/done" "$STATE_DIR/logs"
    echo "[run ] $key  $(date '+%F %T')"
    echo "$*" > "$log"
    local start=$SECONDS
    if "$@" >> "$log" 2>&1; then
        echo "$(date '+%F %T') $((SECONDS - start))s ${SLURM_JOB_ID:-local}" > "$STATE_DIR/done/$key"
        echo "[ ok ] $key  $((SECONDS - start))s"
        COMPLETED=$((COMPLETED + 1))
        return 0
    else
        local rc=$?
        echo "[FAIL] $key  exit $rc after $((SECONDS - start))s, see $log"
        FAILED+=("$key")
        return 1
    fi
}

# ---------- checks ----------

# Inputs every unit in the current selection needs. Prints every problem, not just the first.
preflight_inputs() {
    local ok=0 axiom loss cfg rep rule
    [[ -f learn_rule.py && -f test_model.py ]] || { echo "Run from the repo root."; return 1; }
    [[ -f "$DATA" ]] || { echo "Missing tied data file: $DATA"; ok=1; }
    for axiom in $AXIOMS; do
        loss=$(eval_loss "$axiom") || { ok=1; continue; }
        for cfg in "$(finetune_config "$axiom")" \
                   "utils/parameters/11c44v_pretrained_rules_eval_${loss}.json" \
                   "utils/parameters/11c44v_finetuned_rules_eval_${loss}.json"; do
            [[ -f "$cfg" ]] || { echo "Missing config: $cfg"; ok=1; }
        done
    done
    for rep in $REPS; do
        for rule in $RULES; do
            [[ -f "$(base_model "$rep" "$rule")" ]] || { echo "Missing base model: $(base_model "$rep" "$rule")"; ok=1; }
        done
    done
    return $ok
}

require_cuda() {
    python3 -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "No CUDA device visible; refusing to run on CPU."; return 1; }
}

# Array tasks: sets AXIOM, RULE, REP, DATA and BASE_RUN from row <index> (1-based) of a manifest written by
# axiom_replicates_submit.sh. DATA/BASE_RUN travel in the manifest rather than through --export, because the
# data filename contains '=' characters.
read_manifest_row() {  # read_manifest_row <manifest> <index>
    local row
    row=$(sed -n "$2p" "$1")
    [[ -n "$row" ]] || { echo "No row $2 in manifest $1"; return 1; }
    IFS=$'\t' read -r AXIOM RULE REP DATA BASE_RUN <<< "$row"
}
