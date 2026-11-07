#!/bin/bash
# Move the existing rule-replicate outputs aside so a new replicate sweep cannot overwrite them.
#
# Moves (never copies or deletes):
#   experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v_rep{1..5}/   -> experiments/archive_<tag>/
#   loni_output/11c44v/11c44v_120kp_learn_rules*_rep{1..5}.{o,e}             -> loni_output/archive_<tag>/11c44v/
#
# Folder names are kept, so the archived run can still be read with
#   python evaluate_replicates.py --experiment-base experiments/archive_<tag>/11c_44v_120000p_impartial_culture_L1Loss_11c44v
#   python paper_tables_replicates.py --experiment-base experiments/archive_<tag>/11c_44v_120000p_impartial_culture_L1Loss_11c44v
#
# Usage (from the repo root on the cluster):
#   bash utils/job_scripts/archive_replicates.sh jul2026            # dry run: list what would move
#   bash utils/job_scripts/archive_replicates.sh jul2026 --apply    # move it
set -euo pipefail

TAG="${1:-}"
APPLY="${2:-}"
if [[ -z "$TAG" || "$TAG" == --* ]]; then
    echo "Usage: bash $0 <tag> [--apply]" >&2
    exit 1
fi

RUN="experiments/11c_44v_120000p_impartial_culture_L1Loss_11c44v"
LOG_DIR="loni_output/11c44v"
EXP_DEST="experiments/archive_${TAG}"
LOG_DEST="loni_output/archive_${TAG}/11c44v"

shopt -s nullglob
dirs=()
for rep in 1 2 3 4 5; do
    [[ -d "${RUN}_rep${rep}" ]] && dirs+=("${RUN}_rep${rep}")
done
logs=( "$LOG_DIR"/11c44v_120kp_learn_rules*_rep[1-5].o "$LOG_DIR"/11c44v_120kp_learn_rules*_rep[1-5].e )

if [[ ${#dirs[@]} -eq 0 && ${#logs[@]} -eq 0 ]]; then
    echo "Nothing to archive: no ${RUN}_rep* folders and no replicate logs in ${LOG_DIR}/."
    exit 0
fi

# Refuse to merge into an existing archive, so nothing in it can be overwritten.
for d in "${dirs[@]}"; do
    if [[ -e "$EXP_DEST/$(basename "$d")" ]]; then
        echo "Refusing: $EXP_DEST/$(basename "$d") already exists. Pick another tag." >&2
        exit 1
    fi
done
for f in "${logs[@]}"; do
    if [[ -e "$LOG_DEST/$(basename "$f")" ]]; then
        echo "Refusing: $LOG_DEST/$(basename "$f") already exists. Pick another tag." >&2
        exit 1
    fi
done

echo "Model folders (${#dirs[@]}) -> $EXP_DEST/"
for d in "${dirs[@]}"; do
    echo "    $d  ($(ls "$d/models" 2>/dev/null | wc -l | tr -d ' ') models)"
done
echo "Job logs (${#logs[@]}) -> $LOG_DEST/"

if [[ "$APPLY" != "--apply" ]]; then
    echo
    echo "Dry run: nothing moved. Re-run with --apply to move."
    exit 0
fi

mkdir -p "$EXP_DEST" "$LOG_DEST"
for d in "${dirs[@]}"; do mv -n "$d" "$EXP_DEST/"; done
for f in "${logs[@]}"; do mv -n "$f" "$LOG_DEST/"; done
echo
echo "Moved. Check: ls $EXP_DEST $LOG_DEST | head"
