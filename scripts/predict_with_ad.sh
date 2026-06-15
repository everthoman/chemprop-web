#!/usr/bin/env bash
# Predict with a chemprop web-app checkpoint, adding conformal intervals,
# ensemble uncertainty, and Tanimoto nearest-neighbour AD in one go.
#
# Usage:
#   predict_with_ad.sh --list
#   predict_with_ad.sh --input FILE --ckpt_id ID --output FILE [--alpha 0.15] [--no_cuda]
#
# Arguments:
#   --list      Show available checkpoints and exit
#   --input     CSV with smiles (and optionally compound_id) columns
#   --ckpt_id   Checkpoint ID (use --list to see available models)
#   --output    Path for the predictions CSV
#   --alpha     Conformal error rate (default 0.15 = 85% coverage)
#   --no_cuda   Force CPU inference
#
# Example:
#   bash scripts/predict_with_ad.sh --list
#   bash scripts/predict_with_ad.sh \
#       --input ~/Downloads/compounds.csv \
#       --ckpt_id 127 \
#       --output ~/Downloads/predictions.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHEMPROP_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$CHEMPROP_DIR/chemprop/web/app/web_checkpoints"
DB_PATH="$CHEMPROP_DIR/chemprop/web/chemprop.sqlite3"
PREDICT_PY="$CHEMPROP_DIR/predict.py"
AD_SCRIPT="$SCRIPT_DIR/tanimoto_ad.py"

INPUT=""
CKPT_ID=""
OUTPUT=""
ALPHA="0.15"
CUDA_FLAG=""
LIST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)    LIST=1;        shift ;;
        --input)   INPUT="$2";   shift 2 ;;
        --ckpt_id) CKPT_ID="$2"; shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --alpha)   ALPHA="$2";   shift 2 ;;
        --no_cuda) CUDA_FLAG="--no_cuda"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ $LIST -eq 1 ]]; then
    python3 - "$DB_PATH" "$CHECKPOINT_DIR" <<'EOF'
import sqlite3, os, sys

db_path, ckpt_dir = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db_path)
ckpts = conn.execute('SELECT id, ckpt_name, class, training_size, ensemble_size FROM ckpt ORDER BY id').fetchall()

print(f"{'ID':>4}  {'Name':<30}  {'Type':<15}  {'Train size':>10}  {'Ensemble':>8}  {'Conformal':>9}  {'AD':>6}")
print("-" * 90)
for ckpt_id, name, cls, train_size, ensemble in ckpts:
    has_cal = os.path.exists(os.path.join(ckpt_dir, f'{ckpt_id}_calibration.csv'))
    has_ad  = os.path.exists(os.path.join(ckpt_dir, f'{ckpt_id}_train_test_preds.csv'))
    models  = conn.execute('SELECT id FROM model WHERE associated_ckpt = ?', (ckpt_id,)).fetchall()
    model_ids = ', '.join(str(r[0]) for r in models)
    print(f"{ckpt_id:>4}  {name:<30}  {cls:<15}  {train_size:>10}  {ensemble:>8}  {'yes' if has_cal else 'no':>9}  {'yes' if has_ad else 'no':>6}")
EOF
    exit 0
fi

if [[ -z "$INPUT" || -z "$CKPT_ID" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --list"
    echo "       $0 --input FILE --ckpt_id ID --output FILE [--alpha 0.15] [--no_cuda]"
    exit 1
fi

# Resolve model .pt paths from the database
MODEL_PATHS=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
rows = conn.execute('SELECT id FROM model WHERE associated_ckpt = ?', ($CKPT_ID,)).fetchall()
if not rows:
    raise SystemExit(f'No models found for ckpt_id $CKPT_ID')
print(' '.join('$CHECKPOINT_DIR/' + str(r[0]) + '.pt' for r in rows))
")

CAL_PATH="$CHECKPOINT_DIR/${CKPT_ID}_calibration.csv"
TRAIN_PREDS_PATH="$CHECKPOINT_DIR/${CKPT_ID}_train_test_preds.csv"
TMPDIR="$(dirname "$OUTPUT")"
TMP_CONFORMAL="$TMPDIR/.tmp_conformal_$$.csv"
TMP_ENSEMBLE="$TMPDIR/.tmp_ensemble_$$.csv"

cleanup() { rm -f "$TMP_CONFORMAL" "$TMP_ENSEMBLE"; }
trap cleanup EXIT

echo "=== Checkpoint $CKPT_ID: models $MODEL_PATHS ==="

# Step 1: conformal prediction intervals
if [[ -f "$CAL_PATH" ]]; then
    echo "--- Step 1/3: conformal prediction (alpha=$ALPHA) ---"
    python3 "$PREDICT_PY" \
        --test_path "$INPUT" \
        --checkpoint_paths $MODEL_PATHS \
        --preds_path "$TMP_CONFORMAL" \
        --calibration_path "$CAL_PATH" \
        --calibration_method conformal_regression \
        --conformal_alpha "$ALPHA" \
        --num_workers 4 \
        $CUDA_FLAG
else
    echo "--- Step 1/3: conformal skipped (no calibration set) — using standard predictions ---"
    python3 "$PREDICT_PY" \
        --test_path "$INPUT" \
        --checkpoint_paths $MODEL_PATHS \
        --preds_path "$TMP_CONFORMAL" \
        --num_workers 4 \
        $CUDA_FLAG
fi

# Step 2: ensemble uncertainty
echo "--- Step 2/3: ensemble uncertainty ---"
python3 "$PREDICT_PY" \
    --test_path "$INPUT" \
    --checkpoint_paths $MODEL_PATHS \
    --preds_path "$TMP_ENSEMBLE" \
    --uncertainty_method ensemble \
    --num_workers 4 \
    $CUDA_FLAG

# Merge ensemble variance column into conformal output
python3 - "$TMP_CONFORMAL" "$TMP_ENSEMBLE" "$OUTPUT" <<'EOF'
import pandas as pd, sys

conformal = pd.read_csv(sys.argv[1])
ensemble  = pd.read_csv(sys.argv[2])
output    = sys.argv[3]

unc_cols = [c for c in ensemble.columns if c.endswith('_ensemble_uncal_var')]
assert len(conformal) == len(ensemble), \
    f"Row mismatch: {len(conformal)} vs {len(ensemble)}"

pd.concat([conformal, ensemble[unc_cols]], axis=1).to_csv(output, index=False)
EOF

# Step 3: Tanimoto AD
if [[ -f "$TRAIN_PREDS_PATH" ]]; then
    echo "--- Step 3/3: Tanimoto nearest-neighbour AD ---"
    python3 "$AD_SCRIPT" \
        --train_path "$TRAIN_PREDS_PATH" \
        --preds_path "$OUTPUT"
else
    echo "--- Step 3/3: AD skipped (no train/test predictions CSV found) ---"
fi

echo "=== Done. Output: $OUTPUT ==="
