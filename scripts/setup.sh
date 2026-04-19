#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "iresnet50"
  "mobilefacenet"
  "ghostfacenet"
  "edgeface_s"
  "swin_tiny"
  "facelivtv2_s"
)

echo "Select model:"
for i in "${!MODELS[@]}"; do
  printf "  %d) %s\n" "$((i + 1))" "${MODELS[$i]}"
done

MODEL_INDEX="${1:-}"
if [[ -z "${MODEL_INDEX}" ]]; then
  read -r -p "Enter number [1-${#MODELS[@]}]: " MODEL_INDEX
fi

if ! [[ "${MODEL_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: model choice must be a number"
  exit 1
fi

if [[ "${MODEL_INDEX}" -lt 1 || "${MODEL_INDEX}" -gt "${#MODELS[@]}" ]]; then
  echo "ERROR: choose a number between 1 and ${#MODELS[@]}"
  exit 1
fi

MODEL_NAME="${MODELS[$((MODEL_INDEX - 1))]}"
TRAIN_CONFIG="config/${MODEL_NAME}.yaml"
DATA_CONFIG="config/data_subset.yaml"
TRAIN_LOG_DIR="logs"
TRAIN_OUT_DIR="results/${MODEL_NAME}"
PRUNE_OUT_DIR="results_prune_subset/${MODEL_NAME}"

python -m pip install -r requirements.txt
chmod +x scripts/download_dataset.sh
chmod +x scripts/validate_recordio.py
mkdir -p "$TRAIN_LOG_DIR"

./scripts/download_dataset.sh

python train/train.py \
  --config "$TRAIN_CONFIG" \
  --data_config "$DATA_CONFIG" \
  2>&1 | tee "${TRAIN_LOG_DIR}/${MODEL_NAME}_subset.log"

CHECKPOINT="${TRAIN_OUT_DIR}/best_model.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

python pruning/prune_iterative.py \
  --config "$TRAIN_CONFIG" \
  --data_config "$DATA_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$PRUNE_OUT_DIR" \
  --step_ratio 0.1 \
  --bn_recal_batches 5 \
  2>&1 | tee "${TRAIN_LOG_DIR}/${MODEL_NAME}_subset_prune.log"

echo "Done."
