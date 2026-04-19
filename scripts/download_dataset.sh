#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_dataset.sh

DATA_DIR="data/ms1mv3"
EVAL_DIR="data/eval"
EVAL_EXTRA_DIR="data/eval_extra"
mkdir -p "$DATA_DIR" "$EVAL_DIR" "$EVAL_EXTRA_DIR"

# Training data (MS1MV3 RecordIO)
if [[ -f "$DATA_DIR/train.rec" ]]; then
  echo "Training data exists"
else
  pip install -q "huggingface_hub>=1.0"
  hf download gaunernst/ms1mv3-recordio \
    --repo-type dataset \
    --local-dir "$DATA_DIR"
fi

# Eval bins from Hugging Face mirrors
EXPECTED_BINS=("lfw.bin" "cfp_fp.bin" "agedb_30.bin" "calfw.bin" "cplfw.bin")
ALL_PRESENT=1
for b in "${EXPECTED_BINS[@]}"; do
  [[ -f "$EVAL_DIR/$b" ]] || ALL_PRESENT=0
done

if [[ $ALL_PRESENT -eq 1 ]]; then
  echo "All 5 eval bins exist"
else
  pip install -q "huggingface_hub>=1.0"
  echo "Downloading eval .bin files from Hugging Face..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

repo_id = "gaunernst/face-recognition-eval"
dst = Path("data/eval")
dst.mkdir(parents=True, exist_ok=True)

files = ["lfw.bin", "cfp_fp.bin", "agedb_30.bin", "calfw.bin", "cplfw.bin"]
for name in files:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=name,
    )
    (dst / name).write_bytes(Path(path).read_bytes())
    print(f"Downloaded {name}")
PY
  echo "Found:"
  ls "$EVAL_DIR"/*.bin
fi

# XQLFW
if [[ -f "$EVAL_EXTRA_DIR/xqlfw/xqlfw_pairs.txt" ]]; then
  echo "XQLFW exists"
else
  mkdir -p "$EVAL_EXTRA_DIR/xqlfw"
  curl -L -o "$EVAL_EXTRA_DIR/xqlfw/xqlfw_aligned_112.zip" \
    https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_aligned_112.zip
  curl -L -o "$EVAL_EXTRA_DIR/xqlfw/xqlfw_pairs.txt" \
    https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_pairs.txt
  unzip -o "$EVAL_EXTRA_DIR/xqlfw/xqlfw_aligned_112.zip" -d "$EVAL_EXTRA_DIR/xqlfw/"
  rm "$EVAL_EXTRA_DIR/xqlfw/xqlfw_aligned_112.zip"
fi

echo ""
echo "Done."
echo "  Training:    $DATA_DIR"
echo "  Eval (main): $EVAL_DIR"
echo "  Eval (extra, post-training): $EVAL_EXTRA_DIR/xqlfw"
