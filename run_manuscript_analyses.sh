#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/Users/sakiko/Dropbox/研究/22_AMPA_MultiVariateAnalysis/UMAP/data"
OUT_ROOT="./result"

python ampa_umap_pipeline.py all \
  --wb-data "${DATA_DIR}/HCPSY218_WBSUVR_Dataset.csv" \
  --wm-data "${DATA_DIR}/HCPSY218_WMSUVR_Dataset_MaskMean.csv" \
  --out-root "${OUT_ROOT}" \
  --nperm 1000 \
  --seed 0 \
  --representative-perm 1000 \
  --save-perm-every 10
