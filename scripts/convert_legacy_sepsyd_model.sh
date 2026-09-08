#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conversion_dir="$(mktemp -d)"
trap 'rm -rf "${conversion_dir}"' EXIT

docker run --rm --platform linux/amd64 \
  -v "${project_root}:/workspace:ro" \
  -v "${conversion_dir}:/output" \
  python:3.7-slim sh -lc \
  "apt-get update >/dev/null && \
   apt-get install -y --no-install-recommends libgomp1 >/dev/null && \
   pip install --no-cache-dir -q 'numpy<1.22' 'scipy<1.8' 'xgboost==0.90' && \
   python /workspace/scripts/convert_legacy_sepsyd_model.py \
     /workspace/vendor/sepsyd_original/f120d4e02n8010val434.pickle.dat /output/model.bin"

docker run --rm --platform linux/amd64 \
  -v "${conversion_dir}:/output" \
  python:3.9-slim sh -lc \
  "apt-get update >/dev/null && \
   apt-get install -y --no-install-recommends libgomp1 >/dev/null && \
   pip install --no-cache-dir -q 'xgboost==1.7.6' && \
   python -c \"import xgboost as xgb; b=xgb.Booster(); b.load_model('/output/model.bin'); b.save_model('/output/model.json')\""

cp "${conversion_dir}/model.json" "${project_root}/artifacts/models/v1/model.json"
echo "Portable model written to artifacts/models/v1/model.json"
