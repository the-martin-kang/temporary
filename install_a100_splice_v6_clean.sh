#!/usr/bin/env bash
set -euo pipefail

source /tools/anaconda3/etc/profile.d/conda.sh

# Recreate because previous failed resolver attempts left the env incomplete.
conda deactivate 2>/dev/null || true
conda env remove -n mlbio_a100 -y 2>/dev/null || true
conda create -n mlbio_a100 python=3.10 -y
conda activate mlbio_a100

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_a100_splice_v6_cu121_v6_resolved.txt

python - <<'PY'
import torch
import esm
from rdkit import Chem
from tdc.multi_pred import DTI
from transformers import AutoTokenizer, AutoModel
import accelerate

print("python ok")
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("accelerate:", accelerate.__version__)
print("fair-esm import name esm: OK")
print("rdkit OK")
print("pytdc OK")
print("transformers OK")
PY
