- 처음 가상환경 설치 및 세팅
```
conda create -n mlbio_hw3_a100 python=3.10 -y
conda activate mlbio_hw3_a100

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

- 가상환경 세팅 확인
```
python - <<'PY'
import torch
import esm
from rdkit import Chem
from tdc.multi_pred import DTI
from transformers import AutoTokenizer, AutoModel

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("esm import OK")
print("rdkit import OK")
print("pytdc import OK")
print("transformers import OK")
PY
```