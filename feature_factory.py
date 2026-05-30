"""
Feature extraction helper for ML in Bioinformatics HW3.

Purpose
-------
Keep the main notebook clean while running feature ablations such as:

- Morgan only / Morgan + MolFormer
- MolFormer selected layer embeddings, e.g. [12] or [3, 6, 12]
- ESM2 selected layer embeddings, e.g. [3] or [1, 3, 6]
- ESM2 long-protein handling with sliding windows over sequences longer than 1022 aa

This file keeps feature extraction and optional student-model training helpers
in one importable module. The training helpers preserve the notebook split,
random seed, train-only label standardization, and evaluation metric function
when called with the TA-provided `splits` and `compute_metrics`.

Dependencies already expected in the HW3 environment:
    rdkit, transformers, fair-esm imported as esm, torch, numpy, pandas, tqdm

Do NOT install the PyPI package named `esm`; HW3 uses `fair-esm`, whose import
name is `esm`.

Patch note: training loss casts predictions/targets to float32 before MSELoss so
BF16 autocast on H100 does not trigger dtype-mismatch errors during backward.
"""

from __future__ import annotations

import pickle
import re
import copy
import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


ArrayDict = Dict[str, np.ndarray]
LayerSpec = Union[str, Sequence[int], int, None]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _as_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _safe_name(text: str) -> str:
    """Make a string safe for cache filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _ensure_float32_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _chunked(items: Sequence[int], chunk_size: Optional[int]) -> Iterable[List[int]]:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(items):
        yield list(items)
        return

    for start in range(0, len(items), chunk_size):
        yield list(items[start:start + chunk_size])


def suggest_low_mid_high_layers(num_layers: int) -> List[int]:
    """
    Return numeric low/mid/high layer IDs for a model with `num_layers` layers.

    Examples:
        ESM2-t6  -> [1, 3, 6]
        ESM2-t12 -> [3, 6, 12]
        12-layer MolFormer -> [3, 6, 12]
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    low = max(1, int(round(num_layers * 0.25)))
    mid = max(1, int(round(num_layers * 0.50)))
    high = num_layers

    # Preserve order and remove accidental duplicates for tiny models.
    out: List[int] = []
    for layer in [low, mid, high]:
        if layer not in out:
            out.append(layer)
    return out


def normalize_layer_spec(layer_spec: LayerSpec, num_layers: int) -> List[int]:
    """
    Normalize layer selection.

    Accepted forms:
        None          -> final layer only, [num_layers]
        int           -> [int]
        [1, 3, 6]     -> same numeric list
        "final"       -> [num_layers]
        "high"        -> [num_layers]
        "low_mid_high"-> numeric low/mid/high suggestion
        "all"         -> [1, ..., num_layers]

    The main recommended usage for the notebook is a numeric list, e.g. [3].
    """
    if layer_spec is None:
        layers = [num_layers]
    elif isinstance(layer_spec, int):
        layers = [layer_spec]
    elif isinstance(layer_spec, str):
        key = layer_spec.lower().replace("-", "_").strip()
        if key in {"final", "high", "last"}:
            layers = [num_layers]
        elif key in {"low_mid_high", "lmh", "lowmidhigh"}:
            layers = suggest_low_mid_high_layers(num_layers)
        elif key == "all":
            layers = list(range(1, num_layers + 1))
        else:
            raise ValueError(
                f"Unknown layer spec {layer_spec!r}. Use numeric list, 'final', "
                "'low_mid_high', or 'all'."
            )
    else:
        layers = [int(x) for x in layer_spec]

    bad = [x for x in layers if x < 1 or x > num_layers]
    if bad:
        raise ValueError(
            f"Invalid layer IDs {bad}. This model has valid layer IDs 1..{num_layers}."
        )

    return layers


def slice_feature_dict(feature_dict: Mapping[str, Any], start: int, end: int) -> ArrayDict:
    """Slice each vector in a feature dictionary."""
    out: ArrayDict = {}
    for k, v in feature_dict.items():
        arr = _ensure_float32_1d(v)
        if arr.shape[0] < end:
            raise ValueError(
                f"Feature for key {k!r} has dim {arr.shape[0]}, "
                f"but requested slice [{start}:{end}]."
            )
        out[str(k)] = arr[start:end].astype(np.float32)
    return out


def concat_feature_dicts_by_key(
    left_features: Mapping[str, Any],
    right_features: Mapping[str, Any],
    required_keys: Optional[Iterable[str]] = None,
    left_name: str = "left",
    right_name: str = "right",
) -> ArrayDict:
    """
    Concatenate two feature dictionaries key-by-key.

    This is useful for Morgan + MolFormer drug features.
    """
    if required_keys is None:
        keys = sorted(set(map(str, left_features.keys())) | set(map(str, right_features.keys())))
    else:
        keys = sorted([str(k) for k in required_keys])

    missing_left = [k for k in keys if k not in left_features]
    missing_right = [k for k in keys if k not in right_features]

    if missing_left:
        raise KeyError(f"{left_name} is missing {len(missing_left)} keys. Example: {missing_left[:3]}")
    if missing_right:
        raise KeyError(f"{right_name} is missing {len(missing_right)} keys. Example: {missing_right[:3]}")

    out: ArrayDict = {}
    for k in keys:
        left_vec = _ensure_float32_1d(left_features[k])
        right_vec = _ensure_float32_1d(right_features[k])
        out[k] = np.concatenate([left_vec, right_vec], axis=0).astype(np.float32)
    return out


def build_xy_from_feature_dicts(
    dataframe: pd.DataFrame,
    drug_features: Mapping[str, Any],
    protein_features: Mapping[str, Any],
    drug_col: str = "Drug",
    target_col: str = "Target",
    label_col: str = "Y",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model-ready arrays compatible with the TA skeleton.

    Returns:
        drug_x:    float32 array [num_pairs, drug_dim]
        protein_x: float32 array [num_pairs, protein_dim]
        y:         float32 array [num_pairs]
    """
    drug_x = np.stack([
        _ensure_float32_1d(drug_features[str(x)]) for x in dataframe[drug_col].values
    ]).astype(np.float32)

    protein_x = np.stack([
        _ensure_float32_1d(protein_features[str(x)]) for x in dataframe[target_col].values
    ]).astype(np.float32)

    y = dataframe[label_col].values.astype(np.float32)
    return drug_x, protein_x, y


# -----------------------------------------------------------------------------
# Morgan fingerprint
# -----------------------------------------------------------------------------


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint numpy vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def build_or_load_morgan_features(
    dataframe: pd.DataFrame,
    cache_path: Union[str, Path],
    drug_col: str = "Drug",
    radius: int = 2,
    n_bits: int = 1024,
    force_rebuild: bool = False,
) -> ArrayDict:
    cache_path = _as_path(cache_path)

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached Morgan features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    unique_smiles = sorted([str(x) for x in dataframe[drug_col].dropna().unique()])

    features: ArrayDict = {}
    for smi in tqdm(unique_smiles, desc="Morgan fingerprints"):
        features[smi] = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)

    return features


# -----------------------------------------------------------------------------
# MolFormer
# -----------------------------------------------------------------------------


MOLFORMER_MODEL_CANDIDATES = [
    "ibm-research/MoLFormer-XL-both-10pct",
    "ibm/MoLFormer-XL-both-10pct",
]


def canonicalize_smiles_for_molformer(smiles: str, remove_isomeric: bool = True) -> Optional[str]:
    """
    RDKit-canonicalize a SMILES for MolFormer input.

    The original dataframe SMILES remains the dictionary key.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_isomeric)


def load_molformer_model(
    device: torch.device,
    model_name_or_candidates: Union[str, Sequence[str], None] = None,
):
    """Load MolFormer tokenizer/model with robust fallback candidates."""
    from transformers import AutoModel, AutoTokenizer

    if model_name_or_candidates is None:
        candidates = MOLFORMER_MODEL_CANDIDATES
    elif isinstance(model_name_or_candidates, str):
        candidates = [model_name_or_candidates]
    else:
        candidates = list(model_name_or_candidates)

    last_error = None
    for model_name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    deterministic_eval=True,
                    trust_remote_code=True,
                )
            except TypeError:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

            model.eval()
            model.to(device)
            print(f"Loaded MolFormer: {model_name}")
            return tokenizer, model, model_name
        except Exception as e:  # pragma: no cover - useful in notebooks
            last_error = e
            print(f"Failed to load MolFormer candidate {model_name}: {repr(e)}")

    raise RuntimeError(f"Could not load any MolFormer candidate. Last error: {last_error}")


def _masked_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _pool_molformer_outputs(outputs: Any, attention_mask: torch.Tensor, pooling: str = "pooler") -> torch.Tensor:
    """
    Pool MolFormer outputs.

    pooling='pooler': use pooler_output if available, else masked mean.
    pooling='mean':   masked mean over token states.
    pooling='cls':    first token representation.
    """
    pooling = pooling.lower()

    if pooling == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    hidden = outputs.last_hidden_state

    if pooling in {"pooler", "mean"}:
        return _masked_mean_pool(hidden, attention_mask)
    if pooling == "cls":
        return hidden[:, 0, :]

    raise ValueError(f"Unknown MolFormer pooling: {pooling!r}")


def _molformer_forward_hidden_states(model: torch.nn.Module, inputs: Mapping[str, torch.Tensor]) -> Any:
    """Run MolFormer and request hidden_states."""
    try:
        outputs = model(**inputs, output_hidden_states=True)
    except TypeError:
        old_flag = getattr(model.config, "output_hidden_states", False)
        model.config.output_hidden_states = True
        outputs = model(**inputs)
        model.config.output_hidden_states = old_flag

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError(
            "MolFormer did not return hidden_states. Check whether the loaded HF "
            "model supports output_hidden_states=True."
        )
    return outputs


def build_or_load_molformer_layer_bank(
    dataframe: pd.DataFrame,
    cache_path: Union[str, Path],
    drug_col: str = "Drug",
    model_name_or_candidates: Union[str, Sequence[str], None] = None,
    max_length: int = 202,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    canonicalize: bool = True,
    remove_isomeric: bool = True,
    pooling: str = "mean",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build/load pooled MolFormer embeddings for ALL transformer layers.

    Cache format:
        {
            "kind": "molformer_layer_bank",
            "model_name": used_model_name,
            "layers": [1, 2, ..., n_layers],
            "hidden_dim": hidden_dim,
            "pooling": pooling,
            "features": {original_smiles: np.ndarray [n_layers, hidden_dim]},
        }

    Later, select arbitrary numeric layers with `select_layers_from_bank`.
    """
    cache_path = _as_path(cache_path)

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached MolFormer layer bank from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and "features" in cached and "layers" in cached:
            return cached
        raise ValueError(
            f"Cache at {cache_path} exists but is not a layer-bank cache. "
            "Use another cache path or force_rebuild=True."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cpu")

    tokenizer, model, used_model_name = load_molformer_model(device, model_name_or_candidates)
    original_smiles = sorted([str(x) for x in dataframe[drug_col].dropna().unique()])

    prepared: List[Tuple[str, str]] = []
    invalid: List[str] = []
    for smi in original_smiles:
        model_smi = canonicalize_smiles_for_molformer(smi, remove_isomeric=remove_isomeric) if canonicalize else smi
        if model_smi is None:
            invalid.append(smi)
        else:
            prepared.append((smi, model_smi))

    print(f"Unique drugs: {len(original_smiles)}")
    print(f"Valid SMILES for MolFormer: {len(prepared)}")
    print(f"Invalid SMILES: {len(invalid)}")

    features: ArrayDict = {}
    layer_ids: Optional[List[int]] = None
    hidden_dim: Optional[int] = None

    with torch.no_grad():
        for start in tqdm(range(0, len(prepared), batch_size), desc="MolFormer all-layer bank"):
            batch_pairs = prepared[start:start + batch_size]
            original_batch = [p[0] for p in batch_pairs]
            model_batch = [p[1] for p in batch_pairs]

            inputs = tokenizer(
                model_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = _molformer_forward_hidden_states(model, inputs)
            hidden_states = outputs.hidden_states
            n_layers = len(hidden_states) - 1  # hidden_states[0] is embedding output

            if layer_ids is None:
                layer_ids = list(range(1, n_layers + 1))
                hidden_dim = int(hidden_states[1].shape[-1])
                print("MolFormer transformer layers:", n_layers)
                print("MolFormer hidden dim:", hidden_dim)

            pooled_by_layer: List[torch.Tensor] = []
            for layer in layer_ids:
                hidden = hidden_states[layer]
                if pooling == "mean":
                    pooled = _masked_mean_pool(hidden, inputs["attention_mask"])
                elif pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    raise ValueError("Layer-bank MolFormer pooling must be 'mean' or 'cls'.")
                pooled_by_layer.append(pooled.detach().cpu().float())

            # [B, n_layers, hidden_dim]
            batch_bank = torch.stack(pooled_by_layer, dim=1).numpy().astype(np.float32)

            for smi, bank in zip(original_batch, batch_bank):
                features[smi] = bank

    if layer_ids is None or hidden_dim is None:
        # Empty dataset fallback.
        layer_ids = []
        hidden_dim = int(getattr(model.config, "hidden_size", 768) or 768)

    zero_bank = np.zeros((len(layer_ids), hidden_dim), dtype=np.float32)
    for smi in invalid:
        features[smi] = zero_bank.copy()

    cache = {
        "kind": "molformer_layer_bank",
        "model_name": used_model_name,
        "layers": layer_ids,
        "hidden_dim": hidden_dim,
        "pooling": pooling,
        "features": features,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved MolFormer layer bank to {cache_path}")
    return cache


# -----------------------------------------------------------------------------
# ESM2
# -----------------------------------------------------------------------------


ESM2_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "t6": {"fn": "esm2_t6_8M_UR50D", "num_layers": 6, "dim": 320, "max_len": 1022},
    "8m": {"fn": "esm2_t6_8M_UR50D", "num_layers": 6, "dim": 320, "max_len": 1022},
    "esm2_t6_8M_UR50D": {"fn": "esm2_t6_8M_UR50D", "num_layers": 6, "dim": 320, "max_len": 1022},

    "t12": {"fn": "esm2_t12_35M_UR50D", "num_layers": 12, "dim": 480, "max_len": 1022},
    "35m": {"fn": "esm2_t12_35M_UR50D", "num_layers": 12, "dim": 480, "max_len": 1022},
    "esm2_t12_35M_UR50D": {"fn": "esm2_t12_35M_UR50D", "num_layers": 12, "dim": 480, "max_len": 1022},

    "t30": {"fn": "esm2_t30_150M_UR50D", "num_layers": 30, "dim": 640, "max_len": 1022},
    "150m": {"fn": "esm2_t30_150M_UR50D", "num_layers": 30, "dim": 640, "max_len": 1022},
    "esm2_t30_150M_UR50D": {"fn": "esm2_t30_150M_UR50D", "num_layers": 30, "dim": 640, "max_len": 1022},

    "t33": {"fn": "esm2_t33_650M_UR50D", "num_layers": 33, "dim": 1280, "max_len": 1022},
    "650m": {"fn": "esm2_t33_650M_UR50D", "num_layers": 33, "dim": 1280, "max_len": 1022},
    "esm2_t33_650M_UR50D": {"fn": "esm2_t33_650M_UR50D", "num_layers": 33, "dim": 1280, "max_len": 1022},
}


def get_esm2_spec(model_size: str) -> Dict[str, Any]:
    key = str(model_size).lower()
    if key not in ESM2_MODEL_SPECS:
        valid = sorted(set(ESM2_MODEL_SPECS.keys()))
        raise ValueError(f"Unknown ESM2 model_size={model_size!r}. Valid keys include: {valid}")
    return dict(ESM2_MODEL_SPECS[key])


def load_esm2_model(model_size: str, device: torch.device):
    """Load an ESM2 model from fair-esm. No PyPI `esm` package is required."""
    import esm  # fair-esm import name

    spec = get_esm2_spec(model_size)
    loader = getattr(esm.pretrained, spec["fn"])
    model, alphabet = loader()
    model.eval()
    model.to(device)
    print(f"Loaded ESM2: {spec['fn']} | layers={spec['num_layers']} | dim={spec['dim']}")
    return model, alphabet, spec


def make_esm_windows(seq: str, window_len: int = 1022, stride: int = 768) -> List[Tuple[int, str]]:
    """
    Sliding windows that cover the full protein sequence.

    The final window is shifted to include the C-terminal end.
    """
    seq = str(seq)
    L = len(seq)

    if L <= window_len:
        return [(0, seq)]

    starts = [0]
    while starts[-1] + window_len < L:
        next_start = min(starts[-1] + stride, L - window_len)
        if next_start <= starts[-1]:
            break
        starts.append(next_start)

    return [(s, seq[s:s + window_len]) for s in starts]


def _precompute_window_counts(length: int, windows: Sequence[Tuple[int, str]]) -> torch.Tensor:
    counts = torch.zeros((length, 1), dtype=torch.float32)
    for pos, chunk in windows:
        counts[pos:pos + len(chunk)] += 1.0
    return counts.clamp(min=1.0)


def build_or_load_esm2_layer_bank(
    dataframe: pd.DataFrame,
    cache_path: Union[str, Path],
    target_col: str = "Target",
    model_size: str = "t6",
    window_len: int = 1022,
    stride: int = 768,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    poolings: Sequence[str] = ("mean", "max"),
    overlap_strategy: str = "position",
    repr_layer_chunk_size: Optional[int] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build/load pooled ESM2 embeddings for ALL layers of the selected model.

    Cache format:
        {
            "kind": "esm2_layer_bank",
            "model_size": "t6",
            "model_fn": "esm2_t6_8M_UR50D",
            "layers": [1, 2, ..., n_layers],
            "hidden_dim": 320,
            "poolings": ["mean", "max"],
            "per_layer_dim": 640,
            "window_len": 1022,
            "stride": 768,
            "features": {protein_sequence: np.ndarray [n_layers, per_layer_dim]},
        }

    Long proteins are processed with sliding windows.

    overlap_strategy:
        "position"        -> de-duplicate overlapping windows by averaging each
                             residue position before global pooling. More accurate.
        "window_weighted" -> length-weighted mean over windows. Lower memory.

    For t30/t33, caching every layer can be slow/heavy. If memory is tight, pass
    repr_layer_chunk_size=4 or 6, which repeats forward passes but stores fewer
    layer activations at once.
    """
    cache_path = _as_path(cache_path)
    spec = get_esm2_spec(model_size)
    num_layers = int(spec["num_layers"])
    hidden_dim = int(spec["dim"])
    layer_ids = list(range(1, num_layers + 1))
    poolings = tuple(p.lower() for p in poolings)

    valid_poolings = {"mean", "max"}
    unknown_poolings = [p for p in poolings if p not in valid_poolings]
    if unknown_poolings:
        raise ValueError(f"Unknown ESM pooling(s): {unknown_poolings}. Use 'mean' and/or 'max'.")

    per_layer_dim = hidden_dim * len(poolings)

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached ESM2 layer bank from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and "features" in cached and "layers" in cached:
            return cached
        raise ValueError(
            f"Cache at {cache_path} exists but is not an ESM2 layer-bank cache. "
            "Use another cache path or force_rebuild=True."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cpu")

    if model_size.lower() in {"t30", "150m", "t33", "650m"}:
        print(
            "Warning: selected ESM2 size is large. Full all-layer caching may be slow/heavy. "
            "Use batch_size=1 and repr_layer_chunk_size=4 if memory is tight."
        )

    model, alphabet, spec = load_esm2_model(model_size, device)
    batch_converter = alphabet.get_batch_converter()

    unique_targets = sorted([str(x) for x in dataframe[target_col].dropna().unique()])
    lengths = [len(x) for x in unique_targets]
    if lengths:
        print("Unique targets:", len(unique_targets))
        print("Max target length:", max(lengths))
        print(f"Targets longer than {window_len} aa:", sum(L > window_len for L in lengths))

    features: ArrayDict = {}

    with torch.no_grad():
        for target_idx, seq in enumerate(tqdm(unique_targets, desc="ESM2 all-layer window bank")):
            seq = str(seq)
            L = len(seq)

            if L == 0:
                features[seq] = np.zeros((num_layers, per_layer_dim), dtype=np.float32)
                continue

            windows = make_esm_windows(seq, window_len=window_len, stride=stride)

            if overlap_strategy == "position":
                counts = _precompute_window_counts(L, windows)
                layer_sums = {
                    layer: torch.zeros((L, hidden_dim), dtype=torch.float32)
                    for layer in layer_ids
                }
                layer_maxs = {
                    layer: torch.full((hidden_dim,), -float("inf"), dtype=torch.float32)
                    for layer in layer_ids
                }
            elif overlap_strategy == "window_weighted":
                layer_weighted_mean_sums = {
                    layer: torch.zeros((hidden_dim,), dtype=torch.float32)
                    for layer in layer_ids
                }
                layer_weight_sums = {layer: 0.0 for layer in layer_ids}
                layer_maxs = {
                    layer: torch.full((hidden_dim,), -float("inf"), dtype=torch.float32)
                    for layer in layer_ids
                }
            else:
                raise ValueError("overlap_strategy must be 'position' or 'window_weighted'.")

            for start in range(0, len(windows), batch_size):
                batch_windows = windows[start:start + batch_size]
                batch_data = [
                    (f"target_{target_idx}_window_{start + j}_pos_{pos}", chunk)
                    for j, (pos, chunk) in enumerate(batch_windows)
                ]
                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)

                for layer_chunk in _chunked(layer_ids, repr_layer_chunk_size):
                    results = model(
                        batch_tokens,
                        repr_layers=list(layer_chunk),
                        return_contacts=False,
                    )

                    reps_by_layer = {
                        layer: results["representations"][layer].detach().cpu().float()
                        for layer in layer_chunk
                    }

                    for i, (pos, chunk) in enumerate(batch_windows):
                        chunk_len = len(chunk)
                        s0 = pos
                        s1 = pos + chunk_len

                        for layer in layer_chunk:
                            residue_repr = reps_by_layer[layer][i, 1:chunk_len + 1]

                            if overlap_strategy == "position":
                                layer_sums[layer][s0:s1] += residue_repr
                            else:
                                layer_weighted_mean_sums[layer] += residue_repr.mean(dim=0) * float(chunk_len)
                                layer_weight_sums[layer] += float(chunk_len)

                            if "max" in poolings:
                                layer_maxs[layer] = torch.maximum(
                                    layer_maxs[layer],
                                    residue_repr.max(dim=0).values,
                                )

                    del results, reps_by_layer

            layer_features: List[torch.Tensor] = []
            for layer in layer_ids:
                parts: List[torch.Tensor] = []

                if "mean" in poolings:
                    if overlap_strategy == "position":
                        per_position_avg = layer_sums[layer] / counts
                        layer_mean = per_position_avg.mean(dim=0)
                    else:
                        denom = max(layer_weight_sums[layer], 1.0)
                        layer_mean = layer_weighted_mean_sums[layer] / denom
                    parts.append(layer_mean)

                if "max" in poolings:
                    parts.append(layer_maxs[layer])

                layer_features.append(torch.cat(parts, dim=0))

            # [num_layers, per_layer_dim]
            seq_bank = torch.stack(layer_features, dim=0).numpy().astype(np.float32)
            features[seq] = seq_bank

    cache = {
        "kind": "esm2_layer_bank",
        "model_size": model_size,
        "model_fn": spec["fn"],
        "layers": layer_ids,
        "hidden_dim": hidden_dim,
        "poolings": list(poolings),
        "per_layer_dim": per_layer_dim,
        "window_len": window_len,
        "stride": stride,
        "overlap_strategy": overlap_strategy,
        "features": features,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved ESM2 layer bank to {cache_path}")
    return cache


# -----------------------------------------------------------------------------
# Layer-bank selection and final feature pack
# -----------------------------------------------------------------------------


def select_layers_from_bank(bank: Mapping[str, Any], layers: LayerSpec) -> ArrayDict:
    """
    Select numeric layers from a MolFormer or ESM2 layer-bank cache.

    Returns a feature dict where selected layer vectors are concatenated:
        selected [3, 6, 12] -> [layer3, layer6, layer12]
    """
    layer_ids = list(bank["layers"])
    if not layer_ids:
        raise ValueError("Layer bank has no layers.")

    max_layer = max(layer_ids)
    selected_layers = normalize_layer_spec(layers, max_layer)
    row_by_layer = {layer: i for i, layer in enumerate(layer_ids)}

    missing = [layer for layer in selected_layers if layer not in row_by_layer]
    if missing:
        raise ValueError(f"Layer bank does not contain requested layers: {missing}")

    features = bank["features"]
    out: ArrayDict = {}
    for k, arr in features.items():
        arr = np.asarray(arr, dtype=np.float32)
        selected = [arr[row_by_layer[layer]].reshape(-1) for layer in selected_layers]
        out[str(k)] = np.concatenate(selected, axis=0).astype(np.float32)
    return out


@dataclass
class FeaturePack:
    """Container returned by HW3FeatureFactory.build_feature_pack(...)."""

    drug_features: ArrayDict
    protein_features: ArrayDict
    config: Dict[str, Any]
    drug_col: str = "Drug"
    target_col: str = "Target"
    label_col: str = "Y"

    @property
    def drug_dim(self) -> int:
        return int(next(iter(self.drug_features.values())).shape[0])

    @property
    def protein_dim(self) -> int:
        return int(next(iter(self.protein_features.values())).shape[0])

    def build_xy(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return build_xy_from_feature_dicts(
            dataframe,
            self.drug_features,
            self.protein_features,
            drug_col=self.drug_col,
            target_col=self.target_col,
            label_col=self.label_col,
        )

    def make_xy_builder(self):
        def _xy_builder(dataframe: pd.DataFrame):
            return self.build_xy(dataframe)
        return _xy_builder


class HW3FeatureFactory:
    """
    Main user-facing feature builder.

    Typical notebook use:
        factory = HW3FeatureFactory(df, DATASET_NAME, CACHE_DIR, device=device)
        pack = factory.build_feature_pack(
            use_morgan=True,
            morgan_features=drug_features,        # reuse TA Morgan cache if already built
            use_molformer=True,
            molformer_layers=[12],                # or [3, 6, 12]
            esm_model_size="t6",
            esm_layers=[3],                       # ESM2-mid for t6
            esm_poolings=("mean", "max"),
            esm_window_len=1022,
            esm_stride=768,
        )
        student_drug_features = pack.drug_features
        student_protein_features = pack.protein_features
        STUDENT_DRUG_DIM = pack.drug_dim
        STUDENT_PROTEIN_DIM = pack.protein_dim
        student_xy_builder = pack.make_xy_builder()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        cache_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        drug_col: str = "Drug",
        target_col: str = "Target",
        label_col: str = "Y",
    ):
        self.df = dataframe
        self.dataset_name = str(dataset_name)
        self.cache_dir = _as_path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device is not None else torch.device("cpu")
        self.drug_col = drug_col
        self.target_col = target_col
        self.label_col = label_col

    def _default_morgan_cache(self, n_bits: int, radius: int) -> Path:
        return self.cache_dir / f"{self.dataset_name.lower()}_drug_morgan_r{radius}_{n_bits}.pkl"

    def _default_molformer_bank_cache(
        self,
        model_name: Optional[Union[str, Sequence[str]]],
        pooling: str,
        max_length: int,
        canonicalize: bool,
        remove_isomeric: bool,
    ) -> Path:
        if model_name is None:
            model_key = "molformer_xl_both10pct"
        elif isinstance(model_name, str):
            model_key = _safe_name(model_name)
        else:
            model_key = _safe_name(str(list(model_name)[0]))
        iso_key = "noiso" if remove_isomeric else "iso"
        canon_key = "canon" if canonicalize else "raw"
        return self.cache_dir / (
            f"{self.dataset_name.lower()}_drug_{model_key}_all_layers_"
            f"{pooling}_len{max_length}_{canon_key}_{iso_key}.pkl"
        )

    def _default_esm_bank_cache(
        self,
        model_size: str,
        poolings: Sequence[str],
        window_len: int,
        stride: int,
        overlap_strategy: str,
    ) -> Path:
        pool_key = "".join(poolings)
        return self.cache_dir / (
            f"{self.dataset_name.lower()}_protein_esm2_{_safe_name(model_size)}_"
            f"all_layers_{pool_key}_win{window_len}_stride{stride}_{overlap_strategy}.pkl"
        )

    def build_morgan(
        self,
        n_bits: int = 1024,
        radius: int = 2,
        cache_path: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> ArrayDict:
        if cache_path is None:
            cache_path = self._default_morgan_cache(n_bits=n_bits, radius=radius)
        return build_or_load_morgan_features(
            self.df,
            cache_path=cache_path,
            drug_col=self.drug_col,
            radius=radius,
            n_bits=n_bits,
            force_rebuild=force_rebuild,
        )

    def build_molformer_layer_bank(
        self,
        model_name_or_candidates: Union[str, Sequence[str], None] = None,
        max_length: int = 202,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
        canonicalize: bool = True,
        remove_isomeric: bool = True,
        pooling: str = "mean",
        cache_path: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        if cache_path is None:
            cache_path = self._default_molformer_bank_cache(
                model_name=model_name_or_candidates,
                pooling=pooling,
                max_length=max_length,
                canonicalize=canonicalize,
                remove_isomeric=remove_isomeric,
            )
        return build_or_load_molformer_layer_bank(
            self.df,
            cache_path=cache_path,
            drug_col=self.drug_col,
            model_name_or_candidates=model_name_or_candidates,
            max_length=max_length,
            batch_size=batch_size,
            device=device if device is not None else self.device,
            canonicalize=canonicalize,
            remove_isomeric=remove_isomeric,
            pooling=pooling,
            force_rebuild=force_rebuild,
        )

    def build_esm2_layer_bank(
        self,
        model_size: str = "t6",
        window_len: int = 1022,
        stride: int = 768,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        poolings: Sequence[str] = ("mean", "max"),
        overlap_strategy: str = "position",
        repr_layer_chunk_size: Optional[int] = None,
        cache_path: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        if cache_path is None:
            cache_path = self._default_esm_bank_cache(
                model_size=model_size,
                poolings=poolings,
                window_len=window_len,
                stride=stride,
                overlap_strategy=overlap_strategy,
            )
        return build_or_load_esm2_layer_bank(
            self.df,
            cache_path=cache_path,
            target_col=self.target_col,
            model_size=model_size,
            window_len=window_len,
            stride=stride,
            batch_size=batch_size,
            device=device if device is not None else self.device,
            poolings=poolings,
            overlap_strategy=overlap_strategy,
            repr_layer_chunk_size=repr_layer_chunk_size,
            force_rebuild=force_rebuild,
        )

    def build_feature_pack(
        self,
        *,
        # Drug options
        use_morgan: bool = True,
        morgan_features: Optional[Mapping[str, Any]] = None,
        morgan_bits: int = 1024,
        morgan_radius: int = 2,
        use_molformer: bool = True,
        molformer_layers: LayerSpec = None,
        molformer_model_name_or_candidates: Union[str, Sequence[str], None] = None,
        molformer_max_length: int = 202,
        molformer_batch_size: int = 64,
        molformer_device: Optional[torch.device] = None,
        molformer_canonicalize: bool = True,
        molformer_remove_isomeric: bool = True,
        molformer_pooling: str = "mean",
        molformer_bank_cache_path: Optional[Union[str, Path]] = None,
        # Protein options
        esm_model_size: str = "t6",
        esm_layers: LayerSpec = None,
        esm_poolings: Sequence[str] = ("mean", "max"),
        esm_window_len: int = 1022,
        esm_stride: int = 768,
        esm_batch_size: int = 8,
        esm_device: Optional[torch.device] = None,
        esm_overlap_strategy: str = "position",
        esm_repr_layer_chunk_size: Optional[int] = None,
        esm_bank_cache_path: Optional[Union[str, Path]] = None,
        # General
        force_rebuild_molformer: bool = False,
        force_rebuild_esm: bool = False,
        force_rebuild_morgan: bool = False,
    ) -> FeaturePack:
        """
        Build final drug/protein feature dictionaries for the student model.

        Returns a FeaturePack with `drug_features`, `protein_features`, dimensions,
        and `build_xy(...)` / `make_xy_builder()` helpers.
        """
        required_drug_keys = [str(x) for x in self.df[self.drug_col].dropna().unique()]

        drug_parts: List[Tuple[str, ArrayDict]] = []

        if use_morgan:
            if morgan_features is None:
                morgan_features = self.build_morgan(
                    n_bits=morgan_bits,
                    radius=morgan_radius,
                    force_rebuild=force_rebuild_morgan,
                )
            morgan_features = {str(k): _ensure_float32_1d(v) for k, v in morgan_features.items()}
            drug_parts.append(("Morgan", dict(morgan_features)))

        molformer_bank = None
        selected_molformer_layers = None
        if use_molformer:
            molformer_bank = self.build_molformer_layer_bank(
                model_name_or_candidates=molformer_model_name_or_candidates,
                max_length=molformer_max_length,
                batch_size=molformer_batch_size,
                device=molformer_device,
                canonicalize=molformer_canonicalize,
                remove_isomeric=molformer_remove_isomeric,
                pooling=molformer_pooling,
                cache_path=molformer_bank_cache_path,
                force_rebuild=force_rebuild_molformer,
            )
            selected_molformer_layers = normalize_layer_spec(
                molformer_layers,
                max(molformer_bank["layers"]),
            )
            molformer_features = select_layers_from_bank(molformer_bank, selected_molformer_layers)
            drug_parts.append((f"MolFormer{selected_molformer_layers}", molformer_features))

        if not drug_parts:
            raise ValueError("At least one of use_morgan or use_molformer must be True.")

        # Concatenate drug parts in order: Morgan, then MolFormer, unless Morgan disabled.
        drug_features = drug_parts[0][1]
        drug_name = drug_parts[0][0]
        for part_name, part_features in drug_parts[1:]:
            drug_features = concat_feature_dicts_by_key(
                drug_features,
                part_features,
                required_keys=required_drug_keys,
                left_name=drug_name,
                right_name=part_name,
            )
            drug_name = f"{drug_name}+{part_name}"

        esm_bank = self.build_esm2_layer_bank(
            model_size=esm_model_size,
            window_len=esm_window_len,
            stride=esm_stride,
            batch_size=esm_batch_size,
            device=esm_device,
            poolings=esm_poolings,
            overlap_strategy=esm_overlap_strategy,
            repr_layer_chunk_size=esm_repr_layer_chunk_size,
            cache_path=esm_bank_cache_path,
            force_rebuild=force_rebuild_esm,
        )
        selected_esm_layers = normalize_layer_spec(esm_layers, max(esm_bank["layers"]))
        protein_features = select_layers_from_bank(esm_bank, selected_esm_layers)

        pack = FeaturePack(
            drug_features=drug_features,
            protein_features=protein_features,
            config={
                "drug_order": [name for name, _ in drug_parts],
                "use_morgan": use_morgan,
                "use_molformer": use_molformer,
                "molformer_layers": selected_molformer_layers,
                "molformer_pooling": molformer_pooling,
                "esm_model_size": esm_model_size,
                "esm_layers": selected_esm_layers,
                "esm_poolings": list(esm_poolings),
                "esm_window_len": esm_window_len,
                "esm_stride": esm_stride,
                "esm_overlap_strategy": esm_overlap_strategy,
            },
            drug_col=self.drug_col,
            target_col=self.target_col,
            label_col=self.label_col,
        )

        print("Feature pack built.")
        print("  Drug feature dim:", pack.drug_dim)
        print("  Protein feature dim:", pack.protein_dim)
        print("  Config:", pack.config)
        return pack

# =============================================================================
# Student model and memory-efficient training helpers
# =============================================================================
# These helpers are intended for heavy student experiments where the protein
# feature vector is large, e.g. ESM2-t33 all-layer flattened features:
#     protein_x shape = [B, 33 * (1280 mean + 1280 max)] = [B, 84480]
# The model reshapes this flattened input back to [B, 33, 2560] internally.
#
# The training loop below preserves the important TA logic:
#   - train/valid/test split is supplied by the notebook and not modified
#   - label standardization uses train split statistics only
#   - metrics are computed by the notebook's compute_metrics function
# It only changes how large feature matrices are served to the model:
# unique drug/protein feature banks are stored once, and pair datasets carry IDs.

import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block used in the student model."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.15):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CrossAttentionBlock(nn.Module):
    """One pre-norm cross-attention block plus feed-forward residual block."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.15):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(memory_tokens)
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )
        query_tokens = query_tokens + self.drop(attn_out)
        query_tokens = query_tokens + self.ff(self.ff_norm(query_tokens))
        return query_tokens


class ESMAllLayerFusionDTA(nn.Module):
    """
    Heavy all-layer ESM2 fusion model for DTA regression.

    Expected drug_x:
        MolFormer final-layer embedding only, shape [B, 768].

    Expected protein_x:
        Flattened all-layer ESM2 features.
        For ESM2-t33 with mean+max pooling:
            protein_dim = 33 * (1280 + 1280) = 84480
        The model reshapes this to [B, 33, 2560].

    Compatible with the TA-style construction:
        model_class(drug_dim=..., protein_dim=..., **model_kwargs)
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        esm_num_layers: int = 33,
        d_model: int = 512,
        n_heads: int = 8,
        protein_encoder_layers: int = 2,
        cross_layers: int = 2,
        hidden_dim: int = 1024,
        dropout: float = 0.15,
    ):
        super().__init__()

        if protein_dim % esm_num_layers != 0:
            raise ValueError(
                f"protein_dim={protein_dim} is not divisible by "
                f"esm_num_layers={esm_num_layers}."
            )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")

        self.drug_dim = int(drug_dim)
        self.protein_dim = int(protein_dim)
        self.esm_num_layers = int(esm_num_layers)
        self.esm_per_layer_dim = int(protein_dim // esm_num_layers)
        self.d_model = int(d_model)

        # Drug side: a single MolFormer final-layer vector becomes one query token.
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(d_model, dropout=dropout),
        )

        # Protein side: each ESM layer vector becomes one layer token.
        self.protein_layer_proj = nn.Sequential(
            nn.Linear(self.esm_per_layer_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.protein_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layer_embed = nn.Parameter(torch.zeros(1, esm_num_layers + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.protein_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=protein_encoder_layers,
        )

        # Sample-wise learned layer gate over all ESM layers.
        self.layer_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Combine CLS token, gated layer summary, and max-over-layer summary.
        self.protein_summary = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(d_model, dropout=dropout),
        )

        # Drug query attends to protein layer tokens.
        self.drug_to_protein_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(cross_layers)
        ])

        # Drug-conditioned modulation of protein summary.
        self.film = nn.Linear(d_model, d_model * 2)
        self.film_norm = nn.LayerNorm(d_model)

        fusion_dim = d_model * 8
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        nn.init.normal_(self.protein_cls, std=0.02)
        nn.init.normal_(self.layer_embed, std=0.02)

    def forward(self, drug_x: torch.Tensor, protein_x: torch.Tensor) -> torch.Tensor:
        batch_size = drug_x.size(0)

        z_d = self.drug_proj(drug_x)       # [B, d_model]
        drug_query = z_d.unsqueeze(1)      # [B, 1, d_model]

        # Flattened protein feature -> ESM layer tokens.
        protein_layers = protein_x.reshape(
            batch_size,
            self.esm_num_layers,
            self.esm_per_layer_dim,
        )                                  # [B, 33, 2560] for ESM2-t33 mean+max

        protein_tokens = self.protein_layer_proj(protein_layers)
        protein_cls = self.protein_cls.expand(batch_size, -1, -1)
        protein_tokens = torch.cat([protein_cls, protein_tokens], dim=1)
        protein_tokens = protein_tokens + self.layer_embed
        protein_tokens = self.protein_encoder(protein_tokens)

        p_cls = protein_tokens[:, 0, :]
        p_layers = protein_tokens[:, 1:, :]

        gate_logits = self.layer_gate(p_layers).squeeze(-1)  # [B, num_layers]
        gate = torch.softmax(gate_logits, dim=1)
        p_gated = torch.sum(gate.unsqueeze(-1) * p_layers, dim=1)
        p_max = torch.max(p_layers, dim=1).values

        z_p = self.protein_summary(torch.cat([p_cls, p_gated, p_max], dim=1))

        q = drug_query
        for block in self.drug_to_protein_blocks:
            q = block(q, p_layers)
        d_ctx = q.squeeze(1)

        gamma, beta = self.film(z_d).chunk(2, dim=1)
        z_p_film = self.film_norm(z_p * (1.0 + torch.tanh(gamma)) + beta)

        pair_fusion = torch.cat(
            [
                z_d,
                z_p,
                d_ctx,
                z_p_film,
                z_d * z_p,
                torch.abs(z_d - z_p),
                z_d * d_ctx,
                torch.abs(z_d - d_ctx),
            ],
            dim=1,
        )

        return self.predictor(pair_fusion).squeeze(-1)


# Legacy alias kept for fallback. The final MyImprovedModel alias is defined at the bottom.
LegacyESMAllLayerFusionDTA = ESMAllLayerFusionDTA


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _torch_dtype_from_name(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key in {"float32", "fp32", "f32"}:
        return torch.float32
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {dtype!r}")


def make_feature_banks(
    drug_features: Mapping[str, Any],
    protein_features: Mapping[str, Any],
    drug_dtype: Optional[Union[str, torch.dtype]] = None,
    protein_dtype: Optional[Union[str, torch.dtype]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], Dict[str, int]]:
    """
    Build compact unique feature banks.

    Instead of materializing pair-level arrays like [num_pairs, 84480], store:
        drug_bank    [num_unique_drugs, drug_dim]
        protein_bank [num_unique_targets, protein_dim]
    and let the Dataset return integer IDs for each DTA pair.
    """
    drug_features = {str(k): _ensure_float32_1d(v) for k, v in drug_features.items()}
    protein_features = {str(k): _ensure_float32_1d(v) for k, v in protein_features.items()}

    drug_keys = sorted(drug_features.keys())
    protein_keys = sorted(protein_features.keys())

    drug_to_id = {k: i for i, k in enumerate(drug_keys)}
    protein_to_id = {k: i for i, k in enumerate(protein_keys)}

    drug_bank = torch.tensor(
        np.stack([drug_features[k] for k in drug_keys]).astype(np.float32),
        dtype=_torch_dtype_from_name(drug_dtype),
    )
    protein_bank = torch.tensor(
        np.stack([protein_features[k] for k in protein_keys]).astype(np.float32),
        dtype=_torch_dtype_from_name(protein_dtype),
    )

    print("drug_bank:", tuple(drug_bank.shape))
    print("protein_bank:", tuple(protein_bank.shape))
    return drug_bank, protein_bank, drug_to_id, protein_to_id


class PairIdDataset(Dataset):
    """DTA pair dataset that returns compact drug/protein IDs plus standardized y."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        drug_to_id: Mapping[str, int],
        protein_to_id: Mapping[str, int],
        y_values: np.ndarray,
        drug_col: str = "Drug",
        target_col: str = "Target",
    ):
        self.drug_ids = torch.tensor(
            [drug_to_id[str(x)] for x in dataframe[drug_col].values],
            dtype=torch.long,
        )
        self.protein_ids = torch.tensor(
            [protein_to_id[str(x)] for x in dataframe[target_col].values],
            dtype=torch.long,
        )
        self.y = torch.tensor(y_values, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.drug_ids[idx], self.protein_ids[idx], self.y[idx]


def _get_amp_dtype(device: torch.device, use_amp: bool = True) -> Optional[torch.dtype]:
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover - compatibility with older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _forward_with_optional_amp(
    model: nn.Module,
    drug_x: torch.Tensor,
    protein_x: torch.Tensor,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if amp_dtype is None:
        return model(drug_x, protein_x)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        return model(drug_x, protein_x)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[Mapping[str, Any]],
    epochs: int,
    steps_per_epoch: int,
):
    if scheduler_config is None:
        scheduler_config = {"type": "onecycle"}

    scheduler_type = str(scheduler_config.get("type", "onecycle")).lower()
    if scheduler_type in {"none", "off", "null"}:
        return None, None

    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(scheduler_config.get("max_lr", optimizer.param_groups[0]["lr"])),
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            pct_start=float(scheduler_config.get("pct_start", 0.10)),
            div_factor=float(scheduler_config.get("div_factor", 10.0)),
            final_div_factor=float(scheduler_config.get("final_div_factor", 100.0)),
        )
        return scheduler, "batch"

    if scheduler_type == "cosine":
        total_steps = max(1, int(epochs) * int(steps_per_epoch))
        warmup_steps = int(scheduler_config.get("warmup_steps", max(1, 2 * steps_per_epoch)))
        min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.05))

        def lr_lambda(step: int) -> float:
            step = step + 1
            if step <= warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, "batch"

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("lr_patience", 3)),
            threshold=float(scheduler_config.get("threshold", 1e-4)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
        return scheduler, "epoch"

    raise ValueError(f"Unknown scheduler type: {scheduler_type!r}")


@torch.no_grad()
def predict_idbank(
    model: nn.Module,
    loader: DataLoader,
    drug_bank_device: torch.Tensor,
    protein_bank_device: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for drug_ids, protein_ids, y in loader:
        drug_ids = drug_ids.to(device, non_blocking=True)
        protein_ids = protein_ids.to(device, non_blocking=True)
        drug_x = drug_bank_device[drug_ids]
        protein_x = protein_bank_device[protein_ids]
        pred = _forward_with_optional_amp(model, drug_x, protein_x, amp_dtype)
        preds.append(pred.detach().float().cpu().numpy())
        ys.append(y.numpy())

    return np.concatenate(ys), np.concatenate(preds)


def train_one_split_idbank(
    split_name: str,
    split: Mapping[str, pd.DataFrame],
    model_class: type,
    drug_bank: torch.Tensor,
    protein_bank: torch.Tensor,
    drug_to_id: Mapping[str, int],
    protein_to_id: Mapping[str, int],
    compute_metrics_fn,
    device: torch.device,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    max_grad_norm: float = 1.0,
    scheduler_config: Optional[Mapping[str, Any]] = None,
    amp: bool = True,
    bank_device_dtype: Optional[str] = None,
    save_model_dir: Optional[Union[str, Path]] = None,
    save_model_filename: str = "model.pth",
    num_workers: int = 0,
    drug_col: str = "Drug",
    target_col: str = "Target",
    label_col: str = "Y",
):
    """
    Train one split using compact feature banks.

    This is a student-model replacement for the TA train_one_split function when
    pair-level protein_x would be too large. It preserves the core label logic:
        y_mean/y_std are computed from split['train'] only.

    Pass the notebook's compute_metrics function as compute_metrics_fn to keep
    the exact same evaluation metric implementation.

    save_model_dir:
        Optional directory used to save the full trained model object after this
        split finishes. The model is saved with torch.save(model, path), not as
        a state_dict. To avoid overwriting across splits, the final path is
        <save_model_dir>/<split_name>/<save_model_filename>.
    """
    if compute_metrics_fn is None:
        raise ValueError("Pass the notebook's compute_metrics function as compute_metrics_fn.")
    if model_kwargs is None:
        model_kwargs = {}

    print(f"\n===== Training on split: {split_name} =====")

    y_train_raw = split["train"][label_col].values.astype(np.float32)
    y_valid_raw = split["valid"][label_col].values.astype(np.float32)
    y_test_raw = split["test"][label_col].values.astype(np.float32)

    # Same rule as TA skeleton: standardize using train split statistics only.
    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std() + 1e-8
    y_train = (y_train_raw - y_mean) / y_std
    y_valid = (y_valid_raw - y_mean) / y_std
    y_test = (y_test_raw - y_mean) / y_std

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        PairIdDataset(split["train"], drug_to_id, protein_to_id, y_train, drug_col, target_col),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        PairIdDataset(split["valid"], drug_to_id, protein_to_id, y_valid, drug_col, target_col),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        PairIdDataset(split["test"], drug_to_id, protein_to_id, y_test, drug_col, target_col),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    drug_dim = int(drug_bank.shape[1])
    protein_dim = int(protein_bank.shape[1])
    model = model_class(drug_dim=drug_dim, protein_dim=protein_dim, **dict(model_kwargs)).to(device)

    print("drug_dim:", drug_dim)
    print("protein_dim:", protein_dim)
    print("Trainable parameters:", count_trainable_parameters(model))

    amp_dtype = _get_amp_dtype(device, use_amp=amp)

    # Optional: keep huge feature banks in reduced precision on GPU to cut memory
    # bandwidth and speed up the gather/forward path. This does not change labels
    # or metrics; loss is still computed in float32.
    device_bank_dtype = None
    if bank_device_dtype is not None:
        key = str(bank_device_dtype).lower()
        if key in {"amp", "auto"}:
            device_bank_dtype = amp_dtype
        else:
            device_bank_dtype = _torch_dtype_from_name(bank_device_dtype)

    drug_bank_device = drug_bank.to(device, dtype=device_bank_dtype, non_blocking=True)
    protein_bank_device = protein_bank.to(device, dtype=device_bank_dtype, non_blocking=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler, scheduler_step_unit = _build_scheduler(
        optimizer,
        scheduler_config=scheduler_config,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    scaler = _make_grad_scaler(enabled=(amp_dtype == torch.float16))

    print("AMP dtype:", amp_dtype)
    print("Bank device dtype:", device_bank_dtype if device_bank_dtype is not None else "float32")
    print("Initial lr:", optimizer.param_groups[0]["lr"])
    print("Scheduler:", scheduler_config if scheduler_config is not None else {"type": "onecycle"})

    best_valid_loss = float("inf")
    best_state = None
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []

        for drug_ids, protein_ids, y in train_loader:
            drug_ids = drug_ids.to(device, non_blocking=True)
            protein_ids = protein_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            drug_x = drug_bank_device[drug_ids]
            protein_x = protein_bank_device[protein_ids]

            optimizer.zero_grad(set_to_none=True)
            pred = _forward_with_optional_amp(model, drug_x, protein_x, amp_dtype)
            loss = criterion(pred.float(), y.float())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            if scheduler is not None and scheduler_step_unit == "batch":
                scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for drug_ids, protein_ids, y in valid_loader:
                drug_ids = drug_ids.to(device, non_blocking=True)
                protein_ids = protein_ids.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                drug_x = drug_bank_device[drug_ids]
                protein_x = protein_bank_device[protein_ids]
                pred = _forward_with_optional_amp(model, drug_x, protein_x, amp_dtype)
                valid_losses.append(float(criterion(pred.float(), y.float()).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))

        if scheduler is not None and scheduler_step_unit == "epoch":
            scheduler.step(valid_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": float(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "lr": current_lr,
        })

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train {train_loss:.4f} | valid {valid_loss:.4f} | lr {current_lr:.2e}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)

    _, valid_pred_std = predict_idbank(
        model,
        valid_loader,
        drug_bank_device,
        protein_bank_device,
        device,
        amp_dtype=amp_dtype,
    )
    _, test_pred_std = predict_idbank(
        model,
        test_loader,
        drug_bank_device,
        protein_bank_device,
        device,
        amp_dtype=amp_dtype,
    )

    valid_pred = valid_pred_std * y_std + y_mean
    test_pred = test_pred_std * y_std + y_mean

    valid_metrics = compute_metrics_fn(y_valid_raw, valid_pred)
    test_metrics = compute_metrics_fn(y_test_raw, test_pred)

    print("Valid metrics:", valid_metrics)
    print("Test metrics:", test_metrics)

    saved_model_path = None
    if save_model_dir is not None:
        # Save one full model object per split. The filename itself stays
        # "model.pth", while the split-specific subdirectory prevents random,
        # cold_drug, and cold_target from overwriting each other.
        split_save_dir = _as_path(save_model_dir) / _safe_name(split_name)
        split_save_dir.mkdir(parents=True, exist_ok=True)
        saved_model_path = split_save_dir / str(save_model_filename)
        torch.save(model, saved_model_path)
        print(f"Saved full model object to {saved_model_path}")

    # Keep the TA-style test metric columns (RMSE/MAE/Pearson/Spearman) for the
    # final comparison table, but also store validation metrics explicitly so
    # model selection can be justified without looking at the test set.
    result = {
        "Split": split_name,
        **test_metrics,
        "Valid_RMSE": float(valid_metrics["RMSE"]),
        "Valid_MAE": float(valid_metrics["MAE"]),
        "Valid_Pearson": float(valid_metrics["Pearson"]),
        "Valid_Spearman": float(valid_metrics["Spearman"]),
        "BestValidLoss_standardized_MSE": best_valid_loss,
        "EpochsTrained": len(history),
        "SavedModelPath": str(saved_model_path) if saved_model_path is not None else "",
    }
    history_df = pd.DataFrame(history)
    return result, model, history_df


# =============================================================================
# Version 3 model: ESM all-layer channel-CNN + Morgan/MolFormer staged fusion
# =============================================================================
# This version is intentionally simpler than the earlier all-layer Transformer:
#   protein_x: [B, 33 * 2560] -> [B, 33, 2560]
#   layer-axis Conv1d: 33 -> 120 -> 1
#   compressed protein vector: [B, 2560]
# Then:
#   Stage 1: Morgan + compressed ESM protein -> MLP
#   Stage 2: previous pair vector + MolFormer-final -> MLP -> KIBA score


def _valid_group_count(num_channels: int, preferred: int = 8) -> int:
    """Pick a GroupNorm group count that divides num_channels."""
    preferred = max(1, min(int(preferred), int(num_channels)))
    for g in range(preferred, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ESMChannelCNNCompressor(nn.Module):
    """
    Compress all ESM2 layer features into one ESM-layer-sized vector.

    Input:
        protein_layers: [B, esm_num_layers, esm_per_layer_dim]
                       e.g. [B, 33, 2560]

    Channel-CNN over ESM layers:
        Conv1d channels 33 -> 120 -> 1, kernel_size=1

    Output:
        [B, esm_per_layer_dim]
        e.g. [B, 2560]

    kernel_size=1 is intentional: ESM embedding dimensions are not spatially
    ordered like image pixels. The convolution acts as a shared layer-channel
    mixer across the 33 ESM layer channels for every embedding coordinate.
    """

    def __init__(
        self,
        esm_num_layers: int,
        esm_per_layer_dim: int,
        protein_cnn_channels: int = 120,
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        group_norm_groups: int = 8,
        residual_final_layer: bool = True,
    ):
        super().__init__()

        self.esm_num_layers = int(esm_num_layers)
        self.esm_per_layer_dim = int(esm_per_layer_dim)
        self.protein_cnn_channels = int(protein_cnn_channels)
        self.protein_layer_dropout = float(protein_layer_dropout)
        self.residual_final_layer = bool(residual_final_layer)

        groups = _valid_group_count(self.protein_cnn_channels, preferred=group_norm_groups)

        self.input_norm = nn.LayerNorm(self.esm_per_layer_dim)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.esm_num_layers,
                out_channels=self.protein_cnn_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=self.protein_cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=self.protein_cnn_channels,
                out_channels=1,
                kernel_size=1,
                bias=True,
            ),
        )
        self.output_norm = nn.LayerNorm(self.esm_per_layer_dim)

        # Safety path: start close to the strong final-layer ESM baseline while
        # the 33->120->1 CNN learns an all-layer correction.
        if self.residual_final_layer:
            self.final_scale = nn.Parameter(torch.tensor(1.0))
            self.cnn_scale = nn.Parameter(torch.tensor(0.25))
        else:
            self.final_scale = None
            self.cnn_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, protein_layers: torch.Tensor) -> torch.Tensor:
        if protein_layers.ndim != 3:
            raise ValueError(
                f"Expected protein_layers [B, L, D], got shape {tuple(protein_layers.shape)}"
            )
        if protein_layers.size(1) != self.esm_num_layers:
            raise ValueError(
                f"Expected {self.esm_num_layers} ESM layers, got {protein_layers.size(1)}"
            )
        if protein_layers.size(2) != self.esm_per_layer_dim:
            raise ValueError(
                f"Expected ESM per-layer dim {self.esm_per_layer_dim}, got {protein_layers.size(2)}"
            )

        x = self.input_norm(protein_layers)

        # Optional regularization: randomly drop whole ESM-layer channels only in training.
        # This does not alter validation/test behavior.
        if self.training and self.protein_layer_dropout > 0.0:
            keep_prob = 1.0 - self.protein_layer_dropout
            mask = torch.empty(
                x.size(0), x.size(1), 1,
                dtype=x.dtype,
                device=x.device,
            ).bernoulli_(keep_prob)
            x = x * mask / max(keep_prob, 1e-6)

        # Exact requested channel path: 33 -> 120 -> 1.
        cnn_out = self.channel_mixer(x).squeeze(1)

        if self.residual_final_layer:
            final_layer = protein_layers[:, -1, :]
            out = self.final_scale * final_layer + self.cnn_scale * cnn_out
        else:
            out = self.cnn_scale * cnn_out

        out = self.output_norm(out)
        return out


class ESMLayerCNNMorganMolFormerDTA(nn.Module):
    """
    Morgan + MolFormer final + ESM2 all-layer channel-CNN model.

    Expected drug_x order from HW3FeatureFactory.build_feature_pack:
        [Morgan 1024-d, MolFormer-final 768-d]

    Expected protein_x:
        flattened all-layer ESM2 features.
        For ESM2-t33 with mean+max pooling:
            [B, 33 * 2560] = [B, 84480]

    Internal protein processing:
        protein_x -> [B, 33, 2560]
        channel CNN: 33 -> 120 -> 1
        optional residual final-layer skip
        output -> [B, 2560]

    Fusion order:
        1. Morgan + compressed ESM protein -> Morgan-protein pair vector.
        2. Pair vector + MolFormer final vector -> KIBA regression score.

    Compatible with:
        model_class(drug_dim=..., protein_dim=..., **model_kwargs)
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        esm_num_layers: int = 33,
        morgan_dim: int = 1024,
        protein_cnn_channels: int = 120,
        proj_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        residual_final_layer: bool = True,
    ):
        super().__init__()

        self.drug_dim = int(drug_dim)
        self.protein_dim = int(protein_dim)
        self.esm_num_layers = int(esm_num_layers)
        self.morgan_dim = int(morgan_dim)
        self.molformer_dim = self.drug_dim - self.morgan_dim

        if self.molformer_dim <= 0:
            raise ValueError(
                f"drug_dim={drug_dim}, morgan_dim={morgan_dim}. "
                "This model expects drug_x = [Morgan, MolFormer]. "
                "Build features with use_morgan=True and use_molformer=True."
            )
        if self.protein_dim % self.esm_num_layers != 0:
            raise ValueError(
                f"protein_dim={protein_dim} is not divisible by esm_num_layers={esm_num_layers}."
            )

        self.esm_per_layer_dim = self.protein_dim // self.esm_num_layers

        self.protein_cnn = ESMChannelCNNCompressor(
            esm_num_layers=self.esm_num_layers,
            esm_per_layer_dim=self.esm_per_layer_dim,
            protein_cnn_channels=protein_cnn_channels,
            dropout=dropout,
            protein_layer_dropout=protein_layer_dropout,
            residual_final_layer=residual_final_layer,
        )

        self.morgan_encoder = nn.Sequential(
            nn.Linear(self.morgan_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.protein_encoder = nn.Sequential(
            nn.Linear(self.esm_per_layer_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.morgan_protein_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.molformer_encoder = nn.Sequential(
            nn.Linear(self.molformer_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, drug_x: torch.Tensor, protein_x: torch.Tensor) -> torch.Tensor:
        batch_size = drug_x.size(0)

        # drug_x = [Morgan, MolFormer-final]
        x_morgan = drug_x[:, :self.morgan_dim]
        x_molformer = drug_x[:, self.morgan_dim:]

        # protein_x = flattened [layer1, layer2, ..., layer33]
        protein_layers = protein_x.reshape(
            batch_size,
            self.esm_num_layers,
            self.esm_per_layer_dim,
        )
        x_protein = self.protein_cnn(protein_layers)

        z_morgan = self.morgan_encoder(x_morgan)
        z_protein = self.protein_encoder(x_protein)

        # Stage 1: Morgan + compressed ESM protein.
        mp_fusion = torch.cat(
            [
                z_morgan,
                z_protein,
                z_morgan * z_protein,
                torch.abs(z_morgan - z_protein),
            ],
            dim=1,
        )
        z_mp = self.morgan_protein_fusion(mp_fusion)

        # Stage 2: fuse the Morgan-protein pair vector with MolFormer.
        z_molformer = self.molformer_encoder(x_molformer)
        final_fusion = torch.cat(
            [
                z_mp,
                z_molformer,
                z_mp * z_molformer,
                torch.abs(z_mp - z_molformer),
            ],
            dim=1,
        )

        return self.predictor(final_fusion).squeeze(-1)


# Notebook-friendly alias for the v3 CNN experiment.
MyImprovedModel = ESMLayerCNNMorganMolFormerDTA
# =============================================================================
# V4: SpliceAI-inspired sequence-aware ESM all-layer CNN
# =============================================================================
# Motivation
# ----------
# The v3 ChannelCNN compressed [33 ESM layers] -> [1 ESM-layer-sized vector]
# after global pooling over the whole protein. That is fast and strong, but it
# cannot learn local sequence context because residue/bin order has already been
# pooled away.
#
# V5-lite keeps a coarser sequence axis by binning residue embeddings along the full
# protein sequence:
#     ESM2-t33 all layers -> [num_bins, 33, 1280]
# flattened as a single protein feature vector for TA-style compatibility.
# The model reshapes it back to [B, num_bins, 33, 1280] and uses dilated CNNs
# over the sequence-bin axis before drug-conditioned weighted pooling.
#
# This is inspired by the SpliceAI-style idea of expanding receptive field with
# residual/dilated CNNs, but it is still a DTA feature-fusion model: no external
# affinity labels, no DTA checkpoint, and no split/metric changes.


def _bin_ids_for_sequence(length: int, num_bins: int) -> np.ndarray:
    """Map each residue position 0..L-1 to a fixed sequence bin 0..num_bins-1."""
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    ids = (np.arange(length, dtype=np.int64) * int(num_bins)) // int(length)
    return np.clip(ids, 0, int(num_bins) - 1).astype(np.int64)


def build_or_load_esm2_sequence_layer_bank(
    dataframe: pd.DataFrame,
    cache_path: Union[str, Path],
    target_col: str = "Target",
    model_size: str = "t33",
    sequence_bins: int = 128,
    window_len: int = 1022,
    stride: int = 511,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    overlap_strategy: str = "position_weighted",
    repr_layer_chunk_size: Optional[int] = 6,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build/load a sequence-aware ESM2 all-layer bank.

    Cache format:
        {
            "kind": "esm2_sequence_layer_bank",
            "model_size": "t33",
            "layers": [1, ..., 33],
            "hidden_dim": 1280,
            "sequence_bins": 128,
            "features": {
                protein_sequence: np.ndarray [sequence_bins, num_layers, hidden_dim]
            }
        }

    Compared with build_or_load_esm2_layer_bank(...), this does NOT globally pool
    over the full protein sequence. Instead, each residue is assigned to a fixed
    sequence bin and ESM residue embeddings are averaged within each bin.

    Sliding-window overlap handling:
        position_weighted:
            each residue's repeated window appearances are weighted by
            1 / count(position), so each residue contributes total weight 1.
            This reduces overlap bias while avoiding a huge [L, layers, dim]
            residue-level cache.
    """
    cache_path = _as_path(cache_path)
    spec = get_esm2_spec(model_size)
    num_layers = int(spec["num_layers"])
    hidden_dim = int(spec["dim"])
    layer_ids = list(range(1, num_layers + 1))
    sequence_bins = int(sequence_bins)

    if sequence_bins <= 0:
        raise ValueError("sequence_bins must be positive.")

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached ESM2 sequence layer bank from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("kind") == "esm2_sequence_layer_bank":
            return cached
        raise ValueError(
            f"Cache at {cache_path} exists but is not an ESM2 sequence-layer bank. "
            "Use another cache path or force_rebuild=True."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cpu")

    if model_size.lower() in {"t30", "150m", "t33", "650m"}:
        print(
            "Warning: sequence-aware ESM2 all-layer caching is very heavy. "
            "For H100/G4, start with batch_size=1 and repr_layer_chunk_size=4 or 6."
        )

    if overlap_strategy != "position_weighted":
        raise ValueError("Currently only overlap_strategy='position_weighted' is supported.")

    model, alphabet, spec = load_esm2_model(model_size, device)
    batch_converter = alphabet.get_batch_converter()

    unique_targets = sorted([str(x) for x in dataframe[target_col].dropna().unique()])
    lengths = [len(x) for x in unique_targets]
    if lengths:
        print("Unique targets:", len(unique_targets))
        print("Max target length:", max(lengths))
        print(f"Targets longer than {window_len} aa:", sum(L > window_len for L in lengths))
        print("Sequence bins:", sequence_bins)
        print("Per-target flattened protein dim:", sequence_bins * num_layers * hidden_dim)

    features: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for target_idx, seq in enumerate(tqdm(unique_targets, desc="ESM2 sequence-bin all-layer bank")):
            seq = str(seq)
            L = len(seq)

            if L == 0:
                features[seq] = np.zeros((sequence_bins, num_layers, hidden_dim), dtype=np.float32)
                continue

            windows = make_esm_windows(seq, window_len=window_len, stride=stride)
            pos_counts = _precompute_window_counts(L, windows).squeeze(1)  # [L]
            bin_ids_np = _bin_ids_for_sequence(L, sequence_bins)
            bin_counts_np = np.bincount(bin_ids_np, minlength=sequence_bins).astype(np.float32)
            safe_bin_counts = torch.tensor(bin_counts_np, dtype=torch.float32).clamp(min=1.0).view(sequence_bins, 1)

            # Store bin sums for each ESM layer. This is much smaller than a full
            # residue-level [L, layers, hidden_dim] tensor.
            layer_bin_sums = {
                layer: torch.zeros((sequence_bins, hidden_dim), dtype=torch.float32)
                for layer in layer_ids
            }

            for start in range(0, len(windows), batch_size):
                batch_windows = windows[start:start + batch_size]
                batch_data = [
                    (f"target_{target_idx}_window_{start + j}_pos_{pos}", chunk)
                    for j, (pos, chunk) in enumerate(batch_windows)
                ]

                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)

                for layer_chunk in _chunked(layer_ids, repr_layer_chunk_size):
                    results = model(
                        batch_tokens,
                        repr_layers=list(layer_chunk),
                        return_contacts=False,
                    )

                    reps_by_layer = {
                        layer: results["representations"][layer].detach().cpu().float()
                        for layer in layer_chunk
                    }

                    for i, (pos, chunk) in enumerate(batch_windows):
                        chunk_len = len(chunk)
                        s0 = int(pos)
                        s1 = int(pos + chunk_len)

                        bin_idx = torch.tensor(bin_ids_np[s0:s1], dtype=torch.long)
                        weights = (1.0 / pos_counts[s0:s1].clamp(min=1.0)).view(chunk_len, 1)

                        for layer in layer_chunk:
                            # ESM token representations include BOS/EOS.
                            residue_repr = reps_by_layer[layer][i, 1:chunk_len + 1]
                            weighted_repr = residue_repr * weights
                            layer_bin_sums[layer].index_add_(0, bin_idx, weighted_repr)

                    del results, reps_by_layer

            # Normalize each bin by the number of residues assigned to that bin.
            # Empty bins remain zero because numerator is zero and denominator is 1.
            layer_bin_features = []
            for layer in layer_ids:
                layer_bin_mean = layer_bin_sums[layer] / safe_bin_counts
                layer_bin_features.append(layer_bin_mean)

            # [sequence_bins, num_layers, hidden_dim]
            seq_bank = torch.stack(layer_bin_features, dim=1).numpy().astype(np.float32)
            features[seq] = seq_bank

    cache = {
        "kind": "esm2_sequence_layer_bank",
        "model_size": model_size,
        "model_fn": spec["fn"],
        "layers": layer_ids,
        "hidden_dim": hidden_dim,
        "sequence_bins": sequence_bins,
        "window_len": window_len,
        "stride": stride,
        "overlap_strategy": overlap_strategy,
        "features": features,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved ESM2 sequence layer bank to {cache_path}")
    return cache


def select_sequence_layers_from_bank(bank: Mapping[str, Any], layers: LayerSpec = "all") -> ArrayDict:
    """
    Select layers from an ESM2 sequence-layer bank and flatten for TA compatibility.

    Input bank feature shape:
        [sequence_bins, num_layers, hidden_dim]

    Output feature dict value shape:
        [sequence_bins * selected_layers * hidden_dim]

    The model later reshapes it back to:
        [B, sequence_bins, selected_layers, hidden_dim]
    """
    if bank.get("kind") != "esm2_sequence_layer_bank":
        raise ValueError("Expected kind='esm2_sequence_layer_bank'.")

    layer_ids = list(bank["layers"])
    max_layer = max(layer_ids)
    selected_layers = normalize_layer_spec(layers, max_layer)
    row_by_layer = {layer: i for i, layer in enumerate(layer_ids)}

    missing = [layer for layer in selected_layers if layer not in row_by_layer]
    if missing:
        raise ValueError(f"Sequence bank does not contain requested layers: {missing}")

    out: ArrayDict = {}
    for k, arr in bank["features"].items():
        arr = np.asarray(arr, dtype=np.float32)  # [T, L, H]
        selected = arr[:, [row_by_layer[layer] for layer in selected_layers], :]
        out[str(k)] = selected.reshape(-1).astype(np.float32)
    return out




def coarsen_sequence_layer_bank(
    bank: Mapping[str, Any],
    output_sequence_bins: int,
    mode: str = "nonzero_mean",
) -> Dict[str, Any]:
    """
    Downsample a sequence-layer ESM bank without rerunning ESM2.

    Input feature shape per protein:
        [source_bins, num_layers, hidden_dim]

    Output feature shape per protein:
        [output_sequence_bins, num_layers, hidden_dim]

    This is useful when a heavy cache was already created with 128 bins but
    training is too slow. For example, 128 -> 64 halves the flattened protein
    feature size and roughly halves the sequence-CNN compute.

    mode="nonzero_mean" averages only non-empty source bins inside each output
    bin. This avoids shrinking short proteins just because some fixed bins were
    empty.
    """
    if bank.get("kind") != "esm2_sequence_layer_bank":
        raise ValueError("Expected kind='esm2_sequence_layer_bank'.")

    source_bins = int(bank["sequence_bins"])
    output_sequence_bins = int(output_sequence_bins)

    if output_sequence_bins <= 0:
        raise ValueError("output_sequence_bins must be positive.")
    if output_sequence_bins > source_bins:
        raise ValueError(
            f"Cannot coarsen from {source_bins} to {output_sequence_bins}; "
            "output_sequence_bins must be <= source_bins."
        )
    if output_sequence_bins == source_bins:
        return dict(bank)

    new_features: Dict[str, np.ndarray] = {}

    # Map source bin i -> output bin floor(i * output_bins / source_bins).
    out_ids = (np.arange(source_bins, dtype=np.int64) * output_sequence_bins) // source_bins
    out_ids = np.clip(out_ids, 0, output_sequence_bins - 1)

    for key, arr in bank["features"].items():
        arr = np.asarray(arr, dtype=np.float32)  # [source_bins, L, H]
        if arr.shape[0] != source_bins:
            raise ValueError(
                f"Feature for key {key!r} has {arr.shape[0]} bins, expected {source_bins}."
            )

        out = np.zeros((output_sequence_bins, arr.shape[1], arr.shape[2]), dtype=np.float32)

        # A source bin is considered valid if any layer/hidden entry is non-zero.
        valid = np.abs(arr).sum(axis=(1, 2)) > 0

        for j in range(output_sequence_bins):
            idx = np.where(out_ids == j)[0]
            if idx.size == 0:
                continue

            if mode == "mean":
                use_idx = idx
            elif mode == "nonzero_mean":
                nz = idx[valid[idx]]
                use_idx = nz if nz.size > 0 else idx
            else:
                raise ValueError("mode must be 'mean' or 'nonzero_mean'.")

            out[j] = arr[use_idx].mean(axis=0)

        new_features[str(key)] = out.astype(np.float32)

    new_bank = dict(bank)
    new_bank["features"] = new_features
    new_bank["source_sequence_bins"] = source_bins
    new_bank["sequence_bins"] = output_sequence_bins
    new_bank["coarsen_mode"] = mode
    return new_bank

def _factory_default_esm_sequence_bank_cache(
    self: HW3FeatureFactory,
    model_size: str,
    sequence_bins: int,
    window_len: int,
    stride: int,
    overlap_strategy: str,
) -> Path:
    return self.cache_dir / (
        f"{self.dataset_name.lower()}_protein_esm2_{_safe_name(model_size)}_"
        f"seqbins{sequence_bins}_all_layers_mean_win{window_len}_stride{stride}_{overlap_strategy}.pkl"
    )


def _factory_build_esm2_sequence_layer_bank(
    self: HW3FeatureFactory,
    model_size: str = "t33",
    sequence_bins: int = 128,
    window_len: int = 1022,
    stride: int = 511,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    overlap_strategy: str = "position_weighted",
    repr_layer_chunk_size: Optional[int] = 6,
    cache_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if cache_path is None:
        cache_path = _factory_default_esm_sequence_bank_cache(
            self,
            model_size=model_size,
            sequence_bins=sequence_bins,
            window_len=window_len,
            stride=stride,
            overlap_strategy=overlap_strategy,
        )
    return build_or_load_esm2_sequence_layer_bank(
        self.df,
        cache_path=cache_path,
        target_col=self.target_col,
        model_size=model_size,
        sequence_bins=sequence_bins,
        window_len=window_len,
        stride=stride,
        batch_size=batch_size,
        device=device if device is not None else self.device,
        overlap_strategy=overlap_strategy,
        repr_layer_chunk_size=repr_layer_chunk_size,
        force_rebuild=force_rebuild,
    )


def _factory_build_sequence_feature_pack(
    self: HW3FeatureFactory,
    *,
    # Drug options
    use_morgan: bool = True,
    morgan_features: Optional[Mapping[str, Any]] = None,
    morgan_bits: int = 1024,
    morgan_radius: int = 2,
    use_molformer: bool = True,
    molformer_layers: LayerSpec = None,
    molformer_model_name_or_candidates: Union[str, Sequence[str], None] = None,
    molformer_max_length: int = 202,
    molformer_batch_size: int = 64,
    molformer_device: Optional[torch.device] = None,
    molformer_canonicalize: bool = True,
    molformer_remove_isomeric: bool = True,
    molformer_pooling: str = "mean",
    molformer_bank_cache_path: Optional[Union[str, Path]] = None,
    # Protein sequence-bank options
    esm_model_size: str = "t33",
    esm_layers: LayerSpec = "all",
    esm_sequence_bins: int = 128,
    esm_source_sequence_bins: Optional[int] = None,
    esm_output_sequence_bins: Optional[int] = None,
    esm_window_len: int = 1022,
    esm_stride: int = 511,
    esm_batch_size: int = 1,
    esm_device: Optional[torch.device] = None,
    esm_overlap_strategy: str = "position_weighted",
    esm_repr_layer_chunk_size: Optional[int] = 6,
    esm_sequence_bank_cache_path: Optional[Union[str, Path]] = None,
    # General
    force_rebuild_molformer: bool = False,
    force_rebuild_esm: bool = False,
    force_rebuild_morgan: bool = False,
) -> FeaturePack:
    """
    Build a feature pack for the V4 Splice-style sequence CNN experiment.

    Drug feature order:
        [Morgan, MolFormer selected layers]

    Protein feature order:
        flattened [sequence_bin, ESM_layer, hidden_dim]
    """
    required_drug_keys = [str(x) for x in self.df[self.drug_col].dropna().unique()]
    drug_parts: List[Tuple[str, ArrayDict]] = []

    if use_morgan:
        if morgan_features is None:
            morgan_features = self.build_morgan(
                n_bits=morgan_bits,
                radius=morgan_radius,
                force_rebuild=force_rebuild_morgan,
            )
        morgan_features = {str(k): _ensure_float32_1d(v) for k, v in morgan_features.items()}
        drug_parts.append(("Morgan", dict(morgan_features)))

    molformer_bank = None
    selected_molformer_layers = None
    if use_molformer:
        molformer_bank = self.build_molformer_layer_bank(
            model_name_or_candidates=molformer_model_name_or_candidates,
            max_length=molformer_max_length,
            batch_size=molformer_batch_size,
            device=molformer_device,
            canonicalize=molformer_canonicalize,
            remove_isomeric=molformer_remove_isomeric,
            pooling=molformer_pooling,
            cache_path=molformer_bank_cache_path,
            force_rebuild=force_rebuild_molformer,
        )
        selected_molformer_layers = normalize_layer_spec(
            molformer_layers,
            max(molformer_bank["layers"]),
        )
        molformer_features = select_layers_from_bank(molformer_bank, selected_molformer_layers)
        drug_parts.append((f"MolFormer{selected_molformer_layers}", molformer_features))

    if not drug_parts:
        raise ValueError("At least one of use_morgan or use_molformer must be True.")

    drug_features = drug_parts[0][1]
    drug_name = drug_parts[0][0]
    for part_name, part_features in drug_parts[1:]:
        drug_features = concat_feature_dicts_by_key(
            drug_features,
            part_features,
            required_keys=required_drug_keys,
            left_name=drug_name,
            right_name=part_name,
        )
        drug_name = f"{drug_name}+{part_name}"

    # Build/load a source sequence-bin cache, then optionally coarsen it for faster training.
    # Example: source 128 bins -> output 64 bins reuses the expensive 128-bin cache
    # without rerunning ESM2, while halving the training-time protein feature size.
    source_sequence_bins = int(esm_sequence_bins if esm_source_sequence_bins is None else esm_source_sequence_bins)
    output_sequence_bins = int(esm_sequence_bins if esm_output_sequence_bins is None else esm_output_sequence_bins)

    esm_sequence_bank = _factory_build_esm2_sequence_layer_bank(
        self,
        model_size=esm_model_size,
        sequence_bins=source_sequence_bins,
        window_len=esm_window_len,
        stride=esm_stride,
        batch_size=esm_batch_size,
        device=esm_device,
        overlap_strategy=esm_overlap_strategy,
        repr_layer_chunk_size=esm_repr_layer_chunk_size,
        cache_path=esm_sequence_bank_cache_path,
        force_rebuild=force_rebuild_esm,
    )

    if output_sequence_bins != source_sequence_bins:
        print(f"Coarsening ESM sequence bins: {source_sequence_bins} -> {output_sequence_bins}")
        esm_sequence_bank = coarsen_sequence_layer_bank(
            esm_sequence_bank,
            output_sequence_bins=output_sequence_bins,
            mode="nonzero_mean",
        )

    selected_esm_layers = normalize_layer_spec(esm_layers, max(esm_sequence_bank["layers"]))
    protein_features = select_sequence_layers_from_bank(esm_sequence_bank, selected_esm_layers)

    pack = FeaturePack(
        drug_features=drug_features,
        protein_features=protein_features,
        config={
            "kind": "sequence_feature_pack_v4",
            "drug_order": [name for name, _ in drug_parts],
            "use_morgan": use_morgan,
            "use_molformer": use_molformer,
            "molformer_layers": selected_molformer_layers,
            "molformer_pooling": molformer_pooling,
            "esm_model_size": esm_model_size,
            "esm_layers": selected_esm_layers,
            "esm_sequence_bins": output_sequence_bins,
            "esm_source_sequence_bins": source_sequence_bins,
            "esm_hidden_dim": esm_sequence_bank["hidden_dim"],
            "esm_window_len": esm_window_len,
            "esm_stride": esm_stride,
            "esm_overlap_strategy": esm_overlap_strategy,
            "protein_layout": "flattened [sequence_bin, esm_layer, hidden_dim]",
        },
        drug_col=self.drug_col,
        target_col=self.target_col,
        label_col=self.label_col,
    )

    print("Sequence feature pack built.")
    print("  Drug feature dim:", pack.drug_dim)
    print("  Protein feature dim:", pack.protein_dim)
    print("  Config:", pack.config)
    return pack


# Monkey-patch methods into the existing factory class so older code remains valid.
HW3FeatureFactory._default_esm_sequence_bank_cache = _factory_default_esm_sequence_bank_cache
HW3FeatureFactory.build_esm2_sequence_layer_bank = _factory_build_esm2_sequence_layer_bank
HW3FeatureFactory.build_sequence_feature_pack = _factory_build_sequence_feature_pack


def _valid_group_count_v4(num_channels: int, preferred: int = 8) -> int:
    return _valid_group_count(num_channels, preferred=preferred)


class SpliceStyleLayerSequenceMixer(nn.Module):
    """
    ESM all-layer sequence compressor.

    Input:
        protein_seq: [B, T, L, H]
            T = sequence bins, L = ESM layers, H = ESM hidden dim.

    Operation:
        Treat ESM layers as CNN channels and sequence bins as the spatial axis.
        Conv2d uses kernels (k_seq, 1), so it expands receptive field along the
        protein sequence while sharing the same operation across embedding dims.

    Output:
        [B, T, H]
    """

    def __init__(
        self,
        esm_num_layers: int = 33,
        esm_hidden_dim: int = 1280,
        layer_cnn_channels: int = 120,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        residual_final_layer: bool = True,
        group_norm_groups: int = 8,
    ):
        super().__init__()
        self.esm_num_layers = int(esm_num_layers)
        self.esm_hidden_dim = int(esm_hidden_dim)
        self.layer_cnn_channels = int(layer_cnn_channels)
        self.protein_layer_dropout = float(protein_layer_dropout)
        self.residual_final_layer = bool(residual_final_layer)
        self.dilations = tuple(int(d) for d in dilations)

        groups = _valid_group_count_v4(self.layer_cnn_channels, preferred=group_norm_groups)

        self.input_norm = nn.LayerNorm(self.esm_hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=self.esm_num_layers,
                out_channels=self.layer_cnn_channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dilated_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=self.layer_cnn_channels,
                    out_channels=self.layer_cnn_channels,
                    kernel_size=(3, 1),
                    padding=(d, 0),
                    dilation=(d, 1),
                    bias=False,
                ),
                nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=self.layer_cnn_channels,
                    out_channels=self.layer_cnn_channels,
                    kernel_size=(1, 1),
                    bias=True,
                ),
                nn.Dropout(dropout),
            )
            for d in self.dilations
        ])

        self.collapse = nn.Conv2d(
            in_channels=self.layer_cnn_channels,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        self.output_norm = nn.LayerNorm(self.esm_hidden_dim)

        if self.residual_final_layer:
            self.final_scale = nn.Parameter(torch.tensor(1.0))
            self.cnn_scale = nn.Parameter(torch.tensor(0.25))
        else:
            self.final_scale = None
            self.cnn_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, protein_seq: torch.Tensor) -> torch.Tensor:
        if protein_seq.ndim != 4:
            raise ValueError(
                f"Expected protein_seq [B, T, L, H], got shape {tuple(protein_seq.shape)}"
            )
        if protein_seq.size(2) != self.esm_num_layers:
            raise ValueError(f"Expected {self.esm_num_layers} ESM layers, got {protein_seq.size(2)}")
        if protein_seq.size(3) != self.esm_hidden_dim:
            raise ValueError(f"Expected hidden dim {self.esm_hidden_dim}, got {protein_seq.size(3)}")

        x = self.input_norm(protein_seq)

        # Drop whole ESM-layer channels during training.
        if self.training and self.protein_layer_dropout > 0.0:
            keep_prob = 1.0 - self.protein_layer_dropout
            mask = torch.empty(
                x.size(0), 1, x.size(2), 1,
                dtype=x.dtype,
                device=x.device,
            ).bernoulli_(keep_prob)
            x = x * mask / max(keep_prob, 1e-6)

        # [B, T, L, H] -> [B, L, T, H], where L is treated as channels.
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.stem(x)

        # Dilated residual CNN along the sequence-bin axis.
        for block in self.dilated_blocks:
            x = x + block(x)

        cnn_out = self.collapse(x).squeeze(1)  # [B, T, H]

        if self.residual_final_layer:
            final_seq = protein_seq[:, :, -1, :]  # final ESM layer per sequence bin
            out = self.final_scale * final_seq + self.cnn_scale * cnn_out
        else:
            out = self.cnn_scale * cnn_out

        out = self.output_norm(out)
        return out


class DilatedResidualConv1DBlock(nn.Module):
    """Depthwise-separable residual dilated Conv1d block over sequence tokens."""

    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.20):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y = self.norm(x).transpose(1, 2).contiguous()
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise(y)
        y = self.drop(y).transpose(1, 2).contiguous()
        return x + y


class SpliceStyleESMSequenceCNNMorganMolFormerDTA(nn.Module):
    """
    V4 sequence-aware ESM all-layer CNN model.

    Expected drug_x order:
        [Morgan 1024-d, MolFormer-final 768-d]

    Expected protein_x layout:
        flattened [sequence_bin, ESM_layer, ESM_hidden]
        For ESM2-t33, sequence_bins=128:
            protein_dim = 128 * 33 * 1280 = 5,406,720

    Internal protein flow:
        protein_x -> [B, T, 33, 1280]
        Splice-style layer/sequence CNN with dilations -> [B, T, 1280]
        token projection + dilated sequence CNN -> [B, T, seq_model_dim]
        drug-conditioned weighted sum + mean/max pooling -> protein summary

    Fusion flow follows the user's requested order:
        Morgan + protein summary -> MLP pair vector
        pair vector + MolFormer -> final MLP -> KIBA score
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        morgan_dim: int = 1024,
        sequence_bins: int = 128,
        esm_num_layers: int = 33,
        esm_hidden_dim: int = 1280,
        layer_cnn_channels: int = 120,
        layer_cnn_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        seq_model_dim: int = 512,
        seq_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        proj_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        residual_final_layer: bool = True,
    ):
        super().__init__()

        self.drug_dim = int(drug_dim)
        self.protein_dim = int(protein_dim)
        self.morgan_dim = int(morgan_dim)
        self.molformer_dim = self.drug_dim - self.morgan_dim
        self.sequence_bins = int(sequence_bins)
        self.esm_num_layers = int(esm_num_layers)
        self.esm_hidden_dim = int(esm_hidden_dim)

        if self.molformer_dim <= 0:
            raise ValueError(
                f"drug_dim={drug_dim}, morgan_dim={morgan_dim}. "
                "This model expects drug_x = [Morgan, MolFormer]."
            )

        expected_protein_dim = self.sequence_bins * self.esm_num_layers * self.esm_hidden_dim
        if self.protein_dim != expected_protein_dim:
            raise ValueError(
                f"Expected protein_dim={expected_protein_dim}, got {protein_dim}. "
                "Check sequence_bins, esm_num_layers, esm_hidden_dim, and feature construction."
            )

        self.layer_sequence_mixer = SpliceStyleLayerSequenceMixer(
            esm_num_layers=self.esm_num_layers,
            esm_hidden_dim=self.esm_hidden_dim,
            layer_cnn_channels=layer_cnn_channels,
            dilations=layer_cnn_dilations,
            dropout=dropout,
            protein_layer_dropout=protein_layer_dropout,
            residual_final_layer=residual_final_layer,
        )

        self.protein_token_proj = nn.Sequential(
            nn.Linear(self.esm_hidden_dim, seq_model_dim),
            nn.LayerNorm(seq_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.seq_conv_blocks = nn.ModuleList([
            DilatedResidualConv1DBlock(seq_model_dim, dilation=d, dropout=dropout)
            for d in seq_dilations
        ])

        self.morgan_encoder = nn.Sequential(
            nn.Linear(self.morgan_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.molformer_encoder = nn.Sequential(
            nn.Linear(self.molformer_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        # Drug-conditioned query used only for protein sequence weighted pooling.
        self.drug_query = nn.Sequential(
            nn.Linear(proj_dim * 4, seq_model_dim),
            nn.LayerNorm(seq_model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.protein_key = nn.Linear(seq_model_dim, seq_model_dim, bias=False)

        self.protein_summary = nn.Sequential(
            nn.Linear(seq_model_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.morgan_protein_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(proj_dim, dropout=dropout),
        )

        self.predictor = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim // 2, dropout=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], mask: [B, T]
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        very_neg = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
        x_masked = x.masked_fill(~mask.unsqueeze(-1), very_neg)
        out = x_masked.max(dim=1).values
        # If a sample somehow has no valid bins, fall back to zeros.
        no_valid = ~mask.any(dim=1)
        if no_valid.any():
            out = out.clone()
            out[no_valid] = 0.0
        return out

    def forward(self, drug_x: torch.Tensor, protein_x: torch.Tensor) -> torch.Tensor:
        batch_size = drug_x.size(0)

        # drug_x = [Morgan, MolFormer-final]
        x_morgan = drug_x[:, :self.morgan_dim]
        x_molformer = drug_x[:, self.morgan_dim:]

        z_morgan = self.morgan_encoder(x_morgan)
        z_molformer = self.molformer_encoder(x_molformer)

        # protein_x = flattened [sequence_bin, ESM_layer, ESM_hidden]
        protein_seq = protein_x.reshape(
            batch_size,
            self.sequence_bins,
            self.esm_num_layers,
            self.esm_hidden_dim,
        )

        # Valid bins are non-zero in the raw cached representation.
        # This handles short proteins where some fixed bins are empty.
        valid_mask = protein_seq.detach().abs().sum(dim=(2, 3)) > 0
        if not valid_mask.any(dim=1).all():
            # Make sure softmax never receives all -inf for pathological empty input.
            valid_mask = valid_mask.clone()
            empty_rows = ~valid_mask.any(dim=1)
            valid_mask[empty_rows, 0] = True

        # Splice-style layer/sequence CNN: [B, T, 33, 1280] -> [B, T, 1280]
        protein_seq = self.layer_sequence_mixer(protein_seq)
        seq_tokens = self.protein_token_proj(protein_seq)  # [B, T, C]

        for block in self.seq_conv_blocks:
            seq_tokens = block(seq_tokens)

        # Drug-conditioned weighted sequence pooling.
        drug_pair = torch.cat(
            [
                z_morgan,
                z_molformer,
                z_morgan * z_molformer,
                torch.abs(z_morgan - z_molformer),
            ],
            dim=1,
        )
        q = self.drug_query(drug_pair)  # [B, C]
        k = self.protein_key(seq_tokens)

        attn_logits = (k * q.unsqueeze(1)).sum(dim=-1) / np.sqrt(float(k.size(-1)))
        attn_logits = attn_logits.masked_fill(~valid_mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        z_p_attn = torch.sum(attn.unsqueeze(-1) * seq_tokens, dim=1)

        z_p_mean = self._masked_mean(seq_tokens, valid_mask)
        z_p_max = self._masked_max(seq_tokens, valid_mask)
        z_protein = self.protein_summary(torch.cat([z_p_attn, z_p_mean, z_p_max], dim=1))

        # Stage 1: Morgan + sequence-aware protein summary.
        mp_fusion = torch.cat(
            [
                z_morgan,
                z_protein,
                z_morgan * z_protein,
                torch.abs(z_morgan - z_protein),
            ],
            dim=1,
        )
        z_mp = self.morgan_protein_fusion(mp_fusion)

        # Stage 2: fuse Morgan-protein pair vector with MolFormer.
        final_fusion = torch.cat(
            [
                z_mp,
                z_molformer,
                z_mp * z_molformer,
                torch.abs(z_mp - z_molformer),
            ],
            dim=1,
        )

        return self.predictor(final_fusion).squeeze(-1)


# Notebook-friendly alias for the v5-lite Splice-style experiment.
MyImprovedModel = SpliceStyleESMSequenceCNNMorganMolFormerDTA


# =============================================================================
# V6: circular long-protein windows + selected ESM2 layers + circular sequence CNN
# =============================================================================
# This section is intentionally appended as overrides, so the older v5-lite code
# remains available above.  The public names HW3FeatureFactory.build_sequence_feature_pack
# and MyImprovedModel are rebound below.
#
# Key changes vs v5-lite:
#   1) sequence bins can be reduced to 16.
#   2) ESM2 sequence bank can cache only selected layers, e.g.
#      [4, 8, 12, 16, 20, 24, 28, 33], instead of all 33 layers.
#   3) Long proteins can be processed with circular sliding windows.  For a
#      long sequence, windows are taken on a conceptual ring, so windows near
#      the C-terminus can wrap around and include the N-terminus.
#   4) The sequence-bin CNN can use circular padding along the bin axis, which
#      is consistent with the circular-window hypothesis.

import math
import torch.nn.functional as F


def _layers_key_for_cache(layers: LayerSpec, num_layers: int) -> str:
    selected = normalize_layer_spec(layers, num_layers)
    if selected == list(range(1, num_layers + 1)):
        return "all"
    return "layers" + "-".join(str(x) for x in selected)


def make_esm_circular_windows(
    seq: str,
    window_len: int = 1022,
    stride: int = 511,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Sliding windows on a circularized protein sequence.

    Returns a list of tuples:
        (start_position_on_ring, chunk_sequence, original_position_indices)

    For L <= window_len, the normal full sequence is returned once.  For
    L > window_len, start positions are 0, stride, 2*stride, ... < L, and each
    window has exactly window_len residues, wrapping around the end when needed.
    The original_position_indices array maps every residue token in the chunk
    back to its original 0..L-1 position.
    """
    seq = str(seq)
    L = len(seq)
    if L == 0:
        return []
    if L <= window_len:
        return [(0, seq, np.arange(L, dtype=np.int64))]

    stride = max(1, int(stride))
    window_len = int(window_len)
    starts = list(range(0, L, stride))

    windows: List[Tuple[int, str, np.ndarray]] = []
    for start in starts:
        pos_idx = (start + np.arange(window_len, dtype=np.int64)) % L
        # Build the wrapped sequence chunk without assuming contiguous slicing.
        # KIBA has only 229 unique targets, so this is fine and keeps the mapping explicit.
        chunk = "".join(seq[int(i)] for i in pos_idx)
        windows.append((int(start), chunk, pos_idx))
    return windows


def _make_position_mapped_windows(
    seq: str,
    window_len: int = 1022,
    stride: int = 511,
    circular: bool = False,
) -> List[Tuple[int, str, np.ndarray]]:
    """Return windows with an explicit token->original-position mapping."""
    if circular:
        return make_esm_circular_windows(seq, window_len=window_len, stride=stride)

    out: List[Tuple[int, str, np.ndarray]] = []
    for start, chunk in make_esm_windows(seq, window_len=window_len, stride=stride):
        pos_idx = np.arange(start, start + len(chunk), dtype=np.int64)
        out.append((int(start), chunk, pos_idx))
    return out


def _precompute_position_counts_from_mapped_windows(
    length: int,
    windows: Sequence[Tuple[int, str, np.ndarray]],
) -> torch.Tensor:
    counts_np = np.zeros((int(length),), dtype=np.float32)
    for _, chunk, pos_idx in windows:
        if len(chunk) != len(pos_idx):
            raise ValueError("Window chunk length and position-index length differ.")
        np.add.at(counts_np, pos_idx.astype(np.int64), 1.0)
    return torch.tensor(counts_np, dtype=torch.float32).clamp(min=1.0)


def build_or_load_esm2_sequence_layer_bank_v6(
    dataframe: pd.DataFrame,
    cache_path: Union[str, Path],
    target_col: str = "Target",
    model_size: str = "t33",
    sequence_bins: int = 16,
    window_len: int = 1022,
    stride: int = 511,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    overlap_strategy: str = "position_weighted",
    repr_layers: LayerSpec = (4, 8, 12, 16, 20, 24, 28, 33),
    repr_layer_chunk_size: Optional[int] = None,
    circular_windows: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build/load a sequence-aware ESM2 bank with optional circular windows.

    Feature shape per protein:
        [sequence_bins, selected_esm_layers, hidden_dim]

    This function is the V6 replacement for the v5-lite sequence bank builder.
    It can store only selected ESM2 layers and can circularize long-protein
    sliding windows.
    """
    cache_path = _as_path(cache_path)
    spec = get_esm2_spec(model_size)
    num_layers = int(spec["num_layers"])
    hidden_dim = int(spec["dim"])
    layer_ids = normalize_layer_spec(repr_layers, num_layers)
    sequence_bins = int(sequence_bins)
    circular_windows = bool(circular_windows)

    if sequence_bins <= 0:
        raise ValueError("sequence_bins must be positive.")
    if overlap_strategy != "position_weighted":
        raise ValueError("Currently only overlap_strategy='position_weighted' is supported.")

    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached ESM2 sequence layer bank from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("kind") == "esm2_sequence_layer_bank":
            cached_layers = list(cached.get("layers", []))
            cached_bins = int(cached.get("sequence_bins", -1))
            cached_circular = bool(cached.get("circular_windows", False))
            if cached_layers != layer_ids or cached_bins != sequence_bins or cached_circular != circular_windows:
                raise ValueError(
                    "Cache metadata does not match requested V6 sequence bank. "
                    f"requested layers={layer_ids}, bins={sequence_bins}, circular={circular_windows}; "
                    f"cached layers={cached_layers}, bins={cached_bins}, circular={cached_circular}. "
                    "Use a different cache path or force_rebuild=True."
                )
            return cached
        raise ValueError(
            f"Cache at {cache_path} exists but is not an ESM2 sequence-layer bank. "
            "Use another cache path or force_rebuild=True."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cpu")

    if model_size.lower() in {"t30", "150m", "t33", "650m"}:
        print(
            "Warning: sequence-aware ESM2-t33 extraction is still heavy. "
            "V6 stores selected layers only; use batch_size=1. "
            "Set repr_layer_chunk_size=None to avoid repeated forward passes when memory allows."
        )

    model, alphabet, spec = load_esm2_model(model_size, device)
    batch_converter = alphabet.get_batch_converter()

    unique_targets = sorted([str(x) for x in dataframe[target_col].dropna().unique()])
    lengths = [len(x) for x in unique_targets]
    if lengths:
        print("Unique targets:", len(unique_targets))
        print("Max target length:", max(lengths))
        print(f"Targets longer than {window_len} aa:", sum(L > window_len for L in lengths))
        print("Sequence bins:", sequence_bins)
        print("Selected ESM layers:", layer_ids)
        print("Circular windows:", circular_windows)
        print("Per-target flattened protein dim:", sequence_bins * len(layer_ids) * hidden_dim)

    features: Dict[str, np.ndarray] = {}
    layer_chunks = list(_chunked(layer_ids, repr_layer_chunk_size))

    with torch.no_grad():
        for target_idx, seq in enumerate(tqdm(unique_targets, desc="ESM2 V6 circular/selected-layer bank")):
            seq = str(seq)
            L = len(seq)

            if L == 0:
                features[seq] = np.zeros((sequence_bins, len(layer_ids), hidden_dim), dtype=np.float32)
                continue

            windows = _make_position_mapped_windows(
                seq,
                window_len=window_len,
                stride=stride,
                circular=circular_windows,
            )
            pos_counts = _precompute_position_counts_from_mapped_windows(L, windows)  # [L]
            bin_ids_np = _bin_ids_for_sequence(L, sequence_bins)
            bin_counts_np = np.bincount(bin_ids_np, minlength=sequence_bins).astype(np.float32)
            safe_bin_counts = torch.tensor(bin_counts_np, dtype=torch.float32).clamp(min=1.0).view(sequence_bins, 1)

            # Store bin sums for each selected ESM layer.
            layer_bin_sums = {
                layer: torch.zeros((sequence_bins, hidden_dim), dtype=torch.float32)
                for layer in layer_ids
            }

            for start in range(0, len(windows), batch_size):
                batch_windows = windows[start:start + batch_size]
                batch_data = [
                    (f"target_{target_idx}_window_{start + j}_pos_{pos}", chunk)
                    for j, (pos, chunk, _) in enumerate(batch_windows)
                ]

                _, _, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)

                for layer_chunk in layer_chunks:
                    results = model(
                        batch_tokens,
                        repr_layers=list(layer_chunk),
                        return_contacts=False,
                    )

                    reps_by_layer = {
                        layer: results["representations"][layer].detach().cpu().float()
                        for layer in layer_chunk
                    }

                    for i, (_, chunk, pos_idx_np) in enumerate(batch_windows):
                        chunk_len = len(chunk)
                        if chunk_len != len(pos_idx_np):
                            raise ValueError("Window chunk length and position-index length differ.")

                        pos_idx = torch.tensor(pos_idx_np, dtype=torch.long)
                        bin_idx = torch.tensor(bin_ids_np[pos_idx_np], dtype=torch.long)
                        weights = (1.0 / pos_counts[pos_idx].clamp(min=1.0)).view(chunk_len, 1)

                        for layer in layer_chunk:
                            # ESM token representations include BOS/EOS.
                            residue_repr = reps_by_layer[layer][i, 1:chunk_len + 1]
                            weighted_repr = residue_repr * weights
                            layer_bin_sums[layer].index_add_(0, bin_idx, weighted_repr)

                    del results, reps_by_layer

            layer_bin_features = []
            for layer in layer_ids:
                layer_bin_mean = layer_bin_sums[layer] / safe_bin_counts
                layer_bin_features.append(layer_bin_mean)

            # [sequence_bins, selected_layers, hidden_dim]
            seq_bank = torch.stack(layer_bin_features, dim=1).numpy().astype(np.float32)
            features[seq] = seq_bank

    cache = {
        "kind": "esm2_sequence_layer_bank",
        "version": "v6_circular_selected_layers",
        "model_size": model_size,
        "model_fn": spec["fn"],
        "layers": layer_ids,
        "hidden_dim": hidden_dim,
        "sequence_bins": sequence_bins,
        "window_len": window_len,
        "stride": stride,
        "overlap_strategy": overlap_strategy,
        "circular_windows": circular_windows,
        "features": features,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved ESM2 V6 sequence layer bank to {cache_path}")
    return cache


def _factory_default_esm_sequence_bank_cache_v6(
    self: HW3FeatureFactory,
    model_size: str,
    sequence_bins: int,
    window_len: int,
    stride: int,
    overlap_strategy: str,
    repr_layers: LayerSpec = (4, 8, 12, 16, 20, 24, 28, 33),
    circular_windows: bool = True,
) -> Path:
    spec = get_esm2_spec(model_size)
    layer_key = _layers_key_for_cache(repr_layers, int(spec["num_layers"]))
    circular_key = "circular" if circular_windows else "linear"
    return self.cache_dir / (
        f"{self.dataset_name.lower()}_protein_esm2_{_safe_name(model_size)}_"
        f"seqbins{int(sequence_bins)}_{layer_key}_mean_win{int(window_len)}_"
        f"stride{int(stride)}_{circular_key}_{overlap_strategy}.pkl"
    )


def _factory_build_esm2_sequence_layer_bank_v6(
    self: HW3FeatureFactory,
    model_size: str = "t33",
    sequence_bins: int = 16,
    window_len: int = 1022,
    stride: int = 511,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    overlap_strategy: str = "position_weighted",
    repr_layers: LayerSpec = (4, 8, 12, 16, 20, 24, 28, 33),
    repr_layer_chunk_size: Optional[int] = None,
    circular_windows: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if cache_path is None:
        cache_path = _factory_default_esm_sequence_bank_cache_v6(
            self,
            model_size=model_size,
            sequence_bins=sequence_bins,
            window_len=window_len,
            stride=stride,
            overlap_strategy=overlap_strategy,
            repr_layers=repr_layers,
            circular_windows=circular_windows,
        )
    return build_or_load_esm2_sequence_layer_bank_v6(
        self.df,
        cache_path=cache_path,
        target_col=self.target_col,
        model_size=model_size,
        sequence_bins=sequence_bins,
        window_len=window_len,
        stride=stride,
        batch_size=batch_size,
        device=device if device is not None else self.device,
        overlap_strategy=overlap_strategy,
        repr_layers=repr_layers,
        repr_layer_chunk_size=repr_layer_chunk_size,
        circular_windows=circular_windows,
        force_rebuild=force_rebuild,
    )


def _factory_build_sequence_feature_pack_v6(
    self: HW3FeatureFactory,
    *,
    # Drug options
    use_morgan: bool = True,
    morgan_features: Optional[Mapping[str, Any]] = None,
    morgan_bits: int = 1024,
    morgan_radius: int = 2,
    use_molformer: bool = True,
    molformer_layers: LayerSpec = None,
    molformer_model_name_or_candidates: Union[str, Sequence[str], None] = None,
    molformer_max_length: int = 202,
    molformer_batch_size: int = 64,
    molformer_device: Optional[torch.device] = None,
    molformer_canonicalize: bool = True,
    molformer_remove_isomeric: bool = True,
    molformer_pooling: str = "mean",
    molformer_bank_cache_path: Optional[Union[str, Path]] = None,
    # Protein sequence-bank options
    esm_model_size: str = "t33",
    esm_layers: LayerSpec = (4, 8, 12, 16, 20, 24, 28, 33),
    esm_sequence_bins: int = 16,
    esm_source_sequence_bins: Optional[int] = None,
    esm_output_sequence_bins: Optional[int] = None,
    esm_window_len: int = 1022,
    esm_stride: int = 511,
    esm_batch_size: int = 1,
    esm_device: Optional[torch.device] = None,
    esm_overlap_strategy: str = "position_weighted",
    esm_repr_layer_chunk_size: Optional[int] = None,
    esm_sequence_bank_cache_path: Optional[Union[str, Path]] = None,
    esm_circular_windows: bool = True,
    # General
    force_rebuild_molformer: bool = False,
    force_rebuild_esm: bool = False,
    force_rebuild_morgan: bool = False,
) -> FeaturePack:
    """
    Build a V6 sequence feature pack.

    Drug feature order:
        [Morgan, MolFormer selected layers]

    Protein feature order:
        flattened [sequence_bin, selected_esm_layer, hidden_dim]
    """
    required_drug_keys = [str(x) for x in self.df[self.drug_col].dropna().unique()]
    drug_parts: List[Tuple[str, ArrayDict]] = []

    if use_morgan:
        if morgan_features is None:
            morgan_features = self.build_morgan(
                n_bits=morgan_bits,
                radius=morgan_radius,
                force_rebuild=force_rebuild_morgan,
            )
        morgan_features = {str(k): _ensure_float32_1d(v) for k, v in morgan_features.items()}
        drug_parts.append(("Morgan", dict(morgan_features)))

    molformer_bank = None
    selected_molformer_layers = None
    if use_molformer:
        molformer_bank = self.build_molformer_layer_bank(
            model_name_or_candidates=molformer_model_name_or_candidates,
            max_length=molformer_max_length,
            batch_size=molformer_batch_size,
            device=molformer_device,
            canonicalize=molformer_canonicalize,
            remove_isomeric=molformer_remove_isomeric,
            pooling=molformer_pooling,
            cache_path=molformer_bank_cache_path,
            force_rebuild=force_rebuild_molformer,
        )
        selected_molformer_layers = normalize_layer_spec(
            molformer_layers,
            max(molformer_bank["layers"]),
        )
        molformer_features = select_layers_from_bank(molformer_bank, selected_molformer_layers)
        drug_parts.append((f"MolFormer{selected_molformer_layers}", molformer_features))

    if not drug_parts:
        raise ValueError("At least one of use_morgan or use_molformer must be True.")

    drug_features = drug_parts[0][1]
    drug_name = drug_parts[0][0]
    for part_name, part_features in drug_parts[1:]:
        drug_features = concat_feature_dicts_by_key(
            drug_features,
            part_features,
            required_keys=required_drug_keys,
            left_name=drug_name,
            right_name=part_name,
        )
        drug_name = f"{drug_name}+{part_name}"

    # In V6, circular windows and selected layers change the feature definition;
    # old 128-bin all-layer linear-window caches should not be silently reused.
    source_sequence_bins = int(esm_sequence_bins if esm_source_sequence_bins is None else esm_source_sequence_bins)
    output_sequence_bins = int(esm_sequence_bins if esm_output_sequence_bins is None else esm_output_sequence_bins)
    selected_esm_layers = normalize_layer_spec(esm_layers, get_esm2_spec(esm_model_size)["num_layers"])

    esm_sequence_bank = _factory_build_esm2_sequence_layer_bank_v6(
        self,
        model_size=esm_model_size,
        sequence_bins=source_sequence_bins,
        window_len=esm_window_len,
        stride=esm_stride,
        batch_size=esm_batch_size,
        device=esm_device,
        overlap_strategy=esm_overlap_strategy,
        repr_layers=selected_esm_layers,
        repr_layer_chunk_size=esm_repr_layer_chunk_size,
        circular_windows=esm_circular_windows,
        cache_path=esm_sequence_bank_cache_path,
        force_rebuild=force_rebuild_esm,
    )

    if output_sequence_bins != source_sequence_bins:
        print(f"Coarsening ESM sequence bins: {source_sequence_bins} -> {output_sequence_bins}")
        esm_sequence_bank = coarsen_sequence_layer_bank(
            esm_sequence_bank,
            output_sequence_bins=output_sequence_bins,
            mode="nonzero_mean",
        )

    protein_features = select_sequence_layers_from_bank(esm_sequence_bank, selected_esm_layers)

    pack = FeaturePack(
        drug_features=drug_features,
        protein_features=protein_features,
        config={
            "kind": "sequence_feature_pack_v6_circular_selected_layers",
            "drug_order": [name for name, _ in drug_parts],
            "use_morgan": use_morgan,
            "use_molformer": use_molformer,
            "molformer_layers": selected_molformer_layers,
            "molformer_pooling": molformer_pooling,
            "esm_model_size": esm_model_size,
            "esm_layers": selected_esm_layers,
            "esm_sequence_bins": output_sequence_bins,
            "esm_source_sequence_bins": source_sequence_bins,
            "esm_hidden_dim": esm_sequence_bank["hidden_dim"],
            "esm_window_len": esm_window_len,
            "esm_stride": esm_stride,
            "esm_overlap_strategy": esm_overlap_strategy,
            "esm_circular_windows": bool(esm_circular_windows),
            "protein_layout": "flattened [sequence_bin, selected_esm_layer, hidden_dim]",
        },
        drug_col=self.drug_col,
        target_col=self.target_col,
        label_col=self.label_col,
    )

    print("V6 sequence feature pack built.")
    print("  Drug feature dim:", pack.drug_dim)
    print("  Protein feature dim:", pack.protein_dim)
    print("  Config:", pack.config)
    return pack


# Monkey-patch V6 methods into the existing factory class.
HW3FeatureFactory._default_esm_sequence_bank_cache = _factory_default_esm_sequence_bank_cache_v6
HW3FeatureFactory.build_esm2_sequence_layer_bank = _factory_build_esm2_sequence_layer_bank_v6
HW3FeatureFactory.build_sequence_feature_pack = _factory_build_sequence_feature_pack_v6


class CircularConv2dSeq(nn.Module):
    """Conv2d with circular padding along the sequence-bin dimension only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 1),
        dilation: Tuple[int, int] = (1, 1),
        bias: bool = False,
    ):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.dilation = tuple(dilation)
        if self.kernel_size[1] != 1 or self.dilation[1] != 1:
            raise ValueError("CircularConv2dSeq only supports kernels over sequence bins, i.e. (*, 1).")
        self.pad_seq = ((self.kernel_size[0] - 1) * self.dilation[0]) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=(0, 0),
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_seq > 0:
            # F.pad order for [B, C, T, H] is (H_left, H_right, T_left, T_right).
            x = F.pad(x, (0, 0, self.pad_seq, self.pad_seq), mode="circular")
        return self.conv(x)


class CircularSpliceStyleLayerSequenceMixer(nn.Module):
    """
    Splice-style layer/sequence mixer with circular padding over sequence bins.

    Input:  [B, T, L, H]
    Output: [B, T, H]
    """

    def __init__(
        self,
        esm_num_layers: int = 8,
        esm_hidden_dim: int = 1280,
        layer_cnn_channels: int = 96,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        residual_final_layer: bool = True,
        group_norm_groups: int = 8,
    ):
        super().__init__()
        self.esm_num_layers = int(esm_num_layers)
        self.esm_hidden_dim = int(esm_hidden_dim)
        self.layer_cnn_channels = int(layer_cnn_channels)
        self.protein_layer_dropout = float(protein_layer_dropout)
        self.residual_final_layer = bool(residual_final_layer)
        self.dilations = tuple(int(d) for d in dilations)

        groups = _valid_group_count_v4(self.layer_cnn_channels, preferred=group_norm_groups)
        self.input_norm = nn.LayerNorm(self.esm_hidden_dim)

        self.stem = nn.Sequential(
            CircularConv2dSeq(
                in_channels=self.esm_num_layers,
                out_channels=self.layer_cnn_channels,
                kernel_size=(3, 1),
                dilation=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dilated_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
                nn.GELU(),
                CircularConv2dSeq(
                    in_channels=self.layer_cnn_channels,
                    out_channels=self.layer_cnn_channels,
                    kernel_size=(3, 1),
                    dilation=(d, 1),
                    bias=False,
                ),
                nn.GroupNorm(num_groups=groups, num_channels=self.layer_cnn_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=self.layer_cnn_channels,
                    out_channels=self.layer_cnn_channels,
                    kernel_size=(1, 1),
                    bias=True,
                ),
                nn.Dropout(dropout),
            )
            for d in self.dilations
        ])

        self.collapse = nn.Conv2d(
            in_channels=self.layer_cnn_channels,
            out_channels=1,
            kernel_size=(1, 1),
            bias=True,
        )
        self.output_norm = nn.LayerNorm(self.esm_hidden_dim)

        if self.residual_final_layer:
            self.final_scale = nn.Parameter(torch.tensor(1.0))
            self.cnn_scale = nn.Parameter(torch.tensor(0.25))
        else:
            self.final_scale = None
            self.cnn_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, protein_seq: torch.Tensor) -> torch.Tensor:
        if protein_seq.ndim != 4:
            raise ValueError(
                f"Expected protein_seq [B, T, L, H], got shape {tuple(protein_seq.shape)}"
            )
        if protein_seq.size(2) != self.esm_num_layers:
            raise ValueError(f"Expected {self.esm_num_layers} ESM layers, got {protein_seq.size(2)}")
        if protein_seq.size(3) != self.esm_hidden_dim:
            raise ValueError(f"Expected hidden dim {self.esm_hidden_dim}, got {protein_seq.size(3)}")

        x = self.input_norm(protein_seq)

        if self.training and self.protein_layer_dropout > 0.0:
            keep_prob = 1.0 - self.protein_layer_dropout
            mask = torch.empty(
                x.size(0), 1, x.size(2), 1,
                dtype=x.dtype,
                device=x.device,
            ).bernoulli_(keep_prob)
            x = x * mask / max(keep_prob, 1e-6)

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, T, H]
        x = self.stem(x)
        for block in self.dilated_blocks:
            x = x + block(x)
        cnn_out = self.collapse(x).squeeze(1)  # [B, T, H]

        if self.residual_final_layer:
            final_seq = protein_seq[:, :, -1, :]
            out = self.final_scale * final_seq + self.cnn_scale * cnn_out
        else:
            out = self.cnn_scale * cnn_out

        return self.output_norm(out)


class CircularDilatedResidualConv1DBlock(nn.Module):
    """Depthwise-separable residual dilated Conv1d block with circular padding."""

    def __init__(self, dim: int, dilation: int = 1, dropout: float = 0.20):
        super().__init__()
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=0,
            dilation=self.dilation,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2).contiguous()  # [B, C, T]
        if self.dilation > 0:
            y = F.pad(y, (self.dilation, self.dilation), mode="circular")
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise(y)
        y = self.drop(y).transpose(1, 2).contiguous()
        return x + y


class CircularSpliceStyleESMSequenceCNNMorganMolFormerDTA(SpliceStyleESMSequenceCNNMorganMolFormerDTA):
    """
    V6 sequence-aware model.

    Same fusion design as v5-lite, but the sequence-bin convolutions use circular
    padding. This matches the feature extractor's circular long-protein window
    assumption, where the N- and C-termini are treated as adjacent on a ring.
    """

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        morgan_dim: int = 1024,
        sequence_bins: int = 16,
        esm_num_layers: int = 8,
        esm_hidden_dim: int = 1280,
        layer_cnn_channels: int = 96,
        layer_cnn_dilations: Sequence[int] = (1, 2, 4, 8),
        seq_model_dim: int = 384,
        seq_dilations: Sequence[int] = (1, 2, 4, 8),
        proj_dim: int = 384,
        hidden_dim: int = 768,
        dropout: float = 0.20,
        protein_layer_dropout: float = 0.05,
        residual_final_layer: bool = True,
        circular_padding: bool = True,
    ):
        super().__init__(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            morgan_dim=morgan_dim,
            sequence_bins=sequence_bins,
            esm_num_layers=esm_num_layers,
            esm_hidden_dim=esm_hidden_dim,
            layer_cnn_channels=layer_cnn_channels,
            layer_cnn_dilations=layer_cnn_dilations,
            seq_model_dim=seq_model_dim,
            seq_dilations=seq_dilations,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            protein_layer_dropout=protein_layer_dropout,
            residual_final_layer=residual_final_layer,
        )
        self.circular_padding = bool(circular_padding)
        if self.circular_padding:
            self.layer_sequence_mixer = CircularSpliceStyleLayerSequenceMixer(
                esm_num_layers=self.esm_num_layers,
                esm_hidden_dim=self.esm_hidden_dim,
                layer_cnn_channels=layer_cnn_channels,
                dilations=layer_cnn_dilations,
                dropout=dropout,
                protein_layer_dropout=protein_layer_dropout,
                residual_final_layer=residual_final_layer,
            )
            self.seq_conv_blocks = nn.ModuleList([
                CircularDilatedResidualConv1DBlock(seq_model_dim, dilation=d, dropout=dropout)
                for d in seq_dilations
            ])


# Notebook-friendly alias for the V6 experiment.
MyImprovedModel = CircularSpliceStyleESMSequenceCNNMorganMolFormerDTA
