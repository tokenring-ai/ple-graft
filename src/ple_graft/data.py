"""Packed uint32 token streams for causal LM windows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(
    os.environ.get("PLE_GRAFT_DATA_DIR", "/mnt/llm-cache/ple-graft/data")
).expanduser()


def load_tokens(path: Path) -> np.memmap:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint32, mode="r")


def n_windows(n_tokens: int, seq_len: int) -> int:
    if n_tokens < seq_len + 1:
        return 0
    return (n_tokens - 1) // seq_len


def window_at(tokens: np.ndarray, index: int, seq_len: int) -> np.ndarray:
    """Return `seq_len` tokens starting at `index * seq_len` (non-overlapping)."""
    start = int(index) * seq_len
    end = start + seq_len
    if end > tokens.shape[0]:
        raise IndexError(f"window {index} out of range")
    return np.asarray(tokens[start:end], dtype=np.int64)


def batch_windows(tokens: np.ndarray, indices: list[int], seq_len: int) -> np.ndarray:
    return np.stack([window_at(tokens, i, seq_len) for i in indices], axis=0)


def write_manifest(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
