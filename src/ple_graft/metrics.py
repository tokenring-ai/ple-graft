"""Metrics helpers for run-of-record logging."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def nll_to_ppl(nll: float) -> float:
    return math.exp(nll)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def paired_delta_stats(nll_a, nll_b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired Δ = mean(nll_a - nll_b). Negative means A is better (lower NLL)."""
    import numpy as np

    a = np.asarray(nll_a, dtype=np.float64).reshape(-1)
    b = np.asarray(nll_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = a - b
    mean = float(d.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[draws].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n_windows": int(d.size),
        "mean_delta": mean,
        "ci95": [float(lo), float(hi)],
        "frac_a_better": float((d < 0).mean()),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0,
    }
