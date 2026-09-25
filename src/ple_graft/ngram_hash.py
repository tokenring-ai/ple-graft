"""Exact Flash-Next PLE n-gram hash.

Reference: llama.cpp / SGLang. Arithmetic is uint64 multiply, XOR, then
`mixed % vocab + offset`. Do not reconstruct multipliers; pass checkpoint values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .donor_assets import PleConfig, expected_ple_config

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _as_u64(x: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.uint64).reshape(-1)


def context_windows(
    token_ids: Sequence[int] | np.ndarray,
    ngram_size: int,
    eos_token_id: int,
) -> np.ndarray:
    """Per-position n-gram windows.

    `out[t, 0]` is token t. `out[t, j]` is token t-j, or EOS if t-j < 0 or if
    any token in (t-j, t] is EOS (n-gram cut). Shape `[T, ngram_size]`.
    """
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    t_len = int(ids.shape[0])
    out = np.empty((t_len, ngram_size), dtype=np.int64)
    eos = int(eos_token_id)
    for t in range(t_len):
        out[t, 0] = ids[t]
        cut = False
        for s in range(1, ngram_size):
            if cut or t - s < 0:
                out[t, s] = eos
            else:
                out[t, s] = ids[t - s]
            if out[t, s] == eos:
                cut = True
    return out


def hash_windows(
    windows: np.ndarray,
    multipliers: Sequence[int],
    head_vocab_sizes: Sequence[int],
    head_offsets: Sequence[int],
    heads_per_ngram: int,
) -> np.ndarray:
    """Hash `[T, ngram_size]` windows to `[T, n_heads]` global row ids."""
    win = np.asarray(windows, dtype=np.int64)
    if win.ndim != 2:
        raise ValueError(f"windows must be [T, ngram], got {win.shape}")
    t_len, ngram_size = win.shape
    mult = _as_u64(multipliers)
    vocabs = _as_u64(head_vocab_sizes)
    offsets = _as_u64(head_offsets)
    if mult.shape[0] != ngram_size:
        raise ValueError("multiplier count must equal ngram_size")
    n_heads = int(vocabs.shape[0])
    if n_heads != (ngram_size - 1) * heads_per_ngram:
        raise ValueError("head count must be (ngram_size-1)*heads_per_ngram")
    tok_u = win.astype(np.uint64, copy=False)
    out = np.empty((t_len, n_heads), dtype=np.int64)
    for n in range(2, ngram_size + 1):
        mixed = tok_u[:, 0] * mult[0]
        for j in range(1, n):
            mixed = np.bitwise_xor(mixed, tok_u[:, j] * mult[j])
        mixed = mixed & MASK64
        base = (n - 2) * heads_per_ngram
        for g in range(heads_per_ngram):
            h = base + g
            out[:, h] = (mixed % vocabs[h] + offsets[h]).astype(np.int64)
    return out


def hash_token_ids(
    token_ids: Sequence[int] | np.ndarray,
    cfg: PleConfig | None = None,
) -> np.ndarray:
    """Token ids `[T]` → row ids `[T, 16]`."""
    cfg = cfg or expected_ple_config()
    windows = context_windows(token_ids, cfg.ngram_size, cfg.eos_token_id)
    return hash_windows(
        windows,
        cfg.layer_multipliers,
        cfg.head_vocab_sizes,
        cfg.head_offsets,
        cfg.heads_per_ngram,
    )


def hash_batch(
    token_ids: np.ndarray,
    cfg: PleConfig | None = None,
) -> np.ndarray:
    """Token ids `[B, S]` → row ids `[B, S, 16]`. Each row is hashed independently."""
    cfg = cfg or expected_ple_config()
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise ValueError(f"expected [B, S], got {ids.shape}")
    batch, seq = ids.shape
    out = np.empty((batch, seq, cfg.n_heads), dtype=np.int64)
    for b in range(batch):
        out[b] = hash_token_ids(ids[b], cfg)
    return out
