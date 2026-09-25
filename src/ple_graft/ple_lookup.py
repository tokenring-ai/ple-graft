"""Token ids → PLE rows. Control modes (zero / random / shuffled) are lookup-side."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .donor_assets import PleConfig, expected_ple_config
from .ngram_hash import hash_batch, hash_token_ids
from .ple_store import PleStore, open_store


class LookupMode(str, Enum):
    REAL = "real"
    ZERO = "zero"
    RANDOM = "random"
    SHUFFLED = "shuffled"


@dataclass
class LookupStats:
    n_tokens: int = 0
    n_row_ids: int = 0
    n_unique: int = 0
    bytes_read: int = 0
    hash_ms: float = 0.0
    gather_ms: float = 0.0


@dataclass
class LookupResult:
    row_ids: np.ndarray  # [B, S, 16] or [S, 16]
    rows: np.ndarray  # [..., 16, 160]
    concat: np.ndarray  # [..., 2560]
    stats: LookupStats = field(default_factory=LookupStats)


def _as_numpy_ids(token_ids) -> np.ndarray:
    if hasattr(token_ids, "detach"):
        return token_ids.detach().to("cpu").numpy().astype(np.int64, copy=False)
    return np.asarray(token_ids, dtype=np.int64)


class PleLookup:
    def __init__(
        self,
        store: PleStore | None = None,
        cfg: PleConfig | None = None,
        mode: LookupMode = LookupMode.REAL,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or (store.cfg if store is not None else expected_ple_config())
        self.store = store
        self.mode = LookupMode(mode)
        self.seed = int(seed)
        self.last_stats = LookupStats()
        self.head_mask: np.ndarray | None = None
        self._perm: np.ndarray | None = None
        # SHUFFLED and RANDOM both apply a stored permutation of real row ids so
        # the map is independent of batch shape. RANDOM uses a different stream
        # than SHUFFLED (seed, 0xC1) so they are not the same control.
        if self.mode == LookupMode.SHUFFLED:
            rng = np.random.default_rng(self.seed)
            self._perm = rng.permutation(self.cfg.logical_rows).astype(np.int64)
        elif self.mode == LookupMode.RANDOM:
            rng = np.random.default_rng((int(self.seed), 0xC1))
            self._perm = rng.permutation(self.cfg.logical_rows).astype(np.int64)

    def row_ids(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim == 1:
            hashed = hash_token_ids(ids, self.cfg)
        elif ids.ndim == 2:
            hashed = hash_batch(ids, self.cfg)
        else:
            raise ValueError(f"token_ids rank must be 1 or 2, got {ids.shape}")
        if self.mode in (LookupMode.SHUFFLED, LookupMode.RANDOM):
            assert self._perm is not None
            hashed = self._perm[hashed]
        return hashed

    def set_head_mask(self, keep: Sequence[int] | None) -> None:
        """Keep listed heads (0..15); others zeroed in concat. None = all heads."""
        if keep is None:
            self.head_mask = None
            return
        mask = np.zeros(self.cfg.n_heads, dtype=bool)
        for h in keep:
            hi = int(h)
            if hi < 0 or hi >= self.cfg.n_heads:
                raise IndexError(h)
            mask[hi] = True
        self.head_mask = mask

    def lookup(self, token_ids) -> LookupResult:
        ids = _as_numpy_ids(token_ids)
        t0 = time.perf_counter()
        row_ids = self.row_ids(ids)
        hash_ms = (time.perf_counter() - t0) * 1e3
        cfg = self.cfg
        t1 = time.perf_counter()
        if self.mode == LookupMode.ZERO:
            rows = np.zeros(row_ids.shape + (cfg.head_dim,), dtype=np.float32)
            n_unique = 0
        else:
            if self.store is None:
                raise RuntimeError("lookup needs a store")
            rows = self.store.rows_f32(row_ids)
            n_unique = int(np.unique(row_ids.reshape(-1)).size)
        gather_ms = (time.perf_counter() - t1) * 1e3
        concat = rows.reshape(*row_ids.shape[:-1], cfg.n_heads * cfg.head_dim)
        if self.head_mask is not None:
            concat = np.array(concat, copy=True, dtype=np.float32)
            hd = cfg.head_dim
            for h, keep in enumerate(self.head_mask):
                if not keep:
                    concat[..., h * hd : (h + 1) * hd] = 0
        stats = LookupStats(
            n_tokens=int(ids.size),
            n_row_ids=int(row_ids.size),
            n_unique=n_unique,
            bytes_read=n_unique * cfg.head_dim,
            hash_ms=hash_ms,
            gather_ms=gather_ms,
        )
        self.last_stats = stats
        return LookupResult(row_ids=row_ids, rows=rows, concat=concat, stats=stats)

    def lookup_torch(self, token_ids, device, dtype):
        import torch

        result = self.lookup(token_ids)
        concat = torch.from_numpy(np.ascontiguousarray(result.concat))
        return concat.to(device=device, dtype=dtype), result


def default_lookup(mode: LookupMode = LookupMode.REAL, seed: int = 0) -> PleLookup:
    store = open_store()
    return PleLookup(store=store, mode=mode, seed=seed)
