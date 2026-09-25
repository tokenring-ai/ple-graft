"""Logical-row access over the packed FP8 PLE table (mmap, host-side)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .donor_assets import PLE_BIN, PleConfig, expected_ple_config
from .fp8 import bf16_bytes_to_f32, e4m3fn_to_f32


@dataclass
class PleStore:
    path: Path
    cfg: PleConfig
    scale: float
    _mmap: np.memmap

    @property
    def head_dim(self) -> int:
        return self.cfg.head_dim

    @property
    def physical_rows(self) -> int:
        return self.cfg.physical_rows

    @property
    def logical_rows(self) -> int:
        return self.cfg.logical_rows

    def close(self) -> None:
        if hasattr(self._mmap, "_mmap") and self._mmap._mmap is not None:
            self._mmap._mmap.close()

    def raw_rows(self, row_ids: np.ndarray) -> np.ndarray:
        """Gather FP8 bytes. `row_ids` any shape → `[..., head_dim]` uint8."""
        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if flat.size == 0:
            return np.empty(ids.shape + (self.head_dim,), dtype=np.uint8)
        if np.any(flat < 0) or np.any(flat >= self.cfg.physical_rows):
            raise IndexError(
                f"PLE row id out of range [0, {self.cfg.physical_rows}): "
                f"min={int(flat.min())} max={int(flat.max())}"
            )
        gathered = np.ascontiguousarray(self._mmap[flat])
        return gathered.reshape(ids.shape + (self.head_dim,))

    def rows_f32(self, row_ids: np.ndarray) -> np.ndarray:
        """Dequantized rows: e4m3fn * tensor scale. Float32.

        Unique row IDs are gathered once (output-preserving).
        """
        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if flat.size == 0:
            return np.empty(ids.shape + (self.head_dim,), dtype=np.float32)
        unique, inverse = np.unique(flat, return_inverse=True)
        raw = self.raw_rows(unique)
        decoded = e4m3fn_to_f32(raw)
        if self.cfg.scale_convention != "multiply":
            raise ValueError(f"unsupported scale convention {self.cfg.scale_convention}")
        decoded = decoded * np.float32(self.scale)
        # e4m3fn exp=15 is NaN; a single NaN would poison 0 @ W_up. Flush to 0.
        decoded = np.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
        return decoded[inverse].reshape(ids.shape + (self.head_dim,))


def open_store(path: Path | None = None, cfg: PleConfig | None = None) -> PleStore:
    cfg = cfg or expected_ple_config()
    path = Path(path) if path is not None else PLE_BIN
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    expected = cfg.payload_nbytes + cfg.scale_offset - cfg.payload_nbytes + 2
    # scale lives at scale_offset; file must cover payload + 2-byte scale.
    min_size = cfg.scale_offset + 2
    if size < min_size:
        raise ValueError(f"{path} is {size} bytes, need at least {min_size}")
    with path.open("rb") as fh:
        fh.seek(cfg.scale_offset)
        scale = bf16_bytes_to_f32(fh.read(2))
    mmap = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=0,
        shape=(cfg.physical_rows, cfg.head_dim),
    )
    return PleStore(path=path, cfg=cfg, scale=float(scale), _mmap=mmap)
