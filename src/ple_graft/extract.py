"""Load PLE constants from the pack manifest and HF safetensors; fail on mismatch."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from .donor_assets import (
    EXPECTED_HEAD_OFFSETS,
    EXPECTED_HEAD_VOCAB_SIZES,
    EXPECTED_LAYER_MULTIPLIERS,
    EXPECTED_LOGICAL_ROWS,
    HF_DONOR_CONFIG,
    HF_DONOR_DIR,
    HF_DONOR_INDEX,
    HF_HEAD_OFFSETS,
    HF_HEAD_VOCAB_SIZES,
    HF_LAYER_MULTIPLIERS,
    HF_SHARD_KEY,
    HF_WEIGHT_SCALE,
    PACK_MANIFEST,
    PACKED_FILE_NBYTES,
    PACKED_PHYSICAL_ROWS,
    PLE_BIN,
    PLE_EOS_TOKEN_ID,
    PLE_LAYER_INDEX_0BASED,
    SPLIT_NGRAM_PARTS,
    PleConfig,
    expected_ple_config,
)
from .fp8 import bf16_bytes_to_f32


def sha256_file(path: Path, buf: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_pack_manifest(path: Path = PACK_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text())


def pack_ple_config(manifest: dict[str, Any] | None = None) -> PleConfig:
    man = manifest if manifest is not None else load_pack_manifest()
    cfg = man["config"]
    tensor = next(t for t in man["tensors"] if t["name"] == "per_layer_token_embd.weight")
    fp8 = tensor["fp8"]
    return PleConfig(
        ngram_size=int(cfg["qwen4exp.ple.ngram_size"]),
        heads_per_ngram=int(cfg["qwen4exp.ple.heads_per_ngram"]),
        head_dim=int(cfg["qwen4exp.embedding_length_per_layer_input"]),
        eos_token_id=int(cfg["qwen4exp.ple.eos_token_id"]),
        layer_index_0based=int(cfg["qwen4exp.ple.layers"][0]),
        layer_multipliers=tuple(int(x) for x in cfg["qwen4exp.ple.layer_multipliers"]),
        head_offsets=tuple(int(x) for x in cfg["qwen4exp.ple.head_offsets"]),
        head_vocab_sizes=tuple(int(x) for x in cfg["qwen4exp.ple.head_vocab_sizes"]),
        physical_rows=int(tensor["rows"]),
        payload_nbytes=int(fp8["payload"]["nbytes"]),
        scale_offset=int(fp8["scales"]["offset"]),
        scale_convention=str(fp8["scale_convention"]),
        scale_type=str(fp8["scale_type"]),
        extra={
            "file": tensor["file"],
            "tensor_offset": tensor["offset"],
            "nbytes": tensor["nbytes"],
            "identity": man.get("identity"),
            "source_path": man.get("source", {}).get("path"),
        },
    )


def _tensor_file(index: dict[str, Any], name: str) -> Path:
    rel = index["weight_map"][name]
    return HF_DONOR_DIR / rel


def _read_i64(path: Path, name: str) -> list[int]:
    with safe_open(str(path), framework="numpy") as fh:
        arr = fh.get_tensor(name)
    return [int(x) for x in np.asarray(arr).reshape(-1).tolist()]


def _read_bf16_scalar(path: Path, name: str) -> float:
    """Read a 1-element BF16 tensor without requiring numpy's bfloat16 dtype."""
    from .fp8 import bf16_bytes_to_f32

    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    info = header[name]
    begin, end = info["data_offsets"]
    if end - begin < 2:
        raise ValueError(f"{name} is not a bf16 scalar")
    with path.open("rb") as fh:
        fh.seek(8 + n + int(begin))
        return bf16_bytes_to_f32(fh.read(2))


def hf_ple_constants() -> dict[str, Any]:
    index = json.loads(HF_DONOR_INDEX.read_text())
    hf_cfg = json.loads(HF_DONOR_CONFIG.read_text())
    text = hf_cfg["text_config"]
    multipliers = _read_i64(_tensor_file(index, HF_LAYER_MULTIPLIERS), HF_LAYER_MULTIPLIERS)
    offsets = _read_i64(_tensor_file(index, HF_HEAD_OFFSETS), HF_HEAD_OFFSETS)
    vocabs = _read_i64(_tensor_file(index, HF_HEAD_VOCAB_SIZES), HF_HEAD_VOCAB_SIZES)
    scale_path = _tensor_file(index, HF_WEIGHT_SCALE)
    scale = _read_bf16_scalar(scale_path, HF_WEIGHT_SCALE)
    return {
        "ple_layer_ids": list(text["ple_layer_ids"]),
        "ngram_size": int(text["ngram_size"]),
        "heads_per_ngram": int(text["heads_per_ngram"]),
        "ple_embed_dim": int(text["ple_embed_dim"]),
        "eos_token_id": int(text["eos_token_id"]),
        "layer_multipliers": multipliers,
        "head_offsets": offsets,
        "head_vocab_sizes": vocabs,
        "weight_scale": scale,
        "split_ngram_parts": int(text["split_ngram_parts"]),
        "ple_embedding_dtype": text.get("ple_embedding_dtype"),
    }


def hf_shard_layout() -> list[dict[str, Any]]:
    """Prefix-sum of ngram_embedding.shard_i.weight rows across the 128 shards."""
    index = json.loads(HF_DONOR_INDEX.read_text())
    layout = []
    cursor = 0
    for i in range(SPLIT_NGRAM_PARTS):
        name = HF_SHARD_KEY.format(i=i)
        path = _tensor_file(index, name)
        header = _safetensors_header(path)
        info = header[name]
        rows, cols = info["shape"]
        layout.append(
            {
                "index": i,
                "name": name,
                "file": str(path),
                "rows": int(rows),
                "cols": int(cols),
                "row_start": cursor,
                "dtype": info["dtype"],
            }
        )
        cursor += int(rows)
    return layout


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def _safetensors_mmap_rows(path: Path, name: str) -> np.memmap:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    info = header[name]
    rows, cols = info["shape"]
    begin, _end = info["data_offsets"]
    offset = 8 + n + int(begin)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(int(rows), int(cols)))


def read_hf_rows(layout: list[dict[str, Any]], row_ids: list[int]) -> np.ndarray:
    """Read raw FP8 rows `[N, 160]` from HF shards for the given global row ids."""
    if not row_ids:
        return np.empty((0, 160), dtype=np.uint8)
    starts = np.array([s["row_start"] for s in layout], dtype=np.int64)
    rows_per = np.array([s["rows"] for s in layout], dtype=np.int64)
    ends = starts + rows_per
    out = np.empty((len(row_ids), layout[0]["cols"]), dtype=np.uint8)
    groups: dict[int, list[tuple[int, int]]] = {}
    for out_i, rid in enumerate(row_ids):
        shard = int(np.searchsorted(ends, rid, side="right"))
        if shard >= len(layout) or rid < starts[shard]:
            raise IndexError(f"row {rid} not in HF shard layout")
        groups.setdefault(shard, []).append((out_i, int(rid - starts[shard])))
    opened: dict[tuple[str, str], np.memmap] = {}
    try:
        for shard, items in groups.items():
            spec = layout[shard]
            key = (spec["file"], spec["name"])
            mm = opened.get(key)
            if mm is None:
                mm = _safetensors_mmap_rows(Path(spec["file"]), spec["name"])
                opened[key] = mm
            for out_i, local in items:
                out[out_i] = np.array(mm[local], dtype=np.uint8, copy=True)
    finally:
        for mm in opened.values():
            if getattr(mm, "_mmap", None) is not None:
                mm._mmap.close()
    return out


def packed_scale(path: Path = PLE_BIN, cfg: PleConfig | None = None) -> float:
    cfg = cfg or expected_ple_config()
    with path.open("rb") as fh:
        fh.seek(cfg.scale_offset)
        return bf16_bytes_to_f32(fh.read(2))


def mismatches(pack: PleConfig, hf: dict[str, Any]) -> list[str]:
    problems = []
    if pack.layer_multipliers != tuple(hf["layer_multipliers"]):
        problems.append("layer_multipliers pack != HF")
    if pack.head_offsets != tuple(hf["head_offsets"]):
        problems.append("head_offsets pack != HF")
    if pack.head_vocab_sizes != tuple(hf["head_vocab_sizes"]):
        problems.append("head_vocab_sizes pack != HF")
    if pack.layer_multipliers != EXPECTED_LAYER_MULTIPLIERS:
        problems.append("layer_multipliers != pinned expected")
    if pack.head_offsets != EXPECTED_HEAD_OFFSETS:
        problems.append("head_offsets != pinned expected")
    if pack.head_vocab_sizes != EXPECTED_HEAD_VOCAB_SIZES:
        problems.append("head_vocab_sizes != pinned expected")
    if pack.eos_token_id != PLE_EOS_TOKEN_ID or hf["eos_token_id"] != PLE_EOS_TOKEN_ID:
        problems.append(f"eos mismatch pack={pack.eos_token_id} hf={hf['eos_token_id']}")
    if pack.layer_index_0based != PLE_LAYER_INDEX_0BASED:
        problems.append(f"layer index {pack.layer_index_0based} != {PLE_LAYER_INDEX_0BASED}")
    hf_layers = [int(x) - 1 for x in hf["ple_layer_ids"]]
    if hf_layers != [pack.layer_index_0based]:
        problems.append(f"HF ple_layer_ids {hf['ple_layer_ids']} vs pack layer {pack.layer_index_0based}")
    if pack.logical_rows != EXPECTED_LOGICAL_ROWS:
        problems.append(f"logical rows {pack.logical_rows} != {EXPECTED_LOGICAL_ROWS}")
    if pack.physical_rows != PACKED_PHYSICAL_ROWS:
        problems.append(f"physical rows {pack.physical_rows} != {PACKED_PHYSICAL_ROWS}")
    if pack.physical_rows < pack.logical_rows:
        problems.append("physical table shorter than logical heads")
    return problems


def build_manifest(
    include_sha256: bool = False,
) -> dict[str, Any]:
    pack = pack_ple_config()
    hf = hf_ple_constants()
    problems = mismatches(pack, hf)
    scale = packed_scale(PLE_BIN, pack)
    size = PLE_BIN.stat().st_size
    if size != PACKED_FILE_NBYTES and size < pack.scale_offset + 2:
        problems.append(f"ple.bin size {size} too small")
    out: dict[str, Any] = {
        "pack": {
            "manifest": str(PACK_MANIFEST),
            "ple_bin": str(PLE_BIN),
            "ple_bin_bytes": size,
            "config": {
                "ngram_size": pack.ngram_size,
                "heads_per_ngram": pack.heads_per_ngram,
                "head_dim": pack.head_dim,
                "eos_token_id": pack.eos_token_id,
                "layer_index_0based": pack.layer_index_0based,
                "layer_multipliers": list(pack.layer_multipliers),
                "head_offsets": list(pack.head_offsets),
                "head_vocab_sizes": list(pack.head_vocab_sizes),
                "logical_rows": pack.logical_rows,
                "physical_rows": pack.physical_rows,
                "payload_nbytes": pack.payload_nbytes,
                "scale_offset": pack.scale_offset,
                "scale_convention": pack.scale_convention,
                "packed_scale": scale,
            },
            "extra": pack.extra,
        },
        "hf": hf,
        "ok": not problems,
        "problems": problems,
    }
    if include_sha256:
        out["pack"]["ple_bin_sha256"] = sha256_file(PLE_BIN)
    return out
