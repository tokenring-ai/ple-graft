"""Pinned local paths and checkpoint-derived PLE constants.

Hash multipliers, head offsets, and vocab sizes are loaded from the donor
checkpoint / pack manifest. The literals below are the expected values used
as a fail-fast check; they are not a substitute for the on-disk tensors.

Local directories are overridable with `PLE_GRAFT_*` environment variables
(see README). Defaults match the original experiment machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


PACK_DIR = _env_path("PLE_GRAFT_PACK_DIR", "/opt/llm/tr-infer/flashnext-nvfp4")
PACK_MANIFEST = PACK_DIR / "manifest.json"
PLE_BIN = PACK_DIR / "ple.bin"
PACK_TOKENIZER = PACK_DIR / "tokenizer.json"

HF_DONOR_DIR = _env_path("PLE_GRAFT_DONOR_DIR", "/opt/llm/Qwen3.8-Flash-Next-NVFP4")
HF_DONOR_CONFIG = HF_DONOR_DIR / "config.json"
HF_DONOR_TOKENIZER = HF_DONOR_DIR / "tokenizer.json"
HF_DONOR_TOKENIZER_CONFIG = HF_DONOR_DIR / "tokenizer_config.json"
HF_DONOR_INDEX = HF_DONOR_DIR / "model.safetensors.index.json"

RECIPIENT_DIR = _env_path("PLE_GRAFT_RECIPIENT_DIR", "/mnt/llm-cache/models/Qwen3.5-4B-Base")
RECIPIENT_REPO_ID = "Qwen/Qwen3.5-4B-Base"
MOE_RECIPIENT_DIR = _env_path("PLE_GRAFT_MOE_DIR", "/opt/llm/Qwen3.6-35B-A3B")
MOE_RECIPIENT_REPO_ID = "Qwen/Qwen3.6-35B-A3B"
MOE_HIDDEN_SIZE = 2048
PLE_CONCAT_DIM = 2560

SMOKE_CORPUS = _env_path(
    "PLE_GRAFT_SMOKE_CORPUS",
    "/home/mdierolf/gitprojects/ninfer/eval/corpora/perplexity-1m",
)

# HF names for the hash-constant tensors (0-based layer 1).
HF_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding."
HF_LAYER_MULTIPLIERS = HF_PLE_PREFIX + "layer_multipliers"
HF_HEAD_OFFSETS = HF_PLE_PREFIX + "ngram_heads_offsets"
HF_HEAD_VOCAB_SIZES = HF_PLE_PREFIX + "ngram_heads_vocab_sizes"
HF_WEIGHT_SCALE = HF_PLE_PREFIX + "ngram_embedding.weight_scale"
HF_SHARD_KEY = HF_PLE_PREFIX + "ngram_embedding.shard_{i}.weight"

# PLE hash resets n-grams on this id (HF text_config.eos_token_id / <|endoftext|>).
# Chat generation EOS (im_end=248046) is a different token and must not be used here.
PLE_EOS_TOKEN_ID = 248044
PLE_IMAGE_TOKEN_ID = 248056

NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 16
HEAD_DIM = 160
PLE_EMBED_DIM = N_HEADS * HEAD_DIM  # 2560
SPLIT_NGRAM_PARTS = 128

# 1-based HF ple_layer_ids=[2] → 0-based recipient/donor layer index 1.
PLE_LAYER_INDEX_0BASED = 1

# Packed table geometry (from per_layer_token_embd.weight in the pack manifest).
PACKED_PHYSICAL_ROWS = 320_001_536
PACKED_PAYLOAD_NBYTES = PACKED_PHYSICAL_ROWS * HEAD_DIM  # 51_200_245_760
PACKED_SCALE_OFFSET = PACKED_PAYLOAD_NBYTES
PACKED_SCALE_NBYTES = 2
PACKED_FILE_NBYTES = PACKED_PAYLOAD_NBYTES + PACKED_SCALE_NBYTES

EXPECTED_LAYER_MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
EXPECTED_HEAD_OFFSETS = (
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
)
EXPECTED_HEAD_VOCAB_SIZES = (
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
)
EXPECTED_LOGICAL_ROWS = sum(EXPECTED_HEAD_VOCAB_SIZES)  # 320_001_446

PADDED_VOCAB_SIZE = 248_320
HIDDEN_SIZE = 2560


@dataclass(frozen=True)
class PleConfig:
    ngram_size: int
    heads_per_ngram: int
    head_dim: int
    eos_token_id: int
    layer_index_0based: int
    layer_multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    physical_rows: int
    payload_nbytes: int
    scale_offset: int
    scale_convention: str = "multiply"
    scale_type: str = "bf16"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def logical_rows(self) -> int:
        return int(sum(self.head_vocab_sizes))

    @property
    def embed_dim(self) -> int:
        return self.n_heads * self.head_dim


def expected_ple_config() -> PleConfig:
    return PleConfig(
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        head_dim=HEAD_DIM,
        eos_token_id=PLE_EOS_TOKEN_ID,
        layer_index_0based=PLE_LAYER_INDEX_0BASED,
        layer_multipliers=EXPECTED_LAYER_MULTIPLIERS,
        head_offsets=EXPECTED_HEAD_OFFSETS,
        head_vocab_sizes=EXPECTED_HEAD_VOCAB_SIZES,
        physical_rows=PACKED_PHYSICAL_ROWS,
        payload_nbytes=PACKED_PAYLOAD_NBYTES,
        scale_offset=PACKED_SCALE_OFFSET,
    )
