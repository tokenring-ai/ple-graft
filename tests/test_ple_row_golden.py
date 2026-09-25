from __future__ import annotations

import numpy as np

from ple_graft.donor_assets import expected_ple_config
from ple_graft.extract import hf_ple_constants, hf_shard_layout, packed_scale, read_hf_rows
from ple_graft.fp8 import e4m3fn_to_f32
from ple_graft.ngram_hash import hash_token_ids
from ple_graft.ple_store import open_store
from ple_graft.tokenizer_compat import encode_ids, load_tokenizer
from ple_graft.donor_assets import HF_DONOR_TOKENIZER


def test_packed_scale_matches_hf_weight_scale():
    hf = hf_ple_constants()
    packed = packed_scale()
    np.testing.assert_allclose(packed, hf["weight_scale"], rtol=0, atol=0)


def test_physical_table_covers_logical_heads(ple_bin):
    store = open_store(ple_bin)
    cfg = store.cfg
    assert store.physical_rows >= store.logical_rows
    last = cfg.head_offsets[-1] + cfg.head_vocab_sizes[-1] - 1
    row = store.raw_rows(np.array([0, last], dtype=np.int64))
    assert row.shape == (2, 160)
    store.close()


def test_hashed_rows_match_hf_shards(ple_bin):
    tok = load_tokenizer(HF_DONOR_TOKENIZER)
    store = open_store(ple_bin)
    layout = hf_shard_layout()
    texts = [
        "Hello, world.",
        "def add(a, b):\n    return a + b\n",
        "The n-gram embedding is addressed from raw token ids.",
    ]
    ids: list[int] = []
    for text in texts:
        row_ids = hash_token_ids(encode_ids(tok, text))
        ids.extend(int(x) for x in row_ids[:3].reshape(-1))
    # also pin endpoints of head 0 and last head
    cfg = expected_ple_config()
    ids.extend([0, cfg.head_vocab_sizes[0] - 1, cfg.logical_rows - 1])
    packed = store.raw_rows(np.array(ids, dtype=np.int64))
    hf = read_hf_rows(layout, ids)
    np.testing.assert_array_equal(packed, hf)
    # dequant is a scaled view of the same bytes
    dec = e4m3fn_to_f32(packed) * np.float32(store.scale)
    np.testing.assert_allclose(dec, store.rows_f32(np.array(ids, dtype=np.int64)), rtol=0, atol=0)
    store.close()
