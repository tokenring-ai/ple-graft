from __future__ import annotations

import numpy as np

from ple_graft.donor_assets import EXPECTED_LOGICAL_ROWS, expected_ple_config
from ple_graft.extract import mismatches, pack_ple_config, hf_ple_constants
from ple_graft.ngram_hash import MASK64, context_windows, hash_token_ids, hash_windows

MASK = int(MASK64)


def _py_hash(ctx: list[int], cfg) -> list[int]:
    rows = []
    for n in range(2, cfg.ngram_size + 1):
        mixed = (int(ctx[0]) * int(cfg.layer_multipliers[0])) & MASK
        for j in range(1, n):
            mixed ^= (int(ctx[j]) * int(cfg.layer_multipliers[j])) & MASK
            mixed &= MASK
        base = (n - 2) * cfg.heads_per_ngram
        for g in range(cfg.heads_per_ngram):
            h = base + g
            rows.append(mixed % int(cfg.head_vocab_sizes[h]) + int(cfg.head_offsets[h]))
    return rows


def test_pack_matches_hf_and_pinned_expected():
    pack = pack_ple_config()
    hf = hf_ple_constants()
    assert mismatches(pack, hf) == []
    assert pack.logical_rows == EXPECTED_LOGICAL_ROWS
    assert pack.physical_rows >= pack.logical_rows


def test_numpy_hash_matches_python_uint64():
    cfg = expected_ple_config()
    ids = [12, 99, 248044, 7, 8, 9, 100000]
    windows = context_windows(ids, cfg.ngram_size, cfg.eos_token_id)
    got = hash_windows(
        windows,
        cfg.layer_multipliers,
        cfg.head_vocab_sizes,
        cfg.head_offsets,
        cfg.heads_per_ngram,
    )
    for t, ctx in enumerate(windows.tolist()):
        assert got[t].tolist() == _py_hash(ctx, cfg)


def test_row_ids_in_head_ranges():
    cfg = expected_ple_config()
    ids = list(range(50, 80)) + [cfg.eos_token_id] + list(range(1000, 1020))
    rows = hash_token_ids(ids, cfg)
    assert rows.shape == (len(ids), 16)
    for h in range(16):
        lo = cfg.head_offsets[h]
        hi = lo + cfg.head_vocab_sizes[h]
        assert rows[:, h].min() >= lo
        assert rows[:, h].max() < hi
    assert rows.max() < cfg.logical_rows


def test_position_zero_pads_history_with_eos():
    cfg = expected_ple_config()
    windows = context_windows([123], cfg.ngram_size, cfg.eos_token_id)
    assert windows[0].tolist() == [123, cfg.eos_token_id, cfg.eos_token_id]


def test_eos_cuts_further_history_but_keeps_tokens_after():
    cfg = expected_ple_config()
    eos = cfg.eos_token_id
    ids = [10, eos, 20, 30]
    windows = context_windows(ids, cfg.ngram_size, eos)
    assert windows[2].tolist() == [20, eos, eos]
    assert windows[3].tolist() == [30, 20, eos]
