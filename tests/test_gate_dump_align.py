from __future__ import annotations

from pathlib import Path

import pytest

from ple_graft.data import DEFAULT_DATA_DIR, load_tokens, n_windows
from ple_graft.gate_analysis import (
    DUMP_DIR,
    N_POS_EXPECTED,
    SEQ_LEN,
    assert_aligned,
    labels_from_val,
    load_dumps,
)


def test_labels_from_val_are_window_shift_one():
    tokens = load_tokens(DEFAULT_DATA_DIR / "val.bin")
    lab = labels_from_val(tokens, 0, SEQ_LEN)
    assert lab.shape == (SEQ_LEN - 1,)
    assert lab.tolist() == tokens[1:SEQ_LEN].astype("int32").tolist()


@pytest.mark.skipif(not (DUMP_DIR / "tok.bin").is_file(), reason="gate dumps missing")
def test_real_dumps_align_with_val_bin():
    tokens = load_tokens(DEFAULT_DATA_DIR / "val.bin")
    n_win = n_windows(len(tokens), SEQ_LEN)
    dumps = load_dumps(DUMP_DIR, n_win=n_win, seq_len=SEQ_LEN)
    assert dumps["tok"].size == N_POS_EXPECTED
    stats = assert_aligned(dumps, tokens)
    assert stats["n"] == N_POS_EXPECTED
    assert 0.2 < stats["mean_gate"] < 0.4
    assert stats["mean_delta"] < 0
