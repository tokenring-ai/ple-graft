from __future__ import annotations

import numpy as np

from ple_graft.ple_store import open_store


def test_unique_gather_matches_naive(ple_bin):
    store = open_store(ple_bin)
    ids = np.array([[0, 1, 0, 7], [1, 1, 3, 0]], dtype=np.int64)
    naive = store._mmap[ids.reshape(-1)].reshape(ids.shape + (store.head_dim,))
    # rows_f32 unique path vs raw dequant of naive bytes
    from ple_graft.fp8 import e4m3fn_to_f32

    got = store.rows_f32(ids)
    want = e4m3fn_to_f32(np.ascontiguousarray(naive)) * np.float32(store.scale)
    want = np.nan_to_num(want, nan=0.0, posinf=0.0, neginf=0.0)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    store.close()
