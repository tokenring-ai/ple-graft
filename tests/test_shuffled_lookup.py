from __future__ import annotations

import numpy as np

from ple_graft.ple_lookup import LookupMode, PleLookup


def test_shuffled_row_ids_are_not_the_real_addresses():
    ids = np.arange(32, dtype=np.int64)
    real = PleLookup(store=None, mode=LookupMode.REAL)
    shuf = PleLookup(store=None, mode=LookupMode.SHUFFLED, seed=0)
    a = real.row_ids(ids)
    b = shuf.row_ids(ids)
    assert a.shape == b.shape == (32, 16)
    assert not np.array_equal(a, b)
    # Same permutation is deterministic; a different seed is a different map.
    shuf2 = PleLookup(store=None, mode=LookupMode.SHUFFLED, seed=1)
    c = shuf2.row_ids(ids)
    assert not np.array_equal(b, c)
    np.testing.assert_array_equal(shuf.row_ids(ids), b)


def test_head_mask_zeros_masked_slices():
    from ple_graft.donor_assets import HEAD_DIM, N_HEADS
    from ple_graft.ple_store import open_store

    store = open_store()
    lookup = PleLookup(store=store, mode=LookupMode.REAL)
    ids = np.arange(8, dtype=np.int64)
    full = lookup.lookup(ids).concat
    lookup.set_head_mask(range(8))  # bigram only
    bi = lookup.lookup(ids).concat
    assert bi.shape == full.shape
    hd = HEAD_DIM
    np.testing.assert_array_equal(bi[..., : 8 * hd], full[..., : 8 * hd])
    np.testing.assert_array_equal(bi[..., 8 * hd :], np.zeros_like(bi[..., 8 * hd :]))
    lookup.set_head_mask(None)
    again = lookup.lookup(ids).concat
    np.testing.assert_array_equal(again, full)
    store.close()


def test_random_row_ids_are_batch_shape_invariant():
    """RANDOM must not depend on tensor rank/batch the way the old per-call RNG did."""
    seq = np.arange(32, dtype=np.int64)
    b1 = PleLookup(store=None, mode=LookupMode.RANDOM, seed=0)
    b4 = PleLookup(store=None, mode=LookupMode.RANDOM, seed=0)
    one = b1.row_ids(seq)
    four = b4.row_ids(np.stack([seq, seq + 100, seq + 200, seq + 300], axis=0))
    np.testing.assert_array_equal(four[0], one)
    # Different from real and from shuffled (different perm stream).
    real = PleLookup(store=None, mode=LookupMode.REAL).row_ids(seq)
    shuf = PleLookup(store=None, mode=LookupMode.SHUFFLED, seed=0).row_ids(seq)
    assert not np.array_equal(one, real)
    assert not np.array_equal(one, shuf)
