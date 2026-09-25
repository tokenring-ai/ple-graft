from __future__ import annotations

import numpy as np

from ple_graft.ngram_hash import hash_token_ids


def test_future_tokens_do_not_change_past_row_ids():
    prefix = [11, 22, 33, 44, 55, 66, 77]
    a = hash_token_ids(prefix + [100, 200, 300])
    b = hash_token_ids(prefix + [999, 888, 777])
    t = len(prefix)
    np.testing.assert_array_equal(a[:t], b[:t])
    assert not np.array_equal(a[t:], b[t:])


def test_hash_uses_only_current_and_two_predecessors():
    # At t=5, only ids[3:6] can matter for trigrams (current + 2 past).
    base = [1, 2, 3, 4, 5, 6, 7, 8]
    alt = [90, 91, 92, 4, 5, 6, 7, 8]
    a = hash_token_ids(base)
    b = hash_token_ids(alt)
    np.testing.assert_array_equal(a[5], b[5])
    assert not np.array_equal(a[2], b[2])
