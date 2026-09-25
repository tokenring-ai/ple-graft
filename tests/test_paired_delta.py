from __future__ import annotations

import numpy as np

from ple_graft.metrics import paired_delta_stats


def test_paired_delta_recovers_constant_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(2.3, 0.2, size=2000)
    real = base - 0.02
    stats = paired_delta_stats(real, base, n_boot=2000, seed=1)
    assert stats["n_windows"] == 2000
    np.testing.assert_allclose(stats["mean_delta"], -0.02, atol=1e-12)
    # Constant shift → bootstrap CI collapses onto the mean.
    np.testing.assert_allclose(stats["ci95"], [-0.02, -0.02], atol=1e-12)
    assert stats["frac_a_better"] == 1.0


def test_paired_delta_rejects_shape_mismatch():
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        paired_delta_stats([1.0, 2.0], [1.0])
