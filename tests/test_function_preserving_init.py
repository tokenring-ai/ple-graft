from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ple_graft.adapter import PleAdapter


def test_zero_init_adapter_is_identity():
    adapter = PleAdapter(hidden_size=32, rank=8, gate_bias=-3.0)
    hidden = torch.randn(2, 5, 32)
    ple = torch.randn(2, 5, 32)
    out = adapter(hidden, ple, enabled=True)
    torch.testing.assert_close(out, hidden, rtol=0, atol=0)
    # enabled=False is also identity
    torch.testing.assert_close(adapter(hidden, ple, enabled=False), hidden, rtol=0, atol=0)


@pytest.mark.parametrize("rank", [256, 512, 1024])
def test_bottleneck_ranks_zero_up_is_identity(rank):
    adapter = PleAdapter(hidden_size=64, rank=rank, arch="bottleneck", gate_bias=0.0)
    hidden = torch.randn(2, 3, 64)
    ple = torch.randn(2, 3, 64)
    torch.testing.assert_close(adapter(hidden, ple, enabled=True), hidden, rtol=0, atol=0)


def test_per_head_zero_mix_is_identity():
    adapter = PleAdapter(hidden_size=32, arch="per_head", gate_bias=0.0)
    hidden = torch.randn(2, 5, 32)
    ple = torch.randn(2, 5, 32)
    torch.testing.assert_close(adapter(hidden, ple, enabled=True), hidden, rtol=0, atol=0)
    torch.nn.init.normal_(adapter.w_mix.weight, std=0.1)
    assert not torch.allclose(adapter(hidden, ple, enabled=True), hidden)


def test_bottleneck_width_mismatch_zero_up_is_identity():
    """35B-A3B residual is 2048; PLE concat stays 2560."""
    adapter = PleAdapter(hidden_size=2048, ple_dim=2560, rank=256, arch="bottleneck", gate_bias=0.0)
    hidden = torch.randn(2, 3, 2048)
    ple = torch.randn(2, 3, 2560)
    torch.testing.assert_close(adapter(hidden, ple, enabled=True), hidden, rtol=0, atol=0)
    assert adapter.w_down.weight.shape == (256, 2560)
    assert adapter.w_up.weight.shape == (2048, 256)
    torch.nn.init.normal_(adapter.w_up.weight, std=0.05)
    assert not torch.allclose(adapter(hidden, ple, enabled=True), hidden)


def test_dense_zero_init_is_identity():
    adapter = PleAdapter(hidden_size=32, arch="dense", gate_bias=0.0)
    hidden = torch.randn(2, 5, 32)
    ple = torch.randn(2, 5, 32)
    torch.testing.assert_close(adapter(hidden, ple, enabled=True), hidden, rtol=0, atol=0)
    torch.nn.init.normal_(adapter.w_full.weight, std=0.1)
    assert not torch.allclose(adapter(hidden, ple, enabled=True), hidden)


def test_nonzero_up_changes_output():
    adapter = PleAdapter(hidden_size=32, rank=8, gate_bias=0.0)
    torch.nn.init.normal_(adapter.w_up.weight, std=0.1)
    hidden = torch.randn(2, 5, 32)
    ple = torch.randn(2, 5, 32)
    out = adapter(hidden, ple, enabled=True)
    assert not torch.allclose(out, hidden)
