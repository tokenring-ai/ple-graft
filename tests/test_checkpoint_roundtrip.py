from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ple_graft.adapter import PleAdapter


def test_adapter_state_dict_roundtrip(tmp_path):
    a = PleAdapter(hidden_size=64, rank=16, gate_bias=-2.5)
    torch.nn.init.normal_(a.w_down.weight, std=0.05)
    path = tmp_path / "adapter.pt"
    torch.save(a.state_dict(), path)
    b = PleAdapter(hidden_size=64, rank=16, gate_bias=0.0)
    b.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
