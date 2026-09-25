from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from ple_graft.lora import LoRALinear, apply_lora, lora_state_dict, load_lora_state_dict


def test_zero_B_is_identity():
    linear = nn.Linear(8, 16, bias=True)
    x = torch.randn(3, 5, 8)
    base = linear(x).detach()
    wrapped = LoRALinear(linear, rank=4, alpha=4.0)
    torch.testing.assert_close(wrapped(x), base, rtol=0, atol=0)


def test_nonzero_B_changes_output():
    linear = nn.Linear(8, 16, bias=False)
    x = torch.randn(2, 8)
    wrapped = LoRALinear(linear, rank=4, alpha=4.0)
    before = wrapped(x).detach()
    nn.init.normal_(wrapped.lora_B, std=0.1)
    assert not torch.allclose(wrapped(x), before)


def test_apply_lora_on_first_layers_and_roundtrip():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.w(x)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    m = M()
    x = torch.randn(2, 8)
    before = m(x).detach()
    n = apply_lora(m.layers[0], rank=2, alpha=2.0) + apply_lora(m.layers[1], rank=2, alpha=2.0)
    assert n == 2
    torch.testing.assert_close(m(x), before, rtol=0, atol=0)
    sd = lora_state_dict(m)
    assert any("lora_A" in k for k in sd)
    nn.init.normal_(m.layers[0].w.lora_B, std=0.2)
    changed = m(x).detach()
    assert not torch.allclose(changed, before)
    with torch.no_grad():
        m.layers[0].w.lora_B.zero_()
    load_lora_state_dict(m, sd)
    torch.testing.assert_close(m(x), before, rtol=0, atol=0)
