from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from ple_graft.adapter import PleAdapter
from ple_graft.ple_lookup import LookupMode, PleLookup
from ple_graft.qwen35_patch import PleGraft


class _FakeGDN(nn.Module):
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return hidden_states * 0  # mixer contribution zero → residual add is identity


class _FakeMLP(nn.Module):
    def forward(self, x):
        return x * 0


class _FakeNorm(nn.Module):
    def forward(self, x):
        return x


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_type = "linear_attention"
        self.linear_attn = _FakeGDN()
        self.mlp = _FakeMLP()
        self.input_layernorm = _FakeNorm()
        self.post_attention_layernorm = _FakeNorm()

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _FakeAttn(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        position_embeddings=None,
        **kwargs,
    ):
        return hidden_states * 0, None


class _FakeFullLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_type = "full_attention"
        self.self_attn = _FakeAttn()
        self.mlp = _FakeMLP()
        self.input_layernorm = _FakeNorm()
        self.post_attention_layernorm = _FakeNorm()

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _FakeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer(), _FakeLayer()])

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        x = inputs_embeds if inputs_embeds is not None else input_ids.float().unsqueeze(-1).expand(-1, -1, 8)
        for layer in self.layers:
            x = layer(x, position_embeddings=None)
        return SimpleNamespace(last_hidden_state=x, logits=x)


class _ToyLookup:
    def lookup(self, token_ids):
        from ple_graft.ple_lookup import LookupResult, LookupStats
        import numpy as np

        if token_ids.ndim == 1:
            b, s = 1, int(token_ids.shape[0])
            row_ids = np.zeros((s, 16), dtype=np.int64)
            concat = np.zeros((s, 8), dtype=np.float32)
        else:
            b, s = int(token_ids.shape[0]), int(token_ids.shape[1])
            row_ids = np.zeros((b, s, 16), dtype=np.int64)
            concat = np.zeros((b, s, 8), dtype=np.float32)
        return LookupResult(
            row_ids=row_ids,
            rows=np.zeros(row_ids.shape + (160,), dtype=np.float32),
            concat=concat,
            stats=LookupStats(),
        )


def test_disabled_graft_is_identity_on_toy_stack():
    model = _FakeText()
    hidden = torch.randn(2, 4, 8)
    before = model(inputs_embeds=hidden).last_hidden_state.detach().clone()

    adapter = PleAdapter(hidden_size=8, rank=4)
    graft = PleGraft(model, adapter, _ToyLookup(), layer_index=1, enabled=False)
    ids = torch.randint(0, 50, (2, 4))
    after = model(input_ids=ids, inputs_embeds=hidden).last_hidden_state
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert graft.last_row_ids is not None
    graft.remove()
    restored = model(inputs_embeds=hidden).last_hidden_state
    torch.testing.assert_close(restored, before, rtol=0, atol=0)


def test_eval_batch1_then_train_batch2_looks_up_again():
    """Eval windows are B=1; the next train step is B=2. Stale concat must not be reused."""
    model = _FakeText()
    adapter = PleAdapter(hidden_size=8, rank=4)
    torch.nn.init.normal_(adapter.w_up.weight, std=0.05)
    graft = PleGraft(model, adapter, _ToyLookup(), layer_index=1, enabled=True)
    h1 = torch.randn(1, 4, 8)
    ids1 = torch.randint(0, 50, (1, 4))
    out1 = model(input_ids=ids1, inputs_embeds=h1)
    assert out1.last_hidden_state.shape == (1, 4, 8)
    assert graft.last_row_ids.shape[0] == 1
    h2 = torch.randn(2, 4, 8)
    ids2 = torch.randint(0, 50, (2, 4))
    out2 = model(input_ids=ids2, inputs_embeds=h2)
    assert out2.last_hidden_state.shape == (2, 4, 8)
    assert graft.last_row_ids.shape == (2, 4, 16)
    graft.remove()


def test_zero_init_graft_is_identity_at_full_attention_layer():
    """Layer 3 is full_attention in Qwen3.5-4B; zero-init must still match the ungrafted forward."""
    model = _FakeText()
    model.layers = nn.ModuleList([_FakeLayer(), _FakeLayer(), _FakeLayer(), _FakeFullLayer()])
    hidden = torch.randn(2, 4, 8)
    before = model(inputs_embeds=hidden).last_hidden_state.detach().clone()
    adapter = PleAdapter(hidden_size=8, rank=4)
    graft = PleGraft(model, adapter, _ToyLookup(), layer_index=3, enabled=True)
    assert model.layers[3].block_type == "full_attention"
    ids = torch.randint(0, 50, (2, 4))
    after = model(input_ids=ids, inputs_embeds=hidden).last_hidden_state
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    graft.remove()


def test_zero_init_graft_is_identity_at_layer_0():
    model = _FakeText()
    hidden = torch.randn(2, 4, 8)
    before = model(inputs_embeds=hidden).last_hidden_state.detach().clone()
    adapter = PleAdapter(hidden_size=8, rank=4)
    graft = PleGraft(model, adapter, _ToyLookup(), layer_index=0, enabled=True)
    ids = torch.randint(0, 50, (2, 4))
    after = model(input_ids=ids, inputs_embeds=hidden).last_hidden_state
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    graft.remove()


def test_inject_rejects_stale_concat_shape():
    model = _FakeText()
    adapter = PleAdapter(hidden_size=8, rank=4)
    graft = PleGraft(model, adapter, _ToyLookup(), layer_index=1, enabled=True)
    graft._ple_concat = torch.zeros(1, 4, 8)
    graft._ple_token_shape = (1, 4)
    with pytest.raises(RuntimeError, match="does not match"):
        graft._inject(torch.randn(2, 4, 8))
    graft.remove()
