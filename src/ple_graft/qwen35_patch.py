"""Hook PLE lookup + adapter into Qwen3.5 at 0-based layer 1.

Injection is after the token-mixer residual add of that layer (before FFN),
matching donor `ple_layer_ids=[2]` (1-based → layer index 1). Phase 1 keeps
`enabled=False` so hidden states — and therefore logits — are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .donor_assets import PLE_LAYER_INDEX_0BASED

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def text_model(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model
    raise AttributeError("cannot find Qwen3.5 text backbone on model")


def transformer_layers(model: Any):
    text = text_model(model)
    if hasattr(text, "layers"):
        return text.layers
    if hasattr(text, "model") and hasattr(text.model, "layers"):
        return text.model.layers
    raise AttributeError("cannot find transformer layers")


class PleGraft:
    """Installs a lookup pre-hook and a layer-1 mixer-residual injection.

    Does not wrap the root module, so `model.generate` / HuggingFace APIs keep working.
    """

    def __init__(
        self,
        model: Any,
        adapter: Any,
        lookup: Any,
        layer_index: int = PLE_LAYER_INDEX_0BASED,
        enabled: bool = False,
    ) -> None:
        if nn is None:
            raise ImportError("PleGraft requires torch")
        self.model = model
        self.adapter = adapter
        self.lookup = lookup
        self.layer_index = int(layer_index)
        self.enabled = bool(enabled)
        self.last_row_ids = None
        self.last_stats = None
        self._ple_concat = None
        self._hooks: list[Any] = []
        self._layer = None
        self._orig_forward = None
        self._ple_token_shape = None
        self._install()

    def _install(self) -> None:
        layers = transformer_layers(self.model)
        if self.layer_index < 0 or self.layer_index >= len(layers):
            raise IndexError(f"PLE layer {self.layer_index} out of range ({len(layers)} layers)")
        self._layer = layers[self.layer_index]
        block = getattr(self._layer, "block_type", "linear_attention")
        if block not in ("linear_attention", "full_attention"):
            raise ValueError(f"layer {self.layer_index} has unsupported block_type {block}")
        self._orig_forward = self._layer.forward
        layer = self._layer
        graft = self

        def patched_forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if getattr(layer, "block_type", "linear_attention") == "full_attention":
                hidden_states, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            else:
                hidden_states = layer.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            hidden_states = residual + hidden_states
            hidden_states = graft._inject(hidden_states)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = residual + hidden_states
            return hidden_states

        self._layer.forward = patched_forward

        # One pre-hook on the module the caller invokes. Do not cache by
        # id(input_ids): CPython reuses PyObject ids, so an eval batch-1
        # tensor can collide with the next train batch-2 and skip lookup.
        self._hooks.append(self.model.register_forward_pre_hook(self._capture_ids, with_kwargs=True))

        if self.adapter is not None and hasattr(self.model, "add_module"):
            # Keep adapter on the same device as the backbone without wrapping the model.
            self.model.add_module("ple_adapter", self.adapter)

    def _capture_ids(self, _mod, args, kwargs):
        input_ids = None
        if kwargs:
            input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            cand = args[0]
            if torch.is_tensor(cand) and cand.dtype in (torch.int32, torch.int64) and cand.ndim == 2:
                input_ids = cand
        if input_ids is not None:
            self.prepare_lookup(input_ids)
        return None

    def prepare_lookup(self, input_ids) -> None:
        result = self.lookup.lookup(input_ids)
        self.last_row_ids = result.row_ids
        self.last_stats = result.stats
        if hasattr(input_ids, "shape"):
            self._ple_token_shape = tuple(int(x) for x in input_ids.shape)
        else:
            self._ple_token_shape = None
        if not self.enabled:
            self._ple_concat = None
            return
        concat = torch.from_numpy(np.ascontiguousarray(result.concat))
        device = input_ids.device if hasattr(input_ids, "device") else "cpu"
        self._ple_concat = concat.to(device=device, dtype=torch.float32)

    def _inject(self, hidden):
        if self._ple_concat is None or self.adapter is None:
            return hidden
        if not self.enabled:
            return hidden
        ple_vec = self._ple_concat
        token_shape = tuple(int(x) for x in hidden.shape[:-1])
        if self._ple_token_shape is not None and self._ple_token_shape != token_shape:
            raise RuntimeError(
                f"PLE lookup shape {self._ple_token_shape} does not match hidden {token_shape}"
            )
        n_tok = 1
        for d in token_shape:
            n_tok *= d
        if ple_vec.numel() != n_tok * ple_vec.shape[-1]:
            raise RuntimeError(
                f"PLE concat has {ple_vec.numel()} values, need {n_tok * ple_vec.shape[-1]} for hidden {tuple(hidden.shape)}"
            )
        ple_vec = ple_vec.reshape(*token_shape, ple_vec.shape[-1])
        out = self.adapter(
            hidden.to(dtype=next(self.adapter.parameters()).dtype),
            ple_vec.to(device=hidden.device, dtype=next(self.adapter.parameters()).dtype),
            enabled=True,
        )
        return out.to(dtype=hidden.dtype)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if self._layer is not None and self._orig_forward is not None:
            self._layer.forward = self._orig_forward
            self._orig_forward = None
        if hasattr(self.model, "ple_adapter"):
            try:
                del self.model.ple_adapter
            except AttributeError:
                pass

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad_(True)

    def trainable_parameter_names(self) -> list[str]:
        names = []
        if self.adapter is not None:
            names.extend(n for n, p in self.adapter.named_parameters() if p.requires_grad)
        return names
