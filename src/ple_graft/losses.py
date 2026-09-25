"""Causal LM loss helpers."""

from __future__ import annotations

from typing import Any


def causal_nll(logits, labels, ignore_index: int = -100):
    import torch
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def token_nll_unreduced(logits, labels, ignore_index: int = -100):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return loss.view(shift_labels.shape), shift_labels
