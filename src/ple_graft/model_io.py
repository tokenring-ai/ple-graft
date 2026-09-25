"""Load the pinned Qwen3.5-4B-Base recipient the same way every script does."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .donor_assets import RECIPIENT_DIR


def load_recipient(path: Path | None = None, device: str = "cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    path = Path(path) if path is not None else RECIPIENT_DIR
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    load_kw: dict[str, Any] = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **load_kw)
    except (ValueError, KeyError):
        model = AutoModelForImageTextToText.from_pretrained(str(path), **load_kw)
    model.to(device)
    model.eval()
    return model, tok
