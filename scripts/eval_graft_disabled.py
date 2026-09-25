#!/usr/bin/env python3
"""Phase 1: grafted lookup with residual hard-off must match untouched logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.adapter import PleAdapter  # noqa: E402
from ple_graft.donor_assets import PLE_CONCAT_DIM, RECIPIENT_DIR  # noqa: E402
from ple_graft.metrics import write_json  # noqa: E402
from ple_graft.model_io import load_recipient  # noqa: E402
from ple_graft.ngram_hash import hash_batch  # noqa: E402
from ple_graft.ple_lookup import LookupMode, default_lookup  # noqa: E402
from ple_graft.qwen35_patch import PleGraft, transformer_layers  # noqa: E402

SNIPPETS = [
    "Hello, world.",
    "def add(a, b):\n    return a + b\n",
    "The n-gram embedding is addressed from raw token ids.",
    "Qwen3.5-4B hidden size is 2560.",
]


def _logits(model, input_ids):
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids)
        return out.logits.float().cpu()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=RECIPIENT_DIR)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "metrics" / "phase1_disabled.json",
    )
    args = p.parse_args()

    import torch

    print(f"loading {args.model} on {args.device}")
    model, tok = load_recipient(args.model, device=args.device)
    layers = transformer_layers(model)
    layer1 = layers[1]
    print(f"layer1 type={layer1.block_type} mixer={type(layer1.linear_attn).__name__}")

    encoded = []
    for text in SNIPPETS:
        ids = tok(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )["input_ids"].to(args.device)
        if ids.size(1) < 2:
            continue
        encoded.append((text, ids))

    baseline = [(text, ids, _logits(model, ids)) for text, ids in encoded]

    lookup = default_lookup(LookupMode.REAL)
    cfg = getattr(model.config, "text_config", model.config)
    hidden = int(getattr(cfg, "hidden_size", 2560))
    adapter = PleAdapter(hidden_size=hidden, ple_dim=PLE_CONCAT_DIM, rank=256, gate_bias=0.0).to(
        device=args.device, dtype=torch.bfloat16
    )
    graft = PleGraft(model, adapter, lookup, enabled=False)

    rows = []
    max_abs = 0.0
    all_ids_ok = True
    for text, ids, base in baseline:
        got = _logits(model, ids)
        diff = (got - base).abs()
        this_max = float(diff.max())
        max_abs = max(max_abs, this_max)
        row_ids = graft.last_row_ids
        expected = hash_batch(ids.detach().cpu().numpy())
        ids_ok = row_ids is not None and (row_ids == expected).all()
        all_ids_ok = all_ids_ok and bool(ids_ok)
        stats = graft.last_stats
        rows.append(
            {
                "text": text,
                "tokens": int(ids.size(1)),
                "max_abs_logit_diff": this_max,
                "row_ids_match": bool(ids_ok),
                "unique_rows": None if stats is None else stats.n_unique,
                "bytes_read": None if stats is None else stats.bytes_read,
                "hash_ms": None if stats is None else stats.hash_ms,
                "gather_ms": None if stats is None else stats.gather_ms,
            }
        )
        print(f"{text!r:60s} max_abs={this_max:.3e} ids_ok={ids_ok}")

    # Zero-init adapter with enabled=True should still be numerically a no-op.
    graft.enabled = True
    zero_max = 0.0
    for text, ids, base in baseline:
        got = _logits(model, ids)
        this_max = float((got - base).abs().max())
        zero_max = max(zero_max, this_max)

    graft.remove()
    payload = {
        "device": args.device,
        "n_sequences": len(rows),
        "disabled_max_abs_logit_diff": max_abs,
        "zero_init_enabled_max_abs_logit_diff": zero_max,
        "row_ids_match": all_ids_ok,
        "layer_index": 1,
        "layer_type": layer1.block_type,
        "sequences": rows,
        "torch": torch.__version__,
    }
    write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in (
        "disabled_max_abs_logit_diff",
        "zero_init_enabled_max_abs_logit_diff",
        "row_ids_match",
    )}, indent=2))
    ok = all_ids_ok and max_abs == 0.0
    # zero-init add may be exact 0 in bf16; allow a tiny slop if not
    ok = ok and zero_max < 1e-5
    print("GO" if ok else "NO-GO", f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
