#!/usr/bin/env python3
"""Untouched Qwen3.5-4B-Base NLL on the smoke corpus (and optional val jsonl)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.donor_assets import RECIPIENT_DIR, SMOKE_CORPUS  # noqa: E402
from ple_graft.metrics import nll_to_ppl, write_json  # noqa: E402


def load_smoke_texts() -> list[tuple[str, str]]:
    out = []
    for path in sorted(SMOKE_CORPUS.glob("data/*/*.txt")):
        out.append((path.stem, path.read_text(encoding="utf-8", errors="replace")))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=RECIPIENT_DIR)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "metrics" / "baseline.json",
    )
    args = p.parse_args()
    if not (args.model / "config.json").is_file():
        print(f"missing recipient model at {args.model}", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    print(f"loading {args.model} on {args.device}")
    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    load_kw = dict(dtype=torch.bfloat16, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(args.model), **load_kw)
    except (ValueError, KeyError):
        model = AutoModelForImageTextToText.from_pretrained(str(args.model), **load_kw)
    model.to(args.device)
    model.eval()

    texts = load_smoke_texts()
    n_tokens = 0
    nll_sum = 0.0
    per_stream = []
    t0 = time.time()
    with torch.no_grad():
        for name, text in texts:
            enc = tok(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(args.device)
            if input_ids.size(1) < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            loss = float(out.loss.item())
            tok_count = int(input_ids.size(1) - 1)
            nll_sum += loss * tok_count
            n_tokens += tok_count
            per_stream.append({"id": name, "tokens": tok_count, "nll": loss, "ppl": math.exp(loss)})
    elapsed = time.time() - t0
    mean_nll = nll_sum / max(n_tokens, 1)
    payload = {
        "model": str(args.model),
        "device": args.device,
        "max_length": args.max_length,
        "tokens": n_tokens,
        "nll": mean_nll,
        "ppl": nll_to_ppl(mean_nll),
        "seconds": elapsed,
        "tokens_per_s": n_tokens / elapsed if elapsed else None,
        "streams": per_stream,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("nll", "ppl", "tokens", "seconds")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
