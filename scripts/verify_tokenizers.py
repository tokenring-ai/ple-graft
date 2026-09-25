#!/usr/bin/env python3
"""Compare donor Flash-Next tokenizer with recipient Qwen3.5-4B-Base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.donor_assets import HF_DONOR_TOKENIZER, RECIPIENT_DIR, SMOKE_CORPUS  # noqa: E402
from ple_graft.tokenizer_compat import compare_tokenizers  # noqa: E402


def iter_snippets(limit: int) -> list[str]:
    texts: list[str] = []
    if SMOKE_CORPUS.is_dir():
        for path in sorted(SMOKE_CORPUS.glob("data/*/*.txt")):
            raw = path.read_text(encoding="utf-8", errors="replace")
            for para in raw.split("\n\n"):
                para = para.strip()
                if para:
                    texts.append(para)
                if len(texts) >= limit:
                    return texts
    builtins = [
        "Hello, world.",
        "def add(a, b):\n    return a + b\n",
        "Qwen3.5-4B and Qwen3.8-Flash-Next share hidden size 2560.",
        "中文维基百科测试句子。",
        "for i in range(10):\n    print(i)\n",
        "<|endoftext|>",
        "SELECT * FROM users WHERE id = 1;",
        "https://qwen.ai/blog",
        "The n-gram embedding is addressed from raw token ids.",
        "0xDEADBEEF " * 20,
    ]
    while len(texts) < limit:
        texts.extend(builtins)
    return texts[:limit]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snippets", type=int, default=10_000)
    p.add_argument(
        "--recipient",
        type=Path,
        default=None,
        help="tokenizer.json or a model dir containing it",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "manifests" / "tokenizer_compat.json",
    )
    args = p.parse_args()
    recip = args.recipient
    if recip is None:
        recip = RECIPIENT_DIR / "tokenizer.json"
    elif recip.is_dir():
        recip = recip / "tokenizer.json"
    if not recip.is_file():
        print(f"NO-GO: missing recipient tokenizer at {recip}", file=sys.stderr)
        return 2
    report = compare_tokenizers(HF_DONOR_TOKENIZER, recip, iter_snippets(args.snippets), args.snippets)
    payload = report.__dict__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not report.graftable:
        print("NO-GO: shared token IDs or snippet encodings differ", file=sys.stderr)
        return 1
    if not report.equal:
        print("GO-WITH-NOTE: shared IDs match; extra specials recorded (no remap)")
        return 0
    print("GO: tokenizer identity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
