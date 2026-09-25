#!/usr/bin/env python3
"""Tokenize a frozen FineWeb-Edu train/val split plus the ninfer smoke corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.data import DEFAULT_DATA_DIR, write_manifest  # noqa: E402
from ple_graft.donor_assets import PLE_EOS_TOKEN_ID, RECIPIENT_DIR, SMOKE_CORPUS  # noqa: E402
from ple_graft.tokenizer_compat import encode_ids, load_tokenizer  # noqa: E402


def _iter_fineweb():
    from datasets import load_dataset

    tries = [
        dict(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True),
        dict(path="HuggingFaceFW/fineweb-edu", split="train", streaming=True),
        dict(path="allenai/c4", name="en", split="train", streaming=True),
    ]
    last = None
    for kw in tries:
        try:
            print(f"loading {kw}", flush=True)
            ds = load_dataset(**kw)
            return ds, kw
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"failed {kw}: {exc}", flush=True)
    raise RuntimeError(f"could not load a text dataset: {last}")


def _iter_smoke_texts():
    if not SMOKE_CORPUS.is_dir():
        return
    for path in sorted(SMOKE_CORPUS.glob("data/*/*.txt")):
        yield path.read_text(encoding="utf-8", errors="replace")


def _flush(buf: list[int], fh, written: int) -> tuple[list[int], int]:
    if not buf:
        return buf, written
    arr = np.asarray(buf, dtype=np.uint32)
    fh.write(arr.tobytes())
    return [], written + arr.size


def tokenize_stream(texts, tok, out_path: Path, limit: int, eos: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    written = 0
    with out_path.open("wb") as fh:
        for text in texts:
            if not text or not str(text).strip():
                continue
            ids = encode_ids(tok, str(text))
            if not ids:
                continue
            buf.extend(ids)
            buf.append(eos)
            if len(buf) >= 1_000_000:
                buf, written = _flush(buf, fh, written)
                print(f"  {out_path.name}: {written:,} tokens", flush=True)
            if written + len(buf) >= limit:
                need = limit - written
                buf = buf[:need]
                buf, written = _flush(buf, fh, written)
                break
        buf, written = _flush(buf, fh, written)
    return written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-tokens", type=int, default=80_000_000)
    p.add_argument("--val-tokens", type=int, default=8_000_000)
    p.add_argument("--tokenizer", type=Path, default=RECIPIENT_DIR / "tokenizer.json")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    tok = load_tokenizer(args.tokenizer)
    eos = PLE_EOS_TOKEN_ID
    args.out_dir.mkdir(parents=True, exist_ok=True)

    smoke_path = args.out_dir / "smoke_val.bin"
    n_smoke = tokenize_stream(_iter_smoke_texts(), tok, smoke_path, 2_000_000, eos)
    print(f"smoke_val {n_smoke} tokens -> {smoke_path}")
    if args.smoke_only:
        write_manifest(
            args.out_dir / "manifest.json",
            {
                "tokenizer": str(args.tokenizer),
                "eos_token_id": eos,
                "dataset": "ninfer-ppl-1m",
                "files": {"smoke_val": {"path": str(smoke_path), "tokens": n_smoke}},
            },
        )
        return 0 if n_smoke > 1000 else 1

    ds, ds_kw = _iter_fineweb()
    # One pass: do not re-iterate the streaming dataset (it would restart and leak val into train).
    stream = (ex.get("text") or ex.get("content") or "" for ex in ds)

    val_path = args.out_dir / "val.bin"
    n_val = tokenize_stream(stream, tok, val_path, args.val_tokens, eos)
    print(f"val {n_val} tokens -> {val_path}")

    train_path = args.out_dir / "train.bin"
    n_train = tokenize_stream(stream, tok, train_path, args.train_tokens, eos)
    print(f"train {n_train} tokens -> {train_path}")

    man = {
        "tokenizer": str(args.tokenizer),
        "eos_token_id": eos,
        "dataset": ds_kw,
        "files": {
            "train": {"path": str(train_path), "tokens": n_train},
            "val": {"path": str(val_path), "tokens": n_val},
            "smoke_val": {"path": str(smoke_path), "tokens": n_smoke},
        },
    }
    write_manifest(args.out_dir / "manifest.json", man)
    print(json.dumps(man, indent=2))
    if n_val < args.val_tokens * 0.9 or n_train < args.train_tokens * 0.9:
        print("incomplete token dump", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
