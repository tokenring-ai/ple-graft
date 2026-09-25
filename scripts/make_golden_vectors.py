#!/usr/bin/env python3
"""Write golden n-gram row ids and a small set of PLE row values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.donor_assets import HF_DONOR_TOKENIZER  # noqa: E402
from ple_graft.extract import hf_shard_layout, packed_scale, read_hf_rows  # noqa: E402
from ple_graft.ngram_hash import hash_token_ids  # noqa: E402
from ple_graft.ple_store import open_store  # noqa: E402
from ple_graft.tokenizer_compat import encode_ids, load_tokenizer  # noqa: E402

SEQUENCES = [
    "Hello, world.",
    "def add(a, b):\n    return a + b\n",
    "The n-gram embedding is addressed from raw token ids.",
    "中文测试。",
    "a",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "golden" / "hash_rows.json")
    args = p.parse_args()
    tok = load_tokenizer(HF_DONOR_TOKENIZER)
    store = open_store()
    layout = hf_shard_layout()
    scale = packed_scale()
    records = []
    for text in SEQUENCES:
        ids = encode_ids(tok, text)
        row_ids = hash_token_ids(ids)
        # keep first 4 positions fully, plus a few raw rows
        pos = min(4, row_ids.shape[0])
        sample_ids = row_ids[:pos].reshape(-1).tolist()
        packed = store.raw_rows(np.array(sample_ids, dtype=np.int64))
        hf = read_hf_rows(layout, sample_ids)
        records.append(
            {
                "text": text,
                "token_ids": ids,
                "row_ids": row_ids.tolist(),
                "sample_row_ids": sample_ids,
                "packed_fp8_hex": [bytes(r).hex() for r in packed],
                "hf_fp8_hex": [bytes(r).hex() for r in hf],
                "packed_matches_hf": bool(np.array_equal(packed, hf)),
            }
        )
    payload = {
        "scale": scale,
        "n_sequences": len(records),
        "all_rows_match": all(r["packed_matches_hf"] for r in records),
        "sequences": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} all_rows_match={payload['all_rows_match']}")
    store.close()
    return 0 if payload["all_rows_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
