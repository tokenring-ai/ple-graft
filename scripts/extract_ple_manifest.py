#!/usr/bin/env python3
"""Extract and pin PLE hash constants from the pack and HF donor tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.extract import build_manifest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "manifests" / "ple.json",
    )
    p.add_argument("--sha256", action="store_true", help="hash the 48G ple.bin (slow)")
    args = p.parse_args()
    man = build_manifest(include_sha256=args.sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    if man["problems"]:
        print("NO-GO: PLE manifest mismatches:", file=sys.stderr)
        for item in man["problems"]:
            print(f"  - {item}", file=sys.stderr)
        print(f"wrote {args.out}", file=sys.stderr)
        return 1
    print(f"GO: wrote {args.out}")
    print(
        "logical_rows={logical_rows} physical_rows={physical_rows} scale={packed_scale}".format(
            **man["pack"]["config"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
