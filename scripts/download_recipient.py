#!/usr/bin/env python3
"""Download a recipient checkpoint (default: Qwen/Qwen3.5-4B-Base)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.donor_assets import RECIPIENT_DIR, RECIPIENT_REPO_ID  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=RECIPIENT_REPO_ID)
    p.add_argument("--out", type=Path, default=RECIPIENT_DIR)
    p.add_argument("--revision", default=None)
    args = p.parse_args()
    from huggingface_hub import snapshot_download

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(repo_id=args.repo, local_dir=str(args.out))
    if args.revision:
        kw["revision"] = args.revision
    snapshot_download(**kw)
    print(f"downloaded {args.repo} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
