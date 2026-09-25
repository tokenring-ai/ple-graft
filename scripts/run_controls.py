#!/usr/bin/env python3
"""Phase 2B causal controls. Not used until Phase 2 adapter training exists."""

from __future__ import annotations

import sys


def main() -> int:
    print("run_controls.py: Phase 2B. Requires a trained adapter and lookup modes real/zero/random/shuffled.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
