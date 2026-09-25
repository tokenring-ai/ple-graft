#!/usr/bin/env python3
"""Print donor/recipient checkpoint facts used by Phase 0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.donor_assets import (  # noqa: E402
    HF_DONOR_CONFIG,
    HF_DONOR_DIR,
    PACK_MANIFEST,
    PLE_BIN,
    RECIPIENT_DIR,
)


def main() -> int:
    print("donor HF dir:", HF_DONOR_DIR, "exists", HF_DONOR_DIR.is_dir())
    print("pack manifest:", PACK_MANIFEST, "exists", PACK_MANIFEST.is_file())
    print("ple.bin:", PLE_BIN, "bytes", PLE_BIN.stat().st_size if PLE_BIN.is_file() else None)
    print("recipient:", RECIPIENT_DIR, "exists", RECIPIENT_DIR.is_dir())
    if HF_DONOR_CONFIG.is_file():
        cfg = json.loads(HF_DONOR_CONFIG.read_text())
        text = cfg["text_config"]
        print("donor hidden", text["hidden_size"], "vocab", text["vocab_size"])
        print("donor ple_layer_ids", text["ple_layer_ids"], "ngram", text["ngram_size"])
    rec_cfg = RECIPIENT_DIR / "config.json"
    if rec_cfg.is_file():
        cfg = json.loads(rec_cfg.read_text())
        text = cfg.get("text_config", cfg)
        print("recipient hidden", text.get("hidden_size"), "vocab", text.get("vocab_size"))
        print("recipient layers", text.get("num_hidden_layers"), "arch", cfg.get("architectures"))
    else:
        print("recipient config missing — download Qwen/Qwen3.5-4B-Base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
