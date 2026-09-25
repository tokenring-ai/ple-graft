#!/usr/bin/env python3
"""Tabulate baseline / real-PLE / shuffled-PLE val NLL from train logs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_RE = re.compile(
    r"eval (\S+) step=(\d+) seen=([0-9,]+) val_nll=([0-9.]+) ppl=([0-9.]+)"
)


def parse_evals(log: Path) -> list[dict]:
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "tag": m.group(1),
                "step": int(m.group(2)),
                "seen": int(m.group(3).replace(",", "")),
                "val_nll": float(m.group(4)),
                "ppl": float(m.group(5)),
            }
        )
    return rows


def best_after_train(rows: list[dict]) -> dict | None:
    trained = [r for r in rows if r["seen"] > 0]
    if not trained:
        return None
    return min(trained, key=lambda r: r["val_nll"])


def main() -> int:
    real_log = ROOT / "artifacts" / "metrics" / "train_real_s0_open_gate_8M.log"
    shuf_log = ROOT / "artifacts" / "metrics" / "train_shuffled_s0.log"
    if not real_log.is_file() or not shuf_log.is_file():
        print("missing logs", file=sys.stderr)
        return 2
    real = parse_evals(real_log)
    shuf = parse_evals(shuf_log)
    base = real[0]["val_nll"] if real else None
    shuf_base = shuf[0]["val_nll"] if shuf else None
    real_best = best_after_train(real)
    shuf_best = best_after_train(shuf)
    if base is None or real_best is None or shuf_best is None:
        print("incomplete evals", json.dumps({"real": real, "shuffled": shuf}, indent=2))
        return 1
    table = {
        "shared_before_train_val_nll": base,
        "shuffled_before_train_val_nll": shuf_base,
        "baseline": {"val_nll": base, "delta": 0.0},
        "real_ple_best": {**real_best, "delta": real_best["val_nll"] - base},
        "shuffled_ple_best": {**shuf_best, "delta": shuf_best["val_nll"] - base},
        "real_evals": real,
        "shuffled_evals": shuf,
    }
    # 32-window gap of ~0.016 is within noise; require a clearly larger real advantage.
    gap = shuf_best["val_nll"] - real_best["val_nll"]
    table["shuffled_minus_real"] = gap
    table["real_clearly_better"] = bool(gap > 0.05)
    table["verdict"] = (
        "continue" if table["real_clearly_better"] else "dead_end_real_not_clearly_better_than_shuffled"
    )
    out = ROOT / "artifacts" / "metrics" / "phase2_run_of_record.json"
    out.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
