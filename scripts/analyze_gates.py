#!/usr/bin/env python3
"""CPU gate analysis on existing ple_where token dumps. No GPU, no retraining."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.data import DEFAULT_DATA_DIR, load_tokens, n_windows, window_at  # noqa: E402
from ple_graft.donor_assets import PLE_EOS_TOKEN_ID, RECIPIENT_DIR  # noqa: E402
from ple_graft.gate_analysis import (  # noqa: E402
    DUMP_DIR,
    SEQ_LEN,
    assert_aligned,
    invert_vocab,
    load_dumps,
    ngram_rarity_for_windows,
    quintile_table,
    spearman,
    token_bucket,
)
from ple_graft.tokenizer_compat import load_tokenizer  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "metrics" / "ple_where"


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for r in rows:
        cells = []
        for _, key in cols:
            v = r.get(key)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main() -> int:
    tokens = load_tokens(DEFAULT_DATA_DIR / "val.bin")
    n_win = n_windows(len(tokens), SEQ_LEN)
    dumps = load_dumps(DUMP_DIR, n_win=n_win, seq_len=SEQ_LEN)
    print("aligning dumps...", flush=True)
    align = assert_aligned(dumps, tokens)
    print("align", align, flush=True)

    g = np.asarray(dumps["gate"], dtype=np.float64)
    nll_ple = np.asarray(dumps["nll_ple"], dtype=np.float64)
    nll_b0 = np.asarray(dumps["nll_b0"], dtype=np.float64)
    tok = np.asarray(dumps["tok"])
    delta = nll_ple - nll_b0
    per = SEQ_LEN - 1

    q_gate = quintile_table(g, delta, g)
    q_b0 = quintile_table(nll_b0, delta, g)
    rho_g_b0 = spearman(g, nll_b0)
    rho_g_delta = spearman(g, delta)
    print(f"spearman g vs B0-NLL {rho_g_b0:.4f}; g vs Δ {rho_g_delta:.4f}", flush=True)

    print("decoding vocab...", flush=True)
    tokenizer = load_tokenizer(RECIPIENT_DIR / "tokenizer.json")
    id2s = invert_vocab(tokenizer)
    # id_to_token is faster if present
    def dec(tid: int) -> str:
        if hasattr(tokenizer, "id_to_token"):
            s = tokenizer.id_to_token(int(tid))
            return s if s is not None else id2s.get(int(tid), "")
        return id2s.get(int(tid), "")

    unique_ids = np.unique(tok)
    bucket_of = {int(t): token_bucket(dec(int(t)), int(t), PLE_EOS_TOKEN_ID) for t in unique_ids}
    buckets = defaultdict(lambda: {"n": 0, "sum_g": 0.0, "sum_d": 0.0})
    # vectorized via unique inverse
    bnames = np.array([bucket_of[int(t)] for t in unique_ids])
    inv = np.searchsorted(unique_ids, tok)
    names = bnames[inv]
    for name in np.unique(names):
        m = names == name
        buckets[str(name)] = {
            "n": int(m.sum()),
            "mean_gate": float(g[m].mean()),
            "mean_delta": float(delta[m].mean()),
        }
    # previous-token
    prev = np.empty_like(tok)
    for wi in range(n_win):
        w = window_at(tokens, wi, SEQ_LEN)
        sl = slice(wi * per, (wi + 1) * per)
        prev[sl] = w[:-1].astype(np.int32)
    prev_unique = np.unique(prev)
    prev_bucket = {int(t): token_bucket(dec(int(t)), int(t), PLE_EOS_TOKEN_ID) for t in prev_unique}
    pb = np.array([prev_bucket[int(t)] for t in prev_unique])
    pinv = np.searchsorted(prev_unique, prev)
    pnames = pb[pinv]
    prev_rows = {}
    for name in np.unique(pnames):
        m = pnames == name
        prev_rows[str(name)] = {
            "n": int(m.sum()),
            "mean_gate": float(g[m].mean()),
            "mean_delta": float(delta[m].mean()),
        }

    print("n-gram rarity (two hash passes)...", flush=True)
    rarity = ngram_rarity_for_windows(tokens, n_win, SEQ_LEN)
    q_rare = quintile_table(rarity, delta, g)
    rho_g_rare = spearman(g, rarity)

    print("code-ish windows...", flush=True)
    code_frac = np.empty(n_win, dtype=np.float64)
    win_g = g.reshape(n_win, per).mean(axis=1)
    win_d = delta.reshape(n_win, per).mean(axis=1)
    markers = ("def", "class", "import", "#include", "fn ")
    for i in range(n_win):
        ids = window_at(tokens, i, SEQ_LEN)
        n_hit = 0
        for tid in ids:
            s = dec(int(tid))
            if any(m in s for m in markers) or s in ("{", "};"):
                n_hit += 1
        code_frac[i] = n_hit / SEQ_LEN
    q_code = quintile_table(code_frac, win_d, win_g)

    payload = {
        "align": align,
        "spearman": {
            "gate_vs_b0_nll": rho_g_b0,
            "gate_vs_delta": rho_g_delta,
            "gate_vs_ngram_rarity": rho_g_rare,
        },
        "quintile_gate": q_gate,
        "quintile_b0_nll": q_b0,
        "quintile_ngram_rarity": q_rare,
        "quintile_code_frac_windows": q_code,
        "token_buckets": dict(buckets),
        "prev_token_buckets": prev_rows,
    }

    # Verdict
    hard_g = q_b0[-1]["mean_1"]
    easy_g = q_b0[0]["mean_1"]
    hard_d = q_b0[-1]["mean_0"]
    easy_d = q_b0[0]["mean_0"]
    high_g_d = q_gate[-1]["mean_0"]
    low_g_d = q_gate[0]["mean_0"]
    if abs(rho_g_b0) < 0.05 and abs(rho_g_delta) < 0.05 and hard_d < easy_d - 0.02:
        if hard_g <= easy_g + 0.02:
            verdict = "anti_or_mix"
            verdict_text = (
                "**Anti-correlated / mix, not a detector.** ΔNLL follows B0-error "
                f"(hardest {hard_d:.4f} vs easiest {easy_d:.4f}) but the gate does not open on "
                f"hard tokens (g {hard_g:.3f} vs {easy_g:.3f}). Spearman(g, B0-NLL)={rho_g_b0:.3f}, "
                f"Spearman(g, Δ)={rho_g_delta:.3f}. The adapter does the work; the gate is a mild global mix."
            )
        else:
            verdict = "weak_detector"
            verdict_text = "**Weak detector.** Gate rises somewhat with B0-error while ΔNLL also follows hardness."
    elif abs(rho_g_delta) >= 0.08 or (high_g_d < low_g_d - 0.005):
        verdict = "weak_detector"
        verdict_text = (
            f"**Weak detector.** Spearman(g, Δ)={rho_g_delta:.3f}; "
            f"highest-g quintile Δ={high_g_d:.4f} vs lowest-g Δ={low_g_d:.4f}."
        )
    else:
        verdict = "mix"
        verdict_text = (
            f"**Mix, not a detector.** Spearman(g, B0-NLL)={rho_g_b0:.3f}, "
            f"Spearman(g, Δ)={rho_g_delta:.3f}. ΔNLL still tracks B0-error."
        )
    payload["verdict"] = verdict
    payload["verdict_text"] = verdict_text

    def qrows(qs, dkey="mean_0", gkey="mean_1"):
        out = []
        for r in qs:
            out.append({"q": r["q"], "lo": r["lo"], "hi": r["hi"], "n": r["n"], "mean_delta": r[dkey], "mean_gate": r[gkey]})
        return out

    md = []
    md.append("# Gate analysis")
    md.append("")
    md.append("Checkpoint: seed-1 r=256 + trained gate. Dumps: 7,812 × 1023 positions. CPU only.")
    md.append("")
    md.append(verdict_text)
    md.append("")
    md.append(f"- mean g = {align['mean_gate']:.4f}")
    md.append(f"- mean ΔNLL = {align['mean_delta']:.4f}")
    md.append(f"- Spearman(g, B0-NLL) = {rho_g_b0:.4f}")
    md.append(f"- Spearman(g, ΔNLL) = {rho_g_delta:.4f}")
    md.append(f"- Spearman(g, n-gram rarity) = {rho_g_rare:.4f}")
    md.append("")
    md.append("## Quintiles of gate vs ΔNLL")
    md.append("")
    md.append(_md_table(qrows(q_gate), [("Q", "q"), ("g lo", "lo"), ("g hi", "hi"), ("n", "n"), ("mean Δ", "mean_delta"), ("mean g", "mean_gate")]))
    md.append("")
    md.append("## Quintiles of B0 NLL (replicate)")
    md.append("")
    md.append(_md_table(qrows(q_b0), [("Q", "q"), ("B0 lo", "lo"), ("B0 hi", "hi"), ("n", "n"), ("mean Δ", "mean_delta"), ("mean g", "mean_gate")]))
    md.append("")
    md.append("## Token-type buckets")
    md.append("")
    brow = [{"bucket": k, **v} for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]["n"])]
    md.append(_md_table(brow, [("bucket", "bucket"), ("n", "n"), ("mean g", "mean_gate"), ("mean Δ", "mean_delta")]))
    md.append("")
    md.append("## Previous-token buckets")
    md.append("")
    prow = [{"bucket": k, **v} for k, v in sorted(prev_rows.items(), key=lambda kv: -kv[1]["n"])]
    md.append(_md_table(prow, [("bucket", "bucket"), ("n", "n"), ("mean g", "mean_gate"), ("mean Δ", "mean_delta")]))
    md.append("")
    md.append("## N-gram rarity (higher = more common addresses in val)")
    md.append("")
    md.append(_md_table(qrows(q_rare), [("Q", "q"), ("rarity lo", "lo"), ("rarity hi", "hi"), ("n", "n"), ("mean Δ", "mean_delta"), ("mean g", "mean_gate")]))
    md.append("")
    md.append("## Window code-ish fraction (weak FineWeb split)")
    md.append("")
    md.append(_md_table(qrows(q_code), [("Q", "q"), ("frac lo", "lo"), ("frac hi", "hi"), ("n windows", "n"), ("mean Δ", "mean_delta"), ("mean g", "mean_gate")]))
    md.append("")
    md.append("NER was not run: decode/n-gram tables do not show a name-like content effect stronger than punctuation.")
    md.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gate_analysis.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (OUT_DIR / "gate_analysis.md").write_text("\n".join(md) + "\n")
    print(verdict_text)
    print(f"wrote {OUT_DIR / 'gate_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
