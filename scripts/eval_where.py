#!/usr/bin/env python3
"""Stage A: head knockouts and per-token gate/error dumps on the s1_gate adapter."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.adapter import PleAdapter  # noqa: E402
from ple_graft.data import DEFAULT_DATA_DIR, batch_windows, load_tokens, n_windows  # noqa: E402
from ple_graft.donor_assets import RECIPIENT_DIR  # noqa: E402
from ple_graft.losses import token_nll_unreduced  # noqa: E402
from ple_graft.metrics import paired_delta_stats, write_json  # noqa: E402
from ple_graft.model_io import load_recipient  # noqa: E402
from ple_graft.ple_lookup import LookupMode, PleLookup  # noqa: E402
from ple_graft.ple_store import open_store  # noqa: E402
from ple_graft.qwen35_patch import PleGraft  # noqa: E402

CKPT = ROOT / "artifacts" / "checkpoints" / "adapter_real_bottleneck_r256_s1_gate" / "adapter.pt"
FULL16_NPY = ROOT / "artifacts" / "metrics" / "expanded_val" / "P1_r256_s1_gate.npy"
B0_NPY = ROOT / "artifacts" / "metrics" / "expanded_val" / "B0_baseline.npy"

HEAD_RUNS = {
    "full16": list(range(16)),
    "bigram": list(range(8)),
    "trigram": list(range(8, 16)),
    "rand8": sorted(np.random.default_rng(0).choice(16, size=8, replace=False).tolist()),
}


def _load_adapter(path: Path, device: str):
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    ck = blob.get("args") or {}
    adapter = PleAdapter(
        rank=int(ck.get("rank", 256)),
        gate_bias=float(ck.get("gate_bias", 0.0)),
        arch=ck.get("arch", "bottleneck"),
    ).to(device=device, dtype=torch.bfloat16)
    adapter.load_state_dict(blob["adapter"])
    adapter.eval()
    return adapter


def _eval_windows(model, tokens, n, seq_len, batch_size, device, graft=None, dump=None):
    import torch

    win = np.empty(n, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            ids = torch.from_numpy(batch_windows(tokens, idx, seq_len)).to(device)
            logits = model(input_ids=ids).logits
            tok_loss, _ = token_nll_unreduced(logits, ids)
            win[start : start + len(idx)] = tok_loss.float().mean(dim=-1).cpu().numpy()
            if dump is not None:
                nll = tok_loss.float().cpu().numpy()
                gates = None
                if graft is not None and graft.adapter is not None and graft.adapter.last_gate is not None:
                    g = graft.adapter.last_gate.float().cpu().numpy()
                    # last_gate is [B,S,1] on the unshifted sequence; align to labels[:-1]
                    gates = g[:, 1:, 0]
                for bi, w in enumerate(idx):
                    sl = slice(w * (seq_len - 1), (w + 1) * (seq_len - 1))
                    dump["nll"][sl] = nll[bi]
                    dump["tok"][sl] = ids[bi, 1:].detach().cpu().numpy().astype(np.int32)
                    if gates is not None:
                        dump["gate"][sl] = gates[bi]
            if (start // batch_size) % 50 == 0:
                print(f"    window {start}/{n}", flush=True)
    return win


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--windows", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "metrics" / "ple_where")
    p.add_argument("--skip-heads", action="store_true")
    p.add_argument("--skip-tokens", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch

    tokens = load_tokens(DEFAULT_DATA_DIR / "val.bin")
    n_all = n_windows(len(tokens), args.seq_len)
    n = n_all if args.windows <= 0 else min(args.windows, n_all)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"windows={n}/{n_all}", flush=True)

    model, _ = load_recipient(RECIPIENT_DIR, device=args.device)
    store = open_store()
    lookup = PleLookup(store=store, mode=LookupMode.REAL, seed=1)
    adapter = _load_adapter(CKPT, args.device)
    graft = PleGraft(model, adapter, lookup, enabled=True)

    base = np.load(B0_NPY)[:n]
    full16 = np.load(FULL16_NPY)[:n]
    report = {"windows": n, "head_keep": {k: v for k, v in HEAD_RUNS.items()}, "heads": {}, "tokens": {}}

    if not args.skip_heads:
        for name, keep in HEAD_RUNS.items():
            npy = args.out_dir / f"head_{name}.npy"
            if name == "full16" and not args.force:
                arr = full16
                print(f"{name}: reuse {FULL16_NPY} mean={arr.mean():.4f}", flush=True)
            elif npy.is_file() and not args.force and np.load(npy).shape[0] == n:
                arr = np.load(npy)
                print(f"{name}: loaded mean={arr.mean():.4f}", flush=True)
            else:
                lookup.set_head_mask(keep)
                print(f"{name}: keep={keep}", flush=True)
                t0 = time.time()
                arr = _eval_windows(model, tokens, n, args.seq_len, args.batch_size, args.device)
                np.save(npy, arr)
                print(f"{name}: mean={arr.mean():.4f} seconds={time.time()-t0:.1f}", flush=True)
            lookup.set_head_mask(None)
            report["heads"][name] = {
                "nll": float(arr.mean()),
                "vs_B0": paired_delta_stats(arr, base),
                "vs_full16": paired_delta_stats(arr, full16),
            }

    if not args.skip_tokens:
        per = n * (args.seq_len - 1)
        dump_dir = args.out_dir / "tokens"
        dump_dir.mkdir(exist_ok=True)
        paths = {k: dump_dir / f"{k}.bin" for k in ("nll_ple", "nll_b0", "gate", "tok")}
        need = args.force or not all(p.is_file() and p.stat().st_size == per * 4 for p in paths.values())
        if not need:
            print("token dumps already present", flush=True)
        else:
            nll_ple = np.memmap(paths["nll_ple"], dtype=np.float32, mode="w+", shape=(per,))
            gate = np.memmap(paths["gate"], dtype=np.float32, mode="w+", shape=(per,))
            tok = np.memmap(paths["tok"], dtype=np.int32, mode="w+", shape=(per,))
            print("dumping PLE per-token nll/gate", flush=True)
            lookup.set_head_mask(None)
            _eval_windows(
                model,
                tokens,
                n,
                args.seq_len,
                args.batch_size,
                args.device,
                graft=graft,
                dump={"nll": nll_ple, "gate": gate, "tok": tok},
            )
            nll_ple.flush()
            gate.flush()
            tok.flush()
            graft.remove()
            graft = None
            print("dumping B0 per-token nll", flush=True)
            nll_b0 = np.memmap(paths["nll_b0"], dtype=np.float32, mode="w+", shape=(per,))
            tok_b0 = np.memmap(dump_dir / "tok_b0.bin", dtype=np.int32, mode="w+", shape=(per,))
            _eval_windows(
                model,
                tokens,
                n,
                args.seq_len,
                args.batch_size,
                args.device,
                dump={"nll": nll_b0, "tok": tok_b0},
            )
            nll_b0.flush()
            tok_b0.flush()

        nll_ple = np.memmap(paths["nll_ple"], dtype=np.float32, mode="r")
        nll_b0 = np.memmap(paths["nll_b0"], dtype=np.float32, mode="r")
        gate = np.memmap(paths["gate"], dtype=np.float32, mode="r")
        tok = np.memmap(paths["tok"], dtype=np.int32, mode="r")
        # unigram counts on the whole val stream
        counts = np.bincount(np.asarray(tokens), minlength=int(tok.max()) + 1)
        uni = counts[np.clip(tok, 0, counts.size - 1)].astype(np.float64)
        delta = nll_ple.astype(np.float64) - nll_b0.astype(np.float64)

        def quintile_table(x, y_delta, y_gate):
            qs = np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            rows = []
            for i in range(5):
                lo, hi = qs[i], qs[i + 1]
                m = (x >= lo) & (x <= hi if i == 4 else x < hi)
                rows.append(
                    {
                        "q": i,
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(m.sum()),
                        "mean_delta": float(y_delta[m].mean()) if m.any() else None,
                        "mean_gate": float(y_gate[m].mean()) if m.any() else None,
                    }
                )
            return rows

        report["tokens"] = {
            "n": int(per),
            "mean_gate": float(np.mean(gate)),
            "mean_delta": float(delta.mean()),
            "by_unigram": quintile_table(uni, delta, gate),
            "by_b0_nll": quintile_table(nll_b0.astype(np.float64), delta, gate),
        }

    if graft is not None:
        graft.remove()
    write_json(args.out_dir / "report.json", report)
    print(json.dumps(report, indent=2)[:4000])
    print(f"wrote {args.out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
