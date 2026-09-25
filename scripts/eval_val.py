#!/usr/bin/env python3
"""Expanded frozen-val NLL for baseline and saved adapters, with paired CIs."""

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
from ple_graft.lora import apply_lora, load_lora_state_dict  # noqa: E402
from ple_graft.losses import token_nll_unreduced  # noqa: E402
from ple_graft.metrics import nll_to_ppl, paired_delta_stats, write_json  # noqa: E402
from ple_graft.model_io import load_recipient  # noqa: E402
from ple_graft.ple_lookup import LookupMode, PleLookup  # noqa: E402
from ple_graft.ple_store import open_store  # noqa: E402
from ple_graft.qwen35_patch import PleGraft, transformer_layers  # noqa: E402


def _model_hidden(model) -> int:
    cfg = getattr(model.config, "text_config", model.config)
    return int(getattr(cfg, "hidden_size", 2560))

CONDITIONS = [
    {"name": "B0_baseline", "ckpt": None, "mode": None, "seed": 0},
    {"name": "P1_real_s0", "ckpt": "adapter_real_s0", "mode": "real", "seed": 0},
    {"name": "P1_real_s1", "ckpt": "adapter_real_s1", "mode": "real", "seed": 1},
    {"name": "C1_random_s0", "ckpt": "adapter_random_s0", "mode": "random", "seed": 0},
    {"name": "C2_shuffled_s0", "ckpt": "adapter_shuffled_s0", "mode": "shuffled", "seed": 0},
    {"name": "P1_r512_s0", "ckpt": "adapter_real_bottleneck_r512_s0", "mode": "real", "seed": 0},
    {"name": "P1_r1024_s0", "ckpt": "adapter_real_bottleneck_r1024_s0", "mode": "real", "seed": 0},
    {"name": "P1_dense_s0", "ckpt": "adapter_real_dense_s0", "mode": "real", "seed": 0},
    {"name": "P1_r256_s1_gate", "ckpt": "adapter_real_bottleneck_r256_s1_gate", "mode": "real", "seed": 1},
    {"name": "P1_L0_s0", "ckpt": "adapter_real_bottleneck_r256_s0_L0", "mode": "real", "seed": 0},
    {"name": "P1_L3_s0", "ckpt": "adapter_real_bottleneck_r256_s0_L3", "mode": "real", "seed": 0},
    {"name": "lora_only_s0", "ckpt": "adapter_lora_r16_n8_s0", "mode": None, "seed": 0},
    {"name": "ple_lora_L3_s0", "ckpt": "adapter_real_bottleneck_r256_s0_L3_lora_r16", "mode": "real", "seed": 0},
    {"name": "P1_35b_s0", "ckpt": "adapter_35b_real_bottleneck_r256_s0", "mode": "real", "seed": 0},
    {"name": "C2_35b_s0", "ckpt": "adapter_35b_shuffled_bottleneck_r256_s0", "mode": "shuffled", "seed": 0},
    {"name": "lora_only_35b", "ckpt": "adapter_35b_lora_r16_n8_s0", "mode": None, "seed": 0},
    {"name": "ple_lora_35b", "ckpt": "adapter_35b_ple_lora_r16_s0", "mode": "real", "seed": 0},
]


def _window_nlls(model, tokens, seq_len: int, n: int, batch_size: int, device: str) -> np.ndarray:
    import torch

    out = np.empty(n, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            ids = torch.from_numpy(batch_windows(tokens, idx, seq_len)).to(device)
            logits = model(input_ids=ids).logits
            tok_loss, _ = token_nll_unreduced(logits, ids)
            out[start : start + len(idx)] = tok_loss.float().mean(dim=-1).cpu().numpy()
            if (start // batch_size) % 50 == 0:
                print(f"    window {start}/{n}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=RECIPIENT_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt-dir", type=Path, default=ROOT / "artifacts" / "checkpoints")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "metrics" / "expanded_val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--windows", type=int, default=0, help="0 = all non-overlapping val windows")
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--gate-bias", type=float, default=0.0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import torch

    tokens = load_tokens(args.data_dir / "val.bin")
    n_all = n_windows(len(tokens), args.seq_len)
    n = n_all if args.windows <= 0 else min(args.windows, n_all)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"val tokens={len(tokens)} windows={n}/{n_all} seq={args.seq_len} bs={args.batch_size}", flush=True)

    print(f"loading {args.model}", flush=True)
    model, _tok = load_recipient(args.model, device=args.device)
    store = open_store()
    graft = None
    means = {}
    t_all = time.time()
    lora_ready = False

    for cond in CONDITIONS:
        npy = args.out_dir / f"{cond['name']}.npy"
        if npy.is_file() and not args.force and np.load(npy).shape[0] == n:
            arr = np.load(npy)
            print(f"{cond['name']}: loaded {npy} mean={arr.mean():.4f}", flush=True)
        else:
            if graft is not None:
                graft.remove()
                graft = None
            if cond["ckpt"] is None:
                print(f"{cond['name']}: baseline (no PLE)", flush=True)
            else:
                ckpt_path = args.ckpt_dir / cond["ckpt"] / "adapter.pt"
                if not ckpt_path.is_file():
                    print(f"{cond['name']}: skip missing {ckpt_path}", flush=True)
                    continue
                blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ck_args = blob.get("args") or {}
                lora_r = int(ck_args.get("lora_r") or 0)
                if lora_r > 0 and _model_hidden(model) != int(ck_args.get("hidden_size") or 2560):
                    print(
                        f"{cond['name']}: skip LoRA ckpt for hidden {_model_hidden(model)}",
                        flush=True,
                    )
                    continue
                if lora_r > 0:
                    if not lora_ready:
                        layers = transformer_layers(model)
                        n_wrap = min(int(ck_args.get("lora_layers") or 8), len(layers))
                        for i in range(n_wrap):
                            apply_lora(layers[i], rank=lora_r, alpha=float(ck_args.get("lora_alpha") or 16.0))
                        lora_ready = True
                        print(f"applied LoRA r={lora_r} on first {n_wrap} layers", flush=True)
                    load_lora_state_dict(model, blob.get("lora") or {})
                if blob.get("adapter") is not None and cond.get("mode"):
                    arch = ck_args.get("arch", "bottleneck")
                    rank = int(ck_args.get("rank", args.rank))
                    hidden = int(ck_args.get("hidden_size") or 2560)
                    if hidden != _model_hidden(model):
                        print(
                            f"{cond['name']}: skip width mismatch adapter hidden={hidden} model={_model_hidden(model)}",
                            flush=True,
                        )
                        continue
                    adapter = PleAdapter(
                        hidden_size=hidden,
                        ple_dim=int(ck_args.get("ple_dim") or 2560),
                        rank=rank,
                        gate_bias=args.gate_bias,
                        arch=arch,
                    ).to(device=args.device, dtype=torch.bfloat16)
                    adapter.load_state_dict(blob["adapter"])
                    adapter.gate_weight.requires_grad_(False)
                    adapter.gate_bias.requires_grad_(False)
                    lookup = PleLookup(store=store, mode=LookupMode(cond["mode"]), seed=cond["seed"])
                    layer = int(ck_args.get("layer", 1))
                    graft = PleGraft(model, adapter, lookup, enabled=True, layer_index=layer)
                    print(f"{cond['name']}: {ckpt_path} mode={cond['mode']} layer={layer} lora_r={lora_r}", flush=True)
                else:
                    print(f"{cond['name']}: {ckpt_path} LoRA-only lora_r={lora_r}", flush=True)
            t0 = time.time()
            arr = _window_nlls(model, tokens, args.seq_len, n, args.batch_size, args.device)
            np.save(npy, arr)
            print(f"{cond['name']}: mean={arr.mean():.4f} first32={arr[:32].mean():.4f} seconds={time.time()-t0:.1f}", flush=True)
        means[cond["name"]] = {"nll": float(arr.mean()), "ppl": nll_to_ppl(float(arr.mean())), "first32": float(arr[: min(32, n)].mean()), "path": str(npy)}

    if graft is not None:
        graft.remove()

    base = np.load(args.out_dir / "B0_baseline.npy")
    comparisons = {}
    for name in (
        "P1_real_s0",
        "P1_real_s1",
        "C1_random_s0",
        "C2_shuffled_s0",
        "P1_r512_s0",
        "P1_r1024_s0",
        "P1_dense_s0",
        "P1_r256_s1_gate",
        "P1_L0_s0",
        "P1_L3_s0",
        "lora_only_s0",
        "ple_lora_L3_s0",
        "P1_35b_s0",
        "C2_35b_s0",
        "lora_only_35b",
        "ple_lora_35b",
    ):
        npy_path = args.out_dir / f"{name}.npy"
        if not npy_path.is_file():
            continue
        other = np.load(npy_path)
        comparisons[f"{name}_minus_B0"] = paired_delta_stats(other, base)
        if name != "P1_real_s0" and (args.out_dir / "P1_real_s0.npy").is_file():
            comparisons[f"{name}_minus_P1_real_s0"] = paired_delta_stats(
                other, np.load(args.out_dir / "P1_real_s0.npy")
            )
    def _pair(a_name: str, b_name: str, key: str) -> None:
        pa, pb = args.out_dir / f"{a_name}.npy", args.out_dir / f"{b_name}.npy"
        if pa.is_file() and pb.is_file():
            comparisons[key] = paired_delta_stats(np.load(pa), np.load(pb))

    _pair("P1_real_s0", "C1_random_s0", "P1_real_s0_minus_C1_random_s0")
    _pair("P1_real_s0", "C2_shuffled_s0", "P1_real_s0_minus_C2_shuffled_s0")
    _pair("P1_real_s0", "P1_real_s1", "P1_real_s0_minus_P1_real_s1")

    report = {
        "windows": n,
        "seq_len": args.seq_len,
        "val_tokens": int(len(tokens)),
        "seconds": time.time() - t_all,
        "means": means,
        "paired": comparisons,
    }
    out = args.out_dir / "report.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
