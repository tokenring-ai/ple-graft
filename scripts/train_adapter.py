#!/usr/bin/env python3
"""Phase 2 adapter-only training. Backbone and PLE table stay frozen."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ple_graft.adapter import PleAdapter  # noqa: E402
from ple_graft.data import DEFAULT_DATA_DIR, batch_windows, load_tokens, n_windows  # noqa: E402
from ple_graft.donor_assets import RECIPIENT_DIR  # noqa: E402
from ple_graft.lora import apply_lora, lora_state_dict  # noqa: E402
from ple_graft.metrics import nll_to_ppl, write_json  # noqa: E402
from ple_graft.model_io import load_recipient  # noqa: E402
from ple_graft.ple_lookup import LookupMode, default_lookup  # noqa: E402
from ple_graft.qwen35_patch import PleGraft, transformer_layers  # noqa: E402


def _trainable(model) -> tuple[list[str], list]:
    names, tensors = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            tensors.append(p)
    return names, tensors


def _batch(tokens, rng, seq_len: int, batch_size: int, replace: bool = True) -> np.ndarray:
    n = n_windows(len(tokens), seq_len)
    if n <= 0:
        raise RuntimeError("not enough tokens for seq_len")
    idx = rng.integers(0, n, size=batch_size) if replace else rng.choice(n, size=min(batch_size, n), replace=False)
    return batch_windows(tokens, [int(i) for i in np.asarray(idx).tolist()], seq_len)


def _forward_nll(model, input_ids):
    import torch

    out = model(input_ids=input_ids, labels=input_ids)
    return out.loss, out.logits


def _eval_nll(model, tokens, seq_len: int, max_windows: int, device: str) -> dict:
    import torch

    n = min(n_windows(len(tokens), seq_len), max_windows)
    if n == 0:
        raise RuntimeError("empty eval")
    losses = []
    n_tok = 0
    model.eval()
    with torch.no_grad():
        for i in range(n):
            ids = torch.from_numpy(batch_windows(tokens, [i], seq_len)).to(device)
            loss, _ = _forward_nll(model, ids)
            losses.append(float(loss.item()))
            n_tok += seq_len - 1
    mean = float(np.mean(losses))
    return {"nll": mean, "ppl": nll_to_ppl(mean), "windows": n, "tokens": n_tok}


def _gate_stats(adapter) -> dict:
    g = adapter.last_gate
    if g is None:
        return {}
    x = g.float().reshape(-1)
    return {
        "gate_mean": float(x.mean()),
        "gate_median": float(x.median()),
        "gate_p10": float(x.quantile(0.1)),
        "gate_p90": float(x.quantile(0.9)),
        "adapter_rms": None if adapter.last_adapter_rms is None else float(adapter.last_adapter_rms),
        "hidden_rms": None if adapter.last_hidden_rms is None else float(adapter.last_hidden_rms),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=RECIPIENT_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-tokens", type=int, default=50_000_000)
    p.add_argument("--eval-every-tokens", type=int, default=2_000_000)
    p.add_argument("--eval-windows", type=int, default=32)
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=2560, help="recipient residual width")
    p.add_argument("--ple-dim", type=int, default=2560, help="PLE concat width (16×160)")
    p.add_argument("--arch", default="bottleneck", choices=["bottleneck", "dense", "per_head"])
    p.add_argument("--gate-bias", type=float, default=-3.0)
    p.add_argument("--freeze-gate", action="store_true", help="Keep the scalar gate fixed so it cannot collapse to 0")
    p.add_argument("--init-ckpt", type=Path, default=None, help="Load an existing adapter.pt before training")
    p.add_argument("--train-only-gate", action="store_true", help="Freeze adapter maps; train only the scalar gate")
    p.add_argument("--gate-lr-mult", type=float, default=0.1, help="If the gate is trained, use this multiple of --lr")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--ple-mode", default="real", choices=[m.value for m in LookupMode])
    p.add_argument("--layer", type=int, default=1, help="0-based decoder layer to inject after the mixer residual")
    p.add_argument("--no-ple", action="store_true", help="Do not install the PLE graft (LoRA-only control)")
    p.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 disables LoRA")
    p.add_argument("--lora-layers", type=int, default=8, help="Apply LoRA to the first N decoder layers")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--freeze-adapter", action="store_true", help="Keep the PLE adapter frozen (PLE+LoRA)")
    p.add_argument("--run-tag", default=None, help="Override checkpoint/log name")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    args = p.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_path = args.data_dir / "train.bin"
    val_path = args.data_dir / "val.bin"
    if args.sanity:
        src = args.data_dir / "smoke_val.bin"
        if not src.is_file() or src.stat().st_size == 0:
            print(f"missing {src}; run scripts/prepare_data.py --smoke-only", file=sys.stderr)
            return 2
        tokens = load_tokens(src)
        train_tokens = tokens[: args.seq_len * 4]
        val_tokens = tokens[: args.seq_len * 2]
    else:
        if not train_path.is_file() or not val_path.is_file():
            print(f"missing packed data in {args.data_dir}; run scripts/prepare_data.py", file=sys.stderr)
            return 2
        train_tokens = load_tokens(train_path)
        val_tokens = load_tokens(val_path)

    print(f"loading model {args.model} on {args.device}", flush=True)
    model, _tok = load_recipient(args.model, device=args.device)
    adapter = None
    graft = None
    lookup = None
    n_lora = 0
    if args.lora_r > 0:
        layers = transformer_layers(model)
        n_wrap = min(args.lora_layers, len(layers))
        for i in range(n_wrap):
            n_lora += apply_lora(layers[i], rank=args.lora_r, alpha=args.lora_alpha)
        print(f"LoRA r={args.lora_r} on first {n_wrap} layers, wrapped {n_lora} linears", flush=True)
    if not args.no_ple:
        lookup = default_lookup(LookupMode(args.ple_mode), seed=args.seed)
        adapter = PleAdapter(
            hidden_size=args.hidden_size,
            ple_dim=args.ple_dim,
            rank=args.rank,
            gate_bias=args.gate_bias,
            arch=args.arch,
        ).to(device=args.device, dtype=torch.bfloat16)
        if args.init_ckpt is not None:
            blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            adapter.load_state_dict(blob["adapter"])
            print(f"loaded adapter from {args.init_ckpt}", flush=True)
        print("layer", args.layer, flush=True)
        graft = PleGraft(model, adapter, lookup, enabled=True, layer_index=args.layer)
        graft.freeze_backbone()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        print("no-ple: graft skipped", flush=True)
    if args.lora_r > 0:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
    if adapter is not None:
        if args.freeze_adapter:
            for p in adapter.parameters():
                p.requires_grad_(False)
            print("PLE adapter frozen", flush=True)
        elif args.train_only_gate:
            for n, p in adapter.named_parameters():
                p.requires_grad_(n.startswith("gate_"))
            with torch.no_grad():
                adapter.gate_weight.zero_()
                adapter.gate_bias.fill_(0.0)
            print("train-only-gate: maps frozen, gate re-inited to 0.5", flush=True)
        elif args.freeze_gate:
            adapter.gate_weight.requires_grad_(False)
            adapter.gate_bias.requires_grad_(False)
    names, params = _trainable(model)
    if not params:
        print("NO-GO: no trainable parameters", file=sys.stderr)
        return 1
    print("arch", args.arch, "rank", args.rank, "lora_r", args.lora_r)
    print("trainable:", names[:40], ("..." if len(names) > 40 else ""))
    print("n_trainable", sum(p.numel() for p in params))

    model.eval()
    max_abs = 0.0
    if graft is not None and args.init_ckpt is None:
        probe = torch.from_numpy(batch_windows(val_tokens, [0], min(args.seq_len, 64))).to(args.device)
        with torch.no_grad():
            graft.enabled = True
            loss_on, logits_on = _forward_nll(model, probe)
            graft.enabled = False
            loss_off, logits_off = _forward_nll(model, probe)
            graft.enabled = True
        max_abs = float((logits_on - logits_off).abs().max())
        print(f"step0 max_abs_logit_diff={max_abs:.3e} nll_on={float(loss_on):.4f} nll_off={float(loss_off):.4f}", flush=True)
        if max_abs > 1e-5:
            print("NO-GO: zero-init adapter changed logits", file=sys.stderr)
            return 1
    else:
        print("skip step0 identity (warm start or no-ple LoRA)", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if args.no_ple:
        run_tag = f"lora_r{args.lora_r}_n{args.lora_layers}_s{args.seed}"
    else:
        run_tag = (
            f"{args.ple_mode}_{args.arch}_s{args.seed}"
            if args.arch in ("dense", "per_head")
            else f"{args.ple_mode}_{args.arch}_r{args.rank}_s{args.seed}"
        )
        if args.layer != 1:
            run_tag = f"{run_tag}_L{args.layer}"
        if args.train_only_gate:
            run_tag = f"{run_tag}_gate"
        if args.lora_r > 0:
            run_tag = f"{run_tag}_lora_r{args.lora_r}"
    if args.run_tag:
        run_tag = args.run_tag
    ckpt_dir = args.out_dir / "checkpoints" / f"adapter_{run_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "metrics" / f"train_{run_tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tokens_per_step = args.batch_size * args.seq_len
    target_tokens = args.seq_len * args.batch_size * 40 if args.sanity else args.train_tokens
    eval_every = tokens_per_step * 5 if args.sanity else args.eval_every_tokens
    seen = 0
    step = 0
    t0 = time.time()
    best_val = None
    history = []

    def run_eval(tag: str) -> dict:
        metrics = _eval_nll(model, val_tokens, args.seq_len, args.eval_windows, args.device)
        metrics["tag"] = tag
        metrics["step"] = step
        metrics["seen_tokens"] = seen
        print(
            f"eval {tag} step={step} seen={seen:,} val_nll={metrics['nll']:.4f} ppl={metrics['ppl']:.3f}",
            flush=True,
        )
        return metrics

    first_eval = run_eval("before_train")
    history.append(first_eval)

    model.train()
    while seen < target_tokens:
        ids_np = _batch(train_tokens, rng, args.seq_len, args.batch_size)
        ids = torch.from_numpy(ids_np).to(args.device)
        opt.zero_grad(set_to_none=True)
        loss, _ = _forward_nll(model, ids)
        if not torch.isfinite(loss):
            print(f"FAILED nonfinite loss at step {step}: {loss}", flush=True)
            return 1
        loss.backward()
        gn = {}
        n_logged = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            gn[n] = 0.0 if p.grad is None else float(p.grad.float().norm())
            n_logged += 1
            if n_logged >= 12:
                break
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        seen += tokens_per_step
        step += 1
        rec = {
            "step": step,
            "seen_tokens": seen,
            "train_nll": float(loss.item()),
            "seconds": time.time() - t0,
            "lookup": None if graft is None or graft.last_stats is None else graft.last_stats.__dict__,
            "grads": gn,
            **(_gate_stats(adapter) if adapter is not None else {}),
        }
        history.append(rec)
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if step % 5 == 0 or args.sanity:
            gather = None if graft is None or graft.last_stats is None else round(graft.last_stats.gather_ms, 1)
            print(
                f"step {step:5d} seen={seen:,} train_nll={rec['train_nll']:.4f} "
                f"gate_mean={rec.get('gate_mean')} gather_ms={gather}",
                flush=True,
            )
        if seen % eval_every < tokens_per_step or seen >= target_tokens:
            model.eval()
            ev = run_eval(f"seen_{seen}")
            history.append(ev)
            if graft is not None:
                graft._ple_concat = None
                graft._ple_token_shape = None
            if best_val is None or ev["nll"] < best_val:
                best_val = ev["nll"]
                payload = {
                    "adapter": None if adapter is None else adapter.state_dict(),
                    "lora": lora_state_dict(model) if args.lora_r > 0 else {},
                    "step": step,
                    "seen_tokens": seen,
                    "val_nll": ev["nll"],
                    "args": vars(args),
                }
                torch.save(payload, ckpt_dir / "adapter.pt")
            model.train()

    summary = {
        "ple_mode": None if args.no_ple else args.ple_mode,
        "no_ple": args.no_ple,
        "lora_r": args.lora_r,
        "lora_layers": args.lora_layers,
        "arch": args.arch,
        "rank": args.rank,
        "layer": args.layer,
        "run_tag": run_tag,
        "seed": args.seed,
        "sanity": args.sanity,
        "n_trainable": sum(p.numel() for p in params),
        "trainable": names,
        "step0_max_abs_logit_diff": max_abs,
        "first_eval": first_eval,
        "last_eval": next(h for h in reversed(history) if "windows" in h),
        "best_val_nll": best_val,
        "seen_tokens": seen,
        "steps": step,
        "seconds": time.time() - t0,
        "log": str(log_path),
        "ckpt": str(ckpt_dir / "adapter.pt"),
    }
    out = args.out_dir / "metrics" / f"train_{run_tag}_summary.json"
    write_json(out, summary)
    print(json.dumps({k: summary[k] for k in ("best_val_nll", "seen_tokens", "step0_max_abs_logit_diff", "seconds")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
