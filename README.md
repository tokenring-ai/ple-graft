# ple-graft

Graft Qwen3.8-Flash-Next’s frozen ~51B n-gram PLE table onto smaller same-family Qwen3 models.

**Result:** the table is a portable, address-aligned memory. A tiny frozen-backbone adapter improves FineWeb-Edu NLL a little, shuffled addresses make it worse, and **LoRA-only beats PLE+LoRA** on both a dense 4B and a 35B-A3B MoE. Research result, not a product path.

| Writeup | Recipient | Frozen PLE ΔNLL | vs LoRA-only |
|---|---|---|---|
| [MEMO.md](MEMO.md) | Qwen3.5-4B-Base | **−0.011** | PLE+LoRA **worse** (+0.0017) |
| [MEMO_35B_A3B.md](MEMO_35B_A3B.md) | Qwen3.6-35B-A3B (instruct) | **−0.007** | PLE+LoRA **worse** (+0.0050) |

Primary metric: next-token NLL on a frozen 8M-token FineWeb-Edu val (7,812 windows × 1024). 95% CIs are paired bootstrap over windows.

This is a research experiment, not an inference engine. llama.cpp, ninfer, SGLang, and tr-infer-rs are read-only oracles.

## What this repo contains

- Graft code: hash, mmap FP8 lookup, identity-init adapter, Qwen3.5 / Qwen3.5-MoE hook
- Packed-data helpers and train/eval scripts
- Paper-style memos and the per-window NLL arrays behind the CIs
- Unit tests for tokenizer identity, hash/row goldens, causal alignment, LoRA, paired Δ

**Not in git:** the 48 GiB `ple.bin` table, recipient weights, adapter checkpoints, train `.jsonl` / `.log`, and per-token gate dumps. Do not publish a combined 4B+PLE or 35B+PLE weight dump.

## Headline numbers (full 7,812-window val)

**4B-Base** — B0 = 2.3086

| Condition | NLL | Δ vs B0 | 95% CI |
|---|---|---|---|
| Real PLE, layer 1, seed 1 | 2.2974 | −0.0111 | [−0.0115, −0.0108] |
| Shuffled addresses | 2.4567 | +0.148 | [+0.147, +0.149] |
| LoRA-only (r=16, first 8 layers) | **2.2905** | **−0.018** | — |
| PLE+LoRA | 2.2922 | −0.016 | vs LoRA +0.00165 [+0.00149, +0.00182] |

**35B-A3B instruct** — B0 = 2.1264. Pre-registered amplify bar was CI(Δ) entirely below −0.025.

| Condition | NLL | Δ vs B0 | 95% CI |
|---|---|---|---|
| Real PLE, layer 1, seed 0 | 2.1195 | −0.00685 | [−0.00766, −0.00602] |
| Shuffled addresses | 2.4257 | +0.299 | [+0.297, +0.302] |
| LoRA-only | **2.0909** | **−0.0354** | [−0.0360, −0.0348] |
| PLE+LoRA | 2.0959 | −0.0304 | vs LoRA +0.00500 [+0.00477, +0.00522] |

Cite shuffled as the causal control. Do not cite the 32-window random-row 2.319 as C1; that sampler was batch-shape-dependent.

## Setup

Python 3.12+, a CUDA GPU (4B is light; 35B-A3B is ~70 GiB BF16), and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest tests
```

Tests that need local checkpoints or `ple.bin` skip when those files are missing.

### External assets

| Asset | Env var | Default on this machine |
|---|---|---|
| Packed PLE table (`ple.bin` + pack tokenizer) | `PLE_GRAFT_PACK_DIR` | `/opt/llm/tr-infer/flashnext-nvfp4` |
| Flash-Next HF donor (hash tensors) | `PLE_GRAFT_DONOR_DIR` | `/opt/llm/Qwen3.8-Flash-Next-NVFP4` |
| Qwen3.5-4B-Base | `PLE_GRAFT_RECIPIENT_DIR` | `/mnt/llm-cache/models/Qwen3.5-4B-Base` |
| Qwen3.6-35B-A3B | `PLE_GRAFT_MOE_DIR` | `/opt/llm/Qwen3.6-35B-A3B` |
| Packed FineWeb `train.bin` / `val.bin` | `PLE_GRAFT_DATA_DIR` | `/mnt/llm-cache/ple-graft/data` |

Download a recipient with:

```bash
uv run python scripts/download_recipient.py --repo Qwen/Qwen3.5-4B-Base --out /path/to/Qwen3.5-4B-Base
uv run python scripts/download_recipient.py --repo Qwen/Qwen3.6-35B-A3B --out /path/to/Qwen3.6-35B-A3B
```

`ple.bin` is a Flash-Next pack file (~48 GiB FP8 E4M3, 320,001,536 physical rows × 160, tensor BF16 scale). It is not on Hugging Face as a standalone file in this experiment.

## Reproduce

Hash/row goldens and tokenizer identity:

```bash
uv run python scripts/extract_ple_manifest.py
uv run python scripts/verify_tokenizers.py
uv run python scripts/make_golden_vectors.py
uv run pytest tests/test_tokenizer_identity.py tests/test_hash_golden.py \
  tests/test_ple_row_golden.py tests/test_causal_alignment.py
```

Identity graft (lookup on, residual off or zero `W_up`):

```bash
uv run python scripts/eval_graft_disabled.py
```

Packed FineWeb-Edu (8M val, 80M train, sequential split — do not re-iterate a streaming dataset):

```bash
uv run python scripts/prepare_data.py --train-tokens 80000000 --val-tokens 8000000
```

Adapter-only (4B, freeze-gate 0.5, rank 256):

```bash
uv run python scripts/train_adapter.py --seq-len 1024 --batch-size 2 \
  --train-tokens 8000000 --eval-every-tokens 2000000 --seed 0 \
  --ple-mode real --rank 256 --arch bottleneck --freeze-gate
uv run python scripts/eval_val.py --batch-size 4
```

35B-A3B uses the same scripts with `--model`, `--hidden-size 2048`, `--ple-dim 2560`, `--batch-size 1`. See `configs/moe_35b_a3b.yaml` for the pre-registered bars.

LoRA-only vs PLE+LoRA: `scripts/train_adapter.py` with `--lora-r 16` (and `--no-ple` for the control). Routed MoE experts are not wrapped.

## Method (short)

Lookup is 16 heads (8 bigram + 8 trigram), 160-d FP8 rows, concat 2560. Hash is checkpoint `uint64` multiply–XOR–mod, EOS 248044, insertion at 0-based layer 1 (`ple_layer_ids=[2]` is 1-based). Unique-row gather; e4m3fn NaNs flushed to 0.

Adapter after the mixer residual, before FFN (before MoE on 35B):

```
z = RMSNorm(E)
a = W_up(SiLU(W_down(z)))     # 2560→256→hidden, W_up = 0 at init
h' = h + 0.5 · a
```

Flash-Next’s PLE contextualizer (`key_proj`, conv, HyperConnection) is **not** copied.

## License

Code and memos: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party model/data terms. The PLE table and Qwen checkpoints stay with their upstream licenses; this repo does not bundle them.
