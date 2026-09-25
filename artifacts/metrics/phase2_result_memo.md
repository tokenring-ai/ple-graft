# Phase 2 result memo (seed 0)

**Superseded by [MEMO.md](../../MEMO.md).** This note is the 32-window train-echo. Cite the 7,812-window numbers in `MEMO.md`. Do not cite the 32-window random-row 2.319 as C1 (batch-shape RNG).

Question: with Qwen3.5-4B and the 51B PLE table frozen, does a ~1.3M adapter improve held-out NLL using the real Flash-Next lookup?

## Setup
- Recipient: Qwen3.5-4B-Base, all backbone weights frozen
- Adapter: 2560→256→2560, RMSNorms trainable, W_up zero-init
- Gate: frozen open (bias 0 → mix 0.5); a learned gate collapsed to 0 on the first attempt
- Data: FineWeb-Edu packed tokens, 8M val / 80M train sequential split
- Eval: 32 fixed windows × 1024 tokens (32,736 tokens), identical across runs
- Lookup: mmap FP8 `ple.bin`, unique-row gather, e4m3fn NaNs flushed to 0

## Seed-0 val NLL (lower is better)

| Run | Best val NLL | Δ vs B0 | Tokens at best |
|---|---|---|---|
| B0 baseline (no PLE / before-train) | 2.2955 | 0 | 0 |
| P1 real donor lookup | **2.2797** | **−0.0158** | 6.0M |
| C1 random frozen rows | 2.3194 | +0.0239 | 8.0M |
| C2 shuffled addresses | 2.4360 | +0.1405 | 8.0M |

P1 > B0, P1 > C1, P1 > C2.

## Interpretation
Real addresses are not interchangeable with random or shuffled rows. Shuffled and random both **hurt** relative to the untouched 4B; real PLE is the only condition that does not. That is evidence the pretrained table has address-aligned content, not merely extra adapter capacity.

The gain versus the frozen 4B is small and flattened after ~2–6M tokens (8M real was slightly worse than 6M). Do not spend the remaining 42M real-PLE tokens. Do not unfreeze the backbone or start LoRA/distill to chase a −0.016 ΔNLL.

## Seed 1 (same recipe, different data order)
Best val NLL **2.2824** at 2M tokens (Δ **−0.0131**). Then 2.2829 / 2.2833 / 2.2841 at 4/6/8M — same flatten. Both seeds beat baseline by ~0.013–0.016.

## Phase-2 go/no-go
**GO on Q2/Q3 at this scale:** real lookup beats shuffled and random, and the small held-out NLL gain repeats across two seeds.

**STOP escalation:** do not start LoRA, distillation, or unfreeze the backbone. The gain is small and saturates by ~2–6M tokens. Negative shuffled/random results are part of the record (`artifacts/metrics/phase2_run_of_record.json`).
