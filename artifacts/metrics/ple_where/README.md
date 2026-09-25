# Stage A: where the 0.010 lives

Checkpoint: `adapter_real_bottleneck_r256_s1_gate`. Val: 7,812 windows × 1024 (same `val.bin`).

## Heads (eval-time knockout)

| Keep | NLL | Δ vs B0 (2.3086) | Δ vs full16 (2.2961) |
|---|---|---|---|
| full 16 | **2.2961** | **−0.0125** | 0 |
| trigram 8–15 | 2.3143 | +0.0057 | +0.0182 |
| rand 8 of 16 | 2.3292 | +0.0206 | +0.0331 |
| bigram 0–7 | 2.3528 | +0.0442 | +0.0567 |

Any 8-head subset is **worse than no PLE**. Trigrams are less bad than bigrams, but the −0.0125 needs the full concat. Not a bland residual: the 16 hash heads are doing different work together.

## Tokens (7,991,676 next-token positions)

Mean gate **0.284** (did not collapse). Mean ΔNLL **−0.0125**.

**By B0 error quintile** (this is the structure):

| Q | B0 NLL range | mean Δ | mean gate |
|---|---|---|---|
| 0 easiest | 0–0.17 | +0.0006 | 0.283 |
| 1 | 0.17–0.88 | −0.0002 | 0.296 |
| 2 | 0.88–2.09 | −0.0070 | 0.290 |
| 3 | 2.09–4.13 | −0.0172 | 0.281 |
| 4 hardest | 4.13–24.5 | **−0.0387** | 0.267 |

Almost all of the gain is on tokens the frozen 4B already finds hard. Easy tokens are unchanged or slightly worse.

**By unigram frequency:** mixed (rarest −0.015, mid −0.019, common −0.011). Gate rises slightly with frequency (0.25→0.32). Rarity is not the main axis.

## Decision

Stage A **continue**: gain is concentrated (hard tokens) and head-complete (all 16). Next is Stage B insertion placement (layers 0 / 1 / 3), not LoRA.

**Gate analysis** (CPU, existing dumps): [gate_analysis.md](gate_analysis.md). Verdict: anti-correlated mix — adapter helps on hard tokens; the gate does not open there.
