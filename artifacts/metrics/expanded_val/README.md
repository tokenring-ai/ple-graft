# Expanded frozen val (8,000,000 tokens, 7,812 windows × 1024)

Same FineWeb-Edu `val.bin` used in training, but every non-overlapping window instead of the first 32. Per-window NLL arrays are the `.npy` files. First-32 means reproduce the training evals.

| Condition | First 32 | Full 7,812 | Δ vs B0 (full) | 95% CI |
|---|---|---|---|---|
| B0 baseline | 2.2951 | 2.3086 | 0 | — |
| P1 real s0 | 2.2796 | 2.2989 | −0.0097 | [−0.0100, −0.0093] |
| P1 real s1 | 2.2823 | 2.2974 | −0.0111 | [−0.0115, −0.0108] |
| C1 random s0 | 2.6674 | 2.6823 | +0.3737 | [+0.366, +0.381] |
| C2 shuffled s0 | 2.4361 | 2.4567 | +0.1481 | [+0.147, +0.149] |

Real lookup is better than baseline on 82–88% of windows. Random/shuffled are worse on almost every window.

## Why C1 random 32-window (2.319) ≠ full val (2.682)

Those are **not** the same random table. The original RANDOM sampler did `default_rng(seed).integers(..., size=row_ids.shape)` on **every** lookup, so the draw depended on batch shape:

- training used batch 2
- the 32-window train eval used batch 1
- expanded val used batch 4

Baseline and real lookup have no RNG, which is why their first-32 means match training to ~0.0001. Shuffled used a stored permutation and is the trustworthy negative control (full-val 2.457). RANDOM is now the same kind of stored permutation (different stream from shuffled) in `ple_lookup.py`; the old random checkpoint was trained under the shape-dependent sampler and should not be cited as C1.
