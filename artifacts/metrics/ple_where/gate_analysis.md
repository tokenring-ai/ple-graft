# Gate analysis

Checkpoint: seed-1 r=256 + trained gate. Dumps: 7,812 × 1023 positions. CPU only.

**Anti-correlated mix, not a detector.** The adapter helps when the frozen 4B is already lost (hardest B0 quintile ΔNLL **−0.039** vs **+0.001** on the easiest). The gate does **not** open there: Spearman(g, B0-NLL) = **−0.184**, and g is *lower* on hard tokens (0.267 vs 0.283). Spearman(g, ΔNLL) = **−0.002** — opening the gate is uncorrelated with benefit. N-gram rarity does not move g (ρ = 0.015). Token type: slightly higher g on punctuation (0.33) than words (0.28), without a matching ΔNLL story. The gate is a mild global mix; the adapter, not the gate, is the policy.

- mean g = 0.2835
- mean ΔNLL = -0.0125
- Spearman(g, B0-NLL) = -0.1837
- Spearman(g, ΔNLL) = -0.0023
- Spearman(g, n-gram rarity) = 0.0146

## Quintiles of gate vs ΔNLL

| Q | g lo | g hi | n | mean Δ | mean g |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0630 | 0.2480 | 1519056 | -0.0190 | 0.2280 |
| 1 | 0.2480 | 0.2695 | 1560683 | -0.0101 | 0.2578 |
| 2 | 0.2695 | 0.2871 | 1551699 | -0.0071 | 0.2767 |
| 3 | 0.2871 | 0.3105 | 1690345 | -0.0058 | 0.2969 |
| 4 | 0.3105 | 0.9297 | 1669893 | -0.0206 | 0.3509 |

## Quintiles of B0 NLL (replicate)

| Q | B0 lo | B0 hi | n | mean Δ | mean g |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.1729 | 1597777 | 0.0006 | 0.2835 |
| 1 | 0.1729 | 0.8828 | 1595526 | -0.0002 | 0.2960 |
| 2 | 0.8828 | 2.0938 | 1596623 | -0.0070 | 0.2902 |
| 3 | 2.0938 | 4.1250 | 1600430 | -0.0172 | 0.2809 |
| 4 | 4.1250 | 24.5000 | 1601320 | -0.0387 | 0.2672 |

## Token-type buckets

| bucket | n | mean g | mean Δ |
| --- | --- | --- | --- |
| word | 6792479 | 0.2776 | -0.0115 |
| punct | 793642 | 0.3299 | -0.0086 |
| digit | 397969 | 0.2860 | -0.0037 |
| eos | 7586 | 0.5795 | -1.7299 |

## Previous-token buckets

| bucket | n | mean g | mean Δ |
| --- | --- | --- | --- |
| word | 6792474 | 0.2817 | -0.0098 |
| punct | 793631 | 0.2961 | -0.0304 |
| digit | 397986 | 0.2901 | -0.0106 |
| eos | 7585 | 0.2264 | -0.6469 |

## N-gram rarity (higher = more common addresses in val)

| Q | rarity lo | rarity hi | n | mean Δ | mean g |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.6931 | 1.0398 | 1598157 | -0.0113 | 0.2826 |
| 1 | 1.0398 | 1.7530 | 1598381 | -0.0111 | 0.2835 |
| 2 | 1.7530 | 2.7229 | 1598457 | -0.0112 | 0.2848 |
| 3 | 2.7229 | 4.3796 | 1598334 | -0.0114 | 0.2847 |
| 4 | 4.3796 | 9.7185 | 1598347 | -0.0174 | 0.2820 |

## Window code-ish fraction (weak FineWeb split)

| Q | frac lo | frac hi | n windows | mean Δ | mean g |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0000 | 0.0000 | 0 | None | None |
| 1 | 0.0000 | 0.0010 | 2782 | -0.0114 | 0.2840 |
| 2 | 0.0010 | 0.0010 | 0 | None | None |
| 3 | 0.0010 | 0.0029 | 3435 | -0.0134 | 0.2830 |
| 4 | 0.0029 | 0.0400 | 1595 | -0.0123 | 0.2839 |

NER was not run: decode/n-gram tables do not show a name-like content effect stronger than punctuation.

