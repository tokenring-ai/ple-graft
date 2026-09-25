# Grafting Flash-Next n-gram memory onto Qwen3.5-4B

**Technical memo** · 13 September 2026  
**Repo:** `ple-graft` · **Recipient:** `Qwen/Qwen3.5-4B-Base` · **Donor table:** Qwen3.8-Flash-Next PLE (~51B FP8 rows)

## Abstract

Qwen3.8-Flash-Next stores ~51B parameters of n-gram embedding memory (PLE) that is addressed from raw token history, not from donor hidden states. We asked whether that table can function as a portable knowledge module for Qwen3.5-4B, which shares hidden width 2560 and padded vocab 248,320. With the 4B backbone and the table frozen, a 1.3M-parameter adapter improves held-out FineWeb NLL by **−0.010 to −0.012**. The gain is causal: shuffled or random row addresses *raise* NLL. It is concentrated on tokens the frozen 4B already finds hard, and it requires all 16 hash heads together. Widening the adapter, going full-rank, and changing insertion layer do not unlock a large extra effect. Once the backbone can adapt, the table is not an extra ingredient: **LoRA-only** (r=16 on the first 8 layers) reaches **−0.018**, and **PLE+LoRA is strictly worse** than LoRA-only (Δ +0.0017, 95% CI entirely above 0). Address-aligned PLE is transferable into a frozen smaller LM; it is not a product path for a better 4B.

## 1. Question

Flash-Next’s PLE is a sparse, deterministically addressed n-gram table used near the start of the network. Unlike transplanting an FFN or attention block, lookup keys are token n-grams, so in principle the table need not live in the donor’s residual geometry.

The operational questions were:

| ID | Question |
|---|---|
| Q1 | Exact tokenizer IDs, hash, offsets, and retrieved rows? |
| Q2 | Frozen table + tiny adapter improves 4B NLL? |
| Q3 | Gain from pretrained rows, not extra parameters? |
| Q4 | Does LoRA on the 4B unlock more value from the same frozen table? |

Primary metric: next-token NLL on a **frozen** FineWeb-Edu val split (8,000,000 tokens, 7,812 non-overlapping windows of 1024). Training used the following 80M tokens of the same stream (no overlap). 95% CIs are paired bootstrap over windows. First-32 window means of full-val arrays reproduce train-time 32-window evals to ~0.0001–0.0003.

We did **not** copy Flash-Next’s PLE contextualizer (`key_proj`, conv, HyperConnection). Those weights were trained against a 4-branch residual 4B does not have.

## 2. Method

**Lookup.** 16 heads (8 bigram + 8 trigram), 160-d FP8 rows, concat 2560. Hash is checkpoint `uint64` multiply–XOR–mod, EOS 248044, insertion originally 0-based layer 1 (HF `ple_layer_ids=[2]` is 1-based). Table: mmap `/opt/llm/tr-infer/flashnext-nvfp4/ple.bin`; unique-row gather; e4m3fn NaNs flushed to 0.

**Adapter (MVP).** After the token-mixer residual, before FFN:

```
z = RMSNorm(E)
a = W_up(SiLU(W_down(z)))     # 2560→256→2560, W_up = 0 at init
g = σ(w⊤ RMSNorm(h) + b)
h' = h + g · a
```

~1.32M trainable parameters. Backbone and table frozen. A learned gate from a cold start collapsed to ~0; recorded adapter-only runs freeze `g = 0.5` unless noted.

**Controls.** B0: no PLE. C2: stored permutation of real row ids (shuffled). C1 random was originally batch-shape-dependent and is **not** cited as the causal control; shuffled is.

## 3. Results

### 3.1 Q1 — Compatibility (GO, with a tokenizer note)

Pack multipliers, offsets, and vocab sizes match the HF tensors. Packed `ple.bin` rows match HF FP8 shards on hashed ids. Position `t` uses only `x_≤t`. Graft with residual hard-off is logit-identical to the untouched 4B (max abs diff 0).

Tokenizers: 10k snippets encode identically; shared token IDs match. Donor has 11 extra specials (`<think>`, tool, tts, audio). No remap. Direct graft is valid for LM text.

### 3.2 Q2 / Q3 — Frozen adapter (GO, small)

Full val, B0 = **2.3086**:

| Condition | NLL | Δ vs B0 | 95% CI |
|---|---|---|---|
| B0 baseline | 2.3086 | 0 | — |
| Real PLE, layer 1, seed 0 | 2.2989 | **−0.0097** | [−0.0100, −0.0093] |
| Real PLE, layer 1, seed 1 | 2.2974 | **−0.0111** | [−0.0115, −0.0108] |
| Shuffled addresses | 2.4567 | +0.148 | [+0.147, +0.149] |
| Random rows (flawed sampler) | 2.6823 | +0.374 | — |

Real lookup is better than baseline on 82–88% of windows. Shuffled/random are worse on almost every window. Two seeds agree. The 32-window train echo (−0.016) was slightly optimistic; the 8M number is the one to cite.

A scalar gate trained **after** the adapter exists stays in 0.27–0.58 (no collapse) and adds **−0.0014** vs frozen seed 1 (CI excludes 0). Full-val NLL **2.2961**.

### 3.3 Where the 0.011 lives

Eval-time head knockouts on the seed-1 + gate checkpoint (same 7,812 windows):

| Keep | NLL | Δ vs B0 |
|---|---|---|
| All 16 heads | **2.2961** | **−0.0125** |
| Trigrams only | 2.3143 | +0.006 |
| Random 8 of 16 | 2.3292 | +0.021 |
| Bigrams only | 2.3528 | +0.044 |

Any 8-head subset is **worse than no PLE**. The concat is not a bland extra residual.

Per-token ΔNLL vs frozen-4B error (7.99M positions):

| B0 difficulty quintile | mean ΔNLL |
|---|---|
| easiest 20% | +0.001 |
| hardest 20% | **−0.039** |

Almost all of the gain is on tokens the 4B already fails. Unigram rarity is a weak axis.

### 3.4 Interface capacity — not the bottleneck

Same frozen table, layer 1, freeze-gate 0.5, 8M train, full val:

| Interface | Trainable | NLL | vs r=256 s0 |
|---|---|---|---|
| r=256 (ref) | 1.32M | 2.2989 | 0 |
| r=512 | 2.63M | 2.2997 | worse (CI > 0) |
| r=1024 | 5.25M | 2.3026 | worse |
| Dense 2560×2560 | 6.56M | 2.576 | failed to train |
| 16× Linear(160,160) + mix | 6.97M | ~2.42 (32-win) | failed to train |

The ~0.010 is **not** a 256-d bottleneck. A full linear from zero does not yield −0.025.

### 3.5 Placement

r=256, freeze-gate 0.5, 8M, full val vs layer-1 seed 1 (2.2974):

| Site | Block | NLL | Δ vs L1 s1 | Win vs s1? |
|---|---|---|---|---|
| Layer 0 | GDN | 2.3000 | +0.0025 | no |
| Layer 1 | GDN (ref) | 2.2974 | 0 | — |
| Layer 3 | full attention | **2.2965** | **−0.00094** | yes (CI entirely < 0) |

Layer 3 is a small, significant win over ungated layer 1. It does not beat layer-1 + trained gate (2.2961).

### 3.6 Q4 — LoRA (NO-GO)

LoRA r=16 on the first 8 decoder layers, 8M tokens, lr 2e-4. PLE+LoRA freezes the layer-3 adapter and table.

| Run | NLL | Δ vs B0 |
|---|---|---|
| PLE-only (L3) | 2.2965 | −0.012 |
| PLE+LoRA | 2.2922 | −0.016 |
| **LoRA-only (no PLE)** | **2.2905** | **−0.018** |

PLE+LoRA − LoRA-only = **+0.00165**, 95% CI **[+0.00149, +0.00182]**. LoRA-only is strictly better. The 4B can exceed the frozen-PLE gain without the table.

## 4. Limitations

- Val is FineWeb-Edu LM NLL, not MMLU/HumanEval. The original spec treated NLL as the primary signal; we did not claim downstream accuracy.
- 8M train tokens; adapter curves flattened by 2–6M.
- One GPU, one val stream. CIs are over windows, not over datasets.
- Random-row 32-window 2.319 is **not** comparable to full-val 2.682 (batch-shape RNG). Cite shuffled (2.457) as C2.
- Donor contextualizer and distillation were never in scope after Q4 failed.
- Combined 4B+PLE weights were not redistributed.

## 5. Conclusion

**Q1 GO. Q2 GO (small). Q3 GO. Q4 NO-GO.**

A 51B n-gram table addressed from tokens can be read by a frozen 4B through a 1.3M adapter. The transferable signal is real, local, and small. It is not unlocked by a fatter interface. It is dominated by ordinary LoRA on the 4B.

Do not build a 4B+PLE product on this evidence. Keep the causal controls: they show that *when the backbone is frozen*, pretrained row content is not interchangeable with noise. That is the result.

## Artifacts

| What | Where |
|---|---|
| Code | this repo |
| Packed data | `$PLE_GRAFT_DATA_DIR` (default `/mnt/llm-cache/ple-graft/data/`) |
| Full-val arrays + CIs | `artifacts/metrics/expanded_val/` |
| Head/token analysis | `artifacts/metrics/ple_where/` |
| Placement | `artifacts/metrics/placement_comparison.json` |
| LoRA decision | `artifacts/metrics/lora_comparison.json` |
| Spec | `~/PLE_Graft_Experiment_Qwen3_5_4B.txt` |
| 35B-A3B follow-up | `MEMO_35B_A3B.md` |
