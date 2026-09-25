# Grafting Flash-Next n-gram memory onto Qwen3.6-35B-A3B

**Technical memo (addendum to `MEMO.md`)** · 14 September 2026  
**Repo:** `ple-graft` · **Recipient:** `Qwen/Qwen3.6-35B-A3B` (instruct; no public Base) · **Donor table:** Qwen3.8-Flash-Next PLE (same frozen mmap as the 4B study)

## Abstract

The 4B study showed that Flash-Next’s 51B n-gram table is portable into a frozen same-family dense LM (ΔNLL **−0.011**) but is not an extra ingredient once the backbone can adapt (LoRA-only beat PLE+LoRA). The remaining question was whether a **same-family hybrid MoE** would *use* the 2560-d n-gram vector better than the dense 4B.

It does not. On the same 8M FineWeb-Edu val, frozen 35B-A3B + a 1.18M adapter improves NLL by **−0.00685** (95% CI **[−0.00766, −0.00602]**). The gain is causal (shuffled addresses **+0.299**) and smaller than 4B’s −0.011. The pre-registered amplify bar (CI entirely below **−0.025**, or ≥2× the 4B Δ with CIs disjoint) is **not** met. With LoRA r=16 on the first 8 decoder layers, **LoRA-only** reaches **−0.0354**; **PLE+LoRA is strictly worse** (Δ **+0.00500**, CI **[+0.00477, +0.00522]**). **P2 NO-GO.** MoE experts seeing a PLE residual does not amplify the portable n-gram signal, and it does not change the product conclusion.

## 1. Question

| ID | Question |
|---|---|
| Q1 | Tokenizer IDs match Flash-Next / 4B so we can reuse `val.bin`? Graft is identity at step 0 on the 35B? |
| Q2 | Frozen table + tiny 2560→2048 adapter improves 35B NLL? |
| Q3 | Gain from pretrained rows (real ≪ shuffled)? |
| Amplify | Is frozen-MoE Δ **clearly larger** than 4B’s −0.011? Pre-registered: 95% CI on Δ vs B0 entirely below **−0.025**, or ≥2× the 4B Δ with CI excluding the 4B interval. |
| Q4 / P2 | Does LoRA on the 35B unlock extra value from the same frozen table? GO only if PLE+LoRA beats LoRA-only with CI entirely below 0. |

Primary metric: next-token NLL on the **same** packed FineWeb-Edu val as the 4B study (8,000,000 tokens, 7,812 windows × 1024), no chat template. Training used the following 80M of the same stream. Backbone, routed experts, routers, and the PLE table stay frozen. We did **not** copy Flash-Next’s contextualizer.

Do not claim a product 35B+PLE unless P2 GO.

## 2. Recipient vs 4B

| | 4B (done) | 35B-A3B | Flash-Next PLE |
|---|---|---|---|
| Hidden | 2560 | **2048** | concat **2560** = 16×160 |
| Vocab (padded) | 248320 | 248320 | 248320 |
| EOS | 248044 | 248044 | 248044 |
| Layout | 8× (3 GDN + 1 attn) | **10×** (3 GDN + 1 attn) | GDN + QSA + HC |
| FFN | dense | **MoE 256 / top-8 + 1 shared** | MoE |
| Active params | 4B | **3B** | 6B + 51B PLE |
| Checkpoint | Base | **post-trained instruct only** | — |
| Weights | ~8GB BF16 | ~70GB BF16 at `/opt/llm/Qwen3.6-35B-A3B` rev `995ad96` | mmap `ple.bin` |

Two new confounds vs 4B: **width mismatch** (adapter must map 2560→2048) and **instruct, not Base**. Both are documented; neither is mixed with a chat template on FineWeb.

Adapter (identity at step 0):

```
z = RMSNorm(E)                 # 2560
a = W_up(SiLU(W_down(z)))      # 2560→256→2048, W_up = 0
g = 0.5 (frozen)
h' = h + g · a                 # h ∈ R^{2048}
```

Trainable **1,184,256**. Insert after the token-mixer residual, **before MoE FFN**, so experts see the PLE residual. Primary site: 0-based **layer 1** (second GDN). Layer 3 was optional only if frozen Δ sat in the 4B ballpark; it did not, and was not run.

LoRA: r=16 on the first 8 decoder layers, wrapping `nn.Linear` only (attention + **shared** expert). Routed experts and routers are not wrapped. 70 linears, **4,495,488** trainable. Batch 1, seq 1024, text-only load.

## 3. Results

### 3.1 Q1 — Compatibility (GO, with a tokenizer note)

10k snippets encode identically vs Flash-Next. Shared token IDs match. Donor has 7 extra specials (`<tts_pad>`, `<tts_text_bos>`, `<tts_text_bos_single>`, `<tts_text_eod>`, `<|audio_end|>`, `<|audio_pad|>`, `<|audio_start|>`). No remap. Reused the 4B `val.bin` / `train.bin`.

Graft with `enabled=False` and with enabled zero `W_up`: max abs logit diff **0.0** on four sequences (`artifacts/metrics/phase1_disabled_35b.json`). `Qwen3_5MoeDecoderLayer.mlp` can return a tuple; the graft unpacks `hidden_states[0]`.

### 3.2 Q2 / Q3 — Frozen adapter (GO, smaller than 4B)

Full val, B0 = **2.1264** (first-32 **2.1233**, matching train-time before-train 2.1233):

| Condition | NLL | Δ vs B0 | 95% CI | First 32 |
|---|---|---|---|---|
| B0 baseline | 2.1264 | 0 | — | 2.1233 |
| Real PLE, layer 1, seed 0 | **2.1195** | **−0.00685** | **[−0.00766, −0.00602]** | 2.1060 |
| Shuffled addresses | 2.4257 | +0.299 | [+0.297, +0.302] | 2.4134 |

Real lookup is better than baseline on **70.1%** of windows (4B was 82–88%). Shuffled is worse on essentially every window. Causal: pretrained rows, not extra parameters.

32-window P1 curve (same 32 windows as B0 2.1233): **2.1233 → 2.1158 (2M) → 2.1060 (4M) → 2.1159 (6M) → 2.1156 (8M)**. Best-ckpt 4M val_nll 2.1060 is the full-val checkpoint; first-32 of the full-val array is 2.1060.

### 3.3 Amplify — NO

Pre-registered bar: CI(Δ vs B0) entirely **< −0.025**, or at least 2× 4B’s −0.0111 with CIs disjoint.

| Recipient | Frozen Δ vs B0 | 95% CI |
|---|---|---|
| 4B-Base, layer 1, seed 1 | −0.0111 | [−0.0115, −0.0108] |
| 35B-A3B instruct, layer 1, seed 0 | **−0.00685** | **[−0.00766, −0.00602]** |

The 35B interval lies entirely **above** −0.025 and entirely **above** the 4B interval. MoE did **not** amplify. The portable signal is small on both recipients; here it is smaller.

### 3.4 Q4 / P2 — LoRA (NO-GO)

Same recipe as 4B: LoRA r=16, first 8 layers, 8M tokens, lr 2e-4. PLE+LoRA freezes the P1 adapter and table.

| Run | NLL | Δ vs B0 | First 32 |
|---|---|---|---|
| B0 | 2.1264 | 0 | 2.1233 |
| PLE-only (P1) | 2.1195 | −0.00685 | 2.1060 |
| PLE+LoRA | 2.0959 | −0.0304 | 2.0887 |
| **LoRA-only (no PLE)** | **2.0909** | **−0.0354** | **2.0836** |

PLE+LoRA − LoRA-only = **+0.00500**, 95% CI **[+0.00477, +0.00522]**. LoRA-only is strictly better. The 32-window echo agrees (2.0887 vs 2.0836). Same sign as 4B (+0.00165), larger gap.

**P2 verdict: NO-GO.** MoE does not make the frozen table an extra ingredient.

## 4. Limitations

- Instruct checkpoint, not Base. FineWeb NLL without a chat template; the instruct confound is not removed.
- Width 2048 vs PLE concat 2560; only the bottleneck adapter was used (dense/per-head require matching width).
- Layer 3 was not run. The plan made that optional after a 4B-sized frozen Δ; we observed a *smaller* Δ.
- Val is FineWeb-Edu LM NLL, not MMLU/HumanEval.
- 8M train tokens; P1 32-window best at 4M, then slightly worse.
- Routed experts never trained. That was a pre-registered non-goal, not a failed attempt.
- Combined 35B+PLE weights were not redistributed.

## 5. Conclusion

**Q1 GO-WITH-NOTE. Q2 GO (small). Q3 GO. Amplify NO. P2 NO-GO.**

A 51B n-gram table addressed from tokens can be read by a frozen 35B-A3B MoE through a 1.18M 2560→2048 adapter. The transferable signal is real and smaller than on the dense 4B. Putting that residual in front of 256 experts does not turn −0.007 into −0.025. Ordinary LoRA on the first 8 layers (shared expert + attention only) beats PLE+LoRA by a clean +0.005.

Do not build a 35B+PLE product on this evidence. The two-recipient result is: address-aligned PLE is a small frozen-backbone curiosity in this family, not a module that MoE “knows how to use.”

## Artifacts

| What | Where |
|---|---|
| Code | this repo |
| Recipient | `$PLE_GRAFT_MOE_DIR` (HF `Qwen/Qwen3.6-35B-A3B` rev `995ad96`) |
| Config (pre-registered bars) | `configs/moe_35b_a3b.yaml` |
| Tokenizer | `artifacts/manifests/tokenizer_compat_35b_a3b.json` |
| Step-0 identity | `artifacts/metrics/phase1_disabled_35b.json` |
| Full-val arrays + CIs | `artifacts/metrics/expanded_val_35b/` |
| P1 vs B0 / C2 | `artifacts/metrics/expanded_val_35b/p1_c2_vs_b0.json` |
| P2 decision | `artifacts/metrics/expanded_val_35b/lora_comparison.json` |
| Checkpoints | `artifacts/checkpoints/adapter_35b_*` |
| 4B parent memo | `MEMO.md` |
