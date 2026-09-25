# Artifacts in git

Tracked:

- `golden/` — hash/row identity goldens
- `manifests/` — tokenizer compat, pack constants, pinned revisions
- `metrics/**/*.json`, `*.npy`, `*.md` — full-val NLL arrays, paired CIs, train summaries, gate analysis

Not tracked (see `.gitignore`):

- `checkpoints/` — adapter and LoRA `.pt` (a few MB to ~18 MB each; local only)
- `metrics/**/*.jsonl` and `*.log` — per-step train traces
- `metrics/ple_where/tokens/` — ~150 MB per-token gate/NLL dumps
- Packed FineWeb `train.bin` / `val.bin` (`PLE_GRAFT_DATA_DIR`)
- `ple.bin` and recipient weights
