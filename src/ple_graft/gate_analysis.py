"""Load and slice per-token gate dumps. CPU only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_DIR, load_tokens, n_windows, window_at
from .donor_assets import RECIPIENT_DIR
from .ngram_hash import hash_token_ids

SEQ_LEN = 1024
N_WIN_EXPECTED = 7812
PER_WIN = SEQ_LEN - 1  # 1023
N_POS_EXPECTED = N_WIN_EXPECTED * PER_WIN
_REPO_ROOT = Path(__file__).resolve().parents[2]
DUMP_DIR = _REPO_ROOT / "artifacts" / "metrics" / "ple_where" / "tokens"
MEAN_DELTA_REF = -0.01249250399688918


def _mmap(path: Path, dtype, n: int):
    arr = np.memmap(path, dtype=dtype, mode="r")
    if arr.size != n:
        raise ValueError(f"{path} has {arr.size} values, expected {n}")
    return arr


def load_dumps(dump_dir: Path | None = None, n_win: int = N_WIN_EXPECTED, seq_len: int = SEQ_LEN):
    dump_dir = Path(dump_dir) if dump_dir is not None else DUMP_DIR
    per = n_win * (seq_len - 1)
    gate = _mmap(dump_dir / "gate.bin", np.float32, per)
    nll_ple = _mmap(dump_dir / "nll_ple.bin", np.float32, per)
    nll_b0 = _mmap(dump_dir / "nll_b0.bin", np.float32, per)
    tok = _mmap(dump_dir / "tok.bin", np.int32, per)
    tok_b0 = _mmap(dump_dir / "tok_b0.bin", np.int32, per)
    return {
        "gate": gate,
        "nll_ple": nll_ple,
        "nll_b0": nll_b0,
        "tok": tok,
        "tok_b0": tok_b0,
        "n_win": n_win,
        "seq_len": seq_len,
        "per_win": seq_len - 1,
    }


def labels_from_val(tokens: np.ndarray, window_index: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Target ids stored in tok.bin for one window: window[1:]."""
    return window_at(tokens, window_index, seq_len)[1:].astype(np.int32, copy=False)


def assert_aligned(
    dumps: dict,
    tokens: np.ndarray,
    window_sample: tuple[int, ...] = (0, 1, 100, 7811),
    n_random: int = 10_000,
    seed: int = 0,
) -> dict:
    tok = np.asarray(dumps["tok"])
    tok_b0 = np.asarray(dumps["tok_b0"])
    if tok.size != tok_b0.size:
        raise AssertionError("tok/tok_b0 length mismatch")
    if not np.array_equal(tok, tok_b0):
        n_bad = int(np.count_nonzero(tok != tok_b0))
        raise AssertionError(f"tok != tok_b0 at {n_bad} positions")
    n_win = int(dumps["n_win"])
    seq_len = int(dumps["seq_len"])
    per = int(dumps["per_win"])
    for wi in window_sample:
        if wi < 0 or wi >= n_win:
            continue
        want = labels_from_val(tokens, wi, seq_len)
        got = tok[wi * per : (wi + 1) * per]
        if not np.array_equal(got, want):
            raise AssertionError(f"window {wi} labels do not match val.bin")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_win, size=min(n_random, n_win))
    for wi in np.unique(idx):
        want = labels_from_val(tokens, int(wi), seq_len)
        got = tok[int(wi) * per : (int(wi) + 1) * per]
        if not np.array_equal(got, want):
            raise AssertionError(f"random window {int(wi)} labels do not match val.bin")
    g = np.asarray(dumps["gate"])
    if not np.all(np.isfinite(g)):
        raise AssertionError("non-finite gate")
    if float(g.min()) <= 0 or float(g.max()) >= 1:
        raise AssertionError(f"gate out of (0,1): {float(g.min())} {float(g.max())}")
    delta = np.asarray(dumps["nll_ple"], dtype=np.float64) - np.asarray(dumps["nll_b0"], dtype=np.float64)
    if not np.all(np.isfinite(delta)):
        raise AssertionError("non-finite NLL")
    mean_delta = float(delta.mean())
    if abs(mean_delta - MEAN_DELTA_REF) > 0.001:
        raise AssertionError(f"mean ΔNLL {mean_delta} != {MEAN_DELTA_REF} ± 0.001")
    return {
        "n": int(tok.size),
        "mean_gate": float(g.mean()),
        "mean_delta": mean_delta,
        "gate_min": float(g.min()),
        "gate_max": float(g.max()),
    }


def quintile_table(x: np.ndarray, *ys: np.ndarray) -> list[dict]:
    x = np.asarray(x, dtype=np.float64)
    qs = np.quantile(x, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rows = []
    for i in range(5):
        lo, hi = float(qs[i]), float(qs[i + 1])
        m = (x >= lo) & (x <= hi if i == 4 else x < hi)
        row = {"q": i, "lo": lo, "hi": hi, "n": int(m.sum())}
        for j, y in enumerate(ys):
            yy = np.asarray(y, dtype=np.float64)
            row[f"mean_{j}"] = float(yy[m].mean()) if m.any() else None
        rows.append(row)
    return rows


def spearman(x: np.ndarray, y: np.ndarray, n_sample: int = 2_000_000, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("spearman length mismatch")
    if x.size > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=n_sample, replace=False)
        x, y = x[idx], y[idx]
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def invert_vocab(tokenizer) -> dict[int, str]:
    vocab = tokenizer.get_vocab()
    return {int(i): s for s, i in vocab.items()}


def token_bucket(text: str, tid: int, eos_id: int) -> str:
    if tid == eos_id:
        return "eos"
    if text.startswith("<") and text.endswith(">"):
        return "special"
    if not text:
        return "other"
    if text in ("\n", "\r", "\r\n") or text.isspace():
        return "whitespace"
    if all(ch.isdigit() for ch in text.strip()) and any(ch.isdigit() for ch in text):
        return "digit"
    punct = set('.,;:!?()[]{}"\'`-_/\\@#$%^&*+=~|<>')
    stripped = text.strip()
    if stripped and all(ch in punct for ch in stripped):
        return "punct"
    return "word"


CODE_SNIPPETS = ("def ", "class ", "import ", "#include", "};", "fn ", "func ")


def window_code_frac(decoded_tokens: list[str]) -> float:
    n = 0
    for s in decoded_tokens:
        if any(p in s for p in ("def", "class", "import", "#include")) or s in ("{", "};", "();"):
            n += 1
    return n / max(len(decoded_tokens), 1)


def ngram_rarity_for_windows(
    tokens: np.ndarray,
    n_win: int,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Per dump-position rarity: mean over 16 heads of log1p(count of that row id).

    Lookup at hidden position k (0-based in the window) predicts token k+1.
    Dump slot k uses rows[k].
    """
    from collections import Counter

    per = seq_len - 1
    counts = [Counter() for _ in range(16)]
    for i in range(n_win):
        rows = hash_token_ids(window_at(tokens, i, seq_len))
        for h in range(16):
            u, c = np.unique(rows[:, h], return_counts=True)
            counts[h].update({int(a): int(b) for a, b in zip(u.tolist(), c.tolist())})
    rarity = np.empty(n_win * per, dtype=np.float32)
    for i in range(n_win):
        rows = hash_token_ids(window_at(tokens, i, seq_len))
        acc = np.zeros(per, dtype=np.float64)
        for h in range(16):
            ids = rows[:per, h]
            acc += np.log1p(np.array([counts[h][int(x)] for x in ids], dtype=np.float64))
        rarity[i * per : (i + 1) * per] = (acc / 16.0).astype(np.float32)
    return rarity
