"""Tokenizer identity checks between donor Flash-Next and recipient Qwen3.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer

from .donor_assets import HF_DONOR_TOKENIZER, PADDED_VOCAB_SIZE, RECIPIENT_DIR


@dataclass
class TokenizerReport:
    equal: bool
    graftable: bool
    vocab_size_a: int
    vocab_size_b: int
    n_id_mismatches: int
    n_snippet_mismatches: int
    n_snippets: int
    notes: list[str]


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def vocab_id_mismatches(a: Tokenizer, b: Tokenizer, limit: int = 32) -> list[tuple[str, int, int]]:
    va = a.get_vocab()
    vb = b.get_vocab()
    mismatches: list[tuple[str, int, int]] = []
    keys = set(va) | set(vb)
    for tok in keys:
        ia = va.get(tok)
        ib = vb.get(tok)
        if ia != ib:
            mismatches.append((tok, -1 if ia is None else ia, -1 if ib is None else ib))
            if len(mismatches) >= limit:
                break
    return mismatches


def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def compare_tokenizers(
    path_a: Path,
    path_b: Path,
    snippets: Iterable[str],
    snippet_limit: int = 10_000,
) -> TokenizerReport:
    notes: list[str] = []
    a = load_tokenizer(path_a)
    b = load_tokenizer(path_b)
    va = a.get_vocab()
    vb = b.get_vocab()
    if len(va) != len(vb):
        notes.append(f"vocab sizes differ: {len(va)} vs {len(vb)}")
    if len(va) != PADDED_VOCAB_SIZE and len(va) > PADDED_VOCAB_SIZE:
        notes.append(f"tokenizer A vocab {len(va)} exceeds padded {PADDED_VOCAB_SIZE}")
    shared = set(va) & set(vb)
    only_a = sorted(set(va) - set(vb))
    only_b = sorted(set(vb) - set(va))
    shared_id_mismatches = [(t, va[t], vb[t]) for t in shared if va[t] != vb[t]]
    id_mismatches = vocab_id_mismatches(a, b, limit=64)
    n_snip = 0
    n_snip_mm = 0
    for text in snippets:
        if n_snip >= snippet_limit:
            break
        n_snip += 1
        if encode_ids(a, text) != encode_ids(b, text):
            n_snip_mm += 1
    if only_a:
        notes.append(f"tokens only in A ({len(only_a)}): {only_a[:16]!r}")
    if only_b:
        notes.append(f"tokens only in B ({len(only_b)}): {only_b[:16]!r}")
    # Direct-graft GO: every shared token has the same id, and real text encodes identically.
    # Extra specials on one side are recorded; they are not remapped.
    graftable = not shared_id_mismatches and n_snip_mm == 0
    equal = graftable and not only_a and not only_b
    if shared_id_mismatches:
        notes.append(f"shared-id mismatches: {shared_id_mismatches[:8]!r}")
    notes.append(f"graftable={graftable} extra_a={len(only_a)} extra_b={len(only_b)}")
    return TokenizerReport(
        equal=equal,
        graftable=graftable,
        vocab_size_a=len(va),
        vocab_size_b=len(vb),
        n_id_mismatches=len(id_mismatches),
        n_snippet_mismatches=n_snip_mm,
        n_snippets=n_snip,
        notes=notes,
    )


def recipient_tokenizer_path() -> Path:
    p = RECIPIENT_DIR / "tokenizer.json"
    return p


def donor_tokenizer_path() -> Path:
    return HF_DONOR_TOKENIZER
