from __future__ import annotations

from pathlib import Path

from ple_graft.donor_assets import HF_DONOR_TOKENIZER, PACK_TOKENIZER, PADDED_VOCAB_SIZE, SMOKE_CORPUS
from ple_graft.tokenizer_compat import compare_tokenizers, load_tokenizer


def _snippets(limit: int = 256) -> list[str]:
    out: list[str] = []
    if SMOKE_CORPUS.is_dir():
        for path in sorted(SMOKE_CORPUS.glob("data/*/*.txt")):
            raw = path.read_text(encoding="utf-8", errors="replace")
            for para in raw.split("\n\n"):
                para = para.strip()
                if para:
                    out.append(para)
                if len(out) >= limit:
                    return out
    out.extend(
        [
            "Hello, world.",
            "def add(a, b):\n    return a + b\n",
            "中文维基百科",
            "SELECT 1;",
            "The n-gram embedding is addressed from raw token ids.",
        ]
    )
    return out[:limit]


def test_donor_and_pack_tokenizers_are_identical():
    report = compare_tokenizers(HF_DONOR_TOKENIZER, PACK_TOKENIZER, _snippets(2000), 2000)
    assert report.equal, report


def test_donor_vocab_covers_padded_ids():
    tok = load_tokenizer(HF_DONOR_TOKENIZER)
    vocab = tok.get_vocab()
    # padded size is the embedding width; tokenizer vocab may equal or be slightly smaller
    assert max(vocab.values()) < PADDED_VOCAB_SIZE
    assert len(vocab) <= PADDED_VOCAB_SIZE


def test_recipient_matches_donor_if_downloaded(recipient_tokenizer_path: Path):
    report = compare_tokenizers(
        HF_DONOR_TOKENIZER,
        recipient_tokenizer_path,
        _snippets(10_000),
        10_000,
    )
    # Direct graft needs identical IDs on the shared vocabulary and identical
    # encodings of real text. Extra donor-only specials are recorded, not remapped.
    assert report.graftable, report
    assert report.n_snippet_mismatches == 0, report
