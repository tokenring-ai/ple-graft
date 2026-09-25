from __future__ import annotations

from pathlib import Path

import pytest

from ple_graft.donor_assets import HF_DONOR_TOKENIZER, PLE_BIN, RECIPIENT_DIR, SMOKE_CORPUS


@pytest.fixture(scope="session")
def donor_tokenizer_path() -> Path:
    if not HF_DONOR_TOKENIZER.is_file():
        pytest.skip(f"missing donor tokenizer {HF_DONOR_TOKENIZER}")
    return HF_DONOR_TOKENIZER


@pytest.fixture(scope="session")
def recipient_tokenizer_path() -> Path:
    p = RECIPIENT_DIR / "tokenizer.json"
    cfg = RECIPIENT_DIR / "config.json"
    if not p.is_file() or not cfg.is_file():
        pytest.skip(f"recipient checkpoint incomplete at {RECIPIENT_DIR}")
    return p


@pytest.fixture(scope="session")
def ple_bin() -> Path:
    if not PLE_BIN.is_file():
        pytest.skip(f"missing {PLE_BIN}")
    return PLE_BIN


@pytest.fixture(scope="session")
def smoke_corpus() -> Path:
    if not SMOKE_CORPUS.is_dir():
        pytest.skip(f"missing {SMOKE_CORPUS}")
    return SMOKE_CORPUS
