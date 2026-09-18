"""Fixtures for the rules-layer suite.

These tests exercise pure functions only - no Postgres, no Ollama. They run inside
the worker container because the modules' default config paths point at /app.
Where a test needs its own vocabulary it passes a fixture path explicitly.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def lex():
    """A small lexicon. load() is lru_cached, so the cache must be cleared or the
    production lexicon leaks in from whichever test ran first."""
    from pipeline.rules import lexicon

    lexicon.load.cache_clear()
    out = lexicon.load(FIXTURES / "lexicon.yml")
    yield out
    lexicon.load.cache_clear()


@pytest.fixture
def domain(monkeypatch):
    from pipeline.rules import domain as dom

    dom.load.cache_clear()
    monkeypatch.setattr(dom, "DOMAIN_PATH", FIXTURES / "domain.yml")
    yield dom.load(FIXTURES / "domain.yml")
    dom.load.cache_clear()
