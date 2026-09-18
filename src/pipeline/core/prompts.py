"""Prompt loading and document rendering.

PROMPT_VERSION is part of the extraction cache key, so bumping it invalidates
cached rows and forces a clean re-extract. Bump it whenever extract.md changes
in a way that should change output.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Env-overridable so any earlier version is recoverable without a code edit:
# PROMPT_VERSION=v5 plus `make normalize profile` rebuilds that data product with
# zero model calls.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v7")

_PROMPT_PATH = Path("/app/config/prompts/extract.md")


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def render_document(
    *,
    doc_type: str,
    author_id: str,
    title: str | None,
    body: str,
    parent_title: str | None = None,
) -> str:
    """Render one document, with parent context for replies.

    Replies carry their thread's subject line and nothing else from the parent.
    """
    parts: list[str] = []
    if doc_type == "reply" and parent_title:
        # Subject line only. Sending the parent's BODY was measured (300-reply
        # A/B, v5 vs v6-noparent) to buy a 4% gain in real signal while carrying
        # 95% of the false self-attribution: mentions of a drug named only in the
        # post being answered were credited to the replier. The subject gives the
        # thread topic without transplanting anyone else's treatment names.
        parts.append(
            "=== THREAD SUBJECT (context only - do NOT extract from this) ===\n"
            f"{parent_title}\n"
        )
    parts.append(f"=== DOCUMENT (written by {author_id}) ===")
    if title:
        parts.append(title)
    parts.append(body)
    return "\n".join(parts)
