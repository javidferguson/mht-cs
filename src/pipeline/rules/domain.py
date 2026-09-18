"""Domain configuration - everything condition-specific that is not an entity.

Paired with config/lexicon.yml, this is the whole surface you edit to point the
pipeline at a different drug or disease area.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

DOMAIN_PATH = Path("/app/config/domain.yml")


@dataclass(frozen=True)
class Domain:
    member_label: str
    decoys: dict[str, tuple[str, ...]]
    umbrella_superseded_by: dict[str, frozenset[str]]


@lru_cache(maxsize=1)
def load(path: Path = DOMAIN_PATH) -> Domain:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Domain(
        member_label=raw.get("member_label", "community member"),
        decoys={k.lower(): tuple(v) for k, v in (raw.get("decoys") or {}).items()},
        umbrella_superseded_by={
            k: frozenset(v) for k, v in (raw.get("umbrella_superseded_by") or {}).items()
        },
    )
