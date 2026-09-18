"""Stage 5: Postgres -> Parquet + CSV, one file per grain, plus a manifest.

The manifest records model, prompt version, lexicon digest and row counts, so any
figure in the deck can be traced back to the run that produced it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.rules.lexicon import load as load_lexicon
from pipeline.core.prompts import PROMPT_VERSION

GRAINS = {
    "documents": "SELECT * FROM documents ORDER BY doc_id",
    "mentions": "SELECT * FROM mentions ORDER BY mention_id",
    "switch_events": "SELECT * FROM switch_events ORDER BY switch_id",
    "demographic_claims": "SELECT * FROM demographic_claims ORDER BY claim_id",
    "user_profiles": "SELECT * FROM user_profiles ORDER BY author_id",
    "treatment_summary": "SELECT * FROM treatment_summary ORDER BY n_mentions DESC",
}

# JSONB columns must be serialised before Parquet will accept them.
JSON_COLS = {
    "current_treatments", "past_treatments", "considering_treatments",
    "rejected_treatments", "symptoms", "stance_counts", "sentiment_counts",
}


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def run(cfg: Config, run_id: str | None = None) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = cfg.out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    files: list[str] = []

    with connect(cfg) as conn:
        for name, sql in GRAINS.items():
            cur = conn.execute(sql)
            cols = [c.name for c in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
            for col in df.columns:
                if col in JSON_COLS:
                    df[col] = df[col].map(lambda v: json.dumps(v) if v is not None else None)
            df.to_parquet(out / f"{name}.parquet", index=False)
            df.to_csv(out / f"{name}.csv", index=False)
            counts[name] = len(df)
            files += [f"{name}.parquet", f"{name}.csv"]

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg.extraction_model,
        "prompt_version": PROMPT_VERSION,
        "lexicon_digest": load_lexicon().digest,
        "git_rev": _git_rev(),
        "row_counts": counts,
        "files": sorted(files),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # No "latest" pointer. Run directories sort lexicographically by timestamp,
    # so consumers use max(out.glob("2*Z")) - one line, no symlink that behaves
    # differently on Linux, and nothing to go stale.
    return manifest
