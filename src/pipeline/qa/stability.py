"""Run-to-run stability: the number that justifies the hybrid design.

Extracts the same documents N times and measures Jaccard agreement over
canonical ids. Reported at two levels:
  * canonical - what the pipeline actually emits, after the lexicon
  * surface   - what an LLM-only pipeline would aggregate on
The gap between them is what deterministic canonicalisation buys.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.rules.lexicon import load as load_lexicon
from pipeline.core.models import extraction_schema
from pipeline.core.ollama import OllamaClient
from pipeline.core.prompts import render_document, system_prompt

SAMPLE = """
SELECT d.doc_id, d.doc_type, d.author_id, d.title, d.body, p.title
  FROM documents d
  LEFT JOIN documents p ON p.doc_id = d.parent_id
  JOIN extractions e ON e.doc_id = d.doc_id AND e.parsed_ok
 ORDER BY md5(d.doc_id) LIMIT %s
"""


@dataclass
class StabilityResult:
    n_docs: int
    runs: int
    canonical: list[float]
    surface: list[float]

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    @property
    def canonical_mean(self) -> float:
        return self._mean(self.canonical)

    @property
    def surface_mean(self) -> float:
        return self._mean(self.surface)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


async def run(cfg: Config, n_docs: int = 50, runs: int = 3) -> StabilityResult:
    lex = load_lexicon()
    with connect(cfg) as conn:
        rows = conn.execute(SAMPLE, (n_docs,)).fetchall()

    schema = extraction_schema()
    sem = asyncio.Semaphore(cfg.ollama_num_parallel)

    async def one(row) -> tuple[set, set]:
        doc_id, dt, au, ti, bo, pti = row
        prompt = render_document(doc_type=dt, author_id=au, title=ti, body=bo,
                                 parent_title=pti)
        async with sem:
            g = await client.generate(prompt, schema=schema, system=system_prompt())
        d = g.parsed
        surfaces, canon = set(), set()
        for m in d.get("treatments", []) + d.get("symptoms", []):
            s = m["surface"].strip().lower()
            surfaces.add(s)
            hit = lex.match(s)
            canon.add(hit.id if hit else f"OOV:{s}")
        return canon, surfaces

    per_run: list[list[tuple[set, set]]] = []
    async with OllamaClient(cfg) as client:
        for _ in range(runs):
            per_run.append(list(await asyncio.gather(*(one(r) for r in rows))))

    canon_j, surf_j = [], []
    for i in range(len(rows)):
        for a in range(runs):
            for b in range(a + 1, runs):
                canon_j.append(_jaccard(per_run[a][i][0], per_run[b][i][0]))
                surf_j.append(_jaccard(per_run[a][i][1], per_run[b][i][1]))

    return StabilityResult(n_docs=len(rows), runs=runs, canonical=canon_j, surface=surf_j)
