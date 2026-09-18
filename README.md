# Community treatments & demographics pipeline

Turns forum posts and replies (JSONL) into a per-member dataset of **treatments,
symptoms, switch events and demographics**.

A local Ollama model does schema-constrained extraction, one call per document. A
deterministic lexicon does canonicalisation. The split is the point: **the model
reads language, the lexicon decides identity.** `Vagifem` and `Estring` are separate
entities; `vaginal estrogen` with no brand named is a third. Because that decision
lives in a YAML file rather than in the model, aggregates are inspectable, and
retuning them costs no model calls.

The pipeline is domain-agnostic — the extraction prompt uses `DRUG-A`/`DRUG-B`
placeholders and never names a real product. Point it at a different drug or
condition by editing `config/lexicon.yml` and `config/domain.yml`.

## Architecture

```
worker (py3.13) ──┐
notebook (py3.13) ├── postgres:17 + pgvector      ./data:/data (ro) · ./out:/out
                  └── OLLAMA_BASE_URL ──► host.docker.internal:11434 (native, Metal)
```

Docker on macOS cannot pass through the Apple GPU, so Ollama runs **natively on the
host** and containers reach it through Docker Desktop's proxy. The job queue is a
Postgres table claimed with `FOR UPDATE SKIP LOCKED` — no Redis, no Celery. Postgres
and JupyterLab publish to `127.0.0.1` only.

```
src/pipeline/
  core/     config · db · ollama · models · prompts · schema.sql
  rules/    lexicon · domain · negation · context · cohort   <- deterministic, no model
  stages/   ingest · extract · normalize · profile · export
  qa/       bench · ablate · stability · gold · score
```

## Setup (macOS)

```bash
brew install ollama
```

`OLLAMA_NUM_PARALLEL` must be set **on the server**. The `.env` value is only a
client-side semaphore; without this the server runs `-np 1` and serialises every
request:

```bash
OLLAMA_NUM_PARALLEL=8 ollama serve
```

Leave that running. The default `127.0.0.1` bind is correct — containers still reach
it, and nothing is exposed to your network. Then:

```bash
make setup            # build image, start postgres
make pull-model       # ~4.9 GB into ~/.ollama
make doctor           # postgres / ollama / model / input / disk / concurrency
make hello            # JSON round-trip to Ollama from inside the container
```

Put the input JSONL in `data/` (or point `DATA_DIR` elsewhere). Expected fields:
posts carry `id, type, author_id, title, body, created_at`; replies carry the same
plus `parent_id`.

## Running it

```bash
make ingest                  # JSONL -> documents + job queue      (~3s, idempotent)
make bench                   # measure throughput, print a real ETA
make extract LIMIT=10        # small slice to hand-read first
make extract                 # full corpus (resumable, content-hash cached)
make normalize               # canonicalise + guards               (no model calls)
make profile                 # user_profiles + treatment_summary + narratives
make export                  # parquet + csv + manifest -> out/<run_id>/
make evaluate                # named-entity recall, negative controls, coverage
make test                    # rules-layer suite (no database, no model)
make lab                     # JupyterLab -> http://localhost:8888
```

`make all` runs ingest → extract → normalize → profile → export. `make help` lists
every target.

**Reference run:** 9,991 documents, 5h45m at 0.48 docs/s on an M5 Max with
`llama3.1:8b`, 8 parallel slots, `num_ctx=4096`. 100% parse success.

## Iterating

Editing `config/lexicon.yml` is free — `make normalize` rebuilds every downstream
table from stored extractions with **zero model calls**. Use the out-of-vocabulary
report it prints as the backlog.

Editing `config/prompts/extract.md` is not free. Bump `PROMPT_VERSION` in
`src/pipeline/core/prompts.py` (or set it in the environment); it is part of the
extraction cache key, so a new value forces a full re-extract. Every prior version
stays in `extractions`, so any earlier run is recoverable by setting
`PROMPT_VERSION` back and re-running `normalize profile`.

## Output

Each run writes a timestamped directory under `out/` with parquet + csv per grain
and a `manifest.json` recording model, prompt version, lexicon hash and row counts.

| File | Grain | |
|---|---|---|
| `documents.parquet` | one row per source document | ingested input |
| `mentions.parquet` | one row per extracted entity | **the workhorse** |
| `switch_events.parquet` | one row per switch | directional: from → to, reason |
| `demographic_claims.parquet` | one row per claim | includes rejected claims and why |
| `user_profiles.parquet` | one row per member | **the data product** |
| `treatment_summary.parquet` | one row per entity | members, mentions, stance breakdown |

`surface` is the verbatim string the member wrote; `canonical_id` is what it
resolves to; `context` is located deterministically in the source text so it always
shows the term in use (98.5% coverage). Only `experienced_by='self'` counts toward
a member's own regimen.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data model, the hybrid
split, and what the evaluation does and does not prove.
