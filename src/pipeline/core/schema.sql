-- Phase 1 schema: documents + the job queue.
-- The queue lives in Postgres (claimed with FOR UPDATE SKIP LOCKED) rather than
-- Redis/Celery, which keeps the stack at three services.

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    doc_type     TEXT        NOT NULL CHECK (doc_type IN ('post', 'reply')),
    author_id    TEXT        NOT NULL,
    -- Self-referencing FK: a reply whose parent is missing cannot be inserted, so
    -- "0 orphan replies" is enforced by the database rather than asserted by a test.
    parent_id    TEXT        REFERENCES documents (doc_id),
    title        TEXT,
    body         TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    content_hash TEXT        NOT NULL,
    source_file  TEXT        NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((doc_type = 'post' AND parent_id IS NULL)
        OR (doc_type = 'reply' AND parent_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS documents_author_idx ON documents (author_id);
CREATE INDEX IF NOT EXISTS documents_parent_idx ON documents (parent_id);
CREATE INDEX IF NOT EXISTS documents_type_idx   ON documents (doc_type);

CREATE TABLE IF NOT EXISTS jobs (
    doc_id     TEXT        NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    stage      TEXT        NOT NULL,
    status     TEXT        NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending', 'running', 'done', 'error')),
    attempts   INT         NOT NULL DEFAULT 0,
    locked_at  TIMESTAMPTZ,
    locked_by  TEXT,
    error      TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, stage)
);

-- Supports the SKIP LOCKED claim query.
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs (stage, status);

-- Phase 2: raw LLM output, kept for audit and replay.
-- The primary key IS the cache key: re-running only calls the model for documents
-- whose text, prompt version or model actually changed.
CREATE TABLE IF NOT EXISTS extractions (
    doc_id            TEXT        NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content_hash      TEXT        NOT NULL,
    prompt_version    TEXT        NOT NULL,
    model             TEXT        NOT NULL,
    raw_json          JSONB,
    parsed_ok         BOOLEAN     NOT NULL,
    error             TEXT,
    prompt_tokens     INT,
    completion_tokens INT,
    latency_ms        INT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, content_hash, prompt_version, model)
);

CREATE INDEX IF NOT EXISTS extractions_ok_idx ON extractions (parsed_ok);

-- Phase 3: normalised output. Rebuilt deterministically from extractions, so it
-- can be dropped and regenerated with zero LLM cost while the lexicon is tuned.
CREATE TABLE IF NOT EXISTS mentions (
    mention_id     BIGSERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    author_id      TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('treatment', 'symptom')),
    surface        TEXT NOT NULL,
    canonical_id   TEXT,                -- NULL = out-of-vocabulary, see the OOV report
    canonical_class TEXT,
    stance         TEXT,                -- treatments only
    status         TEXT,                -- symptoms only
    experienced_by TEXT NOT NULL,
    dose           TEXT,
    duration       TEXT,
    sentiment      TEXT,
    evidence       TEXT NOT NULL,
    lexicon_hash   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS mentions_author_idx    ON mentions (author_id);
CREATE INDEX IF NOT EXISTS mentions_canonical_idx ON mentions (canonical_id);
CREATE INDEX IF NOT EXISTS mentions_kind_idx      ON mentions (kind);
CREATE INDEX IF NOT EXISTS mentions_doc_idx       ON mentions (doc_id);

CREATE TABLE IF NOT EXISTS switch_events (
    switch_id      BIGSERIAL PRIMARY KEY,
    doc_id         TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    author_id      TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    from_surface   TEXT NOT NULL,
    to_surface     TEXT NOT NULL,
    from_canonical TEXT,
    to_canonical   TEXT,
    reason         TEXT,
    experienced_by TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    lexicon_hash   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS switch_from_idx ON switch_events (from_canonical);
CREATE INDEX IF NOT EXISTS switch_to_idx   ON switch_events (to_canonical);

-- Per-document demographic claims, after the negation guard. Rolled up per author
-- in Phase 4.
CREATE TABLE IF NOT EXISTS demographic_claims (
    claim_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    author_id   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    rejected    BOOLEAN NOT NULL DEFAULT FALSE,
    reject_note TEXT
);

CREATE INDEX IF NOT EXISTS demo_author_idx ON demographic_claims (author_id, field);

-- Deterministic context: where the surface actually sits in the source document.
-- Added because the model's `evidence` omitted the surface in ~63 percent of
-- treatment mentions, so it could not show how the term was used.
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS char_start INT;
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS char_end INT;
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS context TEXT;
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS surface_found BOOLEAN;

ALTER TABLE switch_events ADD COLUMN IF NOT EXISTS from_context TEXT;
ALTER TABLE switch_events ADD COLUMN IF NOT EXISTS to_context TEXT;
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS context_source TEXT;

-- Phase 4: the data product. Rebuilt deterministically from mentions; the
-- narrative column is the only part that costs model calls (one per author).
CREATE TABLE IF NOT EXISTS user_profiles (
    author_id       TEXT PRIMARY KEY,
    n_docs          INT NOT NULL,
    n_posts         INT NOT NULL,
    n_replies       INT NOT NULL,
    first_seen      TIMESTAMPTZ,
    last_seen       TIMESTAMPTZ,
    age             INT,
    city            TEXT,
    region          TEXT,
    country         TEXT,
    profession      TEXT,
    race_ethnicity  TEXT,
    has_partner     BOOLEAN,
    caregiving      TEXT,
    prescriber_type TEXT,
    current_treatments   JSONB NOT NULL DEFAULT '[]',
    past_treatments      JSONB NOT NULL DEFAULT '[]',
    considering_treatments JSONB NOT NULL DEFAULT '[]',
    rejected_treatments  JSONB NOT NULL DEFAULT '[]',
    symptoms             JSONB NOT NULL DEFAULT '[]',
    n_treatment_mentions INT NOT NULL DEFAULT 0,
    n_symptom_mentions   INT NOT NULL DEFAULT 0,
    n_switches           INT NOT NULL DEFAULT 0,
    narrative       TEXT,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS treatment_summary (
    canonical_id    TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    entity_class    TEXT,
    manufacturer    TEXT,
    n_users_self    INT NOT NULL DEFAULT 0,
    n_mentions      INT NOT NULL DEFAULT 0,
    n_mentions_self INT NOT NULL DEFAULT 0,
    stance_counts   JSONB NOT NULL DEFAULT '{}',
    sentiment_counts JSONB NOT NULL DEFAULT '{}',
    n_switch_from   INT NOT NULL DEFAULT 0,
    n_switch_to     INT NOT NULL DEFAULT 0,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- FALSE = a NAMED entity whose name does not appear in the text. Kept for audit, excluded
-- from every aggregate.
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS supported BOOLEAN NOT NULL DEFAULT TRUE;

-- `tolerability` was tried here and removed: as an LLM field it was only ~47%
-- grounded in the source text, and its commonest label (site_irritation) fired on
-- drug side effects rather than product handling. Adhesion is already carried by
-- switch_events.reason (28 adhesion-driven switches) and is a one-line query over
-- documents.body, so nothing in the pipeline needs to compute it.
ALTER TABLE mentions DROP COLUMN IF EXISTS tolerability;

-- Provenance: which extraction run `normalize` built these rows from. Without it
-- nothing downstream can tell whether the mentions table matches the run being
-- scored, and `evaluate --prompt-version v5` silently printed v7's coverage.
ALTER TABLE mentions ADD COLUMN IF NOT EXISTS prompt_version TEXT;
