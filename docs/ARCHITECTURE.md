# Architecture

How the pipeline is put together and why, including the measurements that settled
the debatable decisions.

## The hybrid split

The model and the lexicon do different jobs, and keeping them separate is what
makes the output auditable.

| | Job | Property |
|---|---|---|
| **LLM** (`stages/extract`) | find mentions, read stance, attribute them | handles language; needs no vocabulary maintenance |
| **Lexicon** (`rules/`) | decide identity, guard demographics, locate evidence | deterministic; re-runs free; inspectable in a YAML file |

The model emits the **surface** — the member's own words — and never canonicalises.
Asking it to do both was tried and rejected: across runs it returned `"Climara"` and
`"Climara patch"` as different answers, so the same product split into different
rows non-deterministically.

### Why `canonical_id` exists

People do not write a product name the same way twice. In the reference corpus:

```
climara                  1,032 mentions written  19 different ways
micronized progesterone    448 mentions written  71 different ways
```

Grouped by raw surface instead, Climara splits across 19 table rows — the largest
holding 576 of its 1,032 mentions. It would be **understated by 44%** and rank #5
instead of #3 in "most discussed", sitting below generic phrases that are partly the
same product. It is also what makes switch edges possible at all: `from → to` needs
stable endpoints.

Cost: 1,963 distinct surfaces map to 98 entities, and ~6% of treatment mentions map
to nothing. Those keep their surface and appear in the OOV report rather than being
dropped.

## The four grains

```
  documents ──┐  LLM: ONE call per document. Independent -> parallel,
              │  resumable, retryable, content-hash cacheable.
              ▼  extract
  mentions  ──┐  tidy/long: one row per extracted entity
              ▼  normalize  <- lexicon, deterministic, NO model calls
     group by (author_id, canonical_id)
       ┌──────────┼───────────┐
       ▼          ▼           ▼
   profiles   switch edges  treatment summary
```

Extraction is **per document, not per member**: each document is independent, so it
parallelises, caches on content hash, and retries alone. Batching a member's ~100
documents would mean a 30k-token prompt where one malformed document corrupts that
whole member.

`parent_id` links a reply to its post but is used **only** for thread context and FK
integrity. Every aggregate partitions by `(author_id, canonical_id)` — there is no
thread-level aggregation and no coreference resolution. The accurate name for the
rollup is *per-author entity rollup*.

## Why replies do not carry their parent's body

Replies originally included the parent's title plus 600 characters of its body so
bare references ("the patch", "it") could resolve. The prompt said not to extract
from it. The model did anyway.

A 300-reply A/B with the decision rule fixed in advance:

| | with parent body | subject only |
|---|---|---|
| self-attributed, name **absent** from the reply (contamination) | 39 | **2** (−95%) |
| self-attributed, name **present** (real signal) | 84 | **81** (−4%) |

The parent body bought 4% of signal for 95% of the false attribution, so it went.
Attribution corrections downstream fell from 4,241 to 588 — the errors stopped being
generated rather than being corrected. The cost is that an unresolvable "it" stays
`patch_unspecified` instead of being guessed.

## Named entities vs described entities

Entities split into ones you **name** and ones you **describe**. There is no
paraphrase for "Climara"; "my hormones" names nothing. Only the first kind can be
checked against the text, so `name_must_appear: true` marks them and two things key
off it:

- **a data-quality guard** — a named entity whose name is absent from the document is
  flagged `supported=false` and excluded from aggregates (kept for audit)
- **the recall gold standard** — their presence is objectively checkable, so recall
  needs no annotator

"Named" is about lexical distinctiveness, not trademarks: `Climara` is a brand,
`spironolactone` is a generic molecule, `Midi` is a company, and all three behave
identically here.

## What the evaluation proves — and what it does not

`make evaluate` is scoped to a single `prompt_version`; the `mentions` table records
which run built it, and the coverage block warns when those differ.

| Measured deterministically | Not measured |
|---|---|
| named-entity recall | **stance accuracy** |
| entities emitted with no supporting text | **`experienced_by` accuracy** |
| negation-guard behaviour, both directions | symptom recall |
| canonicalisation stability run-to-run | sentiment accuracy |

Recall is a real gold standard — a distinctive name is either in the document or it
is not, no judgement involved. It is a **proxy for extraction quality, not a score of
it**: high recall means the model reliably spots named products, and says nothing
about whether it read the stance correctly. Stance and sentiment need a labelled
sample, which does not exist yet; the attribution figure in the report is explicitly
labelled a tendency rather than an accuracy.

## Determinism

Every model call runs at `temperature=0`, so decoding is greedy — the same document
extracted three times returns byte-identical output. Reproducibility is therefore not
the open question; **validity** is.

Two generation guards exist because the corpus found them: `num_predict` bounds
output (two documents drove the model into a repetition loop that ran until the
timeout), and a parse failure retries once with `temperature=0.4` and
`repeat_penalty=1.3`, which clears it.

`num_ctx=4096` is not a tuning preference. The model's default 131,072-token context
reserves ~22 GB of KV cache per slot; at 8 parallel slots that exceeds 64 GB of
unified memory. Capping it drops the footprint to 5.3 GB and is what makes
parallelism possible at all.

## Testing

`make test` covers `rules/` — the deterministic layer — with no database and no
model, in well under a second.
