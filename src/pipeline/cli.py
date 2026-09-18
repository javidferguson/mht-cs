"""Pipeline CLI. Phase 0 ships `doctor` and `hello`; later phases add
ingest / extract / normalize / profile / export / bench.
"""

from __future__ import annotations

import asyncio
import shutil
import sys

import typer
from rich.console import Console
from rich.table import Table

from pipeline.core.config import Config
from pipeline.core.ollama import OllamaClient

app = typer.Typer(add_completion=False, help="Patient-community treatment/symptom extraction pipeline.")
console = Console()

OK, BAD, WARN = "[green]OK[/green]", "[red]FAIL[/red]", "[yellow]WARN[/yellow]"


def _fmt_gb(n: float) -> str:
    return f"{n / 1_000_000_000:.1f} GB"


@app.command()
def doctor() -> None:
    """Verify Postgres, Ollama, the model, input data and disk before any real run."""
    cfg = Config.from_env()
    rows: list[tuple[str, str, str]] = []
    failed = False

    # --- Postgres -----------------------------------------------------------
    try:
        from pipeline.core.db import server_version

        v = server_version(cfg).split(",")[0]
        rows.append(("postgres", OK, f"{cfg.pg_dsn_redacted} · {v}"))
    except Exception as e:  # noqa: BLE001
        failed = True
        rows.append(("postgres", BAD, f"{cfg.pg_dsn_redacted} · {type(e).__name__}: {e}"))

    # --- Ollama reachability + model ---------------------------------------
    async def _check_ollama() -> tuple[str, str, str, str]:
        async with OllamaClient(cfg) as c:
            models = await c.list_models()
            names = [m.get("name", "?") for m in models]
            wanted = cfg.extraction_model
            wanted_full = wanted if ":" in wanted else f"{wanted}:latest"
            hit = next((m for m in models if m.get("name") == wanted_full), None)
            if hit:
                model_row = (OK, f"{wanted_full} · {_fmt_gb(hit.get('size', 0))}")
            elif names:
                model_row = (BAD, f"{wanted_full} NOT pulled. Have: {', '.join(names)}. Run: make pull-model")
            else:
                model_row = (BAD, f"{wanted_full} NOT pulled (no models at all). Run: make pull-model")
            return (OK, f"{cfg.ollama_base_url} · {len(models)} model(s)", *model_row)

    try:
        reach_s, reach_d, model_s, model_d = asyncio.run(_check_ollama())
        rows.append(("ollama", reach_s, reach_d))
        rows.append(("model", model_s, model_d))
        if model_s == BAD:
            failed = True
    except Exception as e:  # noqa: BLE001
        failed = True
        rows.append(
            (
                "ollama",
                BAD,
                f"{cfg.ollama_base_url} unreachable · {type(e).__name__}: {e}\n"
                f"  On Mac: is `ollama serve` running on the host?",
            )
        )
        rows.append(("model", WARN, "skipped - Ollama unreachable"))

    # --- Input data ---------------------------------------------------------
    if cfg.input_dir.is_dir():
        jsonl = sorted(p.name for p in cfg.input_dir.glob("*.jsonl"))
        if jsonl:
            rows.append(("input", OK, f"{cfg.input_dir} · {', '.join(jsonl)}"))
        else:
            failed = True
            rows.append(("input", BAD, f"{cfg.input_dir} has no *.jsonl (check DATA_DIR)"))
    else:
        failed = True
        rows.append(("input", BAD, f"{cfg.input_dir} not mounted (check DATA_DIR)"))

    # --- Disk ---------------------------------------------------------------
    free = shutil.disk_usage("/").free
    rows.append(("disk", OK if free > 8e9 else WARN, f"{_fmt_gb(free)} free in container"))

    # --- Concurrency settings ----------------------------------------------
    rows.append(
        (
            "concurrency",
            OK,
            f"num_ctx={cfg.ollama_num_ctx} x {cfg.ollama_num_parallel} parallel slots",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail")
    for name, status, detail in rows:
        table.add_row(name, status, detail)
    console.print(table)

    if failed:
        console.print("\n[red]doctor failed[/red] - fix the above before running the pipeline.")
        raise typer.Exit(1)
    console.print("\n[green]all checks passed[/green]")


@app.command()
def hello() -> None:
    """One schema-constrained JSON round-trip to Ollama, from inside this container.

    This is the Phase 0 gate: it proves cross-container networking, the
    host.docker.internal hop, and the model weights all work together.
    """
    cfg = Config.from_env()
    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "model_said": {"type": "string"},
        },
        "required": ["ok", "model_said"],
    }

    async def _run() -> None:
        async with OllamaClient(cfg) as c:
            console.print(f"[dim]-> {cfg.ollama_base_url} · {cfg.extraction_model}[/dim]")
            g = await c.generate(
                'Reply with ok=true and model_said="hello from ollama".',
                schema=schema,
            )
            parsed = g.parsed
            secs = g.total_duration_ns / 1e9
            tps = g.completion_tokens / secs if secs else 0.0
            console.print(f"[green]round-trip OK[/green]  {parsed}")
            console.print(
                f"[dim]{g.prompt_tokens} prompt + {g.completion_tokens} completion tokens "
                f"in {secs:.2f}s ({tps:.1f} tok/s)[/dim]"
            )

    try:
        asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]hello failed[/red] {type(e).__name__}: {e}")
        console.print(f"[dim]OLLAMA_BASE_URL={cfg.ollama_base_url}[/dim]")
        sys.exit(1)


@app.command("init-db")
def init_db() -> None:
    """Apply schema.sql (idempotent)."""
    cfg = Config.from_env()
    from pipeline.core.db import init_schema

    init_schema(cfg)
    console.print("[green]schema applied[/green]")


@app.command()
def ingest() -> None:
    """PHASE 1 GATE: load JSONL into documents and seed the job queue."""
    cfg = Config.from_env()
    from pipeline.core.db import init_schema
    from pipeline.stages.ingest import ingest as run_ingest, stats

    init_schema(cfg)
    res = run_ingest(cfg)

    console.print(
        f"[dim]read {', '.join(res.files)} from {cfg.input_dir}[/dim]\n"
        f"inserted [bold]{res.inserted}[/bold] · "
        f"updated [bold]{res.updated}[/bold] · "
        f"unchanged [bold]{res.unchanged}[/bold]  "
        f"(jobs seeded {res.jobs_seeded}, reset {res.jobs_reset})"
    )
    if res.inserted == 0 and res.updated == 0 and res.unchanged:
        console.print("[green]re-run was a no-op[/green] - idempotency confirmed")

    s = stats(cfg)
    table = Table(show_header=True, header_style="bold")
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    table.add_column("expected", style="dim")

    expectations = {
        "documents": 9991,
        "posts": 2001,
        "replies": 7990,
        "authors": 100,
        "threads": 2000,
        "orphan_replies": 0,
    }
    failed = False
    for key, value in s.items():
        exp = expectations.get(key)
        if exp is None:
            table.add_row(key, str(value), "")
            continue
        good = value == exp
        failed = failed or not good
        mark = "[green]OK[/green]" if good else "[red]MISMATCH[/red]"
        table.add_row(key, f"{value} {mark}", str(exp))
    console.print(table)

    if failed:
        console.print("\n[red]ingest gate failed[/red] - counts do not match the source files.")
        raise typer.Exit(1)
    console.print("\n[green]phase 1 gate passed[/green]")


@app.command()
def bench(n: int = typer.Option(None, help="docs to time (default BENCH_N)")) -> None:
    """Measure real throughput and extrapolate a full-corpus ETA."""
    cfg = Config.from_env()
    from pipeline.qa.bench import run as run_bench

    console.print(f"[dim]warming up, then timing at {cfg.ollama_num_parallel} parallel...[/dim]")
    r = asyncio.run(run_bench(cfg, n))
    eta_m = r.eta_s / 60

    table = Table(show_header=True, header_style="bold")
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    table.add_row("docs timed", str(r.n))
    table.add_row("parallel slots", str(r.parallel))
    table.add_row("wall clock", f"{r.wall_s:.1f}s")
    table.add_row("throughput", f"{r.docs_per_s:.2f} docs/s")
    table.add_row("completion tok/s", f"{r.completion_tps:.0f}")
    table.add_row("prompt tok/doc", f"{r.prompt_tokens / r.n:.0f}")
    table.add_row("completion tok/doc", f"{r.completion_tokens / r.n:.0f}")
    table.add_row("latency p50 / p95", f"{r.p50_ms / 1000:.1f}s / {r.p95_ms / 1000:.1f}s")
    table.add_row(
        f"ETA for {r.corpus_total} docs",
        f"[bold]{eta_m:.0f} min[/bold]" if eta_m < 90 else f"[bold]{eta_m / 60:.1f} hrs[/bold]",
    )
    console.print(table)


@app.command()
def extract(
    limit: int = typer.Option(None, help="stop after N docs (default LIMIT env, blank = all)"),
    doc: list[str] = typer.Option(None, "--doc", help="re-extract specific doc_id(s)"),
) -> None:
    """PHASE 2: run schema-constrained extraction over pending documents."""
    cfg = Config.from_env()
    from pipeline.stages.extract import run as run_extract, queue_counts
    from pipeline.core.prompts import PROMPT_VERSION

    lim = limit if limit is not None else cfg.limit
    before = queue_counts(cfg)
    console.print(
        f"[dim]model={cfg.extraction_model} prompt={PROMPT_VERSION} "
        f"parallel={cfg.ollama_num_parallel} num_ctx={cfg.ollama_num_ctx} "
        f"limit={lim or 'all'}[/dim]"
    )
    console.print(f"[dim]queue before: {before}[/dim]")

    with console.status("extracting...") as status:
        def tick(s) -> None:
            done = s.processed + s.cached
            rate = s.processed / s.wall_s if s.wall_s else 0
            status.update(
                f"extracting... {done} done ({s.cached} cached, {s.failed} failed)"
                + (f" · {rate:.1f} docs/s" if rate else "")
            )

        s = asyncio.run(run_extract(cfg, limit=lim, progress=tick, doc_ids=list(doc) if doc else None))

    rate = s.processed / s.wall_s if s.wall_s else 0
    console.print(
        f"processed [bold]{s.processed}[/bold] · cached [bold]{s.cached}[/bold] · "
        f"failed [bold]{s.failed}[/bold] in {s.wall_s:.1f}s"
        + (f" ({rate:.2f} docs/s)" if rate else "")
    )
    if s.processed:
        console.print(
            f"[dim]{s.prompt_tokens / s.processed:.0f} prompt + "
            f"{s.completion_tokens / s.processed:.0f} completion tok/doc[/dim]"
        )
    if s.processed == 0 and s.cached:
        console.print("[green]re-run was a no-op[/green] - extraction cache confirmed")
    console.print(f"[dim]queue after: {queue_counts(cfg)}[/dim]")
    if s.failed:
        raise typer.Exit(1)


@app.command()
def normalize(oov: int = typer.Option(25, help="how many OOV surfaces to list")) -> None:
    """PHASE 3 GATE: canonicalise mentions, build switch edges, guard demographics."""
    cfg = Config.from_env()
    from pipeline.core.db import init_schema
    from pipeline.rules.lexicon import load as load_lexicon
    from pipeline.stages.normalize import run as run_normalize

    init_schema(cfg)
    lex = load_lexicon()
    r = run_normalize(cfg)

    table = Table(show_header=True, header_style="bold")
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")
    table.add_row("documents normalised", str(r.docs))
    table.add_row("treatment mentions", str(r.treatments))
    table.add_row("symptom mentions", str(r.symptoms))
    table.add_row("mapped to lexicon", f"{r.mapped} ({r.mapped_pct:.1f}%)")
    table.add_row("out of vocabulary", f"{r.oov_total} ({len(r.oov)} distinct)")
    table.add_row("names absent from the text", f"{r.unattested_name} (excluded from aggregates)")
    table.add_row("attribution corrected self->other", str(r.attribution_fixed))
    table.add_row("kind corrected from lexicon", str(r.kind_fixed))
    tot_loc = r.surface_located + r.surface_missing
    pct = 100.0 * r.surface_located / tot_loc if tot_loc else 0.0
    table.add_row("surface located in text", f"{r.surface_located} ({pct:.1f}%)")
    cpct = 100.0 * r.context_any / tot_loc if tot_loc else 0.0
    table.add_row("mentions with context", f"{r.context_any} ({cpct:.1f}%)")
    table.add_row("switch edges kept", str(r.switches))
    table.add_row("switch edges dropped", str(r.switches_dropped))
    table.add_row("demographic claims kept", str(r.demo_accepted))
    table.add_row("demographic claims rejected", str(r.demo_rejected))
    table.add_row("lexicon", f"{len(lex.entries)} entries · {lex.digest}")
    console.print(table)

    if r.oov:
        console.print(f"\n[bold]top out-of-vocabulary surfaces[/bold] (lexicon backlog)")
        for surface, n in r.oov.most_common(oov):
            console.print(f"  {n:>4}  {surface}")


@app.command()
def profile(
    narrate_: bool = typer.Option(True, "--narrate/--no-narrate", help="run the per-author LLM pass"),
) -> None:
    """PHASE 4: roll mentions up into user_profiles and treatment_summary."""
    cfg = Config.from_env()
    from pipeline.core.db import init_schema
    from pipeline.stages.profile import build, narrate as run_narrate

    init_schema(cfg)
    r = build(cfg)
    console.print(
        f"profiles [bold]{r.profiles}[/bold] · treatments rolled [bold]{r.treatments_rolled}[/bold]"
        f" · symptoms rolled [bold]{r.symptoms_rolled}[/bold]"
    )
    if narrate_:
        with console.status("writing narratives...") as status:
            def tick(s) -> None:
                status.update(f"narratives... {s.narratives}")
            n = asyncio.run(run_narrate(cfg, progress=tick))
        console.print(
            f"narratives [bold]{n.narratives}[/bold] · failed [bold]{n.narrative_failures}[/bold]"
        )


@app.command()
def export(run_id: str = typer.Option(None, help="name the output directory")) -> None:
    """PHASE 4 GATE: write parquet + csv per grain, plus manifest.json."""
    cfg = Config.from_env()
    from pipeline.stages.export import run as run_export

    m = run_export(cfg, run_id)
    table = Table(show_header=True, header_style="bold")
    table.add_column("grain", style="cyan")
    table.add_column("rows", justify="right")
    for k, v in m["row_counts"].items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(
        f"[dim]run_id={m['run_id']} · model={m['model']} · prompt={m['prompt_version']}"
        f" · lexicon={m['lexicon_digest']}[/dim]"
    )
    console.print(f"[green]wrote {len(m['files'])} files[/green] to {cfg.out_dir}/{m['run_id']}")


@app.command()
def stability(
    docs: int = typer.Option(50, help="documents to repeat"),
    runs: int = typer.Option(3, help="repeat count"),
) -> None:
    """Measure run-to-run agreement, canonical vs raw surface forms."""
    cfg = Config.from_env()
    from pipeline.qa.stability import run as run_stability

    console.print(f"[dim]extracting {docs} docs x {runs} runs...[/dim]")
    r = asyncio.run(run_stability(cfg, docs, runs))
    table = Table(show_header=True, header_style="bold")
    table.add_column("aggregation level", style="cyan")
    table.add_column("mean Jaccard", justify="right")
    table.add_row("canonical ids (this pipeline)", f"{r.canonical_mean:.3f}")
    table.add_row("raw surfaces (LLM-only)", f"{r.surface_mean:.3f}")
    console.print(table)
    delta = r.canonical_mean - r.surface_mean
    console.print(f"[dim]{r.n_docs} docs x {r.runs} runs · canonicalisation adds {delta:+.3f}[/dim]")


@app.command()
def ablate() -> None:
    """A/B whether parent-post context earns its complexity. Writes only to
    extractions - never touches mentions or the job queue."""
    import asyncio as _a

    from pipeline.qa.ablate import run as run_ablate

    cfg = Config.from_env()
    base, abl, n = _a.run(run_ablate(cfg))

    t = Table(show_header=True, header_style="bold",
              title=f"parent-context ablation · {n} replies", title_justify="left")
    t.add_column("measure", style="cyan")
    t.add_column(base.label, justify="right")
    t.add_column(abl.label, justify="right")
    t.add_column("change", justify="right")

    def row(name: str, a: int, b: int, good_down: bool) -> None:
        if a == 0:
            delta = "-"
        else:
            pct = 100.0 * (b - a) / a
            better = (pct < 0) if good_down else (pct > -10)
            colour = "green" if better else "red"
            delta = f"[{colour}]{pct:+.0f}%[/{colour}]"
        t.add_row(name, str(a), str(b), delta)

    row("PRIMARY  self, name NOT in own text", base.self_unattested, abl.self_unattested, True)
    row("GUARDRAIL self, name in own text", base.self_attested, abl.self_attested, False)
    t.add_row("total treatment mentions", str(base.mentions), str(abl.mentions), "")
    t.add_row("bare references (the patch, it)", str(base.bare_refs), str(abl.bare_refs), "")
    console.print(t)

    prim_ok = abl.self_unattested < base.self_unattested
    guard_ok = base.self_attested == 0 or (
        (abl.self_attested - base.self_attested) / base.self_attested > -0.10
    )
    console.print(
        f"\nprimary {'[green]improved[/green]' if prim_ok else '[red]did not improve[/red]'} · "
        f"guardrail {'[green]held[/green]' if guard_ok else '[red]broke[/red]'}"
    )
    console.print(
        "[bold]-> simplify: drop parent body[/bold]" if (prim_ok and guard_ok)
        else "[bold]-> keep parent context[/bold]"
    )


@app.command()
def evaluate(
    prompt_version: str = typer.Option(None, help="Score a specific run (default: current)."),
) -> None:
    """PHASE 5 GATE: named-entity recall, hallucination, negative controls, coverage."""
    cfg = Config.from_env()
    from pipeline.qa.score import run as run_eval

    r = run_eval(cfg, prompt_version)
    console.print(f"[dim]scoring prompt_version={r.prompt_version} · model={cfg.extraction_model}[/dim]")

    cov_note = ("" if r.mentions_match_pv else
                f"  [yellow]-- these rows come from the mentions table, built from "
                f"{r.mentions_built_from or 'an unknown run'}, NOT {r.prompt_version}[/yellow]")
    cov = Table(show_header=True, header_style="bold",
                title="coverage" + cov_note, title_justify="left")
    cov.add_column("metric", style="cyan"); cov.add_column("value", justify="right")
    cov.add_row("documents extracted", f"{r.docs_extracted:,} / {r.docs_total:,}"
                f"  ({100*r.docs_extracted/r.docs_total:.1f}%)")
    cov.add_row("parse success", f"{r.parse_ok:,} / {r.parse_total:,}"
                f"  ({100*r.parse_ok/max(r.parse_total,1):.2f}%)")
    cov.add_row("mentions mapped to lexicon", f"{r.mapped_pct:.1f}%")
    cov.add_row("mentions with context", f"{r.context_pct:.1f}%")
    cov.add_row("surface forms -> canonical", f"{r.distinct_surfaces} -> {r.distinct_canonical}"
                f"  ({r.distinct_surfaces/max(r.distinct_canonical,1):.1f}:1)")
    console.print(cov)

    bt = Table(show_header=True, header_style="bold",
               title="named-entity recall (deterministic gold: the name appears in the document's own text)",
               title_justify="left")
    bt.add_column("entity", style="cyan"); bt.add_column("docs", justify="right")
    bt.add_column("found", justify="right"); bt.add_column("recall", justify="right")
    bt.add_column("emitted w/o text", justify="right")
    for b in sorted(r.entities.values(), key=lambda x: -x.docs_containing):
        if not b.docs_containing and not b.emitted_without_text:
            continue
        rec = "-" if b.docs_containing == 0 else f"{100*b.recall:.0f}%"
        style = "" if b.docs_containing == 0 or b.recall >= 0.8 else "yellow"
        bt.add_row(b.entity, str(b.docs_containing), str(b.docs_found), f"[{style}]{rec}[/{style}]" if style else rec,
                   str(b.emitted_without_text))
    console.print(bt)
    console.print(f"  overall named-entity recall [bold]{100*r.recall_overall:.1f}%[/bold]"
                  f"   ·   mentions whose name is absent from the text [bold]{r.hallucinated}[/bold]")

    console.print(f"\n[bold]negative controls[/bold]  {r.neg_pass}/{r.neg_total} pass")
    for f in r.neg_failures:
        console.print(f"  [red]FAIL[/red] {f}")

    if r.parent_only_total:
        pct = 100 * r.parent_only_non_self / r.parent_only_total
        console.print(
            f"\n[bold]attribution tendency[/bold] (proxy, not accuracy): of {r.parent_only_total} "
            f"mentions where the name appears only in the PARENT post, "
            f"[bold]{pct:.0f}%[/bold] are attributed to someone other than the reply author."
        )


@app.command()
def reclaim() -> None:
    """Return jobs abandoned by a crashed worker to the queue."""
    cfg = Config.from_env()
    from pipeline.stages.extract import reclaim_stale

    n = reclaim_stale(cfg, older_than_s=0)
    console.print(f"reclaimed [bold]{n}[/bold] stale job(s)")


@app.command("show")
def show(n: int = typer.Option(10, help="how many extractions to print")) -> None:
    """Print raw extractions for hand-reading - the Phase 2 gate."""
    import json as _json

    cfg = Config.from_env()
    from pipeline.core.db import connect

    with connect(cfg) as conn:
        rows = conn.execute(
            """
            SELECT e.doc_id, d.doc_type, d.author_id, coalesce(d.title, left(d.body, 70)),
                   e.raw_json, e.completion_tokens
              FROM extractions e JOIN documents d ON d.doc_id = e.doc_id
             WHERE e.parsed_ok ORDER BY e.created_at LIMIT %s
            """,
            (n,),
        ).fetchall()

    for doc_id, dtype, author, label, raw, ct in rows:
        console.rule(f"[bold]{doc_id}[/bold] · {dtype} · {author} · {ct} tok")
        console.print(f"[dim]{label}[/dim]")
        console.print_json(_json.dumps(raw))


if __name__ == "__main__":
    app()
