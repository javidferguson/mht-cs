# Mac dev assumes Ollama runs natively on the host (Docker on macOS cannot pass
# through the Apple GPU). EC2 runs it as a container: `make up-gpu`.

SHELL := /bin/bash
COMPOSE := docker compose
RUN := $(COMPOSE) run --rm worker python -m pipeline

-include .env
export

.DEFAULT_GOAL := help

.PHONY: help
help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment
.PHONY: env
env:  ## create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

.PHONY: build
build: env  ## build the worker/notebook image
	$(COMPOSE) build

.PHONY: up
up: env  ## start postgres (Mac: Ollama runs on the host, not here)
	$(COMPOSE) up -d postgres

.PHONY: up-gpu
up-gpu: env  ## EC2: start postgres + containerised Ollama with GPU
	$(COMPOSE) --profile ollama up -d postgres ollama

.PHONY: down
down:  ## stop containers (keeps volumes: pg data + model weights)
	$(COMPOSE) down

.PHONY: setup
setup: env build up  ## build + start everything needed for phase 0

# --------------------------------------------------------------------- ollama
.PHONY: pull-model
pull-model:  ## pull EXTRACTION_MODEL (host Ollama on Mac, container on EC2)
	@if $(COMPOSE) ps --services --filter status=running 2>/dev/null | grep -qx ollama; then \
	  echo "-> pulling $(EXTRACTION_MODEL) into the ollama container volume"; \
	  $(COMPOSE) exec ollama ollama pull "$(EXTRACTION_MODEL)"; \
	else \
	  command -v ollama >/dev/null || { echo "ollama not installed on host. Run: brew install ollama"; exit 1; }; \
	  echo "-> pulling $(EXTRACTION_MODEL) into host ~/.ollama"; \
	  ollama pull "$(EXTRACTION_MODEL)"; \
	fi

# ----------------------------------------------------------------- phase gates
.PHONY: test
test:  ## run the rules-layer test suite (no database, no model)
	$(COMPOSE) run --rm worker python -m pytest tests -q

.PHONY: doctor
doctor:  ## verify postgres, ollama, model, input data, disk
	$(RUN) doctor

.PHONY: hello
hello:  ## PHASE 0 GATE: one JSON round-trip to Ollama from inside the container
	$(RUN) hello

.PHONY: init-db
init-db:  ## apply the database schema (idempotent)
	$(RUN) init-db

.PHONY: ingest
ingest:  ## PHASE 1 GATE: load JSONL into documents, seed the job queue
	$(RUN) ingest

.PHONY: bench
bench:  ## measure real throughput and extrapolate a full-corpus ETA
	$(RUN) bench

.PHONY: stop-workers
stop-workers:  ## stop orphaned worker containers (a killed `make extract` leaves one running)
	@docker ps -q --filter name=$(notdir $(CURDIR))-worker-run | xargs -r docker stop >/dev/null && echo "stopped orphaned workers" || echo "no orphaned workers"
	@$(RUN) reclaim >/dev/null 2>&1 || true

.PHONY: extract
extract:  ## run extraction (LIMIT=n to cap, e.g. make extract LIMIT=10)
	$(RUN) extract $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: normalize
normalize:  ## PHASE 3 GATE: canonicalise mentions + switch edges (no LLM cost)
	$(RUN) normalize

.PHONY: profile
profile:  ## PHASE 4: roll up user_profiles + treatment_summary (+narratives)
	$(RUN) profile

.PHONY: export
export:  ## PHASE 4 GATE: write parquet + csv + manifest to out/
	$(RUN) export

.PHONY: all
all: ingest extract normalize profile export  ## full pipeline end to end

.PHONY: ablate
ablate:  ## A/B parent-context (writes only to extractions; never normalize)
	$(COMPOSE) run --rm -e PROMPT_VERSION=v6-noparent -e PARENT_CONTEXT_CHARS=0 worker python -m pipeline ablate

.PHONY: evaluate
evaluate:  ## PHASE 5 GATE: recall, hallucination, negative controls, coverage
	$(RUN) evaluate

.PHONY: stability
stability:  ## measure run-to-run agreement (canonical vs surface)
	$(RUN) stability

.PHONY: show
show:  ## PHASE 2 GATE: print extractions for hand-reading (N=10)
	$(RUN) show --n $(or $(N),10)

# ------------------------------------------------------------------- dev shell
.PHONY: shell
shell:  ## bash inside the worker container
	$(COMPOSE) run --rm worker bash

.PHONY: psql
psql:  ## psql into the pipeline database
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

.PHONY: lab
lab: env  ## start JupyterLab on $(JUPYTER_PORT)
	$(COMPOSE) up -d notebook
	@echo "JupyterLab -> http://localhost:$(JUPYTER_PORT)"

.PHONY: logs
logs:  ## tail all container logs
	$(COMPOSE) logs -f
