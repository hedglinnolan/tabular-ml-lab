# TurboTab

An experimental rebuild of Tabular ML Lab as a single scrolling interview — the app
asks one question at a time, the answers accumulate into a document, and that document
*is* the manuscript in embryo. TurboTax for tabular research data.

This folder is the design and transition record. **Nothing here has been implemented.**
No application code has been written, moved, or deleted. Every file is analysis,
specification, or a clickable prototype driven by synthetic data.

To get from here to something runnable, see [`LOOP.md`](LOOP.md) §"Loop 3 — the walking
skeleton". The ledger loops verify and fix; only Loop 3 builds.

Baseline for all analysis: `origin/main` @ `24c3446` (PR #145 merged).

---

## Start here

| Read | For |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | What survives the rebuild, what dies, what must be built. The census of the existing codebase and the six-component target architecture. |
| [`TRANSITION_PLAN.md`](TRANSITION_PLAN.md) | The delicate parts, named. Live bugs, structural facts, landmine classes, and a gated sequence. |
| [`FINDINGS_LEDGER.md`](FINDINGS_LEDGER.md) | All 385 findings, tracked to completion. Nothing closes without a regression test named after it. |
| [`prototypes/interview-feed.html`](prototypes/interview-feed.html) | Open in a browser. The interaction model, working, with synthetic data. |
| [`LOOP.md`](LOOP.md) | **How to run this as an unsupervised agent loop.** Three loops: verify the ledger, fix the live bugs, or build the walking skeleton — the last is the one that produces a running app. |
| [`ROADMAP.md`](ROADMAP.md) | **All twelve loops, the three decision gates, and what "done" means.** Read this to see the whole road, not just the next step. |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Provenance: what was measured, what was verified by hand, what must still be re-checked. |

## Prototypes

Static HTML, no build step, no network. Open directly in a browser.

- **`interview-feed.html`** — the full arc: upload branch → coach noticing → EDA findings
  with before/after previews → feature question → preprocess (deferred reminders resurface)
  → model selection → training → results → drafted manuscript. Flagged figures actually
  land in the manuscript in journal format. Try "change" on an early decision to see the
  stale cascade.
- **`design-language.html`** — the written design system: palette, the three-voice type
  rule (app speaks serif, user acts sans, data speaks mono), component vocabulary, motion
  and voice rules. Includes a live slice of the feed.
- **`train-compare-mockup.html`** — the first static mockup, showing the hierarchy fixes
  applied to the existing Train & Compare page. Superseded by the feed prototype but kept
  because it isolates the hierarchy argument from the interaction argument.

## Machine-readable data

- `data/findings.json` — the ledger as records (`id`, `area`, `sev`, `item`, `detail`,
  `ev`, `act`, `status`). Use this to drive the verification loop.
- `data/raw-analysis.json` — the full unfiltered output of ten agent passes over the
  repository (~820 KB). Function inventories, invariants, landmines, stage contracts.
  The ledger is a deduplicated summary of this; go here when you need the detail.
- `data/import-graph.json` / `data/reverse-deps.json` — measured module dependency graph.

---

## State of play

**Settled.**

- The modeling engine is already UI-independent: of ~18,700 lines in `ml/` and `models/`,
  41 reference Streamlit. `visualizations.py` returns figure objects and never renders to
  Streamlit, so the entire plotting layer is portable.
- Coupling is concentrated in five modules. Four are shallow — `insight_ledger.py` and
  `workflow_provenance.py` are clean domain objects with a `get_*()` singleton bolted to
  the bottom, and both already have `to_dict`/`from_dict`.
- `utils/session_state.py` is the `AnalysisProject` model in disguise. Its dataclasses are
  already the project schema; `reset_downstream_results()` is already the invalidation DAG,
  written imperatively.
- Per-model preprocessing pipelines are real in the data model. The *global fallback slot*
  is the flaw, not the design.
- The coach can **order** questions but cannot **gate** them. Promoting it to Router is new
  construction, not a refactor.

**Open, and blocking.**

- Row identity uses two incompatible conventions (lockbox seals index *labels*; splits store
  *positions*). Highest silent-corruption risk in the migration. Pick one before writing
  `AnalysisProject`.
- Three live bugs are shipping on `main` today — see `TRANSITION_PLAN.md` §01. They are
  independent of the rebuild.
- The safety net is thinner than the coverage number: no test calls the production
  `reset_downstream_results()`, `tests/integration/conftest.py` injects a bare `Ridge` where
  the app stores wrapper objects, and `tests/test_insight_id_integrity.py` will pass
  vacuously after a rename.
- `docs/FINDINGS_LEDGER.md` (the *existing* app ledger, not this one) has an open tail on
  the multi-file/JSON import path from two audit runs whose results were lost. Treat
  `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and `pages/01` as
  engine-move-only until that closes.

---

## The loop

The rule this folder runs on, inherited from `docs/FINDINGS_LEDGER.md`:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false**.

And its corollary for this work: **a finding that is documented but not tracked is a finding
that gets lost.** Every finding lands in `FINDINGS_LEDGER.md` with a disposition, and nothing
is closed without a regression test named after it.

The ledger has two tiers:

- **Tier 0 (15)** — re-verified by hand against `origin/main` after PR #145. Status is real.
- **Tier 1 (370)** — from ten agent passes at commit `fbe422a`, all marked `UNVERIFIED`.
  They predate PR #145, which changed `utils/test_lockbox.py` by +312 lines and added
  `utils/replay.py`. They must be re-checked before they are trusted.

**The first loop iteration is verifying Tier 1 against main** and dispositioning each row as
`OPEN` / `FIXED` / `NOT-A-DEFECT` / `WONTFIX`. That converts a research artifact into a real
backlog. See [`LOOP.md`](LOOP.md) for the prompt to hand an agent.

Work `data/findings.json` through `tools/ledger.py`; `FINDINGS_LEDGER.md` is generated.

```bash
python docs/turbotab/tools/ledger.py stats     # progress
python docs/turbotab/tools/ledger.py next --n 15
python docs/turbotab/tools/ledger.py check     # schema guard; run before every commit
```

The full route to a finished app — twelve loops and three decision gates — is in
[`ROADMAP.md`](ROADMAP.md). The short version, from `TRANSITION_PLAN.md` §06:

1. Fix the three live bugs on the current app (independent of the rebuild).
2. Write characterization tests **before** moving any code.
3. Settle row identity, then design `AnalysisProject`.
4. Extract the split block (`pages/06:380-760`) and the step state machine (`utils/theme.py:685`).
5. Cut the record singletons, add the job queue.
6. Build the Router against EDA only.
7. Port the frontend one step at a time.

Each step in the plan carries an exit gate — a thing that must be demonstrably true before
the next begins. Don't skip the gates; they are where a delicate migration proves it hasn't
silently broken something.
