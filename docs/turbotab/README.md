# TurboTab

An experimental rebuild of Tabular ML Lab as a single scrolling interview — the app
asks one question at a time, the answers accumulate into a document, and that document
*is* the manuscript in embryo. TurboTax for tabular research data.

This folder is the design and transition record. **The app is running.** A driver goes upload →
lens → orientation → target → purpose → grain → eligibility → seal against real fixtures without
touching code. The interaction spine is real; the back half — figures, the manuscript chain, the
domain packs — is in progress. See [`ROADMAP.md`](ROADMAP.md) §"The map" for where the line is.

*(This paragraph read "Nothing here has been implemented" for far longer than it was true. A README
is a claim like any other and decays the same way — silently, while the people who already know the
state keep working. If you are picking this up and it looks wrong again, it is.)*

**New here?** Read `PRODUCT_VISION.md`, `ROADMAP.md`, then [`LOOP.md`](LOOP.md) §02 (how a loop is
shaped), §06 (how to judge the report) and §03 (what has already run).

**Streamlit is never deleted.** Existing users depend on it, so this is not a replacement
project — it is an *extraction* project. The goal is one shared core with two front doors, and
the rules that keep that from becoming two divergent apps are in
[`ROADMAP.md`](ROADMAP.md) §"Two front doors, one core".

Baseline for all analysis: `origin/main` @ `24c3446` (PR #145 merged).

---

## Start here

| Read | For |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | What survives the rebuild, what dies, what must be built. The census of the existing codebase and the six-component target architecture. |
| [`TRANSITION_PLAN.md`](TRANSITION_PLAN.md) | The delicate parts, named. Live bugs, structural facts, landmine classes, and a gated sequence. |
| [`FINDINGS_LEDGER.md`](FINDINGS_LEDGER.md) | Every finding, tracked to completion (618 at last count — `tools/ledger.py stats` is authoritative). Nothing closes without a regression test named after it. |
| [`prototypes/interview-feed.html`](prototypes/interview-feed.html) | Open in a browser. The interaction model, working, with synthetic data. |
| [`LOOP.md`](LOOP.md) | **The operator's manual.** How a loop is shaped (four parts), the log of what has run, the guardrails, how domain research is cited, and **how to adjudicate a report** — the half of the job that was unwritten longest. |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) | **Do the intelligent features carry over?** Capability vs orchestration vs exposure, and the register that stops a feature going missing quietly. |
| [`DOMAIN_SCIENCE.md`](DOMAIN_SCIENCE.md) | **What the domain research means for the product.** Four literatures converged on seven structural facts; those are the primitives. §03b routes every finding to the app surface it lands on. The four authoritative research files are in [`research/`](research/) and are cited by section from every pack-building loop. |
| [`ROADMAP.md`](ROADMAP.md) | **The twelve phases, the domain track that runs through them, the three decision gates, and what "done" means.** Read this to see the whole road, not just the next step. Contains both constitutions: routing (what the app asks) and lockbox (what it may know, and when). |
| [`ASSEMBLY_SPEC.md`](ASSEMBLY_SPEC.md) | **Multi-file assembly.** The research, the interaction, and the seven acceptance criteria the audit produced. The grain question it specifies is the same one the lockbox seal needs. |
| [`COPY_DECK.md`](COPY_DECK.md) | **Every user-facing string in the Guided door, by step and by state, with the condition that triggers it.** So copy can be reviewed without running the app. Half generated from source, half hand-assembled and probe-checked — `tools/copydeck.py regen` after changing a string. |
| [`OPENING_SEQUENCE.md`](OPENING_SEQUENCE.md) | **Everything before the seal** — upload to drawn lockbox, in one place because the order is load-bearing. Nine questions, four to six firing, with copy, conditions and fixtures. |
| [`DOMAIN_PACKS.md`](DOMAIN_PACKS.md) | **How the app becomes field-aware without becoming a different app.** The opening lens question, what a pack may and may not change, and the filter that decides which science earns its place. |
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

**Settled since — these read as open for months after they closed.**

- **Row identity.** Decision A is answered and reframed: the question was not *labels or
  positions* but that four repair kinds renumber rows mid-analysis. The identity barrier is the
  ruling; `AnalysisProject` shipped at L5. See `ROADMAP.md` §"Decision A".
- **The three live bugs** are `FIXED` with named tests, folded into L7.
- **The lost audit was never lost.** Both runs wrote to `docs/audit/` and were committed the whole
  time this file said otherwise. That error cost a loop of rediscovery and is the origin of the
  rule that a record pointing at ephemeral storage will eventually lie — and lie toward *"the work
  is gone."* The freeze on `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and
  `pages/01` still holds, for the untriaged defect tail rather than for missing evidence; its one
  definition and the three gates that lift it are in `TRANSITION_PLAN.md` §05.

**Open, and blocking.**

- **The back half.** Figures beyond the first two, the manuscript chain, and three of four domain
  packs. `ROADMAP.md` §"The map" is authoritative.
- **Verification debt in the research.** Every numeric threshold in `research/` is search-surfaced
  rather than read from primary text; items marked `[verify-at-build]` may not ship as constants.
  The DRI tables in particular must ship as data read from NASEM, not as prose.
- **The safety net is thinner than the coverage number**, and the shape of the thinness keeps
  changing. Six times a guard has turned out to be testing its own description rather than the
  app — most recently three frontend tests that passed against a page emptied to `<body></body>`.
  The six axes are in `FEATURE_PARITY.md`; the practical answer is the revert probe.

---

## The loop

The rule this folder runs on, inherited from `docs/FINDINGS_LEDGER.md`:

> The app may be **silent**, and it may **refuse**, but it must never **assert something false**.

And its corollary for this work: **a finding that is documented but not tracked is a finding
that gets lost.** Every finding lands in `FINDINGS_LEDGER.md` with a disposition, and nothing
is closed without a regression test named after it.

The ledger has two tiers:

- **Tier 0 (15)** — re-verified by hand against `origin/main` after PR #145. Status is real.
- **Tier 1 (370)** — from ten agent passes at commit `fbe422a`, all marked `UNVERIFIED` at
  the time. **Loop 1 re-verified all 370 against HEAD (2026-07-27):** 289 OPEN, 31 PARTIAL,
  50 FIXED with named tests, 0 NOT-A-DEFECT. The ledger is now a real backlog, not research.
  Four stale Tier-0 rows the verifier flagged were closed by the adjudicator against their
  named tests; see `LOOP.md` §03 for the full result.

Work `data/findings.json` through `tools/ledger.py`; `FINDINGS_LEDGER.md` is generated.

```bash
python docs/turbotab/tools/ledger.py stats     # progress
python docs/turbotab/tools/ledger.py next --n 15
python docs/turbotab/tools/ledger.py check     # schema guard; run before every commit
```

The full route — twelve phases, the domain track that runs through them, and three decision gates —
is in [`ROADMAP.md`](ROADMAP.md). **Phases L1–L8 are done and all three decisions are answered.**
What remains is L9 (the interaction layer, in progress, one step per loop), the domain track D1–D5
running alongside it, then L10–L12.

Each phase carries an exit gate — a thing that must be demonstrably true before the next begins.
Don't skip the gates; they are where a delicate migration proves it hasn't silently broken
something.

**How the work actually happens** is `LOOP.md`: one prompt of four parts, run unattended, reporting
once, adjudicated against the code before it is accepted.
