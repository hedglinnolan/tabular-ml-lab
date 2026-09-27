# TurboTab Next — blueprint

The contract every agent builds against. Short on purpose: if a rule matters, it lives here or in a
test, not in a new essay. Branch: `turbotab-next`. Owner: Nolan. Orchestrator: Claude.

## North star — "This app is supposed to be beautiful and useful" (Nolan, 2026-09-27)

Every milestone serves these ambitions at once; a feature that serves one while failing another is
not done.

1. **The full breadth of modeling nutrition research needs** — prediction and inference;
   regression, binary, multiclass, ordinal, time-to-event; repeated measures; p ≫ n omics;
   compositional and substitution models; survey-weighted estimation. A model family is a plug-in
   that declares what it needs, what it assumes (its inductive bias), and when it is a poor fit, so
   the shelf can grow wide without the app getting harder to use.
2. **Every choice is domain-informed and teaches while it asks.** Each question carries the one
   sentence needed to answer it; a *why?* that opens in place; and **the consequence shown on the
   user's own data in the pipeline panel**. The pipeline panel is the main teaching surface: *what
   did my choice just do to my rows, my columns, my model?* Deeper concept pages open in a drawer and
   are never required. The packs (`docs/turbotab/research/`) are the source of what is taught, with
   their evidence badges.
3. **The design communicates.** Motion shows cause and effect, type separates the app's voice from
   the user's actions and the data, every color is a claim. A milestone that works but reads badly is
   not done: the orchestrator reviews the screenshots and the `/lab` motion before Nolan drives it.
4. **Modeling decision provenance.** Nolan's term (2026-09-27) for a reproducibility problem in the
   field: a published result rarely lets a reader reconstruct which choices turned the raw columns
   and rows into the model's inputs. TurboTab records every choice as a decision, and the lineage
   diagram and participant flow *draw* it. That makes provenance a deliverable, not a by-product.
   The export carries it: the lineage and row flow as figures, and a machine-readable provenance
   record (decisions, lineage, software versions) that a reviewer can replay to regenerate the
   same model matrix.

## 0 · Rulings (2026-09-27) — do not re-litigate

- **Keep the Python engine and domain packs; replace the frontend.** React + TypeScript + Vite.
  The old `turbotab/api.py` + `turbotab/web/index.html` keep running untouched until the new app
  passes them, then they are deleted.
- **Parity with Classic (Streamlit) is not binding.** Reshape anything. Do not edit `pages/`, `app.py`,
  or Classic-only modules. Reuse `ml/` and `turbotab/*.py` domain code by import; if a reused module
  needs a change, prefer a wrapper in `turbotab/core/` over editing a module Classic also imports.
- **All five lenses** — metabolomics, genomics, dietary, clinical, survey — are "nutrition research"
  and are all first-class.
- **Centerpiece: the live pipeline panel.** Interview/record on the left; on the right, always
  visible: **Rows** (participant flow: n at each step), **Columns** (lineage from raw columns to the
  model matrix X), **Results**. Changing any decision visibly propagates downstream.
- **Local compute first, arbitrarily large data.** The backend runs on the user's machine and reads
  files from disk; the same backend deploys to a university server when the laptop can't cope.
- **Velocity over ceremony.** Fixed-scope milestones. Proportional testing (§6). New ideas go to
  `INBOX.md`, not into the current milestone, unless they are blockers.

Principles carried forward from the old docs (these still bind): never assert falsely — every
rendered claim traces to a recorded decision or a computed fact; the user's answer is the answer
(detection suggests, never pre-selects below high confidence); the shelf is never shortened
(judgment is order and stated concern, not absence); preview before apply; the past is editable and
never silently destroyed (stale, not deleted); anything fit on data is fit on training rows only;
the lockbox is sealed before exploration and sealed once.

## 1 · Layout

```
turbotab/core/            engine host: workspace, datastore, decisions, graph, jobs, events, stages
turbotab/core/methods/    statistical methods (energy adjustment, substitution curves, …)
turbotab/core/stages/     stage implementations registered on the graph
turbotab/core/tests/      pytest
turbotab/server/          FastAPI app (HTTP + SSE only; no statistics here)
turbotab/server/tests/    pytest (TestClient)
turbotab/frontend/        Vite + React app; builds to turbotab/frontend/dist, served by the server
docs/turbotab-next/       BLUEPRINT.md (this), INBOX.md, milestone notes
```

Python: `venv/bin/python` (3.13; pandas 2.3, sklearn 1.9, duckdb, pyarrow, fastapi, lightgbm,
xgboost, shap already installed). New Python deps go in `turbotab/server/requirements.txt`.
Node 25 / npm 11.

## 2 · Data layer — arbitrary size

- **Workspace.** `TURBOTAB_HOME` (default `~/.turbotab`) → `projects/<pid>/` holding `project.json`
  (metadata), `decisions.jsonl` (append-only log), `data/raw.parquet`, `cache/<stage>/<key>/`.
  The cache is disposable; an exported project carries decisions + inputs, never fitted objects.
- **Ingest once, columnar.** CSV/TSV/TXT via DuckDB `read_csv` (parallel, out-of-core); Parquet
  read in place; Excel via pandas. Write `data/raw.parquet` with an added `__row_id` int64 column
  (0..n-1 in file order) — the stable row identity everything downstream keys on. Never hold the
  whole table in the server process to answer a UI question.
- **UI reads are queries.** Row windows, column summaries, histograms are DuckDB queries over the
  Parquet file. The browser never receives the full table.
- **Modeling materializes only what it needs** (`materialize(columns, rows)` → pandas/numpy), under
  a memory budget (default: 50% of available RAM, `TURBOTAB_MEMORY_BUDGET`). Exceeding it raises
  `MemoryBudgetExceeded` carrying the estimate, the budget, and the sentence "this dataset needs
  more memory than this machine has free — run TurboTab on a server" (server mode is the answer, not
  a silent sample).
- **Bridge to legacy domain code.** `packs.findings(df, lens)`, `engine.diagnose`, etc. take a
  pandas frame; the bridge materializes under the same budget. Wide omics tables (p ≫ n) are a
  known M2 benchmark item — M0 only has to ingest the genomics/metabolomics fixtures correctly.

## 3 · Decisions — event-sourced

- Every user answer is a **Decision**: pydantic v2 model, discriminated union on `kind`.
  Stored as a `DecisionRecord { id, seq, at, kind, payload…, note? }` line in `decisions.jsonl`.
- **State is a fold.** `ProjectState = fold(records)`: each decision writes one **slot**
  (`set_lens`→`lens`, `set_target`→`target`, `set_purpose`→`purpose`, …); later writes win.
  Changing an answer = appending a new record. `revert {decision_id}` restores the slot's previous
  value. The full log is the Record the UI renders as the transcript.
- Refusals (the "refuse" rung) are HTTP 409 `{error: {code, message, exits: [...]}}` and are not
  recorded.
- M0 kinds: `set_lens {lenses: Lens[]}` (non-empty), `set_target {column}`,
  `set_task {column, task: regression|binary|multiclass}` (override of detection for that
  column; the `task` slot holds the latest answer naming the current target, else null),
  `set_purpose {purpose: prediction|inference}`, `revert {decision_id}`. Later milestones add
  kinds; adding a kind must not require editing a giant if-chain — one handler per kind, registered.

## 4 · The stage graph — what makes "live" possible

A stage is a pure function of (upstream artifacts, the decision slots it reads).

```python
Stage(name, version, deps: tuple[str, ...], reads: tuple[str, ...], fn, heavy: bool = False,
      requires: tuple[str, ...] = ())   # slots that must be set, else status "blocked"
```

- `key = sha256(name, version, [key(d) for d in deps], {slot: state[slot] for slot in reads},
  dataset fingerprint)`. No manual invalidation lists: a changed decision changes keys downstream.
- **Status per stage:** `fresh` (artifact for current key exists) · `stale` (no artifact for the
  current key but an older one exists — served with `fresh: false`, never deleted) · `queued` ·
  `running` · `blocked` (with `missing: [slots]`) · `error` (with message) · `idle` (never computed).
- **On every decision:** recompute keys; emit a `stage` event for each stage whose status changed;
  automatically (re)compute every unblocked stage whose key changed, in dependency order; **cancel
  in-flight jobs whose key is no longer current** (rapid toggling must not pile up fits).
- Artifacts: small → JSON; tables → Parquet; fitted estimators → joblib (cache only).
- `heavy=False` stages run in a thread; `heavy=True` stages run as jobs in worker processes.
- M0 stages: `ingest` (DatasetInfo), `profile` (column summaries, lens hints via
  `turbotab.packs.suggest`, task detection via `turbotab.engine.detect_task_type` once a target is
  set), `findings` (reads `lens`, `target`: `turbotab.packs.findings` + `turbotab.engine.diagnose`).

## 5 · Jobs and events

- **JobRunner:** a small pool of worker processes (default `cpu_count - 1`, `TURBOTAB_WORKERS`),
  each pre-importing numpy/pandas/sklearn so a job starts fast. Workers read inputs from the
  workspace and write artifacts to the cache (no large pickles across the process boundary).
  `ctx.progress(fraction, message)` reports progress; **cancel stops CPU within ~2 s**
  (cooperative flag checked at progress calls, then terminate-and-respawn the worker).
- **EventBus → SSE.** `GET /api/projects/{pid}/events` (text/event-stream), events:
  `decision {record}` · `stage {stage, status, key, fresh}` · `job {job_id, stage, state, progress,
  message}` · heartbeat every 15 s. The frontend invalidates queries from these; no polling.

## 6 · HTTP API (M0) — `/api/*`, typed

Every request/response is a pydantic model; the OpenAPI document is the contract and the frontend's
types are generated from it (`npm run gen:api` ← `python -m turbotab.server.openapi`).

| Method & path | Returns |
|---|---|
| `GET /api/health` | `{version, mode: local|server, workers}` |
| `GET /api/projects` | `ProjectSummary[]` |
| `POST /api/projects` `{path}` — local mode only (403 in server mode) | `ProjectSummary` (ingest starts as a job) |
| `POST /api/projects/upload` multipart, streamed to disk, no size cap in local mode | `ProjectSummary` |
| `GET /api/projects/{pid}` | `ProjectView {summary, state, decisions: DecisionRecord[], stages: {name: StageStatus}}` |
| `POST /api/projects/{pid}/decisions` `Decision` | `ProjectView` · 409 refusal |
| `GET /api/projects/{pid}/stages/{stage}` | `StageResult {stage, key, fresh, status, artifact}` |
| `POST /api/projects/{pid}/stages/{stage}/run` — retry after `error` or a cancel (`cancelled: true`) | `StageStatus` |
| `GET /api/projects/{pid}/table?offset&limit&columns` | `TableWindow {columns, rows, total_rows}` |
| `GET /api/projects/{pid}/columns` | `ColumnSummary[]` |
| `GET /api/projects/{pid}/columns/{name}/histogram?bins=` | `Histogram {edges, counts, n_missing}` |
| `GET /api/projects/{pid}/jobs/{jid}` · `POST …/cancel` | `JobView` |
| `GET /api/projects/{pid}/events` | SSE |
| `GET /api/fs/list?path=` — local mode only | `{path, parent, entries: [{name, is_dir, size}]}` |

Local mode binds `127.0.0.1` only and rejects requests whose `Host` is not `localhost`/`127.0.0.1`
(DNS-rebinding guard), and any non-GET request a page on another site sent (`Origin` not
localhost/127.0.0.1, or `Sec-Fetch-Site: cross-site`): a form upload needs no preflight.
No CORS in production; the Vite dev server proxies `/api`.
Launch: `venv/bin/python -m turbotab.server --port 8787 [--open] [--mode local|server]`.

## 7 · Frontend

- **Stack:** Vite, React 19, TypeScript strict, `motion` (Motion for React), TanStack Query,
  TanStack Table + `@tanstack/react-virtual`, `d3-scale`/`d3-shape`/`d3-array` (math only — React
  renders the SVG), Radix primitives, CSS Modules + `tokens.css`. Vitest + Testing Library;
  Playwright for real-browser journeys. ESLint (typescript-eslint, react-hooks) + Prettier.
- **Design language carries over** from `docs/turbotab/DESIGN_LANGUAGE.md`: §02 color tokens (every
  hue is a claim — `--accent` now, `--ok` recorded, `--warn` coach, `--stop` invalid-downstream
  only), §03 three voices (app speaks serif, user acts sans, data speaks mono; Inter and JetBrains
  Mono are vendored in `static/fonts/`), §04 components, §09 question grammar. Light and dark are
  both first-class.
- **Layout:** header (project, job chips) · left: **the Record** (questions → decision sentences,
  findings) · right, sticky: **the Pipeline panel** (Rows / Columns / Results). Narrow screens stack.
- **Motion's job is identity continuity** (DESIGN_LANGUAGE §05.2): *settle* (the question becomes its
  decision sentence — shared `layoutId`), *arrive* (new content grows from its cause), *propagate*
  (staleness sweeps downstream in order), and the working table/lineage morphing under a change.
  Nothing else moves. `prefers-reduced-motion` → instant.
- **Server state** lives in TanStack Query keyed by `[pid, …]`; SSE events invalidate/patch it.
  No global mutable singletons.

## 8 · Testing — proportional

- **Tier A, rigorous, known-answer:** statistical methods, leakage (anything fit on data sees
  training rows only), lockbox/split, graph invalidation semantics, ingest fidelity. These are where
  a wrong answer ends up in a paper.
- **Tier B, light:** one TestClient test per route family; component tests only for real logic.
- **Tier C:** one Playwright journey per milestone, with screenshots for Nolan's review.
- **Don't:** run the legacy suites (`turbotab/test_*.py`, `tests/`) — they take ~2 h and pin every
  core on a machine beside Nolan's bed; write a test per UI-polish tweak; add prose rules.
- Fast checks: `venv/bin/python -m pytest turbotab/core turbotab/server -q` and, in
  `turbotab/frontend`, `npm run check` (tsc + eslint + vitest). Both should stay under ~1 minute.

## 9 · Milestones

| | Scope — a milestone is done when Nolan can drive it on real fixtures |
|---|---|
| **M0 Foundations** | core (workspace, datastore, decisions, graph, jobs, events), typed server, frontend shell with the Record + Pipeline panel, open-by-path + upload, lens/target/purpose questions, virtualized table, motion primitives |
| **M1 Energy-adjustment slice** | dietary lens: energy adjustment (the five specs in `research/NUTRITION_PACK.md` §04) as an in-fold pipeline step; split; fit a linear model and a tree ensemble; Rows/Columns/Results panels live; switching the spec rewrites the model matrix, refits, and redraws a substitution curve |
| **M2 Opening sequence, all lenses** | orientation, grain/repeats, eligibility & exclusions, the seal; findings with preview-before-apply and deferral; wide omics data path benchmarked |
| **M3 Explore & prepare** | explore stack, missingness routing, preprocessing recipes, survey weights as `sample_weight` |
| **M4 Models & meaning** | model shelf at scale, comparison deck, explainability (inductive-bias curves, substitution curves), sensitivity/instability |
| **M5 Report & ship** | manuscript + checklists + journal-format figure export; desktop launcher; university server deployment (Docker, auth) |

Each milestone runs as: build (parallel agents by area, in worktrees) → integrate → one review
drive (blockers only) → fix → the orchestrator drives it with Playwright → Nolan drives it when he
can (his findings jump the queue; the next milestone does not wait for him).

## 10 · Source control — so v2 ships as one clean merge

- `main` is the released Classic app (v1.0.0). Classic fixes keep landing there.
- `turbotab-next` is the v2 integration branch, pushed to `origin` after every merged phase. Agents
  work in worktree branches and merge in; merged worktree branches are deleted.
- Merge `main` into `turbotab-next` at each milestone boundary so the final merge stays small.
- Classic must keep working on `turbotab-next`: do not edit its code (§0), so merging v2 never breaks
  v1 users before we decide Classic's fate.
- **Never push a `v*` tag from this branch** — `.github/workflows/release.yml` publishes a release for
  any `v*` tag. Milestone markers are `next-m0`, `next-m1`, ….
- v2 release: PR `turbotab-next` → `main`, CI green, tag `v2.0.0`.

## 11 · Guidance without walls — the design doctrine

The hardest design problem in this product, in Nolan's words: *"It's hard to present all of the
options and guidance for each without inundating a user with information they will likely not
read."* The answer is not better prose. **The explanation of an option is its effect on the user's
own data.** Rules, in priority order:

1. **Show the consequence, then say only what the picture can't.** Hovering or focusing an option
   asks the consequence planner (`turbotab/core/consequences.py`) what that option would do and
   renders it as a labeled *preview* — nothing is recorded. Arrow keys move through options, so
   the user learns by contrast: flip, watch, flip back.
2. **A closed vocabulary of views, chosen by measured change.** Rows (`row_flow`), columns
   (`lineage`), values (`distribution`, `table_focus`), relationships (`relationship`), and later
   structure (`embedding`) and outputs (`metric`, `curve`). At most three per preview, one primary.
   A new decision kind declares how it transforms a sample; the generic diff picks what changed most.
   Domain builders override only where a specific picture *is* the teaching.
3. **Wide data shows what the choice touched** — the affected columns (≤ 12) with a count of the
   rest; never the whole table.
4. **Word budgets, enforced by tests:** question ≤ 14 words · one-line why ≤ 22 · option
   consequence ≤ 16 · in-place *why?* ≤ 60 · finding summary ≤ 20 · preview caption ≤ 20.
5. **Four opt-in layers:** (0) question + one line; (1) each option's one-line consequence and its
   preview; (2) *why?* in place; (3) the concept drawer (pack content, sourced, badged). Nothing past
   layer 1 is needed to answer.
6. **Their data before theory.** A sentence citing their table ("`fat_total` correlates 0.71 with
   `kcal`") beats a general one; general theory lives in the drawer.
7. **One voice at a time.** One open question; ≤ 3 pushed findings, the rest counted and typed;
   same-kind findings share one paged card; a finding is a one-line claim plus its lever.
8. **Terms define themselves:** dotted underline → a one-sentence card on hover/focus.
9. **Judgment is order and emphasis, never absence** — usual choices first, the rest one step away.

### 11.1 · Rulings on the consequence-preview prototypes (Nolan, 2026-09-27)

From the screenshots of `explore/stage`, `explore/inline` and `explore/scrub`:

- **`stage` is the base.** *"It provides a larger window for visualizing the change with each
  decision."*
- **No per-option preview thumbnails** (`inline`'s small multiples are rejected): *"the typical user
  for this app during exploratory mode will rifle through each option regardless and can see the
  effects of a decision in the larger window."* The design should make rifling fast: one key per
  option, and the big window morphs.
- **`stage`'s lineage diagram is the canonical lineage** — *"the best version I have seen of that
  lineage diagram design."* It is also the modeling-decision-provenance figure (North star 4).
- **A transform player, not a slider** (settled over three exchanges). Nolan first ruled in a global
  slider over a multi-view stage: *"It could allow us to show a multi-lens view of the transformation
  from different plots with one large slider at the top… giving us more space to work since we are not
  cutting the 'widgets' window in half… As long as users get the option to save any plot."* He then
  refined it: *"maybe a slider is not the right button design for the actions to show a transformation
  since it is ultimately a binary decision, but something is. And the animation between the states is a
  useful tool to see the transformation in action. I genuinely believe in that as a pedagogical
  vehicle."* The design:
  1. **Flip** — a two-state toggle, *Your data now ⇄ With this choice (preview)*, in the stage's header.
     Click or Space flips it, and every flip animates. It drives every view at once (scatter,
     distribution, lineage, table); each view shows one state at full width.
  2. **▶ Watch it happen** — plays the transformation as a **storyboard of the method's own real,
     labeled steps**, never an interpolated half-state. For example, residual: fit each nutrient on
     energy (the line draws) → keep what energy does not explain (points drop to residuals) → add back
     the average (the cloud re-centers; r settles at 0.00). A method declares its storyboard; a generic
     transform's storyboard is just before → after. Consequence views therefore gain an optional
     `steps: [{label, …view data}]`.
  3. **Save** captures any labeled state — now, a storyboard step, after, or a before/after pair —
     captioned with the choice that produced it (a provenance figure).
  4. **Two axes, two controls:** arrow keys move between options; the flip moves between before and
     after, and its position holds across options, so at *after* flipping options morphs one method's
     result into the next.
  5. **Headline numbers stay pinned** in a small readout (r 0.86 → 0.14; n 21,348 → 2,943).
  The table keeps `scrub`'s column-identity morph (`fat_total` → `fat_total_adj`).
- **Direction to prototype: a pipeline banner.** `stage` gives up the "once over the world look at
  the pipeline you built". Nolan's idea: put that overview into the banner at the top of the app,
  so that *"everything below the banner image is the working window."* The banner is a compact,
  always-visible strip: row counts through the flow, the column path to the model matrix, the
  models, the result. It doubles as the map of where you are (DRIVE_RUBRIC §2.4) and as a
  provenance summary. Clicking a node navigates there.
