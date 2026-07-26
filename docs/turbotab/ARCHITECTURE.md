# TurboTab — core inventory and target architecture

What survives the rebuild, what dies, and what has to be built.

**Headline: the modeling engine is already UI-independent.** Of ~18,700 lines in `ml/` and
`models/`, 41 lines reference Streamlit. TurboTab is not a rewrite of the engine — it is a new
host for an engine that has been quietly portable all along.

Baseline: `origin/main` @ `24c3446`.

---

## 01 · The census

| Layer | Files | Lines | Streamlit-coupled | Verdict |
|---|---:|---:|---:|---|
| `ml/` | 35 | 17,735 | 4 direct + 3 transitive | **core.** See the correction below — grep understates this |
| `models/` | 7 | 1,000 | 0 | **core.** `BaseModelWrapper` already defines a clean contract |
| `visualizations.py` | 1 | ~700 | 0 | **core.** Returns figure objects; never calls `st.pyplot`/`st.plotly_chart` |
| `utils/` — domain | 9 | 1,227 | 0 | **core.** `combine`, `combine_preview`, `persistence`, `reconcile`, `datasets`, `column_utils` |
| `utils/` — records | 2 | 2,076 | 2 | **shallow.** `insight_ledger` (7 refs), `workflow_provenance` (5). Both have `to_dict`/`from_dict` |
| `utils/` — state | 2 | 950 | 2 | **rewrite.** `session_state` (92 refs), `test_lockbox` (8) |
| `utils/` — UI & infra | 16 | ~7,200 | 16 | **mixed.** `*_ui.py`, `theme`, `perf_cache`, `session_manager`, `dataset_db` |
| `pages/` + `app.py` | 13 | 19,835 | 13 | **extraction backlog.** All UI, with domain logic embedded |

Proportions: **18,735 lines portable core · ~8,000 coupled · 19,835 page layer.**

### The five modules that actually resist

| Module | Lines | `st.` refs | What the coupling is | Effort |
|---|---:|---:|---|---|
| `session_state.py` | 657 | 92 | Pervasive. *This module is the project model in disguise* — already holds `DataConfig`, `SplitConfig`, `ModelConfig`, `TaskTypeDetection`, `CohortStructureDetection`, plus `reset_downstream_results()` | rewrite |
| `publication.py` | 1,887 | 32 | Export orchestration interleaved with download widgets | split |
| `test_lockbox.py` | 295 | 8 | Reads `exploratory_mode`, `test_lockbox`, `test_lockbox_fraction`, `random_seed`; writes the sealed lockbox back. Only `render_lockbox_status()` is true UI | easy |
| `insight_ledger.py` | 1,406 | 7 | 1,400 lines of pure domain logic, then a `get_ledger()` singleton at the bottom | trivial |
| `workflow_provenance.py` | 672 | 5 | 13 dataclasses and a recorder, then a `get_provenance()` singleton | trivial |

### Correction (twice): what "coupled" actually means here

**This section was wrong, and the walking-skeleton loop caught it.** Both corrections are worth
keeping visible, because both were methodology errors rather than typos.

**First error — I counted function-level imports as if they were module-level.** A static pass
over all `import` statements reported `ml.model_coach` as transitively tainted through
`utils.insight_ledger`. It is not: `model_coach` imports the ledger *lazily, inside two
functions* (`ml/model_coach.py:634`, `:1080`), and its only module-level imports are
`dataclasses`, `typing` and `enum`. It loads clean with Streamlit blocked. My own empirical run
had already shown `model_coach` importing successfully; I published the static analysis over the
measurement, which is exactly backwards.

Counting **module-level imports only**, the real figure:

| | Count | Modules |
|---|---:|---|
| Directly coupled | 2 | `ml.eda_actions`, `ml.macro_shape` |
| Transitively coupled | 2 | `ml.narrative_engine` → `utils.insight_ledger` · `ml.manuscript_validator` → `narrative_engine` → `insight_ledger` |
| Headless-clean | 38 | everything else in `ml/`, `models/`, `visualizations`, `data_processor` |

So **4 of 42 core modules cannot import without Streamlit, not 7** — and the Router's basis,
`ml.model_coach`, is not one of them. The manuscript chain is the real cluster, and all of it
traces to one module-level `import streamlit` at `utils/insight_ledger.py:46` (note: the singleton
at the bottom of that file is *not* its only coupling — the top-level import is).

**Second error — the reproduce snippet did not reproduce anything.** It defined
`find_module`/`load_module`, a finder protocol Python stopped consulting in 3.12. Run as printed
on a modern interpreter it reports success whether or not a module is coupled. And on a machine
without Streamlit installed the whole test passes vacuously, because the import fails for the
wrong reason.

A blocker that actually bites:

```python
import importlib, importlib.abc, importlib.machinery, sys, types

# 1. Make sure a *real* streamlit is importable, so the test cannot pass vacuously.
if "streamlit" not in sys.modules:
    try:
        importlib.import_module("streamlit")
    except ImportError:
        sys.modules["streamlit"] = types.ModuleType("streamlit")  # stub stands in

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):      # 3.12+ protocol
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(f"BLOCKED: {name}")

for k in [k for k in sys.modules if k == "streamlit" or k.startswith("streamlit.")]:
    del sys.modules[k]
sys.meta_path.insert(0, Blocker())

# 2. Prove the blocker bites before trusting any result from it.
try:
    importlib.import_module("streamlit")
    raise SystemExit("blocker is not working — results would be meaningless")
except ImportError:
    pass

importlib.import_module("ml.model_coach")   # clean at HEAD
```

**The lesson, which generalises to the whole ledger:** static analysis over-reports coupling, and
a test that cannot fail proves nothing. Prefer the runtime check, and always assert the guard is
active before trusting what it says.

### Structural facts

- **163 distinct `st.session_state` keys.** That is the migration surface.
- **34 of 70 core modules** are imported by any test. All of `models/` is uncovered, along
  with `ml.splits`, `ml.triage`, `ml.preprocess_operators`, `ml.feature_steps`,
  `ml.stats_tests`, `ml.outliers`, `visualizations`.
- **Highest fan-in modules are the coupled ones**: `session_state` (16 importers, 10 pages),
  `insight_ledger` (14/7), `workflow_provenance` (12/10), `theme` (12/11), `test_lockbox`
  (12/10), `storyline` (11/10), `table_export` (9/9), `data_processor` (8/6).
- **Three import cycles, all through the state layer**:
  `ml.publication ↔ utils.insight_ledger`, `utils.cohorts ↔ utils.test_lockbox`,
  `utils.cohorts ↔ utils.session_state ↔ utils.test_lockbox`.

---

## 02 · Invariants the rebuild may not break

The functions are re-writable. These are not — they are accumulated scar tissue from real
failures, several enforced by named regression tests.

- **The lockbox is sealed before exploration, and sealed once.** `ensure_lockbox()` freezes
  row labels keyed by a signature over `(df, target, task_type, fraction, seed, group_col)`.
  A redraw sets `_lockbox_redrawn` because a silent re-draw invalidates every downstream number.
- **Repeated subjects group the split.** `detect_repeated_subjects()` exists so a patient
  cannot appear in both partitions. Must survive the multi-file join path, where a one-to-many
  join creates repeats that existed in neither input.
- **`high` confidence is the only tier the UI may pre-select.** Implemented as
  `ShapeFinding.auto_suggestable`. *The app may be silent, and it may refuse, but it must never
  assert something false.* **Auto-advancing an interview is pre-selection** — this binds the Router.
- **Diagnosis never mutates; fixes are reversible and explicit.** The import doctor returns
  findings and proposed fixes and applies nothing on its own. This is already the contract the
  preview-before-apply interaction needs.
- **Some findings must refuse rather than guess.** Mixed units produce a *critical* finding with
  `fix_kind='none'`; an ambiguous decimal comma is forced to low confidence because guessing
  wrong is a silent 1000× rescale.
- **Column iteration is positional, not label-based.** `_each_column()` yields by position so
  duplicate labels give a `Series`, not a sub-`DataFrame`. Reaching for `df[col]` reintroduces
  a critical defect that already shipped once (ledger finding 13).
- **Footer/row fixes carry positions, not index labels.** A non-`RangeIndex` frame breaks
  label-based slicing.
- **Provenance precedes narrative.** The narrative engine may only assert what the record
  contains; everything else is `[AUTHOR REQUIRED]`.
- **Invalidation is a cascade, not a reset.** `reset_downstream_results()` takes flags
  (`clear_feature_engineering`, `restore_pre_fe_features`, `clear_feature_selection`) —
  **partial invalidation is a real call**, so a naive full-cascade DAG cannot replace it.
- **Plane authority order is advise → witness → publish.** The Advisory plane must never write
  into the Narrative plane. This forbids exactly the Router→Record shortcut the promotion invites.
- **Nothing is written to disk.** A `_NEVER_PERSIST` session contract exists and is stronger
  than its documentation. A job queue is the most likely thing ever to violate it.
- **`get_data()` applies the active cohort filter**, with exactly two `full_study=True` escapes,
  enforced by nothing but a default parameter.

---

## 03 · Target architecture

Six components, in dependency order. Only two are new.

| Component | Status | Job | Source |
|---|---|---|---|
| **Engine** | exists | Stateless functions over dataframes. Diagnose, profile, transform, fit, evaluate, plot. Knows nothing about projects, users, or HTTP. | `ml/`, `models/`, `visualizations.py` |
| **Project** | rewrite | One serializable object holding data handles, every config, the lockbox, the active cohort filter, and the stage DAG. Owns invalidation. Replaces `st.session_state`. | from `utils/session_state.py` + `test_lockbox.py` |
| **Record** | exists | The ledger and the provenance log. Append-mostly, already serializable. The transcript the user scrolls *is* a rendering of this. | `insight_ledger.py`, `workflow_provenance.py` |
| **Router** | **NEW** | Given a Project and a Record, decides which question is asked next and which options are offered. | new construction (see §04) |
| **Jobs** | **NEW** | Anything over ~1s runs as a cancellable job with progress and a result handle. | no equivalent today |
| **Feed** | prototyped | The interview. Renders the Record forward, poses the Router's next question, shows job progress. Holds no analysis state. | `prototypes/interview-feed.html` |

```
  answer ──▶ Project.apply(decision)          state becomes true
                 │
                 ├──▶ Project.invalidate(downstream)  stale, not destroyed
                 │
                 ├──▶ Record.append(decision)         history, append-only
                 │
                 ├──▶ Jobs.submit(recompute)          observable, cancellable
                 │         └──▶ Engine.f(df, cfg)     pure, unchanged
                 │                    └──▶ Record.upsert(insights)
                 ▼
            Router.next(Project, Record)       which question comes next
                 ▼
              Feed renders                     one card at a time
```

### Why this shape

- **Engine and Project stay separate** because the census says they already are. Putting a
  project handle into 18,700 working lines couples them to the host for no gain.
- **Record is not derived from Project.** The record is what happened; the project is what is
  currently true. Changing an answer rewrites the project and *appends to* the record.
  Conflating them loses the ability to claim "the lockbox was sealed before exploration" — a
  statement about history, not state.
- **Router is separate from Engine** because the engine's job is to compute honestly and the
  router's is to decide what to ask. Fusing them puts UI sequencing inside statistical
  functions, which is the mistake the page layer already made.
- **Jobs must exist before the frontend.** Every "is it broken?" complaint, every lost state on
  refresh, and every frozen page traces to Streamlit having no way to own long work. This is the
  component whose absence caused the migration.

---

## 04 · What must be built that does not exist

| Component | Why nothing covers it today | Risk |
|---|---|---|
| `AnalysisProject` | `st.session_state` is a global string-keyed dict across 163 keys. The new model must be typed, serializable, and survive a refresh. The dataclasses exist; the container does not. | high |
| Job queue | Streamlit reruns instead of scheduling. Nothing owns a long task, reports progress, or supports cancel. (The existing Cancel button is decorative — see `TRANSITION_PLAN.md` §01.) | high |
| Router | `model_coach` ranks models and `insight_ledger` holds noticings, but nothing decides *which question is next*. Page order is hard-coded in filenames. **The coach is a pure annotator: it can order questions but cannot gate them.** No `blocker` severity, no confidence tier of its own, and 100% of its trigger logic lives in `pages/`. Scope this as a build, not a move. | high |
| Preview/diff engine | The prototype's before/after tables need real `(before, after, stats)` triples. The import doctor already returns reversible fixes — extend that shape to every transform. | medium |
| Figure export contract | `visualizations.py` returns figures; the interactive/journal-format split needs one renderer with two targets. | low |
| Persistence | `session_manager`/`dataset_db` are Streamlit-bound; projects need server-side save/reopen without violating the no-disk contract. | medium |

---

## 05 · The extraction backlog

19,835 lines of page layer. Most is layout, but the domain logic mixed into it is code TurboTab
does not have — and it will not appear in any `ml/` inventory.

| Page | Lines | Logic density | Likely trapped there |
|---|---:|---:|---|
| `11_Theory_Reference` | 5,162 | 152 | Mostly static explanatory content. **A content asset — migrate as data, not code.** |
| `06_Train_and_Compare` | 2,784 | 382 | Highest density. **~370 lines of split logic at 380–760**, CV wiring, the per-model-vs-borrowed pipeline rule, result assembly |
| `10_Report_Export` | 2,536 | 326 | 24 local functions. Manuscript assembly and export decisions living in the view |
| `02_EDA` | 1,764 | 222 | Which analyses run, in what order, what counts as notable. **The Router's raw material** |
| `05_Preprocess` | 1,326 | 131 | Per-model pipeline construction and defaults |
| `03_Feature_Engineering` | 1,293 | 144 | Transform catalogue, applicability rules, **a hand-rolled partial cascade** |
| `01_Upload_and_Audit` | 1,264 | 115 | Orchestration over import/join doctors — **frozen, open defect backlog** |
| `07_Explainability` | 1,234 | 160 | SHAP orchestration, per-model applicability |
| `09_Hypothesis_Testing` | 1,026 | 79 | Test selection rules |
| `08_Sensitivity` + `04_Feature_Selection` | 1,081 | 129 | Thin over `ml/`; cheapest to port |

**Do not port these pages.** Each page is a set of questions. The migration unit is not
"port page 02" — it is "what does the app ask at exploration time, and what does the Router
need to ask it?" Extracting the logic and rewriting the layout are the same task, done once,
in the interview idiom.

Also note `utils/theme.py:685 render_sidebar_workflow` is **not styling** — it holds the only
implementation of the step-completion state machine (ten predicates over session state) plus
the quick/advanced split. That is the Router's readiness function, filed under CSS.
