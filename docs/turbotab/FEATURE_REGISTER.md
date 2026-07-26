# TurboTab feature register

**Generated from `data/register.json` — do not hand-edit.** Update via `tools/register.py`, then `regen`. Rationale and rules: `FEATURE_PARITY.md`.

| State | Meaning | Count |
|---|---|---:|
| `both` | exposed in Classic and Guided | 8 |
| `core` | extracted into the shared core | 4 |
| `classic-only` | a claim to be justified, never a shrug | 13 |
| `guided-only` | a debt owed back to Classic | 4 |

## Data & Target (Classic: pages/01, Step 4) — 22

| ID | Capability | Classic | State | Reason |
|---|---|---|---|---|
| `audit-preview` | Preview before apply | (none) | **guided-only** | Classic applies straight from a button — the blind consent PRODUCT_VISION §04 argues against. Debt owed to Classic: T0-CLASSIC-001. |
| `audit-undo` | Undo an applied repair | (none) | **guided-only** | Engine always supported it (apply_fix never mutates); Classic never exposed it. Owed to Classic with the preview. |
| `audit-apply-repair` | Apply a proposed repair | 'Suggested Actions' | **both** | import_doctor.apply_fix, same nine fix kinds |
| `audit-diagnosis` | Structural diagnosis | import_doctor.diagnose | **both** | Same function, same findings — asserted field-for-field by test_findings_match_a_direct_engine_call |
| `audit-profile` | Dataset profile (types, missingness, cardinality, numeric stats) | Step 3 expanders | **both** | compute_dataset_profile; rendered as one ranked stack rather than six expanders |
| `ingest-single-file` | Single delimited file upload | st.file_uploader (Step 1) | **both** | engine.read_table, plain pandas inference to match what the doctor expects |
| `target-selection` | Target column selection | Step 4 | **both** |  |
| `target-task-detection` | Task-type detection | triage.detect_task_type | **both** |  |
| `target-task-override` | Task-type override | 'Override Task Type' expander | **both** | Added because this register found it missing — offered whenever confidence is below high; detection and answer both recorded. |
| `audit-cardinality-table` | Cardinality table, per column | Step 3 expander | **classic-only** | Guided surfaces the high-cardinality finding but not the full table. Pull-based exploration, deferred. |
| `audit-duplicate-rows` | Duplicate-row detection | Step 3 expander (inline) | **classic-only** | No engine home — computed inline in pages/01, orchestration still trapped (T0-PAGES-001). |
| `ingest-excel-sheets` | Excel .xlsx + sheet selection | Step 1 | **classic-only** | RESUME.md names Excel-sheet × transpose as untested; exposing it in Guided first would build on that gap. |
| `ingest-json-records` | JSON + records-key selection | Step 1 | **classic-only** | Same door as multi-file import; frozen with it. |
| `ingest-large-file-guard` | Large-file guard ('Load anyway') | Step 1 | **classic-only** | Guided uses a hard 64 MB ceiling instead — frame is held in memory, never spooled. Different mechanism, same intent. |
| `ingest-multi-join` | Multi-file join + key detection | Step 2, join_doctor | **classic-only** | Frozen (§05). |
| `ingest-multi-roster` | Multi-file upload + roster (rename / remove / replace) | Step 1 | **classic-only** | Multi-file path frozen pending the open ledger tail (TRANSITION_PLAN §05). Deliberate, not forgotten. |
| `ingest-practice-data` | Built-in practice datasets | Step 1 | **classic-only** | Guided ships one messy sample table (turbotab/sample_data/). |
| `ingest-repair-ledger-log` | Import-repair logging to the insight ledger | _log_import_repairs | **classic-only** | Guided records decisions; the ledger singleton is not cut yet (L7). |
| `ingest-transpose` | Transpose on import | Step 1 | **classic-only** | Belongs with the Excel path above. |
| `target-feature-picklist` | Feature selection (select all / clear) | Step 4 | **classic-only** | Not yet asked in Guided. |
| `target-goal-selection` | Goal selection (Prediction vs Hypothesis Testing) | Step 4 | **classic-only** | Guided assumes prediction; the hypothesis-testing branch is a different interview. |
| `target-lockbox-settings` | Test-holdout / lockbox settings | Step 4 expander | **classic-only** | Updated after L5: the project now OWNS a lockbox — sealing raises the identity barrier and the apply path refuses pre-barrier repairs — but the fraction/seed settings and the act of sealing are not yet asked in the Guided interview. Half-watched, no longer unmodelled. |

## Cross-step infrastructure — 7

| ID | Capability | Classic | State | Reason |
|---|---|---|---|---|
| `cancel-training` | Stopping a long run | removed — the button set a flag nothing read | **guided-only** | T0-LIVE-002. Classic's button is removed rather than wired: Streamlit runs one script per session on one thread, so during training no widget is interactive and the button could not be clicked at all. The page now says a run cannot be interrupted, which is true. Guided has real cancellation via the job queue. |
| `job-queue` | Observable, cancellable background work with explicit per-job RNG | none — Streamlit reruns instead of scheduling | **guided-only** | turbotab/jobs.py. Owed back to Classic in the sense that Classic cannot have it without a client/server split — this is the component whose absence caused the migration. Jobs that touch process-global RNG are serialized, because snapshot/restore does not isolate threads sharing one RNG; measured, not assumed. |
| `macro-shape-cache` | PCA / UMAP / persistence / Mapper caching keyed on dataset content | pages/02_EDA.py _macro_fp | **both** | T0-LIVE-001. The engine's caches keyed on nothing and served the first dataset's results to every later dataset and user. Caching moved to the host, which is what knows when a dataset changed; the fingerprint hashes values, not shape, because two cohort runs of one study share shape and columns. |
| `core-invalidation-dag` | Downstream invalidation cascade | utils/session_state.py reset_downstream_results | **core** | L5 DAG reproduces the production cascade key-for-key across all four flag combinations; carries a live list of the 15 keys pages/03's hand-rolled version forgets. |
| `core-lockbox-barrier` | Lockbox + identity barrier | utils/test_lockbox.py (no barrier) | **core** | L5: sealing raises the barrier; promote_header/melt_repeated unreachable behind it, enforced at the API apply path. Guided-ahead-of-Classic on the barrier itself. |
| `core-readiness-model` | Step-completion / readiness model (ten predicates + quick/advanced) | utils/theme.py:685 render_sidebar_workflow | **core** | Extracted to turbotab/readiness.py in L6; the page asks instead of computing, and a test asserts the expressions are gone. The Router's first real input. |
| `engine-headless` | Engine imports and runs with no Streamlit in the process | n/a — this is the core | **core** | All 45 core modules import with streamlit blocked, enforced by tests/test_engine_is_headless.py with a stub-first blocker so it cannot pass vacuously. Six were tainted; four went with two deleted lines (insight_ledger's module-level import, which also freed narrative_engine and manuscript_validator, and eda_actions' dead one). |
