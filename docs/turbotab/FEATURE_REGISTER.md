# TurboTab feature register

**Generated from `data/register.json` — do not hand-edit.** Update via `tools/register.py`, then `regen`. Rationale and rules: `FEATURE_PARITY.md`.

| State | Meaning | Count |
|---|---|---:|
| `both` | exposed in Classic and Guided | 25 |
| `core` | extracted into the shared core | 4 |
| `classic-only` | a claim to be justified, never a shrug | 23 |
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
| `target-lockbox-settings` | Test-holdout / lockbox settings | Step 4 expander | **classic-only** | Updated after L5: the project now OWNS a lockbox — sealing raises the identity barrier and the apply path refuses pre-barrier repairs — but the fraction/seed settings and the act of sealing are not yet asked in the Guided interview. Half-watched, no longer unmodeled. |

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

## Explore / EDA (Classic: pages/02) — 27

| ID | Capability | Classic | State | Reason |
|---|---|---|---|---|
| `explore-eda-class-imbalance` | EDA trigger: eda_class_imbalance (warning) | pages/02_EDA.py:415 | **both** | Raised by the profile and rendered; the imbalance finding is in the Explore stack. |
| `explore-eda-leakage-col` | EDA trigger: eda_leakage_<var> (blocker) | pages/02_EDA.py:279 | **both** | T0-ROUTE-001 built: leakage is now a pushed question of consequence, not a palette offer. The Router raises one question per column correlating above 0.95 with the target, naming the column, ranked first in the plan. router.audit refuses a blocker offered as pull or skipped at any confidence. Leaving the step with one unresolved is permitted and recorded — the tool does not refuse the user's judgment, it refuses silence, and the acknowledgment is what the manuscript carries as a stated limitation. Classic still only offers a Run leakage detection action and never names the leaking column, so the Classic side of this row remains owed at L11 convergence of pages/02. |
| `explore-eda-missing-moderate` | EDA trigger: eda_missing_moderate (info) | pages/02_EDA.py:380 | **both** | Same source, info severity. |
| `explore-eda-missing-severe` | EDA trigger: eda_missing_severe (warning) | pages/02_EDA.py:363 | **both** | Surfaced as a profile finding in the Explore stack and as the missingness pull card. |
| `explore-eda-sufficiency-borderline` | EDA trigger: eda_sufficiency_borderline (warning) | pages/02_EDA.py:261 | **both** | Same source as the insufficient case, at warning severity. |
| `explore-eda-sufficiency-insufficient` | EDA trigger: eda_sufficiency_insufficient (blocker) | pages/02_EDA.py:245 | **both** | Raised by the profile as a critical warning and rendered in Explore. One of the two blocker-severity triggers that only the page emitted; it is engine-derived now. |
| `explore-rec-r1-plausibility` | EDA recommendation card: Physiologic Plausibility Check | ml/eda_recommender.py:267 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r10-baselines` | EDA recommendation card: Quick Baseline Models | ml/eda_recommender.py:541 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r2-missingness` | EDA recommendation card: Missingness Pattern Analysis | ml/eda_recommender.py:290 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r3-cohort-structure` | EDA recommendation card: Longitudinal Data Split Guidance | ml/eda_recommender.py:322 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r4-leakage` | EDA recommendation card: Target Leakage Risk Assessment | ml/eda_recommender.py:345 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r5-target-classification` | EDA recommendation card: Class Balance & Baseline | ml/eda_recommender.py:403 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r5-target-regression` | EDA recommendation card: Target Distribution & Outliers | ml/eda_recommender.py:378 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r6-dose-response` | EDA recommendation card: Dose-Response Trends | ml/eda_recommender.py:428 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r7-interactions` | EDA recommendation card: Stratified Trends by Demographics | ml/eda_recommender.py:458 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r8-collinearity` | EDA recommendation card: Collinearity Heatmap | ml/eda_recommender.py:483 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-rec-r9-outlier-influence` | EDA recommendation card: Outlier Influence Analysis | ml/eda_recommender.py:517 (already engine) | **both** | Already engine code; the page only rendered it. Guided offers it as a pull affordance in the Explore palette — present, and never counted as a question. |
| `explore-eda-corr-cluster-col` | EDA trigger: eda_corr_cluster_<var> (warning) | pages/02_EDA.py:338 | **classic-only** | Collinearity clusters. Offered via the collinearity heatmap card; not yet a question. |
| `explore-eda-low-dimensionality` | EDA trigger: eda_low_dimensionality (opportunity) | pages/02_EDA.py:1228 | **classic-only** | Raised from the macro-shape panel, which Guided does not render yet. |
| `explore-eda-opportunity-balanced` | EDA trigger: eda_opportunity_balanced (opportunity) | pages/02_EDA.py:563 | **classic-only** | Metric guidance; Train step. |
| `explore-eda-opportunity-clean-data` | EDA trigger: eda_opportunity_clean_data (opportunity) | pages/02_EDA.py:475 | **classic-only** | An encouragement, not a decision. Deliberately not surfaced: an interview that congratulates the user on clean data is ceremony, which the clean-dataset guard forbids. |
| `explore-eda-opportunity-high-np` | EDA trigger: eda_opportunity_high_np (opportunity) | pages/02_EDA.py:544 | **classic-only** | Model-selection guidance; Train step. |
| `explore-eda-opportunity-nonlinear` | EDA trigger: eda_opportunity_nonlinear (opportunity) | pages/02_EDA.py:525 | **classic-only** | Model-selection guidance; Train step. |
| `explore-eda-opportunity-strong-signal` | EDA trigger: eda_opportunity_strong_signal (opportunity) | pages/02_EDA.py:503 | **classic-only** | Model-selection guidance; Train step. |
| `explore-eda-skew-group` | EDA trigger: eda_skew_group (info) | pages/02_EDA.py:451 | **classic-only** | Feature-level skew grouping. Its action is a Preprocess transform; deferred to that step. |
| `explore-eda-target-skew` | EDA trigger: eda_target_skew (warning) | pages/02_EDA.py:395 | **classic-only** | The action is a target transform on the Train page, which is outside the explore window. Belongs to the Train step's pre-list. |
| `explore-eda-tda-loops` | EDA trigger: eda_tda_loops (opportunity) | pages/02_EDA.py:1300 | **classic-only** | Raised from the persistence diagram; requires the optional TDA extra and the macro-shape panel. Neither is in Guided. |
