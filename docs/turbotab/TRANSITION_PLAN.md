# TurboTab — transition plan

The delicate parts, named. Ten agents read the repository function by function: 663 functions,
76 invariants, 231 critical findings. This is the subset that changes what we do.

Baseline: `origin/main` @ `24c3446` (after PR #145). Section 01 and 02 were re-verified by hand
against that commit.

---

## 01 · Live bugs, not migration issues

Found while mapping. All three verified directly against source. **These are shipping today and
are independent of the rebuild.**

### LIVE-001 — Macro-shape plots serve the wrong dataset's results · CRITICAL

`compute_pca`, `compute_umap`, `compute_persistence` and `compute_mapper` are decorated
`@st.cache_data`, and their only non-default argument is `_df_numeric` — the leading underscore
is Streamlit's marker for *do not hash this*. Every call site passes nothing else.
**The cache key is therefore constant.**

`st.cache_data` is process-global, so the first dataset's PCA, UMAP, persistence diagram and
Mapper graph are returned for every subsequent dataset — and, in the multi-user university
Docker deployment, across users.

The codebase already knows this hazard: `pages/02_EDA.py:122` carries the comment *"Streamlit
skips hashing _-prefixed params (like _df), so we pass this as a non-prefixed param to ensure
cache misses on dataset switch"* and threads a `_data_fingerprint` through its own caches.
`macro_shape` never got the same fix.

> `ml/macro_shape.py:82, 240, 337, 495` · call sites `pages/02_EDA.py:1163, 1205, 1226, 1265` ·
> contrast `pages/02_EDA.py:118-124`

**Fix:** add a non-underscore fingerprint parameter, exactly as `pages/02` already does.

### LIVE-002 — The Cancel Training button does nothing · HIGH

`st.session_state.cancel_training` is initialised at line 1289 and set to `True` by the button
at 1301. **It is never read anywhere in the repository.** Training runs to completion regardless.

Worth fixing in place rather than waiting for the rebuild: it is the exact promise TurboTab's
job queue is meant to keep, and shipping a decorative cancel teaches users the app lies about control.

> `pages/06_Train_and_Compare.py:1289-1301` — grep for `cancel_training` returns only these four lines

### LIVE-003 — The neural-net sklearn adapter's `fit()` does not train · HIGH

`SklearnCompatibleNNRegressor.fit()` and `...Classifier.fit()` set `is_fitted_ = True`, record
`n_features_in_`, and return. The docstring says this is deliberate — real training happens in
`wrapper_instance.fit()` beforehand.

It is a legitimate adapter, but it only holds while every caller knows the protocol. **Any
sklearn idiom that clones and refits produces a silently untrained model that still answers
`predict()`** — no exception, just wrong numbers. Sensitivity analysis already calls `clone()`
on wrapper objects.

> `models/nn_whuber.py` — `fit()` in both regressor and classifier; reachable via
> `NNWeightedHuberWrapper.get_model()`

**Fix:** raise from `fit()` unless an explicit "already trained" flag is set, so the unsafe path
is loud instead of silent.

---

## 02 · Five structural facts that reorder the plan

### 1 · Per-model pipelines are real — the *fallback slot* is the flaw

**Question settled.** The per-model chain is genuine end to end — built per model at
`pages/05:870-1005`, stored by `set_preprocessing_pipelines()`, read by
`get_preprocessing_pipeline(model_key)`, applied at `pages/06:1286`. A test asserts two models
receive genuinely different transformed matrices. **`AnalysisProject` must have per-model
pipeline slots; a single global slot is a regression.**

But the UI warning was also literally true, for a narrower reason. `session_state.py:501` sets
the global default to `pipelines_by_model.get('default') or next(iter(pipelines_by_model.values()))`
— **an arbitrary member chosen by dict insertion order, which is checkbox render order.** Worse,
it hands back the same *object*, and `pages/06:1312` calls `.fit()` on it in place, so two
models' fitted pipelines can alias one instance.

PR #145 added *disclosure* (`pages/06:1813-1821` now names the borrowers) but did not change the
root cause.

**Fix in the rebuild:** delete the global slot. Make pipeline resolution total — every selected
model gets an explicit spec, auto-derived from its registry capabilities if the user never opened
its panel — and store the *spec* (serializable, diffable, hashable) rather than a live fitted object.

### 2 · Row identity uses two incompatible conventions · HIGHEST RISK

The lockbox stores index **labels** (`train_row_mask()` tests `lbl not in test_set`). The splits
store **positions** (`original_indices = np.where(mask)[0]` at `pages/06:398`, stored at 696-702).
Page 07 then does `df_raw.iloc[test_indices]` against a freshly fetched frame (`pages/07:185, 196`).

These agree only while the frame's index is a clean `RangeIndex` that never changes — which cohort
filtering, row-dropping repairs and joins all violate.

PR #145 fixed one *instance*: `apply_plausibility_filter` ended with `.reset_index(drop=True)`,
now removed and pinned by `tests/test_row_labels_are_identities.py`. **The class remains open** —
any future operation that changes row order or count between split time and page 07 reintroduces
the corruption.

**Decide one convention before writing `AnalysisProject`.** Labels are safer — they survive filtering.

### 3 · There is no split function — it lives in a page

`ml/splits.py` is 20 lines and contains exactly one helper, `to_numpy_1d`. The real splitting
logic is **`pages/06_Train_and_Compare.py:380-760`** — four mutually exclusive branches in
priority order:

1. grouped (`GroupShuffleSplit`) when cohort structure is longitudinal and an entity id exists
2. chronological sort-and-slice when `use_time_split` and a datetime column
3. lockbox-respecting — the frozen test labels *are* the test set, only train/val are drawn
4. plain / stratified `train_test_split`

That is ~370 lines of unextracted, untested engine logic implementing the app's most
safety-critical behavior, inside a Streamlit script. First to extract, first to characterization-test.

### 4 · The step state machine is filed under CSS

`utils/theme.py:685 render_sidebar_workflow` is not styling. It holds the **only** implementation
of the app's step-completion model — ten predicates over session state — plus the quick/advanced
split that decides which questions are optional. That is the Router's readiness function. Delete
`theme.py` as "just styling" during the cut and the step model goes with it.

### 5 · The coach can order questions but cannot gate them

The feasibility verdict on the central architectural bet. The coach is **a pure annotator with
one accidental act of control flow** — `pages/05_Preprocess.py:291-294` auto-checks model
checkboxes, the only place it steers rather than advises.

It is deterministic and explainable enough to **order** questions. It cannot **gate** them: it
never emits a `blocker` severity (only `pages/02_EDA.py` does, at 212 and 246), it has no
confidence tier of its own, and **100% of its trigger logic lives in `pages/`**.

**Consequence:** "promote the coach to Router" is not a refactor, it is new construction —
triggers must be lifted out of eleven pages, a severity/confidence model added, gating designed.
And the governing rule binds it: *high confidence is the only tier the UI pre-selects*, and
**auto-advancing an interview is pre-selection**.

---

## 03 · The safety net is thinner than the numbers suggest

34 of 70 modules have some test. That figure is optimistic in four specific ways.

| Problem | Detail | Consequence |
|---|---|---|
| Invalidation DAG untested | No test calls the production `reset_downstream_results()`. Three separate re-implementations test themselves. | The single most important behavior to preserve has zero real coverage. |
| Tests use a shape the app never makes | `tests/integration/conftest.py:82,98` injects a bare sklearn `Ridge` into `trained_models`; the app stores wrapper objects. `CODE_REVIEW.md` already records this. | Downstream suites pass against a fiction. This is how a `clone()` breakage stayed invisible. |
| A test that will pass vacuously | `tests/test_insight_id_integrity.py:23` is an AST scanner keyed on `SCAN_DIRS = ["pages","utils","ml"]` and the literal name `Insight`. | After the rename-heavy rebuild it finds zero ids and **passes** — green, guarding nothing, against exactly the defect class a refactor produces. |
| `models/` has no coverage at all | Every file in `models/` is untested, alongside `ml.splits`, `ml.triage`, `ml.preprocess_operators`, `ml.feature_steps`, `ml.stats_tests`, `visualizations`. | The layer that must move *intact* is the least protected. |

**Also: page 03 hand-rolls its own cascade.** The Feature Engineering save handler clears 18 keys
inline instead of calling `reset_downstream_results()`, missing at least `pdp_results`,
`bootstrap_results`, `baseline_results`, `calibration_results`, `cv_results`,
`train/val/test_indices`, `target_transformer`, `feature_names_by_model`, `cv_strategy`,
`eda_results`, `dataset_profile` — plus ledger rollback and provenance clearing. The cascade is
not merely untested, it is **already inconsistent between call sites.** Do not port either
version; derive one declarative DAG and verify it against both.

---

## 04 · Landmine classes

110 catalogued. These are the recurring classes, each with a rule that neutralises the whole class.

| Class | Where it bites | Rule for the rebuild |
|---|---|---|
| Cache keys that omit the data | `macro_shape` (live bug), `perf_cache`, six caches in `pages/02` | No caching layer survives the cut. Re-add keyed on an explicit content fingerprint, never on object identity. |
| Global RNG mutation | `utils/datasets.py`, `utils/seed.py`, `models/nn_whuber.py:314` call `np.random.seed` / `torch.manual_seed` on process state | Safe with one run at a time; **silently corrupting under a job-worker pool**. Pass explicit `Generator` objects. Reproducibility is a manuscript claim, so this fails where it costs most. |
| Exceptions swallowed to a clean default | 91 `except Exception` in core; ~24 return a plausible empty value | This is ledger finding 13 exactly — `diagnose()` swallowed a crash and reported a clean bill of health. Swallowing must record a degraded-result marker the UI can show. |
| Positional argument alignment | `build_unit_harmonization_config` / `build_plausibility_bounds` align factors to columns by list order | Silently applies one variable's unit conversion to another. Key by column name, never by position. |
| Read-never-written state keys | `llm_ui.py:225` reads `working_df`; page 01 writes `working_table` | Fails closed, returns a thinner plausible answer. A typed project model eliminates the class outright — the strongest single argument for doing `AnalysisProject` first. |
| Silent semantic links by string | `theory_anchors.infer_theory_anchor` falls back to substring matching on finding text | Reword a finding during the rewrite and its theory link vanishes with no error. Make the anchor mandatory at construction. |

---

## 05 · Do not touch yet

**The multi-file import path has an open defect backlog.** `docs/FINDINGS_LEDGER.md` carries a
"Still open" section fed by two audit runs whose results never landed — the scratchpad they wrote
to no longer exists. The original set was 102 raw / 48 confirmed; 11 are marked FIXED. A tail
remains untriaged, specifically on the multi-file and JSON import path.

`docs/audit/RESUME.md` adds two named gaps (Excel-sheet × transpose untested; the untriaged tail)
and one trap: *"still_broken" verdicts on findings 5, 7, 11, 13, 27 refer to pre-fix code* — read
cold, they will cause already-fixed code to be re-fixed.

> **Rule:** freeze `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and `pages/01`
> as *engine-move-only* — no signature changes — until that ledger closes.

**Cohort runs are the newest subsystem and they redefine `get_data()`.** The last several commits
are all cohort work. `get_data()` now applies an active cohort filter (`session_state.py:216`),
with exactly two `full_study=True` escapes, enforced only by a default parameter. Cohorts are also
entangled with the lockbox through two of the three import cycles. **A `Project` that models "the
working table" without modeling "the active cohort filter" silently deletes the newest feature.**

**Clear drops.** `utils/dataset_db.py` (797 loc, **zero importers**, superseded by
`session_projects.py`; keeping it reintroduces a disk path that contradicts the no-persistence
privacy model) · `setup.py` (`python_requires ">=3.8,<3.10"` against a 3.12 repo — currently
uninstallable) · `decision_curve_analysis` (README-advertised, zero production callers) ·
`example_data.csv` (10 rows, below the app's own 50-row minimum).

---

## 06 · The sequence

Each step has an **exit gate** — a thing that must be demonstrably true before the next begins.

**S1 · Fix the three live bugs on the current app.**
Cache keys in `macro_shape`, wire or remove Cancel, guard the NN adapter against clone-and-refit.
None of this is migration work; all of it is shipping wrong answers now, and the cache bug is
worst in the multi-user deployment.
*Gate: a second dataset in one session produces its own PCA.*

**S2 · Write the characterization tests before moving anything.**
Golden-output tests on the split block, `reset_downstream_results()` called for real, the lockbox
seal/redraw signature, and every model wrapper against *real* wrapper objects rather than a bare
`Ridge`. Port `test_insight_id_integrity` first, with a non-zero-count assertion so it cannot pass
vacuously.
*Gate: the suite fails when you deliberately break the cascade.*

**S3 · Settle row identity, then design `AnalysisProject`.**
One convention (labels), the cohort filter as a first-class field, per-model pipeline *specs*
rather than fitted objects, and a declarative DAG that can express partial invalidation.
*Gate: the new DAG reproduces both existing cascade implementations, including page 03's.*

**S4 · Extract the split block and the step state machine.**
`pages/06:380-760` becomes `ml/splits.py` for real; `theme.py:685`'s ten predicates become the
Project's readiness model.
*Gate: a headless script runs CSV → trained models with no Streamlit imported.*

**S5 · Cut the record singletons and add jobs.**
Ledger, provenance and lockbox lose their `st` accessors (~20 lines). Then the job queue, with
explicit RNG passed to every worker.
*Gate: two concurrent jobs produce identical results to two sequential ones.*

**S6 · Build the Router against EDA only.**
Not a move — new construction: lift triggers out of the pages, add a severity model, design gating
under the "high confidence only" rule.
*Gate: the router's chosen next question is explainable from the record alone.*

**S7 · Port the frontend one step at a time.**
Upload & Audit *last* among early candidates — not first, because its defect backlog is open.
Start with Preprocess or Train, which are well-specified and where jobs pay off most.

---

## 07 · What changed from the first draft

- **Upload & Audit is no longer the pilot.** Its ledger tail is open; piloting there means
  rebuilding on top of known-unfixed defects.
- **The Router moved later and grew.** The feasibility verdict says construction, not promotion,
  so it cannot be an early cheap win.
- **Characterization tests moved ahead of all extraction.** The safety net is weakest exactly
  where the blast radius is largest.
- **Fixing live bugs became step 1.** They are independent of the rebuild and wrong today.
- **"Cut the singletons first" was demoted** from step 1 to step 5. Still nearly free — but not
  worth doing before a test can prove the cut changed nothing.
