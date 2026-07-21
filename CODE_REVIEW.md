# Tabular ML Lab — Comprehensive Code Review

**Scope:** `main` @ `ddb922c` · reviewed 2026-07-16, ahead of conference demo
**Question under review:** Does the app deliver real utility, and is what it communicates (README, ARCHITECTURE.md, in-app copy, generated manuscripts) actually borne out by the implementation?
**Method:** 8-dimension claim-vs-implementation review by parallel subagents, followed by independent line-level verification of every critical/major finding (all evidence below re-checked against source), plus dynamic testing: 32 adversarial AppTest click-around scenarios, a stale-state probe, and the repo's own integration (73 passed / 3 skipped) and workflow (89 passed / 2 failed) suites on freshly installed current dependencies.

---

## Executive summary

The app is genuinely differentiated and much of its engineering is excellent: the per-model pipeline architecture is real and correctly implemented, the BCa bootstrap is a true implementation, page gating is robust (zero uncaught exceptions across 32 out-of-order navigation scenarios), and the session save/load layer is security-conscious well beyond its documentation. **The app provides real utility.**

However, its two flagship promises are not currently true:

1. **"No data leakage" is broken by the workflow's own page order.** Feature Engineering (page 03) and Feature Selection (page 04) run on the entire dataset because no split exists until page 06. Test rows shape the engineered features and choose the predictors, so reported held-out metrics are optimistically biased — while the manuscript asserts leakage-free rigor.
2. **"No stale state" is broken on several reachable paths.** Re-uploads, target changes, and cleaning actions can silently leave results from a previous dataset/outcome flowing into the UI and the exported manuscript.

A third theme: **the generated manuscript silently omits or misstates things it promises** (split/seed description, calibration, baselines, reproducibility manifest), which matters most for the app's core "CSV → defensible paper" pitch.

None of the critical findings crash the app; they produce *wrong or misleading science quietly*, which is worse for a tool whose value proposition is methodological trustworthiness. All are fixable, most with small diffs; a ranked pre-demo punch list is at the end.

---

## Remediation status (updated 2026-07-16, same day)

**All critical and major findings below have been fixed on this branch**, in a
series of commits following this document (see `git log` from `1f35cf7`).
Highlights:

- **Both leakage criticals** are resolved structurally by the **test-set
  lockbox** (`utils/test_lockbox.py`): seeded test rows are frozen at upload;
  EDA target views, feature-engineering fits (PCA/UMAP/binning/TDA), and all
  feature selectors are train-scoped; Train & Compare consumes the frozen
  labels as THE test set (verified end-to-end at the widget level:
  `tests/integration/test_lockbox_split.py`). An explicit, watermarked
  Exploratory mode preserves fast full-data screening, and toggling it in
  either direction resets downstream results so exploratory selection can
  never launder into a clean manuscript.
- **All state-integrity criticals** fixed: complete downstream reset (single
  helper), content-fingerprinted re-uploads, target/task-aware invalidation,
  cleaning actions that stick, ledger resolution rollback + EDA insight
  pruning so the manuscript can never assert actions invalidated mid-session.
- **All statistical majors** fixed: NaN-safe BCa bootstrap, per-fold CV
  preprocessing, honest seed sensitivity (re-splits per seed), working feature
  dropout (pipeline-respecting ablation), NN best-weight restoration,
  data-driven weighted-Huber, α wiring, revived model-selection guidance.
- **All manuscript majors** fixed: split/seed recorded, calibration and
  auto-computed baselines exported, honest best-model claims, no placeholders,
  corrected Strengths/Limitations, real reproducibility manifest (versions +
  SHA-256 data hash), lockbox stated in Methods and tracked for TRIPOD.
- **Docs corrected** (README/ARCHITECTURE/QUICKSTART) to match the code.

A second, Fable-model design-review round (fresh-context reviewers over the
coaching/provenance system and the UI/UX, plus an author-independent
adversarial pass over the fix series) produced a further wave of fixes:
one-vocabulary model families, blocker-safe acknowledge gates, per-insight
deduped grouped coaching, flash messages that survive reruns, data-scope
captions on every major number, lockbox/exploratory status on all result
pages, and validation-gated downloads.

**Verification state:** 515 tests passing (including 28 new regression tests
pinning every fix), 32/32 adversarial click-around scenarios, 13/13 lockbox
invariants, end-to-end lockbox widget tests.

**Ultrawide-shape wave (2026-07-17, driven by a 3000×34 browser stress
test):** the first-10 feature default (which silently dropped predictors on
wide data and made the EDA tiles describe a subset) now defaults to all
features; a stale sufficiency vocabulary meant the p≫n blocker never fired —
fixed against the real `DataSufficiencyLevel` enum and verified firing on
3000×34; per-rerun O(columns) scans (audit loops, `df.describe`, per-column
target correlations, uploaded-file re-parse) were vectorized and cached in
`utils/perf_cache.py`, cutting EDA warm reruns from ~10s to ~2.5s and
page-01 clicks from 11–22s to 3–6s at 3000 columns; RFE-CV (measured 80s at
this shape) auto-disables above 500 features; the LASSO path plot caps at
the 20 strongest paths; metric tiles no longer truncate and the Missing
tile's tooltip names the worst column. Verified end-to-end in a real
browser session (upload → EDA → Feature Selection) plus 8 new regression
tests. Test count: 525 passing.

**Bucket-1 hardening wave (2026-07-18):** downstream ultrawide smoke showed
training/CV/bootstrap/SHAP all fast at 28×3000 but permutation importance at
~141s (features × repeats evaluations) — both Train & Compare and
Explainability now show upfront wide-feature advisories (>500 features)
instead of appearing hung; a high-cardinality one-hot guard on Preprocess
warns and writes a self-healing ledger insight when categoricals exceed 50
levels; the `use_container_width` deprecation was swept repo-wide (the API
is already past its removal date); a new AST-based CI scan
(`tests/test_insight_id_integrity.py`) fails on insight ids that nothing
produces — it caught a third live instance of the stale-identifier class on
its first run (`train_prefer_simpler`, produced via a dict-literal pattern
the scanner now understands); and long cold loads are designed out rather
than hidden: every cached heavy computation carries a named spinner
("Profiling dataset structure (one-time per dataset)…") and wide datasets
get an explicit first-visit caption on EDA. Test count: 532 passing.

**Manuscript-trust wave (2026-07-18):** the manuscript layer now operates as
a compiler from provenance to prose with an explicit ownership contract.
(1) Register separation: `Insight.manuscript_text` carries reviewer-facing
phrasing for every auto-generated producer (EDA, coach diagnostics,
preprocessing guards); coaching voice ("a reviewer would question…",
"consider…") can no longer reach the Discussion verbatim, and the regex
cleaner is now a fallback rather than the mechanism. (2) Overclaims
removed: the Conclusions no longer assert the model "can effectively
predict" regardless of performance — they state the recorded result and
hand the adequacy judgment to the author; negative R² is reported as
"below a mean-only baseline"; "strongest" is claimed only when multiple
models were actually compared. (3) Author-owned passages are standardized
`[AUTHOR REQUIRED — …]` scaffolds that cite the study's own evidence
(headline metric for prior-work comparison, top predictors with an
explicit not-causal guard for implications); page 10 counts remaining
author inputs. (4) Every draft ships an ownership preamble (markdown
blockquote / LaTeX comments) stating the compiled-vs-author contract.
(5) A per-section **evidence map** artifact traces each compiled section to
the recorded events and values behind it, admitting "NOT RECORDED" where
the pipeline holds no evidence — downloadable beside the draft.
(6) `ManuscriptDraft.to_latex()` escapes LaTeX specials before command
conversion, so hostile column names (`feat_0042`, "15%") compile.
17 new tests in `tests/test_manuscript_trust.py`. Test count: 549 passing.

**Model-coach assessment wave (2026-07-19):** an empirical audit (the
coach's verbatim output across seven dataset shapes) found the visible
coach reacting to whichever signal it checked first rather than the
dataset's dominant constraint: feature outliers hijacked the pick to Huber
(which is robust to TARGET outliers — a statistical confusion), a 34×300
dataset got an unpenalized robust regression with no mention of p≫n,
a 6%-minority classification with EPV 1.6 was told logistic regression's
"probability outputs are well-calibrated" with zero mention of imbalance,
the collinearity branch was dead (the profile has no such field), and the
scaling advice told users to do what the default pipeline already does.
`select_top_picks` was rewritten shape-first: it now returns a HEADLINE
naming the dominant constraint with the numbers (p≫n, EPV vs the ≥10
guideline, imbalance ratio, small-n CV-spread warning), picks penalized
models for wide data (LASSO + Ridge comparison, trees skip-listed with an
explanation), triggers Huber only on measured outcome outliers, keeps the
lineup deliberately small at low EPV, claims calibration only when the
event count supports checking it, and gives skip-list reasons that cite
the dataset's own numbers. Preprocessing insights now acknowledge pipeline
defaults, carry a small-n outlier-detector caveat, and use a consistent
native-NaN vocabulary. The parallel dead recommendation apparatus (~830
lines never rendered by any page) was removed, closing the drift surface.
10 new shape-awareness tests. Test count: 568 passing. (Known pre-existing:
one stale pandas-3 dtype expectation in `scripts/smoke_check.py`'s
categorical-target check — the pytest suite covering the same path is
green.)

**Known remaining work (deliberately deferred, none demo-blocking):**
explicit theory anchors at every EDA producer (inference fallback is
currently benign but fragile); coaching badge counts vs scoped body
coherence; removing or wiring the dead `DatasetDB` /
`decision_curve_analysis` code; Pandas 4 deprecation warnings in
`data_processor.py`. (Resolved since first written: the manuscript-register
separation, the insight-ID CI scan, and `ManuscriptDraft.to_latex()`
escaping — see the manuscript-trust wave below.)

---

## What genuinely holds (verified strengths)

| Claim | Verdict | Evidence |
|---|---|---|
| Per-model pipeline forking (core differentiator) | **Holds** | `05_Preprocess.py:738-874` builds distinct pipelines per model; `06_Train_and_Compare.py:1050-1059` fits each model's own pipeline on train only and trains on its own transform |
| Pipelines fit on training data only | **Holds** | `06:1056` `model_pipeline.fit(X_train)`; val/test only transformed |
| BCa bootstrap is genuine (z0 + jackknife acceleration) | **Holds** | `ml/bootstrap.py:55` (bias correction), `:60-64` (acceleration), `:70-85` (adjusted percentiles); run on held-out test predictions (`06:1707-1711`) |
| Optuna tunes on validation, never test | **Holds** | `06:988-993` objective scores `X_val_transformed` |
| Target transform fit on training targets only | **Holds** | `06:478` |
| "22 models" | **Holds** | exactly 22 `ModelSpec` registrations in `ml/model_registry.py:180-733` |
| Compilable LaTeX (escaping of user strings) | **Holds** | `ml/latex_report.py:129-140` `_escape_latex` applied to prose, tables, abstract |
| Privacy: "Nothing is written to disk" | **Substantially holds** | The SQLite layer (`utils/dataset_db.py`) and disk-persistence module (`utils/persistence.py`) are **dead code with no callers**; page 01 uses `utils/session_projects.py` ("All state lives in st.session_state — nothing written to disk"); session archives are built in-memory for browser download; PDF compiles in an auto-cleaned tempdir. Caveats below. |
| Session save/load security | **Holds (better than documented)** | `utils/session_manager.py` uses ZIP/JSON/Parquet, **rejects** pickle, validates against zip-bombs/path traversal, `_NEVER_PERSIST` drops fitted models |
| Out-of-order navigation is safe | **Holds (dynamically verified)** | 32/32 adversarial AppTest scenarios (cold session, data-only jumps, classification, degenerate data: n=25, constant col, all-NaN col) — zero uncaught exceptions; every downstream page gates with `st.stop()` |
| Optional deps (giotto-tda, umap) degrade gracefully | **Holds** | function-local imports with `st.error` guards (`03:843-845`, `03:1048`) |
| Four views of one insight ledger | **Holds** | `utils/insight_ledger.py:956,1016,1066,1158` all consume the same `_insights` list |
| Advertised features reachable (TDA, UMAP, PCA, Optuna, subgroup, external validation, stability, consensus) | **Holds** | verified reachable in UI (exceptions: decision curves and calibration placement — see findings) |

Privacy caveats (README wording, not behavior): "All processing happens in your browser session" is inaccurate for server-side Streamlit — for the hosted demo URL, uploaded CSVs are processed on the remote server; and the cloud-LLM opt-in sends column names and up to ~20 raw rows to the chosen API (`utils/llm_ui.py:334-349,443-448`), which the parenthetical only loosely discloses.

---

## Critical findings (verified, wrong-science class)

### C1. Feature Engineering leaks the test set into engineered features
`pages/03_Feature_Engineering.py:991` (PCA `fit_transform` on all rows; likewise UMAP `:1024-1030`, polynomial `:342-347`, binning `:702-709`, TDA `:850-872`) → persisted as `df_engineered` (`:1285-1294`) → served by `get_data()` (`utils/session_state.py:194-206`) → split happens only later (`06:424-448`). PCA loadings, bin edges, scaler statistics, and manifold embeddings are all estimated using rows that later become the test set. Reported test metrics are optimistically biased; the manuscript on the same run claims "no data leakage" (README:76,143).

### C2. Feature Selection chooses predictors using the test rows
`pages/04_Feature_Selection.py:121-129` builds `X` from **all rows** (median-imputed on all rows), then LASSO / RFE-CV / univariate / stability selection run on it (`:206-248`; `ml/feature_selection.py`), and the chosen set is written into the modeling config (`:404-406`) that page 06 splits on. LASSO/RFE/univariate are checked by default, so this is the default path. Classic selection leakage (Ambroise & McLachlan 2002; ESL §7.10.2): invalidates the "held-out" evaluation of the selected feature set — and is most severe on exactly the wide datasets the step is designed for.

### C3. The generated Methods section omits the split design and seed
`record_split` (`utils/workflow_provenance.py:286`) is called **only in tests** — no page invokes it (repo-wide grep). `NarrativeEngine._gen_study_design` gates the entire split/seed description on `if split_prov:` (`ml/narrative_engine.py:485-562`), so the exported manuscript's Methods contains no train/val/test statement and no random seed — the most basic ML-methods element, and one the app explicitly promises. Tests create the provenance manually, giving false confidence.

### C4. Model-aware coaching — the flagship Layer-2 behavior — never activates
`utils/coaching_ui.py:34` reads `st.session_state.get("selected_models", [])`; **no code anywhere writes that key** (repo-wide grep). The model-grouping branch (`:112`) is unreachable, `get_for_models`/`coaching_summary_for_models` are dead in the UI, and coaching renders the same flat list regardless of which models the user selects. The "BMI skew matters for Ridge/MLP but not RF" story in ARCHITECTURE.md cannot happen in the running app.

### C5. Cascade invalidation misses result keys → manuscripts can mix datasets
`reset_data_dependent_state()` (`utils/session_state.py:229-291`) — the documented single invalidation entry point — never clears `shap_results`, `bootstrap_results`, `baseline_results`, `sensitivity_seed_results`, `hypothesis_test_results`, `methodology_log`, or `workflow_provenance`. (Page 01's own ad-hoc clear list at `01:1602-1616` *does* clear `shap_results`/`sensitivity_seed_results`, proving they're understood to be downstream — but it too misses `hypothesis_test_results`, `bootstrap_results`, `baseline_results`.) **Dynamically demonstrated:** after a schema-change re-upload, stale values from the prior dataset survive; user retrains on new data, passes Report Export's gate, and the manuscript embeds the previous dataset's SHAP plots, robustness numbers, and provenance narrative with no warning. Also note `07:600` writes `shap_results`, confirming staleness is reachable.

### C6. Changing the target (or task type) does not invalidate anything
Invalidation on page 01 is keyed solely on a hash of `feature_cols` (`01:1598`). Target selectbox and task-type override don't participate. Switch the target from `glucose` to `cholesterol` (when the new target wasn't a selected feature): models, splits (`y_train` still glucose), metrics, and SHAP all survive, and Report Export labels the old model's results with the new outcome name. The validator's metric-vs-task check catches only a regression↔classification flip, not a same-task target swap. Silent scientific-integrity failure.

### C7. Same-schema re-upload is ignored downstream
`set_data()` triggers the reset only when the **column set** changes (`utils/session_state.py:222-226`). Re-uploading a corrected file with identical columns (the standard path, `01:726`) resets nothing: trained models, results, and the ledger keep describing the old data — and if Feature Engineering was applied, `get_data()` keeps returning the **old engineered dataframe**, so every page silently ignores the new upload. **Dynamically demonstrated.** Related: the one-click cleaning actions pass `is_schema_change=False` (`01:1380`), so e.g. "Drop duplicate rows" after FE has zero effect on what is trained (raw_data is cleaned; `df_engineered` still drives everything). The intended safety net, `reconcile_state_with_df()`, is imported but never called by any page (only `scripts/smoke_check.py`).

---

## Major findings (verified)

### Statistical reporting
- **CV preprocessing fit outside the folds** — pipeline fit once on all of train (`06:1056`), the transformed matrix fed to `cross_val_score` (`06:1278-1281` → `ml/eval.py:108`). Fold-held-out rows contributed to imputer/scaler/PCA/outlier-bound fitting; CV means and the paired model-comparison test are optimistically biased. (Target transform is correctly per-fold via `TransformedTargetRegressor`; features are not.)
- **AUC bootstrap CI collapses to `[nan, nan]` on imbalanced/small test sets** — the AUC helper *returns* NaN for single-class resamples (`ml/bootstrap.py:231-234`) instead of raising, so the exception-substitution at `:138` never fires and `np.percentile` propagates NaN into both bounds (`:84-85`). At 90/10 prevalence with n≈50, effectively guaranteed across 1000 resamples. Renders as `[nan, nan]` in the UI CI table and the manuscript.
- **Model-selection guidance and poor-performance diagnostics are dead code** — `metric_col = 'AUC (val)' / 'R² (val)'` (`06:1925`) never matches `comparison_df`'s real columns (`RMSE`, `R2`, `ROC-AUC`, built at `06:1597-1615`), so the bootstrap-CI-overlap "How to Choose Your Model" block and the diagnostics assistant never execute; after training, the page shows "Train models to see selection guidance."
- **α selector ignored by 4 of 6 hypothesis tests** — two-sample (`09:489`), ANOVA (`:634`), categorical (`:750`), paired (`:961`) hardcode `p < 0.05` and the literal text "at α=0.05"; only correlation honors the sidebar `alpha_level` (`:294-337`).
- **Seed sensitivity measures the wrong thing vs. its own description** — UI text promises detection of "lucky/unlucky train-test split" (`08:88-92`); code re-seeds the estimator on a **fixed** split (`08:121-127`). Deterministic models (Ridge, GLM, kNN, LDA, NB) show CV=0% → "Highly robust / publication-ready" verdict regardless of actual split fragility.
- **Feature Dropout is broken for every selectable model** — `clone(model_obj)` (`08:382`) is called on the app's wrapper objects, which don't implement `get_params` (only the NN shims do — grep of `models/`), so sklearn raises `TypeError` per feature; swallowed → all impacts 0 → empty results. The seed block correctly unwraps via `get_model()` (`08:104`); dropout doesn't. Even if cloned, the design compares a pipeline-transformed baseline against raw-feature retrains — not a valid comparison.
- **NN early stopping never restores best weights** — `best_model_state = self.model.state_dict().copy()` (`models/nn_whuber.py:416,511`) is a shallow copy of live tensors mutated by every optimizer step; `load_state_dict(best_model_state)` (`:515,528`) is a no-op. Returned model is last-epoch (up to `patience` epochs past optimum) while `best_val_rmse` reports the best epoch — a metric/model mismatch. Fix: `copy.deepcopy` or per-tensor `.clone()`.
- **Weighted-Huber loss is hardcoded to a glucose target** — `t0=180.0, s=20.0, alpha=2.5` (`models/nn_whuber.py:68`), call sites pass no overrides (`:445,478`). For targets not on a ~180 scale the weighting silently degenerates to plain Huber; for targets spanning 180 it biases toward mid-range values — neither matches the "emphasizes high-value targets" tooltip.
- **LASSO selection is broken on current scikit-learn** — `LassoCV(n_alphas=...)` (`ml/feature_selection.py:40`); the argument was removed in sklearn 1.9. `requirements.txt` allows `>=1.3.0`, README badges 1.8+ — fresh installs get a raw `TypeError` in a warning box and LASSO silently drops out of the consensus. (Found by running the repo's own test suite: 2 failures, both here.)

### Manuscript integrity
- **Calibration never reaches the export** — computed and displayed on page 06 (`:1804-1833`) but never written to session state; `calibration_results` is only ever **read** (`10:760,778`; `publication.py:1470`). The promised "calibration analysis" is absent from every manuscript.
- **Baselines are manual and never exported** — behind a button in a collapsed expander (`06:1747-1753`), stored as `baseline_results`, which Report Export never reads; `publication.py:1087-1094` looks in `model_results`, where baselines never appear. "Automatic comparison against null baselines" is neither automatic nor in the paper.
- **Reproducibility over-claim** — the LaTeX supplementary asserts the package includes "software versions, random seeds, and data hashes" (`ml/latex_report.py:987`); the exported bundle contains seed + configs but no versions and no data hash; `generate_reproducibility_manifest` (`utils/persistence.py:116`) — which would capture them — has zero callers.
- **"Best overall performance" asserted for whatever model the user picks as primary** — `narrative_engine.py:1008` emits it unconditionally; the comparative block can then emit "highest prediction error" for the same model (`:1056-1071`). Choosing an interpretable-but-not-top primary (which the UI help text encourages) produces a self-contradicting Results section.
- **Permanent placeholder in generated Results** — `"[Feature importance analysis pending — requires explainability provenance integration.]"` appended unconditionally (`narrative_engine.py:1090-1092`); flows into the promoted Methods download; not caught by the validator (which only scans LaTeX for `[NOTE`/`[PLACEHOLDER`).
- **Discussion "Strengths" are inverted** — unresolved `severity=='info'` insights are classified as strengths (`insight_ledger.py:1118`) while genuine EDA opportunities are created `resolved=True` (`02_EDA.py:392,413,435,450,465`) and skipped (`:1111`); result: "N features are heavily skewed" can appear under Strengths while real positives are omitted.
- **Skew-resolution ID mismatch** — EDA creates `eda_skew_group` (`02:367`); Preprocess resolves `eda_skew_individual`/`eda_skew_batch` (`05:1039`), which are never created. Handling skew on the Preprocess page (a documented path) never resolves the insight. Worse, the same loop resolves `eda_target_skew` (a **target** insight) whenever a **feature** power transform is configured (`05:1035-1048`) — the Methods can then claim a target transform that never happened.

### Documentation truthfulness
- **ARCHITECTURE.md describes a persistence/security model the code deliberately does not implement** — "SQLite for project/dataset persistence" (`:13,171`) is dead code; sessions are described as "pickle-based… preserving trained models" (`:177-178`) while the implementation is a no-pickle ZIP/Parquet format that **rejects** `.pkl` and never persists models. The code is better than the doc; the doc is the liability (especially for university-IT evaluators the README courts).
- **README advertises "decision curves" on Explainability** — `decision_curve_analysis` (`ml/calibration.py:250`) has zero callers; no UI control exists. "Calibration" is listed on page 07 but lives on page 06.
- **ARCHITECTURE.md file map lists `ml/preprocessing.py` and `ml/training.py`** — neither exists.
- **QUICKSTART tells users to run `preflight.py`** (4 places) — the file does not exist; also contains a stale project name (`glucose-mlp-interactive`).

### Demo/UX robustness
- **AI "Deep Analysis" fails out of the box** — default Ollama model `qwen3.5:9b` (`utils/llm_ui.py:21`) must be pre-pulled; there is no availability check and the sidebar value resets each session; failure degrades to a generic warning (the helpful `__unavailable__` guidance path is unreachable — that value is never set). Also: if Ollama is installed but stopped, the app `Popen`s `ollama serve` on the request thread and can block up to 120s behind a spinner that promises ~60s (`:522-530,584,740`).
- **Task-type detection mislabels low-cardinality integer targets** — any integer target with ≤10 unique values → "classification" with a confident green banner (`ml/triage.py:53-56`; `01:1570`), regardless of n. Counts/scores in large datasets get steered to classifiers; extreme case: 10-unique targets hard-error at the stratified split.
- **`example_data.csv` is a trap** — 10 rows (below the app's stated 50-row minimum), referenced by nothing in the app; the built-in practice datasets come from `get_builtin_datasets()` instead. Reaching for the obviously-named file on stage dead-ends.
- **Optuna shows no per-trial progress** — up to 30 trials behind a static status line (`06:1067`); reads as a hang for minutes (opt-in, pre-warned).
- **Figure export can silently drop or stall** — `save_plotly_fig` returns None on any exception (figures silently missing from ZIP) and kaleido's known hang mode has no timeout (`10:2030-2035`).

### Corrections from verification (findings killed or downgraded)
- **Refuted:** "Table 1 silently drops categorical p-values when a variable has missing values" — `pd.crosstab` on independently-`dropna`'d *Series* aligns on the index intersection (verified empirically on pandas 3.0.3); the complete-case p-value is computed correctly.
- **Retracted (my own earlier interim note):** "uploads write metadata to `datasets.db` on disk" — `DatasetDB` has no live callers; the running app persists nothing to disk.
- **Benign:** the jackknife cap at 200 in the bootstrap fills the tail with the mean (`ml/bootstrap.py:150-151`) — an approximation, not a bug.
- Minor (reported by reviewers, spot-checked, low stakes): `p = 0.0000` formatting in two report paths (`publication.py:1500`; `10:1255`); multiclass calibration uses last-class proba against multiclass labels (`ml/calibration.py:56-76`); Fisher's odds ratio labeled χ² in Table 1 footnotes (`09:762`); baselines computed on transformed-target scale while the leaderboard is back-transformed (`06:1756-1777` vs `:1215-1228`); theory-anchor `feature_scale` key doesn't exist (works by accidental text match); several theory breadcrumbs point to section titles that don't exist on page 11; coaching renders via the shared component only on pages 04–09; the validator's "split counts reconcile" check compares a value to itself; vacuous `Pandas4Warning` deprecations in `data_processor.py` (future breakage risk).

---

## Dynamic testing summary

| Test | Result |
|---|---|
| 32 adversarial click-around scenarios (cold/partial/classification/degenerate states, all 12 pages) | **32/32 pass, zero uncaught exceptions** |
| Stale-state probe (same-schema re-upload; schema-change reset) | Confirmed C5/C7: same-schema re-upload resets nothing; schema-change reset leaves 5 result keys stale |
| Repo integration suite (`tests/integration`, `test_page_imports`) | 73 passed, 3 skipped |
| Repo workflow suite (`tests/workflow`) | 89 passed, **2 failed** (LASSO `n_alphas` — sklearn 1.9 incompatibility) |

Note on test fidelity: `tests/integration/conftest.py` injects a *bare sklearn Ridge* into `trained_models`, but the real Train page stores wrapper objects — the suites exercise downstream pages against a state shape the app never produces, which is how the Feature Dropout `clone()` breakage stayed invisible.

---

## Pre-demo punch list (two weeks, ranked by risk × effort)

**Do before the demo (small, safe diffs):**
1. `LassoCV` sklearn-1.9 fix (`alphas=` or pin `scikit-learn<1.9` in requirements) — one line; unbreaks a headline feature on fresh installs.
2. Add the missing keys to `reset_data_dependent_state()` (`shap_results`, `bootstrap_results`, `baseline_results`, `sensitivity_seed_results`, `hypothesis_test_results`, `methodology_log`, `workflow_provenance`) and mirror them into page 01's clear list.
3. Include target/task-type in the invalidation hash (`01:1598`).
4. Call `record_split` from page 06 after the split — restores the Methods split/seed paragraph.
5. Remove the unconditional placeholder and the unconditional "best overall performance" sentence in `narrative_engine.py` (guard on actual best-by-metric).
6. Write `calibration_results` to session state on page 06; read `baseline_results` in Report Export.
7. Fix the `metric_col` names (`06:1925`) to `'ROC-AUC'`/`'R2'` — revives model-selection guidance.
8. README edits: drop "decision curves," move "calibration" to Train & Compare, fix "browser session" phrasing, qualify "no data leakage" until the split-first change ships. ARCHITECTURE.md: fix persistence/session/file-map sections. QUICKSTART: remove `preflight.py`.
9. `deepcopy` fix for NN early stopping; NaN-drop fix in `_bca_ci` (use `np.nanpercentile` + drop NaN resamples).
10. Wire the α selector into the four hardcoded verdicts; reword the seed-sensitivity header to say what it measures.

**Demo-day behavioral rules (no code needed):**
- Never re-upload data or change the target mid-session — restart the browser session to switch datasets.
- Use a built-in practice dataset, not `example_data.csv`.
- Pre-pull the exact Ollama model tag and test a "Deep Analysis" click before going on stage; keep Optuna off unless you narrate the wait.
- Don't click "Run Feature Dropout" (all-zero output); skip target trimming; avoid demoing weighted-Huber on non-glucose data.
- Show Explainability (page 07) — it's a genuine strength, correctly pairing each model with its own pipeline and post-transform feature names.

**The structural fix (start now, land post-demo unless it lands cleanly early):**
Lock the test set at upload — draw and freeze seeded test indices on page 01 right after target confirmation; pages 02–04 scope all target-aware computation (target correlations, FE fits, all selectors) to training rows with a one-line caption; page 06 consumes the frozen indices. Add an explicit, labeled "Exploratory mode" toggle for full-data screening that watermarks downstream metrics and the manuscript. This single change repairs C1, C2, and (via a natural `record_split` call site) C3, and converts the app's biggest liability into its best stage moment: "the app quarantined the test set before it let me touch anything."

---

## Design principles for future front-end changes

Distilled from this review's confirmed defects; each rule traces to a real instance above.

1. **Defaults carry the rigor; captions carry the disclosure; options carry the speed.** The zero-click path must be the methodologically correct one. Never make rigor opt-in.
2. **State first, widgets second.** Before drawing UI, name the session-state keys: who writes, who reads, who invalidates; extend ARCHITECTURE.md's contract table in the same PR. (Root cause of: `calibration_results` read-never-written, `selected_models` read-never-written, skew-ID mismatches, reset-list gaps.)
3. **Every number on screen names its data** — train/val/test/full-cohort chip or caption. (Root cause of: CV bias invisibility, seed-sensitivity mislabel, baseline scale mismatch.)
4. **UI copy is a contract with code.** Any sentence describing a computation is part of the diff that changes the computation. (Seed-sensitivity header; Ollama sidebar caption; spinner "~60s" vs 120s timeout.)
5. **Never render a dead branch; gate visibly.** "Requires X — run it on page Y" beats silent omission. (Coefficient table, selection guidance, calibration in export.)
6. **The manuscript is the regression test.** Every UI result either flows to the export or is explicitly UI-only; add an export-side assertion when introducing a new result key. (Calibration, baselines, best-model contradiction.)
7. **Docs are claims.** README/ARCHITECTURE statements should be greppable to their implementation; when code moves on, the doc edit belongs in the same commit. (SQLite/pickle drift, decision curves, preflight.py, file map.)

---

*Review conducted with parallel subagent analysis, adversarial verification, independent line-level re-verification of all critical/major findings, and dynamic AppTest execution. Two subagent findings were refuted during verification and are documented above rather than silently dropped.*
