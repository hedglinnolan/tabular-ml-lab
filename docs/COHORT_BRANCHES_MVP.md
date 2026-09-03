# Cohort Branches — MVP Implementation Plan

*Drafted 2026-09-03 from a code audit of the cohort-run feature. Every claim below cites the line it was read from. Line numbers are as of commit `081c2cc` (main).*

## 0. What this MVP does and does not do

**Today** a cohort run is a destructive sequence: "Now run the same analysis on Male" wipes every result the Female run produced and rebuilds from the shared decisions. What survives of Female is a `CohortRun` holding the label, two row counts, the constant-in-group predictors, and **one metric** (`pages/06_Train_and_Compare.py:3047-3053`). The export and manuscript describe whichever cohort is active at download time; nothing in the export path reads `cohort_runs_done`.

**After this MVP** a cohort run is a persistent branch:

- Switching cohorts archives the current branch and restores the target. Nothing fitted is lost.
- Pages 07 (Explainability) and 08 (Sensitivity) carry a cohort picker; picking a cohort *is* the switch.
- The exported bundle carries every banked branch (`cohorts/<label>/…`), a comparison table, and the multiplicity caveats. `manuscript.tex` stays single-cohort with an explicit pointer — this is the **(b+)** scope.
- The seal's open count is reported per cohort, with the study total alongside.
- Four manuscript-correctness bugs in the current design are fixed in the same change.

**Not in this MVP:** the up-front cohort declaration on page 01, seal stratification on cohort, a multi-cohort `manuscript.tex`, cross-session persistence of branches. See §10.

---

## 1. The invalidation rule (the load-bearing decision)

Persistence is easy to build and easy to get wrong. The failure mode is not "a branch was lost" — it is "a branch survived a change that made it stale, and its numbers reached the manuscript." So the first thing to fix is *what invalidates a branch*, and only then how to store one.

### 1.1 The graph already exists and is test-pinned

`turbotab/cascade.py` declares the analysis as a dependency-ordered stage graph with an explicit `produces` set per stage:

```
data → feature_engineering → feature_selection → preprocessing → split → training → {explainability, evaluation} → report
data → analysis           (EDA, Table 1, profile — descriptive, stale on any data change)
data → coach              (probe verdicts — describe the data they were measured on)
```

`cascade.keys_for_reset_downstream_results(flags)` is pinned **key-for-key** against the production `utils/session_state.py::reset_downstream_results` by `tests/integration/test_cascade_dag_equivalence.py::test_the_dag_matches_the_production_cascade`, for every flag combination. A second gate, `test_the_dag_declares_every_key_the_production_cascade_clears`, names the 14 keys the production reset clears that the graph does not yet own (`_NOT_YET_DECLARED_IN_THE_DAG`, same file :182-206). So the result-key inventory is **verified, not hand-maintained**, and this plan derives from it rather than re-listing it.

The graph has no cohort concept. That is the one thing this MVP adds to it.

### 1.2 Every caller of the reset, classified

> **Corrected during implementation.** This section originally listed fourteen call sites. An AST scan that resolves import aliases — page 01 calls the reset as `_rdr` inside a widget callback, which a name grep misses — finds **nineteen, in nine files**. The five that were missed are marked ★. None of them changes the conclusion; every one is a "drop all branches" caller, and each got the safe behavior for free from §1.3 without being touched. That is the argument for the rule, made by the plan's own undercount.

| # | Call site | What changed | Kind |
|---|-----------|--------------|------|
| ★ | `pages/01_Upload_and_Audit.py:183`, `:208` — a new file is loaded | the data | **drop all branches** |
| 1 | `pages/01_Upload_and_Audit.py:1546` — config save, feature/target hash changed | shared decision (features, target) | **drop all branches** |
| 2 | `pages/01_Upload_and_Audit.py:1750` — exploratory-mode flip (called as `_rdr`) | quarantine regime | **drop all branches** |
| ★ | `pages/02_EDA.py:98` — staleness guard, target column left the frame | the data | **drop all branches** |
| 3 | `pages/03_Feature_Engineering.py:319` — FE Reset button | shared decision (recipe) | **drop all branches** |
| 4 | `pages/03_Feature_Engineering.py:332` — FE Skip button | shared decision | **drop all branches** |
| 5 | `pages/03_Feature_Engineering.py:1394` — FE Apply | shared decision | **drop all branches** |
| 6 | `pages/04_Feature_Selection.py:644` — FS apply (consensus) | shared decision (selection) | **drop all branches** |
| 7 | `pages/04_Feature_Selection.py:718` — FS apply (manual) | shared decision | **drop all branches** |
| 8 | `pages/05_Preprocess.py:981` — row-filter rule changed rows | shared rule (plausibility bounds) | **drop all branches** |
| ★ | `utils/session_state.py:411` — `set_data`, schema changed | the data | **drop all branches** |
| 9 | `utils/session_state.py:436` — `set_data`, same schema, values changed | the data | **drop all branches** |
| 10 | `utils/session_state.py:759` — `reset_data_dependent_state`, new dataset | the data | **drop all branches** (already pops `cohort_runs_done`) |
| ★ | `utils/session_manager.py:766` — a saved session is being restored | the whole session | **drop all branches** |
| 11 | `utils/test_lockbox.py:899` — seal stale, sealed labels left the frame | the seal | **drop all branches** |
| 12 | `utils/test_lockbox.py:1194` — seal redrawn with different labels | the seal | **drop all branches** |
| 13 | `utils/cohort_ui.py:235` — `_switch_to` (page-01 chooser) | who the rows are | **archive / restore** |
| 14 | `utils/cohort_ui.py:434` — `_advance_to` ("Now run on X") | who the rows are | **archive / restore** |

Seventeen callers mean "the question changed under every branch." Two mean "same question, different people." There is no third kind.

**Case 12 deserves emphasis.** `ensure_lockbox` refuses to redraw *only while a cohort is active* (`utils/test_lockbox.py:996-1016`). The sequence "run Female, run Male, click *Go back to analyzing everyone*, change the test fraction" therefore redraws the seal and fires this reset — with no cohort active. Every archived branch was scored against a seal that no longer exists. Under the current code this also resets `opened_count` to 0 (`utils/test_lockbox.py:1182-1183`), after which the Methods says the set was "accessed only for the final evaluation" (`ml/narrative_engine.py:909-911`). The archive must go in this case, and §4 makes the count survive it.

### 1.3 The rule: default-destructive, opt-in preservation

Do **not** teach twelve call sites about branches. Make the reset itself drop the archive, and let the two cohort-switch callers opt out:

```python
def reset_downstream_results(clear_feature_engineering=True,
                             restore_pre_fe_features=True,
                             clear_feature_selection=True,
                             preserve_branches=False):          # NEW
    ...
    if not preserve_branches:
        st.session_state.pop(BRANCH_ARCHIVE_KEY, None)
        # A branch is only comparable under the question it answered.
        # Every caller that reaches here without the flag changed that question.
```

Only `utils/cohorts.py::switch_branch` (§3.2) passes `preserve_branches=True`, and only when the target is a *new* branch that needs the ordinary reset to make room. Every existing caller, and every future one written without knowing branches exist, gets the safe behavior for free. That property — a new caller cannot leak a stale branch by omission — is the reason to put the drop inside the reset rather than beside it.

### 1.4 Why not extend `CohortRun.question` instead

`CohortRun.question` is `(column, target_col, task_type, _raw_data_fingerprint)` (`utils/cohorts.py:409-411`), and `completed_runs()` filters banked runs by it at read time (`:552`). It was designed for exactly this problem and it is the right idea for the *comparison table*. But it does not see preprocessing config, the feature selection, the FE recipe, the seal signature, or the exploratory flag — every one of the twelve callers above can change the question without moving the fingerprint. Extending the fingerprint to cover all of them means hashing widget state and pipeline configs, which is fragile and would drift. The archive drop in §1.3 covers the same ground with one line and no hashing. Keep `question` as the *comparison* guard; use the reset as the *staleness* guard.

### 1.5 Consequence for the two switch callers

`_switch_to` and `_advance_to` currently do: pop `filtered_data` → `stage_for_replay` → start/clear cohort → record provenance → `reset_downstream_results(clear_feature_engineering=True)` → `restore_decisions` → rerun (`utils/cohort_ui.py:200-238`, `:404-436`). Both become a call to `switch_branch(target)`, which decides whether the target has a snapshot (restore it; no reset, no replay) or is new (snapshot current, then the existing reset-and-replay path with `preserve_branches=True`).

The replay machinery (`utils/replay.py`) is untouched. It already answers "what are the shared decisions" (§2.3) and is only needed when a branch is *created*.

---

## 2. The branch / shared boundary

### 2.1 The rule

**A key is per-branch iff its value was computed from rows.** Decisions — a recipe, a config, a selection, a pick — are shared and never archived. Fits, frames, matrices, scores, figures and the records that describe them are per-branch.

This cuts *across* the cascade's stages rather than at a boundary, because `preprocessing` owns both `preprocessing_config` (a decision) and `preprocessing_pipeline` (a fit), and `feature_engineering` owns both the recipe (shared, in `fe_recipe`) and `df_engineered` (that cohort's rows). The key names already encode the distinction; the archive just has to respect it.

### 2.2 Per-branch keys (derived)

The branch key set is **computed at import, not hand-listed**, so it cannot drift from the graph the tests pin:

```python
BRANCH_KEYS = (
    cascade.all_result_keys()
    | _NOT_YET_DECLARED_IN_THE_DAG          # the 14, promoted out of the test into cascade.py
    | {"feature_engineering_applied", "engineered_feature_names"}   # describe df_engineered
) - SHARED_DECISION_KEYS
```

Spelled out, by stage, so the reader can check it against `turbotab/cascade.py`:

| Stage | Per-branch keys | Note |
|-------|-----------------|------|
| feature_engineering | `df_engineered`, `engineering_log`, `feature_engineering_applied`, `engineered_feature_names` | the frame is this cohort's rows; `get_data()` filters it (`test_cohort_runs::test_engineered_frame_is_filtered_too`) |
| preprocessing | `preprocessing_pipeline`, `preprocessing_pipelines_by_model`, `preprocess_built_model_keys`, `preprocessing_summary` | fitted objects and the list of what has them |
| split | all 20 keys in `cascade.stage("split").produces`, plus `split_trim_record` | this branch's slice of the one seal |
| training | `trained_models`, `model_results`, `fitted_estimators`, `fitted_preprocessing_pipelines`, `cv_results` | |
| explainability | `permutation_importance`, `partial_dependence`, `explainability_robustness`, `shap_results`, `shap_matplotlib_figs`, `pdp_results`, `external_validation_results` | |
| evaluation | `bootstrap_results`, `baseline_results`, `calibration_results`, `sensitivity_seed_results`, `sensitivity_dropout_results`, `sensitivity_dropout_baseline`, `bland_altman_results` | |
| analysis | `eda_results`, `eda_insights`, `dataset_profile`, `dataset_profile_scope`, `hypothesis_test_results`, `table1_df`, `table1_metadata`, `custom_table1_tests`, `table1_custom_test_footnotes` | descriptive of this cohort's rows |
| coach | `coach_probe_result`, `_coach_applied` | the probe verdict describes the rows it measured |
| report | all 11 keys in `cascade.stage("report").produces`, plus `manuscript_export_context`, `compiled_pdf`, `manuscript_table1_df`, `manuscript_table1_metadata` | describe this branch's models |
| (row filter) | `filtered_data` | this cohort's surviving rows |

Plus three **non-key slices** that are per-branch and need their own snapshot/restore (§3.4): the insight ledger's entries whose `source_page ∈ BRANCH_PAGES`, the `methodology_log` entries whose step maps to those pages, and the provenance sections from `workflow_provenance.downstream_sections()` minus the two flagged shared ones.

### 2.3 Shared keys (never archived)

Everything `utils/replay.py::stage_for_replay` captures (`:165-206`, `:209-247`) is by construction a decision:

- `fe_recipe` (the `Step` list), `engineered_feature_transforms`
- `preprocessing_config`, `preprocessing_config_by_model`, every `preprocess_*` widget key except `preprocess_built_model_keys`
- model picks, per-model hyperparameter widget keys, `preprocess_config_mode`, `interpretability_mode`
- `cohort_decisions_pending`, `cohort_replay_pending`

Plus configuration the reset never touched: `data_config`, `selected_features`, `pre_fe_feature_cols`, `raw_data`. Plus the study-wide records: `test_lockbox` (the seal — one per study, by invariant), `cohort_runs_done`, `cohort_run`.

```python
SHARED_DECISION_KEYS = {
    "preprocessing_config", "preprocessing_config_by_model",
    "feature_selection_results", "consensus_features",      # see 2.4
}
```

### 2.4 Straddles, and the MVP simplification for each

- **`feature_selection_results` / `consensus_features`.** The *selection* is a shared decision (`selected_features`). The *results table* was computed on whichever cohort's training rows were active when Feature Selection ran. Strictly it is per-branch; but making it per-branch means a freshly created branch has no selection results, page 04 renders empty, and the researcher is invited to re-select — which would change the shared decision under every other branch. **MVP: shared.** Page 04 gets a one-line caption when a cohort is active: "This ranking was computed on `<cohort>`'s training rows; the selection it produced applies to every group."
- **`df_engineered`.** Per-branch data built from a shared recipe. Archive it rather than replaying on every restore (2 cohorts × one frame is cheap; replay is not free). New branches still get it via replay, as today.
- **`preprocess_built_model_keys`.** State, not a choice — `replay._NOT_A_CHOICE` already says so (`:49`). Per-branch.
- **`filtered_data`.** Per-branch rows, produced by a shared rule. Archive the rows; a rule change is caller #8 and drops everything.

---

## 3. Storage: swap the branch in and out

### 3.1 Why swap, not prefix

The alternative — namespacing every result key by cohort (`trained_models[label]`) — changes every read site. `reset_downstream_results`' own docstring counts 10 bare reads of `model_results` on page 06 alone, 5 of `eda_results`, 4 of `trained_models` (`utils/session_state.py:521-539`). That is the rewrite this plan avoids.

Instead: **the flat keys are the active branch's view.** On a switch, snapshot every `BRANCH_KEY` into an archive keyed by `(column, label)`, then either restore the target's snapshot into the flat keys or (new branch) run the ordinary reset. Pages 02–10 do not change a line. The picker calls the switch.

### 3.2 `switch_branch`

New in `utils/cohorts.py`:

```python
BRANCH_ARCHIVE_KEY = "cohort_branch_archive"     # Dict[Tuple[str, str], Snapshot]

@dataclass
class Snapshot:
    keys: Dict[str, Any]                  # BRANCH_KEYS present at snapshot time
    ledger_entries: List[Insight]         # source_page in BRANCH_PAGES
    methodology: List[dict]               # steps mapping to BRANCH_PAGES
    provenance: Dict[str, Any]            # sections from downstream_sections()
    seal_opens: List[str]                 # this branch's opened_at entries (§4)

def snapshot_current() -> Optional[Snapshot]: ...
def archive_current() -> None:
    """Snapshot the live branch under its (column, label). No-op with no active run."""

def switch_branch(target: Optional[Tuple[df, plan, cell, target_col, dropped]]) -> None:
    """The one door. Both cohort_ui callers go through it."""
    archive_current()                                     # 1. never lose what is live
    key = _branch_key(target)                             # None => 'everyone'
    snap = _archive().get(key)
    if snap is not None:                                  # 2a. known branch: restore
        _clear_branch_keys()                              #     fresh objects, no reset
        _restore(snap)
        start_cohort(...) if target else clear_cohort()
    else:                                                 # 2b. new branch: today's path
        _replay.stage_for_replay(reason="cohort switch")
        start_cohort(...) if target else clear_cohort()
        reset_downstream_results(clear_feature_engineering=True,
                                 preserve_branches=True)  #     the ONLY True caller
        _replay.restore_decisions()
    get_provenance().record_cohort_restriction()
    st.rerun()
```

`_switch_to` and `_advance_to` in `utils/cohort_ui.py` shrink to argument-marshalling plus `switch_branch(...)`. The `filtered_data` pop they both do first (`:213`, `:417`) moves inside `archive_current` (it is a branch key now) — the hazard it guarded against (`apply_cohort` reading the previous cohort's frame) cannot arise once the frame is archived before the run label changes.

**"Everyone" is a branch too.** `key=None` archives the whole-study analysis like any other. Clicking "Go back to analyzing everyone" after two cohorts restores it if it existed, rather than presenting an empty app. This costs nothing extra and removes a class of confusion.

### 3.3 Two rules that keep the swap honest

1. **Snapshot before reset, always.** `archive_current()` is the first line of `switch_branch`, unconditionally. The existing reset assigns fresh objects (`= {}`, `= None`, `pop`) and never `.clear()`s in place — a snapshot therefore holds the old objects, not emptied shells. This becomes a stated invariant of `reset_downstream_results` with a test (§9).
2. **Aliasing is intended.** After a restore, the live key and the archived snapshot reference the same object; a page that mutates `model_results` in place mutates the archive. That is correct — the archive should track the live branch — and it is why `archive_current()` before every switch is idempotent rather than redundant.

### 3.4 The three non-key slices

- **Insight ledger.** Every `Insight` carries `source_page` and `auto_generated` (`utils/insight_ledger.py:761`, `:777`). `BRANCH_PAGES = {"05_Preprocess", "06_Train_and_Compare", "07_Explainability", "08_Sensitivity_Analysis", "09_Hypothesis_Testing"}` — the same set as the reset's `_rollback_pages` (`utils/session_state.py:639-644`). Snapshot = entries with `source_page ∈ BRANCH_PAGES`; restore = remove the live ones with those pages, extend with the snapshot. This also **closes the leak**: today the reset prunes only `{02, 05, 06}` (`:648`) and merely un-resolves 07/08/09, so female insights from those pages are printed into the male `report.md` (`pages/10_Report_Export.py:1506-1520`). Under the swap they travel with their branch instead of surviving into the next one. For the *new-branch* path, add 07/08/09 to `_pruned_pages` so the ordinary reset stops leaking them too.
- **`methodology_log`.** Filter by `_STEP_TO_PAGE[entry["step"]] ∈ BRANCH_PAGES` (`utils/session_state.py:916-928`); same snapshot/restore shape.
- **Provenance.** Sections named by `workflow_provenance.downstream_sections()` (`:1058-1071`) minus `feature_engineering` and `feature_selection` (shared decisions). Snapshot the section objects; restore by `setattr`. `upload` is preserved by the reset already and holds the cohort restriction, which `record_cohort_restriction` rewrites on every switch (`:393-428`).

### 3.5 Memory

Two branches of tabular fits and SHAP output are fine in session. The archive is bounded by `plan.viable` (≤ `MAX_SENSIBLE_CELLS = 6`, `utils/cohorts.py:53`) plus "everyone"; at six branches on wide data this needs revisiting (spill inactive snapshots to the scratch dir). Log the archive's approximate size in the sidebar chip once it exceeds one branch so the cost is visible. Not an MVP blocker; a stated limit.

---

## 4. Seal accounting per cohort

`opened_count` and `opened_at` live on the lockbox dict (`utils/test_lockbox.py:801-805`), which is study-wide by invariant and survives every reset except a new dataset. `record_lockbox_open(source)` appends a timestamp plus a bare page name; nothing records *which slice* was opened. The chip and the Methods both read the raw total.

### 4.1 Record the cohort with the opening

```python
def record_lockbox_open(source: str = "") -> ...:
    run = active_cohort()
    tag = f"{run['column']}={run['label']}" if run else "everyone"
    opens.append(f"{ts} ({source}) [{tag}]")
    lb["opened_count"] += 1
    lb.setdefault("opened_by_cohort", {})[tag] = lb["opened_by_cohort"].get(tag, 0) + 1
```

Both call sites (`pages/06_Train_and_Compare.py:1746`, `pages/08_Sensitivity_Analysis.py:260`) are unchanged — they already pass a source; the cohort is read from state.

### 4.2 Report both numbers

- **Chip** (`utils/test_lockbox.py:1353-1381`, `:1402-1431`): when a cohort is active, the red/blue decision uses *this branch's* scoring-open count; the sentence states both: "This run's held-out slice has been scored once. Across all groups the sealed set has been opened 2 times (Female 1, Male 1) — disjoint slices, no row scored twice." The caption at `:1428` that currently says "every run is evaluated against the same held-out people" is rewritten to say what the code computes: "each run is evaluated against its own slice of one held-out set drawn before the study was split."
- **Methods** (`ml/narrative_engine.py:900-911` and the Limitations entry at `:1993-1999`): read `opened_by_cohort`. With cohorts: "A single held-out set was drawn on the full study before it was split by *sex*; each group's slice was accessed *n* times (Female 1, Male 1). Because the groups were analyzed sequentially and Male was run after Female's held-out results were observed, …" The sequential-order disclosure stays until the up-front declaration (§10) makes it unnecessary.
- **Survive a redraw.** Case 12 in §1.2 rebuilds the lockbox with `opened_count: 0`. Carry the previous seal's `opened_by_cohort` forward under a `retired_seals` list so the Methods can say a prior seal existed and was opened, rather than "accessed only for the final evaluation."

### 4.3 What the counter still does not see

Page 07 computes SHAP/PDP/permutation on `X_test` and page 08's **Feature Dropout** tab scores `X_test` once per ablated feature (`pages/08_Sensitivity_Analysis.py:625-643`, `:685-710`); neither calls `record_lockbox_open`. That is a separate defect and out of scope here — but the per-cohort count must not be described as complete. The chip's wording says "scoring runs at Train & Compare," which is what it counts.

---

## 5. The picker on pages 07 and 08

Both pages gate on `trained_models` near the top (`pages/07_Explainability.py:163`, `pages/08_Sensitivity_Analysis.py:100`). The picker renders **above that gate**, so a researcher whose active branch is untrained can still switch to one that is:

```python
# utils/cohort_ui.py
def render_branch_picker(page_key: str) -> None:
    run = active_cohort()
    branches = available_branches()        # order from run["order"] + "everyone"; from archive ∪ active
    if len(branches) < 2:
        return
    labels = [f"{b.label}  ({'trained' if b.has_models else 'not yet trained'})" for b in branches]
    idx = st.selectbox("Cohort", labels, index=branches.index(current), key=f"branch_pick_{page_key}")
    if branches[idx] != current:
        switch_branch(branches[idx].target)
```

- Picking an untrained branch performs the switch and the page then stops at its own gate with the existing "train first" message — correct, and it says where to go.
- The picker is the switch (§3.2), so the sidebar chip, `get_data()`, provenance and export all agree with what the page shows. No second source of truth.
- **Known debt:** `active_cohort()` also filters pages 02–05. In a branch model those pages should not be per-cohort at all (decisions are shared). Acceptable while the flow is still sequential; the declaration step (§10) retires it.
- Page 08 additionally gets `render_cohort_note()` beside its tables (it calls neither today; page 07's only in-page statement is the lockbox caption at `:73-74`).

---

## 6. Export: one bundle, per-cohort sections — the (b+) scope

`pages/10_Report_Export.py` builds the zip in one block (`:2798-3031`) from the live keys. It stays the **full bundle for the active branch**. For every *other* archived branch under the current question, add:

```
cohorts/<column>=<label>/
    metrics.csv                 from snapshot.keys["model_results"]
    predictions/<model>.csv     from the snapshot's estimators + X_test/y_test
    models/<model>.joblib       from snapshot.keys["trained_models"]
    manifest.json               n_train, n_test, seal opens for this slice, constant features
cohort_comparison.csv           one row per banked CohortRun under completed_runs(column)
```

And in `report.md`:

- A **"Cohort analyses"** section listing every branch with its n, its best model and metric, and its seal-opens — the table `utils/cohort_ui.py::_runs_table` already builds and today renders only as a `st.dataframe` on page 06 with no download.
- The `comparison_caveats` sentences (`utils/cohorts.py:598-629`), which currently exist only as `st.warning` at `utils/cohort_ui.py:319` and reach no artifact.
- The `| Rows | N |` line under "Dataset Summary" (`:1469`) gains the cohort label; `metadata.json`'s `dataset.n_rows` (`:244-250`) gains `cohort` and `n_study`.

`manuscript.tex` and `latex_report` are **not** made multi-cohort in this MVP (the narrative engine, `latex_report`, `publication` and page 10 are the four largest, most contract-tested files; a two-column Results table waits for the declaration step, when the Methods must be rewritten anyway). The restriction sentence (`utils/workflow_provenance.py:63-95`) gains one clause: "Results for the other group(s) are in `cohorts/` in the accompanying bundle and must be reported together with these."

Artifacts are generated **from the snapshot dicts**, never by swapping the live branch during export. The three writers involved (metrics CSV, predictions, joblib) already take a dict-shaped input; give each a `state: Mapping` parameter defaulting to `st.session_state`.

---

## 7. Fix batch folded into this change

These are defects in the *current* design that corrupt a live manuscript, independent of persistence. They are small and they belong in the same PR series because each is "inherit the right things":

1. **Ledger leak** (07/08/09 insights survive the switch) — closed by §3.4; add the pages to `_pruned_pages` for the new-branch path.
2. **`accessed N times` sentence** (`ml/narrative_engine.py:900-911`, `:1993-1999`) — rewritten by §4.2.
3. **`dropped_features` recorded, never applied** (`utils/cohorts.py:474`; every other use is display/serialization). Apply at branch creation: remove the flat columns from the branch's *working* feature list for training only, keep `data_config.feature_cols` intact, and state it on page 06 and in the manifest. Page 07's importance ranking then stops showing `sex` at ~0 as if it were a finding.
4. **Page 07 External Validation ignores the cohort** (`pages/07_Explainability.py:1839-1979`): scores a single-sex model against the whole external file. Filter the external frame by the active cohort's column/value when both exist; refuse with a stated reason when the column is absent.

Deferred, with reasons: page 07 Subgroup Analysis offering a constant stratifier (needs the declaration step's notion of "the grouping variable"); stale `llm_result_*` prose (opt-in export path, off by default, reset by the switch — screen-only today).

---

## 8. What changes where

| File | Change |
|------|--------|
| `turbotab/cascade.py` | Promote the 14 `_NOT_YET_DECLARED_IN_THE_DAG` keys onto their stages (each already names its stage in the test's comment). Add `BRANCH_KEYS` / `SHARED_DECISION_KEYS` derivation. |
| `utils/session_state.py` | `reset_downstream_results(..., preserve_branches=False)`; the archive pop; 07/08/09 into `_pruned_pages`; docstring invariant "assigns fresh objects, never clears in place." |
| `utils/cohorts.py` | `Snapshot`, `BRANCH_ARCHIVE_KEY`, `snapshot_current`, `archive_current`, `switch_branch`, `available_branches`. `CohortRun` gains `seal_opens: int`. |
| `utils/cohort_ui.py` | `_switch_to` / `_advance_to` delegate to `switch_branch`; drop the "artifacts will not be kept" warning (`:351-359`, no longer true); `render_branch_picker`. |
| `utils/test_lockbox.py` | `record_lockbox_open` tags the cohort and maintains `opened_by_cohort`; chip reads per-branch count; caption at `:1428` corrected; `retired_seals` on redraw. |
| `ml/narrative_engine.py` | Per-cohort access sentence and Limitations entry; sequential-order disclosure. |
| `utils/workflow_provenance.py` | Restriction sentence's bundle pointer clause. |
| `pages/07_Explainability.py`, `pages/08_Sensitivity_Analysis.py` | `render_branch_picker` above the `trained_models` gate; `render_cohort_note` on 08; external-validation cohort filter on 07. |
| `pages/10_Report_Export.py` | `cohorts/` tree, `cohort_comparison.csv`, "Cohort analyses" section + caveats in `report.md`, labeled row counts. |
| `pages/04_Feature_Selection.py` | One caption (§2.4). |
| `pages/06_Train_and_Compare.py` | Apply `dropped_features` to the training feature list; bank `seal_opens` into `CohortRun`. |

Pages 02, 03, 05, 09 and `app.py`: no change.

---

## 9. Tests

**Keep green, unchanged** — these pin invariants the MVP preserves:
`tests/test_cohorts.py` (planning/vetting), `tests/test_cohort_lockbox_invariant.py` (seal never redrawn mid-run), `tests/test_cohort_run_identity.py` (comparison table filtering), `tests/test_the_cohort_switch_keeps_the_decisions_it_promises.py` (replay contract), `tests/test_manuscript_discloses_cohort.py`, `tests/integration/test_cascade_dag_equivalence.py` (after promoting the 14 keys, `_NOT_YET_DECLARED_IN_THE_DAG` becomes empty and its staleness assertion should be inverted to "nothing is excluded").

**Re-read for intent before touching:**
- `tests/integration/test_the_next_cohort_rebuilds_the_same_pipeline.py::test_nothing_fitted_on_the_women_survived` — asserts the *live* keys hold nothing from Female. The swap satisfies this (live keys are Male's). The invariant is "no Female fit is *used* for Male," not "no Female fit *exists*." Add an assertion that the archive holds Female's, so the test states the new contract.
- `tests/test_cohort_runs.py::test_corrected_values_keep_the_run_but_kill_the_results` — caller #9. Extend: the archive is empty afterwards.
- `tests/test_session_carries_the_run.py` — the archive is *not* saved (§10); assert a restored session has an empty archive and says so.

**New:**
- ~~One test per caller in §1.2, cases 1–12~~ → **replaced by a static gate plus four behavioral ones**, in `tests/test_a_branch_does_not_outlive_its_question.py`. Booting twelve Streamlit pages to assert one key's absence pins the callers that exist today; an AST scan over every source file pins the property itself — *no call site outside `utils/cohorts.py` passes `preserve_branches=True`* — and keeps pinning it for callers written next year. A second scan classifies every caller as drop-all or archive/restore and fails on an unclassified one, which is how the five missing from §1.2 were found. The behavioral half covers the four flag combinations, `reset_data_dependent_state`, and `set_data`'s same-schema branch.
- `switch_branch` round trip: Female → Male → Female restores identical objects for every `BRANCH_KEY`, the ledger slice, the methodology slice, and the provenance sections.
- Snapshot-before-reset: a switch to a *new* branch archives the previous one; `reset_downstream_results` never `.clear()`s (AST scan, the same shape as `test_page_03_no_longer_hand_rolls_its_own_cascade`).
- Picker: selecting an untrained branch switches and stops at the page gate; selecting a trained one renders that branch's models.
- Seal: two cohorts scored once each → `opened_by_cohort == {F:1, M:1}`, chip is not red, Methods sentence names both counts; a redraw after clearing the cohort carries the counts into `retired_seals`.
- Export: with Male active and Female archived, the zip contains `cohorts/sex=Female/metrics.csv` whose numbers equal Female's banked `model_results`; `report.md` contains every `comparison_caveats` sentence; no artifact under `cohorts/` is generated by touching the live keys (assert live `trained_models` is Male's before and after).
- `dropped_features` applied: a single-sex branch's fitted models do not have `sex` among `feature_names`.

---

## 10. Out of scope, and known limits

- **Up-front cohort declaration on page 01** (multi-select, chooser moved above `ensure_lockbox` at `pages/01:1697`, seal stratified on cohort × outcome, "everyone" as a checkbox, obligations for declared-but-unrun groups). This is the scientifically stronger design — it closes the forking path that the sequential-order disclosure in §4.2 can only *describe* — and it is deferred because it cascades through every page. Everything in this MVP is prerequisite to it.
- **Multi-cohort `manuscript.tex`** (two-column Results, both N's in the abstract). Waits for the declaration step.
- **Cross-session persistence of branches.** Every branch key is in `session_manager._NEVER_PERSIST` (`utils/session_manager.py:167-191`) or absent from the save allowlist; only `cohort_runs.json` (the thin `CohortRun` list) is written. Branches live for the session. The restore path should say so.
- **`active_cohort()` filtering pages 02–05.** Sequential-model behavior retained; retire with the declaration step.
- **Counter blind spots** (page 07 SHAP on `X_test`, page 08 Feature Dropout scoring) — separate defect, §4.3.
- **Subgroup Analysis in a single-sex run; stale LLM prose** — deferred, §7.

---

## 11. PR sequencing

Each PR is independently mergeable and leaves the app no worse than before it.

1. **Invalidation and the archive drop.** `preserve_branches` flag; the pop inside the reset; promote the 14 keys into `cascade.py`; `BRANCH_KEYS` derivation; the twelve caller tests. *No behavior change yet* — the archive key never exists — but every future PR is safe.
2. **`switch_branch` and the swap.** `Snapshot`, `archive_current`, the two `cohort_ui` callers delegate, ledger/methodology/provenance slices, 07/08/09 into `_pruned_pages`, "everyone" as a branch, round-trip tests. After this PR nothing is lost on a switch; there is still no picker.
3. **Per-cohort seal accounting.** `opened_by_cohort`, chip, Methods, `retired_seals`, corrected caption. Small and self-contained; could land before 2.
4. **The picker.** Pages 07/08; `render_cohort_note` on 08.
5. **Export (b+).** `cohorts/` tree, comparison CSV, caveats into `report.md`, labeled counts, restriction-sentence clause.
6. **Fix batch.** `dropped_features` applied; External Validation cohort filter. (The ledger leak and the access sentence are already closed by 2 and 3.)

---

## 12. Implementation notes — where the plan was wrong

*Added 2026-09-03 after building all six PRs. Every item below is a place the
plan above says something the code did not support. They are recorded rather
than edited away, because the pattern in them is the useful part: every one is
a case of a plan reading an interface and the code having a second one behind
it.*

**§1.2 undercounted the callers.** Fourteen listed, nineteen real, in nine
files. The five missed were page 01's two new-file resets, page 02's staleness
guard, `set_data`'s schema branch, and the session restore. A name grep also
misses page 01's exploratory toggle, which calls the reset through the alias
`_rdr` inside a nested callback — the AST scan in
`tests/test_a_branch_does_not_outlive_its_question.py` resolves aliases for
exactly that reason. The default-destructive rule covered all five for free,
which is the argument for it made by its own author's undercount.

**§9's "one test per caller" is not reachable, and a static gate is better.**
Seven of the callers sit inside `if st.button(...)` bodies in page scripts, and
one is a nested closure no test can call by name. More importantly, twelve
page-boot tests pin the callers that exist *today*; an AST scan asserting *no
call site outside `utils/cohorts.py` passes `preserve_branches=True`* pins the
property, and keeps pinning it for callers written next year.

**§3.2's `switch_branch` does not `st.rerun()`.** It returns a bool — restored,
or built — and the two UI callers rerun. `st.rerun()` raises inside the
function, which would make every test of the swap catch an exception.

**§3.4's `BRANCH_PAGES` could not be `_rollback_pages`.** That set is seven
pages at runtime, not five: it adds `03_Feature_Engineering` and
`04_Feature_Selection` when `clear_feature_selection` is true, which it is on
the cohort-switch path. Reusing it would have archived the FE recipe's and the
selection's insights per branch — contradicting §2.3, which calls both shared.
It is a literal in `turbotab/cascade.py`.

**§3.4's ledger restore needed a new method.** `prune_auto_generated` filters on
`auto_generated AND source_page`, so a hand-written note on 06–09 survived the
removal and was then duplicated by the restored slice. The ledger gains
`prune_pages` (page only) and `entries_from_pages`. The splice uses `upsert`,
not `add` — `add` returns False and silently discards on an id collision. And
the snapshot holds live `Insight` objects: `to_list()`/`from_list()` round-trip
through dicts and re-run `__post_init__`, which refills an emptied
`tripod_keys` from the category map, so a restored insight would not equal the
archived one.

**§3.4's provenance list comes from the record, not the cascade.**
`downstream_sections()` is structural — it finds every `Optional[*Provenance]`
field — and returns nine branch sections including `sensitivity`,
`statistical_validation` and `external_validation`, three that
`turbotab/cascade.py`'s own `provenance_sections` never declared. Deriving from
the cascade would have silently failed to archive them.

**§4.1's tag could not go in the `opened_at` string.** The chip parses those
strings for the source name and requires each entry to END with `)`
(`utils/test_lockbox.py`). A ` [sex=Female]` suffix empties `_source_counts`,
which re-classifies page 08's seed sweep as a scoring run and flips its blue
notice to a red warning. The tag lives in a structured `opened_by_cohort` field;
a test pins the string format.

**§4.2 named one destruction path; there are two.** The redraw rebuilds the dict
with `opened_count: 0` — `existing` is still in scope there — but the
stale-label refusal `pop`s the dict outright and returns None, with no successor
to carry anything into. `retired_seals` is a session key beside the lockbox, not
a field inside it, so both paths can reach it.

**§5's picker had to go above the TASK-MODE gate, not the trained-models one.**
Page 07 stops on task mode five lines earlier. And the selectbox is seeded
rather than defaulted: passing both `index=` and `key=` makes Streamlit log a
conflict every render, and the widget also has to re-seed when the branch
changes underneath it, because the sidebar chooser and page 06's button switch
too.

**§6's "give each writer a `state: Mapping` parameter" was not viable.** Two of
the three writers are not functions — the metrics CSV and the predictions loop
are inline top-level code closing over module globals — and `generate_report`
reads `st.session_state` at eleven points and the module-global `df`. Threading
a mapping through all of it is a much larger change than the plan budgeted.
Instead `utils/cohort_export.py` builds every other branch's artifacts from its
`Snapshot`, and **does not import Streamlit at all**. That is a stronger
guarantee than the rule §6 states: a module with no access to session state
cannot swap a branch in, so the export cannot leave the researcher in a cohort
they did not choose. A test asserts the import is absent.

**§7.3's `dropped_features` has two application sites, not one.** The split
builds `feature_cols`; `feature_names_by_model` re-reads `selected_features`
from scratch several hundred lines later and shares no variable with it.
Filtering only the first leaves the exported per-model feature names listing a
column the model was never fitted on. Both go through one
`cohorts.training_features()`. Applying it also made page 06's
`reconcile_pipeline_columns` backstop warn about an intended decision on every
training run, so that warning now separates a cohort's deliberate drops from a
real drift.

**§7.4's filter belongs on `ext_df`, not on the score inputs.** Both the stored
`external_validation_results` record and the provenance event read
`ext_df.shape[0]`, so filtering further down would have written the UNFILTERED
external N into the manuscript — over-stating the validation cohort, which is
the defect the fix exists to remove. The missing-column check also had to be
new: the existing one tests `selected_features + target_col`, and the grouping
column is in neither.

**§10's persistence claim is false as written.** Seven per-branch keys ARE in
`session_manager`'s save allowlist (`df_engineered`, `filtered_data`,
`feature_engineering_applied`, `engineered_feature_names`,
`preprocessing_config_by_model`, `methodology_log`,
`preprocess_built_model_keys`). The narrower claim the MVP actually needs does
hold and is what the test asserts: `cohort_branch_archive` is in none of the
four allowlists, so branches live for the session.

### Still open after this MVP

Everything in §10, unchanged — the up-front declaration on page 01 is the
scientifically stronger design and remains the next step. Plus one found while
building: `tests/test_drive8_explainability.py` and
`tests/test_paper_risk_report.py` read page sources without
`encoding="utf-8"`, so 17 tests die with `UnicodeDecodeError` on Windows rather
than checking anything. That predates this series and is a separate one-line
fix.
