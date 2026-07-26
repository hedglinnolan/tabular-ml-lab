# Do the intelligent features get ported?

Short answer: **they don't get ported — they get shared.** That distinction is the whole point of
the extraction, and it is also where the one real risk lives.

The question splits three ways, and each part has a different answer.

---

## 1 · Capability — shared, automatically

Almost every intelligent feature already lives in the engine, not in the UI. Both doors call the
same function. Nothing is copied, nothing is reimplemented, and a fix lands once for both.

| Capability | Home | Lines | Shared today |
|---|---|---:|---|
| Structural diagnosis + reversible repairs | `ml/import_doctor.py` | 1,029 | yes |
| Key detection, join confidence, relationship types | `ml/join_doctor.py` | 1,088 | yes |
| Model ranking, bucketing, viability | `ml/model_coach.py` | 1,443 | yes |
| Pre-training probe | `ml/coach_probe.py` | 243 | yes |
| Dataset diagnostics | `ml/dataset_profile.py` | 754 | yes |
| Task-type + cohort-structure detection | `ml/triage.py` | 282 | yes |
| EDA recommendations | `ml/eda_recommender.py` | 563 | yes |
| EDA actions | `ml/eda_actions.py` | 1,328 | yes (4 `st` refs to strip) |
| Table One | `ml/table_one.py` | 375 | yes |
| Statistical tests | `ml/stats_tests.py` | 169 | yes |
| Outlier detection | `ml/outliers.py` | 100 | yes |
| Calibration | `ml/calibration.py` | 338 | yes |
| Bootstrap CIs | `ml/bootstrap.py` | 263 | yes |
| Sensitivity analysis | `ml/sensitivity.py` | 132 | yes |
| Manuscript generation + `[AUTHOR REQUIRED]` | `ml/narrative_engine.py` | 1,975 | after detaint |
| Manuscript validation | `ml/manuscript_validator.py` | 426 | after detaint |
| LaTeX export | `ml/latex_report.py` | 1,066 | after detaint |
| PCA / UMAP / persistence / Mapper | `ml/macro_shape.py` | 723 | yes (5 `st` refs; also `T0-LIVE-001`) |
| Feature selection | `ml/feature_selection.py` | 295 | yes |
| NN configuration advice | `ml/nn_recommender.py` | 199 | yes |
| Plot narration | `ml/plot_narrative.py` | 469 | yes |
| Regime detection | `ml/regime.py` | 193 | yes |
| Clinical units + physiology reference | `ml/clinical_units.py`, `ml/physiology_reference.py` | 353 | yes |
| Insight lifecycle (the coach's memory) | `utils/insight_ledger.py` | 1,408 | after singleton cut |
| Provenance record | `utils/workflow_provenance.py` | 759 | after singleton cut |
| Test-set lockbox | `utils/test_lockbox.py` | 554 | after `st` reads removed |
| Cohort runs | `utils/cohorts.py` | 629 | after detaint |
| Replay engine | `utils/replay.py` | 405 | after detaint |

**~19,000 lines of intelligence, and essentially all of it is already engine code.** The "after
detaint" entries are the L7 work, and the deepest of them is a singleton at the bottom of a file.

## 2 · Orchestration — trapped, and this is the real work

What is *not* in the engine is the intelligence about **which analysis runs when, in what order,
with what defaults, and which options are offered.** That lives in `pages/`, and it is 19,835
lines of it.

| Trapped capability | Where | Risk |
|---|---|---|
| The whole split strategy — grouped / chronological / lockbox-respecting / stratified | `pages/06:380-760` (~370 loc) | **high** — safety-critical, untested, no `ml/splits.py` equivalent exists |
| Step-completion model + quick/advanced disclosure | `utils/theme.py:685` | **high** — the Router's readiness function, filed under CSS |
| Which EDA analyses run and what counts as notable | `pages/02` (222 logic markers) | high — this *is* the Router's raw material |
| Report assembly decisions | `pages/10` (24 local functions) | medium |
| SHAP orchestration + per-model applicability | `pages/07` (160 markers) | medium |
| Statistical test selection rules | `pages/09` (79 markers) | medium |
| Per-model pipeline defaults | `pages/05` (131 markers) | medium |
| Transform catalogue + applicability | `pages/03` (144 markers) | medium |

**Nothing here is lost — but nothing here is free either.** Each one has to be extracted to the
core before either door can share it, and until it is extracted it exists only in Streamlit.

## 3 · Exposure — a decision per feature

Even once a capability is shared, the Guided door only surfaces what its interview asks about.
That is a design choice, not an accident: the entire premise is that fewer, better-ordered
questions beat eleven pages of everything (`PRODUCT_VISION.md` §01).

So "is feature X in TurboTab?" is really three questions:

1. **Is the logic in the core?** Mostly yes today.
2. **Has the orchestration been extracted?** Mostly no.
3. **Does the Guided interview ask about it?** Per-feature decision, made when that step is built.

A capability can be fully shared and deliberately not surfaced in Guided — Classic remains the
door for it. That is a legitimate outcome, and it is not a regression, as long as it is
**recorded** rather than forgotten.

---

## The risk, stated plainly

> **A feature that exists only in `pages/` and is never touched will never reach the Guided door,
> and nothing will announce that.**

The lazy-migration policy (`ROADMAP.md` rule 4) says pages move to the core when you touch them.
That is the right policy for maintenance cost — and it means untouched pages stay Streamlit-only
indefinitely. Combined with the exposure decision in §3, a capability can go missing from Guided
for two entirely reasonable-sounding reasons at once, and nobody notices until a user asks.

### The mitigation

**A feature register, maintained like the ledger.** Every capability gets a row with three
states: `core` (extracted), `classic-only` (still trapped, or deliberately not surfaced), and
`both` (extracted and exposed in Guided). Then:

- Building any Guided step **starts** by listing what the corresponding Classic page can do, and
  **ends** by recording each item as `both` or `classic-only` with a reason.
- The parity harness covers `both` rows. `classic-only` rows are excluded from parity *by
  explicit entry*, never by omission — so the exclusion list is readable and arguable.
- "We forgot" stops being possible, because a capability with no row fails the register check.

This is the same trick as the findings ledger: the failure mode is silence, so make silence a
test failure.

### Two specific things to watch

- **The pedagogy layer** — `utils/theory_anchors.py` (532 loc) and `utils/theory_demos.py`
  (869 loc) are a 19-key registry pair with **no test asserting the keys match**, plus a
  substring-matching fallback that silently drops a theory link when a finding string is reworded.
  It is the most fragile intelligent feature in the app and the most likely to quietly not
  survive a rewrite.
- **Cohort runs and the replay engine** — the newest subsystem (`utils/cohorts.py`,
  `utils/replay.py`), entangled with the lockbox through two import cycles. A `Project` that
  models "the working table" without modelling "the active cohort filter" deletes it silently.
  Already flagged in `TRANSITION_PLAN.md` §05; repeated here because it is a *feature*, not just
  a state-model detail.

---

## The register — Data & Target step (L3 walking skeleton)

The first step to be built under the register rule. Classic counterpart:
`pages/01_Upload_and_Audit.py` (1,264 loc, frozen). Every capability that page holds is listed;
a capability with no row here is the failure mode the register exists to prevent.

**Ingestion**

| Capability | Classic | State | Reason |
|---|---|---|---|
| Single delimited file upload | `st.file_uploader` | **both** | `engine.read_table`, plain pandas inference to match what the doctor expects |
| Multi-file upload + roster (rename / remove / replace) | Step 1 | classic-only | The multi-file path is frozen pending the open ledger tail (`TRANSITION_PLAN.md` §05). Deliberate, not forgotten. |
| Excel `.xlsx` + sheet selection | Step 1 | classic-only | Not yet exposed. `docs/audit/RESUME.md` names Excel-sheet × transpose as untested; exposing it in Guided first would build on that gap. |
| JSON + records-key selection | Step 1 | classic-only | Same door as multi-file import; frozen with it. |
| Transpose on import | Step 1 | classic-only | Belongs with the Excel path above. |
| Large-file guard ("Load anyway") | Step 1 | classic-only | Guided uses a hard 64 MB ceiling instead, because the frame is held in memory and never spooled to disk. Different mechanism, same intent. |
| Built-in practice datasets | Step 1 | classic-only | Guided ships one messy sample table instead (`turbotab/sample_data/`). |
| Import-repair logging to the insight ledger | `_log_import_repairs` | classic-only | Guided records *decisions*; the ledger singleton is not cut yet (L7). |
| Multi-file join + key detection | Step 2, `join_doctor` | classic-only | Frozen (§05). |

**Audit**

| Capability | Classic | State | Reason |
|---|---|---|---|
| Structural diagnosis | `import_doctor.diagnose` | **both** | Same function, same findings — asserted field-for-field by `test_findings_match_a_direct_engine_call` |
| Apply a proposed repair | "Suggested Actions" | **both** | `import_doctor.apply_fix`, same nine fix kinds |
| **Preview before apply** | — | **guided-only** | Classic applies straight from a button. See the note below: the register has no state for this. |
| **Undo an applied repair** | — | **guided-only** | Same. Classic has no undo for a suggested action. |
| Dataset profile (types, missingness, cardinality, numeric stats) | Steps 3 expanders | **both** | `compute_dataset_profile`; rendered as one ranked stack rather than six expanders |
| Duplicate-row detection | Step 3 expander | classic-only | Not surfaced in Guided. **No engine home** — it is computed inline in the page, so this is orchestration still trapped in `pages/`. |
| Cardinality table, per column | Step 3 expander | classic-only | Guided surfaces the profile's high-cardinality *finding* but not the full table. Pull-based exploration, deferred. |

**Target & task**

| Capability | Classic | State | Reason |
|---|---|---|---|
| Target column selection | Step 4 | **both** | |
| Task-type detection | `triage.detect_task_type` | **both** | |
| **Task-type override** | "Override Task Type" expander | **both** | *Added because this register found it missing.* See below. |
| Goal selection (Prediction vs Hypothesis Testing) | Step 4 | classic-only | Guided assumes prediction; the hypothesis-testing branch is a different interview. |
| Feature selection (select all / clear) | Step 4 | classic-only | Not yet asked in Guided. |
| Test-holdout / lockbox settings | Step 4 expander | classic-only | **The one to watch.** The lockbox is not modelled in the skeleton at all, and it is the invariant with the most scar tissue behind it. |

### What the register caught on its first use

**A missing override, which was a correctness bug and not a scope decision.** `ml/triage.py:53-64`
returns `low` confidence for a low-cardinality integer target and says so in its own words —
*"counts or ordinal scores should be treated as regression. Verify or override below."* Classic has
that override. Guided reported the verdict and offered no way to contradict it, which is the app
deciding at a confidence tier `PRODUCT_VISION.md` §07.1 reserves for the user. Filing it
`classic-only` would have recorded a violation of the governing rule as a legitimate exclusion, so
it was built instead: the override is offered whenever confidence is below `high`, and both the
detection and the user's answer are kept in the record.

That is the register working exactly as intended on its first outing — the gap was invisible while
the feature was being built and obvious the moment the capabilities were listed side by side.

**The register needs a fourth state.** It has `core`, `classic-only` and `both`, all of which
assume capability flows Classic → Guided. Preview-before-apply and undo flow the other way: the
engine always supported them (`apply_fix` returns a new frame and never mutates), Classic never
exposed them, Guided now does. Recording those as `both` would be false and `classic-only` absurd.
Suggest **`guided-only`**, with the same obligation attached — a reason, and a note on whether
Classic should get it. For these two the answer is probably yes: applying a repair to a research
dataset from a single button, with no diff and no undo, is the blind consent
`PRODUCT_VISION.md` §04 argues against, and it is shipping in Classic today.

**One capability has no engine home.** Duplicate-row detection is computed inline in `pages/01`,
so it is not "shared automatically" — it is orchestration in the §2 sense, and it will not appear
in any `ml/` inventory. Recorded as `classic-only` rather than assumed portable.

---

## The one-line answer

**The algorithms are safe — they were always engine code and both doors will call the same
functions. The orchestration is not, and neither is exposure. Those two need a register, or
"ported" quietly becomes "most of it."**
