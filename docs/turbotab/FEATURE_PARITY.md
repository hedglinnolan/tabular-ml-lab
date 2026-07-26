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

**A feature register, maintained like the ledger.** Every capability gets a row with four
states: `core` (extracted), `both` (extracted and exposed in Guided), `classic-only` (still
trapped, or deliberately not surfaced), and `guided-only` (built in the Guided door and owed
back to Classic). Then:

The fourth state is not symmetry for its own sake — **convergence runs both ways.**
Preview-before-apply and undo are Guided-first, and Classic today applies a repair from a single
button with no diff and no undo, which is precisely the blind consent `PRODUCT_VISION.md` §04
argues against. A `guided-only` row is a debt owed to Classic, and naming it stops the Guided
door quietly becoming the only place the product's own principles hold.

Also note what the register caught on its very first use: `triage` returns low confidence and
tells the user to *verify or override*; Classic offers that override, Guided did not. Filing it
`classic-only` would have recorded **a governing-rule violation as a legitimate exclusion**. The
register works because it forces that comparison — but only if `classic-only` is treated as a
claim to be justified, never as a shrug.

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

## The one-line answer

**The algorithms are safe — they were always engine code and both doors will call the same
functions. The orchestration is not, and neither is exposure. Those two need a register, or
"ported" quietly becomes "most of it."**
