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

**The register lives at [`FEATURE_REGISTER.md`](FEATURE_REGISTER.md), generated from
`data/register.json` via `tools/register.py` — never hand-edited.** It earned that structure the
hard way: the first register, written as a markdown table in this file, was destroyed by a branch
merge that blind-copied an older revision over it. The ledger survived the same merge because it
is data worked through a tool; the register now is too, with a `check` that fails when a built
step has no rows or the markdown goes stale.

- Building any Guided step **starts** by listing what the corresponding Classic page can do
  (`register.py add` per capability), and **ends** with every item dispositioned — and the step
  added to `BUILT_STEPS` so an empty step is a failure, not a silence.
- The parity harness covers `both` rows. `classic-only` rows are excluded from parity *by
  explicit entry*, never by omission — so the exclusion list is readable and arguable.
- "We forgot" stops being possible, because a capability with no row fails the register check.

This is the same trick as the findings ledger: the failure mode is silence, so make silence a
test failure.

**Corollary — the principle-locality rule** (from the T0-LIVE-005 close): *a principle written in
one place and applied in another is the same silence as a capability with no row.* The content-key
principle lived in a docstring on four caches while eight caches beside it violated it. When a
principle is worth a docstring, it is worth a test that applies it everywhere it binds — one
fingerprint for the page, not one per author.

**Corollary — the expiring-guarantee rule** (from the `T0-PREREG-002` close): *a protection that
depends on "X does not exist yet" expires the moment X exists, and nothing will tell you.*
`tests/integration/test_routing_baseline.py` wrote the pre-registered Classic baseline and said
why in its docstring — *"written now, while the Router does not exist, so it cannot be fitted
to."* True when written. False from the moment L8 landed, and no test, check or reviewer noticed:
for three loops afterwards every suite run re-measured the reference the value check is judged
against, with the Router present, and committed the new numbers over the old.

The sibling relationship to principle-locality is exact — both are **silence rather than
failure**, and both are invisible precisely because the thing that was true is still written
down. The difference is the axis: locality fails across *space* (stated here, violated there),
this one fails across *time* (true then, false now, unannounced).

Three defenses, in order of strength:

- **Do not write a temporal guarantee you cannot enforce.** Split the acts: measurement and
  comparison must not share a code path, or the comparison silently becomes a re-measurement.
- **Name the expiry condition in the artifact**, not in prose. A baseline carries `measured_at`;
  a test asserts it against the document that banks thresholds on it.
- **Audit every "before X exists" claim when X ships.** A loop that lands a component should grep
  for guarantees phrased against its absence. `router.py` landing at L8 should have triggered
  exactly that sweep and did not.

**Corollary — the frozen-measurement rule** (from the same close, after the first "never edited,
ever" proved too blunt to state what it protected): *the measurements are frozen; the envelope may
gain labels, never lose or alter one.*

A pre-registered data file has two layers and only one of them is the experiment. The
`measurements` are the result and are immutable — a value that moves is a re-measurement, and a
re-measurement is a new row, never a restatement. Everything around them is envelope: schema
version, `measured_at`, `prereg`, and whatever provenance a later loop finds it needs.
`c8c5f51` added `pull_affordances` and `mode` to `routing-baseline.json` without moving a number,
and L9b added `measured_at`; both are legitimate, and a rule that forbade them would have forced
the provenance into a second file nobody checks.

**Why adding a label is safe here, stated because it is the whole justification:** a self-declared
stamp proves nothing on its own. Anyone who can swap the file can swap the stamp with it. So the
stamp is a convenience, and the load-bearing assertion is the git-read values check —
`test_the_frozen_baseline_is_the_one_the_prereg_names` reads the file as of the commit the
pre-registration names and compares every metric the prereg quotes. The envelope may be edited
precisely because nothing depends on trusting it.

### Two specific things to watch

- **The pedagogy layer** — `utils/theory_anchors.py` (532 loc) and `utils/theory_demos.py`
  (869 loc) are a 19-key registry pair with **no test asserting the keys match**, plus a
  substring-matching fallback that silently drops a theory link when a finding string is reworded.
  It is the most fragile intelligent feature in the app and the most likely to quietly not
  survive a rewrite.
- **Cohort runs and the replay engine** — the newest subsystem (`utils/cohorts.py`,
  `utils/replay.py`), entangled with the lockbox through two import cycles. A `Project` that
  models "the working table" without modeling "the active cohort filter" deletes it silently.
  Already flagged in `TRANSITION_PLAN.md` §05; repeated here because it is a *feature*, not just
  a state-model detail.

---

## The register — state model (L5 / L6)

Not a Guided *step*, but the layer both doors read. Registered because "extract
the core so the UI becomes a choice" only holds if the **state model** is shared
too: two state models is the same failure as two engines, one level down.

| Capability | Classic | State | Reason |
|---|---|---|---|
| Step-completion model (ten predicates) | `utils/theme.py:685` | **core** | Extracted to `turbotab/readiness.py`; `theme.py` now asks instead of computing, and a test asserts the expressions are gone rather than merely unused. This is the Router's readiness function. |
| Quick / advanced disclosure split | `utils/theme.py:685` | **core** | Moved with the predicates — it decides which questions are optional, which the Router inherits. |
| Invalidation cascade | `utils/session_state.py`, plus a copy in `pages/03` | **core** | `turbotab/cascade.py` declares the graph once and reproduces the production function key for key across all four flag combinations. The `pages/03` copy is **not yet reconciled**: it misses fifteen result keys, and the DAG names them. |
| Partial invalidation (`clear_feature_*=False`) | `utils/session_state.py` | **core** | Expressed as `keep={stage}` and pinned per flag, because a naive full-cascade DAG cannot express it. |
| Session save / restore (zip archive) | `utils/session_manager.py` | **both** | `turbotab/archive.py` ports the schema — same members, same names, same version — so the doors read each other's archives. |
| Sealed lockbox, held by row label | `utils/test_lockbox.py` | **both** | Same dict shape, so one lockbox satisfies both doors; round-trips verbatim. |
| Active cohort filter | `utils/cohorts.py` | **both** | First-class field on the project, with `working_table` derived from it, so "which rows is this number about?" has one answer. |
| Per-model pipeline **specs** | `pages/05` (stores fitted objects) | **guided-only** | Guided stores serializable specs and has no global fallback slot. Classic still stores fitted pipelines behind a global slot that lets two models alias one instance (`TRANSITION_PLAN.md` §02.1). A debt owed back, and a live defect there. |
| Identity barrier (`T0-ID-001`) | — | **guided-only** | Classic has no phase rule: nothing stops a structural repair running after the lockbox is sealed, and the stale lockbox still looks well-formed. Owed back. |
| Serialization guard (no participant data in the record) | — | **guided-only** | New. `session_manager` drops derivatives for safety and cost; it does not check that what remains is free of cell values. Owed back. |

Three `guided-only` rows, all debts rather than luxuries — and two of them
(`pipeline specs`, `identity barrier`) describe defects that are **live in
Classic today**, not merely absent from it.

> **Note.** The Data & Target step register added in `9211566` is not in this
> file any more; it was dropped in `b4bff25`. Not restored here, because that
> may have been deliberate — but a step with no rows is the failure mode this
> register exists to prevent, so it needs either restoring or an explicit entry
> saying where it went.

---

## The one-line answer

**The algorithms are safe — they were always engine code and both doors will call the same
functions. The orchestration is not, and neither is exposure. Those two need a register, or
"ported" quietly becomes "most of it."**
