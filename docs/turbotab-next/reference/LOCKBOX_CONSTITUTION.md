# The lockbox constitution

What the app is allowed to know, and when. Binding for TurboTab v2 wherever BLUEPRINT.md and
MODELING_SEQUENCE.md do not supersede it; BLUEPRINT §12 ruling 4 supersedes the unscoped "never"
§07 once carried, and the text below already states the purpose-conditional rule.

Extracted verbatim from the legacy roadmap (`docs/turbotab/archive/ROADMAP.md`, §"The lockbox
constitution", and the routing constitution in its Decision B) when the legacy app was retired
(BLUEPRINT §9.1). Code and docs cite it as "the lockbox constitution §NN"; the section numbers
are the roadmap's. Documents it names that are not in this folder are in
`docs/turbotab/archive/`, except `DOMAIN_SCIENCE.md`, which stays in `docs/turbotab/`.

---

## The constitution

The routing constitution governs which questions get asked. This one governs **what the app is
allowed to know, and when.** It exists because the seal is the load-bearing claim of the whole
product: every held-out number, every manuscript metric, rests on the assertion that the test rows
were never seen. `IMPORT-020` proved that assertion could be false while a lock icon rendered
cleanly, which is the governing rule's own failure at the deepest point in the app.

Grounded in TRIPOD+AI (eligibility reporting), Harrell RMS (extrapolation), sklearn's pipeline
doctrine and Kapoor & Narayanan's leakage taxonomy (fold-local fitting), Sisk/Sperrin/van Smeden
and Groenwold (missingness under prediction vs inference), and Steyerberg (outcome excluded from
imputation).

### 01 · The pre-seal sequence is fixed

> **lens → structural repairs and the impossibility pass → target → grain → *(repeats or time
> points)* → *(unit of analysis)* → *(aggregation)* → *(temporal prediction)* → eligibility →
> SEAL → EDA**

Bracketed steps fire only when the shape calls for them; most datasets see four to six questions
in total. The full specification, with copy, firing conditions and fixtures, is
[`OPENING_SEQUENCE.md`](OPENING_SEQUENCE.md).

Nothing may be resequenced. Two of those steps are pre-seal for reasons that are easy to miss:

- **The impossibility pass**, not for leakage reasons — setting a physiologically impossible value
  to missing is row-local and leaks nothing — but because a stratified or grouped split computed
  over corrupted values is a worse split, and impossible entries are normally an exclusion that
  changes N, which belongs in the flow diagram before anything is sealed.
- **Grain**, because the seal cannot be drawn correctly without it. See §02.

**Checked against the code, L12 — Classic contradicts this clause today.** `STATE-101`,
`STATE-102`, `GUIDED-011`. Filed, not fixed; the check was file-don't-fix.

| Clause-01 step | Where it actually runs in Classic |
|---|---|
| load | `pages/01` |
| structural repairs | `pages/01`, via `utils/import_ui.render_import_doctor` — **in order** |
| the impossibility pass | `pages/02_EDA.py:1758`, and its row-dropping form at `pages/05:894` — **after the seal** |
| grain | inferred, never asked (§02; `IMPORT-022`) |
| eligibility | not a distinct step |
| **SEAL** | `pages/01_Upload_and_Audit.py:1106`, at target selection |
| EDA | `pages/02` |

The seal is drawn *first* of the three, not last. Measured consequence: 400 rows with 60 carrying
an impossible value seal 60 test rows at 15%; the page-05 plausibility filter then drops 7 of them
and evaluation runs on 53 while the status chip still reports `n=60`. Two clauses are engaged, not
one — §04 says a robustness trim touches the training partition only and that trimming the test set
to match is *permanently off the menu*, and this filter trims both because it filters the frame
wholesale into `filtered_data`, which `get_data()` serves to every page.

Guided has no seal step at all (`register.json` → `target-lockbox-settings`), so the ordering
cannot yet be got wrong there — or right. That is the one place being behind Classic is an
advantage: build the sequence *with* the seal rather than after it. Classic is the warning, because
its seal and its impossibility pass grew on different pages at different times and reordering them
now means moving a step other pages already depend on.

### 02 · Grain is asked, never inferred

> **"Can one person appear in more than one row?"**

This is the same question multi-file assembly asks (`IMPORT-005`, `IMPORT-015`) and the same one
the lockbox needs. It is asked **once**, pre-seal, and both consumers read the one recorded answer:
a project that arrives through assembly has already answered it and the seal inherits it.

The heuristics (`detect_repeated_subjects`, `rank_grouping_candidates`) are **demoted from source
of truth to two lesser roles**: a *suggestion* offered to the human, and a *contradiction detector*
when the human's answer disagrees with the data's shape. A user who says "one row per person"
while a column repeats three times per value is evidence that somebody is wrong — that earns an
interruption, by the same rule that governs join drops: escalate on evidence of error, never on
the magnitude of a consequence.

Name lists and ratio bounds cannot close this and must not be tuned as though they could. The
engine was guessing at something the user simply knows.

### 03 · The seal states its own basis — three states, never two

> **Grouped by column X · repetition found but grouping abandoned · undetermined**

`undetermined` is first-class: persisted in the lockbox record (never as `group_col: None`, which
a consumer cannot tell from a verified cross-sectional seal), asserted by a test, and **never
rendered as a clean lock.** The failure `IMPORT-020` names is not that detection is hard — it is
that failure to detect was indistinguishable from success.

The asymmetry that settled it: `IMPORT-021` leaks too, and closes anyway, because it *says so*.
Leaking and disclosing is the governing rule's **refuse** branch. Leaking behind a lock icon is
its **assert something false** branch. An undetermined seal is an advisory with exploratory
labeling, not a hard block — a user who genuinely does not know their own data's shape should get
honest numbers, not a locked door.

### 04 · Eligibility and robustness trims are different objects

Two operations that look identical in a spreadsheet and are not:

| | Eligibility criterion | Robustness trim |
|---|---|---|
| What it says | who the model is *for* | how the fit is *stabilized* |
| Applied to | the whole dataset, **pre-seal** | the training partition only, **post-seal** |
| Changes N | yes — reported in the flow diagram with its reason | no |
| Test set | obeys it | never touched |

TRIPOD+AI names continuous-variable restrictions ("e.g. age range") as an eligibility item
reported in participant flow. The eligibility question is asked in **scientific terms** — *does
your research question restrict who is studied?* — with the target's distribution **withheld**,
because an eligibility criterion comes from the research question and not from the histogram. A
rule on the outcome's own range is refused: selecting on the outcome biases the estimates, and no
one whose outcome is still unknown could be screened by it (audit RO-01). If a
user needs to see the shape to decide where to cut, that is data-driven cohort selection, which is
its own publishable bias. The app may show what is needed to answer *"is this data corrupted?"*
(observed min/max, impossible-value flags) and not what is needed to answer *"where should I cut?"*

**"Also trim the test set to match" is permanently off the menu.** A user who truly wants the
narrower population is routed back to the pre-seal eligibility question, which requires a re-seal
and is therefore its own hard, logged decision.

### 05 · The extrapolation obligation fires at the report, not at the trim

A train-only trim is a **legitimate choice**, so it does not earn a blocker — friction is spent
where an operation is almost certainly an error, and this one is not. What is illegitimate is
reporting a single aggregate metric afterward as though nothing happened.

So the trim is a CHOICE that silently **arms a requirement**, and the blocker fires at export if
the stratified in-range / out-of-range breakdown is absent. Same protection, spent at the point
where the error actually occurs, and no tax on a researcher doing something defensible.

### 06 · Declaration and execution are separate, and execution is bound to a data scope

The litmus test, automatable:

> **Does this transform's output for row *i* depend on any other row?**

- **No — structural repair.** Row-local, deterministic, label-free: parse `True`/`False` to
  boolean, coerce a type, fix units, rename, split a delimited field. Zero leakage pathway, so it
  **executes immediately** on the working table and posts a receipt.
- **Yes — statistical transform.** Imputation, scaling, winsorizing, trimming, target encoding,
  feature selection all learn from a distribution. They are **recorded as decisions now and
  executed inside per-model pipelines fit on training folds only.** Materializing one on the
  working table pre-split is the canonical preprocessing leak.

**The router defaults to deferral when unsure.** The user still gets the immediate point-and-fix;
the decision sentence carries the timing as methods prose — *"Missing `age` will be imputed with
the training-fold median"* — which is simultaneously the receipt, the schedule, and the manuscript
line. Never hidden, never a lecture. Forcing a stateful transform to materialize early is a
blocker; a **read-only preview not persisted to the modeling table** is the only permitted
override, and it is labeled *preview, not applied*.

### 07 · Missingness routes by dtype **and** mechanism

Prediction is not inference, and the distinction is load-bearing: the missing-indicator method
discouraged for causal estimation is defensible and often helpful for prediction under informative
missingness.

- **Binary / categorical** — ask first whether the missingness is informative (*"could a blank here
  mean something?"*); in EHR data it usually is. Default to an explicit `Missing` category or a
  missing indicator, which preserve the signal. Imputing an informatively-missing field is a
  blocker with typed acknowledgment, and the **stability assumption** — that missingness means the
  same thing at deployment — is recorded as a methods assumption, because it may not hold across
  sites.
- **Numeric** — the outcome's place in the imputation model is the purpose's, not a universal
  rule (BLUEPRINT §12 ruling 4; audit WP7, which superseded this section's unscoped "never"):

  > The outcome's place in the imputation model depends on the purpose. Under inference, missing predictors are multiply imputed with the outcome and total energy in the imputation model, and the analyses pooled by Rubin's rules (Moons et al. 2006, via Harrell). Under prediction, they are imputed inside each training fold without the outcome, so the fitted pipeline can impute a new row as it was developed (Sisk et al. 2023).

  Under inference a single fill or a missing indicator is blocked and recorded, and complete cases
  stay available with their assumption stated. Under prediction the in-fold, outcome-free fill and
  the indicators stand. Either way an energy-bearing nutrient is filled from its line on total
  energy, and values below a detection limit are never filled by the median without a reason
  (`turbotab/core/methods/missing.py`).

### 08 · What this does not settle

No source gives a missingness rate at which an indicator beats imputation; the app asks rather than
infers. Mechanism stability at deployment is unverifiable at build time and is recorded, not
checked. Whether a train-only trim is worth its extrapolation cost is a per-dataset judgment. And
whether a non-persisted preview biases the analyst's later model choice is an unstudied
cognitive-leakage question — previews stay conservative and labeled.

---

## Appendix · The routing constitution

The lockbox constitution's first sentence refers to this one, which governs which questions get
asked (the legacy roadmap's Decision B, "Router gating policy", from its canonical refinement on).

**Refinement from the L8 implementation — the fact/choice distinction, now canonical:**

> A `high`-confidence finding can settle a question of **fact** — "is this categorical?" — because
> the engine is certain and the transcript can state it. It can never settle a question of
> **choice**. Whether to apply a repair is the user's decision however confident the engine is,
> because applying without preview is the blind consent the preview exists to end.

Repairs are always asked; only detected facts are skippable. And with the palette came the second
clause: **a pull affordance may never be skipped or deferred — ignoring one has to be free, or it
was never pull.**

**Third clause, from the Explore step's register dispositioning — blocker severity:**

> A `blocker`-severity finding is a question of **consequence**. It is always pushed, never
> offered — *a blocker that only offers is not gating.* The tool does not hard-refuse to proceed
> (the user may know the flagged column is legitimate), but passing an unresolved blocker requires
> an **explicit recorded acknowledgment**, and that acknowledgment flows into the record so the
> manuscript can carry it as a limitation. Silence past a blocker is impossible; overriding one is
> a decision the transcript owns.

Fact → skippable at `high` confidence. Choice → always asked. Consequence → always asked, and
exit past it unresolved is itself a recorded decision. This is the routing constitution in three
clauses, and `router.audit()` enforces all of it before any run is scored — a run that breaks a
rule has no number, it has a failure.

Two refinements from the T0-ROUTE-001 build, now binding:

- **Certainty does not make a question of consequence moot.** Being certain a column leaks is a
  reason to ask, not a reason to stay quiet — the one place where `high` confidence must not
  skip. This is why consequence could never fit under fact.
- **Blockers rank first.** A blocker third in a list of nine is a blocker in name only. Ordering
  is part of the gate.
