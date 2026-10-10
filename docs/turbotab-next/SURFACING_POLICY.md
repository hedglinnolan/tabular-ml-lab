# The surfacing policy: what to serve when, where, and why

Written 2026-10-09 against `turbotab-next` at `d0ae80ac`, read-only. It answers Nolan's three
framings: "see what the engine needs to surface that a user decides, dynamically build the tapestry
to show the consequences of their decision-making"; the fear that routing is "if this then that
pathways"; and the thesis that "some of the 'art' we think of as modeling decisions can actually be
decomposed into a science we just haven't bothered to check before". It builds on FOUNDATION, the
crosswalk, the understanding layer, the First look brief, the exploration rule table, the model
contract's §2 and the two engine modules that already carry surfacing logic
(`turbotab/core/consequences.py`, `turbotab/core/quest.py`). It cites `file:symbol` so each claim
can be checked.

The orchestrator's stated approach is: declarations, not paths; order within a stage by consequence,
blockers first, under a load cap; invariants checked by a path fuzzer. §8 pressure-tests it. The
short version: the approach is right in shape and not yet real in substance. The "declarations" are
708 prose strings in `crosswalk.json` plus 15 code predicates in `quest.py:DECLARATIONS`; the
"consequence" that orders a stage is a hand-typed `order` float, because the planner's footprint
(`consequences.diff_views`) exists only per option, on hover; and nothing yet decides, in numbers,
whether a noticing changes the result. So the fuzzer has nothing executable to fuzz, and the triage
sweep Nolan ruled on 2026-10-09 has no engine behind its three dispositions. This document supplies
the missing piece, a **materiality function with a predicted-versus-realized ledger**, and makes
everything else (tier, order, timing, load, canvas, post-fit sequencing, invariants) a projection of
it.

---

## 0 · The policy on one page

1. **One quantity drives surfacing: materiality.** For an item *i* in state *s*, `M(i, s)` is the
   largest movement any alternative to its current answer (or default) would cause in the target
   quantity *Q*, in units of *Q*'s own uncertainty. *Q* is the estimand under Estimate, the honest
   score under Predict, the descriptive quantity under Describe. Before the lock, `M` is computed
   from outcome-blind proxies and theory bounds; after it, from refits. The two are recorded side
   by side in a **materiality ledger**, and the pre-lock proxies are calibrated against the
   post-lock truth on the reference journeys. That ledger is the "science we haven't bothered to
   check": it measures, journey by journey, which modeling decisions mattered.
2. **Tier is a function of blocking, who can decide, and materiality.** A blocker is Decide. An item
   the engine cannot settle by rule (it needs a *meaning* only the researcher has) with `M > 0` is
   Decide. A default with an alternative above τ_confirm is Confirm. Everything else that fired is
   For the record. Nothing that fired is dropped (the coverage test).
3. **Order is a priority topological sort.** The engine's dependency order (`interview.QUESTION_KEYS`,
   `quest.question_reads`) is a hard partial order; within it, the key is
   `(blocker, reach, materiality band, default's uncertainty, dependents, novelty, catalog order)`.
   The same key ranks First look highlights, card rows, Confirm lines, triage rows and exhibits.
   Ranks recompute only when the data or an answer changes, never while browsing.
4. **Timing: Notice at the earliest legal evidence; Own at the first decision it changes; Return as
   one context line per card it shaped.** A noticing has a surface budget (highlight ≤ 1, row ≤ 1,
   context line ≤ 1 per card, triage row ≤ 1) and never appears on the quest line or the bar.
5. **The triage sweep is the Confirm sweep for noticings, in two phases.** Pre-lock the engine
   recommends a disposition from legal evidence and records it; post-lock it verifies every
   "doesn't change your numbers here" and "could bias" by refit where legal, and raises a label on
   the exhibit when the realized movement leaves the predicted band. Never silently.
6. **Load is managed by the same key.** Caps per surface; overflow merges by (subject, consumer),
   then becomes a stated phrase, then For the record. Expertise and speed modes change disclosure
   level, never tier: the set of Decide and Confirm items is identical in every mode.
7. **The canvas follows the footprint, which is the materiality vector** (rows, columns, routing,
   angles). Its argmax picks the layout; captions state the movement in design units before the
   lock and in *Q*-units after it.
8. **After the fit, exhibits are ordered by the floor, then claim strength, then materiality of the
   finding; checks that pass go to the Supplement; post-fit intelligence labels exhibits and drafts
   Discussion sentences and never becomes a Decide.** "Prediction plus explainability begets
   further inference" is allowed exactly as `MODEL_FAMILY_CONTRACT.md` §2.2's leash rows allow it.
9. **Ten invariants, one fuzzer over the fold**, and a handful of goodness metrics that come free
   from the ledger: false-alarm and false-reassurance rates, decisions changed by a noticing, time
   to a bundle that passes the manuscript gate.
10. **The smallest proof** is the dietary NHANES Models stage with three noticings through the
    whole loop (notice, materiality, triage, lock, verification, ledger) and a 2,000-journey fuzzer
    over `decisions.fold`.

---

## 1 · The decision model

### 1.1 What is being decided

At any moment the screen is a function of the state. Four questions per candidate item:

| Question | Output | Decided by |
|---|---|---|
| Is it here at all? | `fires(i, s) ∈ {0, 1}` | applicability: lens, goal, table facts, earlier answers |
| Which label? | `tier(i, s) ∈ {Decide, Confirm, For the record, hidden}` | blocking, who can decide, materiality |
| Where in the stage? | `rank(i, s)`, a lexicographic key | the priority topological sort |
| How much is drawn? | `level(i, s) ∈ {1, 2, 3}` and the views | the one open line is at 3; the rest at 1; pointing lifts to 2 |

The candidate set is the union of four shapes the repo already has: Router questions and other
decision kinds (`quest.QUESTIONS`, `quest.OTHER_KINDS`), declarations with no Router key
(`quest.DECLARATIONS`), findings (`stages/findings.py`, placed by `quest.finding_place`), and the
catalog threads (`quest_noticings.json`, `quest.noticing_place`). Exhibits join after the fit (§6).
Each shape should answer the four questions through one protocol, below.

### 1.2 The inputs, and which are computable today

| Input | Definition | Today | Needs |
|---|---|---|---|
| **blocker** | some feed of *i* refuses with an exit for this purpose, or a refusal kind would fire (`routing_leash`, `estimand.direct_questions`, `methods/batch.py:batch_confounded_with_outcome`) | computable for engine refusals; T1 for threads is prose (`tier` field in the catalogs) | U1: `Feed.effect == "refuses_option"` or `"blocks"` |
| **answerable** | every question *i* reads is answered and every stage its card needs is fresh | `quest.question_reads`, `quest.READ_BY_GATE`, `Declaration.reads`; the Router's `sequence._answers_in_order` | none; extend to threads' `needs` |
| **gate** | the view classes O0–O4 *i* may draw now, by purpose and lock (`UNDERSTANDING_LAYER.md` §1.3's legality matrix) | `consequences.estimates_unseen`, `estimand.served_gate`, `estimand.HOLDS` | U1: a `view: ViewClass` on every item, not only threads |
| **reach** | 3 = touches the exposure (or the outcome for O1/O2), 2 = the adjustment set or predictors, 1 = other analyzed columns, 0 = unused | computable from settled roles (`readings.settled_columns`, `readings.PREDICTOR_ROLES`) | a shared function |
| **materiality `M`** | §2 | partial: `consequences.diff_views` scores per option on a sample; `stages/sensitivity.py:changes_for` post-lock; E-value and robustness value in `models/effects.py` | U4 `notices`, the ledger (§2.6) |
| **uncertainty** | the default's confidence, or the detector's power on these rows | `readings.Reading.confidence ∈ {high, medium, low}`; `readings.by_values` verdicts | a detector-power field per thread (`Notice.power`) |
| **novelty** | changed since you looked (within a project); seen and settled the same way before (across projects, opt-in) | `invalidates` relations exist for contracts; `diff_views` can diff two notice sets | a per-user, per-lens memory of confirmed readings (S); a tolerance per thread |
| **prior answers** | the log | `decisions.fold`, `ProjectState` | none |
| **expertise** | an explicit mode only (§4.4) | none | one setting |
| **load cap** | per surface (§4.1) | the numbers in `UNDERSTANDING_LAYER.md` §2.8; no gate | U5 |

Everything above is a pure function of `(state, facts, artifacts)`, as `quest.quest_log` already
is. That is the property that makes the policy testable: the fuzzer feeds states and checks outputs.

### 1.3 The tier rule

```
tier(i, s):
  if not fires(i, s):                                  hidden        # never listed; "checked clean" if it is a thread
  elif blocks(i, s):                                    Decide        # T1: nothing downstream is valid until it is answered
  elif decides_by(i) == "meaning" and M(i, s) > 0:      Decide        # a reading only the researcher can settle, and it matters
  elif changes_question(i):                             Decide        # substitution vs addition, mean vs median: never graded by a number
  elif has_default(i, s) and M_alt(i, s) >= tau_confirm: Confirm      # a default whose best alternative moves a number
  elif has_default(i, s) or settled_by_values(i, s):    For the record
  else:                                                 Decide        # it fired, nothing can answer it but the person
```

Where `decides_by` is the understanding layer's `DecidesBy` (`UNDERSTANDING_LAYER.md` §1.2:
meaning, fixed_rule, resampled_candidate, disclosure), `has_default` means a contract option or a
`stated` tier exists, and `M_alt` is the materiality of the best alternative to the default (§2).
This replaces three hand-kept tables in `quest.py`: `STATED` (task and clusters labeled For the
record when skipped), `TIER_RULINGS` (two overrides with prose reasons) and the Confirm-or-Record
choice in `_question_lines` (`STATED.get(key, CONFIRM)`). Overrides may stay, but each must carry a
test asserting that the computed tier would differ, so a ruling that stops being needed is noticed.

For a thread the rule reads through its feeds (`Feed.stage`, `Feed.consumer`, `Feed.effect`):

- **Decide** (a row on its card) while a number-changing feed for this purpose is unanswered and
  `M > 0`;
- **context line** (one per card, the highest-ranked) once that consumer is answered and the thread
  shaped it, from the consumed-by log (`reads_threads` on the stage artifact, §1.4 of the
  understanding layer);
- **triage row** if still open at the gate (§3.3);
- **For the record, "checked clean"** if it did not fire, with its measure kept for the supplement.

### 1.4 The rank key

Within a stage, items are sorted by a **priority topological sort**: the dependency order is a hard
constraint (an item is never listed above a question it reads; `quest.question_reads` and
`Declaration.reads` give the edges, and `sequence._answers_in_order` already refuses the reverse),
and among the items free to move, the key is

```
rank(i, s) = (
  0 if blocks(i, s) else 1,              # blockers first, within the answerable frontier
  -reach(i, s),                          # 3 exposure/outcome, 2 adjustment/predictors, 1 other analyzed, 0 unused
  -band(M(i, s)),                        # 2 act on it, 1 could bias, 0 below noise (§2.3)
  -uncertainty(i, s),                    # a low-confidence default asks earlier than a high-confidence one
  -dependents(i),                        # how many later items read its answer (from the stage graph)
  -novelty(i, s),                        # changed since you looked > never seen > settled the same way before
  catalog_order(i),                      # the crosswalk's `order`, kept as the stable tiebreak
)
```

Three things follow. The key is explainable: its first differing component is the item's "Why is
this here?" line, as `FIRST_LOOK_BRIEF.md` §3.2 already does ("Changes a number (the energy model) ·
touches your exposure · every row"). It is stable: `reach`, `band` and `uncertainty` change only when
data or an answer changes, so the list never re-sorts while someone browses (brief §3.4 rule 2). And
it is one key for every surface, so a noticing that is highlight 1 in First look cannot be row 11 on
its Models card: `FIRST_LOOK_BRIEF.md` §3.2's `(tier, reach, share, excess, column order)` becomes a
projection of this key, with `(share, excess)` folded into `M`.

**Example, dietary NHANES (the `dietary-inference` capture: 21,849 adults, `sugar` → `glucose`).**
In Models, the Router's order is estimand, adjustment, time-varying, energy, forms, families, then
the declarations (`CROSSWALK.md` §5). The policy keeps that spine (it is the dependency order) and
orders the noticings that land on those cards by the key. `diet-energy-carries-the-nutrient`
(reach 3: it touches `sugar`; `changes_question`: substitution versus addition; r(sugar, kcal) =
0.64 on the brief's extract) sits first on the energy card as a Decide. `diet-implausible-reporters`
(reach 1 before the lock: the Goldberg screen reads weight and height, not the outcome; M from the
flow: 5,212 of 21,849 rows leave, 23.9%) is a Decide on Who's in's exclusions card, with the every-
row sensitivity analysis pre-filled as a Confirm in Models (`decision:set_sensitivity`, Banna 2017).
`diet-day-to-day-variance` cannot measure λ on this extract (one recall day, `set_column_unit`
recorded `days: 1`), so its uncertainty is maximal and its band is "could bias (not measurable
here)": it never becomes a row, it goes straight to the triage sweep as a pre-filled limitation,
and the related `diet-borrowed-validity-coefficients` sentence waits for the fit (§6).

**Example, metabolomics prediction (`metabolomics_untargeted.csv`: 80 injections, 72 participants,
8 pooled QCs, two batches, 392 features, `responder`).** `metab-run-order-aligned-with-outcome` is
T1 when it fires (an O2 design verdict, legal before the draw as counts). On this fixture order
versus responder gives AUC 0.545: it does not fire, so it is For the record with one methods line.
`metab-qc-drift` (64% of features drift against about 0.7% by chance; reach 2) is Decide on the
drift card, where QC-RLSC reads "Not available yet" (4 QCs per batch, the engine needs 5;
`methods/qc_drift.py`) and batch-as-term is the exit. `metab-left-censored-nondetects` (ρ = −0.99
between blank share and log mean intensity) is Decide on the fill card in Models.
`metab-wide-noise-ceiling` (p > n) is surfaced, never asked: its feed is a baseline and a label, so
it is For the record before the fit and an exhibit label after it.

### 1.5 Disclosure level

- **Level 3** for exactly one line per screen: the open Decide (FOUNDATION §3). The triage sweep and
  the Confirm sweep are at level 3 as a whole, with their lines at level 1.
- **Level 1** for every other fired item: the line, its answer or default, one clause of reason for
  a Confirm.
- **Level 2** on pointing or focus: one clause on what it does to this table, the quiet technical
  name, the tapestry lit.
- **Not listed individually:** items whose `reads` are unanswered. FOUNDATION §3 lists each as
  "Waiting for: [question]"; this policy collapses them to one line per stage, "N more questions
  wait for earlier answers", with the list behind it (§8.4, text noise). An item that *becomes*
  answerable moves into the list at its rank, which is what "lines still to come wait below" means.

### 1.6 One protocol

The four shapes of §1.1 should implement one interface, in code, not prose:

```python
class Surfaceable(Protocol):
    key: str
    def fires(self, state, facts) -> bool: ...          # applicability (quest.Declaration.applies today)
    def reads(self) -> tuple[str, ...]: ...              # questions it waits for (Declaration.reads)
    def needs(self) -> tuple[str, ...]: ...              # stages whose artifact its card reads (interview.NEEDS)
    def view(self) -> ViewClass: ...                     # O0–O4: what its evidence looks at
    def holds(self, state) -> bool: ...                  # whether a recorded answer still stands (Declaration.holds)
    def decides_by(self, purpose) -> DecidesBy: ...
    def materiality(self, ctx) -> Materiality: ...       # §2: pre-lock proxies, post-lock refit
    def consumers(self) -> tuple[Consumer, ...]: ...     # first consumer sets the stage; later ones return it
    def sentences(self) -> Sentences: ...
```

`quest.Declaration` already has `applies`, `reads`, `holds`, `counted`; the thread contract's
`Notice`, `Meaning`, `Feed` have `fires`, `needs`, `view`, `decides_by`. The protocol unifies the two
and adds `materiality`. The crosswalk's `fires_when`, `depends_on`, `reopens_when` stay as
documentation until each item gains a `rule: "module:function"` field that points at its predicate;
the registry test treats a prose-only item as unwired.

---

## 2 · Materiality: deciding between "doesn't change your numbers", "could bias" and "act on it"

This is the hardest call because the honest answer, "fit it both ways and compare", is illegal before
the lock under Estimate (FOUNDATION §5 rule 6; `consequences.estimates_unseen`) and is the very
thing the triage sweep must recommend before the lock. The resolution is to separate **what we may
measure before the lock** from **what we verify after it**, record both, and let the second calibrate
the first.

### 2.1 The definition

For item *i* with current answer or default `a₀` and alternatives `A`, and target quantity *Q*:

```
M(i, s) = max_{a ∈ A} | Q(a) − Q(a₀) | / scale(Q)
```

- Under **Estimate**, *Q* is the declared estimate (the primary coefficient, risk difference or
  ratio) and `scale(Q)` is the half-width of its 95% interval. A movement of 1 means the alternative
  moves the point estimate to the edge of the primary's interval.
- Under **Predict**, *Q* is the declared result (`models/selection.py:declared_result`: BBC-corrected
  or nested-CV score) and `scale(Q)` is the fold-to-fold standard deviation of the paired difference
  on the same folds (`fit.objects["comparison"]`, as `stages/evaluation.py` already pairs the
  benchmark against each family).
- Under **Describe**, *Q* is the descriptive quantity (a weighted mean, a prevalence, the usual-intake
  percentile) and `scale(Q)` its interval half-width.

Two special values. **`changes_question`**: when an alternative changes what *Q* is (substitution
versus addition, `diet-energy-carries-the-nutrient`; mean versus median through a robust loss;
total versus direct effect; consumers-only domain), `M` is undefined and the item is Decide
regardless (§1.3). **`not_measurable`**: when the detector's power on these rows is too low to
measure its proxy (one recall day for λ; fewer QCs than the correction needs), `M` is unknown, and
unknown is handled as "could bias" with the reason "not measurable here" (§2.4).

### 2.2 Two regimes, one ledger

| Regime | When | What may be read | Instruments |
|---|---|---|---|
| **Predicted** (`M_pre`) | before the lock under Estimate and Describe; before the draw under Predict; and on hover at any time | O0 always; O1 counts and O2 design counts; under Predict O1 and O3 on training rows after the draw | theory bounds and design-space movement (§2.3) |
| **Realized** (`M_post`) | after the lock under Estimate and Describe; in-fold at any time under Predict | the fitted model, as a labeled secondary or a paired refit | refit movement, E-value-style benchmarks (§2.3) |

Both land in the **materiality ledger**: one row per (noticing or default, alternative, journey or
project) with `M_pre`, its instrument, the recommended disposition, the recorded disposition,
`M_post`, its instrument, and whether the realized band matched the predicted one. The ledger lives
in the record (a stage artifact derived from the log, never a new mutable store, as the understanding
layer's consumed-by log does), so replay reproduces it and the supplement can print it.

### 2.3 The instruments, by legality

**Legal before the lock (outcome-blind, or counts the matrix allows):**

1. **Theory bounds.** Where a result gives the movement without the outcome, use it exactly.
   - Regression dilution: with one error-prone exposure and reliability λ, the slope is attenuated
     by 1 − λ (Hutcheon et al. 2010; `MODEL_FAMILY_CONTRACT.md` §2.4's entry). λ from the recall
     variance decomposition (`diet-day-to-day-variance`; `usual_intake` and
     `methods/calibration.py`) gives `M_pre = (1 − λ) · |β| / half-width`, which still needs |β|;
     before the lock report the attenuation factor itself and band it: 1 − λ ≥ 0.5 is "act on it"
     (declare regression calibration), 0.2–0.5 "could bias", < 0.2 "doesn't change your numbers
     here". On `dietary_recalls.csv` λ for a two-day mean is 0.36–0.39, so 1 − λ ≈ 0.6: act.
   - Precision bounds: Kish's effective sample size under weights, events per parameter, Riley's
     minimum (`models/selection.py:shelf_order`), the rank of a closed composition (an all-parts
     model with energy is singular: a refusal, hence a blocker).
   - Known refusals: CLR in an unpenalized GLM with an intercept; a screen that reads the outcome
     (`stages/sensitivity.py`, audit RO-01); an outcome-protected ComBat
     (`methods/batch.py:outcome_combat_leaks`). These are blockers, not bands.
2. **Design-space movement**, from the planner's own footprint on the 5,000-row sample
   (`consequences.PreviewContext.sample_size`, `consequences.diff_views`):
   - rows: the share of rows the alternative changes (`_rows_view`), times the largest standardized
     mean difference between leavers and stayers on the adjustment set (`shared-missingness-
     mechanism`'s own rank signal, "rows lost × max |SMD|"). A rule that removes 23.9% of rows whose
     leavers differ from stayers by SMD 0.4 on BMI is in a different band from one that removes
     0.3% with SMD < 0.05.
   - values: the standardized Wasserstein shift of the exposure column under the alternative
     (`consequences._shifts`: W1 / sd), and the change in exposure–covariate correlation
     (`_relationship_change`: |Δr|, already thresholded at `RELATIONSHIP_DELTA = 0.2`).
   - routing: columns moving in or out of the model (`_columns_view`), weighted by reach.
   - design: imbalance of a process variable over the outcome, as counts (O2): batch × outcome,
     run order × class as rank-biserial r (`metab-run-order-aligned-with-outcome`).
   - excess over reference: the detector's measure beyond its chance share or convention
     (`detectors/assay.py`'s drift reading is the template: 64% of features drift against 0.7% by
     chance).
3. **Under Predict, in-fold refits** are legal at any time on training rows and are the realized
   instrument from the start: the alternative runs as a candidate inside each fold, and
   `M = |Δscore| / sd_fold(Δ)` paired by fold, exactly as `models/selection.py:interpretable_cost`
   pairs the benchmark. U9's honest-score ladder is this instrument for process baselines.

**Legal after the lock (Estimate and Describe):**

4. **Refit movement.** The alternative is fit as a labeled secondary on the locked plan's state
   (C7b), and `M_post = |Q_alt − Q_primary| / half-width`. `stages/sensitivity.py:changes_for`
   already does this for exclusion rules (Banna 2017's every-row analysis beside the screen);
   `exhibit:which_decisions_mattered` is the same over every declared alternative; `stages/
   calibration.py` gives the corrected coefficient beside the uncorrected one. None of it changes
   the primary.
5. **E-value-style benchmarks** for what cannot be refit because it was never measured: the
   Cinelli–Hazlett robustness value with each adjusted covariate as a benchmark
   (`models/effects.py:robustness_value`, `BenchmarkResult`), the E-value for the estimate and for
   the limit nearer the null (`e_values`). For a noticing about a *mishandled measured* factor
   (treated values, a mismeasured confounder, a season unevenly spread), the benchmark "a confounder
   as strong as `age` would move it to X" is the movement bound, read against the half-width.
6. **Detectability.** When no instrument applies, the realized row says "not verifiable", and the
   pre-lock disposition stands as recorded, with its reason.

### 2.4 The bands, and the disposition they recommend

| Band | `M` | Pre-lock recommendation | After verification |
|---|---|---|---|
| 0 · below noise | `M < τ₀` | "Doesn't change your numbers here" → a supplement line (checked, with the numbers) | confirmed, or **upgraded** with a label on the exhibit if `M_post ≥ τ₀` |
| 1 · could bias | `τ₀ ≤ M < τ₁`, or `not_measurable` | "Could bias the estimate" → a sensitivity analysis is declared where one exists (then a supplement line with its result); a limitation sentence only where nothing can be done (Nolan, 2026-10-09) | confirmed, or **downgraded** to band 0 wording ("checked; it did not") |
| 2 · act on it | `M ≥ τ₁`, or a sign change, or crossing the null, or `changes_question` | "Act on it" → points to the decision; cannot be confirmed past | n/a before the lock; after the lock a band-2 realization becomes a labeled secondary and a Discussion sentence |
| blocker | any | must be resolved | n/a |

The thresholds are conventions, to be set on the reference journeys (§2.6) and recorded with their
reason, as every other convention in the repo is (`MODEL_FAMILY_CONTRACT.md` §2.1 sets its "most"
at 0.8 the same way). Starting values: `τ₀ = 0.1`, `τ₁ = 0.5` in *Q*-units for realized movement;
for the pre-lock proxies, per instrument: rows `share × SMD ≥ 0.05` for band 1 and `≥ 0.2` for
band 2; exposure shift W1/sd `≥ 0.1` / `≥ 0.5`; |Δr| `≥ 0.1` / `≥ 0.3`; attenuation 1 − λ `≥ 0.2`
/ `≥ 0.5`; design imbalance rank-biserial `|r| ≥ 0.1` / `≥ 0.3`. Pre-lock, "doesn't change your
numbers here" is a claim about the *design*, and the supplement line says so: "This rule removes 61
rows (0.3%) who differ from the others by at most 0.04 standard deviations on what is adjusted
for; the estimate was checked on both sets after the plan was fixed and moved by 0.02 of its
interval."

### 2.5 Per family: evidence, instruments, failure modes

The 17 families (`UNDERSTANDING_LAYER.md` §3.1) group by where error enters, which is also where
their materiality instrument comes from.

| Family | Pre-lock evidence (legal class) | Post-lock verification | Default band when it fires | The failure each guards against |
|---|---|---|---|---|
| S1 Shortcut | O2 counts: process variable × outcome imbalance; under Predict a process-only in-fold baseline (U9) | the baseline beside the model, paired by fold | 2 if the baseline is within τ₁ of the model; T1 under perfect alignment | a model that learns the instrument; crying wolf on a randomized run (metabolomics: AUC 0.545, silent) |
| S2 Leak in time | the timing reading per column (meaning); under Predict a column's in-fold association ordering the timing rows | leave-one-period-out | T1 for a column recorded after the moment of use, else 1 | a predictor that is the outcome's shadow |
| S3 Leak in meaning | the meaning reading (a diagnosis column defines the outcome) | none needed: it is a definition | T1 | tautology |
| S4 Not independent | repeats and groupings from structure (O0); design effect | cluster-robust versus naive interval ratio | 2 when the unit repeats and no clustering is declared | intervals too narrow by the design effect |
| S5 Who is in | rows share × SMD of leavers versus stayers (O0) | the every-row analysis (`stages/sensitivity.py`) | by the rows proxy | selecting on the outcome; 5,212 rows leaving quietly |
| S6 Done before upload | readings of pre-processing (meaning) | a reversal where one exists (already-transformed values) | 1, or 2 when a transform would be applied twice | double logging; a residual fitted on residuals |
| S7 Drift and transport | drift share against chance (O0, reference rows); QC RSD | correction on versus off, paired by fold, or after the lock as a disclosed pair | by excess over chance; 2 when > 50% of features drift | drift read as biology |
| K1 What a value means | the reading's confidence and its value test (`readings.by_values`) | n/a: a wrong unit is a wrong number everywhere | 2 when unsettled and read by a number-changing consumer (`readings.Consumer.changes`) | pounds read as kilograms |
| K2 What a zero or blank means | the day-1 × day-2 zero table against independence (O0); missingness against intensity | the fill repeated under a second scheme; the two-part model beside the one-part | 2 for episodic zeros that a transform would log | log(x+1) on days without the food |
| K3 How well it measures | λ, QC RSD, D-ratio (O0, reference rows) | the calibrated coefficient beside the uncorrected one | by 1 − λ; `not_measurable` with one measurement | a null read as no effect when the slope is attenuated by half |
| K4 Causal place | the adjustment answers (meaning); overlap and positivity (O0) | the "further adjusted for" model (`stages/secondary.py`); benchmarks | 2 for a mediator in the set; 1 for unknown timing | adjusting away the effect |
| K5 Structure among variables | |r| ≥ 0.9 pairs (`stages/explore.py:COLLINEAR`), closure, nesting | grouped importance; the set with and without the pair | 1; T1 for a singular design | two strong predictors each called minor |
| K6 Outcome and clock | follow-up spread, event counts (O1 counts) | the time-to-event model beside the yes/no one | 2 when follow-up varies and a yes/no model is declared | censoring read as absence |
| K7 Reference and context | weights, strata, the population answer (O0) | weighted beside unweighted | 2 when weights exist and are not used | describing the sample as the country |
| E1 Support | events per parameter, Riley's n, p/n (O1 counts) | the spline benchmark versus the flexible family | 1 below Riley's minimum; 2 at p ≥ n for an unpenalized fit | a flexible family on too few rows |
| E2 Noise and multiplicity | the family size, the permutation ceiling's expectation at this n and p (O0) | the permutation null of the honest score; q-values | 2 for a declared family with no multiplicity answer | the best of 392 null features reported |
| E3 Reading the result | the declared smallest effect that matters (meaning), if any | the interval against it (`shared-null-is-inconclusive`) | silent until the fit; 1 after it when the interval includes the SESOI | "no association" for an inconclusive interval |

**Two failure modes the bands must balance.** *False reassurance*: the engine says "doesn't change
your numbers here" and the realized movement is 0.6 of the interval. The guard is structural: band 0
is never final until verified, the verification is mandatory wherever an instrument exists, and an
upgrade is a label on the exhibit plus a limitation draft, never a silent edit. *Crying wolf*: every
noticing is "could bias", the paper carries nine limitation sentences, and the user stops reading
them. The guards: a limitation sentence appears only when `M ≥ τ₀` *and nothing was done* (Nolan's
rule); a sensitivity analysis counts as something done and moves the row to the supplement; the
triage sweep caps its rows (§4.1) and merges by family; and the calibration (§2.6) is run against a
budget on limitation sentences per journey.

### 2.6 Calibration on the reference journeys

The twelve reference journeys (`review-packets/captures/*.json`, `CROSSWALK.md` "The reference
journeys against the load caps") are the calibration set. For every noticing and default that fires
on them:

1. compute `M_pre` with its instrument at the moment the triage sweep would show it;
2. lock, fit, and compute `M_post` with its instrument;
3. record both in the ledger (`materiality_calibration.json`, committed; the registry test reads
   it).

Then choose `τ₀`, `τ₁` and the per-instrument proxy thresholds to **minimize false reassurance
(band 0 predicted, band ≥ 1 realized) at a budget on cry-wolf** (at most *k* limitation sentences
per journey, with *k* = 3 for the dietary inference journey as the understanding layer's "4
surfaced" already implies). Report the confusion matrix per instrument in the calibration file.
A threshold change is a recorded methods decision with a reason. Where the calibration set has no
case for an instrument, the proxy is marked uncalibrated and its band is capped at 1: the engine
may say "could bias" but never "doesn't change your numbers here" from an uncalibrated proxy.

**Amended 2026-10-10 (WAVE_C6A_PLAN §7 ruling 2).** An instrument that is exact by theorem is not
floored at band 1. Examples are Frisch–Waugh–Lovell, and a reparameterization of a column that is
not focal (not what you study, not the outcome, not a declared modifier): either leaves the model
matrix's column space unchanged, so the focal estimate, its interval and the fitted values move by
exactly 0. Such an instrument (`invariance` in `materiality_calibration.json`) may say "doesn't
change your numbers here" without calibration cases. `calibrate` keeps it as it is, and the ledger
names the theorem (the entry's `theorem`, carried on each movement it measures). Every proxy that is
not exact keeps the floor.

This is the concrete form of Nolan's thesis. Each journey adds rows that say which decisions moved
which estimates by how much. After a dozen journeys the ledger is a small, honest, replayable
dataset on the materiality of modeling decisions in nutrition and omics data, and it is what
`exhibit:which_decisions_mattered` prints for one paper.

---

## 3 · Timing and placement

### 3.1 The rule

An item is **noticed** at the earliest moment when its evidence is computable under its gate and
cheap enough to compute on this path (`Notice.cost`: instant or pass before the seal; fit and heavy
scheduled, never on hover). It is **owned** (asked) at the first decision it changes for this
purpose: the earliest stage among its number-changing feeds, by `quest.kind_place(feed.consumer)`.
It **returns** at every later feed as the card's one context line, with "See it" bringing its view
back as the secondary view, and at the sentence it shaped as the manuscript's "noticed" mark.

The three moments are usually different stages, and that is the design, not a defect: First look is
where O0, O1-count and O2 noticings are shown ("Comes up at Energy"); Models is where most are
answered; Results and Write-up are where they return. First look's "Decide now" opens the owning
card early, and is legal only when every question that card reads is answered
(`quest._declaration_lines`' `unanswered` check, and the Router's `sequence._answers_in_order`)
and its preview obeys the gate (`consequences.estimates_unseen`).

### 3.2 Where each shape lives

Unchanged from the stage registry: Router questions by `quest.QUESTIONS`, other kinds by
`quest.OTHER_KINDS`, findings by `quest.finding_place` (held for a question, routed to it, else Your
data), noticings by `quest_noticings.json`. Two additions: the stage of a thread is derived from its
first number-changing feed (so the JSON becomes a generated check, not a source), and a thread with
no number-changing feed for this purpose (its tier is `surfaced`) lives where its label or
disclosure lands: on the exhibit in Results, or in the IDA paragraph.

### 3.3 The triage sweep's place and shape

Under Estimate and Describe, just before the analysis flowchart; under Predict, in Results before the
held-out rows open (crosswalk "Settled here"). It lists every noticing still open that feeds the
plan, each with its recommended disposition and reason, grouped by family when more than six, with
blockers first and "Confirm all" clearing the rest. Its state machine adds two decision kinds to the
understanding layer's `leave_open`: `dispose(thread, subject, disposition, reason)` and the
system's `verify(thread, subject, M_post, band)` written after the fit. Every row's disposition is in
the log before `lock_plan`, and the lock's digest covers the list (`plan_lock.plan_of` reads every
slot upstream of an estimate stage, so adding the dispositions as a slot puts them under the hash
without a new list to keep).

### 3.4 How it avoids nagging

A noticing has a **surface budget**, enforced by the state machine of `UNDERSTANDING_LAYER.md` §2.1
and a `seen` set per (thread, subject, surface) kept in the derived record:

| Surface | Budget | When it returns anyway |
|---|---|---|
| First look highlight | 1 | only as "Changed since you looked", when its measure crosses its own tolerance after a decision reruns `notices` |
| Row on its card | 1 per (card, kind, subject) | re-asked when stale (`invalidates`), with the reason |
| Context line | 1 per card it shaped | never twice on one card; the rest under "Why does this matter?" as "Also shaped by" |
| Triage row | 1 | never, once disposed |
| Exhibit label | 1 per exhibit | an upgrade from verification replaces the label, never adds one |
| Manuscript mark | 1 per sentence | n/a |
| Quest line, progress bar | 0 | never |

Two noticings that share a subject and a first consumer merge, the second becoming the first's
context view (brief §3.3). A specialization merges into its shared thread (understanding layer
Appendix B). A noticing the user dismissed with a value ("does not apply") is settled, not open: it
does not return in the triage sweep.

---

## 4 · Load management and calm

### 4.1 Caps, per surface

| Surface | Cap | Overflow goes to |
|---|---|---|
| Open Decide lines at level 3 | 1 | the rest at level 1 |
| Noticing rows on one card | 3 | merged by (subject, consumer), else a stated phrase under the card |
| First look highlights | 3, at most 2 per group | the group's list, "and N more" |
| Context lines per card | 1 | "Also shaped by" under "Why does this matter?" |
| Confirm sweep lines shown | 8 | "and N more, each read from your data", collapsed, all still confirmable (audit §4.3: above eight it is a settings page) |
| Triage sweep rows | 6 | grouped by family with a count, each group expandable |
| Views on the tapestry | 3 (`consequences.MAX_VIEWS`) | "More angles" |
| Coach notes per view | 2 (`consequences.MAX_COACH`) | dropped, never cut mid-claim (`coach.note`) |
| Exhibits open in Results at level 3 | 1 | the list |
| "Waiting for" lines | 0 individually | one count line per stage |
| Limitation sentences drafted | *k* per journey (3 for dietary inference), by the calibration budget | a supplement line with its numbers |

Overflow never hides: the coverage test (U13) requires every fired item to be a row, a context
line, a triage row, a stated phrase, an exhibit label or a checked-clean record. The U5 gate reports
asked rows per reference journey against the catalogs' targets (`UNDERSTANDING_LAYER.md` §2.8) and
fails the build when a journey exceeds them.

### 4.2 Decide load, not only noticing load

The caps above are mostly about noticings, which is where the understanding layer put them. The
larger load in Models is the Decide lines themselves: 71 Decide items fit the dietary inference
journey's goal and lens in `crosswalk.json` before any `fires_when` is read, against 8 engine cards
answered in the capture. The difference is `fires`. So the single most effective load control is
an **executable `fires`** per item (§1.6): the quest log lists what fires, counts only what fires,
and the fuzzer measures the Decide count per journey. A stage whose fired Decide count exceeds a
target (Models: 10 under Estimate) fails the U5 gate just as a noticing overflow does.

### 4.3 Text noise

The audit (§4.2) is right that FOUNDATION §3 replaced color noise with text noise. Three controls,
each a test:

- **A screen word budget.** The card at rest (every line at level 1, one at level 3) for each
  reference journey's each stage renders under a budget (120 words is a starting point), measured
  by a test over `quest.quest_log` plus the card copy, the way `consequences` tests enforce
  `CAPTION_WORDS` and `TITLE_WORDS`.
- **A device budget.** At most four distinct label devices on one screen (a quiet label on an option,
  "changed by you", a reopen line, a "noticed" mark); the rest wait for pointing. The purpose
  registry entry of each device names which it displaces when the budget is hit.
- **One word for one thing.** "Decide · Confirm · For the record" are the only section heads; the
  triage sweep is a Confirm sweep of noticings and uses the same head, so "noticings" is named in
  the two places the DoD allows and nowhere else.

### 4.4 Expertise and speed modes, without hiding anything material

One explicit setting, "I know this field", and one per-user, per-lens, opt-in memory of readings
confirmed with the same value on earlier projects. They compress **disclosure**, never **tier**:

- a Decide whose recommended answer equals the user's answer on ≥ 2 earlier projects in this lens
  starts collapsed at level 1 *with that answer pre-filled and marked "as before"*, still counted,
  still one click to open, still recorded as the user's answer only when Continue is pressed;
- "Why does this matter?" and the two-register term stay behind their disclosures as they are; the
  mode does not remove them;
- the Confirm sweep is unchanged, because it is already one click;
- the First look walk ("Look at the first → Next") becomes "Continue" on a revisit (brief §9 Q2).

The invariant the fuzzer checks: for every state, the set of items with tier Decide or Confirm is
identical across modes; only `level` differs. Speed of answering is not an input: it is unreliable
and would be read as surveillance.

---

## 5 · The canvas

### 5.1 Footprint is the materiality vector

FOUNDATION §5 defines the footprint as rows, columns, routing and angles, and picks the layout by
it. The policy makes that literal: the footprint of an option is the vector of design-space
movements §2.3 computes, (rows share × SMD, max column shift W1/sd, routing count × reach, declared
angles), and the layout is its argmax, Flow, Focus or Strip (one column versus several), Routing,
Angles. `consequences.diff_views` already scores rows, columns, values and relationships on one scale
and sorts views by it; the footprint is those scores kept, not discarded after sorting, and exposed
on `PreviewResult` as `footprint: Footprint` so the card's "Why is this here?" and the layout read
the same numbers.

### 5.2 At rest, on pointing, on click

- **At rest** the tapestry shows the open line's subject in gray in the layout its options use
  (FOUNDATION §5 rule 8), with the one line saying what it is. For a Decide shaped by a settled
  noticing, the context line's "See it" view is the secondary view at rest, so the lesson is on
  screen before anything is pointed at.
- **On pointing** an option, its footprint lights in indigo and the caption states the movement in
  design units before the lock ("removes 5,212 rows; those leaving are 7 years older") and in
  *Q*-units after it ("moves the estimate by 0.3 of its interval").
- **On click** the option's three views, linked by column and step keys (rule 4), with one flip and
  one storyboard for all three (rule 3).
- **Nothing changes** says so in one line when the footprint is below τ₀ in every component
  (rule 7), and that line is the same words the supplement will use.

### 5.3 Noticings on the canvas

A thread declares its layout (`Thread.layout`; the catalogs' `layout` field) and its measure beside
its reference is the view (brief §3.2). Where it returns as a context line, "See it" restores that
view as the secondary. The triage sweep shows the selected row's view. After the fit, a noticing that
labels an exhibit draws nothing of its own: the label sits on the exhibit (§6).

### 5.4 Gates on the canvas

Every view's class is declared (O0–O4) and checked against the gate before it is served: the outcome
alone opens after Who's in under Estimate, beside a column after the lock, both after the draw under
Predict, the strictest among several tracks (FOUNDATION §5 rule 6; `crosswalk.json:order_audit.
gates`). The policy adds nothing here except the test: the fuzzer asks the planner for every option
of every open line in every generated state and asserts no view of a class the gate forbids.

---

## 6 · After the fit

### 6.1 The pivot and the floor

Results opens with the lock line and digest (`plan_lock.digest`) under Estimate and Describe, with
the declared cross-validated result under Predict. The floor is fixed and first: the locked primary
(`exhibit:table2`) or the declared score, always in Results, placement not editable; every analysis
run is listed in the supplement; no family trimmed by p-value; explanations never worded as effects;
causal wording for trials only (FOUNDATION §8).

### 6.2 Ordering the exhibits

Exhibits are lines, so they take the same key, projected:

```
rank(exhibit) = (
  0 if on the floor else 1,
  -claim_strength,                       # estimated effect > association > descriptive > describes the model > secondary > inconclusive
  -materiality_of_finding,               # how far it moves what a reader would conclude: |Q − null| / half-width for a result; M_post for a sensitivity row
  0 if pre_included else 1,              # declared before the lock
  catalog_order,
)
```

Default placement follows the crosswalk's "The exhibits" table and the audit's advice: most go to
the Supplement with placement pre-filled, the user promotes. Checks that pass are clean checks and
stay in the Supplement (ruling of 2026-10-06). The verification rows of §2 are themselves exhibits:
a sensitivity row whose `M_post` reached band 1 or 2 is promoted by default from the Supplement to a
Results sentence ("The estimate was 0.31 with the Goldberg screen and 0.19 without it"), which is
how the triage sweep's promise is kept in the paper.

**Dietary inference, in order.** The lock line; Table 2 (floor); the substitution curve, declared
before Fit (`decision:substitution-pair`: 100 kcal of sugar for protein); the sensitivity table,
Goldberg against every row (Banna), promoted if `M_post ≥ τ₀`; unmeasured confounding (robustness
value with age, gender and kcal as benchmarks; the E-value), one Discussion sentence and the
Supplement table, never a pass or fail (`stages/effects.py:sensitivity_reading`); diagnostics, clean
checks to the Supplement; "Which of my decisions mattered?" over the declared alternatives
(Supplement); regression calibration not computable on one recall day, so the drafted limitation
("there is no second measure here; a null cannot rule out a slope several times larger",
`diet-borrowed-validity-coefficients`) is the Discussion draft; the inconclusive-null label fires
only if a smallest effect that matters was declared (`diet-meaningful-increment`), else silent.

**Metabolomics prediction, in order.** The declared result with its basis (BBC-CV AUC over repeated
folds); the noise-ceiling label on it (the best of 392 null features reaches about 0.69 here; a top
feature at 0.70 is "within what noise reaches"); the process-only baseline beside the model (run
order and batch alone: silent here as one methods line, since order versus responder was 0.545);
drift correction on versus off, paired by fold (the materiality verification of `metab-qc-drift`
under the batch-as-term exit, since QC-RLSC was not available); the fill scheme repeated (half-
minimum against QRILC); calibration; the decision curve only if an intended use was declared;
explanations labeled "describes the model", with adducts grouped (`shared-collinear-predictors`);
the Rashomon label if two families tie across 50 refits; panel instability across refits; the
hypothesis noticing (§6.3) only if a flexible family clearly beats the spline benchmark and 72 rows
meet Riley's minimum, which they almost certainly do not: the noticing stays silent and the card
says nothing.

### 6.3 Post-fit intelligence and its leash

`MODEL_FAMILY_CONTRACT.md` §2 defines four readouts. The policy places them without adding a Decide:

| Readout | Tier after the fit | Surface | Condition to appear |
|---|---|---|---|
| Target alignment (§2.1) | For the record | behind "More angles" on the shrinkage view; one card sentence | Σ âⱼ above its permutation floor; thresholds set on the journeys before any card fires |
| The hypothesis noticing (§2.2) | For the record under Predict; never under Estimate | an "Exploratory: suggested by these rows" label on the explanations; a Write-up sentence; one action, "Check it across refits" | the three conditions: the flexible family clearly beats the benchmark, the non-additive share concentrates on 2–4 predictors, Riley's minimum met |
| The shrink view (§2.3) | shown, not an objective | a Focus on the penalty | a penalized family was fit |
| Named phenomena (§2.4) | the quiet name on an existing concern or label | "Known as …" on focus | the entry's detector fired and its guard allows the label |

"Prediction plus explainability begets further inference" is the leash table of §2.2, restated as
policy: under Predict the noticing labels and discloses, counts as an estimate shown (so a later
Estimate track locks as declared after estimates were seen), is tested once on the held-out rows at
the opening as a labeled secondary with a valid p-value, and never adds a candidate, prunes, or
re-ranks the shelf; under Estimate it is only an exploratory secondary "selected among N candidates
on these rows", with no p-value shown, never the locked primary. Both write one sentence for new
data with the same measurement protocol. The overclaiming guard is the claim-strength ladder on
wordings and the gate on "Write my own" (FOUNDATION §8); a wording stronger than its exhibit's
strength raises a noticing on the exhibit, and keeping it is recorded.

### 6.4 Noticings born after the fit

They label the exhibit they concern, add a disclosure, or add a sensitivity analysis (crosswalk,
"Noticings born after the fit"), and a new analysis they ask for is a secondary. Their materiality is
realized by construction (they read the fit), so they enter the ledger with `M_post` only and no
pre-lock row, which is correct and the calibration must not count them as predictions.

---

## 7 · Invariants and verification

### 7.1 The fuzzer

Two layers, so that thousands of journeys run in minutes and a few dozen run end to end:

- **Offline, over the fold.** Generate decision sequences from the registered kinds
  (`decisions.register_kind`'s registry and validators), apply them with `decisions.fold_onto`,
  route with `interview.route` on mocked stage artifacts (the `Facts` and `artifacts` that
  `quest.quest_log` already takes), and evaluate the surfacing functions on every intermediate
  state. Fixtures: the committed NHANES fixture and `metabolomics_untargeted.csv`, with their
  `Truth` (`tests/truths.py`) answering readings as the drives do. Target: 2,000 journeys under
  two minutes on the PR tier.
- **End to end, over the server.** `tests/acceptance/server_drive.py:Drive` on a sample of generated
  journeys, nightly, asserting the same properties on served responses (`PreviewResult`, the quest
  endpoint, the artifacts).

### 7.2 The properties

| # | Property | How it is checked |
|---|---|---|
| I1 | **No estimate before the lock.** While `consequences.estimates_unseen(state)`, no `ESTIMATE_STAGES` artifact is served, no `PreviewResult` holds an O3 or O4 view, no quest line's text holds a number from an estimate stage | `Stage.serves == ESTIMATE` derives the list; every view declares its class |
| I2 | **Nothing material is hidden.** Every item with `fires ∧ M > 0` (or `changes_question`, or a blocker) appears as a Decide line, a Confirm line, a row, a context line, a triage row, a stated phrase or an exhibit label; every fired item appears somewhere or as a checked-clean record | the coverage test (U13) over the fuzzer's states |
| I3 | **Caps hold** on every surface in every state (§4.1) | counts over `quest_log` and the planner |
| I4 | **Every Decide is answerable or says so.** A listed Decide has every `reads` answered and every `needs` fresh, or is in the one "N more wait" line | `quest.question_reads`, `Declaration.reads` |
| I5 | **Every disposition is recorded.** Before `lock_plan` (or `open_seal`), every noticing that fired has a `settled`, `dismissed`, `dispose` or `leave_open` record; the lock's digest covers them | `plan_lock.plan_of` includes the dispositions slot |
| I6 | **The display-order rule.** Nothing shown depends on an unanswered decision without saying so; every engine-filled answer is visible in Confirm or For the record; no view touches rows or the outcome before its gate | the three conditions of FOUNDATION §10 as assertions over lines and views |
| I7 | **Surface budgets.** A (thread, subject) appears at most once per surface unless "Changed since you looked" | the `seen` set |
| I8 | **Determinism and replay.** The same log yields the same quest log, the same ranks, the same materiality ledger, on every platform within tolerance | the replay harness (`export/replay.py`) extended to the ledger |
| I9 | **Stability.** A decision that writes no slot an item reads leaves that item's rank unchanged | `quest.written_slots` against `question_reads` |
| I10 | **Tier is mode-independent.** The Decide and Confirm sets are identical across expertise modes; only `level` differs | run the same state under each mode |
| I11 | **Progress is monotone except by reopen.** A stage's answered count never falls without a `Reopened` record naming the cause. Its required count may grow as answers make new lines apply within the stage being worked. A complete stage that gains a line has been reopened, and says why (a `Reopened` record in it) | `quest.Progress`, `quest.Reopened`; the path fuzzer's I11 and its hand-built journeys |
| I12 | **The triage recommendation is reproducible** from the ledger row alone, and never band 0 from an uncalibrated instrument | the calibration file |
| I13 | **Verification never silently edits.** When `M_post` leaves the predicted band, an exhibit label and a draft sentence exist; the primary is unchanged | the exhibit model (C7a) |
| I14 | **Quest log latency.** `quest.quest_log` under 200 ms on the NHANES journey's longest log | a timing test, like `test_previews_4_latency.py` |

### 7.3 Metrics that say the surfacing is good, not only safe

All computable from the record, the ledger and the drives, with no new instrumentation except a
`shown` sidecar (what was listed, pointed at, opened, changed; a derived record, not a decision).

| Metric | Definition | Source | Target to start |
|---|---|---|---|
| Time to a defensible result | wall clock from upload to a bundle that passes the manuscript gate (X5) | drives | the capture took 17 s of compute on dietary inference; the human time is the number to learn |
| Decisions per journey | fired Decide lines plus triage rows answered | `quest_log`, the log | within the catalogs' targets (`UNDERSTANDING_LAYER.md` §2.8) |
| False-alarm rate | share of band ≥ 1 predictions whose `M_post < τ₀` | the ledger | ≤ 0.3 after calibration |
| False-reassurance rate | share of band 0 predictions whose `M_post ≥ τ₀` | the ledger | ≤ 0.05 |
| Decisions changed by a noticing | share of Decide answers that differ from the default on a card carrying a context line or row, against cards without | the log and `shown` | > 0 is the proof that Own happens; the number itself is the finding |
| Limitation sentences per paper | drafted and kept | the manuscript model | ≤ *k* |
| Reopen count | `Reopened` records per journey | `quest_log` | falls as `fires` improves |
| Reverts after a preview | `revert` records within one card of a preview | the log | a proxy for confusion |
| "Why does this matter?" opens per Decide | from `shown` | | high early, falling on repeat journeys |
| Five-second test | a first-time user says what to click within five seconds | drives (DRIVE_RUBRIC) | 18/18 |

### 7.4 With real users

The DRIVE_RUBRIC and the R3 drives are the instrument. Three additions cost little:

- **Think-aloud on the Models stage of the dietary journey first**, drawn as one static screen
  (audit recommendation 8), before P0.7, with Nolan and one colleague; record where they hesitate
  and what they read aloud, against the rank key's "Why is this here?" lines.
- **A counterbalanced order test with the methodologists (R4).** Two orderings of the same Models
  stage, the policy's and the crosswalk's hand order, on the same journey; ask which decisions they
  would challenge as reviewers and whether anything felt hidden. This is the only way to learn
  whether `reach` should outrank `band` or the reverse.
- **The ledger as a review packet section.** Each lens's packet already shows what the journey
  decided; add the predicted-versus-realized table so the expert can say where the proxies are
  wrong for their field.

---

## 8 · Pressure-testing the current design

### 8.1 Where the declarative approach breaks

1. **The declarations are prose.** All 708 `fires_when` strings are free text; 380 of 933
   `depends_on` tokens are prose rather than item ids; 149 `reopens_when` and 63 `display_gate`
   fields are paragraphs. Only `quest.DECLARATIONS` (15 kinds) and the Router (`interview.
   QUESTION_KEYS`, `NEEDS`, `sequence._answers_in_order`) are executable. A fuzzer can only fuzz
   code. The fix is §1.6: a `rule` per item, landed as items are wired, with the registry test
   counting prose-only items as unwired.
2. **"Order by consequence" has no consequence to read.** The crosswalk's `order` is hand-assigned;
   the planner's footprint exists per option on hover and is discarded after sorting views; the
   catalogs' `rank_signal` is prose; P0.5's would-change-a-number test does not exist. Until §2's
   `M` exists, the stage order is the Router's order plus the catalog's guess, which is what the
   crosswalk already admits (disagreement 18: "the quest log adopts the Router's order").
3. **Three rank keys compete.** The brief's `(tier, reach, share, excess, column order)` for First
   look, the crosswalk's `order` for lines, `diff_views`' score for views. A user will see the same
   noticing ranked differently on two screens. One key, three projections (§1.4).
4. **The triage sweep has no engine.** Nolan's 2026-10-09 ruling names three dispositions; nothing
   in the repo computes them, and under Estimate the honest computation is illegal before the lock.
   §2's two-regime design is the only way to keep the ruling and the leash both.
5. **"Blockers first" collides with the dependency order** unless stated as "blockers first within
   the answerable frontier". A T1 noticing that reads an unanswered question cannot be first; it is
   in the "N more wait" line until it can be.
6. **The caps are on the wrong count.** `UNDERSTANDING_LAYER.md` §2.8 caps surfaced noticings; the
   load in Models is Decide lines (71 candidates for dietary inference before `fires`). §4.2.
7. **The thread registry's tier is set per purpose by the pack** (`Meaning.tier`), checked against
   feeds (R4). It should be computed from `M` and the feeds, with a pack allowed to tighten only
   (R4 already says so). Keep the rule, drop the hand field.

### 8.2 The if-this-then-that that remains

It did not disappear; it moved. Each is a place where a predicate must replace prose:

- `reopens_when` on 149 items, where `quest._reopened_by` already does the honest thing by replaying
  the log and asking `holds`; the prose should become `holds` functions.
- The 63 `display_gate` paragraphs, which are the O-class gate per item, and should be one `view`
  field plus the legality matrix.
- The several-goals rule ("every shared view follows the strictest gate among the tracks") and the
  Describe-without-outcome path, which are two more purposes in every purpose switch; D2's track id
  must make the gate a function of the track set, not a branch per combination.
- First look's "Decide now" exception and Predict's two sweep departures (FOUNDATION §3), each a
  special case the fuzzer must cover explicitly.
- The exhibit placement table, 50 rows of default and allowed placements, which should be one rule
  (floor, pre-included, claim strength) with exceptions recorded.

### 8.3 Over-built

- **Ten new view kinds designed before one quest-log screen exists** (P0.3b). Design the table,
  forest and curve the first slice needs; the rest when an exhibit draws them.
- **Seventeen sentinel families each placed and sized** before any sentinel has caught a planted
  leak on a fixture. Build U10's sentinel proof for S1–S3 first; let the rest follow the journeys.
- **The phenomena registry's teaching-only entries** (double descent, spectral bias) in 2.0. Keep
  the registry; land the entries with detectors.
- **The hypothesis noticing (MC-9) before the first slice.** Its conditions are sound; its place is
  after C7d, as the contract already says.
- **Per-item "Waiting for" lines, "May change as you answer" lists for unreached stages, the
  manuscript rail's count badge.** Each earns its place alone; together they are the text noise the
  audit named. §1.5 and §4.3 collapse them.

### 8.4 Under-built

- **The materiality function and ledger** (§2). Nothing in the repo computes a disposition.
- **Executable `fires`, `reads`, `holds` for noticings and defaults** (§1.6).
- **P0.5's would-change-a-number test**, which is `M_alt ≥ τ_confirm` on the planner's footprint
  and is implementable today for every kind with a `register_transform`.
- **U5's asked-rows gate** and a Decide-count gate per stage (§4.2).
- **A `shown` record** for the metrics (§7.3). Without it, "decisions changed by a surfaced item"
  cannot be measured.
- **The pre-lock verification hook into the sensitivity, calibration and secondary stages**, which
  already compute the right numbers and only need to write ledger rows.

### 8.5 Concrete changes to the three documents

**FOUNDATION.md**

- §3, "Decide · Confirm · For the record": state the tier rule of §1.3 and that Confirm's "another
  choice would change a number" means `M_alt ≥ τ_confirm` on the planner's footprint.
- §3, the Confirm sweep: cap the shown lines at 8, the rest collapsed "and N more, each read from
  your data".
- §3, "Waiting for": one count line per stage, the list behind it; drop "May change as you answer".
- §7, the triage sweep: add the two phases, the verification after the fit, the "not measurable
  here" reason, the cap of 6 rows grouped by family, and that it is the Confirm sweep of noticings.
- §5: name the footprint as the materiality vector; captions carry the movement in design units
  before the lock and in *Q*-units after; "nothing changes" is `M < τ₀` in every component.
- §8: the verification rows are exhibits; a band-1 or band-2 realization is promoted to a Results
  sentence by default.
- §10: add I1–I14 as the properties the fuzzer holds the design to.

**CROSSWALK.md and crosswalk.json**

- Add `rule`, `view`, `materiality` and `changes_question` fields to the item schema; keep the
  prose fields as documentation; the registry test counts items without `rule` as unwired.
- Replace the hand `order` by the computed key, keeping `order` as the tiebreak; regenerate the
  per-stage "Decide, in order" lists from the engine and diff them against the hand lists once.
- "The reference journeys against the load caps": add the fired-Decide count per stage and the
  limitation-sentence count per journey as the two numbers the U5 gate reads.
- Question 4's re-ruling makes T2s the plan; add the materiality calibration to T1's definition of
  done ("each proof lands with its ledger row").

**UNDERSTANDING_LAYER.md**

- §1.2: `Notice` gains `materiality: str` (the instrument, `module:function`) and `power: str`;
  `Feed` gains `changes_question: bool`; `Meaning.tier` becomes derived, with `tighten` the only
  hand field.
- §1.3 R4: the tier follows the feeds *and* `M`; add R11, "every number-changing feed names a
  materiality instrument or declares `not_measurable`", and R12, "a band-0 recommendation requires
  a calibrated instrument".
- §2.2: the rank key is §1.4's key; `(share, excess)` fold into `M`.
- §2.6: rewrite for the triage sweep (the 2026-10-09 ruling), with `dispose` and `verify`.
- §2.8: add the Decide-count cap per stage and the limitation budget per journey.
- §5: U4 gains the pre-lock instruments; U8 becomes the triage engine; add U15, the materiality
  ledger and calibration, and U16, the path fuzzer.

---

## 9 · Implementation plan

Sizes on SIZING's scale (S = 1, S–M = 2, M = 3, M–L = 5.5, L = 8, XL = 20). Everything is engine
work on top of what exists; the shell (P0.7) reads the result through `GET /quest`, extended.

| # | Work | Builds on | Package | Size |
|---|---|---|---|---|
| 1 | `turbotab/core/surfacing.py`: the `Surfaceable` protocol, `tier`, `rank`, `band`, `reach` as pure functions over `(state, facts, measures)`; `quest.DECLARATIONS` adapted to it; `STATED` and `TIER_RULINGS` re-expressed as overrides with a test each | `quest.py` | P0.4 (extension) | M |
| 2 | The Confirm test: `M_alt` from the planner's footprint on the sample for every kind with a `register_transform`; `Footprint` kept on `PreviewResult`; `τ_confirm` recorded; kinds in `UNPREVIEWED` are For the record by construction | `consequences.diff_views`, `_shifts`, `_relationship_change` | P0.5 | M |
| 3 | The `notices` stage: measure, reference, reach, O-class, cost; the pre-lock instruments of §2.3 (rows × SMD, W1/sd, |Δr|, λ, O2 counts, excess over chance); cached per data state; heavy passes scheduled | `stages/findings.py`, `detectors/assay.py`, `stages/explore.py`'s outcome-free findings, `usual_intake` | U4 | M–L |
| 4 | The thread registry, minimal: `Notice`, `Meaning`, `Feed`, `Thread` with `materiality` and `changes_question`; R1–R12; two threads (`diet-energy-carries-the-nutrient`, `diet-implausible-reporters`) and one metabolomics thread (`metab-qc-drift`) | `contracts.register_contract`, `readings.KIND_RULES` | U1, U2 | M |
| 5 | The triage engine: `dispose` and `verify` kinds; the recommendation from the ledger row; the dispositions as a slot under `plan_lock.plan_of`; sentences for the three dispositions and "not measurable here" | `plan_lock`, `decisions.register_kind` | U8 (widened) | M |
| 6 | Post-lock verification: ledger rows written by `stages/sensitivity.py:changes_for`, `stages/calibration.py`, `stages/secondary.py`, and under Predict by the paired fold differences in `stages/evaluation.py`; the exhibit label on a band change | the stages named | C7a (a part), U9 | M |
| 7 | The ledger and calibration: `materiality_calibration.json` built from the twelve captures; the confusion matrix per instrument; the registry test that caps uncalibrated instruments at band 1 | the captures, `server_drive.Drive` | new U15 | S–M |
| 8 | The U5 gate: fired Decide count per stage and asked rows per journey against the targets; limitation sentences per journey against the budget | `quest_log`, the captures | U5 | S–M |
| 9 | The path fuzzer: offline generator over `decisions.fold_onto` and `interview.route` with mocked artifacts; I1–I14; seeds; 2,000 journeys under two minutes; a nightly end-to-end sample over `server_drive` | `tests/acceptance/test_previews_leash*.py` as the pattern | new U16 | M |
| 10 | The `shown` sidecar and the metrics script (§7.3) | the log | new | S–M |
| 11 | `GET /quest` extended with `rank`, `tier`, `band`, `why_here`, `footprint` per line; the shell reads them | `quest.QuestLine` | P0.7 (a part) | S–M |
| 12 | The coverage test over the fuzzer's states (I2) | U13 | U13 | S |

Order: 1, 2 and 3 can run in parallel; 4 after 1; 5 after 3 and 4; 6 after 5; 7 and 8 after 6; 9
after 1 (growing as 2–6 land); 10 and 11 any time after 1; 12 after 4. Total about 30 units, of
which about 14 are P0.5, P0.9 and U-packages already on the road; the new units are the ledger, the
calibration, the fuzzer and the sidecar, about 9.

### The smallest end-to-end proof

One journey, one stage, three noticings, the whole loop, before any catalog rollout:

1. Load the committed NHANES fixture, answer the `dietary-inference` capture up to Models
   (`server_drive.Drive`, `Truth`).
2. `notices` fires `diet-energy-carries-the-nutrient` (r(sugar, kcal); `changes_question`),
   `diet-implausible-reporters` (5,212 rows; leavers' SMD on age, gender, BMI), and
   `diet-day-to-day-variance` as `not_measurable` (one recall day).
3. The quest log lists Models with the key of §1.4: the energy question first among the noticing-
   shaped Decides, the exclusions sensitivity as a Confirm in Models, and the triage sweep with
   three rows: "act on it" (energy, already answered: it drops out), "doesn't change your numbers
   here" or "could bias" for the screen by the rows proxy, "could bias (not measurable here)" for
   the measurement error, pre-filled.
4. Confirm all; press Fit; the lock's digest covers the dispositions.
5. The sensitivity stage writes the realized row for the screen; the ledger compares bands; if the
   realized band is higher, Table 2's sensitivity exhibit carries the label and the Discussion
   draft gains the sentence.
6. The fuzzer runs 2,000 offline journeys on the same fixture and holds I1–I14.
7. The calibration file has its first three rows, and the review packet for dietary prints them.

Everything in this proof uses functions that exist today except `surfacing.py`, the `notices`
stage's instruments, the `dispose` kind and the ledger. It is the dietary slice the audit asked for,
with the intelligence inside it rather than after it.

---

## Appendix · Sources in the repo this policy rests on

- `turbotab/core/consequences.py`: `diff_views`, `_shifts`, `_relationship_change`, `RELATIONSHIP_DELTA`, `MAX_VIEWS`, `MAX_COACH`, `PreviewContext.sample_size`, `estimates_unseen`, `register_transform`, `register_consequence`, `UNPREVIEWED`, `plan`.
- `turbotab/core/quest.py`: `Declaration`, `DECLARATIONS`, `QUESTIONS`, `OTHER_KINDS`, `STATED`, `TIER_RULINGS`, `READ_BY_GATE`, `question_reads`, `stage_reads`, `written_slots`, `_reopened_by`, `_hold_the_families`, `_frontier`, `quest_log`, `QuestLine`, `Progress`, `Reopened`, `noticing_place`.
- `turbotab/core/readings.py`: `Reading.confidence`, `KIND_RULES`, `by_values`, `Consumer`, `CONSUMERS`, `Unsettled`, `settled_columns`, `PREDICTOR_ROLES`.
- `turbotab/core/ask.py`: `CONSUMERS`, the one card where the first consumer needs it.
- `turbotab/core/interview.py`: `QUESTION_KEYS`, `NEEDS`, `SLOT_OF`, `route`; `turbotab/core/sequence.py`: `_answers_in_order`, `ANSWERED_ANY_TIME`.
- `turbotab/core/estimand.py`: `HOLDS`, `served_gate`, `ESTIMATE_STAGES` derived from `Stage.serves == ESTIMATE` (`turbotab/core/graph.py:Stage`, `turbotab/core/stages/__init__.py`).
- `turbotab/core/plan_lock.py`: `plan_slots`, `plan_of`, `digest`.
- `turbotab/core/stages/sensitivity.py`: `changes_for`, `SensitivityChange`, Banna 2017; `turbotab/core/stages/effects.py`: `EValueResult`, `RobustnessResult`, `BenchmarkResult`, `sensitivity_reading`; `turbotab/core/models/effects.py`: `unmeasured_confounding`, `robustness_value`, `e_values`.
- `turbotab/core/stages/evaluation.py`: the spline benchmark paired by fold; `turbotab/core/models/selection.py`: `interpretable_cost`, `shelf_order` (Riley), `declared_result`, `note_seen`.
- `turbotab/core/stages/explore.py`: `hand_levers`, `COLLINEAR`, `RARE_CLASS`, `GROUP_GAP`; `turbotab/core/stages/findings.py`: the severity sort to replace (`SEVERITY_RANK`, lines 225–232).
- `turbotab/core/decisions.py`: `fold`, `fold_onto`, `state_after`, `register_kind`, `register_validator`, `register_completion`, `disclose`.
- `turbotab/core/tests/acceptance/server_drive.py`: `Drive`, `local_server`, `settle_post`; `turbotab/core/tests/truths.py:Truth`.
- `docs/turbotab-next/crosswalk/crosswalk.json`: 708 items; `fires_when` prose on all; `depends_on` 553 ids and 380 prose tokens; `reopens_when` on 149; `display_gate` on 63; `returns_at` on 322; `order_audit.gates`.
- `docs/turbotab-next/understanding/catalogs/dietary.json` and `metabolomics.json`: the threads and numbers cited; `review-packets/captures/dietary-inference.json`: the 28 answers cited.
