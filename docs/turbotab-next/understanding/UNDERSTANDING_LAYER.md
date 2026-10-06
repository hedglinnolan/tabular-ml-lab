# The understanding layer: threads, from a noticing to the paper

*Design · TurboTab Next · 2026-10-05. The repo (turbotab-next at caa9563) was only read. I ran no computation on any fixture: every number on a user's data below is quoted from the six final catalogs. The one script I ran sorted the 333 catalog ids into families and counted them.*

**Inputs.**
- **Both research runs had finished, so no polling was needed.**
  - The exploration design brief (wf_1b3d0fc3, "First look"): its Look contract becomes the noticing-and-layout part of a thread.
  - The verified rule table (wf_e039e6d2): it becomes the legality matrix the thread registry enforces.
- **BLUEPRINT** §11–§14 and **FOUNDATION** §3–§5 (calm worktree).
- **MODELING_SEQUENCE** §1.
- **Engine modules:** `readings.py`, `contracts.py`, `stages/findings.py`, `stages/explore.py`, `plan_lock.py`, `interview.py`, `ask.py`, `export/checklists.py` and `packs.py`.

---

## 0 · The answer on one page

1. **A thread is declared once and lives through the whole analysis, the way a method contract does (§13).** It has seven parts:
   - how it is noticed: a measured detector, with a reference;
   - what the researcher confirms: a reading in the ledger;
   - every later decision it feeds, by stage and by purpose;
   - the failure a score-driven pipeline would commit without it, and why its score cannot see that failure;
   - the moment: two numbers on the user's own data;
   - the canvas layout that shows it;
   - its sentences: methods, limitation, dismissal, clean check, and checklist anchors.

2. **The magic has three beats, and AutoML has only the first.**
   - **Notice:** a measure beside its reference, before any model runs.
   - **Own:** the researcher answers in their own terms, with each answer's consequence previewed beside it.
   - **Return:** later cards and the paper visibly descend from that answer: "Because you said alcohol's zeros are days, the two-part model is first."

   A pipeline that never asks has nothing to return to. That gap is the difference between TurboTab and "auto-ml but with more steps".

3. **The readings ledger is the backbone.** §14.1 already guarantees that a number-changing consumer reads only settled readings. Threads extend the ledger in four ways:
   - noticings become computed facts that sit beside readings;
   - a subject can be wider than one column;
   - meaning kinds register through `KIND_RULES` with their alternatives;
   - a consumed-by log lets later cards, the manuscript and the supplement say what each answer changed.

4. **The answer to "what if there are more":** the 333 catalog threads are instances of **17 families**.
   - **Where the families come from.** They come from where error enters an analysis. Three things a score cannot do (see what is wrong with its data, know what the data mean, explain why) split into 7 + 7 + 3 families.
   - **Sentinels.** Every family gets a lens-agnostic detector. A situation no catalog names still raises a generic thread of its family.
   - **Coverage as data.** Coverage is held in a registry, and a test fails on any unmapped item. The registry holds:
     - PROBAST's 20 signalling questions;
     - ROBINS-E's and ROBINS-I V2's seven domains each;
     - the eight leakage types of Kapoor and Narayanan;
     - TRIPOD+AI's 52 items;
     - STROBE-nut's 58 rows;
     - the 14 stage × purpose cells.
   - **Gaps.** Mapping them here found 7 gaps (§4.8), and each one becomes a thread.

5. **The purpose rules are enforced when a thread registers, not in review.** A thread whose detector looks at the outcome cannot declare a feed that shapes a predictor's representation or prunes a candidate. A feed that changes a number is decided in one of three ways, never by eye:
   - by meaning: a reading the researcher confirms;
   - by a fixed rule that the replay re-runs, refit in-fold where it learns;
   - by a candidate that the resampling tests.

6. **New threads enter as declarations in a lens pack, and no screen is redesigned.** These render any thread:
   - the First look guide;
   - the rows of the card families;
   - the context line on later cards;
   - the manuscript mark;
   - the list of open threads before the lock.

   Only a new card family or a new view kind needs a design decision.

7. **Build order.**
   1. The contract, the ledger extension and the census, built with two threads (one per purpose).
   2. One thread per remaining lens, plus a sentinel proof: set the lens to "Something else", and the family sentinels must still catch the leak.
   3. A second thread per lens, each from a different family.

---

## 1 · The thread contract

### 1.1 The seven parts, beside the method contract

| Part | Thread field | Parallel in the method contract (§13) | Answers |
|---|---|---|---|
| How it is noticed | `notice`: the measure, its reference, the firing rule, the view class, which rows it reads, the settled readings it needs, its lens, its cost, two fixtures | `scope` (what it may learn from), `needs` | Is something here, compared with what? |
| What the researcher confirms | `meaning`: the reading kind and its subject, the question (≤ 14 words), the alternatives including "does not apply", `settled_by`, the card family, a tier per purpose, the conservative path while unsettled | `routing` (question, options with both labels, rungs) | What does it mean? |
| What it feeds | `feeds`: for each, the stage, purposes, consumer and effect; what changes for each reading value (`when`); the context line (`says`); and `decides_by` | `slot`, `relations` (implies, enables, disables, invalidates, conflicts) | Which later decisions change, and how? |
| The failure | `failure`: what a score-driven pipeline does; why the score is blind to it (the family's reason by default); the counterfactual measure; the fixtures where it is measured | `options[*].sound` | What goes wrong without it? |
| The moment | `moment`: a template, the two contrasting numbers, and the stage where it first appears | `storyboard` | When does the user get it? |
| The canvas | `layout`: the footprint (Focus, Strip, Flow, Routing or Angles), views from the closed vocabulary, and context variables (never the outcome) | `storyboard` steps | Which picture shows it? |
| The sentences | `sentences`: one for each reading value, plus open at the lock, dismissed, and checked clean; the anchors (TRIPOD+AI, STROBE-nut, PROBAST, ROBINS, leakage) | `sentence` | What does the paper say? |
| Family and links | `family`, `specializes`, `relations` (merges_with, implies, invalidates, conflicts) | `relations`, `package` | What is it an instance of? |

The brief's Look contract (§2.3) maps onto these parts:
- **Look = `notice` + `layout` + `claim`.** Its `thread` field becomes the first feed's consumer.
- **The Look's `tier`** is derived from the feeds (§2.2).

### 1.2 The declaration

```python
ViewClass = Literal["O0", "O1", "O2", "O3", "O4"]   # what the detector looks at (§1.3)
Tier      = Literal["asked", "surfaced", "stated", "silent"]
Stage     = Literal["opening", "representation", "selection", "model",
                    "evaluation", "explanation", "reporting"]
Effect    = Literal["asks", "sets_default", "ranks_options", "refuses_option",
                    "changes_rows", "changes_representation", "changes_split",
                    "adds_candidate", "adds_baseline", "adds_sensitivity",
                    "labels", "discloses"]
DecidesBy = Literal["meaning", "fixed_rule", "resampled_candidate", "disclosure"]

@dataclass(frozen=True)
class Notice:
    measure: str            # "module:function" → Measure(value, numbers, subject, reach, share)
    reference: Reference    # chance share | named convention (+source) | field range | codebook
    fires: str              # the declared firing rule, in words and as code
    view: ViewClass
    rows: Literal["all", "reference", "training", "analysis"]
    needs: tuple[str, ...]  # settled readings and roles it requires
    lens: tuple[str, ...]   # ("*",) for a family sentinel
    cost: Literal["instant", "pass", "fit", "heavy"]   # heavy → scheduled, never on hover
    fixtures: tuple[str, str]                          # fires on · stays silent on

@dataclass(frozen=True)
class Meaning:
    kind: str               # a readings.KIND_RULES kind (new or existing)
    subject: Literal["column", "columns", "rows", "design", "table"]
    question: str           # ≤ 14 words
    alternatives: tuple[Alternative, ...]   # key, label, consequence ≤ 16 words; includes not_applicable
    settled_by: Literal["values", "user", "codebook"]   # values only with a discriminating test
    card: str               # card family key
    tier: Mapping[str, Tier]                 # per purpose; checked against the feeds
    conservative: Mapping[str, str]          # consumer → what it does while unsettled

@dataclass(frozen=True)
class Feed:
    stage: Stage
    purposes: tuple[str, ...]
    consumer: str           # a method-contract key, a QUESTION_KEY, or a readings.CONSUMERS name
    effect: Effect
    when: Mapping[str, str] # reading value → what changes ("episodic" → "two-part ranks first")
    says: str               # the context line template, ≤ 22 words
    decides_by: DecidesBy

@dataclass(frozen=True)
class Thread:
    key: str; family: str; specializes: str | None; label: str
    claim: str              # ≤ 20 words, filled from the measure's numbers
    notice: Notice
    meaning: Meaning | None # None: a silent guard, or a disclosure-only thread
    feeds: tuple[Feed, ...]
    failure: Failure        # pipeline_does, score_blind_because, counterfactual, measurable_on
    moment: Moment          # template, contrast (two numbers), first_at
    layout: Layout          # footprint, views, context
    sentences: Sentences    # resolved{value}, open_at_lock, dismissed, checked_clean, answers
    relations: tuple[ThreadRelation, ...] = ()
    sources: tuple[str, ...] = ()
    package: str = ""

def register_thread(t: Thread) -> Thread: ...   # rules R1–R10 below; mirrors register_contract
```

`Pack` gains `threads: tuple[Thread, ...]` beside `looks_for`, `detectors`, `priors`, `reframings`, `hedges` and `recipes`. The existing two-way test `test_a_pack_names_what_it_will_look_for` extends so that every thread has a `looks_for` phrase.

### 1.3 What the registry enforces

The registry checks the following at import time and raises on any violation, as `register_contract` refuses a training-fold method before the seal.

- **R1 · Family.**
  - Every thread names one of the 17 families (§3.1).
  - A blank `failure.score_blind_because` inherits the family's reason.
- **R2 · Purpose legality.** Every feed passes the legality matrix below for its view class, purpose, stage and effect.
- **R3 · Decided by a rule, never by eye.** A feed that changes a number declares how it is decided:
  - `meaning`: the researcher's knowledge, through a reading;
  - `fixed_rule`: a declared function the replay re-runs, with a training-fold scope under prediction if it reads other rows;
  - `resampled_candidate`: prediction only;
  - `disclosure`.
- **R4 · The tier follows the feeds.**
  - `asked`: only for a purpose where some feed changes a number (§14.2 rule 1: ask only where it matters).
  - `silent`: only for a guard whose behavior is fixed and offers no alternatives (for example, "median fill is never used on an energy-residualized nutrient").
  - `surfaced`: when the feeds are disclosure, sensitivity or a baseline.
  - `stated`: a number-changing default with a changeable phrase.
  - A pack may tighten a tier, with a reason. It may never loosen one.
- **R5 · Settled only (§14.1 restated).** Each feed's consumer is registered in `readings.CONSUMERS` with this thread's reading kind and reads it through the ledger. The structural census checks both directions.
- **R6 · Corroboration discriminates (§14.3).**
  - A meaning kind settles by its values only when it declares a value test that rejects every alternative, each with a fixture.
  - Most meaning kinds settle only by the user or the codebook. Values can make "episodic zeros" likely; they cannot exclude "mixed".
- **R7 · Card family.**
  - An asked thread names an existing card family, with one row per (family, kind, subject).
  - A new card family is a design decision, like a new view kind in `consequences.py`.
- **R8 · Two fixtures.**
  - Each thread has a committed fixture on which it fires and one on which it stays silent or stated.
  - The moment's two numbers are computed on the positive fixture. Otherwise the moment says it is illustrative: never assert falsely.
- **R9 · Every outcome has a sentence.**
  - One per reading value, plus open-at-the-lock, dismissed and checked-clean.
  - At least one anchor to a checklist item or a bias-tool domain, or a stated reason why there is none.
- **R10 · Word budgets** (§11 rule 4):

  | Element | Budget |
  |---|---|
  | claim | ≤ 20 words |
  | question | ≤ 14 words |
  | `says` (the context line) | ≤ 22 words |
  | option consequence | ≤ 16 words |

**The legality matrix (R2).** These are the purpose rules, written as the effects each view class may have. Within a cell, effects are separated by semicolons.

| The detector looks at | Inference, before the lock | Inference, after the lock | Prediction (training rows, holdout sealed) |
|---|---|---|---|
| **O0** · predictors, design, structure or reference rows (outcome-blind) | asks; sets a default; changes rows; changes the representation (declared, frozen at the lock, learned on the full analysis sample); adds a sensitivity analysis | discloses; adds a sensitivity analysis (recorded "after the estimates were seen") | asks; sets a default; changes the representation as an in-fold rule (learned parts have training-fold scope); changes the split; adds a candidate |
| **O1** · the outcome alone | asks about Y's data quality and raises the estimand question its shape suggests (mean or geometric mean, hurdle or not), answered as an estimand, never chosen by the histogram; sets the df budget | diagnostics conditional on X (zero inflation given the fitted mean); discloses; prespecified sensitivity | adds candidate families and losses; stratifies folds; sets the df budget |
| **O2** · outcome × design or process, as counts or a process-only baseline | asks; blocks (T1, perfect confounding); changes the validation design; adds a baseline; adds a sensitivity analysis. Never selects predictors or sets forms | discloses | the same as before the lock. A count of cases per batch may read every row, because it is a design fact. Anything fitted, such as a process-only AUC, is computed in-fold on training rows, like any score |
| **O3** · predictor × outcome | nothing: the view sits behind the recorded door | diagnostics; labels; discloses; exploratory sensitivity, labeled as such | adds a candidate and never prunes; labels; discloses; every view recorded |
| **O4** · an outcome-model estimate | nothing | labels; discloses; sensitivity | labels; discloses; baseline comparison |

**Two cells need the separate purpose-rule verification to rule on them.** Each is narrow and stated here so it can be checked.

1. **O1 informing family or link.** The task's rule says the outcome alone "may inform its family/link". The verified table says that under inference it "does not inform the loss or the model family by its shape".
   - The contract reconciles them by letting the O1 view raise the question. The answer is an estimand under inference, or a resampled candidate under prediction.
2. **The audit sentinel under prediction** (families S2 and S3).
   - **What it does.** A single column's in-fold association with the outcome (O3, recorded) may only order the rows of a meaning question that applies to every candidate predictor: "When was this recorded relative to the moment of use?"
   - **What decides.** The answer is a reading, and the drop is applied by an outcome-blind rule ("columns recorded after the moment of use leave"). The association never prunes anything itself.
   - **Under inference** the sentinel is not needed: the existing `after_exposure` card asks about timing for every covariate.
   - **Basis in the verified table.** Library size against outcome group appears in its §4 as "a design-balance check that decides the normalization, never the features". That is the precedent for class O2.

### 1.4 How it extends the readings ledger (§14.1)

1. **A noticing is a fact; a reading is an interpretation.**
   - A new stage, `notices`, returns one record per fired thread and subject: the measure, reference, reach, rank key and the claim's numbers.
     - Before the seal it computes the O0 measures and the O2 design counts.
     - Under prediction it also computes the O1 and O3 measures on training rows, after the split.
   - It is deterministic, replayable and cached per data state.
   - The noticing is the evidence that the reading's guess leads with (`Reading.evidence`). The card shows the user's own numbers, not a name match.
2. **Subjects widen.** `Reading.subject` is a tuple of columns today (`readings.py:103`). Threads add four subject shapes:
   - **a column set:** `fat_total` with energy;
   - **a row set:** named by its rule plus a digest of its row keys. A confirmation then settles exactly the rows it lists, which is §14.2's block rule applied to rows ("2 persons, each with the odd visit pre-selected").
   - **a design element:** batch × outcome;
   - **the table:** the instrument kind.
3. **Meaning kinds enter `KIND_RULES`.** Each new kind declares its alternatives, `settled_by` and any value test. Then `by_values`, the ask card, "read from your data" and the codebook path work unchanged. The catalogs need these kinds:
   - `zero_meaning`, `timing_vs_index`, `causal_place`, `instrument_kind`, `process_role`;
   - `reference_group`, `outcome_target` (disease or diagnosis), `outcome_course` (event, state or recurrent);
   - `sample_timing`, `deployment_measure`, `metabolome_role`, `scale_key`.
4. **"Does not apply" is a value, not an absence.**
   - Dismissing a thread confirms its `not_applicable` alternative, which has its own sentence.
   - Leaving a thread open at the lock is a separate decision, `leave_open`, and it writes a limitation (§2.6).
5. **A consumed-by log.**
   - Each consumer that reads a thread writes `reads_threads: [{thread, subject, value, effect}]` into its stage artifact.
   - That log, not the UI, is the only source of the context line, the manuscript mark and the supplement's "what it changed" column.
   - This is event-sourced as in BLUEPRINT §3: answers are decisions, notices and reads are stage artifacts, and the thread's record is derived, never a new mutable store.
6. **Staleness.** A decision that changes rows or values (an exclusion, a repair) reruns `notices`.
   - If a settled thread's numbers move past the tolerance its contract declares, or it stops firing, its `invalidates` relation re-asks it. It is never silently kept (§13).
   - If a thread starts firing after a decision, it returns as "Changed since you looked" (brief §4.5).
7. **Conservative paths.** While a thread is unsettled, each consumer takes the path the thread declares for it. For example:
   - no two-part default;
   - QC-RLSC marked "Not available yet";
   - the fit waits on a flagged column whose timing is unknown under prediction, a T1 row.

   This is §14.1's "asks, or takes the conservative path", declared per thread instead of per reader.

### 1.5 How consumers read it

```python
from turbotab.core import threads

v = threads.settled(state, "dietary.zero_is_a_day", ("alcohol_pct_kcal",))  # value or None
v = threads.require(state, key, subject, consumer="form")    # raises readings.Unsettled, exits = the card row
threads.record_read(ctx, key, subject, v, effect="ranks_options")          # the consumed-by log
threads.context(state, "form")       # ≤ 1 context line for the card, the rest for "Why does this matter?"
threads.open(state, before="lock")   # or before="seal": the open threads (§2.6)
threads.sentences(state)             # methods clauses, limitations, supplement rows
```

`Consumer` in `readings.CONSUMERS` gains `threads=(...)`. Two tests check the wiring:

- **The census, in both directions.** Every feed names a consumer that calls `threads.require` or `threads.settled` for that key. Every consumer whose behavior branches on a thread is declared by a feed.
- **The honored-confirmation test (§14.3).** For each alternative, confirming it on the positive fixture must produce that alternative's declared `when` behavior at every feed. The drivers answer from the fixture's truth.

Method contracts can also depend on threads:
- A contract's `needs` may name a thread reading. For example, NCI's one-part or two-part choice needs `zero_meaning`.
- An option's rung may depend on one. For example, regression calibration is refused when `instrument_kind` says "two FFQs a year apart". That is a reproducibility pair, not repeat recalls (diet-instrument-kind).

### 1.6 Two worked declarations

**`dietary.zero_is_a_day`** (family K2; specializes `shared.zero_mass_predictor`). The numbers are the dietary catalog's, on `dietary_recalls.csv`.

```python
register_thread(Thread(
  key="dietary.zero_is_a_day", family="K2", specializes="shared.zero_mass_predictor",
  claim="{col} is zero on {day_share:.0%} of recall days; {both:.1%} of people are zero on both.",
  notice=Notice(
    measure="turbotab.core.detectors.dietary:zero_days_vs_people",
    reference=Reference.chance("people zero on both days if zeros were independent days"),
    fires="zero-day share ≥ 5% (NCHS Series 2 No. 178) and both-days share ≤ 1.5 × chance",
    view="O0", rows="all", needs=("grain:recall_day", "unit:{col}"),
    lens=("dietary",), cost="pass",
    fixtures=("dietary_recalls.csv", "nhanes_dietary.csv")),   # fires · dormant (one day only)
  meaning=Meaning(
    kind="zero_meaning", subject="column", question="What does a zero in {col} mean?",
    alternatives=(
      Alt("episodic", "A day without it", "Two parts: any on a day, then how much"),
      Alt("never", "A person who never has it", "Non-consumers become their own group"),
      Alt("mixed", "Both kinds of zero", "Never and former split, if a lifetime column exists"),
      Alt("not_applicable", "Zeros are ordinary amounts here", "One line through zero")),
    settled_by="user", card="what_a_zero_means",
    tier={"inference": "asked", "prediction": "asked"},
    conservative={"form": "no two-part default; the card says the zeros are unconfirmed"}),
  feeds=(
    Feed("representation", ("inference", "prediction"), "usual_intake", "ranks_options",
         {"episodic": "two-part NCI first"}, "You said {col}'s zeros are days, not people.", "meaning"),
    Feed("model", ("inference",), "form", "ranks_options",
         {"episodic": "two-part form first", "never": "non-consumers as their own category first"},
         "From First look: {col}'s zeros are days ({day_share:.0%} of days, {both:.1%} of people).",
         "meaning"),
    Feed("model", ("prediction",), "levers", "adds_candidate",
         {"episodic": "any-day indicator and amount enter as candidates"},
         "Zeros are days: frequency and amount enter as candidates.", "resampled_candidate"),
    Feed("selection", ("inference",), "adjustment", "asks",
         {"mixed": "a never-or-former row joins the card when a lifetime column exists"},
         "Your zero group mixes never-drinkers with people who stopped.", "meaning"),
    Feed("reporting", ("inference", "prediction"), "methods", "discloses", {}, "", "disclosure")),
  failure=Failure(
    pipeline_does="codes every zero as 'non-drinker' and uses two dry days as the reference group",
    counterfactual="the share of the 'non-drinker' group that drank on one of its two days",
    measurable_on=("dietary_recalls.csv",)),
  moment=Moment(contrast=("{both:.1%} are zero on both days", "{expected:.1%} expected by chance"),
                first_at="opening"),
  layout=Layout("focus", views=("distribution", "table_focus"), context=("recall_number", "sex")),
  sentences=Sentences(
    resolved={"episodic": "{Col} was zero on {day_share:.0%} of recall days but for {both:.1%} of "
              "participants on both days (expected {expected:.1%} if zeros were independent days), "
              "so zeros were treated as days without intake and {col} was modeled in two parts."},
    open_at_lock="Whether zeros in {col} mark non-consumers or days without intake was not established.",
    dismissed="Zeros in {col} were treated as amounts.",
    checked_clean="{col}: fewer than 5% of recall days were zero.",
    answers=("STROBE-nut nut-11", "STROBE-nut nut-14", "STROBE-nut nut-12.2", "ROBINS-E exposure measurement")),
  relations=(merges_with("survey.former_users_in_the_reference"),
             merges_with("dietary.former_consumers_in_reference"))))
```

Filled with the catalog's numbers, the moment reads: "Alcohol is zero on 31% of recall days, but only 9.7% of people are zero on both, about what chance alone predicts (9.5%)."

**`clinical.predictor_after_prediction_time`** (family S2) behaves differently under the two purposes. The table shows one thread under each.

| Part | Prediction (`leaky_sepsis.csv`) | Inference |
|---|---|---|
| Notice | The audit sentinel (O3, in-fold, recorded): `abx_escalation_score` agrees with `sepsis` on all 160 admissions. A name or date rule (O0) flags `los_days`, which is known only at discharge (AUC 0.51). | Not run as an outcome view. The existing `after_exposure` card asks about timing for every covariate (O0). |
| Meaning | `timing_vs_index`, one row per flagged column on the opening card, once the moment of use is settled: before, after, or don't know (the fit waits). | `causal_place` on the existing card. No new question. |
| Feeds | **Opening:** the candidate set, by meaning, applied to every column by rule. **Evaluation:** the score with and without the column. **Explanation:** "not a predictor: recorded after the moment of use". **Reporting:** TRIPOD+AI 9b and 27a, PROBAST 2.3, KN-L2. | The adjustment set, and the label of the possible-mediator model (Model 3). |
| Failure | Keeps the column and reports 1.00. Never asks about `los_days`, because its AUC is 0.51. | Adjusts for a consequence of the exposure. |
| Moment | "abx_escalation_score agrees with sepsis on all 160 admissions. When was it recorded?" One tap, and the score settles from 1.00 to 0.81. | none |
| Contrast fixture | `clinical_risk.csv`: `length_of_stay_days` is legitimate for a 30-day readmission model used at discharge, and a leak for one used at admission. The same column means different things at two moments of use. | none |

---

## 2 · The lifecycle in the calm design

### 2.1 States

```
declared ──run──▶ checked-clean            (did not fire; measure kept for the supplement)
             └──▶ noticed ──▶ surfaced (First look, or on its card)
                         └──▶ asked (a row on its card family, where its first consumer needs it)
                                  ├─▶ settled (confirmed · read from data · codebook)
                                  ├─▶ dismissed ("does not apply": a value, with a sentence)
                                  └─▶ left open (at the lock or the seal → a limitation)
settled ──▶ consumed (each feed's read is logged) ──▶ reported (sentence · supplement · checklist)
any state ──data changed──▶ re-measured ──▶ unchanged | "Changed since you looked" | stale: re-asked
```

"Asked" is not "asked at First look". A thread is asked where its first consumer needs it, as the ask card is placed today (`ask.py`). First look shows it as worth a look, and the item reads "Comes up at Energy". The legacy product vision called this the loop that matters: deferred items "resurface, pre-checked and attributed, at the step they target". The thread makes that loop structural.

### 2.2 "Worth a look" (First look, brief §1–§4)

- **What a highlight shows.** Each is the thread's claim, plus "Affects …", which names the stage of its first feed in plain words. The measure appears beside its reference on the canvas, in the thread's layout.
- **The rank key is the brief's: (tier, reach, share, excess, column order).** The tier is derived from the feeds, never set by hand:

  | Tier | When | Note |
  |---|---|---|
  | **T1** | some feed blocks, or refuses with an exit, for this purpose | "Decide this first" |
  | **T2** | some feed changes a number | |
  | **T3** | the feeds only label or disclose | |

  A thread that is already settled drops one tier and reads "Settled".
- **Limits.**
  - At most three highlights, at most two from any one group.
  - O3 and O4 threads never appear here.
  - Silent threads never appear.
  - Two threads that share a subject and a first consumer merge: the second becomes the first's context view.
- **Groups.** The six groups (Who, Missing, Each variable, Together, Over time and batch, The outcome) list every noticed thread with its count. A group with nothing to show states what it checked: "27 columns: no reading beyond its reference". Those are the checked-clean records.

### 2.3 Asked, on its card family

- **One row per (card family, kind, subject).** Each row shows:
  - the guess;
  - the evidence, which is the moment's two numbers on the user's data;
  - each option's consequence (≤ 16 words), previewed on the canvas when the option is pointed at.
- **The card families come from the catalogs.**
  - **Shared:** "Tell me about these columns" (with "read from your data"), the outcome card, the follow-up card, the treatment card, the design card, Who is in (Flow).
  - **Dietary:** instrument and grain, energy accounting, measurement error, what a zero means.
  - **Metabolomics:** the assay card, reference rows and acquisition, the study question.
  - **Genomics:** "What drives your leading components" (Angles over one PC × covariate R² matrix), "Who are your samples", "Where did these come from".
  - **Survey:** the codes question, the scale-declaration form, one skip-gate Flow per module, "the outcome's condition in the questionnaire".
  - **Prediction:** the deployment question.
- **Rows are ordered by consequence.** Mastery unlocks block confirmation (§11.4), which settles exactly the rows it lists. Mastery never changes what is asked.

### 2.4 The context line on a later card

This is beat three, and the part of the design that most makes it feel like magic. When a card's options, order, default or availability were shaped by a settled thread, the card carries one plain line under the question, in the app's voice. It uses no chip and no color. It has a quiet "See it" link that brings the thread's view back as the secondary view on the canvas.

```
┌────────────────────────────────────────────────────┬──────────────────────────────────────┐
│ Model · step 2 of 4                                │ CANVAS                               │
│ How should alcohol enter the model?                │ (pointing at an option previews it;  │
│ From First look: alcohol's zeros are days, not     │  "See it" puts the zero-days view    │
│ people (31% of days, 9.7% of people).    See it    │  beside it as the second view)       │
│                                                    │                                      │
│ ● Two parts: any on a day, then how much  Recommended                                     │
│ ○ One straight line through zero                   │                                      │
│ ○ Quintiles, with zero-intake days as reference    │                                      │
│     Ranked lower: mixes two dry days with abstainers                                      │
│ Why does this matter?                              │                                      │
│                                        [ Continue ]│                                      │
└────────────────────────────────────────────────────┴──────────────────────────────────────┘
```

- **At most one context line per card,** from the highest-ranked thread that shaped it. The others are listed under "Why does this matter?" as "Also shaped by: …".
- **The shelf is never shortened.** A thread reorders options and labels them. It refuses only through a declared conflict, and then with an exit.
- **An unsettled thread** that a card needs shows instead: "Not settled yet: {claim}. {conservative path}." A quiet "Decide now" opens its row.
- **Purpose registry.** The context line answers question 3 ("Why does that matter for my result?") and question 5 ("What did I decide?").

### 2.5 The manuscript: citation and supplement

- **In the app's manuscript rail.** A sentence shaped by a thread ends with a plain-text "noticed" link. Hovering shows the claim and the measure; clicking returns to the First look item. Green stays reserved for recorded sentences (FOUNDATION §4).
- **In the exported methods** there is no marker. The evidence lives inside the sentence: "Alcohol was zero on 31% of recall days but for 9.7% of participants on both days (9.5% expected if zeros were independent days), so zeros were treated as days without intake and alcohol was modeled in two parts."
- **The IDA paragraph** (brief §6.5) becomes generated text: the groups checked, then "the outcome's associations were not examined".
- **Supplement table S1, "What the data showed and what it changed."** One row per noticed thread:
  - the noticing;
  - its measure, with the reference;
  - the researcher's reading;
  - the decisions it shaped, from the consumed-by log;
  - where it is reported, by checklist item.

  Whether a "Checked; nothing beyond its reference" list follows the table is owner question 3.
- **Checklist rules read thread anchors.** `export/checklists.py` currently maps decision kinds to items. It gains thread anchors:
  - a settled thread whose sentence is present answers its items;
  - a thread left open makes them "partly answered", and names what remains.
- **The PROBAST and ROBINS evidence table.** For each domain, it lists the threads that bear on it and their state. It is never a rating: the app does not stamp pass or fail, and reviewers rate.

### 2.6 Open threads, listed before the lock and before the seal

Under inference, the lock happens on the first estimate shown (`plan_lock.py`). Just before it, one card lists the threads still open that feed the plan. Under prediction, the same card appears before the held-out rows are scored (`open_seal`), and lists the open threads that change the honest score.

```
┌────────────────────────────────────────────────────┬──────────────────────────────────────┐
│ Before you see estimates                           │ CANVAS: the selected noticing's view │
│ The plan is fixed the first time estimates appear. │                                      │
│ Two noticings still shape it.                      │                                      │
│                                                    │                                      │
│ 1 Sodium's two recalls barely agree (r 0.01).      │                                      │
│   Affects how its result reads.                    │                                      │
│   Decide now · Leave as a limitation               │                                      │
│ 2 One in six follow a special or diabetic diet.    │                                      │
│   Affects who is analyzed.                         │                                      │
│   Decide now · Leave as a limitation               │                                      │
│                                                    │                                      │
│ Settled 6 · checked, nothing found 23              │                                      │
│                        [ Fix the plan and show estimates ]                                │
└────────────────────────────────────────────────────┴──────────────────────────────────────┘
```

- **Leaving a thread open** writes its `open_at_lock` sentence into the limitations draft. The lock records which threads were left open, and that list is covered by the plan's hash.
- **A T1 thread cannot be left open.** Its exit is to decide it, or to change the analysis it blocks.
- **This is §11.4's objective list** ("the open slots are the objectives"), shown at the one moment it matters most.

### 2.7 Threads born late

Some threads are noticed only once there is a fit:
- the p-value histogram;
- unstable panels and signatures;
- reversed expected directions;
- the noise ceiling;
- a model no better than the baseline;
- the trust row on a top feature.

These are O3 and O4 threads. They never enter First look.
- **Under inference** they appear only after the lock: on the Results card, as its one context line, or in "Which of my decisions mattered?".
- **Under prediction** they appear on the evaluation ladder (§5, U9).
- **Their feeds** are labels, disclosures and sensitivity analyzes only (R2).

### 2.8 Load

- **Caps:**
  - at most 3 "worth a look" highlights;
  - at most 1 context line per card;
  - at most 3 surfaced noticings per stage (the shared catalog's rule);
  - at most one row per (family, kind, subject).
- **The gate reports asked rows for each reference journey** (§14.3 demands this). The catalogs' targets after merging:

  | Journey | Asked | Surfaced |
  |---|---|---|
  | Dietary, NHANES, inference | about 5 new cards | 4 |
  | Clinical, multi-site EHR, prediction (the heaviest) | 8–9 rows on 5 cards | 4–6 |
  | Metabolomics | 2–4 cards (3–5 under prediction) | 5–7 |
  | Genomics, shipped fixture | 2–3 | 0–1 |
  | Survey, NHANES | about 8 | about 4 |
  | Shared, with batching | 8–12 | none given |

- **A thread that would push a reference journey past its target** must join an existing row or be stated with a changeable phrase. The registry test checks this against the reference journeys.

---

## 3 · What is generative

### 3.1 The seventeen families

The families decompose the three things a score cannot do. Each family's reason is structural: no amount of resampling changes it. That is what makes the list closed under new data, where 333 examples alone could not be.

| Family | The question it makes the researcher answer | Why no score can see it | Lens-agnostic sentinel | n |
|---|---|---|---|---|
| **S1 Shortcut** | Does a process line up with the outcome? | The process is in every fold, so resampling rewards it | A process-only baseline in-fold, on the settled structural columns (batch, site, date, plate, run order, interviewer, device, cycle), plus an outcome-free PC × covariate R² matrix for structure no column names | 16 |
| **S2 Leak in time** | Was this known at the moment of use, or before the outcome? | Information from after the index sits in every fold | A timing question for every candidate (§1.3 sentinel); dates later than the index | 7 |
| **S3 Leak in meaning** | Is the outcome written inside a predictor? | A definition predicts its own label in every fold | A finder for an exact rule (the outcome as a threshold or composite of columns, asked as a meaning question); near-identity with the outcome; a feature count far below the platform's vocabulary | 8 |
| **S4 Not independent** | What is one unit, and which rows belong together? | Row-wise folds put one unit on both sides, so the resampling itself is wrong | Repeated identifiers; the nearest-neighbour identity rate; near-duplicate rows; the gap between grouped and random folds | 22 |
| **S5 Who is in** | Who is in these rows, and how did they get here? | The score is computed on the selected rows only | The flow: who leaves at each step and how they differ (predictors only); design columns; prevalence against the stated population | 24 |
| **S6 Done before upload** | What was done to these values before you got them? | It happened before any fold existed | Provenance fingerprints: identical within-group spreads, constant column sums, already-logged ranges, quantile-identical distributions, flag columns | 6 |
| **S7 Drift and transport** | Will the place, time or instrument of use look like this? | A score estimates performance in the data's own distribution | Leave-one-period-out or site-out spread; a predictors-only shift classifier; the deployment question | 9 |
| **K1 What a value means** | What does this value say, in what unit, about whom? | A model fits any coding, and a wrong unit still predicts (and can turn into a site shortcut) | The readings ledger: units, codes, sentinel codes, sex coding, identifiers | 49 |
| **K2 What a zero or blank means** | Is this a fact, a skip, a non-detection or a gap? | The mechanism is not in the data; any default fits | Missingness against other columns (predictors only); gate columns; zero mass; missingness against abundance | 22 |
| **K3 How well it measures** | How much of this value is the person, and how much the instrument? | Noise weakens coefficients and caps scores without announcing itself | Any replicate of one quantity → a reliability; reference rows → technical noise; self-report beside a measured value | 41 |
| **K4 Causal place** | Where does this sit between the exposure and the outcome? | A confounder, a mediator and a consequence predict equally well | The adjustment card's questions for every covariate; treatment columns; exposure–covariate overlap (predictors only) | 25 |
| **K5 Structure among variables** | Which columns are built from others? | What a coefficient means (substitution or addition) is not a property of fit | Exact and near identities (sums, ratios, cuts); collinearity clusters; closure to a constant | 23 |
| **K6 Outcome and clock** | What is the outcome, and when does its clock start and stop? | The score is defined relative to the outcome you chose | The outcome card: follow-up, dates, the rule finder, the levels | 27 |
| **K7 Reference and context** | Compared with whom, on what scale, under what physiology? | The comparison group changes the estimate, not the score | Ever/current column pairs; life-stage and population tables; the unit of the contrast | 12 |
| **E1 Support** | Can these data support this claim where it is made? | A score is an average, and regions without data are invisible in it | Events per parameter (Riley); sparse cells; tails; overlap; effective n | 18 |
| **E2 Noise and multiplicity** | Does the signal beat noise and the number of looks taken? | The best of many looks beats a one-look null, and a score cannot count looks | A counter of looks; a label-permutation null (scheduled); instability across refits; the no-predictor baseline | 16 |
| **E3 Reading the result** | What can this explanation or null claim, and for whom? | An explanation attributes to the model's features, not to the world | Labels on explanations (weighting, grouping, support); diagnostics; data quality by group | 8 |

The n column sums to 333. Appendix A lists every thread by family.

### 3.2 Specialization and merging

- **Inheritance.** A lens thread `specializes` a shared one. It inherits:
  - the failure statement;
  - the generic feeds;
  - the anchors;
  - the purpose legality.

  It overrides the detector, the meaning's wording and values, the moment and the sentence.
- **Merging.** Two threads with the same (family, kind, subject) merge into one row. The merged row keeps the most specific claim and the union of the feeds.
- **Why this matters.** An NHANES journey that runs the survey, dietary and clinical lenses together asks "Was this blank because the gate said skip?" once, not three times.
- **Appendix B** lists the merges the catalogs imply. There are at least 60 lens threads that specialize a shared one.

### 3.3 Sentinels catch what no catalog names

Each family has at least one detector that runs under every lens, including "Something else, or not sure". When it fires with no lens specialization, it raises a generic thread of its family:
- the claim: "`plate_id` alone predicts the outcome at AUC 0.79";
- the family's question;
- generic options;
- the family's failure statement;
- a generic sentence.

The catalogs add specific meaning and the specific moment. The sentinels guarantee that a family never misses a situation silently.

The prototype proves this directly (§6, the sentinel proof): with the lens set to "Something else", the S2/S3 sentinels must still catch `abx_escalation_score`, and the S1 sentinel must still catch an aligned run order.

### 3.4 What a pack writes, and what never changes

**A pack writes, to add a thread:**
1. the `Thread` declaration;
2. the detector function, with its reference;
3. two committed fixtures: one where it fires and one where it stays silent;
4. if needed, a reading kind with its alternatives, plus a value test or none;
5. feeds to existing consumers, or a new method contract through §13 when no consumer exists;
6. the sentence templates;
7. its sources.

**Never changes:**
- the First look guide and its six groups;
- the card-family row grammar;
- the context line;
- the manuscript mark;
- supplement table S1;
- the open-threads card;
- the five footprint layouts;
- the closed view vocabulary.

Adding a new card family or a new view kind is the only screen work, and each is a recorded design decision.

### 3.5 Coverage as a test

- **The frameworks live as data in the coverage registry:**
  - PROBAST's 20 signalling questions;
  - the domains of ROBINS-E and ROBINS-I V2;
  - Kapoor and Narayanan's 8 leakage types;
  - TRIPOD+AI's 52 items and STROBE-nut's 58 rows (both already in `export/data/*.json`);
  - the 14 stage × purpose cells;
  - the family × lens matrix.
- **The test fails when** an item maps to no thread and no out-of-scope reason.
- **An out-of-scope reason comes from a closed list:**
  - the author supplies it, because the record cannot know it;
  - it is enforced by construction (name the guard and its test);
  - it is outside TurboTab's scope (name what).
- **The test also fails when** a thread anchors to nothing, which is "every element earns its place" applied to threads.
- **A new framework version** (PROBAST+AI, a TRIPOD update) turns the test red until it is mapped. That makes comprehensiveness a property of the build, not of anyone's memory.

---

## 4 · Coverage report

### 4.1 Family × lens matrix (counts of catalog threads)

| | dietary | clinical | metabolomics | genomics | survey | shared | total |
|---|---|---|---|---|---|---|---|
| S1 shortcut | 3 | 1 | 2 | 6 | **0** | 4 | 16 |
| S2 leak in time | **0** | 2 | 2 | 1 | 1 | 1 | 7 |
| S3 leak in meaning | 1 | 1 | 1 | 3 | 1 | 1 | 8 |
| S4 not independent | 3 | 1 | 3 | 4 | 2 | 9 | 22 |
| S5 who is in | 3 | 5 | 1 | 1 | 6 | 8 | 24 |
| S6 done before upload | 1 | **0** | 3 | 1 | **0** | 1 | 6 |
| S7 drift and transport | 1 | 3 | 1 | 1 | 1 | 2 | 9 |
| K1 value meaning | 7 | 6 | 4 | 9 | 7 | 16 | 49 |
| K2 zero or blank | 3 | 3 | 4 | 1 | 5 | 6 | 22 |
| K3 measurement | 13 | 5 | 9 | 3 | 7 | 4 | 41 |
| K4 causal place | 5 | 5 | 3 | 5 | 1 | 6 | 25 |
| K5 structure | 7 | 1 | 3 | 3 | 4 | 5 | 23 |
| K6 outcome and clock | **0** | 12 | **0** | **0** | 2 | 13 | 27 |
| K7 reference and context | 4 | 4 | **0** | **0** | 3 | 1 | 12 |
| E1 support | 2 | 3 | 1 | 1 | 4 | 7 | 18 |
| E2 noise and multiplicity | 1 | 1 | 4 | 4 | 1 | 5 | 16 |
| E3 reading the result | 1 | 1 | 1 | 1 | 2 | 2 | 8 |
| **total** | 55 | 54 | 42 | 44 | 47 | 91 | 333 |

Genomics includes the catalog's shared-sampled-after-diagnosis and shared-score-within-noise. Survey includes shared-panel-attrition and shared-ema-within-between.

**The empty cells:**
- **S1 survey:** covered by shared-process-aligned-with-outcome (interviewer, mode) and shared-season.
- **S2 dietary:** covered by shared-predictor-after-the-index. Diet's own timing hazard sits in K4 (diet-changed-because-of-disease).
- **K6 dietary, metabolomics and genomics:** covered by the 13 shared outcome-and-clock threads, which are lens-agnostic by nature.
- **K7 metabolomics and genomics:** physiological state and life stage arrive through clin-physiological-state and shared-children-among-adults in mixed journeys. No omics-specific gap was found.
- **S6 clinical and survey:** a real gap (G7 below). Values filled by the source system carry no flag: EHR last-observation-carried-forward, or agency-edited survey values.

### 4.2 PROBAST (Wolff et al. 2019): the 20 signalling questions

| Item | Signalling question | Threads (family) |
|---|---|---|
| 1.1 | Appropriate data sources? | shared-case-control-sampling, clin-outcome-dependent-sampling, survey-nonprobability-sample (S5); clin-diagnostic-or-prognostic, shared-randomized-arm (K6) |
| 1.2 | All inclusions and exclusions appropriate? | shared-selection-flow, shared-selection-on-a-consequence, clin-prevalent-cases-at-baseline, diet-implausible-reporters, metab-reference-rows, shared-non-data-rows (S5) |
| 2.1 | Predictors defined and assessed similarly for all? | shared-process-aligned-with-outcome, clin-site-differences (S1); shared-method-change (S7); clin-mixed-units (K1); clin-self-report-vs-measured, survey-mode-proxy-nonequivalence (K3); shared-quality-differs-by-group (E3) |
| 2.2 | Predictors assessed without knowledge of the outcome? | metab-sample-timing-vs-diagnosis, shared-sampled-after-diagnosis (S2); survey-same-sitting-reverse-causation (K4); amendment G6 (recall after diagnosis in case–control designs) |
| 2.3 | All predictors available at the intended moment of use? | shared-predictor-after-the-index, clin-predictor-after-prediction-time (S2); diet-deployment-measurement-heterogeneity, genomics-new-site-new-platform (S7); new G1 |
| 3.1 | Outcome determined appropriately? | clin-diagnosed-not-diseased, shared-outcome-ascertainment, survey-self-reported-diagnosis-outcome (K6); clin-partial-verification (S5) |
| 3.2 | Prespecified or standard outcome definition? | the outcome card (K6); clin-population-specific-cutoffs (K7); shared-coarsened-copy (K5) |
| 3.3 | Predictors excluded from the outcome definition? | shared-outcome-proxy, clin-outcome-defined-by-predictors, metab-outcome-defined-by-a-feature, genomics-outcome-defined-from-features (S3) |
| 3.4 | Outcome defined and determined similarly for all? | clin-coding-system-transition, shared-method-change (S7); shared-outcome-ascertainment (K6) |
| 3.5 | Outcome determined without knowledge of predictors? | new G2; clin-informative-test-ordering (K2) in part |
| 3.6 | Time interval appropriate? | shared-follow-up-varies, clin-diagnostic-or-prognostic, clin-time-zero (K6); clin-preclinical-disease-before-event (S2) |
| 4.1 | Reasonable number of participants with the outcome? | shared-events-not-rows, survey-weight-effective-n (E1) |
| 4.2 | Continuous and categorical predictors handled appropriately? | shared-coarsened-copy (K5); diet-quantiles-sort-by-sex-and-size (K7); shared-sparse-levels, shared-tails-and-support (E1); shared-codes-or-amounts (K1) |
| 4.3 | All enrolled participants included? | shared-selection-flow (S5); shared-missingness-mechanism (K2) |
| 4.4 | Missing data handled appropriately? | the K2 family |
| 4.5 | Selection on univariable analysis avoided? | genomics-features-preselected-on-outcome (S3); shared-selection-unstable (E2); also by construction (the selection menu refuses selection outside the resampling) |
| 4.6 | Complexities accounted for (censoring, competing risks, control sampling)? | shared-competing-death, shared-follow-up-varies, clin-event-found-at-visits (K6); shared-case-control-sampling, shared-survey-population-design (S5); shared-unit-repeats (S4) |
| 4.7 | Relevant performance measures? | clin-rare-outcome-threshold (E1); shared-score-within-noise (E2); also by construction (calibration for every task) |
| 4.8 | Overfitting and optimism accounted for? | metab-wide-noise-ceiling, shared-model-no-better-than-baseline (E2); shared-unit-repeats (S4); also by construction (nested CV, BBC-CV, the seal) |
| 4.9 | Final model matches the multivariable analysis? | **Out of scope, by construction.** The exported model is the fitted object, and the replay regenerates it. |

PROBAST+AI (Moons et al., BMJ 2025) keeps the four domains and adds concerns about data quality, fairness and leakage. Those map to S6, K3, E3 and the S-family leakage threads. Its exact item list must be checked against the publication before its rows enter the coverage registry. I have not verified the item numbers.

### 4.3 ROBINS-E and ROBINS-I V2: the seven domains each

| Domain (ROBINS-E name · ROBINS-I V2 name) | Threads |
|---|---|
| Confounding · Confounding | the K4 family: shared-mediator-or-confounder, diet-healthy-lifestyle-cluster (with the E-value), shared-missing-confounder, clin-confounding-by-indication, genomics-ancestry-structure, genomics-cell-mix-drives-signal, genomics-exposure-written-in-the-profile (confounder measured with error), shared-time-varying-feedback (time-varying confounding); shared-positivity-overlap (E1) |
| Measurement of the exposure · Classification of interventions | the K3 family (diet-day-to-day-variance, diet-instrument-kind, diet-misreporting-tracks-body-size, clin-single-reading-dilution, survey-reliability-attenuation); the K1 family (units, equivalents); shared-method-change (S7); diet-supplements-in-totals |
| Selection of participants into the study or analysis · same | the S5 family; clin-time-zero and shared-time-zero (K6: follow-up and exposure start at different times) |
| Post-exposure interventions · Deviations from intended interventions | shared-treatment-paradox, clin-treatment-during-follow-up (K4); clin-intercurrent-events, clin-crossover-design, shared-randomized-arm (K6) |
| Missing data · same | the K2 family; shared-informative-dropout, clin-loss-to-follow-up (S5) |
| Measurement of the outcome · same | clin-diagnosed-not-diseased, shared-outcome-ascertainment, survey-self-reported-diagnosis-outcome (K6); diet-intake-as-outcome (K3); clin-coding-system-transition (S7); new G2 |
| Selection of the reported result · same | the E2 family (shared-exposure-family, diet-nutrient-wide-scan, survey-question-wide-scan, shared-curvature-seen-in-explore, shared-subgroup-seen-in-explore); also by construction (the plan lock, the hash, and "after the estimates were seen") |

ROBINS-E is primary for this app, because diet is an exposure, not an intervention. Under ROBINS-I V2, the "deviations" domain applies only to trial-like designs: the randomized arm, crossover and intercurrent events.

### 4.4 Leakage types

| Type (Kapoor & Narayanan 2023; Kaufman et al. 2012) | Threads, or the guard |
|---|---|
| L1.1 No test set | **By construction:** the seal (prediction); the selection-corrected estimate when nothing is held out |
| L1.2 Preprocessing on training and test together | **By construction:** in-fold steps, with scope tested by perturbation (`contracts.observed_scope`). **Threads, for preprocessing done before upload (S6):** genomics-normalized-across-all-samples, metab-pre-corrected-batches, diet-delivered-already-adjusted, metab-already-transformed |
| L1.3 Feature selection on training and test together | **By construction:** selection outside the resampling is refused. **Threads:** genomics-features-preselected-on-outcome, genomics-polygenic-score-provenance (S3) |
| L1.4 Duplicates | shared-duplicate-records, genomics-near-identical-samples, metab-technical-replicates, shared-imputed-copies (S4) |
| L2 Illegitimate features (Kaufman: leakage in features) | **S2:** shared-predictor-after-the-index, metab-treatment-marker, survey-items-downstream-of-the-outcome, metab-sample-timing-vs-diagnosis. **S3:** shared-outcome-proxy, clin-outcome-defined-by-predictors, diet-ratio-features-share-the-outcome, survey-item-overlap-with-outcome. **K1:** shared-identifier-as-predictor. **S1:** diet-recall-process-shortcuts |
| L3.1 Temporal leakage | clin-predictor-after-prediction-time (S2); clin-calendar-drift (S7, time-ordered validation); G4 (a window before the index for dense series) |
| L3.2 Non-independence between training and test (Kaufman: leakage in training examples) | the S4 family: shared-unit-repeats, clin-encounters-not-patients, metab-person-fingerprint, genomics-relatedness, diet-meal-level-rows, shared-spatial-dependence, shared-grouping-above-the-person |
| L3.3 Sampling bias in the test distribution | shared-case-control-sampling, clin-outcome-dependent-sampling (S5); shared-case-mix-shift, genomics-new-site-new-platform, metab-cross-batch-transport (S7) |
| Outcome in the imputation, under prediction | **By construction:** BLUEPRINT §12 ruling 4 |
| A test set reused for tuning | **By construction:** nested tuning, and the seal opened once |
| A shortcut a process carries (not in the taxonomy; the commonest omics leak) | the S1 family, with a process-only baseline in evaluation |

### 4.5 TRIPOD+AI: all 52 items

| Items | Mapping |
|---|---|
| 1, 2, 4, 17, 18a, 18b, 18e, 19, 25, 27b, 27c | **Out of scope, author.** The title, abstract, objectives, ethics, funding, conflicts of interest, data sharing, patient and public involvement, interpretation, user interaction and next steps are facts the record cannot know. The development-or-evaluation term in item 1 and the open-threads list (useful for 27c) are offered as drafts. |
| 18c, 18d | **Out of scope, author.** The plan's hash supports registration; it is not a thread. |
| 18f, 12g, 22 | **Out of scope, by construction.** The export bundle and the replay; the fitted model object is exported. |
| 3a | clin-diagnostic-or-prognostic, shared-follow-up-varies (K6) |
| 3b | survey-population-or-sample (S5); intended use (clin-rare-outcome-threshold); G1. The author adds the clinical context. |
| 3c | shared-quality-differs-by-group (E3), on the data side; the author adds the background |
| 5a | shared-case-control-sampling, survey-population-or-sample, shared-randomized-arm (S5, K6); the author names the sources |
| 5b | clin-calendar-drift (S7, the date span); shared-follow-up-varies; the author gives accrual dates |
| 6a | shared-grouping-above-the-person, clin-site-differences (S4, S1); the author describes the setting |
| 6b | shared-selection-flow, diet-implausible-reporters, clin-physiological-state, shared-children-among-adults, metab-reference-rows |
| 6c | clin-treated-measurements, shared-treatment-paradox, clin-treatment-during-follow-up (K4) |
| 7 | the K1, K2, K3 and S6 families, with the IDA paragraph; across groups: shared-quality-differs-by-group |
| 8a | the outcome card (K6): clin-outcome-defined-by-predictors, clin-diagnosed-not-diseased, clin-state-or-event, shared-competing-death, the horizon |
| 8b, 8c | new G2 (asked on the design card when the outcome is adjudicated or a judgment); the author completes it |
| 9a | genomics-features-preselected-on-outcome (S3); clin-incremental-value (E2); shared-exposure-family |
| 9b | shared-predictor-after-the-index (S2); units (K1); diet-instrument-kind (K3); G1 |
| 9c | clin-self-report-vs-measured, shared-proxy-respondent, survey-mode-proxy-nonequivalence (K3) |
| 10 | shared-events-not-rows, shared-predictors-outnumber-rows, survey-weight-effective-n (E1) |
| 11 | the K2 family |
| 12a | shared-unit-repeats (grouped folds, S4); clin-calendar-drift (time-ordered, S7); the seal by construction |
| 12b | the K1 and K5 families, shared-zero-mass-predictor (K2), shared-tails-and-support (E1), shared-curvature-seen-in-explore (E2) |
| 12c | **By construction:** the contract sentences, tuning and the shelf. shared-selection-unstable (E2) |
| 12d, 23b | shared-grouping-above-the-person (leave-one-cluster-out spread), clin-site-differences, genomics-new-site-new-platform |
| 12e | clin-rare-outcome-threshold (E1); shared-score-within-noise (E2); by construction (proper scores and calibration) |
| 12f, 24 | diet-deployment-measurement-heterogeneity (recalibration at deployment noise), shared-case-mix-shift (S7) |
| 13 | clin-rare-outcome-threshold, shared-events-not-rows (E1); Explore's rare-class lever |
| 14 | shared-quality-differs-by-group (E3); new G3 |
| 15 | clin-rare-outcome-threshold (the threshold at intended use) |
| 16, 20c | shared-case-mix-shift, genomics-new-site-new-platform, metab-cross-batch-transport (S7) |
| 20a | shared-selection-flow (the flow figure) |
| 20b | shared-quality-differs-by-group (E3); clin-site-differences (S1); shared-missingness-mechanism (K2) |
| 21 | shared-events-not-rows (E1) |
| 23a | shared-score-within-noise (E2); G3 |
| 26 | every thread's limitation sentence, through the open-threads card; the author edits |
| 27a | G1; diet-deployment-measurement-heterogeneity; clin-informative-test-ordering |

All 52 items are covered: 1, 2, 3a–c, 4, 5a–b, 6a–c, 7, 8a–c, 9a–c, 10, 11, 12a–g, 13–17, 18a–f, 19, 20a–c, 21, 22, 23a–b, 24, 25, 26 and 27a–c.

### 4.6 STROBE-nut: all 58 rows

| Rows | Mapping |
|---|---|
| 1b, 2, 3, 18, 22, nut-22.1 | **Out of scope, author.** The abstract, background, objectives, key results, funding and ethics. The plan lock lists the declared exposure and estimand to help with row 3. |
| 1a, nut-1 | The author writes the title. The design term comes from the design card (K6/S5), and the assessment method from diet-instrument-kind (K3). |
| nut-22.2 | The author provides the instruments. survey-instrument-published-scoring (K7) states the instrument version. |
| 4, 12d, 6b | shared-randomized-arm, clin-crossover-design (K6); shared-case-control-sampling, metab-matched-sets, survey-population-or-sample, shared-informative-dropout (S5, S4) |
| 5 | the date span (S7); site (S1); the author describes the setting |
| nut-5 | diet-season-of-assessment, shared-season (S1) |
| 6a | shared-selection-flow, clin-prevalent-cases-at-baseline, shared-informative-dropout (S5) |
| nut-6 | clin-physiological-state (K7); diet-changed-because-of-disease (K4); diet-implausible-reporters (S5) |
| 7 | the outcome card (K6) and the adjustment card (K4: shared-mediator-or-confounder) |
| nut-7.1 | diet-name-is-not-a-nutrient, diet-nutrient-equivalents, diet-supplements-in-totals (K1) |
| nut-7.2 | diet-patterns, diet-quality-index-is-a-density (K5); G5 |
| 8 | the K1 and K3 families; shared-method-change (S7) |
| nut-8.1 | diet-instrument-kind, diet-recall-completeness, diet-recall-nuisance-effects (K3) |
| nut-8.2 | diet-assessment-batch, for the food-composition release (S1); diet-nutrient-equivalents (K1) |
| nut-8.3 | diet-dri-life-stage (K7); diet-usual-intake-distribution (K3) |
| nut-8.4 | clin-specimen-quality-flags, clin-season-of-draw, clin-fasting-status (K3); clin-inflammation-adjustment (K7); shared-preanalytical-handling (S1) |
| nut-8.5 | diet-healthy-lifestyle-cluster (K4); shared-predictor-after-the-index (S2, timing); clin-self-report-vs-measured (K3) |
| nut-8.6 | diet-ffq-recall-substudy, diet-recovery-biomarker, shared-validation-subsample (K3) |
| 9 | every S and K thread's sentence (the bias paragraph) |
| nut-9 | diet-misreporting-tracks-body-size (K3); diet-implausible-reporters (S5); diet-changed-because-of-disease (K4) |
| 10 | shared-events-not-rows (E1); the author justifies the size |
| 11, 16b | diet-quantiles-sort-by-sex-and-size, diet-meaningful-increment (K7); shared-coarsened-copy (K5) |
| nut-11 | diet-mass-at-zero, diet-zero-is-a-day (K2); diet-former-consumers-in-reference (K7) |
| 12a | the K4 family; by construction (contract sentences) |
| 12b | shared-interaction-support (E1); shared-subgroup-seen-in-explore (E2) |
| 12c, 14b | the K2 family; shared-missingness-mechanism |
| 12e, 17 | every thread whose feed adds a sensitivity analysis (diet-implausible-reporters, diet-changed-because-of-disease, clin-preclinical-disease-before-event, …); subgroups seen in Explore are labeled |
| nut-12.1 | diet-day-to-day-variance (K3); diet-food-rows-sum-to-days (S4); diet-repeated-ffq-cumulative (K4) |
| nut-12.2 | diet-energy-carries-the-nutrient (K5); diet-usual-intake-distribution (K3); diet-dietary-weight-tier (S5) |
| nut-12.3 | diet-day-to-day-variance, diet-too-noisy-to-correct, diet-few-replicate-persons (K3) |
| 13a, 13b, 13c | shared-selection-flow (the flow figure); the author gives recruitment reasons |
| nut-13 | diet-implausible-reporters (S5); diet-recall-completeness (K3) |
| 14a, nut-14 | diet-healthy-lifestyle-cluster (Table 1 by exposure, predictors only, K4); diet-mass-at-zero (consumers against non-consumers) |
| 14c | shared-follow-up-varies (K6) |
| 15 | shared-events-not-rows (E1) |
| 16a | **By construction:** Table 2 (crude, Model 1, Model 2). The K4 family sets the adjustment. |
| 16c | shared-common-outcome-measure (K6) |
| nut-16 | diet-supplements-in-totals (K1) |
| nut-17 | diet-implausible-reporters (S5); the K2 family (imputation) |
| 19, nut-19 | every limitation sentence; diet-invisible-exposure (E3); diet-instrument-kind (K3) |
| 20 | shared-exposure-family, diet-nutrient-wide-scan (E2); the author interprets |
| nut-20 | diet-meaningful-increment (K7); the author judges relevance |
| 21 | survey-population-or-sample, survey-nonprobability-sample (S5); the author discusses |

All 58 rows are covered.

### 4.7 The 14 stage × purpose cells

| Stage | Inference | Prediction |
|---|---|---|
| **Opening** (ingest, units, roles, rows, eligibility, the seal) | **Families:** K1, K2, S4, S5, S6, K6, K7. **Effects:** asks, sets defaults, changes rows. Eligibility is decided before the lock from predictors-only contrasts. **Example:** diet-changed-because-of-disease moves the flag to an eligibility sensitivity analysis before any estimate. | **Also S2 and S7:** the moment of use, and a time-ordered split. S4 makes the seal grouped. **Example:** clin-predictor-after-prediction-time; shared-unit-repeats leads to a split by person. |
| **Representation** (repairs, transforms, measurement correction, features) | **Families:** K1, K2, K3, K5, S1, S6. Declared, frozen at the lock, learned without the outcome on the full sample. **Example:** metab-qc-drift (reference rows); survey-reverse-keying (the key comes from the instrument). | **The same, with learned parts refit in-fold.** **Example:** genomics-batch-aligned-with-case (outcome-free batch adjustment in-fold); metab-left-censored-nondetects (QRILC in-fold). |
| **Selection** (predictors, adjustment set, dimension) | **Families:** K4 (the adjustment set), E2 (the exposure family and its multiplicity), K5 (dimension by a declared rule). No data-driven selection. **Example:** diet-food-and-its-nutrients moves nutrients into a declared secondary model. | **Families:** E2 (candidates added, never pruned), S3 (the preselection question), E1 (p ≫ n), S7 and G1 (availability at use). **Example:** metab-one-compound-many-features (grouped candidates); survey-items-or-score. |
| **Model** (family, inductive bias, form, tuning) | **By estimand:** K6 (outcome kind → family), K2 (two-part), S4 (mixed model or GEE), S5 (design-based, case–control), K3 (measurement-error model as a secondary analysis), E1 (Firth or penalization when sparse). | **The same families, as resampled candidates.** diet-energy-carries-the-nutrient states that the energy-model choice matters little here. |
| **Evaluation** (validation, calibration, sensitivity, the honest score) | K3 (regression calibration as a secondary analysis); S5 (Goldberg as a sensitivity analysis); K4 (E-value); S1 (batch sensitivity); E3 (diagnostics); E2 (the specification curve, after the lock) | S4 (grouped vs random); S1 (process-only baselines, leave-one-batch-out); S7 (time-ordered, deployment noise, transport); E2 (noise ceiling, baseline); G3 (subgroup performance) |
| **Explanation** (importance, effects, ALE/SHAP, the architecture lane) | Effects only as the estimand states them. K7 (the meaningful increment); E3 (invisible exposure, expected directions); E1 (effect curves end where the support ends) | K5 (grouped importance: one compound, modules, items); S2 (a treatment marker revealed); S1 (features that are the end of the run); E3 (the trust row; explanations describe the sample) |
| **Reporting** (methods sentences, checklists, figures) | Every thread's sentence; STROBE-nut and ROBINS-E anchors; limitations from open threads; the IDA paragraph; supplement S1 | Every thread's sentence; TRIPOD+AI and PROBAST anchors; the same supplement |

Every cell is fed by at least two families, so none is out of scope.

### 4.8 Gaps the mapping found, each becoming a thread

| Id | Thread | Family | Why it is needed |
|---|---|---|---|
| G1 | **shared-available-at-use.** A predictor is measured here but will not be available, or not the same, at the moment of use. Asked under prediction; silent under inference. | S7 | PROBAST 2.3, TRIPOD+AI 27a. The catalogs cover timing, but not routine availability (a research assay, a two-day recall). |
| G2 | **shared-outcome-assessed-knowing-predictors.** The outcome was adjudicated or judged by someone who knew the predictors. A design-card row, asked when the outcome is a judgment. | K6 | PROBAST 3.5; TRIPOD+AI 8b, 8c |
| G3 | **shared-performance-differs-by-group.** Calibration or discrimination differs across the groups confirmed in Explore. Surfaced at evaluation, under prediction. | E3 | TRIPOD+AI 14, 23a, 3c. The catalogs cover data quality by group, but not performance. |
| G4 | **shared-dense-series-sufficiency.** CGM, accelerometer or diary series need a sufficiency rule (14 days with at least 70% wear), a standard duration, a window before the index, the sampling interval, and a pinned MAGE implementation. | S4/K3 | The verified rule table §3.3 and its repeated-measures row. No catalog covers dense series. |
| G5 | **shared-low-dimensional-structure.** A few components carry most of the predictors' variance. Under inference, k comes from a declared rule (parallel analysis or MAP, not eigenvalue > 1). Under prediction, k is tuned with an unreduced comparator. UMAP and t-SNE are QC proposals only. | K5 | The rule table's dimension-reduction row; STROBE-nut nut-7.2 |
| G6 | **Amendment to shared-case-control-sampling:** add the row "Was the exposure recalled after diagnosis?" | S5 | PROBAST 2.2 (differential recall) |
| G7 | **shared-filled-by-the-source.** Values carried forward or edited by the source system with no flag: runs of identical values across visits beyond plausibility, or spikes at a model's output. | S6 | The empty S6 cells for clinical and survey |

---

## 5 · Engine work, as work packages

**What exists today.** The ledger's machinery, the method contracts and their relations, and most domain consumers already exist. By a module census, roughly 70 of the 333 catalog threads have a detector, a reading or a question in the engine today: about one in five, an approximate count. These include:
- dietary: implausible intake, energy adjustment, Atwater, compositional shares, survey weights and lonely PSU, nesting;
- clinical: censored values, mixed units, default mass, impossible versus extreme, number format;
- metabolomics: run-order drift with its chance share, redundancy, left-censoring, pooled QCs and sample roles, no run order, repeated subjects, zeros, already transformed, duplicates, empty blocks, ion modes;
- genomics: data type, gene-id damage, p over n, single cell, batch confounding;
- survey: sentinel codes, ordinal declared;
- shared: identifiers, flags, grain and repeats, codes or amounts, units by magnitude, sex coding, imputed copies, groupings, follow-up, after-exposure timing, Explore's outcome-free checks, Riley n, positivity, instability, the baseline.

None has the thread's end-to-end wiring.

**Consumers the catalogs need that have no code today** (searched by keyword in `turbotab/core`):
- competing risks (only a caveat line exists, in `dietary_caveats.py`);
- re-anchoring for time zero and immortal time;
- IPCW;
- leave-one-batch-out or leave-one-site-out validation;
- the specification curve promised in §11.4;
- parallel analysis or MAP;
- re-scoring at deployment noise.

| WP | Work | Status | Builds on | Size | Done when |
|---|---|---|---|---|---|
| U1 | Thread registry and contract (§1.2–1.3), with the legality matrix and the 17-family table | new | `contracts.register_contract`; `Pack` | M | One refusal test per rule R1–R10; one legality test per matrix cell |
| U2 | Ledger extension: subject shapes, meaning kinds in `KIND_RULES`, `leave_open`, staleness through `invalidates`, conservative paths | partial | `Reading`, `KIND_RULES`, `Unsettled`, `confirm_readings`, `read_from_data`, the codebook path | M | Row-set confirmation settles exactly the rows it lists; a stale thread is re-asked |
| U3 | The census in both directions, and the honored-confirmation test per alternative | partial | `CONSUMERS` and the structural census; §14.3's honored test | S | Red on any undeclared branch or any ignored confirmation |
| U4 | The `notices` stage (First look's engine): measure, reference, reach, rank key, rows by O-class, a cache, heavy cost scheduled. Findings migrate into threads by gaining a family, feeds and sentences. | partial | the `findings` stage, pack detectors, Explore's outcome-free findings, the assay drift chance share, `evidence.py` | M–L | The rank key replaces severity sorting (`findings.py:225`); the outcome is never drawn from `/columns` |
| U5 | Card families in the Router: one question key whose rows come from threads, placed by NEEDS, with block confirmation | partial | `ask.py`, `covariate_guesses.py`, the adjustment, follow-up, grain and intended-use questions | M | The reference journeys' asked rows are reported, and within the catalogs' targets |
| U6 | Context lines and option ranking from feeds; purpose-registry entries | new | `Relation.says`; per-purpose option `order` | S–M | At most one line per card; word budgets; "See it" restores the view |
| U7 | Sentences, the IDA paragraph, supplement S1, provenance `threads` block; checklist rules read the anchors; the PROBAST/ROBINS evidence table | partial | `provenance.py`, `export/bundle.py`, `export/checklists.py`, `export/replay.py` | M | The replay re-measures every notice and matches; checklist items flip with thread state |
| U8 | The open-threads card at the lock and at the seal | partial | `plan_lock.py`, `open_seal` | S | `leave_open` writes the limitation; a T1 thread cannot be left open |
| U9 | The honest score ladder: process-only baselines, leave-one-group-out, grouped minus random, deployment-noise re-score; one ladder showing only deltas above a declared margin | partial | `baseline.py`, `validation.py`, `design_cv.py`, `folds.py`, `decision_curve.py` | M | Shows the leaky_sepsis, clinical_labs and genomics deltas |
| U10 | The 17 family sentinels with negative controls; the sentinel-proof test | new (pieces exist) | assay drift, redundancy, the structure stage, groupings | M | Lens "other" still catches the S1/S2/S3 positives |
| U11 | Missing consumers, each through §13: competing risks, time zero, IPCW, LOBO, the specification curve, parallel analysis/MAP, deployment re-score | new | the method-contract registry | L, staged by the prototype | Each has a contract, a sentence and a chain test |
| U12 | Fixtures (§6), including derived and committed NHANES fixtures (`_tt_tmp_nhanes.csv` is untracked) | partial | `sample_data/`, `make_*_siblings.py` | S each | Every prototype thread has one fixture where it fires and one where it stays silent |
| U13 | The coverage registry and its test (§3.5), with PROBAST, ROBINS-E, ROBINS-I V2, the leakage types and the cells as data | new | `export/data/tripod_ai.json`, `strobe_nut.json` | S | Red on any unmapped item or orphan thread |
| U14 | Frontend, calm: the First look guide, card-family rows, the context line, the manuscript mark, the open-threads card, the supplement view | new | the brief's First look; `PreviewGrid`; Record | M–L | The five-second test (FOUNDATION §1); the pedagogy reviewer passes |

**Order.**
1. U1, U2, U3.
2. U4, U5, U6, with the two threads of phase 0.
3. U7, U8.
4. U9, U10.
5. U13 runs alongside from the start.
6. U11 lands only as a prototype thread needs it.

**Compute.** These runs are long and CPU-heavy on the machine beside the bed, so they are scheduled with Nolan and never run on hover:
- whole-pipeline label permutation (the noise ceiling);
- in-fold ComBat with leave-one-batch-out.

Until a slot is agreed, use fewer permutations with a Monte Carlo interval.

---

## 6 · Prototype plan: prove the magic end to end

**A thread is proven when all of these hold:**
1. Its detector reproduces the catalog's numbers on the positive fixture and stays silent (or stated) on the negative one.
2. Its reading is asked on its card family, with the evidence shown, and every alternative changes downstream behavior (U3).
3. At least two later stages show its context line, and their option order or labels change with the answer.
4. The failure's two numbers are computed once and shown.
5. The methods sentence cites the evidence; supplement S1 has its row; at least one checklist item is answered from it.
6. The replay reproduces the notice and the decision.
7. Under the other purpose, it behaves as its feeds declare.
8. The reference journey's asked count is reported.
9. Nolan drives it in the calm design.

**Phase 0: the contract slice.** One thread per purpose.

| Thread | Fixture: fires · silent or contrast | What it proves |
|---|---|---|
| diet-day-to-day-variance, with diet-too-noisy-to-correct (K3, inference) | `dietary_recalls.csv` · `nhanes_dietary.csv` (one day: dormant, so an "invisible exposure" limitation instead) | **An O0 noticing changes a plan.** "About three-quarters of the spread in one day's energy is day-to-day noise; a two-day mean carries about 39% of the true slope on the log scale." It feeds the measurement-error card, the regression-calibration secondary analysis (the `calibration` stage exists), and the explanation label "uninformative null" for sodium (r 0.01) and fiber (−0.11). It answers STROBE-nut nut-12.3 and nut-8.6. Under prediction it becomes the deployment-instrument question. |
| clin-predictor-after-prediction-time (S2, prediction) | `leaky_sepsis.csv` · `clinical_risk.csv` (contrast: the moment of use decides) | **The audit sentinel, the timing reading, the honest score from 1.00 to 0.81, and `los_days`** (AUC 0.51, but known at discharge). It answers TRIPOD+AI 9b and PROBAST 2.3. |

**Phase 1: one thread per remaining lens, plus the sentinel proof.**

| Lens | Thread | Fixture | What it proves |
|---|---|---|---|
| Metabolomics | metab-qc-drift with metab-qc-design-insufficient (K3) | `metabolomics_untargeted.csv` (4 QCs per batch; 18 injections after their batch's last QC) | Reference-rows scope; QC-RLSC "Not available yet" with an exit; the feature flow; the explanation trust row flags top features from the end of the run |
| Genomics | genomics-batch-aligned-with-case (S1, an O2 guard) | A confounded sibling made by `make_genomics_siblings.py`, which fires: random folds 0.91, leave-one-batch-out with in-fold ComBat 0.61 (scheduled compute). `genomics_expression.csv` is balanced, so it is stated: "Batch is 97% of PC1, but every batch holds 10 cases and 10 controls, so it costs you power, not truth." | The O2 legality, the change of validation design, and the Angles card "What drives your leading components" |
| Survey | survey-sentinel-codes with survey-reverse-keying (K1) | `survey_sentinels.csv`: item_14's 33 nines move its mean from 3.05 to 3.70; under the codebook's key, α goes from 0.80 to 0.94 · `survey_instrument.csv` (no sentinels: stated) | The codes question, then the scale form, then reliability and attenuation (the `scales` stage exists). The key comes from the instrument, never from the correlations. |
| Shared | **Sentinel proof** | `leaky_sepsis.csv` and the metabolomics run-order sibling, with lens "Something else" | The S2/S3 and S1 sentinels raise generic threads without any pack. This is the direct answer to "what will go un-missed". |

**Phase 2: a second thread per lens, from a different family.**

| Lens | Thread | Fixture | What it proves |
|---|---|---|---|
| Dietary | diet-zero-is-a-day with diet-mass-at-zero (K2) | `dietary_recalls.csv` (`alcohol_pct_kcal`) | §1.6 in full: two-part form, the never-or-former row, nut-11 and nut-14 |
| Clinical | clin-mixed-units, which dissolves clin-site-differences (a K1 → S1 chain) | `clinical_labs.csv`: SOUTH's 102 low glucose values; C 1.00, then 0.53 without glucose | A value-meaning answer removes a shortcut, and the chain is drawn |
| Metabolomics | metab-run-order-aligned-with-outcome (S1) | The sibling fires; `metabolomics_untargeted.csv` stays silent (AUC 0.545, one methods line) | A clean check in the paper, and the process-only baseline on the ladder |
| Genomics | genomics-outlier-sample (K3) | `genomics_expression.csv` (GS031, z = −3.8) | A row-set subject; asked once, in the Flow |
| Survey | survey-skip-pattern-blanks with shared-skip-pattern-structural-zero (K2) | A new committed, derived NHANES fixture: `meds_chol` blank for the never-told, and complete cases keeping 2,943 of 21,348 rows | One Flow per module; the fill by rule; no MI inventing drinking for abstainers |

**After phase 2, the families roll out in order of how often they fire in the reference journeys:** K1, K2, S4, S5, K3, K4 and K6. New fixtures each catalog already asks for:
- an outcome defined by predictors;
- time zero;
- encounters with a patient-level signal;
- a derived fixture of NHANES treated blood pressure.

---

## 7 · Questions for Nolan (product and purpose only), ruled on 2026-10-06

1. **What happens to an open thread at the lock, or before the seal?**
   - **My recommendation:** "Leave as a limitation" is always allowed except for T1 threads. It writes the limitation sentence, and the lock records which threads were left open.
   - **The alternative:** every thread that feeds the plan must be decided or dismissed before estimates appear.
   - **The trade-off:** a tighter leash and a cleaner paper, against a stall at the most exciting moment.
   - **Ruled: the alternative.** Every open thread that feeds the plan must be decided or dismissed before estimates appear, and under prediction before the seal opens. The lifecycle (§2) and the lock change to match. The open list before the lock becomes a list the user clears.
2. **Should clean checks appear in the paper?**
   - **My recommendation:** the supplement lists what was checked and found nothing beyond its reference. An example: "run order and responder were balanced, AUC 0.545." The methods text stays short.
   - **The alternatives:** list only noticings that fired, or put clean checks in the methods text too.
   - **Ruled: as recommended,** in the supplement.
3. **Are noticings something the user sees by name?**
   - **My recommendation:** name them "noticings" in exactly two places: First look's "Worth a look", and the open list before the lock. Elsewhere they appear only as context lines and "noticed" links, never as a panel or a counter.
   - **The alternative:** keep them unnamed infrastructure. That is calmer, but the user cannot point at "the thing the app told me".
   - **Ruled: as recommended,** named in those two places.

---

## Appendix A · The 333 catalog threads by family

- **S1** (16)
  - *dietary:* assessment-batch, season-of-assessment, recall-process-shortcuts
  - *clinical:* site-differences
  - *metabolomics:* run-order-aligned-with-outcome, preanalytical-aligned
  - *genomics:* batch-aligned-with-case, depth-tracks-case, sample-quality-drives-components, unrecorded-structure, draw-conditions, mixed-tissue-types
  - *shared:* season, preanalytical-handling, omics-sample-total-tracks-outcome, process-aligned-with-outcome
- **S2** (7)
  - *clinical:* predictor-after-prediction-time, preclinical-disease-before-event
  - *metabolomics:* treatment-marker, sample-timing-vs-diagnosis
  - *genomics:* shared-sampled-after-diagnosis
  - *survey:* items-downstream-of-the-outcome
  - *shared:* predictor-after-the-index
- **S3** (8)
  - *dietary:* ratio-features-share-the-outcome
  - *clinical:* outcome-defined-by-predictors
  - *metabolomics:* outcome-defined-by-a-feature
  - *genomics:* features-preselected-on-outcome, outcome-defined-from-features, polygenic-score-provenance
  - *survey:* item-overlap-with-outcome
  - *shared:* outcome-proxy
- **S4** (22)
  - *dietary:* food-rows-sum-to-days, household-level-intake, meal-level-rows
  - *clinical:* encounters-not-patients
  - *metabolomics:* technical-replicates, matched-sets, person-fingerprint
  - *genomics:* paired-or-repeated-samples, near-identical-samples, cells-are-not-replicates, relatedness
  - *survey:* interviewer-clustering, shared-ema-within-between
  - *shared:* unit-repeats, grouping-above-the-person, exposure-varies-between-groups, repeats-or-time-points, imputed-copies, one-value-per-unit, duplicate-records, join-changes-the-row, spatial-dependence
- **S5** (24)
  - *dietary:* implausible-reporters, implausible-day-vs-person, dietary-weight-tier
  - *clinical:* prevalent-cases-at-baseline, loss-to-follow-up, outcome-dependent-sampling, index-event-selection, partial-verification
  - *metabolomics:* reference-rows
  - *genomics:* excluded-libraries-by-group
  - *survey:* population-or-sample, nonprobability-sample, informative-design, subsample-weight, exclusion-is-a-domain, shared-panel-attrition
  - *shared:* linkage-quality, survey-population-design, survey-which-weight, case-control-sampling, selection-flow, non-data-rows, selection-on-a-consequence, informative-dropout
- **S6** (6)
  - *dietary:* delivered-already-adjusted
  - *metabolomics:* merge-artifacts, already-transformed, pre-corrected-batches
  - *genomics:* normalized-across-all-samples
  - *shared:* flags-describe-the-data
- **S7** (9)
  - *dietary:* deployment-measurement-heterogeneity
  - *clinical:* coding-system-transition, assay-change, calendar-drift
  - *metabolomics:* cross-batch-transport
  - *genomics:* new-site-new-platform
  - *survey:* instrument-changed-across-cycles
  - *shared:* case-mix-shift, method-change
- **K1** (49)
  - *dietary:* energy-unit-and-days, kcal-per-unit-of-each-source, energy-is-intake-not-expenditure, name-is-not-a-nutrient, supplements-in-totals, ffq-frequency-not-a-scale, nutrient-equivalents
  - *clinical:* censored-lab-results, mixed-units, default-value-entries, digit-preference, impossible-vs-extreme, implausible-trajectory
  - *metabolomics:* internal-standards, subdomain, variance-structure, limit-flags-in-cells
  - *genomics:* what-the-numbers-are, sample-sheet-alignment, sex-from-expression, matrix-orientation, gene-identifiers-damaged, qpcr-ct-values, methylation-beta, allele-harmonization, mean-variance-scaling
  - *survey:* reverse-keying, sentinel-codes, dont-know-meaning, top-codes, ordinal-predictor-spacing, heaped-answers, recall-period-mismatch
  - *shared:* identifier-as-predictor, codes-or-amounts, missing-value-codes, sas-transport-zeros, units-by-magnitude, mixed-units, codebook-contradicted, sex-coding, impossible-vs-extreme, heaping-digit-preference, ambiguous-dates, transposed-assay-table, lens-contradicts-table, omics-value-scale, number-format, top-coded
- **K2** (22)
  - *dietary:* mass-at-zero, zero-is-a-day, ffq-blank-means-never
  - *clinical:* informative-test-ordering, coded-history-absence, informative-visit-process
  - *metabolomics:* zeros-meaning, left-censored-nondetects, missing-by-batch, group-specific-detection
  - *genomics:* array-detection-floor
  - *survey:* item-nonresponse, breakoff, skip-pattern-blanks, check-all-that-apply, imputation-carries-the-design
  - *shared:* left-censoring, skip-pattern-structural-zero, informative-missingness, missingness-mechanism, missing-in-a-derived-term, zero-mass-predictor
- **K3** (41)
  - *dietary:* day-to-day-variance, too-noisy-to-correct, usual-intake-distribution, misreporting-tracks-body-size, recall-nuisance-effects, ffq-recall-substudy, recovery-biomarker, instrument-kind, recall-completeness, error-grows-with-intake, few-replicate-persons, intake-as-outcome, single-baseline-long-follow-up
  - *clinical:* single-reading-dilution, fasting-status, season-of-draw, self-report-vs-measured, specimen-quality-flags
  - *metabolomics:* qc-drift, qc-design-insufficient, qc-representativeness, feature-technical-reliability, blank-contamination, dilution-linearity, failed-injection, dilution, low-biological-icc
  - *genomics:* outlier-sample, genotype-qc, imputed-dosages
  - *survey:* careless-responding, acquiescence, reliability-attenuation, repeat-or-reference-measurement, dif, floor-ceiling, mode-proxy-nonequivalence
  - *shared:* replicate-reliability, omics-outlier-sample, validation-subsample, proxy-respondent
- **K4** (25)
  - *dietary:* energy-related-outcome, changed-because-of-disease, repeated-ffq-cumulative, food-and-its-nutrients, healthy-lifestyle-cluster
  - *clinical:* treated-measurements, confounding-by-indication, treatment-during-follow-up, biomarker-on-the-path, diagnosis-changes-exposure
  - *metabolomics:* normalizer-carries-biology, clinical-factors, metabolome-role
  - *genomics:* genes-are-the-outcomes, cell-mix-drives-signal, ancestry-structure, genotype-shapes-the-diet, exposure-written-in-the-profile
  - *survey:* same-sitting-reverse-causation
  - *shared:* mediator-or-confounder, time-varying-feedback, treated-values, treatment-paradox, prevalent-disease-reverse-causation, missing-confounder
- **K5** (23)
  - *dietary:* energy-budget-gap, energy-carries-the-nutrient, nested-parts, compositional-shares, median-fill-breaks-the-residual, patterns, quality-index-is-a-density
  - *clinical:* derived-clinical-variables
  - *metabolomics:* one-compound-many-features, ratio-features, feature-flow
  - *genomics:* few-transcripts-take-the-reads, many-probes-one-gene, low-expression-filter
  - *survey:* dimensionality, wording-method-factor, formative-or-reflective, items-or-score
  - *shared:* near-constant-predictors, collinear-predictors, derived-predictors, coarsened-copy, compositional-parts
- **K6** (27)
  - *clinical:* diagnosed-not-diseased, unequal-follow-up, diagnostic-or-prognostic, time-zero, competing-death, age-time-scale, randomized-arm, state-or-event, regression-to-the-mean, crossover-design, intercurrent-events, event-found-at-visits
  - *survey:* ordinal-outcome, self-reported-diagnosis-outcome
  - *shared:* outcome-ascertainment, follow-up-varies, competing-death, outcome-scale-skew, ordered-or-multiclass-outcome, common-outcome-measure, zero-mass-outcome, count-outcome-exposure-time, bounded-outcome, baseline-outcome-present, time-zero, randomized-arm, entry-on-a-high-reading
- **K7** (12)
  - *dietary:* dri-life-stage, former-consumers-in-reference, quantiles-sort-by-sex-and-size, meaningful-increment
  - *clinical:* inflammation-adjustment, physiological-state, pediatric-growth-scale, population-specific-cutoffs
  - *survey:* instrument-published-scoring, score-direction, former-users-in-the-reference
  - *shared:* children-among-adults
- **E1** (18)
  - *dietary:* substitution-support, rare-events-sparse-categories
  - *clinical:* sparse-levels-separation, rare-outcome-threshold, contraindication-positivity
  - *metabolomics:* detectable-effect
  - *genomics:* more-genes-than-samples
  - *survey:* small-sample-psychometrics, weight-effective-n, design-df-budget, estimate-reliability
  - *shared:* lonely-psu, events-not-rows, predictors-outnumber-rows, sparse-levels, tails-and-support, interaction-support, positivity-overlap
- **E2** (16)
  - *dietary:* nutrient-wide-scan
  - *clinical:* incremental-value
  - *metabolomics:* correction-honest-check, enrichment-background, wide-noise-ceiling, panel-instability
  - *genomics:* signature-instability, pvalue-distribution, enrichment-background, shared-score-within-noise
  - *survey:* question-wide-scan
  - *shared:* exposure-family, curvature-seen-in-explore, subgroup-seen-in-explore, model-no-better-than-baseline, selection-unstable
- **E3** (8)
  - *dietary:* invisible-exposure
  - *clinical:* expected-directions
  - *metabolomics:* annotation-confidence
  - *genomics:* coexpression-modules
  - *survey:* common-method, explanations-describe-the-sample
  - *shared:* quality-differs-by-group, primary-model-diagnostics

## Appendix B · Specializations that merge (the shared thread first)

| Shared thread | Lens threads that specialize it |
|---|---|
| predictor-after-the-index | clin-predictor-after-prediction-time |
| outcome-proxy | clin-outcome-defined-by-predictors, metab-outcome-defined-by-a-feature, genomics-outcome-defined-from-features |
| skip-pattern-structural-zero | survey-skip-pattern-blanks |
| mixed-units | clin-mixed-units |
| impossible-vs-extreme | clin-impossible-vs-extreme |
| treated-values | clin-treated-measurements |
| competing-death | clin-competing-death |
| time-zero | clin-time-zero |
| follow-up-varies | clin-unequal-follow-up |
| randomized-arm | clin-randomized-arm |
| entry-on-a-high-reading | clin-regression-to-the-mean |
| prevalent-disease-reverse-causation | clin-diagnosis-changes-exposure, diet-changed-because-of-disease, survey-same-sitting-reverse-causation |
| (no shared thread) | diet-former-consumers-in-reference merges with survey-former-users-in-the-reference |
| heaping-digit-preference | clin-digit-preference, survey-heaped-answers |
| left-censoring | clin-censored-lab-results, metab-left-censored-nondetects, metab-limit-flags-in-cells |
| case-control-sampling | clin-outcome-dependent-sampling |
| informative-missingness | clin-informative-test-ordering |
| replicate-reliability | clin-single-reading-dilution |
| omics-sample-total-tracks-outcome | genomics-depth-tracks-case, metab-dilution |
| omics-outlier-sample | genomics-outlier-sample |
| process-aligned-with-outcome | genomics-batch-aligned-with-case, metab-run-order-aligned-with-outcome, diet-assessment-batch |
| method-change | clin-assay-change, survey-instrument-changed-across-cycles |
| season | clin-season-of-draw, diet-season-of-assessment |
| preanalytical-handling | metab-preanalytical-aligned, genomics-draw-conditions, clin-fasting-status, clin-specimen-quality-flags |
| derived-predictors | clin-derived-clinical-variables |
| mediator-or-confounder | clin-biomarker-on-the-path, diet-energy-related-outcome |
| treatment-paradox | clin-treatment-during-follow-up |
| positivity-overlap | clin-contraindication-positivity |
| sparse-levels | clin-sparse-levels-separation, diet-rare-events-sparse-categories |
| unit-repeats | clin-encounters-not-patients, metab-technical-replicates, genomics-paired-or-repeated-samples |
| grouping-above-the-person | clin-site-differences, survey-interviewer-clustering |
| selection-on-a-consequence | clin-index-event-selection |
| case-mix-shift | clin-calendar-drift |
| compositional-parts | diet-compositional-shares |
| survey-population-design | survey-population-or-sample |
| survey-which-weight | survey-subsample-weight, diet-dietary-weight-tier |
| missing-value-codes | survey-sentinel-codes |
| top-coded | survey-top-codes |
| zero-mass-predictor | diet-mass-at-zero, diet-zero-is-a-day |
| exposure-family | diet-nutrient-wide-scan, survey-question-wide-scan |
| selection-unstable | metab-panel-instability, genomics-signature-instability |
| score-within-noise | metab-wide-noise-ceiling |
| validation-subsample | diet-ffq-recall-substudy, diet-recovery-biomarker |
| proxy-respondent | survey-mode-proxy-nonequivalence |
| children-among-adults | clin-pediatric-growth-scale |
| informative-dropout | clin-loss-to-follow-up, shared-panel-attrition |
| outcome-ascertainment | clin-diagnosed-not-diseased, survey-self-reported-diagnosis-outcome |

## Sources

**Repo**, read only:
- `docs/turbotab-next/BLUEPRINT.md`: §11 to §11.4, §12, §13, §14 to §14.3.
- `docs/turbotab-next/MODELING_SEQUENCE.md` §1 and §1.1.
- `.worktrees/calm/docs/turbotab-next/calm/FOUNDATION.md` §1 to §5.
- Engine modules: `turbotab/core/readings.py` (Reading 103, KindRule 375, CONSUMERS 3226), `contracts.py`, `stages/findings.py`, `stages/explore.py`, `plan_lock.py`, `interview.py` (QUESTION_KEYS 99), `ask.py`, `coach.py`, `evidence.py`, `routing_leash.py`, `export/checklists.py`, `export/data/tripod_ai.json`, `export/data/strobe_nut.json`.
- `turbotab/packs.py`: Pack 5111, PACKS 5217.
- `turbotab/test_a_deferred_noticing_comes_back_where_it_said.py`: the legacy precedent of PRODUCT_VISION §04.
- `turbotab/sample_data/*`.

**The workflow:**
- The First look design brief (wf_1b3d0fc3).
- The verified rule table of what exploration may decide (wf_e039e6d2).
- The six final catalogs supplied in the task.

**Literature:**
- Wolff RF et al., PROBAST, Ann Intern Med 2019;170:51–58, and Moons KGM et al., explanation and elaboration, Ann Intern Med 2019;170:W1–W33.
- Moons KGM et al., PROBAST+AI, BMJ 2025. The item list is to be verified before its rows enter the registry.
- Higgins JPT et al., ROBINS-E, Environ Int 2024.
- ROBINS-I V2 (Sterne et al., 2024–25 release).
- Kapoor S, Narayanan A, Leakage and the reproducibility crisis in ML-based science, Patterns 2023;4:100804.
- Kaufman S et al., Leakage in data mining, ACM TKDD 2012.
- Collins GS et al., TRIPOD+AI, BMJ 2024;385:e078378.
- Lachat C et al., STROBE-nut, PLoS Med 2016;13:e1002036.
- Heinze G et al., Regression without regrets, BMC Med Res Methodol 2024;24:178.
- Rubin DB 2008.
- Gelman A, Loken E 2013.
- NCHS Series 2 No. 178, and Tooze JA 2006 (the episodic-food rule).
- Battelino T 2019 (CGM sufficiency).
- Weiss S 2017 (library size as a design-balance check).
