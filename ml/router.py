"""
Given a project and a record, decide which question comes next.

The differentiator, and the only component in this migration with no existing
implementation to be equivalent to. `TRANSITION_PLAN.md` §02.5 is the feasibility
verdict: the coach is *a pure annotator with one accidental act of control flow*
— it can **order** questions but cannot **gate** them, it never emits a `blocker`
severity, it has no confidence tier of its own, and 100% of its trigger logic
lives in `pages/`. So this is new construction, and the triggers are lifted here
from `pages/02_EDA.py::_auto_generate_insights` (inventoried in
`docs/turbotab/data/explore-triggers.json`).

**Decision B, and it binds everything below.**

> Skip a question only where a `high`-confidence finding makes it moot. Every
> skip is visible and reversible in the transcript. Nothing below `high` ever
> auto-advances.

Auto-advancing an interview *is* pre-selection (`PRODUCT_VISION.md` §07.1), so
the confidence tier that governs pre-selection governs skipping too. The rule is
enforced in one place — :func:`_skip_is_permitted` — and asserted, not trusted:
a skip without a high-confidence finding raises rather than being quietly
emitted.

**The routing constitution — three clauses, one per kind of question.**

* **Fact** — skippable at `high` confidence. The engine is certain and the
  transcript can state it: *"is this column categorical?"*
* **Choice** — always asked. Whether to apply a repair is the user's however
  confident the engine is, because applying without preview is the blind consent
  `PRODUCT_VISION.md` §04 rules out.
* **Consequence** — always **pushed, never offered**, and leaving the step with
  one unresolved is itself a recorded decision. A `blocker` is a finding that can
  make the whole analysis wrong; offering it beside a distribution gallery is
  not gating, it is decoration. Leakage is the canonical case — near-perfect
  discrimination from a column that encodes the outcome is how a prediction
  paper becomes wrong.

The third clause deliberately does **not** hard-refuse. The user may know the
flagged column is legitimate, and a tool that blocks on a correct analysis is a
tool people route around. What it refuses is *silence*: exiting past a blocker
writes an acknowledgment into the record, so the manuscript can carry it as a
stated limitation rather than omitting it. Overriding a blocker is allowed;
overriding one quietly is not.

**Determinism.** `plan()` is a pure function of (findings, detection, record).
Same inputs, same questions, same order — derivable from the record alone, with
no clock, no RNG and no session state.

Headless: no Streamlit, no project object, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Steps of the exploration phase, in order. The Router asks at the earliest step
# that can act on a question, which is what makes deferral meaningful: a
# deferred item has somewhere later to go.
STEPS: Sequence[str] = ("data", "explore", "features", "preprocess")

# Every step a deferral can name, built or not, with the words the interface
# uses for it. A deferral affordance says "Decide at Preprocess" rather than
# "Remind me later" — the API has always required a target_step, so the
# information existed at click time and the button simply did not say it
# (GUIDED-008). Naming a step that is not built yet is correct: it is where the
# item resurfaces, and pretending otherwise would be the vaguer answer.
STEP_LABELS: Dict[str, str] = {
    "data": "Data",
    "explore": "Explore",
    "preprocess": "Preprocess",
    "features": "Features",
    "train": "Train",
}

# Severity order for ranking. The engine's own vocabulary; no new tiers.
_SEVERITY_RANK = {"critical": 0, "blocker": 0, "warning": 1, "caution": 2, "info": 3,
                  "opportunity": 4}
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


class RouterError(Exception):
    """The Router was asked to do something the governing rules forbid."""


@dataclass
class Question:
    """One thing the interview puts to the user, or explains why it did not."""

    key: str
    title: str
    why: str
    step: str
    kind: str                       # target | task_type | repair | explore
    # push | pull. "Push the notable, pull the rest" (`PRODUCT_VISION.md` §04):
    # a pushed item is asked and awaits an answer; a pull affordance is offered
    # beside it and costs nothing to ignore. The interview's question count is
    # pushed items only — otherwise offering a distribution gallery would read
    # as the interview becoming more talkative when it is doing the opposite.
    mode: str = "push"
    triggering_finding: Optional[str] = None
    confidence: Optional[str] = None
    severity: Optional[str] = None
    options: List[str] = field(default_factory=list)
    # The VALUE each option submits, parallel to `options`. Empty means the
    # label is the value, which is true of every question written before the
    # lens — and the lens is why this exists: its labels are prose
    # ("Metabolomics or proteomics") and its values are keys ("metabolomics"),
    # so a page rendering labels had nothing to submit. `DRIVE-001` is what that
    # costs when a question is servable and unanswerable.
    option_values: List[str] = field(default_factory=list)
    # One sentence per option, parallel to `options`, saying what THAT answer
    # leads the app to look for. Empty means the question-level effect sentence
    # is the whole truth, which is the case for every question whose options are
    # variants of one operation. `GUIDED-039`: the lens is the case where it is
    # not — its six options select five different packs, and one shared hover
    # said nothing about any of them.
    option_notes: List[str] = field(default_factory=list)
    multi_select: bool = False
    # For a multi-select: how many answers the record requires, and the reason
    # in the words of whatever enforces it.
    #
    # `GUIDED-038`. The two multi-selects DISAGREE, which is why this cannot be
    # a property of the interface: reverse-coding's zero is a real answer — *"no
    # items are reverse-coded"*, the recorded-absence rule — and the lens's zero
    # is refused by `packs.normalize`, because an empty selection is
    # indistinguishable from never having asked. A page that hardcoded either
    # would hold a second copy of a rule that already exists, and the copy is
    # what drifts.
    min_selections: int = 0
    min_reason: Optional[str] = None
    # Who consumes this answer, and what for. DESIGN_LANGUAGE §09: "Every FACT
    # carries a 'Why we ask' disclosure that names who consumes the answer and
    # what for. A FACT that cannot state its consumer is a question we have no
    # right to ask." Held on the Question rather than written into the page, so
    # `audit()` can refuse a plan that asks one without it.
    consumer: Optional[str] = None
    # The constitution clause that REQUIRES this question, when one does.
    #
    # A fourth origin, and naming it is the L16 ruling on `GUIDED-018`. The
    # measurement harness assumed every legitimate question originates from a
    # finding: `irrelevant_questions = asked − required_decisions`, and
    # `required_decisions` is built from what the engine found. Clause §02's
    # grain question originates somewhere else — it is asked because the app
    # CANNOT KNOW, and asking it only when a detector fires is precisely
    # `IMPORT-020` and `IMPORT-022`. So it cites no finding, has no inventory
    # key by design, and scored as noise.
    #
    # The fix is to name the category, not to add a key per clause: eligibility
    # (§04) is the second one, missingness routing (§07) is likely the third,
    # and a denominator that gains an entry per constitutional question is a
    # denominator that moves every loop.
    clause: Optional[str] = None

    # asked | skipped | deferred — every one of which is visible in the
    # transcript. There is no fourth state, because a question that is neither
    # asked nor accounted for is the silence this whole design removes.
    status: str = "asked"
    skip_reason: Optional[str] = None
    defer_target: Optional[str] = None
    deferred_from: Optional[str] = None

    @property
    def is_findings_driven(self) -> bool:
        return bool(self.triggering_finding)

    @property
    def is_visible(self) -> bool:
        """A skip or deferral must carry its reason to count as visible."""
        if self.status == "skipped":
            return bool(self.skip_reason)
        if self.status == "deferred":
            return bool(self.defer_target)
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "title": self.title, "why": self.why,
            "step": self.step, "kind": self.kind, "mode": self.mode,
            "triggering_finding": self.triggering_finding,
            "confidence": self.confidence, "severity": self.severity,
            "status": self.status, "skip_reason": self.skip_reason,
            "defer_target": self.defer_target, "deferred_from": self.deferred_from,
            "options": list(self.options),
            "option_values": list(self.option_values or self.options),
            "option_notes": list(self.option_notes),
            "multi_select": bool(self.multi_select),
            "min_selections": int(self.min_selections),
            "min_reason": self.min_reason,
            "consumer": self.consumer,
            "clause": self.clause,
        }


# Severities that are questions of consequence. `blocker` is the engine's own
# word, and `ARCHITECTURE.md` records that only `pages/02` ever emitted it — the
# coach cannot gate. Now the Router can.
BLOCKER_SEVERITIES = frozenset({"blocker"})

# The three types of DESIGN_LANGUAGE §09, by the `kind` this module already
# used. FACT is a question of fact — the lightest object on screen, answered;
# CHOICE is a repair, decided; CONSEQUENCE is a blocker, resolved or attested.
# Named here so the audit rules can speak in the design language's vocabulary
# rather than re-listing kinds at each site.
FACT_KINDS = frozenset({"target", "task_type", "grain", "eligibility",
                        "missingness", "lens", "reverse_coding",
                        # Questions 4, 5 and 7. Each is a question about what
                        # the data IS, and each names what reads its answer.
                        "repeat_kind", "unit_of_analysis",
                        "temporal_prediction"})
# `preparation_mode` is deliberately a CHOICE and not a FACT. The engine has a
# recommendation — per-model, because a model handicapped by preparation it does
# not suit is not informative either — and a recommendation is not certainty.
# What comparison you want to make is not a property of the data, so no
# confidence in the engine could make it skippable.
# `aggregation` is a CHOICE and not a FACT, deliberately. The engine has a
# recommendation and, for time points, an argument for having none — but which
# summary of a person's records answers your question is not a property of the
# table, so no confidence in the engine could make it skippable. It also
# REWRITES THE TABLE, which is the other half of what makes a choice a choice.
CHOICE_KINDS = frozenset({"repair", "preparation_mode", "aggregation"})
CONSEQUENCE_KINDS = frozenset({"blocker"})


def _skip_is_permitted(confidence: Optional[str], kind: str) -> bool:
    """Decision B, in one place.

    Two conditions, both necessary:

    * the finding is `high` confidence — anything less may not auto-advance,
      because auto-advancing is pre-selection;
    * the question is one of *fact*, not of *choice*. A repair is a choice no
      matter how certain the engine is; skipping it would apply a change the
      user never saw.

    **`missingness` was added at L20, and it is the one tier change this rule
    has taken since Decision B was written**, so the argument is recorded here
    rather than in a commit message.

    The mechanism question — *"could a blank here mean something?"* — is a
    question of FACT, and until a lens is set the app genuinely cannot know the
    answer, which is why it was asked unconditionally. A pack prior at `derived`
    confidence changes that for the columns its own detector named: on an
    untargeted panel, missingness tracking abundance rank at rho = -0.99 IS the
    engine being certain, and `DOMAIN_PACKS.md` §02 shows the intended result in
    as many words — with a metabolomics lens, *"how should missing values be
    filled?"* leaves the question list and becomes a rendered skip carrying its
    reason.

    Three things keep this inside Decision B rather than beside it:

    * **`derived` only.** `convention` and `offered` priors never reach here, so
      the clinical pack's *not ordered* prior informs the question and does not
      remove it.
    * **Column by column.** The skip is granted per column, by the columns the
      detector named — never to the table (`GUIDED-027`).
    * **Visible and reversible.** The skip carries the pack's own reason and the
      reopen affordance, which is what `audit()` demands of every skip.
    """
    return confidence == "high" and kind in ("task_type", "missingness")


# ─────────────────────────────────────────────────────────────────────────────
# The plan
# ─────────────────────────────────────────────────────────────────────────────

def palette(recommendations: Sequence[Any], step: str = "explore") -> List[Question]:
    """The pull side: what else the user may look at, offered not asked.

    Built from `ml.eda_recommender.recommend_eda`, which is already engine code
    and already ranks its cards — the page only rendered them. Each is returned
    with ``mode="pull"``, so it is present in the interface and absent from the
    question count.

    This is the second half of *"push the notable, pull the rest"*. Without it
    the interview only ever shows what it found, which is the wall-of-plots
    problem inverted: nothing to explore unless something was wrong.
    """
    out: List[Question] = []
    for rec in recommendations:
        rid = getattr(rec, "id", None) or (rec.get("id") if isinstance(rec, dict) else None)
        if not rid:
            continue
        if isinstance(rec, dict):
            def get(k, d=None, _r=rec):
                return _r.get(k, d)
        else:
            def get(k, d=None, _r=rec):
                return getattr(_r, k, d)
        if get("enabled", True) is False:
            continue
        why = get("why") or []
        out.append(Question(
            key=f"look::{rid}", kind="explore", step=step, mode="pull",
            title=get("title") or rid,
            why="; ".join(why) if isinstance(why, (list, tuple)) else str(why or ""),
            options=["show me"]))
    return out


def blockers(signals: Any, step: str = "explore") -> List[Question]:
    """Questions of consequence, from the signals that carry `blocker` severity.

    Today that is leakage. `compute_dataset_signals` reports
    `leakage_candidate_cols` for anything correlating above 0.95 with the
    target, which `pages/02` raised as a blocker insight and Guided previously
    only *offered* as a palette card.

    Pushed, one per column, each citing the column that raised it.
    """
    out: List[Question] = []
    cols = list(getattr(signals, "leakage_candidate_cols", None) or [])
    flags = list(getattr(signals, "leakage_flags", None) or [])
    for col in cols:
        out.append(Question(
            key=f"blocker::leakage::{col}", kind="blocker", step=step, mode="push",
            severity="blocker", confidence="high",
            title=f"Does '{col}' encode the outcome?",
            why=("It correlates above 0.95 with the target. If it was recorded "
                 "after the outcome was known, any model using it will look "
                 "near-perfect and predict nothing — the canonical way a "
                 "prediction paper becomes wrong. If it is a legitimate "
                 "measurement, say so and it will be carried as a stated "
                 "limitation. " + " ".join(flags)),
            triggering_finding=f"eda_leakage_{col}",
            options=["drop it", "it is legitimate — record why", "show me the correlation"]))
    return out


def plan(
    findings: Sequence[Dict[str, Any]],
    *,
    target: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None,
    step: str = "data",
    deferred: Optional[Dict[str, str]] = None,
    answered: Sequence[str] = (),
    recommendations: Sequence[Any] = (),
    signals: Any = None,
    missing_columns: Sequence[str] = (),
    lens_block: Optional[Dict[str, Any]] = None,
    repeats: Optional[Dict[str, Any]] = None,
    missingness_priors: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    missingness_groups: Optional[List[Dict[str, Any]]] = None,
    missingness_exceptions: Optional[Dict[str, Dict[str, Any]]] = None,
    missingness_settled: Optional[List[Dict[str, Any]]] = None,
) -> List[Question]:
    """Every question this step asks, in order, derived from the record.

    `deferred` maps a question key to the step it was deferred *to*; a question
    deferred to this step reappears here, which is the whole promise of deferral
    as a first-class disposition.

    `answered` is the keys already settled — the record, replayed.

    `lens_block` is the survey pack's detector result, passed in rather than
    computed: this module is headless and takes no dataframe, which is what
    keeps `plan()` a pure function of the record. `None` means either no survey
    lens or no shared response scale, and both mean the same thing here — the
    reverse-coding question is not in the plan.
    """
    if step not in STEPS:
        raise RouterError(f"Unknown step {step!r}; expected one of {list(STEPS)}.")
    deferred = dict(deferred or {})
    answered = set(answered)
    out: List[Question] = []

    # ── the lens, before everything, because the DIAGNOSIS is field-sensitive
    #    (`DOMAIN_PACKS.md` §01, clause `lockbox-01`). 400 columns across 80
    #    rows reads as malformed to a general-purpose import doctor and is the
    #    expected shape for an assay panel; setting the lens first turns a false
    #    alarm into a correct reading.
    #
    #    A FACT, and never skippable — `_skip_is_permitted` admits only
    #    `task_type`. The same reasoning as the grain: the app CANNOT KNOW, and
    #    asking only when a detector fires is how `IMPORT-020` happened. The
    #    heuristic is a suggestion and a contradiction detector, never the
    #    answer.
    if step == "data" and "state_lens" not in answered:
        from turbotab import packs as _packs
        spec = _packs.question()
        out.append(Question(
            key=spec["key"], kind="lens", step="data", clause=spec["clause"],
            title=spec["title"], why=spec["why"], consumer=spec["consumer"],
            multi_select=True,
            min_selections=spec["min_selections"],
            min_reason=spec["min_reason"],
            options=[o["label"] for o in spec["options"]],
            option_values=[o["key"] for o in spec["options"]],
            option_notes=[o["note"] for o in spec["options"]]))

    # ── the target, first among the questions about the analysis ────────────
    if step == "data" and "choose_target" not in answered:
        out.append(Question(
            key="choose_target", kind="target", step="data",
            title="What are you predicting?",
            why=("Pick the column your paper is about. Everything after this — "
                 "which findings matter, which models are viable, what the test "
                 "set is drawn against — follows from it."),
            consumer=(
                "`ml.triage.detect_task_type` reads this column to decide whether "
                "the problem is classification or regression; "
                "`ml.dataset_profile.compute_dataset_profile` uses it to compute "
                "class balance or target distribution, which is what the model "
                "coach ranks models against; and the lockbox draws the held-out "
                "test set stratified on it. Choosing a different column later "
                "recomputes all three and marks everything below stale."),
            options=["<column>"]))

    # ── the task type: a question of FACT, so skippable at high confidence ──
    if target and detection and "confirm_task_type" not in answered:
        conf = detection.get("confidence")
        q = Question(
            key="confirm_task_type", kind="task_type", step="data",
            title=f"Is {target} a {detection.get('detected')} problem?",
            why=" ".join(detection.get("reasons") or []),
            confidence=conf,
            consumer=(
                "The answer chooses the whole downstream vocabulary: which "
                "metrics are computed (AUC and calibration, or R² and residuals), "
                "which models `ml.model_registry` offers, and whether the split "
                "is stratified. Getting it wrong does not raise an error — it "
                "produces a complete set of numbers for a question you did not "
                "ask."),
            options=["classification", "regression"])
        if _skip_is_permitted(conf, "task_type"):
            q.status = "skipped"
            q.skip_reason = (
                f"The engine reads {target} as {detection.get('detected')} at high "
                f"confidence: {' '.join(detection.get('reasons') or [])} Stated in "
                "the transcript rather than asked — change it there if it is wrong.")
        out.append(q)

    # ── the grain, before the seal, because the seal cannot be drawn without
    #    it. Constitution §02: ASKED, never inferred. It is a FACT, so `audit()`
    #    demands it name its consumer — but it is NOT skippable at any
    #    confidence, because `_skip_is_permitted` admits only `task_type`. That
    #    is deliberate and is the whole clause: `IMPORT-020` and `IMPORT-022`
    #    are the app having inferred this and rendered the guess as a clean lock.
    if step == "data" and target and "state_grain" not in answered:
        out.append(Question(
            key="state_grain", kind="grain", step="data", clause="lockbox-02",
            title="Can one person appear in more than one row?",
            why=("This decides how your held-out rows are chosen. If the same "
                 "person lands on both sides, your held-out numbers will look "
                 "better than the model is."),
            consumer=(
                "The lockbox reads this to decide whether the held-out set is "
                "drawn by person or by row, and records it as the seal's stated "
                "basis. Multi-file assembly reads the same answer rather than "
                "asking again. Answering wrongly does not raise an error — it "
                "produces held-out numbers that are optimistic by an amount "
                "nothing on screen can show you."),
            options=["No, one row per person",
                     "Yes, people repeat",
                     "I'm not sure",
                     "My design isn't described here"],
            option_values=["one_row_per_person", "people_repeat", "not_sure",
                           "design_not_described"]))

    # ── questions 4 to 7, all of which fire only when people repeat ─────────
    #
    # `repeats` is the chain's state, resolved by the caller: this module is
    # headless and takes no dataframe, so the EVIDENCE for question 4 is read
    # where the frame is and passed in. `None` means the grain says people do
    # not repeat, and then none of the four is in the plan at all — which is
    # `OPENING_SEQUENCE.md` §02's whole claim, that the count tracks the shape
    # of the study.
    if step == "data" and target and repeats and "state_grain" in answered:
        chain = _repeat_chain(repeats, answered)
        out.extend(chain)

    # ── eligibility, between the grain and the seal. Constitution §01 fixes
    #    that position and §04 says what the question may show: the target's
    #    distribution is WITHHELD, because a criterion chosen from the histogram
    #    is data-driven cohort selection rather than an eligibility criterion.
    #    Asked after the grain because the two are independent and the sequence
    #    is the constitution's, not a preference.
    if (step == "data" and target and "state_grain" in answered
            and "state_eligibility" not in answered):
        from turbotab import eligibility as _elig
        spec = _elig.question()
        out.append(Question(
            key=spec["key"], kind="eligibility", step="data",
            clause=spec["clause"],
            title=spec["title"],
            why=spec["why"] + " " + spec["withheld"],
            consumer=spec["consumer"],
            options=list(spec["options"])))

    # ── reverse-coding: the ONE question a pack is allowed to add ───────────
    #
    # Guard #1 says a pack may not add interview components, and this is the
    # deliberate exception rather than a hole in it. Reverse-coding needs a
    # codebook the app does not have, so it cannot be a stated fact and cannot
    # be a default; it is a question, and it exists only where its own detector
    # fires. On any table without a shared declared response scale it is not in
    # the plan at all, which is what keeps guard #2 true.
    #
    # And it is NEVER inferred from item correlations. On a unidimensional
    # instrument that inference is right; on two subscales measuring opposing
    # constructs it is confidently wrong, and nothing in the numbers separates
    # the two. `survey_instrument.csv` is built so the wrong answer looks right.
    if (step == "data" and lens_block and "state_reverse_coding" not in answered):
        out.append(Question(
            key="state_reverse_coding", kind="reverse_coding", step="data",
            clause="lockbox-01",
            title="Are any of these items reverse-coded?",
            why=(f"{len(lens_block['columns']):,} columns share one "
                 f"{len(lens_block['scale'])}-point response scale. If some of "
                 f"them are worded so that agreeing means the opposite, they "
                 f"have to be flipped before the scale means anything."),
            consumer=(
                "Scoring reads this to decide which items to reverse before "
                "combining them, and the methods section carries the list. "
                "Nothing here is inferred: the app can see that some items "
                "correlate negatively with the rest, and that is the same "
                "evidence two subscales measuring opposing constructs produce. "
                "Answering wrongly does not raise an error — it produces a "
                "scale score that means nothing, with every downstream number "
                "computed from it."),
            options=list(lens_block["columns"]), multi_select=True))

    # ── the Features step, constitution §06 ─────────────────────────────────
    # Two questions, because the clause draws two different objects. Building
    # a derived column is a CHOICE the user makes and sees applied; choosing a
    # selection rule is a CHOICE whose EXECUTION is deferred. Collapsing them
    # into one "configure features" question would hide exactly the distinction
    # the clause exists to draw.
    if step == "features" and target and "choose_features" not in answered:
        out.append(Question(
            key="choose_features", kind="grain", step="features",
            title="Are there quantities your question is really about?",
            why=("A ratio or an interaction you already reason about clinically "
                 "is usually a better feature than the columns it came from. "
                 "Anything built here from one row at a time is applied now "
                 "and shown to you; anything that learns from the column's "
                 "distribution is recorded and fitted inside the training "
                 "folds instead."),
            consumer=(
                "Row-local columns are added to the working table immediately "
                "and every later step sees them, so `ml.dataset_profile` "
                "profiles them and the models receive them as ordinary "
                "columns. Distribution-dependent ones are stored as a spec "
                "that the per-model preprocessing pipeline fits inside each "
                "training fold — nothing is computed over the held-out rows. "
                "Adding or removing a column marks every downstream result "
                "stale."),
            options=["build a feature", "skip this step"]))

    if step == "features" and target and "choose_selection" not in answered:
        out.append(Question(
            key="choose_selection", kind="grain", step="features",
            title="Should the models be given every column, or a chosen subset?",
            why=("Narrowing to the strongest features can help a small study. "
                 "The catch is that choosing them using all your data lets the "
                 "held-out rows influence which columns exist — so the choice "
                 "is recorded now and made again inside each training fold."),
            consumer=(
                "The answer becomes a selection spec on the project, which the "
                "per-model pipeline reads and refits per training fold. It "
                "also becomes a sentence in the methods section naming the "
                "method and the timing. Nothing is selected at the moment you "
                "answer — a set chosen now would have been chosen with the "
                "held-out rows in view."),
            options=["every column", "a chosen subset"],
            option_values=["every_column", "chosen_subset"]))

    # ── Preprocess, constitution §07 ────────────────────────────────────────
    # One question per column with blanks, and the MECHANISM comes first: "could
    # a blank here mean something?" is asked before "how should it be filled?",
    # because the answer decides which strategies are even legitimate. Asking
    # them the other way round is how a column that carried information gets a
    # median written over it by a well-meaning default.
    #
    # A FACT that is never skippable, for the same reason as the grain: the app
    # cannot know, and `_skip_is_permitted` admits only `task_type`.
    # Model selection comes FIRST at this step: per-model preprocessing has
    # nothing to hang off until the user says what they intend to train, and
    # `PRODUCT_VISION.md` makes Train execution rather than choice. A CHOICE,
    # so never skippable at any confidence — the shelf's ORDER carries the
    # judgment and the selection stays the user's.
    if step == "preprocess" and target and "choose_models" not in answered:
        from turbotab import models as _models
        out.append(Question(
            key="choose_models", kind="repair", step="preprocess",
            title="Which models do you want to train?",
            why=(_models.SHELF_DISCLOSURE),
            options=["<model keys>"]))

    # Asked ONCE, after the models are chosen, because it is a property of the
    # COMPARISON rather than of any model in it.
    if (step == "preprocess" and target and "choose_models" in answered
            and "choose_preparation_mode" not in answered):
        out.append(Question(
            key="choose_preparation_mode", kind="preparation_mode",
            step="preprocess",
            title=("Should each model get the preparation it needs, or should "
                   "they all get the same preparation so the comparison is "
                   "about the models?"),
            why=("Per-model is the usual choice and what we recommend: a model "
                 "handicapped by preparation it does not suit is not "
                 "informative either. The cost is that a difference between "
                 "two models then reflects the model and its preparation "
                 "together — so if you pick it, that caveat is written into "
                 "your methods section automatically."),
            consumer=(
                "The answer decides how `turbotab.recipes` resolves each "
                "model's operations: per-model resolves each against its own "
                "capabilities, uniform resolves every model against the first "
                "selected model's settings. It also decides whether the "
                "comparison caveat is added to the manuscript — choosing "
                "per-model adds it, choosing uniform does not, because under "
                "uniform there is nothing to caveat."),
            options=["Each model gets the preparation it needs (recommended)",
                     "All models get the same preparation"]))

    if step == "preprocess" and target:
        from turbotab import missingness as _miss
        out.extend(_missingness_questions(
            missing_columns, missingness_priors, missingness_groups,
            missingness_exceptions, answered, _miss,
            settled=missingness_settled))

    # ── one question per repairable finding, ranked by the engine ──────────
    for f in _rank(findings):
        key = f"repair::{f['id']}"
        if key in answered:
            continue
        if not _is_repairable(f):
            continue

        home = _home_step(f)
        target_step = deferred.get(key, home)
        if target_step != step:
            # Not this step's business yet. Recorded as deferred so it is
            # visible, rather than silently absent.
            if home == step:
                q = _repair_question(f, step)
                q.status = "deferred"
                q.defer_target = target_step
                out.append(q)
            continue

        q = _repair_question(f, step)
        if key in deferred:
            q.deferred_from = "data"
        out.append(q)

    # Questions of consequence come FIRST when they exist: a blocker that is
    # third in a list of nine is a blocker in name only.
    if signals is not None:
        blocking = [q for q in blockers(signals, step=step)
                    if q.key not in answered]
        out = blocking + out

    # The pull palette sits beside the questions, never among them.
    if recommendations:
        out.extend(palette(recommendations, step=step))

    return out


def unresolved_blockers(questions: Sequence[Question],
                        answered: Sequence[str] = ()) -> List[Question]:
    """Blockers still open at the point of leaving the step.

    The caller uses this to require an acknowledgment. Nothing here blocks —
    the refusal is of silence, not of the user's judgment.
    """
    done = set(answered)
    return [q for q in questions
            if q.kind == "blocker" and q.key not in done and q.status == "asked"]


def acknowledgment_required(questions: Sequence[Question],
                            answered: Sequence[str] = ()) -> Optional[str]:
    """The sentence the record must carry if the step is left as it stands."""
    open_ones = unresolved_blockers(questions, answered)
    if not open_ones:
        return None
    subjects = ", ".join(q.title for q in open_ones)
    return (f"{len(open_ones)} question(s) of consequence were left unresolved: "
            f"{subjects} Recorded so the manuscript can carry it as a stated "
            "limitation rather than omitting it.")


def _is_repairable(f: Dict[str, Any]) -> bool:
    """A finding is a question only when the engine proposes a choice.

    `fix_kind == 'none'` is the engine refusing to guess — a report, not a fork.
    `info` findings are noticings; surfacing each as a question is how an
    interview becomes the wall of prompts it exists to replace.
    """
    return (f.get("severity") in ("critical", "warning")
            and bool(f.get("fix_label"))
            and f.get("fix_kind") not in (None, "none"))


def defer_destination(finding: Dict[str, Any]) -> Tuple[str, str]:
    """Where a deferred finding resurfaces, and the label the button shows.

    The Router owns this because it owns "which step can act on this". The
    frontend must not invent a destination: a deferral whose target is chosen by
    the renderer is a deferral the record cannot honor.

    Two rules:

    * A **structural** repair changes what the table *is*, so it has to be
      settled before rows acquire identities (`T0-ID-001`). Its home step is
      `data`; deferring it moves it to `explore`, the last step where the table
      can still be repaired.
    * A **profile** finding — missingness, distribution, plausibility — is
      answered by a statistical transform, and those are recorded now and fitted
      inside the per-model pipeline. They belong to `preprocess`.
    """
    source = finding.get("source")
    category = ((finding.get("params") or {}).get("category") or "")
    if source == "structure":
        step = "explore"
    elif category.startswith("missing") or category in (
            "physiologic_plausibility", "outliers", "distribution", "skew"):
        step = "preprocess"
    else:
        step = "preprocess"
    return step, STEP_LABELS.get(step, step.title())


def _home_step(f: Dict[str, Any]) -> str:
    """The earliest step that can act on this finding.

    Structural repairs belong at `data`: they change what the table *is*, and
    `T0-ID-001`'s barrier means they must happen before rows acquire identities.
    """
    return "data"



def _missingness_questions(missing_columns, priors_by_column, groups,
                           exceptions, answered, _miss,
                           settled=None) -> List[Question]:
    """Clause §07's question, asked over SETS rather than over columns.

    `GUIDED-029`. Before this, one question per column with blanks — 308 of them
    on `metabolomics_untargeted.csv`, which is roughly ten times the ~32 this
    project calls Classic's indictment. The count is now **O(1) in the column
    count**: at most one bulk question per dtype branch, plus one exceptions
    question per branch, plus a genuine per-column question only where a group
    has fewer members than a rule is worth writing.

    `groups` arrives resolved against the frame, because this module takes no
    dataframe. `None` means the caller has not built them — the per-column path
    is then the whole behavior, which is what every test written before this
    finding expects.
    """
    out: List[Question] = []
    if groups is None:
        for col in (missing_columns or []):
            key = f"missingness::{col}"
            if key not in answered:
                out.append(_one_column_question(col, priors_by_column, _miss))
        return out

    covered = set()

    # THE SKIP SCALES TOO. A pack prior settling 306 columns is ONE stated
    # fact, not 306 — a rendered skip is still a rendered thing, and §09 wants
    # skips to group so their density reads as machine work at a glance.
    for block in (settled or []):
        covered.update(block["columns"])
        if block["key"] in answered:
            continue
        q = Question(
            key=block["key"], kind="missingness", step="preprocess",
            clause="lockbox-07", confidence="high",
            title=block["title"],
            why=_miss.MECHANISM_WHY,
            consumer=_miss.MECHANISM_CONSUMER,
            options=list(_miss.MECHANISM_OPTIONS))
        q.status = "skipped"
        q.skip_reason = (
            f"Not asked for {block['n']:,} columns: {block['reason']} Stated "
            f"from the {block['label'].lower()} lens rather than asked — "
            f"change it here if it is wrong.")
        out.append(q)

    for group in groups:
        if not group["is_bulk"]:
            # A rule over one column is a column with extra words in front of
            # it. Asked individually below, and no rule is invented for it.
            continue
        covered.update(group["members"])
        if group["key"] in answered:
            continue
        out.append(Question(
            key=group["key"], kind="missingness", step="preprocess",
            clause="lockbox-07", title=group["title"],
            why=(f"{_miss.MECHANISM_WHY} You are answering once for "
                 f"{group['n']:,} columns — the rule is *{group['rule']}*, and "
                 f"you edit the rule rather than the list."),
            consumer=(
                _miss.MECHANISM_CONSUMER
                + " Answered here for a set rather than a column, so the "
                  "transcript carries one sentence about "
                  f"{group['n']:,} columns rather than {group['n']:,} sentences "
                  "about one each — which is also the sentence a reader wants."),
            options=list(_miss.MECHANISM_OPTIONS)))

    # Everything no BULK group claimed: a branch with too few columns to be a
    # rule, and any column a group did not reach.
    for col in (missing_columns or []):
        if col in covered or f"missingness::{col}" in answered:
            continue
        out.append(_one_column_question(col, priors_by_column, _miss))

    # THE EXCEPTIONS, as a group. A single answer across 294 columns is not
    # always true — and 500 exceptions asked one at a time would be the
    # unbounded interview arriving through the back door.
    for branch, detail in (exceptions or {}).items():
        key = f"missingness_exceptions::{branch}"
        if key in answered or not detail.get("columns"):
            continue
        out.append(Question(
            key=key, kind="missingness", step="preprocess", clause="lockbox-07",
            title=(f"{len(detail['columns']):,} of those columns look like "
                   f"exceptions"),
            why=detail["sentence"],
            consumer=(
                "The same consumer as the bulk answer, for these columns only. "
                "They were included in it and the evidence disagrees, so this "
                "is the chance to answer them differently before a median is "
                "written over a column that carried signal. Escalated on "
                "evidence that a reading is wrong, never on the size of the "
                "consequence."),
            options=list(_miss.MECHANISM_OPTIONS)))
    return out


def _one_column_question(col, priors_by_column, _miss) -> Question:
    """The per-column question, unchanged — including its rendered skip."""
    column_priors = list((priors_by_column or {}).get(col) or [])
    derived = next((p for p in column_priors
                    if p.get("marker") == "derived"), None)
    q = Question(
        key=f"missingness::{col}", kind="missingness", step="preprocess",
        clause="lockbox-07",
        title=_miss.MECHANISM_QUESTION.format(column=col),
        why=_miss.MECHANISM_WHY,
        consumer=_miss.MECHANISM_CONSUMER,
        confidence="high" if derived else None,
        options=list(_miss.MECHANISM_OPTIONS))
    if derived and len(column_priors) == 1:
        q.status = "skipped"
        q.skip_reason = (
            f"Not asked: {derived['reason']} Stated from the "
            f"{derived['label'].lower()} lens rather than asked — "
            f"change it here if it is wrong.")
    return q


def _repeat_chain(repeats: Dict[str, Any],
                  answered: Sequence[str]) -> List[Question]:
    """Questions 4 to 7 of `OPENING_SEQUENCE.md`, each gating the next.

    `repeats` carries the reading resolved against the frame: `reading`
    (`repeats` | `time_points` | `None` when the evidence is thin), `sentence`,
    plus whatever the record already holds — `kind`, `unit`, `aggregated`.

    **Question 4 is a rendered skip, not a skipped question**, and the
    distinction is `audit()`'s. Decision B permits a skip only for `task_type`,
    which is correct and is not what this is: the app STATES a reading and
    offers to be asked anyway, so the question is `asked` with its reading
    attached. A user who agrees presses through in one gesture; a user who does
    not sees the evidence that was used. Marking it `status="skipped"` would
    both break the audit and claim an authority the evidence does not carry.
    """
    out: List[Question] = []
    reading = repeats.get("reading")
    kind = repeats.get("kind")

    if "state_repeat_kind" not in answered:
        q = Question(
            key="state_repeat_kind", kind="repeat_kind", step="data",
            clause="lockbox-01",
            title="Are these repeats or different time points?",
            why=(repeats.get("sentence") or "")
                + (" " if repeats.get("sentence") else "")
                + ("What varies between one person's rows decides whether "
                   "averaging them is correct: averaging replicate "
                   "measurements reduces measurement error, and averaging a "
                   "trajectory destroys it."),
            consumer=(
                "The aggregation menu reads this and inverts on it — the mean "
                "is recommended for replicates with the measurement-error "
                "reason stated, and has NO default for time points because "
                "averaging them destroys the signal. The temporal-prediction "
                "question fires only when this says time points. Answering "
                "wrongly does not raise an error; it produces a table whose "
                "rows are averages of a trajectory."),
            confidence=repeats.get("confidence"),
            options=["Repeated measurements of the same quantity",
                     "Different time points"],
            option_values=["repeats", "time_points"])
        # The reading, and the affordance to overturn it, travel WITH the
        # question so an interface cannot render one without the other.
        if reading:
            q.skip_reason = repeats.get("sentence")
        out.append(q)
        return out

    if "state_unit_of_analysis" not in answered:
        out.append(Question(
            key="state_unit_of_analysis", kind="unit_of_analysis", step="data",
            clause="lockbox-01",
            title="When you analyze this, what is one row?",
            why=("You told us people appear more than once. That leaves two "
                 "honest options, and they lead to different analyses."),
            consumer=(
                "The seal reads it to decide what it is drawing over, and it "
                "is the last thing that can change what a row IS — after the "
                "seal, rows have identities and combining them is impossible. "
                "There is NO DEFAULT here: guessing at grain is what produced "
                "the leak this whole constitution exists to prevent, and the "
                "same reasoning binds one level down."),
            options=["One row per person", "One row per record",
                     "My design isn't described here"],
            option_values=["person", "record", "not_described"]))
        return out

    unit = repeats.get("unit")
    if unit == "person" and "state_aggregation" not in answered:
        menu = repeats.get("menu") or {}
        out.append(Question(
            key="state_aggregation", kind="aggregation", step="data",
            clause="lockbox-01",
            title="How should each person's rows be combined?",
            why=menu.get("reason", ""),
            confidence="high" if menu.get("recommended") else None,
            consumer=(
                "This rewrites the table before anything is held out — one row "
                "per person from here on — and the receipt says how many rows "
                "became how many. It happens now because it changes what a row "
                "IS, and a seal drawn beforehand would name rows that no "
                "longer exist."),
            options=[o["label"] for o in menu.get("options", [])],
            option_values=[o["key"] for o in menu.get("options", [])]))
        return out

    if (kind == "time_points" and unit == "record"
            and "state_temporal_prediction" not in answered):
        out.append(Question(
            key="state_temporal_prediction", kind="temporal_prediction",
            step="data", clause="lockbox-01",
            title=("Are you predicting something that happens later from "
                   "measurements taken earlier?"),
            why=("A random split — even one grouped by person — is optimistic "
                 "when the task is predicting a later outcome from earlier "
                 "measurements."),
            consumer=(
                "`ml/splits.py` reads this to choose between its chronological "
                "and grouped strategies. Both already exist; what has been "
                "missing is the routing that decides when each applies."),
            options=["Yes", "No"], option_values=["yes", "no"]))
    return out


def _repair_question(f: Dict[str, Any], step: str) -> Question:
    return Question(
        key=f"repair::{f['id']}", kind="repair", step=step,
        title=f["title"],
        why=f.get("why_it_matters") or f.get("detail", ""),
        triggering_finding=f["id"],
        confidence=f.get("confidence"),
        severity=f.get("severity"),
        options=["show me what changes", "remind me later", "dismiss"])


def _rank(findings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The engine's severity, then its confidence, then id. A total order."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_RANK.get(f.get("severity"), 99),
                       _CONFIDENCE_RANK.get(f.get("confidence"), 1),
                       str(f.get("id"))))


def next_question(
    findings: Sequence[Dict[str, Any]],
    *,
    target: Optional[str] = None,
    detection: Optional[Dict[str, Any]] = None,
    step: str = "data",
    deferred: Optional[Dict[str, str]] = None,
    answered: Sequence[str] = (),
) -> Optional[Question]:
    """The single next question, or None when the step is done.

    Derivable from the record alone: `answered` and `deferred` are the record
    replayed, and nothing else is consulted.
    """
    for q in plan(findings, target=target, detection=detection, step=step,
                  deferred=deferred, answered=answered):
        if q.status == "asked":
            return q
    return None


def audit(questions: Sequence[Question]) -> None:
    """Assert the governing rules on a plan. Raises rather than reporting.

    Called by the Router's own tests and by the measurement harness, so a
    violation cannot be scored — a run that breaks Decision B has no number, it
    has a failure.
    """
    for q in questions:
        # Consequence is checked first: both rules would reject a
        # skipped blocker, and the specific message is the useful one.
        if q.kind == "blocker" and q.mode != "push":
            raise RouterError(
                f"{q.key} is a blocker offered as {q.mode!r}. A blocker is a "
                "question of consequence: always pushed, never offered. A "
                "blocker that only offers is not gating.")
        if q.kind == "blocker" and q.status == "skipped":
            raise RouterError(
                f"{q.key} is a blocker and was skipped. Consequence is the one "
                "kind that high confidence does not make moot — being certain a "
                "column leaks is a reason to ask, not a reason to stay quiet.")
        if q.status == "skipped":
            if not _skip_is_permitted(q.confidence, q.kind):
                raise RouterError(
                    f"{q.key} was skipped at confidence {q.confidence!r} and kind "
                    f"{q.kind!r}. Decision B allows a skip only where a high-confidence "
                    "finding makes a question of fact moot.")
            if not q.skip_reason:
                raise RouterError(
                    f"{q.key} was skipped with no reason in the transcript. Every "
                    "skip must be visible and reversible.")
        if q.mode == "pull" and q.status != "asked":
            raise RouterError(
                f"{q.key} is a pull affordance with status {q.status!r}. Pull "
                "affordances are offered, never skipped or deferred — ignoring "
                "one is free, which is what makes it pull rather than push.")
        if q.status == "deferred" and not q.defer_target:
            raise RouterError(
                f"{q.key} was deferred with no target step, so it would never "
                "resurface. Deferral is a disposition, not a discard.")
        if q.kind in FACT_KINDS and q.mode == "push" and not q.consumer:
            # DESIGN_LANGUAGE §09: a FACT that cannot name who consumes its
            # answer is a question we have no right to ask. Enforced here rather
            # than left to the page, so a new FACT cannot ship without one.
            raise RouterError(
                f"{q.key} is a question of fact with no stated consumer. A FACT "
                "must be able to name what reads its answer and what changes as "
                "a result; one that cannot is a question we have no right to ask.")
        if not q.is_visible:
            raise RouterError(f"{q.key} is not visible in the transcript.")
