"""turbotab.models — the shelf, ordered and never shortened.

`PRODUCT_VISION.md`, *"the shelf is never shortened"*:

> The shape of the data changes how models are **ranked**. It never changes
> which models are **available.**

Silently withholding a model is the app deciding in the user's name, which
`DESIGN_LANGUAGE.md` §06 forbids. Offering it at the bottom with the reason
stated is the app doing its job. **Ordering and prominence carry the judgment**
— which is exactly what Classic's Train page failed at: `ml.model_coach` was
already bucketing into good / ok / poor with an evidence-bearing clause per
model, and the layout rendered taxonomy instead. The bucketing was never the
missing piece. The rendering was.

So this module adds almost no intelligence. It calls `model_coach.model_viability`
— which already returns `{model_key: (verdict, clause)}` where the clause cites
*this dataset's* numbers — and turns it into an order, with every model still on
the shelf and every concern stated in full.

**Where this sits on the three-rung ladder.** Model choice is the bottom rung —
*rank and state the concern* — and the ladder's own test is why: the top rung is
not about severity but about whether a competent researcher could have a reason.
There are many analyses in which a tree ensemble at p ≫ n is the right call
anyway. So nothing here refuses, and nothing here blocks.

## L55-B — the recorded design reaches the shelf

Until now this module read the **shape** of the table and nothing the user had
*said*. Three answers the opening sequence deliberately asks for — the purpose
(question 2.5), the repeat kind (question 4) and the unit of analysis
(question 5) — stopped at `AnalysisProject` and were read by the missingness
route and by nothing else. `grep -c lens`, `purpose`, `repeat_kind` and
`unit_of_analysis` in this file and in `ml/model_registry.py` all returned **0**.

Three rules govern what carrying them in may do, and they are what stop this
becoming a filter:

1. **The shelf is never shortened.** A model that does not fit the recorded
   design **ranks lower and says why**. It is never removed, never disabled and
   never hidden. Reordering happens **inside** the coach's bucket, so the
   engine's verdict about the *shape* is not overwritten by a verdict about the
   *design* — two different claims, kept apart.
2. **The sentence is the deliverable, not the ordering.** A reordered shelf with
   no account of why is a black box wearing a lens. Every entry that moved
   carries a clause naming which recorded answer moved it, and that clause
   **quotes the recorded decision** — `Decision.text`, the string the record
   actually kept — rather than recomposing it here. Where an answer bears on
   every model equally, it is a shelf-level statement instead of the same
   sentence repeated once per row, because a caveat printed on everything makes
   a real concern and a routine one read identically.
3. **An unanswered question changes nothing.** `None` is not an answer, and
   neither is an answer whose recorded sentence cannot be found: this module
   will not reorder on something it cannot quote. With no design, the order is
   byte-identical to what it was before this section was written.

**And one thing that is deliberately not here.** A recorded repeated-measures
design does not reorder anything, because it does not differentiate: every model
in the registry treats rows as independent, so the concern is uniform and
demoting all of them is a no-op dressed as a judgment. What it does instead is
**say the number** — how many rows the order was computed from, and how many
people those rows are. `GUIDED-235` is the row for re-ranking on the effective
sample size; that is a change to what `model_coach` is asked, not a sentence,
and it is not made in the same loop as the sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The three buckets, in the words the coach already uses on screen. The mapping
# from the engine's verdict is one place rather than one per renderer.
RECOMMENDED = "recommended"
WORTH_TRYING = "worth_trying"
NOT_RECOMMENDED = "not_recommended"

_BUCKET_FROM_VERDICT = {"good": RECOMMENDED, "ok": WORTH_TRYING,
                        "poor": NOT_RECOMMENDED}

BUCKET_LABEL = {
    RECOMMENDED: "Recommended",
    WORTH_TRYING: "Worth trying",
    NOT_RECOMMENDED: "Not recommended for this data",
}

# Rendered above the third group so the ordering cannot be misread as a
# shortlist. The list is not filtered and the app says so, because a user who
# believes a model is unavailable will not think to look for it.
SHELF_DISCLOSURE = (
    "Every model is available. This order is about your data, not about which "
    "models are any good — a model low on this list is one whose concern "
    "applies to a table this shape, and you may have a reason it does not "
    "apply to yours. Select whatever you intend to train.")

_BUCKET_ORDER = {RECOMMENDED: 0, WORTH_TRYING: 1, NOT_RECOMMENDED: 2}


class ModelSelectionError(Exception):
    """The shelf was asked for something it cannot honestly provide."""


# ─────────────────────────────────────────────────────────────────────────────
# The recorded design — three answers, each with the sentence the record kept
# ─────────────────────────────────────────────────────────────────────────────

#: The three questions, in the words the app asked them. Named here so a clause
#: can say WHICH question moved a model rather than naming a field.
QUESTION_TITLE = {
    "purpose": "What is this model for?",
    "repeat_kind": "Are these repeats or different time points?",
    "unit_of_analysis": "When you analyze this, what is one row?",
}


@dataclass(frozen=True)
class RecordedDesign:
    """What the user *said*, as opposed to what the table looks like.

    Every `*_said` field is the **recorded sentence** — `Decision.text`, built
    once by the method that recorded the answer and never rebuilt here. An
    answer with no sentence beside it is treated as unanswered: a clause that
    cannot quote the record has nothing to quote, and paraphrasing is the one
    thing `L36`, `L53-B` and `L54-B` all ruled against on three other surfaces.

    `n_rows` / `n_people` describe the rows the shelf was RANKED on, not the
    table — the seal is drawn before the shelf is offered and the order is
    computed on the training rows (`GUIDED-088`).
    """
    purpose: Optional[str] = None
    purpose_said: Optional[str] = None
    repeat_kind: Optional[str] = None
    repeat_said: Optional[str] = None
    unit_of_analysis: Optional[str] = None
    unit_said: Optional[str] = None
    n_rows: Optional[int] = None
    n_people: Optional[int] = None
    group_column: Optional[str] = None

    def answered(self, which: str) -> bool:
        """Is this question answered AND quotable?

        Both halves, and the second is the load-bearing one. An answer whose
        recorded sentence is missing is treated exactly like no answer: the
        shelf does not reorder on something it cannot quote, because the clause
        that would explain the reorder is the deliverable.
        """
        return bool(getattr(self, which, None)
                    and getattr(self, _SAID[which], None))

    def said(self, which: str) -> str:
        """The recorded sentence for this question, verbatim."""
        return str(getattr(self, _SAID[which]))


#: question -> the field holding its recorded sentence. A map rather than a
#: naming convention, because `unit_of_analysis`'s sentence field is `unit_said`
#: and a convention that needs an exception is not a convention.
_SAID = {"purpose": "purpose_said",
         "repeat_kind": "repeat_said",
         "unit_of_analysis": "unit_said"}

NO_DESIGN = RecordedDesign()


@dataclass(frozen=True)
class DesignNote:
    """Why a recorded answer moved this model, quoting the record.

    `quote` is the record's own sentence and is never edited on the way through.
    `clause` is this module's account of what that sentence means for this one
    model, and it is the only composed text in the note.
    """
    answer: str
    question: str
    quote: str
    clause: str
    ranks_lower: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "question": self.question,
                "quote": self.quote, "clause": self.clause,
                "ranks_lower": self.ranks_lower}


@dataclass(frozen=True)
class ShelfEntry:
    """One model, with its verdict and the concern in full."""
    key: str
    name: str
    group: str
    bucket: str
    # The engine's own clause, citing this dataset's numbers. Never summarized
    # and never truncated: a concern shortened to fit a badge is the failure
    # this module exists to correct.
    concern: str
    requires_scaled_numeric: bool
    recommended_for_high_dim: bool
    interpretability: str
    # `L55-B`. Empty for every entry the recorded design did not move, which is
    # every entry when nothing was recorded. A note is emitted ONLY where the
    # answer moved this model — an entry that stayed put is accounted for by the
    # shelf-level statement, not by a clause repeated down the column.
    design_notes: Tuple[DesignNote, ...] = field(default_factory=tuple)

    @property
    def design_rank(self) -> int:
        """0 for an entry the recorded design left alone, 1 for one it lowered.

        A SECOND SORT KEY AND NOT A BUCKET CHANGE. The bucket is the coach's
        verdict about the SHAPE of the table; this is a verdict about the
        recorded DESIGN. Letting one rewrite the other would make a model the
        engine calls a good fit for this shape read as a poor one, which is a
        different claim than the app has evidence for.
        """
        return 1 if any(n.ranks_lower for n in self.design_notes) else 0

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "group": self.group,
                "bucket": self.bucket, "bucket_label": BUCKET_LABEL[self.bucket],
                "concern": self.concern,
                "requires_scaled_numeric": self.requires_scaled_numeric,
                "recommended_for_high_dim": self.recommended_for_high_dim,
                "interpretability": self.interpretability,
                "design_notes": [n.to_dict() for n in self.design_notes],
                "ranked_lower_by_design": bool(self.design_rank)}


PREDICTION = "prediction"
INFERENCE = "inference"
UNIT_RECORD = "record"
UNIT_PERSON = "person"
UNIT_NOT_DESCRIBED = "not_described"


def _purpose_note(name: str, caps: Any, design: RecordedDesign
                  ) -> Optional[DesignNote]:
    """The one recorded answer that differentiates between models.

    Only under `inference`, and only downwards. Under `prediction` nothing here
    separates one model from another — every model on this shelf predicts — so
    the answer is reported at the shelf level as *read and changed nothing*,
    which is a real state and gets said rather than left as silence.
    """
    # `answered`, NOT `design.purpose ==`. The first draft compared the answer
    # alone and reordered a shelf whose clause would have had nothing to quote
    # — rule 3's second half failing in the code that rule 3 is about. The
    # question is not *what did they answer* but *what did the record keep*.
    if not design.answered("purpose") or design.purpose != INFERENCE:
        return None
    exposes = getattr(caps, "exposes_coefficients", None)
    # UNDECLARED IS NOT A NO. A model whose registry entry has not answered the
    # question takes no part in this ordering and the shelf says nothing about
    # its coefficients — the alternative is asserting an absence from ignorance.
    if exposes is not False:
        return None
    return DesignNote(
        answer="purpose",
        question=QUESTION_TITLE["purpose"],
        quote=design.said("purpose"),
        clause=(
            f"{name} exposes no per-predictor coefficient once fitted, so there "
            f"is no association estimate to read off it and the model "
            f"coefficients figure is not drawn for it. It is ranked below the "
            f"models that can answer the question you recorded, and it is still "
            f"here and still selectable — a model can be the right choice for a "
            f"paper whose objective is an association without being the thing "
            f"the association is estimated from."),
        ranks_lower=True)


def design_notes(name: str, caps: Any,
                 design: Optional[RecordedDesign]) -> Tuple[DesignNote, ...]:
    """Every clause the recorded design attaches to THIS model.

    Empty when nothing was recorded, which is what makes rule 3 — *an
    unanswered question changes nothing* — a property of the code rather than a
    promise in a docstring.
    """
    if design is None:
        return ()
    note = _purpose_note(name, caps, design)
    return (note,) if note is not None else ()


def design_statement(design: Optional[RecordedDesign]) -> List[Dict[str, Any]]:
    """What the recorded design did to the WHOLE shelf, quoting the record.

    Separate from the per-entry notes on purpose. An answer that bears on every
    model equally belongs here: the same sentence printed once per row would
    make a real concern and a routine one read identically, which is the
    second, uncalibrated layer of caution this project forbids elsewhere.
    """
    if design is None:
        return []
    out: List[Dict[str, Any]] = []

    if design.answered("purpose"):
        if design.purpose == INFERENCE:
            effect = ("Models that expose one coefficient per predictor are "
                      "ordered first within each group, because that is what "
                      "an association estimate is read from. Nothing was "
                      "removed; the models that do not are lower down with the "
                      "reason on them.")
        else:
            effect = ("The order below is unchanged by this answer. Every model "
                      "here predicts, so nothing about a prediction objective "
                      "separates one from another — this is said rather than "
                      "left silent, because an answer that changed nothing and "
                      "an answer nobody read look the same from outside.")
        out.append({"answer": "purpose",
                    "question": QUESTION_TITLE["purpose"],
                    "quote": design.said("purpose"),
                    "effect": effect})

    if design.answered("repeat_kind"):
        # THE ORDER IS NOT TOUCHED AND THE REASON IS THAT IT WOULD BE A NO-OP.
        # Every model in the registry treats rows as independent, so this
        # concern lands on all of them equally; demoting all of them would move
        # nothing while reading as a judgment. What is available instead is the
        # number, which nothing else on this surface says.
        rows = design.n_rows
        people = design.n_people
        if rows is not None and people is not None:
            counted = (f"This order was computed from {rows:,} rows, which are "
                       f"{people:,} people"
                       + (f" identified by `{design.group_column}`. "
                          if design.group_column else ". "))
        elif rows is not None:
            counted = (f"This order was computed from {rows:,} rows, and how "
                       f"many people those rows are was not recorded. ")
        else:
            counted = ""
        quote = design.said("repeat_kind")
        if design.answered("unit_of_analysis"):
            quote = quote + " " + design.said("unit_of_analysis")
        out.append({
            "answer": "repeat_kind",
            "question": QUESTION_TITLE["repeat_kind"],
            "quote": quote,
            "effect": (
                counted +
                ("Every model on this shelf treats rows as independent — none "
                 "of them carries a person-level random effect or a "
                 "cluster-robust term — so this answer does not separate one "
                 "model from another and the order below is unchanged by it. "
                 "The rows-versus-people count is stated because the ranking "
                 "was computed on the row count."
                 if design.unit_of_analysis != UNIT_PERSON else
                 "Each person's records are combined into one before anything "
                 "is held out, so the rows this order was computed on are "
                 "already one per person and no model here is being asked to "
                 "treat correlated rows as independent. The order below is "
                 "unchanged by this answer.")),
        })

    return out


def shelf(profile: Any, task_type: str, probe: Any = None,
          design: Optional[RecordedDesign] = None) -> List[ShelfEntry]:
    """Every model the registry offers, ordered by fit and never filtered.

    `profile` is a `ml.dataset_profile` profile object — the same one the coach
    reads — so the clauses quote real numbers rather than adjectives.

    `design` is what the user SAID (`RecordedDesign`). `None` — and a design
    with nothing answered — leave the result byte-identical to what it was
    before `L55-B`: same models, same order, no notes.
    """
    from ml.model_registry import get_registry
    from ml import model_coach

    registry = get_registry()
    try:
        verdicts = model_coach.model_viability(profile, probe)
    except Exception as exc:
        # A profile the coach cannot read must not empty the shelf. Silence
        # about ORDER is recoverable; an empty list is the app withholding
        # every model, which is the thing this module forbids.
        #
        # Recoverable is not the same as invisible: every model then appears
        # unranked and unconcerned, and a shelf with no concerns on it looks
        # like a shelf where the engine had nothing to say.
        from turbotab import devchecks
        devchecks.swallowed(
            "models.shelf::model_viability", exc,
            "every model is presented with no ranking and no concern, which "
            "reads as 'the engine had nothing to say about these'")
        verdicts = {}

    supports = ("supports_classification" if task_type == "classification"
                else "supports_regression")
    out: List[ShelfEntry] = []
    for key, spec in registry.items():
        caps = spec.capabilities
        if not getattr(caps, supports, False):
            # NOT a judgment and NOT a shortening: a classifier cannot fit a
            # continuous outcome at all, so offering it would be offering
            # something that raises rather than something that fits poorly.
            # The ladder's top rung — no legitimate use exists.
            continue
        verdict, clause = verdicts.get(key, ("ok", ""))
        out.append(ShelfEntry(
            key=key, name=spec.name, group=spec.group,
            bucket=_BUCKET_FROM_VERDICT.get(verdict, WORTH_TRYING),
            concern=clause or "No specific concern for a table this shape.",
            requires_scaled_numeric=bool(caps.requires_scaled_numeric),
            recommended_for_high_dim=bool(caps.recommended_for_high_dim),
            interpretability=str(caps.interpretability_tier),
            design_notes=design_notes(spec.name, caps, design)))

    # `design_rank` sits BETWEEN the bucket and the alphabet, which is the whole
    # of what the recorded design is allowed to do to this list. It moves an
    # entry down inside its group; it never moves it out of one, never drops it,
    # and with no design recorded it is 0 for every entry, so the key collapses
    # to the one that was here before.
    out.sort(key=lambda e: (_BUCKET_ORDER[e.bucket], e.design_rank,
                            e.group, e.name))
    return out


def grouped(entries: Sequence[ShelfEntry]) -> List[Dict[str, Any]]:
    """The shelf as three labeled groups, in order, none of them omitted.

    An empty group is RETURNED EMPTY rather than dropped: "no model is
    recommended for this data" is a real and informative state, and a renderer
    that only sees two groups cannot say it.
    """
    return [{"bucket": b, "label": BUCKET_LABEL[b],
             "models": [e.to_dict() for e in entries if e.bucket == b]}
            for b in (RECOMMENDED, WORTH_TRYING, NOT_RECOMMENDED)]


def validate_selection(entries: Sequence[ShelfEntry],
                       chosen: Sequence[str]) -> List[str]:
    """Accept any model on the shelf, in the order the shelf presents them.

    The only refusals are structural — an unknown key, or an empty selection —
    and neither is a judgment about fit. Selecting three `not_recommended`
    models is a legitimate thing to do and this returns them.
    """
    known = {e.key: e for e in entries}
    unknown = [k for k in chosen if k not in known]
    if unknown:
        raise ModelSelectionError(
            f"{unknown} are not models this task can use. Available: "
            f"{', '.join(sorted(known))}.")
    if not chosen:
        raise ModelSelectionError(
            "Choose at least one model. Preprocessing is configured per model, "
            "so there is nothing to configure until you say what you intend to "
            "train.")
    order = [e.key for e in entries]
    return sorted(set(chosen), key=order.index)


def selection_note(entries: Sequence[ShelfEntry],
                   chosen: Sequence[str]) -> Optional[str]:
    """What the record says when a low-ranked model is chosen.

    Not a warning and not a confirmation step — the choice is already made and
    already legitimate. This is the sentence the methods section carries, so
    that the concern travels with the result rather than being an on-screen
    caveat the reader never sees.
    """
    by_key = {e.key: e for e in entries}
    poor = [by_key[k] for k in chosen
            if k in by_key and by_key[k].bucket == NOT_RECOMMENDED]
    if not poor:
        return None
    lines = "; ".join(f"{e.name} — {e.concern}" for e in poor)
    return (f"{len(poor)} of the selected model(s) carry a stated concern for a "
            f"table this shape: {lines}. Selected deliberately; the concern is "
            f"recorded so it can be reported rather than discovered.")
