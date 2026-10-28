"""The corpus record: one JSON document per swarm run (DRIVER_PLAN D3; SWARM_PLAN §4).

A record is assembled after the run from the server's own data and the driver's run directory
(``assemble.py``), never from the analyst's account of it, and validated here before it is kept.
Version 2 (SWARM_PLAN §4, the build's package 1) adds the fields marked new. Its fields:

* ``run``: the run's id and design, the engine's commit, the analyst's model, **its persona**
  (new; both redacted in the view a blind reader is shown, :func:`redacted`, whose times keep
  their order only), effort and prompt version, **the vocabulary its tags were taken under**
  (new, ``vocabulary.DIGEST``), when it started and ended, its spend (fresh and cached tokens,
  counted from the analyst's transcript once per message) and its fits and their seconds;
* ``dataset``: the files the run read, each with its SHA-256 and where it came from, and, from the
  intake, whether the dataset is a known-answer control or a null decoy (never on the analyst's
  card), and **the declared measurement markers** (new, SWARM_PLAN §5: each quoted, or recorded
  as not stated);
* ``question``: the goal G (the purpose) and the estimand or intended use Q the analyst declared;
* ``regime``: the engine's readings of the table: rows, rows analyzed, events, blanks, the design,
  the lenses; and **each marker of the vocabulary the engine reads, tagged live or post hoc**
  (new, :class:`Marker`: its value with where the engine served it, or why it has none);
* ``decisions``: each decision record of the log with its option, the options examined for it
  (previewed, **with the preview's reason**, new, or fitted), its reason (the record's ``note``),
  **the analyst's tag at the decision: its marker and the alternative it rejected** (new,
  :class:`Tag`, from the driver's ``answer``), its phase, who recorded it and its materiality M;
* ``attempts``: each fit's record as ``GET …/fits`` serves it (but a penalized family's coefficient
  rows, which it serves among the fit's ``effects``, kept apart under ``penalized``: they are
  shrunk by the penalty and estimate no effect), with its change from the fit before
  it both ways (the records that entered force and those that left it, each credited to the record
  that made the change: itself, the record that replaced it, or the revert that withdrew or
  restored it) and the reasons of the records credited; **the live markers at that fit** (new,
  read by the driver while the fit's answers stood); and **the signed change in each family's
  primary score from the fit before** (new, :class:`SignedChange`: paired over the same folds,
  else unpaired with the Nadeau–Bengio inflation, ``signed.py``);
* ``result``: the reported result, once: the first opening's held-out scores under Predict; under
  Estimate the number the app reports for the estimand the lock's plan declares, on its measure's
  scale (a marginal risk difference or ratio as ``GET …/results`` serves the effects stage's
  standardization, a conditional measure as the locked fit's exposure rows), each with its
  interval; or why there is none and of which kind (none reported, none served, or none the
  record can know);
* ``claims``: every registry claim rendered for the project, as the analyst log serves it;
* ``surfacings``: each refusal the driver met and the command that followed it, each concern
  the materiality ledger measured, and **each fit that failed while the driver watched, classed
  refused (the engine's typed refusal: its code and ways forward) or failed** (new,
  :class:`FailedFit`);
* ``wants`` (new): each option, check, reading or eligibility rule the analyst said the engine
  lacks, with its reason and marker, as the driver's ``want`` recorded it;
* ``account`` (new): the analyst's end-of-run account, as the driver's ``account`` recorded it,
  never the agent's final message, marked whether it was written after the results were read and
  after the seal; context for hypotheses, never evidence;
* ``export``: the bundle's SHA-256, its methods text and its checklist, and whether the analyst's
  own export reached it;
* ``replication``: for a replication (design 1), the paper's decisions against ours, filled by its
  grader; null otherwise.

**The replays are not the record's.** A final plan replayed on fresh draws of its fixture's
generator (SWARM_PLAN §7) is made after the record, from it and its bundle, by
``score_pilot.py replay``, which writes it to a file of its own and scores it from there; no field
of the record holds it, so a record that carries one fails its schema. A record written before the
field was dropped (the dry run's) holds ``replay: null``, which records nothing, and is read as one
without it.

**Who fills what.** The analyst fills G, Q, D and O through its answers; the engine fills the
regime R and the materiality M; **no field holds a validity V**. Every model here forbids fields it
does not declare, so a record that carries a validity, or any field a run invents, fails its schema.
Only a mechanical check (``checks.py``) or an independent grader judges a run. Every tag and want
names a marker of the vocabulary (``vocabulary.py``), and a want a kind of it; a record that names
another fails.

**Materiality is the engine's number.** A decision's ``materiality`` holds either what the engine
measured (``GET …/fits``' paired movement for the fit whose whole change from the fit before is
that decision's, a revert's included, or ``GET …/mattered``' specifications for a choice the curve
varies), with at least one number, or why it has none; a record holding neither, or both, or a
measure of whys alone, fails its schema.

**A record that fails its schema is a failed run** (``assemble.write``): it is written to
``research/corpus/failed/`` with the error, and never edited into one that passes.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research.swarm import vocabulary as V

VERSION = 2

# When a decision was made, by the decision log's own marks (SWARM_PLAN §4's phases):
# "after_seal", recorded after the held-out rows were opened (``post_seal``); "after_estimates",
# recorded with the inference estimates in view (``after_estimates``: after the plan's lock, once
# an estimate was shown under it); "before", neither.
Phase = Literal["before", "after_estimates", "after_seal"]
Control = Literal["known_answer", "null_decoy"]
# Why a result has no numbers (``assemble.result_of``): "not_reported", the run reported none (the
# seal never opened, the plan never locked, no goal, no estimand in the lock's plan);
# "not_served", the engine served none (the opening recorded no held-out scores and its fit serves
# none; the engine refused the fit of the locked plan, ``assemble.fit_refused_at_lock``; the
# locked fit serves no exposure rows and withholds none; the effects stage served no marginal risk
# for any family); "not_known", the record cannot say what the run reported (no fit record, or
# several, is marked the one the lock was recorded on, and the engine did not refuse that fit;
# the route withholds the locked fit's rows now; the opening recorded no scores while its fit
# serves held-out scores; the results no longer serve the locked plan's marginal risks).
RefusedKind = Literal["not_reported", "not_served", "not_known"]
# SWARM_PLAN §6's personas: stances toward the field's defaults, under one protocol.
Persona = Literal["conventional", "skeptical", "explorer"]
# What a redacted field holds in the view a blind reader is shown (:func:`redacted`).
REDACTED = "redacted"
REDACTED_SPEND = "Redacted for a blind reader: a run's spend can tell its model."
# The two scales of an effect measure (``estimand.measure_facts``, MODELING_SEQUENCE ruling 9), and
# the value of no effect on each.
Scale = Literal["difference", "ratio"]
NULL_OF: dict[str, float] = {"difference": 0.0, "ratio": 1.0}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── run ──────────────────────────────────────────────────────────────────────


class Spend(_Strict):
    """Tokens, counted from the analyst agent's transcript once per message id
    (``assemble.spend``): ``fresh`` is input, cache creation and output; ``cached`` is cache
    reads."""

    fresh: int = Field(ge=0)
    cached: int = Field(ge=0)
    messages: int = Field(ge=0)


class Run(_Strict):
    id: str = Field(min_length=1)
    design: Literal[1, 2, 3]  # SWARM_PLAN §1: replication, agent analysts, the full shelf
    engine_commit: str | None  # the server's git revision, from the export's provenance
    engine_modified: bool | None  # whether tracked files differed from it
    engine_version: str | None  # GET /health's version
    analyst_model: str = Field(min_length=1)
    # The persona the analyst ran under; None for a run under none (design 1's sweep, design 3).
    persona: Persona | None = None
    effort: str | None
    prompt_version: str = Field(min_length=1)
    vocabulary: str = Field(min_length=1)  # the vocabulary's digest (``vocabulary.DIGEST``)
    # The view a blind reader is shown (:func:`redacted`): every time in it is its order only.
    redacted: bool = False
    started: datetime
    ended: datetime | None
    spend: Spend | None
    spend_why: str | None  # why there is no spend
    fits: int = Field(ge=0)  # the fit records the server kept (one per distinct fit)
    fit_seconds: float = Field(ge=0)  # summed over the fits whose seconds the server saw
    fits_unseen: int = Field(ge=0)  # fits whose seconds the server did not see
    fit_budget: Any = None  # the driver's budget, as driver.json holds it
    help: list[str] = []  # each intervention by a person, as recorded; none recorded is none
    server: dict[str, Any] | None = None  # the driver's server: its pid, port, mode and workers

    @model_validator(mode="after")
    def _spend_or_why(self) -> "Run":
        if (self.spend is None) == (self.spend_why is None):
            raise ValueError("a run holds its spend or why it has none, not both and not neither")
        hidden = (self.analyst_model == REDACTED and self.persona is None and self.effort is None
                  and self.prompt_version == REDACTED and self.spend is None
                  and self.spend_why == REDACTED_SPEND)
        if self.redacted and not hidden:
            raise ValueError("a redacted run holds no model, persona, effort, prompt version or "
                             "spend")
        if not self.redacted and REDACTED in (self.analyst_model, self.prompt_version):
            raise ValueError("a run that is not redacted names its model and prompt version")
        return self


# ── dataset, question, regime ────────────────────────────────────────────────


class DatasetFile(_Strict):
    """One input file as the export's ``provenance.json`` lists it (else the ingest's own)."""

    role: str
    name: str
    sha256: str | None
    bytes: int | None = None
    url: str | None = None
    release_date: str | None = None
    license: str | None = None


DeclaredName = Literal[V.DECLARED]  # type: ignore[valid-type]


class DeclaredMarker(_Strict):
    """A declared measurement marker (SWARM_PLAN §5), as the intake reads it from the paper's
    methods or the NCHS documentation: its value with the passage quoted and where it is, or, where
    the source does not say, ``not_stated`` with neither; whether a referee checked the quote."""

    name: DeclaredName
    status: Literal["declared", "not_stated"]
    value: str | None = None
    quote: str | None = None
    source: str | None = None
    checked: bool | None = None

    @model_validator(mode="after")
    def _quoted_or_not_stated(self) -> "DeclaredMarker":
        if self.status == "declared" and not (self.value and self.quote and self.source):
            raise ValueError("a declared marker holds its value, the passage quoted and where it is")
        if self.status == "not_stated" and (self.value is not None or self.quote is not None):
            raise ValueError("a marker the source does not state holds no value and no quote")
        return self


class Dataset(_Strict):
    id: str = Field(min_length=1)
    # From the intake's manifest, never the analyst's card: a known-answer control or a null decoy.
    control: Control | None = None
    # The fixture's file name for a known-answer control (``truths.FIXTURE_TRUTHS``' key).
    fixture: str | None = None
    # A known-answer control's declared answer, where its intake declares one (``checks``).
    known: dict[str, Any] | None = None
    files: list[DatasetFile]
    codebook: str | None = None
    card: dict[str, Any] | None = None  # the question card the analyst was given
    declared: list[DeclaredMarker] = []  # SWARM_PLAN §5, from the intake's version 2

    @model_validator(mode="after")
    def _each_declared_once(self) -> "Dataset":
        names = [m.name for m in self.declared]
        if len(names) != len(set(names)):
            raise ValueError("a declared marker is listed once")
        return self


class Question(_Strict):
    purpose: str | None  # G
    target: str | None
    estimand: dict[str, Any] | None  # Q under Estimate: the estimand as recorded
    intended_use: dict[str, Any] | None  # Q under Predict: the intended use as recorded
    stated: str | None = None  # the card's question, in its words
    paper: str | None = None  # a replication's paper


MarkerName = Literal[V.NAMES]  # type: ignore[valid-type]


class Marker(_Strict):
    """One marker of the vocabulary as the engine reads it (``markers.py``): its name, its axis and
    whether it is known live or post hoc, as the vocabulary fixes them; its value, with where the
    engine served it (``basis``), or why it has none. A declared marker is the dataset's
    (:class:`DeclaredMarker`), never one the engine reads."""

    name: MarkerName
    axis: V.Axis
    timing: Literal["live", "post_hoc"]
    value: Any = None
    basis: str | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _as_the_vocabulary_fixes_it(self) -> "Marker":
        fixed = V.BY_NAME[self.name]
        if (self.axis, self.timing) != (fixed.axis, fixed.timing):
            raise ValueError(f"the marker {self.name} is {fixed.axis} and {fixed.timing} in the "
                             f"vocabulary")
        if (self.value is None) == (self.why is None):
            raise ValueError(f"the marker {self.name} holds its value or why it has none, not "
                             f"both and not neither")
        if self.value is not None and not self.basis:
            raise ValueError(f"the marker {self.name}'s value says where the engine served it")
        return self


def _once(markers: list[Marker], where: str) -> None:
    names = [m.name for m in markers]
    if len(names) != len(set(names)):
        raise ValueError(f"{where} lists a marker more than once")


def has_table(family: Any) -> bool:
    """Whether ``family`` gives a coefficient table under inference, as the effects stage reads it
    (``stages.effects.has_coefficient_table``), so its fit's rows of the exposure are an estimate
    of the effect: a penalized family's shrunk rows are not (the ridge, the elastic nets, Huber's),
    nor are those of a family the engine does not register."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.models.base import get_family
    from turbotab.core.stages.effects import has_coefficient_table

    try:
        return has_coefficient_table(get_family(str(family)))
    except KeyError:
        return False


def split_effects(effects: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """A fit's rows as ``GET …/fits`` serves them (its ``effects``: each family's coefficient rows,
    or None where it reports none), split in two: the rows of each family that gives a coefficient
    table (:func:`has_table`), and each other family's rows, which are shrunk by its penalty and
    are no estimate of an effect (the ruling of 2026-10-23); a family with no rows stays with the
    first, holding none."""
    tabled: dict[str, Any] = {}
    penalized: dict[str, Any] = {}
    for family, rows in (effects or {}).items():
        (penalized if rows and not has_table(family) else tabled)[str(family)] = rows
    return tabled, penalized


class Regime(_Strict):
    """The engine's readings, each its own number as served, and (version 2) each marker of the
    vocabulary it reads at the end of the run, live and post hoc."""

    n_rows: int | None  # the ingest's rows
    n_analyzed: int | None  # the participant flow's final count
    task: str | None
    event: str | None  # a yes/no outcome's event level
    events: int | None  # rows at the event level, from the outcome's own value counts
    missing: dict[str, int]  # each column with blanks, and how many
    design: dict[str, Any]  # survey, clusters, grain, repeats, unit and design, as recorded
    lenses: list[str]
    markers: list[Marker]

    @model_validator(mode="after")
    def _markers_once(self) -> "Regime":
        _once(self.markers, "the regime")
        return self


# ── decisions and attempts ───────────────────────────────────────────────────


class Answer(_Strict):
    """One question a decision answers, and its value (``assemble.answers_of``):
    ``<reading>:<column>`` for a reading (the keys of a fixture's declared truth,
    ``truths.Truth``), else where the server keeps the answer: its slot, and the entry's key where
    the slot is keyed (``adjust:<column>`` for a covariate's adjustment answer)."""

    question: str
    value: Any


class Examined(_Strict):
    """An option examined for a question a decision answers: previewed (``GET …/analyst-log``'
    previews) while the decision was being chosen or while it stood, after the question's previous
    answer and before its next; or fitted (an earlier answer in force in a fit that completed
    before the decision). ``alternative``: it is not the option chosen and, for a preview, it
    previewed rather than being refused."""

    question: str
    value: Any
    how: Literal["preview", "fit"]
    at: datetime
    seq: int | None = None  # how == "fit": the earlier record that held the option
    attempt: str | None = None  # how == "fit": the fit it was in force in
    refused: str | None = None  # how == "preview": the refusal's code
    # how == "preview": the reason the driver's ``preview`` took with it; None where no preview
    # command of the run's is found for it (a preview made outside the driver).
    reason: str | None = None
    alternative: bool


class Materiality(_Strict):
    """How far the engine measured a decision to move the result, or why it has no measure.

    ``source == "fits"``: ``movement`` is the paired movement ``GET …/fits`` served for the fit
    the decision first took effect in (a revert: the fit whose change holds its withdrawal), where
    the fit's whole change from the fit before was that one record's (``Attempt.changed`` is that
    record alone, and every record that entered or left force is credited to a record). ``source
    == "mattered"``: ``departures`` are ``GET …/mattered``' specifications that
    change this decision's choices, beside its ``primary``. A measure holds at least one number;
    where every family or departure has only a why, the materiality is that why."""

    source: Literal["fits", "mattered"] | None
    attempt: str | None = None
    movement: list[dict[str, Any]] | None = None
    primary: dict[str, Any] | None = None
    departures: list[dict[str, Any]] | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _number_or_why(self) -> "Materiality":
        listed = bool(self.movement) or bool(self.departures)
        if listed == (self.why is not None):
            raise ValueError("a materiality holds the engine's measure or why there is none, not "
                             "both and not neither")
        numbers = any(m.get("movement") is not None for m in self.movement or ()) or any(
            d.get("spec") is not None for d in self.departures or ())
        if listed and not numbers:
            raise ValueError("a measure holds at least one number the engine measured; whys "
                             "alone are a why")
        if listed and self.source is None:
            raise ValueError("a measured materiality names the route that measured it")
        return self


class Tag(_Strict):
    """What the analyst said at a decision beside its reason (SWARM_PLAN §4, the analyst's account
    written at each decision, before it had seen what mattered): the marker of the vocabulary the
    decision answers to, and the alternative it rejected (an option offered for the same question,
    by its id and the decision it would record), as the driver's ``answer`` took them."""

    marker: MarkerName
    rejected: dict[str, Any] | None = None
    command: Literal["answer"] = "answer"
    at: datetime | None = None


class Decision(_Strict):
    id: str
    seq: int = Field(ge=1)
    at: datetime
    kind: str
    option: dict[str, Any]  # the decision as recorded
    answers: list[Answer]
    reason: str | None  # the record's note
    source: Literal["analyst", "app"]  # recorded by the analyst ("you"), or by TurboTab itself
    phase: Phase
    post_seal: bool
    after_estimates: bool
    early: str | None = None
    examined: list[Examined]
    materiality: Materiality
    # The analyst's tag, for a record the driver's ``answer`` posted; None for every other (one
    # TurboTab recorded itself, a revisit, the seal).
    tag: Tag | None = None


class Reason(_Strict):
    seq: int
    note: str


class Shift(_Strict):
    """A record that left force (``Attempt.withdrawn``) or came back into it
    (``Attempt.reinstated``) between two fits, and ``by``, the record credited with it: the revert
    that withdrew or restored it, or the later record that replaced it; null when no record of the
    log is found to credit."""

    seq: int
    by: int | None


class Attempt(_Strict):
    """One fit's record as ``GET …/fits`` serves it, with its change from the fit before it
    (``assemble.change_of``), both ways:

    * ``changed``: the records credited with the change, each record in force here that is new
      since the fit before, and each record credited with a shift (every record in force, for the
      first fit), held to the server's own rule: one record alone only where that record alone
      makes this fit's records from the fit before's, else the records this fit read since the
      fit before, which moved together; null where either fit's records are not served, or the
      decision log does not account for them (``assemble.Prefixes``: no prefix of it holds them,
      or two that do fold to different answers);
    * ``withdrawn``: each record in force at the fit before and not here, with the record credited
      (none where ``changed`` is null);
    * ``reinstated``: each record in force here, recorded before the fit before and not in force
      at it, with the revert credited (none where ``changed`` is null);
    * ``reasons``: the notes of the records in ``changed``."""

    id: str
    seq: int = Field(ge=1)
    at: datetime
    purpose: str | None
    target: str | None
    fit_key: str
    plan_digest: str | None
    decisions: list[int] | None
    changed: list[int] | None
    withdrawn: list[Shift]
    reinstated: list[Shift]
    reasons: list[Reason]
    press: dict[str, Any] | None
    families: list[str]
    seconds: float | None
    workers: int
    primary_metric: str | None
    folds: str | None
    seed: int | None
    held_out: str | None
    scores: dict[str, dict[str, Any]]
    holdout: dict[str, dict[str, float | None] | None] | None
    holdout_why: str | None
    changed_after_seal: bool
    at_opening: dict[str, Any] | None
    # Each family's rows of the fit that estimate an effect: those of a family with a coefficient
    # table (:func:`has_table`). ``GET …/fits`` serves a penalized family's rows under its
    # ``effects`` too; they are shrunk by the penalty and estimate no effect (the ruling of
    # 2026-10-23), so the record keeps them under ``penalized``, never here (:func:`split_effects`,
    # :meth:`_shrunk_rows_apart`), and no reader of ``effects`` reads one as an effect.
    effects: dict[str, list[dict[str, Any]] | None]
    penalized: dict[str, list[dict[str, Any]]] = {}
    movement: list[dict[str, Any]]
    movement_why: str | None
    withheld: str | None
    # Estimate: GET …/fits marks this the fit the plan's lock in force was recorded on.
    locked: bool = False
    # The live markers at this fit, as the driver read them while its answers stood (a command
    # whose current fit was this one), or why there are none.
    markers: list[Marker] | None = None
    markers_why: str | None = None
    # Under Predict, each family's signed change in its primary score from the fit before
    # (:class:`SignedChange`), or why there is none (the first fit; no score served).
    change: list["SignedChange"] = []
    change_why: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _shrunk_rows_apart(cls, data: Any) -> Any:
        """A penalized family's rows given under ``effects`` (as ``GET …/fits`` serves them, and
        as a record assembled before 2026-10-28 holds them) are held under ``penalized``; a
        family's rows given under both fail."""
        if not isinstance(data, dict) or not isinstance(data.get("effects"), dict):
            return data
        tabled, shrunk = split_effects(data["effects"])
        given = data.get("penalized") or {}
        twice = sorted(set(shrunk) & set(given)) if isinstance(given, dict) else []
        if twice:
            raise ValueError(f"fit #{data.get('seq')} holds the rows of {', '.join(twice)} both "
                             f"as effects and as a penalized family's")
        if not shrunk:
            return data
        return {**data, "effects": tabled, "penalized": {**given, **shrunk}}

    @model_validator(mode="after")
    def _markers_or_why(self) -> "Attempt":
        if (self.markers is None) == (self.markers_why is None):
            raise ValueError(f"fit #{self.seq} holds its live markers or why it has none, not "
                             f"both and not neither")
        if self.markers is not None:
            _once(self.markers, f"fit #{self.seq}")
            if any(m.timing != "live" for m in self.markers):
                raise ValueError(f"fit #{self.seq}'s markers are live markers: a post hoc one is "
                                 f"not known at a fit")
        return self


class SignedChange(_Strict):
    """One family's signed change in the primary score from the fit before (``against``),
    computed from the two fits' per-fold scores (``signed.change``, SWARM_PLAN §7): ``paired``
    over the same folds, else unpaired with each arm's Nadeau–Bengio inflation; ``difference`` is
    this fit's mean less the earlier's on the metric's scale, ``better`` whether this fit scored
    better by the metric's direction; ``half_width`` is half the earlier fit's own interval
    width; ``test_share`` is each arm's n_test/n_train (this fit's first), the engine's measured
    share on the folds the split drew (``signed.test_share``). Without a number, ``why``: among
    others, folds whose share the split's facts do not fix (time-ordered, internal–external,
    grouped), a pooled score that is not the mean of its fold scores, and a change with no
    fold-to-fold spread."""

    family: str
    metric: str
    against: str
    paired: bool
    difference: float | None
    better: bool | None
    se: float | None
    df: float | None
    ci_low: float | None
    ci_high: float | None
    excludes_zero: bool | None
    folds: list[int] | None
    half_width: float | None
    test_share: list[float] | None = None
    why: str | None

    @model_validator(mode="after")
    def _number_or_why(self) -> "SignedChange":
        if (self.difference is None) == (self.why is None):
            raise ValueError("a signed change holds its number or why it has none, not both and "
                             "not neither")
        return self


# ── result, claims, surfacings, export, replication ──────────────────────────


class Estimate(_Strict):
    """One number the app reports for the estimand (``assemble.result_of``), on the measure's
    scale (``Result.scale``): under a marginal measure one contrast of the effects stage's
    standardization (g-computation) as ``GET …/results`` serves it, its ``rd`` or ``rr`` with the
    bootstrap's percentile interval; under a conditional measure one exposure row of the locked
    fit as ``GET …/fits`` serves it, on a ratio scale its ``ratio`` columns (exp of the coefficient
    and of its interval's ends), on a difference scale the coefficient itself. ``why``: why it has
    no number, or no interval, in the engine's words where it gives them."""

    family: str
    feature: str | None  # the exposure's row; a marginal contrast's exposure
    setting: str | None = None  # a marginal contrast: the settings it compares, as served
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    why: str | None = None


class Result(_Strict):
    """The reported result, once (SWARM_PLAN §4): under Predict the held-out scores the first
    opening of the seal recorded (``open_seal``'s ``scores``), from the fit it was recorded on, with
    the intervals beside them only while the scores served are the opening's; under Estimate the
    number the app reports for the estimand the plan's lock in force recorded (``measure``, on its
    ``scale``), from the fit the lock was recorded on, the one fit record ``GET …/fits`` marks so
    (``Attempt.locked``), under the lock's digest: each family's :class:`Estimate`, read where
    ``source`` says (``"results"``: ``GET …/results``' effects stage, served for the fit
    ``served_fit``; ``"fits"``: the locked fit's record). The fit's coefficient rows stay its
    attempt's (``Attempt.effects``). ``refused`` says why there is none, and ``refused_kind`` which
    kind of none it is (:data:`RefusedKind`): a check judges a result none was reported or served
    for as refused, and does not judge one it cannot know."""

    purpose: str | None
    attempt: str | None  # the fit record the result is from
    plan_digest: str | None
    decisions: list[int] | None  # the records in force at that fit
    opening: int | None = None  # Predict: the opening's record seq
    family: str | None = None  # Predict: the final model declared at the opening
    metric: str | None = None  # Predict: the primary metric
    n_holdout: int | None = None
    scores: dict[str, dict[str, float | None]] | None = None
    intervals: dict[str, Any] | None = None
    intervals_why: str | None = None
    lock: int | None = None  # Estimate: the lock's record seq
    measure: str | None = None  # Estimate: the estimand's effect measure, as the lock's plan holds it
    scale: Scale | None = None  # ... its scale, and whether it is conditional or marginal
    conditioning: str | None = None
    estimates: list[Estimate] | None = None
    source: Literal["results", "fits"] | None = None  # where the estimates were read
    served_fit: str | None = None  # source "results": the fit key the results served them with
    refused: str | None = None
    refused_kind: RefusedKind | None = None

    @model_validator(mode="after")
    def _numbers_or_refused(self) -> "Result":
        has = bool(self.scores) or any(e.estimate is not None for e in self.estimates or ())
        if has == (self.refused is not None):
            raise ValueError("a result holds its numbers or why it has none, not both and not "
                             "neither")
        if (self.refused is None) != (self.refused_kind is None):
            raise ValueError("a result with no numbers says which kind of none it is, and only "
                             "then")
        if self.estimates and (self.measure is None or self.scale is None or self.source is None):
            raise ValueError("a result's estimates name their measure, its scale and where they "
                             "were read")
        return self


class Claim(_Strict):
    """One render of a registry claim, as ``GET …/analyst-log`` serves it."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    at: datetime
    id: str
    claim_register: str = Field(alias="register")
    status: str | None
    grade: str | None
    fallback: str | None
    version: int | None
    hash: str | None
    refused: str | None


class Refusal(_Strict):
    """A refusal the driver met (its ``commands.jsonl``), and the command that followed it."""

    at: datetime | None
    command: str
    code: str
    message: str
    exits: list[str]
    then: str | None  # the next command the driver ran; null when none followed
    # Who refused: the server at the answer, the driver, or the engine at fit time (a stage's
    # typed refusal, ``stage`` the stage that refused).
    by: Literal["server", "driver", "engine"] | None = None
    stage: str | None = None


class FailedFit(_Strict):
    """A fit that failed while the driver watched (a command whose current fit, or a stage the
    open question waits on, ended in error), classed by the stage's status alone: ``refused``
    where it served a typed refusal (``refusal``: its code, words and ways forward, as the engine
    wrote them), else ``failed`` (an error the engine did not type as a refusal)."""

    at: datetime | None
    command: str
    stage: str
    key: str | None
    error: str
    kind: Literal["refused", "failed"]
    refusal: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _classed_by_the_refusal(self) -> "FailedFit":
        if (self.kind == "refused") != (self.refusal is not None):
            raise ValueError("a failed fit is refused exactly where the engine served a refusal")
        if self.refusal is not None and not {"code", "message", "exits"} <= set(self.refusal):
            raise ValueError("a fit's refusal holds its code, its words and its ways forward")
        return self


class Surfacings(_Strict):
    refusals: list[Refusal]
    concerns: list[dict[str, Any]]  # the materiality ledger's rows, as served
    failed_fits: list[FailedFit] = []


class Export(_Strict):
    reached: bool  # the analyst's own ``export`` command returned the bundle
    sha256: str | None  # of the bundle the server serves after the run
    methods: str | None  # its methods.md
    checklist: dict[str, Any] | None  # its checklist's JSON report
    refused: dict[str, Any] | None  # the server's refusal of the export after the run

    @model_validator(mode="after")
    def _bundle_or_refused(self) -> "Export":
        if (self.sha256 is None) == (self.refused is None):
            raise ValueError("an export holds the bundle's SHA-256 or the refusal, not both and "
                             "not neither")
        return self


WantKind = Literal[V.WANT_KINDS]  # type: ignore[valid-type]


class Want(_Strict):
    """An option, a check, a reading or an eligibility rule the engine lacks, as the analyst named
    it with the driver's ``want``: its kind, what, why and the marker it answers to."""

    at: datetime
    kind: WantKind
    what: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    marker: MarkerName


class Account(_Strict):
    """The analyst's end-of-run account (SWARM_PLAN §4), as the driver's ``account`` recorded it:
    its source is the driver, never the agent's final message. ``after_results``: a ``results``
    command had returned before it; ``after_seal``: the held-out rows were open. It sits beside the
    engine's record, never in its place: context for hypotheses, never evidence."""

    at: datetime
    text: str = Field(min_length=1)
    after_results: bool
    after_seal: bool
    source: Literal["driver"] = "driver"


class Replication(_Strict):
    paper: str
    stated: list[dict[str, Any]]
    unstated: list[dict[str, Any]]
    published: dict[str, Any] | None = None
    ours: dict[str, Any] | None = None


class Record(_Strict):
    version: Literal[2]
    run: Run
    dataset: Dataset
    question: Question
    regime: Regime
    decisions: list[Decision]
    attempts: list[Attempt]
    result: Result
    claims: list[Claim]
    surfacings: Surfacings
    export: Export
    replication: Replication | None = None
    wants: list[Want] = []
    account: Account | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_replay(cls, data: Any) -> Any:
        """A record written before ``replay`` was dropped holds it null, which records nothing:
        read as one without it (the module docstring). One that holds a replay keeps it, and
        fails."""
        if isinstance(data, dict) and "replay" in data and data["replay"] is None:
            return {k: v for k, v in data.items() if k != "replay"}
        return data

    @model_validator(mode="after")
    def _references_hold(self) -> "Record":
        seqs = [d.seq for d in self.decisions]
        if seqs != sorted(set(seqs)):
            raise ValueError("the decisions' seqs repeat or are out of order")
        known = set(seqs)
        for a in self.attempts:
            named = set(a.decisions or ()) | set(a.changed or ()) | {r.seq for r in a.reasons}
            named |= {n for s in a.withdrawn + a.reinstated for n in (s.seq, s.by)
                      if n is not None}
            stray = sorted(named - known)
            if stray:
                raise ValueError(f"fit #{a.seq} names records {stray} the decision log does not "
                                 f"hold")
        stray = sorted(set(self.result.decisions or ()) - known)
        if stray:
            raise ValueError(f"the result reads records {stray} the decision log does not hold")
        ids = {a.id for a in self.attempts}
        if self.result.attempt is not None and self.result.attempt not in ids:
            raise ValueError("the result names a fit no attempt holds")
        for d in self.decisions:
            if d.materiality.attempt is not None and d.materiality.attempt not in ids:
                raise ValueError(f"record #{d.seq}'s materiality names a fit no attempt holds")
            if d.tag is not None and d.source != "analyst":
                raise ValueError(f"record #{d.seq} was recorded by TurboTab itself, so it holds "
                                 f"no analyst's tag")
        for a in self.attempts:
            stray = sorted({c.against for c in a.change} - ids)
            if stray:
                raise ValueError(f"fit #{a.seq}'s change is against fits {stray} no attempt holds")
        return self


def validate(data: Any) -> Record:
    """``data`` as a :class:`Record`, or pydantic's ``ValidationError`` saying what fails."""
    return Record.model_validate(data)


# A time as a record's JSON holds it (``datetime``'s ISO form, the driver's and the server's), in a
# field of its own or inside words (a refusal that says when the account was given).
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?")
# Where the redacted view's order of times starts: its n-th distinct time is n seconds after it.
ORDER_FROM = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _instant(text: str) -> datetime:
    t = datetime.fromisoformat(text.replace(" ", "T"))
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _stamps(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _stamps(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _stamps(v)]
    return STAMP.findall(obj) if isinstance(obj, str) else []


def _ordered(obj: Any, place: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _ordered(v, place) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ordered(v, place) for v in obj]
    return STAMP.sub(lambda m: place[m.group()], obj) if isinstance(obj, str) else obj


def order_only(data: Any) -> Any:
    """``data`` (a record's JSON) with every time it holds, in a field or inside words, replaced by
    its place in the record's order of times: the n-th distinct instant is ``ORDER_FROM`` plus n
    seconds, so two times compare as they did and no gap between them is kept."""
    instants = {t: _instant(t) for t in _stamps(data)}
    rank = {t: n for n, t in enumerate(sorted(set(instants.values())), start=1)}
    place = {text: (ORDER_FROM + timedelta(seconds=rank[t])).isoformat().replace("+00:00", "Z")
             for text, t in instants.items()}
    return _ordered(data, place)


def neutral_id(run_id: str) -> str:
    """A run's id as a blind reader is shown it: ``redacted-`` and 32 hex digits of the id's
    SHA-256, the same for the same run and naming neither its label nor its secret."""
    return f"{REDACTED}-{hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:32]}"


def _named_by(run_id: str) -> list[str]:
    """What names the run in a record's words: its id, and its label where the id is the driver's
    (``driver.new_run_id``: the label, a hyphen and a 32-hex-digit secret), longest first."""
    label, _, secret = run_id.rpartition("-")
    return [run_id, label] if label and re.fullmatch(r"[0-9a-f]{32}", secret) else [run_id]


def _renamed(obj: Any, names: list[str], token: str) -> Any:
    """``obj`` with each of ``names`` that stands as a word of its own in its strings replaced by
    ``token``."""
    if isinstance(obj, dict):
        return {k: _renamed(v, names, token) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_renamed(v, names, token) for v in obj]
    if isinstance(obj, str):
        for name in names:
            obj = re.sub(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_])", token, obj)
        return obj
    return obj


def redacted(record: Record | Any) -> dict[str, Any]:
    """The view of ``record`` a blind reader is shown (SWARM_PLAN §7: the stuck runs' classing,
    the referee): the analyst's model and persona, its effort, its prompt version (which can name
    the persona) and its spend (whose size can tell the model) taken out, ``run.redacted`` set;
    and every time it holds, the run's start and end and each ``at`` (of its decisions, previews,
    fits, claims, refusals, failed fits, tags, wants and account) included, turned into its order
    only (:func:`order_only`), since with an agent analyst the gaps between its commands are its
    model's response time. The run's start order is taken out too, since a reader who knows the
    schedule could name a run's model from it: its id, whose label counts the runs started before
    it, is a neutral token wherever it stands (:func:`neutral_id`; its label alone too), and its
    server's PID and port, which grow with the start order, are dropped (its mode and workers
    kept). It validates as a record, and the checks read it as they read the record. What the
    analyst wrote (its reasons, wants and account) is kept as written but for a time or its run's
    name in it: a reader is told the model was redacted, not that no word of the analyst's can
    reveal it."""
    rec = record if isinstance(record, Record) else validate(record)
    data = rec.model_dump(mode="json", by_alias=True)
    data = _renamed(data, _named_by(rec.run.id), neutral_id(rec.run.id))
    server = data["run"].get("server")
    data["run"].update(analyst_model=REDACTED, persona=None, effort=None, prompt_version=REDACTED,
                       spend=None, spend_why=REDACTED_SPEND, redacted=True,
                       server=None if server is None else {
                           k: v for k, v in server.items() if k not in ("pid", "port")})
    data = order_only(data)
    validate(data)
    return data


__all__ = ["Account", "Answer", "Attempt", "Claim", "Dataset", "DatasetFile", "DeclaredMarker",
           "Decision", "Estimate", "Examined", "Export", "FailedFit", "Marker", "Materiality",
           "NULL_OF", "ORDER_FROM", "Persona", "Question", "REDACTED", "REDACTED_SPEND", "Reason",
           "Record", "RefusedKind", "Refusal", "Regime", "Replication", "Result", "Run",
           "STAMP", "Scale", "Shift", "SignedChange", "Spend", "Surfacings", "Tag", "VERSION",
           "Want", "neutral_id", "order_only", "redacted", "validate"]
