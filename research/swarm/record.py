"""The corpus record: one JSON document per swarm run (DRIVER_PLAN D3; SWARM_PLAN §4).

A record is assembled after the run from the server's own data (``assemble.py``), never from the
analyst's account of it, and validated here before it is kept. Its fields are SWARM_PLAN §4's:

* ``run``: the run's id and design, the engine's commit, the analyst's model, effort and prompt
  version, when it started and ended, its spend (fresh and cached tokens, counted from the
  analyst's transcript once per message) and its fits and their seconds;
* ``dataset``: the files the run read, each with its SHA-256 and where it came from, and, from the
  intake, whether the dataset is a known-answer control or a null decoy (never on the analyst's
  card);
* ``question``: the goal G (the purpose) and the estimand or intended use Q the analyst declared;
* ``regime``: the engine's readings of the table: rows, rows analyzed, events, blanks, the design,
  the lenses;
* ``decisions``: each decision record of the log with its option, the options examined for it
  (previewed or fitted), its reason (the record's ``note``), its phase, who recorded it and its
  materiality M;
* ``attempts``: each fit's record as ``GET …/fits`` serves it, with the records that changed since
  the fit before it and their reasons;
* ``result``: the reported result, once: the first opening's held-out scores under Predict, the
  locked plan's estimate under Estimate, or why there is none;
* ``claims``: every registry claim rendered for the project, as the analyst log serves it;
* ``surfacings``: each refusal the driver met and the command that followed it, and each concern
  the materiality ledger measured;
* ``export``: the bundle's SHA-256, its methods text and its checklist, and whether the analyst's
  own export reached it;
* ``replication``: for a replication (design 1), the paper's decisions against ours, filled by its
  grader; null otherwise.

**Who fills what.** The analyst fills G, Q, D and O through its answers; the engine fills the
regime R and the materiality M; **no field holds a validity V**. Every model here forbids fields it
does not declare, so a record that carries a validity, or any field a run invents, fails its schema.
Only a mechanical check (``checks.py``) or an independent grader judges a run.

**Materiality is the engine's number.** A decision's ``materiality`` holds either what the engine
measured (``GET …/fits``' paired movement for the fit the decision first took effect in, or
``GET …/mattered``' specifications for a choice the curve varies) or why it has none; a record
holding neither, or both, fails its schema.

**A record that fails its schema is a failed run** (``assemble.write``): it is written to
``research/corpus/failed/`` with the error, and never edited into one that passes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSION = 1

# When a decision was made, by the decision log's own marks (SWARM_PLAN §4's phases):
# "after_seal", recorded after the held-out rows were opened (``post_seal``); "after_estimates",
# recorded with the inference estimates in view (``after_estimates``: after the plan's lock, once
# an estimate was shown under it); "before", neither.
Phase = Literal["before", "after_estimates", "after_seal"]
Control = Literal["known_answer", "null_decoy"]


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
    effort: str | None
    prompt_version: str = Field(min_length=1)
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


class Question(_Strict):
    purpose: str | None  # G
    target: str | None
    estimand: dict[str, Any] | None  # Q under Estimate: the estimand as recorded
    intended_use: dict[str, Any] | None  # Q under Predict: the intended use as recorded
    stated: str | None = None  # the card's question, in its words
    paper: str | None = None  # a replication's paper


class Regime(_Strict):
    """The engine's readings, each its own number as served."""

    n_rows: int | None  # the ingest's rows
    n_analyzed: int | None  # the participant flow's final count
    task: str | None
    event: str | None  # a yes/no outcome's event level
    events: int | None  # rows at the event level, from the outcome's own value counts
    missing: dict[str, int]  # each column with blanks, and how many
    design: dict[str, Any]  # survey, clusters, grain, repeats, unit and design, as recorded
    lenses: list[str]


# ── decisions and attempts ───────────────────────────────────────────────────


class Answer(_Strict):
    """One question a decision answers, and its value: ``<reading>:<column>`` for a reading (the
    keys of a fixture's declared truth, ``truths.Truth``), else the decision's kind and the fields
    that say which question it answers."""

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
    alternative: bool


class Materiality(_Strict):
    """How far the engine measured a decision to move the result, or why it has no measure.

    ``source == "fits"``: ``movement`` is the paired movement ``GET …/fits`` served for the fit
    the decision first took effect in, where it was the one record that changed since the fit
    before. ``source == "mattered"``: ``departures`` are ``GET …/mattered``' specifications that
    change this decision's choices, beside its ``primary``."""

    source: Literal["fits", "mattered"] | None
    attempt: str | None = None
    movement: list[dict[str, Any]] | None = None
    primary: dict[str, Any] | None = None
    departures: list[dict[str, Any]] | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _number_or_why(self) -> "Materiality":
        measured = bool(self.movement) or bool(self.departures)
        if measured == (self.why is not None):
            raise ValueError("a materiality holds the engine's measure or why there is none, not "
                             "both and not neither")
        if measured and self.source is None:
            raise ValueError("a measured materiality names the route that measured it")
        return self


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


class Reason(_Strict):
    seq: int
    note: str


class Attempt(_Strict):
    """One fit's record as ``GET …/fits`` serves it, with ``changed``: the records in force at this
    fit and not at the one before it (every record in force, for the first), and their reasons."""

    id: str
    seq: int = Field(ge=1)
    at: datetime
    purpose: str | None
    target: str | None
    fit_key: str
    plan_digest: str | None
    decisions: list[int] | None
    changed: list[int] | None
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
    effects: dict[str, list[dict[str, Any]] | None]
    movement: list[dict[str, Any]]
    movement_why: str | None
    withheld: str | None


# ── result, claims, surfacings, export, replication ──────────────────────────


class Result(_Strict):
    """The reported result, once (SWARM_PLAN §4): under Predict the held-out scores the first
    opening of the seal recorded (``open_seal``'s ``scores``), with the intervals beside them only
    while the scores served are the opening's; under Estimate each family's exposure rows of the fit
    the plan's lock in force was recorded on, under the lock's digest. ``refused`` says why there
    is none."""

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
    effects: dict[str, list[dict[str, Any]] | None] | None = None
    refused: str | None = None

    @model_validator(mode="after")
    def _numbers_or_refused(self) -> "Result":
        has = bool(self.scores) or bool(self.effects)
        if has == (self.refused is not None):
            raise ValueError("a result holds its numbers or why it has none, not both and not "
                             "neither")
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


class Surfacings(_Strict):
    refusals: list[Refusal]
    concerns: list[dict[str, Any]]  # the materiality ledger's rows, as served


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


class Replication(_Strict):
    paper: str
    stated: list[dict[str, Any]]
    unstated: list[dict[str, Any]]
    published: dict[str, Any] | None = None
    ours: dict[str, Any] | None = None


class Record(_Strict):
    version: Literal[1]
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

    @model_validator(mode="after")
    def _references_hold(self) -> "Record":
        seqs = [d.seq for d in self.decisions]
        if seqs != sorted(set(seqs)):
            raise ValueError("the decisions' seqs repeat or are out of order")
        known = set(seqs)
        for a in self.attempts:
            stray = sorted(set(a.decisions or ()) - known)
            if stray:
                raise ValueError(f"fit #{a.seq} reads records {stray} the decision log does not "
                                 f"hold")
        ids = {a.id for a in self.attempts}
        if self.result.attempt is not None and self.result.attempt not in ids:
            raise ValueError("the result names a fit no attempt holds")
        for d in self.decisions:
            if d.materiality.attempt is not None and d.materiality.attempt not in ids:
                raise ValueError(f"record #{d.seq}'s materiality names a fit no attempt holds")
        return self


def validate(data: Any) -> Record:
    """``data`` as a :class:`Record`, or pydantic's ``ValidationError`` saying what fails."""
    return Record.model_validate(data)


__all__ = ["Answer", "Attempt", "Claim", "Dataset", "DatasetFile", "Decision", "Examined", "Export",
           "Materiality", "Question", "Reason", "Record", "Refusal", "Regime", "Replication",
           "Result", "Run", "Spend", "Surfacings", "VERSION", "validate"]
