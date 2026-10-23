"""The structured proposal (SWARM_PLAN §9 step 1): what the theorist proposes, in fields a script reads.

A proposal is a surfacing claim (SWARM_PLAN §1): in a region of the live markers, a decision moves
the result by a material amount, or does not. Its fields:

* ``id`` (unique in its proposals file) and ``author`` (the theorist's agent id, so the registry's
  referee gate can tell its author from its referees);
* ``decision``: a decision kind the engine records (``turbotab.core.decisions``), other than those
  that record the protocol and not a modeling choice a sweep changes (:data:`PROTOCOL`);
* ``region``: one bound or more, each a **live** marker of the vocabulary (``vocabulary.LIVE``)
  with its comparison (:data:`OPS`) and threshold, each marker once. A promoted claim keys the app
  through a predicate on markers read before the decision (SWARM_PLAN §1), so a post hoc marker
  keys nothing and a declared one is the documentation's, not read live: **a proposal with no live
  marker is refused as not a surfacing claim**. A share's threshold lies within 0 and 1;
* ``direction``: ``material`` (in the region the decision moves the result by a material amount)
  or ``not_material``; ``size``: how far, in the loss's units, at least 0;
* ``loss``: the purpose, with the primary score under Predict (a metric the engine scores) or the
  estimand's measure under Estimate (one the engine estimates);
* ``refits``: the decision's refits in the region, **not counted** or **counted as the sweeps count
  them**. SWARM_PLAN §9 asks a proposal for the share of its decision's refits refused in its
  region, and only the sweeps' tables hold those refits, which the claim cycle's tools do not
  read: a proposal from them leaves its refits not counted (``null``), and the validator labels it
  so in its own words (:data:`NOT_COUNTED`), never with a zero. A count is taken only where the
  sweeps' own count for the proposal is given (:func:`validate`'s ``sweeps``) and it states that
  count; else it is refused as a count no one checked. A counted proposal states how many refits,
  how many the engine refused, **the share refused** (refused unless it is the counts' own, to
  three decimals, and then kept as the counts give it; none where there was no refit), each
  refusal's code with its count, and **whether the refusals sit in one marker region**
  (``one_region``, and that ``region``, drawn as a claim's is), said exactly where some refit was
  refused. A refusal that is the engine's gap (:data:`ENGINE_GAPS`, the code the engine refuses an
  empty selection with) is not part of any claim (SWARM_PLAN §9), so a count holding one is
  refused under every decision; a refusal that is the method's own limit (backward elimination
  where the columns outnumber the fold's rows) is part of the claim, and counts;
* ``evidence``: the datasets the proposal rests on, by the labels the reading tools give them
  (``view.py``), and its runs;
* ``scope``: **a claim about variable selection** (``set_selection``) only. Until the engine fits
  the intercept alone where a selection keeps no predictor (SWARM_PLAN §8), such a claim is scoped
  to the datasets where every swept selection kept a predictor, and says so (SWARM_PLAN §9): it is
  refused without a scope and where its evidence names a dataset outside it. The validator reads
  no sweep, so it cannot check that the scope's datasets are those, and its words (``says``) say
  the scope is its author's and not checked against the sweeps. The words are the validator's,
  never the proposal's;
* ``basis``: in the theorist's words, what in the corpus the proposal rests on;
* ``status`` and ``grade``: **proposed and unsupported**, the only ones a proposal enters with
  (CLAIMS_REGISTRY §8): only the mechanical tests, the two-referee gate and the basis audit move it.

Every model here forbids a field it does not declare, so a proposal carrying a validity, or any
field a theorist invents, is refused. A refusal says what fails, in words (:class:`ProposalRefused`).

**The proposals file** is JSON lines, one validated proposal a line, named ``.jsonl``:
:func:`append` validates a proposal and appends it once under its id, and :func:`read_proposals`
reads every line back through the validator. A file that is not one, the shuffled copy's sealed
file among them, is neither read nor appended to, and its contents are never repeated.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (BaseModel, ConfigDict, Field, ValidationError, ValidationInfo,
                      field_validator, model_validator)

from research.swarm import vocabulary as V

STATUS = "proposed"
GRADE = "unsupported"
# A bound's comparisons: below (<), at most (<=), above (>), at least (>=) a number; between two
# numbers, the lower included and the upper not; is one value; in a list of values.
OPS = ("below", "at_most", "above", "at_least", "between", "is", "in")
NUMERIC = ("below", "at_most", "above", "at_least")
DIRECTIONS = ("material", "not_material")
# The decision kinds that record the protocol, not a modeling choice a sweep changes: a revert of
# an earlier record, the seal's opening and a new seal, the plan's lock, a view of the outcome.
PROTOCOL = ("revert", "open_seal", "reseal", "lock_plan", "view_outcome")
SELECTION = "set_selection"
# A fit-time refusal that is the engine's gap, not the method's own limit (SWARM_PLAN §9): its code
# (``turbotab.core.models.variable_selection``, a selection that keeps no predictor) and what the
# engine lacks.
ENGINE_GAPS = {"nothing_kept": "the intercept-only fit where a selection keeps no predictor "
                               "(SWARM_PLAN §8, after the pilot)"}
SHARE_DIGITS = 3  # a stated share is the counts' own to this many decimals
# What the validator writes beside refits not counted: what the proposal lacks, and why.
NOT_COUNTED = ("Not counted: the proposal states no count of its decision's refits in its region, "
               "so no share of them refused, which SWARM_PLAN §9 asks of a proposal; the claim "
               "cycle's tools read no sweeps table to count them.")
# The counts a sweeps table gives for a proposal, and a count states.
COUNTS = ("refits", "refused", "codes")


class ProposalRefused(ValueError):
    """A proposal, or a proposals file, the validator refuses, in words."""


@lru_cache(maxsize=1)
def decision_kinds() -> frozenset[str]:
    """Every decision kind the engine records (``turbotab.core.decisions.Decision``)."""
    from typing import get_args

    from turbotab.core.decisions import Decision

    return frozenset(c.model_fields["kind"].default for c in get_args(get_args(Decision)[0]))


@lru_cache(maxsize=1)
def metrics() -> frozenset[str]:
    """Every metric the engine scores, under any task (``models.metrics.METRICS``)."""
    from turbotab.core.models.metrics import METRICS

    return frozenset(m for names in METRICS.values() for m in names)


@lru_cache(maxsize=1)
def measures() -> frozenset[str]:
    """Every effect measure the engine estimates (``estimand.MEASURE_WORDS``)."""
    from turbotab.core.estimand import MEASURE_WORDS

    return frozenset(MEASURE_WORDS)


def _listed(names: Any) -> str:
    return ", ".join(sorted(names))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))


class Bound(_Strict):
    """One live marker of a region, its comparison and its threshold."""

    marker: str
    op: Literal[OPS]  # type: ignore[valid-type]
    threshold: Any

    @model_validator(mode="after")
    def _live_and_well_formed(self) -> "Bound":
        name, op, t = self.marker, self.op, self.threshold
        fixed = V.BY_NAME.get(name)
        if fixed is None:
            raise ValueError(V.unknown(name, V.LIVE))
        if fixed.timing == "post_hoc":
            raise ValueError(f"`{name}` is known only after a run, so it keys nothing (SWARM_PLAN "
                             f"§1); a region is drawn on live markers.")
        if fixed.timing == "declared":
            raise ValueError(f"`{name}` is declared from the dataset's documentation, not read "
                             f"live before the decision; a region is drawn on live markers.")
        if op in NUMERIC:
            if not _number(t):
                raise ValueError(f"`{name}` {op} takes a number, not {t!r}.")
            numbers = [t]
        elif op == "between":
            if not (isinstance(t, list) and len(t) == 2 and all(_number(x) for x in t)):
                raise ValueError(f"`{name}` between takes two numbers, the lower bound below its "
                                 f"upper, not {t!r}.")
            if not t[0] < t[1]:
                raise ValueError(f"`{name}` between takes two numbers, the lower bound below its "
                                 f"upper, not {t!r}.")
            numbers = list(t)
        elif op == "is":
            if not _scalar(t):
                raise ValueError(f"`{name}` is takes one value, a word, a number or true or false, "
                                 f"not {t!r}.")
            numbers = [t] if _number(t) else []
        else:
            if not (isinstance(t, list) and t and all(_scalar(x) for x in t)):
                raise ValueError(f"`{name}` in takes a list of at least one value, each a word, a "
                                 f"number or true or false.")
            numbers = [x for x in t if _number(x)]
        if any(not math.isfinite(x) for x in numbers):
            raise ValueError(f"`{name}`'s threshold is a finite number.")
        if name.endswith("_share") and any(not 0 <= x <= 1 for x in numbers):
            raise ValueError(f"`{name}` is a share, so its threshold lies between 0 and 1.")
        return self


def _once(bounds: list[Bound], where: str) -> None:
    names = [b.marker for b in bounds]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"{where} names `{twice[0]}` more than once; a band is one bound, "
                         f"`between`.")


class Loss(_Strict):
    """The loss a proposal is judged on: under Predict the primary score, under Estimate the
    estimand's effect measure, each as the engine names it."""

    purpose: Literal["prediction", "inference"]
    metric: str | None = None
    measure: str | None = None

    @model_validator(mode="after")
    def _the_engines(self) -> "Loss":
        if self.purpose == "prediction":
            if self.metric not in metrics() or self.measure is not None:
                raise ValueError(f"A prediction's loss names its metric and no measure: one the "
                                 f"engine scores, {_listed(metrics())}.")
        elif self.measure not in measures() or self.metric is not None:
            raise ValueError(f"An inference's loss names the estimand's measure and no metric: one "
                             f"the engine estimates, {_listed(measures())}.")
        return self


def _share_of(refused: int, refits: int) -> float:
    return refused / refits


class Refits(_Strict):
    """The decision's refits in the proposal's region (the module docstring): not counted
    (``refits`` null; every other count null, and ``why`` the validator's :data:`NOT_COUNTED`), or
    counted as the sweeps count them (SWARM_PLAN §6: each refit scored, refused with the engine's
    reason, or failed): how many, how many refused, the share refused, each refusal's code and
    count, and where the refusals sit (``why`` null)."""

    refits: int | None = Field(default=None, ge=0)
    refused: int | None = Field(default=None, ge=0)
    share: float | None = None
    codes: dict[str, int] = {}
    one_region: bool | None = None
    region: list[Bound] | None = None
    why: str | None = None

    @model_validator(mode="after")
    def _counted(self) -> "Refits":
        n, k = self.refits, self.refused
        if n is None:
            stated = [f for f in ("refused", "share", "one_region", "region")
                      if getattr(self, f) is not None] + (["codes"] if self.codes else [])
            if stated:
                raise ValueError(f"Its refits are not counted (refits null), so it states no "
                                 f"{', '.join(stated)}: each is read from a count.")
            self.why = NOT_COUNTED
            return self
        self.why = None
        if k is None:
            raise ValueError(f"It counts {n} refit{'' if n == 1 else 's'}: say how many were "
                             f"refused (refused).")
        if k > n:
            raise ValueError(f"It counts more refits refused than refitted ({k} of {n}).")
        if n == 0:
            if self.share is not None:
                raise ValueError("With no refit in its region there is no share refused; the "
                                 "share is null.")
        else:
            exact = _share_of(k, n)
            if self.share is None or round(abs(self.share - exact), 9) >= 0.5 * 10 ** -SHARE_DIGITS:
                stated = "null" if self.share is None else f"{self.share:g}"
                raise ValueError(f"{k} of {n} refits refused is a share of "
                                 f"{round(exact, SHARE_DIGITS):g}, not {stated}.")
            self.share = exact
        if any(c < 1 for c in self.codes.values()):
            raise ValueError("Each refusal code counts at least one refit.")
        total = sum(self.codes.values())
        if total != k:
            raise ValueError(f"Its codes count {total} refusal{'' if total == 1 else 's'}, and it "
                             f"says {k} {'was' if k == 1 else 'were'} refused.")
        if k == 0:
            if self.one_region is not None or self.region is not None:
                raise ValueError("It says where the refusals sit, but no refit was refused; "
                                 "one_region and region are null.")
            return self
        if self.one_region is None:
            raise ValueError(f"{k} refit{' was' if k == 1 else 's were'} refused: say whether "
                             f"they sit in one marker region (one_region), and if so which "
                             f"(region).")
        if self.one_region and not self.region:
            raise ValueError("The refusals sit in one marker region: name that region (region), "
                             "drawn on live markers as a claim's is.")
        if not self.one_region and self.region is not None:
            raise ValueError("The refusals do not sit in one region, so the region is null.")
        if self.region:
            _once(self.region, "The refusals' region")
        return self


def _counts(c: dict[str, Any]) -> str:
    n, k = c.get("refits"), c.get("refused")
    codes = ", ".join(f"{code} {m}" for code, m in sorted((c.get("codes") or {}).items()))
    return (f"{n} refit{'' if n == 1 else 's'}, {k} refused"
            + (f" ({codes})" if codes else ""))


class Scope(_Strict):
    """A claim about variable selection's scope (SWARM_PLAN §9): the datasets its author states are
    those where every swept selection kept a predictor, and the validator's words for it."""

    datasets: list[str] = Field(min_length=1)
    says: str | None = None


def scope_words(datasets: list[str]) -> str:
    """What a selection claim's scope says, from its datasets: its author's, not checked."""
    named = sorted(set(datasets))
    n = len(named)
    those = ("the 1 dataset {}, which it states is the one" if n == 1 else
             f"the {n} datasets {{}}, which it states are those").format(", ".join(named))
    return (f"Scoped by its author to {those} where every swept selection kept a predictor; that "
            f"is not checked against the sweeps. Until the engine fits the intercept alone where "
            f"a selection keeps none (SWARM_PLAN §8), the claim says nothing of a dataset where "
            f"one kept none.")


class Evidence(_Strict):
    """What the proposal rests on: the datasets, by the labels the reading tools give, and runs."""

    datasets: list[str] = Field(min_length=1)
    runs: list[str] = []


ENTRY = (f"A proposal enters as {STATUS} and {GRADE} (CLAIMS_REGISTRY §8): only the mechanical "
         f"tests, the two-referee gate and the basis audit move it.")


class Proposal(_Strict):
    id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    decision: str
    region: list[Bound]
    direction: Literal[DIRECTIONS]  # type: ignore[valid-type]
    size: float = Field(ge=0, allow_inf_nan=False)
    loss: Loss
    refits: Refits
    evidence: Evidence
    scope: Scope | None = None
    basis: str = Field(min_length=1)
    status: str = STATUS
    grade: str = GRADE

    @model_validator(mode="before")
    @classmethod
    def _enters_proposed(cls, data: Any) -> Any:
        if isinstance(data, dict) and (data.get("status", STATUS) != STATUS
                                       or data.get("grade", GRADE) != GRADE):
            raise ValueError(ENTRY)
        return data

    @field_validator("refits", mode="before")
    @classmethod
    def _null_is_not_counted(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _a_surfacing_claim(self, info: ValidationInfo) -> "Proposal":
        if not self.region:
            raise ValueError("The region names no live marker, so it is not a surfacing claim: a "
                             "claim keys the app on live markers read before the decision "
                             "(SWARM_PLAN §1, §9).")
        _once(self.region, "The region")
        if self.decision not in decision_kinds():
            raise ValueError(f"`{self.decision}` is not a decision the engine records; it is one "
                             f"of {_listed(decision_kinds() - set(PROTOCOL))}.")
        if self.decision in PROTOCOL:
            raise ValueError(f"`{self.decision}` records the protocol, not a modeling choice a "
                             f"sweep changes.")
        self._checked_against((info.context or {}).get("sweeps"))
        gaps = {c: k for c, k in self.refits.codes.items() if c in ENGINE_GAPS}
        if gaps:
            code, k = sorted(gaps.items())[0]
            gap = (f"{k} of its refits {'was' if k == 1 else 'were'} refused with `{code}`, the "
                   f"engine's gap ({ENGINE_GAPS[code]}), not the method's own limit, so not part "
                   f"of a claim (SWARM_PLAN §9)")
            if self.decision == SELECTION:
                raise ValueError(f"{gap}: scope the claim to the datasets where every swept "
                                 f"selection kept a predictor, whose refits hold no such refusal.")
            raise ValueError(f"{gap}: a claim about `{self.decision}` counts no refit refused "
                             f"for the engine's gap.")
        if self.decision != SELECTION:
            if self.scope is not None:
                raise ValueError(f"Only a claim about variable selection (`{SELECTION}`) is scoped "
                                 f"to the datasets where every swept selection kept a predictor; "
                                 f"this one is about `{self.decision}`.")
            return self
        if self.scope is None:
            raise ValueError(f"A claim about variable selection is scoped to the datasets where "
                             f"every swept selection kept a predictor, until the engine fits the "
                             f"intercept alone where a selection keeps none (SWARM_PLAN §8, §9): "
                             f"name them in scope.datasets.")
        outside = sorted(set(self.evidence.datasets) - set(self.scope.datasets))
        if outside:
            raise ValueError(f"Its evidence names {', '.join(outside)}, outside its scope: a claim "
                             f"about variable selection rests only on the datasets where every "
                             f"swept selection kept a predictor.")
        self.scope.says = scope_words(self.scope.datasets)
        return self

    def _checked_against(self, sweeps: Any) -> None:
        """A count is the sweeps' own for this proposal, or refused (the module docstring)."""
        n = self.refits.refits
        if n is None:
            return
        counts = f"It counts {n} refit{'' if n == 1 else 's'}"
        if not isinstance(sweeps, dict):
            raise ValueError(f"{counts}, but no sweeps table is read here, so nothing checks the "
                             f"count: leave the refits not counted (refits null).")
        theirs = sweeps.get(self.id)
        if not isinstance(theirs, dict):
            raise ValueError(f"{counts}, but the sweeps table read here holds no count for "
                             f"{self.id}, so nothing checks it: leave the refits not counted "
                             f"(refits null).")
        stated = {"refits": n, "refused": self.refits.refused, "codes": dict(self.refits.codes)}
        given = {k: (dict(theirs.get(k) or {}) if k == "codes" else theirs.get(k))
                 for k in COUNTS}
        if stated != given:
            raise ValueError(f"It counts {_counts(stated)}; the sweeps count {_counts(given)}.")


def _words(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        loc = ".".join(str(x) for x in err.get("loc") or ())
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "The proposal is refused: " + "; ".join(parts)


Sweeps = dict[str, dict[str, Any]]


def validate(data: Any, sweeps: Sweeps | None = None) -> Proposal:
    """``data`` as a :class:`Proposal`, or :class:`ProposalRefused` saying what fails. ``sweeps``,
    where a sweeps table has been read, gives for each proposal id its decision's refits in its
    region as the sweeps count them (:data:`COUNTS`: ``refits``, ``refused`` and ``codes``); a
    proposal that counts its refits is taken only where it states that count. The claim cycle's
    tools read no sweeps table, so they give none."""
    if not isinstance(data, dict):
        raise ProposalRefused("The proposal is refused: a proposal is one JSON object.")
    try:
        return Proposal.model_validate(data, context={"sweeps": sweeps})
    except ValidationError as exc:
        raise ProposalRefused(_words(exc)) from None


def proposable(kind: str) -> bool:
    """Whether a theorist may propose about the decision ``kind``: one the validator takes (a kind
    the engine records, not one that records the protocol), and not variable selection, which the
    theorist's prompt sets aside, since its scope is the sweeps' to show and the tools read none."""
    return kind in decision_kinds() and kind not in PROTOCOL and kind != SELECTION


# ── the proposals file ───────────────────────────────────────────────────────


def _not_sealed(path: Path, text: str) -> None:
    from research.swarm.claim_cycle.view import SEALED_REFUSAL, is_sealed

    try:
        whole = json.loads(text)
    except ValueError:
        return
    if is_sealed(whole):
        raise ProposalRefused(SEALED_REFUSAL)


def read_proposals(path: str | Path, sweeps: Sweeps | None = None) -> list[Proposal]:
    """Every proposal of the proposals file at ``path``, each through the validator (beside
    ``sweeps``, :func:`validate`'s), in its order; a file that is not one is refused, and its
    contents are not repeated."""
    path = Path(path)
    if not path.is_file():
        raise ProposalRefused(f"{path.name} is not a proposals file: there is no file there.")
    text = path.read_text("utf-8")
    _not_sealed(path, text)
    out: list[Proposal] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except ValueError:
            raise ProposalRefused(f"Line {i} of {path.name} is not a proposal: it is not one JSON "
                                  f"object, so {path.name} is not a proposals file.") from None
        try:
            got = validate(data, sweeps)
        except ProposalRefused as exc:
            raise ProposalRefused(f"Line {i} of {path.name}: {exc}") from None
        if got.id in {p.id for p in out}:
            raise ProposalRefused(f"Line {i} of {path.name} proposes {got.id} again.")
        out.append(got)
    return out


def append(path: str | Path, data: Any, sweeps: Sweeps | None = None) -> Proposal:
    """``data`` validated (beside ``sweeps``, :func:`validate`'s) and appended to the proposals
    file at ``path`` (created if its folder exists), once under its id; refused, and nothing
    written, otherwise."""
    path = Path(path)
    existing = read_proposals(path, sweeps) if path.exists() else []
    if path.suffix != ".jsonl":
        raise ProposalRefused(f"A proposals file is JSON lines, named .jsonl, not {path.name}.")
    if not path.parent.is_dir():
        raise ProposalRefused(f"The proposals file's folder, {path.parent}, does not exist.")
    got = validate(data, sweeps)
    if got.id in {p.id for p in existing}:
        raise ProposalRefused(f"{got.id} is proposed already in {path.name}; a proposal is "
                              f"appended once, under an id of its own.")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(got.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return got


__all__ = ["Bound", "COUNTS", "DIRECTIONS", "ENGINE_GAPS", "ENTRY", "Evidence", "GRADE", "Loss",
           "NOT_COUNTED", "NUMERIC", "OPS", "PROTOCOL", "Proposal", "ProposalRefused", "Refits",
           "SELECTION", "STATUS", "Scope", "Sweeps", "append", "decision_kinds", "measures",
           "metrics", "proposable", "read_proposals", "scope_words", "validate"]
