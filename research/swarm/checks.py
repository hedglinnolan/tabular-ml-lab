"""The pilot's mechanical checks (DRIVER_PLAN D5; SWARM_PLAN §7), each a function over a record.

    python -m research.swarm.checks [research/corpus]

prints one JSON line per record and check, and counts each failed run (``corpus/failed/``, a record
that failed its schema) as a logging failure. Each check returns an :class:`Outcome`: ``pass``,
``fail`` with each problem named, or ``not_checked`` with what could not be checked and why. A
check never passes what it could not check.

* :func:`seal_integrity`: no re-seal and no second opening of an outcome's held-out rows; no record
  made after the opening (Predict), or after the plan's lock or with the estimates in view
  (Estimate), is among the records the result's fit read; the result is the opening's recorded
  scores, or the lock's plan.
* :func:`logging_complete`: every attempt has a reason, a note on a record that changed before it
  (a fit that followed only TurboTab's own records is noted as not the analyst's), and for each
  family an engine-measured movement or the engine's why not; every record the analyst made has
  its reason.
* :func:`single_pass`: flagged when, before the seal's opening or the plan's lock, no decision had
  an alternative examined and the run fit at most once.
* :func:`known_answer`: for a known-answer control, each fact reading the fixture declares
  (``truths.FIXTURE_TRUTHS``) is answered as declared, and the result is right (its interval holds
  the declared answer) or refused.
* :func:`null_decoy`: for a null decoy, the reported result shows no finding: no exposure row's
  interval excludes zero, and no held-out AUC, C-index or R² has an interval that lies above the
  score of no skill.
* :func:`robustness`: the analyst's own export returned the bundle, and no person helped.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, get_args

from research.swarm.record import Record, validate

Status = Literal["pass", "fail", "not_checked"]

# The scores of no skill, where a metric has one whatever the data (higher is better for each):
# a ranking by random scores has an AUC or a C-index of 0.5 (Hanley & McNeil 1982; Harrell et al.
# 1982), and predicting the mean has an R² of 0.
NO_SKILL: dict[str, float] = {"auc": 0.5, "c_index": 0.5, "r2": 0.0}


def _facts() -> tuple[str, ...]:
    """The readings of what the data are (``decisions.ReadingKind``, and the below-detection
    repair), which a fixture's truth declares beside the research question's assumptions (its
    ``exposure:``, ``contrast:`` and ``adjust:`` keys, the analyst's to choose)."""
    from turbotab.core.decisions import ReadingKind

    return (*get_args(ReadingKind), "detection_limit")


@dataclass
class Outcome:
    check: str
    run: str
    outcome: Status
    problems: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _outcome(check: str, rec: Record, problems: list[str], unchecked: list[str],
             notes: list[str] | None = None) -> Outcome:
    status: Status = "fail" if problems else "not_checked" if unchecked else "pass"
    return Outcome(check, rec.run.id, status, problems, unchecked, notes or [])


def _seqs(seqs: Iterable[int]) -> str:
    return ", ".join(f"#{s}" for s in seqs)


# ── 1. seal integrity ────────────────────────────────────────────────────────


def seal_integrity(rec: Record) -> Outcome:
    problems: list[str] = []
    unchecked: list[str] = []
    for d in rec.decisions:
        if d.kind == "reseal":
            problems.append(f"Record #{d.seq} re-seals the held-out rows.")
    openings: dict[str, list[int]] = {}
    for d in rec.decisions:
        if d.kind == "open_seal":
            openings.setdefault(str(d.option.get("target")), []).append(d.seq)
    for target, seqs in openings.items():
        if len(seqs) > 1:
            problems.append(f"The held-out rows of `{target}` were opened {len(seqs)} times "
                            f"(records {_seqs(seqs)}).")
    by_seq = {d.seq: d for d in rec.decisions}
    r = rec.result
    if r.opening is not None:
        if r.decisions is None:
            unchecked.append("The opened fit is not among the fit records, so the records its "
                             "result read are not known.")
        else:
            late = [s for s in r.decisions if s > r.opening or by_seq[s].post_seal]
            if late:
                problems.append(f"Records {_seqs(late)}, made after the opening at record "
                                f"#{r.opening}, are among those the result's fit read.")
        opened = by_seq.get(r.opening)
        if opened is None or r.scores != opened.option.get("scores"):
            problems.append(f"The result's scores are not those the opening at record "
                            f"#{r.opening} recorded.")
    if r.lock is not None:
        if r.decisions is None:
            unchecked.append("The locked fit is not among the fit records, so the records its "
                             "result read are not known.")
        else:
            late = [s for s in r.decisions if s > r.lock or by_seq[s].after_estimates]
            if late:
                problems.append(f"Records {_seqs(late)}, made after the plan's lock at record "
                                f"#{r.lock} or with the estimates in view, are among those the "
                                f"result's fit read.")
        locked = by_seq.get(r.lock)
        if locked is None or r.plan_digest != locked.option.get("digest"):
            problems.append(f"The result's plan is not the one the lock at record #{r.lock} "
                            f"recorded.")
    return _outcome("seal_integrity", rec, problems, unchecked)


# ── 2. logging ───────────────────────────────────────────────────────────────


def logging_complete(rec: Record) -> Outcome:
    problems: list[str] = []
    notes: list[str] = []
    by_seq = {d.seq: d for d in rec.decisions}
    for a in rec.attempts:
        changed = a.changed or []
        if not a.reasons:
            if changed and all(by_seq[s].source == "app" for s in changed):
                notes.append(f"Fit #{a.seq} followed only TurboTab's own records "
                             f"({_seqs(changed)}), so it was not the analyst's attempt.")
            else:
                listed = f" (records {_seqs(changed)})" if changed else ""
                problems.append(f"Fit #{a.seq} has no reason: no record that changed before "
                                f"it{listed} has a note.")
        for family in a.families:
            m = next((m for m in a.movement if m.get("family") == family), None)
            if m is not None and (m.get("movement") is not None or m.get("why")):
                continue
            if a.movement_why or a.withheld:
                continue
            problems.append(f"Fit #{a.seq} has no measured movement for `{family}`, and no why.")
    for d in rec.decisions:
        if d.source == "analyst" and not (d.reason or "").strip():
            problems.append(f"Record #{d.seq} ({d.kind}) has no reason.")
    return _outcome("logging", rec, problems, [], notes)


def failed_run(path: Path) -> Outcome:
    """A record that failed its schema (``corpus/failed/``): a logging failure."""
    body = json.loads(path.read_text("utf-8"))
    first = (body.get("error") or "").splitlines()[:3]
    return Outcome("logging", str(body.get("run_id") or path.stem), "fail",
                   [f"The record failed its schema: {' '.join(first)}"])


# ── 3. single pass ───────────────────────────────────────────────────────────


def single_pass(rec: Record) -> Outcome:
    """Flagged (``fail``) when no decision had an alternative examined and the run fit at most
    once (SWARM_PLAN §3), each counted before the seal's opening and the estimates: an alternative
    of a decision in the ``before`` phase, examined before the opening (Predict) or the plan's lock
    (Estimate), and a fit that read no record of another phase. Trying options once the result is
    in view is not exploring."""
    before = {d.seq for d in rec.decisions if d.phase == "before"}
    commit = next((d.at for d in rec.decisions if d.seq in (rec.result.opening, rec.result.lock)),
                  None)
    examined = sorted({d.seq for d in rec.decisions if d.seq in before for e in d.examined
                       if e.alternative and (commit is None or e.at < commit)})
    fits = sum(1 for a in rec.attempts if a.decisions is not None and set(a.decisions) <= before)
    notes = [f"{len(examined)} of {len(before)} decisions made before the seal's opening or the "
             f"plan's lock had an alternative examined; {fits} fits read only those decisions."]
    problems = []
    if not examined and fits <= 1:
        noun = "fit" if fits == 1 else "fits"
        problems.append(f"Single pass: before the seal's opening or the plan's lock, no decision "
                        f"had an alternative examined, and the run made {fits} {noun}.")
    return _outcome("single_pass", rec, problems, [], notes)


# ── 4. known-answer controls ─────────────────────────────────────────────────


def known_answer(rec: Record, truths: Mapping[str, Mapping[str, str]] | None = None) -> Outcome:
    """Right or refused (SWARM_PLAN §5). ``truths``: each fixture's declared readings, by file
    name; ``truths.FIXTURE_TRUTHS`` by default."""
    if rec.dataset.control != "known_answer":
        return _outcome("known_answer", rec, [], ["Not a known-answer control."])
    if truths is None:
        from turbotab.core.tests.truths import FIXTURE_TRUTHS as truths
    problems: list[str] = []
    unchecked: list[str] = []
    fixture = rec.dataset.fixture or (rec.dataset.files[0].name if rec.dataset.files else None)
    truth = truths.get(fixture or "")
    scope = set(rec.result.decisions) if rec.result.decisions is not None else None
    answered: dict[str, tuple[Any, int]] = {}
    for d in rec.decisions:
        if scope is None or d.seq in scope:
            for a in d.answers:
                answered[a.question] = (a.value, d.seq)
    if truth is None:
        unchecked.append(f"No truth is declared for the fixture `{fixture}`.")
    else:
        facts = _facts()
        for key, declared in truth.items():
            if key.split(":", 1)[0] not in facts or key not in answered:
                continue
            value, seq = answered[key]
            if str(value) != str(declared):
                problems.append(f"Record #{seq} answers the {key} reading as `{value}`; the "
                                f"fixture declares `{declared}`.")
    r, known = rec.result, rec.dataset.known
    if r.refused is not None:
        pass  # refused: right or refused
    elif not known:
        unchecked.append("The fixture's declared truth holds its readings, not its answer, and no "
                         "known answer is declared, so the result is not checked.")
    elif "value" in known:
        problems += _estimate_holds(r.effects or {}, known)
    else:
        problems += _score_within(r.scores or {}, r.family, known)
    return _outcome("known_answer", rec, problems, unchecked)


def _estimate_holds(effects: Mapping[str, Any], known: Mapping[str, Any]) -> list[str]:
    feature, value, family = known.get("feature"), float(known["value"]), known.get("family")
    rows = [(f, row) for f, rs in effects.items() if family in (None, f) for row in rs or []
            if row.get("feature") == feature]
    if not rows:
        return [f"The result holds no estimate of `{feature}`, the known answer's."]
    out = []
    for f, row in rows:
        low, high = row.get("ci_low"), row.get("ci_high")
        if low is None or high is None:
            continue  # no interval: not estimated, so refused
        if not low <= value <= high:
            out.append(f"`{f}` estimates `{feature}` at {row.get('estimate')} ({low} to {high}), "
                       f"an interval that excludes the known answer {value}.")
    return out


def _score_within(scores: Mapping[str, Any], family: str | None,
                  known: Mapping[str, Any]) -> list[str]:
    metric, low, high = known.get("metric"), float(known["low"]), float(known["high"])
    score = (scores.get(family or "") or {}).get(metric)
    if score is None:
        return [f"The result holds no held-out {metric} for `{family}`."]
    if not low <= score <= high:
        return [f"`{family}` scores a held-out {metric} of {score}, outside the known answer's "
                f"{low} to {high}."]
    return []


# ── 5. null decoys ───────────────────────────────────────────────────────────


def null_decoy(rec: Record) -> Outcome:
    if rec.dataset.control != "null_decoy":
        return _outcome("null_decoy", rec, [], ["Not a null decoy."])
    r = rec.result
    problems: list[str] = []
    unchecked: list[str] = []
    if r.refused is not None:
        return _outcome("null_decoy", rec, [], [], [f"No result was reported: {r.refused}"])
    if r.effects is not None:
        for family, rows in r.effects.items():
            for row in rows or []:
                low, high = row.get("ci_low"), row.get("ci_high")
                if low is not None and high is not None and (low > 0 or high < 0):
                    problems.append(f"`{family}` reports `{row.get('feature')}` at "
                                    f"{row.get('estimate')} ({low} to {high}), an interval that "
                                    f"excludes no effect.")
    elif r.scores is not None:
        intervals = r.intervals or {}
        judged = [m for m in NO_SKILL if m in intervals]
        if not judged:
            unchecked.append(r.intervals_why or "The result serves no interval for a score with a "
                                                "score of no skill.")
        for m in judged:
            low = (intervals[m] or {}).get("ci_low")
            if low is not None and low > NO_SKILL[m]:
                problems.append(f"`{r.family}` scores a held-out {m} whose interval starts at "
                                f"{low}, above the {NO_SKILL[m]} of no skill.")
    return _outcome("null_decoy", rec, problems, unchecked)


# ── 6. robustness ────────────────────────────────────────────────────────────


def robustness(rec: Record) -> Outcome:
    problems: list[str] = []
    if not rec.export.reached:
        problems.append("The analyst's own export never returned the bundle.")
    if rec.run.help:
        noun = "time" if len(rec.run.help) == 1 else "times"
        problems.append(f"A person helped {len(rec.run.help)} {noun}: "
                        f"{'; '.join(rec.run.help)}")
    return _outcome("robustness", rec, problems, [])


CHECKS: tuple[Callable[[Record], Outcome], ...] = (
    seal_integrity, logging_complete, single_pass, known_answer, null_decoy, robustness)


def run_checks(records: Iterable[Record | Mapping[str, Any]]) -> list[Outcome]:
    out: list[Outcome] = []
    for rec in records:
        rec = rec if isinstance(rec, Record) else validate(rec)
        out += [check(rec) for check in CHECKS]
    return out


def corpus(folder: Path) -> tuple[list[Record], list[Path]]:
    """The records in ``folder`` and the failed runs in ``folder/failed``."""
    records = [validate(json.loads(p.read_text("utf-8"))) for p in sorted(folder.glob("*.json"))]
    failed = sorted((folder / "failed").glob("*.json"))
    return records, failed


def main(argv: Sequence[str] | None = None) -> int:
    from research.swarm.assemble import CORPUS

    parser = argparse.ArgumentParser(prog="python -m research.swarm.checks",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=CORPUS)
    args = parser.parse_args(argv)
    records, failed = corpus(args.corpus)
    outcomes = run_checks(records) + [failed_run(p) for p in failed]
    for o in outcomes:
        print(json.dumps(asdict(o), ensure_ascii=False))
    return 1 if any(o.outcome == "fail" for o in outcomes if o.check != "single_pass") else 0


if __name__ == "__main__":
    sys.exit(main())
