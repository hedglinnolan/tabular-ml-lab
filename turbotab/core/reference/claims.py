"""The claims reference, the claims' history and their frozen status, generated from the claims
registry (CLAIMS_REGISTRY.md §5, §6; R1_PLAN.md R1-G).

    python -m turbotab.core.reference.claims          # writes the reference and the frozen status,
                                                      # and appends the history
    python -m turbotab.core.reference.claims --check  # exits 1 when any is stale; writes nothing

writes ``docs/turbotab-next/reference/CLAIMS_REFERENCE.md``: every claim the registry
(``turbotab.core.claims``) holds, by status (built; ruled, not built; retracted), each with its
sentences in both registers, its origin, its parameters, its regime, its decision, its parts and
their grades, basis audits and referees, its tests and their truth, the code that enforces it, its
version and its content hash.

``turbotab/core/claims/history.jsonl`` keeps every version of every claim the generator has
written, oldest first, one JSON object a line (``id``, ``version``, ``hash``, ``claim``). When a
claim's content hash changes, its old version stays where it is and the new one is appended after
it, so an old project's claims-in-force manifest resolves its hash to the words and grade it ran
under (§5).

The history also keeps every grade change, a **grade line** per part (``id``, ``part``, ``grade``,
``was``, ``date``, ``grader``, ``reason``), each after its claim's version line. A part's first
line records the grade it holds, moving from no grade (``was`` is empty), with its reason
:data:`FIRST_RECORDED` unless its ``regraded`` gives one; after that, a line is written only when
the part's grade differs from its last line's, upgrades and downgrades alike, saying the grade it
moved from. A line's grader and date are the "<who>, YYYY-MM-DD" of the part's grader in force;
only a first line whose part names no dated grader (one its author graded judgment or unsupported)
takes the author as its grader and the day it is recorded as its date (``--date``, today unless
given). A change needs a dated grader, a grader and date other than the last line's (a grader in
force left from the grade before is no record of who made this change, or when), a date no
earlier than the last line's, and a reason in ``regraded`` other than the last line's.

A part its sentence no longer asserts is graded ``retired`` (``claims.RETIRED``), a change like
any other, and once its line is written the claim may leave it out; a part the history holds that
the claim no longer names, and whose last line does not retire it, is dropped or renamed, and is
refused. A part the history never recorded cannot enter retired, and a swarm claim's part is
retired only by someone other than its author. A change without what it needs is refused, and so
is a swarm claim's part entering above unsupported, and a version of a swarm claim that edits
what the history recorded of its review (``claims.review_edits``: its origin or author, a
referee or a basis audit dropped or changed, or the grader of a part above unsupported while its
grade stands); where any is refused, the generator writes nothing and it and ``--check`` exit 1.
Meta-test 8 refuses a grade that differs from its part's last line, a part dropped without a line
that retires it, and such an edited review, where a history written otherwise holds one.

``turbotab/core/claims/status.json``, the frozen status file, holds each claim's id, derived status
and content hash. The shipped app holds no tests, so a status cannot be derived there: a sentence
renders from this file (``Registry.frozen_status``), and the generator, which runs where the source
is complete, is what writes it.

``--check`` fails when the reference or the frozen status file differs from the registry's, or
when the history's newest version of a claim is not the one in force. An id the history holds is
never dropped (§1: a retired id stays, marked retracted), so where the registry no longer registers
one, the generator writes nothing and it and ``--check`` both exit 1.

The reference and the frozen file are pure functions of the registry and the files its status is
derived from: nothing they hold reads the clock, and claims are listed by id. Only a first grade
line with no dated grader holds the day it is recorded on, and what ``--check`` decides never
turns on the day: a line is stale by its grade alone.
``turbotab/core/tests/test_claims_reference.py`` regenerates the reference and fails when the
committed copy is stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from turbotab.core import claims as registry_module
from turbotab.core.claims import (FROZEN, GENERATOR, GRADED_FORM, GRADES, ORIGINS, RETIRED, Claim,
                                  Registry, audits, checked, content, content_hash, dated, fill,
                                  frozen_records, frozen_text, review_edits, same_person)
from turbotab.core.claims import HISTORY as HISTORY_PATH

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "turbotab-next" / "reference" / "CLAIMS_REFERENCE.md"
HISTORY = REPO / HISTORY_PATH
COMMAND = GENERATOR
HASH_SHOWN = 12  # hex digits of a content hash the reference prints
# The reason on a part's first grade line, where its source gives none.
FIRST_RECORDED = "first recorded: the grade the part held when the history first recorded it"
SECTIONS = (
    ("built", "Built",
     "Claims whose enforcers resolve and whose tests exist (CLAIMS_REGISTRY.md §6)."),
    ("proposed", "Ruled, not built",
     "Claims whose code or tests do not exist yet, and swarm claims with a part still "
     "unsupported. Each is a ruling, or a claim on its way in; none renders a sentence until it "
     "is built, and a doc that describes one as current is ahead of the code."),
    ("retracted", "Retracted",
     "Claims withdrawn, each with its date and reason. A retired id stays, and is never reused."),
)


def code(text: str) -> str:
    return f"`{text}`"


def _sentence(c: Claim, register: str) -> str:
    template = getattr(c, register)
    return fill(template, c.params) if template else "none"


def _part(p: Any, c: Claim) -> str:
    b = p.basis
    basis = [*(f"cites {code(k)}" for k in b.cites)]
    for label, value in (("where", b.where), ("reason", b.reason), ("theorem", b.theorem),
                         ("derivation", b.derivation)):
        if value:
            basis.append(f"{label}: {value}")
    basis += [f"test {code(t)}" for t in b.tests]
    basis += [f"held out on {h.fixture}: {h.rate:g} over {h.cases} cases" for h in b.heldout]
    grader = p.graded_by or c.graded_by or "not graded"
    now = checked(c, p)
    changed = ", on a version of the part since changed"
    checks = []
    for audit in audits(p):  # oldest first; the last is in force
        checks.append(f"basis checked against its source by {audit.who}"
                      + ("" if audit.checked == now else changed))
    groups: dict[tuple[str, bool, str], list[str]] = {}  # (verdict, read since, grade): who
    for r in p.referees:
        stale = r.checked != now
        groups.setdefault((r.verdict, stale, r.grade), []).append(
            r.who + (f" (a fork of {r.fork_of})" if r.fork_of else ""))
    for verdict, label in (("upheld", "upheld"), ("not_upheld", "not upheld")):
        for stale in (False, True):
            for (v, s, grade), who in groups.items():
                if (v, s) == (verdict, stale):
                    checks.append(f"{label} as {grade} by {', '.join(who)}"
                                  + (changed if stale else ""))
    if p.regraded:
        checks.append(f"regraded: {p.regraded}")
    return (f"  - {p.name}: *{p.grade}*, graded by {grader}"
            + (f" ({'; '.join(basis)})" if basis else "")
            + (f"; {'; '.join(checks)}" if checks else ""))


def claim_section(c: Claim, status: str) -> list[str]:
    """One claim's entry."""
    lines = [f"### {code(c.id)}", "",
             f"*{c.kind}* · from *{c.origin}* · weakest part *{c.grade}* · {status} · "
             f"version {c.version} · hash {code(content_hash(c)[:HASH_SHOWN])}", "",
             f"- **Card:** {_sentence(c, 'card')}",
             f"- **Methods:** {_sentence(c, 'methods')}"]
    if c.params:
        lines.append("- **Parameters:** " + "; ".join(
            f"{code(k)} = {p.value} {p.unit}" + (f", read by {code(p.constant)}" if p.constant
                                                 else "")
            for k, p in sorted(c.params.items())))
    regime = c.regime.words or "not stated"
    if c.regime.requires:
        regime += "; predicates " + ", ".join(code(r) for r in c.regime.requires)
    if c.regime.disjoint_from:
        regime += "; disjoint from " + ", ".join(code(r) for r in c.regime.disjoint_from)
    lines.append(f"- **Regime:** {regime}")
    if c.decision or c.options or c.verdict or c.goal:
        what = [f"decision {code(c.decision)}" if c.decision else "",
                "options " + ", ".join(code(o) for o in c.options) if c.options else "",
                f"verdict *{c.verdict}*" if c.verdict else "",
                "goal " + ", ".join(c.goal) if c.goal else ""]
        lines.append("- **Decision:** " + "; ".join(w for w in what if w))
    lines.append("- **Parts:**")
    lines += [_part(p, c) for p in c.parts]
    if c.tests:
        lines.append("- **Tests:** " + "; ".join(
            f"{t.side} {code(t.test)} ({t.truth})" for t in c.tests))
    if c.enforced_by:
        lines.append("- **Enforced by:** " + ", ".join(code(e) for e in c.enforced_by))
    for label, ids in (("Supersedes", c.supersedes), ("Falls back to", (c.fallback,))):
        if any(ids):
            lines.append(f"- **{label}:** " + ", ".join(code(i) for i in ids if i))
    lines.append(f"- **Author:** {c.author}")
    if c.ruled:
        lines.append(f"- **Ruled:** {c.ruled}")
    if c.retracted:
        lines.append(f"- **Retracted:** {c.retracted}")
    lines.append("")
    return lines


def render(reg: Registry) -> str:
    """The claims reference, as Markdown."""
    status = {c.id: reg.status(c.id) for c in reg}
    counts = {s: sum(1 for v in status.values() if v == s) for s, _, _ in SECTIONS}
    lines = [
        "# TurboTab v2 claims reference", "",
        f"*Generated by `{COMMAND}` from the claims registry (`turbotab/core/claims/`, "
        "CLAIMS_REGISTRY.md). Do not edit it by hand: "
        "`turbotab/core/tests/test_claims_reference.py` regenerates it and fails when this copy "
        "is stale. Every version of every claim, and every change of a part's grade with its "
        "date, grader and reason, is kept in `turbotab/core/claims/history.jsonl`, and the "
        f"status each renders with in `{FROZEN.as_posix()}`.*", "",
    ]
    if not len(reg):
        lines += ["No claim is registered yet.", ""]
    else:
        noun = "claim" if len(reg) == 1 else "claims"
        lines += [f"It holds {len(reg)} {noun}: {counts['built']} built, {counts['proposed']} "
                  f"ruled and not built, {counts['retracted']} retracted.", ""]
    lines += [
        "## How to read an entry", "",
        "- **Kind**: a *rule* (a guardrail, refusal, routing or default), a *parameter* (a number "
        "with a unit) or *teaching* (a statement of fact about a method).",
        "- **Origin** (" + ", ".join(f"*{o}*" for o in ORIGINS) + "): written by the engine's "
        "builders from its own code, registered for a finding an audit made, or proposed by the "
        "research swarm. A swarm claim enters with every part unsupported; each of its parts "
        "leaves unsupported only as graded by someone other than its author. A swarm claim's "
        "part rises above judgment only once two distinct referees, neither its author nor a "
        "fork, upheld it at that grade, and someone other than its author checked its basis "
        "against the source. Each of them read the part as it now stands (its sentence, regime, "
        "decision, grade and basis), on or after the day it took its previous grade. Once the "
        "history records a version of a swarm claim, its author stays as recorded, the grader of "
        "a part above unsupported changes only with its grade, and every referee and basis audit "
        "stays as recorded: a later verdict is a referee added, and a re-audit an audit appended "
        "after the others, the last in force, never one edited or removed.",
        "- **Grade**: each part of a sentence is graded on its own (" + ", ".join(
            f"*{g}*" for g in GRADES) + ", strongest first), and the sentence shows its weakest "
        "part's. Above judgment, someone other than the author grades. A part's grade moves "
        "only through a line of the history that gives its date, grader and reason. Each line "
        "is dated no earlier than the line before it, with a grader or date other than that "
        f"line's. A part the sentence no longer asserts is *{RETIRED}* through such a line, "
        "never dropped.",
        "- **Status** is derived: *built* once every enforcer resolves and every test exists and "
        "is not marked to skip or xfail unconditionally, *ruled, not built* until then, as is a "
        "swarm claim with a part still unsupported; *retracted* is the only status anyone "
        "declares. The app ships without its tests, so it renders a claim from the status and "
        "hash the generator froze, never from the source.",
        "- **Card** and **Methods**: the sentence in its two registers, filled from its "
        "parameters.",
        f"- **Hash**: the first {HASH_SHOWN} hex digits of the SHA-256 of the claim's content; a "
        "manifest that names it resolves through the history.", "",
    ]
    for key, title, about in SECTIONS:
        lines += [f"## {title}", "", about, ""]
        chosen = [c for c in reg if status[c.id] == key]
        if not chosen:
            lines += ["None.", ""]
        for c in chosen:
            lines += claim_section(c, title.lower())
    return "\n".join(lines)


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_history(path: Path = HISTORY) -> list[dict[str, Any]]:
    """The history's versions, oldest first (its grade lines left out)."""
    return [r for r in _lines(path) if "part" not in r]


def read_grades(path: Path = HISTORY) -> list[dict[str, Any]]:
    """The history's grade lines, oldest first."""
    return [r for r in _lines(path) if "part" in r]


def grade_lines(reg: Registry, grades: list[dict[str, Any]],
                today: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The grade lines the history lacks, in id order, and the changes refused, one sentence each.
    ``today`` dates a first line whose part names no dated grader."""
    last = {(g["id"], g["part"]): g for g in grades}
    held: dict[str, list[str]] = {}  # each claim's parts the history holds, as it first names them
    for id, part in last:
        held.setdefault(id, []).append(part)
    lines, refused = [], []
    for c in reg:
        named = {p.name for p in c.parts}
        for name in held.get(c.id, []):
            gone = last[(c.id, name)]
            if name not in named and gone["grade"] != RETIRED:
                refused.append(f"{c.id}: part {name!r}, last graded {gone['grade']}, is in the "
                               "history, but the claim no longer names it: a part is retired, "
                               f"never dropped or renamed, so name it graded {RETIRED}, with its "
                               "dated grader and its reason in regraded, until the history "
                               "records that")
        for p in c.parts:
            prev = last.get((c.id, p.name))
            if prev is not None and prev["grade"] == p.grade:
                continue
            graded = dated(p.graded_by or c.graded_by)
            if prev is None:
                if p.grade == RETIRED:
                    refused.append(f"{c.id}: part {p.name!r} is retired, but the history holds "
                                   "no line for it: a part the history never recorded is left "
                                   "out, not retired")
                    continue
                if c.origin == "swarm" and p.grade != "unsupported":
                    refused.append(f"{c.id}: part {p.name!r} is graded {p.grade}, but a swarm "
                                   "claim's part enters unsupported (R1_PLAN.md R1-G): register "
                                   "it so, and grade it after")
                    continue
                who, when = graded if graded else (c.author, today)
                reason, was = p.regraded.strip() or FIRST_RECORDED, ""
            else:
                missing = []
                if graded is None:
                    missing.append(f"a dated grader, {GRADED_FORM}")
                elif graded == (prev["grader"], prev["date"]):
                    missing.append(f"a grader and date other than its last line's, "
                                   f"{prev['grader']}, {prev['date']}, to say who made this "
                                   "change and when")
                elif graded[1] < prev["date"]:
                    missing.append(f"a date no earlier than its last line's, {prev['date']}, not "
                                   f"{graded[1]}")
                elif (c.origin == "swarm" and p.grade == RETIRED
                      and same_person(graded[0], c.author)):
                    missing.append(f"a grader other than its author, not {graded[0]!r}: a swarm "
                                   "claim's part is retired only by someone else")
                if not p.regraded.strip() or p.regraded.strip() == prev["reason"]:
                    missing.append("a reason in its regraded other than its last line's")
                if missing:
                    refused.append(f"{c.id}: part {p.name!r} moves from {prev['grade']} to "
                                   f"{p.grade}, so it names {' and '.join(missing)}")
                    continue
                (who, when), reason, was = graded, p.regraded.strip(), prev["grade"]
            lines.append({"id": c.id, "part": p.name, "grade": p.grade, "was": was, "date": when,
                          "grader": who, "reason": reason})
    return lines, refused


def pending(reg: Registry, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The versions in force that the history's newest entry for their claim is not."""
    newest = {h["id"]: h["hash"] for h in history}
    return [{"id": c.id, "version": c.version, "hash": content_hash(c), "claim": content(c)}
            for c in reg if newest.get(c.id) != content_hash(c)]


def dropped(reg: Registry, history: list[dict[str, Any]]) -> list[str]:
    """The ids the history holds that the registry no longer registers, sorted."""
    return sorted({h["id"] for h in history} - {c.id for c in reg})


def frozen_differs(reg: Registry, text: str) -> list[str]:
    """The ids whose frozen status or hash in ``text`` is not the registry's now, sorted (every
    id, when ``text`` does not read)."""
    derived = {r["id"]: (r["status"], r["hash"]) for r in frozen_records(reg)}
    try:
        held = {r["id"]: (r["status"], r["hash"]) for r in json.loads(text)["claims"]}
    except (ValueError, KeyError, TypeError):
        held = {}
    return sorted(i for i in derived.keys() | held.keys() if derived.get(i) != held.get(i))


def line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _day(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a day written YYYY-MM-DD") from None


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the reference or the history is stale, or the history "
                             "holds an id the registry dropped; write nothing")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--history", type=Path, default=None,
                        help="the history to read and append to (the registry's own unless given)")
    parser.add_argument("--date", type=_day, default=None,
                        help="the day a first grade line with no dated grader is recorded on "
                             "(YYYY-MM-DD; today unless given)")
    args = parser.parse_args(argv)
    reg = registry_module.registry() if registry is None else registry
    if args.history is None:
        args.history = reg.history_path
    frozen_path = reg.frozen_path  # where rendering reads it
    text, frozen = render(reg), frozen_text(reg)
    history = read_history(args.history)
    todo = pending(reg, history)
    gone = dropped(reg, history)
    graded, refused = grade_lines(reg, read_grades(args.history),
                                  args.date or date.today().isoformat())
    edited = review_edits(reg, args.history)
    if gone:
        print(f"{args.history} holds {', '.join(gone)}, which the registry no longer registers: "
              "an id is never dropped or reused (CLAIMS_REGISTRY.md §1), so restore the claim "
              "and retract it", file=sys.stderr)
        if not args.check:
            return 1
    for message in refused:
        print(f"{message}: a grade moves only through a history line with its date, grader and "
              "reason", file=sys.stderr)
    for message in edited:
        print(f"{message}; so no version that edits it is written", file=sys.stderr)
    if (refused or edited) and not args.check:
        return 1
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        stale = []
        if current != text:
            stale.append(f"{args.out} is stale: run {COMMAND}")
        if todo:
            stale.append(f"{args.history} lacks the version in force of "
                         f"{', '.join(r['id'] for r in todo)}: run {COMMAND}")
        if graded:
            moved = sorted({g["id"] for g in graded})
            stale.append(f"{args.history} lacks the grade lines of {', '.join(moved)}: run "
                         f"{COMMAND}")
        frozen_now = frozen_path.read_text(encoding="utf-8") if frozen_path.is_file() else ""
        if frozen_now != frozen:
            moved = frozen_differs(reg, frozen_now)
            stale.append(f"{frozen_path} is stale" + (f" for {', '.join(moved)}" if moved else "")
                         + f": run {COMMAND}")
        for message in stale:
            print(message, file=sys.stderr)
        if stale or gone or refused or edited:
            return 1
        print(f"{args.out}, {args.history} and {frozen_path} are current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as f:
        for c in reg:  # each claim's version line, then its grade lines
            f.writelines(line(r) for r in todo if r["id"] == c.id)
            f.writelines(line(g) for g in graded if g["id"] == c.id)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(frozen, encoding="utf-8")
    print(f"wrote {args.out} ({len(text.encode('utf-8')):,} bytes) and {frozen_path} ({len(reg)} "
          f"claim(s)); appended {len(todo)} version(s) and {len(graded)} grade line(s) to "
          f"{args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
