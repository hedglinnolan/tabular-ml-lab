"""`TEST-109` and `TEST-079` — the two fields the tool could create and not fix.

`cmd_add` was the sole writer of `ev` and of `act`. `cmd_set` wrote `status`,
`note`, `test` and `verified_ev` and touched neither. So a citation that was
true when it was filed became permanently wrong **through the only writer this
project sanctions** — a file a tool owns has exactly one writer, and for these
two fields that writer could not correct itself. The instrument built to keep
the ledger honest was structurally unable to repair the field carrying the line
numbers.

## The foot-gun in front of the fix, and the ruling on it

**`set --ev` already worked**, and it did not mean what it looks like. argparse
resolves an unambiguous prefix, so `--ev` was `--evidence`, and `--evidence`
writes `verified_ev`. Adding a literal `--ev` for the `ev` field would have
silently moved the destination of a flag that worked that day — same spelling,
different field, no error and no warning. Driven in a sandboxed copy of the tool
before the flag existed and again after.

**The ruling is `allow_abbrev=False` on the `set` subparser**, so the two are
exact matches and cannot be confused, plus an explicit `--verified-ev` alias so
the pair is named in `--help` rather than remembered. **What that breaks, said
rather than discovered:** every other prefix on that subcommand — `--st`,
`--no`, `--ev` meaning `--evidence`. Nothing in the repository used one;
`LOOP.md`, the tool's own docstring and every prompt spell the flags in full,
checked before the change. `add` keeps abbreviation because no pair of its
options is a prefix of another.

## Why `regen` changed too

`cmd_regen` rendered `ev` and **never** rendered `verified_ev`, so every
correction filed through `set --evidence` since that flag existed was invisible
in the generated ledger — including `L64`'s own note that `GUIDED-245`'s
citation was stale, which is the row this file's sweep re-found. 476 rows carry
a re-verified citation that differs from the original.

## This file writes nothing itself

The ledger's own `save()` does the writing, and `ledger.DATA` is pointed at
`tmp_path`. That keeps `tests/test_no_test_writes_a_path_git_tracks.py`'s
unresolved-destination count where it is — the corpus sits at exactly its
ceiling, so a `shutil.copy` into a `tmp_path`-derived name here would turn that
guard red. Stated rather than left for someone to rediscover.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "docs" / "turbotab" / "tools" / "ledger.py"
TERMINAL = {"FIXED", "NOT-A-DEFECT", "WONTFIX"}


def _ledger():
    spec = importlib.util.spec_from_file_location("_ledger_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """The real tool, pointed at a throwaway ledger."""
    ledger = _ledger()
    monkeypatch.setattr(ledger, "DATA", tmp_path / "findings.json")
    monkeypatch.setattr(ledger, "OUT", tmp_path / "FINDINGS_LEDGER.md")
    def row(row_id, item):
        return {"id": row_id, "area": "TEST", "sev": "low", "item": item,
                "detail": "", "ev": "ORIGINAL CITATION some_file.py:1",
                "act": "ORIGINAL REMEDY", "status": "OPEN", "note": "",
                "verified_against": "", "test": "", "verified_ev": ""}

    # PADDED, because `cmd_regen` refuses to write a markdown under 1,024 bytes
    # — its own guard against a truncated ledger, and a one-row sandbox trips
    # it. The padding rows are inert and the assertions all name `TEST-000`.
    ledger.save([row("TEST-000", "a row to repair")]
                + [row(f"TEST-9{i:02d}",
                       f"padding row {i} — the regen guard refuses a markdown "
                       f"smaller than a kilobyte, which a single-row ledger is")
                   for i in range(12)])
    return ledger


def _run(ledger, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["ledger.py", *argv])
    return ledger.main()


def _row(ledger, row_id="TEST-000"):
    return next(r for r in ledger.load() if r["id"] == row_id)


# ═══════════ 1 · THE TWO FIELDS THE TOOL COULD NOT REPAIR ═══════════

def test_set_writes_the_evidence_citation_it_could_only_create(sandbox,
                                                               monkeypatch):
    """`TEST-109`. The field carrying the line numbers, made correctable."""
    assert _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
                "--ev", "REPAIRED turbotab/api.py:29") == 0
    row = _row(sandbox)
    assert row["ev"] == "REPAIRED turbotab/api.py:29"
    assert row["verified_ev"] == "", (
        "`--ev` landed in `verified_ev`, which is the collision this flag was "
        "ruled on: it was an abbreviation of `--evidence` before it was an "
        "option")


def test_set_writes_the_remedy_the_spelling_gate_cannot_see(sandbox,
                                                            monkeypatch):
    """`TEST-079`. `act` is not rendered into the generated markdown, so the
    spelling gate cannot reach it — and until now neither could the tool."""
    assert _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
                "--act", "REPAIRED REMEDY") == 0
    assert _row(sandbox)["act"] == "REPAIRED REMEDY"


def test_evidence_still_writes_the_reverified_field(sandbox, monkeypatch):
    """The behavior that already existed, pinned so the new flag cannot take
    it over by accident."""
    _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
         "--evidence", "RE-VERIFIED AT HEAD")
    row = _row(sandbox)
    assert row["verified_ev"] == "RE-VERIFIED AT HEAD"
    assert row["ev"] == "ORIGINAL CITATION some_file.py:1", (
        "`--evidence` now overwrites the original citation, which destroys the "
        "history the two fields exist to keep apart")

    _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
         "--verified-ev", "VIA THE ALIAS")
    assert _row(sandbox)["verified_ev"] == "VIA THE ALIAS"


def test_an_abbreviation_is_refused_rather_than_silently_redirected(
        sandbox, monkeypatch):
    """**The ruling on the collision, asserted as behavior.**

    Before `allow_abbrev=False`, `--ev` was a prefix of `--evidence` and landed
    in a different field with no error. The failure mode of a silent redirect
    is that the tool succeeds, the gate stays green, and the record is quietly
    wrong about itself — which is the shape this project has already paid for
    three times in ledger notes.
    """
    with pytest.raises(SystemExit) as exit_info:
        _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
             "--evid", "AMBIGUOUS")
    assert exit_info.value.code != 0
    assert _row(sandbox)["ev"] == "ORIGINAL CITATION some_file.py:1"
    assert _row(sandbox)["verified_ev"] == ""

    # And the two full spellings are still both accepted, so the fix is not
    # "reject everything".
    assert _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
                "--ev", "A") == 0
    assert _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
                "--evidence", "B") == 0
    assert (_row(sandbox)["ev"], _row(sandbox)["verified_ev"]) == ("A", "B")


def test_regen_renders_a_reverified_citation_when_it_differs(sandbox,
                                                             monkeypatch):
    """`cmd_regen` rendered `ev` and never `verified_ev`, so every correction
    filed through `set --evidence` reached no reader."""
    _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
         "--evidence", "CORRECTED CITATION ml/router.py:237")
    _run(sandbox, monkeypatch, "regen")
    markdown = sandbox.OUT.read_text(encoding="utf-8")
    assert "re-verified" in markdown, markdown[-1500:]
    assert "CORRECTED CITATION ml/router.py:237" in markdown
    assert "ORIGINAL CITATION some_file.py:1" in markdown, (
        "the original citation was replaced rather than joined")

    # NOT when they agree — `cmd_add` seeds both from one argument, so an
    # unconditional second cell would double 943 of 992 rows for nothing.
    _run(sandbox, monkeypatch, "set", "TEST-000", "--status", "OPEN",
         "--evidence", "ORIGINAL CITATION some_file.py:1")
    _run(sandbox, monkeypatch, "regen")
    assert "re-verified" not in sandbox.OUT.read_text(encoding="utf-8")


# ═══════════ 2 · THE SWEEP THE ROW'S `act` ASKED FOR ═══════════

#: ANCHORED, and the anchoring is the whole of the pattern's honesty. A bare
#: `name:123` fires on prose — `L64-B: 3`, `run 5: 12`, `2026-08-21: 4` — so a
#: citation must carry a token ending in a source extension, then a colon and a
#: line number, with no word character on either side.
CITATION = re.compile(
    r"(?<![\w/.-])"
    r"((?:[\w.-]+/)*[\w.-]+\.(?:py|js|html|css|sh|md|json|yml|yaml|toml|cfg|txt))"
    r":(\d+)(?:-(\d+))?"
    r"(?![\w.])")


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT),
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def _index(paths):
    """Citations are written repo-relative AND package-relative. Both resolve.

    A first pass matched only exact repo paths and reported 23 unresolvable
    citations; every one was a package-relative spelling of a file that exists.
    A sweep whose pattern has not been positively controlled reports its own
    blind spot as a finding.
    """
    by_suffix = defaultdict(list)
    for path in paths:
        parts = path.split("/")
        for i in range(len(parts)):
            by_suffix["/".join(parts[i:])].append(path)
    return set(paths), by_suffix


def _closed_rows():
    data = json.loads((ROOT / "docs" / "turbotab" / "data" / "findings.json")
                      .read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data["findings"]
    return [r for r in rows if r["status"] in TERMINAL], len(rows)


def test_the_citation_matcher_does_not_fire_on_prose():
    """**The negative control, and it comes before the count.**

    `AGENT_ONBOARD.md` §07 trap 5b: a matcher that fires on prose has silence
    that means nothing, and the same is true of one that fires on everything.
    """
    for prose in ("L64-B: 3 of 5", "run 5: 12 surfaces", "2026-08-21: 4 rows",
                  "commit 2761ab8", "13 checks, 0 unmet", "ratio 2.92:1",
                  "see §05:2 of the plan", "version 1.60.0"):
        assert not CITATION.search(prose), (prose, CITATION.search(prose))

    for citation, expected in (
            ("turbotab/api.py:29", ("turbotab/api.py", "29", None)),
            ("ml/router.py:237-240", ("ml/router.py", "237", "240")),
            ("web/index.html:3884", ("web/index.html", "3884", None)),
            (".githooks/lib.sh:90", (".githooks/lib.sh", "90", None))):
        match = CITATION.search(citation)
        assert match and match.groups() == expected, (citation, match)


def test_every_closed_rows_citation_still_resolves(capsys):
    """**The count the row's `act` asked for, and it is zero.**

    RESOLVES, not RIGHT — and the difference is the point rather than a
    caveat. This asks whether the file can be found and the line is inside it.
    It does not read the line and judge whether it still says what the row
    claims; `test_the_sweep_reports_citations_that_point_at_nothing` is the
    weaker structural proxy for that, and it does not return zero.
    """
    closed, total = _closed_rows()
    files, by_suffix = _index(_tracked())

    parsed, prose_only, broken = 0, 0, []
    for row in closed:
        hits = list(CITATION.finditer(row.get("ev") or ""))
        if not hits:
            if (row.get("ev") or "").strip():
                prose_only += 1
            continue
        for match in hits:
            parsed += 1
            path, start, end = match.group(1), int(match.group(2)), match.group(3)
            last = int(end) if end else start
            cite = f"{path}:{match.group(2)}" + (f"-{end}" if end else "")
            found = [path] if path in files else by_suffix.get(path, [])
            if len(found) != 1:
                broken.append((row["id"], cite,
                               f"{len(found)} tracked path(s) match"))
                continue
            lines = len((ROOT / found[0]).read_bytes().splitlines())
            if last > lines:
                broken.append((row["id"], cite,
                               f"{found[0]} has {lines} lines"))

    # THE INSTRUMENT'S OWN CONTROL, before its silence is quoted.
    assert parsed >= 250, (
        f"only {parsed} citations parsed out of {len(closed)} closed rows; the "
        f"pattern is not seeing the corpus and its silence means nothing")

    assert not broken, (
        f"{len(broken)} evidence citation(s) on CLOSED rows no longer resolve. "
        f"`ledger.py set --ev` can repair them now, which is the capability "
        f"TEST-109 was filed for:\n  "
        + "\n  ".join(f"{rid} {cite} — {why}" for rid, cite, why in broken))

    with capsys.disabled():
        print(f"\n  {len(closed)} closed rows of {total} · {parsed} citations "
              f"parsed · {prose_only} rows carry prose-only evidence · "
              f"{len(broken)} do not resolve")


def test_the_sweep_reports_citations_that_point_at_nothing(capsys):
    """The weaker measure, reported rather than gated.

    A `.py` citation whose whole cited span is blank or comment has very likely
    drifted — but **not always**, and that is why this counts rather than
    fails: `TEST-081`'s entire subject is a comment in `ml/router.py`, so
    citing one there is correct. A citation landing on a BLANK line is the
    sharper signal and is reported separately.
    """
    closed, _ = _closed_rows()
    files, by_suffix = _index(_tracked())
    blank, comment = [], []
    for row in closed:
        for match in CITATION.finditer(row.get("ev") or ""):
            path, start, end = match.group(1), int(match.group(2)), match.group(3)
            if not path.endswith(".py"):
                continue
            found = [path] if path in files else by_suffix.get(path, [])
            if len(found) != 1:
                continue
            lines = (ROOT / found[0]).read_text(encoding="utf-8").splitlines()
            span = lines[start - 1:(int(end) if end else start)]
            if not span:
                continue
            cite = f"{row['id']} {path}:{match.group(2)}"
            if all(not s.strip() for s in span):
                blank.append(cite)
            elif all((not s.strip()) or s.strip().startswith("#")
                     for s in span):
                comment.append(cite)

    # The control: this measure must be able to find something, or its number
    # is not a measurement.
    assert blank or comment, (
        "no closed row cites a blank or comment-only span, which would be a "
        "first for this corpus — check the resolver before believing it")

    with capsys.disabled():
        print(f"\n  citations landing on a BLANK line: {len(blank)}")
        for cite in blank:
            print(f"      {cite}")
        print(f"  citations landing on a COMMENT-only span: {len(comment)} "
              f"(some are correct — a row can be about a comment)")


def test_the_documented_flags_are_spelled_in_full(capsys):
    """The premise `allow_abbrev=False` rests on, asserted rather than assumed.

    Turning abbreviation off breaks every prefix on `set`. That was safe
    because nothing used one — and a document that starts using `--st` would
    break silently at the next reader rather than here.
    """
    offenders = []
    for name in ("docs/turbotab/LOOP.md", "docs/turbotab/README.md",
                 "docs/turbotab/prompts/AGENT_ONBOARD.md"):
        path = ROOT / name
        if not path.exists():
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if "ledger.py set" not in line:
                continue
            for flag in re.findall(r"--[a-z-]+", line):
                if flag not in ("--status", "--note", "--test", "--evidence",
                                "--ev", "--act", "--verified-ev"):
                    offenders.append(f"{name}:{number} {flag}")
    assert not offenders, (
        f"these documents pass an abbreviated flag to `ledger.py set`, which "
        f"is now an error rather than a silent redirect: {offenders}")
