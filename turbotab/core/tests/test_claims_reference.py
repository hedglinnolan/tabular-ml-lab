"""The claims reference and the claims' history stay what the registry generates (CLAIMS_REGISTRY.md
§5, §6; R0_PLAN.md R0a acceptance 5).

* The committed ``CLAIMS_REFERENCE.md`` is regenerated from the registry and must be identical, and
  ``turbotab/core/claims/history.jsonl`` must hold the version of every claim now in force.
* The committed frozen status file, ``turbotab/core/claims/status.json``, holds each claim's id,
  derived status and content hash, which is what rendering reads (``test_claims_frozen.py``).
* ``--check`` exits 1 when any of the three is stale and 0 when all are current, and writes
  nothing.
* A claim whose content hash changes keeps its old version in the history: the generator appends
  the new version after it, and an old manifest's hash still resolves to the words it ran under.
* An id the history holds stays registered (§1: a retired id stays, marked retracted): where the
  registry drops one, the generator writes nothing and it and ``--check`` exit 1.
* Proposed claims are listed under "Ruled, not built" (§6).
* Every grade change is a history line the generator writes, with its date, grader and reason and
  the grade it moved from, downgrades included (R1_PLAN.md R1-G): a part's first line records the
  grade it holds; a change whose source names no dated grader, the last line's grader and date
  again, a date before the last line's, or no reason other than the last line's, is refused and
  nothing is written; a swarm claim's part enters unsupported or is refused.
* A part the history holds is retired through a line, never dropped or renamed: the generator
  refuses a claim that no longer names one whose last line does not retire it, and writes nothing;
  a part the history never recorded cannot enter retired, and a swarm claim's part is retired only
  by someone other than its author.
* An entry names each referee's verdict with the grade it judged, and marks a referee or a basis
  audit that read a version of the part since changed.
"""
from __future__ import annotations

import json

import pytest

from turbotab.core import claims
from turbotab.core.reference import claims as reference
from turbotab.core.claims import Audit, Basis, Fires, NotRenderable, Part, Referee
from turbotab.core.tests.test_claims_fixtures import bound, registry, rule, swarm, teaching

REPO = reference.REPO


def test_the_committed_reference_and_history_are_what_the_registry_generates():
    reg = claims.registry()
    committed = reference.OUT.read_text(encoding="utf-8") if reference.OUT.is_file() else ""
    assert committed == reference.render(reg), (
        f"{reference.OUT.relative_to(REPO)} is stale: run `{reference.COMMAND}`")
    assert reference.pending(reg, reference.read_history(reference.HISTORY)) == [], (
        f"{reference.HISTORY.relative_to(REPO)} is stale: run `{reference.COMMAND}`")
    frozen = reg.frozen_path.read_text(encoding="utf-8") if reg.frozen_path.is_file() else ""
    assert frozen == claims.frozen_text(reg), (
        f"{reg.frozen_path.relative_to(REPO)} is stale: run `{reference.COMMAND}`")
    assert reference.main(["--check"]) == 0


def test_an_empty_registry_renders_its_headers_and_says_no_claim_is_registered():
    text = reference.render(claims.Registry())
    assert text.startswith("# TurboTab v2 claims reference\n")
    assert "No claim is registered yet." in text
    for heading in ("## Built", "## Ruled, not built", "## Retracted"):
        assert f"\n{heading}\n" in text


def test_a_proposed_claim_is_listed_as_ruled_not_built_and_a_built_one_as_built(tmp_path):
    proposed = teaching(id="planted.ruled", card="A ruled sentence.", ruled="R0_PLAN.md, planted",
                        tests=(Fires("planted/test_rule.py::test_not_yet", "construction"),))
    text = reference.render(registry(tmp_path, rule(), proposed))
    built, rest = text.split("\n## Ruled, not built\n", 1)
    ruled, retracted = rest.split("\n## Retracted\n", 1)
    assert "`planted.few_clusters`" in built and "`planted.few_clusters`" not in ruled
    assert "`planted.ruled`" in ruled and "R0_PLAN.md, planted" in ruled
    assert "`planted.ruled`" not in built
    assert "No claim is registered yet." not in text
    assert "Fewer than 10 clusters: intervals can still be too narrow." in built


def test_the_count_agrees_in_number_with_the_claims_it_counts(tmp_path):
    one = reference.render(registry(tmp_path, rule()))
    assert "It holds 1 claim: 1 built, 0 ruled and not built, 0 retracted." in one
    two = reference.render(registry(tmp_path, rule(), teaching()))
    assert "It holds 2 claims: 2 built, 0 ruled and not built, 0 retracted." in two


def _paths(tmp_path):
    out, history = tmp_path / "CLAIMS_REFERENCE.md", tmp_path / "history.jsonl"
    return out, history, ["--out", str(out), "--history", str(history)]


def test_check_exits_1_on_a_stale_reference_and_0_on_a_current_one_and_writes_nothing(tmp_path):
    reg = registry(tmp_path, rule())
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0  # writes all three
    before = (out.read_bytes(), history.read_bytes(), reg.frozen_path.read_bytes())
    assert reference.main(["--check", *args], registry=reg) == 0
    out.write_text(out.read_text(encoding="utf-8") + "hand edit\n", encoding="utf-8")
    stale = out.read_bytes()
    assert reference.main(["--check", *args], registry=reg) == 1
    assert (out.read_bytes(), history.read_bytes(), reg.frozen_path.read_bytes()) == (
        stale, *before[1:])


def test_check_exits_1_on_a_frozen_status_the_source_does_not_derive_and_0_on_one_it_does(
        tmp_path, capsys):
    reg = registry(tmp_path, rule(), freeze=False)
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0  # writes the frozen file beside the others
    frozen = reg.frozen_path
    written = frozen.read_text(encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 0
    tampered = written.replace('"status": "built"', '"status": "proposed"')
    assert tampered != written
    frozen.write_text(tampered, encoding="utf-8")
    before = (out.read_bytes(), history.read_bytes())
    capsys.readouterr()
    assert reference.main(["--check", *args], registry=reg) == 1
    err = capsys.readouterr().err
    assert str(frozen) in err and "planted.few_clusters" in err
    assert (out.read_bytes(), history.read_bytes()) == before, "the check writes nothing"
    assert frozen.read_text(encoding="utf-8") == tampered
    frozen.unlink()
    assert reference.main(["--check", *args], registry=reg) == 1
    assert reference.main(args, registry=reg) == 0
    assert frozen.read_text(encoding="utf-8") == written
    assert reference.main(["--check", *args], registry=reg) == 0


def test_check_exits_1_when_the_source_moves_a_status_the_frozen_file_holds(tmp_path):
    """A status change alone (a missing test written) leaves the frozen file stale."""
    late = Fires("planted/test_late.py::test_written_later", "construction")
    reg = registry(tmp_path, rule(tests=(*rule().tests, late)), freeze=False)
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0
    (reg.root / "planted" / "test_late.py").write_text("def test_written_later():\n    pass\n",
                                                       encoding="utf-8")
    out.write_text(reference.render(reg), encoding="utf-8")  # the reference brought up to date
    assert reference.main(["--check", *args], registry=reg) == 1
    assert reference.main(args, registry=reg) == 0
    assert '"status": "built"' in reg.frozen_path.read_text(encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 0


def test_check_exits_1_when_the_history_lacks_a_version_in_force_and_writes_nothing(tmp_path):
    reg = registry(tmp_path, rule())
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0
    history.write_text("", encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 1
    assert history.read_text(encoding="utf-8") == ""


def test_a_changed_hash_keeps_the_old_version_and_appends_the_new(tmp_path):
    out, history, args = _paths(tmp_path)
    old = rule()
    new = rule(card="Fewer than {few} clusters: the intervals can be too narrow.", version=2)
    assert reference.main(args, registry=registry(tmp_path, old)) == 0
    first = reference.read_history(history)
    assert [(h["id"], h["hash"]) for h in first] == [(old.id, claims.content_hash(old))]

    assert reference.main(["--check", *args], registry=registry(tmp_path, new)) == 1
    assert reference.main(args, registry=registry(tmp_path, new)) == 0
    lines = reference.read_history(history)
    assert [h["hash"] for h in lines] == [claims.content_hash(old), claims.content_hash(new)]
    assert lines[0] == first[0], "the old version is kept as it was written"
    assert lines[0]["claim"] == claims.content(old)
    assert lines[0]["claim"]["card"] == old.card and lines[0]["version"] == 1
    assert reference.main(["--check", *args], registry=registry(tmp_path, new)) == 0


def test_generating_twice_appends_nothing_the_second_time(tmp_path):
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule(), teaching())
    assert reference.main(args, registry=reg) == 0
    once = history.read_bytes()
    assert reference.main(args, registry=reg) == 0
    assert history.read_bytes() == once
    assert len(once.decode("utf-8").splitlines()) == 4, "a version and a grade line per claim"


def test_a_status_change_alone_makes_the_reference_stale_but_not_the_history(tmp_path):
    """Status is derived, not content: writing a claim's missing test moves the reference only."""
    out, history, args = _paths(tmp_path)
    late = Fires("planted/test_late.py::test_written_later", "construction")
    reg = registry(tmp_path, rule(tests=(*rule().tests, late)))
    assert reference.main(args, registry=reg) == 0
    (reg.root / "planted" / "test_late.py").write_text("def test_written_later():\n    pass\n",
                                                       encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 1
    assert reference.pending(reg, reference.read_history(history)) == []


def test_each_history_line_is_one_json_object_naming_its_claim(tmp_path):
    """A claim's version line comes first, then a grade line for each of its parts."""
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule())) == 0
    version, grade = (json.loads(x) for x in history.read_text(encoding="utf-8").splitlines())
    assert set(version) == {"id", "version", "hash", "claim"}
    assert version["id"] == "planted.few_clusters" and version["claim"]["id"] == version["id"]
    assert set(grade) == {"id", "part", "grade", "was", "date", "grader", "reason"}
    assert grade["id"] == version["id"] and grade["part"] == "value"


def test_a_claim_deleted_from_the_registry_fails_the_check_and_the_generator(tmp_path, capsys):
    """An id is never dropped (§1): once the history holds it, the registry must still register
    it, so the generator writes nothing and both it and ``--check`` exit 1."""
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule(), teaching())) == 0
    without = registry(tmp_path, rule(), freeze=False)
    before = (out.read_bytes(), history.read_bytes(), without.frozen_path.read_bytes())
    capsys.readouterr()
    assert reference.main(["--check", *args], registry=without) == 1
    assert reference.main(args, registry=without) == 1
    assert (out.read_bytes(), history.read_bytes(), without.frozen_path.read_bytes()) == before
    err = capsys.readouterr().err
    assert "planted.your_call" in err and "retract" in err
    out.write_text(reference.render(without), encoding="utf-8")  # a reference written regardless
    assert reference.main(["--check", *args], registry=without) == 1


def test_a_retracted_claim_stays_and_passes_the_check(tmp_path):
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule(), teaching())) == 0
    retracted = registry(tmp_path, rule(), teaching(retracted="2026-10-16: a planted retraction"))
    assert reference.main(args, registry=retracted) == 0
    assert reference.main(["--check", *args], registry=retracted) == 0
    assert [h["id"] for h in reference.read_history(history)] == [
        "planted.few_clusters", "planted.your_call", "planted.your_call"]
    assert "`planted.your_call`" in out.read_text(encoding="utf-8").split("\n## Retracted\n")[1]


# ── every grade change is a history line (R1_PLAN.md R1-G) ───────────────────────────────────

ID = "planted.few_clusters"
PASSAGE = Basis(cites=("friedman2001",), where="§10.10, p. 1203")


def _moves(history):
    return [(g["was"], g["grade"], g["date"], g["grader"], g["reason"])
            for g in reference.read_grades(history)]


def test_the_first_run_records_each_part_s_grade_its_grader_and_date(tmp_path):
    """A part's grader and date are its own dated grader's; where it names none, its author's
    grade on the day the generator records it."""
    out, history, args = _paths(tmp_path)
    graded = rule(parts=(Part("value", "sourced", PASSAGE, graded_by="a basis audit, 2026-10-16"),
                         Part("scope", "judgment", Basis(reason="r"))))
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, graded)) == 0
    assert reference.read_grades(history) == [
        {"id": ID, "part": "value", "grade": "sourced", "was": "", "date": "2026-10-16",
         "grader": "a basis audit", "reason": reference.FIRST_RECORDED},
        {"id": ID, "part": "scope", "grade": "judgment", "was": "", "date": "2026-10-18",
         "grader": "the R0a builder", "reason": reference.FIRST_RECORDED}]
    assert [h["id"] for h in reference.read_history(history)] == [ID]


def test_an_upgrade_and_a_downgrade_are_each_a_line_with_the_grade_it_moved_from(tmp_path):
    out, history, args = _paths(tmp_path)
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, rule())) == 0
    up = rule(parts=(Part("value", "sourced", PASSAGE, graded_by="a basis audit, 2026-10-19",
                          regraded="the passage says it in this regime"),))
    assert reference.main(["--check", *args], registry=registry(tmp_path, up)) == 1
    assert reference.main([*args, "--date", "2026-10-20"], registry=registry(tmp_path, up)) == 0
    down = rule(parts=(Part("value", "judgment", Basis(reason="r"),
                            graded_by="a domain reviewer, 2026-10-21",
                            regraded="the passage is about another design"),))
    assert reference.main([*args, "--date", "2026-10-22"], registry=registry(tmp_path, down)) == 0
    assert _moves(history) == [
        ("", "judgment", "2026-10-18", "the R0a builder", reference.FIRST_RECORDED),
        ("judgment", "sourced", "2026-10-19", "a basis audit",
         "the passage says it in this regime"),
        ("sourced", "judgment", "2026-10-21", "a domain reviewer",
         "the passage is about another design")]
    assert reference.main(["--check", *args], registry=registry(tmp_path, down)) == 0


UNRECORDED = {
    "no reason": Part("value", "unsupported", graded_by="a basis audit, 2026-10-19"),
    "the last line's reason again": Part("value", "unsupported",
                                         graded_by="a basis audit, 2026-10-19",
                                         regraded=reference.FIRST_RECORDED),
    "no dated grader": Part("value", "unsupported", regraded="the reason no longer holds"),
}


@pytest.mark.parametrize("case", sorted(UNRECORDED))
def test_a_grade_change_without_its_grader_date_or_reason_is_refused_and_nothing_is_written(
        tmp_path, capsys, case):
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule())
    assert reference.main([*args, "--date", "2026-10-18"], registry=reg) == 0
    before = (out.read_bytes(), history.read_bytes(), reg.frozen_path.read_bytes())
    moved = registry(tmp_path, rule(parts=(UNRECORDED[case],)), freeze=False)
    capsys.readouterr()
    assert reference.main(args, registry=moved) == 1
    assert reference.main(["--check", *args], registry=moved) == 1
    assert (out.read_bytes(), history.read_bytes(), moved.frozen_path.read_bytes()) == before
    assert f"{ID}: part 'value' moves from judgment to unsupported" in capsys.readouterr().err


def test_a_grade_change_with_its_grader_date_and_reason_is_recorded(tmp_path):
    """The other side of the refusals: the same move, with all three, is one line."""
    out, history, args = _paths(tmp_path)
    assert reference.main([*args, "--date", "2026-10-18"], registry=registry(tmp_path, rule())) == 0
    moved = rule(parts=(Part("value", "unsupported", graded_by="a basis audit, 2026-10-19",
                             regraded="the reason no longer holds"),))
    assert reference.main(args, registry=registry(tmp_path, moved)) == 0
    assert _moves(history)[-1] == ("judgment", "unsupported", "2026-10-19", "a basis audit",
                                   "the reason no longer holds")


def test_a_swarm_claim_enters_unsupported_or_is_refused(tmp_path, capsys):
    out, history, args = _paths(tmp_path)
    judged = Part("value", "judgment", Basis(reason="r"), graded_by="the orchestrator, 2026-10-19",
                  regraded="the orchestrator's call, on the swarm's runs")
    capsys.readouterr()
    assert reference.main(args, registry=registry(tmp_path, swarm(parts=(judged,)),
                                                  freeze=False)) == 1
    assert not history.exists() and not out.exists()
    assert "enters unsupported" in capsys.readouterr().err
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, swarm())) == 0
    assert reference.main(args, registry=registry(tmp_path, swarm(parts=(judged,)))) == 0
    assert _moves(history) == [
        ("", "unsupported", "2026-10-18", "theorist A", reference.FIRST_RECORDED),
        ("unsupported", "judgment", "2026-10-19", "the orchestrator",
         "the orchestrator's call, on the swarm's runs")]


def test_check_exits_1_on_a_history_with_versions_but_no_grade_lines_and_the_generator_adds_them(
        tmp_path):
    """The history as R0 left it: every version written, no grade line; the generator records the
    grade each part holds and appends no version."""
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule(), teaching())
    assert reference.main(args, registry=reg) == 0
    versions = [x for x in history.read_text(encoding="utf-8").splitlines()
                if '"hash"' in x]
    history.write_text("".join(x + "\n" for x in versions), encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 1
    assert reference.main([*args, "--date", "2026-10-18"], registry=reg) == 0
    assert reference.main(["--check", *args], registry=reg) == 0
    assert len(reference.read_history(history)) == 2
    assert [(g["id"], g["grade"]) for g in reference.read_grades(history)] == [
        (ID, "judgment"), ("planted.your_call", "judgment")]


def test_each_part_s_first_grade_line_is_the_grade_its_claim_s_first_version_held():
    """In the committed history, the grade lines begin where the versions written before them left
    each part, so recording them moved no grade."""
    first_version, first_line = {}, {}
    for v in reference.read_history(reference.HISTORY):
        first_version.setdefault(v["id"], v)
    for g in reference.read_grades(reference.HISTORY):
        first_line.setdefault((g["id"], g["part"]), g)
    assert len(first_line) == sum(len(c.parts) for c in claims.registry())
    for (id, part), g in first_line.items():
        held = {p["name"]: p["grade"] for p in first_version[id]["claim"]["parts"]}
        assert g["grade"] == held[part], (id, part)


def test_an_entry_names_its_origin_its_basis_audit_its_referees_and_why_it_was_regraded(tmp_path):
    part = Part("value", "sourced", PASSAGE, graded_by="the orchestrator, 2026-10-19",
                audited_by=Audit("a basis audit, 2026-10-19"), regraded="two referees upheld it",
                referees=(Referee("referee A, 2026-10-19", "upheld"),
                          Referee("referee B, 2026-10-19", "upheld"),
                          Referee("referee C, 2026-10-19", "not_upheld", fork_of="referee A")))
    text = reference.render(registry(tmp_path, bound(swarm(parts=(part,))), rule()))
    swarm_entry = text.split("### `planted.swarm_found`")[1]
    assert "*rule* · from *swarm* · weakest part *sourced*" in swarm_entry
    assert ("; basis checked against its source by a basis audit, 2026-10-19; upheld as sourced by "
            "referee A, 2026-10-19, referee B, 2026-10-19; not upheld as sourced by referee C, "
            "2026-10-19 (a fork of referee A); regraded: two referees upheld it") in swarm_entry
    assert "since changed" not in swarm_entry
    plain = text.split("### `planted.few_clusters`")[1].split("### ")[0]
    assert "*rule* · from *engine*" in plain
    for absent in ("basis checked", "upheld", "regraded:"):
        assert absent not in plain


def test_an_entry_marks_a_referee_or_an_audit_that_read_a_version_since_changed(tmp_path):
    """A verdict is shown with the grade it judged; one on another version of the part says so,
    so the reference never shows an old verdict as one on the part as it stands."""
    was = bound(swarm(parts=(Part("value", "sourced", PASSAGE,
                                  graded_by="the orchestrator, 2026-10-19",
                                  audited_by=Audit("a basis audit, 2026-10-19"),
                                  referees=(Referee("referee A, 2026-10-19", "upheld"),)),)))
    old = was.parts[0]
    now = swarm(parts=(Part("value", "proved", Basis(derivation="notes/derivation.md"),
                            graded_by="the orchestrator, 2026-10-20",
                            audited_by=old.audited_by,
                            referees=(*old.referees, Referee("referee B, 2026-10-20", "upheld"))),))
    entry = reference.render(registry(tmp_path, bound(now))).split("### `planted.swarm_found`")[1]
    assert ("basis checked against its source by a basis audit, 2026-10-19, on a version of the "
            "part since changed") in entry
    assert "upheld as proved by referee B, 2026-10-20" in entry
    assert ("upheld as sourced by referee A, 2026-10-19, on a version of the part since changed"
            in entry)


def test_an_entry_shows_a_retired_part_and_the_grade_the_parts_it_asserts_give(tmp_path):
    claim = rule(parts=(Part("value", "judgment", Basis(reason="r")),
                        Part("scope", "retired", graded_by="a domain reviewer, 2026-10-19",
                             regraded="the sentence no longer says it")))
    entry = reference.render(registry(tmp_path, claim)).split("### `planted.few_clusters`")[1]
    assert "· weakest part *judgment* ·" in entry
    assert ("  - scope: *retired*, graded by a domain reviewer, 2026-10-19; regraded: the sentence "
            "no longer says it") in entry


# ── a grade change names who made it, and when (RG-2) ────────────────────────────────────────

VERIFIED = "the R0b verifier, 2026-10-16"
SOURCED = Part("value", "sourced", PASSAGE, graded_by=VERIFIED)


def _recorded_then(tmp_path, capsys, later):
    """Record SOURCED's first line, then run the generator on ``later``: its exit code, whether
    the history, reference and frozen file were left as they were, and what it printed."""
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule(parts=(SOURCED,)))
    assert reference.main([*args, "--date", "2026-10-18"], registry=reg) == 0
    before = (out.read_bytes(), history.read_bytes(), reg.frozen_path.read_bytes())
    moved = registry(tmp_path, rule(parts=later), freeze=False)
    capsys.readouterr()
    code = reference.main([*args, "--date", "2026-10-25"], registry=moved)
    after = (out.read_bytes(), history.read_bytes(), moved.frozen_path.read_bytes())
    return code, after == before, capsys.readouterr().err, history


STALE_CHANGES = {
    "a downgrade under the grader and date that upheld the higher grade": Part(
        "value", "judgment", Basis(reason="r"), graded_by=VERIFIED,
        regraded="2026-10-25 audit: the passage does not say it"),
    "a change dated before the last line": Part(
        "value", "judgment", Basis(reason="r"), graded_by="a domain reviewer, 2019-05-01",
        regraded="the passage does not say it"),
}


@pytest.mark.parametrize("case", sorted(STALE_CHANGES))
def test_a_change_under_its_last_line_s_grader_and_date_or_dated_before_it_is_refused(
        tmp_path, capsys, case):
    code, unchanged, err, _ = _recorded_then(tmp_path, capsys, (STALE_CHANGES[case],))
    assert code == 1 and unchanged
    assert f"{ID}: part 'value' moves from sourced to judgment" in err


@pytest.mark.parametrize("graded_by, when", [
    ("a domain reviewer, 2026-10-25", "2026-10-25"),
    ("the R0b verifier, 2026-10-25", "2026-10-25"),
    ("a domain reviewer, 2026-10-16", "2026-10-16"),
])
def test_the_same_change_under_a_new_grader_or_a_new_day_is_recorded(tmp_path, capsys,
                                                                      graded_by, when):
    """The other side: another grader, or the same one on a later day, records the downgrade on
    its own day, no earlier than the last line's."""
    down = Part("value", "judgment", Basis(reason="r"), graded_by=graded_by,
                regraded="2026-10-25 audit: the passage does not say it")
    code, _, _, history = _recorded_then(tmp_path, capsys, (down,))
    assert code == 0
    assert _moves(history)[-1] == ("sourced", "judgment", when, graded_by.rpartition(",")[0],
                                   "2026-10-25 audit: the passage does not say it")


# ── a part is retired through a line, never dropped or renamed (RG-4) ────────────────────────

JUDGED = Part("value", "judgment", Basis(reason="r"))
SCOPED = Part("scope", "sourced", PASSAGE, graded_by="a basis audit, 2026-10-16")
LEFT = {
    "dropped": (JUDGED,),
    "renamed": (JUDGED, Part("value_scope", "judgment", Basis(reason="r"))),
}


@pytest.mark.parametrize("case", sorted(LEFT))
def test_a_part_dropped_or_renamed_is_refused_and_nothing_is_written(tmp_path, capsys, case):
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule(parts=(JUDGED, SCOPED)))
    assert reference.main([*args, "--date", "2026-10-18"], registry=reg) == 0
    before = (out.read_bytes(), history.read_bytes(), reg.frozen_path.read_bytes())
    left = registry(tmp_path, rule(parts=LEFT[case]), freeze=False)
    capsys.readouterr()
    assert reference.main(args, registry=left) == 1
    assert reference.main(["--check", *args], registry=left) == 1
    assert (out.read_bytes(), history.read_bytes(), left.frozen_path.read_bytes()) == before
    err = capsys.readouterr().err
    assert f"{ID}: part 'scope'" in err and "no longer names it" in err


def test_a_part_retired_through_a_line_may_then_leave_its_claim(tmp_path):
    """The other side: retired with its grader, date and reason, the part's line is written, and
    the claim may then leave it out."""
    out, history, args = _paths(tmp_path)
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, rule(parts=(JUDGED, SCOPED)))) == 0
    retiring = Part("scope", "retired", graded_by="a domain reviewer, 2026-10-19",
                    regraded="the sentence no longer says it")
    assert reference.main(args, registry=registry(tmp_path, rule(parts=(JUDGED, retiring)))) == 0
    assert _moves(history)[-1] == ("sourced", "retired", "2026-10-19", "a domain reviewer",
                                   "the sentence no longer says it")
    gone = registry(tmp_path, rule(parts=(JUDGED,)))
    assert reference.main(args, registry=gone) == 0
    assert reference.main(["--check", *args], registry=gone) == 0
    assert claims.grades_recorded(claims.Registry(list(gone), root=gone.root,
                                                  history=history)) == []


def test_a_part_the_history_never_recorded_cannot_enter_retired(tmp_path, capsys):
    out, history, args = _paths(tmp_path)
    claim = rule(parts=(JUDGED, Part("scope", "retired", graded_by="a domain reviewer, 2026-10-19",
                                     regraded="never said")))
    capsys.readouterr()
    assert reference.main(args, registry=registry(tmp_path, claim, freeze=False)) == 1
    assert not history.exists()
    assert f"{ID}: part 'scope' is retired" in capsys.readouterr().err


SWARM = "planted.swarm_found"


@pytest.mark.parametrize("by, recorded", [("theorist A, 2026-10-19", False),
                                          ("the orchestrator, 2026-10-19", True)])
def test_a_swarm_part_is_retired_only_by_someone_other_than_its_author(tmp_path, capsys, by,
                                                                       recorded):
    out, history, args = _paths(tmp_path)
    entering = swarm(parts=(Part("value", "unsupported"), Part("scope", "unsupported")))
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, entering)) == 0
    retiring = swarm(parts=(Part("value", "unsupported"),
                            Part("scope", "retired", graded_by=by, regraded="not said")))
    capsys.readouterr()
    assert reference.main(args, registry=registry(tmp_path, retiring)) == (0 if recorded else 1)
    if not recorded:
        err = capsys.readouterr().err
        assert f"{SWARM}: part 'scope' moves from unsupported to retired" in err
        assert "a grader other than its author" in err
    assert (_moves(history)[-1][1] == "retired") is recorded


def test_a_swarm_claim_that_drops_its_unsupported_part_stays_unrenderable(tmp_path):
    """The verifier's probe E: a swarm claim with one part through the gate and one unsupported
    is proposed; dropped, the unsupported part would make it built, but the generator refuses the
    drop and freezes nothing, so the claim, whose hash has moved, still renders nowhere."""
    out, history, args = _paths(tmp_path)
    entering = swarm(parts=(Part("value", "unsupported"), Part("scope", "unsupported")))
    assert reference.main([*args, "--date", "2026-10-18"],
                          registry=registry(tmp_path, entering)) == 0
    gated = bound(swarm(parts=(
        Part("value", "sourced", PASSAGE, graded_by="the orchestrator, 2026-10-19",
             regraded="two referees upheld it", audited_by=Audit("a basis audit, 2026-10-19"),
             referees=(Referee("referee A, 2026-10-19", "upheld"),
                       Referee("referee B, 2026-10-19", "upheld"))),
        Part("scope", "unsupported"))))
    reg = registry(tmp_path, gated, freeze=False)
    assert reference.main(args, registry=reg) == 0 and reg.frozen_status(SWARM) == "proposed"
    dropped = registry(tmp_path, swarm(parts=gated.parts[:1]), freeze=False)
    assert dropped.status(SWARM) == "built"
    assert reference.main(args, registry=dropped) == 1
    with pytest.raises(NotRenderable):
        dropped.card(SWARM, strict=False)
