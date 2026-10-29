"""The walker (``research/swarm/walk.py``; the owner's ruling of 2026-10-23, "Go with D"): its oracle
on transcripts, logs and views built here, both ways for each of its five defects; the actions it
offers and draws at a seed; the end of the path's second seal and revisit after it, accepted by a
stub (the real engine cannot be made to accept them) and refused; the report refused where its
engines differ; a run that skips a seed already reported; whole walks on a server modeled here
(:class:`Served`): the round owed where a decision is reverted and no round sent one, and a fault of
the walker's own after the walk, which ends only that walk; a press of Fit whose wait ran out, on
the arms' tests' modeled server (``test_arms.Server``), its fit waited out at the end of the path
or stopped at the run's time budget, and kept in the transcript as what it came to, so Fit refused
again after its way forward is a dead end there too; and, slow, one walk through the driver's own
commands on a small dataset that is not one of the pilot's (``clinical_readmission_predict``, 480
rows), whose second seal and revisit after it the engine and the driver refuse.

Known truth: each command's payload as the driver prints it (``driver.main``'s JSON: ``refused``
with its ``by``, ``status``, ``code``, ``message`` and ``exits``; ``recorded``; a press of Fit's
``pressed`` and ``fit``), built here; a stage's status as the engine serves it (``status``,
``error``, ``refusal``); a decision log appended here with the engine's own log
(``decisions.DecisionLog``) and a fit record written from it, its digest the SHA-256 of the plan's
canonical JSON computed here with hashlib; each draw from numpy's generator at the walker's seed,
drawn here; the modeled server's log the engine's own, and its refusal of a revert of a decision
already reverted the engine's (``decisions._check_revert``); the time waited from the arms' tests'
hand clock, which only the waits move, derived here from the budget."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from research.swarm import arms, walk

RID = "run_01-" + "0" * 32


def argv(command, *ids, **flags):
    """A command line as ``arms.Driven`` builds it: the command, ``--run`` and the run's id, the
    ids, then its flags."""
    out = [command, "--run", RID, *ids]
    for k, v in flags.items():
        out += [f"--{k}", v]
    return out


def refused(code="no_way", message="The answer cannot be recorded.", exits=(), by="server",
            status=409):
    return {"refused": {"by": by, "status": status, "code": code, "message": message,
                        "exits": list(exits)}}


def way(oid, decision, label="A way"):
    return {"id": oid, "label": label, "decision": decision}


def ok(kind="set_missing", rid="r1", seq=3):
    return {"recorded": {"id": rid, "seq": seq, "decision": {"kind": kind}}, "next": None}


def entries(*commands):
    """The transcript of ``commands``, each ``(argv, exit, payload)``, as the walker keeps it."""
    return [walk.entry_of(a, code, payload, n=i) for i, (a, code, payload) in enumerate(commands)]


# ── (1) a 500, the driver's failure, a fit error the engine did not type ─────


def test_a_500_the_drivers_failure_and_an_untyped_fit_error_are_defects():
    t = entries(
        (argv("answer", "impute", reason="r"), 1, refused("http_500", "Internal Server Error",
                                                          status=500)),
        (argv("revisit", "r9", reason="r"), 1, refused("driver_failed", "The driver failed: "
                                                       "KeyError: 'x'", by="driver",
                                                       status=None)),
        (argv("fit", wait="240"), 0, {"pressed": {}, "fit": {"status": "error",
                                                             "error": "KeyError: 'x'"}}))
    found = walk.failures_of(t)
    assert [(d["defect"], d["command"], d["code"]) for d in found] == [
        ("failed", "answer", "http_500"), ("failed", "revisit", "driver_failed"),
        ("failed", "fit", "fit_failed")]
    assert found[0]["request"]["args"][:1] == ["impute"] and found[2]["words"] == "KeyError: 'x'"


def test_a_typed_refusal_and_a_fit_refused_in_words_are_no_failure():
    t = entries(
        (argv("answer", "impute", reason="r"), 1, refused()),
        (argv("fit", wait="240"), 1, {"pressed": {}, "fit": {"status": "error"},
                                      **refused("too_few_rows", "The answers leave 3 rows.",
                                                by="engine", status=None)}),
        (argv("fit", wait="240"), 0, {"pressed": {}, "fit": {"status": "fresh", "key": "k"}}))
    assert walk.failures_of(t) == []


def stage_view(**stages):
    return {"stages": {"working": {"status": "fresh", "key": "w1"}, **{
        name: {"status": "error", "key": f"{name}1", "error": words, "refusal": refusal}
        for name, (words, refusal) in stages.items()}}}


REFUSAL = {"stage": "cohort", "code": "too_few_rows", "message": "The answers leave 3 of 60 rows.",
           "exits": [{"label": "Fill the blanks", "decision": {"kind": "set_missing",
                                                               "strategy": "impute"}}]}


def test_a_stage_in_error_the_engine_did_not_type_is_a_defect_and_a_typed_one_is_not():
    view = stage_view(effects=("ValueError: no coefficient table", None), cohort=(
        "ValueError: The answers leave 3 of 60 rows.", REFUSAL),
        fit=("Needs 'cohort', which failed.", REFUSAL))
    found = walk.stages_of(view)
    assert [(d["defect"], d["stage"], d["words"]) for d in found] == [
        ("stage_failed", "effects", "ValueError: no coefficient table")]
    assert walk.stages_of(stage_view(cohort=("ValueError: refused", REFUSAL))) == []


@pytest.mark.parametrize("upstream, said", [
    (("ValueError: no coefficient table", None), ["effects"]),
    (("ValueError: The answers leave 3 of 60 rows.", REFUSAL), []),
    (None, ["fit"])], ids=["upstream_failed", "upstream_refused", "upstream_not_in_error"])
def test_a_stage_that_only_names_a_failed_upstream_is_not_said_again(upstream, said):
    """A stage whose error only names one upstream ("Needs 'effects', which failed.") is not said
    again where that upstream is in error, untyped (said once, for itself) or a typed refusal (no
    defect); where the stage it names is not in error, its error is its own, and a defect."""
    view = stage_view(fit=("Needs 'effects', which failed.", None),
                      **({"effects": upstream} if upstream else {}))
    assert [d["stage"] for d in walk.stages_of(view) if d["defect"] == "stage_failed"] == said
    # Words that only begin as the upstream's are the stage's own.
    view = stage_view(fit=("Needs 'effects', which failed. KeyError: 'b'", None),
                      effects=("ValueError: no coefficient table", None))
    assert [d["stage"] for d in walk.stages_of(view)] == ["fit", "effects"]


# ── (2) a refusal with no code or no words ───────────────────────────────────


@pytest.mark.parametrize("code, message, wordless", [
    ("no_way", "The answer cannot be recorded.", False), (None, "Words.", True),
    ("no_way", "", True), ("no_way", None, True), ("", "Words.", True)],
    ids=["code_and_words", "no_code", "empty_words", "no_words", "empty_code"])
def test_a_refusal_with_no_code_or_no_words_is_a_defect(code, message, wordless):
    t = entries((argv("answer", "impute", reason="r"), 1, refused(code, message)))
    found = walk.wordless_of(t)
    assert bool(found) is wordless
    if wordless:
        assert found[0]["defect"] == "wordless" and found[0]["command"] == "answer"


def test_a_stop_or_a_stage_refusal_with_no_words_is_a_defect():
    stops = [{"by": "engine", "kind": "refused", "words": "stopped",
              "detail": {"kind": "refused", "code": "too_few_rows", "message": None}}]
    assert [d["defect"] for d in walk.wordless_of([], stops)] == ["wordless"]
    worded = [{"by": "engine", "kind": "refused", "words": "stopped",
               "detail": {"kind": "refused", "code": "too_few_rows", "message": "Words."}}]
    assert walk.wordless_of([], worded) == []
    view = stage_view(cohort=("ValueError: x", {**REFUSAL, "message": ""}))
    assert [d["defect"] for d in walk.stages_of(view)] == ["wordless"]


# Where a stop's refusal came from a command the transcript holds, as the code names it: an answer
# of the arm's walk (``arms.met_answer``, "walk"), the walker's own command (``Walker.check``'s
# default, "walker"), a press of Fit and the export (``Walker.check``'s and the stops of
# ``Walker.settle`` and ``Walker.exported``). A stage's refusal (``arms.met_stages``, "stage") is
# no command's.
@pytest.mark.parametrize("at, said", [
    ("walk", False), ("walker", False), ("fit", False), ("export", False), ("stage", True),
    (None, True)])
def test_a_wordless_stop_is_said_once_where_its_command_is_in_the_transcript(at, said):
    """A stop's refusal with no words is said from the stop only where no command of the
    transcript said it; where it came from a command the transcript holds, that command's entry
    says it, once."""
    stop = {"by": "engine", "kind": "refused", "words": "stopped",
            "detail": {"kind": "refused", "code": "too_few_rows", "message": None, "at": at}}
    assert bool(walk.wordless_of([], [stop])) is said
    t = entries((argv("answer", "impute", reason="r"), 1, refused("too_few_rows", None)))
    found = walk.wordless_of(t, [stop])
    assert [d["command"] for d in found] == ["answer", *([None] if said else [])]


# ── (3) a dead end ───────────────────────────────────────────────────────────

FILL = way("x-fill", {"kind": "set_missing", "strategy": "impute"})
DROP = way("x-drop", {"kind": "set_missing", "strategy": "complete_case"})


def test_a_way_forward_refused_again_with_the_same_code_is_a_dead_end():
    t = entries(
        (argv("answer", "single", reason="r"), 1, refused("single_fill", "No.", [FILL, DROP])),
        (argv("ask", wait="90"), 0, {"question": {"key": "missing", "status": "open"}}),
        (argv("answer", "x-fill", reason="r"), 1, refused("single_fill", "No again.", [FILL])))
    found = walk.dead_ends_of(t)
    assert [(d["defect"], d["code"], d["way"]) for d in found] == [
        ("dead_end", "single_fill", "x-fill")]
    assert found[0]["refused"]["n"] == 0 and found[0]["request"]["n"] == 2


@pytest.mark.parametrize("taken", [
    (0, ok()), (1, refused("another_code", "Another reason.")),
], ids=["the_way_works", "refused_with_another_code"])
def test_a_way_forward_that_works_or_is_refused_for_another_reason_is_no_dead_end(taken):
    t = entries(
        (argv("answer", "single", reason="r"), 1, refused("single_fill", "No.", [FILL, DROP])),
        (argv("answer", "x-fill", reason="r"), *taken))
    assert walk.dead_ends_of(t) == []


@pytest.mark.parametrize("answered, dead_end", [
    ("x-fill", True), ("x-drop", True), ("single", False), ("x-other", False)],
    ids=["its_first_way", "its_second_way", "the_answer_again", "an_id_none_of_its_ways"])
def test_an_answer_that_is_none_of_the_ways_refused_again_is_no_dead_end(answered, dead_end):
    """Rule 3's "the way was taken", both ways: after a refusal whose ways record something, the
    next command that records something is an answer refused again with the same code; it is a
    dead end where that answer names one of the refusal's ways, and none where it names another
    id (the answer refused given again, or an id that is none of its ways): no way was taken."""
    t = entries(
        (argv("answer", "single", reason="r"), 1, refused("single_fill", "No.", [FILL, DROP])),
        (argv("answer", answered, reason="r"), 1, refused("single_fill", "No again.", [FILL])))
    found = walk.dead_ends_of(t)
    assert [(d["code"], d["way"]) for d in found] == ([("single_fill", answered)] if dead_end
                                                      else [])


def test_a_refusal_whose_ways_record_nothing_or_are_not_taken_is_no_dead_end():
    """A way with no decision (a wait) is no way to take; a way is taken only by the next command
    that records something after its refusal (here a revisit, so the later answer of the way is
    not that refusal's)."""
    wait = {"id": None, "label": "Wait for the fit to finish", "decision": None}
    t = entries(
        (argv("seal", "linear", reason="r"), 1, refused("fit_not_fresh", "Not yet.", [wait])),
        (argv("fit", wait="240"), 0, {"pressed": {}, "fit": {"status": "fresh"}}),
        (argv("answer", "single", reason="r"), 1, refused("single_fill", "No.", [FILL])),
        (argv("revisit", "r2", reason="r"), 0, ok("revert")),
        (argv("answer", "x-fill", reason="r"), 1, refused("single_fill", "No.", [FILL])))
    assert walk.dead_ends_of(t) == []


SELECT = way("x-sel", {"kind": "select_models", "models": ["linear"]}, "Fit another family")
TRIAGE = way("x-tri", {"kind": "confirm_sweep"}, "Confirm the triage")
FRESH_FIT = {"pressed": {}, "fit": {"status": "fresh", "key": "k2"}}
EXPORTED = {"exported": {"path": "export.zip", "bytes": 10, "files": []}}


def fit_refused(code="too_few_rows", exits=(FILL,)):
    return {"pressed": {}, "fit": {"status": "error"},
            **refused(code, "The answers leave 3 of 60 rows.", exits, by="engine", status=None)}


def retried(command):
    """A refusal of ``command`` (a press of Fit, the seal or the export) with a way that records
    something; the way taken and recorded, the walk going on as the walker's does (for the export,
    the ask, Fit and the results); and the command given again, ``(exit, payload)`` left to the
    caller."""
    if command == "export":
        return [(argv("export", wait="240"), 1, refused("result_failed", "Table 2 failed.",
                                                        [SELECT])),
                (argv("answer", "x-sel", reason="r", marker="design"), 0, ok("select_models")),
                (argv("ask", wait="90"), 0, {"question": {"key": "models", "status": "settled"},
                                             "fit_reached": True}),
                (argv("fit", wait="240"), 0, FRESH_FIT),
                (argv("results", wait="240"), 0, {"results": {}})], argv("export", wait="240")
    if command == "fit":
        return [(argv("fit", wait="240"), 1, fit_refused()),
                (argv("answer", "x-fill", reason="r", marker="design"), 0, ok("set_missing")),
                (argv("ask", wait="90"), 0, {"question": None, "fit_reached": True})], \
            argv("fit", wait="240")
    return [(argv("seal", "linear", reason="r"), 1, refused("triage_open", "Open noticings.",
                                                            [TRIAGE])),
            (argv("answer", "x-tri", reason="r", marker="design"), 0, ok("confirm_sweep"))], \
        argv("seal", "linear", reason="r")


AGAIN = {"export": (1, refused("result_failed", "Table 2 failed again.", [SELECT])),
         "fit": (1, fit_refused()),
         "seal": (1, refused("triage_open", "Open noticings again.", [TRIAGE]))}
GIVEN = {"export": (0, EXPORTED), "fit": (0, FRESH_FIT), "seal": (0, ok("open_seal"))}
OTHER = {"export": (1, refused("fit_not_fresh", "Not yet.")),
         "fit": (1, fit_refused("separation", ())),
         "seal": (1, refused("fit_not_fresh", "Not yet."))}


@pytest.mark.parametrize("command", ["export", "fit", "seal"])
def test_a_fit_seal_or_export_refused_again_after_its_way_is_taken_is_a_dead_end(command):
    """The motivating shape: a typed refusal whose way forward, taken and recorded, leaves the
    command it refused refused again with the same code."""
    before, again = retried(command)
    t = entries(*before, (again, *AGAIN[command]))
    found = walk.dead_ends_of(t)
    assert [(d["defect"], d["command"], d["code"], d["refused"]["n"], d["taken"]["n"],
             d["request"]["n"]) for d in found] == [
        ("dead_end", command, t[0]["code"], 0, 1, len(t) - 1)]
    assert found[0]["way"] == t[1]["ids"][0] and found[0]["words"] == t[-1]["message"]
    # The walk's limit on its tries (the export's, the seal's) stops it, and it is a defect.
    stop = {"by": "walk", "kind": "stuck", "words": "Refused each time.", "detail": None}
    assert walk.classed(record(transcript=t, stops=[stop]))[0] == "defect"


@pytest.mark.parametrize("command", ["export", "fit", "seal"])
@pytest.mark.parametrize("then", ["given", "another_code", "way_not_taken", "another_answer"])
def test_a_fit_seal_or_export_given_after_its_way_or_without_it_is_no_dead_end(command, then):
    """Silent where the way works (the command given again), where the command is refused again
    for another reason, and where the way was not taken: a revisit recorded first, so the later
    answer of the way and the refusal again are not that refusal's; or an answer recorded in its
    place that is none of the refusal's ways, the command then refused again with the same code."""
    before, again = retried(command)
    if then == "way_not_taken":
        before = [before[0], (argv("revisit", "r2", reason="r"), 0, ok("revert")), *before[1:]]
    if then == "another_answer":
        before = [before[0], (argv("answer", "x-other", reason="r", marker="design"), 0,
                              ok("set_missing")), *before[2:]]
    t = entries(*before, (again, *{"given": GIVEN, "another_code": OTHER}.get(
        then, AGAIN)[command]))
    assert walk.dead_ends_of(t) == []
    assert walk.classed(record(transcript=t, reached=True))[0] == "clean"


# ── (4) the plan's digest, refolded from the log ─────────────────────────────


def plan_sha(records):
    """The plan's digest computed here: SHA-256 over the canonical JSON (keys sorted, no spaces,
    UTF-8) of the plan the records fold to."""
    from turbotab.core import decisions as D, plan_lock

    text = json.dumps(plan_lock.plan_of(D.fold(records)), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def logged(tmp_path):
    from turbotab.core import decisions as D

    log = D.DecisionLog(tmp_path / "decisions.jsonl")
    for d in ({"kind": "set_target", "column": "y"}, {"kind": "set_purpose", "purpose": "prediction"},
              {"kind": "set_exclusions", "rules": []}):
        log.append(d)
    return log


def fit_record(records, *, at, digest=None, key="k1", seq=1):
    from turbotab.core import provenance

    live = provenance.in_force(records)
    return {"id": f"f{seq}", "seq": seq, "at": at.isoformat(), "fit_key": key,
            "plan_digest": digest if digest is not None else plan_sha(records),
            "decisions": sorted(r.seq for r in records if r.id in live)}


def write_fits(tmp_path, *records):
    path = tmp_path / "fits.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    return path


def test_a_fit_record_written_from_the_log_matches_its_refold(tmp_path):
    log = logged(tmp_path)
    records = log.records()
    fits = write_fits(tmp_path, fit_record(records, at=records[-1].at))
    log.append({"kind": "set_missing", "strategy": "complete_case"})  # after the fit: not its plan
    checks = walk.digest_checks(tmp_path / "decisions.jsonl", fits)
    assert [(c["checked"], c["matched"], c["prefix"]) for c in checks] == [(True, True, 3)]
    assert walk.digest_defects(checks) == []


def test_a_fit_record_whose_digest_is_not_its_logs_is_a_defect(tmp_path):
    log = logged(tmp_path)
    records = log.records()
    other = plan_sha([*records, log.append({"kind": "set_missing", "strategy": "complete_case"})])
    fits = write_fits(tmp_path, fit_record(records, at=records[-1].at + timedelta(seconds=1),
                                           digest=other))
    checks = walk.digest_checks(tmp_path / "decisions.jsonl", fits)
    assert [(c["checked"], c["matched"]) for c in checks] == [(True, False)]
    found = walk.digest_defects(checks)
    assert [d["defect"] for d in found] == ["digest"] and found[0]["record_digest"] == other
    assert found[0]["refolded"] == plan_sha(records)


def test_a_fit_record_whose_records_in_force_no_prefix_of_the_log_holds_is_a_defect(tmp_path):
    log = logged(tmp_path)
    records = log.records()
    rec = fit_record(records, at=records[-1].at + timedelta(seconds=1))
    fits = write_fits(tmp_path, {**rec, "decisions": [1, 2, 3, 4]})
    checks = walk.digest_checks(tmp_path / "decisions.jsonl", fits)
    assert [(c["checked"], c["matched"]) for c in checks] == [(True, False)]
    assert [d["defect"] for d in walk.digest_defects(checks)] == ["digest"]


def test_a_record_that_names_no_plan_is_not_compared(tmp_path):
    log = logged(tmp_path)
    records = log.records()
    rec = {**fit_record(records, at=records[-1].at), "plan_digest": None, "decisions": None}
    checks = walk.digest_checks(tmp_path / "decisions.jsonl", write_fits(tmp_path, rec))
    assert [(c["checked"], c["matched"]) for c in checks] == [(False, None)] and checks[0]["why"]
    assert walk.digest_defects(checks) == []
    assert walk.digest_checks(tmp_path / "decisions.jsonl", tmp_path / "none.jsonl") == []


# ── (5) a second seal, and a revisit after it ────────────────────────────────


class Sealing:
    """A stub session for the end of the path: ``seal`` and ``revisit`` answered as given, each kept;
    the real engine refuses a second opening, so only a stub can accept one."""

    def __init__(self, seal, revisit):
        self.said = {"seal": seal, "revisit": revisit}
        self.calls = []

    def seal(self, family, reason):
        self.calls.append(("seal", family))
        return self.said["seal"]

    def revisit(self, record, reason):
        self.calls.append(("revisit", record))
        return self.said["revisit"]


ACCEPTED = arms.Said("ok", {"recorded": {"id": "r9"}})


def refusal(code, message="Words."):
    return arms.Said("refused", {}, by="server", code=code, message=message)


def test_a_second_seal_and_a_revisit_after_it_accepted_are_defects():
    s = Sealing(ACCEPTED, ACCEPTED)
    after = walk.sealed_twice(s, "linear", ["r2", "r5"], np.random.default_rng(3), "walk:3")
    rng = np.random.default_rng(3)
    assert s.calls == [("seal", "linear"), ("revisit", ["r2", "r5"][int(rng.integers(2))])]
    assert [d["what"] for d in walk.sealed_defects(after)] == ["second_seal",
                                                               "revisit_after_seal"]
    assert {d["defect"] for d in walk.sealed_defects(after)} == {"accepted_after_seal"}


def test_a_second_seal_and_a_revisit_after_it_refused_are_no_defect():
    codes = walk.expected_codes()
    s = Sealing(refusal(codes["seal"]), refusal(codes["revisit"]))
    after = walk.sealed_twice(s, "linear", ["r2"], np.random.default_rng(3), "walk:3")
    assert walk.sealed_defects(after) == [] and walk.noted_of(after) == []
    other = Sealing(refusal("another_code"), refusal(codes["revisit"]))
    after = walk.sealed_twice(other, "linear", [], np.random.default_rng(3), "walk:3")
    assert walk.sealed_defects(after) == [] and other.calls == [("seal", "linear")]
    assert [(n["what"], n["code"], n["expected"]) for n in walk.noted_of(after)] == [
        ("second_seal", "another_code", codes["seal"])]


def test_the_codes_expected_after_the_seal_are_the_engines_and_the_drivers():
    """Read from the code that raises them: the engine's validator of an opening once the seal is
    open, and the driver's guard of every change after it."""
    from types import SimpleNamespace

    from research.swarm import driver
    from turbotab.core import seal
    from turbotab.core.decisions import Refusal

    with pytest.raises(Refusal) as again:
        seal._open_seal_once_on_a_fresh_fit(None, {"state": SimpleNamespace(seal_opened=True)})
    with pytest.raises(driver.DriverError) as after:
        driver.guard_sealed({"state": {"seal_opened": True}})
    assert walk.expected_codes() == {"seal": again.value.code, "revisit": after.value.code}


# ── the class ────────────────────────────────────────────────────────────────


def record(**extra):
    base = {"transcript": [], "stops": [], "stages": [], "checks": [], "after_seal": [],
            "reached": False}
    return {**base, **extra}


def test_a_walk_is_clean_refused_or_a_defect():
    assert walk.classed(record(reached=True))[0] == "clean"
    typed = entries((argv("answer", "impute", reason="r"), 1, refused()))
    assert walk.classed(record(reached=True, transcript=typed))[0] == "clean"
    stop = {"by": "engine", "kind": "refused", "words": "stopped",
            "detail": {"kind": "refused", "code": "too_few_rows", "message": "Words."}}
    assert walk.classed(record(stops=[stop]))[0] == "refused"
    failing = entries((argv("answer", "impute", reason="r"), 1, refused("http_500", "x",
                                                                        status=500)))
    kind, defects = walk.classed(record(reached=True, transcript=failing))
    assert kind == "defect" and [d["defect"] for d in defects] == ["failed"]


def test_a_stop_on_a_stage_the_engine_did_not_type_is_a_defect_once():
    stop = {"by": "engine", "kind": "failed", "words": "stopped", "detail": {
        "kind": "failed", "code": "stage_failed", "at": "stage", "stage": "shelf",
        "message": "RuntimeError: boom"}}
    stages = walk.stages_of(stage_view(shelf=("RuntimeError: boom", None)))
    kind, defects = walk.classed(record(stops=[stop], stages=stages))
    assert kind == "defect" and [(d["defect"], d["stage"]) for d in defects] == [
        ("stage_failed", "shelf")]
    kind, defects = walk.classed(record(stops=[stop]))
    assert [(d["defect"], d["stage"]) for d in defects] == [("stage_failed", "shelf")]


def test_a_defect_met_again_is_kept_once_as_first_met_and_another_beside_it():
    """A stage's error met in round 1 (after command 4), again in round 3 at another key (after
    command 9) and in the view as the walk ends is kept once, as first met: its round and the
    command it followed, which the report's words give. Another stage's error, met in round 2, and
    the same command's failure given twice (two requests) are each kept beside it."""
    words = "ValueError: no coefficient table"
    met = [{"defect": "stage_failed", "stage": "effects", "key": "e1", "words": words, "round": 1,
            "after": 4},
           {"defect": "stage_failed", "stage": "shelf", "key": "s1", "words": "RuntimeError: x",
            "round": 2, "after": 6},
           {"defect": "stage_failed", "stage": "effects", "key": "e3", "words": words, "round": 3,
            "after": 9}]
    final = [{"defect": "stage_failed", "stage": "effects", "key": "e3", "words": words}]
    failing = entries(*[(argv("answer", "impute", reason="r"), 1,
                         refused("http_500", "x", status=500))] * 2)
    kind, defects = walk.classed(record(stages_met=met, stages=final, transcript=failing))
    assert kind == "defect"
    assert [(d["defect"], d.get("stage"), d.get("round"), d.get("after"),
             (d.get("request") or {}).get("n")) for d in defects] == [
        ("failed", None, None, None, 0), ("failed", None, None, None, 1),
        ("stage_failed", "effects", 1, 4, None), ("stage_failed", "shelf", 2, 6, None)]
    report = walk.report_of("d", "inference", [1], [{**summary(1, kind=kind),
                                                     "defects": defects}], ENGINE, 1.0)
    text = walk.walk_md(report)
    assert "at the effects stage (met in round 1)" in text and "(met in round 3)" not in text
    assert "at the shelf stage (met in round 2)" in text


class Viewed(walk.Walked):
    """A run of the driver's own whose project is the view the test sets (``current``)."""

    current: dict = {}

    def view(self):
        return self.current


def canned(self, *argv):
    """The driver's answer to each command as the walker gives it: Fit made fresh, the results."""
    return 0, FRESH_FIT if argv[0] == "fit" else {"results": {}, "stages": {"fit": "fresh"}}


FRESH = {"stages": {"working": {"status": "fresh", "key": "w1"},
                    "fit": {"status": "fresh", "key": "f2"},
                    "effects": {"status": "fresh", "key": "e2"}}}


@pytest.mark.parametrize("middle, defect", [
    (stage_view(effects=("ValueError: no coefficient table", None)), True),
    (stage_view(cohort=("ValueError: The answers leave 3 of 60 rows.", REFUSAL)), False),
], ids=["untyped_in_round_1", "typed_in_round_1"])
def test_a_stage_in_error_met_mid_walk_is_held_though_the_final_view_is_fresh(
        tmp_path, monkeypatch, middle, defect):
    """The stages are held after each press of Fit and each read of the results, each with the
    round it was met in: an untyped error in round 1, gone from the final view once a later round
    refits, is still a defect; a typed refusal there is not."""
    monkeypatch.setattr(arms.Driven, "command", canned)
    session = Viewed(tmp_path)
    session.run = RID
    w = record(actions=[], skipped=[])
    walker = walk.Walker(session, arms.Random(arms.Question("y", "inference"), 1),
                         walk.walker_rng(1), w, None)
    session.at, session.current = 1, middle
    assert walker.fitted().ok
    session.at, session.current = "end", FRESH
    assert walker.fitted().ok
    assert walk.stages_of(session.view()) == []
    w.update(transcript=session.transcript, stages=walk.stages_of(session.view()),
             stages_met=session.met, reached=True)
    kind, defects = walk.classed(w)
    if defect:
        assert kind == "defect" and [(d["defect"], d["stage"], d["round"]) for d in defects] == [
            ("stage_failed", "effects", 1)]
        report = walk.report_of("d", "inference", [1], [{**summary(1, kind=kind),
                                                         "defects": defects}], ENGINE, 1.0)
        assert report["defects"][0]["round"] == 1 and report["passed"] is False
        assert "at the effects stage (met in round 1)" in walk.walk_md(report)
    else:
        assert kind == "clean" and defects == [] and session.met == []


# ── the actions offered, and the draws ───────────────────────────────────────


def served_log(tmp_path):
    """A log as the project view serves it, each record's JSON: the outcome and the goal (the
    card's own answers), the exclusions, the missing values, and a revert of the exclusions."""
    from turbotab.core import decisions as D

    log = D.DecisionLog(tmp_path / "decisions.jsonl")
    target = log.append({"kind": "set_target", "column": "y"})
    log.append({"kind": "set_purpose", "purpose": "prediction"})
    excl = log.append({"kind": "set_exclusions", "rules": []})
    missing = log.append({"kind": "set_missing", "strategy": "complete_case"})
    revert = log.append({"kind": "revert", "decision_id": excl.id})
    return log, {"decisions": [r.model_dump(mode="json") for r in log.records()]}, {
        "target": target.id, "excl": excl.id, "missing": missing.id, "revert": revert.id}


def test_the_actions_offered_are_the_drivers_at_that_point(tmp_path):
    _, view, ids = served_log(tmp_path)
    q = {"question": None, "fit_reached": True, "options": [], "also_open": [
        {"source": "finding", "key": "f1", "options": [way("x-d", {"kind": "dismiss_finding",
                                                                   "finding_id": "f1"})]},
        {"source": "declaration", "key": "set_selection", "options": [
            {"id": "set_selection:none", "decision": {"kind": "set_selection", "method": "none"}},
            {"id": "set_selection:custom", "decision": None, "needs": ["column"]}]}]}
    offers = walk.offered_actions(q, view, seen={"impute", "x-d", "none"})
    assert [c["id"] for c in offers["revisit"]] == [ids["missing"]]  # not the card's own answers
    assert [c["id"] for c in offers["undo"]] == [ids["revert"]]
    assert [c["id"] for c in offers["finding"]] == ["x-d"]
    assert [c["id"] for c in offers["declaration"]] == ["set_selection:none",
                                                         "set_selection:custom"]
    assert [c["id"] for c in offers["preview"]] == ["x-d", "set_selection:none",
                                                     "set_selection:custom"]
    assert offers["fit"] == [{}]
    assert [(c["kind"], c["id"]) for c in offers["stale"]] == [
        ("reverted", ids["excl"]), ("option", "impute"), ("option", "none")]


def test_with_nothing_reverted_or_seen_no_undo_and_no_stale_id_is_offered(tmp_path):
    from turbotab.core import decisions as D

    log = D.DecisionLog(tmp_path / "decisions.jsonl")
    log.append({"kind": "set_missing", "strategy": "complete_case"})
    view = {"decisions": [r.model_dump(mode="json") for r in log.records()]}
    q = {"question": {"key": "models", "status": "open"}, "options": [
        {"id": "linear", "decision": {"kind": "select_models", "models": ["linear"]}}]}
    offers = walk.offered_actions(q, view, seen={"linear"})
    assert offers["undo"] == [] and offers["stale"] == [] and offers["finding"] == []
    assert [c["id"] for c in offers["preview"]] == ["linear"]
    assert [n for n in walk.ACTIONS if offers[n]] == ["revisit", "preview", "fit"]


def test_the_rounds_and_each_draw_are_numpys_at_the_walkers_seed():
    for seed in range(12):
        rng = np.random.default_rng([walk.STREAM, seed])
        assert walk.rounds_of(walk.walker_rng(seed)) == int(rng.integers(2, 5))
    got = {walk.rounds_of(walk.walker_rng(s)) for s in range(40)}
    assert got == {2, 3, 4}
    rng = walk.walker_rng(5)
    walk.rounds_of(rng)
    names = ["revisit", "preview", "fit"]
    drawn = [walk.draw(rng, names) for _ in range(30)]
    mine = np.random.default_rng([walk.STREAM, 5])
    mine.integers(2, 5)
    assert drawn == [names[int(mine.integers(3))] for _ in range(30)]


def test_a_stale_ids_kind_is_drawn_first_so_a_reverted_decision_is_sent():
    """Two decisions reverted beside 54 options an earlier ask offered (the smoke run's counts):
    the kind is drawn uniformly among those offered, then an id of that kind, each from numpy's
    generator at the walker's seed, drawn here; so both kinds are sent, the reverted about half
    the time and not about 2 times in 56."""
    stale = [{"kind": "reverted", "id": f"r{i}"} for i in range(2)] + [
        {"kind": "option", "id": f"o{i}"} for i in range(54)]
    rng = walk.walker_rng(7)
    drawn = [walk.stale_of(rng, stale)["id"] for _ in range(40)]
    mine = np.random.default_rng([walk.STREAM, 7])
    expected = []
    for _ in range(40):
        kind = ["reverted", "option"][int(mine.integers(2))]
        ids = [c["id"] for c in stale if c["kind"] == kind]
        expected.append(ids[int(mine.integers(len(ids)))])
    assert drawn == expected
    kinds = [i[0] for i in drawn]
    assert 0 < kinds.count("r") < 40 and kinds.count("r") > 40 * 2 / 56 * 3
    only = [c for c in stale if c["kind"] == "option"]
    assert {walk.stale_of(walk.walker_rng(7), only)["kind"] for _ in range(5)} == {"option"}


def test_an_option_that_needs_values_takes_them_from_fields_or_is_skipped_and_counted():
    from research.swarm import sweep

    class Built:
        def answers(self, point, option):
            if option["id"] == "custom":
                return [sweep.Fielded("custom@built:x", ["custom"], ["column=x", "low=1.0",
                                                                     "high=2.0"], "built",
                                      "built")], None
            return [], "It records nothing by itself: it takes a column."

    skipped = []
    plain = {"id": "none", "decision": {"kind": "set_exclusions", "rules": []}}
    assert walk.answer_of(plain, None, Built(), skipped) == (["none"], [])
    custom = {"id": "custom", "decision": None, "needs": ["column", "low", "high"]}
    point = sweep.Point("question", "exclusions")
    assert walk.answer_of(custom, point, Built(), skipped) == (
        ["custom"], ["column=x", "low=1.0", "high=2.0"])
    levels = {"id": "levels", "decision": None, "needs": ["level_column"]}
    assert walk.answer_of(levels, point, Built(), skipped) is None
    assert skipped == [{"option": "levels", "point": "question:exclusions",
                        "why": "It records nothing by itself: it takes a column."}]


# ── the report and the run ───────────────────────────────────────────────────

ENGINE = {"commit": "c1", "engine_tree": "t1", "engine_modified": False}


def summary(seed, kind="clean", start=ENGINE, end=ENGINE):
    return {"seed": seed, "dataset": "d", "purpose": "inference", "class": kind, "defects": [],
            "engine": {"start": {**start, "at": "a"}, "end": {**end, "at": "b"}},
            "rounds": 2, "actions": [], "stops": [], "skipped": [], "checks": [],
            "after_seal": [], "reached": kind == "clean", "seconds": 1.0, "run": "r"}


def test_a_report_whose_engines_differ_is_refused():
    one = walk.report_of("d", "inference", [1, 2], [summary(1), summary(2)], ENGINE, 2.0)
    assert one["engine_changed"] is None and one["passed"] is True
    assert one["classes"] == {"clean": 2, "refused": 0, "defect": 0}
    moved = {**ENGINE, "engine_tree": "t2", "commit": "c2"}
    two = walk.report_of("d", "inference", [1, 2], [summary(1), summary(2, end=moved)], ENGINE,
                         2.0)
    assert two["engine_changed"] and "t2" in two["engine_changed"] and two["passed"] is False
    assert "engine changed" in walk.walk_md(two).lower()
    same_tree = {**ENGINE, "commit": "c9"}  # a commit that leaves the tree as it was
    three = walk.report_of("d", "inference", [1], [summary(1, start=same_tree)], ENGINE, 1.0)
    assert three["engine_changed"] is None


def test_a_report_with_a_seed_unreported_or_a_defect_does_not_pass():
    part = walk.report_of("d", "inference", [1, 2], [summary(1)], ENGINE, 1.0)
    assert part["unreported"] == [2] and part["complete"] is False and part["passed"] is False
    bad = walk.report_of("d", "inference", [1], [summary(1, kind="defect")], ENGINE, 1.0)
    assert bad["classes"]["defect"] == 1 and bad["passed"] is False


def test_a_run_skips_each_seed_already_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(walk, "stamp", lambda: {**ENGINE, "at": "now"})
    calls = []

    def fake(dataset, seed, out, **kw):
        calls.append(seed)
        return summary(seed)

    walk.run("d", [1, 2], tmp_path, walker=fake, purpose="inference")
    assert calls == [1, 2]
    walk.run("d", [1, 2, 3], tmp_path, walker=fake, purpose="inference")
    assert calls == [1, 2, 3]
    report = json.loads((tmp_path / "walk.json").read_text("utf-8"))
    assert [w["seed"] for w in report["walks"]] == [1, 2, 3] and report["passed"] is True
    assert (tmp_path / "walk.md").read_text("utf-8").startswith("# The walks of d")


def test_an_error_of_the_walkers_own_is_kept_and_the_report_does_not_pass(tmp_path, monkeypatch):
    """A fault in the walker's own code (here at the run's start) ends that walk, kept as the
    walker's error in its words, not the engine's; a report holding one does not pass, and one
    without passes (above)."""
    def broken(self, *a, **k):
        raise KeyError("a fault of the walker's")

    monkeypatch.setattr(walk.Walked, "start", broken)
    table = tmp_path / "t.csv"
    table.write_text("y,x\n1,2\n0,3\n", "utf-8")
    w = walk.walk_one("d", 4, tmp_path, data=table, question=arms.Question("y", "prediction"))
    assert [(s["by"], s["kind"]) for s in w["stops"]] == [("walker", "error")]
    assert "KeyError" in w["stops"][0]["words"] and w["run"] is None and w["defects"] == []
    w = {**summary(4, kind=w["class"]), "stops": w["stops"]}
    report = walk.report_of("d", "prediction", [4], [w], ENGINE, 1.0)
    assert report["walker_errors"] == [4] and report["passed"] is False


def test_the_seeds_read_as_a_range_or_one():
    assert walk.seeds_of("1-3") == [1, 2, 3] and walk.seeds_of("7") == [7]
    with pytest.raises(ValueError):
        walk.seeds_of("3-1")


# ── whole walks on a server modeled here ─────────────────────────────────────


class Served(walk.Walked):
    """A run of the driver's own whose server is modeled here (``serve``, in place of
    ``driver.main``; the fixture ``served``), each command answered as the driver prints it. The
    log is the engine's own (``decisions.DecisionLog``): the card's outcome and goal, then
    ``revisitable`` decisions a revisit may take. Every ask finds Fit reached and nothing open; the
    first offers ``options`` option ids that no later ask offers, so each is a stale id of the
    option kind. A revisit records a revert and the decision answered again (as the walk answers
    its question again), or, where the decision is already reverted, is refused as the engine
    refuses it (``decisions._check_revert``: its code, words and way forward); an answer of a way
    forward kept from the last refusal records it, and any other id is refused by the driver
    (``not_offered``); Fit is made fresh, the results read and the export given."""

    revisitable, options = 3, 0

    def serve(self, *argv):
        from turbotab.core import decisions as D

        command = argv[0]
        ids = list(itertools.takewhile(lambda a: not a.startswith("--"), argv[3:]))
        if command == "start":
            self.runs.mkdir(parents=True, exist_ok=True)
            self.log, self.kept, self.asks = D.DecisionLog(self.runs / "decisions.jsonl"), [], 0
            for d in ({"kind": "set_target", "column": "y"},
                      {"kind": "set_purpose", "purpose": "inference"}):
                self.log.append(d)
            for _ in range(self.revisitable):
                self.log.append({"kind": "set_missing", "strategy": "complete_case"})
            return 0, {"started": {"run": RID}}
        if command == "ask":
            self.asks += 1
            options = [{"id": f"o{i}", "label": f"Option {i}", "decision": {
                "kind": "set_missing", "strategy": "complete_case"}}
                for i in range(self.options if self.asks == 1 else 0)]
            return 0, {"question": None, "fit_reached": True, "options": options,
                       "also_open": []}
        if command == "revisit":
            return self.revisited(ids[0])
        if command == "answer":
            way = next((w for w in self.kept if w["id"] == ids[0]), None)
            if way is None:
                return 1, refused("not_offered", f"No option {ids[0]} is offered now.",
                                  by="driver", status=None)
            r = self.log.append(way["decision"])
            self.kept = []
            return 0, ok(r.decision.kind, r.id, r.seq)
        canned = {"fit": FRESH_FIT, "results": {"results": {}}, "export": EXPORTED,
                  "stop": {"stopped": {"run": RID}}}
        assert command in canned, f"the walk gave {command}, which this server does not model"
        return 0, canned[command]

    def revisited(self, record):
        from turbotab.core import decisions as D

        records = self.log.records()
        target = next(r for r in records if r.id == record)
        try:
            D._check_revert(records, D.Revert(decision_id=record))
        except D.Refusal as exc:
            self.kept = [{"id": f"x-{i}", **e} for i, e in enumerate(exc.exits)]
            return 1, refused(exc.code, exc.message, self.kept)
        r = self.log.append({"kind": "revert", "decision_id": record})
        if target.decision.kind != "revert":
            self.log.append(target.decision.model_dump(mode="json"))
        self.kept = []
        return 0, ok("revert", r.id, r.seq)

    def view(self):
        return {"decisions": [r.model_dump(mode="json") for r in self.log.records()],
                "stages": {"fit": {"status": "fresh", "key": "f1"}}}

    def project_folder(self):
        return self.runs


QUESTION = arms.Question("y", "inference")


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The walker's run is :class:`Served`, the engine stamped as :data:`ENGINE`; the table."""
    monkeypatch.setattr(arms.Driven, "command", lambda self, *argv: self.serve(*argv))
    monkeypatch.setattr(walk, "Walked", Served)
    monkeypatch.setattr(walk, "stamp", lambda: {**ENGINE, "at": "now"})
    table = tmp_path / "t.csv"
    table.write_text("y,x\n1,2\n0,3\n", "utf-8")
    return table


def faulty(where, monkeypatch, out):
    """A fault of the walker's own planted after the walk, at ``where``; its exception's name."""
    def fault(*a, **k):
        raise RuntimeError(f"a fault of the walker's, at {where}")

    if where == "checks":
        monkeypatch.setattr(walk, "digest_checks", fault)
    elif where == "class":
        monkeypatch.setattr(walk, "classed", fault)
    elif where == "stop":
        monkeypatch.setattr(Served, "stop", fault)
    elif where == "defects":  # a defect found, and its folder cannot be made: a file holds its name
        monkeypatch.setattr(walk, "classed", lambda w: ("defect", [{"defect": "failed",
                                                                     "words": "Words."}]))
        out.mkdir(parents=True)
        (out / "defects").write_text("a file, not a folder", "utf-8")
        return "FileExistsError"
    return "RuntimeError"


@pytest.mark.parametrize("where", ["checks", "class", "defects", "stop", None])
def test_a_fault_in_a_walks_tail_ends_only_that_walk_kept_as_the_walkers(served, tmp_path,
                                                                          monkeypatch, where):
    """I3: a fault of the walker's own after the walk (its run's stop, the fit records held to the
    log, the class, the defects' files) ends only that walk: it is kept as the walker's own fault,
    in its words, and written; the run goes on to the next seed, the report holding it does not
    pass, and a run again walks neither seed again. With no fault planted, none is kept and the
    report passes."""
    out = tmp_path / "out"
    name = faulty(where, monkeypatch, out) if where else None
    calls = []

    def walker(*a, **k):
        calls.append(a[1])
        return walk.walk_one(*a, **k)

    report = walk.run("d", [5, 6], out, walker=walker, purpose="inference", data=served,
                      question=QUESTION)
    assert calls == [5, 6] and report["complete"] is True
    for seed in (5, 6):
        w = json.loads((out / "walks" / f"d-{seed}.json").read_text("utf-8"))
        assert w["class"] in walk.CLASSES and w["reached"] is True
        faults = [(s["kind"], s["words"].split(":")[0], (s.get("detail") or {}).get("at"))
                  for s in w["stops"] if s["by"] == "walker"]
        assert faults == ([("error", name, where)] if where else [])
        if where == "defects":
            assert w["class"] == "defect" and [d.get("path") for d in w["defects"]] == [None]
        if where == "class":
            assert (w["class"], w["defects"]) == ("refused", [])
    assert report["walker_errors"] == ([5, 6] if where else [])
    assert report["passed"] is (where is None)
    walk.run("d", [5, 6], out, walker=walker, purpose="inference", data=served,
             question=QUESTION)
    assert calls == [5, 6]


def started(served, tmp_path):
    session = Served(tmp_path / "runs")
    session.start(served, budget_fits=1, budget_seconds=1.0, model="m")
    w = record(actions=[], skipped=[], end=[])
    return session, w, walk.Walker(session, arms.Random(QUESTION, 1), walk.walker_rng(1), w, None)


def test_a_round_more_sends_a_decision_already_reverted_where_no_round_sent_one(served,
                                                                                tmp_path):
    """I1, by construction: one decision of the log reverted (a revisit recorded here) and no
    round having sent one, the round after the drawn rounds sends that decision, through the
    driver's revisit, to the server, which refuses it as the engine does, with a way forward that
    is taken and recorded. Once one is sent, no round more is given; with nothing reverted, none,
    and no command is sent."""
    session, w, walker = started(served, tmp_path)
    target = session.log.records()[2].id  # the first decision a revisit may take
    assert session.revisit(target, "r").ok
    before = len(session.transcript)
    walker.owed(4)
    assert [(a["round"], a["action"], a["stale"], a["owed"], a["id"]) for a in w["actions"]] == [
        (4, "stale", "reverted", True, target)]
    sent = session.transcript[before]
    assert (sent["command"], sent["ids"], sent["kind"], sent["by"]) == (
        "revisit", [target], "refused", "server")
    assert w["actions"][0]["accepted"] is False and w["actions"][0]["way"]["kind"] == "ok"
    n = len(session.transcript)
    walker.owed(5)
    assert len(session.transcript) == n and len(w["actions"]) == 1

    session, w, walker = started(served, tmp_path / "none")
    n = len(session.transcript)
    walker.owed(3)
    assert len(session.transcript) == n and w["actions"] == []


def test_each_walk_whose_rounds_revert_a_decision_sends_one_already_reverted(served, tmp_path,
                                                                             monkeypatch):
    """I1 over seeds 1-40 on the server modeled here (three decisions a revisit may take, two option
    ids an earlier ask offered): a walk sends a decision already reverted exactly where its drawn
    rounds recorded a revert (a revisit or an undo, each recorded here); where none of them sent
    one, the round after them does, through the driver's revisit, refused by the server; a walk
    whose rounds reverted nothing, or sent one, is given no round more. Each case is met at these
    seeds."""
    monkeypatch.setattr(Served, "options", 2)
    cases = {"owed": 0, "sent": 0, "none": 0}
    for seed in range(1, 41):
        w = walk.walk_one("d", seed, tmp_path / str(seed), data=served, question=QUESTION)
        drawn = [a for a in w["actions"] if a["round"] <= w["rounds"]]
        more = [a for a in w["actions"] if a["round"] > w["rounds"]]
        reverted = any(a["action"] in ("revisit", "undo") and a["said"]["kind"] == "ok"
                       for a in drawn)
        sent = any(a.get("stale") == "reverted" for a in drawn)
        assert any(a.get("stale") == "reverted" for a in w["actions"]) is reverted, seed
        if reverted and not sent:
            assert [(a["round"], a["action"], a["stale"], a["owed"]) for a in more] == [
                (w["rounds"] + 1, "stale", "reverted", True)], seed
            assert (more[0]["said"]["kind"], more[0]["said"]["by"]) == ("refused", "server")
            assert any(e["command"] == "revisit" and e["ids"] == [more[0]["id"]]
                       and e["by"] == "server" for e in w["transcript"])
        else:
            assert more == [], seed
        cases["owed" if reverted and not sent else "sent" if sent else "none"] += 1
        assert w["class"] == "clean", seed
    assert all(cases.values()), cases


# ── a fit that outlasts one press's wait ─────────────────────────────────────


def walked(tmp_path, monkeypatch, server, budget, then=None):
    """A walker's run of the driver's own (:class:`walk.Walked`) whose driver is modeled as the
    arms' tests model it (``test_arms.Modeled``: its Fit the driver's own on ``server``, on the
    server's hand clock; ``then``, a call after a command), the project read as ``server`` serves
    it, started with ``budget`` seconds; the session and the model."""
    from research.swarm.tests.test_arms import Modeled, wired

    wired(tmp_path, monkeypatch, server)
    model = Modeled(server.clock, then=then)
    monkeypatch.setattr(arms.Driven, "command", model)
    monkeypatch.setattr(arms.Driven, "read", lambda self, p: server.view() if p is None else None)
    session = walk.Walked(tmp_path / "runs")
    session.start(tmp_path / "t.csv", budget_fits=walk.BUDGET_FITS, budget_seconds=budget,
                  model="mechanical-walk-1")
    return session, model


def ending(tmp_path, monkeypatch, server, budget):
    """A walker at the end of the path (``Walker.settle``) on :func:`walked`'s session, every ask
    finding Fit reached with nothing open, the fit not yet pressed."""
    session, model = walked(tmp_path, monkeypatch, server, budget)
    w = record(actions=[], skipped=[], end=[])
    return walk.Walker(session, arms.Random(QUESTION, 1), walk.walker_rng(1), w, None), model


def test_the_end_of_the_path_waits_out_a_fit_still_running_when_its_press_comes_back(
        tmp_path, monkeypatch):
    """W1 in the walker: the fit not fresh at the end of the path, Fit is pressed, and its press
    comes back with the fit (of 1000 s) still running; the walk waits it out with reads alone and
    goes on to the results and the ask, where it stopped refused before ("Fit was not made for the
    end of the path"). The press's entry in the transcript is what it came to, fresh, with what it
    printed when its wait ran out beside it."""
    from research.swarm.tests.test_arms import Clock, Server

    server = Server(Clock(), purpose=QUESTION.purpose, seconds=1000.0)
    walker, model = ending(tmp_path, monkeypatch, server, walk.BUDGET_SECONDS)
    q = walker.settle()
    assert q["fit_reached"] is True
    assert [a[0] for a in model.argv][-3:] == ["fit", "results", "ask"]
    assert server.posts == [0.0] and server.clock.now >= 1000.0
    fits = [e for e in walker.session.transcript if e["command"] == "fit"]
    assert [(e["kind"], e["fit"]["status"]) for e in fits] == [("ok", "fresh")]
    assert fits[0]["ran_out"] == {
        "kind": "refused", "by": "driver", "code": "fit_not_reached",
        "message": "The fit is running.",
        "fit": {"status": "running", "key": "k1", "progress": round(arms.WAIT / 1000.0, 3)}}


def test_a_fit_that_outlasts_the_runs_time_budget_stops_the_walk_at_its_own_limit(
        tmp_path, monkeypatch):
    """The other way: the fit never done, the run given 500 s, Fit pressed at its start and its
    fit waited for to the budget; the walk stops there in the arm's words, a limit of the walk's
    own (a stuck stop), not a refusal, and the press's entry stays as it printed it."""
    from research.swarm.tests.test_arms import Clock, Server

    budget = 500.0
    server = Server(Clock(), purpose=QUESTION.purpose, end="never")
    walker, model = ending(tmp_path, monkeypatch, server, budget)
    with pytest.raises(arms.Stuck) as got:
        walker.settle()
    assert not isinstance(got.value, arms.Met)
    assert model.waits() == [format(arms.WAIT, "g")] and server.posts == [0.0]
    assert walk.stop_of(got.value) == {
        "by": "walk", "kind": "stuck", "detail": None,
        "words": f"The fit outlasted the run's time budget of {budget:g} s from its start: it was "
                 f"still running {budget:g} s after Fit was pressed."}
    fits = [e for e in walker.session.transcript if e["command"] == "fit"]
    assert [(e["code"], "ran_out" in e) for e in fits] == [("fit_not_reached", False)]


@pytest.mark.parametrize("then, dead_end", [("refused_with_a_way", True), ("fresh", False)],
                         ids=["refused_again", "given"])
def test_fit_waited_out_is_read_by_what_it_came_to_so_refused_again_after_its_way_is_a_dead_end(
        tmp_path, monkeypatch, then, dead_end):
    """Both ways: Fit refused by the engine in words with a way forward, the way taken (answers
    that change the fit's key, to a fit of 1000 s), and Fit pressed again, whose press comes back
    with that fit still running and is waited out. Refused again with the same code there, it is
    a dead end, the request refused again the second press; given, it is none."""
    from research.swarm.tests.test_arms import WORDS, Clock, Server

    server = Server(Clock(), purpose=QUESTION.purpose, seconds=0.0, end="refused_with_a_way")
    session, _ = walked(tmp_path, monkeypatch, server, walk.BUDGET_SECONDS,
                        then={"answer": lambda: server.refit(1000.0, then)})
    first = session.fit()
    assert (first.by, first.code, first.message) == ("engine", "too_few_rows", WORDS)
    assert session.answer([first.exits[0]["id"]], [], "walk: the fit's way forward.").ok
    again = session.fit()
    assert again.ok is not dead_end and server.posts == [0.0, 0.0]
    fits = [e for e in session.transcript if e["command"] == "fit"]
    found = walk.dead_ends_of(session.transcript)
    assert [(d["code"], d["request"]["n"]) for d in found] == (
        [("too_few_rows", fits[-1]["n"])] if dead_end else [])
    assert len(fits) == 2 and fits[1]["ran_out"]["code"] == "fit_not_reached"
    assert walk.wordless_of(session.transcript) == []


# ── one walk through the driver, to its end ──────────────────────────────────


@pytest.mark.slow
def test_a_walk_runs_through_the_driver_to_its_end_and_the_second_seal_is_refused(tmp_path):
    report = walk.run("clinical_readmission_predict", [3], tmp_path)
    w = json.loads((tmp_path / "walks" / "clinical_readmission_predict-3.json").read_text("utf-8"))
    assert w["class"] in ("clean", "refused", "defect") and report["walks"][0]["class"] == w["class"]
    assert 2 <= w["rounds"] <= 4
    drawn = [a for a in w["actions"] if a["round"] <= w["rounds"]]
    assert len(drawn) <= w["rounds"] and all(a.get("owed") for a in w["actions"][len(drawn):])
    if any(a["action"] in ("revisit", "undo") and (a.get("said") or {}).get("kind") == "ok"
           for a in drawn):
        # A decision already reverted is sent, and reaches the server past the driver.
        sent = [a for a in w["actions"] if a.get("stale") == "reverted"]
        assert sent and (sent[0]["said"]["kind"], sent[0]["said"]["by"]) == ("refused", "server")
    assert w["engine"]["start"]["engine_tree"] == w["engine"]["end"]["engine_tree"]
    commands = [e["command"] for e in w["transcript"]]
    assert commands[0] == "start" and commands.count("fit") >= 1
    assert w["family"] and w["after_seal"], "seed 3 reaches the seal on this dataset"
    codes = walk.expected_codes()
    said = {a["what"]: a for a in w["after_seal"]}
    assert (said["second_seal"]["kind"], said["second_seal"]["code"]) == ("refused", codes["seal"])
    assert (said["revisit_after_seal"]["kind"], said["revisit_after_seal"]["code"]) == (
        "refused", codes["revisit"])
    assert not walk.sealed_defects(w["after_seal"])
    assert w["checks"]  # every fit record held to the log's refold
