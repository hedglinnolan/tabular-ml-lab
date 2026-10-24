"""The walker (``research/swarm/walk.py``; the owner's ruling of 2026-10-23, "Go with D"): its oracle
on transcripts, logs and views built here, both ways for each of its five defects; the actions it
offers and draws at a seed; the end of the path's second seal and revisit after it, accepted by a
stub (the real engine cannot be made to accept them) and refused; the report refused where its
engines differ; a run that skips a seed already reported; and, slow, one walk through the driver's
own commands on a small dataset that is not one of the pilot's six
(``clinical_readmission_predict``, 480 rows), whose second seal and revisit after it the engine and
the driver refuse.

Known truth: each command's payload as the driver prints it (``driver.main``'s JSON: ``refused``
with its ``by``, ``status``, ``code``, ``message`` and ``exits``; ``recorded``; a press of Fit's
``pressed`` and ``fit``), built here; a stage's status as the engine serves it (``status``,
``error``, ``refusal``); a decision log appended here with the engine's own log
(``decisions.DecisionLog``) and a fit record written from it, its digest the SHA-256 of the plan's
canonical JSON computed here with hashlib; each draw from numpy's generator at the walker's seed,
drawn here."""
from __future__ import annotations

import hashlib
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


def test_the_seeds_read_as_a_range_or_one():
    assert walk.seeds_of("1-3") == [1, 2, 3] and walk.seeds_of("7") == [7]
    with pytest.raises(ValueError):
        walk.seeds_of("3-1")


# ── one walk through the driver, to its end ──────────────────────────────────


@pytest.mark.slow
def test_a_walk_runs_through_the_driver_to_its_end_and_the_second_seal_is_refused(tmp_path):
    report = walk.run("clinical_readmission_predict", [3], tmp_path)
    w = json.loads((tmp_path / "walks" / "clinical_readmission_predict-3.json").read_text("utf-8"))
    assert w["class"] in ("clean", "refused", "defect") and report["walks"][0]["class"] == w["class"]
    assert 2 <= w["rounds"] <= 4 and len(w["actions"]) <= w["rounds"]
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
