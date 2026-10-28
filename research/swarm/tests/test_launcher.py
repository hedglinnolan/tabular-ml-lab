"""The pilot's launcher (PILOT_READINESS §5; ``research/swarm/launcher.py``): the known answers read
from each dataset's own answers file and checked against the generator that planted them, the
declared measurement markers as the intake wrote them, the crash sweep's report gating a launch
both ways, the fit budget from its measured fits by hand, the pre-registration gating a launch
both ways (``prereg_fill.ready`` asked before each, here a stand-in, and package 6b's own where it
is on the tree) with the fit budget prereg.json holds, and the start command line the launcher
hands the driver: no label of its own, the run named by the driver, the known answer written where
no analyst's command reads it, in the launches folder every checkout shares (the main checkout's,
the first ``git worktree list`` names), with no command-line option to write it elsewhere."""
from __future__ import annotations

import json
import math

import pytest

from research.swarm import driver, launcher as L

ENGINE = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}
# The table the dataset's manifest names, read from the manifest itself.
TABLE = json.loads((L.DATASETS / "outpatient_admission_predict" / "manifest.json")
                   .read_text())["table"]["sha256"]


def report(**changes):
    out = {"dataset": "outpatient_admission_predict", "complete": True, "unvisited": [],
           "defects": [], "passed": True, "engine_tree": "t0", "engine_modified": False,
           "commit": "c0", "fit_seconds": {"fits": 4, "total": 20.0, "median": 4.0, "p90": 7.2,
                                           "max": 7.2},
           "only": None, "data": {"path": "outpatient_visits.csv", "sha256": TABLE}}
    out.update(changes)
    return out


# ── the known answers ────────────────────────────────────────────────────────


def test_the_estimate_fixtures_known_answer_is_its_planted_effect_on_the_engines_measure():
    from research.swarm.generators import ESTIMATE_MODEL
    from turbotab.core.estimand import MEASURE_OF_TASK

    known, source = L.known_answer("walking_bp_estimate")
    answers = json.loads((L.DATASETS / "walking_bp_estimate" / "answers.json").read_text())
    stated = next(t for t in answers["truths"]
                  if t["name"] == "the effect of walks_daily on sbp_mmhg, a difference in means")
    assert known == {"feature": "walks_daily", "measure": MEASURE_OF_TASK["regression"],
                     "value": stated["value"]}
    assert known["value"] == ESTIMATE_MODEL["effect"]  # the generator's own constant
    assert source.startswith("answers.json")


@pytest.mark.parametrize("dataset", ["outpatient_admission_predict", "survey_support_estimate"])
def test_a_null_decoy_and_a_null_effect_are_written_null(dataset):
    """The decoy's outcome reads no column (its answers file says so); the survey fixture states
    the effect of age as none."""
    known, _ = L.known_answer(dataset)
    assert known == {"null": True}


@pytest.mark.parametrize("dataset", ["clinical_readmission_predict", "bang2009", "sullivan2021",
                                     "nhanes_glucose_predict", "plasma_panel_estimate"])
def test_a_dataset_with_no_answer_of_that_form_has_none_and_says_why(dataset):
    known, why = L.known_answer(dataset)
    assert known is None and why


def test_the_checks_read_the_known_answer_as_the_launcher_writes_it():
    """The record's known-answer check compares a result on the known answer's measure with its
    value, and a null one with the scale's no-effect value (``checks._known_value``)."""
    from types import SimpleNamespace

    from research.swarm import checks

    walking, _ = L.known_answer("walking_bp_estimate")
    result = SimpleNamespace(measure="mean_difference", scale="difference")
    assert checks._known_value(result, walking) == (walking["value"], None)
    null, _ = L.known_answer("outpatient_admission_predict")
    assert checks._known_value(result, null)[0] == 0.0
    other = SimpleNamespace(measure="odds_ratio", scale="ratio")
    value, why = checks._known_value(other, walking)
    assert value is None and why


def test_the_records_dataset_holds_the_control_kind_from_the_answers_file():
    meta = L.dataset_meta("outpatient_admission_predict")
    assert meta["control"] == "null_decoy" and meta["known"] == {"null": True}
    assert L.dataset_meta("walking_bp_estimate")["control"] == "known_answer"
    assert L.dataset_meta("bang2009")["control"] is None


# ── the crash sweep gates the launch ─────────────────────────────────────────


def test_a_crash_sweep_that_passed_on_this_engine_lets_the_dataset_launch():
    assert L.crash_passed(report(), "outpatient_admission_predict", ENGINE) is None


@pytest.mark.parametrize("changes, engine, words", [
    ({"dataset": "bang2009"}, ENGINE, "not of"),
    ({"complete": False, "unvisited": ["question:missing=impute"]}, ENGINE, "did not visit"),
    ({"defects": [{"refit": "x"}], "passed": False}, ENGINE, "engine defect"),
    ({"passed": False}, ENGINE, "did not pass"),
    ({"engine_modified": True}, ENGINE, "when the crash sweep ran"),
    ({}, {**ENGINE, "engine_modified": True}, "now"),
    ({"engine_tree": "t1"}, ENGINE, "runs again on the frozen engine"),
    ({"only": ["question:missing", "question:split"]}, ENGINE, "swept only 2"),
    ({"only": []}, ENGINE, "swept only 0"),
    ({"data": {"path": "tiny.csv", "sha256": "0" * 64}}, ENGINE, "not the manifest's"),
    ({"data": None}, ENGINE, "not the manifest's"),
], ids=["other_dataset", "incomplete", "defect", "not_passed", "modified_then", "modified_now",
        "other_engine", "some_decisions", "no_decision", "other_table", "no_table"])
def test_a_crash_sweep_that_has_not_passed_on_this_engine_refuses_the_launch(changes, engine,
                                                                               words):
    why = L.crash_passed(report(**changes), "outpatient_admission_predict", engine)
    assert why is not None and words in why


def test_a_report_that_does_not_say_what_it_swept_refuses_the_launch():
    said = {k: v for k, v in report().items() if k != "only"}
    why = L.crash_passed(said, "outpatient_admission_predict", ENGINE)
    assert why is not None and "does not say whether it swept every decision" in why


def test_a_dataset_whose_manifest_names_no_table_is_not_launched(monkeypatch, tmp_path):
    (tmp_path / "outpatient_admission_predict").mkdir()
    monkeypatch.setattr(L, "DATASETS", tmp_path)
    why = L.crash_passed(report(), "outpatient_admission_predict", ENGINE)
    assert why is not None and "names no table" in why


def test_the_budget_is_its_fits_at_the_slowest_fit_measured_rounded_up():
    budget = L.budget_of(report(), 12)
    assert budget == {"fits": 12, "seconds": float(math.ceil(12 * 7.2)), "slowest_fit": 7.2}
    assert budget["seconds"] == 87.0  # 86.4 rounded up
    with pytest.raises(L.Refused, match="no fit's run time"):
        L.budget_of(report(fit_seconds={"fits": 0, "max": None}), 12)
    with pytest.raises(L.Refused, match="at least one fit"):
        L.budget_of(report(), 0)


# ── the declared measurement markers ─────────────────────────────────────────


@pytest.mark.parametrize("dataset", ["bang2009", "walking_bp_estimate"])
def test_the_records_dataset_holds_the_declared_markers_the_intake_wrote(dataset):
    """Both ways: a dataset the intake declared markers for (``declared.json``) carries them, as
    written; one it wrote none for carries none."""
    from research.swarm.arms import record_meta as arm_meta

    path = L.DATASETS / dataset / "declared.json"
    written = json.loads(path.read_text()) if path.exists() else []
    assert (len(written) > 0) is (dataset == "bang2009")  # the construction
    meta = L.dataset_meta(dataset)
    assert meta["declared"] == written
    assert L.record_meta({"dataset": meta, "analyst_model": "m", "prompt_version": "p"})[
        "dataset"]["declared"] == written
    assert arm_meta(dataset, "mechanical-default")["dataset"]["declared"] == written


# ── the start the launcher hands the driver ──────────────────────────────────

PILOT = "walking_bp_estimate"
PILOT_TABLE = json.loads((L.DATASETS / PILOT / "manifest.json").read_text())["table"]["sha256"]
BUDGET = {"fits": 10, "seconds": 72.0, "slowest_fit": 7.2}


class Prereg:
    """prereg_fill as the launcher calls it, here: ``load`` gives ``prereg``, ``ready`` says
    ``why`` (None: the dataset may launch); each call kept, with the dataset it was asked of."""

    def __init__(self, prereg, why=None):
        self.prereg, self.why, self.calls = prereg, why, []

    def load(self, path=None):
        self.calls.append(("load", path))
        return self.prereg

    def ready(self, prereg, engine=None, *, dataset=None, margins=True):
        self.calls.append(("ready", prereg, dataset))
        return self.why


@pytest.fixture
def prereg(monkeypatch):
    """Installs a :class:`Prereg` as ``research.swarm.prereg_fill`` (the module and the package's
    attribute, where the launcher imports it from); returns a maker."""
    import sys

    import research.swarm as package

    def install(filled=None, why=None):
        fake = Prereg({"filled": {"fit_budget": filled if filled is not None
                                  else {PILOT: dict(BUDGET)}}}, why)
        monkeypatch.setitem(sys.modules, "research.swarm.prereg_fill", fake)
        monkeypatch.setattr(package, "prereg_fill", fake, raising=False)
        return fake

    return install


def started(tmp_path, monkeypatch, *, dataset=PILOT, table=PILOT_TABLE, passed=True):
    """A launch of ``dataset`` with a crash report on its manifest's table, the start command
    lines it handed the driver, and its result (or the refusal)."""
    crash = tmp_path / "crash.json"
    crash.write_text(json.dumps(report(dataset=dataset, data={"path": "t.csv", "sha256": table},
                                       **({} if passed else {"passed": False,
                                                             "defects": [{"refit": "x"}]}))))
    monkeypatch.setattr(L, "engine_now", lambda: dict(ENGINE))
    data = tmp_path / "walking_bp.csv"
    data.write_text("x,y\n1,0\n")
    runs, launches = tmp_path / "runs", tmp_path / "launches"
    given: list[list[str]] = []

    def start(argv):
        given.append(list(argv))
        driver.build_parser().parse_args(argv)  # a start command line the driver reads
        return 0, {"started": {"run": driver.new_run_id(runs)}}

    try:
        out = L.launch(dataset, crash, model="claude-haiku-5-5", persona="skeptical",
                       prompt_version="pilot-1", effort="high", runs=runs, launches=launches,
                       data=data, start=start)
    except L.Refused as exc:
        return given, exc
    return given, out | {"runs": runs, "launches": launches, "data": data}


def test_a_launch_starts_the_run_the_driver_names_with_prereg_jsons_budget(tmp_path,
                                                                           monkeypatch, prereg):
    fake = prereg()
    given, out = started(tmp_path, monkeypatch)
    # Asked of this dataset before the start (the staged pre-registration, 2026-10-28).
    assert fake.calls == [("load", None), ("ready", fake.prereg, PILOT)]
    (argv,) = given
    assert argv == ["start", "--data", str(out["data"]), "--budget-fits", "10",
                    "--budget-seconds", "72", "--model", "claude-haiku-5-5", "--prompt-version",
                    "pilot-1", "--persona", "skeptical", "--effort", "high"]
    label, secret = out["run"].rsplit("-", 1)
    assert driver.RUN_ID.fullmatch(out["run"]) and label == "run_01" and len(secret) == 32
    assert all(word not in out["run"] for word in ("walking", "known", "estimate"))
    kept = json.loads((out["launches"] / f"{out['run']}.json").read_text())
    known, _ = L.known_answer(PILOT)
    assert kept["dataset"]["known"] == known and kept["dataset"]["control"] == "known_answer"
    assert kept["budget"] == BUDGET  # prereg.json's, as it holds it
    assert not (out["runs"] / out["run"]).exists()  # nothing written into the run's own folder
    meta = L.record_meta(kept)
    assert meta["dataset"]["known"] == known and "known_source" not in meta["dataset"]
    assert meta["persona"] == "skeptical" and meta["design"] == 2
    # Under the corpus, where no analyst's command opens a file (the driver's seal).
    assert driver.unread(L.LAUNCHES / "run_01-x.json")


def test_prereg_json_not_ready_refuses_the_launch_in_its_words(tmp_path, monkeypatch, prereg):
    why = "prereg.json holds no r0: the default plan's replay is not measured."
    fake = prereg(why=why)
    given, refused = started(tmp_path, monkeypatch)
    assert isinstance(refused, L.Refused) and str(refused) == f"{PILOT} is not launched: {why}"
    assert given == [] and fake.calls[-1][0] == "ready"


def test_a_dataset_prereg_json_holds_no_fit_budget_for_is_not_launched(tmp_path, monkeypatch,
                                                                       prereg):
    """The decoy, which only the mechanical arms run: no budget is pre-registered for it."""
    prereg()
    given, refused = started(tmp_path, monkeypatch, dataset="outpatient_admission_predict",
                             table=TABLE)
    assert isinstance(refused, L.Refused) and given == []
    assert "prereg.json holds no fit budget for outpatient_admission_predict" in str(refused)


@pytest.mark.parametrize("budget", [{"fits": 0, "seconds": 72.0}, {"fits": 10, "seconds": None},
                                    {"fits": True, "seconds": 72.0},
                                    {"fits": 10, "seconds": float("nan")}],
                         ids=["no_fit", "no_seconds", "a_flag", "not_a_number"])
def test_a_pre_registered_budget_that_is_no_budget_is_refused(budget):
    with pytest.raises(L.Refused, match="not a count of fits and a number of seconds"):
        L.prereg_budget({"filled": {"fit_budget": {PILOT: budget}}}, PILOT)
    assert L.prereg_budget({"filled": {"fit_budget": {PILOT: BUDGET}}}, PILOT) == BUDGET


def test_a_dataset_whose_crash_sweep_did_not_pass_is_not_started(tmp_path, monkeypatch, prereg):
    prereg()
    given, refused = started(tmp_path, monkeypatch, passed=False)
    assert isinstance(refused, L.Refused) and "is not launched" in str(refused)
    assert "engine defect" in str(refused) and given == []


def test_the_committed_pre_registration_gates_the_launch_in_prereg_fills_words(tmp_path,
                                                                                monkeypatch):
    """With package 6b's own prereg_fill and prereg.json, where they are on this tree: the
    launcher refuses in the words ``ready`` gives for the file as committed."""
    P = pytest.importorskip("research.swarm.prereg_fill")
    monkeypatch.setattr(L, "engine_now", lambda: dict(ENGINE))
    why = P.ready(P.load(), dict(ENGINE), dataset=PILOT)
    given, out = started(tmp_path, monkeypatch)
    if why is None:  # pragma: no cover - a pre-registration filled and ready on this engine
        assert isinstance(out, dict) and given
    else:
        assert isinstance(out, L.Refused) and str(out) == f"{PILOT} is not launched: {why}"
        assert given == []


# ── the launches every checkout shares ───────────────────────────────────────


def test_the_launches_are_the_main_checkouts_whichever_checkout_reads_them():
    """The launches folder is research/corpus/launches in the main checkout, the first worktree
    ``git worktree list`` names, from this checkout whichever it is; the driver's seal covers it
    there too."""
    import subprocess
    from pathlib import Path

    out = subprocess.run(["git", "-C", str(L.REPO), "worktree", "list", "--porcelain"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        pytest.skip("not run from a git checkout")
    main = Path(out.stdout.splitlines()[0].split(" ", 1)[1]).resolve()
    assert L.LAUNCHES == L.launches_of() == main / "research" / "corpus" / "launches"
    assert driver.unread(L.LAUNCHES / "run_01-x.json")


def test_a_worktree_whose_main_checkout_git_cannot_name_reads_no_launches(tmp_path):
    """A folder whose ``.git`` is a file git cannot follow (a worktree it cannot place) has no
    launches folder it can be sure every checkout shares, so it is refused; a folder in no git
    repository is its own only checkout."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ".git").write_text("gitdir: /nowhere/.git/worktrees/x\n", "utf-8")
    with pytest.raises(L.Refused, match="cannot tell which checkout is the main one"):
        L.launches_of(broken)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert L.launches_of(plain) == plain.resolve() / "research" / "corpus" / "launches"


def test_the_launchers_command_line_takes_no_launches_folder_of_its_own(tmp_path, capsys):
    """A launch from the command line is written where every checkout reads the graded runs; no
    option writes it elsewhere."""
    with pytest.raises(SystemExit):
        L.main(["--dataset", PILOT, "--crash", str(tmp_path / "crash.json"), "--model", "m",
                "--prompt-version", "p", "--launches", str(tmp_path)])
    assert "--launches" in capsys.readouterr().err
