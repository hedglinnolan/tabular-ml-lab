"""The opening sequence on the five lens fixtures (Tier B): the Router's order, what fires and what
is stated, and the refusals that keep the structural answers coherent (M2_CONTRACT §1, §7)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbotab.core import decisions
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.interview import QUESTION_KEYS, route
from turbotab.core.tests.graph_runner import GraphRun

SAMPLES = Path(__file__).resolve().parents[2] / "sample_data"
FRESH = {"status": "fresh"}


def state(**slots) -> ProjectState:
    return ProjectState.model_validate(slots)


class Project:
    """A fixture through the real graph, and the Router and validators over its artifacts."""

    def __init__(self, source: Path, folder: Path):
        self.run = GraphRun(source, folder)
        self.out: dict = {}

    def at(self, s: ProjectState) -> "Project":
        self.state = s
        self.out = self.run.run(s, upto=["structure", "working", "target_info"])
        return self

    def artifact(self, stage: str):
        value = self.out.get(stage)
        return getattr(value, "data", value)

    def steps(self) -> dict:
        stages = {stage.name: dict(FRESH) for stage in self.run.graph.stages()}
        artifacts = {k: self.artifact(k) for k in ("target_info", "oriented", "structure")
                     if self.artifact(k) is not None}
        steps = route(self.state, stages, artifacts)
        assert [s.key for s in steps] == list(QUESTION_KEYS)
        assert [s.status for s in steps].count("open") <= 1
        return {s.key: s for s in steps}

    def validate(self, decision: dict, **extra) -> object:
        info = self.artifact("working") or self.artifact("oriented")
        ctx = {
            "columns": [c["name"] for c in info["columns"]],
            "column_info": {c["name"]: c for c in info["columns"]},
            "state": self.state, "target": self.state.target,
            "task": self.state.task or (self.artifact("target_info") or {}).get("task"),
            "artifact": self.artifact, **extra,
        }
        return decisions.validate(decision, ctx)


@pytest.fixture
def project(tmp_path):
    made: list[Project] = []

    def make(name: str | Path) -> Project:
        source = name if isinstance(name, Path) else SAMPLES / name
        p = Project(source, tmp_path / f"p{len(made)}")
        made.append(p)
        return p

    yield make
    for p in made:
        p.run.close()


def refused(fn, decision) -> Refusal:
    with pytest.raises(Refusal) as caught:
        fn(decision)
    return caught.value


# ── dietary: repeats stated, one row per person, the mean recommended ─────────


def test_dietary_recalls_repeat_per_person_and_are_combined_by_their_mean(project):
    p = project("dietary_recalls.csv").at(state(lens=["dietary"], target="hba1c",
                                                purpose="prediction"))
    steps = p.steps()
    assert steps["orientation"].status == "not_applicable"  # no assay lens
    assert steps["event"].status == "not_applicable"  # hba1c is continuous
    assert steps["task"].status == "skipped"
    assert steps["grain"].status == "open"  # always asked, never answered for the user
    assert p.artifact("structure")["grain"]["suggested"][0] == "participant_id"

    # "one row per person" contradicts the data: refused, with the attestation exit
    no = refused(p.validate, {"kind": "set_grain", "grain": "one_row_per_unit"})
    assert no.code == "data_repeats" and "`participant_id`" in no.message
    assert [e["decision"]["acknowledged"] for e in no.exits if e["decision"]] == [False, True]
    p.validate({"kind": "set_grain", "grain": "one_row_per_unit", "acknowledged": True})
    assert refused(p.validate, {"kind": "set_grain", "grain": "repeated"}).code == "no_id_column"

    p.at(p.state.model_copy(update={"grain": decisions.GrainSpec(grain="repeated",
                                                                 id_column="participant_id")}))
    steps = p.steps()
    assert steps["grain"].status == "answered"
    assert steps["repeat_kind"].status == "skipped"  # stated: repeats, from the recall spacing
    assert steps["repeat_kind"].reason.startswith("these look like repeated")  # after "Not asked:"
    assert "`recall_date`" in steps["repeat_kind"].reason
    assert steps["unit"].status == "open"  # no default
    assert steps["temporal"].status == "not_applicable"  # repeats, not time points

    p.at(p.state.model_copy(update={"unit": "unit"}))
    steps = p.steps()
    assert steps["aggregation"].status == "open" and steps["temporal"].status == "not_applicable"
    menu = p.artifact("structure")["aggregation"]
    assert menu["kind"] == "repeats" and menu["recommended"] == "mean"
    assert "measurement error" in menu["reason"]
    p.validate({"kind": "set_aggregation", "method": "mean"})  # hba1c is one value per person

    p.at(p.state.model_copy(update={"aggregation": decisions.AggregationSpec(method="mean")}))
    assert p.steps()["roles"].status == "open"
    assert p.artifact("working")["n_rows"] == 300  # energy adjustment then runs per person


# ── clinical: time points stated, rows kept, temporal asked ───────────────────


def test_clinical_visits_are_time_points_and_ask_about_temporal_prediction(project):
    p = project("clinical_longitudinal.csv").at(state(
        lens=["clinical"], target="progressed", purpose="prediction",
        grain={"grain": "repeated", "id_column": "subject_id"}))
    steps = p.steps()
    assert steps["event"].status == "open"  # progressed is binary: which level is the event
    assert steps["repeat_kind"].status == "skipped"
    assert "different time points" in steps["repeat_kind"].reason
    assert "90 days apart" in steps["repeat_kind"].reason

    rows = p.at(p.state.model_copy(update={"event": "1", "unit": "row"})).steps()
    assert rows["aggregation"].status == "not_applicable"
    assert rows["temporal"].status == "open"  # time points stay as rows
    p.validate({"kind": "set_temporal", "temporal": True})  # the visit date orders them
    assert p.artifact("structure")["time_column"] == "visit_date"

    units = p.at(p.state.model_copy(update={"unit": "unit"})).steps()
    assert units["aggregation"].status == "open" and units["temporal"].status == "not_applicable"
    menu = p.artifact("structure")["aggregation"]
    assert menu["kind"] == "time_points" and menu["recommended"] is None  # no default
    assert p.artifact("structure")["outcome"]["varies"] is True
    which = refused(p.validate, {"kind": "set_aggregation", "method": "last"})
    assert which.code == "which_outcome" and "127" in which.message
    assert {e["decision"]["outcome"] for e in which.exits} == {"first", "last"}  # binary: no mean
    assert refused(p.validate, {"kind": "set_aggregation", "method": "last",
                                "outcome": "mean"}).code == "outcome_not_numeric"
    p.validate({"kind": "set_aggregation", "method": "change", "outcome": "last"})


def test_the_event_names_a_level_of_the_binary_outcome(project):
    p = project("clinical_longitudinal.csv").at(state(lens=["clinical"], target="progressed"))
    p.validate({"kind": "set_event", "column": "progressed", "level": "1"})
    p.validate({"kind": "set_event", "column": "progressed", "level": "1.0"})  # one level, two spellings
    assert refused(p.validate, {"kind": "set_event", "column": "progressed",
                                "level": "yes"}).code == "unknown_level"
    assert refused(p.validate, {"kind": "set_event", "column": "sbp",
                                "level": "1"}).code == "not_the_target"


# ── metabolomics: orientation fires only on the turned-around copy ───────────


def test_orientation_fires_only_on_a_feature_major_assay_table(project, tmp_path):
    plain = project("metabolomics_untargeted.csv").at(state(lens=["metabolomics"]))
    steps = plain.steps()
    assert steps["orientation"].status == "not_applicable"
    assert steps["target"].status == "open"
    assert plain.artifact("structure")["grain"]["suggested"] == []  # one row per subject

    m = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv").set_index("sample_id")
    turned = m.select_dtypes("number").T
    turned.index.name = "feature_id"
    source = tmp_path / "metabolomics_T.csv"
    turned.to_csv(source)
    t = project(source).at(state(lens=["metabolomics"]))
    steps = t.steps()
    assert steps["orientation"].status == "open"
    assert steps["target"].status == "waiting" and steps["target"].waiting_on == ["orientation"]
    assert t.artifact("oriented")["reading"]["ratio"] > 4
    clinical = t.at(state(lens=["clinical"])).steps()  # no assay lens: never asked
    assert clinical["orientation"].status == "not_applicable"

    t.at(state(lens=["metabolomics"], orientation="feature_major"))
    steps = t.steps()
    assert steps["orientation"].status == "answered" and steps["target"].status == "open"
    oriented = t.artifact("oriented")
    assert (oriented["n_rows"], oriented["n_cols"]) == (80, 397)
    assert oriented["columns"][0]["name"] == "sample_id"
    # diagnosis ran on the turned table: the findings read samples as rows
    assert "80 rows" in t.artifact("findings")["basis"]
    # turning back is refused once an outcome is chosen on the turned table
    t.state = t.state.model_copy(update={"target": "responder"})
    assert refused(t.validate, {"kind": "set_orientation",
                                "orientation": "sample_major"}).code == "target_exists"


# ── genomics: one row per sample, nothing repeats ─────────────────────────────


def test_genomics_reads_one_row_per_sample_and_the_repeats_chain_stays_quiet(project):
    p = project("genomics_expression.csv").at(state(lens=["genomics"], target="age",
                                                    purpose="prediction",
                                                    grain={"grain": "one_row_per_unit"}))
    steps = p.steps()
    assert steps["orientation"].status == "not_applicable"
    for key in ("repeat_kind", "unit", "aggregation", "temporal"):
        assert steps[key].status == "not_applicable", key
    assert p.artifact("working")["pass_through"] is True


# ── survey: the event level of a binary item ─────────────────────────────────


def test_a_binary_survey_outcome_asks_for_its_event_level(project):
    p = project("survey_instrument.csv").at(state(lens=["survey"], target="sought_support"))
    steps = p.steps()
    assert steps["orientation"].status == "not_applicable"
    assert steps["event"].status == "open"
    levels = [str(c["value"]) for c in p.artifact("target_info")["classes"]]
    assert len(levels) == 2
    p.validate({"kind": "set_event", "column": "sought_support", "level": levels[0]})
    survey = project("survey_sentinels.csv").at(state(lens=["survey"]))
    assert survey.steps()["orientation"].status == "not_applicable"


def test_the_grain_names_a_column_that_repeats(project):
    m = project("metabolomics_untargeted.csv").at(state(lens=["metabolomics"]))
    unique = refused(m.validate, {"kind": "set_grain", "grain": "repeated", "id_column": "sample_id"})
    assert unique.code == "id_column_unique" and "every one of its 80 rows" in unique.message
    assert unique.exits[-1]["decision"]["acknowledged"] is True  # attest: the user is the authority
    m.validate({"kind": "set_grain", "grain": "repeated", "id_column": "sample_id", "acknowledged": True})
    m.validate({"kind": "set_grain", "grain": "one_row_per_unit"})  # nothing repeats like a roster

    p = project("dietary_recalls.csv").at(state(lens=["dietary"], target="hba1c"))
    assert refused(p.validate, {"kind": "set_grain", "grain": "repeated",
                                "id_column": "hba1c"}).code == "target_is_id"
    assert refused(p.validate, {"kind": "set_grain", "grain": "repeated",
                                "id_column": "nope"}).code == "unknown_column"
    assert refused(p.validate, {"kind": "set_unit", "unit": "unit"}).code == "not_repeated"
