"""Findings speak with a lever (M1_CONTRACT §6): the NHANES export's findings route to the
questions that act on them, and the app says what the rubric says NHANES users were never told."""
from __future__ import annotations

import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.stages.finding_words import family, quote_columns
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import NHANES, SAMPLES, Ingested
from turbotab.server.schemas import FindingsArtifact

FLAGS = ["imputed_weight", "imputed_height", "imputed_bmi", "imputed_waist", "imputed_bp_sys",
         "imputed_bp_di"]


@pytest.fixture(scope="module")
def nhanes(tmp_path_factory):
    if not NHANES.is_file():
        pytest.skip("the real NHANES export is not on this machine")
    table = Ingested(NHANES, tmp_path_factory.mktemp("nhanes"))
    artifact = table.run(findings_stage, ProjectState(lens=["dietary", "clinical"], target="glucose"))
    return table, artifact


def by_family(artifact):
    out: dict[str, list[dict]] = {}
    for f in artifact["findings"]:
        out.setdefault(family(f["id"]), []).append(f)
    return out


def test_the_artifact_has_the_contract_shape(nhanes):
    FindingsArtifact.model_validate(nhanes[1])


def test_energy_adjustment_routes_to_its_question(nhanes):
    f, = by_family(nhanes[1])["pack::dietary::energy_adjustment"]
    assert (f["routes_to"], f["lever_label"]) == ("energy_adjustment", "Adjust for energy")
    table, _ = nhanes
    df = pd.read_csv(table.source)
    r = df["kcal"].corr(df["fat_total"])
    assert f["summary"] == (f"`fat_total` correlates {r:.2f} with `kcal`: nutrient effects are "
                            f"tangled with total energy.")
    # The legacy detail counted every numeric column as a candidate nutrient; it names the real ones.
    assert "19 other numeric columns" not in f["detail"]
    assert "`protein`, `carb`, `fat_total`, `fat_sat`, `fat_mon` and `fat_poly` carry energy" in f["detail"]


def test_implausible_intake_routes_to_exclusions(nhanes):
    f, = by_family(nhanes[1])["pack::dietary::implausible_intake"]
    assert f["routes_to"] == "exclusions"
    assert f["summary"] == ("`501` of `21,849` rows report `kcal` below `500` or above `5000` a day.")
    impossible, = by_family(nhanes[1])["pack::clinical::impossible_vs_extreme"]
    assert impossible["routes_to"] == "exclusions"


def test_the_identifier_the_design_and_the_flags_route_to_roles(nhanes):
    families = by_family(nhanes[1])
    seqn, = families["voice::identifier"]
    assert seqn["affected_columns"] == ["SEQN"] and seqn["routes_to"] == "roles"
    design, = families["voice::survey_design_absent"]
    assert design["routes_to"] == "roles" and design["severity"] == "warning"
    assert design["evidence"]["status"] == "SETTLED"
    flags = families["voice::flag"]
    assert sorted(f["affected_columns"] for f in flags) == sorted(
        [c, c.removeprefix("imputed_")] for c in FLAGS)
    assert {f["routes_to"] for f in flags} == {"roles"}
    assert {f["group"] for f in flags} == {"flags"}  # one paged card, not six
    cycles, = families["voice::pooled_cycles"]
    assert cycles["routes_to"] == "roles"
    assert cycles["summary"] == ("`cycle_begin_year` pools `9` survey cycles, `2001` to `2017`; "
                                 "methods may differ across them.")


def test_a_flag_column_is_reported_once(nhanes):
    """`imputed_weight` is a flag of `weight`, not also "a binary variable written as true/false"."""
    flagged = set(FLAGS)
    for f in nhanes[1]["findings"]:
        if family(f["id"]) in ("binary_text", "boolean_as_text"):
            assert not flagged & set(f["affected_columns"]), f["id"]
    reported = [f["id"] for f in nhanes[1]["findings"] if "meds_hbp" in f["affected_columns"]]
    assert reported == ["binary_text__meds_hbp"]


def test_blanks_that_may_mean_not_asked_route_to_missing(nhanes):
    f = next(f for f in nhanes[1]["findings"] if f["id"] == "binary_text__meds_hbp")
    assert f["routes_to"] == "missing"
    assert "`15,552` of `21,849` blank" in f["summary"]


def test_repeated_participants_are_named_on_the_recall_fixture(tmp_path):
    table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path)
    artifact = table.run(findings_stage, ProjectState(lens=["dietary"], target="hba1c"))
    f, = by_family(artifact)["voice::repeats"]
    assert f["summary"] == ("`participant_id` repeats: `600` rows from `300` participants, so rows "
                            "of one participant are not independent.")
    assert f["routes_to"] == "roles"
    assert "voice::survey_design_absent" not in by_family(artifact)  # not an NHANES table


def test_quoted_column_names_become_data_values():
    assert quote_columns("'gender' holds 'male' and 'female'", ["gender"]) == \
        "`gender` holds 'male' and 'female'"
    assert quote_columns("the participant's 'bmi'", ["bmi"]) == "the participant's `bmi`"
