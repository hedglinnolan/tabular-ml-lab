"""The previews keep the leash, second round (calm/FOUNDATION §5 rules 6–8; MODELING_SEQUENCE §4;
BLUEPRINT §14; audit ME-03): the residue the first round's verifier listed, each held to the
stage's own record or a hand computation.

1. **An unanswered purpose is the strictest case.** No outcome-model estimate appears until the
   purpose is answered as prediction or the analysis plan is locked: the first round's walk
   (``test_previews_leash``), repeated on its two tables with the purpose unanswered, finds no
   preview that tells them apart, and the fitted model's influence check is refused as not yet
   shown. Answered as prediction, the same walk sees the estimates a prediction may show.
2. **Never an empty canvas, on settled states.** Under each purpose, answered or not, no answer
   the server accepts previews the planner's generic line: regression calibration says what the
   calibration stage records, an event level what the server says of the outcome, and a
   multiplicity method what its tests wait for.
3. **Cluster units that span PSUs.** Under the surveyed population with every site's rows in more
   than one PSU, the fit refuses every coefficient; every preview of an answer that specifies the
   outcome model shows that refusal, with the sample-only exit and recording the answer as it is.
4. **Population estimand without a design-based estimator: block and record.** With the elastic
   net chosen beside the linear model, the swap and the sensitivity analyses preview the block the
   substitution and sensitivity stages record, with their exits.
5. **The design's warning after the lock.** The energy-dropped residual's warning withholds the
   coefficient gap until the estimates may be seen, and quotes it, as the server serves the design,
   once the plan is locked (audit ME-03); NumPy and statsmodels give the gap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.tests.acceptance import test_previews_leash as L
from turbotab.core.tests.acceptance.preview_harness import Project
from turbotab.core.tests.acceptance.test_previews_leash import influential  # noqa: F401 - fixture
from turbotab.core.voice import question_name

GENERIC = L.GENERIC
SAMPLE_ONLY = {"kind": "set_survey", "estimand": "sample"}
PREVIEW_NOTE = 30  # words (``turbotab.core.teaching``'s preview_note budget)


# ── 1 · an unanswered purpose is the strictest case ──────────────────────────────────────────


def binary_table(repaired: bool) -> pd.DataFrame:
    """The walk's table with a yes/no outcome: glucose at or above its median (the re-paired table
    re-pairs glucose first, so the yes/no outcome keeps its counts and no longer relates to
    anything)."""
    frame = L.walk_table(repaired)
    frame["high_glucose"] = np.where(frame["glucose"] >= frame["glucose"].median(), "yes", "no")
    return frame


BINARY_ROLES = {**L.ROLES, "glucose": "excluded", "high_glucose": "excluded"}


def binary_state(**update: Any) -> d.ProjectState:
    return L.walk_state(target="high_glucose", task="binary", event="yes",
                        roles=dict(BINARY_ROLES), role_confirmations=dict(BINARY_ROLES),
                        estimand=d.EstimandSpec(exposure="carb", contrast="substitution",
                                                measure="odds_ratio"), **update)


# Beyond the walk, the levers and the prediction evaluation's answers that read the outcome model.
EXTRA = [d.SetLevers(forms="inner_cv", imbalance="weights"),
         d.SetIntendedUse(use="decision_support", subgroups=["gender"]),
         d.SetSelection(method="stepwise")]
BINARY_WALK = [L.DROPPED, d.SetUpdating(method="shrinkage"), d.SetSelection(method="elastic_net"),
               *EXTRA]


def pair_of(folder: Path, table: Any, state: d.ProjectState) -> tuple[Project, Project]:
    return (Project(table(False), folder / "paired", state, upto=L.UPTO),
            Project(table(True), folder / "repaired", state, upto=L.UPTO))


@pytest.fixture(scope="module")
def unanswered(tmp_path_factory):
    """The walk's two tables, run through the stage graph with the purpose not answered yet."""
    pair = pair_of(tmp_path_factory.mktemp("leash2_unanswered"), L.walk_table,
                   L.walk_state(purpose=None))
    yield pair
    for project in pair:
        project.close()


@pytest.fixture(scope="module")
def unanswered_binary(tmp_path_factory):
    pair = pair_of(tmp_path_factory.mktemp("leash2_unanswered_binary"), binary_table,
                   binary_state(purpose=None))
    yield pair
    for project in pair:
        project.close()


def untimed(result: dict[str, Any]) -> dict[str, Any]:
    """A served preview without the families' fitting cost (``models.previews._say_the_cost``):
    the shelf's wall-clock measurement of each family on this machine, not an estimate of the
    outcome model, and not the same from run to run."""
    if str(result.get("note") or "").startswith("Fitting "):
        return {**result, "note": None}
    return result


def differing(pair: tuple[Project, Project], state: d.ProjectState,
              decisions: list[Any]) -> list[str]:
    """The decisions whose served preview is not the same on both tables (``L.differing``, its
    timing masked)."""
    out = []
    for decision in decisions:
        one, other = (untimed(L.served(p, decision, state)) for p in pair)
        if json.dumps(one, sort_keys=True) != json.dumps(other, sort_keys=True):
            out.append(f"{decision.kind} ({getattr(decision, 'method', '')}): "
                       f"{json.dumps(one, sort_keys=True)[:400]}")
    return out


def test_1_with_the_purpose_unanswered_no_preview_shows_an_outcome_model_estimate(unanswered):
    assert differing(unanswered, L.walk_state(purpose=None), [*L.WALK, *EXTRA]) == []


def test_1_a_yes_no_outcome_with_the_purpose_unanswered_shows_none_either(unanswered_binary):
    """The same on a yes/no outcome: the fitted model's risks (decision support), the calibration
    slope (shrinkage), what a selection keeps and inner cross-validation's bends."""
    assert differing(unanswered_binary, binary_state(purpose=None), BINARY_WALK) == []


def test_1_answered_as_prediction_the_walk_sees_the_estimates_it_may_show(unanswered):
    """The positive control: under prediction the energy-dropped residual quotes the outcome
    model's coefficients, shrinkage its calibration slope and the elastic net what it keeps, so
    the two tables' previews differ, and the detector sees it."""
    found = differing(unanswered, L.walk_state(purpose="prediction"),
                      [L.DROPPED, d.SetUpdating(method="shrinkage"),
                       d.SetSelection(method="elastic_net")])
    assert [x.split(" ")[0] for x in found] == ["set_energy_adjustment", "set_updating",
                                                "set_selection"]
    assert "coefficient" in found[0]


def test_1_the_energy_dropped_residual_with_the_purpose_unanswered_quotes_no_coefficient(
        unanswered):
    """The method's own picture: carb's correlation with kcal on every row (NumPy; no row is held
    out), none after the residual, and kcal leaving the outcome model."""
    paired, _ = unanswered
    state = L.walk_state(purpose=None, energy_adjustment=None)
    result = L.served(paired, L.DROPPED, state)
    frame = L.walk_table(False)
    r = np.corrcoef(frame["carb"], frame["kcal"])[0, 1]
    assert result["views"][0]["caption"] == (
        f"`carb` correlates {r:.2f} with `kcal`; after residual adjustment, 0.00; `kcal` leaves "
        f"the outcome model.")
    assert "coefficient" not in json.dumps(result)


def test_1_the_influence_check_waits_while_the_purpose_is_unanswered(influential):  # noqa: F811
    """The fitted model's influence check flags one row (Cook's distance by hand), and the effects
    stage reports it failed; with the purpose unanswered the response is refused as not yet shown,
    and its preview counts no row from the fitted model."""
    project, frame = influential
    flagged, _ = L.cooks_flagged(frame)
    assert flagged == 1
    state = project.state.model_copy(update={"purpose": None})
    refused = L.served(project, L.INFLUENCE, state)
    assert refused["refused"]["error"]["code"] == "check_not_shown", refused
    result, _ = project.preview(L.INFLUENCE, ctx=project.context(state=state))
    assert result.views == [] and "Cook" not in json.dumps(result.model_dump(mode="json"))


# ── 2 · never an empty canvas, on settled states ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def purposes(tmp_path_factory, unanswered):
    """The walk's table under each purpose (the unanswered one is the walk's above)."""
    folder = tmp_path_factory.mktemp("leash2_purposes")
    out: dict[str | None, Project] = {None: unanswered[0]}
    for purpose in ("inference", "prediction"):
        out[purpose] = Project(L.walk_table(False), folder / purpose,
                               L.walk_state(purpose=purpose), upto=L.UPTO)
    yield out
    for purpose in ("inference", "prediction"):
        out[purpose].close()


@pytest.mark.parametrize("purpose", [None, "inference", "prediction"])
def test_2_no_settled_preview_ends_on_the_generic_line(purposes, purpose):
    """Every answer of the walk the server accepts under this purpose previews something: a view,
    a caution with its controls, or one line that is not the planner's last resort."""
    project, state = purposes[purpose], L.walk_state(purpose=purpose)
    empty = []
    for decision in L.WALK:
        out = L.served(project, decision, state)
        if "refused" in out:
            continue
        if out["note"] == GENERIC or not (out["views"] or out["note"] or out["caution"]):
            empty.append(decision.kind)
        if out["note"]:
            assert voice.words(out["note"]) <= PREVIEW_NOTE, (decision.kind, out["note"])
    assert empty == []


CALIBRATION = d.SetMeasurementError(method="regression_calibration", exposures=["carb"])


def test_2_regression_calibration_without_repeated_recalls_says_what_the_stage_records(purposes):
    """One record per person, no repeated recalls: recorded, the calibration stage calibrates
    nothing and says why; the preview says the same before it is recorded. With the purpose
    unanswered, it names the question that settles it."""
    project = purposes["inference"]
    result, _ = project.preview(CALIBRATION, ctx=project.context(state=project.state))
    _, out = project.after(CALIBRATION, upto=["calibration"])
    recorded = out["calibration"].data
    assert recorded["applies"] is False and recorded["reason"]
    assert result.views == [] and result.note == recorded["reason"]
    unanswered = purposes[None]
    result, _ = unanswered.preview(CALIBRATION, ctx=unanswered.context(state=unanswered.state))
    assert result.note is not None and question_name("purpose") in result.note


def test_2_an_event_level_of_a_quantity_says_what_the_server_says(purposes):
    """``glucose`` is a quantity: the server, which knows the task, refuses an event level, and
    the preview says its refusal; while the outcome's levels are read again, it says so."""
    project = purposes["inference"]
    decision = d.SetEvent(column="glucose", level="1")
    with pytest.raises(d.Refusal) as refused:
        d.validate(decision, {"state": project.state, "target": "glucose", "task": "regression"})
    result, _ = project.preview(decision, ctx=project.context(state=project.state))
    assert result.views == [] and result.note == refused.value.message
    binary = project.state.model_copy(update={"task": "binary"})
    stale = {k: v for k, v in project.artifacts.items() if k != "target_info"}
    result, _ = project.preview(decision, ctx=project.context(state=binary, artifacts=stale))
    assert result.note not in (None, GENERIC) and "read again" in result.note


def test_2_a_multiplicity_method_says_what_its_tests_wait_for(purposes):
    """Under prediction no exposure's test is reported, so the method changes nothing and the
    preview says it applies under inference; with the purpose unanswered, the purpose question
    settles it. Under inference the family's three exposures draw the thresholds."""
    decision = d.SetMultiplicity(method="bh")
    for purpose in ("prediction", None):
        project = purposes[purpose]
        result, _ = project.preview(decision, ctx=project.context(state=project.state))
        assert result.views == [] and result.note not in (None, GENERIC), purpose
        assert question_name("purpose") in result.note and "inference" in result.note
    project = purposes["inference"]
    result, _ = project.preview(decision, ctx=project.context(state=project.state))
    assert [v.kind for v in result.views] == ["relationship"]
    assert result.views[0].caption.startswith("`3` tests")


# ── 3 · cluster units that span PSUs ─────────────────────────────────────────────────────────

# The walk's nested parts, confirmed as the author knows them, so the swap moves them with their
# totals (``readings``: nested_in).
PARTS = {f"nested_in:{c}": p for c, p in (("sugar", "carb"), ("fat_sat", "fat_total"),
                                           ("fat_mon", "fat_total"), ("fat_poly", "fat_total"))}
SURVEY_UPTO = ["design", "cohort", "split", "fit", "effects"]


def surveyed_state(**update: Any) -> d.ProjectState:
    roles = {**L.ROLES, **L.DESIGN}
    return L.walk_state(roles=roles, role_confirmations=dict(roles), survey=L.SURVEYED,
                        shape_confirmations={"code_or_count:age": "amount", **PARTS}, **update)


@pytest.fixture(scope="module")
def spanning_diet(tmp_path_factory):
    """The walk's table under a survey design drawn row by row (8 strata of 2 PSUs, a weight;
    ``L.with_design``), so every site's rows fall in several PSUs, the surveyed population
    declared and the intervals clustered by site."""
    frame = L.with_design(L.walk_table(False))
    project = Project(frame, tmp_path_factory.mktemp("leash2_spanning"), surveyed_state(),
                      upto=SURVEY_UPTO)
    yield project, frame
    project.close()


SWAP = d.SetSubstitution(donor="fat_total", recipient="carb", step_kcal=100)
SENSITIVITY = d.SetSensitivity(analyses=[d.SensitivityAnalysis(label="800–4,000 kcal", rules=[
    d.ExclusionRule(column="kcal", low=800, high=4000, reason="implausible intakes")])])
# Every answer that specifies the outcome model whose estimates the fit refuses.
SPECIFYING = [
    d.SetAdjustment(exposure="carb", answers={"bmi": d.CovariateAnswers(**L.CONFOUNDER)}),
    d.SetExposureForm(column="carb", form="spline", knots=4),
    d.SetForms(forms={"carb": d.ExposureFormSpec(form="spline", knots=4)}),
    d.SetModification(modifier="gender"),
    d.SetMultiplicity(method="bh"),
    SENSITIVITY,
    d.SetEnergyAdjustment(method="residual", **L.NUTRIENTS),
    SWAP,
    d.SetCategorical(columns=["gender"]),
    # Those that showed it already (the first round and REPAIR-PREVIEWS).
    d.SetEstimand(exposure="carb", contrast="substitution", measure="mean_difference"),
    d.SetModelSequence(exposure="carb", model_1=["age"]),
    d.SelectModels(models=["linear"]),
    d.SetClusters(column="site", adjust="cluster_only"),
    L.POPULATION.model_copy(update={"weight": "wt", "strata": "stratum", "psu": "psu"}),
]


def test_3_every_specifying_preview_shows_the_fits_refusal_when_sites_span_psus(spanning_diet):
    """The fit refuses every coefficient, saying how many sites span PSUs (pandas counts them:
    all 24); every preview of an answer that specifies the outcome model says the same, with the
    sample-only exit and recording the answer as it is, each exit a decision the server accepts."""
    project, frame = spanning_diet
    state = project.state
    held = frame.groupby("site")[["stratum", "psu"]].apply(
        lambda g: len(set(zip(g["stratum"], g["psu"]))))
    spanning = int((held > 1).sum())
    assert spanning == 24
    [model] = project.artifacts["fit"].data["models"]
    refused = model["inference"]["refused"]
    assert model["coefficients"] == [] and refused.startswith(
        f"{spanning} `site` units have rows in more than one PSU")
    ctx = {"state": state, "columns": list(project.store.columns), "target": state.target,
           "artifact": lambda stage: project.artifacts.get(stage)}
    missing = []
    for decision in SPECIFYING:
        out = L.served(project, decision, state)
        assert "refused" not in out, (decision.kind, out)
        caution = out["caution"]
        if caution is None or caution["text"] != refused:
            missing.append(decision.kind)
            continue
        as_recorded = d.validate(decision, ctx).model_dump(mode="json")  # the server's parse
        assert [x["decision"] for x in caution["exits"]] == [SAMPLE_ONLY, as_recorded], \
            decision.kind
    assert missing == []
    assert d.validate(SAMPLE_ONLY, ctx).kind == "set_survey"  # the exit is accepted


def test_3_recorded_the_swap_and_the_sensitivity_analyses_meet_the_same_refusal(spanning_diet):
    """What the previews say is what recording does: the substitution stage draws no curve over a
    design the fit refuses (it drew these participants' curve, with no band, in its place), and
    each sensitivity analysis's table is refused with the fit's words and exits."""
    project, _ = spanning_diet
    [model] = project.artifacts["fit"].data["models"]
    info = model["inference"]
    _, out = project.after(SWAP, upto=["substitution"])
    [curve] = out["substitution"]["models"]
    assert (curve["refused"], curve["exits"]) == (info["refused"], info["exits"])
    assert all(v is None for v in curve["delta"])
    _, out = project.after(SENSITIVITY, upto=["sensitivity"])
    [family] = out["sensitivity"].data["families"]
    assert [(f["inference"]["refused"], f["inference"]["exits"]) for f in family["fits"]] == [
        (info["refused"], info["exits"])] * 2


# ── 4 · population estimand without a design-based estimator: block and record ─────────────


RECORD = L.RECORD
TWO = ["linear", "elastic_net"]


@pytest.fixture(scope="module")
def nested_diet(tmp_path_factory):
    """The walk's table under a survey design drawn per site, so each site's rows sit in one PSU
    and the design stands; the surveyed population declared, the linear model and the elastic net
    chosen."""
    frame = L.with_design(L.walk_table(False), unit="site")
    project = Project(frame, tmp_path_factory.mktemp("leash2_nested"), surveyed_state(models=TWO),
                      upto=SURVEY_UPTO)
    yield project
    project.close()


def test_4_the_swap_previews_the_block_the_substitution_stage_records(nested_diet):
    """Recorded, the substitution stage draws the linear model's population curve and blocks the
    elastic net's, with its reason and exits (the survey-weighted family in its place; the
    sample-only answer); the swap's preview says the same, with recording it as it is. With the
    linear model alone, nothing is blocked."""
    state = nested_diet.state
    result = L.served(nested_diet, SWAP, state)
    _, out = nested_diet.after(SWAP, upto=["substitution"])
    curves = {m["family"]: m for m in out["substitution"]["models"]}
    assert curves["linear"]["refused"] is None and any(
        v is not None for v in curves["linear"]["delta"][1:])
    blocked = curves["elastic_net"]
    assert blocked["refused"].startswith("Elastic net has no design-based estimator, so its "
                                         "substitution curve")
    caution = result["caution"]
    assert caution is not None and caution["text"] == blocked["refused"]
    assert [x["decision"] for x in caution["exits"]] == [
        *(e["decision"] for e in blocked["exits"]), SWAP.model_dump(mode="json")]
    assert caution["exits"][-1]["label"] == RECORD
    alone = L.served(nested_diet, SWAP, state.model_copy(update={"models": ["linear"]}))
    assert alone["caution"] is None and alone["views"]


def test_4_the_sensitivity_analyses_preview_the_block_the_sensitivity_stage_records(nested_diet):
    """Recorded, each sensitivity analysis refits the linear model over the design and blocks the
    elastic net's table, with the fit's reason and exits; the preview says the same beside the
    rows each analysis keeps."""
    state = nested_diet.state
    result = L.served(nested_diet, SENSITIVITY, state)
    _, out = nested_diet.after(SENSITIVITY, upto=["sensitivity"])
    families = {f["family"]: f for f in out["sensitivity"].data["families"]}
    assert all(f["inference"]["refused"] is None for f in families["linear"]["fits"])
    info = families["elastic_net"]["fits"][-1]["inference"]
    assert info["refused"].startswith("Elastic net has no design-based estimator")
    caution = result["caution"]
    assert caution is not None and caution["text"] == info["refused"]
    assert [x["decision"] for x in caution["exits"]] == [
        *(e["decision"] for e in info["exits"]), SENSITIVITY.model_dump(mode="json")]
    assert caution["exits"][-1]["label"] == RECORD
    assert [v["kind"] for v in result["views"]] == ["row_flow"]


# ── 5 · the design's warning after the lock (audit ME-03) ────────────────────────────────────


def gap_design(tmp_path: Path, purpose: str | None) -> Any:
    """The methods gate's s14 fixture through the real split and design stages, the energy-dropped
    residual recorded, 20% held out."""
    from turbotab.core.decisions import EnergyAdjustment, SplitSpec
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import test_methods_gate as G

    frame = G._residual_fixture()
    paths = mf.ingest_frame(frame, tmp_path)
    frame = pd.read_csv(Path(paths["source"]))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=3)
    state = mf.state(roles=G.RESIDUAL_ROLES, target="y", task="regression", purpose=purpose,
                     models=["linear"], split=SplitSpec(holdout=0.2, seed=3, folds=5),
                     energy_adjustment=EnergyAdjustment(method="residual_energy_dropped",
                                                        energy_column="energy_kcal",
                                                        nutrients=["fat_g"]))
    ti = mf.target_info("regression", "y")
    return frame, state, design_stage(mf.context(state, {"split": split, "target_info": ti}, paths))


def gap_line(frame: pd.DataFrame, n: str, rows: str) -> str:
    from turbotab.core.tests.acceptance import test_methods_gate as G

    dropped, standard = G._gap(frame)
    return (f"energy_kcal left the outcome model: on {n} {rows} fat_g_adj's coefficient is "
            f"{G._sig3(dropped)}, against {G._sig3(standard)} with energy_kcal kept (the standard "
            f"model's). The two agree only when no covariate correlates with energy_kcal.")


WITHHELD = ("energy_kcal left the outcome model: each nutrient's coefficient is the standard "
            "model's (with energy_kcal kept) only when no covariate correlates with energy_kcal.")


@pytest.mark.parametrize("purpose", ["inference", None])
def test_5_the_design_withholds_the_gap_until_the_lock_and_serves_it_after(tmp_path, purpose):
    """Under inference, and with the purpose unanswered, the design's warning says what the form
    does without the outcome model's coefficients; served once the plan is locked, it quotes them:
    under inference on every one of the 3,000 analyzed rows (NumPy's residual, statsmodels' OLS:
    the verifier's 0.022624 and 0.021624)."""
    from turbotab.core.stages.modeling import design_as_served

    frame, state, design = gap_design(tmp_path, purpose)
    assert [w for w in design.data["warnings"] if w.startswith("energy_kcal left")] == [WITHHELD]
    assert "fat_g_adj's coefficient" not in json.dumps(design.data)
    assert design_as_served(design.data, design.objects, state) == design.data
    if purpose is None:
        return
    locked = design_as_served(design.data, design.objects,
                              state.model_copy(update={"plan_locked": True}))
    assert [w for w in locked["warnings"] if w.startswith("energy_kcal left")] == [
        gap_line(frame, "3,000", "analyzed rows")]
    assert [w for w in design.data["warnings"] if w.startswith("energy_kcal left")] == [WITHHELD]


def test_5_the_server_serves_the_gap_once_the_plan_is_locked(tmp_path):
    """Through the real server under inference: the design is served without the coefficients
    before the lock and with them after (the lock recorded as a client may record it), the gap
    the one NumPy and statsmodels give on all 3,000 analyzed rows."""
    from turbotab.core.tests.acceptance import test_methods_gate as G
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.truths import Truth

    frame = G._residual_fixture()
    path = tmp_path / "resid.csv"
    frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    dropped = {"kind": "set_energy_adjustment", "method": "residual_energy_dropped",
               "energy_column": "energy_kcal", "nutrients": ["fat_g"]}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, Truth(G.RESIDUAL_TRUTH, fixture="_residual_fixture"))
        answers = {
            "lens": {"kind": "set_lens", "lenses": ["dietary"]},
            "target": {"kind": "set_target", "column": "y"},
            "task": {"kind": "set_task", "column": "y", "task": "regression"},
            "purpose": {"kind": "set_purpose", "purpose": "inference"},
            "grain": {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"},
            "temporal": {"kind": "set_temporal", "temporal": False},
            "roles": {"kind": "set_roles", "roles": G.RESIDUAL_ROLES},
            "survey": {"kind": "set_survey", "estimand": "sample"},
            "exclusions": {"kind": "set_exclusions", "rules": []},
            "missing": {"kind": "set_missing", "strategy": "complete_case"},
            "split": {"kind": "set_split", "holdout": 0.2, "seed": 3, "folds": 5},
            "energy_adjustment": dropped,
            "models": {"kind": "select_models", "models": ["linear"]},
        }
        G._answer_until(drive, "models", answers)

        def said() -> list[str]:
            return [w for w in drive.artifact("design")["warnings"]
                    if w.startswith("energy_kcal left")]

        assert said() == [WITHHELD]
        assert not drive.view()["state"].get("plan_locked")
        # The first estimate served locks the plan (``plan_lock``): the fit's coefficient table.
        fit = drive.artifact("fit")
        assert any(m.get("coefficients") for m in fit["models"])
        assert drive.view()["state"]["plan_locked"] is True
        assert said() == [gap_line(frame, "3,000", "analyzed rows")]
