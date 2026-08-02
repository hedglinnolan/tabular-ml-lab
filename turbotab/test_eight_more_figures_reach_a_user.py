"""L40-C — the eight figures, and the companion that was never registered.

L39 traded these whole and said so; a part deferred twice is how a gap becomes
invisible, so they are Part C this loop and the scope note protected them.
Hardest-first per `LOOP.md` §02: decision curve, the two instability plots it
unblocked, the forest plot, then the ROC, then the survey four.

## What building them found

**`calibration` has named a companion that does not exist since L34.** Its
`companions` was `("discrimination",)` — an id never registered and never
declared pending — so `figures.bundle` held it on every project that could
otherwise draw it. `GUIDED-058`'s class at the companion layer: the figure was
correct, tested, and unreachable, and the thing it was waiting for did not
exist until §A4.4 was built this loop. `GUIDED-128`.

And with it finally admissible, it fails one of its own checklist items —
invisible for six loops because nothing ever admitted it. `GUIDED-129`.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes for the clinical four
(binary classification and a three-class control that must NOT be offered
them), and the survey four run against the survey fixture with the shapes not
covered named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import figure_bundle as FB
from turbotab import figure_specs as F
from turbotab import figures
from turbotab import training as T
from turbotab.project import AnalysisProject

#: `GUIDED-097`. The three-class arm is a CONTROL: L39-D found that nothing
#: declines a multiclass target, and these four are the first figures built
#: after that finding, so they are the first that must decline one.
TARGET_SHAPES = {
    "binary classification": ("leaky_sepsis.csv", "sepsis", "classification",
                              "logreg"),
    # `GUIDED-135`, added at L41. Same target shape as the arm above and the
    # opposite CALIBRATION shape: `leaky_sepsis.csv` separates completely, so
    # every clinical claim in this file had only ever been checked against a
    # model with a C-statistic of exactly 1.000. That is a fixture property, it
    # is not a property any real cohort has, and the figures below are about
    # discrimination and net benefit — quantities a separating model makes
    # degenerate.
    "binary classification, non-separating": ("clinical_risk.csv",
                                              "readmit_30d", "classification",
                                              "logreg"),
    "three-class classification": ("multiclass_stage.csv", "disease_stage",
                                   "classification", "logreg"),
}

#: The two arms that are binary, which is what the clinical figures need. The
#: three-class arm is a CONTROL and must be declined, so it is not in here.
CLINICAL_BINARY_SHAPES = ("binary classification",
                          "binary classification, non-separating")

#: NOT COVERED, said out loud.
#:
#: POLYCHORIC CORRELATIONS. §B5.4 is SETTLED that they are the appropriate
#: choice for Likert items and §B5.5 specifies parallel analysis ON them.
#: Nothing in this repository computes one and no dependency ships one, so the
#: two structure figures use Pearson and SAY Pearson — which is §B5.4's own
#: requirement — with the consequence stated: attenuated correlations
#: understate loadings and retain fewer factors. `GUIDED-127`.
#:
#: CFA FIT INDICES. §B5.5 asks for CFI, TLI, RMSEA with its CI and SRMR as
#: values and explicitly not as PASS/FAIL. This app has no confirmatory factor
#: model, so they are absent with the absence explained rather than
#: approximated.
#:
#: SURVIVAL. `GUIDED-118`; L38's refusal stands and this loop does not lift it.
#:
#: COEFFICIENT INTERVALS. The forest plot draws bare points where the
#: estimator exposes no standard errors, which sklearn's linear models do not.
#: A thinner figure is not a false one; an interval invented from nothing
#: would be.
SHAPES_NOT_COVERED = [
    "polychoric correlations — not computable here; the figures use Pearson "
    "and say so, per §B5.4's own caption requirement (GUIDED-127)",
    "CFA fit indices — no confirmatory factor model exists, so they are "
    "absent with the absence explained rather than approximated",
    "survival / time-to-event — no task type (GUIDED-118)",
    "coefficient confidence intervals — sklearn's linear models expose no "
    "standard errors, so the forest plot draws points without intervals",
]

CLINICAL = ("decision_curve", "roc", "forest")
SURVEY = ("scree", "item_correlations", "floor_ceiling", "item_panel")
INSTABILITY = ("classification_instability", "decision_curve_instability")


def _clinical(shape):
    name, target, task, model = TARGET_SHAPES[shape]
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    p.training_run = T.train(p, [model])
    return p


def _survey():
    from turbotab import packs

    df = pd.read_csv("turbotab/sample_data/survey_instrument.csv")
    p = AnalysisProject.from_dataframe(df, "survey_instrument.csv")
    p.target, p.task_type = "age", "regression"
    # THE LENS DECIDES WHICH FIGURES EXIST FOR THIS PROJECT, per
    # `DOMAIN_PACKS.md` §08 — it is a recorded answer, not an inference from
    # column names, so the fixture has to record it.
    p.set_lens([packs.SURVEY])
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    return p


# ═══════════ EVERY ONE REACHES A USER ═══════════

def test_all_eight_are_registered():
    for figure_id in CLINICAL + SURVEY + INSTABILITY:
        assert figure_id in figures.REGISTRY, f"{figure_id} is not registered"
    assert len(figures.REGISTRY) == 17, sorted(figures.REGISTRY)


def test_the_clinical_figures_reach_a_user_from_a_fitted_project():
    """`LOOP.md` §05, and `GUIDED-058` is why: a figure registered and
    unreachable is a specification with a passing test that no user can see."""
    bundle = FB.render(_clinical("binary classification"))
    drawn = {row["id"] for row in bundle["admitted"] + bundle["held"]}
    for figure_id in ("decision_curve", "roc"):
        assert figure_id in drawn, (
            f"{figure_id} is registered and no project reaches it; "
            f"served lists were {[r['id'] for r in bundle['admitted']]} / "
            f"{[r['id'] for r in bundle['unavailable']]}")
        row = next(r for r in bundle["admitted"] + bundle["held"]
                   if r["id"] == figure_id)
        assert row["caption"] and row["annotations"] and row["payload"]
        assert row["source"].startswith("research/")


def test_the_survey_four_reach_a_user_from_an_upload():
    bundle = FB.render(_survey())
    drawn = {row["id"] for row in bundle["admitted"] + bundle["held"]}
    for figure_id in SURVEY:
        assert figure_id in drawn, (
            f"{figure_id} did not reach the bundle; unavailable said "
            f"{[(r['id'], r['why'][:60]) for r in bundle['unavailable']]}")


@pytest.mark.parametrize("shape", CLINICAL_BINARY_SHAPES)
def test_the_clinical_figures_pass_their_checklists_on_both_binary_fixtures(shape):
    """`GUIDED-135`. These three were verified only against `leaky_sepsis.csv`,
    whose model separates completely — and two of them are *about* separation.

    The ROC of a separating model is a right angle and its C-statistic is 1.000;
    the decision curve of one dominates treat-all at every threshold. Neither
    figure is wrong there, but neither is being exercised: a checklist scored
    only against a degenerate model has a pass set nobody has tested.
    """
    bundle = FB.render(_clinical(shape))
    for figure_id in CLINICAL:
        row = next((r for r in bundle["admitted"] + bundle["held"]
                    if r["id"] == figure_id), None)
        assert row is not None, f"{figure_id} was not drawn on {shape}"
        failed = [item["id"] for item in row["checklist"] if not item["passed"]]
        assert not failed, (
            f"{figure_id} fails its own checklist on {shape}: {failed}")


@pytest.mark.parametrize("figure_id", sorted(CLINICAL + SURVEY))
def test_each_figure_passes_its_own_checklist(figure_id):
    """A checklist whose items are prose is a style guide; these are callables
    and they run against the real payload."""
    project = (_survey() if figure_id in SURVEY
               else _clinical("binary classification"))
    bundle = FB.render(project)
    row = next((r for r in bundle["admitted"] + bundle["held"]
                if r["id"] == figure_id), None)
    if row is None:
        unavailable = {r["id"]: r["why"] for r in bundle["unavailable"]}
        pytest.fail(f"{figure_id} was not drawn: "
                    f"{unavailable.get(figure_id, 'not offered')}")
    failed = [item["id"] for item in row["checklist"] if not item["passed"]]
    assert not failed, f"{figure_id} fails its own checklist: {failed}"


# ═══════════ THE COMPANION THAT WAS NEVER REGISTERED ═══════════

def test_every_declared_companion_is_a_figure_that_exists():
    """**`GUIDED-128`.** `calibration` named `discrimination` as its companion
    from L34, and no such figure was ever registered or declared pending — so
    `figures.bundle` held it on every project that could otherwise draw it,
    for six loops, silently.

    A companion is a hard admissibility requirement rather than a suggestion,
    which is exactly why an unresolvable one is worse here than anywhere else:
    it does not degrade the figure, it removes it.
    """
    for figure_id, spec in figures.REGISTRY.items():
        for companion in spec.companions:
            assert (companion in figures.REGISTRY
                    or companion in figures.PENDING), (
                f"{figure_id} declares `{companion}` as a companion and it is "
                f"neither registered nor declared pending, so {figure_id} can "
                f"never be admitted by any project")


def test_calibration_is_admissible_now_that_the_roc_exists():
    """The consequence of `GUIDED-128`, driven. Before L40 this figure could
    not be admitted by any project at all."""
    bundle = FB.render(_clinical("binary classification"))
    admitted = {row["id"] for row in bundle["admitted"]}
    assert "calibration" in admitted, (
        f"calibration is still not admitted; held={[r['id'] for r in bundle['held']]}")
    assert "roc" in admitted, "its companion is not drawn either"
    assert figures.REGISTRY["calibration"].companions == ("roc",)


# ═══════════ THE SPEC POINTS THAT ARE NOT PRESENTATION ═══════════

def test_the_decision_curve_uses_a_clinical_threshold_range_and_says_so():
    """§A4.5: *choose the threshold range from the clinical decision, not from
    the data*, and plotting across 0–100% is the named anti-pattern."""
    project = _clinical("binary classification")
    row = next(r for r in FB.render(project)["admitted"]
               if r["id"] == "decision_curve")
    payload = row["payload"]

    low, high = payload["threshold_range"]
    assert (low, high) == F.DCA_THRESHOLDS
    assert high <= 0.9, "the range reaches into thresholds no clinician uses"
    # SAY THE NUMBER, and ask for it to be confirmed — the pack's own
    # instruction rather than a nicety.
    assert f"{low:.0%}" in row["caption"] and f"{high:.0%}" in row["caption"]
    assert "Confirm it brackets" in row["caption"]
    assert "from the decision, not from the data" in row["caption"]


def test_the_decision_curve_can_show_a_model_going_negative():
    """§A4.5's most useful negative finding, and an axis clipped at zero hides
    exactly the curves that produce it."""
    row = next(r for r in FB.render(_clinical("binary classification"))["admitted"]
               if r["id"] == "decision_curve")
    assert row["payload"]["y_lower_bound"] <= -0.05
    assert row["payload"]["treat_none"] == [0.0] * len(row["payload"]["thresholds"])


def test_the_net_benefit_formula_is_vickers_and_elkin():
    """Computed here from the definition and compared, so the figure's numbers
    are checked rather than trusted."""
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=float)
    risks = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.05, 0.3])
    payload = F.decision_curve_payload(y, {"m": risks}, low=0.1, high=0.5,
                                       n_points=5)
    for i, t in enumerate(payload["thresholds"]):
        flagged = risks >= t
        tp = float(np.sum(flagged & (y == 1)))
        fp = float(np.sum(flagged & (y == 0)))
        expected = tp / len(y) - (fp / len(y)) * (t / (1 - t))
        # The payload rounds to 6dp for transport, so the tolerance is 1e-6.
        assert abs(payload["models"]["m"][i] - expected) < 1e-6


def test_the_forest_plot_is_labeled_coefficients_and_uses_a_log_axis():
    """§A4.7's two rules that are correctness rather than taste. The label is
    the critical warning most tools omit, and a linear axis makes OR 0.5 and
    OR 2.0 look asymmetric when they are equal and opposite."""
    payload = F.forest_payload([
        {"name": "age", "estimate": 1.4, "low": 1.1, "high": 1.8},
        {"name": "site", "reference": True},
        {"name": "crp", "estimate": 0.5, "low": 0.3, "high": 0.8},
    ])
    assert payload["title"] == "Model coefficients"
    assert payload["not_titled"] == "Risk factors"
    assert payload["x_scale"] == "log"
    assert payload["reference_line"] == 1.0
    assert payload["sorted_by_significance"] is False
    assert payload["n_reference_rows"] == 1
    assert len(payload["numeric_column"]) == payload["n_rows"]
    assert not [i.id for i in F.FOREST.checklist if not i.check(payload)]

    caption = F.FOREST.caption(payload)
    assert "not causal effects" in caption
    assert "risk factors" not in caption.lower().replace("not_titled", "")
    assert "avoid causal language" in caption


def test_a_linear_axis_for_ratio_measures_fails_the_checklist():
    """The revert probe for the non-negotiable rule, in place."""
    payload = F.forest_payload([{"name": "a", "estimate": 2.0,
                                 "low": 1.1, "high": 3.0}])
    payload["x_scale"] = "linear"
    failed = [i.id for i in F.FOREST.checklist if not i.check(payload)]
    assert "log_axis_for_ratios" in failed


def test_the_roc_is_not_presented_as_the_headline():
    """§A4.4 ranks it below calibration and the decision curve, so it declares
    both as companions and its caption says what the C-statistic cannot."""
    spec = figures.REGISTRY["roc"]
    assert set(spec.companions) == {"calibration", "decision_curve"}

    row = next(r for r in FB.render(_clinical("binary classification"))["admitted"]
               if r["id"] == "roc")
    caption = row["caption"]
    assert "discrimination only" in caption
    assert "nothing about whether the predicted risks are correct" in caption
    assert row["payload"]["axis_labels"] == {"y": "Sensitivity",
                                             "x": "1 − Specificity"}
    for anti in ("accuracy", "F1 ", "Youden"):
        assert anti.lower() not in caption.lower(), (
            f"the ROC caption names {anti!r}, which §A4.4 lists as an "
            f"anti-pattern for a clinical risk model")


# ═══════════ L39-D's ZERO, CLOSED FOR THESE FOUR ═══════════

def test_the_clinical_figures_decline_a_three_class_target():
    """L39-D found that NOTHING anywhere declines a multiclass target. These
    are the first figures built after that finding, so they are the first that
    must — and declining is `not_drawn` with a reason, never a wrong panel."""
    bundle = FB.render(_clinical("three-class classification"))
    offered = {row["id"] for row in bundle["admitted"] + bundle["held"]}
    for figure_id in ("decision_curve", "roc"):
        assert figure_id not in offered, (
            f"{figure_id} was offered on a three-class target; net benefit "
            f"and a C-statistic are both defined for a single predicted risk")
    not_drawn = {row["id"] for row in bundle["not_drawn"]}
    assert {"decision_curve", "roc"} <= not_drawn, (
        "the figures are absent AND unexplained, which is the silence L39-D "
        "found everywhere")


# ═══════════ THE SURVEY FOUR'S SPEC POINTS ═══════════

def test_parallel_analysis_decides_the_factor_count_and_kaiser_is_only_shown():
    """§B5.5 is SETTLED that the eigenvalue>1 rule over-extracts, so it is
    reported for comparison and never used to decide."""
    payload = F.scree_payload(pd.read_csv(
        "turbotab/sample_data/survey_instrument.csv")[
            [f"item_{i:02d}" for i in range(1, 21)]])

    assert payload["n_retained"] >= 1
    assert payload["n_kaiser"] >= payload["n_retained"], (
        "Kaiser retained fewer than parallel analysis, which would make the "
        "over-extraction claim unshowable on this fixture")
    assert payload["n_simulations"] == 100 and payload["percentile"] == 95
    caption = F.SCREE.caption(payload)
    assert "parallel analysis" in caption
    assert "over-extracts" in caption or "the same number here" in caption
    assert "Pearson" in caption, "the correlation method is not in the caption"


def test_the_correlation_matrix_is_never_auto_scaled():
    """§B5.4: auto-scaling makes weak matrices look strong. A palette stretched
    to a maximum of 0.22 renders near-zero correlations in saturated color."""
    payload = F.item_correlations_payload(pd.read_csv(
        "turbotab/sample_data/survey_instrument.csv")[
            [f"item_{i:02d}" for i in range(1, 9)]])
    assert payload["color_domain"] == [-1.0, 1.0]
    assert payload["autoscaled"] is False
    assert payload["triangle"] == "lower" and payload["diagonal"] == "blank"
    caption = F.ITEM_CORRELATIONS.caption(payload)
    assert "Pearson" in caption and "polychoric" in caption, (
        "§B5.4 requires the caption to state which was used")
    assert "attenuates" in caption, "the consequence of Pearson is not stated"


def test_floor_and_ceiling_use_theoretical_limits_not_observed_ones():
    """The distinction is the finding. A sample whose observed maximum is 38 on
    a 0–40 scale has NOBODY at the ceiling; computing against the observed
    maximum would report everyone there."""
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    theoretical = F.floor_ceiling_payload(frame, scale_min=2.0, scale_max=10.0)
    observed = F.floor_ceiling_payload(frame)

    # NOBODY is at the ceiling of a 2–10 scale when the highest total is 8 —
    # and the observed-limit version reports one respondent there. That gap IS
    # the finding: computing against what the sample happened to reach turns
    # "this instrument has headroom" into "this respondent is maxed out".
    assert theoretical["scale_max"] == 10.0
    assert theoretical["at_ceiling"] == 0
    assert observed["scale_max"] == 8.0
    assert observed["at_ceiling"] == 1, (
        "the observed-limit fallback no longer differs, so this test cannot "
        "show why the theoretical limits matter")
    assert theoretical["at_floor"] == 1
    assert theoretical["threshold"] == F.FLOOR_CEILING_THRESHOLD == 0.15
    caption = F.FLOOR_CEILING.caption(theoretical)
    assert "convention from Terwee" in caption
    assert "not an empirically derived constant" in caption, (
        "§B5.3 says to say the 15% is a convention; a threshold presented as "
        "a constant is one nobody argues with")


def test_the_item_panel_shows_shape_the_diverging_bar_cannot():
    """§B5.2's reason for existing beside the diverging chart: bimodality is
    visible here and invisible there, because a diverging bar collapses each
    item to agreement against disagreement."""
    # A deliberately bimodal item: everyone at one end or the other.
    frame = pd.DataFrame({"q1": [1] * 20 + [5] * 20,
                          "q2": [3] * 40})
    payload = F.item_panel_payload(frame, scale=[1, 2, 3, 4, 5])
    by_item = {p["item"]: p for p in payload["panels"]}

    assert by_item["q1"]["bimodal"] is True
    assert by_item["q2"]["bimodal"] is False
    assert payload["shared_axis"] == "percentage"
    caption = F.ITEM_PANEL.caption(payload)
    assert "both extremes" in caption and "diverging bar cannot show" in caption


# ═══════════ THE TWO INSTABILITY PLOTS L38 AND L39 DEFERRED ═══════════

def _resampled():
    from turbotab import instability as I

    project = _clinical("binary classification")
    result = I.run(project, "logreg", b=8, seed=42)
    rows = project.training_rows
    rows = rows[rows["sepsis"].notna()]
    positive = sorted(rows["sepsis"].dropna().unique())[-1]
    return result, (rows["sepsis"] == positive).astype(float)


def test_classification_instability_shows_what_the_prediction_plot_cannot():
    """A patient whose predicted risk moves from 0.18 to 0.22 has barely moved
    on the scatter and has crossed a 20% treatment threshold. Spread in a
    prediction and spread in the DECISION are different quantities."""
    result, _y = _resampled()
    payload = F.classification_instability_payload(result, threshold=0.2)

    assert payload["applicable"] is True
    assert payload["threshold"] == 0.2 and payload["threshold_is_default"]
    assert len(payload["flip_rate"]) == payload["n"]
    assert all(0.0 <= v <= 1.0 for v in payload["flip_rate"])
    assert not [i.id for i in F.CLASSIFICATION_INSTABILITY.checklist
                if not i.check(payload)]
    caption = F.CLASSIFICATION_INSTABILITY.caption(payload)
    assert "20%" in caption and "held-out" in caption
    assert "set from your clinical decision" in caption


def test_decision_curve_instability_reuses_one_net_benefit_formula():
    """The grey curves and the bold one must come from one arithmetic, or the
    figure compares two definitions of net benefit."""
    import inspect

    result, y = _resampled()
    payload = F.decision_curve_instability_payload(result, y)
    assert payload["applicable"] is True
    assert len(payload["curves"]) == result["b_completed"]
    assert payload["y_lower_bound"] <= -0.05
    assert not [i.id for i in F.DECISION_CURVE_INSTABILITY.checklist
                if not i.check(payload)]

    source = inspect.getsource(F.decision_curve_instability_payload)
    assert "decision_curve_payload" in source, (
        "net benefit is recomputed here rather than reused, so the grey "
        "curves and the bold one could drift apart")
    assert "TP" not in source, "the formula is written out a second time"


def test_a_regression_run_gets_neither_and_says_why():
    """Return nothing rather than a plot of something else."""
    from turbotab import instability as I

    project = AnalysisProject.from_dataframe(
        pd.read_csv("turbotab/sample_data/survey_instrument.csv"),
        "survey_instrument.csv")
    project.target, project.task_type = "age", "regression"
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    idx = list(project.df.index)
    project.seal_lockbox(idx[:60], fraction=0.2)
    result = I.run(project, "ridge", b=4, seed=42)

    for payload in (F.classification_instability_payload(result),
                    F.decision_curve_instability_payload(
                        result, np.zeros(result["n"]))):
        assert payload["applicable"] is False
        assert len(payload["because"]) > 40
