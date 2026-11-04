"""An exposure family's served effect words are true for its fit (PP2's follow-up package
pp2-family-effect-words, the brief's FW1).

Under the linear model an exposure family's members share one fit: each declared model holds them
all, so each member's estimate is adjusted for the others there. The effects stage served each
model's ``adjusted_for`` without the other members, the unadjusted model as "Unadjusted" with the
note "no other column in the model", and the multiplicity sentence "2 exposures were tested, each
in turn; Benjamini–Hochberg q-values control the false-discovery rate across all 2 …" beside rows
with no q-value at all. Under feature-wise regression each member is fitted in its own model, and
those words were true. Now each model of a joint fit names the members it holds, its unadjusted
model is "Without covariates" with a note that says they are in it together, and the multiplicity
sentence says the members were fit together and that their p-values are not adjusted; feature-wise
regression's words stay as they were, byte for byte; with both chosen, each is said under its
name.

The construction is ``test_featurewise_family_counts_its_tests._frame`` (400 rows: `fiber`,
`sugar` = 0.3 × `fiber` + noise with its own effect on `glucose`, a two-level `arm`), each truth
statsmodels' least squares on the design written out here: the members together (and with the
covariates) for the joint fit, each member alone (and with the covariates) for feature-wise
regression. What each model is adjusted for is read off the design that reproduces its rows: its
columns as the raw columns they come from (`sex_male` is `sex`'s), the row's own member aside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.tests import test_featurewise_family_counts_its_tests as FW
from turbotab.core.tests.acceptance import estimand_fixtures as ef

COVARIATES = ["age", "sex", "smoking", "activity"]  # Model 2's adjustment set, as raw columns
FURTHER = ["bmi"]  # Model 3's possible mediator
RAW = {"sex_male": "sex", "arm_y": "arm"}  # a design column's raw column, where its name differs
DESIGN = {"crude": [], "model_2": FW.MODEL_2, "model_3": [*FW.MODEL_2, *FURTHER]}
NUMERIC = {"fiber": "exposure", "sugar": "exposure"}
BH = "Benjamini & Hochberg 1995, J R Stat Soc B 57:289"
ROTHMAN = "Rothman 1990, Epidemiology 1:43"
TOGETHER = ("2 exposures were fit together in each model, so each estimate is adjusted for the "
            "other, and every member is shown.")
JOINT = {
    "fdr_bh": (f"{TOGETHER} Their p-values are not adjusted for multiplicity: Benjamini–Hochberg "
               f"q-values are computed only where each member is tested in its own model."),
    "count_stated": (f"{TOGETHER} Their p-values are not adjusted for multiplicity and are stated "
                     f"as such (customary for a few declared nutrient hypotheses, {ROTHMAN})."),
    "none": f"{TOGETHER} Their p-values have no multiplicity control, recorded as a limitation.",
}
# The sentence feature-wise regression's tests carry, as they were (``FW.ALL_TWO`` for BH).
EACH = {
    "fdr_bh": FW.ALL_TWO,
    "count_stated": ("2 exposures were tested, each in turn; p-values are not adjusted for "
                     "multiplicity, and all 2 tests are stated (customary for a few declared "
                     f"nutrient hypotheses, {ROTHMAN}), with every member shown."),
    "none": ("2 exposures were tested, each in turn, with no multiplicity control, recorded as a "
             "limitation: at p < 0.05 about 0 would pass by chance alone; every member is shown."),
}
JOINT_NOTE = ("STROBE 16a's estimate without the covariates; `fiber` and `sugar`, the members of "
              "the exposure family, are in it together, so each is adjusted for the other.")
UNADJUSTED_NOTE = "STROBE 16a's unadjusted estimate: no other column in the model."
# The claims no sentence beside a joint fit's rows makes: none carries a q-value.
EACH_CLAIMS = ("each in turn", "q-values control", "tests are stated")


def _fit(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """statsmodels' least-squares estimate of each of ``columns`` in glucose ~ ``columns``."""
    fit = sm.OLS(frame["glucose"].to_numpy(float), FW._design(frame, columns)).fit()
    return {c: float(fit.params[c]) for c in columns}


def _columns(frame: pd.DataFrame, members: list[str]) -> list[str]:
    """The members' design columns: a two-level string as its one indicator."""
    out = []
    for m in members:
        if m == "arm":
            frame["arm_y"] = (frame["arm"] == "y").astype(float)
            out.append("arm_y")
        else:
            out.append(m)
    return out


def _raw(columns: list[str]) -> set[str]:
    return {RAW.get(c, c) for c in columns}


def _recorded(models: list[str], method: str):
    """The family {`fiber`, `sugar`} under ``models`` with ``method`` recorded: the estimand's
    method, or (none) the multiplicity slot ``set_multiplicity`` writes."""
    from turbotab.core import decisions as d

    st = FW._state(NUMERIC, models, family=True,
                   multiplicity="count_stated" if method == "count_stated" else "fdr_bh")
    if method == "none":
        st = st.model_copy(update={"multiplicity": d.MultiplicitySpec(method="none")})
    return st


def _sequence(out: dict, family: str) -> dict:
    index = next(i for i, f in enumerate(out["effects"]["families"]) if f["family"] == family)
    return ef.sequence(out["effects"], index)


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    """{`fiber`, `sugar`} under the linear model alone."""
    frame = FW._frame()
    return frame, ef.run(frame, tmp_path_factory.mktemp("joint"), FW._family(NUMERIC, ["linear"]))


@pytest.fixture(scope="module")
def apart(tmp_path_factory):
    """{`fiber`, `sugar`} under feature-wise regression alone."""
    frame = FW._frame()
    return frame, ef.run(frame, tmp_path_factory.mktemp("apart"),
                         FW._family(NUMERIC, ["featurewise"]))


# ── the joint fit ────────────────────────────────────────────────────────────


def test_a_joint_fits_rows_are_statsmodels_one_fit_of_the_members_together(joint):
    """The premise: under the linear model each declared model's rows are statsmodels' one fit of
    glucose on both members (and the model's covariates), never each member's fit alone."""
    frame, out = joint
    sequence = _sequence(out, "linear")
    for key, adjust in DESIGN.items():
        together = _fit(frame, ["fiber", "sugar", *adjust])
        rows = {r["feature"]: r["estimate"] for r in sequence[key]["effects"]}
        assert list(rows) == ["fiber", "sugar"], key
        for m in ("fiber", "sugar"):
            assert rows[m] == pytest.approx(together[m], rel=1e-9), (key, m)
            alone = _fit(frame, [m, *adjust])[m]
            assert abs(rows[m] - alone) > 1e-3 * abs(alone), (key, m)  # the two fits differ


def test_each_model_of_a_joint_fit_names_the_members_it_holds(joint):
    """Each model's adjustment names the members it holds beside the covariates, so each row's own
    member aside, it is the design that reproduces the row (statsmodels': the other member and the
    covariates); the unadjusted model's is the members alone, where it was empty."""
    frame, out = joint
    sequence = _sequence(out, "linear")
    for key, adjust in DESIGN.items():
        adjusted = sequence[key]["adjusted_for"]
        for m in ("fiber", "sugar"):
            design = _raw(["fiber", "sugar", *adjust])
            assert set(adjusted) - {m} == design - {m}, (key, m, adjusted)
    assert sequence["crude"]["adjusted_for"] == ["fiber", "sugar"]
    assert sequence["model_2"]["adjusted_for"] == [*COVARIATES, "fiber", "sugar"]
    assert sequence["model_3"]["adjusted_for"] == [*COVARIATES, "fiber", "sugar", *FURTHER]


def test_a_joint_fits_unadjusted_model_is_said_without_covariates_with_its_members(joint):
    """The unadjusted model holds both members, so it is "Without covariates", and its note says
    they are in it together, never that no other column is in the model; the methods text follows
    its served words, and Model 3 is still further adjusted for `bmi` alone."""
    _, out = joint
    crude = _sequence(out, "linear")["crude"]
    assert crude["label"] == "Without covariates"
    assert crude["note"] == JOINT_NOTE
    methods = out["effects"]["methods"]
    assert ("models fit on all 400 analyzed rows: without the covariates, adjusted for `fiber` and "
            "`sugar` alone; Model 2, the primary, adjusted for `age`, `sex`, `smoking`, "
            "`activity`, `fiber` and `sugar`; Model 3, further adjusted for `bmi`, a possible "
            "mediator") in methods
    assert "unadjusted;" not in methods and "no other column" not in methods


def test_a_joint_fits_multiplicity_sentence_says_the_members_were_fit_together(joint):
    """No row of a joint fit carries a q-value (the premise), so its sentence at the fit and at the
    effects stage, and the methods text it ends, says the members were fit together, each adjusted
    for the other, and that their p-values are not adjusted, where it said each was tested in turn
    with Benjamini–Hochberg q-values controlling the false-discovery rate."""
    _, out = joint
    for key, fit in _sequence(out, "linear").items():
        assert all(r.get("q") is None for r in fit["effects"]), key
    assert all(r.get("q") is None for r in FW._model(out["fit_raw"], "linear")["coefficients"])
    assert out["effects"]["multiplicity"] == JOINT["fdr_bh"]
    assert out["fit"]["estimand"]["multiplicity"] == JOINT["fdr_bh"]
    assert out["effects"]["methods"].endswith(f" {JOINT['fdr_bh']}")
    for claim in EACH_CLAIMS:
        assert claim not in out["effects"]["methods"], claim


def test_a_category_member_of_a_joint_fit_is_named_by_its_column(tmp_path):
    """{`fiber`, `arm`}: the linear model holds `arm` as its indicator `arm_y` beside `fiber`, its
    unadjusted rows statsmodels' glucose ~ arm_y + fiber, and each model names `arm` by its own
    column, as it names `sex`, where it named neither member."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._family({"fiber": "exposure", "arm": "exposure"}, ["linear"]))
    sequence = _sequence(out, "linear")
    columns = _columns(frame, ["arm", "fiber"])
    for key, adjust in DESIGN.items():
        together = _fit(frame, [*columns, *adjust])
        rows = {r["feature"]: r["estimate"] for r in sequence[key]["effects"]}
        assert list(rows) == columns, key
        for c in columns:
            assert rows[c] == pytest.approx(together[c], rel=1e-9), (key, c)
            design = _raw([*columns, *adjust])
            assert set(sequence[key]["adjusted_for"]) - {RAW.get(c, c)} == design - {RAW.get(c, c)}
    assert sequence["crude"]["adjusted_for"] == ["fiber", "arm"]
    assert sequence["crude"]["label"] == "Without covariates"
    assert sequence["crude"]["note"] == (
        "STROBE 16a's estimate without the covariates; `fiber` and `arm`, the members of the "
        "exposure family, are in it together, so each is adjusted for the other.")


# ── one fit per member: silent ───────────────────────────────────────────────


def test_one_fit_per_member_keeps_every_word(apart):
    """Silent: feature-wise regression fits each member in its own model (statsmodels' glucose ~
    the member alone, and with the covariates), so no model names the other member, the unadjusted
    model is "Unadjusted" with no other column in it, every row carries its q-value, and the
    family's sentence is as it was, byte for byte."""
    frame, out = apart
    sequence = _sequence(out, "featurewise")
    for key, adjust in DESIGN.items():
        rows = {r["feature"]: r for r in sequence[key]["effects"]}
        assert list(rows) == ["fiber", "sugar"], key
        for m in ("fiber", "sugar"):
            assert rows[m]["estimate"] == pytest.approx(_fit(frame, [m, *adjust])[m], rel=1e-9)
            assert rows[m]["q"] is not None, (key, m)
            assert set(sequence[key]["adjusted_for"]) == _raw(adjust), key  # no member among them
    crude = sequence["crude"]
    assert (crude["label"], crude["adjusted_for"], crude["note"]) == ("Unadjusted", [],
                                                                     UNADJUSTED_NOTE)
    assert sequence["model_2"]["adjusted_for"] == COVARIATES
    assert out["effects"]["multiplicity"] == EACH["fdr_bh"]
    assert out["fit"]["estimand"]["multiplicity"] == EACH["fdr_bh"]
    assert out["effects"]["methods"].endswith(f" {EACH['fdr_bh']}")


# ── both chosen ──────────────────────────────────────────────────────────────


def test_with_both_fits_chosen_each_sentence_is_said_under_its_name(tmp_path):
    """The linear model and feature-wise regression together: one sentence is served for both
    tables, so each fit's is said under its family's name, the joint fit's never claiming the
    q-values only the feature-wise tests carry; the linear model names the members it holds, and
    feature-wise regression's models none."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._family(NUMERIC, ["linear", "featurewise"]))
    said = (f"Under the feature-wise regression, {EACH['fdr_bh']} Under the linear model, "
            f"{JOINT['fdr_bh']}")
    assert out["effects"]["multiplicity"] == said
    assert out["fit"]["estimand"]["multiplicity"] == said
    linear, featurewise = _sequence(out, "linear"), _sequence(out, "featurewise")
    assert linear["model_2"]["adjusted_for"] == [*COVARIATES, "fiber", "sugar"]
    assert linear["crude"]["label"] == "Without covariates"
    assert featurewise["model_2"]["adjusted_for"] == COVARIATES
    assert (featurewise["crude"]["label"], featurewise["crude"]["adjusted_for"]) == ("Unadjusted", [])
    together = _fit(frame, ["fiber", "sugar", *FW.MODEL_2])
    alone = _fit(frame, ["fiber", *FW.MODEL_2])
    assert linear["model_2"]["effects"][0]["estimate"] == pytest.approx(together["fiber"], rel=1e-9)
    assert featurewise["model_2"]["effects"][0]["estimate"] == pytest.approx(alone["fiber"],
                                                                              rel=1e-9)


# ── the sentence, for each method ────────────────────────────────────────────


@pytest.mark.parametrize("method", ["fdr_bh", "count_stated", "none"])
def test_the_multiplicity_sentence_names_the_fit_under_each_method(method):
    """``estimand.multiplicity_statement`` for each recorded method: a joint fit's names it and
    claims no adjustment it does not make; feature-wise regression's is as it was; a family with
    no coefficient table (the random forest) leaves the sentence as it was, silent."""
    from turbotab.core import estimand as est

    for models, want in ((["linear"], JOINT[method]), (["featurewise"], EACH[method]),
                         (["random_forest"], EACH[method])):
        st = _recorded(models, method)
        assert est.multiplicity_statement(st, st.estimand, 2) == want, models
    st = _recorded(["featurewise", "linear"], method)
    assert est.multiplicity_statement(st, st.estimand, 2) == (
        f"Under the feature-wise regression, {EACH[method]} Under the linear model, "
        f"{JOINT[method]}")


def test_three_members_fit_together_are_each_adjusted_for_the_others():
    """The count and the plural follow the members: 3 fit together, each adjusted for the others."""
    from turbotab.core import estimand as est

    st = FW._state({**NUMERIC, "activity": "exposure"}, ["linear"], family=True,
                   multiplicity="fdr_bh")
    assert est.multiplicity_statement(st, st.estimand, 3).startswith(
        "3 exposures were fit together in each model, so each estimate is adjusted for the "
        "others, and every member is shown. ")


def test_the_frame_holds_what_the_constructions_say():
    """The construction itself: `sugar` moves with `fiber`, so the joint and the separate fits
    differ, and `arm` has two levels."""
    frame = FW._frame()
    assert np.corrcoef(frame["fiber"], frame["sugar"])[0, 1] > 0.3
    assert sorted(frame["arm"].unique()) == ["x", "y"]
