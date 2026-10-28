"""Every count and claim of tests beside a feature-wise table says what the table tested (the
swarm's tail, package T4a: the verifier's T2V3-1, T2V3-2, T2V3-4, T3-O1, T3-O2, T3V-I2, T2R2-card,
T2R2-fitcount and T2R2-O5, under the orchestrator's rulings (1)–(5)).

Feature-wise regression tests each study factor the model matrix holds as one straight-line
column; a category enters as indicators of its levels, which each test adjusts for, and a study
factor in a declared curve refuses the table. The family's sentences counted every member as
tested regardless: "2 exposures were tested, each in turn; Benjamini–Hochberg q-values control
the false-discovery rate across all 2 …, and every member is shown" beside a table that tested
none, at the fit, at the effects stage, in the caption ("(2 tests)"), on the card and in the
multiplicity preview. Now each counts the members a table tests, names a category as adjusted
for, not tested, and claims no test or method where the table is refused, quoting its refusal.

The constructions are ``estimand_fixtures.cohort`` (400 rows) with a category `grp` (levels a, b,
c; the one-hot step drops the first, so `grp_b` and `grp_c` enter) and a two-level `arm`, each
truth written out here: a sentence from the refusal's own words, an estimate from statsmodels'
least squares on the design written out.

* (1) the family {`grp`, `arm`}, both categories: the table is refused, and no sentence claims a
  test, a count of tests or a multiplicity method; {`fiber` in a 4-knot spline, `sugar`} the same,
  in the formed refusal's words; {`fiber`, `grp`}: 1 test, at the fit as at the effects stage;
  silent for {`fiber`, `sugar`}, every sentence pinned byte for byte.
* (2) the estimand `grp`, `fiber` beside it: the served fit's feature-wise table is refused, as
  the effects stage's is, never showing `fiber`'s test as an adjustment term of `grp`'s model;
  silent for the estimand `fiber`.
* (3) a category's indicators are its own encoding's, never a name's: a covariate `grp_age` in a
  spline (`grp_age'`, `grp_age''`) and a fixed-effects grouping `grp_site` (its intercepts
  `grp_site_s2` … `grp_site_s8`) beside `grp` hold no column of `grp`'s, so the unadjusted model
  holds `fiber` and `grp`'s 2 indicators alone (statsmodels' least squares of glucose on those
  three columns) and the linear family shows `grp`'s indicators as its effects, no term of the
  others among them; each adjusted-for list names `grp` where the model holds its indicators.
* (4) `fib` = 2 × `activity` + 1, nothing left once the covariates are projected out: refused by
  row as not estimable given them, naming them, alone and beside a tested `fiber`.
* (5) values below detection under multiple imputation, the table refused: the sentence states
  the answer and no interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
         "activity": "covariate", "bmi": "covariate"}
MODEL_2 = ["sex_male", "age", "smoking", "activity"]  # the confounders and the precision column
GRP = ["grp_b", "grp_c"]
BH = "Benjamini & Hochberg 1995, J R Stat Soc B 57:289"
NO_FACTOR = ("No study factor to test: feature-wise regression tests the columns with the role of "
             "what you study, one at a time, adjusted for the rest; ")
GRP_AND_ARM = (f"{NO_FACTOR}`grp` and `arm` have that role, but the model matrix holds no column "
               f"of their names.")
GRP_ALONE = f"{NO_FACTOR}`grp` has that role, but the model matrix holds no column of that name."
FORMED_FIBER = ("Feature-wise regression tests each study factor as one straight-line column, one "
                "at a time, adjusted for the rest; `fiber` has that role, and its declared form, a "
                "restricted cubic spline, enters it as its spline terms, not one straight-line "
                "column.")
NONE_OF_TWO = ("Feature-wise regression tested none of the 2 exposures, so no multiplicity method "
               "was applied: ")
BH_ONE = (f"Feature-wise regression tested 1 of the 2 exposures (`fiber`); Benjamini–Hochberg "
          f"q-values control the false-discovery rate across that 1 test ({BH}), and every "
          f"exposure tested is shown, significant or not. `grp` is a category: in each "
          f"feature-wise model it enters as indicators of its levels, not one column, so it is "
          f"adjusted for in each test, not tested.")
ALL_TWO = (f"2 exposures were tested, each in turn; Benjamini–Hochberg q-values control the "
           f"false-discovery rate across all 2 ({BH}), and every member is shown, significant or "
           f"not.")
# The claims a sentence makes only of tests that ran.
CLAIMS = ("were tested", "q-values control", "every member is shown", "tests are stated")


def _frame(**extra) -> pd.DataFrame:
    frame = ef.cohort(N, seed=7)
    rng = np.random.default_rng(11)
    frame["grp"] = rng.choice(["a", "b", "c"], N)
    frame["arm"] = rng.choice(["x", "y"], N)
    frame["glucose"] = frame["glucose"] + np.where(frame["grp"] == "c", 3.0, 0.0)
    frame["sugar"] = 0.3 * frame["fiber"] + rng.normal(0, 3, N)
    frame["glucose"] = frame["glucose"] + 0.2 * frame["sugar"]
    for column, values in extra.items():
        frame[column] = values
    return frame


def _design(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({"(intercept)": np.ones(len(frame))}, index=frame.index)
    for c in columns:
        if c == "sex_male":
            X[c] = (frame["sex"] == "male").astype(float)
        elif c in GRP:
            X[c] = (frame["grp"] == c[-1]).astype(float)
        else:
            X[c] = frame[c].astype(float)
    return X


def _ols(frame: pd.DataFrame, tested: str, adjust: list[str]) -> tuple[float, float]:
    """(estimate, classical standard error) of ``tested`` in glucose ~ tested + adjust."""
    fit = sm.OLS(frame["glucose"].to_numpy(float), _design(frame, [tested, *adjust])).fit()
    return float(fit.params[tested]), float(fit.bse[tested])


def _state(roles: dict[str, str], models: list[str], **kw):
    kw.setdefault("measure", "mean_difference")
    return ef.state(target="glucose", task="regression", roles={**ROLES, **roles}, models=models,
                    **kw)


def _family(roles: dict[str, str], models: list[str], **kw):
    return _state(roles, models, family=True, multiplicity="fdr_bh", **kw)


def _model(fit: dict, family: str = "featurewise") -> dict:
    return next(m for m in fit["models"] if m["family"] == family)


def _sequence(out: dict, family: str = "featurewise") -> dict:
    index = next(i for i, f in enumerate(out["effects"]["families"]) if f["family"] == family)
    return ef.sequence(out["effects"], index)


def _claims_no_test(said: str) -> None:
    for claim in CLAIMS:
        assert claim not in said, (claim, said)


# ── (1) the family's sentences count the members the table tested ────────────


@pytest.mark.parametrize("models", [["linear", "featurewise"], ["featurewise"]])
def test_a_family_of_categories_claims_no_test_at_the_fit_or_the_effects_stage(tmp_path, models):
    """{`grp`, `arm`}: the table holds no column of either's name, so it is refused, and each
    sentence says feature-wise regression tested none, quoting the refusal, where each said both
    were tested with Benjamini–Hochberg q-values and every member shown; no model of the refused
    table is said to adjust for them."""
    out = ef.run(_frame(), tmp_path, _family({"grp": "exposure", "arm": "exposure"}, models))
    assert _model(out["fit_raw"])["inference"]["refused"] == GRP_AND_ARM  # the premise
    said = f"{NONE_OF_TWO}{GRP_AND_ARM}"
    assert out["fit"]["estimand"]["multiplicity"] == said
    assert out["effects"]["multiplicity"] == said
    caption = out["fit"]["estimand"]["caption"]
    assert "none tested, so with no multiplicity method applied" in caption
    assert "tests)" not in caption and "every member shown" not in caption
    methods = out["effects"]["methods"]
    if models[0] == "linear":
        assert methods.endswith(f" {said}")
    else:
        assert methods.startswith(f"No estimate of each exposure is reported: {GRP_AND_ARM}")
    for text in (said, methods, caption):
        _claims_no_test(text)
    for key, model in _sequence(out).items():
        assert model["effects"] is None, key
        assert not {"grp", "arm"} & set(model["adjusted_for"]), key  # no adjustment was made


def test_a_family_with_a_member_in_a_spline_claims_no_test(tmp_path):
    """{`fiber` in a 4-knot spline, `sugar`}: the formed member refuses the table, so the family's
    sentence quotes that refusal and claims no test of `sugar` or of anything."""
    st = _family({"fiber": "exposure", "sugar": "exposure"}, ["featurewise"],
                 exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    out = ef.run(_frame(), tmp_path, st)
    assert _model(out["fit_raw"])["inference"]["refused"] == FORMED_FIBER  # the premise
    said = f"{NONE_OF_TWO}{FORMED_FIBER}"
    assert out["fit"]["estimand"]["multiplicity"] == said
    assert out["effects"]["multiplicity"] == said


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """{`fiber`, `grp`} under feature-wise regression alone."""
    frame = _frame()
    st = _family({"fiber": "exposure", "grp": "exposure"}, ["featurewise"])
    return frame, ef.run(frame, tmp_path_factory.mktemp("mixed"), st)


def test_a_category_among_the_members_is_counted_out_at_the_fit_as_at_the_effects_stage(mixed):
    """The fit's served sentence counts the 1 test, as the effects stage's has since T2, and names
    `grp` as adjusted for, not tested; the caption counts 1 test of the 2 members."""
    _, out = mixed
    assert out["fit"]["estimand"]["multiplicity"] == BH_ONE
    assert out["effects"]["multiplicity"] == BH_ONE
    caption = out["fit"]["estimand"]["caption"]
    assert ("every one tested shown, with Benjamini–Hochberg q-values across the family (1 test of "
            "the 2)") in caption
    assert "(2 tests)" not in caption


def test_each_model_that_holds_the_categorys_indicators_names_it_as_adjusted_for(mixed):
    """The verifier's T2V3-4: the unadjusted model holds `grp`'s indicators, so its row is
    "Without covariates", adjusted for `grp` (its estimate statsmodels' glucose ~ fiber + `grp`'s
    indicators), and Model 2 and Model 3 name `grp` beside the covariates; the methods text says
    so, where it said "unadjusted" and listed the covariates alone."""
    frame, out = mixed
    sequence = _sequence(out)
    crude = sequence["crude"]
    assert crude["adjusted_for"] == ["grp"] and crude["label"] == "Without covariates"
    (row,) = crude["effects"]
    assert row["estimate"] == pytest.approx(_ols(frame, "fiber", GRP)[0], rel=1e-9)
    assert "no other column in the model" not in crude["note"]
    assert sequence["model_2"]["adjusted_for"] == ["age", "sex", "smoking", "activity", "grp"]
    assert sequence["model_3"]["adjusted_for"] == ["age", "sex", "smoking", "activity", "grp",
                                                   "bmi"]
    methods = out["effects"]["methods"]
    assert ("models fit on all 400 analyzed rows: without the covariates, adjusted for `grp` alone; "
            "Model 2, the primary, adjusted for `age`, `sex`, `smoking`, `activity` and `grp`;"
            ) in methods
    assert "unadjusted;" not in methods


def test_a_family_of_numeric_members_keeps_every_sentence(tmp_path):
    """Silent: {`fiber`, `sugar`}, each tested; the family's sentence at the fit and the effects
    stage, the caption's count, the unadjusted model's label and the methods text are as they were,
    byte for byte."""
    out = ef.run(_frame(), tmp_path, _family({"fiber": "exposure", "sugar": "exposure"},
                                              ["linear", "featurewise"]))
    assert out["fit"]["estimand"]["multiplicity"] == ALL_TWO
    assert out["effects"]["multiplicity"] == ALL_TWO
    assert out["effects"]["methods"].endswith(f" {ALL_TWO}")
    assert ("every member shown, with Benjamini–Hochberg q-values across the family (2 tests)"
            in out["fit"]["estimand"]["caption"])
    crude = _sequence(out)["crude"]
    assert (crude["label"], crude["adjusted_for"]) == ("Unadjusted", [])
    assert crude["note"] == "STROBE 16a's unadjusted estimate: no other column in the model."
    assert ": unadjusted; Model 2, the primary, adjusted for" in out["effects"]["methods"]


def test_the_tests_record_reads_each_member_by_why_it_was_not_tested():
    """``estimand.family_tests`` on rows written out: a row with a p-value is a test; a category
    the caption names is adjusted for; a row with no number is not estimable; no row, absent; an
    energy step's name is its member's; the refusal is kept only where nothing was tested."""
    from turbotab.core import estimand as est

    rows = [{"feature": "fiber_adj", "p": 0.01}, {"feature": "fib", "estimate": None, "p": None}]
    tests = est.family_tests(rows, ["fiber", "fib", "grp", "mz_9"], ["fiber", "fib", "grp"],
                             refused="ignored", categories=["grp"])
    assert (tests.tested, tests.categories, tests.unestimable, tests.absent, tests.refused) == (
        ["fiber"], ["grp"], ["fib"], ["mz_9"], None)
    none = est.family_tests([], ["grp", "arm"], refused=GRP_AND_ARM, categories=["grp", "arm"])
    assert (none.tested, none.refused) == ([], GRP_AND_ARM)


def test_the_family_sentence_names_each_kind_of_member_not_tested():
    """The sentence for a table that tested some members: each kind of member it did not test is
    named with why, and the count is of those tested; with every member tested, the estimand's own
    sentence."""
    from turbotab.core import estimand as est

    st = _family({"fiber": "exposure", "fib": "exposure", "grp": "exposure", "mz_9": "exposure"},
                 ["featurewise"])
    members = ["fiber", "fib", "grp", "mz_9"]
    some = est.FamilyTests(members, ["fiber"], ["grp"], ["fib"], ["mz_9"])
    assert est.tested_multiplicity_statement(st, st.estimand, some) == (
        f"Feature-wise regression tested 1 of the 4 exposures (`fiber`); Benjamini–Hochberg "
        f"q-values control the false-discovery rate across that 1 test ({BH}), and every exposure "
        f"tested is shown, significant or not. `grp` is a category: in each feature-wise model it "
        f"enters as indicators of its levels, not one column, so it is adjusted for in each test, "
        f"not tested. `fib` is not estimable given the covariates: nothing of it is left once they "
        f"are projected out, so it is not tested. `mz_9` has no column of its name in the model "
        f"matrix, so it is not tested.")
    every = est.FamilyTests(members, list(members), [], [], [])
    assert est.tested_multiplicity_statement(st, st.estimand, every) == \
        est.multiplicity_statement(st, st.estimand, 4)  # silent


def test_the_fits_entry_names_the_categories_its_caption_says_are_not_tested():
    """The fit's artifact entry records which members enter as indicators only in its caption's
    words (``featurewise.not_tested``): read back, they are the columns the sentence was written
    for, in order, one or several; a caption without the sentence names none."""
    from turbotab.core.models import featurewise as fw

    for entered in ({"grp": ["grp_b", "grp_c"]}, {"g": ["g_b"], "h_x": ["h_x_y"]},
                    {"a": ["a_1"], "b": ["b_1"], "c": ["c_1"]}):
        said = fw.not_tested(entered, [c for cols in entered.values() for c in cols])
        assert fw.categories_said(f"Each of 1 study factors tested. {said} Estimated from all 9 "
                                  f"analyzed rows.") == list(entered)
    assert fw.categories_said("Each of 2 study factors tested on its own with 4 adjustment "
                              "columns.") == []


# ── (1) the card and the multiplicity preview ────────────────────────────────

GENERIC = ("Each of the 2 study factors that enters the model as one column is tested in turn, "
           "adjusted for the covariates but not for the other study factors tested, with its "
           "multiplicity method; every one tested is shown (the feature-wise family).")


def test_the_cards_family_consequence_names_what_feature_wise_regression_will_not_test():
    """The card has no column's values, so it says a category is adjusted for, not tested, and
    names a member the answers record as codes; a member in a declared spline refuses the table,
    so the consequence claims no test and quotes the refusal; and "not for the other study
    factors" is of those tested, a category being adjusted for."""
    from turbotab.core import estimand as est

    def card(**kw):
        return est.estimand_card(_family({"fiber": "exposure", "arm": "exposure"},
                                         ["featurewise"], **kw), "regression")["family"]

    assert card()["consequence"] == (f"{GENERIC} A category among them enters each test as "
                                     f"indicators of its levels: it is adjusted for, not tested.")
    coded = card(amounts={**ef.AMOUNTS, "code_or_count:arm": "code"})["consequence"]
    assert coded == (f"{GENERIC} `arm` is recorded as a category: it enters each test as "
                     f"indicators of its levels, adjusted for, not tested, as any category among "
                     f"them would.")
    formed = card(exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    assert formed["consequence"] == ("None of the 2 study factors would be tested as their forms "
                                     f"stand (the feature-wise family): {FORMED_FIBER}")


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    """The previews fixture's cohort (``preview_fixtures.family_state``, 600 rows) with a
    category `grp` among the family's members beside `fiber` and `supplement`, through the
    split."""
    from turbotab.core.tests.acceptance import preview_fixtures as F
    from turbotab.core.tests.acceptance.preview_harness import Project

    frame = F.estimand_table()
    frame["grp"] = np.random.default_rng(5).choice(["a", "b", "c"], len(frame))
    base = F.family_state()
    roles = {**base.roles, "grp": "exposure"}
    state = base.model_copy(update={"roles": roles, "role_confirmations": dict(roles)})
    project = Project(frame, tmp_path_factory.mktemp("planned"), state, upto=["cohort", "split"])
    yield project
    project.close()


def test_the_preview_counts_the_tests_feature_wise_regression_would_make(planned):
    """{`fiber`, `supplement`, `grp`}: 2 tests, `grp` named as adjusted for, not tested, where the
    preview divided by 3."""
    from turbotab.core.tests.acceptance.preview_harness import view

    result, ctx = planned.preview(d.SetMultiplicity(method="bh"))
    assert ctx.read["multiplicity"] == {"m": 2}
    rel = view(result, "relationship")
    assert rel.caption == ("`2` tests: the i-th smallest p must fall below i × 0.05 / 2; the "
                           "smallest below `0.025`.")
    assert rel.points_after == [(1.0, 0.025), (2.0, 0.05)]
    assert result.note == ("Drawn without the p-values: the method is declared before the results "
                           "are seen. `grp` is a category, adjusted for in each test, not tested.")
    # The columns' types settle it: no value is read (the rows' p-values least of all).
    assert result.basis.startswith("Read from the column names and summaries")


def test_the_preview_draws_no_threshold_where_the_table_would_be_refused(planned):
    """`fiber` in a declared spline: no test would be made, so no threshold is drawn, and the
    caution quotes the refusal with its way forward, a straight line for `fiber`."""
    form = {"fiber": d.ExposureFormSpec(form="spline", knots=4)}
    state = planned.state.model_copy(update={"exposure_forms": form})
    result, _ = planned.preview(d.SetMultiplicity(method="bh"), ctx=planned.context(state=state))
    assert result.views == []
    assert result.caution is not None
    assert result.caution.text == f"No multiplicity method would apply: {FORMED_FIBER}"
    assert [e.decision for e in result.caution.exits] == [
        d.SetExposureForm(column="fiber", form="linear").model_dump(mode="json")]
    assert result.note == ("Feature-wise regression would test none of the `3` study factors, so "
                           "there is no threshold to draw.")


# ── (2) a category estimand at the fit's display ─────────────────────────────


def test_a_category_estimand_refuses_the_fits_feature_wise_display(tmp_path):
    """The estimand `grp`, `fiber` beside it: the fit's table tests `fiber` (its caption true), and
    its served display is refused in the effects stage's words, with no row of `fiber` as an
    effect or an adjustment term."""
    out = ef.run(_frame(), tmp_path, _state({"fiber": "exposure", "grp": "exposure"},
                                            ["linear", "featurewise"], exposure="grp"))
    assert [r["feature"] for r in _model(out["fit_raw"])["coefficients"]] == ["fiber"]  # premise
    served = _model(out["fit"])
    assert served["coefficients"] == [] and served["adjustment_terms"] == []
    assert served["inference"]["refused"] == GRP_ALONE
    assert served["inference"]["caption"] == f"No tests: {GRP_ALONE}"
    assert served["inference"]["exits"] == [{"label": "Choose another model family",
                                             "decision": None}]
    assert all("passes the false-discovery" not in c for c in served["concerns"])
    for key, model in _sequence(out).items():  # the effects stage's, as T2 made them
        assert model["inference"]["refused"] == GRP_ALONE, key


def test_a_numeric_estimand_keeps_the_fits_feature_wise_display(tmp_path):
    """Silent: the estimand `fiber`, `grp` beside it: the served table shows `fiber`'s row, its
    estimate statsmodels' glucose ~ fiber + the covariates + `grp`'s indicators."""
    frame = _frame()
    out = ef.run(frame, tmp_path, _state({"fiber": "exposure", "grp": "exposure"},
                                         ["featurewise"]))
    served = _model(out["fit"])
    (row,) = served["coefficients"]
    assert row["feature"] == "fiber" and served["inference"]["refused"] is None
    assert row["estimate"] == pytest.approx(_ols(frame, "fiber", [*MODEL_2, *GRP])[0], rel=1e-9)


# ── (3) a category's columns are its own encoding's ──────────────────────────


def test_a_longer_predictors_name_never_lends_a_category_its_terms():
    """``estimand.primary_features`` reads a longer predictor's spline terms without their primes:
    `grp_age'` and `grp_age''` are `grp_age`'s; silent for an exposure's own spline terms."""
    from turbotab.core.estimand import primary_features

    assert primary_features(["grp_b", "grp_c", "grp_age", "grp_age'", "grp_age''"], "grp",
                            ["grp", "grp_age"]) == ["grp_b", "grp_c"]
    assert primary_features(["fiber", "fiber'", "fiber''", "fiber_x"], "fiber",
                            ["fiber", "fiber_x"]) == ["fiber", "fiber'", "fiber''"]


def test_a_longer_predictors_spline_terms_are_in_no_unadjusted_model(tmp_path):
    """The verifier's T2V3-2: `grp` beside a covariate `grp_age` in a 4-knot spline. The
    unadjusted feature-wise model holds `fiber` and `grp`'s 2 indicators alone: its `fiber` row is
    statsmodels' glucose ~ fiber + `grp`'s indicators (it was adjusted for `grp_age'` and
    `grp_age''` too); the linear family shows `fiber` and `grp`'s indicators as its effects."""
    frame = _frame().rename(columns={"age": "grp_age"})
    roles = {("grp_age" if c == "age" else c): r for c, r in ROLES.items()}
    answers = {("grp_age" if c == "age" else c): a for c, a in ef.ANSWERS.items()}
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={**roles, "fiber": "exposure", "grp": "exposure"}, answers=answers,
                  models=["linear", "featurewise"], family=True, multiplicity="fdr_bh",
                  exposure_forms={"grp_age": d.ExposureFormSpec(form="spline", knots=4)})
    out = ef.run(frame, tmp_path, st)
    crude = _sequence(out)["crude"]
    (row,) = crude["effects"]
    truth = _ols(frame, "fiber", GRP)
    assert row["estimate"] == pytest.approx(truth[0], rel=1e-9)
    assert row["se"] == pytest.approx(truth[1], rel=1e-9)
    assert crude["inference"]["caption"].startswith(
        "Each of 1 study factors tested on its own with 2 adjustment columns;")
    for key, model in _sequence(out, "linear").items():
        assert sorted(r["feature"] for r in model["effects"]) == ["fiber", *GRP], key
    assert sorted(r["feature"] for r in _model(out["fit"], "linear")["coefficients"]) == \
        ["fiber", *GRP]


def test_a_groupings_intercepts_are_no_categorys_columns(tmp_path):
    """A fixed-effects grouping `grp_site` (8 sites) beside the category `grp`: its intercepts
    `grp_site_s2` … `grp_site_s8` begin with `grp`'s name and are its own, read from the one-hot
    step's encoding. The unadjusted feature-wise model holds `fiber` and `grp`'s 2 indicators (its
    estimate statsmodels' glucose ~ fiber + `grp`'s indicators), not the 7 intercepts; the linear
    family's effects, at the fit and the effects stage, are `fiber` and `grp`'s indicators."""
    rng = np.random.default_rng(13)
    frame = _frame(grp_site=rng.choice([f"s{k}" for k in range(1, 9)], N))
    st = _family({"fiber": "exposure", "grp": "exposure", "grp_site": "identifier"},
                 ["linear", "featurewise"],
                 clusters=d.ClusterSpec(column="grp_site", adjust="fixed_effects"))
    out = ef.run(frame, tmp_path, st)
    crude = _sequence(out)["crude"]
    (row,) = crude["effects"]
    assert row["estimate"] == pytest.approx(_ols(frame, "fiber", GRP)[0], rel=1e-9)
    assert crude["inference"]["caption"].startswith(
        "Each of 1 study factors tested on its own with 2 adjustment columns;")
    for key, model in _sequence(out, "linear").items():
        assert sorted(r["feature"] for r in model["effects"]) == ["fiber", *GRP], key
    assert "grp_site" in _sequence(out, "linear")["model_2"]["adjusted_for"]
    assert sorted(r["feature"] for r in _model(out["fit"], "linear")["coefficients"]) == \
        ["fiber", *GRP]


def test_a_categorys_columns_are_read_from_its_encoder():
    """``featurewise.category_columns`` on fitted encoders: the one-hot step's levels in order,
    the first dropped (`grp`: `grp_b`, `grp_c`; `grp_site`: its 3 sites but the first), and the
    levels step's, blanks a level of their own (`arm`: `arm_y`, `arm_Missing`); a column the
    pipeline does not encode is not listed."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from turbotab.core.models import featurewise as fw
    from turbotab.core.models.pipeline import MissingLevelEncoder

    X = pd.DataFrame({"grp": ["a", "b", "c", "a"], "grp_site": ["s1", "s2", "s3", "s1"],
                      "arm": ["x", "y", None, "x"], "fiber": [1.0, 2.0, 3.0, 4.0]})
    onehot = ColumnTransformer([("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                                         handle_unknown="ignore"),
                                 ["grp", "grp_site"])], remainder="passthrough",
                               verbose_feature_names_out=False)
    pipeline = Pipeline([("levels", MissingLevelEncoder(["arm"])), ("onehot", onehot)]).fit(X)
    assert fw.category_columns(pipeline, ["grp", "arm", "fiber"]) == {
        "arm": ["arm_y", "arm_Missing"], "grp": ["grp_b", "grp_c"]}
    assert fw.category_columns(pipeline)["grp_site"] == ["grp_site_s2", "grp_site_s3"]


# ── (4) a study factor with nothing left after the covariates ────────────────

FIB = ("`fib` is not estimable given the covariates: nothing of it is left once the 4 adjustment "
       "columns (`sex_male`, `age`, `smoking` and `activity`) are projected out, so it is not "
       "tested.")
FIB_AMOUNTS = {**ef.AMOUNTS, "code_or_count:fib": "amount"}  # whole numbers, read as amounts


def test_a_sole_study_factor_with_nothing_left_is_refused_naming_the_covariates(tmp_path):
    """The verifier's T3V-I2: `fib` = 2 × `activity` + 1, the one study factor. The table is
    refused as not estimable given the covariates, naming them (the one-hot step's `sex_male`
    first, then the numbers in the table's order), where its caption said `fib` was tested."""
    frame = _frame()
    frame["fib"] = 2 * frame["activity"] + 1
    out = ef.run(frame, tmp_path, _state({"fib": "exposure"}, ["featurewise"], exposure="fib",
                                         amounts=FIB_AMOUNTS))
    model = _model(out["fit_raw"])
    assert model["coefficients"] == []
    assert model["inference"]["refused"] == FIB
    assert model["inference"]["caption"] == f"No tests: {FIB}"
    assert model["inference"]["exits"] == [
        {"label": "Revisit what each test adjusts for (the adjustment question)", "decision": None},
        {"label": "Choose another model family", "decision": None}]


def test_a_study_factor_with_nothing_left_beside_a_tested_one_is_counted_out(tmp_path):
    """{`fiber`, `fib`}: `fiber` is tested (statsmodels' estimate), `fib`'s row holds no number,
    the caption counts 1 test and names `fib`, and the family's sentence counts 1 test."""
    frame = _frame()
    frame["fib"] = 2 * frame["activity"] + 1
    out = ef.run(frame, tmp_path, _family({"fiber": "exposure", "fib": "exposure"},
                                           ["featurewise"], amounts=FIB_AMOUNTS))
    model = _model(out["fit_raw"])
    rows = {r["feature"]: r for r in model["coefficients"]}
    assert rows["fiber"]["estimate"] == pytest.approx(_ols(frame, "fiber", MODEL_2)[0], rel=1e-9)
    assert (rows["fib"]["estimate"], rows["fib"]["p"], rows["fib"]["q"]) == (None, None, None)
    assert rows["fiber"]["q"] == pytest.approx(rows["fiber"]["p"], rel=1e-12)  # BH over 1 test
    caption = model["inference"]["caption"]
    assert caption.startswith("Each of 1 study factors tested on its own with 4 adjustment "
                              "columns;")
    assert "Benjamini–Hochberg q < 0.05: 1 of 1. " + FIB in caption
    said = out["effects"]["multiplicity"]
    assert said.startswith("Feature-wise regression tested 1 of the 2 exposures (`fiber`);")
    assert said.endswith("`fib` is not estimable given the covariates: nothing of it is left once "
                         "they are projected out, so it is not tested.")
    assert out["fit"]["estimand"]["multiplicity"] == said


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_table_refuses_a_column_in_the_covariates_span_under_either_design(task):
    """On a matrix written out: `x` = 3 `z` − 1 has nothing left once `z` is projected out, under
    the outcome's design and the limma design (where its "difference" was 0 with no variance);
    `w`, independent, is tested beside it, its estimate statsmodels'."""
    from turbotab.core.models import featurewise as fw

    rng = np.random.default_rng(4)
    n = 80
    z = rng.normal(size=n)
    matrix = pd.DataFrame({"x": 3 * z - 1, "w": rng.normal(size=n), "z": z})
    y = (matrix["w"] + rng.normal(size=n) > 0).astype(int) if task == "binary" else \
        0.5 * matrix["w"] + rng.normal(size=n)
    alone = fw.featurewise_table(matrix[["x", "z"]], y, task, ["x"])
    said = ("`x` is not estimable given the covariates: nothing of it is left once the 1 "
            "adjustment column (`z`) is projected out, so it is not tested.")
    assert (alone.rows, alone.info["refused"]) == ([], said)
    both = fw.featurewise_table(matrix, y, task, ["x", "w"])
    rows = {r["feature"]: r for r in both.rows}
    assert rows["x"]["estimate"] is None and rows["x"]["p"] is None
    if task == "regression":
        fit = sm.OLS(y.to_numpy(float), sm.add_constant(matrix[["w", "z"]])).fit()
    else:
        fit = sm.OLS(matrix["w"].to_numpy(), sm.add_constant(
            pd.DataFrame({"z": z, "event": y.to_numpy(float)}))).fit()
    key = "w" if task == "regression" else "event"
    assert rows["w"]["estimate"] == pytest.approx(float(fit.params[key]), rel=1e-9)
    assert both.info["caption"].startswith("Each of 1 study factors tested on its own")
    assert said in both.info["caption"]
    plain = fw.featurewise_table(matrix[["w", "z"]], y, task, ["w"])
    assert "not estimable" not in plain.info["caption"] and plain.info["refused"] is None  # silent


# ── (5) values below detection beside a table refused under multiple imputation ──


def test_the_detection_sentence_states_the_answer_and_no_interval_beside_a_refused_table():
    """The verifier's T3-O2: under multiple imputation a feature-wise table is refused (not pooled)
    and records no missing-values method; the sentence said "A single fill, recorded as a
    limitation: its intervals are too narrow". Now it states the answer recorded and that no
    interval rests on it; a single fill beside a refused table claims no interval either; silent
    beside a table that records its single fill or its imputations."""
    from turbotab.core.decisions import MissingSpec, ProjectState
    from turbotab.core.methods import omics

    design = {"method": "censoring_aware", "columns": ["mz_1", "mz_2"]}
    steps = ["normalize", "detect", "log", "model"]

    def said(strategy, record, acknowledged=True):
        state = ProjectState(lens=["metabolomics"], purpose="inference", missing=MissingSpec(
            strategy=strategy, below_detection="censoring_aware", censored_columns=["mz_1", "mz_2"],
            acknowledged=acknowledged))
        return omics.detection_details(state, "censoring_aware", censored=design,
                                       purpose="inference", missing=record, steps=steps)

    assert said("multiple_imputation", None) == (
        "The missing-values answer is multiple imputation, which draws the values below detection "
        "in 2 features below the limit within each copy; the table was not estimated from the "
        "copies, so no estimate or interval rests on them.")
    filled = ("Values below detection in 2 features were filled after normalization and before the "
              "log, each by its expected value below the limit under a left-censored normal fitted "
              "to the feature's logarithm on the analyzed rows (Lubin et al. 2004).")
    assert said("impute", None) == (f"{filled} A single fill, recorded as a limitation; the table "
                                    f"was not estimated from it, so no interval rests on it.")
    assert said("impute", None, acknowledged=False) == (
        f"{filled} A single fill; the table was not estimated from it, so no interval rests on it.")
    assert said("impute", {"method": "single_fill"}) == (
        f"{filled} A single fill, recorded as a limitation: its intervals are too narrow.")
    assert said("multiple_imputation", {"method": "multiple_imputation", "m": 20}).startswith(
        "Values below detection in 2 features were drawn below the limit from a censored-normal "
        "(Tobit) model within each of the 20 multiple imputations")


def test_the_fits_methods_paragraph_beside_a_table_refused_under_multiple_imputation(tmp_path):
    """Chain 3's inference twin (``test_ms7_chains``), its feature-wise table refused under
    multiple imputation: the paragraph states the answer and no interval, and says no single fill
    was made."""
    from turbotab.core.decisions import ProjectState, Refusal, SetMissing, validate
    from turbotab.core.methods import qc_drift as Q
    from turbotab.core.tests.acceptance.omics_chains import metabolomics_run
    from turbotab.core.tests.acceptance.test_ms7_chains import ctx_of, data
    from turbotab.core.tests.graph_runner import GraphRun

    frame, _ = metabolomics_run()
    path = tmp_path / "run.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, tmp_path / "project")
    try:
        base = {"lens": ["metabolomics"], "target": "case", "purpose": "inference",
                "task": "binary", "event": "1",
                "grain": {"grain": "repeated", "id_column": "participant_id"}, "unit": "row",
                "repeat_kind": {"repeat_kind": "repeats"}}
        feats = [c for c in frame.columns if c.startswith("mz_")]
        roles = {c: "exposure" for c in feats}
        roles.update({"participant_id": "identifier", "sample_id": "excluded",
                      "injection_order": "excluded", "batch": "excluded",
                      "sample_type": "excluded"})
        out = run.run(ProjectState.model_validate({**base, "roles": roles}), upto=["findings"])
        findings = {f["id"]: f for f in data(out["findings"])["findings"]}
        qc = next(o for o in findings[Q.FINDING]["repairs"] if o["key"] == "qc_rlsc_lc")
        scale = next(o for o in findings["omics_scale"]["repairs"] if o["key"] == "pqn_log2")
        applied = ProjectState.model_validate({**base, "roles": roles, "findings": {
            Q.FINDING: {"action": "applied", "option": "qc_rlsc_lc",
                        "params": qc["decision"]["params"]},
            "omics_scale": {"action": "applied", "option": "pqn_log2",
                            "params": scale["decision"]["params"]}}})
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy="multiple_imputation"), ctx_of(applied, frame, out))
        drawn = refused.value.exits[0]["decision"]
        slots = {**applied.model_dump(exclude_none=True),
                 "missing": {k: v for k, v in drawn.items() if k != "kind"},
                 "split": {"holdout": 0.0, "folds": 5, "seed": 0, "validation": "kfold"},
                 "models": ["featurewise"]}
        out = run.run(ProjectState.model_validate(slots), upto=["fit"])
        model = data(out["fit"])["models"][0]
        assert model["inference"]["refused"].startswith(
            "Feature-wise regression is not pooled over multiple imputations here")  # premise
        methods = model["methods"]
        assert "The missing-values answer is multiple imputation, which draws the values below " \
               "detection in " in methods
        assert "the table was not estimated from the copies, so no estimate or interval rests " \
               "on them." in methods
        assert "single fill" not in methods.lower().split("the feature-wise table was refused")[0]
        assert "intervals are too narrow" not in methods
    finally:
        run.close()
