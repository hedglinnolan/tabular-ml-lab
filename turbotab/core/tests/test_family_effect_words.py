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

The repair round (the verifier's B1 and M1): the export's Table 2 (``results/table2.csv`` and
``.md``) wrote each row of a joint model as adjusted for every member, its own among them; now
each row's cell is the design that reproduces its estimate, its own member aside, and a single
exposure's and feature-wise regression's cells are the served columns as they were. Beside
feature-wise regression testing some members or none, the multiplicity sentence said nothing of
the joint fit; now its sentence follows, under its name, as it does with every member tested.
"""
from __future__ import annotations

import csv
import io
import re
from types import SimpleNamespace

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
# A Table 2 that draws a category member as its levels draws none of them (the shell's, the
# verifier's FW-V2-3), so the sentence it footnotes says nothing of every member being shown.
TOGETHER = ("2 exposures were fit together in each model, so each estimate is adjusted for the "
            "other.")
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
JOINT_NOTE = ("STROBE 16a's estimate without the covariates; the study factors `fiber` and `sugar` "
              "are in it together, so each is adjusted for the other.")
UNADJUSTED_NOTE = "STROBE 16a's unadjusted estimate: no other column in the model."
# The claims no sentence beside a joint fit's rows makes: none carries a q-value.
EACH_CLAIMS = ("each in turn", "q-values control", "tests are stated")
# What the methods text says each declared model's estimate is adjusted for, read off the design
# that reproduces its rows, the row's own member aside (the verifier's FW-V1 and FW-V3): under the
# linear model the members together, each adjusted for the other and, from Model 2, the covariates;
# under feature-wise regression each member alone, then with the covariates. Model 3 adds `bmi` to
# Model 2's design under either.
JOINT_WORDS = ("the estimate of each exposure is reported across a declared sequence of models fit "
               "on all 400 analyzed rows: without the covariates, with `fiber` and `sugar` fit "
               "together, each adjusted for the other; Model 2, the primary, adjusted for `age`, "
               "`sex`, `smoking`, `activity` and the other exposure; Model 3, further adjusted for "
               "`bmi`, a possible mediator, so not a total effect.")
APART_WORDS = ("the estimate of each exposure is reported across a declared sequence of models fit "
               "on all 400 analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, "
               "`sex`, `smoking` and `activity`; Model 3, further adjusted for `bmi`, a possible "
               "mediator, so not a total effect.")
# A model said to be adjusted for a member: each member's estimate is never adjusted for itself.
ITSELF = re.compile(r"adjusted for [^;.]*`(?:fiber|sugar)`")


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


def _raw_order(columns: list[str]) -> list[str]:
    """The raw columns of ``columns`` as the served order names them: `sex` (`sex_male`) after
    `age`, as the answers list the covariates (``COVARIATES``), then the rest."""
    raw = _raw(columns)
    return [*(c for c in COVARIATES if c in raw), *(c for c in dict.fromkeys(
        RAW.get(x, x) for x in columns) if c not in COVARIATES)]


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


LABEL = {"crude": "Without covariates", "model_2": "Model 2 (primary)", "model_3": "Model 3"}


def _table2(out: dict) -> tuple[dict[str, dict[str, str]], list[str]]:
    """``results/table2.csv``'s rows by key and ``results/table2.md``'s lines, as the bundle writes
    them from the served effects (``export.bundle.contents``: ``tables.results_tables``, then
    ``to_csv`` and ``markdown``)."""
    from turbotab.core.export import tables

    st = out["state"]
    source = SimpleNamespace(purpose=st.purpose, state=st,
                             artifact=lambda k: out["effects"] if k == "effects" else None)
    (table, *_) = (t for t in tables.results_tables(source) if t.name == "table2")
    rows = {r["key"]: r for r in csv.DictReader(io.StringIO(tables.to_csv(table)))}
    return rows, tables.markdown(table, tables.headers_for(table)).splitlines()


def _md_row(lines: list[str], label: str, cell: str, term: str) -> None:
    """A Markdown row of ``term`` under ``label`` whose cell is ``cell`` (one per family: the
    Markdown draws no family column, so two families' rows can read alike)."""
    assert [x for x in lines if x.startswith(f"| {label} | {cell} | 400 | {term} | ")], (
        label, cell, term)


def _served(fit: dict) -> str:
    """The cell as the export wrote every row before the repair: the model's served columns."""
    return ", ".join(fit["adjusted_for"]) or "nothing"


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
    they are in it together, never that no other column is in the model. The methods text says
    what each model's estimate is adjusted for as the design that reproduces its rows does
    (``test_a_joint_fits_rows_are_statsmodels_one_fit_of_the_members_together``): the members fit
    together, each adjusted for the other, then the covariates and the other exposure, and Model 3
    further adjusted for `bmi` alone. It said each model was adjusted for both members, a row's own
    among them ("adjusted for `fiber` and `sugar` alone"; the verifier's FW-V3)."""
    _, out = joint
    crude = _sequence(out, "linear")["crude"]
    assert crude["label"] == "Without covariates"
    assert crude["note"] == JOINT_NOTE
    assert set(COVARIATES) == _raw(FW.MODEL_2)  # Model 2's design, as the columns it names
    methods = out["effects"]["methods"]
    assert methods.startswith(f"T{JOINT_WORDS[1:]}")
    assert ITSELF.search(methods) is None, ITSELF.search(methods)
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


def test_a_joint_fits_exported_table_2_says_each_row_adjusted_for_the_other_member(joint):
    """The verifier's B1: ``results/table2.csv`` and ``.md`` wrote each row of the linear model as
    adjusted for "fiber, sugar" beside the covariates, so `fiber`'s estimate was said adjusted for
    `fiber`. Each row is statsmodels' one fit of glucose on both members (and the model's
    covariates), so what its estimate is adjusted for is that design without its own member: the
    other member and the covariates, in the served order, in the CSV and the Markdown alike."""
    frame, out = joint
    rows, lines = _table2(out)
    for key, adjust in DESIGN.items():
        together = _fit(frame, ["fiber", "sugar", *adjust])
        for m, other in (("fiber", "sugar"), ("sugar", "fiber")):
            row = rows[f"linear/{key}/{m}"]
            assert float(row["estimate"]) == pytest.approx(together[m], rel=1e-9), (key, m)
            cell = row["adjusted_for"]
            assert set(cell.split(", ")) == _raw(["fiber", "sugar", *adjust]) - {m}, (key, m)
            want = {"crude": [other], "model_2": [*COVARIATES, other],
                    "model_3": [*COVARIATES, other, *FURTHER]}[key]
            assert cell == ", ".join(want), (key, m, cell)
            assert row["model"] == LABEL[key]
            _md_row(lines, LABEL[key], cell, m)
    assert not [x for x in lines if re.search(r"\| [^|]*\bfiber\b[^|]* \| 400 \| fiber \|", x)]


def test_three_members_of_a_joint_fit_are_each_said_adjusted_for_the_others(tmp_path):
    """{`fiber`, `sugar`, `supplement`} under the linear model: each row is statsmodels' one fit of
    glucose on the three (and the model's covariates), so each estimate is adjusted for the 2
    others, and the methods text says the plural where it said each model was adjusted for all
    three, a row's own among them."""
    frame = FW._frame()
    members = ["fiber", "sugar", "supplement"]
    out = ef.run(frame, tmp_path, FW._family(dict.fromkeys(members, "exposure"), ["linear"]))
    sequence = _sequence(out, "linear")
    for key, adjust in DESIGN.items():
        together = _fit(frame, [*members, *adjust])
        rows = {r["feature"]: r["estimate"] for r in sequence[key]["effects"]}
        for m in members:
            assert rows[m] == pytest.approx(together[m], rel=1e-9), (key, m)
    methods = out["effects"]["methods"]
    assert methods.startswith(
        "The estimate of each exposure is reported across a declared sequence of models fit on all "
        "400 analyzed rows: without the covariates, with `fiber`, `sugar` and `supplement` fit "
        "together, each adjusted for the others; Model 2, the primary, adjusted for `age`, `sex`, "
        "`smoking`, `activity` and the other exposures; Model 3, further adjusted for `bmi`, a "
        "possible mediator, so not a total effect."), methods[:400]
    assert re.search(r"adjusted for [^;.]*`(?:fiber|sugar|supplement)`", methods) is None


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
        "STROBE 16a's estimate without the covariates; the study factors `fiber` and `arm` are in "
        "it together, so each is adjusted for the other.")
    # The export's Table 2 (the verifier's B1): `arm_y`'s row is `arm`'s, so its cell is the
    # design without `arm` (the covariates and `fiber`), and `fiber`'s without `fiber`.
    rows, lines = _table2(out)
    for key, adjust in DESIGN.items():
        for c, other in (("arm_y", "fiber"), ("fiber", "arm")):
            cell = rows[f"linear/{key}/{c}"]["adjusted_for"]
            own = RAW.get(c, c)
            assert set(cell.split(", ")) == _raw([*columns, *adjust]) - {own}, (key, c, cell)
            assert cell == ", ".join([*{"crude": [], "model_2": COVARIATES,
                                        "model_3": COVARIATES}[key], other,
                                      *(FURTHER if key == "model_3" else [])]), (key, c, cell)
            _md_row(lines, LABEL[key], cell, c)


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
    # Its methods text says each member alone, then with the covariates, as it did.
    assert out["effects"]["methods"].startswith(f"T{APART_WORDS[1:]}")
    assert out["effects"]["methods"].endswith(f" {EACH['fdr_bh']}")
    # The export's Table 2: each row's cell is its model's served columns, as it was, the design
    # that reproduces the row without its member (the verifier's B1, silent).
    rows, lines = _table2(out)
    for key, adjust in DESIGN.items():
        for m in ("fiber", "sugar"):
            cell = rows[f"featurewise/{key}/{m}"]["adjusted_for"]
            assert cell == _served(sequence[key]) == (", ".join(_raw_order(adjust)) or "nothing")
            _md_row(lines, LABEL[key] if key != "crude" else "Unadjusted", cell, m)


# What feature-wise regression's methods text says of {`fiber`, `grp`, `arm`}: it tests `fiber`
# alone, and each of its models holds the indicators of the two categories, as it did.
CATEGORY_WORDS = ("the estimate of each exposure is reported across a declared sequence of models "
                  "fit on all 400 analyzed rows: without the covariates, adjusted for `grp` and "
                  "`arm` alone; Model 2, the primary, adjusted for `age`, `sex`, `smoking`, "
                  "`activity`, `grp` and `arm`; Model 3, further adjusted for `bmi`, a possible "
                  "mediator, so not a total effect.")
# The linear model's for the same family: the three fit together in each model.
CATEGORY_JOINT_WORDS = ("the estimate of each exposure is reported across a declared sequence of "
                        "models fit on all 400 analyzed rows: without the covariates, with `fiber`, "
                        "`grp` and `arm` fit together, each adjusted for the others; Model 2, the "
                        "primary, adjusted for `age`, `sex`, `smoking`, `activity` and the other "
                        "exposures; Model 3, further adjusted for `bmi`, a possible mediator, so not "
                        "a total effect.")


@pytest.mark.parametrize("models", [["featurewise"], ["linear", "featurewise"],
                                    ["featurewise", "linear"]])
def test_the_categories_feature_wise_regression_holds_are_never_said_fit_together(tmp_path,
                                                                                  models):
    """Silent (the verifier's FW-V2-1): {`fiber`, `grp`, `arm`} under feature-wise regression,
    alone and beside the linear model in either order. Feature-wise regression tests `fiber` alone:
    each of its rows is statsmodels' glucose ~ `fiber` + the indicators of `grp` and `arm` (and the
    model's covariates), never `fiber` without them, so each model names both categories among the
    columns it is adjusted for, two members of the family. Its methods text says each model was
    adjusted for them alone, as it did, never that they were fit together, each adjusted for the
    other: only `effects._together`'s guard that a family testing each member in its own model
    holds none together keeps the joint fit's words off it (with {`fiber`, `sugar`} its models
    name no member, so the guard never runs). Beside it the linear model fits the three together
    (statsmodels' one fit of glucose on `fiber` and both categories' indicators), and its text says
    so."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._family(
        {"fiber": "exposure", "grp": "exposure", "arm": "exposure"}, models))
    held = [*FW.GRP, *_columns(frame, ["arm"])]  # grp_b, grp_c and arm_y
    featurewise = _sequence(out, "featurewise")
    for key, adjust in DESIGN.items():
        (row,) = featurewise[key]["effects"]
        want = _fit(frame, ["fiber", *held, *adjust])["fiber"]
        assert row["feature"] == "fiber" and row["estimate"] == pytest.approx(want, rel=1e-9), key
        alone = _fit(frame, ["fiber", *adjust])["fiber"]
        assert abs(row["estimate"] - alone) > 1e-3 * abs(alone), key  # the categories are held
        assert set(featurewise[key]["adjusted_for"]) == _raw(adjust) | {"grp", "arm"}, key
    assert featurewise["crude"]["label"] == "Without covariates"
    methods = out["effects"]["methods"]
    if models == ["featurewise"]:
        assert methods.startswith(f"T{CATEGORY_WORDS[1:]}"), methods[:400]
    else:
        assert f"Under the feature-wise regression, {CATEGORY_WORDS}" in methods, methods
        linear = _sequence(out, "linear")
        for key, adjust in DESIGN.items():
            together = _fit(frame, [*held, "fiber", *adjust])
            rows = {r["feature"]: r["estimate"] for r in linear[key]["effects"]}
            assert list(rows) == [*held, "fiber"], key
            for c in rows:
                assert rows[c] == pytest.approx(together[c], rel=1e-9), (key, c)
        assert f"Under the linear model, {CATEGORY_JOINT_WORDS}" in methods, methods
    assert "with `grp` and `arm` fit together" not in methods
    # The export's Table 2 (the verifier's B1). Each feature-wise model names two members, `grp`
    # and `arm`, and `fiber`'s row is of neither, so its cell is the served columns, as it was: the
    # design that reproduces it without `fiber` (silent). Each row of the linear model is the
    # design without the row's own member: `grp_b`'s and `grp_c`'s are `grp`'s, `arm_y`'s `arm`'s.
    rows, lines = _table2(out)
    own = {"grp_b": "grp", "grp_c": "grp", "arm_y": "arm", "fiber": "fiber"}
    for key, adjust in DESIGN.items():
        cell = rows[f"featurewise/{key}/fiber"]["adjusted_for"]
        assert cell == _served(featurewise[key]), key
        assert set(cell.split(", ")) == _raw(adjust) | {"grp", "arm"}, key
        if "linear" not in models:
            continue
        for c in [*held, "fiber"]:
            cell = rows[f"linear/{key}/{c}"]["adjusted_for"]
            design = {own.get(x, RAW.get(x, x)) for x in [*held, "fiber", *adjust]}
            assert set(cell.split(", ")) == design - {own[c]}, (key, c, cell)
            assert len(cell.split(", ")) == len(design) - 1, (key, c, cell)  # each named once
            _md_row(lines, LABEL[key], cell, c)


def test_a_single_exposures_exported_table_2_keeps_every_cell(tmp_path):
    """Silent (the verifier's B1): the single exposure `fiber` under the linear model and
    feature-wise regression. Each model's rows are statsmodels' glucose ~ `fiber` (and the
    model's covariates) under either, its served columns name no member, and each row's cell in
    ``results/table2.csv`` and ``.md`` is those columns, as it was: "nothing" without the
    covariates, then the covariates."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._state({"fiber": "exposure"}, ["linear", "featurewise"],
                                            exposure="fiber"))
    rows, lines = _table2(out)
    for family in ("linear", "featurewise"):
        sequence = _sequence(out, family)
        for key, adjust in DESIGN.items():
            row = rows[f"{family}/{key}/fiber"]
            assert float(row["estimate"]) == pytest.approx(_fit(frame, ["fiber", *adjust])["fiber"],
                                                           rel=1e-9), (family, key)
            cell = row["adjusted_for"]
            assert cell == _served(sequence[key]) == (", ".join(_raw_order(adjust)) or "nothing")
            _md_row(lines, sequence[key]["label"], cell, "fiber")


# ── both chosen ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("models", [["linear", "featurewise"], ["featurewise", "linear"]])
def test_with_both_fits_chosen_each_sentence_is_said_under_its_name(tmp_path, models):
    """The linear model and feature-wise regression together, in either order: one sentence is
    served for both tables, so each fit's is said under its family's name, the joint fit's never
    claiming the q-values only the feature-wise tests carry; the linear model names the members it
    holds, and feature-wise regression's models none. Both report from every model, and the
    methods text said the first family's sequence alone, for both: under [linear, feature-wise]
    that each feature-wise estimate was adjusted for both members, under [feature-wise, linear]
    that the linear model's unadjusted model was unadjusted (the verifier's FW-V1). Now it says
    each family's sequence under its name, each as the design that reproduces its rows says:
    statsmodels' fit of the members together, and of each member alone."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._family(NUMERIC, models))
    said = (f"Under the feature-wise regression, {EACH['fdr_bh']} Under the linear model, "
            f"{JOINT['fdr_bh']}")
    assert out["effects"]["multiplicity"] == said
    assert out["fit"]["estimand"]["multiplicity"] == said
    linear, featurewise = _sequence(out, "linear"), _sequence(out, "featurewise")
    assert linear["model_2"]["adjusted_for"] == [*COVARIATES, "fiber", "sugar"]
    assert linear["crude"]["label"] == "Without covariates"
    assert featurewise["model_2"]["adjusted_for"] == COVARIATES
    assert (featurewise["crude"]["label"], featurewise["crude"]["adjusted_for"]) == ("Unadjusted", [])
    for key, adjust in DESIGN.items():
        together = _fit(frame, ["fiber", "sugar", *adjust])
        for m in ("fiber", "sugar"):
            joint_row = next(r for r in linear[key]["effects"] if r["feature"] == m)
            apart_row = next(r for r in featurewise[key]["effects"] if r["feature"] == m)
            assert joint_row["estimate"] == pytest.approx(together[m], rel=1e-9), (key, m)
            assert apart_row["estimate"] == pytest.approx(_fit(frame, [m, *adjust])[m],
                                                          rel=1e-9), (key, m)
    words = {"linear": f"Under the linear model, {JOINT_WORDS}",
             "featurewise": f"Under the feature-wise regression, {APART_WORDS}"}
    methods = out["effects"]["methods"]
    first, second = (words[m] for m in models)
    assert methods.startswith(first), methods[:400]
    assert methods.index(f" {second}") > len(first)
    assert methods.endswith(f" {said}")
    assert ITSELF.search(methods) is None, ITSELF.search(methods)


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


@pytest.mark.parametrize("models", [["featurewise", "linear"], ["linear", "featurewise"]])
def test_beside_feature_wise_regression_testing_some_members_the_joint_fit_is_said_too(tmp_path,
                                                                                     models):
    """The verifier's M1: {`fiber`, `grp`}, in either order. Feature-wise regression tests `fiber`
    alone (statsmodels' glucose ~ `fiber` and `grp`'s indicators, and the model's covariates),
    `grp` a category it adjusts for; the linear model fits both together (statsmodels' one fit of
    glucose on `grp`'s indicators and `fiber`), its rows with p-values and no q-value. The
    sentence at the fit and at the effects stage, and the methods text it ends, said feature-wise
    regression's 1 test alone, nothing of the joint fit; now the joint fit's sentence follows
    under its name, as it does with every member tested."""
    frame = FW._frame()
    out = ef.run(frame, tmp_path, FW._family({"fiber": "exposure", "grp": "exposure"}, models))
    featurewise, linear = _sequence(out, "featurewise"), _sequence(out, "linear")
    for key, adjust in DESIGN.items():
        (row,) = featurewise[key]["effects"]
        want = _fit(frame, ["fiber", *FW.GRP, *adjust])["fiber"]
        assert row["feature"] == "fiber" and row["estimate"] == pytest.approx(want, rel=1e-9)
        assert row["q"] is not None, key
        together = _fit(frame, [*FW.GRP, "fiber", *adjust])
        rows = {r["feature"]: r for r in linear[key]["effects"]}
        assert sorted(rows) == sorted([*FW.GRP, "fiber"]), key
        for c, r in rows.items():
            assert r["estimate"] == pytest.approx(together[c], rel=1e-9), (key, c)
            assert r["p"] is not None and r["q"] is None, (key, c)
    said = f"{FW.BH_ONE} Under the linear model, {JOINT['fdr_bh']}"
    assert out["fit"]["estimand"]["multiplicity"] == said
    assert out["effects"]["multiplicity"] == said
    assert out["effects"]["methods"].endswith(f" {said}")


# Feature-wise regression's sentence where it tested `fiber` and not `sugar`, as it was, under
# each method (``estimand.tested_multiplicity_statement``).
NOT_SUGAR = ("`sugar` is not estimable given the covariates: nothing of it is left once they are "
             "projected out, so it is not tested.")
SOME = {
    "fdr_bh": (f"Feature-wise regression tested 1 of the 2 exposures (`fiber`); Benjamini–Hochberg "
               f"q-values control the false-discovery rate across that 1 test ({BH}), and every "
               f"exposure tested is shown, significant or not. {NOT_SUGAR}"),
    "count_stated": (f"Feature-wise regression tested 1 of the 2 exposures (`fiber`); p-values are "
                     f"not adjusted for multiplicity, and that 1 test is stated (customary for a "
                     f"few declared nutrient hypotheses, {ROTHMAN}), with every exposure tested "
                     f"shown. {NOT_SUGAR}"),
    "none": (f"Feature-wise regression tested 1 of the 2 exposures (`fiber`), with no multiplicity "
             f"control, recorded as a limitation: at p < 0.05 about 0 would pass by chance alone; "
             f"every exposure tested is shown. {NOT_SUGAR}"),
}
NONE_TESTED = ("Feature-wise regression tested none of the 2 exposures, so no multiplicity method "
               "was applied: `fiber` and `sugar` have no column of their names in the model "
               "matrix, so none is tested.")


@pytest.mark.parametrize("method", ["fdr_bh", "count_stated", "none"])
def test_the_sentence_of_some_members_tested_or_none_names_the_joint_fit_beside_it(method):
    """``estimand.tested_multiplicity_statement`` where feature-wise regression tested 1 of the 2
    members, or none, under each method: beside the linear model, in either order, the joint fit's
    sentence follows feature-wise regression's under its name (the verifier's M1); with no joint
    model beside it (feature-wise regression alone, or beside the random forest, which has no
    coefficient table), the sentence is as it was, byte for byte."""
    from turbotab.core import estimand as est

    members = ["fiber", "sugar"]
    for tests, alone in ((est.FamilyTests(members, ["fiber"], [], ["sugar"], []), SOME[method]),
                         (est.FamilyTests(members, [], [], [], members), NONE_TESTED)):
        for models in (["featurewise"], ["featurewise", "random_forest"]):
            st = _recorded(models, method)
            assert est.tested_multiplicity_statement(st, st.estimand, tests) == alone, models
        for models in (["featurewise", "linear"], ["linear", "featurewise"]):
            st = _recorded(models, method)
            assert est.tested_multiplicity_statement(st, st.estimand, tests) == (
                f"{alone} Under the linear model, {JOINT[method]}"), models


def test_the_sequence_sentence_says_the_joint_fit_after_every_feature_wise_model():
    """``estimand.sequence_multiplicity_statement`` where the unadjusted model tested other members
    than Model 2: the sentence is of Model 2, then each such model's tests, then, beside the
    linear model, the joint fit's sentence under its name (the verifier's M1), where it said
    nothing of it with some members tested, and with every member tested in Model 2 said it
    between Model 2's and the unadjusted model's; with no joint model, as it was."""
    from turbotab.core import estimand as est

    members = ["fiber", "fib"]
    some = est.FamilyTests(members, ["fiber"], [], ["fib"], [])
    every = est.FamilyTests(members, members, [], [], [])
    unadjusted = [("The unadjusted model", every)]
    fewer = [("The unadjusted model", some)]
    st = _recorded(["featurewise"], "fdr_bh")
    assert est.sequence_multiplicity_statement(st, st.estimand, some, unadjusted) == FW.FIB_FAMILY
    one = ("The unadjusted model tested 1 of the 2 exposures (`fiber`), with Benjamini–Hochberg "
           "q-values across that 1 test.")
    assert est.sequence_multiplicity_statement(st, st.estimand, every, fewer) == (
        f"In Model 2, the primary, {EACH['fdr_bh']} {one}")
    for models in (["featurewise", "linear"], ["linear", "featurewise"]):
        st = _recorded(models, "fdr_bh")
        assert est.sequence_multiplicity_statement(st, st.estimand, some, unadjusted) == (
            f"{FW.FIB_FAMILY} Under the linear model, {JOINT['fdr_bh']}"), models
        assert est.sequence_multiplicity_statement(st, st.estimand, every, fewer) == (
            f"In Model 2, the primary, under the feature-wise regression, {EACH['fdr_bh']} {one} "
            f"Under the linear model, {JOINT['fdr_bh']}"), models


def test_three_members_fit_together_are_each_adjusted_for_the_others():
    """The count and the plural follow the members: 3 fit together, each adjusted for the others."""
    from turbotab.core import estimand as est

    st = FW._state({**NUMERIC, "activity": "exposure"}, ["linear"], family=True,
                   multiplicity="fdr_bh")
    assert est.multiplicity_statement(st, st.estimand, 3).startswith(
        "3 exposures were fit together in each model, so each estimate is adjusted for the "
        "others. ")


def test_the_frame_holds_what_the_constructions_say():
    """The construction itself: `sugar` moves with `fiber`, so the joint and the separate fits
    differ, and `arm` has two levels."""
    frame = FW._frame()
    assert np.corrcoef(frame["fiber"], frame["sugar"])[0, 1] > 0.3
    assert sorted(frame["arm"].unique()) == ["x", "y"]
