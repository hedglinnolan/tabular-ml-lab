"""C1b: stacking NHANES cycles into one sample (`turbotab.core.methods.cycles`).

Every pooled weight is held to NCHS's printed formulas: Table F of the NHANES Analytic Guidelines
2011–2016 ("If sddsrvyr in (7,8,9) then MEC6YR = 1/3 * WTMEC2YR"; "If sddsrvyr in (1,2) then MEC6YR
= 2/3 * WTMEC4YR"; 1999–2002's four-year weight "Provided on the Public-use Data Files") and the
2017–March 2020 file's guidelines (Akinbami et al. 2022: "If SDDSRVYR = 9 then MEC52Y = (2/5.2) •
WTMEC2YR; If SDDSRVYR = 66 then MEC52Y = (3.2/5.2) • WTMECPRP", and the 7.2-year example). The
design over the stack, with strata kept apart by cycle, is held to R survey's ``svydesign`` with
``strata = ~interaction(cycle, SDMVSTRA)``; every refusal is held to its exits, each of which runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.contracts import contract
from turbotab.core.methods import cycles as C
from turbotab.core.tests.acceptance.cycle_fixtures import four_cycles, nhanes_cycle
from turbotab.core.tests.acceptance.survey_r import needs_r, run_r


def _pooled(s: C.Stacked, code: int) -> pd.Series:
    rows = s.frame[s.frame[C.RELEASE] == code]
    return rows[C.WEIGHT]


# ── the pooled weight, against NCHS's printed formulas ───────────────────────


def test_the_prepandemic_file_takes_3_2_over_the_years_stacked_as_nchs_prints_it():
    files = {9: nhanes_cycle(9), 66: nhanes_cycle(66)}
    s = C.stack_cycles(files, weight="WTMEC2YR")
    assert s.total_years == pytest.approx(5.2)
    np.testing.assert_allclose(_pooled(s, 9), (2 / 5.2) * files[9]["WTMEC2YR"].to_numpy(),
                               rtol=1e-15)
    np.testing.assert_allclose(_pooled(s, 66), (3.2 / 5.2) * files[66]["WTMECPRP"].to_numpy(),
                               rtol=1e-15)
    assert s.weights == {"2015–2016": "WTMEC2YR", "2017–March 2020": "WTMECPRP"}
    assert any("`WTMECPRP`, is used" in n for n in s.notes)
    three = {8: nhanes_cycle(8), 9: nhanes_cycle(9), 66: nhanes_cycle(66)}
    s3 = C.stack_cycles(three, weight={8: "WTMEC2YR", 9: "WTMEC2YR", 66: "WTMECPRP"})
    np.testing.assert_allclose(_pooled(s3, 8), (2 / 7.2) * three[8]["WTMEC2YR"].to_numpy(),
                               rtol=1e-15)
    np.testing.assert_allclose(_pooled(s3, 66), (3.2 / 7.2) * three[66]["WTMECPRP"].to_numpy(),
                               rtol=1e-15)
    assert "(3.2/7.2) × `WTMECPRP` on 2017–March 2020" in s3.weight_note


def test_two_year_cycles_divide_by_the_number_of_cycles_as_table_f_prints_it():
    files = {code: nhanes_cycle(code) for code in (7, 8, 9)}
    s = C.stack_cycles(files, weight="WTMEC2YR")
    for code in (7, 8, 9):
        np.testing.assert_allclose(_pooled(s, code), files[code]["WTMEC2YR"].to_numpy() / 3,
                                   rtol=1e-15)
    one = C.stack_cycles({9: files[9]}, weight="WTMEC2YR")
    np.testing.assert_allclose(one.frame[C.WEIGHT], files[9]["WTMEC2YR"], rtol=1e-15)


def test_1999_2004_takes_two_thirds_of_the_four_year_weight_and_a_third_of_the_two_year():
    files = {1: nhanes_cycle(1, four_year=True), 2: nhanes_cycle(2, four_year=True),
             3: nhanes_cycle(3)}
    s = C.stack_cycles(files, weight="WTMEC2YR", four_year="WTMEC4YR")
    for code in (1, 2):
        np.testing.assert_allclose(_pooled(s, code), (2 / 3) * files[code]["WTMEC4YR"].to_numpy(),
                                   rtol=1e-15)
    np.testing.assert_allclose(_pooled(s, 3), files[3]["WTMEC2YR"].to_numpy() / 3, rtol=1e-15)
    assert s.four_year == "WTMEC4YR"
    assert "§3.1.4" in s.sentence()
    # 1999–2002 alone: the four-year weight itself
    two = C.stack_cycles({1: files[1], 2: files[2]}, weight="WTMEC2YR", four_year="WTMEC4YR")
    np.testing.assert_allclose(two.frame[C.WEIGHT],
                               pd.concat([files[1]["WTMEC4YR"], files[2]["WTMEC4YR"]]).to_numpy(),
                               rtol=1e-15)
    # 2001–2002 with later cycles keeps its two-year weight (Table F, 2001–2004)
    later = C.stack_cycles({2: files[2], 3: files[3]}, weight="WTMEC2YR")
    np.testing.assert_allclose(_pooled(later, 2), files[2]["WTMEC2YR"].to_numpy() / 2, rtol=1e-15)


def test_1999_2000_with_another_cycle_and_no_four_year_weight_is_refused_and_its_exits_run():
    files = {1: nhanes_cycle(1, four_year=True), 3: nhanes_cycle(3)}
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles(files, weight="WTMEC2YR")
    assert "different censuses" in str(refused.value)
    exits = refused.value.exits
    assert exits[0]["four_year"] is None and exits[1]["drop_cycles"] == [1]
    s = C.stack_cycles(files, weight="WTMEC2YR", four_year="WTMEC4YR")
    np.testing.assert_allclose(_pooled(s, 1), files[1]["WTMEC4YR"].to_numpy(), rtol=1e-15)
    C.stack_cycles({3: files[3]}, weight="WTMEC2YR")


# ── strata and PSUs stay distinct by cycle, against R ────────────────────────


@needs_r
def test_strata_numbered_alike_in_two_cycles_stay_apart_as_in_r(tmp_path):
    files = {code: nhanes_cycle(code, strata_start=1) for code in (8, 9, 66)}
    s = C.stack_cycles(files, weight="WTMEC2YR")
    assert any("appear in more than one cycle" in c for c in s.concerns)
    design = s.design()
    assert design.n_strata == 24 and design.n_psu == 51
    from turbotab.core.methods.cycle_trends import cycle_trend

    adults = s.frame["BMXBMI"].where(s.frame["RIDAGEYR"] >= 20)
    r = cycle_trend(adults, s.frame[C.CYCLE], design=design)
    ref = run_r("""
f <- read.csv("stacked.csv")
f$w <- ifelse(f$cycle_release == 66, 3.2 / 7.2, 2 / 7.2) * f$cycle_weight
f$y <- ifelse(f$RIDAGEYR >= 20, f$BMXBMI, NA)
kept <- svydesign(ids = ~SDMVPSU, strata = ~interaction(cycle_release, SDMVSTRA), weights = ~w,
                  nest = TRUE, data = f)
merged <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~w, nest = TRUE, data = f)
a <- subset(kept, !is.na(y)); b <- subset(merged, !is.na(y))
out(list(w = max(abs(f$w - f$pooled_weight)), mean = unname(coef(svymean(~y, a))),
         se = as.vector(SE(svymean(~y, a))), merged_se = as.vector(SE(svymean(~y, b))),
         by = unname(SE(svyby(~y, ~cycle_release, a, svymean))), degf = degf(a),
         merged_degf = degf(b)))
""", {"stacked": s.frame}, tmp_path)
    assert ref["w"] < 1e-9
    by_release = {e.cycle: e.se for e in r.estimates}
    np.testing.assert_allclose([by_release[lab] for lab in ("2013–2014", "2015–2016",
                                                            "2017–March 2020")], ref["by"],
                               rtol=1e-10)
    assert r.df == ref["degf"] == 51 - 24 and ref["merged_degf"] == 17 - 8
    # pooled into one estimate over the stack
    ids = adults.dropna().index
    from turbotab.core.models.survey import domain_of, total_variance

    dom = domain_of(ids, design)
    yv = adults.loc[ids].to_numpy()[dom.keep]
    mu = float(dom.weight @ yv) / dom.weight.sum()
    u = np.zeros((design.n_rows, 1))
    u[dom.at, 0] = dom.weight * (yv - mu) / dom.weight.sum()
    se = float(np.sqrt(total_variance(u, design, dom.mask(design)).meat[0, 0]))
    assert mu == pytest.approx(ref["mean"], rel=1e-12)
    assert se == pytest.approx(ref["se"], rel=1e-10)
    assert abs(se - ref["merged_se"]) > 1e-4  # merging the strata would change the answer


# ── refusals, each with exits that run ───────────────────────────────────────


def test_2017_2018_and_the_prepandemic_file_overlap_and_are_refused():
    files = {10: nhanes_cycle(10), 66: nhanes_cycle(66)}
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles(files, weight="WTMEC2YR")
    assert "count them twice" in str(refused.value)
    for exit_ in refused.value.exits:
        kept = {k: v for k, v in files.items() if k not in exit_["drop_cycles"]}
        C.stack_cycles(kept, weight="WTMEC2YR")


def test_a_participant_in_two_cycles_is_refused():
    a, b = nhanes_cycle(8), nhanes_cycle(9)
    b.loc[0, "SEQN"] = a.loc[5, "SEQN"]
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles({8: a, 9: b}, weight="WTMEC2YR")
    assert "counted twice" in str(refused.value)
    assert [e["drop_cycles"] for e in refused.value.exits] == [[9], [8]]


def test_weights_of_different_samples_are_refused():
    a, b = nhanes_cycle(8), nhanes_cycle(9, weight="WTINT2YR")
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles({8: a, 9: b}, weight={8: "WTMEC2YR", 9: "WTINT2YR"})
    assert "different samples" in str(refused.value)
    assert {e["weight_kind"] for e in refused.value.exits} == {"examination", "interview"}


def test_a_file_without_its_design_columns_or_weight_is_refused():
    a, b = nhanes_cycle(8), nhanes_cycle(9).drop(columns=["SDMVPSU"])
    with pytest.raises(C.StackRefused, match="cannot be placed in the survey design"):
        C.stack_cycles({8: a, 9: b}, weight="WTMEC2YR")
    c = nhanes_cycle(9).rename(columns={"WTMEC2YR": "WTMEC2YR_X"})
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles({8: a, 9: c}, weight="WTMEC2YR")
    assert refused.value.exits[0]["weight"] == {9: "WTMEC2YR_X"}
    C.stack_cycles({8: a, 9: c}, weight={8: "WTMEC2YR", **refused.value.exits[0]["weight"]})


def test_a_cycle_whose_length_is_not_known_is_refused_until_it_is_given():
    later = nhanes_cycle(12, weight="WTMEC2YR")
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles({9: nhanes_cycle(9), 12: later}, weight="WTMEC2YR")
    assert "how many years" in str(refused.value) and refused.value.exits[0]["years"] == {12: None}
    s = C.stack_cycles({9: nhanes_cycle(9), 12: later}, weight="WTMEC2YR", years={12: 2.0})
    assert s.total_years == 4.0
    assert C.cycle_of("2015-2016").release == 9 and C.cycle_of("2017–March 2020").release == 66
    assert C.cycle_of("2017-2020 prepandemic").years == 3.2
    assert C.cycle_of(1).midpoint == 2000.0 and C.cycle_of(66).midpoint == pytest.approx(2018.6)
    with pytest.raises(C.StackRefused):
        C.cycle_of("2009-2012")


def test_a_measurement_declared_incompatible_is_refused_and_each_exit_runs():
    files = {code: nhanes_cycle(code) for code in (7, 8, 9)}
    why = {8: "the scale changed model mid-cycle"}
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles(files, weight="WTMEC2YR", variables=["BMXBMI", "RIDAGEYR"],
                       incompatible={"BMXBMI": why})
    says = str(refused.value)
    assert "not measured the same way" in says and "2013–2014: the scale changed" in says
    exits = refused.value.exits
    assert [e["label"] for e in exits] == ["Leave `BMXBMI` out",
                                          "Stack only the cycles measured alike",
                                          "Declare the documented conversion of `BMXBMI`"]
    out = C.stack_cycles(files, weight="WTMEC2YR", variables=["BMXBMI", "RIDAGEYR"],
                         incompatible={"BMXBMI": why}, exclude=exits[0]["exclude"])
    assert out.variables == ("RIDAGEYR",)
    alike = {k: v for k, v in files.items() if k not in exits[1]["drop_cycles"]}
    C.stack_cycles(alike, weight="WTMEC2YR", variables=["BMXBMI"], incompatible={"BMXBMI": why})
    converted = C.stack_cycles(files, weight="WTMEC2YR", variables=["BMXBMI"],
                               incompatible={"BMXBMI": why}, recodes={8: {"BMXBMI": 1.0}})
    assert [f.kind for f in converted.flags] == ["converted"]


def test_a_variable_a_cycle_lacks_is_refused_with_its_likely_earlier_name():
    files = {code: nhanes_cycle(code) for code in (8, 9)}
    files[8]["LBXGLU"] = 100.0
    files[9]["LBDGLU"] = 100.0
    with pytest.raises(C.StackRefused) as refused:
        C.stack_cycles(files, weight="WTMEC2YR", variables=["LBXGLU"])
    assert "2015–2016 has `LBDGLU`: an earlier name?" in str(refused.value)
    first = refused.value.exits[0]
    s = C.stack_cycles(files, weight="WTMEC2YR", variables=["LBXGLU"], renames=first["renames"])
    assert s.frame["LBXGLU"].notna().all()
    assert [f.kind for f in s.flags] == ["renamed"]
    # Without asking for it, it is left out and listed with the same hint.
    whole = C.stack_cycles(files, weight="WTMEC2YR")
    assert "LBXGLU" not in whole.variables
    hinted = {f.column: f for f in whole.left_out}
    assert hinted["LBXGLU"].detail["candidates"] == {"2015–2016": "LBDGLU"}


def test_codes_and_units_that_differ_between_cycles_are_flagged_and_left_unchanged():
    files = {code: nhanes_cycle(code) for code in (7, 8, 9)}
    files[9]["RIAGENDR"] = files[9]["RIAGENDR"] - 1  # 0/1 in one cycle, 1/2 in the others
    files[8]["LBXGLU"] = 5.5 + np.random.default_rng(1).normal(0, 0.6, len(files[8]))  # mmol/L
    files[7]["LBXGLU"] = 99.0 + np.random.default_rng(2).normal(0, 10, len(files[7]))  # mg/dL
    files[9]["LBXGLU"] = 98.0 + np.random.default_rng(3).normal(0, 10, len(files[9]))
    s = C.stack_cycles(files, weight="WTMEC2YR")
    kinds = {f.column: f for f in s.flags}
    assert kinds["RIAGENDR"].kind == "codes_differ"
    assert kinds["RIAGENDR"].detail["codes"]["2015–2016"] == [0, 1]
    assert kinds["LBXGLU"].kind == "scale_differs" and "a change of unit?" in kinds["LBXGLU"].says
    pd.testing.assert_series_equal(s.frame.loc[s.frame[C.RELEASE] == 9, "RIAGENDR"].reset_index(
        drop=True), files[9]["RIAGENDR"].reset_index(drop=True), check_names=False)
    fixed = C.stack_cycles(files, weight="WTMEC2YR",
                           recodes={9: {"RIAGENDR": {0: 1, 1: 2}}, 8: {"LBXGLU": 18.016}})
    assert {f.kind for f in fixed.flags} == {"converted"}


def test_cycles_that_are_not_adjacent_say_what_the_pooled_estimate_averages_over():
    s = C.stack_cycles({5: nhanes_cycle(5), 7: nhanes_cycle(7)}, weight="WTMEC2YR")
    assert any("not adjacent" in c and "2007–2008 and 2011–2012" in c for c in s.concerns)
    assert any("trend tests across cycles" in n for n in s.notes)


# ── the contract ─────────────────────────────────────────────────────────────


# Each relation the contract declares, and the test here that holds the code to it.
RELATION_TESTS = {
    "pooled_weight": "test_the_prepandemic_file_takes_3_2_over_the_years_stacked_as_nchs_prints_it",
    "strata_kept_apart": "test_strata_numbered_alike_in_two_cycles_stay_apart_as_in_r",
    "four_year": "test_1999_2000_with_another_cycle_and_no_four_year_weight_is_refused_and_its_"
                 "exits_run",
    "overlap": "test_2017_2018_and_the_prepandemic_file_overlap_and_are_refused",
    "weight_kind": "test_weights_of_different_samples_are_refused",
    "incompatible": "test_a_measurement_declared_incompatible_is_refused_and_each_exit_runs",
    "absent": "test_a_variable_a_cycle_lacks_is_refused_with_its_likely_earlier_name",
    "flags": "test_codes_and_units_that_differ_between_cycles_are_flagged_and_left_unchanged",
    "trends": "test_strata_numbered_alike_in_two_cycles_stay_apart_as_in_r",
}


def test_the_contract_declares_the_rules_the_code_enforces():
    import importlib

    c = contract("stack_cycles")
    assert c.slot == "ingest" and c.scope == "row_local" and c.package == "C1b"
    assert {r.name for r in c.relations} == set(RELATION_TESTS)
    assert set(RELATION_TESTS.values()) <= {n for n in globals() if n.startswith("test_")}
    for r in c.relations:
        module, fn = r.enforced_by.split(":")
        assert callable(getattr(importlib.import_module(module), fn))
        if r.kind == "conflicts":
            assert r.rung == "refused" and r.exits
    assert any("Akinbami" in src for src in c.sources)
    assert C.stack_sentence().startswith("NHANES cycles were stacked")


def test_the_stacked_rows_are_numbered_anew_in_cycle_order():
    files = four_cycles()
    s = C.stack_cycles({66: files[66], 7: files[7], 9: files[9], 8: files[8]}, weight="WTMEC2YR")
    assert list(s.frame.index) == list(range(len(s.frame)))
    assert s.labels == ["2011–2012", "2013–2014", "2015–2016", "2017–March 2020"]
    assert s.frame[C.MIDPOINT].drop_duplicates().tolist() == pytest.approx([2012, 2014, 2016,
                                                                            2018.6])
    assert s.frame[C.WEIGHT].sum() == pytest.approx(
        sum(files[k][w].sum() * y / 9.2 for k, w, y in ((7, "WTMEC2YR", 2), (8, "WTMEC2YR", 2),
                                                          (9, "WTMEC2YR", 2),
                                                          (66, "WTMECPRP", 3.2))), rel=1e-12)


def test_a_stacked_row_reads_no_other_row_as_the_declared_scope_says():
    """Lockbox constitution §06's test (``contracts.observed_scope``): change the outcome, then
    every other row's measured values and weights, and see whether a row's pooled weight and values
    move. They do not: the scope is row-local, as declared."""
    from turbotab.core.contracts import observed_scope

    files = {code: nhanes_cycle(code, n_per_psu=8) for code in (7, 8, 9)}
    frame = pd.concat(files.values(), ignore_index=True)

    def fit_transform(f: pd.DataFrame, reference: pd.Series, y: np.ndarray) -> pd.DataFrame:
        parts = {code: g for code, g in f.groupby("SDDSRVYR", sort=False)}
        return C.stack_cycles(parts, weight="WTMEC2YR").frame[[C.WEIGHT, "BMXBMI"]]

    y = np.random.default_rng(0).normal(size=len(frame))
    seen = observed_scope(fit_transform, frame, np.zeros(len(frame), dtype=bool), y, row=3,
                          columns=["BMXBMI", "WTMEC2YR"])
    assert seen == contract("stack_cycles").scope == "row_local"
