"""SMC-FCS's Cox outcome draw refused in words true of the model that fails (package B's fallback
pass: the verifier's V3-1 and V3-2).

A Cox fit of the substantive model that does not converge has no maximum to center the draw on,
and the draw is refused (:func:`smcfcs._cox_draw`). Under the population survey answer that model
holds the survey design's terms beside the analysis model's (``missing.engine_substantive``), and
where one stratum has no event its indicator orders the risk sets: the imputation's model has no
maximum, while the analysis model, which holds no design term, has one. The refusal said "The Cox
model's coefficients cannot be drawn ... its partial likelihood has no maximum", false of the
researcher's model, and named no column, so no way forward but complete cases was offered.

Now the words name the design's terms (``Substantive.terms``). Where the analysis model's own terms,
fit alone to the same completed copy, have a maximum, they name the design term along the ordering
and say the imputation's model is the one that fails, and how the design is left out. Where the
analysis model's own terms fail, they name those terms as the analysis table's
``survival.no_maximum`` does, and the refusal's ``columns`` hold the inputs the terms are made from,
so the stage offers to leave them out. A design with no named terms keeps the unnamed words
(``test_smcfcs_logistic_cox_draw``).

**Truth.** By construction: a stratum, or a group, none of whose rows has the event orders the risk
sets along its indicator, the rows with the event lowest at every event time, so its hazard ratio
runs to zero; the other columns are drawn with events at every value. The silent side is the same
data with that stratum's events kept, and the same data without the design's terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods.imputation import ImputationRefused
from turbotab.core.methods.smcfcs import Outcome, Substantive, Term, Variable, _draw_outcome, impute
from turbotab.core.tests import test_smcfcs_aliased_design as aliased

SAMPLE = "Leaving the design out is the way forward."  # a design term's own leave-out words


def _strata(seed: int, quiet: bool = True, group: bool = False
            ) -> tuple[pd.DataFrame, Outcome, int]:
    """The verifier's construction (``vB3_stratum.py``): 1,500 rows in six strata, x (30% blank)
    and w with log hazard ratios 0.5 and 0.3, censoring uniform on (0, 10). ``quiet``: stratum 5
    has no event (every row of it censored). ``group``: an analysis column g (a fifth of the rows)
    none of whose rows has the event. Returns the data, the outcome and the events."""
    rng = np.random.default_rng(seed)
    n = 1500
    stratum = rng.integers(0, 6, n)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    t = rng.exponential(1 / (0.05 * np.exp(0.5 * x + 0.3 * w)))
    c = rng.uniform(0, 10, n)
    recorded = rng.random(n) < 0.7
    g = (rng.random(n) < 0.2).astype(float)  # drawn last: the verifier's draws are unchanged
    event = (t <= c) & ~((stratum == 5) & quiet) & ~((g == 1) & group)
    out = Outcome("cox", time=np.minimum(t, c), event=event.astype(float), entry=np.zeros(n))
    data = pd.DataFrame({"x": np.where(recorded, x, np.nan), "w": w, "g": g,
                         "__stratum": [f"stratum {s}" for s in stratum]})
    return (data if group else data.drop(columns=["g"])), out, int(event.sum())


def _substantive(out: Outcome, design: bool = True, group: bool = False) -> Substantive:
    """x, w (and g) as the analysis model's terms, then stratum 1 … 5's indicators as the survey
    design's (``design``), named as ``missing.engine_substantive`` names them."""
    analysis = ["x", "w", "g"] if group else ["x", "w"]

    def matrix(f: pd.DataFrame) -> np.ndarray:
        s = f["__stratum"].to_numpy()
        own = [f[c].to_numpy(dtype=float) for c in analysis]
        return np.column_stack([*own, *[(s == f"stratum {k}").astype(float)
                                        for k in range(1, 6) if design]])

    def terms() -> list[Term]:
        own = [Term(f"`{c}`", column=c) for c in analysis]
        return own + [Term(f"the indicator of stratum {k} of `STRATUM`", design=True,
                           group=f"stratum {k}", leave_out=SAMPLE) for k in range(1, 6) if design]

    return Substantive(out, matrix, terms=terms)


def _impute(data: pd.DataFrame, sub: Substantive) -> object:
    return impute(data, [Variable("x", "numeric")], mode="smcfcs", substantive=sub, m=2,
                  iterations=4, seed=1, kinds={"__stratum": "categorical"})


@pytest.mark.parametrize("seed", range(3))
def test_a_stratum_with_no_event_is_refused_as_the_imputation_model_s(seed):
    """Stratum 5 has no event: the imputation's model, which holds stratum 5's indicator, has no
    maximum along it. The words name that indicator and say the analysis model has a maximum on
    the same copy, and end with the design term's own leave-out words. ``columns`` holds no input
    (leaving out x or w would not help), and ``design_terms`` the term. At the base commit the words
    were the analysis model's ("The Cox model's coefficients cannot be drawn ...") and named no
    column."""
    data, out, _ = _strata(seed)
    with pytest.raises(ImputationRefused) as refused:
        _impute(data, _substantive(out))
    assert str(refused.value) == (
        "The Cox model the imputations are drawn from has no maximum on a completed copy of the "
        "data, so its coefficients cannot be drawn: to carry the survey design into the copies, "
        "it holds the indicator of stratum 5 of `STRATUM` beside the analysis model's terms, and "
        "at every event time the rows with the event have the lowest value of it among the rows "
        "at risk (as when no row of stratum 5 has the event), so its hazard ratio runs to zero. "
        "The analysis model itself is not the one that fails: without the survey design's terms, "
        "its partial likelihood has a maximum on the same copy. " + SAMPLE)
    assert not getattr(refused.value, "columns", None)
    assert refused.value.design_terms == ["the indicator of stratum 5 of `STRATUM`"]


@pytest.mark.parametrize("seed", range(3))
def test_the_same_data_impute_where_the_stratum_has_events_or_no_design_term(seed):
    """The guard's silent side: stratum 5's events kept, with the design's terms in; and stratum 5
    without events, with the analysis model's terms only (the copies the sample answer draws). Both
    impute, every blank x drawn to a finite value."""
    data, out, _ = _strata(seed, quiet=False)
    kept = _impute(data, _substantive(out))
    data, out, _ = _strata(seed)
    alone = _impute(data, _substantive(out, design=False))
    blank = data["x"].isna().to_numpy()
    for res in (kept, alone):
        assert len(res.frames) == 2
        assert all(np.isfinite(f["x"].to_numpy()[blank]).all() for f in res.frames)


def test_an_ordering_along_an_analysis_column_names_it_and_its_input():
    """The group construction of ``test_a_cox_model_with_no_maximum_is_refused_in_words`` with
    its columns named: the group's indicator (made from the input ``group``) orders the risk sets,
    its rows lowest, so the words name it and its hazard ratio's direction, as the analysis table's
    ``survival.no_maximum`` does, and ``columns`` is the input, which the stage offers to leave
    out. At the base commit the words named no column and ``columns`` was unset."""
    rng = np.random.default_rng(4)
    n = 400
    group = (rng.random(n) < 0.25).astype(float)
    x = rng.normal(size=n)
    time = rng.exponential(1.0 / (0.1 * np.exp(0.3 * x)))
    censor = rng.uniform(0.0, 20.0, n)
    event = ((time <= censor) & (group == 0)).astype(float)
    out = Outcome("cox", time=np.minimum(time, censor), event=event, entry=np.zeros(n))
    names = [Term("`x`", column="x"), Term("`group_1`", column="group")]
    with pytest.raises(ImputationRefused) as refused:
        _draw_outcome(np.column_stack([x, group]), out, np.random.default_rng(0), lambda: names)
    assert str(refused.value) == (
        "The Cox model's coefficients cannot be drawn for the imputations compatible with it, since "
        "its partial likelihood has no maximum for `group_1` on a completed copy of the data: at "
        "every event time the rows with the event have the lowest `group_1` of the rows at risk "
        "(as when a group has no events), so its hazard ratio runs to zero.")
    assert refused.value.columns == ["group"]


@pytest.mark.parametrize("seed", range(3))
def test_an_ordering_along_an_analysis_column_beside_the_design_terms_names_the_column(seed):
    """g (an analysis column) has no event and every stratum has events: the analysis model's own
    terms fail on the copy as well, so the words are the analysis model's, naming `g`, and
    ``columns`` is ``g``, not a design term."""
    data, out, _ = _strata(seed, quiet=False, group=True)
    with pytest.raises(ImputationRefused) as refused:
        _impute(data, _substantive(out, group=True))
    assert str(refused.value) == (
        "The Cox model's coefficients cannot be drawn for the imputations compatible with it, since "
        "its partial likelihood has no maximum for `g` on a completed copy of the data: at every "
        "event time the rows with the event have the lowest `g` of the rows at risk (as when a "
        "group has no events), so its hazard ratio runs to zero.")
    assert refused.value.columns == ["g"]
    assert not getattr(refused.value, "design_terms", None)


@pytest.mark.parametrize("seed", range(5))
def test_a_fit_with_no_stable_maximum_names_columns_of_the_near_alias(seed):
    """The near-alias construction (:func:`aliased._columns`: a code's five indicators, an energy,
    a number, and the code's recode off their span, here by 1e−6 of its length,
    ``test_smcfcs_logistic_cox_draw.near_columns``), its columns named: the information at the
    engine's Cox estimate is singular to working precision, its coefficients along the near-alias
    past their divergence bound. The refusal names columns, each one of the near-aliased set (the
    indicators and the recode, never the energy or the number), and ``columns`` holds their
    inputs. Which of them it names is the solver's, so it is not pinned. At 1e−5 the likelihood
    has a maximum, which the fit now finds (``survival._spread``), so the construction moved
    closer to the span. Before package B the words named no column."""
    from turbotab.core.tests.test_smcfcs_logistic_cox_draw import near_columns

    M, rng = near_columns(1, 300, 1e-6)
    out = aliased._outcome("cox", M, rng)
    names = [*[Term(f"`code_{k}`", column="code") for k in range(1, 6)],
             Term("`energy`", column="energy"), Term("`number`", column="number"),
             Term("`recode`", column="recode")]
    with pytest.raises(ImputationRefused) as refused:
        _draw_outcome(M, out, np.random.default_rng(seed), lambda: names)
    words = str(refused.value)
    assert words.startswith("The Cox model's coefficients cannot be drawn for the imputations "
                            "compatible with it, since ")
    named = [t.label for t in names if t.label in words]
    assert named and set(named) <= {*(f"`code_{k}`" for k in range(1, 6)), "`recode`"}
    assert set(refused.value.columns) <= {"code", "recode"} and refused.value.columns
