"""WP15 · Claims: every sentence matches its source and the computation (docs/turbotab-next/audit/
AUDIT_REPORT.md §5).

Closes IN-20 (the energy finding said adjustment "is not in dispute" and nothing noticed an
energy-related outcome), IN-21 (density plus energy shown as clean composition), IN-22 ("the mean of
recalls is attenuated but unbiased in direction", and no measurement-error caveat on the results),
IN-23 (TNTC taught as a measurement failure), IN-24 (the temporal methods sentence said "scored on
later data"), IN-25 (the lineage drew operations that did not happen), IN-26 (the survey pack mixed
the reliability with its square root), MI-02 (Willett's 800–4,000 against NHS/HPFS's 800–4,200),
MI-03 (the coach's "mostly" at r 0.3), and the minors G14, G15, G16, G18, G19, F15 and D15. One test,
or a few, per acceptance test in §5, numbered as there.

Every reference is independent of the code under test: a primary source quoted in the docstring
(read for this package: the FDA Bacteriological Analytical Manual's ch. 3 PDF; Tomova et al. 2022,
Freedman et al. 2011, Keogh et al. 2020, Groenwold et al. 2012, Van Calster et al. 2019, Lachat et
al. 2016, Pan et al. 2011, de Koning et al. 2011 and Nygaard et al. 2016 in full text from Europe
PMC or PMC; Banna et al. 2017 from the publisher; Yamamoto et al. 2023, Eekhout et al. 2014,
Zindler et al. 2020, Chalmers 2018, Josse et al. and Varoquaux 2018 from their abstracts), a closed
form, numpy or pandas arithmetic on the fixture's own columns, or a simulation's known truth. What
is measured comes through the app's own path: the served teaching entries, the findings stage's
voice, the proposals, the coach, the methods sentences, the seal's chronological draw, the lineage
tracer and the fit stage.

The fixtures for acceptance 5 are the audit's (``repro.tar.gz``: ``A/r15_split.py`` and
``A-skeptic/s9_units.py``), regenerated here from their seeds; acceptance 6 uses
``B-skeptic/s15_lineage.py``'s, acceptance 7 ``F/sim_atten.py``'s design (n = 400,000, λ = 0.70).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import teaching, voice
from turbotab.core.coach import energy_coach
from turbotab.core.decisions import ProjectState
from turbotab.core.methods.dietary_caveats import (
    ENERGY_WHY,
    energy_related,
    measurement_error_line,
)
from turbotab.core.methods.energy import METHOD_TABLE
from turbotab.core.methods.missing import RULE
from turbotab.core.stages.finding_words import FindingContext
from turbotab.core.stages.proposals import build_proposals, rule_excludes

REPO = Path(__file__).resolve().parents[4]
LEDGER = REPO / "docs/turbotab-next/audit/claims-ledger.md"
PACKS = REPO / "docs/turbotab/research"
SAMPLES = REPO / "turbotab/sample_data"


# ── what the app serves ──────────────────────────────────────────────────────


def taught(key: str) -> str:
    """Every word a teaching entry serves: question, one-liner, why, options, terms, drawer."""
    e = teaching.entry(key)
    parts = [e.title, e.question, e.one_liner, e.why, e.consumer]
    parts += [f"{o.label}: {o.consequence}" for o in e.options]
    parts += [f"{t.term}: {t.definition}" for t in e.terms]
    parts += [f"{s.heading}: {s.body}" for s in (e.drawer.sections if e.drawer else [])]
    return "\n".join(parts)


def section(key: str, heading: str) -> Any:
    e = teaching.entry(key)
    found = [s for s in (e.drawer.sections if e.drawer else []) if s.heading == heading]
    assert found, f"{key} has no drawer section {heading!r}"
    return found[0]


def option(key: str, value: str) -> str:
    found = [o for o in teaching.entry(key).options if o.value == value]
    assert found, f"{key} has no option {value!r}"
    return found[0].consequence


_DATED = re.compile(r"\*\((?:(?!\)\*).)*?2026-10-03(?:(?!\)\*).)*?\)\*", re.S)


def read_pack(name: str) -> str:
    """A research pack as it now reads: the dated notes recording what a claim read until this
    package (``*(Until 2026-10-03 this read …)*``) are history, so they are left out."""
    return _DATED.sub("", (PACKS / name).read_text())


def everything_taught() -> str:
    return "\n".join(taught(k) for k in teaching.TEACHING_KEYS)


def _columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for c in frame.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "categorical"
        out.append({"name": c, "dtype": dtype, "n_unique": int(frame[c].nunique()),
                    "n_missing": int(frame[c].isna().sum())})
    return out


def _findings(frame: pd.DataFrame, lens: list[str], target: str | None) -> list[dict[str, Any]]:
    """The findings stage's voice over the legacy streams, as the stage composes them (with WP14
    merged, the superseded detectors are read by ``turbotab.core.detectors``)."""
    from turbotab import engine
    from turbotab.core import detectors
    from turbotab.core.stages.findings import speak_for

    structural = [engine.shape_finding_to_dict(f) for f in engine.diagnose(frame, target)]
    structural = detectors.reframe(detectors.structural(structural, frame), lens, frame)
    return [f for _, f in speak_for(frame, lens, target, structural,
                                    detectors.pack_findings(frame, lens))]


def _dietary(n: int = 600, seed: int = 15) -> pd.DataFrame:
    """Recalls-like rows: energy and three energy-bearing nutrients that track it, sex, BMI, LDL."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    kcal = np.where(sex == "M", rng.normal(2500, 500, n), rng.normal(1900, 420, n)).clip(300, 6000)
    share = rng.dirichlet([4, 5, 3], n)
    frame = pd.DataFrame({
        "pid": np.arange(n), "sex": sex, "age": rng.uniform(20, 75, n).round(),
        "energy_kcal": kcal.round(1),
        "protein_g": (kcal * share[:, 0] / 4).round(1),
        "carbohydrate_g": (kcal * share[:, 1] / 4).round(1),
        "fat_g": (kcal * share[:, 2] / 9).round(1),
    })
    frame["bmi"] = (24 + 0.002 * (kcal - 2100) + rng.normal(0, 4, n)).round(1)
    frame["ldl"] = (120 + 0.05 * frame["fat_g"] + rng.normal(0, 25, n)).round(1)
    return frame


# ── 1 · Every WRONG, OVERSTATED or SELF-CONTRADICTED ledger row, re-checked ──
#
# One entry per ledger row: where the app serves the claim now, a phrase that must be gone, the
# corrected sentence (or its load-bearing part) that must be served, and the primary source quoted
# beside it. Rows the methods layer already corrected (WP6, WP7, WP9, WP10) are re-checked here
# against the text served today, so a later edit cannot quietly restore the old claim.

_TOMOVA = ("Tomova et al. 2022, AJCN 115:189 (PMC8755101): \"It remains underappreciated that "
           "adjusting for total energy and adjusting for remaining energy intake evaluate very "
           "different causal estimands.\"")
_FREEDMAN = ("Freedman et al. 2011, JNCI 103:1086 (PMC3143422): \"In multivariable disease models "
             "with two or more mismeasured exposures, estimated relative risks may become "
             "attenuated, inflated, or can even change direction\"")
_YAMAMOTO = ("Yamamoto et al. 2023, eLife 12:e83616: \"Significant bias in simulated associations "
             "using self-reported NI was reduced but not completely eliminated by Goldberg cutoffs "
             "in 14 of 24 nutrition-outcome pairs; bias was not reduced for the remaining 10 cases. "
             "… Whether one uses Goldberg cutoffs should therefore be decided based on research "
             "purposes and not general rules.\"")
_BANNA = ("Banna et al. 2017, Front Nutr 4:45, quoting Willett, Nutritional Epidemiology 3rd ed. "
          "(2013): \"the range of 500–3,500 kcal/day may be applied to data from women\" and \"an "
          "allowable range of 800–4,000 kcal/day for men may be used\"")
_PAN = ("Pan et al. 2011, AJCN 94:1088 (PMC3173026), NHS, NHS II and HPFS: \"daily energy intake "
        "<800 or >4200 kcal/d for men and <500 or >3500 kcal/d for women\"")
_BAM = ("FDA Bacteriological Analytical Manual, ch. 3, Aerobic Plate Count: \"When number of CFU "
        "per plate exceeds 250, for all dilutions, record the counts as too numerous to count "
        "(TNTC) for all but the plate closest to 250\"; \"Estimate the APC as greater than 100 "
        "times the highest dilution plated, times the area of the plate.\"")

RECHECK: dict[int, dict[str, Any]] = {
    4: dict(where=lambda: taught("lens") + taught("models"),
            gone=["egularization is mandatory"],
            says=["an unpenalized least-squares or logistic model is degenerate",
                  "Penalization, dimension reduction or one test per feature are the ways through"],
            source="closed form: rank(X) ≤ n < p, so XᵀX is singular and least squares has a "
                   "(p − n)-dimensional set of exact fits (test_4_p_over_n_is_degenerate_for_least_"
                   "squares_only); the app's own feature-wise family is a way through"),
    9: dict(where=lambda: taught("repairs"),
            gone=["TNTC and QNS are measurement failures"],
            says=["the count is above that limit, right-censored", "QNS, quantity not sufficient"],
            source=_BAM),
    11: dict(where=lambda: taught("target"),
             gone=["most-checked figure"],
             says=["STROBE-nut asks for the number excluded for missing, incomplete or implausible "
                   "dietary data, and STROBE suggests a flow diagram"],
             source="Lachat et al. 2016 (STROBE-nut), PLoS Med (PMC4896435), nut-13: \"Report the "
                    "number of individuals excluded based on missing, incomplete, or implausible "
                    "dietary/nutritional data.\" STROBE 13(c): \"Consider use of a flow diagram.\""),
    12: dict(where=lambda: taught("event") + taught("models"),
             gone=["Rank models on calibration"],
             says=["Judge models on calibration as well as discrimination",
                   "reports its calibration intercept and slope beside the AUC that ranks the "
                   "families"],
             source="Van Calster et al. 2019, BMC Med 17:230 (PMC6912996): \"poor calibration may "
                    "make an algorithm less clinically useful than a competitor algorithm that has a "
                    "lower AUC but is well calibrated\"; the app ranks binary fits by AUC "
                    "(models/metrics.py PRIMARY) and reports calibration on every fit"),
    14: dict(where=lambda: option("task", "multiclass"),
             gone=["scored by accuracy"], says=["scored by log loss"],
             source="the code: models/metrics.py PRIMARY['multiclass'] == 'log_loss' (audit ME-10)"),
    16: dict(where=lambda: taught("purpose"),
             gone=[], says=["in a randomized trial it is valid for baseline covariates",
                            "multiple imputation, offered here"],
             source="Groenwold et al. 2012, CMAJ 184:1265 (PMC3414599): the method \"typically "
                    "results in biased estimates in nonrandomized studies\"; \"In randomized "
                    "trials, the missing-indicator method is a valid method to handle missing "
                    "baseline covariate data\""),
    21: dict(where=lambda: taught("aggregation"),
             gone=["unbiased in direction"],
             says=["attenuated toward zero only as the model's one error-prone exposure",
                   "a coefficient can be attenuated, inflated or change sign"],
             source=f"{_FREEDMAN}; Keogh et al. 2020 (STRATOS Part 1, PMC7450672) §3.1.3: \"the "
                    f"estimated coefficients in model (11) may be larger or smaller than the true "
                    f"target values in a rather unpredictable manner\""),
    26: dict(where=lambda: taught("roles") + taught("survey"),
             gone=[], says=["Under inference the survey question then asks whether the estimates "
                            "describe the surveyed population"],
             source="CDC NHANES variance tutorial: estimates \"computed using standard statistical "
                    "software packages that assume simple random sampling are generally too low\"; "
                    "the contradiction in use closed by WP10 (the survey question, "
                    "test_wp10_survey_design.py)"),
    29: dict(where=lambda: taught("exclusions"),
             gone=["Willett and the Nurses' Health Study use", "Willett, by sex"],
             says=["Willett's textbook (2013) gives 500–3,500 kcal a day for women and 800–4,000 "
                   "for men", "Women outside 500–3,500 (NHS) and men outside 800–4,200 (HPFS)"],
             source=f"{_BANNA}; {_PAN}"),
    31: dict(where=lambda: taught("exclusions"),
             gone=["That exclusion is insufficient is settled"],
             says=["Goldberg cut-offs reduced but did not remove bias in 14 of 24",
                   "fixed kcal screens were not evaluated"],
             source=_YAMAMOTO),
    34: dict(where=lambda: taught("exclusions"),
             gone=["the field's standard for misreporting"],
             says=["widely used, and chosen by research purpose"], source=_YAMAMOTO),
    37: dict(where=lambda: taught("missing"),
             gone=[], says=["For inference, mean or median filling understates variance",
                            "For prediction, a fill learned in each training fold is the "
                            "deployable choice"],
             source="Josse et al., arXiv:1902.06931: \"the widely-used method of imputing with a "
                    "constant, such as the mean prior to learning is consistent when missing "
                    "values are not informative\" (purpose-scoped by WP7)"),
    38: dict(where=lambda: RULE,
             gone=[], says=["The outcome's place in the imputation model depends on the purpose",
                            "without the outcome"],
             source="Sisk et al. 2023, Stat Methods Med Res 32:1461: \"When missingness is allowed "
                    "at deployment, omitting the outcome from the imputation model at the "
                    "development was preferred\" (WP7's RULE)"),
    42: dict(where=lambda: everything_taught(),
             gone=["Below about 50 rows", "below about 50 rows"],
             says=["computed on these rows"],
             source="Varoquaux 2018, NeuroImage 180:68: \"sample sizes of many neuroimaging "
                    "studies inherently lead to large error bars, eg±10% for 100 samples\" (the "
                    "threshold was 2–4× too low; WP9 replaced it with intervals on the user's rows)"),
    43: dict(where=lambda: taught("lens") + taught("energy_adjustment"),
             gone=["confounds every nutrient association", "associations are confounded by it"],
             says=["adjusting for it changes what a nutrient's coefficient means",
                   "Adjusting for it changes the question"],
             source=_TOMOVA),
    44: dict(where=lambda: option("energy_adjustment", "residual")
             + option("energy_adjustment", "residual_energy_dropped"),
             gone=[], says=["the standard model's swap", "differs when covariates track energy"],
             source="McCullough & Byrd 2023, AJE 192:1801: \"A variation on the simple nutrient "
                    "residual model proposed by Willett and Stampfer includes the nutrient residual "
                    "plus a term for total energy intake.\" (WP6)"),
    45: dict(where=lambda: option("energy_adjustment", "density_multivariate"),
             gone=["diet composition"], says=["obscure, and still biased"],
             source="Tomova et al. 2022, model 3b: \"the multivariable nutrient density model "
                    "returns a more accurate estimate than the (unadjusted) nutrient density model, "
                    "but one which is still biased\"; Table 2 estimand \"Obscure\""),
    54: dict(where=lambda: option("models", "boosted_trees"),
             gone=["handles missing values"], says=["gives no coefficients"],
             source="the code: models/pipeline.py shared_steps fills or drops blanks before every "
                    "family (test_1_the_trees_never_see_a_blank)"),
    64: dict(where=lambda: METHOD_TABLE["none"]["estimand"],
             gone=[], says=["total energy is not in the model"],
             source="the code: under 'none' every energy-role column leaves the matrix (WP6, "
                    "test_wp6_energy_estimands.py test_1)"),
    66: dict(where=lambda: METHOD_TABLE["residual"]["estimand"]
             + METHOD_TABLE["residual_energy_dropped"]["estimand"],
             gone=["The same substitution as the standard model"],
             says=["the coefficient is the standard model's exactly",
                   "only when no other covariate correlates with energy"],
             source="McCullough & Byrd 2023 (as row 44); WP6 test_2"),
    67: dict(where=lambda: METHOD_TABLE["density_multivariate"]["estimand"],
             gone=["Diet composition…"],
             says=["but one which is still biased", "the all-components model is the paper's "
                   "recommended route"],
             source="Tomova et al. 2022 (as row 45) and its Discussion: \"we would recommend the "
                    "all-components model as the more intuitive and transparent option and the "
                    "least susceptible to misinterpretation\""),
    72: dict(where=lambda: voice.sentence_for(
                 d.SetEnergyAdjustment(method="none", energy_column=None, nutrients=[]),
                 ProjectState(roles={"kcal": "energy", "fat_g": "exposure"})),
             gone=[], says=["`kcal` was left out of the models"],
             source="the code (WP6): under 'none' the energy-role column leaves the models"),
    76: dict(where=lambda: " ".join(p["rule"]["reason"] for p in _presets()["exclusions"]),
             gone=["Willett's sex-specific cut-offs"],
             says=["Willett 2013's sex-specific cut-offs",
                   "the Nurses' Health Study and Health Professionals Follow-up Study cut-offs"],
             source=f"{_BANNA}; {_PAN}"),
    79: dict(where=lambda: _coach_first(0.31) + _coach_first(0.6) + _coach_first(0.8),
             gone=["mostly how much people eat"],
             says=["energy explains `10%` of it", "energy explains `36%` of it",
                   "energy explains most, `64%`"],
             source="arithmetic: the share of a nutrient's variance its line on energy explains is "
                    "r² (0.31² = 0.096; 0.60² = 0.36; 0.80² = 0.64)"),
    86: dict(where=lambda: _energy_finding("ldl")["why_it_matters"],
             gone=["not in dispute", "every nutrient association is confounded"],
             says=["makes a nutrient's coefficient a substitution",
                   "The two answer different questions (Tomova et al. 2022)"],
             source=_TOMOVA),
    87: dict(where=lambda: voice.finish(_energy_summary_without_nutrients()),
             gone=["confounded by it until adjusted"],
             says=["adjusting for it makes each nutrient's effect a swap at fixed energy"],
             source=_TOMOVA),
    96: dict(where=lambda: read_pack("CLINICAL_SURVEY_PACK.md"),
             gone=["[SETTLED that polychoric is appropriate"],
             says=["polyserial correlation, not a polychoric one",
                   "ordinal alpha should not be used in routine reliability analyses"],
             source="Chalmers 2018, Educ Psychol Meas 78:1056 (PMC6293415): \"ordinal alpha should "
                    "not be used in routine reliability analyses and reports\"; a polyserial "
                    "correlation is the one between an ordinal and a continuous variable (the "
                    "item–rest sum). The legacy turbotab/survey.py that printed it is not imported "
                    "by Next (test_1_the_polychoric_claim_is_not_served_by_next)"),
}

# Rows the ledger marked INCOMPLETE or misleading, closed beside the three statuses.
ALSO: dict[int, dict[str, Any]] = {
    73: dict(where=lambda: voice.sentence_for(
                 d.SetEnergyAdjustment(method="residual", energy_column="kcal", nutrients=["fat_g"]),
                 ProjectState(roles={"kcal": "energy", "fat_g": "exposure"})),
             gone=[], says=["with total energy kept in the outcome model"],
             source="STROBE-nut nut-12.2: \"Describe and justify the method for energy "
                    "adjustments\" (WP6)"),
    75: dict(where=lambda: voice.sentence_for(d.SetMissing(strategy="impute"), ProjectState()),
             gone=[], says=["the median for numbers"],
             source="STROBE-nut nut-13: \"any method used to handle missing values\" (WP7)"),
    81: dict(where=lambda: _coach_second("density", 0.62, 0.02),
             gone=["what is left is composition"], says=["it no longer tracks energy"],
             source="Tomova et al. 2022: the density coefficient is \"an obscure quantity\""),
}


def _presets(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = _dietary() if frame is None else frame
    return build_proposals(frame, _columns(frame), lens=["dietary"], target="ldl",
                           roles={"energy_kcal": "energy", "protein_g": "exposure",
                                  "carbohydrate_g": "exposure", "fat_g": "exposure",
                                  "sex": "covariate", "age": "covariate", "bmi": "covariate",
                                  "pid": "identifier"})


def _relationship(r_before: float, r_after: float | None = None) -> NS:
    return NS(kind="relationship", r_before=r_before, r_after=r_after, y_label_before="sodium_mg",
              x_label="energy_kcal", coach=[])


def _coach_first(r: float) -> str:
    view = _relationship(r, 0.0)
    energy_coach(NS(method="residual"), [view], None)
    return view.coach[0].text


def _coach_second(method: str, r_before: float, r_after: float) -> str:
    view = _relationship(r_before, r_after)
    energy_coach(NS(method=method), [view], None)
    return view.coach[1].text


def _energy_finding(target: str) -> dict[str, Any]:
    found = [f for f in _findings(_dietary(), ["dietary"], target)
             if f["id"].startswith("pack::dietary::energy_adjustment")]
    assert len(found) == 1, [f["id"] for f in _findings(_dietary(), ["dietary"], target)]
    return found[0]


def _energy_summary_without_nutrients() -> str:
    """The energy finding's summary when no energy-bearing nutrient is recognized by name."""
    from turbotab.core.stages.finding_words import FAMILIES

    frame = pd.DataFrame({"kcal": [1800.0, 2100.0, 2500.0], "y": [1.0, 2.0, 3.0]})
    finding = {"id": "pack::dietary::energy_adjustment", "affected_columns": ["kcal"],
               "title": "", "detail": ""}
    say = FAMILIES["pack::dietary::energy_adjustment"]
    return say(finding, {"energy_column": "kcal"}, FindingContext(frame=frame, target="y")).summary


def _ledger_rows() -> dict[int, str]:
    """The ledger's rows marked WRONG, OVERSTATED or SELF-CONTRADICTED, read from the ledger."""
    rows = {}
    audited = LEDGER.read_text().split("\n## WP15 re-check", 1)[0]  # the auditor's rows only
    for line in audited.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        if re.search(r"WRONG|OVERSTATED|SELF-CONTRADICTED", " ".join(cells[2:])):
            rows[int(cells[0])] = cells[1]
    return rows


def test_1_every_marked_ledger_row_has_a_recheck():
    """The re-check covers the ledger exactly: 27 rows, read from the ledger itself."""
    rows = _ledger_rows()
    assert len(rows) == 27, sorted(rows)
    assert set(rows) == set(RECHECK), (sorted(set(rows) - set(RECHECK)), sorted(set(RECHECK) - set(rows)))
    for row, entry in {**RECHECK, **ALSO}.items():
        assert entry["source"].strip(), f"row {row} names no source"


@pytest.mark.parametrize("row", sorted({**RECHECK, **ALSO}))
def test_1_the_corrected_sentence_is_served_and_the_old_one_is_gone(row):
    entry = {**RECHECK, **ALSO}[row]
    served = entry["where"]()
    for phrase in entry["gone"]:
        assert phrase not in served, f"ledger row {row} still serves {phrase!r}"
    for phrase in entry["says"]:
        assert phrase in served, f"ledger row {row} does not serve {phrase!r}:\n{served}"


def test_4_p_over_n_is_degenerate_for_least_squares_only():
    """Ledger row 4's corrected sentence, checked in closed form: with p > n, least squares has
    exact fits along a null space of dimension p − n, so two different coefficient vectors fit the
    data exactly; a penalized fit (ridge) is unique. Reference: numpy's SVD and least squares."""
    rng = np.random.default_rng(4)
    n, p = 12, 30
    X, y = rng.normal(size=(n, p)), rng.normal(size=n)
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    _, s, vt = np.linalg.svd(X)
    null = vt[n:]  # rank n, so the last p − n right singular vectors span the null space
    assert null.shape == (p - n, p) and int((s > 1e-10).sum()) == n
    b1 = b0 + 3.0 * null[0]
    assert np.allclose(X @ b0, y) and np.allclose(X @ b1, y) and not np.allclose(b0, b1)
    ridge = np.linalg.solve(X.T @ X + 1.0 * np.eye(p), X.T @ y)  # strictly convex: one solution
    assert np.linalg.matrix_rank(X.T @ X + np.eye(p)) == p and np.isfinite(ridge).all()


def test_1_the_trees_never_see_a_blank():
    """Ledger row 54 (G16): the shared missing-values step runs before every family, so the boosted
    trees' model step reads no blank. Reference: pandas ``isna`` on what reaches the model."""
    from turbotab.core.models import families
    from turbotab.core.models.pipeline import DesignSpec, family_steps, transformer

    frame = _dietary(300)
    frame.loc[::7, "fat_g"] = np.nan
    assert int(frame["fat_g"].isna().sum()) > 0
    trees = next(f for f in families() if f.key == "boosted_trees")
    spec = DesignSpec(predictors=["fat_g", "protein_g", "age"], inputs=["fat_g", "protein_g", "age"],
                      categorical=[], numeric=["fat_g", "protein_g", "age"], energy=None,
                      impute=True,
                      roles={"fat_g": "exposure", "protein_g": "exposure", "age": "covariate"})
    steps = family_steps(spec, trees)
    assert steps[0][0] == "impute"
    seen = transformer(steps).fit_transform(frame[spec.inputs])
    assert int(pd.DataFrame(seen).isna().sum().sum()) == 0


def test_1_the_multiclass_metric_is_the_one_the_code_ranks_by():
    """Ledger row 14 (G16): the option names the primary metric the fit ranks multiclass by."""
    from turbotab.core.models.metrics import LABELS, PRIMARY

    assert PRIMARY["multiclass"] == "log_loss"
    assert LABELS[PRIMARY["multiclass"]].lower() in option("task", "multiclass")


def test_1_calibration_is_reported_beside_the_auc_that_ranks():
    """Ledger row 12: binary families are ranked by AUC (the drawer now says so), and every fit
    carries its out-of-fold calibration (``FittedModel.calibration``; computed in WP9)."""
    from turbotab.core.models.artifacts import FittedModel
    from turbotab.core.models.metrics import PRIMARY

    assert PRIMARY["binary"] == "auc"
    assert "calibration" in FittedModel.model_fields


def test_1_the_polychoric_claim_is_not_served_by_next():
    """Ledger row 96: the SETTLED polychoric sentences live in the legacy ``turbotab/survey.py``,
    which no module under ``turbotab/core`` or ``turbotab/server`` imports; the pack it quoted is
    corrected (row 96 above)."""
    pattern = re.compile(r"^\s*(?:from turbotab import survey|import turbotab\.survey|"
                         r"from turbotab\.survey import)", re.M)
    for folder in ("turbotab/core", "turbotab/server"):
        for path in (REPO / folder).rglob("*.py"):
            assert not pattern.search(path.read_text()), path


def test_1_unbiased_in_direction_fails_with_two_error_prone_exposures():
    """Ledger row 21's old claim, refuted in closed form and by simulation: with two correlated
    exposures and the second measured with error, the first's coefficient changes sign.

    Truth: Y = −0.10·X1 + 0.50·X2 + e, corr(X1, X2) = 0.8, W1 = X1, W2 = X2 + U with var(U) = 1.
    Closed form for jointly normal variables: β* = Σ_W⁻¹ Σ_WX β. Reference: numpy."""
    rho, beta = 0.8, np.array([-0.10, 0.50])
    sxx = np.array([[1.0, rho], [rho, 1.0]])
    sww = sxx + np.diag([0.0, 1.0])
    closed = np.linalg.solve(sww, sxx @ beta)
    assert closed[0] > 0 > beta[0]  # the sign flips
    rng = np.random.default_rng(21)
    n = 200_000
    x = rng.multivariate_normal([0, 0], sxx, n)
    y = x @ beta + rng.normal(0, 1, n)
    w = x + np.column_stack([np.zeros(n), rng.normal(0, 1, n)])
    fitted, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), w]), y, rcond=None)
    assert fitted[1] == pytest.approx(closed[0], abs=0.01)
    assert fitted[1] > 0


# ── 2 · A badge-consistency test ─────────────────────────────────────────────


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text)
    return [re.sub(r"\s+", " ", p.strip().rstrip(".!?").lower()) for p in parts if p.strip()]


def badge_conflicts(claims: list[tuple[str, str, str, str]]) -> list[str]:
    """``(where, text, status, source)`` claims → every identical text, or sentence, with the same
    source that carries more than one badge."""
    seen: dict[tuple[str, str], dict[str, list[str]]] = {}
    for where, text, status, source in claims:
        for unit in [re.sub(r"\s+", " ", text.strip().lower()), *_sentences(text)]:
            if len(unit.split()) < 5:  # a fragment such as "Split by participant" is not a claim
                continue
            seen.setdefault((unit, source), {}).setdefault(status, []).append(where)
    return [f"{unit!r} ({source}): {by}" for (unit, source), by in seen.items() if len(by) > 1]


def served_claims() -> list[tuple[str, str, str, str]]:
    """Every badged claim the app serves: each teaching entry's drawer sections, each entry's own
    badge (its why), and the proposals' badges with their labels."""
    out = []
    for e in teaching.entries():
        for s in (e.drawer.sections if e.drawer else []):
            if s.evidence is not None:
                out.append((f"{e.key}/{s.heading}", s.body, s.evidence.status, s.evidence.source))
    return out


def test_2_identical_claims_with_the_same_source_share_one_badge():
    """Audit G18: "Internal validation must resample the entire modeling pipeline" read SETTLED in
    the temporal card and CONVENTION in the split and open-the-seal cards. The clinical pack, §A5.5:
    "[SETTLED that the full pipeline must be inside the loop; the bootstrap-vs-CV preference is
    CONVENTION.]" Each claim is now one section, one badge."""
    claims = served_claims()
    assert len(claims) > 60
    assert badge_conflicts(claims) == []
    whole = [c for c in claims if "resample the entire modeling pipeline" in c[1]]
    assert {c[0].split("/")[0] for c in whole} == {"temporal", "split", "open_seal"}
    assert {c[2] for c in whole} == {"SETTLED"}
    split = [c for c in claims if "a single train and test split is the weakest" in c[1]]
    assert {c[2] for c in split} == {"CONVENTION"} and len(split) == 2


def test_2_the_check_catches_a_disagreement():
    """The positive control: the same sentence under two badges is reported, and the same words
    under one badge, or under two sources, are not."""
    src = "research/X.md#1 · A"
    text = "Internal validation must resample the entire modeling pipeline."
    assert badge_conflicts([("a", text, "SETTLED", src), ("b", text, "CONVENTION", src)])
    assert badge_conflicts([("a", "Something else entirely here. " + text, "SETTLED", src),
                            ("b", text, "CONVENTION", src)])
    assert not badge_conflicts([("a", text, "SETTLED", src), ("b", text, "SETTLED", src)])
    assert not badge_conflicts([("a", text, "SETTLED", src), ("b", text, "CONVENTION", "other")])


# ── 3 · "Not in dispute" is gone; an energy-related outcome pushes the DISPUTED note ──


def test_3_not_in_dispute_is_served_nowhere():
    """Tomova et al. 2022: "adjusting for total energy and adjusting for remaining energy intake
    evaluate very different causal estimands". No teaching text, and no finding on the dietary
    fixture or the sample recalls, says adjustment is not in dispute."""
    assert "not in dispute" not in everything_taught()
    recalls = pd.read_csv(SAMPLES / "dietary_recalls.csv")
    for frame, target in ((_dietary(), "ldl"), (_dietary(), "bmi"), (recalls, "hba1c")):
        for f in _findings(frame, ["dietary"], target):
            text = " ".join(str(f.get(k) or "") for k in ("title", "summary", "detail",
                                                           "why_it_matters"))
            assert "not in dispute" not in text, f["id"]
            assert "confounded by it until adjusted" not in text


def test_3_the_energy_finding_states_the_choice_and_its_badge_reads_the_outcome():
    plain = _energy_finding("ldl")
    assert plain["why_it_matters"] == voice.finish(ENERGY_WHY)
    assert plain["evidence"]["status"] == "CONVENTION"
    related = _energy_finding("bmi")
    assert related["evidence"] == {"status": "DISPUTED", "source": (
        "research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature")}
    assert related["why_it_matters"].startswith(voice.finish(ENERGY_WHY)[:-1])
    assert "`bmi` reads as BMI" in related["why_it_matters"]
    assert "collider" in related["why_it_matters"]


def test_3_an_energy_related_outcome_pushes_the_disputed_note_onto_the_energy_card():
    """NUTRITION_PACK §04, Diagnostic: "Detect whether the outcome is itself energy-related
    (weight, BMI, adiposity, diabetes) — if so, escalate the mediation/collider warning." The card
    carries the line within its budget, and the typed field carries the DISPUTED badge."""
    frame = _dietary()
    roles = {"energy_kcal": "energy", "protein_g": "exposure", "carbohydrate_g": "exposure",
             "fat_g": "exposure", "sex": "covariate", "age": "covariate", "pid": "identifier"}
    card = build_proposals(frame, _columns(frame), lens=["dietary"], target="bmi",
                           roles={**roles, "ldl": "covariate"}, purpose="inference")["energy"]
    line = ("`bmi` reads as BMI: energy may be on its causal path and a collider, so adjusting "
            "for it is disputed.")
    assert card["notes"] == [line]
    assert voice.words(line) <= teaching.COMPOSED_BUDGETS["card_line"]
    assert card["outcome_dispute"]["evidence"]["status"] == "DISPUTED"
    assert card["outcome_dispute"]["kind"] == "BMI"
    plain = build_proposals(frame, _columns(frame), lens=["dietary"], target="ldl",
                            roles={**roles, "bmi": "covariate"}, purpose="inference")["energy"]
    assert plain["notes"] == [] and plain["outcome_dispute"] is None
    from turbotab.server.schemas import EnergyReading

    EnergyReading.model_validate(card)


@pytest.mark.parametrize("name, kind", [
    ("bmi", "BMI"), ("BMXBMI", "BMI"), ("bmi_change", "BMI"), ("weight_kg", "body weight"),
    ("body_weight", "body weight"), ("BMXWT", "body weight"), ("weight_gain_5y", "body weight"),
    ("waist_cm", "waist size"), ("BMXWAIST", "waist size"), ("fat_mass_kg", "adiposity"),
    ("body_fat_pct", "adiposity"), ("obesity", "adiposity"), ("diabetes", "diabetes"),
    ("incident_t2d", "diabetes"), ("DIQ010", "diabetes"),
])
def test_3_energy_related_outcomes_are_read_by_whole_tokens(name, kind):
    assert energy_related(name) == kind


@pytest.mark.parametrize("name", [
    "fat_g", "sfa_g", "WTMEC2YR", "WTDRD1", "sampling_weight", "survey_weight", "birth_weight",
    "ldl", "glucose", "hba1c", "fatigue_score", "carbohydrate_g", "sbp", "crp",
])
def test_3_other_outcomes_are_not(name):
    assert energy_related(name) is None


# ── 4 · Density plus energy carries Tomova's caveat; "unbiased in direction" is qualified;
#        TNTC is right-censored ───────────────────────────────────────────────────────────


def test_4_density_plus_energy_carries_tomovas_caveat_wherever_it_is_named():
    """Tomova et al. 2022, model 3b: "returns a more accurate estimate than the (unadjusted)
    nutrient density model, but one which is still biased"; Table 2 gives its estimand as
    "Obscure"; the Discussion recommends "the all-components model". The estimand the design
    serves (``describe_model``), the option, the pack table and the coach all say so."""
    from turbotab.core.methods.energy import describe_model

    adj = d.EnergyAdjustment(method="density_multivariate", energy_column="kcal",
                             nutrients=["fat_g"])
    roles = {"kcal": "energy", "fat_g": "exposure", "age": "covariate"}
    served = describe_model(adj, ["fat_g", "kcal", "age"], roles,
                            ["fat_g_per_kcal", "kcal", "age"]).text
    assert "still biased" in served and "all-components model" in served
    assert "diet composition" not in option("energy_adjustment", "density_multivariate").lower()
    pack = read_pack("NUTRITION_PACK.md")
    row = next(line for line in pack.splitlines() if "**Multivariate nutrient density**" in line)
    assert "still biased" in row and "Composition, with total energy" not in row
    assert "composition" not in _coach_second("density_multivariate", 0.62, 0.02)


def test_4_unbiased_in_direction_is_qualified_in_the_app_and_the_pack():
    """Keogh et al. 2020 §3.1.1: under classical error "|βX*| ≤ |βX|" for one covariate; §3.1.3:
    with several, "larger or smaller than the true target values in a rather unpredictable
    manner"."""
    for text in (taught("aggregation"), read_pack("NUTRITION_PACK.md")):
        assert "it\n> is unbiased in direction" not in text and "but unbiased in direction" not in text
    pack = read_pack("NUTRITION_PACK.md")
    assert "attenuated, inflated or change sign" in pack


def test_4_the_inference_table_carries_its_measurement_error_line(tmp_path):
    """IN-22: under inference the coefficient table says its dietary intakes are measured with
    error and uncorrected, through the real design and fit stages; under prediction it does not.
    Reference: the line is the one ``measurement_error_line`` states for these exposures (Freedman
    et al. 2011, quoted at row 21)."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _dietary(400)[["ldl", "age", "energy_kcal", "fat_g", "protein_g"]]
    roles = {"energy_kcal": "energy", "fat_g": "exposure", "protein_g": "exposure",
             "age": "covariate"}
    expected = measurement_error_line(["fat_g", "protein_g"], ["energy_kcal"], reports=1.0)
    assert "attenuated, inflated or change sign (Freedman et al. 2011)" in expected
    seen = {}
    for purpose in ("inference", "prediction"):
        paths = mf.ingest_frame(frame, tmp_path / purpose)
        st = mf.state(roles=roles, target="ldl", task="regression", models=["linear"],
                      energy_adjustment=d.EnergyAdjustment(method="standard",
                                                           energy_column="energy_kcal",
                                                           nutrients=["fat_g", "protein_g"]),
                      purpose=purpose, split=d.SplitSpec(holdout=0.0, seed=0, folds=5))
        split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
        ti = mf.target_info("regression", "ldl")
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        seen[purpose] = fit.data["models"][0]["concerns"]
    assert expected in seen["inference"]
    assert not any("measured with error" in c for c in seen["prediction"])
    one = measurement_error_line(["fat_g"], [], reports=2.0)
    assert "the mean of `2` reports per person" in one and "attenuated toward zero" in one


def test_4_tntc_is_right_censored_in_the_finding_on_the_sample_labs():
    """FDA BAM ch. 3: "When number of CFU per plate exceeds 250, for all dilutions, record the
    counts as too numerous to count (TNTC)"; crowded plates are estimated "as greater than 100
    times the highest dilution plated". On ``clinical_labs.csv`` the censored-values finding counts
    each column's TNTC cells as right-censored and its QNS cells as failures; the counts are
    pandas' over the file's own cells."""
    frame = pd.read_csv(SAMPLES / "clinical_labs.csv")
    cells = frame["wbc"].dropna().astype(str).str.strip().str.upper()
    n_tntc, n_qns = int((cells == "TNTC").sum()), int((cells == "QNS").sum())
    assert n_tntc > 0 and n_qns > 0
    found = [f for f in _findings(frame, ["clinical"], "readmitted")
             if f["id"].startswith("pack::clinical::censored_values")]
    assert len(found) == 1
    detail, summary = found[0]["detail"], found[0]["summary"]
    assert f"`wbc`: `{n_tntc}` too numerous to count, right-censored above the countable range" in detail
    assert f"`{n_qns}` measurement {'failure' if n_qns == 1 else 'failures'} (`qns`) with no value" in detail
    assert "not a missing one" in detail and "measurement failures, and they route" not in detail
    assert "too numerous to count" in summary
    assert voice.words(summary) <= 20


def test_4_a_column_with_only_tntc_and_qns_is_not_titled_as_failures():
    """The legacy reading titled a column whose only text was TNTC and QNS "Measurement failures
    are recorded as text in a lab column"; TNTC is censoring, so the title counts it as such.
    Reference: pandas counts of the cells written."""
    rng = np.random.default_rng(23)
    cfu = rng.integers(25, 250, 120).astype(object)
    cfu[:9], cfu[9:11] = "TNTC", "QNS"
    frame = pd.DataFrame({"cfu": cfu, "age": rng.integers(20, 80, 120), "y": rng.normal(size=120)})
    found = [f for f in _findings(frame, ["clinical"], "y")
             if f["id"].startswith("pack::clinical::censored_values")]
    assert len(found) == 1
    f = found[0]
    assert f["title"] == "`1` analyte carries censored values."
    assert "`cfu`: `9` too numerous to count, right-censored above the countable range; `2` " \
           "measurement failures (`qns`) with no value" in f["detail"]
    assert "Measurement failures are recorded" not in f["title"]


def test_4_tntc_in_the_teaching_and_the_pack():
    body = section("repairs", "Too many to count is a value").body
    assert "right-censored" in body and "over 250 per plate" in body
    pack = read_pack("CLINICAL_SURVEY_PACK.md")
    assert "record the counts as too numerous to count (TNTC)" in pack
    assert '> *"`TNTC` and `QNS` are not censoring at a detection limit' not in pack


# ── 5 · The temporal methods sentence describes what was drawn ───────────────


def _visits_r15() -> tuple[np.ndarray, np.ndarray]:
    """A/r15_split.py: 300 units with 1–5 visits uniform on 0–100 (seed 0; the draws in order)."""
    rng = np.random.default_rng(0)
    units = 300
    per = rng.integers(1, 6, units)
    g = np.repeat([f"p{i}" for i in range(units)], per)
    n = len(g)
    rng.permutation(np.arange(10_000, 10_000 + n))  # the script's ids, drawn to keep the stream
    np.where(rng.random(n) < 0.2, "case", "control")  # its outcome, likewise
    np.sort(rng.choice(np.arange(n), int(n * 0.9), replace=False))  # its cohort, likewise
    return rng.uniform(0, 100, n), g


def _visits_s9() -> tuple[np.ndarray, np.ndarray]:
    """A-skeptic/s9_units.py: 300 units, 2–5 visits spread over up to 40 units of time (seed 4)."""
    rng = np.random.default_rng(4)
    ids, ts = [], []
    for u in range(300):
        start = rng.uniform(0, 60)
        k = rng.integers(2, 6)
        v = start + np.sort(rng.uniform(0, 40, k))
        ids += [u] * k
        ts += list(v)
    return np.array(ts), np.array(ids)


def _split_sentence(chron: Any, *, holdout: float = 0.2, seed: int = 0, group: str | None = "pid") -> str:
    basis = ({"state": "grouped", "column": group, "source": "grain", "exploratory": False,
              "label": f"grouped by `{group}`"} if group else
             {"state": "one_row_per_unit", "column": None, "source": "grain", "exploratory": False,
              "label": "one row per unit"})
    plan = {"basis": basis, "chronology": chron.model_dump(mode="json")}
    return voice.sentence_for(d.SetSplit(holdout=holdout, seed=seed, folds=5),
                              ProjectState(target="y", task="regression"), {"seal_plan": plan})


@pytest.mark.parametrize("fixture, audit", [(_visits_r15, 0.71), (_visits_s9, 0.56)])
def test_5_the_sentence_states_the_share_of_held_out_rows_that_predate_training(fixture, audit):
    """AUDIT_REPORT IN-24: "56–71% of held-out rows were observed before the latest training row".
    The draw's mask is the app's; the share is counted from it with numpy (held-out times strictly
    below the latest training time) and must match the audit's printed share to two decimals."""
    from turbotab.core.seal import chronological_holdout

    times, groups = fixture()
    mask, chron = chronological_holdout(times, groups, 0.2, 0, "visit", dated=False)
    held, train = times[mask], times[~mask]
    n_before = int((held < train.max()).sum())
    assert round(n_before / len(held), 2) == audit
    assert (chron.n_held_rows, chron.n_held_earlier) == (len(held), n_before)
    for share in chron.earlier:  # each offered size, recounted from its own draw
        m, _ = chronological_holdout(times, groups, share.holdout, 0, "visit", dated=False)
        assert (share.n_held_rows, share.n_earlier) == (int(m.sum()),
                                                        int((times[m] < times[~m].max()).sum()))
    text = _split_sentence(chron)
    assert "scored on later data" not in text
    assert "held out whole" in text and "each unit's earlier rows included" in text
    assert f"`{n_before / len(held):.0%}` of the held-out rows (`{n_before:,}` of `{len(held):,}`)" in text
    assert "observed before the latest training row" in chron.sentence


def test_5_an_unplanned_share_is_described_without_a_number_and_ungrouped_rows_are_latest():
    from turbotab.core.seal import chronological_holdout

    times, groups = _visits_s9()
    _, chron = chronological_holdout(times, groups, 0.2, 0, "visit", dated=False)
    text = _split_sentence(chron, holdout=0.25)
    assert "some held-out rows can predate the latest training row" in text
    rng = np.random.default_rng(5)
    t = rng.uniform(0, 10, 500)
    mask, flat = chronological_holdout(t, None, 0.2, 0, "year", dated=False)
    assert t[mask].min() >= t[~mask].max()  # numpy: no held-out row is earlier than a training row
    text = _split_sentence(flat, group=None)
    assert "no held-out row is dated earlier than a training row" in text


# ── 6 · The lineage marks pass-throughs kept and attributes operations to what they touched ──


def _lineage_fixture() -> tuple[pd.DataFrame, Any, Any]:
    """B-skeptic/s15_lineage.py: the stratified log residual with blanks in ``fat_g``."""
    from turbotab.core.models.pipeline import DesignSpec, shared_steps, transformer

    rng = np.random.default_rng(2)
    n = 400
    E = np.exp(rng.normal(np.log(2000), 0.3, n))
    df = pd.DataFrame({"fat_g": 0.34 * E / 9 * np.exp(rng.normal(0, .15, n)), "energy_kcal": E,
                       "sex": rng.choice(["F", "M"], n), "smoker": rng.choice(["yes", "no", None], n)})
    df.loc[:30, "fat_g"] = np.nan
    energy = {"method": "residual", "energy_column": "energy_kcal", "nutrients": ["fat_g"],
              "log_transform": True, "strata": "sex"}
    spec = DesignSpec(predictors=["fat_g", "energy_kcal", "smoker"],
                      inputs=["fat_g", "energy_kcal", "sex", "smoker"],
                      categorical=["sex", "smoker"], numeric=["fat_g", "energy_kcal"], energy=energy,
                      impute=True, roles={"fat_g": "exposure", "energy_kcal": "energy",
                                          "smoker": "covariate"},
                      levels=["smoker"], indicators=True)
    pipe = transformer(shared_steps(spec))
    pipe.fit_transform(df[spec.inputs])
    return df, spec, pipe


def test_6_operations_are_attributed_only_to_the_columns_they_touched():
    """The imputer filled only ``fat_g`` (pandas: 31 blanks; ``energy_kcal`` and ``sex`` none), so
    only ``fat_g``'s edge into ``fat_g_adj`` says "imputed"; the encoder passed ``fat_g_adj``,
    ``energy_kcal`` and the indicator through unchanged (their values equal its inputs'), so those
    edges say "kept"."""
    from turbotab.core.models.lineage import missing_counts, trace

    df, spec, pipe = _lineage_fixture()
    blanks = df[spec.inputs].isna().sum()
    assert (int(blanks["fat_g"]), int(blanks["energy_kcal"]), int(blanks["sex"])) == (31, 0, 0)
    lineage = trace(pipe.steps, spec.inputs, spec.roles, missing_counts(df[spec.inputs]))
    op = {(link.source, link.target): link.operation for link in lineage.links}
    assert op[("raw:fat_g", "adj:fat_g_adj")] == "imputed, energy-adjusted (residual)"
    assert op[("raw:energy_kcal", "adj:fat_g_adj")] == "energy-adjusted (residual)"
    assert op[("raw:sex", "adj:fat_g_adj")] == "energy-adjusted (residual)"
    before = pipe[:-1].transform(df[spec.inputs])  # what the encoder received
    after = pipe.transform(df[spec.inputs])
    for column in ("fat_g_adj", "energy_kcal", "missingindicator_fat_g"):
        assert np.allclose(before[column].to_numpy(float), after[column].to_numpy(float))
        assert op[(f"adj:{column}", f"mx:{column}")] == "kept"
    assert op[("adj:smoker", "mx:smoker_yes")] == "one-hot, blank as a level"


def test_6_the_log_residual_names_the_geometric_mean():
    """Under log the reference energy is the geometric mean, exp(mean log E) over the fitting rows
    (numpy), not the arithmetic mean, and the formula says so with that number."""
    from turbotab.core.models.lineage import missing_counts, trace

    df, spec, pipe = _lineage_fixture()
    lineage = trace(pipe.steps, spec.inputs, spec.roles, missing_counts(df[spec.inputs]))
    formula = next(n.formula for n in lineage.nodes if n.id == "adj:fat_g_adj")
    geometric = float(np.exp(np.log(df["energy_kcal"]).mean()))
    arithmetic = float(df["energy_kcal"].mean())
    assert abs(geometric - arithmetic) > 20
    shown = float(re.search(r"geometric-mean energy_kcal \(([0-9.]+)\)", formula).group(1))
    assert shown == pytest.approx(geometric, rel=1e-4)


def test_6_a_density_sentence_claims_no_stratification():
    """Audit D15: a density is the same ratio within any level, so strata change nothing; the
    methods sentence no longer says "divided by `kcal` within levels of `sex`"."""
    for method in ("density", "density_multivariate"):
        text = voice.sentence_for(
            d.SetEnergyAdjustment(method=method, energy_column="kcal", nutrients=["fat_g"],
                                  strata="sex"),
            ProjectState(roles={"kcal": "energy", "fat_g": "exposure", "sex": "covariate"}))
        assert "within levels of" not in text
        assert "strata apply to the residual method only, so `sex` changed nothing" in text


# ── 7 · The survey pack's attenuation text passes a replay with λ = 0.70 ─────


def test_7_the_attenuation_text_replays_with_reliability_070():
    """F/sim_atten.py's design: X ~ N(0, 1), W = X + U with var(U) = 1/λ − 1 (reliability λ =
    0.70), Y = 0.30·X + e with var(Y) = 1. Closed form: the slope of Y on W is 0.30·λ = 0.210 and
    the correlation (the standardized coefficient) is 0.30·√λ = 0.251. The pack's numbers are read
    from its own text and must match both; the old "by approximately its reliability" with 0.25 for
    the standardized coefficient (0.25 / λ = 0.358, not 0.30) is gone."""
    lam, beta = 0.70, 0.30
    assert beta * lam == pytest.approx(0.210, abs=5e-4)
    assert beta * math.sqrt(lam) == pytest.approx(0.251, abs=5e-4)
    rng = np.random.default_rng(1)
    n = 400_000
    x = rng.normal(size=n)
    w = x + rng.normal(scale=math.sqrt(1 / lam - 1), size=n)
    y = beta * x + rng.normal(scale=math.sqrt(1 - beta ** 2), size=n)
    slope = np.cov(w, y)[0, 1] / np.var(w, ddof=1)
    standardized = np.corrcoef(w, y)[0, 1]
    assert slope == pytest.approx(0.210, abs=0.005)  # SE ≈ 0.0013
    assert standardized == pytest.approx(0.251, abs=0.005)
    pack = read_pack("CLINICAL_SURVEY_PACK.md")
    m = re.search(r"slope on the observed score shows up as about (0\.\d+) \(0\.30 × λ\)", pack)
    s = re.search(r"the correlation, as about (0\.\d+) \(0\.30 × √λ\)", pack)
    assert m and s, "the pack no longer states the slope and the standardized coefficient"
    assert float(m.group(1)) == pytest.approx(slope, abs=0.005)
    assert float(s.group(1)) == pytest.approx(standardized, abs=0.005)
    assert "divide a slope by\n> λ and a standardized coefficient by √λ" in pack
    assert "attenuates its estimated effect by approximately its reliability" not in pack


# ── 8 · The exclusion presets are attributed; the coach states r² ─────────────


def test_8_the_sex_specific_presets_are_attributed_and_count_what_pandas_counts():
    """Banna et al. 2017 quoting Willett 2013: women 500–3,500, men 800–4,000 kcal/d. Pan et al.
    2011 for NHS and HPFS: "<800 or >4200 kcal/d for men and <500 or >3500 kcal/d for women";
    de Koning et al. 2011 (HPFS, Diabetes Care 34:1150, PMC3114491): "implausible energy intake
    (<800 or >4200 kcal/day)". Each preset's count is pandas' over the rows with the outcome."""
    frame = _dietary()
    men_at = frame.index[frame["sex"] == "M"][:5]
    frame.loc[men_at, "energy_kcal"] = 4100.0  # between the two men's upper bounds: they differ here
    presets = {p["key"]: p for p in _presets(frame)["exclusions"]}
    women, men, e = frame["sex"] == "F", frame["sex"] == "M", frame["energy_kcal"]
    for key, high, name in (("willett_2013_by_sex", 4000, "Willett 2013"),
                            ("nhs_hpfs_by_sex", 4200, "NHS/HPFS")):
        p = presets[key]
        outside = (women & ((e < 500) | (e > 3500))) | (men & ((e < 800) | (e > high)))
        assert p["affected"] == int(outside.sum())
        assert p["label"].startswith(f"{name}, by sex: women 500–3,500 and men 800–{high:,}")
        assert p["rule"]["by"]["ranges"] == {"F": [500.0, 3500.0], "M": [800.0, float(high)]}
        assert int(rule_excludes(frame, d.ExclusionRule.model_validate(p["rule"])).sum()) == int(outside.sum())
    assert presets["willett_2013_by_sex"]["affected"] - presets["nhs_hpfs_by_sex"]["affected"] == 5
    rule = d.ExclusionRule.model_validate(presets["nhs_hpfs_by_sex"]["rule"])
    sentence = voice.sentence_for(d.SetExclusions(rules=[rule]), ProjectState(target="ldl"),
                                  {"frame": frame})
    assert sentence.endswith("(the Nurses' Health Study and Health Professionals Follow-up Study "
                             "cut-offs).")
    for value in ("willett_2013_by_sex", "nhs_hpfs_by_sex"):
        assert option("exclusions", value)


@pytest.mark.parametrize("r", [0.31, 0.45, 0.60, 0.70, 0.71, 0.80, -0.75])
def test_8_the_coach_states_r_squared_and_says_most_only_past_half(r):
    """MI-03: the share of the nutrient's variance its line on energy explains is r²; "most" only
    where it is more than half. Reference: r * r."""
    text = _coach_first(r)
    share = r * r
    assert f"`{share:.0%}`" in text
    assert ("most" in text.split(":")[1]) == (share > 0.5)
    assert voice.words(text) <= 12


def test_8_the_teaching_attributes_the_screens():
    body = section("exclusions", "The screens in circulation").body
    assert "Willett's textbook (2013) gives 500–3,500 kcal a day for women and 800–4,000 for men" in body
    assert "Health Professionals Follow-up Study (men) use 500–3,500 and 800–4,200" in body
    pack = read_pack("NUTRITION_PACK.md")
    assert "an allowable range of\n     800–4,000 kcal/day for men may be used" in pack
    assert "Willett / Nurses' Health Study **[CONVENTION]**" not in pack


# ── 9 · The pack citations, corrected ─────────────────────────────────────────


def test_9_nygaard_2016_is_cited_as_published():
    """Nygaard, Rødland & Hovig 2016, Biostatistics 17:29 (PMC4679072): "We re-analyzed the cited
    data (GEO: GSE40566) as described, detecting 2011 differentially expressed genes at 5% FDR.
    When, instead of batch adjusting using ComBat, we blocked for batch effect in limma (Smyth,
    2004), only 11 differentially expressed genes were detected"."""
    genomics = read_pack("GENOMICS_PACK.md")
    metabolomics = read_pack("METABOLOMICS_PACK.md")
    for pack in (genomics, metabolomics):
        assert "GSE40566" in pack and "2,011" in pack and "limma" in pack
    assert "| 1,000 vs 11 DE genes in Nygaard's example |" not in genomics
    assert "In their GSE61901 example the ComBat pipeline" not in genomics
    assert '"inadvertently exaggerate the differences observed."' not in metabolomics
    assert "from 11 genes to over 1,000" not in metabolomics


def test_9_zindler_2020_is_scoped_to_methylation_arrays():
    """Zindler et al. 2020, BMC Bioinformatics 21:271 (PMC7328269): "Using ComBat to correct for
    batch effects in randomly generated samples produced alarming numbers of false discovery rate
    (FDR) and Bonferroni-corrected (BF) false positive results in unbalanced as well as in balanced
    sample distributions", for simulated "Infinium HumanMethylation450 BeadChip (450 K) and
    Infinium MethylationEPIC BeadChip Kit (EPIC) DNAm data"."""
    metabolomics = read_pack("METABOLOMICS_PACK.md")
    assert "Zindler et al., *BMC Bioinformatics* 21:271 (2020)" in metabolomics
    assert "DNA\n  methylation microarray data" in metabolomics
    assert "a 2020 simulation of DNA methylation microarrays" in metabolomics


def test_9_eekhout_2014_threshold_is_the_abstracts():
    """Eekhout et al. 2014, J Clin Epidemiol 67:335: "when a large percentage of subjects had
    missing items (>25%), MI methods applied to the items outperformed methods applied to the
    total score"."""
    survey = read_pack("CLINICAL_SURVEY_PACK.md")
    assert "missing items (>25%), MI methods applied to the items outperformed" in survey
    assert "overestimated standard errors when >50% of participants had\n> missing data, though" not in survey
