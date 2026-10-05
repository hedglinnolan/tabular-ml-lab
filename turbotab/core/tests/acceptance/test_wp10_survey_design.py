"""WP10 · Survey design (docs/turbotab-next/audit/AUDIT_REPORT.md §5; closes ME-06 and the minor
D20/G20).

The package's three acceptance tests, in its order:

1. "With design columns present and purpose = inference, the app asks 'population or this sample';
   until design-based estimation exists, a recorded attestation ('unweighted, sample-only estimand;
   standard errors ignore strata and PSUs') is required and appears in the methods sentence."
   Design-based estimation now exists, so the attestation is what the "these participants" answer
   records; until the question is answered, the inference table is blocked.
2. "Design-based estimation on the informative-weight fixture reproduces the reference: DR1TFIBE
   +0.0065–0.0066 (about 0.000–0.013), matched against an independent implementation (for example
   R ``survey::svyglm``)."
3. "Exclusions under a design become domain flags (rows kept for variance); lonely PSUs are handled
   and reported; pooled 1999–2002 data use the four-year weights. Source check: NHANES Analytic
   Guidelines 2011–2016."

Every project is driven through the real server, as the auditors drove it. References come from
paths independent of ``turbotab/core/models/survey.py``: samplics's ``SurveyGLM`` (a separate
Taylor-linearization implementation, test-only dependency), the definitions written out in
``survey_references.py`` (R's ``svyrecvar`` semantics and Stata's equation (1); R is not installed),
statsmodels for the unweighted and the audit's own cluster-robust numbers, and scipy's t quantiles.

**Source check** (NHANES Analytic Guidelines 2011–2016, https://wwwn.cdc.gov/nchs/data/nhanes/
analyticguidelines/11-16-analytic-guidelines.pdf, read for this package):

* §3: "The complex survey design used for NHANES, including oversampling, stratification, and
  clustering, must be considered when analyzing the data for appropriate variance estimation and to
  calculate statistics representative of the U.S. civilian non-institutionalized population."
* §3.2.3.1: "the entire set of data containing the appropriate weights for a particular survey cycle
  must be used to obtain the correct variance estimates. The estimation procedure must indicate
  which records are in the subgroup of interest."
* §3.2.3.2: "The nominal degrees of freedom can be approximated using the stratum and PSU variables
  on the data file by subtracting the number of strata from the number of PSUs. If an analysis is
  performed on a subgroup of cases, the degrees of freedom should be based on the number of strata
  and PSUs containing the observations of interest."
* §3.1.3: "When combining two or more 2-year cycles from 2001–2002 onward, new multi-year sample
  weights can be computed by simply dividing the 2-year sample weights by the number of 2-year
  cycles in the analysis." Table F: "4 years 1999-2002 Provided on the Public-use Data Files"; "6
  years 1999-2004 If sddsrvyr in (1,2) then MEC6YR = 2/3 * WTMEC4YR; … If sddsrvyr=3 then MEC6YR =
  1/3 * WTMEC2YR".
* §3.1.4: "When combining data from the 1999-2000 NHANES cycle with other cycles, it is recommended
  that the 4-year sample weights be used for 1999-2002 and the 2-year sample weights be used for
  other cycles."
* §3.2.1: "Variance estimates computed using standard statistical software packages that assume
  simple random sampling are generally too low (i.e., significance levels are overstated)".
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.survey import ATTESTATION, read_design
from turbotab.core.tests.acceptance import survey_references as ref
from turbotab.server.tests.conftest import make_client, prepare, wait_for

T975 = 0.975


# ── the audit's fixtures, replayed ───────────────────────────────────────────


def informative_tables(folder: Path) -> dict[str, Path]:
    """The two informative-weight tables the audit ran (``repro.tar.gz``: ``fxI/make.py``, seed 42,
    table F5; ``fxIS/make.py`` = ``I-skeptic/mkfx.py``, seed 2026, table I4), each replayed draw for
    draw from its generator's one stream. Both regenerate the audit's CSVs byte for byte (checked
    against the extracted archive when this test was written). A 50/50 sample from a 10/90
    population: the oversampled group has a fiber slope of −0.10, the rest +0.02, so the weighted
    (population) slope is positive and the unweighted one negative."""
    rng = np.random.default_rng(42)
    n = 600  # F2, F3, F4: drawn as the generator draws them, so F5 gets the audit's numbers
    age = rng.normal(55, 8, n).round(0)
    fiber = rng.gamma(4, 5, n).round(1)
    _ = rng.exponential(1 / (0.02 * np.exp(0.04 * (age - 55) - 0.03 * (fiber - 20))))
    _ = rng.uniform(1, 15, n)
    _ = rng.normal(0, 1, n)
    sites = rng.integers(0, 12, n)
    site_eff = rng.normal(0, 1.0, 12)[sites]
    _ = rng.normal(0, 1, n) + 0.8 * site_eff
    _ = rng.normal(0, 1, n)
    m = 2000
    grp = rng.random(m) < 0.5
    w = np.where(grp, 0.2, 1.8) * 10000
    psu, strata = rng.integers(1, 3, m), rng.integers(1, 15, m)
    kcal, fib = rng.normal(2100, 500, m), rng.gamma(4, 5, m)
    crp = 3 + np.where(grp, -0.10, 0.02) * (fib - 20) + 0.4 * grp + rng.normal(0, 1, m)
    f5 = pd.DataFrame({"SEQN": np.arange(1, m + 1), "DR1TKCAL": kcal.round(1), "DR1TFIBE": fib.round(2),
                       "WTDRD1": w.round(2), "SDMVSTRA": strata, "SDMVPSU": psu, "LBXCRP": crp.round(3)})

    rng = np.random.default_rng(2026)
    n = 4000  # I1, I3: drawn as the generator draws them, so I4 gets the audit's numbers
    _ = rng.normal(50, 10, n), rng.gamma(4, 5, n), rng.normal(0, 5, n)
    n = 3000
    entry = rng.uniform(0, 14, n)
    _ = rng.normal(0, 4, n)
    age3 = rng.normal(55, 8, n)
    _ = rng.exponential(1 / (0.03 * np.exp(0.04 * (age3 - 55))))
    m = 2400
    grp = rng.random(m) < 0.5
    w = np.where(grp, 0.25, 1.75) * 9000
    strata, psu = rng.integers(1, 16, m), rng.integers(1, 3, m)
    kcal, fib = rng.normal(2100, 500, m), rng.gamma(4, 5, m)
    crp = 3 + np.where(grp, -0.10, 0.02) * (fib - 20) + 0.4 * grp + rng.normal(0, 1, m)
    i4 = pd.DataFrame({"SEQN": np.arange(1, m + 1), "DR1TKCAL": kcal.round(1), "DR1TFIBE": fib.round(2),
                       "WTDRD1": w.round(2), "SDMVSTRA": strata, "SDMVPSU": psu, "LBXCRP": crp.round(3)})
    out = {"F5": folder / "survey_weighted.csv", "I4": folder / "survey.csv"}
    f5.to_csv(out["F5"], index=False)
    i4.to_csv(out["I4"], index=False)
    return out


def subgroup_table(folder: Path) -> Path:
    """12 strata × 2 PSUs × 60 people. In strata 1–4 the second PSU holds only people aged 40 or
    more, so restricting to ages 20–39 leaves those PSUs with no analysis row: deleting the rows
    would leave four strata with one PSU, keeping them as a domain does not."""
    rng = np.random.default_rng(1010)
    rows = []
    for h in range(1, 13):
        for i in (1, 2):
            effect = rng.normal(0, 0.6)
            for _ in range(60):
                old_only = h <= 4 and i == 2
                age = rng.uniform(40, 80) if old_only else rng.uniform(20, 80)
                x = rng.normal(10, 3)
                rows.append({"age": round(age, 1), "x": round(x, 3), "h": h, "i": i,
                             "w": round(rng.uniform(500, 3000) * (1.6 if age < 40 else 1.0), 1),
                             "y": round(1 + 0.3 * x + 0.02 * age + effect + rng.normal(0, 1), 3)})
    frame = pd.DataFrame(rows)
    frame.insert(0, "SEQN", np.arange(1, len(frame) + 1))
    frame = frame.rename(columns={"h": "SDMVSTRA", "i": "SDMVPSU", "w": "WTMEC2YR"})
    path = folder / "subgroup.csv"
    frame.to_csv(path, index=False)
    return path


def lonely_table(folder: Path) -> Path:
    """10 strata × 2 PSUs and an 11th stratum with a single PSU, 40 people each."""
    rng = np.random.default_rng(1111)
    rows = []
    for h in range(1, 12):
        for i in ((1,) if h == 11 else (1, 2)):
            effect = rng.normal(0, 0.8)
            for _ in range(40):
                x = rng.normal(5, 2)
                rows.append({"x": round(x, 3), "SDMVSTRA": h, "SDMVPSU": i,
                             "WTMEC2YR": round(rng.uniform(800, 4000), 1),
                             "y": round(2 - 0.4 * x + effect + rng.normal(0, 1), 3)})
    frame = pd.DataFrame(rows)
    frame.insert(0, "SEQN", np.arange(1, len(frame) + 1))
    path = folder / "lonely.csv"
    frame.to_csv(path, index=False)
    return path


def pooled_table(folder: Path, cycles: tuple[int, ...], four_year: bool = True) -> Path:
    """NHANES-shaped rows from the given ``SDDSRVYR`` cycles: each cycle its own 6 strata × 2 PSUs
    (masked variance units differ by cycle), a 2-year exam weight on every row, and the 1999–2002
    four-year weight on the 1999–2000 and 2001–2002 rows. The 1999–2000 two-year weights rest on
    the 1990 census and run 1.6 times too high against 2001–2002's (the reason §3.1.4 gives), and
    the outcome's slope differs by cycle, so the weight used changes the estimate."""
    rng = np.random.default_rng(1999 + sum(cycles))
    rows = []
    for c in cycles:
        for h in range(1, 7):
            for i in (1, 2):
                effect = rng.normal(0, 0.5)
                for _ in range(45):
                    x = rng.normal(20, 6)
                    w2 = rng.uniform(5000, 30000) * (1.6 if c == 1 else 1.0)
                    w4 = w2 / (1.6 if c == 1 else 1.0) / 2 * rng.uniform(0.95, 1.05)
                    rows.append({"SDDSRVYR": c, "SDMVSTRA": 100 * c + h, "SDMVPSU": i,
                                 "WTMEC2YR": round(w2, 1),
                                 "WTMEC4YR": round(w4, 1) if c in (1, 2) else np.nan,
                                 "x": round(x, 2),
                                 "y": round(1 + (0.05 if c == 1 else -0.03) * x + effect
                                            + rng.normal(0, 1), 3)})
    frame = pd.DataFrame(rows)
    if not four_year:
        frame = frame.drop(columns=["WTMEC4YR"])
    frame.insert(0, "SEQN", np.arange(1, len(frame) + 1))
    path = folder / f"pooled_{'_'.join(map(str, cycles))}{'' if four_year else '_no4yr'}.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("wp10_tables")
    return {**informative_tables(folder), "subgroup": subgroup_table(folder),
            "lonely": lonely_table(folder), "pooled_123": pooled_table(folder, (1, 2, 3)),
            "pooled_12": pooled_table(folder, (1, 2)), "pooled_23": pooled_table(folder, (2, 3)),
            "pooled_no4yr": pooled_table(folder, (1, 2, 3), four_year=False)}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("wp10_home"), "local", 2, "http://127.0.0.1") as c:
        yield c


# ── driving the server ───────────────────────────────────────────────────────


def post(client, pid: str, decision: dict) -> tuple[int, dict]:
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    return response.status_code, response.json()


def accepted(client, pid: str, decision: dict) -> dict:
    """Record ``decision`` after the questions before it; the readings it asks about are answered
    from the table's declared truth (``TABLE_TRUTH``; BLUEPRINT §14.3: never a constant)."""
    from turbotab.server.tests.conftest import answer_settled

    prepare(client, pid, decision)
    response = answer_settled(client, pid, None, decision)
    assert response.status_code == 200, response.json()
    return response.json()


# The informative-weight tables' truth (their generator, ``informative_tables``): DR1TKCAL is one
# day's simulated intake in kcal; the design codes are the design's.
TABLE_TRUTH = {"unit:DR1TKCAL": "kcal", "day_count:DR1TKCAL": "1",
               "code_or_count:SDMVSTRA": "code", "code_or_count:SDMVPSU": "code",
               # WP17: total energy shares the diet's common causes with fiber, so it is adjusted
               # for (the generator draws them independently; the author adjusts as the field does)
               "adjust:DR1TKCAL": "unknown,unknown,no",
               # the subgroup table's age sets y only (``subgroup_table``)
               "adjust:age": "no,yes,no"}


def open_project(client, path: Path, target: str, purpose: str, roles: dict[str, str]) -> str:
    response = client.post("/api/projects", json={"path": str(path)})
    assert response.status_code == 200, response.text
    pid = response.json()["id"]
    from turbotab.server.tests.conftest import declare

    declare(pid, TABLE_TRUTH, fixture=path.name)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    accepted(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    accepted(client, pid, {"kind": "set_target", "column": target})
    accepted(client, pid, {"kind": "set_purpose", "purpose": purpose})
    # The readings ledger (BLUEPRINT §14.1): a role recorded exactly as a proposal below high
    # confidence is confirmed, with the role its author gave it (``answer_settled``).
    accepted(client, pid, {"kind": "set_roles", "roles": roles})
    return pid


def step(client, pid: str, key: str) -> dict:
    deadline = time.monotonic() + 60
    while True:
        view = client.get(f"/api/projects/{pid}").json()
        found = next(s for s in view["interview"] if s["key"] == key)
        if found["status"] != "waiting" or time.monotonic() > deadline:
            return found
        time.sleep(0.05)


def survey_options(client, pid: str) -> list[dict]:
    wait_for(client, pid, {"proposals": "fresh"}, timeout=60)
    proposal = client.get(f"/api/projects/{pid}/stages/proposals").json()["artifact"]["survey"]
    return proposal["options"]


def finish(client, pid: str, *, rules: list | None = None, holdout: float = 0.2) -> dict:
    """The questions after the survey answer, then the linear model; the fit's linear model."""
    accepted(client, pid, {"kind": "set_exclusions", "rules": rules or []})
    accepted(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    accepted(client, pid, {"kind": "set_split", "holdout": holdout, "seed": 0, "folds": 5})
    accepted(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    return linear_model(client, pid)


def linear_model(client, pid: str) -> dict:
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    return next(m for m in fit["models"] if m["family"] == "linear")


def sentence(client, pid: str, kind: str) -> str:
    records = client.get(f"/api/projects/{pid}").json()["decisions"]
    return [r for r in records if r["decision"]["kind"] == kind][-1]["sentence"]


def row(model: dict, feature: str) -> dict:
    from turbotab.core.tests.acceptance.server_drive import every_row

    return next(c for c in every_row(model) if c["feature"] == feature)


DIET_ROLES = {"SEQN": "identifier", "DR1TKCAL": "covariate", "DR1TFIBE": "exposure",
              "WTDRD1": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}


def samplics_glm(model: str, y, X, weight, strata, psu):
    """samplics's own Taylor linearization (``SurveyGLM``): estimates and standard errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # samplics announces that it is archived
        from samplics.regression import SurveyGLM
        from samplics.utils.types import ModelType

        glm = SurveyGLM(model=ModelType.LINEAR if model == "linear" else ModelType.LOGISTIC)
        glm.estimate(y=np.asarray(y, float), x=np.asarray(X, float), samp_weight=np.asarray(weight, float),
                     stratum=np.asarray(strata), psu=np.asarray(psu), add_intercept=True)
    return np.asarray(glm.beta["point_est"]), np.asarray(glm.beta["stderror"])


# ── 1 · the question, the block and the attestation ──────────────────────────


def test_1_the_app_asks_population_or_this_sample_and_records_the_attestation(client, tables):
    """With ``WTDRD1``, ``SDMVSTRA`` and ``SDMVPSU`` present and purpose = inference, the Router
    opens the survey question after the roles, offering "surveyed population" (one option per
    weight the names read as) and "these participants". Answering "these participants" records the
    attestation word for word in the decision's sentence (the methods sentence), and the linear
    model's table carries it as a concern; its coefficients are the unweighted ones (statsmodels OLS with
    HC3, an independent path: −0.0414 for fiber, the audit's unweighted number). The same table
    under prediction is not asked (the scores say they are unweighted), and a table with no weight
    column is not asked either.
    """
    import statsmodels.api as sm

    pid = open_project(client, tables["I4"], "LBXCRP", "inference", DIET_ROLES)
    asked = step(client, pid, "survey")
    assert asked["status"] == "open", asked
    options = survey_options(client, pid)
    assert [o["key"] for o in options] == ["population:WTDRD1", "sample"]
    assert options[0]["decision"] == {"kind": "set_survey", "estimand": "population",
                                      "weight": "WTDRD1", "strata": "SDMVSTRA", "psu": "SDMVPSU",
                                      "cycle": None, "four_year_weight": None, "acknowledged": False}
    # The next question waits behind it: the order is held server-side.
    code, body = post(client, pid, {"kind": "set_exclusions", "rules": []})
    assert code == 409 and body["error"]["code"] == "not_yet", body
    accepted(client, pid, options[1]["decision"])
    said = sentence(client, pid, "set_survey")
    assert ATTESTATION == "unweighted, sample-only estimand; standard errors ignore strata and PSUs"
    assert ATTESTATION in said, said
    assert "these participants, not the surveyed population" in said
    model = finish(client, pid, holdout=0.0)
    assert any(ATTESTATION in c for c in model["concerns"]), model["concerns"]
    frame = pd.read_csv(tables["I4"])
    ols = sm.OLS(frame["LBXCRP"], sm.add_constant(frame[["DR1TKCAL", "DR1TFIBE"]])).fit(cov_type="HC3")
    fiber = row(model, "DR1TFIBE")
    assert fiber["estimate"] == pytest.approx(float(ols.params["DR1TFIBE"]), abs=1e-10)
    assert round(fiber["estimate"], 4) == -0.0414
    assert model["inference"]["covariance"] == "HC3"

    # Under prediction: not asked, and every model says its scores are unweighted.
    pid = open_project(client, tables["I4"], "LBXCRP", "prediction", DIET_ROLES)
    skipped = step(client, pid, "survey")
    assert skipped["status"] == "not_applicable" and "not weighted" in skipped["reason"]
    model = finish(client, pid)
    assert any("Scores are unweighted" in c for c in model["concerns"]), model["concerns"]

    # No weight column beside the strata and PSU: a body weight called `weight` is not a survey
    # weight, so no population option is offered; but the table names its design, so the question
    # is still asked (gate repair: a half-read design took the unweighted estimand without a word,
    # BLUEPRINT §11.3), and "these participants" is recorded with its attestation, never assumed.
    frame.drop(columns=["WTDRD1"]).rename(columns={"DR1TKCAL": "weight"}).to_csv(
        tables["I4"].with_name("no_weight.csv"), index=False)
    roles = {"SEQN": "identifier", "weight": "covariate", "DR1TFIBE": "exposure",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    pid = open_project(client, tables["I4"].with_name("no_weight.csv"), "LBXCRP", "inference", roles)
    assert step(client, pid, "survey")["status"] == "open"
    assert [o["key"] for o in survey_options(client, pid)] == ["sample"]
    # With no design column at all, the question does not apply.
    frame.drop(columns=["WTDRD1", "SDMVSTRA", "SDMVPSU"]).rename(
        columns={"DR1TKCAL": "weight"}).to_csv(tables["I4"].with_name("no_design.csv"), index=False)
    roles = {"SEQN": "identifier", "weight": "covariate", "DR1TFIBE": "exposure"}
    pid = open_project(client, tables["I4"].with_name("no_design.csv"), "LBXCRP", "inference", roles)
    assert step(client, pid, "survey")["status"] == "not_applicable"


def test_1_until_it_is_answered_the_inference_table_is_blocked(client, tables):
    """The fit never serves an unweighted table silently under inference. A project fit under
    prediction, whose purpose is then changed to inference, reopens the survey question (changing
    an earlier answer never answers a later one), and its refit linear model reports no coefficient:
    the table is refused with the reason and the way forward until the question is answered
    (block and record, BLUEPRINT §11.3)."""
    pid = open_project(client, tables["I4"], "LBXCRP", "prediction", DIET_ROLES)
    finish(client, pid)
    code, body = post(client, pid, {"kind": "set_purpose", "purpose": "inference"})
    assert code == 200, body
    assert step(client, pid, "survey")["status"] == "open"
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    model = linear_model(client, pid)
    assert model["coefficients"] == []
    refused = model["inference"]["refused"]
    assert "surveyed population or these participants is not answered" in refused
    assert model["inference"]["exits"] and model["concerns"][0] == refused
    assert model["inference"]["covariance"] == "none"


# ── 2 · the informative-weight fixture ───────────────────────────────────────


@pytest.mark.parametrize("fixture", ["F5", "I4"])
def test_2_the_design_based_estimate_reproduces_the_reference(client, tables, fixture):
    """The audit's fixture under "surveyed population": DR1TFIBE **+0.0065–0.0066**, against the
    app's unweighted −0.039 to −0.041 (a sign flip).

    References: samplics's ``SurveyGLM`` (estimates and standard errors to 10⁻⁹ relative) and the
    definition written out (``survey_references.wls_by_definition``); the design degrees of freedom
    counted from the raw file (PSUs minus strata: 28 − 14 = 14 and 30 − 15 = 15); the interval
    ``β̂ ± t(df)·SE`` from scipy. The default 20% holdout is drawn and the table still uses every
    analysis row (BLUEPRINT §12 ruling 3: under inference a holdout is a prediction concept).

    On the interval: the audit's "about 0.000–0.013" (0.0000–0.0132 and 0.0006–0.0125) came from a
    weighted least-squares fit with cluster-robust errors by PSU-within-stratum and normal critical
    values (statsmodels' ``cov_type="cluster"``, the script ``I-skeptic/i4.py``), reproduced here
    to 10⁻⁴. The design-based interval centers the PSU totals within strata and uses t on the
    design degrees of freedom, as NCHS directs, so it is wider: −0.0013 to 0.0145 and −0.0009 to
    0.0140. The point estimate, the reference the package names, is the same in both.
    """
    import statsmodels.api as sm

    frame = pd.read_csv(tables[fixture])
    pid = open_project(client, tables[fixture], "LBXCRP", "inference", DIET_ROLES)
    accepted(client, pid, survey_options(client, pid)[0]["decision"])
    said = sentence(client, pid, "set_survey")
    G = int(frame.groupby("SDMVSTRA")["SDMVPSU"].nunique().sum())
    H = int(frame["SDMVSTRA"].nunique())
    assert f"(`{G}` PSUs in `{H}` strata)" in said, said
    assert "Taylor series linearization" in said
    # Nothing is restricted when the survey is answered, so its sentence claims no domain analysis
    # (repair round: it said "a restriction keeps every row in the design" whatever was restricted;
    # the exclusions and missing-values sentences say it where they restrict, test 3).
    assert "domain analysis" not in said
    model = finish(client, pid)  # the usual 20% holdout
    info = model["inference"]
    assert info["covariance"] == "design" and info["survey"]["n_domain"] == len(frame)

    X = frame[["DR1TKCAL", "DR1TFIBE"]].to_numpy(float)
    beta, se = samplics_glm("linear", frame["LBXCRP"], X, frame["WTDRD1"], frame["SDMVSTRA"],
                            frame["SDMVPSU"])
    Xc = np.column_stack([np.ones(len(frame)), X])
    b_def, V_def = ref.wls_by_definition(Xc, frame["LBXCRP"].to_numpy(float),
                                         frame["WTDRD1"].to_numpy(float), frame["SDMVSTRA"],
                                         frame["SDMVPSU"], np.ones(len(frame), bool))
    df = ref.design_df_by_definition(frame["SDMVSTRA"], frame["SDMVPSU"], [True] * len(frame))
    assert df == G - H == {"F5": 14, "I4": 15}[fixture]
    q = stats.t.ppf(T975, df)
    for j, feature in enumerate(["(intercept)", "DR1TKCAL", "DR1TFIBE"]):
        got = row(model, feature)
        assert got["estimate"] == pytest.approx(beta[j], rel=1e-9) == pytest.approx(b_def[j], rel=1e-9)
        assert got["se"] == pytest.approx(se[j], rel=1e-9)
        assert got["se"] == pytest.approx(float(np.sqrt(V_def[j, j])), rel=1e-9)
        assert got["df"] == df
        assert got["ci_low"] == pytest.approx(beta[j] - q * se[j], rel=1e-9)
        assert got["ci_high"] == pytest.approx(beta[j] + q * se[j], rel=1e-9)
    fiber = row(model, "DR1TFIBE")
    assert 0.0065 <= round(fiber["estimate"], 4) <= 0.0066  # the package's reference
    unweighted = sm.OLS(frame["LBXCRP"], sm.add_constant(frame[["DR1TKCAL", "DR1TFIBE"]])).fit()
    assert unweighted.params["DR1TFIBE"] < -0.035 < 0 < fiber["estimate"]  # the sign flip ME-06 found
    # The audit's approximate interval, by its own method, and why ours is wider.
    wls = sm.WLS(frame["LBXCRP"], sm.add_constant(frame[["DR1TKCAL", "DR1TFIBE"]]),
                 weights=frame["WTDRD1"]).fit(
        cov_type="cluster", cov_kwds={"groups": (frame["SDMVSTRA"] * 10 + frame["SDMVPSU"]).to_numpy()})
    audit = wls.conf_int().loc["DR1TFIBE"].round(4).tolist()
    assert audit == {"F5": [0.0, 0.0132], "I4": [0.0006, 0.0125]}[fixture]
    assert (round(fiber["ci_low"], 4), round(fiber["ci_high"], 4)) == \
        {"F5": (-0.0013, 0.0145), "I4": (-0.0009, 0.014)}[fixture]


def test_2_logistic_and_multinomial_tables_match_their_references(tmp_path):
    """The same estimator for a binary and a three-level outcome on the I4 table (CRP above 3 mg/L;
    CRP in thirds), through the linear family's own inference entry point. Binary: samplics's
    ``SurveyGLM(LOGISTIC)`` and the definition by a general-purpose optimizer
    (``survey_references.logistic_by_definition``). Multinomial: samplics has no multinomial model,
    so the definition alone (``multinomial_by_definition``: BFGS on the weighted likelihood, scores and
    information by explicit loops over rows)."""
    from turbotab.core.models.survey import build_design, survey_table

    frame = pd.read_csv(informative_tables(tmp_path)["I4"])
    frame.index = pd.Index(np.arange(len(frame)), name="row_id")
    design = build_design(frame, frame["WTDRD1"].to_numpy(float), weight_column="WTDRD1",
                          strata_column="SDMVSTRA", psu_column="SDMVPSU")
    X = frame[["DR1TKCAL", "DR1TFIBE"]]
    Xc = np.column_stack([np.ones(len(frame)), X.to_numpy(float)])
    everyone = np.ones(len(frame), bool)
    event = (frame["LBXCRP"] > 3).astype(int).to_numpy()
    table = survey_table("binary", X, event, [0, 1], design)
    beta, se = samplics_glm("logistic", event, X, frame["WTDRD1"], frame["SDMVSTRA"], frame["SDMVPSU"])
    b_def, V_def = ref.logistic_by_definition(Xc, event.astype(float), frame["WTDRD1"].to_numpy(float),
                                              frame["SDMVSTRA"], frame["SDMVPSU"], everyone)
    for j, got in enumerate(table.rows):
        assert got["estimate"] == pytest.approx(beta[j], rel=1e-7, abs=1e-12)
        assert got["estimate"] == pytest.approx(b_def[j], rel=1e-5, abs=1e-9)
        assert got["se"] == pytest.approx(se[j], rel=1e-7)
        assert got["se"] == pytest.approx(float(np.sqrt(V_def[j, j])), rel=1e-5)
        assert got["df"] == 15

    thirds = pd.qcut(frame["LBXCRP"], 3, labels=["low", "mid", "high"]).astype(str).to_numpy()
    levels = ["high", "low", "mid"]
    table = survey_table("multiclass", X, thirds, levels, design)
    codes = pd.Categorical(thirds, categories=levels).codes.astype(np.int64)
    theta, V = ref.multinomial_by_definition(Xc, codes, 3, frame["WTDRD1"].to_numpy(float),
                                             frame["SDMVSTRA"], frame["SDMVPSU"], everyone)
    assert [r["feature"] for r in table.rows][:4] == ["(intercept) [low]", "DR1TKCAL [low]",
                                                     "DR1TFIBE [low]", "(intercept) [mid]"]
    for j, got in enumerate(table.rows):
        assert got["estimate"] == pytest.approx(theta[j], rel=1e-4, abs=1e-8)
        assert got["se"] == pytest.approx(float(np.sqrt(V[j, j])), rel=1e-4)


# ── 3 · domains, lonely PSUs, pooled cycles ──────────────────────────────────


def test_3_exclusions_under_a_design_are_domains(client, tables):
    """An eligibility rule (ages 20–39) under "surveyed population" leaves every row in the design:
    the estimate is the domain's, and the variance still sees every stratum and PSU. Four PSUs hold
    no one aged 20–39, so deleting the rows instead would leave four strata with a single PSU, which
    samplics refuses outright (its default ``single_psu``: error), and a design-based variance from
    the deleted file would be a different, wrong number.

    References: samplics with the weights of rows outside the domain set to zero (R's ``subset()``
    on a design does exactly this), and the definition written out; the degrees of freedom by the
    NCHS rule, counting only PSUs and strata holding domain rows: 20 − 12 = 8. The exclusions'
    sentence says the rows stay in the variance.
    """
    frame = pd.read_csv(tables["subgroup"])
    roles = {"SEQN": "identifier", "age": "covariate", "x": "exposure", "WTMEC2YR": "design",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    pid = open_project(client, tables["subgroup"], "y", "inference", roles)
    accepted(client, pid, survey_options(client, pid)[0]["decision"])
    rule = {"column": "age", "low": 20, "high": 39.99, "reason": "young adults"}
    model = finish(client, pid, rules=[rule])
    assert "keep their strata and PSUs in the variance (a domain analysis)" in \
        sentence(client, pid, "set_exclusions")
    # Said again when the survey is answered after a restriction (the voice, on that state).
    from turbotab.core import voice
    from turbotab.core.decisions import ProjectState, SetSurvey

    restricted = ProjectState(target="y", purpose="inference",
                              exclusions=[{"kind": "range", **rule}])
    after = voice.sentence_for(SetSurvey(estimand="population", weight="WTMEC2YR",
                                         strata="SDMVSTRA", psu="SDMVPSU"), restricted)
    assert after.endswith("stay in the design for the variance (a domain analysis).")
    unrestricted = voice.sentence_for(SetSurvey(estimand="population", weight="WTMEC2YR",
                                                strata="SDMVSTRA", psu="SDMVPSU"),
                                      ProjectState(target="y", purpose="inference"))
    assert "domain analysis" not in unrestricted

    domain = frame["age"].between(20, 39.99).to_numpy()
    assert frame.loc[domain].groupby("SDMVSTRA")["SDMVPSU"].nunique().eq(1).sum() == 4
    info = model["inference"]["survey"]
    assert (info["n_design"], info["n_domain"]) == (len(frame), int(domain.sum()))
    assert (info["n_psu"], info["n_strata"], info["domain_psu"], info["domain_strata"]) == (24, 12, 20, 12)
    assert info["df"] == ref.design_df_by_definition(frame["SDMVSTRA"], frame["SDMVPSU"], domain) == 8
    assert info["lonely_strata"] == []
    assert any("domain analysis" in c for c in model["concerns"]), model["concerns"]

    X = frame[["age", "x"]].to_numpy(float)
    zero_outside = np.where(domain, frame["WTMEC2YR"], 0.0)
    beta, se = samplics_glm("linear", frame["y"], X, zero_outside, frame["SDMVSTRA"], frame["SDMVPSU"])
    Xc = np.column_stack([np.ones(len(frame)), X])
    b_def, V_def = ref.wls_by_definition(Xc, frame["y"].to_numpy(float), frame["WTMEC2YR"].to_numpy(float),
                                         frame["SDMVSTRA"], frame["SDMVPSU"], domain)
    for j, feature in enumerate(["(intercept)", "age", "x"]):
        got = row(model, feature)
        assert got["estimate"] == pytest.approx(beta[j], rel=1e-9)
        assert got["se"] == pytest.approx(se[j], rel=1e-9)
        assert got["se"] == pytest.approx(float(np.sqrt(V_def[j, j])), rel=1e-9)
        assert got["df"] == 8
    # Deleting the rows is a different design: samplics refuses its lonely strata, and a variance
    # from the deleted file ("remove" for the four strata) is not the domain's.
    kept = frame.loc[domain]
    with pytest.raises(Exception):
        samplics_glm("linear", kept["y"], kept[["age", "x"]], kept["WTMEC2YR"], kept["SDMVSTRA"],
                     kept["SDMVPSU"])
    _, V_deleted = ref.wls_by_definition(np.column_stack([np.ones(len(kept)), kept[["age", "x"]]]),
                                         kept["y"].to_numpy(float), kept["WTMEC2YR"].to_numpy(float),
                                         kept["SDMVSTRA"], kept["SDMVPSU"], np.ones(len(kept), bool),
                                         lonely="remove")
    assert abs(np.sqrt(V_deleted[2, 2]) / row(model, "x")["se"] - 1) > 0.02


def test_3_a_lonely_psu_is_handled_and_reported(client, tables):
    """Stratum 11 has a single PSU in the design. The table is still reported: that PSU's total is
    centered at the mean of all PSU totals with ``n_h/(n_h − 1)`` taken as 1 (R's
    ``survey.lonely.psu = "adjust"``, Stata's ``singleunit(centered)``), the stratum is named in a
    concern and in the table's design record, and the survey sentence says how it was handled.

    Reference: the definition written out with R's documented "adjust" rule
    (``survey_references.design_variance_by_definition``; samplics offers only error, skip,
    certainty and combine). Conservative: the standard errors are at least those that leave the
    stratum out (R's "remove"), since the added term is positive semi-definite. Degrees of freedom:
    21 PSUs − 11 strata = 10.
    """
    frame = pd.read_csv(tables["lonely"])
    roles = {"SEQN": "identifier", "x": "exposure", "WTMEC2YR": "design", "SDMVSTRA": "design",
             "SDMVPSU": "design"}
    pid = open_project(client, tables["lonely"], "y", "inference", roles)
    accepted(client, pid, survey_options(client, pid)[0]["decision"])
    assert "`1` stratum with a single PSU was centered at the mean of all PSU totals" in \
        sentence(client, pid, "set_survey")
    model = finish(client, pid)
    info = model["inference"]["survey"]
    assert info["lonely_strata"] == ["11"] and info["lonely_method"] == "centered"
    assert info["df"] == 10
    assert any("single PSU (`11`)" in c for c in model["concerns"]), model["concerns"]

    Xc = np.column_stack([np.ones(len(frame)), frame[["x"]].to_numpy(float)])
    everyone = np.ones(len(frame), bool)
    w, y = frame["WTMEC2YR"].to_numpy(float), frame["y"].to_numpy(float)
    b, V_adjust = ref.wls_by_definition(Xc, y, w, frame["SDMVSTRA"], frame["SDMVPSU"], everyone)
    _, V_remove = ref.wls_by_definition(Xc, y, w, frame["SDMVSTRA"], frame["SDMVPSU"], everyone,
                                        lonely="remove")
    for j, feature in enumerate(["(intercept)", "x"]):
        got = row(model, feature)
        assert got["estimate"] == pytest.approx(b[j], rel=1e-9)
        assert got["se"] == pytest.approx(float(np.sqrt(V_adjust[j, j])), rel=1e-9)
        assert got["se"] >= float(np.sqrt(V_remove[j, j]))


def test_3_pooled_1999_2002_cycles_use_the_four_year_weights(client, tables):
    """Pooling 1999–2000, 2001–2002 and 2003–2004 (``SDDSRVYR`` 1, 2, 3): the survey question's
    option names the cycle column and ``WTMEC4YR``, and the estimate uses 2 × ``WTMEC4YR`` ÷ 3 on
    the 1999–2002 rows and ``WTMEC2YR`` ÷ 3 on the 2003–2004 rows (§3.1.4 and Table F, built here
    from the guideline's formula, independently of the app). Pooling 1999–2000 without the four-year
    weight is refused with the guideline's reason; a table pooling only 1999–2002 uses ``WTMEC4YR``
    itself, and one pooling 2001–2002 with 2003–2004 the 2-year weights ÷ 2. Each estimate matches
    samplics on the weight the guideline gives, and differs from the one the two-year weights alone
    would give. The pooled-cycles finding and the roles' drawer now state the exception.
    """
    from turbotab.core.methods.survey import analysis_weights

    roles = {"SEQN": "identifier", "x": "exposure", "SDDSRVYR": "time", "WTMEC2YR": "design",
             "WTMEC4YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    frame = pd.read_csv(tables["pooled_123"])
    pid = open_project(client, tables["pooled_123"], "y", "inference", roles)
    [population, _] = survey_options(client, pid)
    assert population["decision"]["cycle"] == "SDDSRVYR"
    assert population["decision"]["four_year_weight"] == "WTMEC4YR"
    without = {**population["decision"], "four_year_weight": None}
    code, body = post(client, pid, without)
    assert code == 409 and body["error"]["code"] == "four_year_weight", body
    assert "not comparable" in body["error"]["message"]
    assert body["error"]["exits"][0]["decision"]["four_year_weight"] == "WTMEC4YR"
    accepted(client, pid, population["decision"])
    assert "as NCHS directs for 1999–2000" in sentence(client, pid, "set_survey")
    model = finish(client, pid)
    assert "WTMEC4YR" in model["inference"]["survey"]["weight_note"]

    first_two = frame["SDDSRVYR"].isin([1, 2]).to_numpy()
    guideline = np.where(first_two, 2 * frame["WTMEC4YR"] / 3, frame["WTMEC2YR"] / 3)
    X = frame[["x"]].to_numpy(float)
    beta, se = samplics_glm("linear", frame["y"], X, guideline, frame["SDMVSTRA"], frame["SDMVPSU"])
    got = row(model, "x")
    assert got["estimate"] == pytest.approx(beta[1], rel=1e-9)
    assert got["se"] == pytest.approx(se[1], rel=1e-9)
    two_year, _ = samplics_glm("linear", frame["y"], X, frame["WTMEC2YR"], frame["SDMVSTRA"],
                               frame["SDMVPSU"])
    assert abs(got["estimate"] - two_year[1]) > 0.002

    # 1999–2002 alone: the four-year weight NCHS provides; 2001–2004: the 2-year weights ÷ 2.
    alone = pd.read_csv(tables["pooled_12"])
    pooled = analysis_weights(alone, "WTMEC2YR", "SDDSRVYR", "WTMEC4YR")
    assert pooled.refusal is None and np.allclose(pooled.weights, alone["WTMEC4YR"])
    later = pd.read_csv(tables["pooled_23"])
    pooled = analysis_weights(later, "WTMEC2YR", "SDDSRVYR", None)
    assert pooled.refusal is None and np.allclose(pooled.weights, later["WTMEC2YR"] / 2)
    # No four-year weight in the file: refused, with "these participants" as the way on.
    pid = open_project(client, tables["pooled_no4yr"], "y", "inference",
                       {k: v for k, v in roles.items() if k != "WTMEC4YR"})
    [population, _] = survey_options(client, pid)
    code, body = post(client, pid, population["decision"])
    assert code == 409 and body["error"]["code"] == "four_year_weight", body
    way_on = body["error"]["exits"][-1]["decision"]
    assert (way_on["kind"], way_on["estimand"]) == ("set_survey", "sample")

    # D20/G20: the guidance states the exception and cites the guideline.
    from turbotab.core.stages.finding_words import FindingContext, cycle_findings
    from turbotab.core.teaching import entry

    [finding] = cycle_findings(FindingContext(frame=frame, lens=["dietary"]))
    assert "1999–2000" in finding["why_it_matters"] and "four-year" in finding["why_it_matters"]
    assert "NHANES Analytic Guidelines 2011–2016" in finding["why_it_matters"]
    drawer = {s.heading: s.body for s in entry("roles").drawer.sections}
    assert "four-year weight" in drawer["Pooled cycles"] and "§3.1.3–3.1.4" in drawer["Pooled cycles"]


class _Store:
    """The DataStore surface the survey code reads: ``columns`` and ``materialize``."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, row_ids=None):
        return self.frame[list(columns)] if row_ids is None else self.frame.loc[row_ids, list(columns)]


def test_partial_designs_are_attested_and_a_persons_rows_stay_one_unit():
    """Weights with no strata or PSU named are blocked until attested (the record says what the
    intervals assume); under prediction the question is refused, not answered. With no PSU column
    and a person's rows repeating, the person is the sampling unit (reference: the definition with
    the person as PSU and one stratum, df = persons − 1); a person whose rows sit in two PSUs makes
    the design inconsistent, and the fit says so rather than estimate."""
    from turbotab.core import decisions as d
    from turbotab.core import voice
    from turbotab.core.methods.survey import for_fit
    from turbotab.core.models.survey import survey_table

    rng = np.random.default_rng(77)
    people = np.repeat(np.arange(120), 3)
    frame = pd.DataFrame({"pid": people, "x": rng.normal(0, 1, len(people)),
                          "sampling_weight": np.repeat(rng.uniform(1, 4, 120), 3)})
    frame["y"] = 0.5 * frame["x"] + np.repeat(rng.normal(0, 1, 120), 3) + rng.normal(0, 1, len(frame))
    frame.index = pd.Index(np.arange(len(frame)), name="row_id")
    roles = {"pid": "identifier", "x": "exposure", "sampling_weight": "design"}
    state = d.ProjectState(lens=["clinical"], target="y", purpose="inference", roles=roles)
    ctx = {"state": state, "columns": list(frame.columns), "store": lambda: _Store(frame)}
    weights_only = {"kind": "set_survey", "estimand": "population", "weight": "sampling_weight"}
    with pytest.raises(d.Refusal) as refused:
        d.validate(weights_only, ctx)
    assert refused.value.code == "partial_design"
    attested = refused.value.exits[-1]["decision"]
    assert attested["acknowledged"] is True
    recorded = d.validate(attested, ctx)
    assert "no PSU column" in voice.sentence_for(recorded, state)
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_survey", "estimand": "sample"},
                   {"state": state.model_copy(update={"purpose": "prediction"})})
    assert refused.value.code == "not_inference"

    spec = d.SurveySpec(estimand="population", weight="sampling_weight", acknowledged=True)
    fitted = for_fit(state.model_copy(update={"survey": spec}), _Store(frame), unit_column="pid")
    assert fitted.refusal is None and fitted.design.n_psu == 120
    table = survey_table("regression", frame[["x"]], frame["y"].to_numpy(), None, fitted.design)
    Xc = np.column_stack([np.ones(len(frame)), frame["x"]])
    b, V = ref.wls_by_definition(Xc, frame["y"].to_numpy(float), frame["sampling_weight"].to_numpy(float),
                                 [0] * len(frame), frame["pid"], np.ones(len(frame), bool))
    for j, got in enumerate(table.rows):
        assert got["estimate"] == pytest.approx(b[j], rel=1e-9)
        assert got["se"] == pytest.approx(float(np.sqrt(V[j, j])), rel=1e-9)
        assert got["df"] == 119
    assert "each `pid`'s rows are one sampling unit" in table.info["caption"]

    frame["SDMVPSU"] = np.tile([1, 2, 1], 120)  # every person's rows split over two PSUs
    spec = d.SurveySpec(estimand="population", weight="sampling_weight", psu="SDMVPSU",
                        acknowledged=True)
    fitted = for_fit(state.model_copy(update={"survey": spec}), _Store(frame), unit_column="pid")
    assert fitted.refusal is not None and "more than one PSU" in fitted.refusal


def test_the_names_that_read_as_a_survey_design():
    """What the survey question fires on (the name reading behind the Router's gate): NHANES's own
    weights, with or without other NHANES names; generic sampling-weight names; never a body weight,
    a bare ``WT…`` name outside an NHANES table, or a replicate weight."""
    nhanes = read_design(["SEQN", "WTDRD1", "WTMEC2YR", "WTMEC4YR", "WTIREP01", "SDMVSTRA",
                          "SDMVPSU", "SDDSRVYR", "DR1TKCAL"])
    assert nhanes.weights == ["WTDRD1", "WTMEC2YR"]
    assert nhanes.four_year == {"WTMEC2YR": "WTMEC4YR"}
    assert (nhanes.strata, nhanes.psu, nhanes.cycles) == (["SDMVSTRA"], ["SDMVPSU"], ["SDDSRVYR"])
    assert read_design(["pid", "sampling_weight", "stratum", "psu"]).weights == ["sampling_weight"]
    assert not read_design(["id", "weight", "wt", "WTKG", "height"]).present
    assert read_design(["SEQN", "WTKG"]).weights == ["WTKG"]  # beside SEQN it is NHANES's
    assert not read_design(["WTDRD1"], target="WTDRD1").present
