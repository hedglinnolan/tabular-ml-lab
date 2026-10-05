"""Findings that can be acted on (M2_CONTRACT §4).

Tier A: every row-local repair's SQL equals a pandas reference on its fixture (sentinel codes on
``survey_sentinels``, impossible values on ``clinical_labs``, SAS zeros on the real NHANES export,
kilojoules on ``nhanes_kilojoules``, a binary text predictor on ``clinical_labs``); the rows an
impossible-value exclusion removes are exactly the rows holding one; and missingness by mechanism:
a ``Missing`` level reaches the model matrix for ``meds_hbp`` on NHANES, and in-fold imputation never
sees a held-out row or the outcome.

Tier B (light): the options' word budgets, the validators, the memory (answered by, deferred to).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import repairs
from turbotab.core.decisions import (
    ApplyRepair,
    DecisionRecord,
    DeferFinding,
    DismissFinding,
    FindingDisposition,
    MissingSpec,
    ProjectState,
    Refusal,
    SetRoles,
    left_out,
    validate,
)
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.stage_harness import NHANES, SAMPLES, Ingested

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(), reason="the real NHANES export is not here")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """Each fixture ingested once, with its findings under its lens: name -> (table, artifact)."""
    out = {}
    plan = [("survey_sentinels", SAMPLES / "survey_sentinels.csv", ["survey"], None),
            ("clinical_labs", SAMPLES / "clinical_labs.csv", ["clinical"], None),
            ("clinical_longitudinal", SAMPLES / "clinical_longitudinal.csv", ["clinical"], "glucose"),
            ("nhanes_kilojoules", SAMPLES / "nhanes_kilojoules.csv", ["dietary"], None),
            ("binary_shapes", SAMPLES / "binary_shapes.csv", ["clinical"], None),
            ("clinic_visits", SAMPLES / "clinic_visits.csv", ["clinical"], None)]
    if NHANES.is_file():
        plan.append(("nhanes", NHANES, ["dietary", "clinical"], "glucose"))
    for name, path, lens, target in plan:
        table = Ingested(path, tmp_path_factory.mktemp(name))
        artifact = table.run(findings_stage, ProjectState(lens=lens, target=target))
        out[name] = (table, artifact)
    return out


def finding(artifact, fid):
    return next(f for f in artifact["findings"] if f["id"] == fid)


def applied(f, key=None, which=0):
    """The disposition recording ``f``'s option (its first, or the one keyed ``key``)."""
    options = [o for o in f["repairs"] if key is None or o["key"] == key]
    o = options[which]
    return FindingDisposition(action="applied", option=o["key"], params=o["decision"]["params"])


def sql_frame(table, artifact, dispositions, columns):
    exprs = repairs.column_expressions(artifact, dispositions)
    assert set(columns) <= set(exprs), (columns, sorted(exprs))
    return repairs.evaluate(table.parquet, columns, exprs)


def same(a: pd.Series, b: pd.Series) -> None:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    assert len(x) == len(y)
    np.testing.assert_array_equal(np.isnan(x), np.isnan(y))
    np.testing.assert_array_equal(x[~np.isnan(x)], y[~np.isnan(y)])


# ── Tier A: the SQL is the pandas reference ──────────────────────────────────


def test_sentinel_codes_become_missing_exactly_as_pandas_says(tables):
    table, artifact = tables["survey_sentinels"]
    frame = table.frame()
    pack = finding(artifact, "pack::survey::sentinel_codes")
    codes = {"item_14": [9], "item_05": [9], "item_22": [8], "item_33": [8], "item_09": [7]}
    assert {c: [int(v) for v in vs] for c, vs in pack["repairs"][0]["decision"]["params"]["values"].items()} == codes
    got = sql_frame(table, artifact, {pack["id"]: applied(pack)}, list(codes))
    total = 0
    for column, vs in codes.items():
        reference = frame[column].where(~frame[column].isin(vs))  # the codes blank, the rest as is
        same(got[column], reference)
        total += int(frame[column].isin(vs).sum())
    assert total == 102  # survey_sentinels.md: 102 sentinel values across five items
    # One column's own finding does the same to its column, and the SQL is the same.
    one = finding(artifact, "sentinel_missing__item_14")
    alone = sql_frame(table, artifact, {one["id"]: applied(one)}, ["item_14"])
    same(alone["item_14"], got["item_14"])
    assert not (alone["item_14"] == 9).any() and int(alone["item_14"].isna().sum()) == 33


def test_impossible_values_set_to_missing_exactly_as_pandas_says(tables):
    table, artifact = tables["clinical_labs"]
    frame = table.frame()
    f = finding(artifact, "pack::clinical::impossible_vs_extreme")
    assert [o["key"] for o in f["repairs"]] == ["set_missing", "exclude_rows", "unusable"]
    assert f["repairs"][0]["decision"]["params"]["bands"] == {"sbp": [30.0, 300.0]}
    got = sql_frame(table, artifact, {f["id"]: applied(f, "set_missing")}, ["sbp"])
    x = frame["sbp"]
    reference = x.mask((x < 30) | (x > 300))  # CLINICAL_SURVEY_PACK §A1.2: SBP 30–300
    same(got["sbp"], reference)
    assert int(got["sbp"].isna().sum() - x.isna().sum()) == 4  # clinical_labs.md: 4 impossible sbp
    assert int((got["sbp"] > 200).sum()) == int((x > 200).sum())  # the abnormal but real stay


def test_impossible_values_excluded_leave_the_flow_as_exactly_those_rows(tables):
    from turbotab.core.stages.rows import compute_cohort

    table, artifact = tables["clinical_labs"]
    frame = table.frame()
    f = finding(artifact, "pack::clinical::impossible_vs_extreme")
    state = ProjectState(target="readmitted", findings={f["id"]: applied(f, "exclude_rows")})
    with table.store() as store:
        steps, kept, _ = compute_cohort(store, state, table.info)
    outside = (frame["sbp"] < 30) | (frame["sbp"] > 300)
    reference = frame.index[frame["readmitted"].notna() & ~outside].to_numpy()
    np.testing.assert_array_equal(np.sort(kept), np.sort(reference))
    step = next(s for s in steps if s["key"] == "repair:0")
    assert step["dropped"] == 4 and step["reason"] == "a physiologically impossible value"
    # Marking it unusable takes it out of the predictors instead, and no row leaves.
    state = ProjectState(target="readmitted", roles={"sbp": "covariate", "age": "covariate"},
                         findings={f["id"]: applied(f, "unusable")})
    assert left_out(state) == ["sbp"]
    with table.store() as store:
        steps, kept, preds = compute_cohort(store, state, table.info)
    assert preds == ["age"] and len(kept) == len(frame)


@needs_nhanes
def test_sas_zeros_become_zero_and_compose_with_the_impossible_band(tables):
    table, artifact = tables["nhanes"]
    frame = table.frame()
    sas = finding(artifact, "sas_zeros")
    columns = sas["repairs"][0]["decision"]["params"]["columns"]
    assert columns[0] == "bp_di" and {"kcal", "fat_total"} <= set(columns)
    assert repairs.sas_zero_counts(frame)["bp_di"] == 119
    got = sql_frame(table, artifact, {sas["id"]: applied(sas)}, columns)
    for column in columns:
        x = frame[column]
        same(got[column], x.mask(repairs.is_sas_zero(x.to_numpy(dtype=float, na_value=np.nan)), 0.0))
        assert not repairs.is_sas_zero(got[column].to_numpy(dtype=float, na_value=np.nan)).any()
    assert int((got["bp_di"] == 0).sum()) == 119
    # With the impossible values set to missing too, SAS zeros run first: a zero diastolic is
    # outside 10–200 mmHg (CLINICAL_SURVEY_PACK §A1.2), so it ends blank, as the pandas
    # composition says. Total energy is a dietary report, not a physiology band (audit IN-09):
    # the plausibility repair leaves it as the SAS-zero repair wrote it.
    band = finding(artifact, "pack::clinical::impossible_vs_extreme")
    assert "kcal" not in band["repairs"][0]["decision"]["params"]["bands"]
    both = {sas["id"]: applied(sas), band["id"]: applied(band, "set_missing")}
    got = sql_frame(table, artifact, both, ["bp_di", "kcal"])
    for column, (lo, hi) in (("bp_di", (10, 200)), ("kcal", (-np.inf, np.inf))):
        x = frame[column]
        x = x.mask(repairs.is_sas_zero(x.to_numpy(dtype=float, na_value=np.nan)), 0.0)
        same(got[column], x.mask((x < lo) | (x > hi)))


def test_kilojoules_become_kcal_exactly_as_pandas_says(tables):
    table, artifact = tables["nhanes_kilojoules"]
    frame = table.frame()
    f = finding(artifact, "pack::dietary::atwater")
    got = sql_frame(table, artifact, {f["id"]: applied(f)}, ["DR1TKCAL"])
    same(got["DR1TKCAL"], frame["DR1TKCAL"] / 4.184)
    # After the conversion the energy column reads in kcal: Atwater's reconstruction matches it.
    atwater = 4 * frame["DR1TPROT"] + 4 * frame["DR1TCARB"] + 9 * frame["DR1TTFAT"] + 7 * frame["DR1TALCO"]
    assert abs(float((got["DR1TKCAL"] / atwater).median()) - 1.0) < 0.02


def test_a_binary_text_predictor_is_recoded_as_the_chosen_level(tables):
    from ml.binary_text import _normalize

    table, artifact = tables["clinical_labs"]
    frame = table.frame()
    f = finding(artifact, "binary_text__sex")
    assert [(o["label"], o["decision"]["params"]["one"]) for o in f["repairs"]] == [
        ("`f` counts as 1", "f"), ("`m` counts as 1", "m")]
    for which, one in ((0, "f"), (1, "m")):
        got = sql_frame(table, artifact, {f["id"]: applied(f, which=which)}, ["sex"])
        reference = frame["sex"].map(_normalize).map(lambda t: None if t is None else int(t == one))
        same(got["sex"], reference)
        assert set(got["sex"].dropna().unique()) == {0, 1}


def test_a_binary_outcome_is_left_to_the_event_question(tables):
    _, artifact = tables["clinical_longitudinal"]
    assert finding(artifact, "binary_text__sex")["repairs"]  # a predictor: offered
    f = finding(artifact, "pack::clinical::impossible_vs_extreme")
    unusable = next(o for o in f["repairs"] if o["key"] == "unusable")
    assert "glucose" not in unusable["decision"]["params"]["bands"]  # the outcome is never set aside


# ── Tier A: missingness by mechanism ─────────────────────────────────────────


@needs_nhanes
def test_a_missing_level_reaches_the_model_matrix_for_meds_hbp_on_nhanes(tables):
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.rows import compute_cohort

    table, _ = tables["nhanes"]
    keep_blank = MissingSpec(strategy="complete_case", categorical="missing_category")
    st = mf.state(missing=keep_blank, energy_adjustment=None, models=["linear"])
    plain = mf.state(missing="complete_case", energy_adjustment=None, models=["linear"])
    with table.store() as store:
        _, kept, _ = compute_cohort(store, st, table.info)
        _, kept_plain, _ = compute_cohort(store, plain, table.info)
    frame = table.frame(["meds_hbp", "meds_chol", "glucose"])
    # Complete cases no longer drop a row for a blank meds_hbp or meds_chol: a blank is a level.
    assert int(frame.loc[kept, "meds_hbp"].isna().sum()) > 10_000
    assert len(kept) > 3 * len(kept_plain)
    split = mf.split_bundle(kept, seed=0)
    ctx = mf.context(st, {"split": split, "target_info": mf.target_info("regression")},
                     {"data": str(table.parquet), "source": str(table.source)})
    design = design_stage(ctx)
    matrix = {n["column"] for n in design.data["lineage"]["nodes"] if n["lane"] == "matrix"}
    # True is 1, False the reference, and a blank its own level.
    assert {"meds_hbp_1", "meds_hbp_Missing", "meds_chol_Missing"} <= matrix
    assert "meds_hbp" not in matrix and "meds_hbp_0" not in matrix
    assert "levels" in [s["key"] for s in design.data["models"][0]["steps"]]
    shared = design.objects["pipelines"]["linear"][:-1]
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    from turbotab.core.models.pipeline import modeling_frame

    with table.store() as store:
        X = modeling_frame(store, design.objects["spec"]["inputs"], train)
    out = shared.fit(X).transform(X)
    blank = X["meds_hbp"].isna().to_numpy()
    np.testing.assert_array_equal(out["meds_hbp_Missing"].to_numpy(), blank.astype(float))
    np.testing.assert_array_equal(out["meds_hbp_1"].to_numpy(),
                                  (X["meds_hbp"] == 1).to_numpy().astype(float))
    assert not out.isna().any().any()


def test_in_fold_imputation_never_sees_held_out_rows_or_the_outcome(tmp_path, monkeypatch):
    from sklearn.impute import SimpleImputer

    from turbotab.core.models.pipeline import MissingLevelEncoder
    from turbotab.core.stages.modeling import design_stage, fit_stage

    frame = mf.nhanes_like(300, seed=21, missing=True)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), seed=21)
    a = split.frames["assignment"]
    train = a[a["partition"] == "train"]
    train_ids, folds = train["row_id"].to_numpy(), train["fold"].to_numpy()
    holdout = set(a.loc[a["partition"] != "train", "row_id"].tolist())
    allowed = {frozenset(train_ids[folds != k].tolist()) for k in np.unique(folds)}
    allowed.add(frozenset(train_ids.tolist()))
    fits: list[tuple[str, frozenset, list[str], object]] = []

    def spy(cls, name):
        original = cls.fit

        def fit(self, X, *args, **kwargs):
            fits.append((name, frozenset(pd.Index(X.index).tolist()), [str(c) for c in X.columns], self))
            return original(self, X, *args, **kwargs)
        monkeypatch.setattr(cls, "fit", fit)

    spy(SimpleImputer, "imputer")
    spy(MissingLevelEncoder, "levels")
    spec = MissingSpec(strategy="impute", categorical="missing_category", indicators=True)
    st = mf.state(missing=spec, energy_adjustment=None, models=["linear", "boosted_trees"])
    ti = mf.target_info("regression")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fits.clear()
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    assert {name for name, *_ in fits} == {"imputer", "levels"}
    for name, rows, columns, _ in fits:
        assert rows in allowed, f"{name} was fit on rows that are not one training fold"
        assert not rows & holdout, f"{name} saw held-out rows"
        assert "glucose" not in columns, f"the outcome reached the {name}"
    # Known answer: each numeric imputer's medians are the medians of its own fitting rows, and a
    # number with two values is filled by its most frequent value there (BLUEPRINT §14.3: one
    # indicator either way, so a code and an amount fill it alike; the smallest of tied values,
    # SimpleImputer's own rule), never by a median between the two.
    from turbotab.core.models.pipeline import normalize_frame

    stored = normalize_frame(pd.read_parquet(paths["data"]).set_index("__row_id"))
    checked = 0
    for name, rows, columns, imputer in fits:
        if name != "imputer" or imputer.strategy != "median":
            continue
        part = stored.loc[sorted(rows), columns]
        want = [float(part[c].mode().iloc[0]) if stored[c].dropna().nunique() == 2
                else float(part[c].median()) for c in columns]
        np.testing.assert_allclose(imputer.statistics_, want)
        checked += 1
    assert checked >= 2 * (len(allowed))  # each family, each fold and the refit
    # Blank yes/no answers are a level; numbers are filled and marked.
    matrix = {n["column"] for n in design.data["lineage"]["nodes"] if n["lane"] == "matrix"}
    assert {"meds_hbp_Missing", "gender_Missing", "missingindicator_bmi"} <= matrix
    links = {(l["source"], l["operation"]) for l in design.data["lineage"]["links"]
             if l["target"] == "adj:missingindicator_bmi"}
    assert links == {("raw:bmi", "missing indicator")}
    assert all(m["cv"] for m in fit.data["models"])


def test_the_outcome_cannot_move_the_imputer(tmp_path):
    """Scramble the outcome and refit the whole pipeline: the fitted imputation is the same."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.modeling import design_stage

    frame = mf.nhanes_like(240, seed=22, missing=True)
    split = mf.split_bundle(np.arange(len(frame)), seed=22)
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    st = mf.state(missing=MissingSpec(strategy="impute", indicators=True), energy_adjustment=None,
                  models=["linear"])
    ti = mf.target_info("regression")
    scrambled = frame["glucose"].sample(frac=1, random_state=1).to_numpy() * 50
    stats = []
    for k, outcome in enumerate((frame["glucose"].to_numpy(), scrambled)):
        paths = mf.ingest_frame(frame.assign(glucose=outcome), tmp_path / str(k))
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        pipeline = design.objects["pipelines"]["linear"]
        with DataStore(Path(paths["data"]), 1 << 30) as store:
            X = modeling_frame(store, design.objects["spec"]["inputs"], train)
            y = store.materialize(["glucose"], train)["glucose"]
        keep = y.notna().to_numpy()
        pipeline.fit(X[keep], y[keep])
        imputer = pipeline.named_steps["impute"].named_transformers_["numeric"]
        assert "glucose" not in imputer.feature_names_in_
        stats.append(imputer.statistics_.astype(float))
    np.testing.assert_array_equal(stats[0], stats[1])


# ── Tier B: budgets, validators, memory ──────────────────────────────────────


def test_every_option_is_within_its_budget_and_speaks_plainly(tables):
    from turbotab.core import voice

    seen = 0
    for name, (_, artifact) in tables.items():
        for f in artifact["findings"]:
            assert repairs.NO_LEVER not in f["summary"] or not f["repairs"], (name, f["id"])
            for o in f["repairs"]:
                seen += 1
                assert voice.words(o["label"]) <= repairs.LABEL_WORDS, (name, o["label"])
                assert voice.words(o["consequence"]) <= repairs.CONSEQUENCE_WORDS, (name, o["consequence"])
                for text in (o["label"], o["consequence"], o["sentence"]):
                    assert not voice.machinery(text), (name, text)
                assert o["decision"]["finding_id"] == f["id"] and o["decision"]["option"] == o["key"]
                assert o["row_local"] is True
    assert seen >= 20


def _ctx(artifact, **extra):
    return {"artifact": lambda stage: artifact if stage == "findings" else None, **extra}


def test_apply_repair_is_refused_unless_the_finding_offers_it(tables):
    _, artifact = tables["clinical_labs"]
    f = finding(artifact, "pack::clinical::impossible_vs_extreme")
    ctx = _ctx(artifact)
    with pytest.raises(Refusal) as no_such:
        validate(ApplyRepair(finding_id="nope", option="set_missing"), ctx)
    assert no_such.value.code == "unknown_finding"
    with pytest.raises(Refusal) as wrong:
        validate(ApplyRepair(finding_id=f["id"], option="winsorize"), ctx)
    assert wrong.value.code == "unknown_repair" and len(wrong.value.exits) == 3
    forged = {"bands": {"sbp": [0.0, 400.0]}}
    with pytest.raises(Refusal) as bad:
        validate(ApplyRepair(finding_id=f["id"], option="set_missing", params=forged), ctx)
    assert bad.value.code == "repair_params"
    # Named alone, the option is completed with the finding's own params, so the record says
    # exactly what it does.
    done = validate(ApplyRepair(finding_id=f["id"], option="set_missing"), ctx)
    assert done.params["bands"] == {"sbp": [30.0, 300.0]}
    # A two-form option must be chosen; a part of an offer is admitted (some codes, not all).
    sex = finding(artifact, "binary_text__sex")
    with pytest.raises(Refusal) as two:
        validate(ApplyRepair(finding_id=sex["id"], option="level"), ctx)
    assert two.value.code == "choose_repair" and len(two.value.exits) == 2
    _, survey = tables["survey_sentinels"]
    pack = finding(survey, "pack::survey::sentinel_codes")
    part = ApplyRepair(finding_id=pack["id"], option="set_missing", params={"values": {"item_14": [9.0]}})
    assert validate(part, _ctx(survey)).params == {"values": {"item_14": [9.0]}}
    # A finding with no repair says where it can go instead.
    plain = next(f for f in artifact["findings"] if not f["repairs"] and f["routes_to"])
    with pytest.raises(Refusal) as none:
        validate(ApplyRepair(finding_id=plain["id"], option="x"), ctx)
    assert none.value.code == "no_repair"
    assert [e["decision"]["kind"] for e in none.value.exits] == ["defer_finding", "dismiss_finding"]


def test_deferral_needs_a_question_and_dismissal_a_finding(tables):
    _, artifact = tables["clinical_labs"]
    ctx = _ctx(artifact)
    f = finding(artifact, "pack::clinical::impossible_vs_extreme")
    validate(DeferFinding(finding_id=f["id"], to="exclusions"), ctx)
    with pytest.raises(Refusal) as nowhere:
        validate(DeferFinding(finding_id=f["id"], to="the_moon"), ctx)
    assert nowhere.value.code == "unknown_question"
    assert nowhere.value.exits[0]["decision"]["to"] == f["routes_to"]
    validate(DismissFinding(finding_id=f["id"], reason="known entry errors"), ctx)
    with pytest.raises(Refusal):
        validate(DismissFinding(finding_id="gone"), ctx)


def _record(seq, decision):
    from datetime import datetime, timezone

    return DecisionRecord(id=f"r{seq}", seq=seq, at=datetime.now(timezone.utc), decision=decision)


def test_findings_learn_they_were_answered(tables):
    from turbotab.core.decisions import Revert, fold
    from turbotab.core.interview import route

    _, survey = tables["survey_sentinels"]
    pack = finding(survey, "pack::survey::sentinel_codes")
    item_14 = finding(survey, "sentinel_missing__item_14")
    sex = finding(survey, "binary_text__sex")
    wide = next(f for f in survey["findings"] if f["id"].startswith("wide_repeated"))
    records = [
        _record(1, ApplyRepair(finding_id=pack["id"], option="set_missing",
                               params=pack["repairs"][0]["decision"]["params"])),
        _record(2, DeferFinding(finding_id=sex["id"], to="roles")),
        _record(3, DismissFinding(finding_id=wide["id"])),
    ]
    state = fold(records)
    served = {f["id"]: f for f in repairs.annotate(survey, state, records)["findings"]}
    assert served[pack["id"]]["disposition"]["action"] == "applied"
    assert served[pack["id"]]["answered_by"] == "r1"
    # Its own column's finding is answered by the bulk repair that already did its work.
    assert served[item_14["id"]]["disposition"] is None and served[item_14["id"]]["answered_by"] == "r1"
    assert served[sex["id"]]["answered_by"] == "r2" and served[wide["id"]]["answered_by"] == "r3"
    steps = {s.key: s for s in route(state, {}, {}, records)}
    assert steps["roles"].deferred_findings == [sex["id"]]
    assert all(not s.deferred_findings for k, s in steps.items() if k != "roles")
    # Answering the question the finding routes to, with a matching answer, answers it too.
    roles = SetRoles(roles={"respondent_id": "identifier", "age": "covariate"})
    ident = next(f for f in survey["findings"] if f["id"] == "voice::identifier__respondent_id")
    records.append(_record(4, roles))
    served = {f["id"]: f for f in repairs.annotate(survey, fold(records), records)["findings"]}
    assert served[ident["id"]]["answered_by"] == "r4"
    # Revert the bulk repair: nothing it answered stays answered.
    records.append(_record(5, Revert(decision_id="r1")))
    served = {f["id"]: f for f in repairs.annotate(survey, fold(records), records)["findings"]}
    assert served[pack["id"]]["disposition"] is None and served[pack["id"]]["answered_by"] is None
    assert served[item_14["id"]]["answered_by"] is None


def test_only_an_applied_repair_changes_a_stage_key():
    from turbotab.core.graph import key_value

    deferred = {"a": {"action": "deferred", "option": None, "params": {}, "to": "roles", "reason": None}}
    dismissed = {"b": {"action": "dismissed", "option": None, "params": {}, "to": None, "reason": None}}
    applied_ = {"c": {"action": "applied", "option": "zero", "params": {"columns": ["x"]}, "to": None,
                      "reason": None}}
    assert key_value("findings", deferred) is None
    assert key_value("findings", {**deferred, **dismissed}) is None
    assert key_value("findings", {**deferred, **applied_}) == applied_
