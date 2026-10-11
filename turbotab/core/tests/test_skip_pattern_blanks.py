"""W1-H · honest blanks for skip-pattern questions (WALKABLE_PLAN §1 blocker B5, §3 P1).

On NHANES, ``meds_hbp`` (15,552 blank) and ``meds_chol`` (17,204 blank) are blank where the question
was not asked. A most-frequent fill gives those blanks the commonest answer among the people who
were asked, which is True ("on medication"): a claim about people nobody asked. So where the blanks
read as "not asked":

* the recommended fill for those columns is a level of their own, never the mode: blanks as a level
  is the method ranked first and recommended under prediction, and under inference when they are
  the only blanks (with the attestation the leash asks for); elsewhere nothing changes;
* a fill that leaves its categorical treatment unsaid (``{"strategy": "impute"}``: a client, the
  row floor's "fill the blanks instead" exit) is completed by the server as blanks as a level, and
  run through the real pipeline gives every blank the ``Missing`` level and never True;
* "not asked → No" is a repair the person chooses, with its sentence: afterwards the count of True
  is the answered True count and the blanks are False, both computed with pandas from the CSV;
* the preview of a most-frequent fill, said outright, says what it gives the not-asked blanks.

Every expected number is computed independently with pandas from ``fixtures/nhanes.csv.gz``. The
new names are imported inside each test, so on the base each test fails on its own behavior.
Two tests are guards that hold on the base by design: the fixture's own facts (the stakes) and the
unchanged global default where no blank means not asked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import coach, consequences, repairs, row_previews, sweep  # noqa: F401
from turbotab.core import decisions as d
from turbotab.core.consequences import PreviewContext, plan
from turbotab.core.decisions import FindingDisposition, MissingSpec, ProjectState
from turbotab.core.methods import missing as mm
from turbotab.core.stages.findings import findings_stage
from turbotab.core.stages.proposals import build_proposals, missing_reading
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.stage_harness import NHANES_GZ, Ingested, nhanes_fixture

MEDS = ("meds_hbp", "meds_chol")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """The CSV as pandas reads it: the independent reference."""
    return pd.read_csv(NHANES_GZ)


@pytest.fixture(scope="module")
def nhanes(tmp_path_factory):
    table = Ingested(nhanes_fixture(), tmp_path_factory.mktemp("skip_pattern"))
    columns = table.info["columns"]
    roles = dict(mf.NHANES_ROLES)
    frame = table.frame()
    reading = missing_reading(frame, columns, roles, "glucose")
    return table, columns, roles, reading


def decision_ctx(frame, columns, roles, purpose, target="glucose"):
    """What the server hands a validation (``DecisionContext``): the state, the fresh proposals
    artifact and the ingest's column records."""
    proposals = build_proposals(frame, columns, lens=None, target=target, roles=roles,
                                purpose=purpose)
    state = ProjectState(lens=["dietary"], target=target, roles=roles, purpose=purpose)
    return state, {"state": state, "target": target,
                   "artifact": lambda stage: proposals if stage == "proposals" else None,
                   "column_info": {str(c["name"]): c for c in columns}}


def test_the_reference_blanks_and_answers_are_nhanes_own(raw):
    """The stakes, from pandas alone: most blanks, and True the commonest answer among the asked."""
    assert int(raw["meds_hbp"].isna().sum()) == 15_552
    assert int(raw["meds_chol"].isna().sum()) == 17_204
    for c in MEDS:
        assert raw[c].dropna().value_counts().idxmax() == True  # noqa: E712 - the mode is True


def test_the_recommended_fill_for_a_not_asked_column_is_never_the_mode(nhanes, raw):
    table, columns, roles, reading = nhanes
    info = {str(c["name"]): c for c in columns}
    assert sorted(mm.not_asked_columns(reading, info)) == sorted(MEDS)
    for purpose in ("prediction", "inference", None):
        fills = mm.recommended_fills(reading, info, purpose)
        for c in MEDS:
            assert fills[c] == "missing_category", (purpose, c)
            assert fills[c] != "most_frequent"
    # Blanks as a level is first and the only recommended method under prediction.
    ranked = mm.methods_for("prediction", mm.not_asked_columns(reading, info))
    assert ranked[0]["key"] == "missing_category" and ranked[0]["rung"] == "recommended"
    assert [m["key"] for m in ranked if m["rung"] == "recommended"] == ["missing_category"]
    single = next(m for m in ranked if m["key"] == "impute")
    assert "`meds_chol`" in single["sound"] and "not asked" in single["sound"]
    assert single["decision"] == {"strategy": "impute", "categorical": "impute"}  # said outright
    assert mm.honest_default("prediction", MEDS) == {"kind": "set_missing", "strategy": "impute",
                                                     "categorical": "missing_category"}
    assert mm.honest_default(None, MEDS) == mm.honest_default("prediction", MEDS)
    # From the reading as the proposals hold it: on NHANES the meds are the only predictors with
    # blanks (pandas), so under inference too blanks as a level is first, with the attestation
    # the leash asks for, and the leash accepts it as it stands.
    blank = sorted(c for c in raw.columns if raw[c].isna().any() and c in roles)
    assert blank == sorted(MEDS)
    for purpose in ("prediction", "inference"):
        first = mm.methods_for_reading(purpose, reading, info)[0]
        assert first["key"] == "missing_category" and first["rung"] == "recommended", purpose
        answer = mm.honest_default(purpose, mm.not_asked_columns(reading, info), only=True)
        assert answer == {"kind": "set_missing", **first["decision"]}
        _state, ctx = decision_ctx(table.frame(), columns, roles, purpose)
        accepted = d.validate(answer, ctx)
        assert accepted.categorical == "missing_category"
    mi = next(m for m in mm.methods_for_reading("inference", reading, info)
              if m["key"] == "multiple_imputation")
    assert mi["rung"] == "available" and "not asked" in mi["sound"]
    assert first["decision"]["acknowledged"] is True and "not asked" in first["decision"]["reason"]


def test_nothing_changes_where_no_blank_means_not_asked():
    """A guard (it holds on the base by design): the global default is untouched. No not-asked
    columns, the same ranking; a column with random blanks on a quarter of its rows (the WP7
    fixture's ``meds_hbp``) keeps the mode, and an unsaid fill there stays the most frequent."""
    for purpose in ("prediction", "inference", None):
        assert mm.methods_for(purpose, ()) == mm.methods_for(purpose)
    assert mm.methods_for("prediction")[0]["key"] == "impute"
    assert mm.methods_for("prediction")[0]["rung"] == "recommended"
    assert MissingSpec(strategy="impute").categorical == "impute"
    assert d.SetMissing(strategy="impute").categorical == "impute"
    assert mm.honest_default("prediction", ()) is None
    # Under inference with other blanks, multiple imputation stays first.
    assert mm.honest_default("inference", MEDS) is None
    # The WP7 fixture: seed 707's table with meds_hbp blanked at random on about 25% of rows.
    from turbotab.core.tests.acceptance.test_wp7_missing_data import prediction_fixture

    frame = prediction_fixture()
    share = float(frame["meds_hbp"].isna().mean())
    assert 0.15 < share < 0.35
    assert not mm.blanks_mean_not_asked(frame["meds_hbp"])
    columns = [{"name": c, "dtype": "boolean" if c == "meds_hbp" else "float",
                "n_unique": int(frame[c].nunique()), "n_missing": int(frame[c].isna().sum())}
               for c in frame.columns]
    roles = {c: "covariate" for c in frame.columns if c != "glucose"}
    reading = missing_reading(frame, columns, roles, "glucose")
    info = {c["name"]: c for c in columns}
    assert mm.not_asked_columns(reading, info) == []
    assert mm.recommended_fills(reading, info, "prediction")["meds_hbp"] == "most_frequent"
    assert mm.methods_for_reading("prediction", reading, info) == mm.methods_for("prediction")
    _state, ctx = decision_ctx(frame, columns, roles, "prediction")
    assert d.validate({"kind": "set_missing", "strategy": "impute"}, ctx).categorical == "impute"


def test_an_unsaid_categorical_fill_is_completed_as_blanks_as_a_level_on_nhanes(nhanes):
    """The default fill where blanks mean not asked: ``{"strategy": "impute"}`` leaves the
    categorical treatment unsaid, and the server completes it as blanks as a level. Said outright,
    the most-frequent fill is the person's to choose; with the meds left out nothing is not asked."""
    table, columns, roles, _reading = nhanes
    _state, ctx = decision_ctx(table.frame(), columns, roles, "prediction")
    unsaid = d.validate({"kind": "set_missing", "strategy": "impute"}, ctx)
    assert unsaid.categorical == "missing_category"
    said = d.validate({"kind": "set_missing", "strategy": "impute", "categorical": "impute"}, ctx)
    assert said.categorical == "impute"
    left_out = d.validate({"kind": "set_missing", "strategy": "impute",
                           "drop_columns": list(MEDS)}, ctx)
    assert left_out.categorical == "impute"
    indicators = d.validate({"kind": "set_missing", "strategy": "impute", "indicators": True}, ctx)
    assert indicators.categorical == "missing_category" and indicators.indicators
    # Undeclared purpose: as prediction, the strictest reading for a blank.
    _state, undeclared = decision_ctx(table.frame(), columns, roles, None)
    assert d.validate({"kind": "set_missing", "strategy": "impute"},
                      undeclared).categorical == "missing_category"


def test_the_row_floors_fill_exit_never_offers_the_mode_for_not_asked_blanks(nhanes):
    """``row_floor.fill_exits`` (a refusal's "Fill the blanks instead"; complete cases keep 2,996
    of 21,849 NHANES rows) offers each fill as the record would accept it: blanks as a level."""
    from turbotab.core import row_floor

    table, columns, roles, _reading = nhanes
    state, ctx = decision_ctx(table.frame(), columns, roles, "prediction")
    exits = row_floor.fill_exits(state, ctx)
    assert exits, "a fill is offered"
    for e in exits:
        assert e["decision"]["categorical"] == "missing_category", e
        assert "single fill in each training fold" not in e["label"]
    assert exits[0]["label"] == "Fill the blanks instead: blanks as their own level"


def test_the_default_fill_gives_every_not_asked_blank_its_own_level_in_the_pipeline(nhanes, raw):
    """The fill the server records for ``{"strategy": "impute"}``, run through the real design
    and pipeline: every training-row blank is ``Missing`` and none is read as on medication (the
    base read all 12,469 ``meds_hbp`` and 13,804 ``meds_chol`` training blanks as True)."""
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.modeling import design_stage

    table, columns, roles, _reading = nhanes
    _state, ctx = decision_ctx(table.frame(), columns, roles, "prediction")
    recorded = d.validate({"kind": "set_missing", "strategy": "impute"}, ctx)
    answer = recorded.model_dump(exclude={"kind"})
    st = mf.state(missing=MissingSpec(**answer), energy_adjustment=None, models=["linear"])
    rows = np.arange(len(raw))
    split = mf.split_bundle(rows, seed=0)
    ctx = mf.context(st, {"split": split, "target_info": mf.target_info("regression")},
                     {"data": str(table.parquet), "source": str(table.source)})
    design = design_stage(ctx)
    shared = design.objects["pipelines"]["linear"][:-1]
    train = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    with table.store() as store:
        X = modeling_frame(store, design.objects["spec"]["inputs"], train)
    out = shared.fit(X).transform(X)
    for c in MEDS:
        # pandas, from the CSV: which training rows are blank, and which answered True.
        expected_blank = raw[c].isna().to_numpy()[train]
        expected_true = (raw[c] == True).to_numpy()[train]  # noqa: E712
        np.testing.assert_array_equal(out[f"{c}_Missing"].to_numpy(), expected_blank.astype(float))
        np.testing.assert_array_equal(out[f"{c}_1"].to_numpy(), expected_true.astype(float))
        # No blank is read as on medication.
        assert float(out.loc[expected_blank, f"{c}_1"].sum()) == 0.0


@pytest.fixture(scope="module")
def nhanes_findings(nhanes):
    table, *_ = nhanes
    return table.run(findings_stage, ProjectState(lens=["dietary", "clinical"], target="glucose"))


def _not_asked_option(artifact, column):
    f = next(f for f in artifact["findings"] if f["id"] == f"binary_text__{column}")
    return f, next(o for o in f["repairs"] if o["key"] == "not_asked_no")


def test_not_asked_as_no_is_a_repair_the_person_chooses_with_its_sentence(nhanes, nhanes_findings, raw):
    table, *_ = nhanes
    dispositions = {}
    for c in MEDS:
        f, option = _not_asked_option(nhanes_findings, c)
        n_blank = int(raw[c].isna().sum())
        assert len(option["label"].split()) <= repairs.LABEL_WORDS
        assert len(option["consequence"].split()) <= repairs.CONSEQUENCE_WORDS
        assert f"`{n_blank:,}` blanks in `{c}` become `False`" in option["consequence"]
        assert f"`{n_blank:,}` people not asked" in option["sentence"]
        assert option["decision"]["params"]["relabels"] is False
        # Recoding blanks is never read as a relabeling that changes no number.
        assert not sweep._relabels(f["repairs"])
        dispositions[f["id"]] = FindingDisposition(action="applied", option="not_asked_no",
                                                   params=option["decision"]["params"])
    exprs = repairs.column_expressions(nhanes_findings, dispositions)
    after = repairs.evaluate(table.parquet, list(MEDS), exprs)
    for c in MEDS:
        answered_true = int((raw[c] == True).sum())  # noqa: E712
        answered_false = int((raw[c] == False).sum())  # noqa: E712
        blank = int(raw[c].isna().sum())
        values = after[c]
        assert not values.isna().any()
        assert int((values == True).sum()) == answered_true  # noqa: E712
        assert int((values == False).sum()) == answered_false + blank  # noqa: E712
        # Row by row: an answer stays as given, a blank becomes False.
        expected = raw[c].where(raw[c].notna(), False).astype(bool).to_numpy()
        np.testing.assert_array_equal(values.to_numpy(dtype=bool), expected)


def test_the_recode_is_offered_only_where_blanks_mean_not_asked():
    """Random blanks on a quarter of a yes/no (the WP7 shape) are not a skip pattern; nor is a
    two-valued column whose other value is not a "no"."""
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({"glucose": rng.normal(100, 15, n),
                          "meds_hbp": rng.choice([True, False], n).astype(object),
                          "sex": rng.choice(["Male", "Female"], n).astype(object)})
    frame.loc[rng.random(n) < 0.25, "meds_hbp"] = None
    oc = repairs.OfferContext(frame=frame, target="glucose")
    finding = {"id": "binary_text__meds_hbp", "affected_columns": ["meds_hbp"]}
    params = {"column": "meds_hbp", "written_as": "object_bools", "negative": "false",
              "positive": "true", "levels": ["false", "true"]}
    assert repairs._offer_not_asked(finding, params, oc) == []
    # Asked of a fifth: offered.
    frame.loc[rng.random(n) < 0.8, "meds_hbp"] = None
    assert [o.key for o in repairs._offer_not_asked(finding, params, repairs.OfferContext(
        frame=frame, target="glucose"))] == ["not_asked_no"]
    # A text yes/no keeps its own spelling of no.
    frame["asked"] = np.where(rng.random(n) < 0.2, rng.choice(["Yes", "No"], n), None)
    text = {"column": "asked", "written_as": "object_text", "negative": "no", "positive": "yes",
            "levels": ["no", "yes"]}
    [o] = repairs._offer_not_asked({"id": "binary_text__asked", "affected_columns": ["asked"]}, text,
                                   repairs.OfferContext(frame=frame, target="glucose"))
    assert o.decision.params["no"] == "No" and o.decision.params["kind"] == "text"
    # Male/Female: neither is a "no", so nothing is offered.
    frame["sex"] = np.where(rng.random(n) < 0.2, frame["sex"], None)
    sex = {"column": "sex", "written_as": "object_text", "negative": "male", "positive": "female",
           "levels": ["female", "male"]}
    assert repairs._offer_not_asked({"id": "binary_text__sex", "affected_columns": ["sex"]}, sex,
                                    repairs.OfferContext(frame=frame, target="glucose")) == []
    # The outcome is never recoded this way.
    assert repairs._offer_not_asked(finding, params, repairs.OfferContext(
        frame=frame, target="meds_hbp")) == []


def _ctx(table, state, artifacts):
    return PreviewContext(project_id="skip", state=state, datastore=table.store(),
                          artifact=lambda s: artifacts.get(s), training_row_ids=None,
                          cohort_row_ids=None, sealed_row_ids=None)


def test_the_preview_of_a_most_frequent_fill_says_what_it_gives_the_not_asked_blanks(nhanes, raw):
    table, columns, roles, _reading = nhanes
    state = ProjectState(lens=["dietary"], target="glucose", roles=roles, purpose="prediction")
    frame = table.frame()
    proposals = build_proposals(frame, columns, lens=None, target="glucose", roles=roles,
                                purpose="prediction")
    ctx = _ctx(table, state, {"proposals": proposals})
    most_frequent = d.SetMissing(strategy="impute", categorical="impute")
    said = [(v.kind, n.text, n.anchor.ref) for v in plan(most_frequent, ctx, basis="").views
            for n in v.coach]
    measured = raw["glucose"].notna()
    for c in MEDS:
        n = int((raw[c].isna() & measured).sum())
        mode = raw.loc[measured, c].dropna().value_counts().idxmax()
        assert ("table_focus", f"This fill gives `{n:,}` not-asked `{c}` blanks `{mode}`.", c) in said
    # With the blanks as their own level, nothing is given them: the note says they mean not asked.
    level = d.SetMissing(**{k: v for k, v in mm.honest_default("prediction", MEDS).items()
                            if k != "kind"})
    said = [n.text for v in plan(level, ctx, basis="").views for n in v.coach]
    assert not any(t.startswith("This fill gives") for t in said)
    assert any(t.endswith("likely a question not asked.") for t in said)
    assert mm.METHODS["missing_category"].decision == {"strategy": "impute",
                                                    "categorical": "missing_category"}
