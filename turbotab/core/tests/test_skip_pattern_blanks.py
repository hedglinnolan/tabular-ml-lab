"""W1-H · honest blanks for skip-pattern questions (WALKABLE_PLAN §1 blocker B5, §3 P1).

On NHANES, ``meds_hbp`` (15,552 blank) and ``meds_chol`` (17,204 blank) are blank where the question
was not asked. A most-frequent fill gives those blanks the commonest answer among the people who
were asked, which is True ("on medication"): a claim about people nobody asked. So where the blanks
read as "not asked":

* the recommended fill for those columns is a level of their own, never the mode, and blanks as a
  level is the method ranked first and recommended under prediction; elsewhere nothing changes;
* the recommended answer, run through the real pipeline, gives every blank the ``Missing`` level
  and never True;
* "not asked → No" is a repair the person chooses, with its sentence: afterwards the count of True
  is the answered True count and the blanks are False, both computed with pandas from the CSV;
* the preview of a most-frequent fill says what it gives the not-asked blanks.

Every expected number is computed independently with pandas from ``fixtures/nhanes.csv.gz``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import coach, consequences, repairs, row_previews, sweep  # noqa: F401
from turbotab.core import decisions as d
from turbotab.core.consequences import PreviewContext, plan
from turbotab.core.decisions import FindingDisposition, MissingSpec, ProjectState
from turbotab.core.methods.missing import (
    METHODS, blanks_mean_not_asked, honest_default, methods_for, not_asked_columns,
    recommended_fills,
)
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


def test_the_reference_blanks_and_answers_are_nhanes_own(raw):
    """The stakes, from pandas alone: most blanks, and True the commonest answer among the asked."""
    assert int(raw["meds_hbp"].isna().sum()) == 15_552
    assert int(raw["meds_chol"].isna().sum()) == 17_204
    for c in MEDS:
        assert raw[c].dropna().value_counts().idxmax() == True  # noqa: E712 - the mode is True


def test_the_recommended_fill_for_a_not_asked_column_is_never_the_mode(nhanes, raw):
    _table, columns, _roles, reading = nhanes
    info = {str(c["name"]): c for c in columns}
    assert sorted(not_asked_columns(reading, info)) == sorted(MEDS)
    for purpose in ("prediction", "inference", None):
        fills = recommended_fills(reading, info, purpose)
        for c in MEDS:
            assert fills[c] == "missing_category", (purpose, c)
            assert fills[c] != "most_frequent"
    # Blanks as a level is first and the only recommended method under prediction.
    ranked = methods_for("prediction", not_asked_columns(reading, info))
    assert ranked[0]["key"] == "missing_category" and ranked[0]["rung"] == "recommended"
    assert [m["key"] for m in ranked if m["rung"] == "recommended"] == ["missing_category"]
    single = next(m for m in ranked if m["key"] == "impute")
    assert "`meds_chol`" in single["sound"] and "not asked" in single["sound"]
    assert honest_default("prediction", MEDS) == {"kind": "set_missing", "strategy": "impute",
                                                  "categorical": "missing_category"}
    assert honest_default(None, MEDS) == honest_default("prediction", MEDS)


def test_nothing_changes_where_no_blank_means_not_asked():
    """The global default is untouched: no not-asked columns, the same ranking; a column with
    random blanks on a quarter of its rows (the WP7 fixture's ``meds_hbp``) keeps the mode."""
    for purpose in ("prediction", "inference", None):
        assert methods_for(purpose, ()) == methods_for(purpose)
    assert methods_for("prediction")[0]["key"] == "impute"
    assert methods_for("prediction")[0]["rung"] == "recommended"
    assert MissingSpec(strategy="impute").categorical == "impute"
    assert d.SetMissing(strategy="impute").categorical == "impute"
    assert honest_default("prediction", ()) is None
    assert honest_default("inference", MEDS) is None  # MI stays first under inference
    # The WP7 fixture: seed 707's table with meds_hbp blanked at random on about 25% of rows.
    from turbotab.core.tests.acceptance.test_wp7_missing_data import prediction_fixture

    frame = prediction_fixture()
    share = float(frame["meds_hbp"].isna().mean())
    assert 0.15 < share < 0.35
    assert not blanks_mean_not_asked(frame["meds_hbp"])
    columns = [{"name": c, "dtype": "boolean" if c == "meds_hbp" else "float",
                "n_unique": int(frame[c].nunique()), "n_missing": int(frame[c].isna().sum())}
               for c in frame.columns]
    roles = {c: "covariate" for c in frame.columns if c != "glucose"}
    reading = missing_reading(frame, columns, roles, "glucose")
    info = {c["name"]: c for c in columns}
    assert not_asked_columns(reading, info) == []
    assert recommended_fills(reading, info, "prediction")["meds_hbp"] == "most_frequent"


def test_the_recommended_answer_gives_every_not_asked_blank_its_own_level_in_the_pipeline(nhanes, raw):
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.stages.modeling import design_stage

    table, _columns, _roles, _reading = nhanes
    answer = {k: v for k, v in honest_default("prediction", MEDS).items() if k != "kind"}
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
    said = [(v.kind, n.text, n.anchor.ref) for v in plan(d.SetMissing(strategy="impute"), ctx,
                                                          basis="").views for n in v.coach]
    measured = raw["glucose"].notna()
    for c in MEDS:
        n = int((raw[c].isna() & measured).sum())
        mode = raw.loc[measured, c].dropna().value_counts().idxmax()
        assert ("table_focus", f"This fill gives `{n:,}` not-asked `{c}` blanks `{mode}`.", c) in said
    # With the blanks as their own level, nothing is given them: the note says they mean not asked.
    level = d.SetMissing(**{k: v for k, v in honest_default("prediction", MEDS).items() if k != "kind"})
    said = [n.text for v in plan(level, ctx, basis="").views for n in v.coach]
    assert not any(t.startswith("This fill gives") for t in said)
    assert any(t.endswith("likely a question not asked.") for t in said)
    assert METHODS["missing_category"].decision == {"strategy": "impute",
                                                    "categorical": "missing_category"}
