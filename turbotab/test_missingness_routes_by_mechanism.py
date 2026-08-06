"""Clause §07: missingness routes by dtype **and** mechanism.

`GUIDED-021` recorded that the clause was linked to `tests/test_missingness_
encoding.py`, whose ten tests are about encoding helpers and assert nothing
about the clause. This file is what the clause actually requires.

**The mechanism half is the one that cannot be automated**, and it is asked
first. *"Could a blank here mean something?"* comes before *"how should it be
filled?"*, because the answer decides which strategies are legitimate at all.
Asked in the other order, a column that carried information gets a median
written over it by a well-meaning default.

Prediction is not inference, and clause §07 says so in as many words: the
missing-indicator method discouraged for causal estimation is defensible and
often helpful for prediction under informative missingness. So the two branches
are separate objects and fail differently — the categorical branch fails by
imputing a signal away, the numeric branch by leaking the outcome into the
imputation model.

Assertions here follow the L17 rule: **structure, not prose substrings.** Where
prose IS the deliverable — the blocker, the assumption — the assertion is on the
distinctive claim rather than on a word inside it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import (eligibility as E, engine, grain as G,          # noqa: E402
                      missingness as M)
from turbotab.project import AnalysisProject, ProjectError           # noqa: E402


def study(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 15, n),
        "biopsy_grade": rng.choice(["I", "II", "III"], n),
        "outcome": rng.integers(0, 2, n),
    })
    # A numeric column with accidental blanks, and a categorical one whose
    # blanks are the interesting case: a biopsy not ordered is not a biopsy
    # whose result was lost.
    df.loc[rng.choice(n, 30, replace=False), "glucose"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "biopsy_grade"] = np.nan
    return df


def _sealed() -> AnalysisProject:
    p = AnalysisProject.from_dataframe(study(), "t")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    return p


# ── the survey routes by dtype, and asks the mechanism of nobody ─────────────

def test_the_survey_routes_by_dtype_and_leaves_the_mechanism_unanswered():
    """Half of clause §07 is mechanical and half is not, and the split has to be
    visible in the data rather than in a comment.

    `mechanism: None` on every row is the assertion that matters: the app has
    NOT guessed. Inferring it is the same error the grain question exists to
    prevent, one clause over.

    Clause: `lockbox-07`
    """
    p = _sealed()
    rows = {r["column"]: r for r in p.missingness_survey()}

    assert set(rows) == {"glucose", "biopsy_grade"}, (
        "the survey reported columns with no blanks, or missed one")
    assert rows["glucose"]["branch"] == "numeric"
    assert rows["biopsy_grade"]["branch"] == "categorical"
    assert all(r["mechanism"] is None for r in rows.values()), (
        "the survey filled in a mechanism, so the app guessed at something "
        "only the user knows")

    # The strategies offered differ BY BRANCH — an explicit Missing category is
    # meaningless for a numeric column, and a median for a categorical one.
    assert M.EXPLICIT_CATEGORY in rows["biopsy_grade"]["strategies"]
    assert M.EXPLICIT_CATEGORY not in rows["glucose"]["strategies"]
    assert M.IMPUTE_MEDIAN in rows["glucose"]["strategies"]
    assert M.IMPUTE_MEDIAN not in rows["biopsy_grade"]["strategies"]

    # The outcome is never surveyed: its missingness is a different problem and
    # routing it here would offer to impute the thing being predicted.
    assert "outcome" not in rows


def test_a_column_with_no_blanks_cannot_be_routed():
    """The interview does not invent work.

    Clause: `lockbox-07`
    """
    p = _sealed()
    with pytest.raises(ProjectError, match="no missing values"):
        p.route_missingness("age", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)


# ── the CONSEQUENCE: informative missingness imputed away ────────────────────

def test_imputing_an_informatively_missing_column_is_blocked_with_both_exits():
    """§07's blocker, and `DESIGN_LANGUAGE.md` §09's shape for it.

    The user has just said a blank in this column MEANS something. Filling it
    removes that fact from the data entirely and no model can recover it. But
    they may know something the app does not, so it resolves or is attested —
    never a dead end, and never a silent override.

    Clause: `lockbox-07`
    """
    p = _sealed()
    assert M.blocks(M.INFORMATIVE, M.IMPUTE_MODE) is True

    with pytest.raises(ProjectError) as exc:
        p.route_missingness("biopsy_grade", M.INFORMATIVE, M.IMPUTE_MODE)
    said = str(exc.value)
    assert "no model can recover it afterward" in said, (
        "the blocker does not say what is actually lost")
    assert "`Missing` category keeps the blank as its own answer" in said, (
        "the blocker names no way out, so it is a dead end")

    blk = M.blocker("biopsy_grade", M.INFORMATIVE, M.IMPUTE_MODE, 60)
    kinds = {e["kind"] for e in blk["exits"]}
    assert kinds == {"resolve", "attest"}, (
        f"the interruption offers {kinds}, and §09 requires both a resolution "
        "and an attestation")
    assert blk["acknowledgment_kind"] == "typed", (
        "§07 asks for a TYPED acknowledgment; a click is not one")


def test_the_attested_path_records_the_signal_loss_rather_than_hiding_it():
    """Overriding a blocker is allowed; overriding one quietly is not.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("biopsy_grade", M.INFORMATIVE, M.IMPUTE_MODE,
                        acknowledged=True)
    rec = p.missingness[0]
    assert rec["acknowledged_signal_loss"] is True, (
        "the attestation left no mark, so the override is invisible to the "
        "methods section")
    # `AUDIT-028`. THIS DOOR HAS NO FOLDS. `turbotab/training.py:416`:
    # nothing under `turbotab/` imports `KFold`, `cross_val_score` or
    # `cross_validate`. This assertion read "training folds only" for a
    # dozen loops, which made it a GREEN TEST PINNING THE DEFECT — the
    # shape filed this same loop as `TEST-060`.
    assert rec["defers"] is True and rec["fit_on"] == "training rows only"


def test_not_sure_does_not_block():
    """Deliberate. The user has said they do not know; turning an admission of
    uncertainty into a wall teaches people to stop admitting it — the same
    reasoning that makes "I'm not sure" first-class on the grain question.

    Clause: `lockbox-07`
    """
    assert M.blocks(M.NOT_SURE, M.IMPUTE_MEDIAN) is False
    assert M.blocks(M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN) is False
    p = _sealed()
    p.route_missingness("glucose", M.NOT_SURE, M.IMPUTE_MEDIAN)
    assert p.missingness[0]["mechanism"] == M.NOT_SURE


# ── the stability assumption, recorded as a methods assumption ───────────────

def test_stating_informative_records_the_stability_assumption():
    """§07: *the stability assumption — that missingness means the same thing at
    deployment — is recorded as a methods assumption, because it may not hold
    across sites.*

    An ASSUMPTION rather than a warning, and the distinction is the point: a
    warning is something a user dismisses, an assumption is something a
    manuscript carries.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("biopsy_grade", M.INFORMATIVE, M.EXPLICIT_CATEGORY)
    rec = p.missingness[0]
    assert "assumption" in rec, (
        "an informative mechanism was recorded without its stability "
        "assumption, so nothing carries it into the methods section")
    assert "not checkable from this dataset" in rec["assumption"]
    assert "biopsy_grade" in rec["assumption"], (
        "the assumption does not name the column it is about")

    # And it is NOT recorded where it would be false: a mechanism the user
    # called accidental makes no claim about deployment.
    q = _sealed()
    q.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)
    assert "assumption" not in q.missingness[0]


# ── the numeric branch: the outcome is never in the imputation model ─────────

def test_the_outcome_cannot_enter_the_imputation_model():
    """§07's hard blocker, and the one that is not a judgment call — so it is
    REFUSED rather than offered with an acknowledgment.

    An imputer fitted with the outcome in scope writes the outcome's own
    information into the feature columns, and every number scored afterwards is
    scored against features that encode the answer.

    Clause: `lockbox-07`
    """
    p = _sealed()
    with pytest.raises(ProjectError) as exc:
        p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MICE,
                            uses_columns=["age", "outcome"])
    said = str(exc.value)
    assert "already encode the answer" in said
    assert "not offered as a choice" in said, (
        "the refusal reads as a configurable option rather than a prohibition")

    # And there is no acknowledgment that gets round it — unlike the
    # informative-missingness blocker, which the user may legitimately override.
    with pytest.raises(ProjectError):
        p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MICE,
                            uses_columns=["outcome"], acknowledged=True)

    # The same strategy without the outcome in scope is fine.
    p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MICE,
                        uses_columns=["age", "record_id"])
    assert p.missingness[0]["strategy"] == M.IMPUTE_MICE


# ── clause §06 again: almost nothing executes, and the two that do are local ─

def test_only_the_row_local_strategies_change_the_table():
    """Clause §06's litmus, applied to §07's strategies.

    A median, a mode and a MICE model are all statements about the column's
    distribution. An explicit `Missing` token and a was-it-missing flag use
    nothing but the row's own cell. So two execute and the rest are recorded.

    Clause: `lockbox-07`
    """
    for key in M.NUMERIC_STRATEGIES + M.CATEGORICAL_STRATEGIES:
        spec = M.strategy(key)
        assert spec["defers"] == (key not in M.ROW_LOCAL_STRATEGIES)
        assert len(spec["because"]) > 40, (
            f"{key} states a scope without saying why")

    p = _sealed()
    before = p.fingerprint()
    p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)
    assert p.fingerprint() == before, (
        "a stateful strategy changed the working table, which is the canonical "
        "preprocessing leak clause §06 forbids")

    p.route_missingness("biopsy_grade", M.INFORMATIVE, M.EXPLICIT_CATEGORY)
    assert p.fingerprint() != before, "a row-local strategy changed nothing"
    assert p.df["biopsy_grade"].isna().sum() == 0
    assert (p.df["biopsy_grade"] == "Missing").sum() == 60


def test_the_indicator_is_row_local_and_leaves_the_value_blank():
    """The indicator adds a column and does NOT fill the original — filling it
    would be the imputation the user declined.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("glucose", M.INFORMATIVE, M.INDICATOR)
    assert "glucose_was_missing" in p.df.columns
    assert p.df["glucose"].isna().sum() == 30, (
        "the indicator filled the underlying value as well")
    assert p.df["glucose_was_missing"].sum() == 30


# ── the receipt: honest about reporting zero immediate change ────────────────

def test_the_receipt_is_honest_when_nothing_visibly_changed():
    """The design problem this step has and no previous step did.

    Almost every preprocessing transform is stateful, so a user can answer six
    questions and watch the table not move. A receipt that is embarrassed by
    that zero would be the step apologizing for obeying clause §06; a receipt
    that hides it would be the step lying.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)
    p.route_missingness("biopsy_grade", M.NOT_INFORMATIVE, M.IMPUTE_MODE)
    r = M.plan_receipt(p.missingness, len(p.missingness_survey()))

    assert r["n_applied_now"] == 0 and r["n_deferred"] == 2
    assert "0 column(s) changed now" not in r["headline"], (
        "the headline leads with a zero, which reads as a step that failed")
    assert "2 recorded to be fitted inside the training folds" in r["headline"]

    why = r["why_nothing_changed"]
    assert "over the held-out rows too" in why, (
        "the explanation does not give the REASON nothing changed, so the user "
        "is left to conclude the app did nothing")
    assert "the part that cannot be automated" in why, (
        "the receipt does not tell the user what they actually accomplished")


def test_the_receipt_says_so_when_nothing_was_deferred():
    """The other case, and it must not print the same sentence.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("glucose", M.INFORMATIVE, M.INDICATOR)
    p.route_missingness("biopsy_grade", M.INFORMATIVE, M.EXPLICIT_CATEGORY)
    r = M.plan_receipt(p.missingness, len(p.missingness_survey()))
    assert r["n_deferred"] == 0 and r["n_applied_now"] == 2
    assert "Nothing was deferred" in r["why_nothing_changed"]
    assert "training fold" not in r["why_nothing_changed"], (
        "the receipt promises fold-time work that is not scheduled")


def test_the_receipt_counts_what_is_still_unanswered():
    """Silence about an unrouted column is the failure this project exists to
    remove — a step that looks finished while a column with 30% blanks was
    never asked about.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)
    r = M.plan_receipt(p.missingness, len(p.missingness_survey()))
    assert r["n_unanswered"] == 1
    assert r["outstanding"], "an unanswered column produced no outstanding line"


def test_settling_the_step_is_recorded_even_when_it_is_skipped():
    """The recorded-absence rule (`DESIGN_LANGUAGE.md` §09), which this step
    inherits rather than rediscovers.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.settle_preprocess(skipped=True)
    assert p.preprocess_settled is True
    d = next(x for x in p.decisions if x.kind == "settle_preprocess")
    assert d.text.strip() and "skipped" in d.text

    q = _sealed()
    q.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MEDIAN)
    q.settle_preprocess()
    worked = next(x for x in q.decisions if x.kind == "settle_preprocess")
    assert worked.text != d.text, (
        "a skipped step and a worked step produce the same sentence")


# ── the trim is a CHOICE, and it says what it is not ─────────────────────────

def test_the_trim_says_it_is_not_a_population_restriction():
    """§04's two objects look identical in a spreadsheet, so the trim says which
    one it is at the point of the choice rather than leaving it to be inferred.

    Clause: `lockbox-07`
    """
    p = _sealed()
    p.trim_training_rows("age", minimum=40, maximum=75,
                         reason="The cohort of clinical interest is 40 to 75.")
    ob = p.obligations[0]
    said = ob["not_a_population_restriction"]
    assert "does not change who your study is about" in said
    assert "N is unchanged" in said
    # The full claim, not just the noun: naming "eligibility" without saying it
    # is pre-seal and changes N sends the user somewhere they cannot act.
    assert "asked before the seal" in said and "does change N" in said, (
        "the label says what the trim is NOT and does not tell the user what "
        "the other object is or when it can be reached")


# ── over HTTP ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from turbotab.api import app
    return TestClient(app)


def _drive_to_preprocess(client) -> str:
    df = study()
    pid = client.post("/project", files={
        "file": ("study.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["id"]
    for body in (
        {"kind": "set_target", "payload": {"column": "outcome"}},
        {"kind": "set_grain", "payload": {"answer": G.ONE_ROW_PER_PERSON}},
        {"kind": "set_eligibility", "payload": {"answer": E.EVERYONE}},
        {"kind": "seal"},
    ):
        r = client.post(f"/project/{pid}/decision", json=body)
        assert r.status_code == 200, r.text
    return pid


def test_a_driver_routes_missingness_and_meets_the_blocker(client):
    """The whole step over HTTP: the mechanism question is asked per column, the
    blocker arrives as a 409 with both exits, and the attested path completes.

    Clause: `lockbox-07`
    """
    pid = _drive_to_preprocess(client)

    iv = client.get(f"/project/{pid}/interview?step=preprocess").json()
    keys = {q["key"] for q in iv["questions"]}
    assert {"missingness::glucose", "missingness::biopsy_grade"} <= keys
    q = next(q for q in iv["questions"] if q["key"] == "missingness::biopsy_grade")
    assert q["consumer"], "a FACT must name what reads its answer"
    assert q["clause"] == "lockbox-07"

    pre = client.get(f"/project/{pid}/preprocess").json()
    assert {c["column"] for c in pre["columns"]} == {"glucose", "biopsy_grade"}
    assert all("because" in s for s in pre["strategies"]["numeric"])

    blocked = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": "biopsy_grade", "mechanism": M.INFORMATIVE,
                    "strategy": M.IMPUTE_MODE}})
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert {e["kind"] for e in detail["exits"]} == {"resolve", "attest"}
    assert detail["acknowledgment_kind"] == "typed"

    ok = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": "biopsy_grade", "mechanism": M.INFORMATIVE,
                    "strategy": M.EXPLICIT_CATEGORY}})
    assert ok.status_code == 200, ok.text
    assert ok.json()["missingness"][0]["acknowledged_signal_loss"] is False

    # The answered question retires from the interview.
    iv = client.get(f"/project/{pid}/interview?step=preprocess").json()
    assert "missingness::biopsy_grade" not in {q["key"] for q in iv["questions"]}


def test_a_driver_reads_a_receipt_that_explains_the_zero(client):
    """What a driver sees at the end of a step in which nothing visibly changed.

    Clause: `lockbox-07`
    """
    pid = _drive_to_preprocess(client)
    client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": "glucose", "mechanism": M.NOT_INFORMATIVE,
                    "strategy": M.IMPUTE_MEDIAN}})
    client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness",
        "payload": {"column": "biopsy_grade", "mechanism": M.NOT_SURE,
                    "strategy": M.IMPUTE_MODE}})
    body = client.post(f"/project/{pid}/decision",
                       json={"kind": "settle_preprocess"}).json()

    assert body["preprocess_settled"] is True
    d = body["disclosures"]["preprocess"]
    assert d["n_applied_now"] == 0 and d["n_deferred"] == 2
    assert d["n_unanswered"] == 0
    assert "over the held-out rows too" in d["why_nothing_changed"]
    assert body["n_rows"] == 200, "the table changed after a deferred-only step"


def test_the_declarations_survive_the_save_file():
    """This step is almost entirely declarations, so the archive is the ONLY
    place they live — nothing has executed them. A restored project that lost
    them would have no record that the step happened.

    Clause: `lockbox-07`
    """
    from turbotab import archive
    p = _sealed()
    p.route_missingness("glucose", M.NOT_INFORMATIVE, M.IMPUTE_MICE,
                        uses_columns=["age"])
    p.route_missingness("biopsy_grade", M.INFORMATIVE, M.EXPLICIT_CATEGORY)
    p.settle_preprocess()

    back = archive.from_bytes(archive.to_bytes(p))
    assert back.preprocess_settled is True
    assert len(back.missingness) == 2
    mice = next(d for d in back.missingness if d["strategy"] == M.IMPUTE_MICE)
    assert mice["uses_columns"] == ["age"], (
        "the imputation SCOPE was dropped, so nothing downstream can check the "
        "outcome stayed out of it")
    informative = next(d for d in back.missingness if d["mechanism"] == M.INFORMATIVE)
    assert "assumption" in informative, (
        "the stability assumption did not survive, so the methods section "
        "cannot carry it")
