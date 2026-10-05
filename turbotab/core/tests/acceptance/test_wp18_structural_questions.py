"""WP18 · Structural questions ask where evidence is thin (docs/turbotab-next/audit/AUDIT_REPORT.md §5).

Closes RO-09, RO-10, RO-11 and RO-13, and I18 (minor); and places the readings ledger's one ask card
(BLUEPRINT §14.2) where the first consumer needs it. The acceptance tests, as §5 states them:

0. **The one ask card** (BLUEPRINT §14.2): "One card, 'Tell me about these columns', holds each
   field's best guess, pre-filled with its evidence, ordered by how much the field changes. A
   homogeneous family … is confirmed as one block." The Router places it where the first consumer
   needs it, not as a wall at upload.
1. "Time points with predictors = last and outcome = first: refused under prediction, blocked and
   recorded under inference."
2. "``none/mild/moderate/severe`` triggers 'are these levels ordered?'; hs-CRP triggers an
   outcome-scale question; the 20-class rule agrees between the skip and the explicit answer."
3. "A 'something else / not sure' lens is accepted, runs the generic checks and is stated in the
   methods."
4. "A Case/Control/QC export offers a QC exclusion before the seal and the task becomes binary; the
   served QC text is corrected."
5. "NHANES DXA multiple-imputation copies have a route (Rubin's rules) or are blocked and recorded."
   (MS3 builds the route: under inference copies kept as records are pooled by Rubin's rules, and
   combining them per unit is what is blocked and recorded;
   ``test_ms1_ms3_multiple_imputation.test_10_*`` checks the pooling against R.)

Every project is driven through the real server (the FastAPI app over its job runner). Readings the
server asks about are answered from each fixture's declared truth (``conftest.declare``), never a
constant. References come from paths that share nothing with the code under test: pandas counts of
the fixtures' own values, ``scipy.stats.skew``, NumPy least squares, and the fixtures' generators.

**Source checks** (quoted from the primary source, read 2026-10-03):

* The skewness reference behind "markedly skewed" — Kim HY. Statistical notes for clinical
  researchers: assessing normal distribution (2) using skewness and kurtosis. *Restor Dent Endod*
  2013;38(1):52–54: "West et al. (1996) proposed a reference of substantial departure from normality
  as an absolute skew value > 2."
* NHANES 1999–2006 DXA (CDC, "Multiple Imputation Details", wwwn.cdc.gov/nchs/nhanes/dxa/dxa.aspx):
  "Each of the data files contains FIVE sets of measured and imputed values." … "The extra
  variability due to imputation CANNOT be incorporated by simply analyzing a SINGLE dataset as if the
  imputed values were true values." … "The preferred statistical approach is to analyze EACH OF THE
  FIVE datasets separately … and then combining the estimates and standard errors using the
  combining rules".
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.interview import NEEDS
from turbotab.server.tests.conftest import (
    answer_adjustment_card,
    answer_settled,
    declare,
    make_client,
    usual_answer,
    wait_for,
)

SAMPLES = Path(__file__).resolve().parents[3] / "sample_data"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("wp18_home"), "local", 2, "http://127.0.0.1") as c:
        yield c


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("wp18_tables")


# ── driving the server ───────────────────────────────────────────────────────


def post(client, pid: str, decision: dict) -> tuple[int, dict]:
    r = client.post(f"/api/projects/{pid}/decisions", json=decision)
    return r.status_code, r.json()


def accepted(client, pid: str, decision: dict) -> dict:
    r = answer_settled(client, pid, None, decision)
    assert r.status_code == 200, (decision, r.text[:900])
    return r.json()


def refused(client, pid: str, decision: dict, code: str) -> dict:
    """``decision`` refused for ``code`` once the Router has reached it (a recomputing stage may
    hold it behind an earlier question for a moment)."""
    from turbotab.server.tests.conftest import when_reached

    r = when_reached(lambda: client.post(f"/api/projects/{pid}/decisions", json=decision))
    status, body = r.status_code, r.json()
    assert status == 409, (decision, body)
    assert body["error"]["code"] == code, body
    return body["error"]


def open_table(client, path: Path, truth: dict[str, Any] | None = None) -> str:
    r = client.post("/api/projects", json={"path": str(path)})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    declare(pid, truth or {}, fixture=path.name)
    wait_for(client, pid, {"ingest": "fresh"}, timeout=120)
    return pid


def view(client, pid: str) -> dict:
    return client.get(f"/api/projects/{pid}").json()


def step(client, pid: str, key: str, timeout: float = 120.0) -> dict:
    """The Router's step for ``key`` once no stage it waits on is still computing."""
    end = time.monotonic() + timeout
    while True:
        found = next(s for s in view(client, pid)["interview"] if s["key"] == key)
        if found["status"] != "waiting" or time.monotonic() > end:
            return found
        time.sleep(0.05)


def artifact(client, pid: str, stage: str, timeout: float = 240.0) -> dict:
    end = time.monotonic() + timeout
    while True:
        status = view(client, pid)["stages"][stage]
        if status["status"] == "fresh":
            return client.get(f"/api/projects/{pid}/stages/{stage}").json()["artifact"]
        assert status["status"] != "error", status
        assert time.monotonic() < end, f"{stage} never fresh: {status}"
        time.sleep(0.05)


def walk(client, pid: str, until: str, cards: dict[str, Any] | None = None,
         answers: dict[str, dict] | None = None, timeout: float = 300.0) -> dict:
    """Answer each question the Router opens before ``until`` (its usual answer, or ``answers``'),
    recording the ask card each one carried while it was open; returns the view once ``until`` is
    the open question."""
    end = time.monotonic() + timeout
    while True:
        v = view(client, pid)
        first = next((s for s in v["interview"] if s["status"] in ("open", "waiting")), None)
        assert first is not None, f"nothing left to ask before {until}"
        unread = [n for n in (*first["waiting_on"], *NEEDS.get(first["key"], ()))
                  if v["stages"].get(n, {}).get("status") not in (None, "fresh")]
        if first["status"] == "waiting" or unread:
            failed = [n for n in unread if v["stages"][n]["status"] == "error"]
            assert not failed, v["stages"][failed[0]]
            assert time.monotonic() < end, f"held behind {first}"
            time.sleep(0.05)
            continue
        if cards is not None:
            cards.setdefault(first["key"], first.get("ask"))
        if first["key"] == until:
            return v
        answer = (answers or {}).get(first["key"]) or usual_answer(client, pid, first["key"], v)
        if answer["kind"] == "__adjustment__":  # WP17: each covariate from the declared truth
            answer_adjustment_card(client, pid)
            continue
        r = answer_settled(client, pid, first["key"], answer)
        assert r.status_code == 200, (first["key"], r.text[:900])


def last_sentence(client, pid: str) -> str:
    return view(client, pid)["decisions"][-1]["sentence"]


# ── 0 · the one ask card, where the first consumer needs it ──────────────────


def codes_table(folder: Path) -> tuple[Path, pd.DataFrame]:
    """One row per person: whole-number predictors that are codes (smoking 1–3, education 1–5) or
    amounts (cups 0–6, six 1–5 items), a 0/1 flag, a measured age, and a continuous outcome."""
    rng = np.random.default_rng(18)
    n = 400
    df = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(50, 9, n).round(1),
        "female": rng.integers(0, 2, n),
        "smoking": rng.integers(1, 4, n),
        "education": rng.integers(1, 6, n),
        "cups": rng.integers(0, 7, n),
    })
    for k in range(1, 7):
        df[f"item_{k:02d}"] = rng.integers(1, 6, n)
    df["sbp"] = (120 + 0.4 * (df["age"] - 50) + 2.0 * df["cups"] + rng.normal(0, 8, n)).round(1)
    path = folder / "codes.csv"
    df.to_csv(path, index=False)
    return path, df


CODES_TRUTH = {"code_or_count:smoking": "code", "code_or_count:education": "code",
               "code_or_count:cups": "amount",
               **{f"code_or_count:item_{k:02d}": "amount" for k in range(1, 7)}}


def test_0_the_ask_card_sits_where_the_first_consumer_needs_it_never_at_upload(client, folder):
    """The fit is the first consumer of the codes-or-amounts readings on a one-row-per-person table
    (BLUEPRINT §14.3: "A fit needs code-or-amount settled for every whole-valued predictor").

    Reference (pandas, from the file): the whole-valued columns with two or more values that are
    not exactly 0 and 1, the outcome excepted. The card:
    * appears on no question before the models question (lens, outcome, purpose, roles,
      exclusions, missing values, split): no wall at upload;
    * on the models question lists exactly the reference columns, each with its guess and evidence,
      ordered by consequence (more levels first: more indicators), the six ``item_*`` columns as one
      family line;
    * its block confirmation lists exactly those readings; answered from the fixture's truth, the
      card is gone and the models answer is accepted, and the fit reads codes as indicators and
      amounts as slopes.
    """
    path, df = codes_table(folder)
    pid = open_table(client, path, CODES_TRUTH)
    for d in ({"kind": "set_lens", "lenses": ["clinical"]}, {"kind": "set_target", "column": "sbp"}):
        accepted(client, pid, d)
    cards: dict[str, Any] = {}
    v = walk(client, pid, "models", cards)

    asked_before = {k: c for k, c in cards.items() if k != "models" and c is not None}
    assert not asked_before, asked_before
    assert {"purpose", "roles", "exclusions", "missing", "split"} <= set(cards)

    numeric = df.drop(columns=["sbp"]).select_dtypes("number")
    whole = [c for c in numeric if (numeric[c] == numeric[c].round()).all()
             and numeric[c].nunique() >= 2 and set(numeric[c].unique()) != {0, 1}]
    card = next(s for s in v["interview"] if s["key"] == "models")["ask"]
    assert card is not None and card["question"] == "models" and card["consumer"] == "the fit"
    listed = [c for g in card["groups"] for c in g["columns"]]
    assert sorted(listed) == sorted(whole) == sorted(["smoking", "education", "cups",
                                                      *[f"item_{k:02d}" for k in range(1, 7)]])
    assert all(g["kind"] == "code_or_count" and g["guess"] in ("code", "amount") and g["evidence"]
               for g in card["groups"])
    # Ordered by consequence: more levels first (one indicator per level), names breaking ties.
    levels = [df[g["columns"][0]].nunique() for g in card["groups"]]
    assert levels == sorted(levels, reverse=True)
    families = [g["columns"] for g in card["groups"] if len(g["columns"]) > 1]
    assert families == [[f"item_{k:02d}" for k in range(1, 7)]]
    assert card["text"].startswith("Tell me about these columns:")
    assert "6 columns like `item_01`" in card["text"]
    # The block confirmation lists exactly the readings on the card, each with the value it shows.
    block = card["exits"][0]["decision"]
    assert block["kind"] == "confirm_readings"
    assert sorted(i["column"] for i in block["items"]) == sorted(whole)
    # Read from your data (BLUEPRINT §14.3, amendment: "Readings settled by their values appear on
    # the card under 'read from your data', each with its evidence and a way to change it"): the
    # card carries the readings endpoint's own items for the fit. Reference (pandas): `age` is the
    # one predictor whose values are not whole, so its values settle it an amount, never asked.
    served = client.get(f"/api/projects/{pid}/readings").json()["read_from_data"]
    settled = card["read_from_data"]
    assert all(item in served for item in settled), (settled, served)
    assert [c for c in numeric if (numeric[c] != numeric[c].round()).any()] == ["age"]
    age = next(i for i in settled if i["kind"] == "code_or_count" and i["column"] == "age")
    assert age["value"] == "amount" and age["evidence"] and age["change"]
    assert not {i["column"] for i in settled if i["kind"] == "code_or_count"} & set(whole)

    # The user answers the card from what they know (the fixture's truth), in one block.
    accepted(client, pid, {"kind": "confirm_readings", "items": [
        {"reading": "code_or_count", "column": c, "value": CODES_TRUTH[f"code_or_count:{c}"]}
        for c in listed]})
    assert step(client, pid, "models")["ask"] is None
    accepted(client, pid, {"kind": "select_models", "models": ["linear"]})
    fit = artifact(client, pid, "fit")
    features = {r["feature"] for r in fit["models"][0]["coefficients"]}
    # Codes enter as indicators (k − 1 of them), amounts as one slope each.
    assert {"cups", *[f"item_{k:02d}" for k in range(1, 7)]} <= features
    assert "smoking" not in features and "education" not in features
    assert sum(f.startswith("smoking") for f in features) == df["smoking"].nunique() - 1
    assert sum(f.startswith("education") for f in features) == df["education"].nunique() - 1


def test_0b_each_consumer_card_asks_exactly_what_its_refusal_would():
    """The card is the consumer's own question asked first: for the screens, the energy adjustment
    and the survey design, the readings on the card are exactly those the consumer's refusal asks
    (its confirmation exits), read from the same state; a question whose answer reads no reading
    (the lens, the outcome) has no card; total energy's unit, only proposed, is a line answered by
    ``set_column_unit`` as the screens' refusal answers it."""
    from turbotab.core.ask import AskContext, card
    from turbotab.core.decisions import ProjectState, Refusal, validate
    from turbotab.core.tests.truths import asked

    st = ProjectState(
        lens=["dietary"], target="bmi", purpose="inference",
        roles={"participant_id": "identifier", "energy_kcal": "energy", "fat_g": "exposure",
               "protein_g": "exposure", "wtint2yr": "design", "age": "covariate"},
        roles_unconfirmed=["energy_kcal", "fat_g", "wtint2yr"])
    columns = ["participant_id", "energy_kcal", "fat_g", "protein_g", "wtint2yr", "age", "bmi"]
    proposals = {"energy_unit": {"unit": "kcal", "basis": "magnitude", "days": 1,
                                 "confirmed": False, "sentence": "Only its median says kcal."},
                 "energy": {"energy_column": "energy_kcal"}}
    ctx = AskContext(st, {"proposals": proposals})

    def refusal_asks(decision: dict) -> list[str]:
        with pytest.raises(Refusal) as caught:
            validate(decision, {"state": st, "columns": columns})
        return sorted(c for _, c in asked(caught.value.exits))

    def card_columns(question: str, kind: str = "role") -> list[str]:
        found = card(question, ctx)
        return sorted(c for g in found.groups if g.kind == kind for c in g.columns)

    screen = {"kind": "set_exclusions", "rules": [
        {"column": "energy_kcal", "low": 500, "high": 5000, "reason": "implausible intakes"}]}
    assert card_columns("exclusions") == refusal_asks(screen) == ["energy_kcal"]
    unit = card("exclusions", ctx)
    line = next(g for g in unit.groups if g.kind == "unit")
    assert line.columns == ["energy_kcal"] and line.evidence == "Only its median says kcal."
    assert {"kind": "set_column_unit", "column": "energy_kcal", "unit": "kcal", "days": 1} in [
        e.decision for e in unit.exits]
    energy = {"kind": "set_energy_adjustment", "method": "standard", "energy_column": "energy_kcal",
              "nutrients": ["fat_g", "protein_g"]}
    assert card_columns("energy_adjustment") == refusal_asks(energy) == ["energy_kcal", "fat_g"]
    survey = {"kind": "set_survey", "estimand": "population", "weight": "wtint2yr",
              "acknowledged": True}
    assert card_columns("survey") == refusal_asks(survey) == ["wtint2yr"]
    for question in ("lens", "target", "purpose", "roles", "grain"):
        assert card(question, ctx) is None
    # Once each reading is confirmed on its own, nothing is left to ask.
    settled = st.model_copy(update={"role_confirmations": {"energy_kcal": "energy",
                                                           "fat_g": "exposure",
                                                           "wtint2yr": "design"}})
    assert card("energy_adjustment", AskContext(settled)) is None
    assert card("survey", AskContext(settled)) is None


# ── 1 · predictors summarized after the outcome (RO-09) ───────────────────────


def visits_table(folder: Path) -> tuple[Path, pd.DataFrame]:
    """150 people × 3 dated visits six months apart: sbp (the outcome), sodium and a cigarette
    count change between visits; age stays."""
    rng = np.random.default_rng(9)
    rows = []
    start = pd.Timestamp("2020-01-01")
    for i in range(150):
        age = round(float(rng.normal(55, 8)), 1)
        base = float(rng.normal(130, 12))
        for k in range(3):
            day = start + pd.Timedelta(days=int(180 * k + rng.integers(0, 10)))
            sodium = round(float(rng.normal(3.2, 0.8)), 2)
            rows.append({"pid": f"V{i:04d}", "visit_date": day.strftime("%Y-%m-%d"), "age": age,
                         "sodium_g": sodium, "cigarettes": int(rng.integers(0, 21)),
                         "sbp": round(base + 3 * k + 2 * sodium + float(rng.normal(0, 4)), 1)})
    df = pd.DataFrame(rows)
    path = folder / "visits.csv"
    df.to_csv(path, index=False)
    return path, df


VISITS_ROUTE = {"grain": {"kind": "set_grain", "grain": "repeated", "id_column": "pid"},
                "repeat_kind": {"kind": "set_repeat_kind", "repeat_kind": "time_points",
                                "time_column": "visit_date"},
                "unit": {"kind": "set_unit", "unit": "unit"}}


def _visits_at_aggregation(client, folder: Path, purpose: str) -> tuple[str, dict]:
    path, df = visits_table(folder)
    pid = open_table(client, path, {"code_or_count:cigarettes": "amount"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]}, {"kind": "set_target", "column": "sbp"}):
        accepted(client, pid, d)
    cards: dict[str, Any] = {}
    walk(client, pid, "aggregation", cards,
         answers={**VISITS_ROUTE, "purpose": {"kind": "set_purpose", "purpose": purpose}})
    return pid, cards


def test_1a_the_combining_card_is_asked_where_rows_are_combined(client, folder):
    """With repeated rows combined per person, combining is the first consumer of a count that
    changes within people (``cigarettes``: its mean, or its most frequent value as a code), so the
    card sits on the aggregation question and lists exactly the whole-number columns that change
    within a person (pandas reference), and nothing before it."""
    pid, cards = _visits_at_aggregation(client, folder, "prediction")
    _, df = visits_table(folder)
    assert not {k: c for k, c in cards.items() if k != "aggregation" and c is not None}
    card = cards["aggregation"]
    whole = [c for c in ("age", "sodium_g", "cigarettes")
             if (df[c] == df[c].round()).all()
             and (df.groupby("pid")[c].nunique() > 1).any()]
    assert [c for g in card["groups"] for c in g["columns"]] == whole == ["cigarettes"]
    assert card["consumer"] == "combining each unit's rows"


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_1b_predictors_last_with_the_first_outcome_is_refused_or_blocked_and_recorded(
        client, folder, purpose):
    """Reference (pandas): in every person the last visit is dated after the first, so the last
    predictor value is read from a record later than the first outcome. Under prediction the
    combination is refused (no attestation passes it); under inference it is blocked: refused with
    the attestation exit, and the attested answer is recorded with the reverse-causation limitation
    in its methods sentence. The baseline-predictor exits are accepted under either purpose."""
    pid, _ = _visits_at_aggregation(client, folder, purpose)
    _, df = visits_table(folder)
    dates = pd.to_datetime(df["visit_date"])
    later = dates.groupby(df["pid"]).max() > dates.groupby(df["pid"]).min()
    assert bool(later.all())
    accepted(client, pid, {"kind": "confirm_readings", "items": [
        {"reading": "code_or_count", "column": "cigarettes", "value": "amount"}]})

    bad = {"kind": "set_aggregation", "method": "last", "outcome": "first"}
    error = refused(client, pid, bad, "predictors_after_outcome")
    exits = [e["decision"] for e in error["exits"] if e["decision"]]
    assert {"kind": "set_aggregation", "method": "first", "outcome": "last", "columns": {},
            "acknowledged": False} in exits
    attested = {**bad, "columns": {}, "acknowledged": True}
    if purpose == "prediction":
        assert attested not in exits
        refused(client, pid, attested, "predictors_after_outcome")
        accepted(client, pid, {"kind": "set_aggregation", "method": "first", "outcome": "last"})
        assert "limitation" not in last_sentence(client, pid)
    else:
        assert attested in exits
        accepted(client, pid, attested)
        said = last_sentence(client, pid)
        assert "recorded as a limitation" in said and "reverse causation" in said
        assert view(client, pid)["state"]["aggregation"]["acknowledged"] is True


def test_1c_the_rule_reads_which_records_each_summary_takes():
    """Every combination of the predictors' and the outcome's summaries, against a reference
    computed from what each summary reads: first reads record 1, last reads record m, and mean,
    mode and change read record m too. A combination is "after the outcome" when the predictors
    read a record later than every record the outcome is read from. Replicates have no order, so
    nothing is refused for them."""
    from turbotab.core.decisions import ProjectState, Refusal, validate

    latest = {"first": 1, "last": 3, "mean": 3, "change": 3, "mode": 3}
    structure = {"outcome": {"column": "sbp", "varies": True, "n_units_varying": 40,
                             "numeric": True}}
    for kind in ("time_points", "repeats"):
        for purpose in ("prediction", "inference"):
            st = ProjectState(lens=["clinical"], target="sbp", task="regression", purpose=purpose,
                              grain={"grain": "repeated", "id_column": "pid"},
                              repeat_kind={"repeat_kind": kind, "time_column": "visit_date"},
                              unit="unit")
            ctx = {"columns": ["pid", "visit_date", "sodium_g", "sbp"], "state": st,
                   "target": "sbp", "task": "regression",
                   "column_info": {"sbp": {"dtype": "numeric", "n_unique": 300}},
                   "artifact": lambda s: structure if s == "structure" else None}
            for method in ("first", "last", "mean", "change"):
                for outcome in ("first", "last", "mean"):
                    after = kind == "time_points" and latest[method] > latest[outcome]
                    try:
                        validate({"kind": "set_aggregation", "method": method, "outcome": outcome},
                                 ctx)
                        got = None
                    except Refusal as r:
                        got = r.code
                    assert (got == "predictors_after_outcome") == after, (kind, purpose, method,
                                                                          outcome, got)


# ── 2 · the outcome's order, its scale, and one 20-class rule (RO-10) ─────────


def severity_table(folder: Path) -> Path:
    rng = np.random.default_rng(10)
    n = 360
    x = rng.normal(0, 1, n)
    latent = 0.9 * x + rng.logistic(0, 1, n)
    severity = np.select([latent < -1, latent < 0.5, latent < 2], ["none", "mild", "moderate"],
                         "severe")
    path = folder / "severity.csv"
    pd.DataFrame({"participant_id": [f"S{i:04d}" for i in range(n)], "x": x.round(3),
                  "severity": severity}).to_csv(path, index=False)
    return path


def test_2a_none_mild_moderate_severe_asks_whether_the_levels_are_ordered(client, folder):
    """Reference: the clinical order of the four labels, written here. The task question is asked
    (never skipped) with "Are these levels ordered?", its proposal that order lowest first, and an
    ordinal answer waits for the order before the question is answered; an unordered answer needs
    nothing more."""
    pid = open_table(client, severity_table(folder))
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "severity"}):
        accepted(client, pid, d)
    info = artifact(client, pid, "target_info")
    question = info["order_question"]
    assert question["question"] == "Are these levels ordered?"
    assert question["proposed_order"] == ["none", "mild", "moderate", "severe"]
    assert sorted(question["levels"]) == ["mild", "moderate", "none", "severe"]
    assert {"multiclass", "ordinal"} <= set(info["fits"])
    s = walk(client, pid, "task")
    assert next(x for x in s["interview"] if x["key"] == "task")["status"] == "open"

    accepted(client, pid, {"kind": "set_task", "column": "severity", "task": "ordinal"})
    waiting = step(client, pid, "task")
    assert waiting["status"] == "open" and waiting["followup"] == "order"
    order = question["order_options"][0]["decision"]
    assert order == {"kind": "set_outcome_order", "column": "severity",
                     "levels": ["none", "mild", "moderate", "severe"]}
    accepted(client, pid, order)
    done = step(client, pid, "task")
    assert done["status"] == "answered" and done["followup"] is None
    # Unordered classes need no order.
    accepted(client, pid, {"kind": "set_task", "column": "severity", "task": "multiclass"})
    assert step(client, pid, "task")["status"] == "answered"


def crp_table(folder: Path, name: str = "crp.csv", zero: bool = False) -> tuple[Path, pd.DataFrame]:
    """hs-CRP, log-normal: ln(hs_crp) = 0.6 − 0.03 fiber + 0.012 (age − 50) + N(0, 0.9²), with
    a near-symmetric blood pressure beside it."""
    rng = np.random.default_rng(31)
    n = 500
    fiber = rng.gamma(4, 5, n).round(1)
    age = rng.normal(50, 10, n).round(1)
    crp = np.exp(0.6 - 0.03 * fiber + 0.012 * (age - 50) + rng.normal(0, 0.9, n)).round(3)
    crp = np.maximum(crp, 0.001)
    if zero:
        crp[7] = 0.0
    sbp = (125 + 0.3 * (age - 50) + rng.normal(0, 10, n)).round(1)
    # A near-symmetric glucose with six 999 codes: skewed by its codes, and as skewed on a log.
    glucose = (95 + rng.normal(0, 10, n)).round(0)
    glucose[[3, 77, 150, 260, 333, 444]] = 999
    df = pd.DataFrame({"participant_id": [f"C{i:04d}" for i in range(n)], "fiber_g": fiber,
                       "age": age, "sbp": sbp, "glucose": glucose, "hs_crp": crp})
    path = folder / name
    df.to_csv(path, index=False)
    return path, df


def test_2b_hs_crp_asks_its_scale_and_the_log_scale_is_honored(client, folder):
    """Reference: ``scipy.stats.skew(bias=False)`` of the file's hs-CRP is above West et al.'s 2 and
    of its log within ±2; of its blood pressure below 2; and its glucose, skewed only by six 999
    codes, is as skewed on a log (a code is no scale). hs-CRP's task question stays open with the
    scale question (both answers offered); blood pressure's and glucose's are not asked it. The log scale is honored, not only
    recorded: the outcome becomes ``ln_hs_crp``, whose working-table values are ``numpy.log`` of
    the file's; ``hs_crp`` is proposed excluded and refused as a predictor; and under inference the
    linear table's fiber coefficient equals NumPy least squares of ln(hs-CRP) on fiber and age.
    The original scale is recorded as a difference in means."""
    path, df = crp_table(folder)
    skew_crp = float(stats.skew(df["hs_crp"], bias=False))
    skew_log = float(stats.skew(np.log(df["hs_crp"]), bias=False))
    skew_sbp = float(stats.skew(df["sbp"], bias=False))
    assert skew_crp > 2 > abs(skew_sbp) and abs(skew_log) < 2

    # WP17 under inference: the question is fiber's effect on hs-CRP (on the log scale, the target
    # the scale answer writes), and age is drawn apart from fiber and moves hs-CRP (the generator).
    pid = open_table(client, path, {"exposure:ln_hs_crp": "fiber_g", "adjust:age": "no,yes,no"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "sbp"}):
        accepted(client, pid, d)
    info = artifact(client, pid, "target_info")
    assert info["scale_question"] is None
    walk(client, pid, "purpose")
    assert step(client, pid, "task")["status"] == "skipped"
    assert float(stats.skew(df["glucose"], bias=False)) > 2
    assert float(stats.skew(np.log(df["glucose"]), bias=False)) > 2
    accepted(client, pid, {"kind": "set_target", "column": "glucose"})
    assert artifact(client, pid, "target_info")["scale_question"] is None

    accepted(client, pid, {"kind": "set_target", "column": "hs_crp"})
    info = artifact(client, pid, "target_info")
    question = info["scale_question"]
    assert question["skewness"] == pytest.approx(skew_crp, rel=1e-9)
    assert question["log_skewness"] == pytest.approx(skew_log, rel=1e-9)
    assert question["log_column"] == "ln_hs_crp"
    assert [o["decision"]["scale"] for o in question["options"]] == ["original", "log"]
    s = walk(client, pid, "task")
    task = next(x for x in s["interview"] if x["key"] == "task")
    assert task["status"] == "open" and task["followup"] == "scale"

    accepted(client, pid, {"kind": "set_outcome_scale", "column": "hs_crp", "scale": "original"})
    assert "difference in its mean" in last_sentence(client, pid)
    assert view(client, pid)["state"]["target"] == "hs_crp"
    assert step(client, pid, "task")["status"] in ("skipped", "answered")

    accepted(client, pid, {"kind": "set_outcome_scale", "column": "hs_crp", "scale": "log"})
    assert "ratio of geometric means" in last_sentence(client, pid)
    state = view(client, pid)["state"]
    assert state["target"] == "ln_hs_crp"
    assert state["outcome_scale"] == {"column": "hs_crp", "scale": "log"}
    working = artifact(client, pid, "working")
    assert working["derived"] == [{"column": "ln_hs_crp", "source": "hs_crp", "expression": "ln"}]
    store = client.app.state.service.store(pid)
    got = store.materialize(["hs_crp", "ln_hs_crp"])
    np.testing.assert_allclose(got["ln_hs_crp"].to_numpy(float),
                               np.log(df["hs_crp"].to_numpy(float)), rtol=1e-12)
    info = artifact(client, pid, "target_info")
    assert info["column"] == "ln_hs_crp" and info["scale_question"] is None

    roles = {"participant_id": "identifier", "fiber_g": "exposure", "age": "covariate",
             "sbp": "excluded", "glucose": "excluded", "hs_crp": "covariate"}
    walk(client, pid, "roles", answers={"purpose": {"kind": "set_purpose",
                                                    "purpose": "inference"}})
    proposed = {p["column"]: p for p in artifact(client, pid, "roles")["columns"]}
    assert proposed["hs_crp"]["proposed"] == "excluded"
    assert proposed["hs_crp"]["confidence"] == "high"
    refused(client, pid, {"kind": "set_roles", "roles": roles}, "outcome_as_predictor")
    v = walk(client, pid, "models", answers={"roles": {"kind": "set_roles",
                                                       "roles": {**roles, "hs_crp": "excluded"}}})
    assert v["state"]["purpose"] == "inference"
    accepted(client, pid, {"kind": "select_models", "models": ["linear"]})
    model = artifact(client, pid, "fit")["models"][0]
    X = np.column_stack([np.ones(len(df)), df["fiber_g"].to_numpy(float), df["age"].to_numpy(float)])
    beta = np.linalg.lstsq(X, np.log(df["hs_crp"].to_numpy(float)), rcond=None)[0]
    got = {r["feature"]: r for r in model["coefficients"]}
    assert got["fiber_g"]["estimate"] == pytest.approx(beta[1], rel=1e-8)
    assert got["age"]["estimate"] == pytest.approx(beta[2], rel=1e-8)


def test_2c_the_log_scale_is_refused_where_a_value_has_no_log(client, folder):
    """A value of 0 has no logarithm: the log scale is refused with the original scale as the way
    forward, and the derived name of an existing column is never overwritten."""
    path, _ = crp_table(folder, "crp_zero.csv", zero=True)
    pid = open_table(client, path)
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "hs_crp"}):
        accepted(client, pid, d)
    error = refused(client, pid, {"kind": "set_outcome_scale", "column": "hs_crp", "scale": "log"},
                    "log_of_zero")
    assert error["exits"][0]["decision"] == {"kind": "set_outcome_scale", "column": "hs_crp",
                                             "scale": "original"}


def classes_table(folder: Path, name: str, values: Any) -> Path:
    path = folder / name
    rng = np.random.default_rng(len(values))
    pd.DataFrame({"participant_id": [f"K{i:05d}" for i in range(len(values))],
                  "x": rng.normal(0, 1, len(values)).round(3), "y": values}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("case", ["text_25", "text_20", "number_30"])
def test_2d_the_twenty_class_rule_agrees_between_the_skip_and_the_answer(client, folder, case):
    """Reference: the fixture's own count of distinct values. For each task, the explicit answer's
    verdict (200 or 409 ``task_mismatch``) is exactly membership in the target stage's ``fits``,
    and the Router skips the question only to a task the answer accepts. A text outcome of 25
    labels is never stated as multiclass (which takes at most 20); 20 labels are; a whole number of
    30 values over 3,000 rows is proposed as regression, never stated as a 30-class outcome."""
    rng = np.random.default_rng(3)
    if case == "text_25":
        values = [f"L{int(k):02d}" for k in rng.integers(1, 26, 2500)]
    elif case == "text_20":
        values = [f"L{int(k):02d}" for k in rng.integers(1, 21, 2000)]
    else:
        values = [int(k) for k in rng.integers(1, 31, 3000)]
    n_levels = len(set(values))
    pid = open_table(client, classes_table(folder, f"{case}.csv", values))
    for d in ({"kind": "set_lens", "lenses": ["clinical"]}, {"kind": "set_target", "column": "y"}):
        accepted(client, pid, d)
    info = artifact(client, pid, "target_info")
    numeric = case.startswith("number")
    expected = (["regression"] if numeric else []) + (
        ["multiclass", "ordinal"] if 2 < n_levels <= 20 else [])
    assert sorted(info["fits"]) == sorted(expected)
    walk(client, pid, "task")
    s = step(client, pid, "task")
    if s["status"] == "skipped":
        assert info["detected_task"] in info["fits"]
    assert info["confidence"] != "high" or info["detected_task"] in info["fits"]
    for task in ("binary", "multiclass", "ordinal", "regression", "time_to_event"):
        from turbotab.server.tests.conftest import when_reached

        r = when_reached(lambda: client.post(f"/api/projects/{pid}/decisions",
                                             json={"kind": "set_task", "column": "y", "task": task}))
        status, body = r.status_code, r.json()
        if task in info["fits"]:
            assert status == 200, (task, body)
        else:
            assert status == 409 and body["error"]["code"] == "task_mismatch", (task, body)
    if case == "text_25":
        assert s["status"] == "open" and info["confidence"] == "low"
        assert "at most 20" in info["reason"]
    if case == "number_30":
        assert info["detected_task"] == "regression" and s["status"] == "open"


def test_2e_the_event_and_the_task_wait_for_the_chosen_outcome_to_be_read():
    """Whether an event or a task is asked at all depends on the outcome's reading. Right after an
    outcome is chosen its stage may be idle with every input fresh (about to start, not yet
    pending): the event question was then served open for a regression outcome (seen under load as
    drivers answering an event no one should be asked). Both now wait on the reading of the outcome
    chosen, a reading of another outcome included; once read, the event does not apply and the task
    is skipped."""
    from turbotab.core.decisions import ProjectState
    from turbotab.core.interview import route

    st = ProjectState(lens=["clinical"], target="y")
    stages = {name: {"status": "fresh"} for name in
              ("ingest", "oriented", "profile", "findings", "structure", "working")}
    stages["target_info"] = {"status": "idle"}
    for info in (None, {"column": "x", "task": "regression", "confidence": "high", "reason": "r"}):
        steps = {s.key: s for s in route(st, stages, {"target_info": info} if info else {}, [])}
        assert steps["event"].status == "waiting" and steps["event"].waiting_on == ["target_info"]
        assert steps["task"].status == "waiting" and "target_info" in steps["task"].waiting_on
    stages["target_info"] = {"status": "fresh"}
    read = {"column": "y", "task": "regression", "confidence": "high", "reason": "Continuous."}
    steps = {s.key: s for s in route(st, stages, {"target_info": read}, [])}
    assert steps["event"].status == "not_applicable" and steps["task"].status == "skipped"


# ── 3 · "something else, or not sure" (RO-11) ────────────────────────────────


def generic_table(folder: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 200
    df = pd.DataFrame({"record": [f"R{i:04d}" for i in range(n)],
                       "site": ["A"] * n,
                       "x1": rng.normal(10, 2, n).round(2), "x2": rng.normal(0, 1, n).round(3),
                       "y": rng.normal(5, 1, n).round(3)})
    path = folder / "generic.csv"
    df.to_csv(path, index=False)
    return path


def test_3_something_else_or_not_sure_is_an_answer_that_runs_the_generic_checks(client, folder):
    """Accepted alone (and refused beside a lens that describes the table, with both ways
    forward); the findings stage runs, every finding is the generic structural diagnosis (no pack's
    reading, no lens), it still finds the constant column; the Router asks no assay or dietary
    question; the methods sentence says no field's defaults were applied."""
    pid = open_table(client, generic_table(folder))
    error = refused(client, pid, {"kind": "set_lens", "lenses": ["clinical", "other"]},
                    "lens_other_alone")
    assert {"kind": "set_lens", "lenses": ["other"]} in [e["decision"] for e in error["exits"]]
    accepted(client, pid, {"kind": "set_lens", "lenses": ["other"]})
    said = last_sentence(client, pid)
    assert "generic checks" in said and "no field's defaults" in said
    found = artifact(client, pid, "findings")
    assert found["findings"], "the generic checks found nothing"
    assert all(f["source"] != "pack" and f["lens"] is None for f in found["findings"])
    assert any("site" in f["affected_columns"] for f in found["findings"])
    assert "generic checks only" in found["basis"]
    accepted(client, pid, {"kind": "set_target", "column": "y"})
    steps = {s["key"]: s for s in view(client, pid)["interview"]}
    assert steps["orientation"]["status"] == "not_applicable"
    assert steps["energy_adjustment"]["status"] == "not_applicable"
    assert artifact(client, pid, "roles")["columns"]


# ── 4 · pooled QC rows leave before the seal (RO-13) ─────────────────────────


def qc_table(folder: Path) -> tuple[Path, pd.DataFrame]:
    """The audit's recipe (``repro.tar.gz``: ``F-skeptic/f7_app.py``), draw for draw: the untargeted
    fixture's pooled QCs become ``Class`` = ``QC``, the rest Case or Control at random."""
    df = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv")
    rng = np.random.default_rng(0)
    cls = np.where(df.sample_type.eq("pooled_qc"), "QC",
                   np.where(rng.random(len(df)) < 0.5, "Case", "Control"))
    out = df.drop(columns=["sample_type", "bmi", "responder"], errors="ignore")
    out.insert(1, "Class", cls)
    path = folder / "met_class_qc.csv"
    out.to_csv(path, index=False)
    return path, out


OLD_QC_TEXT = "They stay in the table for quality assessment and out of the modeling rows."


def test_4_a_case_control_qc_export_excludes_its_qc_rows_before_the_seal(client, folder):
    """Reference (pandas): 8 rows where ``Class`` is ``QC``, 72 participants, two classes among
    them. The pooled-QC finding fires on the three-level label (the legacy one needed exactly two),
    its served text no longer says the rows are already out of the model, and it offers the
    exclusion (as does the naming census's finding). Before it is applied the outcome reads three
    classes; applied before the seal, the working table holds the 72 participants, the outcome is
    binary at high confidence (the task question is skipped, the event asked), and the participant
    flow counts the 8 on a line of their own."""
    path, df = qc_table(folder)
    n_qc = int((df["Class"] == "QC").sum())
    participants = df.loc[df["Class"] != "QC", "Class"]
    assert n_qc == 8 and len(participants) == 72 and set(participants) == {"Case", "Control"}

    pid = open_table(client, path)
    accepted(client, pid, {"kind": "set_lens", "lenses": ["metabolomics"]})
    walk(client, pid, "target", answers={
        "orientation": {"kind": "set_orientation", "orientation": "sample_major"}})
    accepted(client, pid, {"kind": "set_target", "column": "Class"})
    info = artifact(client, pid, "target_info")
    assert {c["value"] for c in info["classes"]} == {"Case", "Control", "QC"}

    findings = {f["id"]: f for f in artifact(client, pid, "findings")["findings"]}
    qc = findings["pack::metabolomics::pooled_qc"]
    assert qc["severity"] == "critical" and qc["repairs"]
    assert OLD_QC_TEXT not in (qc.get("why_it_matters") or "")
    assert "Until they are excluded they are analyzed as rows like any other" in qc["why_it_matters"]
    assert not any(OLD_QC_TEXT in (f.get("why_it_matters") or "") for f in findings.values())
    option = qc["repairs"][0]
    assert option["decision"]["params"] == {"column": "Class", "levels": ["QC"]}
    assert option["effect"] == "rows"
    roles_finding = findings.get("pack::metabolomics::sample_roles")
    assert roles_finding is not None
    assert roles_finding["repairs"][0]["decision"]["params"] == {"column": "Class", "levels": ["QC"]}

    accepted(client, pid, option["decision"])
    assert "before the held-out rows were drawn" in last_sentence(client, pid)
    working = artifact(client, pid, "working")
    assert working["n_rows"] == 72
    assert working["reference_rows"] == [{"column": "Class", "levels": ["QC"],
                                          "finding": "pack::metabolomics::pooled_qc", "n": 8}]
    info = artifact(client, pid, "target_info")
    assert info["task"] == info["detected_task"] == "binary" and info["confidence"] == "high"
    assert {c["value"] for c in info["classes"]} == {"Case", "Control"}
    assert [c["count"] for c in sorted(info["classes"], key=lambda c: c["value"])] == [
        int((participants == "Case").sum()), int((participants == "Control").sum())]
    s = walk(client, pid, "event")
    assert next(x for x in s["interview"] if x["key"] == "task")["status"] == "skipped"
    flow = artifact(client, pid, "cohort")["steps"]
    assert flow[0] == {**flow[0], "key": "loaded", "n": 80}
    assert flow[1]["key"] == "reference:0" and flow[1]["dropped"] == 8 and flow[1]["n"] == 72


def test_4b_reference_rows_leave_only_before_the_seal():
    """Once the held-out rows are drawn, excluding reference rows would draw them again: refused."""
    from turbotab.core import repairs  # noqa: F401 - registers the reference-rows family
    from turbotab.core.decisions import ProjectState, Refusal, validate

    st = ProjectState(lens=["metabolomics"], target="Class",
                      split={"holdout": 0.2, "seed": 0, "folds": 5})
    with pytest.raises(Refusal) as caught:
        validate({"kind": "apply_repair", "finding_id": "pack::metabolomics::pooled_qc",
                  "option": "exclude_rows", "params": {"column": "Class", "levels": ["QC"]}},
                 {"state": st})
    assert caught.value.code == "rows_after_the_seal"


# ── 5 · NHANES DXA imputed copies (I18) ──────────────────────────────────────


def dxa_table(folder: Path) -> tuple[Path, pd.DataFrame]:
    """NHANES DXA's shape: each SEQN five times, ``_MULT_`` 1–5; a third of the people have an
    imputed total percent fat that differs between copies; age, sex and fiber are the same in all
    five."""
    rng = np.random.default_rng(1999)
    rows = []
    for i in range(240):
        age = round(float(rng.normal(45, 12)), 1)
        fiber = round(float(rng.gamma(4, 4)), 1)
        fat = 30 + 0.1 * (age - 45) - 0.2 * (fiber - 16) + float(rng.normal(0, 4))
        imputed = rng.random() < 1 / 3
        for k in range(1, 6):
            value = fat + (float(rng.normal(0, 2)) if imputed else 0.0)
            rows.append({"SEQN": 30000 + i, "_MULT_": k, "RIDAGEYR": age,
                         "fiber_g": fiber, "DXDTOPF": round(value, 1)})
    df = pd.DataFrame(rows)
    path = folder / "dxa.csv"
    df.to_csv(path, index=False)
    return path, df


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_5_imputed_copies_are_read_asked_and_blocked_and_recorded(client, folder, purpose):
    """Reference (pandas): every SEQN holds ``_MULT_`` 1 to 5 once, and the outcome differs
    between copies for the imputed third. The repeats reading proposes imputed copies (asked, not
    stated). Under inference the answer is accepted and its sentence says each copy is analyzed with
    its own outcome and pooled by Rubin's rules (MS3); combining the copies per unit is blocked and
    recorded (CDC: a single copy, or copies analyzed as values, cannot carry the imputation's
    variability), its exits keeping the copies as records first, and the attested answer's methods
    sentence states the limitation. The copies have no time order, so the temporal question does not
    apply and first, last and change are refused. Under prediction the answer is accepted and its
    sentence states the concern."""
    path, df = dxa_table(folder)
    per = df.groupby("SEQN")["_MULT_"].apply(lambda s: sorted(s.tolist()))
    assert all(v == [1, 2, 3, 4, 5] for v in per)
    assert (df.groupby("SEQN")["DXDTOPF"].nunique() > 1).sum() > 0

    pid = open_table(client, path, {"code_or_count:_MULT_": "code"})
    for d in ({"kind": "set_lens", "lenses": ["clinical"]},
              {"kind": "set_target", "column": "DXDTOPF"}):
        accepted(client, pid, d)
    walk(client, pid, "repeat_kind", answers={
        "purpose": {"kind": "set_purpose", "purpose": purpose},
        "grain": {"kind": "set_grain", "grain": "repeated", "id_column": "SEQN"}})
    reading = artifact(client, pid, "structure")["repeats"]
    assert reading["reading"] == "imputed_copies" and reading["implicate_column"] == "_MULT_"
    assert reading["confidence"] == "medium"
    assert step(client, pid, "repeat_kind")["status"] == "open"

    answer = {"kind": "set_repeat_kind", "repeat_kind": "imputed_copies",
              "implicate_column": "_MULT_"}
    accepted(client, pid, answer)
    said = last_sentence(client, pid)
    assert "imputed copies" in said and "Rubin's rules" in said
    assert "uncertainty" in said
    assert step(client, pid, "temporal")["status"] == "not_applicable"
    if purpose == "inference":
        assert "pooled by Rubin's rules" in said
        error = refused(client, pid, {"kind": "set_unit", "unit": "unit"}, "imputed_copies_combined")
        assert error["exits"][0]["decision"] == {"kind": "set_unit", "unit": "row"}
        attested = next(e["decision"] for e in error["exits"]
                        if e["decision"] and e["decision"].get("acknowledged"))
        accepted(client, pid, attested)
        assert "recorded as a limitation" in last_sentence(client, pid)
    accepted(client, pid, {"kind": "set_unit", "unit": "unit"})
    walk(client, pid, "aggregation")
    refused(client, pid, {"kind": "set_aggregation", "method": "first", "outcome": "first"},
            "copies_have_no_order")
