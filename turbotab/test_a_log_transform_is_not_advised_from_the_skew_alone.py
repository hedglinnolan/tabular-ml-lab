"""`AUDIT-003` — the R5 target card and the log transform it used to advise blind.

`ml/eda_recommender.py`'s `r5_target_regression` card said the same two things
about **every** regression target, whatever its values were:

    what_you_learn:     "Need for log transformation or robust loss"
    model_implications: "High skew -> consider log transform or robust loss (Huber)"

Both are composed from the skew and nothing else. `wide_assay.csv::probe_017`
holds 38 negative values out of 60; `log(x)` is undefined on every one of them,
and the card recommended the transform anyway. `metabolomics_paired_logged.csv`
is a table this repository ships **already logged** — its columns run 8.2 to
11.2 — and the card recommended a second log on it, which
`research/METABOLOMICS_PACK.md` calls *"a silent catastrophe"* because it
produces numbers and every plot still renders.

## What the corrected card has to do

`research/METABOLOMICS_PACK.md`, "Value-state diagnostics":

> **Already-transformed detection.** Any negative values, or a max below ~40
> with a positive min and low dynamic range, or column means ~ 0 -> probably
> already log-transformed and/or scaled. Warn hard; a second log transform is a
> silent catastrophe.

The two clauses do not carry the same weight on one arbitrary column and this
file checks them **differently**, because collapsing them would be the same
defect one surface over:

1. **Non-positive values are arithmetic.** The advice is withdrawn and the
   count is named. Certain on any table in any field.
2. **A compressed range is a reading, not a verdict.** It is what an
   already-logged assay column looks like AND what `creatinine_mg_dl` (0.3 to
   3.85 mg/dL, raw) looks like. The card reports it, names both readings, and
   says the numbers cannot separate them — it does not assert "already
   transformed".
3. **A strictly positive, widely spread target still gets the recommendation.**
   `test_a_raw_positive_target_is_still_offered_the_transform` is the positive
   control (`GUIDED-045`), and it is why this file cannot be satisfied by
   deleting the word *log*.
4. **Every branch names what it did not check** — provenance. A log applied
   before the CSV was written is recorded nowhere the app can read.

Thresholds are imported from `turbotab/packs.py` rather than restated; none
moved this loop.

## Fixture shapes — `GUIDED-097`

| target shape | fixture | reading required |
|---|---|---|
| **float with negatives** | `wide_assay.csv::probe_017` (38 of 60 <= 0) | `log_undefined` |
| **float, already logged** | `metabolomics_paired_logged.csv::mz_0005` | `compressed_scale` |
| **float, raw and wide** | `dietary_recalls.csv::energy_kcal` (257-7,801) | `no_signature` |
| **int count** | `clinical_risk.csv::length_of_stay_days` (1-20) | `compressed_scale` |

**Not covered.** A *classification* target — R5's regression card does not fire
at all there, so there is no log sentence to correct. A target that is entirely
missing or non-numeric — `compute_dataset_signals` computes no `skew` for it
either, and the card is not built.

## And what this file does NOT claim

The corrected sentence is **not rendered to a user in either door today**, and
the last test in this file says so with `xfail(strict=True)` rather than with
silence (trap 1, `GUIDED-119`'s model):

* **Guided** — `ml/router.py:397` puts the card's `why` on the
  `look::r5_target_regression` pull chip, and `turbotab/api.py`'s capability
  table has no entry for that key, so `built` is `False` and
  `turbotab/web/index.html:6529` renders `not_built_reason` in place of `why`.
* **Classic** — `pages/02_EDA.py:289` computes `recommend_eda(signals)` into
  `eda_recommendations` and never reads the variable again.

The rendered instance of this row's defect lives in **`ml/eda_actions.py:417`
and `:426`** (reached from `pages/02_EDA.py:1817`), which this chunk does not
own. That half is reported blocked, not closed.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.eda_recommender import (                                     # noqa: E402
    compute_dataset_signals, read_log_transform_state, recommend_eda,
)

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: The sentence the card used to compose for every regression target alike.
BLIND = "consider log transform or robust loss (Huber)"

NEGATIVE = ("wide_assay.csv", "probe_017")
LOGGED = ("metabolomics_paired_logged.csv", "mz_0005")
RAW_WIDE = ("dietary_recalls.csv", "energy_kcal")
COUNT = ("clinical_risk.csv", "length_of_stay_days")


def _card(fixture: str, target: str):
    frame = pd.read_csv(DATA / fixture)
    signals = compute_dataset_signals(frame, target, "regression",
                                      "cross_sectional", None)
    cards = [c for c in recommend_eda(signals)
             if c.id == "r5_target_regression"]
    assert cards, "the regression target card was not built at all"
    card = cards[0]
    # GUIDED-045's positive control for every absence assertion below: the
    # lists being swept are non-empty before anything is asserted missing.
    assert card.model_implications, "nothing to sweep in model_implications"
    assert card.what_you_learn, "nothing to sweep in what_you_learn"
    assert card.why, "nothing to sweep in why"
    return card, signals


# ── the reading itself ──────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target,expected", [
    (*NEGATIVE, "log_undefined"),
    (*LOGGED, "compressed_scale"),
    (*RAW_WIDE, "no_signature"),
    (*COUNT, "compressed_scale"),
])
def test_the_values_are_read_before_a_transform_is_named(fixture, target, expected):
    series = pd.read_csv(DATA / fixture)[target]
    state = read_log_transform_state(series)
    assert state["reading"] == expected, state


def test_the_reading_is_carried_on_the_signals_the_card_is_built_from():
    """`recommend_eda` never sees the frame. If the reading is not on
    `target_stats` the card cannot make it, and this is the join."""
    _, signals = _card(*RAW_WIDE)
    assert "log_transform_state" in signals.target_stats, signals.target_stats.keys()


# ── the advice, per situation ───────────────────────────────────────────────

def test_a_target_with_negative_values_is_not_advised_to_take_a_log():
    """38 of 60 values are <= 0. `log` is undefined on them and the card used to
    recommend it regardless — the sharpest form of the row."""
    card, _ = _card(*NEGATIVE)
    said = " ".join(card.model_implications)
    assert BLIND not in said, said
    assert "NOT available" in said, said
    assert "38 of 60" in said, said
    assert "undefined" in said, said


def test_an_already_logged_target_is_not_advised_to_take_a_second_log():
    """`metabolomics_paired_logged.csv` ships logged. The card must not
    recommend the transform outright, and must say what it read."""
    card, _ = _card(*LOGGED)
    said = " ".join(card.model_implications)
    assert BLIND not in said, said
    assert "8.195 to 11.2" in said, said
    assert "silent" in said, said


def test_the_compressed_reading_is_offered_and_not_asserted():
    """A compressed span is also what a raw clinical measurement in its own
    units looks like. The card names both readings rather than picking one,
    which is the whole correction: it says LESS, and it is true."""
    card, _ = _card(*COUNT)
    said = " ".join(card.model_implications)
    assert "cannot tell those two apart" in said, said
    assert "already transformed" not in said.lower(), (
        "the card asserts a verdict the numbers do not support: " + said)


def test_a_raw_positive_target_is_still_offered_the_transform():
    """The positive control. `energy_kcal` runs 257 to 7,801 kcal, strictly
    positive and widely spread — a log transform is a real option and the app
    still offers it. Deleting the advice everywhere would fail here."""
    card, _ = _card(*RAW_WIDE)
    said = " ".join(card.model_implications)
    assert BLIND in said, said
    assert "257 to 7,801" in said, said


@pytest.mark.parametrize("fixture,target",
                         [NEGATIVE, LOGGED, RAW_WIDE, COUNT])
def test_every_branch_names_the_check_it_could_not_make(fixture, target):
    """None of the four readings can see provenance. Saying so is the clause
    that keeps the narrowed claim honest rather than merely quieter."""
    card, _ = _card(fixture, target)
    said = " ".join(card.model_implications)
    assert "Provenance is not checked" in said, said


def test_the_card_does_not_promise_a_conclusion_it_cannot_reach():
    """`what_you_learn` used to promise "Need for log transformation or robust
    loss" on a target where the log is not even defined."""
    card, _ = _card(*NEGATIVE)
    learn = " ".join(card.what_you_learn)
    assert "Need for log transformation" not in learn, learn
    assert "unavailable" in learn, learn


# ── where the sentence gets to, and where it does not ───────────────────────

@pytest.mark.parametrize("fixture,target", [NEGATIVE, LOGGED, RAW_WIDE])
def test_the_reading_survives_the_router_into_the_interview_payload(fixture, target):
    """Driven over HTTP, not grepped. `ml/router.py:397` joins the card's `why`
    into the pull chip; this asserts the reading is in that payload. It does
    NOT assert a person sees it — the test below is what says that."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    body = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_target",
                             "payload": {"column": target}}).json()
    assert body.get("task_type") == "regression", body.get("task_type")
    plan = client.get(f"/project/{pid}/interview?step=explore").json()
    chip = next((q for q in plan["questions"]
                 if q["key"] == "look::r5_target_regression"), None)
    assert chip is not None, [q["key"] for q in plan["questions"]]
    assert "Log transform:" in (chip.get("why") or ""), chip.get("why")


# ── the same row's other instance, and this one IS rendered ─────────────────
#
# `ml/model_coach.py`'s `preprocess_skewness_transform` insight said "apply
# Yeo-Johnson or log transform" for every member of
# `profile.highly_skewed_features`, which `ml/dataset_profile.py:641` fills on
# `abs(skewness) > 1.0`. That set holds LEFT-skewed features, where a log
# lengthens the tail it is meant to shorten, and features holding zeros or
# negatives, where a log is undefined. `pages/05_Preprocess.py:327` calls it and
# writes the result to the insight ledger, so this half reaches a person.

def _skew_frame() -> pd.DataFrame:
    """One right-skewed positive column, one left-skewed, one with negatives.

    Constructed rather than read from `sample_data`, and said so: the three
    populations have to sit in one frame for the split to be observable, and no
    fixture holds all three above the 1.0 skew gate.
    """
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "right_pos": rng.lognormal(0, 1.2, 300),
        "left_pos": 140 - rng.lognormal(0, 1.2, 300),
        "has_neg": rng.lognormal(0, 1.2, 300) - 3.0,
        "y": rng.normal(size=300),
    })


def _skew_action(frame: pd.DataFrame) -> str:
    from ml.dataset_profile import compute_dataset_profile
    from ml.model_coach import generate_preprocessing_insights

    profile = compute_dataset_profile(frame, target_col="y",
                                      task_type="regression")
    assert profile.highly_skewed_features, "nothing was flagged as skewed"
    insights = generate_preprocessing_insights(["ridge", "rf"], profile)
    rows = [i for i in insights if i["id"] == "preprocess_skewness_transform"]
    assert rows, [i["id"] for i in insights]
    return rows[0]["recommended_action"]


def test_the_preprocessing_card_does_not_offer_a_log_to_every_skewed_feature():
    said = _skew_action(_skew_frame())
    assert "apply Yeo-Johnson or log transform" not in said, said
    assert "zero or below, where the log is undefined" in said, said
    assert "skewed LEFT" in said, said


def test_yeo_johnson_is_still_recommended_for_all_of_them():
    """The positive control on this half. Yeo-Johnson IS defined on zeros,
    negatives and either tail, so narrowing the LOG must not narrow it."""
    said = _skew_action(_skew_frame())
    assert "apply Yeo-Johnson to all 3" in said, said


def test_the_log_is_still_offered_where_it_applies():
    """And the second positive control: one of the three is strictly positive
    and skewed right, which is the log's own case."""
    said = _skew_action(_skew_frame())
    assert "A LOG transform is an option for 1 of them (right_pos)" in said, said


def test_the_split_is_machine_readable_and_not_only_prose():
    """Trap 7 — the structured payload beside the sentence dropping half of it.
    `metadata` is what the insight ledger and the manuscript read."""
    from ml.dataset_profile import compute_dataset_profile
    from ml.model_coach import generate_preprocessing_insights

    profile = compute_dataset_profile(_skew_frame(), target_col="y",
                                      task_type="regression")
    row = [i for i in generate_preprocessing_insights(["ridge", "rf"], profile)
           if i["id"] == "preprocess_skewness_transform"][0]
    assert row["metadata"]["log_ok"] == ["right_pos"], row["metadata"]
    assert row["metadata"]["not_positive"] == ["has_neg"], row["metadata"]
    assert row["metadata"]["left_skewed"] == ["left_pos"], row["metadata"]


def test_the_classic_page_source_still_calls_the_preprocessing_coach():
    """A SOURCE read and the name says so (trap 3b). `pages/05_Preprocess.py` is
    frozen Streamlit; what is checked is that it still calls the function whose
    sentence was corrected."""
    page = (ROOT / "pages" / "05_Preprocess.py").read_text(encoding="utf-8")
    assert "generate_preprocessing_insights(_selected_for_coaching, profile)" in page, (
        "the Preprocess page no longer calls the preprocessing coach")


@pytest.mark.xfail(strict=True, reason=(
    "AUDIT-003's correction is composed and not rendered. "
    "turbotab/api.py's capability table has no look::r5_target_regression "
    "entry, so the chip is built=False and turbotab/web/index.html:6529 "
    "shows not_built_reason instead of why. Wiring the chip is another "
    "chunk's file; when it lands this xfail turns green and the sentence "
    "should be re-read on the page before this marker is removed."))
def test_the_guided_chip_that_would_show_the_reading_is_wired():
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    fixture, target = RAW_WIDE
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    plan = client.get(f"/project/{pid}/interview?step=explore").json()
    chip = next(q for q in plan["questions"]
                if q["key"] == "look::r5_target_regression")
    assert chip.get("built") is True, chip.get("not_built_reason")
