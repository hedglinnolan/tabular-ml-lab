"""`AUDIT-012`, the two surfaces the first pass did not reach.

`L52` corrected the Model Coach's Huber sentence (`ml/model_coach.py`) and wired
the impossibility tier onto `TargetProfile`. Two other surfaces compose advice
from the same undifferentiated `ml/outliers.py:39` fence count, and both were
still saying it:

**1 · the Guided finding card** — `ml/dataset_profile.py`'s `outliers` warning:

    detail: "4 numeric features have significant outliers. This can affect
             model performance, especially for distance-based and linear
             models."
    suggested_actions: [..., "Consider winsorizing or capping", ...]

On `clinical_labs.csv` those four columns are `sbp`, `dbp`, `temp_f` and `bnp`.
`sbp` holds four entries outside 40–300 mmHg — entry errors. `dbp`'s fence hits
are all inside 15–220 mmHg — real. `temp_f` and `bnp` match no reference
variable at all, so nothing could have been read for them. One sentence carried
all three, and the winsorizing it offered would hide the first, discard the
second, and act blind on the third.

**2 · the R5 and R9 EDA cards** — `ml/eda_recommender.py`:

    R5 model_implications: "High outlier rate → use Huber loss or winsorization"
    R9 model_implications: "High outlier rate → use Huber loss or robust
                            regression" / "Consider winsorization or outlier
                            removal"

R9's second line is the sharpest form the row has anywhere in the tree: an
offer to **remove** values, composed from a number that never looked at whether
they were possible.

## What the research requires

`research/CLINICAL_SURVEY_PACK.md` §A1.2 keeps two bound sets apart and says
what each is for:

> **Physiological plausibility bounds** — values incompatible with a living
> patient. Use for flagging as suspected data error.
> **Reference interval** — the central 95% of a healthy reference population…
> **Use only for annotation. Never for exclusion.**

and Cross-cutting 7 ranks the collapse **seventh by damage**:

> Excluding abnormal-but-possible clinical values as "outliers" (A1.2).
> **Removes the sickest patients.** Physiologically impossible ≠ abnormal, and
> generic outlier rules (±3 SD, IQR fences) are wrong here.

## What the correction is, and what it is not

It is **not** a retreat from advice. Huber is still recommended in every branch
of the card, all four `suggested_actions` on the warning survive verbatim, and
the fence rate is still reported to three significant figures.
`test_the_shelf_is_not_shortened_in_any_branch` is the positive control
(`GUIDED-045`), and it is why this file cannot be satisfied by deleting a
sentence.

What shrinks is the **claim**: each surface now says which tier it read, and
where it read nothing it says the app cannot tell an impossible entry from an
abnormal-but-real one, and names why.

| branch | fixture | what the sentence must do |
|---|---|---|
| some values impossible | `clinical_labs.csv` (`sbp`, 4 outside 40–300 mmHg) | name the count and the band; send them to repair, not to winsorizing |
| recognized, none impossible | `clinical_labs.csv` (`dbp`, band 15–220 mmHg) | say the fence hits read as real extremes |
| nothing recognized at all | `wide_assay.csv` (22 flagged, none in the reference) | say the app **cannot tell them apart**, and why |

## Fixture shapes — `GUIDED-097`

The warning half is a claim about a **journey step** (the finding list the
Upload step presents), so it runs against three target shapes:

| target shape | fixture | branch reached |
|---|---|---|
| **int64 binary classification** | `clinical_labs.csv::diabetes` | all three at once |
| **float64 regression** | `clinical_labs.csv::sbp` | recognized-clean + unrecognized |
| **float64 regression, wide** | `wide_assay.csv::probe_017` | unrecognized only |

The card half runs against `float64` targets (`sbp`, `dbp`, `bnp`) and one
**constructed `int64`** target, because no shipped fixture has an integer
regression target whose IQR rate clears R9's 5% gate —
`longitudinal_visits.csv::bp_sys` and `clinical_risk.csv::length_of_stay_days`
read 0.0% and 1.9%.

**Not covered, said out loud.**

* A **classification** target: R9 is regression-only and R5 takes its other
  branch, so there is no outlier sentence there to correct. The *warning* half
  is target-independent and IS driven on a classification fixture above.
* A **string-sentinel** column such as `clinical_labs.csv::ferritin` (`">1500"`):
  `object` dtype, so `detect_outliers` returns before fencing and the column
  never enters `features_with_outliers`.
* **Pediatric or pregnancy data**, which §A1.2 marks explicitly: *"never apply
  adult bounds."* The reference bundle read here is adult NHANES, and neither
  the warning nor the card asks whether the cohort is adult. That gap is
  unchanged by this loop and is not claimed to be closed.

## Where each half lands

* **Warning** — `turbotab/engine.py:796` flattens `profile.warnings` into
  `rank_findings`, `turbotab/api.py:291` is the only producer of the finding
  list, and `turbotab/web/index.html:2422` writes `f.detail` into the card.
  Driven over HTTP here, not grepped. **This half reaches a person.**
* **Cards** — `ml/router.py:397` carries only `title` and `why` onto the pull
  chip, and `turbotab/api.py`'s `PULL_CAPABILITIES` has no
  `look::r5_target_regression` or `look::r9_outlier_influence` entry, so the
  chip renders `not_built_reason` instead of `why`;
  `pages/02_EDA.py:289` assigns `recommend_eda(...)` and never reads it. The
  correction is on the wire and not on a screen, and the assertions below say
  only that — `test_the_reading_reaches_the_interview_payload` drives the
  payload and claims nothing about a rendering.
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

from ml.dataset_profile import compute_dataset_profile          # noqa: E402
from ml.eda_recommender import (                                # noqa: E402
    compute_dataset_signals, recommend_eda,
)

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: The warning sentence that carried all three tiers alike.
BLIND_WARNING = "numeric features have significant outliers"

#: The two card lines that did the same.
BLIND_R5 = "High outlier rate → use Huber loss or winsorization"
BLIND_R9_REMOVE = "Consider winsorization or outlier removal"

ALL_THREE = ("clinical_labs.csv", "diabetes", "classification")
RECOGNIZED = ("clinical_labs.csv", "sbp", "regression")
NONE_RECOGNIZED = ("wide_assay.csv", "probe_017", "regression")


# ── the warning: helpers ────────────────────────────────────────────────────

def _warning(fixture: str, target: str, task: str):
    """The `outliers` warning, with `GUIDED-045`'s positive control in front of
    every absence assertion made against it."""
    frame = pd.read_csv(DATA / fixture)
    profile = compute_dataset_profile(frame, target_col=target, task_type=task)
    assert profile.features_with_outliers, (
        "nothing tripped the fence on this fixture, so there is no sentence "
        "to check and an absence assertion below would pass for free")
    found = [w for w in profile.warnings if w.category == "outliers"]
    assert found, [w.category for w in profile.warnings]
    warning = found[0]
    assert warning.detailed_message, "the warning carries no sentence at all"
    assert warning.suggested_actions, "nothing to sweep in suggested_actions"
    return warning, profile


# ── the warning: per situation ──────────────────────────────────────────────

def test_an_impossible_entry_is_not_sent_to_winsorizing():
    """The sharpest form on this surface. Four systolic readings on
    `clinical_labs.csv` sit outside 40–300 mmHg; winsorizing pulls them to a
    percentile and the entry error survives, unrepaired and now invisible."""
    warning, _ = _warning(*ALL_THREE)
    said = warning.detailed_message
    assert BLIND_WARNING not in said, said
    assert "sbp (4 outside 40–300 mmHg)" in said, said
    assert "suspected entry errors" in said, said
    assert "repair them on the plausibility card" in said, said


def test_a_recognized_clean_column_is_named_as_real_extremes():
    """`dbp` trips the same fence and nothing is outside 15–220 mmHg. Same
    remedy, opposite reason — which is the distinction the one sentence could
    not draw."""
    warning, _ = _warning(*ALL_THREE)
    said = warning.detailed_message
    assert "dbp" in said, said
    assert "read as real extremes" in said, said
    assert "removes the sickest rows" in said, said


def test_an_unrecognized_column_says_the_app_cannot_tell_them_apart():
    """The clause the row exists for. `temp_f` and `bnp` match no variable in
    the physiologic reference, so no floor or ceiling was ever consulted."""
    warning, _ = _warning(*ALL_THREE)
    said = warning.detailed_message
    assert "cannot tell an impossible entry from an abnormal-but-real one" in said, said
    assert "matches no variable in the physiologic reference" in said, said
    assert "bnp" in said and "temp_f" in said, said


def test_a_table_the_reference_has_never_heard_of_claims_nothing():
    """`wide_assay.csv` flags 22 probe columns and the reference knows none of
    them. The whole sentence has to be the refusal, not a tier report with an
    empty tier."""
    warning, profile = _warning(*NONE_RECOGNIZED)
    said = warning.detailed_message
    assert BLIND_WARNING not in said, said
    assert "cannot tell an impossible entry from an abnormal-but-real one" in said, said
    assert "published plausibility bounds" not in said, (
        "nothing was outside a band because no band was read; naming one "
        "here would report a check that never ran: " + said)


def test_the_sentence_is_read_on_a_second_target_shape():
    """`GUIDED-097`. The same journey step, a `float64` regression target
    instead of the `int64` binary one — `sbp` is now the target and drops out
    of the feature list, so the sentence has to re-read rather than replay."""
    warning, profile = _warning(*RECOGNIZED)
    said = warning.detailed_message
    assert "sbp" not in profile.features_with_outliers, profile.features_with_outliers
    assert BLIND_WARNING not in said, said
    assert "dbp" in said, said
    assert "read as real extremes" in said, said


def test_the_fence_is_still_named_as_a_fence_and_the_count_is_still_reported():
    """The correction is to what the count is CLAIMED to mean. Losing the
    count would be a different defect wearing this one's clothes."""
    warning, profile = _warning(*ALL_THREE)
    said = warning.detailed_message
    n = len(profile.features_with_outliers)
    assert f"{n} numeric feature" in said, said
    assert "1.5×IQR fence" in said, said
    assert warning.short_message == f"{n} features with outliers", (
        warning.short_message)


# ── the warning: driven to the surface that renders it ──────────────────────

def test_the_corrected_sentence_reaches_the_guided_finding_card():
    """Driven over HTTP, not grepped (trap 5). `turbotab/api.py:291` is the one
    producer of the finding list and `turbotab/web/index.html:2422` writes
    `f.detail` into the card, so this is the rendered instance of the row."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    fixture = ALL_THREE[0]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    body = client.get(f"/project/{pid}").json()
    cards = [f for f in body.get("findings", [])
             if (f.get("params") or {}).get("category") == "outliers"]
    assert cards, [(f.get("params") or {}).get("category")
                   for f in body.get("findings", [])]
    detail = cards[0].get("detail") or ""
    assert BLIND_WARNING not in detail, detail
    assert "40–300 mmHg" in detail, detail
    assert "cannot tell an impossible entry" in detail, detail


# ── the R5 / R9 cards ───────────────────────────────────────────────────────

def _cards(frame: pd.DataFrame, target: str):
    signals = compute_dataset_signals(frame, target, "regression",
                                      "cross_sectional", None)
    built = {c.id: c for c in recommend_eda(signals)}
    return built, signals


def _int_target_frame() -> pd.DataFrame:
    """An `int64` systolic column with entry-error zeros.

    Constructed, and said so: no shipped fixture has an integer regression
    target whose IQR rate clears R9's 5% gate, so the int64 shape cannot be
    driven through R9 from `sample_data` at all.
    """
    rng = np.random.default_rng(19)
    values = rng.integers(108, 142, size=240).astype("int64")
    values[:26] = 0
    return pd.DataFrame({
        "bp_sys": values,
        "age": rng.integers(30, 80, size=240),
        "chol": rng.normal(190, 30, size=240),
    })


def _implications(card) -> str:
    assert card.model_implications, "nothing to sweep in model_implications"
    assert card.why, "nothing to sweep in why"
    return " ".join(card.model_implications)


def test_r9_no_longer_offers_to_remove_values_it_never_classified():
    """The sharpest line in the tree. `clinical_labs.csv::bnp` trips the fence
    at 6.2% and matches no reference variable — so the app had no basis
    whatsoever for offering removal, which §A1.2's Cross-cutting 7 names as the
    way the sickest patients leave the table."""
    built, _ = _cards(pd.read_csv(DATA / "clinical_labs.csv"), "bnp")
    card = built.get("r9_outlier_influence")
    assert card is not None, sorted(built)
    said = _implications(card)
    assert BLIND_R9_REMOVE not in said, said
    assert "cannot tell a physiologically impossible entry" in said, said
    assert "matches no variable in the physiologic reference" in said, said
    assert "not offered on this reading" in said, said


def test_r5_names_the_band_when_one_was_read():
    """`sbp` is recognized and four of its values are outside 40–300 mmHg."""
    built, _ = _cards(pd.read_csv(DATA / "clinical_labs.csv"), "sbp")
    said = _implications(built["r5_target_regression"])
    assert BLIND_R5 not in said, said
    assert "40–300 mmHg" in said, said
    assert "suspected entry errors" in said, said
    assert "downweight" in said.lower() or "DOWNWEIGHT" in said, said


def test_r5_says_the_extremes_read_as_real_where_nothing_is_impossible():
    """`dbp`: same fence, same recommendation, opposite reason."""
    built, _ = _cards(pd.read_csv(DATA / "clinical_labs.csv"), "dbp")
    said = _implications(built["r5_target_regression"])
    assert BLIND_R5 not in said, said
    assert "15–220 mmHg" in said, said
    assert "read as real extremes" in said, said


def test_an_integer_target_takes_the_same_reading_through_r9():
    """`GUIDED-097` on the card half — the constructed `int64` shape, 26 zeros
    below the 40 mmHg floor."""
    built, _ = _cards(_int_target_frame(), "bp_sys")
    card = built.get("r9_outlier_influence")
    assert card is not None, sorted(built)
    said = _implications(card)
    assert BLIND_R9_REMOVE not in said, said
    assert "40–300 mmHg" in said, said
    assert "suspected entry errors" in said, said


def test_the_reading_reaches_the_interview_payload():
    """Driven through the router, not grepped. This asserts the reading is on
    the pull chip's `why` — it does NOT assert anybody sees it, and the module
    docstring says which two doors drop it."""
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "clinical_labs.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("clinical_labs.csv", fh, "text/csv")}).json()["id"]
    body = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_target",
                             "payload": {"column": "sbp"}}).json()
    assert body.get("task_type") == "regression", body.get("task_type")
    plan = client.get(f"/project/{pid}/interview?step=explore").json()
    keys = [q["key"] for q in plan["questions"]]
    chip = next((q for q in plan["questions"]
                 if q["key"] == "look::r5_target_regression"), None)
    assert chip is not None, keys
    why = chip.get("why") or ""
    assert "Outlier rate:" in why, why
    assert "40–300 mmHg" in why, why


# ── the positive control ────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target", [
    ("clinical_labs.csv", "sbp"),
    ("clinical_labs.csv", "dbp"),
    ("clinical_labs.csv", "bnp"),
])
def test_the_shelf_is_not_shortened_in_any_branch(fixture, target):
    """`GUIDED-045` and `PRODUCT_VISION.md`'s *the shelf is never shortened*.

    Huber survives in every branch. This one is GREEN BOTH BEFORE AND AFTER the
    correction on purpose — that is what a control is for: it goes red only if
    a future edit satisfies the assertions above by deleting the advice."""
    built, _ = _cards(pd.read_csv(DATA / fixture), target)
    said = _implications(built["r5_target_regression"])
    assert "Huber" in said, said


@pytest.mark.parametrize("fixture,target", [
    ("clinical_labs.csv", "sbp"),
    ("clinical_labs.csv", "dbp"),
    ("clinical_labs.csv", "bnp"),
])
def test_the_fence_rate_is_still_reported_to_the_tenth(fixture, target):
    """And the number survives too. The old line named a remedy without ever
    printing the rate it was chosen from; the corrected one prints it."""
    built, signals = _cards(pd.read_csv(DATA / fixture), target)
    said = _implications(built["r5_target_regression"])
    rate = signals.target_stats.get("outlier_rate")
    assert rate is not None, sorted(signals.target_stats)
    assert f"{rate:.1%}" in said, (rate, said)


def test_all_four_suggested_actions_survive_on_the_warning():
    """The warning's options are what the Guided card turns into controls
    (`turbotab/actions.py` classifies each one). Narrowing the SENTENCE must
    not quietly narrow the offers, and this is what says so."""
    warning, _ = _warning(*ALL_THREE)
    assert warning.suggested_actions == [
        "Investigate if outliers are errors or genuine",
        "Consider robust models (Huber loss)",
        "Consider winsorizing or capping",
        "Tree models are robust to outliers",
    ], warning.suggested_actions


# ── the reader underneath, checked once ─────────────────────────────────────

def test_the_reader_reports_nothing_rather_than_zero_where_it_read_nothing():
    """Trap 9, on the feature side. Empty tier lists would say "no column here
    is impossible", which is a measurement; `None` is its absence.

    Imported inside the test on purpose: `read_outlier_tiers` is new, so a
    module-level import would make a revert probe die on `ImportError` — red
    for the wrong reason — instead of on the sentence."""
    from ml.dataset_profile import read_outlier_tiers

    frame = pd.read_csv(DATA / "clinical_labs.csv")
    unread = read_outlier_tiers(frame, [])
    assert unread["read"] is False, unread
    assert unread["impossible"] is None, unread
    assert unread["within_band"] is None, unread
    assert unread["unrecognized"] is None, unread
    assert unread["reason"], "the absence is not stated"

    read = read_outlier_tiers(frame, ["sbp", "dbp", "bnp"])
    assert read["read"] is True, read
    assert set(read["impossible"]) == {"sbp"}, read
    assert read["impossible"]["sbp"]["n"] == 4, read
    assert read["within_band"] == ["dbp"], read
    assert read["unrecognized"] == ["bnp"], read
