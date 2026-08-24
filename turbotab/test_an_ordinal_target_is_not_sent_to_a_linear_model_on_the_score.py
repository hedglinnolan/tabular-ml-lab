"""`AUDIT-033` — what the task-type detector says about an ordered outcome.

`ml/triage.py`'s low-cardinality-integer branch is exactly §B6's ordinal
signature: a 0-5 mRS, a 1-5 global rating, a 0-10 NRS, a single Likert item. It
used to answer:

    Target has 6 unique integer values (≤10) — this often means classification,
    but counts or ordinal scores should be treated as regression. Verify or
    override below.

`CLINICAL_SURVEY_PACK.md` §B6 Coaching, **[SETTLED]**:

> For an ordinal outcome, use a cumulative link (proportional odds) model rather
> than a **linear model on the score** or a dichotomization into
> 'responder/non-responder.' The PO model generalizes the Wilcoxon and
> Kruskal-Wallis tests while allowing covariate adjustment, handles arbitrarily
> many ties, and uses the full ordering.

`regression` in this app **is** a linear model on the score —
`ml/model_registry.py`'s regression family is OLS/ridge/boosting on the raw
number — so the sentence recommended, in both doors, the one treatment the
registry marks SETTLED against, and named no alternative.

## What the corrected sentence has to do, and why each clause is checked

The app **cannot fit a cumulative link model**, so it may not simply recommend
one either: `ml/model_registry.py` has no such family, and pointing a user at a
menu item that does not exist is the same class of false assertion one step
over. The honest form is the one the governing rule names — say the true thing,
including the true thing about the software:

1. the numbers cannot tell a class code from a count from an ordinal score;
2. if it is ordinal, **neither offer is right**;
3. what is right is a cumulative link (proportional odds) model, **which this
   app does not fit**, and where one lives;
4. what each offered family costs — regression treats the gaps as equal and can
   invert the ordering of group means (§B4's inversion result, the strongest
   argument in that literature), classification discards the ordering.

Clause 3 is the one this file guards hardest. Deleting the bad recommendation
without naming the right method would satisfy "no longer asserts something
false" and would shorten the shelf, which `PRODUCT_VISION.md` forbids.

## Both doors, because the row is about both

* **Guided** — `POST /decision {set_target}` → `turbotab/api.py:493`
  `engine.detect_task_type` → `project.set_target` stores `task_reasons` →
  `GET /project/{id}/interview` composes `detection` and `ml/router.py:608`
  joins the reasons into the `confirm_task_type` card's `why`, which
  `turbotab/web/index.html` renders. Driven over HTTP here.
* **Classic** — `pages/01_Upload_and_Audit.py:1025` stores `task_result['reasons']`
  and `:1040-1042` renders the last one beside the detected task. That page is
  **frozen** (`TRANSITION_PLAN.md` §05), so the fix is in `ml/triage.py` where
  both doors read it; what is asserted here is that the frozen page still reads
  that list.

## Fixture shapes

`GUIDED-097`. Four target shapes, two of them driven over HTTP:

| shape | fixture | what is required |
|---|---|---|
| low-cardinality **int** (a 1-5 Likert item) | `survey_instrument.csv::item_01` | the full disclosure |
| low-cardinality **int** (a visit index) | `clinical_longitudinal.csv::visit` | the full disclosure |
| **string** labels | `multiclass_stage.csv::disease_stage` | no regression recommendation |
| **continuous float** | constructed | regression IS still recommended |

The last row is the positive control, and it is the reason this file cannot be
satisfied by deleting the word *regression* everywhere.

**Not covered: a low-cardinality FLOAT target** — the same Likert item with one
missing value is a `float64` column and takes `ml/triage.py`'s float branch,
which says only *"Target has 5 unique float values (≤10) - classification"* and
raises no ordinal possibility at all. That is silence rather than a false
assertion, so it is out of this row; it is filed separately.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.triage import detect_task_type                               # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: The sentence §B6 marks SETTLED against, in the words the app used to use.
FORBIDDEN = "should be treated as regression"

#: The two ordered-integer fixtures, driven through the Guided door below.
ORDINAL_FIXTURES = [("survey_instrument.csv", "item_01"),
                    ("clinical_longitudinal.csv", "visit")]


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _reasons_over_http(client, fixture: str, target: str):
    """The reason text as the Guided door receives it, not as `ml/` composes it.

    Returns `(stored, rendered)` **separately**: the list `project.task_reasons`
    keeps, and the `why` the `confirm_task_type` card carries. Concatenating
    them would let either half carry the assertion alone, which is trap 6 — the
    server composing a string the interface never renders.
    """
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    body = client.post(f"/project/{pid}/decision",
                       json={"kind": "set_target",
                             "payload": {"column": target}}).json()
    assert body["target"] == target, body
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    card = next((q for q in plan["questions"]
                 if q["key"] == "confirm_task_type"), None)
    assert card is not None, [q["key"] for q in plan["questions"]]
    return " ".join(body["task_reasons"]), (card.get("why") or "")


# ── the claim, in the module both doors call ────────────────────────────────

@pytest.mark.parametrize("fixture,target", ORDINAL_FIXTURES)
def test_the_detector_does_not_send_an_ordered_score_to_a_linear_model(fixture, target):
    frame = pd.read_csv(DATA / fixture)
    reasons = " ".join(detect_task_type(frame, target)["reasons"])
    assert FORBIDDEN not in reasons, reasons


@pytest.mark.parametrize("fixture,target", ORDINAL_FIXTURES)
def test_it_names_the_model_the_registry_marks_settled_for(fixture, target):
    """Clause 3. Removing the wrong recommendation is half the fix; §B6 names a
    method and the shelf is never shortened."""
    frame = pd.read_csv(DATA / fixture)
    reasons = " ".join(detect_task_type(frame, target)["reasons"])
    assert "cumulative link" in reasons, reasons
    assert "proportional odds" in reasons, reasons


@pytest.mark.parametrize("fixture,target", ORDINAL_FIXTURES)
def test_it_says_this_app_does_not_fit_that_model(fixture, target):
    """And clause 3's other half. `ml/model_registry.py` has no cumulative-link
    family, so recommending one without saying so points the user at a menu item
    that is not there."""
    from ml import model_registry

    frame = pd.read_csv(DATA / fixture)
    reasons = " ".join(detect_task_type(frame, target)["reasons"])
    assert "does not fit" in reasons, reasons
    registry_text = " ".join(str(k) for k in model_registry.get_registry())
    assert "ordinal" not in registry_text.lower(), (
        "a cumulative-link model now exists in the registry; the reason string "
        "still tells the user this app does not fit one")


@pytest.mark.parametrize("fixture,target", ORDINAL_FIXTURES)
def test_it_states_what_each_offered_family_costs(fixture, target):
    """§B4's inversion result is the strongest argument in that literature and
    the reason a user should care: a metric model on an ordered outcome can
    report the ordering of group means backwards."""
    frame = pd.read_csv(DATA / fixture)
    reasons = " ".join(detect_task_type(frame, target)["reasons"])
    assert "backwards" in reasons, reasons
    assert "discards the ordering" in reasons, reasons


# ── both doors ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture,target", ORDINAL_FIXTURES)
def test_the_guided_door_renders_the_corrected_reason(fixture, target):
    """Driven, not grepped. The string has to survive `project.task_reasons`,
    the interview composer and `ml/router.py:608`'s join."""
    stored, rendered = _reasons_over_http(_client(), fixture, target)
    for where, text in (("task_reasons", stored), ("card why", rendered)):
        assert FORBIDDEN not in text, f"{where}: {text}"
        assert "cumulative link" in text, f"{where}: {text}"
        assert "does not fit" in text, f"{where}: {text}"


def test_the_frozen_classic_page_still_reads_the_reason_list():
    """`pages/01_Upload_and_Audit.py` is frozen, so the fix had to land in
    `ml/triage.py`. This asserts the page still renders what that module
    returns — otherwise the Classic half of the row is closed against a path
    that no longer exists."""
    page = (ROOT / "pages" / "01_Upload_and_Audit.py").read_text(encoding="utf-8")
    assert "reasons=task_result['reasons']" in page, (
        "the Classic page no longer stores the detection reasons")
    assert "_det_reasons[-1]" in page, (
        "the Classic page no longer renders a detection reason")


# ── the other target shapes ─────────────────────────────────────────────────

def test_a_string_labeled_outcome_is_not_sent_to_regression_either():
    frame = pd.read_csv(DATA / "multiclass_stage.csv")
    reasons = " ".join(detect_task_type(frame, "disease_stage")["reasons"])
    assert FORBIDDEN not in reasons, reasons


def test_a_genuinely_continuous_target_is_still_routed_to_regression():
    """The positive control, and the reason this file cannot be satisfied by
    deleting the word. A continuous outcome is a regression problem and the app
    still says so."""
    rng = np.random.default_rng(33)
    frame = pd.DataFrame({"x": rng.normal(size=400),
                          "y": rng.normal(size=400)})
    detection = detect_task_type(frame, "y")
    assert detection["detected"] == "regression"
    assert detection["confidence"] == "high"
    assert "regression" in " ".join(detection["reasons"])
