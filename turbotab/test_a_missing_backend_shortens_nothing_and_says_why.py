"""`MODELS-026` — a missing estimator marks a row, never empties an endpoint.

**The failure this closes.** `ml/model_registry.py` imported scikit-learn,
xgboost and lightgbm at module scope, so an interpreter missing any one of them
lost `GET /models` — and with it Train, Explain, the figures and the report —
to an unhandled `ModuleNotFoundError` that reached Starlette as twenty-one
characters of *Internal Server Error*, on every file and every target. That is
why no fixture could reproduce it and three human drives went looking for a
data-dependent cause.

**`TEST-038` is the standard and it is applied rather than invented.**
`utils/seed.py` wraps `import torch` in `try/except ImportError` with a comment
saying it is optional; torch is deliberately absent, that absence is a named
expected condition, and it takes no endpoint down. `TEST-038` also stated the
contradiction this closes: *"one of the two is right about whether torch is
optional, and they cannot both be."* `requirements.txt` declares xgboost and
lightgbm while the environment treats them as optional, and the code treated
them as mandatory.

**And `PRODUCT_VISION.md` decides the direction.** *"The shelf is never
shortened"* is about not hiding a model from a user — hiding one because the
machine is short a package is the same act with a better excuse. So the entry
stays, in its bucket, with its concern, and carries the reason it cannot be
fitted **here**. The two are kept apart deliberately: `concern` is the engine's
judgment about this DATA and reads the same on every machine; the reason is a
fact about the INSTALL.

**The disclosure is the part that would otherwise assert something false.**
`SHELF_DISCLOSURE` opens *"Every model is available"*, which is the one
paragraph whose entire job is to tell a user nothing has been hidden — and in
an install missing xgboost it is a lie. Guarding the import and leaving that
sentence would have produced a complete list under a false claim about itself.

**Driven by removing the backend**, not by reading the guard: `backend_error`
is what the shelf reads and what the factory raises, so it is what these tests
take away.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import model_registry as MR                                # noqa: E402
from turbotab import models as M                                   # noqa: E402

BOOSTING = ("xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf")


@pytest.fixture
def xgboost_absent(monkeypatch):
    """The state of an install with no xgboost, without uninstalling it.

    `_XGBOOST_ERROR` is what the module records at its own guarded import and
    is the single source both `backend_error` and the factories read, so
    setting it reproduces the whole downstream condition rather than a piece
    of it.
    """
    monkeypatch.setattr(MR, "_XGBOOST_ERROR",
                        "ModuleNotFoundError: No module named 'xgboost'")
    monkeypatch.setattr(MR, "XGBRegressor", None)
    monkeypatch.setattr(MR, "XGBClassifier", None)


# ── the registry ────────────────────────────────────────────────────────────

def test_the_registry_still_builds_without_a_boosting_backend(xgboost_absent):
    """**The whole point.** Every model is still specified and the endpoint has
    something to serve; only the fit is unavailable."""
    registry = MR.get_registry()
    for key in BOOSTING:
        assert key in registry, (
            f"{key} left the registry when its backend went missing — the "
            f"shelf was shortened rather than marked")
    assert len(registry) >= 20


def test_the_factory_refuses_with_a_sentence_rather_than_a_name_error(
        xgboost_absent):
    """A factory called for an absent backend raises `ModelUnavailable` with
    the reason, not `TypeError: 'NoneType' object is not callable`."""
    spec = MR.get_registry()["xgb_clf"]
    with pytest.raises(MR.ModelUnavailable) as raised:
        spec.factory("classification", 42)
    said = str(raised.value)
    assert "XGBoost is not available in this install" in said, said
    assert "pip install xgboost" in said, said


def test_only_the_absent_backend_is_marked(xgboost_absent):
    """The negative half. LightGBM is present, so it says nothing — a reason
    on a model that fits fine would be noise that teaches a reader to skip
    the field."""
    assert MR.backend_error("xgb_reg")
    assert MR.backend_error("lgbm_reg") is None
    assert MR.backend_error("ridge") is None


# ── the shelf ───────────────────────────────────────────────────────────────

def _shelf(profile, task="classification"):
    return M.shelf(profile, task)


@pytest.fixture
def profile():
    from ml.dataset_profile import compute_dataset_profile
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x1": rng.normal(0, 1, 200),
                          "x2": rng.normal(0, 1, 200),
                          "y": rng.integers(0, 2, 200)})
    return compute_dataset_profile(frame, target_col="y")


def test_the_shelf_keeps_the_model_and_states_the_reason(xgboost_absent, profile):
    """Ranked, not removed. The row is on the shelf and says why it cannot be
    fitted here — `PRODUCT_VISION.md`'s three-rung ladder, and this is not the
    top rung: there is nothing wrong with choosing XGBoost, the machine simply
    cannot run it."""
    entries = {e.key: e for e in _shelf(profile)}
    assert "xgb_clf" in entries, "the shelf dropped a model it could not fit"
    said = entries["xgb_clf"].unavailable_because
    assert "not available in this install" in said, said
    assert entries["lgbm_clf"].unavailable_because == "", (
        "a model whose backend is present carries an unavailability reason")


def test_the_reason_is_not_folded_into_the_concern(xgboost_absent, profile):
    """Two different claims, kept apart. A user reading *"XGBoost is not
    available here"* inside the concern would take it for a judgment about
    their table."""
    entry = {e.key: e for e in _shelf(profile)}["xgb_clf"]
    assert "install" not in entry.concern.lower(), entry.concern
    assert entry.unavailable_because != entry.concern


def test_the_bucket_does_not_move(xgboost_absent, profile):
    """Availability is not a verdict. Demoting an unavailable model would make
    the coach's ranking about the machine instead of about the data — and the
    ranking is the thing the whole module exists to protect."""
    with_backend = {e.key: e.bucket for e in _shelf(profile)}
    # `xgboost_absent` is already applied, so re-read with the flag cleared.
    MR._XGBOOST_ERROR = None
    try:
        without = {e.key: e.bucket for e in _shelf(profile)}
    finally:
        MR._XGBOOST_ERROR = "ModuleNotFoundError: No module named 'xgboost'"
    assert with_backend == without, (
        "a model changed bucket because its backend went missing")


# ── the disclosure, which is the sentence that would be false ───────────────

def test_the_disclosure_stops_claiming_every_model_is_available(
        xgboost_absent, profile):
    """The governing rule, in the paragraph most exposed to it."""
    entries = _shelf(profile)
    said = M.disclosure(entries)
    assert "Every model is available" not in said, (
        "the shelf still claims every model is available while two of them "
        "cannot be fitted here — the governing rule's assert-something-false "
        "branch, in the sentence whose job is to say nothing was hidden")
    assert "cannot be fitted in this install" in said, said
    assert "Every model is listed" in said, (
        "the disclosure stopped saying the list is complete, which is the "
        "claim that is still true and the one that matters most")


def test_the_disclosure_is_unchanged_on_a_complete_install(profile):
    """**The control, and it is the one that keeps this cheap.** On every
    machine this suite runs on, nothing is missing and the sentence must be
    byte-identical to the constant it always was."""
    entries = _shelf(profile)
    assert all(e.unavailable_because == "" for e in entries)
    assert M.disclosure(entries) == M.SHELF_DISCLOSURE


def test_the_wire_carries_the_reason_so_the_page_can_render_it(
        xgboost_absent, profile):
    """`to_dict` is what the page reads. A reason that reached the shelf and
    not the payload would be trap #6 — the server composing a string the
    interface never renders — which is the shape this whole row is about."""
    payload = {e.key: e.to_dict() for e in _shelf(profile)}
    assert "unavailable_because" in payload["xgb_clf"]
    assert payload["xgb_clf"]["unavailable_because"]
    assert payload["lgbm_clf"]["unavailable_because"] == ""
