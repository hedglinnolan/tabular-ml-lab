"""`DRIVE-031` — two receipts on one page, and they could not both be true.

On a 21,849-row NHANES file with 15,552 rows missing the outcome, a human read:

> **STUDY POPULATION** — No eligibility restriction: all 21849 rows are in the
> study population, **and the held-out set is drawn from all of them.**
>
> **THE HELD-OUT SET · SEALED** — **945 rows (15%)** are held out…

945/6,297 = 15.0%. 945/21,849 = 4.3%. The seal's percentage was right and its
base was never stated; the eligibility receipt stated a base the draw does not
use.

## Which moment owns it, established rather than assumed

The prompt asked whether this is one sentence owed by two rows. It is **one
false clause and two sentences that had to change**, and the record says so:

* `eligibility.n_before == n_after == 21849`. The eligibility module is
  **correct about its own operation** — it excluded nobody, and its first clause
  says exactly that.
* `lockbox.n_total == 6297`. The drop happens inside `engine.draw_holdout`,
  whose first act is `eligible = df.index[y.notna()]`.

So the drop is the **lockbox's**, and the false clause was eligibility
describing a draw it does not perform. Each module now claims only what it did.
Removing the clause alone would have traded a false number for no number, which
is why both moved together rather than one silently.

## The app already knew

The read-as-draft methods paragraph has printed *"Of 6,297 rows with a value for
the outcome, 945 were sealed … and 5,352 were available for fitting"* the whole
time. The engine was right and one composed string was wrong — trap 6 with the
error in the sentence rather than in the render.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "sample_data"


def _decide(client, pid, kind, payload):
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": kind, "payload": payload})
    assert resp.status_code == 200, (kind, resp.status_code, resp.text[:300])
    return resp.json()


#: The tester's shape, not their file — it is not on disk. 21,849 rows, a binary
#: outcome missing on 15,552 of them, which is the 71.2% that made the two
#: sentences visibly disagree.
N_ROWS, N_MISSING = 21_849, 15_552


def _nhanes_shaped() -> bytes:
    rng = np.random.default_rng(7)
    target = np.concatenate([
        np.full(N_MISSING, np.nan),
        rng.choice([0.0, 1.0], size=N_ROWS - N_MISSING, p=[0.122, 0.878])])
    rng.shuffle(target)
    frame = pd.DataFrame({
        "SEQN": np.arange(1, N_ROWS + 1),
        "meds_hbp": pd.array(target, dtype="Float64").astype("Int64"),
        "age": rng.integers(18, 85, N_ROWS),
        "kcal": rng.normal(2100, 600, N_ROWS).round(1),
        "bmi": rng.normal(28, 6, N_ROWS).round(1),
    })
    return frame.to_csv(index=False).encode()


def _sealed_project(client, payload: bytes, target: str):
    pid = client.post("/project", files={
        "file": ("t.csv", io.BytesIO(payload), "text/csv")}).json()["id"]
    _decide(client, pid, "set_target", {"column": target})
    _decide(client, pid, "set_grain", {"answer": "one_row_per_person"})
    after = _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    sealed = _decide(client, pid, "seal", {})
    return pid, after, sealed


@pytest.mark.parametrize("basis,extra", [
    ("cross_sectional", {}),
    ("grouped", {"n_test_groups": 30, "group_noun": "subjects"}),
    ("undetermined", {}),
    ("repetition_found_grouping_abandoned",
     {"n_test_groups": 0, "group_noun": "subjects"}),
])
def test_every_basis_can_name_its_base(basis, extra, capsys):
    """All four, because a sentence that could not format would fall through to
    the `undetermined` guard and silently report the wrong basis — which is
    constitution §03's own failure arriving through a `KeyError`."""
    from turbotab import grain

    lockbox = dict({"seal_basis": basis, "n_test": 945, "fraction": 0.15,
                    "n_total": 6297, "n_rows_before_outcome_drop": 21849},
                   **extra)
    said = grain.seal_disclosure(lockbox)
    assert "6,297" in said and "15,552" in said, (basis, said)
    # The basis is still the one asked for, not the fallback.
    if basis == "grouped":
        assert "chosen by subject" in said
    if basis == "undetermined":
        assert "shape is unknown" in said
    with capsys.disabled():
        print(f"\n  {basis:<38} {said[:72]}…")
