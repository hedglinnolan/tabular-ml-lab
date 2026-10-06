"""`GUIDED-171` — the feature-engineering before/after showed only the FIRST
operand.

Driven on `clinical_labs.csv` before this loop, `GET /project/<id>/feature/
preview?transform=ratio&columns=weight_kg,height_cm` returned, verbatim:

    {"sentence": "The ratio `weight_kg / height_cm` was computed row by row; …",
     "rows": [{"label": 0, "before": 95.8, "after": 0.5682}, …]}

`ratio` declares `n_inputs=2`. **The preview showed one.** 95.8 is `weight_kg`;
the 168.6 of `height_cm` that the division actually used appears nowhere, so a
user looking at the surface where they consent to the transform cannot see what
it consumed. The `after` is arithmetically correct and unexplainable.

And the structured payload was **poorer than the sentence beside it**: the
prose named both columns and the machine-readable form named none — no
`inputs`, no second value, nothing. Trap #7 again, one surface over from
`GUIDED-157`.

## Why nothing caught it

`/features` and `/feature/preview` are outside every field-level gate.

- `test_every_field_the_server_composes_has_a_reader.py::NOT_SWEPT` lists
  *"/features, /recipes, /preprocess — the Features and Preprocess steps, which
  the Explore-step drive does not open"*. The reason is true: that sweep's
  fixture stops at Explore.
- `test_the_three_unswept_payloads_are_swept.py` then drove the journey through
  the seal and **enumerated** all three. But it has no equivalent of L42-B's
  `test_every_unread_family_names_a_reader_or_a_row` — it *prints* its unread
  counts and gates nothing. So an unread field in `/features` is a number in a
  captured stdout block and not a failure.

That is the gap, and it is one surface wide rather than one field wide. It is
**reported, not closed here**: making the late sweep gate its dispositions is a
sweep-wide change with its own disposition table to write, and doing it in the
same loop as the defect it would have caught is the pattern
`AGENT_ONBOARD.md` §08.2 warns about from the other direction.

## Fixtures

`GUIDED-097`: `clinical_labs.csv` on a **classification** target and
`clinic_visits.csv` on a **regression** target. Not covered: a **multiclass**
target, and the `deferred` half of the catalogue — a deferred transform's
preview returns `rows: []` on purpose (clause §06: there is no single set of
values to show before the fold), so the operand table this file is about does
not exist there.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import features as F                                    # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: Two target shapes, and a two-operand formula on each. The arithmetic is
#: named here so the assertion can re-derive the `after` from the operands the
#: preview showed — which is what makes it a claim that those are the values
#: the computation used, rather than that two numbers were rendered.
CASES = {
    "classification": ("clinical_labs", "readmitted", "ratio",
                       ["weight_kg", "height_cm"], lambda a, b: a / b),
    "regression": ("clinic_visits", "hba1c", "product",
                   ["age", "glucose"], lambda a, b: a * b),
}


def test_a_one_column_transform_still_says_exactly_what_it_always_said():
    """The shelf is never shortened. `before` is a shipped field with shipped
    readers, and it stays — as `operands[0]`, computed once, not twice."""
    df = pd.DataFrame({"chol": [120.0, 180.0, 240.0, 90.0, 200.0, 160.0, 210.0]})
    pv = F.preview(df, "log", ["chol"])
    assert pv["inputs"] == ["chol"]
    for row in pv["rows"]:
        assert row["operands"] == [row["before"]]
        assert row["before"] == pytest.approx(float(df.loc[row["label"], "chol"]))


def test_the_operands_are_the_slice_the_computation_actually_reads():
    """Trap #3's rule pointed at a payload: the names the preview publishes
    have to resolve to what ran, not to what the caller asked for.

    `_require_columns` validates `columns[:n_inputs]` and `_compute` reads
    `columns[:n_inputs]`, so a caller passing a third column gets two operands
    — and the preview must report two, or it names a column the arithmetic
    never touched.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "b": [2.0, 2.0, 4.0, 4.0, 5.0, 6.0],
                       "c": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]})
    pv = F.preview(df, "ratio", ["a", "b", "c"])
    assert F.get("ratio").n_inputs == 2
    assert pv["inputs"] == ["a", "b"], (
        f"the preview claims to consume {pv['inputs']}, and `c` reaches no "
        f"arithmetic in this transform")
    for row in pv["rows"]:
        assert len(row["operands"]) == 2


def test_a_deferred_transform_still_shows_no_values_and_says_why():
    """The half this fix deliberately does not touch, asserted so the change
    above cannot leak into it. Clause §06: a preview of a transform fitted
    in-fold has no single set of values to show, and inventing an operand
    table for it would be showing the researcher a picture of their held-out
    data."""
    df = pd.DataFrame({"age": [31.0, 44.0, 52.0, 61.0, 27.0, 38.0, 49.0]})
    pv = F.preview(df, "bin_quantile", ["age"], {"n_bins": 3})
    assert pv["rows"] == []
    assert pv["preview_not_applied"] is True
    assert "inputs" not in pv, (
        "the deferred preview grew an operand list it renders nowhere — a "
        "field with no consumer, which is the trap this loop is avoiding")
