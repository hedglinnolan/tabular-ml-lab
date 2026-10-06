"""`GUIDED-157` — the bulk binary repair recorded WHICH COLUMNS and never WHICH
VALUE BECAME 1.

The record the app kept for a bulk `read_as_binary`, driven on
`clinical_labs.csv` before this loop, verbatim:

    1 feature (`sex`) was read as binary.
    payload: {"fix_kind": "read_as_binary", "label": "read as binary",
              "findings": ["binary_text__sex"], "columns": ["sex"],
              "declined": [...], "n_selected": 1, "n_offered": 2}

`sex` holds `M` and `F`. Nothing in that sentence and nothing in that payload
says which of them is now the 1, so *"is the coefficient on `M` or on `F`"* has
**no answer anywhere in the record** — and the direction of every effect
estimate for that variable follows from it. On the product owner's NHANES
export the column is `gender ∈ {female, male}` and the question is the same one.

**No number in this defect is wrong.** A reported number is made
uninterpretable, which is trap #7's shape — the machine-readable form lossier
than the sentence — with the additional turn that the sentence did not carry it
either, so there was nothing for the payload to be lossier *than*.

## The shape this copies

`GUIDED-165`, at L47: one ambiguous kind became two declared kinds, each with a
machine-readable payload beside the sentence a person reads, because *"I
repaired this"* and *"I left it alone"* had to be distinguishable without
string-matching prose. Here the record already had its own kind; what it lacked
was the payload. So `engine.fix_encoding` reports what the transform does,
`api.apply_bulk` records it, and `repairs.sentence` states it.

## What is asserted, and against what

**The record is checked against the FRAME, never against itself.** A payload
saying ``{"M": 1, "F": 0}`` beside a rewrite that did the opposite would satisfy
any test that reads the receipt — and would be a worse defect than the one being
fixed, because it would assert something false rather than say nothing. So every
mapping claim here is re-derived from the rows: the positions that held `M`
before are read out of the column after, and they are required to hold exactly
the value the record says they hold.

**Two fixtures of different target shape** (`GUIDED-097`): `clinical_labs.csv`
on a **classification** target and `metabolomics_untargeted.csv` on a
**regression** target. The shape not covered is a **multiclass** target — no
fixture in `turbotab/sample_data/` pairs one with a `read_as_binary` group of
two or more members (`multiclass_stage.csv` has exactly one binary-text column,
`sex`, and `repairs.MIN_GROUP` is 2).
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import engine, repairs as R                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: The two target shapes, and the fixture that carries a bulk binary group under
#: each. `GUIDED-097`'s rule: one fixture is one `float()` that happened to
#: succeed.
FIXTURES = {
    "classification": ("clinical_labs", "readmitted"),
    "regression": ("metabolomics_untargeted", "bmi"),
}


# ── the record says which value became 1, and the frame agrees ───────────────


def test_a_repair_with_no_mapping_records_none_rather_than_an_invented_one():
    """Trap 9 at the record layer: return nothing rather than a wrong value.

    `coerce_numeric` rewrites a column and has no 1 and no 0 in it. An encoding
    manufactured for it would be the record asserting a direction the transform
    never chose, which is the defect this loop is closing pointed backwards.
    """
    from ml.import_doctor import ShapeFinding

    df = pd.DataFrame({"weight": ["72 kg", "81 kg", "66 kg", "70 kg", "75 kg"],
                       "sex": ["M", "F", "M", "F", "M"]})
    numeric = ShapeFinding(
        id="numeric_as_text__weight", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="coerce_numeric", confidence="high",
        params={"column": "weight"}, affected_columns=["weight"])
    assert engine.fix_encoding(df, numeric) is None

    # The positive control, so the None above is a refusal and not a broken
    # helper: the same call on the kind that DOES have a mapping returns one.
    binary = ShapeFinding(
        id="binary_text__sex", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="read_as_binary", confidence="medium",
        params={"column": "sex"}, affected_columns=["sex"])
    assert (engine.fix_encoding(df, binary) or {}).get("mapping") == {"M": 1,
                                                                     "F": 0}

    # And a column that is no longer binary gets nothing rather than a stale
    # mapping read off the finding's own params.
    already = pd.DataFrame({"sex": [1, 0, 1, 0, 1]})
    assert engine.fix_encoding(already, binary) is None


def test_the_mapping_names_every_spelling_that_maps_to_a_side():
    """`Male` and `male` are one level and two strings.

    `ml.binary_text` compares on a normalized token, so a column written both
    ways has one level with two spellings — and a record naming the first one
    it happened to see would be a claim that is right about half the rows.
    Written against a frame rather than a fixture file because no shipped
    fixture has this shape and the ones that come close have four levels, not
    two: `clinic_visits.csv`'s `sex` holds `Male`, `male`, `M`, `Female`,
    `female`, `F`, which is why the engine does not read it as binary at all.
    """
    from ml.import_doctor import ShapeFinding

    df = pd.DataFrame({"gender": ["Male", "male", "Female", "female", "MALE",
                                  "Female", "male"]})
    finding = ShapeFinding(
        id="binary_text__gender", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="read_as_binary", confidence="medium",
        params={"column": "gender"}, affected_columns=["gender"])
    enc = engine.fix_encoding(df, finding)
    assert enc is not None
    assert set(enc["positive_values"]) == {"Male", "male", "MALE"}
    assert set(enc["negative_values"]) == {"Female", "female"}
    assert enc["mapping"] == {"Male": 1, "male": 1, "MALE": 1,
                              "Female": 0, "female": 0}
    assert enc["n_positive"] == 4 and enc["n_negative"] == 3

    # And the sentence carries all of them, for the same reason.
    said = R.sentence("read as binary", ["gender"], (), {"gender": enc})
    for spelling in ("Male", "male", "MALE", "Female", "female"):
        assert f"`{spelling}`" in said, said


def test_a_kind_with_no_mapping_keeps_the_sentence_it_always_had():
    """The other repairs are not given punctuation for a distinction they do
    not have. `read_as_binary` is the only kind with a per-column direction."""
    assert (R.sentence("read as numbers", ["a", "b"])
            == "2 features (`a`, `b`) were read as numbers.")
    assert (R.sentence("read as numbers", ["a", "b"], (), {})
            == "2 features (`a`, `b`) were read as numbers.")
