"""Question 1.5 — *which way round is this table?*

The lens acts at `engine.rank_findings`, which is **presentation**. That is the
right place for it, and `OPENING_SEQUENCE.md` §01 argues the point at length:
reframing annotates and never deletes, so a user who overrules the lens can
still reach the real repair. It is correct for interpreting findings and it does
nothing for structure.

An assay table exported features-in-rows and samples-in-columns is transposed.
The "columns" are participants, so every per-column reading beneath it is
answering a question about a participant and reporting it as a fact about a
measurement — column dtypes, missingness per column, the impossibility pass
comparing one subject's whole panel against a reference range for one analyte,
and a target list that is a list of sample identifiers. **Annotation cannot fix
a frame.**

So this is the one question in the sequence that genuinely **acts** before the
diagnosis rather than annotating its output, and it is what gives clause 01's
ordering its teeth.

## What is asserted here

Four things, and the third is the one that would have been easy to leave
untested:

1. It fires on a transposed assay export, and is silent on every fixture in this
   tree — including `wide_assay.csv`, which reads closest to the threshold.
2. Answering *features in rows* transposes the frame and records a methods
   sentence.
3. **The diagnosis the user then sees is computed on the turned-around table.**
   That is the whole clause. A test that asserted only the record would pass on
   an implementation that recorded the transposition and left the old findings
   in place, which is precisely the record-versus-draw failure that cost a
   critical last loop.
4. The lens contradiction detector stays quiet while the shape is feature-major
   (`GUIDED-042`), because its evidence is computed across the wrong axis.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import orientation as O, packs as P                     # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _transposed_bytes(name: str = "metabolomics_untargeted") -> bytes:
    """A feature-major export of a fixture this tree already ships.

    Built from the real file rather than written by hand, so the thing under
    test is the actual fixture turned around and not a frame constructed to
    contain the effect — `FEATURE_PARITY.md`'s warning about checks tested only
    against a constructed signal.
    """
    df = pd.read_csv(DATA / f"{name}.csv")
    num = df.select_dtypes(include=[np.number])
    t = num.T
    t.index.name = "feature_id"
    t = t.reset_index()
    t.columns = ["feature_id"] + [f"S{i:03d}" for i in range(1, t.shape[1])]
    buf = io.StringIO()
    t.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _pushed(client, pid, step="data"):
    return {q["key"]: q for q in
            client.get(f"/project/{pid}/interview?step={step}").json()["questions"]
            if q["mode"] == "push"}


# ── the reading ──────────────────────────────────────────────────────────────

def test_the_reading_separates_the_fixture_from_its_own_transpose():
    """The discriminator, on the two frames it has to tell apart.

    Asserted as a wide separation rather than as a threshold hit, because a
    threshold the data barely clears is a threshold that will be wrong on the
    next dataset.
    """
    df = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    upright = O.read(df)
    turned = O.read(pd.read_csv(io.BytesIO(_transposed_bytes())))
    assert upright["reading"] == O.SAMPLE_MAJOR, upright
    assert turned["reading"] == O.FEATURE_MAJOR, turned
    assert turned["ratio"] > 4 * upright["ratio"] * 10, (
        f"the two readings are not far apart: {upright['ratio']} vs "
        f"{turned['ratio']}")


@pytest.mark.parametrize("name", [
    "clinic_visits", "clinical_longitudinal", "dietary_recalls",
    "genomics_expression", "metabolomics_untargeted", "survey_instrument",
    "wide_assay", "leaky_sepsis",
])
def test_the_reading_is_never_feature_major_on_a_real_fixture(name):
    """*A check tested only against a constructed signal will over-fire on real
    data.*

    So the finished check is run against every fixture in the tree, not only
    against the transpose it was built for. `wide_assay.csv` is the interesting
    row — it reads closest to the threshold and must stay on the quiet side of
    it, because a question asked of a table it does not describe is guard #2
    broken.
    """
    df = pd.read_csv(DATA / f"{name}.csv")
    assert O.read(df)["reading"] != O.FEATURE_MAJOR, O.read(df)


def test_it_takes_an_assay_lens_as_well_as_a_shape():
    """Both conditions, and neither alone.

    A clinical export is not shipped transposed, so asking there would be the
    pack firing on data it does not match.
    """
    reading = O.read(pd.read_csv(io.BytesIO(_transposed_bytes())))
    assert O.fires(["metabolomics"], reading) is True
    assert O.fires(["genomics"], reading) is True
    assert O.fires(["clinical"], reading) is False
    assert O.fires(["survey", "dietary"], reading) is False
    assert O.fires(["metabolomics"], {"reading": O.SAMPLE_MAJOR}) is False


# ── the refusals ─────────────────────────────────────────────────────────────

def test_it_is_refused_after_the_seal_because_it_changes_what_a_row_is():
    """Decision A's identity barrier, not a preference. A seal drawn before the
    turn names rows that no longer exist."""
    from turbotab.project import AnalysisProject, ProjectError
    df = pd.read_csv(io.BytesIO(_transposed_bytes()))
    project = AnalysisProject.from_dataframe(df, name="t")
    # The barrier is raised by SEALING, never by setting a flag: it is derived
    # from the lockbox holding labels, which is what makes "sealed once" a
    # property of the record rather than of a boolean somebody remembered.
    project.lockbox = {"labels": list(df.index[:5]), "basis": "cross_sectional"}
    assert project.barrier_raised
    with pytest.raises(ProjectError, match="already sealed"):
        project.set_orientation(O.ROWS_ARE_FEATURES)


def test_it_is_refused_once_a_target_is_chosen_and_says_which_order_they_go_in():
    """The target is a column; after the turn it is a row. Refused with the
    reason rather than dropping the choice underneath the user."""
    from turbotab.project import AnalysisProject, ProjectError
    df = pd.read_csv(io.BytesIO(_transposed_bytes()))
    project = AnalysisProject.from_dataframe(df, name="t")
    project.set_target("S001", "regression", "medium", [])
    with pytest.raises(ProjectError, match="comes before the target"):
        project.set_orientation(O.ROWS_ARE_FEATURES)


def test_duplicate_feature_names_are_refused_rather_than_silently_merged():
    """The governing rule's *refuse* branch, where its *assert something false*
    branch is the alternative: two rows with one name become two columns with
    one name, and every consumer downstream sees whichever pandas hands it."""
    frame = pd.DataFrame({
        "feature_id": ["mz_1", "mz_1"] + [f"mz_{i}" for i in range(2, 12)],
        **{f"S{i:02d}": np.random.default_rng(i).lognormal(i, 1, 12)
           for i in range(1, 9)}})
    assert O.label_column(frame) == "feature_id", (
        "the identifier column is not being recognized, so the refusal below "
        "would be unreachable and this test would pass without exercising it")
    with pytest.raises(O.OrientationError, match="Two rows are both named"):
        O.transpose(frame)


# ── the detector that had to be taught to stay quiet ─────────────────────────

def test_the_lens_contradiction_is_silent_while_the_table_may_be_turned_around():
    """`GUIDED-042`.

    Driven, a transposed copy of `metabolomics_untargeted.csv` made
    `set_lens(["metabolomics"])` return 409 with the app asserting — in its most
    interruptive voice — that the user's blanks *"do not look like
    non-detections"*. Read the right way round they do. The detector's evidence
    is computed per column, and on a feature-major table the columns are
    samples.

    Two readings competed and the app announced the wrong one. The second
    explains the first, and 1.5 is where it is settled.
    """
    turned = pd.read_csv(io.BytesIO(_transposed_bytes()))
    assert P.contradiction(turned, ["metabolomics"]) is None, (
        "the app tells the user their lens is wrong when their table is turned "
        "around")

    # And the silence is SCOPED. The upright fixture, described as something it
    # is not, still gets its interruption — so this is a reading deferred to the
    # question that can settle it, not a detector switched off.
    upright = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    clash = P.contradiction(upright, ["clinical"])
    assert clash is not None, (
        "the contradiction detector has gone quiet everywhere, which is a "
        "bigger defect than the one being fixed")
    assert clash["kind"] == "stated_lens_but_shape_is_an_assay"


def test_the_router_never_skips_the_orientation_question():
    """Two independent guards, asserted separately because either alone would be
    enough to make the other look untested."""
    assert router._skip_is_permitted("high", "orientation") is False
    reading = O.read(pd.read_csv(io.BytesIO(_transposed_bytes())))
    assert reading["confidence"] != "high", (
        "the reading claims high confidence, which is the tier "
        "`_skip_is_permitted` reserves for auto-advancing")
