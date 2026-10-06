"""The grain question — asked, never inferred. Constitution §02.

`IMPORT-020` and `IMPORT-022` are both the engine having guessed whether one
person can appear in more than one row, and a failed guess rendering as a clean
lock over a real leak. These tests pin the three things that stop that:

* the seal cannot be drawn before the question is answered (§01's ordering, as
  a precondition rather than a comment);
* the contradiction detector is **name-blind**, so it fires on the identifier
  spelling the name lists miss — which is `IMPORT-022`'s whole fixture;
* "I'm not sure" seals, and says `undetermined` rather than pretending.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import eligibility as E, repeats as R, engine, grain as G       # noqa: E402
from turbotab.project import (                                    # noqa: E402
    AnalysisProject, GrainContradiction, ProjectError,
)
from utils.test_lockbox import (                                  # noqa: E402
    SEAL_CROSS_SECTIONAL, SEAL_GROUPED, SEAL_UNDETERMINED,
    BASIS_USER_STATED, detect_repeated_subjects,
)


def longitudinal(key: str = "SUBJ", n_sub: int = 60, visits: int = 3) -> pd.DataFrame:
    """The `IMPORT-022` fixture: repeated measures under an unrecognized name."""
    rows = []
    for s in range(n_sub):
        for v in range(visits):
            rows.append((f"S{s:03d}", 40 + s % 30, 90.0 + v, (s + v) % 2))
    df = pd.DataFrame(rows, columns=[key, "age", "glucose", "outcome"])
    return df


def cross_sectional(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })


def _project(df: pd.DataFrame) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(df, "t")
    p.set_target("outcome", "classification", "high", [])
    return p


# ── the ordering: §01, as a precondition ─────────────────────────────────────

def test_the_seal_cannot_be_drawn_before_the_grain_is_answered():
    """Clause §01 fixes the pre-seal order and §02 is why. Refusing here makes
    that order executable, so a caller cannot seal first and ask afterwards.

    Clause: `lockbox-01`
    """
    p = _project(cross_sectional())
    with pytest.raises(ProjectError, match="grain question"):
        p.seal_lockbox(list(p.df.index[:10]))


def test_the_grain_cannot_be_restated_after_the_seal_is_drawn():
    """The split was drawn against the answer recorded at the time. Changing it
    afterwards would describe a split that was not drawn that way."""
    p = _project(cross_sectional())
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    with pytest.raises(ProjectError, match="already sealed"):
        p.set_grain(G.PEOPLE_REPEAT, "record_id")
        # Clause 01's bracketed steps: people repeat, so questions 4 and 5 come
        # before the seal. The unit is the RECORD, so the rows survive and every
        # count asserted below is unchanged.
        p.set_repeat_kind(R.REPEATS)
        p.set_unit_of_analysis(R.UNIT_RECORD)


# ── the contradiction detector is name-blind ─────────────────────────────────

def test_the_contradiction_detector_fires_on_an_id_name_the_heuristic_misses():
    """`IMPORT-022`, inverted into a guard.

    The name-token gate rejects `SUBJ` before its repetition is ever measured —
    asserted here, so this test fails loudly if someone "fixes" it by extending
    the token list, which constitution §02 forbids in as many words. The
    interruption must fire anyway, on shape alone.

    Clause: `lockbox-02`
    """
    df = longitudinal(key="SUBJ")
    assert detect_repeated_subjects(df) is None, (
        "the name heuristic now recognizes SUBJ — good, but this test exists to "
        "prove the interruption does NOT depend on that; rewrite it against a "
        "spelling the list still misses rather than deleting it")

    clash = G.contradiction(df, G.ONE_ROW_PER_PERSON)
    assert clash is not None, "no interruption on a 60-subject x 3-visit file"
    assert "SUBJ" in clash["columns"]
    assert "60" in clash["message"] and "180" in clash["message"], (
        f"the interruption must show what it saw: {clash['message']}")


@pytest.mark.parametrize("key", ["SUBJ", "ptno", "zzz_9", "Anonymised Code"])
def test_the_interruption_does_not_depend_on_what_the_column_is_called(key):
    """Four spellings, none of which a token list would have anticipated."""
    assert G.contradiction(longitudinal(key=key), G.ONE_ROW_PER_PERSON) is not None


def test_stating_one_row_per_person_on_a_repeating_file_is_refused_not_warned():
    """Escalate on evidence of error. One of the two readings is wrong, and the
    caller has to say which — so it raises rather than logging.

    Clause: `lockbox-02`
    """
    p = _project(longitudinal())
    with pytest.raises(GrainContradiction) as exc:
        p.set_grain(G.ONE_ROW_PER_PERSON)
    assert exc.value.detail["kind"] == "stated_unique_but_data_repeats"
    assert p.grain is None, "a refused answer must not be recorded"


def test_the_interruption_can_be_answered_and_the_answer_is_kept():
    """A blocker that cannot be satisfied teaches contempt for all blockers
    (DESIGN_LANGUAGE §09). The user may say the data is right and proceed —
    and the record keeps that they were interrupted."""
    p = _project(longitudinal())
    p.set_grain(G.ONE_ROW_PER_PERSON, acknowledged_contradiction=True)
    assert p.grain["answer"] == G.ONE_ROW_PER_PERSON
    assert p.grain["contradiction_acknowledged"] is True
    assert p.grain["contradiction"]["columns"], "the evidence stays in the record"


def test_the_detector_also_fires_the_other_way():
    """"People repeat, identified by X" where X is unique per row is the same
    kind of disagreement, and grouping by it would produce the row-level split
    the user was trying to avoid."""
    p = _project(cross_sectional())
    with pytest.raises(GrainContradiction, match="different value on every"):
        p.set_grain(G.PEOPLE_REPEAT, "record_id")
        # Clause 01's bracketed steps: people repeat, so questions 4 and 5 come
        # before the seal. The unit is the RECORD, so the rows survive and every
        # count asserted below is unchanged.
        p.set_repeat_kind(R.REPEATS)
        p.set_unit_of_analysis(R.UNIT_RECORD)


def test_a_genuinely_cross_sectional_file_earns_no_interruption():
    """The detector must be quiet when the answer and the data agree, or it is
    noise and gets ignored when it matters."""
    p = _project(cross_sectional())
    p.set_grain(G.ONE_ROW_PER_PERSON)
    assert p.grain["basis"] == SEAL_CROSS_SECTIONAL
    assert p.grain["contradiction"] is None


def test_a_stratum_is_not_mistaken_for_a_roster():
    """`sex` repeats on every row of any cohort and identifies nobody. If it
    counted as evidence, every cross-sectional file would be interrupted."""
    df = cross_sectional(200)
    df["sex"] = (["M", "F"] * 100)
    df["site"] = (["A", "B", "C", "D"] * 50)
    assert G.contradiction(df, G.ONE_ROW_PER_PERSON) is None, (
        "a two-level and a four-level column were read as a roster of people")


# ── the seal states its own basis ────────────────────────────────────────────

def test_a_stated_repeat_seals_grouped_and_leaks_nobody():
    p = _project(longitudinal())
    p.set_grain(G.PEOPLE_REPEAT, "SUBJ")
    # Clause 01's bracketed steps: people repeat, so questions 4 and 5 come
    # before the seal. The unit is the RECORD, so the rows survive and every
    # count asserted below is unchanged.
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    lb = p.lockbox
    assert lb["seal_basis"] == SEAL_GROUPED
    assert lb["basis_source"] == BASIS_USER_STATED
    assert lb["group_col"] == "SUBJ"

    held = set(lb["labels"])
    train = [i for i in p.df.index if i not in held]
    both = set(p.df.loc[list(held), "SUBJ"]) & set(p.df.loc[train, "SUBJ"])
    assert not both, f"{len(both)} subject(s) on both sides of the seal"


def test_not_sure_seals_anyway_and_says_undetermined():
    """Constitution §03: an advisory with exploratory labeling, not a hard
    block. A user who does not know their data's shape gets honest numbers."""
    p = _project(longitudinal())
    p.set_grain(G.NOT_SURE)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    lb = p.lockbox
    assert lb["seal_basis"] == SEAL_UNDETERMINED
    assert lb["exploratory"] is True, "an undetermined seal is never a clean lock"
    assert lb["n_test"] > 0, "it seals — it does not refuse"


def test_undetermined_is_never_recorded_as_a_missing_group_column():
    """`group_col: None` is what a verified cross-sectional seal looks like too,
    so a consumer cannot tell them apart. The basis carries the difference."""
    a = _project(longitudinal()); a.set_grain(G.NOT_SURE)
    b = _project(cross_sectional()); b.set_grain(G.ONE_ROW_PER_PERSON)
    assert a.grain["group_col"] is None and b.grain["group_col"] is None
    assert a.grain["basis"] != b.grain["basis"], (
        "two different claims rendered identically")


def test_the_seal_reports_the_row_share_it_actually_held_out():
    """A grouped draw's fraction is a proportion of GROUPS. With unequal group
    sizes those differ badly — `IMPORT-255` measured 15% requested against 37%
    of rows held out."""
    rows = []
    for s in range(20):
        for v in range(40 if s == 0 else 5):
            rows.append((f"P{s:02d}", 40 + s, float(v), (s + v) % 2))
    df = pd.DataFrame(rows, columns=["SUBJ", "age", "day", "outcome"])
    p = _project(df)
    p.set_grain(G.PEOPLE_REPEAT, "SUBJ")
    # Clause 01's bracketed steps: people repeat, so questions 4 and 5 come
    # before the seal. The unit is the RECORD, so the rows survive and every
    # count asserted below is unchanged.
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    lb = p.lockbox
    assert abs(lb["fraction"] - lb["n_test"] / lb["n_total"]) < 0.01, (
        f"reports {lb['fraction']:.0%}, held out "
        f"{lb['n_test'] / lb['n_total']:.0%}")
    assert lb["fraction_requested"] == 0.15, "the request survives the achievement"


def test_the_basis_source_is_reachable_for_assembly():
    """`inherited_from_assembly` has no producer yet — assembly is behind an
    unmet freeze gate — so it is pinned as reachable rather than dead, which is
    what stops it being deleted as unused before it can be used.

    Clause: `assembly-05`
    """
    p = _project(cross_sectional())
    p.set_grain(G.ONE_ROW_PER_PERSON, inherited=True)
    assert p.grain["basis_source"] == "inherited_from_assembly"


# ── drivable over HTTP: upload a CSV, reach the end of the step ───────────────


def test_the_grain_and_the_basis_survive_the_save_file():
    """`archive.py`'s lockbox member is an explicit whitelist, so a field added
    to the seal and not added there is dropped on save — the seal would come
    back unable to say what it rests on, which is the `group_col: None`
    ambiguity constitution §03 exists to remove."""
    from turbotab import archive
    p = _project(longitudinal())
    p.set_grain(G.PEOPLE_REPEAT, "SUBJ")
    # Clause 01's bracketed steps: people repeat, so questions 4 and 5 come
    # before the seal. The unit is the RECORD, so the rows survive and every
    # count asserted below is unchanged.
    p.set_repeat_kind(R.REPEATS)
    p.set_unit_of_analysis(R.UNIT_RECORD)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    back = archive.from_bytes(archive.to_bytes(p))
    assert back.grain["answer"] == G.PEOPLE_REPEAT
    assert back.grain["group_col"] == "SUBJ"
    assert back.lockbox["seal_basis"] == SEAL_GROUPED
    assert back.lockbox["basis_source"] == BASIS_USER_STATED
    assert back.barrier_raised is True


def test_an_undetermined_seal_still_says_undetermined_after_a_round_trip():
    from turbotab import archive
    p = _project(longitudinal())
    p.set_grain(G.NOT_SURE)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    back = archive.from_bytes(archive.to_bytes(p))
    assert back.lockbox["seal_basis"] == SEAL_UNDETERMINED
    assert back.lockbox["exploratory"] is True, (
        "a restored undetermined seal renders as a clean lock")
