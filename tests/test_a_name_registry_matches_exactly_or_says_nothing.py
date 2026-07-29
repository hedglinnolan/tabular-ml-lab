"""The remedy for a class this project has now fixed five times, applied as one.

`ml/clinical_units.py`, `ml/physiology_reference.py` and `utils/theory_anchors.py`
were each repaired individually with the same fix — **exact key or declared
alias; an unknown name yields silence** — and the fourth and fifth arrived
anyway, because the remedy lived in three separate modules and nothing carried
it across. `FEATURE_PARITY.md` names that: *a principle written in one place and
applied in another is the same silence as a capability with no row.*

The fourth was `_KNOWN_UNITS` in `ml/import_doctor.py` (`IMPORT-267`): the bare
letters `l` and `s`, so a column of education levels was asserted at `critical`
to mix measurement units.

**The fifth was found by the guard in this file, on its first run**, and it is
the reason the guard scans for a pattern rather than checking a list: the three
lists in `ml/triage.py::detect_cohort_structure` are **local variables**, so no
audit of module-level constants would ever have seen them. `subjective_wellbeing`
contains `subject`, so a wellbeing score was a patient identifier; `yearly_income`
contains `year`, so it was a time column. Both feed whether the data is read as
longitudinal, which decides how the held-out rows are chosen.

## The two halves

`test_every_substring_match_against_a_name_is_declared` is the guard. It walks
the AST for the shape *"is any entry of this collection a substring of that
name?"* and requires every site to appear in `DECLARED` with its disposition. A
sixth fails the suite.

Everything above it is the behavior: exact matching works, near-misses are
silent, and the names that used to false-positive no longer do.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import name_registry as N                                     # noqa: E402
from ml.triage import detect_cohort_structure                         # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── the rule itself ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("spelling", [
    "subject_id", "subject-id", "Subject ID", "SubjectID", "  subject_id  "])
def test_one_name_spelled_five_ways_is_one_name(spelling):
    lookup = N.build({"subject": ["subject_id"]})
    assert N.match(spelling, lookup) == "subject"


@pytest.mark.parametrize("name", [
    "subjective_wellbeing",     # contains `subject`
    "subjects_enrolled_total",  # contains `subject`
    "membership_fee",           # contains `member`
    "encounter_cost",           # contains `encounter`
    "recordkeeping_score",      # contains `record`
])
def test_a_name_that_merely_contains_a_key_is_not_a_match(name):
    """The whole class, in one parametrize. Every one of these is a plausible
    column in a real study and every one used to resolve."""
    lookup = N.build({"subject": ["subject_id"], "member": [], "record": [],
                      "encounter": []})
    assert N.match(name, lookup) is None


def test_an_unknown_name_yields_silence_rather_than_a_neighbours_answer():
    """`None` is an answer: *this vocabulary does not describe this column*.

    A gap is visible to the user and costs a question; a wrong claim is
    invisible and licenses an action — a deleted entry, a stratified split on a
    wellbeing score, a units warning on a column of education levels.
    """
    lookup = N.build({"glucose": ["blood_glucose"]})
    assert N.match("glucose", lookup) == "glucose"
    assert N.match("blood glucose", lookup) == "glucose"
    assert N.match("glucose_proxy", lookup) is None
    assert N.match("hba1c", lookup) is None


def test_two_keys_claiming_one_spelling_is_refused_rather_than_resolved():
    """A silent winner is exactly the confidently-wrong outcome the remedy is
    for."""
    with pytest.raises(ValueError, match="claimed by both"):
        N.build({"weight": ["mass"], "burden": ["mass"]})


def test_nothing_is_stemmed_or_truncated():
    """Every one of those is a substring match wearing a normalizer's clothes."""
    lookup = N.build({"visit": []})
    assert N.match("visits", lookup) is None
    assert N.match("visiting_nurse", lookup) is None
    assert N.match("VISIT", lookup) == "visit"


# ── the fifth registry, repaired ─────────────────────────────────────────────

def _longitudinal_frame() -> pd.DataFrame:
    """A frame where the false positives are actually REACHABLE.

    The first version used `subjective_wellbeing` as a float, and the probe came
    back GREEN: the entity-ID branch has a downstream cardinality-and-dtype gate
    that drops float columns with decimals, so the substring match fired and was
    then discarded for an unrelated reason. The test asserted a property that
    was already true by accident — `MINE-001`'s accidental guard, which moves
    and takes the guarantee with it.

    So the trap columns are high-cardinality and discrete, which is what it
    takes to reach the decision the name is used for.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "subject_id": [f"S{i // 3:03d}" for i in range(60)],
        "visit_date": pd.date_range("2024-01-01", periods=60, freq="D").astype(str),
        # Contains `subject`, and clears the cardinality gate the entity-ID
        # branch applies after the name match.
        "subject_matter_code": [f"SMC-{i:04d}" for i in range(60)],
        "yearly_income": rng.normal(size=60),
        "glucose": rng.normal(95, 12, 60),
    })


def test_a_subject_matter_code_is_not_a_patient_identifier():
    """`subject_matter_code` contains `subject`, is high-cardinality and
    discrete — so it clears every downstream gate and the NAME is the only thing
    deciding. Whether it is read as an entity ID decides whether the data is
    read as longitudinal, which decides how the held-out rows are chosen.
    """
    out = detect_cohort_structure(_longitudinal_frame())
    assert "subject_matter_code" not in out["entity_id_candidates"], (
        "a subject-matter code is being read as a patient identifier")


def test_an_income_column_is_not_a_time_column():
    """`yearly_income` contains `year`."""
    out = detect_cohort_structure(_longitudinal_frame())
    assert "yearly_income" not in out["time_column_candidates"]


def test_the_real_time_column_is_still_found():
    """The remedy has to be narrow in both directions. A repair that silenced
    the true reading would trade a wrong answer for a missing one."""
    out = detect_cohort_structure(_longitudinal_frame())
    assert "visit_date" in out["time_column_candidates"]
    assert out["detected"] == "longitudinal"


def test_a_declared_identifier_spelling_still_resolves():
    df = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(80)],
        "age": np.arange(80) % 50 + 20,
        "outcome": np.arange(80) % 2,
    })
    out = detect_cohort_structure(df)
    assert "participant_id" in out["entity_id_candidates"]


# ── the guard that catches the sixth ─────────────────────────────────────────

# Every site matching "is any entry of this collection a substring of that
# name?", with its disposition. A site not listed here fails the suite.
#
# The value is a REASON, not a rubber stamp: `FEATURE_PARITY.md`'s objection to
# `classic-only` applies here too — a disposition is a claim to be justified,
# never a shrug.
# `ml/triage.py`'s three lists were here and are GONE, because they were
# repaired rather than dispositioned — `test_the_declared_list_has_not_gone_stale`
# refused the entries the moment `detect_cohort_structure` stopped matching by
# substring. That is the list being checked in both directions, and it is why a
# repair cannot leave a permanent excuse behind it.
DECLARED = {
    ("ml/splits.py", "kept_labels"):
        "Not a name registry. `kept_labels` holds ROW LABELS and the test is "
        "membership in an index, not a substring of a name.",
    ("turbotab/packs.py", "_ASSAY_PACKS"):
        "Not a name registry. Membership of a lens key in a two-element tuple "
        "of lens keys, both of which this module defines.",
    ("turbotab/engine.py", "display_cols"):
        "Not a name registry. A local list of the frame's own column labels, "
        "tested for membership in another list of the frame's own labels.",
    ("pages/06_Train_and_Compare.py", "models_to_train"):
        "Not a name registry. Model keys the user selected, tested against "
        "keys the registry defines.",
    ("pages/07_Explainability.py", "expected_cols"):
        "Not a name registry. The frame's own columns against a computed list.",
    ("pages/10_Report_Export.py", "feature_cols"):
        "Not a name registry. The frame's own columns.",
    ("utils/combine_ui.py", "cols"):
        "Not a name registry. The frame's own columns. FROZEN PATH "
        "(TRANSITION_PLAN section 05) — engine-move-only, and there is nothing "
        "here to repair.",
}

SCAN_DIRS = ("ml", "utils", "turbotab", "models", "pages")


def _substring_sites():
    """Every `any(x in NAME for x in COLLECTION)`-shaped site, by AST.

    A pattern rather than a list, and that is the point: the fifth registry was
    three LOCAL variables inside one function, so nothing that enumerated
    module-level constants could have found it.
    """
    found = set()
    for directory in SCAN_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
            if ".venv" in rel or path.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:                            # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.GeneratorExp, ast.ListComp,
                                         ast.SetComp)):
                    continue
                for gen in node.generators:
                    if not (isinstance(gen.target, ast.Name)
                            and isinstance(gen.iter, ast.Name)):
                        continue
                    for sub in ast.walk(node.elt):
                        if (isinstance(sub, ast.Compare)
                                and sub.ops and isinstance(sub.ops[0], ast.In)
                                and isinstance(sub.left, ast.Name)
                                and sub.left.id == gen.target.id):
                            found.add((rel, gen.iter.id))
    return found


def test_every_substring_match_against_a_name_is_declared():
    """The guard that catches the sixth.

    It found the fifth on its first run. Writing the check is half the work;
    the other half is naming the moment it runs, and this one runs with the
    suite.
    """
    undeclared = sorted(s for s in _substring_sites() if s not in DECLARED)
    assert not undeclared, (
        "these sites ask 'is any entry of this collection a substring of that "
        "name?', and nothing says whether that is safe:\n  "
        + "\n  ".join(f"{f} :: {n}" for f, n in undeclared)
        + "\n\nThe remedy is `ml/name_registry.py` — exact key or declared "
          "alias, unknown yields silence. If the site is not a name registry, "
          "add it to DECLARED with the reason it is not. A disposition is a "
          "claim to be justified, never a shrug.")


def test_the_declared_list_has_not_gone_stale():
    """An entry for a site that no longer exists is the register lying.

    Same failure as a `classic-only` row whose reason no longer holds, and the
    same repair: the list is checked in both directions.
    """
    sites = _substring_sites()
    stale = sorted(s for s in DECLARED if s not in sites)
    assert not stale, (
        "these declared sites no longer exist; remove them:\n  "
        + "\n  ".join(f"{f} :: {n}" for f, n in stale))


# ── all five, asserted as a class ────────────────────────────────────────────

def _physiology_lookup():
    from ml.physiology_reference import load_reference_bundle, match_variable_key
    reference = load_reference_bundle()["nhanes"]
    return lambda name: match_variable_key(name, reference)


def _clinical_units_lookup():
    from ml.clinical_units import match_clinical_variable
    return match_clinical_variable


def _triage_entity_lookup():
    """`detect_cohort_structure`'s registry, reached through the behavior."""
    def match(name: str):
        df = pd.DataFrame({name: [f"X{i:04d}" for i in range(80)],
                           "age": np.arange(80) % 50 + 20,
                           "outcome": np.arange(80) % 2})
        out = detect_cohort_structure(df)
        return name if name in out["entity_id_candidates"] else None
    return match


# Every registry that answers "is this column that variable?", the exact or
# aliased spelling it must accept, and a near-miss it must refuse. The near-miss
# is a real column name in each case, not a contrived one.
REGISTRIES = [
    ("physiology_reference", _physiology_lookup, "hba1c", "hba1c_proxy"),
    ("physiology_reference/alias", _physiology_lookup, "bp_sys", "bp_sys_delta"),
    ("clinical_units", _clinical_units_lookup, "glucose", "glucose_change"),
    ("ml.triage entity ids", _triage_entity_lookup, "participant_id",
     "participants_screened"),
]


@pytest.mark.parametrize("label,factory,exact,near_miss", REGISTRIES)
def test_every_registry_accepts_a_declared_spelling_and_refuses_a_near_miss(
        label, factory, exact, near_miss):
    """*"Apply it as a class this time."*

    Three of these were repaired individually with the same fix and the fourth
    and fifth arrived anyway. This asserts the RULE across all of them in one
    place, so the next repair has somewhere to be checked rather than somewhere
    to be repeated.

    **What this does not claim.** `ml/name_registry.py` states the rule once and
    `ml/triage.py` is the only registry built on it; `physiology_reference` and
    `clinical_units` carry their own implementations of the same rule, written
    before it existed. They behave identically — which is what this test checks
    — and they are not sharing code. That is a smaller duplication than the one
    that caused this class and it is worth saying out loud rather than implying
    a migration that did not happen.
    """
    match = factory()
    assert match(exact) is not None, f"{label} rejects its own declared spelling"
    assert match(near_miss) is None, (
        f"{label} resolves {near_miss!r}, so it is still matching by substring "
        f"and the class has a sixth member")


def test_the_shared_rule_and_the_private_ones_agree_on_normalization():
    """The rule stated once, checked against the implementations that predate
    it. Case and separators only — nothing stemmed, nothing truncated."""
    from ml.physiology_reference import load_reference_bundle, match_variable_key
    reference = load_reference_bundle()["nhanes"]
    lookup = N.build({"bp_sys": ["systolic", "sbp"]})
    for spelling in ("bp_sys", "BP_SYS", "bp-sys", "Bp Sys", "bpsys"):
        assert N.match(spelling, lookup) == "bp_sys"
        assert match_variable_key(spelling, reference) == "bp_sys"
