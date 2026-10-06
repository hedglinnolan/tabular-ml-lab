"""L48-C — `GUIDED-180`, scoped to the half that can be driven.

Eighteen live decision kinds had no `ACTION_CONTRACT` row, and **two of them —
`apply_bulk` and `route_missingness_bulk` — rewrite the working table**, so all
three of `devchecks.py`'s contract guards were inert on the highest-blast-radius
paths in the app. `ACTION_CONTRACT.get(kind)` returns `None` for an unlisted
kind and every consumer then returns `[]`.

**Filling all eighteen is not one loop**, and that ruling stands: each row is a
claim about whether a kind touches the table and how many things it makes stale,
and a wrong claim turns a guard into a false alarm, which is how a guard gets
switched off. So this loop does three things instead.

## 1 · The two that mutate, DERIVED BY DRIVING

Not reasoned. The input was changed, the table's content hash and the stale list
were watched, and what moved is what the row says. The numbers are in
`ACTION_CONTRACT`'s own comment and re-derived here.

## 2 · `unlisted` becomes a state instead of a hole

`.get(kind)` returning `None` means **unchecked**, and *unchecked* and
*declared as touching nothing* were rendering as one value. That is the seal's
`group_col: None` exactly, and `DESIGN_LANGUAGE.md` §09's recorded-absence rule
is the pattern — `undetermined` being first-class is the precedent.

Three states now: `declared`, `unclassified` (a decision, with the loop it is
due at, reported on every transition and written to `unclassified.jsonl`), and
`undispositioned` — a kind in neither table, which **is** a violation, because
it means a kind was added and nobody decided.

**Why `unclassified` is not a violation**, stated because the opposite is the
obvious move: `test_the_same_drive_with_no_planted_bug_is_clean` drives
`set_repeat_kind` and `set_unit_of_analysis`, both unclassified, and asserts
zero violations. That test is right — *"a check that fires on a correct drive
gets switched off within a day"* — and a drive using an unclassified kind is a
correct drive. The report is the answer; the alarm is not.

## 3 · The other sixteen are a list with deadlines, not sixteen guesses

`devchecks.UNCLASSIFIED`, and this file gates that it stays complete.

## What is NOT covered

- **`stale` for `apply_bulk` is declared `None`.** Three drives added zero, and
  none of them changed the COLUMN SET — which is the case that would stale a
  recorded selection. Declaring 0 from three runs that could not have produced
  anything else would be a measurement dressed as a contract.
- **The sixteen themselves.** Named, dated, unmeasured.
- **`EFFECTS`' twenty-nine missing sentences** — `GUIDED-181`, deliberately
  untouched, and the same argument applies one layer up.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path


DATA = Path(__file__).resolve().parent / "sample_data"


# ── 2 · unlisted is a state ─────────────────────────────────────────────────

def test_a_kind_in_neither_table_is_a_violation():
    """The hole, closed. A kind nobody dispositioned no longer passes silently."""
    from turbotab import devchecks

    assert devchecks.classification("apply_bulk") == "declared"
    assert devchecks.classification("set_lens") == "unclassified"
    assert devchecks.classification("a_kind_nobody_wrote") == "undispositioned"

    quiet = devchecks.every_decision_kind_has_a_disposition("set_lens", {}, {})
    assert quiet == [], (
        "a DECLARED unclassified kind raises a violation. A drive that uses "
        "one is a correct drive, and a check that fires on a correct drive "
        "gets switched off within a day")
    loud = devchecks.every_decision_kind_has_a_disposition(
        "a_kind_nobody_wrote", {}, {})
    assert len(loud) == 1
    assert loud[0].check == "a_decision_kind_has_no_disposition"
    assert "a_kind_nobody_wrote" in loud[0].message
    # And it runs on every transition rather than only when called by hand.
    source = Path(devchecks.__file__).read_text(encoding="utf-8")
    assert "_guard(every_decision_kind_has_a_disposition" in source, (
        "the check exists and `check_transition` does not call it — a "
        "capability without its consumer, in the file that watches for them")


def test_an_unclassified_transition_is_written_down(tmp_path, monkeypatch):
    """`note_unclassified` records, and the index says so.

    Not a violation, and not silence either. Without this, *"no violations"*
    over a drive of unclassified kinds reads as *"everything was checked"*.
    """
    from turbotab import devchecks

    monkeypatch.setattr(devchecks, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(devchecks, "_SESSION", None)
    monkeypatch.setattr(devchecks, "enabled", lambda: True)

    devchecks.note_unclassified("set_lens", {"kind": "set_lens"})
    devchecks.note_unclassified("apply_bulk", {"kind": "apply_bulk"})
    session = devchecks.session()
    assert [row["kind"] for row in session.unclassified] == ["set_lens"], (
        "either the unclassified kind was not recorded, or a DECLARED kind "
        "was recorded as unclassified")
    assert session.unclassified[0]["due"] == devchecks.UNCLASSIFIED["set_lens"]
    assert (tmp_path / session.started_at / "unclassified.jsonl").exists()

    index = devchecks.write_index()
    text = Path(index).read_text(encoding="utf-8") if index else ""
    assert "no contract watching" in text and "`set_lens`" in text, (
        f"the drive index does not say which transitions ran unwatched: "
        f"{text[:400]!r}")
