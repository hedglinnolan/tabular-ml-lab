"""Upload & Audit has to end with a straight answer about the working table.

Every surface on that page used to be prospective — the join preview explains
what a merge will do, beautifully, before you commit to it. Nothing restated
the result afterwards, so:

  - the committed join step went on rendering as a live control, and its
    headline ("→ 156 rows × 10 columns") described the inputs while sitting
    directly above a table a cleaning action had reduced to 9 columns;
  - the number a researcher most needs — that the 60 people they enrolled
    became 52 in the table — was stated only inside a preview that scrolls
    away;
  - a cleaning action's confirmation was a toast, so the account of what had
    been done to the table survived exactly one interaction;
  - nothing could be undone.

The ledger is the account. These tests hold it to the arithmetic.
"""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils import table_ledger as L


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def enrolled(n=60):
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "subject_id": np.arange(1001, 1001 + n),
        "age": rng.integers(25, 75, n),
        "site": "MAIN",                      # constant: the audit offers to drop it
    })


def joined_to_visits(demo, keep=52, visits=3):
    """The shape an inner join with a repeated-measures file produces."""
    rng = np.random.default_rng(3)
    ids = demo["subject_id"].to_numpy()[:keep]
    return pd.DataFrame({
        "subject_id": np.repeat(ids, visits),
        "age": np.repeat(demo["age"].to_numpy()[:keep], visits),
        "site": "MAIN",
        "kcal": rng.normal(2100, 400, keep * visits),
    })


# ── the arithmetic ───────────────────────────────────────────────────────

def test_a_join_that_loses_people_says_how_many():
    demo = enrolled()
    after = joined_to_visits(demo)
    L.clear()
    step = L.record("Combined 3 files", L.COMBINE, before=demo, after=after)

    assert step.subjects_before == 60 and step.subjects_after == 52
    assert step.subjects_delta == -8
    assert "lost 8 subjects" in step.cost_sentence()
    assert step.loss_sentence() == "8 subjects"


def test_the_loss_line_does_not_advertise_a_gain():
    """A join can lose 8 people and gain a column in the same breath."""
    demo = enrolled()
    after = joined_to_visits(demo)
    L.clear()
    step = L.record("Combined 3 files", L.COMBINE, before=demo, after=after)
    assert "gained" in step.cost_sentence()          # the full account says both
    assert "gain" not in step.loss_sentence()        # the losses list says one


def test_dropping_a_column_names_it():
    demo = enrolled()
    after = demo.drop(columns=["site"])
    L.clear()
    step = L.record("Drop constant columns", L.CLEAN, before=demo, after=after)
    assert step.columns_removed == ["site"]
    assert "`site`" in step.cost_sentence() and "1 column" in step.cost_sentence()
    assert step.subjects_delta == 0


def test_the_first_step_is_a_starting_point_not_a_gain():
    """`before=None` means "this is where the table came from"."""
    demo = enrolled()
    L.clear()
    step = L.record("Started from demographics.csv", L.ADD, before=None, after=demo)
    assert step.rows_before == step.rows_after == 60
    assert step.cost_sentence() == "" and not step.is_lossy


def test_the_net_change_spans_the_whole_account():
    demo = enrolled()
    after = joined_to_visits(demo)
    L.clear()
    L.record("Combined 3 files", L.COMBINE, before=demo, after=after)
    L.record("Drop constant columns", L.CLEAN,
             before=after, after=after.drop(columns=["site"]))
    net = L.net_change()
    assert net["rows_before"] == 60 and net["rows_after"] == 156
    assert net["subjects_before"] == 60 and net["subjects_after"] == 52
    assert net["cols_before"] == 3 and net["cols_after"] == 3
    assert net["n_steps"] == 2


def test_rows_are_not_people():
    after = joined_to_visits(enrolled())
    assert L.subject_column(after) == "subject_id"
    assert L.count_subjects(after) == 52 and len(after) == 156


def test_a_table_with_no_identifier_reports_no_subject_count():
    """Silence is right here; a guessed n would be worse than none."""
    df = pd.DataFrame({"bmi": [22.0, 27.5, 31.0], "y": [0, 1, 0]})
    assert L.subject_column(df) is None
    assert L.count_subjects(df) is None


# ── undo ─────────────────────────────────────────────────────────────────

def test_the_last_step_can_be_undone():
    demo = enrolled()
    after = demo.drop(columns=["site"])
    L.clear()
    L.record("Drop constant columns", L.CLEAN, before=demo, after=after)
    assert L.steps()[0].undoable

    back = L.undo_to(0)
    assert back is not None
    assert list(back.columns) == list(demo.columns)
    assert L.steps() == [], "the undone step is still in the account"


def test_undo_rolls_back_everything_after_the_chosen_step():
    demo = enrolled()
    a = demo.drop(columns=["site"])
    b = a.drop(columns=["age"])
    L.clear()
    L.record("Drop constant columns", L.CLEAN, before=demo, after=a)
    L.record("Drop age", L.CLEAN, before=a, after=b)
    assert len(L.steps()) == 2

    back = L.undo_to(0)
    assert list(back.columns) == list(demo.columns)
    assert L.steps() == []


def test_the_undo_handler_itself_runs():
    """The ledger's undo worked; the button's handler imported a module that
    does not exist (`utils.state_reconciler` for `utils.state_reconcile`), so
    clicking Undo raised before it reached any of it. Exercise the handler, not
    just the data structure underneath it."""
    from utils.working_table_ui import _undo
    demo = enrolled()
    after = demo.drop(columns=["site"])
    st.session_state["raw_data"] = after
    st.session_state["working_table"] = after
    L.clear()
    L.record("Drop constant columns", L.CLEAN, before=demo, after=after)

    try:
        _undo(0)
    except Exception as exc:                       # st.rerun() raises to stop
        if type(exc).__name__ not in ("RerunException", "StopException"):
            raise
    restored = st.session_state["working_table"]
    assert "site" in restored.columns, "the undone column did not come back"
    assert st.session_state.get("_working_table_undo_note"), "the undo was silent"


def test_a_table_too_large_to_copy_is_marked_as_not_undoable():
    """Refused honestly rather than dropped out of the history."""
    big = pd.DataFrame(np.zeros((2000, 3000)))
    L.clear()
    step = L.record("Huge step", L.CLEAN, before=big, after=big.iloc[:, :2999])
    assert not step.undoable
    assert L.steps(), "the step must still appear in the account"
    assert L.undo_to(0) is None


# ── the sign-off ─────────────────────────────────────────────────────────

def test_confirming_records_the_shape_it_was_confirmed_at():
    demo = enrolled()
    L.confirm(demo)
    assert L.is_confirmed(demo)
    assert L.confirmed_shape() == (60, 3)


def test_a_confirmation_is_withdrawn_when_the_table_changes():
    demo = enrolled()
    L.clear()
    L.confirm(demo)
    assert L.is_confirmed(demo)

    after = demo.drop(columns=["site"])
    L.record("Drop constant columns", L.CLEAN, before=demo, after=after)
    assert not L.is_confirmed(after), (
        "a stale tick sat beside changed numbers")
    assert L.confirmed_shape() is None


def test_an_undo_also_withdraws_the_confirmation():
    demo = enrolled()
    after = demo.drop(columns=["site"])
    L.clear()
    L.record("Drop constant columns", L.CLEAN, before=demo, after=after)
    L.confirm(after)
    assert L.is_confirmed(after)
    L.undo_to(0)
    assert L.confirmed_shape() is None


# ── persistence ──────────────────────────────────────────────────────────

def test_the_account_survives_a_save_and_restore():
    from utils.session_manager import _collect_session_data, _restore_session_data
    demo = enrolled()
    after = joined_to_visits(demo)
    L.clear()
    L.record("Combined 3 files", L.COMBINE, before=demo, after=after,
             detail="inner join on subject_id")
    L.record("Drop constant columns", L.CLEAN,
             before=after, after=after.drop(columns=["site"]))
    st.session_state["raw_data"] = after.drop(columns=["site"])

    archive, _ = _collect_session_data()
    st.session_state.clear()
    _restore_session_data(archive)

    back = L.steps()
    assert len(back) == 2, "the history of the table was lost on restore"
    assert back[0].subjects_before == 60 and back[0].subjects_after == 52
    assert back[1].columns_removed == ["site"]
    assert not any(s.undoable for s in back), (
        "undo snapshots are session-local and must not claim otherwise")


# ── the page wiring ──────────────────────────────────────────────────────

def test_the_committed_combine_step_stops_rendering_as_a_control():
    """Its headline described the inputs and looked current forever."""
    src = open("pages/01_Upload_and_Audit.py").read()
    assert "_combine_reopen" in src
    assert "Change how these files are combined" in src


def test_every_shape_changing_site_records_a_step():
    import ast
    src = open("pages/01_Upload_and_Audit.py").read()
    tree = ast.parse(src)
    records = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute)
               and n.func.attr == "record"
               and getattr(n.func.value, "id", "") == "_ledger"]
    assert len(records) >= 3, (
        f"only {len(records)} mutation sites record a ledger step; the combine, "
        f"the single-file adopt and the cleaning actions all must")


def test_the_page_states_the_table_before_it_asks_what_to_do_with_it():
    """The sign-off must sit above the st.stop() that gates Step 4."""
    src = open("pages/01_Upload_and_Audit.py").read()
    signoff = src.index("render_exit_assurance(")
    step4 = src.index('st.header("Step 4: Configure Analysis")')
    assert signoff < step4, (
        "the sign-off is below the analysis-type gate, so a researcher who has "
        "not chosen one never sees it")
