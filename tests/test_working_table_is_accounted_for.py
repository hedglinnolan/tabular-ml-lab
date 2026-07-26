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


# ── what happened to the files before the table existed ──────────────────

def test_a_transpose_and_a_repair_reach_the_sign_off():
    """Both happen per-file, and neither shows up in any before/after count.

    A transpose inverts the table; a recode rewrites values in place. The
    first version looked repairs up by dataset id, filename and name — but the
    Import Doctor keys its work by "demographics_csv_0", so nothing ever
    matched and both were invisible from the moment the file was committed.
    """
    from utils.working_table_ui import _surprises
    df = enrolled()
    st.session_state["_dataset_origin"] = {
        "d1": {"filename": "ffq_wide.csv", "transposed": True, "repairs": []},
        "d2": {"filename": "survey.csv", "transposed": False,
               "repairs": ["Recoded 999999 to missing in `income`"]},
    }
    notes = " ".join(_surprises(df, "subject_id"))
    assert "transposed on import" in notes and "ffq_wide.csv" in notes
    assert "value-level repair" in notes and "income" in notes


def test_the_commit_carries_the_import_doctor_key():
    """The last moment the file key and the dataset id coexist."""
    import ast
    src = open("pages/01_Upload_and_Audit.py").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_commit_dataset")
    assert any(a.arg == "file_key" for a in fn.args.args + fn.args.kwonlyargs), (
        "_commit_dataset cannot record repairs it is never told about")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_commit_dataset"]
    upload_calls = [c for c in calls if any(k.arg == "file_key" for k in c.keywords)]
    assert len(upload_calls) >= 2, (
        f"{len(upload_calls)} of {len(calls)} commit sites pass a file key; both "
        f"upload paths must")


# ── one place answers "what have I got" ──────────────────────────────────

def test_only_the_card_states_the_shape():
    """Two places invited the drift this whole change exists to end."""
    page = open("pages/01_Upload_and_Audit.py").read()
    combine = open("utils/combine_ui.py").read()
    assert "Combined table ready" not in combine, (
        "the combine summary restates the shape beside the card")
    assert 'st.caption(f"Shape: {working_df.shape[0]:,} rows' not in page, (
        "Step 2 restates the shape beside the card")


def test_a_step_that_changed_nothing_does_not_draw_an_arrow():
    demo = enrolled()
    L.clear()
    step = L.record("Started from study.csv", L.ADD, before=None, after=demo)
    assert "→" not in step.shape_sentence()
    assert step.shape_sentence() == "60 × 3"


def test_a_handful_of_repeats_is_not_reported_as_repeated_measures():
    """"each person contributes about 1.0 rows" is noise, not information."""
    from utils.working_table_ui import _surprises
    demo = enrolled()
    with_dupes = pd.concat([demo, demo.iloc[:12]], ignore_index=True)
    notes = " ".join(_surprises(with_dupes, "subject_id"))
    assert "1.0 rows" not in notes
    assert "share a `subject_id` with another row" in notes


def test_a_real_repeated_measures_design_still_says_so():
    from utils.working_table_ui import _surprises
    after = joined_to_visits(enrolled())
    notes = " ".join(_surprises(after, "subject_id"))
    assert "Your n is 52, not 156" in notes and "3.0 rows" in notes


# ── the shape change that hides best ─────────────────────────────────────

def _stacked():
    """Two cycles sharing most columns — the classic NHANES stack."""
    from utils.combine_ui import SOURCE_COLUMN
    rng = np.random.default_rng(2)
    a = pd.DataFrame({"seqn": np.arange(10001, 10301),
                      "age": rng.normal(50, 10, 300),
                      "vitamin_d": rng.normal(60, 15, 300),
                      "crp": np.nan, SOURCE_COLUMN: "cycle_2017"})
    b = pd.DataFrame({"seqn": np.arange(20001, 20281),
                      "age": rng.normal(50, 10, 280),
                      "vitamin_d": np.nan,
                      "crp": rng.gamma(2, 1.5, 280), SOURCE_COLUMN: "cycle_2019"})
    return pd.concat([a, b], ignore_index=True)


def test_a_column_measured_in_only_one_file_is_named():
    """Nothing is empty, nothing is dropped, the row count rises exactly as
    promised — and half the sample has no value for two of the columns. This
    used to pass as "nothing about this table looks surprising"."""
    from utils.working_table_ui import _surprises
    notes = " ".join(_surprises(_stacked(), None))
    assert "`vitamin_d` was not measured in cycle_2019" in notes
    assert "`crp` was not measured in cycle_2017" in notes
    assert "280 of 580" in notes and "300 of 580" in notes


def test_a_column_present_everywhere_is_not_flagged():
    from utils.working_table_ui import _columns_absent_from_a_source
    flagged = [c for c, _, _ in _columns_absent_from_a_source(_stacked())]
    assert "age" not in flagged and "seqn" not in flagged


def test_an_unstacked_table_has_no_source_column_to_reason_about():
    from utils.working_table_ui import _columns_absent_from_a_source
    assert _columns_absent_from_a_source(enrolled()) == []


def test_the_subject_note_never_says_none():
    """count_subjects re-derives the column when given None; the text has to
    be attributed to the same one or it reads "`None` repeats"."""
    from utils.working_table_ui import _surprises
    notes = " ".join(_surprises(joined_to_visits(enrolled()), None))
    assert "`None`" not in notes


# ── invalidation ─────────────────────────────────────────────────────────

def test_forgetting_the_table_forgets_its_account_too():
    """Three buttons dropped the table and left the ledger standing. The next
    combine then appended a second "Combined N files" step to an account that
    still described the first, giving a chain that did not join up."""
    import ast
    src = open("pages/01_Upload_and_Audit.py").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_forget_working_table")
    body = ast.dump(fn)
    assert "working_table" in body and "clear" in body

    # every site that drops the working table must go through it
    drops = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "pop"
             and n.args and isinstance(n.args[0], ast.Constant)
             and n.args[0].value == "working_table"]
    assert len(drops) <= 1, (
        f"{len(drops)} places pop working_table directly; they must call "
        f"_forget_working_table so the ledger goes with it")


def test_removing_a_file_says_what_it_costs():
    src = open("pages/01_Upload_and_Audit.py").read()
    assert "is discarded and you will combine the remaining files" in src, (
        "Remove destroys the working table on one unlabeled click")


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
