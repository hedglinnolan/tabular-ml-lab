"""Row labels are people, and a loop must bind a name before it reads it.

Two defects the pre-PR audit found, kept fixed here.

The first is arithmetic on identities. apply_plausibility_filter ended with
.reset_index(drop=True), which is harmless in a script and ruinous in this app:
the test-set lockbox seals a set of row LABELS at upload, and a cohort run
banks the labels of the people in the group. Renumber the survivors and both
sets go on pointing at whatever now sits at those positions. The audit drove
it: a run the sidebar called "sex = Female, 376 of 760" was 189 men and 187
women, with nothing anywhere to say so.

The second is plainer. The "Add all N files to project" button read _name one
line before assigning it, so the NameError was swallowed by the loop's own
except and every file in every multi-file import failed, every time.
"""
import ast
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from ml.pipeline import apply_plausibility_filter
from utils.cohorts import plan_cohorts, start_cohort
from utils.session_state import get_data


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def study(n=800):
    """A study where the impossible BMIs are concentrated in one group."""
    rng = np.random.default_rng(11)
    sex = np.array(["Female"] * (n // 2) + ["Male"] * (n - n // 2))
    bmi = rng.normal(27, 4, n)
    bad = np.r_[rng.choice(np.where(sex == "Female")[0], 30, replace=False),
                rng.choice(np.where(sex == "Male")[0], 10, replace=False)]
    bmi[bad] = 999.0                       # a units mix-up in one source file
    return pd.DataFrame({"sex": sex, "bmi": bmi,
                         "age": rng.integers(20, 80, n),
                         "y": rng.integers(0, 2, n)})


BOUNDS = {"lower_bounds": [15.0, 18.0], "upper_bounds": [50.0, 90.0]}


def test_the_filter_keeps_the_labels_it_was_given():
    df = study()
    kept = apply_plausibility_filter(df, ["bmi", "age"], BOUNDS)
    assert len(kept) < len(df), "the fixture should lose the impossible rows"
    assert list(kept.index) == [i for i in df.index if i in set(kept.index)]
    assert set(kept.index) <= set(df.index), "filtering invented row labels"
    # the surviving rows are still the same people
    pd.testing.assert_frame_equal(kept, df.loc[kept.index])


def test_a_cohort_started_from_a_filtered_frame_holds_only_that_group():
    """The regression in full: filter first, then start the run."""
    df = study()
    st.session_state["raw_data"] = df
    filtered = apply_plausibility_filter(df, ["bmi", "age"], BOUNDS)
    st.session_state["filtered_data"] = filtered

    plan = plan_cohorts(filtered, "sex", "y", "classification")
    chosen = next(c for c in plan.viable if c.label == "Female")
    start_cohort(filtered, plan, chosen, "y")

    # what the real switch does: the per-model gate is rebuilt on page 05 for
    # the new group, so cohort_ui drops the old filtered frame and the banked
    # labels are resolved against raw_data from here on.
    st.session_state.pop("filtered_data", None)

    seen = get_data()
    assert set(seen["sex"].unique()) == {"Female"}, (
        f"the Female run holds {seen['sex'].value_counts().to_dict()}")
    assert (seen["bmi"] < 100).all(), "an excluded row came back into the run"


def _add_all_loop(tree):
    """The for-loop inside the 'Add all N files to project' button block."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Call)):
            continue
        for kw in node.test.keywords:
            if (kw.arg == "key" and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "add_all_files"):
                # the loop over the uploads, not the later `for msg in failed`
                # — ast.walk is breadth-first, so ask for the earliest one.
                return min((n for n in ast.walk(node) if isinstance(n, ast.For)),
                           key=lambda n: n.lineno)
    raise AssertionError("the add-all-files button block is gone")


def test_the_multi_file_loop_binds_every_name_before_it_reads_it():
    src = open("pages/01_Upload_and_Audit.py").read()
    loop = _add_all_loop(ast.parse(src))

    # ast.walk is breadth-first, so take the minimum line rather than the first
    # node handed back — reading in document order is the whole point here.
    first_store, first_load = {}, {}
    for n in ast.walk(loop):
        if not isinstance(n, ast.Name):
            continue
        seen = first_store if isinstance(n.ctx, ast.Store) else first_load
        seen[n.id] = min(seen.get(n.id, n.lineno), n.lineno)
    # the loop target counts as bound at the top
    for t in ast.walk(loop.target):
        if isinstance(t, ast.Name):
            first_store[t.id] = loop.lineno

    too_early = {name: (first_load[name], line) for name, line in first_store.items()
                 if name in first_load and first_load[name] < line}
    assert not too_early, (
        "read before it is assigned, inside a try that swallows the NameError: "
        + ", ".join(f"{n} (read L{r}, assigned L{a})"
                    for n, (r, a) in sorted(too_early.items())))
