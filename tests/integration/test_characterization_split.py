"""Characterization tests for the split block — written BEFORE it moves.

`pages/06_Train_and_Compare.py:380-760` is ~370 lines of the most
safety-critical logic in the app, it is not a function, and nothing tests it.
`ml/splits.py` is 20 lines holding one array helper.

These tests pin what the block does **today**, by driving the real page through
AppTest and recording the partition it produces. That matters more than it
sounds: a characterization test written against a transcription of the block
would pin the transcription, and if the transcription were wrong the extraction
would be "verified" against the same mistake. So nothing here re-implements
anything — the page runs, the button is clicked, and the resulting session state
is the golden output.

The extraction (L6) must reproduce every assertion in this file **unchanged**.
If a test here has to be edited to make the extracted code pass, the extraction
changed behavior and is wrong.

Four branches, in the page's own priority order:

1. grouped          — `GroupShuffleSplit`, when cohort structure is longitudinal
2. chronological    — sort-and-slice, when `use_time_split` and a datetime column
3. lockbox          — the frozen labels *are* the test set; only train/val drawn
4. stratified/plain — `train_test_split`

Run:  venv/bin/python -m pytest tests/integration/test_characterization_split.py -v
"""
import numpy as np
import pandas as pd
import pytest

from tests.integration.conftest import build_test_dataframe, inject_data_state

pytestmark = pytest.mark.timeout(300)


# ── harness ──────────────────────────────────────────────────────────────

@pytest.fixture
def train_page():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file("pages/06_Train_and_Compare.py")


def _click(at, label_fragment):
    for b in at.button:
        if label_fragment in (b.label or ""):
            return b.click()
    raise AssertionError(
        f"No button matching {label_fragment!r}. Buttons: {[b.label for b in at.button]}")


def _inject_pipeline(at, df, target_col="glucose"):
    """Satisfy the page's preprocessing gate with a minimal per-model pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    numeric = [c for c in df.columns if c != target_col and df[c].dtype.kind in "fi"]
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), numeric)],
        remainder="drop",
    )
    at.session_state["preprocessing_pipelines_by_model"] = {"ridge": pre}
    at.session_state["preprocessing_config_by_model"] = {"ridge": {}}


def _route_grouped(at, entity_col):
    """Steer the page into the grouped branch the way the page decides it.

    `pages/06:186-189` reads `cohort_structure_detection.final` and
    `.entity_id_final`, not a `use_group_split` flag — seeding the flag does
    nothing. This seeds the detection object the page actually consults.
    """
    from utils.session_state import CohortStructureDetection
    at.session_state["cohort_structure_detection"] = CohortStructureDetection(
        detected="longitudinal", confidence="high", entity_id_detected=entity_col)


def _route_time(at, datetime_col):
    """Steer the page into the chronological branch.

    Two things are needed: a datetime column on `data_config`, and the
    "Use Time-Based Split" checkbox, whose default the page reads from
    `train_use_time_split` (`pages/06:201-208`).

    Note the page *rebuilds* `split_config` from its own widgets at `:272`, so
    setting `split_config.use_time_split` directly is overwritten on the next
    run. The branch has to be chosen through the widget state, as a user would.
    """
    at.session_state["data_config"].datetime_col = datetime_col
    at.session_state["train_use_time_split"] = True


def _prepare(at):
    at.run()
    _click(at, "Prepare Splits")
    at.run()
    return at


def _ss(at, key, default=None):
    """`AppTest.session_state` has no `.get()` — it reads "get" as a key name."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def _partition(at):
    """The golden output: sizes, index sets, and the CV scheme chosen.

    Index lists are compared as *sets* only where the branch does not promise an
    order, and as lists where it does (the chronological branch promises order).
    """
    missing = [k for k in ("X_train", "X_val", "X_test",
                           "train_indices", "val_indices", "test_indices")
               if _ss(at, k) is None]
    if missing:
        raise AssertionError(
            f"the page produced no splits (missing {missing}). "
            f"Exceptions: {[str(e.value) for e in at.exception]}")
    groups = _ss(at, "cv_groups_train")
    return {
        "n_train": len(at.session_state["X_train"]),
        "n_val": len(at.session_state["X_val"]),
        "n_test": len(at.session_state["X_test"]),
        "train_indices": list(at.session_state["train_indices"]),
        "val_indices": list(at.session_state["val_indices"]),
        "test_indices": list(at.session_state["test_indices"]),
        "cv_strategy": _ss(at, "cv_strategy"),
        "cv_groups_train": None if groups is None else list(groups),
    }


def _assert_partition_is_sane(p, n_rows):
    """Invariants every branch must satisfy, whatever else it does."""
    tr, va, te = set(p["train_indices"]), set(p["val_indices"]), set(p["test_indices"])
    assert tr and va and te, "a partition left one side empty"
    assert not (tr & va), "train and validation overlap"
    assert not (tr & te), "train and test overlap — this is the leak"
    assert not (va & te), "validation and test overlap"
    assert len(tr) == p["n_train"] and len(va) == p["n_val"] and len(te) == p["n_test"]
    assert (tr | va | te) <= set(range(n_rows))


# ── branch 4: plain / stratified random ──────────────────────────────────

def test_plain_random_split_is_stable_and_disjoint(train_page):
    at = train_page
    df = build_test_dataframe(n=200)
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    p = _partition(_prepare(at))

    _assert_partition_is_sane(p, len(df))
    assert p["cv_strategy"] == "standard"
    assert p["cv_groups_train"] is None
    # The whole of the non-missing target is used: nothing is silently dropped.
    assert p["n_train"] + p["n_val"] + p["n_test"] == int(df["glucose"].notna().sum())


def test_plain_random_split_is_reproducible(train_page):
    """Same seed, same data, same partition — a manuscript claim, not a nicety."""
    df = build_test_dataframe(n=200)
    from streamlit.testing.v1 import AppTest

    runs = []
    for _ in range(2):
        at = AppTest.from_file("pages/06_Train_and_Compare.py")
        inject_data_state(at, df, target_col="glucose", task_type="regression")
        _inject_pipeline(at, df)
        runs.append(_partition(_prepare(at)))

    assert runs[0]["train_indices"] == runs[1]["train_indices"]
    assert runs[0]["test_indices"] == runs[1]["test_indices"]


def test_stratified_split_preserves_class_balance(train_page):
    from tests.integration.conftest import build_classification_dataframe

    at = train_page
    df = build_classification_dataframe(n=240)
    target = "outcome" if "outcome" in df.columns else df.columns[-1]
    inject_data_state(at, df, target_col=target, task_type="classification")
    _inject_pipeline(at, df, target_col=target)
    # stratify is not a user toggle: the page sets it for classification
    # whenever the split is neither grouped nor chronological (:277).
    p = _partition(_prepare(at))

    _assert_partition_is_sane(p, len(df))
    assert p["cv_strategy"] == "standard"

    # Stratification means the partitions carry the population's class mix.
    y = df[target].reset_index(drop=True)
    overall = y.value_counts(normalize=True)
    for key in ("train_indices", "test_indices"):
        part = y.iloc[p[key]].value_counts(normalize=True)
        for cls in overall.index:
            assert abs(part.get(cls, 0) - overall[cls]) < 0.12, (
                f"{key} lost the class balance for {cls!r}")


# ── branch 3: the lockbox ────────────────────────────────────────────────

def _seal_lockbox(at, df, labels):
    at.session_state["test_lockbox"] = {
        "labels": sorted(labels),
        "fraction": len(labels) / len(df),
        "seed": 42,
        "n_total": len(df),
        "n_test": len(labels),
        "signature": "characterization-fixture",
    }
    at.session_state["exploratory_mode"] = False


def test_lockbox_labels_are_the_test_set(train_page):
    """The invariant with the most scar tissue behind it.

    When a lockbox is sealed, its labels *are* the test set. The page divides
    only what is left into train and validation — it never redraws the test set.
    """
    at = train_page
    df = build_test_dataframe(n=200)
    rng = np.random.RandomState(42)
    labels = sorted(rng.choice(df.index.values, size=30, replace=False).tolist())
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    _seal_lockbox(at, df, labels)
    p = _partition(_prepare(at))

    _assert_partition_is_sane(p, len(df))
    # `train_indices` etc. are POSITIONS into the target-complete frame. Map the
    # sealed labels through the same mask the page applies.
    mask = df["glucose"].notna()
    kept_labels = list(df.index[mask])
    expected_test_pos = {i for i, lbl in enumerate(kept_labels) if lbl in set(labels)}
    assert set(p["test_indices"]) == expected_test_pos, (
        "the test set is not exactly the sealed labels")
    assert not (set(p["train_indices"]) & expected_test_pos)


def test_exploratory_mode_ignores_the_lockbox(train_page):
    at = train_page
    df = build_test_dataframe(n=200)
    rng = np.random.RandomState(42)
    labels = sorted(rng.choice(df.index.values, size=30, replace=False).tolist())
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    _seal_lockbox(at, df, labels)
    at.session_state["exploratory_mode"] = True
    p = _partition(_prepare(at))

    _assert_partition_is_sane(p, len(df))
    mask = df["glucose"].notna()
    kept_labels = list(df.index[mask])
    sealed_pos = {i for i, lbl in enumerate(kept_labels) if lbl in set(labels)}
    assert set(p["test_indices"]) != sealed_pos, (
        "exploratory mode still honored the lockbox")


# ── branch 1: grouped ────────────────────────────────────────────────────

def test_grouped_split_keeps_a_subject_on_one_side(train_page):
    """`detect_repeated_subjects` exists so a patient cannot appear in both
    partitions. This is that promise, at the partition level."""
    at = train_page
    base = build_test_dataframe(n=120)
    # Three rows per subject: the shape that makes a naive split leak.
    df = pd.concat([base, base, base], ignore_index=True)
    df["subject_id"] = np.tile(np.arange(len(base)), 3)

    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    _route_grouped(at, "subject_id")
    p = _partition(_prepare(at))

    assert p["cv_strategy"] == "group", (
        "the page did not route to the grouped branch — this test pins that "
        "branch and skipping it would leave a quarter of the split logic "
        "unpinned before the extraction")

    _assert_partition_is_sane(p, len(df))
    subj = df["subject_id"].reset_index(drop=True)
    tr = set(subj.iloc[p["train_indices"]])
    va = set(subj.iloc[p["val_indices"]])
    te = set(subj.iloc[p["test_indices"]])
    assert not (tr & te), "a subject appears in both train and test"
    assert not (tr & va) and not (va & te), "a subject spans partitions"
    # CV must inherit the grouping or the folds leak too.
    assert p["cv_groups_train"] is not None
    assert len(p["cv_groups_train"]) == p["n_train"]


# ── branch 2: chronological ──────────────────────────────────────────────

def test_time_split_is_ordered_and_never_trains_on_the_future(train_page):
    at = train_page
    df = build_test_dataframe(n=200)
    df["visit_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    # Shuffle the rows so ordering can only come from the split, not the frame.
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    _route_time(at, "visit_date")
    p = _partition(_prepare(at))

    assert p["cv_strategy"] == "time", (
        "the page did not route to the chronological branch")

    _assert_partition_is_sane(p, len(df))
    dates = df["visit_date"].reset_index(drop=True)
    assert dates.iloc[p["train_indices"]].max() <= dates.iloc[p["val_indices"]].min()
    assert dates.iloc[p["val_indices"]].max() <= dates.iloc[p["test_indices"]].min()


# ── branch selection itself ──────────────────────────────────────────────

def test_branch_priority_group_beats_time(train_page):
    """Four mutually exclusive branches in a fixed priority order. Grouping wins
    over a time split, because a subject spanning partitions is the worse leak.
    """
    at = train_page
    base = build_test_dataframe(n=90)
    df = pd.concat([base, base], ignore_index=True)
    df["subject_id"] = np.tile(np.arange(len(base)), 2)
    df["visit_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    _route_grouped(at, "subject_id")
    _route_time(at, "visit_date")
    p = _partition(_prepare(at))

    assert p["cv_strategy"] in ("group", "standard")
    assert p["cv_strategy"] != "time", "the time branch outranked grouping"


def test_target_rows_with_missing_values_are_excluded(train_page):
    at = train_page
    df = build_test_dataframe(n=200)
    df.loc[df.index[:25], "glucose"] = np.nan
    inject_data_state(at, df, target_col="glucose", task_type="regression")
    _inject_pipeline(at, df)
    p = _partition(_prepare(at))

    assert p["n_train"] + p["n_val"] + p["n_test"] == 175
