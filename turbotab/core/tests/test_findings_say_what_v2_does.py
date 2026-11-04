"""Every served ``why_it_matters`` names only stages, views and diagnostics v2 has (pp2-why-true,
W1). The legacy reasons the findings stage serves were written for Classic, and some named what v2
does not have:

* the two-level reason placed a column's blanks "at Preprocess" (``ml/binary_text.py``); v2 has
  no Preprocess stage (``quest.STAGES``) and asks it in Who's in, at the missing-values question,
  whose "Blanks as their own level" keeps a blank as a value of a two-valued column
  (``models.pipeline.level_columns``);
* the metabolomics acquisition inventory and the no-run-order finding named a run-order overlay
  on a PCA scores plot (``turbotab/packs.py``); v2 draws no such plot (its one PCA figure is the
  batch figure, colored by batch: ``methods.batch.batch_figure``). What v2 computes against an
  injection order is the drift reading (``detectors.assay.drift_reading``) and QC-RLSC
  (``methods.qc_drift``, which reads ``readings.injection_order_reading``), so "adding the column
  recovers all three" became the two that read it;
* the compositional reason said a collinearity figure "is drawn on log-ratios" and the parts "are
  not offered as independent predictors": v2 has no log-ratio figure, and the parts are offered
  (its lever is the roles question's "Leave one part out");
* the counts reason said "regularized models rank first and distance-based ones last": the shelf
  has no distance-based family (``turbotab.core.models``), and under Estimate the feature-wise
  tests rank first. What holds under either purpose, or none yet, is that on every outcome the
  shelf has a regularized family for, every unregularized fit ranks below the regularized ones
  (``models.base.rank``); a time to event has none, and the Cox model, unregularized, ranks alone;
* the repeated-subjects reason said the grain question was one "the app has already asked", with
  "the answer you gave there": the finding is placed in Your data, at the roles question, and v2
  asks the grain question later, in Who's in (``quest.QUESTIONS``);
* the boolean-as-text reason said the column "is neither a number nor a label" and quoted the
  error ("Unknown label type") modeling, correlation and feature selection raise refusing it
  (``ml/import_doctor.py``): v2's ingest reads True/False with blanks as a yes/no column and its
  design reads that predictor as a number, so none of the reason is served;
* the temporal-plausibility reason said the app "has never done" temporal plausibility, on the
  finding that does it (``turbotab/clinical.py``).

Each truth is by construction (a table built here, or a shipped sample table whose companion says
what it holds), by numpy (least squares at p ≥ n has no unique solution), or by the code's own
structure (the stage that holds the missing-values question, the readings that need an order,
the stages the grain question and a finding are placed in, the families each outcome's shelf
registers). The true sentences of each reason stay as they were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.stages.finding_words import family
from turbotab.core.stages.findings import findings_stage, structural_finding
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

# What v2 does not have, as the legacy reasons named it. None may be served.
UNTRUE = ("Preprocess", "PCA scores", "PCA overlay", "overlay", "recovers all three",
          "distance-based", "log-ratios", "independent predictors", "Unknown label type",
          "refuse it", "neither a number nor a label", "has never done", "already asked",
          "answer you gave", "re-asked")


def _artifact(tmp: Path, source: Path, lens: list[str], target: str | None = None) -> dict:
    table = Ingested(source, tmp)
    return table.run(findings_stage, ProjectState(lens=lens, target=target))


def _served(artifact: dict, fam: str) -> list[dict]:
    return [f for f in artifact["findings"] if family(f["id"]) == fam]


def _one(artifact: dict, fam: str) -> dict:
    found = _served(artifact, fam)
    assert len(found) == 1, (fam, [f["id"] for f in artifact["findings"]])
    return found[0]


N, N_BLANK = 120, 12


def two_level_table() -> pd.DataFrame:
    """120 rows: `sex` M/F with 12 blanks (rows 0–11), `smoker` current/never with none, a
    one-level `screened` (True on 48 rows, blank on the rest) and two numbers."""
    rng = np.random.default_rng(7)
    sex = rng.choice(["F", "M"], N).astype(object)
    sex[:N_BLANK] = None
    screened = np.where(np.arange(N) % 5 < 2, "True", None)
    return pd.DataFrame({"age": rng.integers(20, 80, N), "sbp": rng.normal(125, 15, N).round(1),
                         "sex": sex, "smoker": rng.choice(["current", "never"], N),
                         "screened": screened})


@pytest.fixture(scope="module")
def two_level(tmp_path_factory) -> tuple[pd.DataFrame, dict, dict[str, str], Ingested]:
    """The construction, its served findings, the legacy reasons as the legacy engine writes them
    (``turbotab.engine.diagnose``, the stream the findings stage speaks for), and the ingest."""
    from turbotab import engine

    tmp = tmp_path_factory.mktemp("two_level")
    frame = two_level_table()
    path = tmp / "two_level.csv"
    frame.to_csv(path, index=False)
    read = Ingested(path, tmp / "ingested")
    with read.store() as store:
        written = store.materialize().reset_index(drop=True)
    legacy = {f["id"]: f["why_it_matters"] for f in
              (structural_finding(engine.shape_finding_to_dict(s))
               for s in engine.diagnose(written, None))}
    return frame, read.run(findings_stage, ProjectState(lens=[])), legacy, read


def test_the_blanks_are_asked_where_v2_asks_them(two_level):
    """`sex` has 12 blanks by construction. The reason places their question where v2 asks it:
    the stage that holds the missing-values question, by that question's name; and the stage it
    names is one of v2's (Preprocess is not)."""
    from turbotab.core.quest import QUESTIONS, STAGE_NAMES
    from turbotab.core.voice import question_name

    frame, artifact, legacy, _ = two_level
    assert int(frame["sex"].isna().sum()) == N_BLANK
    why = {f["id"]: f for f in artifact["findings"]}["binary_text__sex"]["why_it_matters"]
    stage = STAGE_NAMES[QUESTIONS["missing"].stage]
    assert stage == "Who's in" and "Preprocess" not in STAGE_NAMES.values()
    clause = (f"The {N_BLANK} blank(s) are a separate question, asked in {stage} by "
              f"{question_name('missing')}: whether not-answered is itself informative.")
    assert why.endswith(" " + clause), why
    assert why.count("blank(s)") == 1 and "Preprocess" not in why
    # The two-level reading's own sentences are as the legacy reason writes them.
    before = legacy["binary_text__sex"]
    assert "asked at Preprocess" in before
    assert why[: -len(clause)] == before[: before.index("The 12 blank(s)")]


def test_a_two_level_column_with_no_blanks_keeps_its_reason_whole(two_level):
    """The guard's silent side: `smoker` has no blank, so its reason has no blanks clause and is
    the legacy reason, word for word."""
    frame, artifact, legacy, _ = two_level
    assert int(frame["smoker"].isna().sum()) == 0
    why = {f["id"]: f for f in artifact["findings"]}["binary_text__smoker"]["why_it_matters"]
    assert why == legacy["binary_text__smoker"]
    assert "blank" not in why


def test_text_booleans_serve_no_reason_v2_does_not_bear_out(two_level):
    """`screened` is True on 48 rows and blank on 72 by construction: text booleans with one level,
    which the two-level reading does not claim. The legacy reason said the column "is neither a
    number nor a label" and quoted an error from modeling, correlation and feature selection
    refusing it. v2 reads it as a yes/no column at the ingest, and its design reads it as a number,
    1 where the construction wrote True and blank where it left a blank, so no sentence of that
    reason is true of v2 and none is served."""
    from turbotab.core.models.pipeline import design_spec, modeling_frame

    frame, artifact, legacy, read = two_level
    true_rows = np.arange(N) % 5 < 2
    assert (frame["screened"] == "True").tolist() == true_rows.tolist() and true_rows.sum() == 48
    info = {c["name"]: c for c in read.info["columns"]}
    assert info["screened"]["dtype"] == "boolean"
    predictors = ["age", "sbp", "smoker", "screened"]
    with read.store() as store:
        X = modeling_frame(store, predictors, np.arange(N))
    np.testing.assert_array_equal(X["screened"].to_numpy(dtype=float),
                                  np.where(true_rows, 1.0, np.nan))
    spec = design_spec(ProjectState(lens=[], target="sex"), X, predictors, column_info=info)
    assert "screened" in spec.numeric and "screened" not in spec.categorical
    served = _one(artifact, "boolean_as_text")
    assert served["affected_columns"] == ["screened"]
    before = legacy["boolean_as_text__screened"]
    assert before.startswith("Stored this way the column is neither a number nor a label.")
    assert "Unknown label type" in before
    assert served["why_it_matters"] is None


# ── metabolomics: what an injection order is read by ─────────────────────────


@pytest.fixture(scope="module")
def no_order(tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    """`metabolomics_no_qc.csv`: an untargeted panel with no run order, no timestamp and no
    column that numbers the rows (its companion, ``metabolomics_no_qc.md``)."""
    source = SAMPLES / "metabolomics_no_qc.csv"
    return pd.read_csv(source), _artifact(tmp_path_factory.mktemp("no_order"), source,
                                          ["metabolomics"])


def test_no_run_order_names_what_reads_an_order(no_order):
    frame, artifact = no_order
    why = _one(artifact, "pack::metabolomics::no_run_order")["why_it_matters"]
    assert why == (
        "Instrument drift is often the largest single variance component in a run, and every way "
        "of seeing it is ordered by injection: drift diagnostics and QC-RLSC correction both "
        "become impossible without one. The order is in the instrument's sequence file — if you "
        f"still have it, adding it as a run-order column that runs 1 to {len(frame):,} lets both "
        "read it.")


def test_the_two_readings_need_an_order_and_read_one_that_runs_1_to_n(no_order):
    """The sentence's claim, both ways: on the table as shipped neither the drift reading nor
    QC-RLSC's order reading finds an order; with a column numbering the rows 1 to n added, both
    read it."""
    from turbotab.core.detectors.assay import drift_reading
    from turbotab.core.readings import injection_order_reading

    frame, _ = no_order
    assert drift_reading(frame) is None and injection_order_reading(frame) is None
    ordered = frame.assign(run_order=np.arange(1, len(frame) + 1))
    assert drift_reading(ordered)["column"] == "run_order"
    assert injection_order_reading(ordered).value == "run_order"


@pytest.fixture(scope="module")
def untargeted(tmp_path_factory) -> dict:
    """`metabolomics_untargeted.csv`: a run order, a batch and pooled QCs (its companion)."""
    return _artifact(tmp_path_factory.mktemp("untargeted"),
                     SAMPLES / "metabolomics_untargeted.csv", ["metabolomics"])


def test_the_acquisition_inventory_names_what_v2_computes(untargeted):
    why = _one(untargeted, "pack::metabolomics::acquisition_design")["why_it_matters"]
    assert why == (
        "Run order, batch and plate are what drift correction and batch correction are computed "
        "against, and the group-by-batch crosstab is what shows a confound before it becomes a "
        "result. Nothing here is changed — this is the inventory the manuscript's "
        "data-description sentence is written from.")


@pytest.fixture(scope="module")
def paired(tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    """`metabolomics_paired_logged.csv`: 72 samples from 36 subjects, two each, by `subject_id`
    (its companion)."""
    source = SAMPLES / "metabolomics_paired_logged.csv"
    return pd.read_csv(source), _artifact(tmp_path_factory.mktemp("paired"), source,
                                          ["metabolomics"])


def test_the_repeated_subjects_reason_says_where_v2_asks_the_grain(paired):
    """`subject_id` repeats by construction, so the finding is served; it is placed in Your data,
    at the roles question it routes to, and the grain question is asked later, in Who's in. The
    reason names that stage and question as v2 names them, says nothing was asked before, and
    keeps the legacy reason's true sentences word for word."""
    from turbotab import packs
    from turbotab.core.quest import QUESTIONS, STAGE_INDEX, STAGE_NAMES, finding_place
    from turbotab.core.voice import question_name

    frame, artifact = paired
    subjects = frame["subject_id"].dropna()
    assert (subjects.nunique(), len(subjects)) == (36, 72)
    served = _one(artifact, "pack::metabolomics::repeated_subjects")
    place, question = finding_place(served)
    grain = QUESTIONS["grain"]
    assert (place.stage, question) == ("data", "roles")
    assert STAGE_INDEX[place.stage] < STAGE_INDEX[grain.stage]
    stage = STAGE_NAMES[grain.stage]
    assert stage == "Who's in"
    kept = ("It decides how the held-out set is drawn, because a subject appearing in both halves "
            "makes the held-out estimate optimistic.",
            "this is the same question seen from the assay side, and the reason it matters more "
            "in an assay is that the technical replicate and the repeat visit look identical in "
            "the table.")
    legacy = packs._repeated_subjects(frame)["why_it_matters"]
    assert legacy.startswith("This is the question the app has already asked")
    assert all(sentence in legacy for sentence in kept)
    assert served["why_it_matters"] == (
        f"Whether one person can appear in more than one row is asked in {stage}, by "
        f"{question_name('grain')}, and the answer you give there is the one that governs. "
        f"{kept[0]} That question is not asked here; {kept[1]}")


# ── dietary, genomics and clinical ───────────────────────────────────────────


@pytest.fixture(scope="module")
def recalls(tmp_path_factory) -> dict:
    """`dietary_recalls.csv`: four energy shares that sum to 100 (its companion)."""
    return _artifact(tmp_path_factory.mktemp("recalls"), SAMPLES / "dietary_recalls.csv",
                     ["dietary"])


def test_the_compositional_reason_names_no_figure(recalls):
    why = _one(recalls, "pack::dietary::compositional")["why_it_matters"]
    assert why == (
        "These are compositional — parts of a whole — and ordinary correlation between them is not "
        "interpretable: raising one necessarily lowers another, so the negative correlation is "
        "arithmetic rather than dietary.")


@pytest.fixture(scope="module")
def expression(tmp_path_factory) -> dict:
    """`genomics_expression.csv`: raw counts, many more genes than samples (its companion)."""
    return _artifact(tmp_path_factory.mktemp("expression"), SAMPLES / "genomics_expression.csv",
                     ["genomics"])


def test_the_counts_reason_ranks_only_families_the_shelf_has(expression):
    why = _one(expression, "pack::genomics::counts_p_over_n")["why_it_matters"]
    assert why.startswith(
        "Two consequences, and only one of them is ours to settle. Model ranking is: at this p/n "
        "an unregularized fit is not merely optimistic but degenerate, so it ranks below the "
        "regularized ones — ordered, never filtered. For a time to event there are no regularized "
        "ones: the Cox model, unregularized, ranks alone. Normalization is NOT: "), why


UNREGULARIZED = {"linear", "huber", "proportional_odds", "mixed", "gee", "cox"}
REGULARIZED = {"elastic_net", "ridge"}


def _penalized(family, task: str) -> bool:
    """Whether the estimator ``family`` builds for ``task`` penalizes its coefficients: a positive
    ``alpha`` (scikit-learn's ridge and elastic net), or a finite ``C`` (its logistic regression,
    where C is the inverse of the penalty's strength and only C = inf fits without one)."""
    built = family.build(task, "prediction", 30, 60)
    params = (built[-1] if hasattr(built, "steps") else built).get_params(deep=False)
    return params.get("alpha", 0) > 0 or bool(np.isfinite(params.get("C", np.inf)))


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass", "ordinal",
                                  "time_to_event"])
@pytest.mark.parametrize("purpose", ["inference", "prediction", None])
def test_at_p_over_n_every_unregularized_fit_ranks_below_the_regularized(task, purpose):
    """What the counts reason says, on the shelf: 60 predictors for 30 rows (p/n 2, the reason's
    floor), under either purpose or none yet (the finding is served before the goal is chosen).
    Least squares there has no unique solution (numpy: a 30 × 61 design has rank at most 30). On
    every outcome but a time to event the shelf registers a regularized family (ridge's and the
    elastic net's estimators penalize their coefficients), and every unregularized family ranks
    below every regularized one; a time to event registers the Cox model alone, whose estimator
    takes no penalty, and it ranks alone. Nothing is filtered, and the shelf has no distance-based
    family."""
    from turbotab.core.models import Situation, families, get_family, rank  # registers them
    from turbotab.core.models.base import TASKS

    assert set(TASKS) == {"regression", "binary", "multiclass", "ordinal", "time_to_event"}
    rng = np.random.default_rng(1)
    design = np.column_stack([np.ones(30), rng.poisson(20, (30, 60))])
    assert np.linalg.matrix_rank(design) <= 30 < design.shape[1]
    keys = {f.key for f in families()}
    assert REGULARIZED <= keys and not keys & {"knn", "k_nearest", "svm", "kernel"}
    shelf = {f.key for f in families(task)}
    situation = Situation(task=task, purpose=purpose, n_rows=30, n_features=60,
                          n_events=12 if task in ("binary", "time_to_event") else None,
                          lenses=("genomics",))
    order = [f.key for f, _ in rank(situation)]
    assert set(order) == shelf and len(order) == len(shelf)
    if task == "time_to_event":
        assert order == ["cox"] and not shelf & REGULARIZED
        assert get_family("cox").build(task, purpose, 30, 60).get_params() == {}
        return
    assert shelf & REGULARIZED and all(_penalized(get_family(k), task) for k in shelf & REGULARIZED)
    below = [order.index(k) for k in UNREGULARIZED if k in order]
    above = [order.index(k) for k in REGULARIZED if k in order]
    assert below and above and max(above) < min(below), order


@pytest.fixture(scope="module")
def labs(tmp_path_factory) -> dict:
    """`clinical_labs.csv`: repeated heights that cannot belong to one adult (its companion)."""
    return _artifact(tmp_path_factory.mktemp("labs"), SAMPLES / "clinical_labs.csv", ["clinical"])


def test_the_temporal_reason_does_not_deny_itself(labs):
    why = _one(labs, "pack::clinical::temporal_implausibility")["why_it_matters"]
    assert why == (
        "Kahn et al.'s framework separates ATEMPORAL plausibility — is this value believable — "
        "from TEMPORAL plausibility — is this trajectory believable. Each value flagged here is "
        "individually plausible and the sequence is not, so no per-value check can find them: a "
        "height of 160 cm and a height of 169 cm are both ordinary, and they cannot both belong "
        "to the same adult.")


# ── every reason served on these tables ──────────────────────────────────────


@pytest.mark.parametrize("name", ["two_level", "no_order", "untargeted", "paired", "recalls",
                                  "expression", "labs"])
def test_no_served_reason_names_what_v2_does_not_have(name, request):
    found = request.getfixturevalue(name)
    artifact = found[1] if isinstance(found, tuple) else found
    problems = [(f["id"], word) for f in artifact["findings"] for word in UNTRUE
                if word in (f.get("why_it_matters") or "")]
    assert not problems, problems
