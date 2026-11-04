"""Every served ``why_it_matters`` says only what v2 does, under either goal and before one is
chosen, in every lens, with and without a target (pp2-why-audit, WA1).

The findings stage reads no purpose (its reads: ``stages.build_graph``), so one reason is served
whichever goal the researcher holds or will choose; a sentence whose truth depends on the goal says
the condition itself. The re-scan of every family's served reason found these false or goal-blind:

* numbers stored as text: the app's own reason (``core/repairs.py``) and the legacy one
  (``ml/import_doctor.py``) said the column "cannot be modeled" and sits out of the analysis; v2's
  design reads a text predictor as a category, each distinct value a level of its own
  (``models.pipeline.category_inputs``), until a repair reads it as numbers. The clinical pack's
  number-format reason said such a column "silently" becomes one-hot levels, beside the app's own
  finding about the same column;
* the repeated-subjects reason said the answer "decides how the held-out set is drawn", and the
  pooled-QC reason that "a held-out set could contain them", removed "before the held-out rows are
  drawn": under Estimate no holdout is offered first (``seal.split_offer``), and under Predict a
  small table leads with resampling too, while every fold keeps a unit whole;
* the redundancy reason named "the data-description sentence", which v2 does not write;
* the counts reason said an unregularized fit ranks below the regularized ones, where under
  Estimate the feature-wise tests, unregularized and one feature at a time, rank first
  (``models.base.rank``);
* the zeros reason said the app "needs to be told which export wrote this file" and the
  already-transformed reason that it "needs to know ... by which tool": v2 asks neither;
* the empty-blocks reason said a constant feature survives "every filter that is written in terms
  of variance ratios": v2's D-ratio filter drops one (``methods.qc_drift.DRatioFilter``);
* the clinical mixed-units and default-value reasons said the artifact passes "every distributional
  check the app has", each beside the app's own distributional check that found it;
* the temporal reason's 169 cm was typed beside the jump limit it illustrates
  (``clinical.HEIGHT_JUMP_CM``);
* the survey ordinal reason said the encoding "is applied now", where nothing is applied at the
  finding;
* the energy-unit reason said total energy is "the denominator of every energy adjustment" and that
  a unit error "propagates into every result": a residual adjustment is unchanged by a unit's
  constant factor (least squares, numpy);
* the plausibility reason called the unusual values "the case mix the model exists to learn", and
  the two-level outcome's reason asked "which one you are predicting": goal-blind under Estimate.

Each truth is by construction (a table built here, or a shipped sample whose companion says what it
holds), by numpy or scikit-learn, or by the code's own structure; the true sentences of each reason
stay as they were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.stages.finding_words import family
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

# What the served reasons said that v2 does not bear out. None may be served.
UNTRUE = ("cannot be modeled", "sit out of", "held-out set is drawn",
          "like any other, and a held-out set", "before the held-out rows are drawn", "enters the data-description",
          "which export wrote", "by which tool", "survive every filter",
          "every distributional check", "169 cm", "applied now", "denominator of every energy",
          "propagates into every result", "exists to learn", "which one you are predicting",
          "silently becomes", "an unregularized fit is not")


def _run(tmp: Path, source: Path, lens: list[str], target: str | None = None) -> tuple[Ingested, dict]:
    read = Ingested(source, tmp)
    return read, read.run(findings_stage, ProjectState(lens=lens, target=target))


def _one(artifact: dict, fam: str) -> dict:
    found = [f for f in artifact["findings"] if family(f["id"]) == fam]
    assert len(found) == 1, (fam, [f["id"] for f in artifact["findings"]])
    return found[0]


# ── numbers stored as text ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def visits(tmp_path_factory) -> tuple[Ingested, dict, dict]:
    """`clinic_visits.csv`: `income` written with thousands separators (`110,000`), `weight` with
    its unit (`107 kg`), and a two-level `outcome` (its companion), read under the clinical lens,
    before a target is chosen and with `outcome` as the target."""
    read, before = _run(tmp_path_factory.mktemp("visits"), SAMPLES / "clinic_visits.csv",
                        ["clinical"])
    return read, before, read.run(findings_stage, ProjectState(lens=["clinical"], target="outcome"))


TEXT_NUMBERS_WHY = ("As a predictor left as text, it enters the models as a category, each distinct "
                    "value a level of its own, rather than as a number, until it is read as "
                    "numbers.")
NUMERIC_AS_TEXT_WHY = ("As a predictor left as text, it enters the models as a category, each "
                       "distinct value a level of its own, rather than as a number.")


def test_numbers_stored_as_text_are_said_to_enter_as_a_category(visits):
    """The served reasons, before a target and with one: `income` (the app's own text-number
    finding) and `weight` (the legacy numeric-as-text finding, units the text-number reading does
    not read) enter the models as categories, as the next test shows."""
    _, before, after = visits
    for artifact in (before, after):
        assert _one(artifact, "text_numbers")["why_it_matters"] == TEXT_NUMBERS_WHY
        assert _one(artifact, "numeric_as_text")["why_it_matters"] == NUMERIC_AS_TEXT_WHY
        assert _one(artifact, "pack::clinical::number_format")["why_it_matters"] == (
            "A predictor that does not parse as numeric enters the models as a category, so a "
            "continuous measurement left as text becomes hundreds of one-hot levels. The "
            "decimal-comma case is worse than that: it can parse successfully and be wrong by a "
            "factor of a hundred, which no error anywhere would report.")


def test_both_legacy_spellings_of_the_numbers_as_text_reason_are_restated():
    """The legacy reader writes the reason two ways (``ml/import_doctor.py``: every value a plain
    number, or most of them once units and commas are stripped); each is served as the v2 reason.
    The clause stays silent on a reason it does not name: the same words under another family are
    served as written."""
    from turbotab.core.stages.finding_words import FindingContext, speak

    fc = FindingContext(frame=pd.DataFrame({"dose": ["5 mg", "10 mg"]}))
    spellings = ("As text this column cannot be modeled, correlated or plotted — it will silently "
                 "sit out of your analysis while appearing perfectly fine.",
                 "As text this column cannot be modeled, correlated, or plotted — it will silently "
                 "sit out of your analysis.")
    for why in spellings:
        finding = {"id": "numeric_as_text__dose", "title": "x", "detail": "x",
                   "why_it_matters": why, "affected_columns": ["dose"]}
        assert speak(finding, None, fc)["why_it_matters"] == NUMERIC_AS_TEXT_WHY
    other = {"id": "category_variants__dose", "title": "x", "detail": "x",
             "why_it_matters": spellings[1], "affected_columns": ["dose"]}
    assert speak(other, None, fc)["why_it_matters"] == spellings[1]


def test_a_text_predictor_is_a_category_until_its_repair_reads_it_as_numbers(visits):
    """What the reasons say, on the design every family shares: each text column is a category
    (``category_inputs``) whose one-hot step, in full coding, makes one column per distinct value,
    counted by pandas; once the text-number repair the finding offers is applied, `income` is read
    as numbers (110,000 as 110000) and is no category, while `weight`, which no repair reads, stays
    one. Neither sits out of the model."""
    from sklearn.pipeline import Pipeline

    from turbotab.core import repairs
    from turbotab.core.models.pipeline import (category_inputs, design_spec, modeling_frame,
                                               shared_steps)

    read, _, after = visits
    predictors = ["income", "weight", "age"]
    state = ProjectState(lens=["clinical"], target="outcome")
    info = {c["name"]: c for c in read.info["columns"]}
    with read.store() as store:
        X = modeling_frame(store, predictors, np.arange(read.info["n_rows"]))
    assert X["income"].dtype == object and X["weight"].dtype == object
    assert category_inputs(state, X, predictors) == ["income", "weight"]
    spec = design_spec(state, X, predictors, column_info=info)
    coded = Pipeline(shared_steps(spec, onehot_drop=None)).set_output(transform="pandas").fit_transform(X)
    for c in ("income", "weight"):
        assert sum(str(k).startswith(f"{c}_") for k in coded.columns) == X[c].nunique()
    option = next(o for o in _one(after, "text_numbers")["repairs"] if o["key"] == "thousands")
    d = option["decision"]
    repaired = ProjectState(lens=["clinical"], target="outcome",
                            numbers_read={d["finding_id"]: {"option": d["option"],
                                                            "params": d["params"]}})
    frame = read.frame()
    numbers = repairs.numbers_read(frame, repaired)
    assert frame["income"].iloc[0] == "110,000" and numbers["income"].iloc[0] == 110000.0
    assert category_inputs(repaired, numbers, predictors) == ["weight"]


def test_the_two_level_outcome_is_asked_which_level_is_the_event(visits):
    """With `outcome` as the target its two levels are asked as the event (never "predicted",
    which is one goal's word); the rest of the legacy reason stands as written."""
    _, _, after = visits
    why = _one(after, "positive_class")["why_it_matters"]
    assert why == (
        "Which of these is the event decides the sign of every effect estimate, what sensitivity "
        "and specificity are measuring, and what the model is trained to detect. Nothing in the "
        "file says which one is the event. By convention 'responder' is usually the event, and it "
        "is the smaller group here — but convention is a suggestion, not an answer.")


# ── how rows are split: under either goal ────────────────────────────────────


@pytest.fixture(scope="module")
def paired(tmp_path_factory) -> tuple[Ingested, dict]:
    """`metabolomics_paired_logged.csv`: 36 subjects sampled twice by `subject_id`, pooled QCs,
    and values already logged (its companion)."""
    return _run(tmp_path_factory.mktemp("paired"), SAMPLES / "metabolomics_paired_logged.csv",
                ["metabolomics"])


def test_the_repeated_subjects_reason_holds_under_either_goal(paired):
    from turbotab.core.quest import QUESTIONS, STAGE_NAMES
    from turbotab.core.voice import question_name

    _, artifact = paired
    assert _one(artifact, "pack::metabolomics::repeated_subjects")["why_it_matters"] == (
        f"Whether one person can appear in more than one row is asked in "
        f"{STAGE_NAMES[QUESTIONS['grain'].stage]}, by {question_name('grain')}, and the answer you "
        f"give there is the one that governs. It decides how the rows are split for scoring: the "
        f"cross-validation folds, and the held-out set where one is drawn, keep each subject's rows "
        f"on one side, because a subject appearing on both sides makes the score optimistic. That "
        f"question is not asked here; this is the same question seen from the assay side, and the "
        f"reason it matters more in an assay is that the technical replicate and the repeat visit "
        f"look identical in the table.")


def test_the_grain_answer_keeps_each_subject_in_one_fold(paired):
    """The sentence's claim, both ways: with the grain answered as repeated by `subject_id`, the
    seal groups by it and every subject's two samples fall in one fold; folded by row instead, some
    subject's samples land in two."""
    from turbotab.core.decisions import GrainSpec
    from turbotab.core.models.folds import kfold_assignment
    from turbotab.core.seal import seal_inputs

    read, _ = paired
    frame = read.frame()
    state = ProjectState(lens=["metabolomics"], target="responder",
                         grain=GrainSpec(grain="repeated", id_column="subject_id"))
    with read.store() as store:
        draw = seal_inputs(state, np.arange(len(frame)), store, "binary", holdout=0.2, seed=0)
    assert draw.grouped_by == "subject_id"
    subjects = frame["subject_id"]
    named = subjects.notna().to_numpy()
    assert subjects[named].value_counts().eq(2).all() and subjects[named].nunique() == 36

    def spread(groups) -> int:
        fold = kfold_assignment(len(frame), groups=groups, folds=5, seed=0)
        return int(pd.Series(fold[named]).groupby(subjects[named].to_numpy()).nunique().max())

    assert spread(draw.groups) == 1
    assert spread(None) == 2


@pytest.mark.parametrize("n", [100, 2_000, 20_000])
def test_a_held_out_set_is_drawn_only_where_the_split_holds_one_out(n):
    """Why the reasons say "where one is drawn": under Estimate the split offers no holdout first
    at every size (Shmueli 2010, ``seal.INFERENCE_SPLIT_REASON``) and leads with one
    cross-validation run; under Predict a large table leads with a holdout and a small one with
    resampling. Folds are drawn under both."""
    from turbotab.core.seal import split_offer

    estimate = split_offer("inference", "binary", n, [n // 2, n - n // 2])
    predict = split_offer("prediction", "binary", n, [n // 2, n - n // 2])
    assert estimate.options[0].holdout == 0 and estimate.order[0] == "kfold"
    assert "kfold" in predict.order
    assert (predict.options[0].holdout > 0) == (n == 20_000)


@pytest.fixture(scope="module")
def merged(tmp_path_factory) -> tuple[Ingested, dict]:
    """`metabolomics_merged_modes.csv`: pooled QCs, exact zeros and an empty block (its
    companion)."""
    return _run(tmp_path_factory.mktemp("merged"), SAMPLES / "metabolomics_merged_modes.csv",
                ["metabolomics"])


def test_the_pooled_qc_reason_holds_under_either_goal(merged):
    """The reference rows leave in the working table (``reference_rows.reference_filter``), a
    stage the split stage, which draws the folds and any held-out rows, reads."""
    from turbotab.core.stages import build_graph

    _, artifact = merged
    assert _one(artifact, "pack::metabolomics::pooled_qc")["why_it_matters"] == (
        "They are not participants: modeling them is an error with no legitimate reading. Until "
        "they are excluded they are analyzed as rows like any other, and a cross-validation fold "
        "or a held-out set could contain them. Excluding them here removes them before the rows "
        "are split and counts them in the participant flow; they stay in the file for quality "
        "assessment.")
    graph = build_graph()
    assert "working" in graph.upstream("split") and "working" in graph.upstream("seal_plan")


# ── what the app asks, writes and filters ────────────────────────────────────


def test_zeros_are_non_detections_only_once_you_say_so(merged):
    """The zeros reason says nothing reads a zero as a non-detection unless the researcher says
    so: the missing-values methods' censored columns hold the zeros' columns only once the repair
    the finding offers ("Zeros mean not detected") is applied."""
    from turbotab.core.decisions import FindingDisposition
    from turbotab.core.methods.missing import censored_columns

    _, artifact = merged
    zeros = _one(artifact, "pack::metabolomics::zeros_or_missing")
    assert zeros["why_it_matters"] == (
        "A zero that means \"below the detection limit\" and a zero that means \"measured as "
        "none\" want opposite treatments — the first is imputed from the detection limit, the "
        "second is a real value — and defaulting wrong corrupts every step after it. Nothing here "
        "has defaulted: no zero is read as a non-detection unless you say so.")
    option = next(o for o in zeros["repairs"] if o["key"] == "nondetect")
    assert option["label"] == "Zeros mean not detected"
    columns = option["decision"]["params"]["columns"]
    found = {"findings": [{"id": zeros["id"], "affected_columns": zeros["affected_columns"]}]}
    assert not set(columns) & set(censored_columns(found, ProjectState(lens=["metabolomics"])))
    applied = ProjectState(lens=["metabolomics"], findings={zeros["id"]: FindingDisposition(
        action="applied", option="nondetect", params={"columns": columns})})
    assert set(columns) <= set(censored_columns(found, applied))


def test_the_transformed_reason_asks_for_nothing_the_app_does_not_ask(paired):
    """The legacy reason said the app "needs to know" what was done and by which tool: the finding
    offers no repair and routes to no question, so its summary says it has no control."""
    from turbotab.core.stages.finding_words import NO_LEVER

    _, artifact = paired
    found = _one(artifact, "pack::metabolomics::already_transformed")
    assert found["why_it_matters"] == (
        "A second log transform on already-logged data is a silent catastrophe: it produces "
        "numbers, every plot still renders, and the fold changes underneath are wrong by an amount "
        "nothing in the output reveals. Negative values also break the log outright. Before "
        "anything is transformed here, what was already done to this table needs to be known.")
    assert not found.get("repairs") and found["routes_to"] is None
    assert found["summary"].endswith(NO_LEVER)


def test_the_empty_blocks_reason_claims_no_filter_lets_them_through(merged):
    _, artifact = merged
    assert _one(artifact, "pack::metabolomics::empty_blocks")["why_it_matters"] == (
        "A feature that is zero everywhere and a feature that holds one value everywhere cannot "
        "separate anything. A sample that is empty across every feature is a failed injection or a "
        "merge that matched nothing, and it is worth knowing which before it becomes a row in a "
        "model.")


def test_the_d_ratio_filter_drops_a_feature_of_one_value():
    """Why "both survive every filter written in terms of variance ratios" is not served: v2's
    in-fold D-ratio filter drops a feature that holds one value on every row (its ratio, 0 / 0, is
    not below the limit), and keeps a feature whose pooled QCs vary far less than its samples."""
    from turbotab.core.methods.qc_drift import D_RATIO_MAX, DRatioFilter, qc_sd

    rng = np.random.default_rng(3)
    n = 40
    qc = np.zeros(n, dtype=bool)
    qc[:8] = True
    varied = rng.lognormal(3, 0.5, n)
    varied[qc] = 20 + rng.normal(0, 0.2, 8)
    X = pd.DataFrame({"flat": np.full(n, 5.0), "varied": varied})
    assert np.var(X["flat"]) == 0.0
    sds = {c: float(s) for c, s in zip(X.columns, qc_sd(X.to_numpy(), qc)) if np.isfinite(s)}
    fitted = DRatioFilter(columns=list(X.columns), qc_sd=sds).fit(X[~qc])
    assert fitted.dropped_ == ["flat"]
    assert fitted.ratio_["varied"] < D_RATIO_MAX


@pytest.fixture(scope="module")
def redundant(tmp_path_factory) -> dict:
    """`metabolomics_redundant.csv`: features that are one compound's several ions (its
    companion). Its redundancy reading alone takes about 40 s, so the tests that read it are
    slow."""
    return _run(tmp_path_factory.mktemp("redundant"), SAMPLES / "metabolomics_redundant.csv",
                ["metabolomics"])[1]


@pytest.mark.slow
def test_the_redundancy_reason_names_the_manuscripts_sentence(redundant):
    assert _one(redundant, "pack::metabolomics::redundancy")["why_it_matters"] == (
        "Untargeted features are not independent: one compound leaves the source as several ions "
        "and isotopologues. The count enters the manuscript's data-description sentence and the "
        "denominator of a multiple-testing correction. Nothing is collapsed here.")


# ── the counts reason under Estimate ─────────────────────────────────────────


@pytest.fixture(scope="module")
def expression(tmp_path_factory) -> dict:
    """`genomics_expression.csv`: raw counts, many more genes than samples, with `condition` as the
    target (its companion)."""
    return _run(tmp_path_factory.mktemp("expression"), SAMPLES / "genomics_expression.csv",
                ["genomics"], "condition")[1]


def test_the_counts_reason_ranks_fits_of_all_the_features_at_once(expression):
    why = _one(expression, "pack::genomics::counts_p_over_n")["why_it_matters"]
    assert why.startswith(
        "Two consequences, and only one of them is ours to settle. Model ranking is: at this p/n "
        "an unregularized fit of all the features at once is not merely optimistic but "
        "degenerate, so it ranks below the regularized ones — ordered, never filtered. For a time "
        "to event there are no regularized ones: the Cox model, unregularized, ranks alone. "), why


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_under_estimate_the_feature_wise_tests_rank_first(task):
    """Why the reason says "of all the features at once": at p/n 2 under Estimate the feature-wise
    tests, which fit one exposure at a time with no penalty and make no predictions, rank above
    the regularized families, and every unregularized family that fits every feature in one model
    ranks below them; under Predict the feature-wise tests rank last."""
    from turbotab.core.models import Situation, get_family, rank

    def order(purpose: str) -> list[str]:
        return [f.key for f, _ in rank(Situation(
            task=task, purpose=purpose, n_rows=30, n_features=60,
            n_events=12 if task == "binary" else None, lenses=("genomics",)))]

    estimate, predict = order("inference"), order("prediction")
    assert get_family("featurewise").predicts is False
    assert estimate[0] == "featurewise" and predict[-1] == "featurewise"
    regularized = [estimate.index(k) for k in ("elastic_net", "ridge")]
    at_once = [estimate.index(k) for k in ("linear", "huber", "mixed", "gee") if k in estimate]
    assert at_once and max(regularized) < min(at_once)


# ── the clinical reasons ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def labs(tmp_path_factory) -> dict:
    """`clinical_labs.csv`: glucose in two units, a pile-up of blood pressures at a default, and
    heights that jump between visits (its companion), with `readmitted` as the target."""
    return _run(tmp_path_factory.mktemp("labs"), SAMPLES / "clinical_labs.csv", ["clinical"],
                "readmitted")[1]


def test_the_clinical_reasons_say_only_what_the_app_checks(labs):
    from turbotab.clinical import HEIGHT_JUMP_CM

    assert _one(labs, "pack::clinical::mixed_units")["why_it_matters"] == (
        "A mixed-unit predictor is not a noisy predictor. It is a variable whose meaning changes "
        "between rows, and no amount of regularization repairs that — the model learns a "
        "relationship that is true of neither population.")
    assert _one(labs, "pack::clinical::default_value_mass")["why_it_matters"] == (
        "A spike at a default value is a measurement that was probably never taken, and it biases "
        "every summary toward the default. It is also invisible to every range check, because a "
        "spike at a plausible value is inside every plausible range. The count is what makes it "
        "actionable — a reviewer asking how vitals were captured wants this number.")
    second = 160 + 2 * HEIGHT_JUMP_CM
    assert _one(labs, "pack::clinical::temporal_implausibility")["why_it_matters"] == (
        "Kahn et al.'s framework separates ATEMPORAL plausibility — is this value believable — "
        "from TEMPORAL plausibility — is this trajectory believable. Each value flagged here is "
        "individually plausible and the sequence is not, so no per-value check can find them: a "
        f"height of 160 cm and a height of {second:g} cm are both ordinary, and they cannot both "
        "belong to the same adult.")
    assert _one(labs, "pack::clinical::impossible_vs_extreme")["why_it_matters"] == (
        "Physiologically impossible and statistically extreme are different categories, and no "
        "generic outlier rule can tell them apart. The impossible values are a data-quality repair "
        "with a count that belongs in the paper; the unusual ones are the case mix the model is "
        "fit on. Every limit here is a convention, with its source.")


def test_the_unit_check_sees_two_populations_and_not_one():
    """The mixed-units finding's own check is distributional: glucose in mg/dL beside glucose in
    mmol/L, a factor of 18 apart, splits; one population does not."""
    from turbotab.clinical import read_unit_split

    rng = np.random.default_rng(5)
    two = pd.Series(np.r_[rng.normal(100, 12, 60), rng.normal(100 / 18, 0.7, 40)], name="glucose")
    split = read_unit_split(two)
    assert split is not None and (split.n_low, split.n_high) == (40, 60) and split.factor == 18
    assert read_unit_split(pd.Series(rng.normal(100, 12, 100), name="glucose")) is None


def test_a_range_check_passes_a_default_value_spike():
    """The default-value check finds 40 readings of 120 among 200; the plausibility check, a range
    check, finds none of them and finds three values outside its limits once they are written."""
    from turbotab.clinical import read_default_mass
    from turbotab.core.detectors.plausibility import impossible_vs_extreme_finding

    rng = np.random.default_rng(5)
    sbp = rng.normal(125, 15, 200).round()
    sbp[:40] = 120.0
    table = pd.DataFrame({"patient_id": np.arange(200), "age": rng.integers(30, 70, 200), "sbp": sbp})
    spike = read_default_mass(table["sbp"])
    assert [h["value"] for h in spike] == [120.0] and spike[0]["n"] >= 40
    assert impossible_vs_extreme_finding(table) is None
    written = sbp.copy()
    written[40:43] = [400.0, 5.0, 900.0]
    found = impossible_vs_extreme_finding(table.assign(sbp=written))
    assert found is not None and found["detail"].startswith("3 values in `sbp`")


def test_the_height_example_is_beyond_the_jump_limit():
    """The example's second height is the first plus twice the limit, so the temporal reading
    flags it; a change of the limit itself is not flagged."""
    from turbotab.clinical import HEIGHT_JUMP_CM, read_temporal

    rows = []
    pairs = [(160.0, 160.0 + 2 * HEIGHT_JUMP_CM), (160.0, 160.0 + HEIGHT_JUMP_CM)] + [(170.0, 170.5)] * 6
    for p, heights in enumerate(pairs):
        for date, h in zip(("2020-01-01", "2020-06-01"), heights):
            rows.append({"patient_id": f"p{p}", "visit_date": date, "age": 40, "height_cm": h,
                         "weight_kg": 70})
    jumps = read_temporal(pd.DataFrame(rows)).height_jumps
    assert [(j["person"], j["from"], j["to"]) for j in jumps] == [
        ("p0", 160.0, 160.0 + 2 * HEIGHT_JUMP_CM)]


# ── survey and dietary ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def survey(tmp_path_factory) -> dict:
    """`survey_instrument.csv`: a block of Likert items on one response scale, with
    `sought_support` as the target (its companion)."""
    return _run(tmp_path_factory.mktemp("survey"), SAMPLES / "survey_instrument.csv", ["survey"],
                "sought_support")[1]


def test_the_ordinal_reason_says_nothing_is_applied(survey):
    assert _one(survey, "pack::survey::ordinal_declared")["why_it_matters"] == (
        "The order comes from the instrument, not from the data — which makes the encoding "
        "row-local: the number for a row depends on that row's own answer and on nothing else.")


@pytest.fixture(scope="module")
def kilojoules(tmp_path_factory) -> dict:
    """`nhanes_kilojoules.csv`: total energy written in kilojoules beside grams of the
    macronutrients (its companion)."""
    return _run(tmp_path_factory.mktemp("kilojoules"), SAMPLES / "nhanes_kilojoules.csv",
                ["dietary"])[1]


def test_the_energy_unit_reason_names_what_a_unit_moves(kilojoules):
    assert _one(kilojoules, "pack::dietary::atwater")["why_it_matters"] == (
        "Total energy is the denominator of every nutrient density and the scale of every "
        "implausible-intake screen, so a unit error here moves both and is invisible in the final "
        "tables.")


def test_a_units_factor_moves_densities_and_the_screen_but_not_a_residual():
    """By least squares (numpy): total energy written in kJ, 4.184 times its kcal, leaves a
    nutrient's residual on energy unchanged, so not every adjustment moves; it divides every
    density by 4.184 and moves a 2,000 kcal day outside the screen's kcal range."""
    from turbotab import packs

    rng = np.random.default_rng(11)
    kcal = rng.normal(2000, 400, 300)
    protein = 0.04 * kcal + rng.normal(0, 8, 300)
    kj = 4.184 * kcal

    def residual(energy: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones_like(energy), energy])
        beta, *_ = np.linalg.lstsq(design, protein, rcond=None)
        return protein - design @ beta

    np.testing.assert_allclose(residual(kj), residual(kcal), atol=1e-8)
    np.testing.assert_allclose(protein / kj, protein / kcal / 4.184)
    low, high = packs._PLAUSIBLE_KCAL
    assert low <= 2000 <= high and 2000 * 4.184 > high


# ── every reason served on these tables ──────────────────────────────────────


@pytest.mark.parametrize("name", ["visits", "labs", "paired", "merged",
                                  pytest.param("redundant", marks=pytest.mark.slow),
                                  "expression", "survey", "kilojoules"])
def test_no_served_reason_says_what_v2_does_not_do(name, request):
    """Every reason served on these tables, before a target and with one where the fixture has
    one (`clinic_visits.csv` both ways)."""
    found = request.getfixturevalue(name)
    artifacts = [x for x in (found if isinstance(found, tuple) else (found,)) if isinstance(x, dict)]
    problems = [(f["id"], phrase) for artifact in artifacts for f in artifact["findings"]
                for phrase in UNTRUE if phrase in (f.get("why_it_matters") or "")]
    assert artifacts and not problems, problems
