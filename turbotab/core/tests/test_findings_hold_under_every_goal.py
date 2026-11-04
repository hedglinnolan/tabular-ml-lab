"""Every served ``why_it_matters`` says only what v2 does, under either goal and before one is
chosen, in every lens, with and without a target (pp2-why-audit, WA1).

The findings stage reads no purpose (its reads: ``stages.build_graph``), so one reason is served
whichever goal the researcher holds or will choose; a sentence whose truth depends on the goal says
the condition itself. The re-scan of every family's served reason found these false or goal-blind:

* numbers stored as text: the app's own reason (``core/repairs.py``) and the legacy one
  (``ml/import_doctor.py``) said the column "cannot be modeled" and sits out of the analysis; v2's
  design reads a text predictor as a category, each distinct value a level of its own
  (``models.pipeline.category_inputs``), until a repair reads it as numbers or, for numbers
  ``parse_text_numbers`` reads, the fit's codes-or-amounts question does: the fit waits on it,
  offering amounts first (``readings.code_question``, ``predictors_or_ask``), reads an "amounts"
  answer (``working.text_amounts``) or asks how commas that read two ways read, whose answers are
  the finding's own repairs, and does not ask it of an assay's exposures
  (``readings.unsettled_codes``). The app's own reason said a text predictor "enters the models as
  a category", and the clinical pack's number-format reason that such a column "silently" becomes
  "hundreds of one-hot levels", beside the app's own finding about the same column, where the
  columns it is served on hold 88 and 165 distinct values;
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
  check the app has", each beside the app's own distributional check that found it, and the
  mixed-units reason called its column "a mixed-unit predictor", served alike when that column is
  the outcome (the pack never sees the target);
* the temporal reason's 169 cm was typed beside the jump limit it illustrates
  (``clinical.HEIGHT_JUMP_CM``), and the implausible-intake reason's 300 kcal beside the floor it
  illustrates (``packs._PLAUSIBLE_KCAL``);
* the survey ordinal reason said the encoding "is applied now", where nothing is applied at the
  finding;
* the energy-unit reason said total energy is "the denominator of every energy adjustment" and that
  a unit error "propagates into every result": a residual adjustment is unchanged by a unit's
  constant factor (least squares, numpy);
* the plausibility reason called the unusual values "the case mix the model exists to learn", and
  the two-level outcome's reason asked "which one you are predicting": goal-blind under Estimate.

The second round's re-scan found reasons whose truth depends on the table, restated from it
(``finding_words.READ_CLAUSES``), and more that said what v2 does not do:

* the number-format reason named a repair and the fit's codes-or-amounts question for every
  column it flags, where the app reads neither `57,689.59` nor `1,250,000` as numbers; and its
  decimal comma "no error anywhere would report", where v2 reads a comma as a thousands separator
  only on the researcher's answer;
* the no-run-order reason said a run order lets QC-RLSC read it, on tables with no pooled QC
  (QC-RLSC fits no curve to fewer than five, Broadhurst et al. 2018);
* the below-detection reason said the blanks are "left out", which only complete cases do;
* the energy reason said "models that keep total energy predict alike", where the multivariate
  density model does not (least squares, numpy);
* the duplicate-ids reason said a duplicated feature makes the correction anticonservative, where
  under Benjamini–Hochberg the direction depends on the copy's p-value (statsmodels);
* the genomics data-type reasons asked which pipeline or tool produced the matrix, beside a card
  that takes no answer;
* the wide-shape reason advised one row per measurement beside a lens's note that rules the reshape
  out.

Each truth is by construction (a table built here, or a shipped sample whose companion says what it
holds), by numpy or scikit-learn, or by the code's own structure; the true sentences of each reason
stay as they were.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
          "silently becomes", "an unregularized fit is not", "hundreds of one-hot",
          "mixed-unit predictor", "a recall of 300 kcal", "it enters the models as a category, "
          "each distinct value a level of its own, rather than as a number, until",
          # Round 2.
          "left out as if unknown", "predict alike", "in the anticonservative direction",
          "no error anywhere would report", "You know; they do not", "Which tool produced this "
          "matrix?", "Which did your pipeline produce", "you know, the matrix does not")


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


TEXT_NUMBERS_WHY = ("As a predictor it can enter the models only as a category, one level per "
                    "distinct value, until it is read as numbers: here, or through the fit's "
                    "question whether it holds codes or amounts, which is not asked of an assay's "
                    "study factors.")
NUMERIC_AS_TEXT_WHY = ("As a predictor left as text, it enters the models as a category, each "
                       "distinct value a level of its own, rather than as a number.")
#: The number-format reason's second sentence, which holds whichever columns it flags: the comma is
#: read as a thousands separator only on the researcher's answer (the round-2 tests below).
NUMBER_FORMAT_COMMA = (
    "The decimal-comma case is worse in a reader that takes the comma for a thousands separator: "
    "there it can parse and be wrong by a factor of a hundred with no error. This app reads no "
    "comma as a thousands separator unless you answer that it is one.")
NUMBER_FORMAT_ROUTES = ("by a repair or through the fit's question whether it holds codes or "
                        "amounts (not asked of an assay's study factors)")
#: Every column it flags is one the app reads as numbers (the text-number finding's).
NUMBER_FORMAT_WHY = (
    "A predictor that does not parse as numeric can enter the models only as a category until it "
    f"is read as numbers, {NUMBER_FORMAT_ROUTES}, so a continuous measurement left as text becomes "
    f"a one-hot level for each distinct value. {NUMBER_FORMAT_COMMA}")
#: No column it flags is one the app reads as numbers: no route is named.
NUMBER_FORMAT_NO_ROUTE = (
    "A predictor that does not parse as numeric enters the models as a category, so a continuous "
    "measurement left as text becomes a one-hot level for each distinct value.")


@pytest.fixture(scope="module")
def labs_glucose(tmp_path_factory) -> tuple[Ingested, dict, dict]:
    """`clinical_labs.csv` under the clinical lens before a target is chosen and with `glucose`,
    the column its mixed-units finding flags, as the target."""
    read, before = _run(tmp_path_factory.mktemp("labs_glucose"), SAMPLES / "clinical_labs.csv",
                        ["clinical"])
    return read, before, read.run(findings_stage, ProjectState(lens=["clinical"], target="glucose"))


def test_numbers_stored_as_text_are_said_to_be_a_category_until_read_as_numbers(visits, labs,
                                                                                labs_glucose):
    """The served reasons, before a target and with one, on both tables they are served on:
    `income` (`clinic_visits.csv`) and `platelets` (`clinical_labs.csv`), the app's own text-number
    findings, which the fit asks about (the next test), and `weight`, the legacy numeric-as-text
    finding, units the text-number reading does not read, which the fit does not ask about and
    which enters the models as a category (the test after)."""
    _, before, after = visits
    _, labs_before, _ = labs_glucose
    for artifact in (before, after):
        assert _one(artifact, "text_numbers")["why_it_matters"] == TEXT_NUMBERS_WHY
        assert _one(artifact, "numeric_as_text")["why_it_matters"] == NUMERIC_AS_TEXT_WHY
    for artifact in (before, after, labs, labs_before):
        found = _one(artifact, "pack::clinical::number_format")
        assert found["why_it_matters"] == NUMBER_FORMAT_WHY
    platelets = [f for f in labs["findings"] if f["id"] == "text_numbers__platelets"]
    assert [f["why_it_matters"] for f in platelets] == [TEXT_NUMBERS_WHY]


def test_the_fit_asks_codes_or_amounts_of_a_text_predictor(visits):
    """What the text-number reason says, both ways. The fit waits on whether `income` (numbers
    with thousands separators) holds codes or amounts, leading with amounts, and asks nothing of
    `weight` (`107 kg`, which ``parse_text_numbers`` does not read); under an assay lens it does
    not ask of `income` as an exposure and asks of it as a covariate. Once "amounts" is answered,
    the fit asks how its commas read (every one could split thousands), and its two answers are
    this finding's own repairs; a "codes" answer keeps it a category, a level per distinct value
    (pandas counts 88)."""
    from turbotab.core import readings
    from turbotab.core.models.pipeline import category_inputs
    from turbotab.core.readings import Unsettled, predictors_or_ask, unsettled_codes, whole_facts

    read, _, after = visits
    info = {c["name"]: c for c in read.info["columns"]}
    with read.store() as store:
        facts = whole_facts(["income", "weight"], info, store=store)
    asked = unsettled_codes(ProjectState(lens=["clinical"], target="outcome"),
                            ["income", "weight"], facts)
    assert [(r.column, r.value) for r in asked] == [("income", "amount")]
    for role, want in (("exposure", []), ("covariate", ["income"])):
        assay = ProjectState(lens=["metabolomics"], target="outcome", roles={"income": role})
        assert [r.column for r in unsettled_codes(assay, ["income"], facts)] == want

    def ask(answer: dict[str, str]) -> Unsettled:
        state = ProjectState(lens=["clinical"], target="outcome", roles={"income": "covariate"},
                             reading_confirmations=answer)
        with read.store() as store, pytest.raises(Unsettled) as waited:
            predictors_or_ask(state, read.info, store=store)
        return waited.value

    first = ask({})
    assert [x["decision"] for x in first.exits] == [x["decision"] for x in
                                                    readings.code_or_count_exits("income")]
    assert first.exits[0]["label"].startswith("`income` is a count or an amount")
    commas = ask({"code_or_count:income": "amount"})
    offered = [x["decision"]["option"] for x in commas.exits
               if (x["decision"] or {}).get("finding_id") == "text_numbers__income"]
    own = [o["key"] for o in _one(after, "text_numbers")["repairs"]]
    assert offered == ["thousands", "decimal_comma"] and set(offered) <= set(own)

    frame = read.frame()
    assert frame["income"].dtype == object and frame["income"].nunique() == 88
    codes = ProjectState(lens=["clinical"], target="outcome",
                         reading_confirmations={"code_or_count:income": "code"})
    assert category_inputs(codes, frame[["income"]], ["income"]) == ["income"]


def test_an_amounts_answer_reads_text_numbers_whose_commas_read_one_way(tmp_path):
    """The other path the reason names: forty text numbers whose commas can only mark decimals
    (`4,73`: two digits after a comma cannot be a thousands group) are read as numbers on an
    "amounts" answer alone, as pandas reads them once each comma is a point (the SAS `.` blank),
    and are no category; without the answer, they stay text and a category."""
    from turbotab.core.datastore import ROW_ID
    from turbotab.core.models.pipeline import category_inputs
    from turbotab.core.repairs import evaluate
    from turbotab.core.stages.working import text_amounts

    drawn = np.random.default_rng(0).uniform(0.5, 9.5, 40)
    values = [f"{v:.2f}".replace(".", ",") for v in drawn] + [".", None]
    path = tmp_path / "oriented.parquet"
    pd.DataFrame({"income": values, ROW_ID: np.arange(len(values))}).to_parquet(path)
    assert text_amounts({}, path, ["income"], {}) == {}
    assert category_inputs({}, pd.DataFrame({"income": values}), ["income"]) == ["income"]
    amounts = {"reading_confirmations": {"code_or_count:income": "amount"}}
    sql = text_amounts(amounts, path, ["income"], {})
    assert set(sql) == {"income"}
    got = evaluate(path, ["income"], sql)["income"]
    want = pd.to_numeric(pd.Series(values).str.replace(",", ".", regex=False), errors="coerce")
    np.testing.assert_array_equal(got.to_numpy(dtype=float), want.to_numpy(dtype=float))
    assert category_inputs(amounts, pd.DataFrame({"income": got}), ["income"]) == []


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


MIXED_UNITS_WHY = (
    "A mixed-unit column is not a noisy column. It is a variable whose meaning changes between "
    "rows, and no amount of regularization repairs that — the model learns a relationship that is "
    "true of neither population.")


def test_the_mixed_units_reason_holds_whatever_the_columns_role(labs, labs_glucose):
    """The pack never sees the target (``detectors.pack_findings``), so one reason is served on
    `glucose` before a target, with `readmitted` as the target and with `glucose` itself, the
    flagged column, as the target: it names no role."""
    _, before, flagged = labs_glucose
    for artifact in (before, labs, flagged):
        found = _one(artifact, "pack::clinical::mixed_units")
        assert found["affected_columns"][0] == "glucose"
        assert found["why_it_matters"] == MIXED_UNITS_WHY


def test_the_clinical_reasons_say_only_what_the_app_checks(labs):
    from turbotab.clinical import HEIGHT_JUMP_CM

    assert _one(labs, "pack::clinical::mixed_units")["why_it_matters"] == MIXED_UNITS_WHY
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


@pytest.fixture(scope="module")
def recalls(tmp_path_factory) -> dict:
    """`dietary_recalls.csv`: 24-hour recalls with `energy_kcal`, some outside the plausible
    range (its companion), under the dietary lens."""
    return _run(tmp_path_factory.mktemp("recalls"), SAMPLES / "dietary_recalls.csv",
                ["dietary"])[1]


def _intake_why(low: float) -> str:
    return (f"These are possible days and poor estimates: a recall under {low:,g} kcal is "
            "under-reporting rather than starvation. Excluding them is an eligibility criterion, "
            "which changes N and is reported in participant flow — so it is offered here and never "
            "applied. Nothing is filtered unless you say so.")


def test_the_implausible_intake_example_is_the_floor_the_finding_uses(recalls, kilojoules):
    """Both tables it is served on: the example is the floor below which the finding counts a
    recall (``packs._PLAUSIBLE_KCAL``), here 500 kcal, so it is one the finding flags."""
    from turbotab import packs

    low, _ = packs._PLAUSIBLE_KCAL
    for artifact in (recalls, kilojoules):
        assert _one(artifact, "pack::dietary::implausible_intake")["why_it_matters"] == (
            _intake_why(low))


def test_the_intake_example_moves_with_the_floor_and_only_on_its_own_reason(monkeypatch):
    """With the floor moved to 1,200 kcal (by construction), the pack counts the recalls below
    1,200 and the example names 1,200; the same legacy words under another family are served as
    written."""
    from turbotab import packs
    from turbotab.core.stages.finding_words import FindingContext, speak

    monkeypatch.setattr(packs, "_PLAUSIBLE_KCAL", (1200.0, 5000.0))
    frame = pd.DataFrame({"energy_kcal": [300.0, 900.0, 1100.0, 2000.0, 2500.0, 6000.0]})
    raw = packs._implausible_intake(frame)
    assert raw is not None and raw["params"]["n_flagged"] == 4
    legacy = raw["why_it_matters"]
    fc = FindingContext(frame=frame)
    finding = {"id": raw["id"], "title": raw["title"], "detail": raw["detail"],
               "why_it_matters": legacy, "affected_columns": ["energy_kcal"]}
    assert speak(finding, raw, fc)["why_it_matters"] == _intake_why(1200.0)
    other = {"id": "pack::dietary::atwater", "title": "x", "detail": "x",
             "why_it_matters": legacy, "affected_columns": ["energy_kcal"]}
    assert speak(other, None, fc)["why_it_matters"] == legacy


@pytest.mark.parametrize("fam, legacy", [
    ("pack::clinical::mixed_units",
     "A mixed-unit predictor is not a noisy predictor. It is a variable whose meaning changes "
     "between rows."),
    ("pack::clinical::number_format",
     "A column that does not parse as numeric is treated as a category by everything downstream, "
     "so a continuous measurement silently becomes hundreds of one-hot levels."),
], ids=["mixed_units", "number_format"])
def test_the_restated_clauses_stay_silent_under_another_family(fam, legacy):
    """Each clause restates only its own family's reason: the same words under its family are
    restated, and under another are served as written."""
    from turbotab.core.stages.finding_words import FindingContext, speak

    fc = FindingContext(frame=pd.DataFrame({"glucose": [1.0, 2.0]}))

    def served(fid: str) -> str:
        return speak({"id": fid, "title": "x", "detail": "x", "why_it_matters": legacy,
                      "affected_columns": ["glucose"]}, None, fc)["why_it_matters"]

    assert served(fam) != legacy
    assert served("category_variants__glucose") == legacy


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


# ── reasons whose truth depends on the table (round 2) ───────────────────────


def _table(tmp: Path, columns: dict[str, Any], lens: list[str], target: str | None = None
           ) -> tuple[Ingested, dict]:
    path = tmp / "table.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return _run(tmp, path, lens, target)


def _number_formats(tmp: Path, **text: list[str]) -> tuple[Ingested, dict, dict]:
    """A clinical table of 60 rows with the text columns ``text``, `age` and a two-level
    `outcome`, read under the clinical lens before a target and with `outcome`."""
    rng = np.random.default_rng(1)
    read, before = _table(tmp, {"patient_id": np.arange(60), "age": rng.integers(20, 80, 60),
                                **text, "outcome": rng.choice(["yes", "no"], 60)}, ["clinical"])
    return read, before, read.run(findings_stage, ProjectState(lens=["clinical"], target="outcome"))


def test_the_number_format_reason_names_no_route_where_the_app_reads_none(tmp_path):
    """By construction `cost` (`57,689.59`: a thousands group beside a decimal point) and `big`
    (`1,250,000`: two thousands groups) are numbers the app's text-number reading does not read:
    no text-number finding reads them, no repair the served findings offer reads them, and the
    fit's codes-or-amounts question is asked of neither. The number-format reason flags both, and
    names neither a repair nor that question."""
    from turbotab.core.readings import unsettled_codes, whole_facts

    rng = np.random.default_rng(2)
    cost = [f"{v:,.2f}" for v in rng.uniform(1_000, 90_000, 60)]
    big = [f"{v:,}" for v in rng.integers(1_000_000, 9_000_000, 60)]
    assert cost[0].count(",") == 1 and "." in cost[0] and big[0].count(",") == 2
    read, before, after = _number_formats(tmp_path, cost=cost, big=big)
    info = {c["name"]: c for c in read.info["columns"]}
    with read.store() as store:
        facts = whole_facts(["cost", "big"], info, store=store)
    assert unsettled_codes(ProjectState(lens=["clinical"], target="outcome"), ["cost", "big"],
                           facts) == []
    for artifact in (before, after):
        found = _one(artifact, "pack::clinical::number_format")
        assert found["affected_columns"] == ["big", "cost"]
        assert found["why_it_matters"] == f"{NUMBER_FORMAT_NO_ROUTE} {NUMBER_FORMAT_COMMA}"
        assert not [f["id"] for f in artifact["findings"] if family(f["id"]) == "text_numbers"]
        offered = [(f["id"], o["key"]) for f in artifact["findings"]
                   if {"cost", "big"} & set(f["affected_columns"]) for o in f.get("repairs") or []]
        assert offered == []


def test_the_number_format_reason_names_the_route_only_for_the_columns_it_has(tmp_path):
    """Beside `cost`, `crp` holds numbers in exponent form with five `<LOD` marks: the app's own
    finding reads it, with its repair, and the fit asks whether it holds codes or amounts. The
    reason names the route for `crp` alone."""
    from turbotab.core.readings import unsettled_codes, whole_facts

    rng = np.random.default_rng(1)
    cost = [f"{v:,.2f}" for v in rng.uniform(1_000, 90_000, 60)]
    crp = [f"{v:.2E}" for v in rng.uniform(1, 50, 60)]
    crp[:5] = ["<LOD"] * 5
    read, before, after = _number_formats(tmp_path, cost=cost, crp=crp)
    info = {c["name"]: c for c in read.info["columns"]}
    with read.store() as store:
        facts = whole_facts(["cost", "crp"], info, store=store)
    asked = unsettled_codes(ProjectState(lens=["clinical"], target="outcome"), ["cost", "crp"],
                            facts)
    assert [r.column for r in asked] == ["crp"]
    for artifact in (before, after):
        assert [o["key"] for o in _one(artifact, "text_numbers")["repairs"]] == ["read_numbers"]
        found = _one(artifact, "pack::clinical::number_format")
        assert found["affected_columns"] == ["cost", "crp"]
        assert found["why_it_matters"] == (
            f"{NUMBER_FORMAT_NO_ROUTE[:-1]}; `crp` can be read as numbers first, "
            f"{NUMBER_FORMAT_ROUTES}. {NUMBER_FORMAT_COMMA}")


def test_no_comma_is_read_as_a_thousands_separator_without_an_answer(tmp_path):
    """The second sentence, both ways, by construction. Forty values with a comma that can only
    mark decimals (`4,73`) are read at ingest as pandas reads them once the comma is a point; forty
    whose every comma could split thousands (`8,654`) stay text, and the text-number finding offers
    both readings; read as numbers without an answer they stay text, and on the "commas split
    thousands" answer `8,654` is 8654."""
    from turbotab.core import repairs

    rng = np.random.default_rng(0)
    decimal = [f"{v:.2f}".replace(".", ",") for v in rng.uniform(0.5, 9.5, 40)]
    grouped = [f"{v:,}" for v in rng.integers(1_000, 9_999, 40)]
    read, artifact = _table(tmp_path, {"decimal": decimal, "grouped": grouped,
                                       "y": rng.normal(size=40)}, ["clinical"])
    frame = read.frame()
    want = pd.Series(decimal).str.replace(",", ".", regex=False).astype(float)
    np.testing.assert_array_equal(frame["decimal"].to_numpy(dtype=float), want.to_numpy())
    assert frame["grouped"].tolist() == grouped
    found = _one(artifact, "text_numbers")
    assert found["affected_columns"] == ["grouped"]
    assert [o["key"] for o in found["repairs"]] == ["thousands", "decimal_comma"]
    unanswered = repairs.numbers_read(frame, ProjectState(lens=["clinical"]))
    assert unanswered["grouped"].tolist() == grouped
    d = next(o for o in found["repairs"] if o["key"] == "thousands")["decision"]
    answered = repairs.numbers_read(frame, ProjectState(lens=["clinical"], numbers_read={
        d["finding_id"]: {"option": d["option"], "params": d["params"]}}))
    np.testing.assert_array_equal(answered["grouped"].to_numpy(dtype=float),
                                  [float(v.replace(",", "")) for v in grouped])


def test_the_below_detection_reason_holds_under_every_missing_values_answer(labs, labs_glucose):
    """`hs_crp` (`clinical_labs.csv`) has values below a detection limit, before a target and with
    one; the reason no longer says the blanks are left out, which only complete cases do."""
    _, before, _ = labs_glucose
    for artifact in (before, labs):
        assert _one(artifact, "below_detection")["why_it_matters"] == (
            "Read as blanks they are treated as unknown, dropped with their rows or filled from "
            "the other values, though each is known to be small; replaced by one value they all "
            "become that value, which narrows their spread.")


def test_complete_cases_drop_the_blanks_rows_and_the_fills_keep_them_filled():
    """The truth both ways, by construction (seven blanks in 60 rows): complete cases drop exactly
    those rows (``rows.cohort_flow``); the single fill in each training fold and multiple
    imputation keep every row, and the in-fold fill (the design every family shares) and each of
    multiple imputation's completed tables leave no blank."""
    from sklearn.pipeline import Pipeline

    from turbotab.core.decisions import MissingSpec
    from turbotab.core.methods.imputation import chained_equations
    from turbotab.core.models.pipeline import design_spec, shared_steps
    from turbotab.core.stages.rows import cohort_flow

    rng = np.random.default_rng(5)
    hs = rng.lognormal(0, 0.6, 60)
    y = hs + rng.normal(0, 0.5, 60)
    hs[:7] = np.nan
    X = pd.DataFrame({"hs_crp": hs, "age": rng.normal(50, 10, 60), "y": y})
    blank = set(range(7))
    for strategy in ("complete_case", "impute", "multiple_imputation"):
        _, kept = cohort_flow(X, target="y", rules=None, missing=strategy,
                              predictor_columns=["hs_crp"])
        kept = set(int(i) for i in kept)
        assert kept == (set(range(60)) - blank if strategy == "complete_case" else set(range(60)))
        spec = design_spec(ProjectState(target="y", missing=MissingSpec(strategy=strategy)),
                           X[["hs_crp", "age"]], ["hs_crp", "age"])
        assert spec.impute is (strategy != "complete_case")
        if spec.impute:
            filled = Pipeline(shared_steps(spec)).set_output(transform="pandas").fit_transform(
                X[["hs_crp", "age"]])
            assert len(filled) == 60 and not filled.isna().to_numpy().any()
    drawn = chained_equations(X, m=3, seed=1)
    assert len(drawn.frames) == 3
    assert all(len(f) == 60 and not f.isna().to_numpy().any() for f in drawn.frames)


# ── the run order and the pooled QCs ─────────────────────────────────────────


RUN_ORDER_LEAD = (
    "Instrument drift is often the largest single variance component in a run, and every way of "
    "seeing it is ordered by injection: drift diagnostics and QC-RLSC correction both become "
    "impossible without one. The order is in the instrument's sequence file — if you still have "
    "it, adding it as a run-order column that runs 1 to ")


@pytest.fixture(scope="module")
def no_qc(tmp_path_factory) -> dict:
    """`metabolomics_no_qc.csv`: 72 participants, no pooled QC and no run order (its companion)."""
    return _run(tmp_path_factory.mktemp("no_qc"), SAMPLES / "metabolomics_no_qc.csv",
                ["metabolomics"])[1]


def test_without_pooled_qcs_a_run_order_is_not_said_to_let_qc_rlsc_read_it(no_qc):
    """By its companion the table has no pooled QC, and none is found; the drift reading reads a
    run order on it, and QC-RLSC (which needs at least five pooled QC injections; Broadhurst et
    al. 2018) is not promised."""
    from turbotab.core.detectors.assay import drift_reading

    assert not [f for f in no_qc["findings"] if family(f["id"]) == "pack::metabolomics::pooled_qc"]
    assert _one(no_qc, "pack::metabolomics::no_run_order")["why_it_matters"] == (
        f"{RUN_ORDER_LEAD}72 lets the drift diagnostics read it; QC-RLSC also needs at least 5 "
        "pooled QC injections, and none is found in this table.")
    frame = pd.read_csv(SAMPLES / "metabolomics_no_qc.csv")
    assert drift_reading(frame.assign(run_order=np.arange(1, len(frame) + 1))) is not None


def _untargeted(keep_qc: Sequence[int]) -> pd.DataFrame:
    """`metabolomics_untargeted.csv` in its run order up to its last pooled QC (row 70 of 80, so
    the run begins and ends with one), with only the QCs at ``keep_qc`` and no run-order column."""
    frame = pd.read_csv(SAMPLES / "metabolomics_untargeted.csv").sort_values("run_order")
    frame = frame.reset_index(drop=True).iloc[:71]
    qc = np.flatnonzero(frame["sample_type"] == "pooled_qc")
    assert list(qc) == [0, 10, 20, 30, 40, 50, 60, 70]
    gone = [i for i in qc if i not in keep_qc]
    return frame.drop(index=gone, columns=["run_order"]).reset_index(drop=True)


@pytest.mark.parametrize("keep_qc, n_qc", [((0, 10, 20, 30, 40, 50, 60, 70), 8),
                                           ((0, 30, 50, 70), 4)], ids=["eight", "four"])
def test_qc_rlsc_is_promised_where_the_pooled_qcs_can_fit_its_curve(tmp_path, keep_qc, n_qc):
    """Eight pooled QCs that begin and end the run keep the promise: with a run order added,
    QC-RLSC fits one curve over the run (an option with no refusal) and the drift reading reads
    the order. Four are found but are too few: every QC-RLSC option is refused, and the reason
    says so."""
    from turbotab.core.detectors.assay import drift_reading

    frame = _untargeted(keep_qc)
    n = len(frame)
    _, artifact = _table(tmp_path, frame.to_dict("list"), ["metabolomics"])
    assert _one(artifact, "pack::metabolomics::pooled_qc")["title"].startswith(f"{n_qc} rows")
    why = _one(artifact, "pack::metabolomics::no_run_order")["why_it_matters"]
    if n_qc >= 5:
        assert why == f"{RUN_ORDER_LEAD}{n} lets both read it."
    else:
        assert why == (f"{RUN_ORDER_LEAD}{n} lets the drift diagnostics read it; QC-RLSC also "
                       f"needs at least 5 pooled QC injections, and {n_qc} are found in this "
                       f"table.")
    ordered = tmp_path / "ordered"
    ordered.mkdir()
    _, with_order = _table(ordered, frame.assign(run_order=np.arange(1, n + 1)).to_dict("list"),
                           ["metabolomics"])
    rlsc = [o for o in _one(with_order, "pack::metabolomics::pooled_qc")["repairs"]
            if o["key"].startswith("qc_rlsc")]
    assert rlsc
    assert any(not o["decision"]["params"].get("problems") for o in rlsc) is (n_qc >= 5)
    assert drift_reading(frame.assign(run_order=np.arange(1, n + 1))) is not None


@pytest.mark.parametrize("n_qc", [4, 5])
def test_qc_rlsc_refuses_a_run_with_fewer_than_five_pooled_qcs(n_qc):
    """Broadhurst et al. 2018: "at least five pooled QCs distributed evenly across a single
    batch". A run of 40 injections that begins and ends with a QC: four QCs are refused, five
    are not."""
    from turbotab.core.methods.qc_drift import Injections, sufficiency

    qc = np.zeros(40, dtype=bool)
    qc[np.linspace(0, 39, n_qc).round().astype(int)] = True
    inj = Injections(order=np.arange(1.0, 41.0), batch=np.full(40, "", dtype=object), qc=qc)
    assert bool(sufficiency(inj)) is (n_qc < 5)


# ── the dietary, metabolomics and genomics reasons ───────────────────────────


def test_the_energy_reason_names_the_models_that_predict_alike(recalls):
    why = _one(recalls, "pack::dietary::energy_adjustment")["why_it_matters"]
    assert why.endswith(
        "For prediction, the standard model, the residual model with total energy kept and the "
        "partition model are one linear model in different terms, so a linear fit without a "
        "penalty predicts the same under each; the multivariate density model is a different "
        "model and can predict differently."), why


def test_the_standard_residual_and_partition_models_predict_alike_and_density_does_not():
    """By least squares (numpy), on v2's own energy steps over constructed intakes: the residual
    model with energy kept and the partition model reproduce the standard model's predictions
    (they span one design), and the multivariate density model's differ."""
    from turbotab.core.methods.energy import EnergyAdjuster

    rng = np.random.default_rng(7)
    kcal = rng.normal(2000, 400, 400)
    fat = 0.035 * kcal + rng.normal(0, 12, 400)
    age = rng.normal(50, 10, 400)
    y = 0.02 * fat + 0.001 * kcal + 0.03 * age + rng.normal(0, 1, 400)
    X = pd.DataFrame({"fat_g": fat, "energy_kcal": kcal, "age": age})

    def predictions(method: str, **kw: Any) -> np.ndarray:
        Z = EnergyAdjuster(method, "energy_kcal", ["fat_g"], **kw).fit(X).transform(X)
        design = np.column_stack([np.ones(len(X)), np.asarray(Z, dtype=float)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return design @ beta

    standard = predictions("standard")
    for method, kw in (("residual", {}), ("partition", {"atwater": {"fat_g": 9.0}})):
        np.testing.assert_allclose(predictions(method, **kw), standard, atol=1e-9)
    assert np.max(np.abs(predictions("density_multivariate") - standard)) > 0.1 * np.std(y)


def test_the_duplicate_ids_reason_says_the_correction_moves_either_way(merged):
    _, artifact = merged
    assert _one(artifact, "pack::metabolomics::duplicate_ids")["why_it_matters"] == (
        "A feature counted twice is counted twice in the number of metabolites you report and "
        "twice in every multiple-testing correction, which moves the other features' adjusted "
        "p-values: down when the copy's own p-value is among the smallest, up when it is among "
        "the largest. A sample id on two rows is two injections of one sample, and whichever way "
        "a later join resolves it — fanning out, or keeping one arbitrarily — it does so without "
        "saying which.")


@pytest.mark.parametrize("copied", ["smallest", "largest"])
def test_a_duplicated_feature_moves_benjamini_hochberg_both_ways(copied):
    """statsmodels' Benjamini–Hochberg (the feature-wise tests' correction) over 100 constructed
    p-values: a copy of the smallest lowers the others' adjusted p-values and raises none; a copy
    of the largest raises them and lowers none."""
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(2)
    p = np.sort(np.r_[rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    k = 0 if copied == "smallest" else len(p) - 1
    before = multipletests(p, method="fdr_bh")[1]
    after = multipletests(np.append(p, p[k]), method="fdr_bh")[1][:len(p)]
    others = np.delete(np.arange(len(p)), k)
    lower = int((after[others] < before[others] - 1e-12).sum())
    higher = int((after[others] > before[others] + 1e-12).sum())
    assert (lower > 90 and higher == 0) if copied == "smallest" else (higher > 90 and lower == 0)


@pytest.fixture(scope="module", params=["genomics_cpm.csv", "genomics_estimated_counts.csv",
                                        "genomics_fpkm.csv"])
def data_type(request, tmp_path_factory) -> tuple[str, dict]:
    """Each matrix whose data-type card says the matrix cannot settle what it is (its
    companion), under the genomics lens."""
    name = request.param
    return name, _run(tmp_path_factory.mktemp(name.split(".")[0]), SAMPLES / name,
                      ["genomics"])[1]


STATE_IT = "is yours to state in the methods; the matrix cannot settle it."


def test_the_data_type_reasons_ask_no_question_the_app_takes_no_answer_to(data_type):
    """The card offers no repair and routes to no question (its summary says there is no control),
    so its reason asks none: what the matrix cannot settle is the researcher's to state."""
    from turbotab.core.stages.finding_words import NO_LEVER

    name, artifact = data_type
    found = _one(artifact, "pack::genomics::data_type")
    assert found["routes_to"] is None and not found.get("repairs")
    assert found["summary"].endswith(NO_LEVER)
    why = found["why_it_matters"]
    assert "?" not in why and "You know" not in why
    if name == "genomics_cpm.csv":
        assert why.endswith(
            "CPM and TPM both rescale every sample to exactly one million, and TPM divides by "
            "effective length before that rescale, so the trace is gone from the matrix — no test "
            f"on these numbers separates them. Which one your pipeline produced {STATE_IT}"), why
    else:
        assert why.endswith(
            "What separates them here is the spread of the library sizes, and that is evidence "
            f"rather than a determination. Which tool produced this matrix {STATE_IT}"), why


def test_the_log_expression_reason_asks_no_question(tmp_path):
    """A constructed log2(CPM + 1) matrix (negative-binomial counts, 24 samples, 200 genes): the
    data-type card reads it as log CPM or TPM, which cannot be told apart once logged, and its
    reason asks nothing."""
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(5, 0.01, size=(24, 200)).astype(float)
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    genes = {f"ENSG{i:011d}": np.log2(cpm[:, i] + 1) for i in range(200)}
    _, artifact = _table(tmp_path, {"condition": rng.choice(["a", "b"], 24), **genes},
                         ["genomics"])
    found = _one(artifact, "pack::genomics::data_type")
    assert found["title"].startswith("What your numbers are: log2(CPM or TPM + 1)")
    assert found["why_it_matters"].endswith(
        "Which pipeline produced these values is yours to state in the methods: CPM and TPM are "
        "indistinguishable once logged, and the matrix cannot settle which."), found
    assert "?" not in found["why_it_matters"]


@pytest.fixture(scope="module")
def cpm(tmp_path_factory) -> dict:
    return _run(tmp_path_factory.mktemp("cpm"), SAMPLES / "genomics_cpm.csv", ["genomics"])[1]


@pytest.mark.parametrize("name, note", [
    ("expression", "These are different genes"), ("cpm", "These are different genes"),
    ("survey", "These are the items of one instrument"), ("no_qc", "These are different analytes")])
def test_a_reframed_wide_shape_serves_its_note_and_not_the_reshape_advice(name, note, request):
    """Under a lens that reads the wide shape as expected the card's detail carries the lens's note
    (genes, items, analytes: reshaping would rebuild what a row is), and the legacy reason, which
    advises a reshape v2 does not have, is not served."""
    found = _one(request.getfixturevalue(name), "wide_repeated_measures")
    assert found["title"] == "The wide shape is expected here." and found["lens"]
    assert note in found["detail"]
    assert found["why_it_matters"] is None


def test_the_wide_shape_reason_stands_where_no_lens_reframes_it(visits):
    _, before, after = visits
    for artifact in (before, after):
        found = _one(artifact, "wide_repeated_measures")
        assert found["lens"] is None
        assert found["why_it_matters"] == (
            "Wide repeated measures are fine for prediction, but if each row is a subject measured "
            "several times you may want one row per measurement instead — and clustered rows need "
            "a group-aware split.")


# ── every reason served on these tables ──────────────────────────────────────


@pytest.mark.parametrize("name", ["visits", "labs", "labs_glucose", "paired", "merged",
                                  pytest.param("redundant", marks=pytest.mark.slow),
                                  "expression", "survey", "kilojoules", "recalls", "no_qc", "cpm"])
def test_no_served_reason_says_what_v2_does_not_do(name, request):
    """Every reason served on these tables, before a target and with one where the fixture has
    one (`clinic_visits.csv` both ways, `clinical_labs.csv` before one and with its flagged
    `glucose` as the target)."""
    found = request.getfixturevalue(name)
    artifacts = [x for x in (found if isinstance(found, tuple) else (found,)) if isinstance(x, dict)]
    problems = [(f["id"], phrase) for artifact in artifacts for f in artifact["findings"]
                for phrase in UNTRUE if phrase in (f.get("why_it_matters") or "")]
    assert artifacts and not problems, problems
