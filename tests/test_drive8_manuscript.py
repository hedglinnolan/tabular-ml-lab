"""Drive 8 · the manuscript and report surfaces.

`docs/audit/DRIVE8_CLASSIC_SURFACING.md` findings 14, 15, 17, 20, 21, 28, 29,
30, 31, 32, 33. One governing rule across all of them: **the app may be silent
and it may refuse, but it must never assert something false.** Each of these
was a surface stating something the session did not do, or stating one quantity
two ways in the same panel:

- **21** the abstract reported *"feature engineering yielded 19 candidates"* on
  a run where page 03 was never opened. `candidate` is resolved from the
  SELECTION record's `n_features_before` — the columns selection could rank —
  and every FE clause fired on `candidate != original`.
- **17** the Limitations list carried an internal preprocessing-consistency
  note, complete with its own full stops, inside a semicolon list of study
  limitations.
- **20** an override re-run of ONE comparison was counted as a second
  independent test in the multiplicity sentence.
- **14/30** the improbability-band facts were re-stated in the "NHANES
  reference" vocabulary the caption above them disavows (`MISC-018`), and the
  nudge pointing at "plausibility filtering" named a control page 05 labeled
  something else.
- **28** one Table 1 variable block mixed denominators, so its percentages
  summed to 178% with nothing on screen saying why.
- **31** the reproducibility manifest reported `'commit': 'n/a'`.
- **32** p-values rendered as `0.0000`, which asserts p = 0.
- **33** *"Estimated new features ~190"* beside *"will create ~209 features"*.
- **15** a raw sklearn exception, truncated mid-word, printed as guidance.
- **29** coach cards whose "resolution" restates the finding verbatim, one of
  them with the analysis name missing entirely.
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _source(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def _extract_functions(relative: str, names, extra_globals=None):
    """Exec named module-level functions out of a Streamlit page.

    The pages run their whole UI at import, so they cannot be imported in a
    test. Parsing the shipped file and executing only the definitions under
    test keeps this a test of the REAL source rather than of a copy.
    """
    tree = ast.parse(_source(relative))
    wanted = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in set(names)]
    assert len(wanted) == len(set(names)), (
        f"{relative} no longer defines all of {sorted(set(names))} at module "
        f"level; found {[n.name for n in wanted]}"
    )
    namespace = {"__file__": str(REPO / relative)}
    namespace.update(extra_globals or {})
    exec(compile(ast.Module(body=wanted, type_ignores=[]), relative, "exec"),
         namespace)
    return namespace


# ── 21 · the abstract may only describe stages the provenance records ────────

def test_drive8_21_the_abstract_names_no_feature_engineering_stage_without_engineered_columns():
    from ml.latex_report import _format_abstract_predictor_sentence

    # The drive's own numbers: 27 predictors, a selection record whose
    # `n_features_before` is 19 (the numeric-rankable subset), 27 retained, and
    # NOTHING engineered — page 03 was skipped.
    skipped = _format_abstract_predictor_sentence(
        {"original": 27, "candidate": 19, "selected": 27, "engineered": 0}, None)
    assert "feature engineering" not in skipped.lower(), (
        "the abstract asserts a feature-engineering stage on a run where no "
        f"columns were engineered: {skipped!r}"
    )
    assert "27" in skipped, (
        "the FE clause was removed without stating the counts the record does "
        f"support: {skipped!r}"
    )


def test_drive8_21_the_abstract_still_reports_engineering_that_did_happen():
    """Positive control — the fix must not silence a real stage."""
    from ml.latex_report import _format_abstract_predictor_sentence

    ran = _format_abstract_predictor_sentence(
        {"original": 26, "candidate": 35, "selected": 23, "engineered": 9}, None)
    assert "feature engineering yielded 35 candidates" in ran, ran


def test_drive8_21_the_methods_predictor_paragraph_claims_no_skipped_stage():
    from utils.workflow_provenance import WorkflowProvenance
    from ml.narrative_engine import NarrativeEngine

    prov = WorkflowProvenance()
    prov.record_upload("meds_hbp", "classification",
                       [f"p{i}" for i in range(27)], 21849)
    # Selection ran over the 19 numeric predictors and kept all 27 features.
    prov.record_feature_selection(
        method="consensus", n_before=19, n_after=27,
        features_kept=[f"p{i}" for i in range(27)],
        consensus_methods=[],
    )
    engine = NarrativeEngine(prov, None)
    paragraph = engine._gen_predictor_variables()
    assert "feature engineering" not in paragraph.lower(), (
        "the Methods draft asserts feature engineering on a run whose "
        f"provenance holds no feature-engineering record: {paragraph!r}"
    )


def test_drive8_21_the_methods_predictor_paragraph_reports_engineering_that_ran():
    """Positive control for the Methods half."""
    from utils.workflow_provenance import WorkflowProvenance
    from ml.narrative_engine import NarrativeEngine

    prov = WorkflowProvenance()
    prov.record_upload("y", "regression", [f"p{i}" for i in range(26)], 1000)
    prov.record_feature_engineering(
        transforms=["polynomial"], n_created=9, n_before=26, n_after=35)
    prov.record_feature_selection(
        method="consensus", n_before=35, n_after=23,
        features_kept=[f"p{i}" for i in range(23)],
        consensus_methods=["lasso", "rfe"],
    )
    engine = NarrativeEngine(prov, None)
    paragraph = engine._gen_predictor_variables()
    assert "Feature engineering" in paragraph, paragraph


def test_drive8_21_the_shared_predicate_reads_engineered_columns_only():
    from ml.publication import feature_engineering_ran

    assert feature_engineering_ran({"original": 27, "candidate": 19,
                                    "selected": 27, "engineered": 0}) is False
    # No engineered count resolved at all: engineering only ADDS columns, so a
    # candidate set SMALLER than the original one is not evidence of a stage.
    assert feature_engineering_ran({"original": 27, "candidate": 19,
                                    "selected": 27}) is False
    assert feature_engineering_ran({"original": 26, "candidate": 27}) is True
    assert feature_engineering_ran(None) is False
    assert feature_engineering_ran({"engineered": 9}) is True


# ── 21b · a recorded zero is a result, not an absence ────────────────────────
#
# Addendum from the selection repair agent: `final_count = n_after_sel or
# n_final` swallowed a genuine zero, so a selection that retained NOTHING fell
# through to the pre-selection list and the Methods draft read "All 27
# candidate predictors were retained for final modeling" — the same false
# consensus by a second route. Zero is a real count; absence is None, the same
# discipline `_lockbox_open_count` keeps in the same file.

def _zero_selection_provenance():
    from utils.workflow_provenance import WorkflowProvenance

    prov = WorkflowProvenance()
    prov.record_upload("meds_hbp", "classification",
                       [f"p{i}" for i in range(27)], 21849)
    prov.record_feature_selection(
        method="consensus", n_before=19, n_after=0, features_kept=[],
        consensus_methods=["lasso", "rfe"],
    )
    return prov


def test_drive8_21b_a_selection_that_kept_nothing_does_not_draft_as_all_retained():
    from ml.narrative_engine import NarrativeEngine

    paragraph = NarrativeEngine(_zero_selection_provenance(),
                                None)._gen_predictor_variables()
    assert "were retained for final modeling" not in paragraph, (
        "a selection record of 0 retained predictors drafts as a retained "
        f"predictor set: {paragraph!r}"
    )
    assert "retained none of the 19 candidate predictors" in paragraph, paragraph


def test_drive8_21b_the_zero_survives_the_manuscript_context_route_too():
    """Page 10 hands the counts in as `feature_counts`, not only as provenance."""
    from ml.narrative_engine import NarrativeEngine

    paragraph = NarrativeEngine(
        _zero_selection_provenance(), None,
        manuscript_context={"feature_counts": {"original": 27, "candidate": 19,
                                               "selected": 0, "engineered": 0}},
    )._gen_predictor_variables()
    assert "All 27" not in paragraph and "all 27" not in paragraph, paragraph
    assert "no selected predictor set was produced" in paragraph, paragraph


def test_drive8_21b_a_real_selection_is_still_reported_normally():
    """Positive control — the None-vs-zero fix must not silence a real count."""
    from utils.workflow_provenance import WorkflowProvenance
    from ml.narrative_engine import NarrativeEngine

    prov = WorkflowProvenance()
    prov.record_upload("y", "classification", [f"p{i}" for i in range(27)], 1000)
    prov.record_feature_selection(
        method="consensus", n_before=27, n_after=19,
        features_kept=[f"p{i}" for i in range(19)],
        consensus_methods=["lasso", "rfe"],
    )
    paragraph = NarrativeEngine(prov, None)._gen_predictor_variables()
    assert "retained 19 predictors for final modeling" in paragraph, paragraph


def test_drive8_21b_the_counts_resolver_does_not_swallow_a_recorded_zero():
    from ml.publication import _resolve_workflow_feature_counts

    counts = _resolve_workflow_feature_counts(
        [f"p{i}" for i in range(27)],
        {"Feature Selection": [{"details": {"n_features_before": 19,
                                            "n_features_after": 0}}]},
        {"feature_cols": [f"p{i}" for i in range(27)]},
    )
    assert counts["selected"] == 0, (
        f"a recorded zero was replaced by a fallback count: {counts}"
    )


def test_drive8_21b_the_methods_section_states_the_zero_rather_than_listing_predictors():
    from ml.publication import generate_methods_section

    text = generate_methods_section(
        data_config={"feature_cols": [f"p{i}" for i in range(27)]},
        preprocessing_config={"missing_data": {"label": "median imputation"}},
        model_configs={"logreg": {}},
        split_config={},
        n_total=6297, n_train=4407, n_val=945, n_test=945,
        feature_names=[f"p{i}" for i in range(27)],
        target_name="meds_hbp", task_type="classification",
        metrics_used=["F1"],
        manuscript_context={"feature_counts": {"original": 27, "candidate": 19,
                                               "selected": 0, "engineered": 0}},
    )
    assert "no selected predictor set was produced" in text, text
    assert "predictor variables were included" not in text, text


# ── the fourth n, and the fourth type count ──────────────────────────────────
#
# Addendum from the explainability repair agent. `assess_data_sufficiency`
# printed "Large sample (n=20,904). All model types are viable." into a
# manuscript whose Study Design paragraph says 6,297 observations — a fourth n
# for one study, naming no population. And `compute_dataset_profile` split the
# columns with bare `is_numeric_dtype`, which calls a bool column numeric while
# the preprocessing pipeline one-hot encodes it, producing a fourth type count.

def _outcome_missing_frame(n: int = 4000):
    rng = np.random.default_rng(7)
    outcome = np.where(rng.random(n) < 0.30, rng.integers(0, 2, n), np.nan)
    return pd.DataFrame({
        "age": rng.normal(50, 12, n),
        "smoker": rng.random(n) < 0.4,      # bool: categorical to the pipeline
        "gender": rng.choice(["m", "f"], n),
        "outcome": outcome,
    })


def test_the_sufficiency_sentence_names_the_population_its_n_counts():
    from ml.dataset_profile import compute_dataset_profile

    frame = _outcome_missing_frame()
    profile = compute_dataset_profile(
        frame, target_col="outcome", feature_cols=["age", "smoker", "gender"],
        task_type="classification")
    n_with_outcome = int(frame["outcome"].notna().sum())

    assert profile.n_analysis_rows == n_with_outcome, profile.n_analysis_rows
    narrative = profile.sufficiency_narrative
    assert f"n={n_with_outcome:,}" in narrative, (
        "the sufficiency sentence quotes a row count that is not the analysis "
        f"cohort: {narrative!r}"
    )
    assert f"n={len(frame):,}" not in narrative, (
        "the sufficiency sentence still quotes every uploaded row, including "
        f"rows in no analysis cohort: {narrative!r}"
    )
    assert "observations with a recorded outcome" in narrative, (
        f"the n names no population: {narrative!r}"
    )


def test_the_dimensionality_ratio_shares_that_population():
    """A p/n over rows the models never saw is a different study's ratio."""
    from ml.dataset_profile import compute_dataset_profile

    frame = _outcome_missing_frame()
    profile = compute_dataset_profile(
        frame, target_col="outcome", feature_cols=["age", "smoker", "gender"],
        task_type="classification")
    assert profile.p_n_ratio == pytest.approx(3 / profile.n_analysis_rows)


def test_the_profile_counts_column_types_the_way_the_pipeline_splits_them():
    from ml.dataset_profile import compute_dataset_profile
    from data_processor import get_numeric_columns

    frame = _outcome_missing_frame()
    profile = compute_dataset_profile(
        frame, target_col="outcome", feature_cols=["age", "smoker", "gender"],
        task_type="classification")
    pipeline_numeric = set(get_numeric_columns(frame))

    assert "smoker" not in pipeline_numeric, (
        "the premise moved: data_processor no longer treats a bool column as "
        "categorical, so this row needs re-reading"
    )
    assert profile.n_numeric == 1 and profile.n_categorical == 2, (
        f"the profile counts types its own way: numeric={profile.n_numeric}, "
        f"categorical={profile.n_categorical}"
    )
    assert "is_numeric_dtype(df[col])" not in _source("ml/dataset_profile.py"), (
        "the profile still splits columns with a rule the preprocessing "
        "pipeline does not use"
    )


# ── 17 · an internal preprocessing note is not a study limitation ────────────

def test_drive8_17_the_preprocessing_consistency_card_is_marked_audit_only():
    """Composition test: read the shipped page, not a transcription of it."""
    source = _source("pages/05_Preprocess.py")
    marker = 'id="preprocess_model_checks"'
    assert marker in source, (
        "the insight this row is about is gone from pages/05_Preprocess.py; "
        "re-read the finding rather than deleting this test"
    )
    block = source[source.index(marker): source.index(marker) + 700]
    assert '"audit_only": True' in block, (
        "the preprocessing-consistency card is not marked audit-only, so it "
        "reaches the Discussion's Limitations list as a study limitation"
    )


def test_drive8_17_an_audit_only_note_never_becomes_a_study_limitation():
    from utils.insight_ledger import Insight, InsightLedger

    finding = ("HISTGB_CLF: scaling robust; the recipe table does not require "
               "scaling here. LOGREG: scaling standard, as the recipe table "
               "requires.")
    kwargs = dict(
        id="preprocess_model_checks", source_page="05_Preprocess",
        category="methodology", severity="info", finding=finding,
        implication="Review that preprocessing matches each model.",
        relevant_pages=["06_Train_and_Compare"],
    )

    # Positive control: without the flag this note IS collected as a limitation.
    unflagged = InsightLedger()
    unflagged.upsert(Insight(**kwargs))
    assert any("recipe table" in text for text
               in unflagged.discussion_points_for_manuscript()["limitations"]), (
        "the mechanism this row is about no longer reproduces; the fix below "
        "would then be asserting nothing"
    )

    flagged = InsightLedger()
    flagged.upsert(Insight(metadata={"audit_only": True}, **kwargs))
    assert not any(
        "recipe table" in text for text
        in flagged.discussion_points_for_manuscript()["limitations"]), (
        "an internal preprocessing note is spliced into the study limitations"
    )


# ── 20 · multiplicity counts comparisons, not renders ────────────────────────

def _two_sample_record(test_name: str, overridden: bool) -> dict:
    """One page-09 two-sample record, as `record_statistical_test` writes it."""
    return {
        "test_name": test_name,
        "variable": "glucose by gender",
        "statistic": 4.2,
        "p_value": 1e-9,
        "parametric": overridden,
        "assumption_basis": "male: Shapiro-Wilk p=1.09e-76",
        "assumption_overridden": overridden,
    }


def test_drive8_20_an_override_rerun_is_one_comparison_not_two():
    from ml.narrative_engine import _distinct_comparisons

    records = [_two_sample_record("Mann-Whitney U", False),
               _two_sample_record("t-test (ind.)", True)]
    comparisons = _distinct_comparisons(records)
    assert len(comparisons) == 1, (
        "the assumption check's run and the author's override re-run of the "
        f"same comparison count as two tests: {comparisons}"
    )
    # The record that survives is the one the page reports.
    assert comparisons[0]["test_name"] == "t-test (ind.)"


def test_drive8_20_two_different_nulls_on_one_variable_stay_two_tests():
    """Positive control — Shapiro-Wilk and Breusch-Pagan are not one question."""
    from ml.narrative_engine import _distinct_comparisons

    records = [{"test_name": "Shapiro-Wilk", "variable": "residuals", "p_value": 0.03},
               {"test_name": "Breusch-Pagan", "variable": "residuals", "p_value": 0.12}]
    assert len(_distinct_comparisons(records)) == 2


def test_drive8_20_the_page_09_family_warning_counts_comparisons_too():
    """The on-page family-wise warning had the same double-count."""
    source = _source("pages/09_Hypothesis_Testing.py")
    block_start = source.index("# Family-wise error rate warning")
    block = source[block_start:block_start + 900]
    assert "len(_custom_tests) > 1" not in block, (
        "the family-wise warning still counts rows added to Table 1 rather "
        "than distinct comparisons"
    )
    assert "_comparisons" in block and "distinct comparisons" in block, block


def test_drive8_20_the_multiplicity_sentence_counts_distinct_comparisons():
    from utils.workflow_provenance import WorkflowProvenance
    from ml.narrative_engine import NarrativeEngine

    prov = WorkflowProvenance()
    prov.record_upload("y", "classification", ["glucose", "gender"], 6297)
    for record in (_two_sample_record("Mann-Whitney U", False),
                   _two_sample_record("t-test (ind.)", True)):
        prov.record_statistical_test(
            test_name=record["test_name"], variable=record["variable"],
            statistic=record["statistic"], p_value=record["p_value"],
            details={k: record[k] for k in
                     ("parametric", "assumption_basis", "assumption_overridden")},
        )
    paragraph = NarrativeEngine(prov, None)._gen_statistical_validation()
    assert "across the 2 tests" not in paragraph, (
        "one comparison run two ways is reported as two independent tests: "
        f"{paragraph!r}"
    )
    assert "across the 1 test reported here" in paragraph, paragraph
    # Both tests were performed and both are still named.
    assert "Mann-Whitney U" in paragraph and "t-test (ind.)" in paragraph


# ── 14/30 · one vocabulary for the improbability band ────────────────────────

_DISAVOWED = re.compile(r"NHANES reference", re.IGNORECASE)


def test_drive8_14_the_recommender_states_the_band_in_the_register_it_belongs_to():
    source = _source("ml/eda_recommender.py")
    flag = source[source.index("physio_plausibility_flags.append"):][:300]
    assert "improbability band" in flag, flag
    assert not _DISAVOWED.search(flag), (
        "the plausibility flag re-states the fact in the vocabulary "
        f"ml/physiology_reference.py disavows (`MISC-018`): {flag!r}"
    )


def test_drive8_14_the_dataset_profile_flag_uses_the_same_words():
    source = _source("ml/dataset_profile.py")
    flag = source[source.index("physio_flags.append"):][:300]
    assert "improbability band" in flag, flag
    assert not _DISAVOWED.search(flag), flag


def test_drive8_14_the_three_producers_of_this_fact_agree_word_for_word():
    """`ml/eda_actions.py` is the register; the other two must match it."""
    phrase = "values outside the NHANES improbability band"
    for path in ("ml/eda_actions.py", "ml/eda_recommender.py",
                 "ml/dataset_profile.py"):
        assert phrase in _source(path), (
            f"{path} states the improbability-band fact in different words "
            f"from the other producers of it"
        )


def test_drive8_30_the_preprocess_control_carries_the_name_the_nudge_uses():
    """`pages/02_EDA.py` sends the reader to 'plausibility filtering'."""
    eda = _source("pages/02_EDA.py")
    assert "plausibility filtering is on Preprocess" in eda, (
        "the nudge this row is about is gone; re-read the finding"
    )
    preprocess = _source("pages/05_Preprocess.py")
    labels = re.findall(r'st\.checkbox\(\s*\n?\s*"([^"]+)"', preprocess)
    assert any("Plausibility filtering" in label for label in labels), (
        "no control on pages/05_Preprocess.py is labeled with the word the "
        f"nudge sends the reader looking for; labels present: {labels}"
    )
    block_start = preprocess.index('"Plausibility filtering')
    block = preprocess[block_start:block_start + 600]
    assert not _DISAVOWED.search(block), (
        "the control's help text uses the disavowed 'NHANES reference' "
        f"vocabulary: {block!r}"
    )


# ── 28 · every Table 1 percentage carries its denominator ────────────────────

def _table1_frame() -> pd.DataFrame:
    """A categorical column with heavy missingness, like the drive's."""
    values = ["False"] * 1000 + ["True"] * 3600 + [None] * 17200
    return pd.DataFrame({"meds_chol": values,
                         "age": np.linspace(20, 80, len(values))})


def test_drive8_28_a_categorical_block_does_not_mix_denominators_silently():
    from ml.table_one import Table1Config, generate_table1

    table, metadata = generate_table1(
        _table1_frame(),
        Table1Config(categorical_vars=["meds_chol"], show_pvalues=False,
                     show_missing=True),
    )
    cells = {str(idx): str(row.iloc[0]) for idx, row in table.iterrows()}
    category_cells = [v for k, v in cells.items()
                      if k.strip() in ("False", "True")]
    missing_cell = next(v for k, v in cells.items() if k.strip().startswith("Missing"))

    assert len(category_cells) == 2, cells
    for cell in category_cells + [missing_cell]:
        assert re.fullmatch(r"\d+/\d+ \(\d+\.\d%\)", cell), (
            f"a Table 1 percentage prints without its denominator: {cell!r}"
        )

    # The category rows share ONE denominator and sum to 100% of it; the
    # Missing row is over a different one and says so.
    denominators = {cell.split("/")[1].split(" ")[0] for cell in category_cells}
    assert len(denominators) == 1, denominators
    percentages = [float(re.search(r"\((\d+\.\d)%\)", c).group(1))
                   for c in category_cells]
    assert abs(sum(percentages) - 100.0) < 0.2, percentages
    assert missing_cell.split("/")[1].split(" ")[0] != denominators.pop(), (
        "the Missing row now shares the category rows' denominator, so the "
        "distinction this row is about has been erased rather than stated"
    )
    assert metadata.get("denominator_note"), (
        "no note records which denominator each percentage uses"
    )


# ── 31 · the reproducibility manifest reports a commit ───────────────────────

def _git_info_namespace():
    return _extract_functions(
        "pages/10_Report_Export.py", ["_git_commit", "get_git_info"],
        extra_globals={"os": __import__("os"), "subprocess": subprocess,
                       "Dict": typing.Dict, "Optional": typing.Optional,
                       "_WORKING_TREE_SUFFIX": "+wt"},
    )


def test_drive8_31_the_manifest_reports_the_commit_it_was_built_at():
    namespace = _git_info_namespace()
    info = namespace["get_git_info"]()
    assert info["commit"] != "n/a", (
        "the reproducibility manifest still declines to say which code "
        "produced the numbers it describes"
    )
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=REPO, capture_output=True, text=True)
    if head.returncode == 0 and head.stdout.strip():
        assert info["commit"].startswith(head.stdout.strip()), info


def test_drive8_31_a_dirty_tree_is_stamped_rather_than_claimed_clean():
    """The L67 adjudication's `+wt`: a stamp announces its own limitation."""
    namespace = _git_info_namespace()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=REPO, capture_output=True, text=True)
    if dirty.returncode != 0:
        pytest.skip("not a git checkout")
    commit = namespace["_git_commit"]()
    assert commit is not None
    if dirty.stdout.strip():
        assert commit.endswith("+wt"), (
            "the export claims a commit a reader could check out while the "
            f"tree it was taken from has uncommitted changes: {commit!r}"
        )
    else:
        assert not commit.endswith("+wt"), commit


def test_drive8_31_absence_stays_absence_outside_a_repository():
    """An installed copy or a downloaded zip has no commit, and says so."""
    class _NoRepo:
        SubprocessError = subprocess.SubprocessError

        @staticmethod
        def run(cmd, *args, **kwargs):
            class _Result:
                returncode = 128
                stdout = ""
            return _Result()

    namespace = _extract_functions(
        "pages/10_Report_Export.py", ["_git_commit", "get_git_info"],
        extra_globals={"os": __import__("os"), "subprocess": _NoRepo,
                       "Dict": typing.Dict, "Optional": typing.Optional,
                       "_WORKING_TREE_SUFFIX": "+wt"},
    )
    assert namespace["_git_commit"]() is None, (
        "a commit is invented outside a git checkout"
    )
    assert namespace["get_git_info"]()["commit"] == "not a git checkout"


# ── 32 · p-values render a floor, never 0.0000 ───────────────────────────────

def test_drive8_32_a_tiny_p_value_renders_as_a_floor_not_as_zero():
    from ml.table_one import format_pvalue

    assert format_pvalue(1.09e-76) == "< 0.0001"
    assert format_pvalue(0.0) == "< 0.0001"
    assert format_pvalue(0.00004) == "< 0.0001"
    assert format_pvalue(0.0234) == "0.0234"
    assert format_pvalue(None) == "—"
    assert format_pvalue(float("nan")) == "—"
    assert format_pvalue(1e-9, decimals=3) == "< 0.001"


def test_drive8_32_the_hypothesis_page_prints_no_raw_four_decimal_p():
    source = _source("pages/09_Hypothesis_Testing.py")
    reader_facing = [line for line in source.splitlines()
                     if "results['p']:.4f" in line]
    assert not reader_facing, (
        "a p-value still renders through `:.4f`, which prints 0.0000 for "
        f"anything below 5e-5: {reader_facing}"
    )
    assert "format_pvalue" in source


def test_drive8_32_the_latex_statistical_validation_line_floors_its_p():
    source = _source("ml/latex_report.py")
    assert "p_value:.4f" not in source, (
        "the LaTeX Methods still renders p-values with `:.4f`"
    )


# ── 33 · one polynomial estimator ────────────────────────────────────────────

def test_drive8_33_the_polynomial_estimate_matches_what_the_button_creates():
    from sklearn.preprocessing import PolynomialFeatures

    namespace = _extract_functions("pages/03_Feature_Engineering.py",
                                   ["_poly_new_feature_count"],
                                   extra_globals={"math": math})
    estimate = namespace["_poly_new_feature_count"]

    columns = [f"c{i}" for i in range(6)]
    frame = np.random.default_rng(0).random((12, len(columns)))
    for degree in (2, 3):
        for interaction_only in (False, True):
            poly = PolynomialFeatures(degree=degree,
                                      interaction_only=interaction_only,
                                      include_bias=False).fit(frame)
            names = list(poly.get_feature_names_out(columns))
            created = [n for n in names if n not in columns]
            assert estimate(len(columns), degree, interaction_only) == len(created), (
                f"degree={degree} interaction_only={interaction_only}: the "
                f"panel's estimate disagrees with what the run creates"
            )
    assert estimate(0, 2, False) == 0


def test_drive8_33_the_panel_holds_one_estimator_and_prints_one_number():
    source = _source("pages/03_Feature_Engineering.py")
    assert "_poly_new_feature_count" in source
    # The two disagreeing inline formulas are gone.
    assert "len(numeric_features) * (len(numeric_features) + 1) // 2" not in source
    assert "n_numeric * 4" not in source
    block_start = source.index('st.metric("Estimated new features"')
    block = source[block_start - 800: block_start + 1400]
    printed = set(re.findall(r"~\{([a-z_]+):,\}", block))
    assert printed == {"est_new"}, (
        f"the panel prints more than one estimate of the same quantity: {printed}"
    )


# ── 15 · an exception is not guidance ────────────────────────────────────────

def test_drive8_15_the_interaction_error_is_not_a_truncated_exception():
    source = _source("pages/02_EDA.py")
    assert "Interaction detection" in source
    block_start = source.index("Interaction detection")
    # Walk back to the `except` that guards it.
    guard = source[max(0, block_start - 400): block_start + 800]
    assert "str(e)[:80]" not in guard, (
        "pages/02_EDA.py still prints a raw sklearn exception truncated to 80 "
        "characters — mid-word — as user guidance"
    )
    assert "expander" in guard, (
        "the exception is no longer truncated but it is also no longer kept "
        "whole anywhere the user can read it"
    )
    assert "did not run" in guard, (
        "the message does not name the operation that failed"
    )


# ── 29 · a resolution that restates the finding is not a resolution ──────────

def test_drive8_29_an_echo_resolution_is_not_printed_twice():
    from utils.insight_ledger import Insight, resolution_text

    echoed = Insight(id="method_explainability", source_page="07_Explainability",
                     category="explainability", severity="info",
                     finding="Ran permutation_importance on 3 models",
                     implication="Logged methodology decision",
                     resolved=True,
                     resolved_by="Ran permutation_importance on 3 models")
    assert resolution_text(echoed) == "", (
        "a coach card renders '~~X~~ → X', restating the finding as its own "
        "resolution"
    )

    real = Insight(id="eda_skew", source_page="02_EDA", category="distribution",
                   severity="warning", finding="3 features are right-skewed",
                   implication="May affect linear models",
                   resolved=True, resolved_by="Applied log1p transform")
    assert resolution_text(real) == "Applied log1p transform"


def test_drive8_29_an_action_with_no_subject_says_so():
    from utils.insight_ledger import name_empty_slots

    assert name_empty_slots("Ran  on 3 models") == "Ran (not recorded) on 3 models"
    # Typographic double spaces after sentence punctuation are left alone.
    assert name_empty_slots("Done.  Next step") == "Done.  Next step"
    assert name_empty_slots("") == ""
    assert name_empty_slots("Ran SHAP on 3 models") == "Ran SHAP on 3 models"


def test_drive8_29_the_render_paths_all_go_through_the_same_decision():
    for path in ("utils/coaching_ui.py", "pages/10_Report_Export.py"):
        source = _source(path)
        assert "resolution_text" in source or "name_empty_slots" in source, (
            f"{path} renders a resolution without asking whether it is one"
        )
    assert "~~{ins.finding}~~ → {ins.resolved_by}" not in _source("utils/coaching_ui.py")
