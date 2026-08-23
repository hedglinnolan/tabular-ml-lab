"""CONTRACT-021 / STATE-031 / STATE-008 / RECORD-007: the manuscript reports
what the analysis did, or it refuses.

The Report Export page used to RE-DERIVE the numbers it printed. Three separate
re-derivations, one per finding:

- the analysis cohort (`_build_analysis_cohort_df`) recomputed the target-trim
  quantiles over the whole current frame and dropped both tails, while
  `ml/splits.make_split` had computed the trim from TRAINING rows only and
  exempted every sealed test row. Two populations, both well-formed, and the
  one that reached Table 1 and the manuscript N was the one no model was fitted
  on (`CONTRACT-021`, `STATE-031`).
- the preprocessing recipe (`ml.pipeline.get_pipeline_recipe`) introspected the
  fitted objects with a branch per option it happened to know, so Yeo-Johnson,
  min-max scaling, ordinal and target encoding and the entire passthrough block
  printed nothing, and MAD capping printed a hardcoded `3` for a threshold that
  was 3.5 by default (`STATE-008`).
- the .tex sectioning (`ml.latex_report`) split the compiled draft at the first
  `## Results`, so the Discussion — with every `[AUTHOR REQUIRED]` scaffold the
  workflow generated — landed in a variable only used when there were no model
  results, i.e. never in a real run (`RECORD-007`).

The rule these share: a decision implemented twice is a decision that can be
made two ways, and the paper gets whichever one runs last.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.splits import (SplitIdentityError, SplitSpec, make_split,
                       resolve_analysis_cohort)

REPO = Path(__file__).resolve().parent.parent


def _skewed_study(n: int = 200) -> pd.DataFrame:
    """A regression cohort with a long right tail, so a trim has work to do."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.normal(50, 12, n),
            "bmi": rng.normal(27, 5, n),
            "glucose": np.concatenate(
                [rng.normal(100, 10, n - 10), rng.normal(400, 20, 10)]),
        },
        index=pd.RangeIndex(n),
    )


def _trimmed_split(df: pd.DataFrame, lockbox_labels):
    spec = SplitSpec(target_trim_enabled=True, target_trim_lower=0.02,
                     target_trim_upper=0.98)
    return make_split(df, ["age", "bmi"], "glucose", "regression", spec,
                      lockbox_labels)


def _page10_source(func: str) -> str:
    """The body of one function in the Report Export page.

    The page is a Streamlit script and cannot be imported; its cohort rule is
    pinned at the source level, which is also where the defect lived.
    """
    src = (REPO / "pages" / "10_Report_Export.py").read_text()
    start = src.index(f"def {func}(")
    nxt = src.find("\ndef ", start + 1)
    return src[start:nxt if nxt != -1 else len(src)]


# ── CONTRACT-021 · the cohort is the one the split realized ──────────────

class TestContract021AnalysisCohortIsTheRealizedOne:

    def test_the_recorded_cohort_keeps_every_sealed_test_row(self):
        """The split exempts test rows from the trim. Re-deriving does not."""
        study = _skewed_study()
        lockbox = list(study.index[:40])
        split = _trimmed_split(study, lockbox)
        assert split.lockbox_applied, "fixture must take the lockbox branch"
        assert split.n_trimmed_rows > 0, "fixture must actually trim"

        realized = set(split.analysis_labels)
        assert set(split.test_labels) <= realized
        # No sealed row was trimmed: the whole lockbox is still in the cohort.
        assert set(lockbox) & set(study.index) == set(lockbox)
        assert set(split.test_labels) == set(lockbox)

        # The old page-10 rule, verbatim, so the fixture is known to bite.
        target = pd.to_numeric(study["glucose"], errors="coerce")
        q_lo = float(target.quantile(0.02))
        q_hi = float(target.quantile(0.98))
        rederived = set(study.index[(target >= q_lo) & (target <= q_hi)])
        assert rederived != realized, (
            "fixture no longer diverges, so it proves nothing")
        # And the divergence is in the direction that matters: rows the models
        # were evaluated on that the re-derived Table 1 would have excluded.
        assert set(split.test_labels) - rederived

    def test_the_cohort_reads_back_by_label_not_by_rule(self):
        study = _skewed_study()
        split = _trimmed_split(study, list(study.index[:40]))

        cohort = resolve_analysis_cohort(study, split.analysis_labels)
        assert set(cohort.index) == set(split.analysis_labels)
        assert len(cohort) == sum(split.sizes.values())
        # Frame order, which is the order Table 1 describes the rows in.
        assert list(cohort.index) == [i for i in study.index
                                      if i in set(split.analysis_labels)]

    def test_the_realized_trim_thresholds_are_recorded(self):
        """The thresholds the split used, kept where a report can read them."""
        study = _skewed_study()
        split = _trimmed_split(study, list(study.index[:40]))

        assert split.trim_record is not None
        assert split.trim_record["basis"] == "training rows only"
        assert split.trim_record["test_rows_exempt"] is True
        assert split.trim_record["n_trimmed"] == split.n_trimmed_rows
        lo, hi = split.trim_record["thresholds"]
        assert lo < hi

        # No trim, no record — the field never claims a trim that did not run.
        plain = make_split(study, ["age", "bmi"], "glucose", "regression",
                           SplitSpec(), None)
        assert plain.trim_record is None

    def test_page10_no_longer_recomputes_the_trim(self):
        body = _page10_source("_build_analysis_cohort_df")
        assert "quantile(" not in body, (
            "the report recomputes the trim thresholds it should be reading")
        assert "target_trim_enabled" not in body
        assert "resolve_analysis_cohort" in body, (
            "the report must read the cohort the split recorded")
        assert "_row_labels" in body


# ── STATE-031 · a cohort that cannot be read back is refused ─────────────

class TestState031ExportRefusesRatherThanRederiving:

    def test_no_recorded_split_means_no_cohort(self):
        with pytest.raises(SplitIdentityError, match="No analysis cohort row labels"):
            resolve_analysis_cohort(_skewed_study(), [])
        with pytest.raises(SplitIdentityError, match="no active dataset"):
            resolve_analysis_cohort(None, [0, 1, 2])

    def test_a_row_set_change_after_the_split_is_refused_not_absorbed(self):
        """Page 05's plausibility filter writes a new frame with no reset.

        Re-deriving absorbed that silently: different rows, different quantiles,
        a different Table 1 beside Results computed on the other population.
        """
        study = _skewed_study()
        split = _trimmed_split(study, list(study.index[:40]))

        filtered = study.drop(index=list(split.train_labels[:5]))
        with pytest.raises(SplitIdentityError) as exc:
            resolve_analysis_cohort(filtered, split.analysis_labels)
        assert "Prepare Splits" in str(exc.value)

    def test_page10_surfaces_the_refusal_instead_of_a_number(self):
        body = _page10_source("_build_analysis_cohort_df")
        assert "SplitIdentityError" in body
        readiness = _page10_source("_table1_readiness")
        assert "cohort_refusal" in readiness, (
            "the readiness caption must report the refusal, not an N derived "
            "some other way")
        table1 = _page10_source("_build_manuscript_table1")
        assert "cohort_unavailable" in table1


# ── STATE-008 · the recipe names every step, or names it as unrecorded ───

class TestState008RecipeOmitsNothing:

    @staticmethod
    def _fitted(**kwargs):
        from ml.pipeline import build_preprocessing_pipeline
        rng = np.random.default_rng(3)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60),
            "c": rng.choice(["x", "y"], 60), "eng": rng.normal(0, 1, 60),
        })
        y = rng.normal(0, 1, 60)
        pipe = build_preprocessing_pipeline(**kwargs)
        pipe.fit(X, y)
        return pipe

    def test_every_option_reaches_the_methods_section(self):
        from ml.pipeline import get_pipeline_recipe

        recipe = get_pipeline_recipe(self._fitted(
            numeric_features=["a", "b"], categorical_features=["c"],
            numeric_imputation="median", numeric_scaling="minmax",
            numeric_power_transform="yeo-johnson",
            categorical_encoding="ordinal",
            passthrough_numeric_features=["eng"],
        ))
        assert "yeo-johnson" in recipe            # was invisible
        assert "Min-max scaling" in recipe        # was invisible
        assert "Ordinal encoding" in recipe       # was invisible
        assert "passthrough" in recipe.lower()    # whole block was invisible
        assert "NaN guard" in recipe              # a second imputation, unnamed

    def test_target_encoding_is_never_silently_omitted(self):
        """Encoding a predictor with the OUTCOME is the most leakage-sensitive
        choice on the Preprocess page; the Methods may not lose it."""
        pytest.importorskip("sklearn.preprocessing", reason="TargetEncoder")
        from ml.pipeline import _HAS_TARGET_ENCODER, get_pipeline_recipe
        if not _HAS_TARGET_ENCODER:
            pytest.skip("scikit-learn without TargetEncoder")

        recipe = get_pipeline_recipe(self._fitted(
            numeric_features=["a"], categorical_features=["c"],
            categorical_encoding="target", categorical_target_type="continuous",
        ))
        assert "Target encoding" in recipe
        assert "uses the outcome" in recipe

    def test_the_outlier_threshold_printed_is_the_one_applied(self):
        from ml.pipeline import get_pipeline_recipe

        for threshold in (2.0, 5.0):
            recipe = get_pipeline_recipe(self._fitted(
                numeric_features=["a", "b"], categorical_features=[],
                numeric_outlier_treatment="mad",
                numeric_outlier_params={"threshold": threshold},
            ))
            assert f"{threshold}× MAD" in recipe, recipe
        # The default is 3.5, and the recipe used to assert 3.
        default_recipe = get_pipeline_recipe(self._fitted(
            numeric_features=["a", "b"], categorical_features=[],
            numeric_outlier_treatment="mad",
        ))
        assert "3.5× MAD" in default_recipe
        assert "3× MAD" not in default_recipe

    def test_percentile_capping_reports_its_bounds(self):
        from ml.pipeline import get_pipeline_recipe

        recipe = get_pipeline_recipe(self._fitted(
            numeric_features=["a", "b"], categorical_features=[],
            numeric_outlier_treatment="percentile",
            numeric_outlier_params={"lower_q": 0.05, "upper_q": 0.95},
        ))
        assert "Percentile clip (5th–95th)" in recipe

    def test_an_imputer_without_a_strategy_does_not_crash_the_recipe(self):
        from ml.pipeline import _HAS_ITERATIVE, get_pipeline_recipe
        if not _HAS_ITERATIVE:
            pytest.skip("scikit-learn without IterativeImputer")

        recipe = get_pipeline_recipe(self._fitted(
            numeric_features=["a", "b"], categorical_features=[],
            numeric_imputation="iterative",
        ))
        assert "Iterative imputation" in recipe

    def test_an_unknown_step_is_named_not_dropped(self):
        """The failure mode was structural: an option added to the builder was
        absent from the recipe until someone remembered to add a branch."""
        from sklearn.preprocessing import QuantileTransformer

        from ml.pipeline import get_pipeline_recipe

        pipe = self._fitted(numeric_features=["a", "b"], categorical_features=[])
        numeric_pipe = pipe.named_steps["preprocessor"].transformers_[0][1]
        numeric_pipe.steps.append(("mystery_transform", QuantileTransformer(n_quantiles=10)))

        recipe = get_pipeline_recipe(pipe)
        assert "unrecorded step: mystery_transform (QuantileTransformer)" in recipe

    def test_row_filtering_is_part_of_the_recipe_in_the_export(self):
        page10 = (REPO / "pages" / "10_Report_Export.py").read_text()
        assert page10.count("get_pipeline_recipe(") >= 2
        assert not re.search(r"get_pipeline_recipe\(pl\)", page10), (
            "the export drops the 'rows filtered' line by omitting the mode")
        assert "plausibility_mode=cfg.get" in page10


# ── RECORD-007 · the .tex carries what the .md carries ───────────────────

_DRAFT = """> **How to read this draft.** Compiled from the recorded workflow.

## Methods

### Study Design
A retrospective cohort of adults was analyzed.

## Results

### Model Performance
Ridge regression reached R2 = -0.03 on the held-out test set.

## Discussion

### Principal Findings
Ridge regression reached R2 = -0.03, which is below a mean-only baseline: the
model did not explain outcome variance in held-out data.

### Strengths and Limitations
[AUTHOR REQUIRED - the analysis was run in exploratory mode: the held-out test
set was not quarantined from feature engineering and selection.]
"""


def _tex(draft: str = _DRAFT, **kwargs):
    from ml.latex_report import generate_latex_report
    params = dict(
        methods_section=draft,
        model_results={"ridge": {"metrics": {"RMSE": 1.2, "R2": -0.03}}},
        task_type="regression", n_total=200, n_train=140, n_val=30, n_test=30,
    )
    params.update(kwargs)
    return generate_latex_report(**params)


class TestRecord007CompiledSectionsSurviveIntoTheTex:

    def test_the_discussion_is_not_dropped_when_there_are_model_results(self):
        """The `elif draft_results:` branch was dead in every real run."""
        source = _tex()
        assert "below a mean-only baseline" in source, (
            "the compiled Discussion was replaced by generic placeholders")
        assert "[AUTHOR REQUIRED" in source, (
            "the author-input count the page advertises was untrue of the .tex")
        assert "exploratory mode" in source

    def test_the_results_prose_is_omitted_out_loud_and_keeps_author_input(self):
        """The metrics table restates the draft's Results numbers, so the prose
        stays out — but the omission is disclosed, and an author-input scaffold
        inside it is carried rather than dropped."""
        source = _tex()
        assert "tab:model_performance" in source, (
            "the structured metrics table must still be generated")
        assert "not reproduced here" in source, (
            "the .tex drops the compiled Results narrative without saying so")

        with_scaffold = _DRAFT.replace(
            "Ridge regression reached R2 = -0.03 on the held-out test set.",
            "Ridge regression reached R2 = -0.03 on the held-out test set.\n\n"
            "[AUTHOR REQUIRED - state the clinically meaningful error margin.]")
        carried = _tex(with_scaffold)
        assert "clinically meaningful error margin" in carried
        assert "Outstanding author input" in carried

    def test_the_methods_are_still_converted(self):
        source = _tex()
        assert "retrospective cohort" in source
        assert r"\section{Methods}" in source

    def test_a_section_with_no_home_is_printed_not_lost(self):
        draft = _DRAFT + """
## Data Availability

The dataset is available on request from the corresponding author.
"""
        source = _tex(draft)
        assert "Unmapped Compiled Draft Sections" in source
        assert "available on request" in source

    def test_the_placeholder_discussion_is_only_a_fallback(self):
        """Without a compiled Discussion the skeleton still has one — and with
        one, the two do not both print under the same subsection headings."""
        no_discussion = _DRAFT.split("## Discussion")[0]
        fallback = _tex(no_discussion)
        assert "[PLACEHOLDER: Summarize the main results" in fallback

        compiled = _tex()
        assert compiled.count(r"\subsection{Principal Findings}") <= 1

    def test_an_explicit_limitations_argument_is_still_appended(self):
        source = _tex(limitations="Single-centre data; no external validation.")
        assert "Single-centre data" in source

    def test_the_split_is_structural_not_a_first_match_on_results(self):
        from ml.latex_report import _convert_markdown_to_latex

        parts = _convert_markdown_to_latex(_DRAFT)
        assert "retrospective" in parts["methods"]
        assert "Ridge regression reached" in parts["results"]
        assert "mean-only baseline" in parts["discussion"]
        # The Discussion is not swept into Results, which is the whole defect.
        assert "mean-only baseline" not in parts["results"]
        assert parts["unmapped"] == []
