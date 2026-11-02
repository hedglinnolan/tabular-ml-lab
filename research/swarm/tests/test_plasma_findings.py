"""The plasma fixture's findings judge columns on its participants, not its pooled-QC injections
(the findings-rows package; ``turbotab/core/tests/test_findings_read_the_participants.py`` holds
the constructed tables).

The fixture's 28 pooled QCs (``generators.panel_layout``'s ``qc``, by construction) carry no age,
sex, BMI or outcome. Read over all 128 rows, `sex` routed to the missing-values question as "28 of
128 blank; a blank may mean not asked" and the left-censoring finding named `age`, `bmi` and the
outcome `disease` among 39 censored columns. On the 100 participants, counted here with pandas,
`sex` has no blank and 36 columns have one, every one a feature: those are what the findings say,
and the left-censoring detail's numbers are the 150 features' alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import generators as G  # noqa: E402


@pytest.fixture(scope="module")
def plasma(tmp_path_factory):
    """The fixture at its committed seed, as a CSV reader reads it, its QC rows by construction,
    and its findings under the metabolomics lens with `disease` the outcome (the pilot's)."""
    from turbotab.core.decisions import ProjectState
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.tests.stage_harness import Ingested

    folder = tmp_path_factory.mktemp("plasma")
    path = folder / G.PANEL_FILE
    G.metabolite_panel().to_csv(path, index=False)
    df = pd.read_csv(path)
    qc = G.panel_layout()["qc"].to_numpy()
    table = Ingested(path, folder / "ingested")
    artifact = table.run(findings_stage, ProjectState(lens=["metabolomics"],
                                                      target=G.PANEL_OUTCOME))
    return df, qc, {f["id"]: f for f in artifact["findings"]}


def test_sex_blank_only_on_the_qc_rows_reads_as_two_level_text(plasma):
    df, qc, found = plasma
    assert (int(df.loc[qc, "sex"].isna().sum()), int(df.loc[~qc, "sex"].isna().sum())) == (28, 0)
    # The rows set aside are the ones the pooled-QC finding's exclusion names: the 28 QCs.
    exclusion = next(o for o in found["pack::metabolomics::pooled_qc"]["repairs"]
                     if o["key"] == "exclude_rows")["decision"]["params"]
    named = df[exclusion["column"]].astype(str).isin(exclusion["levels"]).to_numpy()
    assert (named == qc).all()
    sex = found["binary_text__sex"]
    assert sex["routes_to"] is None
    assert sex["summary"].startswith("`sex` is two-level text (")
    assert sex["summary"].endswith("which level counts as 1 is not asked yet.")
    assert [o["key"] for o in sex["repairs"]] == ["level", "level"]


def test_left_censoring_names_only_the_features_blank_among_the_participants(plasma):
    df, qc, found = plasma
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    among = {c for c in numeric if df.loc[~qc, c].isna().any()}
    assert len(among) == 36 and all(c.startswith("mz_") for c in among)
    assert {"age", "bmi", G.PANEL_OUTCOME} <= {c for c in numeric if df[c].isna().any()}
    censored = found["pack::metabolomics::left_censored"]
    assert set(censored["censored_columns"]) == among


def test_the_left_censoring_detail_says_it_reads_the_100_participants(plasma):
    """The detail's numbers are the 150 features' (``generators.PANEL``'s ``features``, the `mz_`
    columns by construction) on the 100 rows outside the 28 QCs (pandas, scipy): 150 features, a
    correlation of -0.74, 36 features with a blank and a highest rate of 86%. It names those rows,
    since read on all 128 rows the highest rate is another number. Read with `run_order`, `age`,
    `bmi` and the outcome among its columns, it said 154 features and -0.68."""
    import numpy as np
    from scipy import stats

    df, qc, found = plasma
    features = [c for c in df.columns if c.startswith("mz_")]
    assert len(features) == G.PANEL["features"] == 150

    def read_on(rows: pd.DataFrame) -> str:
        rate, mean = rows[features].isna().mean(), rows[features].mean()
        usable = [c for c in features if pd.notna(mean[c]) and mean[c] > 0]
        rho = stats.spearmanr(rate[usable].to_numpy(), np.log(mean[usable].to_numpy())).statistic
        blank = [c for c in features if rate[c] > 0]
        return (f"{len(usable):,} features, a feature's missing rate tracks its abundance rank at "
                f"a rank correlation of {rho:.2f}. {len(blank):,} features have blanks; the "
                f"highest rate is {rate[blank].max():.0%}, on one of the least abundant.")

    assert int((~qc).sum()) == 100
    assert read_on(df.loc[~qc]) == (
        "150 features, a feature's missing rate tracks its abundance rank at a rank correlation "
        "of -0.74. 36 features have blanks; the highest rate is 86%, on one of the least abundant.")
    assert read_on(df.loc[~qc]) != read_on(df)
    detail = found["pack::metabolomics::left_censored"]["detail"]
    assert detail == f"On the 100 non-QC rows, across {read_on(df.loc[~qc])}"
