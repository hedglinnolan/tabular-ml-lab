"""Tests for Table 1 generation."""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.table_one import (
    Table1Config,
    generate_feature_table1,
    generate_table1,
    partition_table1_variables,
    table1_to_csv,
    table1_to_latex,
)


@pytest.fixture
def sample_df():
    """Create a sample clinical dataset."""
    np.random.seed(42)
    n = 200
    group = np.random.choice(['Treatment', 'Control'], size=n)
    age = np.random.normal(55, 10, n)
    bmi = np.random.normal(27, 4, n)
    glucose = np.where(group == 'Treatment',
                       np.random.normal(100, 20, n),
                       np.random.normal(120, 25, n))
    sex = np.random.choice(['Male', 'Female'], size=n, p=[0.6, 0.4])

    df = pd.DataFrame({
        'Group': group,
        'Age': age,
        'BMI': bmi,
        'Glucose': glucose,
        'Sex': sex,
    })
    # Add some missing values
    df.loc[5:10, 'BMI'] = np.nan
    df.loc[15:17, 'Sex'] = np.nan
    return df


def test_basic_table1(sample_df):
    """Test basic Table 1 generation without grouping."""
    config = Table1Config(
        continuous_vars=['Age', 'BMI', 'Glucose'],
        categorical_vars=['Sex'],
    )
    table, metadata = generate_table1(sample_df, config)
    assert table is not None
    assert len(table) > 0
    assert f"Overall (N={len(sample_df)})" in table.columns


def test_stratified_table1(sample_df):
    """Test Table 1 with grouping variable."""
    config = Table1Config(
        grouping_var='Group',
        continuous_vars=['Age', 'BMI', 'Glucose'],
        categorical_vars=['Sex'],
        show_pvalues=True,
    )
    table, metadata = generate_table1(sample_df, config)
    assert "P-value" in table.columns
    assert len(metadata["tests_used"]) > 0


def test_smd(sample_df):
    """Test SMD calculation."""
    config = Table1Config(
        grouping_var='Group',
        continuous_vars=['Glucose'],
        show_smd=True,
    )
    table, metadata = generate_table1(sample_df, config)
    assert "SMD" in table.columns


def test_missing_counts(sample_df):
    """Test that missing data is reported."""
    config = Table1Config(
        continuous_vars=['BMI'],
        show_missing=True,
    )
    table, metadata = generate_table1(sample_df, config)
    # Should have a "Missing" row
    assert any("Missing" in str(idx) for idx in table.index)


def test_csv_export(sample_df):
    """Test CSV export."""
    config = Table1Config(continuous_vars=['Age'])
    table, _ = generate_table1(sample_df, config)
    csv = table1_to_csv(table)
    assert isinstance(csv, str)
    assert len(csv) > 0


def test_latex_export(sample_df):
    """Test LaTeX export."""
    config = Table1Config(continuous_vars=['Age'])
    table, _ = generate_table1(sample_df, config)
    latex = table1_to_latex(table)
    assert "\\begin{table}" in latex


def test_empty_vars():
    """Test with no variables specified."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    config = Table1Config()
    table, metadata = generate_table1(df, config)
    assert len(table) == 0


def test_partition_table1_variables_preserves_feature_order():
    df = pd.DataFrame({
        'age': [50, 60],
        'sex': ['F', 'M'],
        'bmi': [25.0, 31.2],
        'group': ['A', 'B'],
    })

    continuous, categorical = partition_table1_variables(
        df,
        ['sex', 'age', 'bmi', 'group'],
        grouping_var='group',
    )

    assert continuous == ['age', 'bmi']
    assert categorical == ['sex']


def test_generate_feature_table1_uses_requested_final_features():
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'age': [50, 60, 55, 65],
        'sex': ['F', 'M', 'F', 'M'],
        'bmi': [25.0, 31.2, 27.5, 29.4],
        'glucose': [100, 110, 120, 130],
    })

    table, metadata, config = generate_feature_table1(
        df,
        ['age', 'sex', 'bmi'],
        grouping_var='group',
        show_pvalues=False,
        show_smd=False,
        show_missing=False,
    )

    assert config.continuous_vars == ['age', 'bmi']
    assert config.categorical_vars == ['sex']
    assert "Overall (N=4)" in table.columns
    assert any(str(idx).startswith('age,') for idx in table.index)
    assert any(str(idx).startswith('sex,') for idx in table.index)
    assert not any('glucose' in str(idx).lower() for idx in table.index)
