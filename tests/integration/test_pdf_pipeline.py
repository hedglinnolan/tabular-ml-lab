"""Integration test for the manuscript -> pdflatex pipeline.

Closes the audit-flagged gap that nothing previously exercised the
``pdflatex`` subprocess used by Page 10 to render the manuscript PDF.

The test calls the same code path the live app uses:
``ml.latex_report.generate_latex_report`` to build the LaTeX source, then
``ml.latex_report.compile_latex_to_pdf`` to actually invoke pdflatex.

Auto-skips when ``pdflatex`` is not on PATH so the test is portable
across dev machines that don't have a TeX install.
"""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from ml.latex_report import compile_latex_to_pdf, generate_latex_report

pytestmark = pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex not installed -- skipping PDF round-trip test",
)


def _build_synthetic_args() -> dict:
    """Build a realistic-but-tiny set of args for generate_latex_report."""
    table1 = pd.DataFrame({
        "Variable": ["age", "bmi", "glucose"],
        "Overall (n=100)": ["50.2 ± 12.3", "27.5 ± 4.1", "98.4 ± 14.2"],
    })
    model_results = {
        "Ridge": {
            "metrics": {"RMSE": 12.5, "MAE": 9.8, "R2": 0.42},
        },
        "Random Forest": {
            "metrics": {"RMSE": 11.1, "MAE": 8.6, "R2": 0.51},
        },
    }
    return dict(
        title="Smoke Test Manuscript",
        authors="Test",
        affiliation="Test",
        abstract="Synthetic abstract used only to verify pdflatex compiles.",
        methods_section="Synthetic methods text. Models trained on synthetic data.",
        table1_df=table1,
        model_results=model_results,
        bootstrap_results=None,
        task_type="regression",
        feature_names=["age", "bmi"],
        target_name="glucose",
        n_total=100,
        n_train=70,
        n_val=15,
        n_test=15,
        data_config={"target_col": "glucose", "feature_cols": ["age", "bmi"]},
        calibration_text="",
        limitations="Synthetic data; no clinical inference intended.",
    )


def test_latex_source_contains_required_sections():
    """Sanity check on the LaTeX source before we ask pdflatex to consume it."""
    latex = generate_latex_report(**_build_synthetic_args())
    assert r"\documentclass" in latex
    assert r"\begin{document}" in latex
    assert r"\end{document}" in latex
    assert "Smoke Test Manuscript" in latex


def test_pdf_round_trip():
    """End-to-end: generate LaTeX, compile via pdflatex, assert valid PDF."""
    latex = generate_latex_report(**_build_synthetic_args())
    pdf_bytes = compile_latex_to_pdf(latex, timeout=60)
    assert pdf_bytes is not None, (
        "compile_latex_to_pdf returned None -- pdflatex compile failed. "
        "Check that texlive-latex-base, texlive-latex-recommended, and "
        "texlive-fonts-recommended are installed."
    )
    assert pdf_bytes[:4] == b"%PDF", (
        f"Expected PDF magic bytes, got {pdf_bytes[:8]!r}"
    )
    # A real manuscript should be at least a few KB. Anything smaller usually
    # means an empty page slipped through.
    assert len(pdf_bytes) > 2048, (
        f"PDF is suspiciously small ({len(pdf_bytes)} bytes) -- likely empty"
    )


def test_compile_latex_to_pdf_returns_none_for_garbage():
    """Malformed LaTeX should not raise -- it should return None."""
    pdf_bytes = compile_latex_to_pdf(r"\not real latex at all \end")
    assert pdf_bytes is None
