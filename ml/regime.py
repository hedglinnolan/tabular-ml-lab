"""
Dataset regime detection — drives adaptive layout decisions.

Computed once on page load. All UI components read the regime
to decide what to render (full heatmap vs top-N list, etc).
"""
from dataclasses import dataclass
from typing import List, Literal, Optional
import pandas as pd
import numpy as np


@dataclass
class DatasetRegime:
    """Describes the shape of the dataset for adaptive UI decisions.
    
    Computed from the active DataFrame + feature/target configuration.
    Immutable for the duration of a page render.
    """
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_missing_cols: int          # columns with any missing values
    n_high_missing_cols: int     # columns with >5% missing
    has_target: bool
    target_type: Optional[str]   # "numeric", "categorical", None

    # -- Feature regime ----------------------------------------------------

    @property
    def feature_regime(self) -> Literal["narrow", "medium", "wide", "ultra_wide"]:
        """How many features the dataset has — drives gallery/matrix decisions."""
        if self.n_features <= 15:
            return "narrow"
        elif self.n_features <= 50:
            return "medium"
        elif self.n_features <= 200:
            return "wide"
        else:
            return "ultra_wide"

    # -- Row regime --------------------------------------------------------

    @property
    def row_regime(self) -> Literal["tiny", "standard", "large", "massive"]:
        """How many rows — drives sampling/plotting decisions."""
        if self.n_rows < 100:
            return "tiny"
        elif self.n_rows < 10_000:
            return "standard"
        elif self.n_rows < 100_000:
            return "large"
        else:
            return "massive"

    # -- Derived properties ------------------------------------------------

    @property
    def needs_sampling(self) -> bool:
        """Whether scatter plots should sample data."""
        return self.row_regime in ("large", "massive")

    @property
    def sample_size(self) -> int:
        """Recommended sample size for scatter plots."""
        if self.row_regime == "massive":
            return 5_000
        elif self.row_regime == "large":
            return 5_000
        return self.n_rows  # no sampling needed

    @property
    def show_full_corr_matrix(self) -> bool:
        """Whether to show full NxN correlation heatmap."""
        return self.feature_regime in ("narrow", "medium")

    @property
    def show_macro_shape(self) -> bool:
        """Whether to show PCA/UMAP/TDA section."""
        return self.feature_regime != "narrow"

    @property
    def gallery_page_size(self) -> int:
        """Number of feature charts per page in distribution gallery."""
        return 9  # 3×3 grid

    @property
    def show_sample_size_warning(self) -> bool:
        """Whether to warn about small sample size."""
        return self.row_regime == "tiny"

    @property
    def use_hexbin(self) -> bool:
        """Whether scatter plots should use hexbin instead of points."""
        return self.row_regime == "massive"

    @property
    def distribution_mode(self) -> Literal["gallery", "summary"]:
        """How to show feature distributions."""
        if self.feature_regime == "ultra_wide":
            return "summary"  # summary-of-summaries view
        return "gallery"  # small multiples grid

    @property
    def corr_top_n(self) -> int:
        """How many correlation pairs to show in list view."""
        if self.feature_regime == "wide":
            return 30
        elif self.feature_regime == "ultra_wide":
            return 50
        return 0  # not used when showing full matrix

    @property
    def target_relationship_top_n(self) -> int:
        """How many features to auto-show in target relationship gallery."""
        if self.feature_regime == "ultra_wide":
            return 10
        return 0  # show all (paginated)

    @property
    def macro_shape_tiers(self) -> List[str]:
        """Which macro-shape views to offer."""
        if self.feature_regime == "narrow":
            return []
        elif self.feature_regime == "medium":
            return ["pca"]
        elif self.feature_regime == "wide":
            return ["pca", "umap"]
        else:
            return ["pca", "umap", "persistence", "mapper"]

    # -- Description -------------------------------------------------------

    def describe(self) -> str:
        """Human-readable description of the regime."""
        parts = []
        parts.append(f"{self.n_rows:,} rows × {self.n_features} features")
        parts.append(f"({self.n_numeric} numeric, {self.n_categorical} categorical)")
        parts.append(f"Feature regime: {self.feature_regime}")
        parts.append(f"Row regime: {self.row_regime}")
        if self.n_high_missing_cols > 0:
            parts.append(f"{self.n_high_missing_cols} columns with >5% missing")
        return " · ".join(parts)


def detect_regime(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
) -> DatasetRegime:
    """Detect the dataset regime from the active DataFrame.
    
    Args:
        df: Active DataFrame (may include target + feature columns)
        feature_cols: List of feature column names (excludes target)
        target_col: Target column name, or None
        
    Returns:
        DatasetRegime instance
    """
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    datetime_cols = df[feature_cols].select_dtypes(
        include=["datetime64", "datetimetz"]
    ).columns.tolist()

    missing_counts = df[feature_cols].isnull().sum()
    n_missing_cols = int((missing_counts > 0).sum())
    n_high_missing = int((missing_counts / max(len(df), 1) > 0.05).sum())

    has_target = target_col is not None and target_col in df.columns
    target_type = None
    if has_target:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            target_type = "numeric"
        else:
            target_type = "categorical"

    return DatasetRegime(
        n_rows=len(df),
        n_features=len(feature_cols),
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        n_datetime=len(datetime_cols),
        n_missing_cols=n_missing_cols,
        n_high_missing_cols=n_high_missing,
        has_target=has_target,
        target_type=target_type,
    )
