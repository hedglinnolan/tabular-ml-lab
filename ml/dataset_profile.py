"""
Dataset Profile: Computes comprehensive dataset diagnostics for intelligent model coaching.

This module provides the foundation for all coach recommendations by analyzing:
- Dataset dimensions (n rows, p features)
- Feature types (numeric vs categorical)
- Missingness patterns
- Target characteristics (type, balance)
- Cardinality for categoricals
- Outlier detection for numerics
- Data sufficiency indicators
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pandas.api import types as _pdt
from enum import Enum
from ml.clinical_units import infer_unit
from ml.physiology_reference import load_reference_bundle, match_variable_key, get_improbability_band
from ml.outliers import detect_outliers


class DataSufficiencyLevel(Enum):
    """Indicates how sufficient the data is for various model types."""
    ABUNDANT = "abundant"      # Plenty of data for any model
    ADEQUATE = "adequate"      # Sufficient for most models
    LIMITED = "limited"        # May constrain complex models
    SCARCE = "scarce"          # Strong regularization needed
    CRITICAL = "critical"      # Only simplest models viable


class WarningLevel(Enum):
    """Warning severity levels."""
    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DataWarning:
    """A warning or caution flag about the dataset."""
    category: str  # e.g., "sample_size", "imbalance", "missingness", "dimensionality"
    level: WarningLevel
    short_message: str  # Brief message for badges/tags
    detailed_message: str  # Full explanation for expandable sections
    affected_models: List[str] = field(default_factory=list)  # Model families affected
    suggested_actions: List[str] = field(default_factory=list)  # What to do about it


@dataclass
class FeatureProfile:
    """Profile for a single feature."""
    name: str
    dtype: str
    is_numeric: bool
    is_categorical: bool
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_ratio: float  # unique_count / n
    
    # Numeric-specific
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    has_outliers: bool = False
    outlier_count: int = 0
    outlier_rate: float = 0.0
    
    # Categorical-specific
    is_high_cardinality: bool = False
    top_categories: Optional[List[Tuple[str, int]]] = None
    
    # Potential issues
    is_constant: bool = False
    is_id_like: bool = False


@dataclass
class TargetProfile:
    """Profile for the target variable."""
    name: str
    task_type: str  # "regression" or "classification"
    n_unique: int
    
    # Regression-specific
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    has_outliers: bool = False
    outlier_rate: float = 0.0

    # `AUDIT-012`: the SECOND tier of the outlier reading. `outlier_rate` above
    # is an IQR fence count and cannot tell a systolic pressure of 0 (an entry
    # error) from one of 244 (a hypertensive crisis, and real). These four say
    # what the physiologic reference could add, and they are `None` — never
    # `0.0` — when it could add nothing, because `0.0` would assert "no
    # impossible values here" where the truth is "no band was read" (trap 9).
    #: "matched" | "unrecognized" | "unread" | "not_regression"
    physio_read: str = "unread"
    #: The reference variable this target IS, exact key or declared alias only.
    physio_variable: Optional[str] = None
    impossible_count: Optional[int] = None
    impossible_rate: Optional[float] = None
    #: (floor, ceiling, unit) in the reference's units, or None.
    impossibility_band: Optional[Tuple[float, float, str]] = None

    # Classification-specific
    n_classes: Optional[int] = None
    class_counts: Optional[Dict[Any, int]] = None
    class_balance_ratio: Optional[float] = None  # max/min class ratio
    minority_class_size: Optional[int] = None
    is_imbalanced: bool = False
    imbalance_severity: Optional[str] = None  # "mild", "moderate", "severe"


@dataclass
class DatasetProfile:
    """
    Comprehensive dataset profile for intelligent model coaching.
    
    This is the central data structure that drives all recommendations.
    """
    # Basic dimensions
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    
    # Feature-to-sample ratios
    p_n_ratio: float  # p/n
    
    # Missingness summary
    total_missing_rate: float
    n_features_with_missing: int
    n_features_high_missing: int  # >10% missing
    n_features_critical_missing: int  # >50% missing
    
    # Optional fields with defaults

    #: `CLINICAL_SURVEY_PACK.md` §A5.4's *count parameters, not variables* — a
    #: numeric column is 1, a k-level factor is k−1. Distinct from `n_features`,
    #: which is and stays the COLUMN count: "8 predictors" is true of columns
    #: and "8 candidate parameters" is a different claim (`AUDIT-020`).
    n_candidate_parameters: Optional[int] = None

    #: Minority-class events per CANDIDATE PARAMETER, never per column. `None`
    #: where the quotient would be a guess — see `ml.sample_size`.
    events_per_variable: Optional[float] = None

    #: Rows that carry an outcome value — the population every sufficiency and
    #: dimensionality claim on this profile describes. `n_rows` is the frame
    #: handed in; a row with no outcome is in no analysis cohort, so a verdict
    #: computed over it describes a study that does not exist. `None` where no
    #: target was known and the two are the same number.
    n_analysis_rows: Optional[int] = None

    # Target info
    target_profile: Optional[TargetProfile] = None
    
    # Feature details
    feature_profiles: Dict[str, FeatureProfile] = field(default_factory=dict)
    
    # Cardinality issues
    high_cardinality_features: List[str] = field(default_factory=list)
    constant_features: List[str] = field(default_factory=list)
    id_like_features: List[str] = field(default_factory=list)
    
    # Numeric issues
    features_with_outliers: List[str] = field(default_factory=list)
    highly_skewed_features: List[str] = field(default_factory=list)

    #: `AUDIT-012`. Which of `features_with_outliers` carry entries outside the
    #: published plausibility bounds, which are recognized and clean, and which
    #: the reference has never heard of. `None` means the read never ran — it
    #: is not the same claim as "no column here is impossible", and
    #: `read_outlier_tiers` keeps the two apart.
    outlier_tiers: Optional[Dict[str, Any]] = None

    physio_plausibility_flags: List[str] = field(default_factory=list)
    physio_reference_version: Optional[str] = None
    
    # Data sufficiency assessment
    data_sufficiency: DataSufficiencyLevel = DataSufficiencyLevel.ADEQUATE
    sufficiency_narrative: str = ""
    
    # Warnings and flags
    warnings: List[DataWarning] = field(default_factory=list)
    
    # Recommended model families (preliminary)
    recommended_families: List[str] = field(default_factory=list)
    cautioned_families: List[str] = field(default_factory=list)
    discouraged_families: List[str] = field(default_factory=list)
    
    # Metadata
    profile_timestamp: Optional[str] = None


def compute_feature_profile(df: pd.DataFrame, col: str, n: int, outlier_method: str = "iqr") -> FeatureProfile:
    """Compute profile for a single feature."""
    series = df[col]
    dtype = str(series.dtype)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_categorical = not is_numeric or series.nunique() <= 10
    
    missing_count = series.isna().sum()
    missing_rate = missing_count / n if n > 0 else 0.0
    unique_count = series.nunique()
    cardinality_ratio = unique_count / n if n > 0 else 0.0
    
    profile = FeatureProfile(
        name=col,
        dtype=dtype,
        is_numeric=is_numeric,
        is_categorical=is_categorical and not is_numeric,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_count=unique_count,
        cardinality_ratio=cardinality_ratio,
        is_constant=(unique_count <= 1),
        # WHAT MAKES A COLUMN A ROW'S NAME (`GUIDED-120`).
        #
        # This required an INTEGER dtype and so answered False for every string
        # identifier — `respondent_id`, `admission_id`, `patient_id`,
        # `sample_id`, all `object`. Guided compensated with its own arithmetic
        # for text columns, which is a private copy of a core rule and the
        # thing `ROADMAP.md` "One core, no forks" forbids. Widened here so
        # Classic gets the same answer rather than a narrower one.
        #
        # Two conditions, and the second is why numerics still need the dtype
        # test. A column is a row's name when every value is different AND the
        # model cannot use the values' ORDER:
        #
        #   * a NON-NUMERIC column with one level per row has no order to use —
        #     one-hot encoding it spends n-1 parameters, each true for exactly
        #     one row;
        #   * a NUMERIC column with one level per row is normally a continuous
        #     MEASUREMENT, which is a perfectly good predictor. Requiring an
        #     integer dtype there is what separates a row number from a
        #     measurement, and dropping it would have flagged 88 assay columns
        #     on `metabolomics_untargeted.csv` — the study's own predictors.
        #
        # A nullable Int64 column is still integer, and `dtype in
        # ['int64','int32']` says it is not, which is why the predicate is
        # asked rather than the dtype compared.
        is_id_like=(
            unique_count == n
            and not _pdt.is_bool_dtype(series)
            and (not is_numeric
                 or _pdt.is_integer_dtype(series))
        )
    )
    
    if is_numeric:
        valid = series.dropna()
        if len(valid) > 0:
            profile.mean = float(valid.mean())
            profile.std = float(valid.std())
            profile.min_val = float(valid.min())
            profile.max_val = float(valid.max())
            profile.median = float(valid.median())
            
            # Skewness
            #
            # `AUDIT-003`. BEFORE: `except: profile.skewness = 0.0` — a failed
            # computation reported PERFECT SYMMETRY, which is trap 9 in its
            # exact form: a value returned from ignorance, and the one value
            # every downstream reader treats as "nothing to see". AFTER:
            # `skewness` stays `None`, which is what "not measured" already
            # means on this field — `highly_skewed_features` below is gated on
            # `is not None`, so an unmeasurable column is left out of the
            # transform advice instead of being asserted symmetric into it.
            if len(valid) > 2:
                try:
                    profile.skewness = float(valid.skew())
                except Exception:
                    profile.skewness = None

            # Outlier detection (skip boolean columns - quantile fails on bool)
            if valid.dtype != bool and not (hasattr(valid.dtype, 'kind') and valid.dtype.kind == 'b'):
                outlier_mask, _ = detect_outliers(valid, method=outlier_method)
                profile.outlier_count = int(outlier_mask.sum())
            else:
                profile.outlier_count = 0
            profile.outlier_rate = profile.outlier_count / len(valid) if len(valid) > 0 else 0.0
            profile.has_outliers = profile.outlier_rate > 0.01  # >1% outliers
    
    if profile.is_categorical or (not is_numeric):
        # High cardinality check
        profile.is_high_cardinality = cardinality_ratio > 0.5 or unique_count > 50
        
        # Top categories
        top_cats = series.value_counts().head(5)
        profile.top_categories = list(zip(top_cats.index.astype(str), top_cats.values.tolist()))
    
    return profile


def read_target_impossibility(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """The impossible tier for one column, or an honest record that there is none.

    `AUDIT-012`. `ml/outliers.py`'s IQR fences answer *"is this value far from
    the others"*; `ml/physiology_reference.py`'s impossibility band answers
    *"could a living person produce this"*. Those are different questions with
    opposite remedies, and the coach had only the first.

    This does not re-derive the second — `ml/card_evidence.py`'s
    `plausibility_report` is the reader `turbotab/engine.plausibility` already
    serves to both doors, and it is called here so the coach's number and the
    plausibility card's number are the same number. Unit conversion, the
    exact-or-alias name match and the whole-column-suspect exclusion all come
    with it.

    Returns `impossible_count` / `impossible_rate` as **None** unless a band was
    actually read. `0` would say "nothing here is impossible" on a column whose
    name matches nothing in the reference, which is the app asserting a
    measurement it never took.
    """
    out: Dict[str, Any] = {"physio_read": "unread", "physio_variable": None,
                           "impossible_count": None, "impossible_rate": None,
                           "impossibility_band": None}
    try:
        from ml.card_evidence import plausibility_report
        from ml.physiology_reference import get_impossibility_band

        reference = load_reference_bundle()["nhanes"]
        var_key = match_variable_key(str(target_col), reference)
        if not var_key:
            out["physio_read"] = "unrecognized"
            return out
        band = get_impossibility_band(reference, var_key)
        if band is None:
            # Recognized, but no floor/ceiling is published for it. Named
            # separately from "unrecognized" because the two are different
            # gaps and the sentence downstream says which one it is.
            out.update(physio_read="unrecognized", physio_variable=var_key)
            return out

        report = plausibility_report(df, columns=[target_col],
                                     reference=reference)
        present = int(df[target_col].dropna().shape[0])
        count = int(report.get("n_impossible") or 0)
        out.update(physio_read="matched", physio_variable=var_key,
                   impossible_count=count,
                   impossible_rate=(count / present) if present else None,
                   impossibility_band=band)
    except Exception:
        # Left as "unread". The coach then says the band was not read rather
        # than reporting a zero it did not measure.
        return {"physio_read": "unread", "physio_variable": None,
                "impossible_count": None, "impossible_rate": None,
                "impossibility_band": None}
    return out


#: The reading `read_outlier_tiers` returns when nothing could be read. Every
#: tier list is `None` rather than `[]`: an empty list would say "no column
#: here carries an impossible value", which is a measurement, and this is its
#: absence (trap 9).
OUTLIER_TIERS_UNREAD: Dict[str, Any] = {
    "read": False, "reason": None, "impossible": None,
    "within_band": None, "unrecognized": None, "reference_version": None,
}


def read_outlier_tiers(df: pd.DataFrame,
                       columns: Optional[List[str]]) -> Dict[str, Any]:
    """Split fence-flagged COLUMNS into the tiers an IQR fence cannot see.

    `AUDIT-012`, the feature half. `read_target_impossibility` above does this
    for the one target column; the profile's `features_with_outliers` list is
    the same undifferentiated fence count spread over every numeric column, and
    the warning composed from it offered winsorizing and capping to all of them
    alike.

    `research/CLINICAL_SURVEY_PACK.md` §A1.2 keeps **two separate,
    differently-purposed bound sets** and says what each is for:

    > **Physiological plausibility bounds** — values incompatible with a living
    > patient. Use for flagging as suspected data error.
    > **Reference interval** — the central 95% of a healthy reference
    > population. *By construction ~5% of healthy people fall outside.*
    > **Use only for annotation. Never for exclusion.**

    and Cross-cutting 7 ranks the collapse seventh by damage: *"Excluding
    abnormal-but-possible clinical values as 'outliers'. Removes the sickest
    patients. Physiologically impossible ≠ abnormal, and generic outlier rules
    (±3 SD, IQR fences) are wrong here."*

    Returns three disjoint groups over `columns`:

    ``impossible``    {column: {"n", "band"}} — at least one entry outside the
                      published floor/ceiling. Repaired, not downweighted.
    ``within_band``   recognized, and every fence hit is inside what a living
                      person can produce — abnormal but real.
    ``unrecognized``  matched no reference variable, or no band is published
                      for it. **The app cannot tell the two apart here** and
                      the sentence says so rather than keeping wording that
                      implies it checked.

    The read is not re-derived: `ml/card_evidence.plausibility_report` is the
    same reader `turbotab/engine.plausibility` serves to both doors, so this
    warning and the plausibility card cannot report different numbers for one
    frame. `whole_column_suspect` blocks are excluded exactly as that reader
    excludes them — a mis-united column is a units finding, not an entry error.
    """
    cols = [c for c in (columns or []) if c in df.columns]
    if not cols:
        return {**OUTLIER_TIERS_UNREAD,
                "reason": "no column tripped the fence, so there was nothing to read"}
    try:
        from ml.card_evidence import plausibility_report
        from ml.physiology_reference import get_impossibility_band

        reference = load_reference_bundle()["nhanes"]
        report = plausibility_report(df, columns=cols, reference=reference)

        impossible: Dict[str, Dict[str, Any]] = {}
        for block in report.get("impossible") or []:
            if block.get("whole_column_suspect"):
                continue
            col, n = block.get("column"), int(block.get("n_flagged") or 0)
            if col in cols and n > 0:
                impossible[col] = {
                    "n": n,
                    "band": (block.get("low"), block.get("high"),
                             block.get("unit")),
                    "variable": block.get("variable"),
                }

        within_band, unrecognized = [], []
        for col in cols:
            if col in impossible:
                continue
            var_key = match_variable_key(str(col), reference)
            band = get_impossibility_band(reference, var_key) if var_key else None
            (within_band if band is not None else unrecognized).append(col)

        return {"read": True, "reason": None, "impossible": impossible,
                "within_band": within_band, "unrecognized": unrecognized,
                "reference_version": reference.get("version")}
    except Exception as exc:                                  # pragma: no cover
        return {**OUTLIER_TIERS_UNREAD,
                "reason": f"the physiologic reference could not be read ({exc.__class__.__name__})"}


def outlier_tier_sentence(n_flagged: int, tiers: Optional[Dict[str, Any]]) -> str:
    """What the profile may say about `n_flagged` fence-flagged columns.

    `AUDIT-012`. BEFORE — one sentence for every column alike, and it named a
    consequence for the model without ever naming what the values were:

        "{n} numeric features have significant outliers. This can affect model
         performance, especially for distance-based and linear models."

    AFTER — the same subject, split by what was actually read, and where
    nothing was read the sentence says so. The IQR count is still reported: the
    correction is to what the app CLAIMS the count means, not to the count.
    """
    head = (f"{n_flagged} numeric feature{'' if n_flagged == 1 else 's'} "
            f"{'has' if n_flagged == 1 else 'have'} "
            f"values outside the 1.5×IQR fence. That fence answers *is this "
            f"value far from the others*; it cannot answer *could a living "
            f"person produce it*, and the two have opposite remedies — an "
            f"impossible entry is a data error and is repaired, an extreme but "
            f"attainable measurement is the phenomenon under study and is kept "
            f"(CLINICAL_SURVEY_PACK §A1.2).")

    if not tiers or not tiers.get("read"):
        why = tiers.get("reason") if tiers else None
        return (f"{head} The physiologic reference was not read for these "
                f"columns"
                + (f" ({why})" if why else "")
                + ", so the app cannot tell those two apart here and this "
                  "count is the fence count and nothing more.")

    impossible = tiers.get("impossible") or {}
    within = tiers.get("within_band") or []
    unknown = tiers.get("unrecognized") or []
    parts = [head]

    if impossible:
        named = ", ".join(
            f"{col} ({d['n']} outside {d['band'][0]:g}–{d['band'][1]:g} "
            f"{d['band'][2]})" for col, d in sorted(impossible.items()))
        parts.append(f"{len(impossible)} of them "
                     f"{'carries' if len(impossible) == 1 else 'carry'} "
                     f"entries outside the "
                     f"published plausibility bounds — {named}. Those are "
                     f"suspected entry errors: repair them on the plausibility "
                     f"card rather than winsorizing them, which hides them at "
                     f"the fence instead of correcting them.")
    if within:
        parts.append(f"{len(within)} of them ({', '.join(sorted(within))}) "
                     f"{'matches' if len(within) == 1 else 'match'} "
                     f"a reference variable and every fence hit is "
                     f"inside what a living person can produce — those read as "
                     f"real extremes, and excluding them removes the sickest "
                     f"rows in the table.")
    if unknown:
        parts.append(f"For {len(unknown)} of them "
                     f"({', '.join(sorted(unknown))}) the app cannot tell an "
                     f"impossible entry from an abnormal-but-real one: the "
                     f"column matches no variable in the physiologic "
                     f"reference, so there is no floor or ceiling to check "
                     f"against and the fence count is all that was measured.")
    return " ".join(parts)


def compute_target_profile(df: pd.DataFrame, target_col: str, task_type: str, outlier_method: str = "iqr") -> TargetProfile:
    """Compute profile for the target variable."""
    series = df[target_col]
    n_unique = series.nunique()
    
    profile = TargetProfile(
        name=target_col,
        task_type=task_type,
        n_unique=n_unique
    )
    
    valid = series.dropna()
    
    if task_type == 'regression':
        if len(valid) > 0:
            profile.mean = float(valid.mean())
            profile.std = float(valid.std())
            profile.min_val = float(valid.min())
            profile.max_val = float(valid.max())
            profile.median = float(valid.median())
            
            # `AUDIT-003`, the target's copy of the same fabrication. It is the
            # one that reaches a person: `pages/06_Train_and_Compare.py:866`
            # renders `f", target skew {abs(_target_prof.skewness):.2f}"`, so a
            # failed computation printed "target skew 0.00" — a measurement
            # asserted, not taken. `None` is already the field's "not measured"
            # and both readers (that page and `ml/nn_recommender.py:146`) guard
            # on it, so the app goes quiet here instead of confident.
            if len(valid) > 2:
                try:
                    profile.skewness = float(valid.skew())
                except Exception:
                    profile.skewness = None

            # Outlier detection (configurable method)
            outlier_mask, _ = detect_outliers(valid, method=outlier_method)
            profile.outlier_rate = float(outlier_mask.sum() / len(valid)) if len(valid) > 0 else 0.0
            profile.has_outliers = profile.outlier_rate > 0.05

            # `AUDIT-012`. The rate above is a fence count. Beside it sits a
            # published impossibility band that nothing on the coaching path
            # read, so one number carried two situations that want opposite
            # advice: an impossible entry is REPAIRED, an abnormal-but-real
            # extreme is MODELED. Read here, reported separately, and left
            # `None` where the reference has nothing to say.
            reading = read_target_impossibility(df, target_col)
            profile.physio_read = reading["physio_read"]
            profile.physio_variable = reading["physio_variable"]
            profile.impossible_count = reading["impossible_count"]
            profile.impossible_rate = reading["impossible_rate"]
            profile.impossibility_band = reading["impossibility_band"]

    else:  # classification
        profile.physio_read = "not_regression"
        profile.n_classes = n_unique
        profile.class_counts = valid.value_counts().to_dict()
        
        if profile.class_counts:
            max_count = max(profile.class_counts.values())
            min_count = min(profile.class_counts.values())
            profile.class_balance_ratio = max_count / min_count if min_count > 0 else float('inf')
            profile.minority_class_size = min_count
            
            # Imbalance assessment
            if profile.class_balance_ratio > 10:
                profile.is_imbalanced = True
                profile.imbalance_severity = "severe"
            elif profile.class_balance_ratio > 5:
                profile.is_imbalanced = True
                profile.imbalance_severity = "moderate"
            elif profile.class_balance_ratio > 2:
                profile.is_imbalanced = True
                profile.imbalance_severity = "mild"
            else:
                profile.is_imbalanced = False
    
    return profile


def assess_data_sufficiency(
    n: int,
    p: int,
    task_type: str,
    minority_class_size: Optional[int] = None,
    n_parameters: Optional[int] = None,
    population: str = "",
) -> Tuple[DataSufficiencyLevel, str]:
    """
    Assess data sufficiency for modeling.

    Uses practical heuristics (not formal power calculations):
    - Events per candidate PARAMETER for classification
    - Observations per parameter for regression
    - Feature-to-sample ratio

    `AUDIT-020` / `AUDIT-021`, and they are one design — see `ml/sample_size.py`.

    `p` is the number of predictor COLUMNS and stays that, because every
    sentence here that says "features" or "dimensionality" is true of columns.
    `n_parameters` is `CLINICAL_SURVEY_PACK.md` §A5.4's *count parameters, not
    variables* — a 5-level factor is 4 — and it is the only denominator EPV may
    use. **A `None` parameter count reports no EPV at all** rather than falling
    back to `p`: a silent fall back to the wrong denominator is the defect this
    signature exists to remove, and the app may be silent.

    The docstring caveat above ("not formal power calculations") used to be the
    only disclosure that this is a heuristic, and it lived here where no user
    can read it. `sample_size.SUPERSEDED` is that disclosure attached to the
    string that ships.

    **`population` names what `n` counts, and it is in every sentence that
    quotes it.** The drive found *"Large sample (n=20,904). All model types are
    viable."* in a manuscript whose Study Design paragraph says 6,297
    observations and whose Table 1 is built on them — a fourth `n` for one
    study, naming no population, so a reader cannot tell which rows it is a
    verdict about. Empty means the caller had nothing to disclose.
    """
    from ml import sample_size as _ss

    p_n_ratio = p / n if n > 0 else float('inf')
    _pop = f" {population.strip()}" if population and population.strip() else ""

    narratives = []
    level = DataSufficiencyLevel.ADEQUATE

    # Basic sample size check
    if n < 50:
        level = DataSufficiencyLevel.CRITICAL
        narratives.append(f"Very small sample (n={n:,}{_pop}). Only the simplest models are viable.")
    elif n < 100:
        level = DataSufficiencyLevel.SCARCE
        narratives.append(f"Small sample (n={n:,}{_pop}). Strong regularization recommended.")
    elif n < 500:
        level = DataSufficiencyLevel.LIMITED
        narratives.append(f"Modest sample size (n={n:,}{_pop}). Complex models may overfit.")
    elif n < 5000:
        level = DataSufficiencyLevel.ADEQUATE
        narratives.append(f"Adequate sample size (n={n:,}{_pop}) for most model types.")
    else:
        level = DataSufficiencyLevel.ABUNDANT
        narratives.append(f"Large sample (n={n:,}{_pop}). All model types are viable.")
    
    # Feature-to-sample ratio check
    if p_n_ratio > 1.0:
        level = min(level, DataSufficiencyLevel.CRITICAL, key=lambda x: list(DataSufficiencyLevel).index(x))
        narratives.append(f"More features than samples (p/n={p_n_ratio:.2f}). Regularization essential; consider dimensionality reduction.")
    elif p_n_ratio > 0.5:
        level = min(level, DataSufficiencyLevel.SCARCE, key=lambda x: list(DataSufficiencyLevel).index(x))
        narratives.append(f"High dimensionality (p/n={p_n_ratio:.2f}). Regularized models preferred.")
    elif p_n_ratio > 0.1:
        narratives.append(f"Moderate dimensionality (p/n={p_n_ratio:.2f}).")
    else:
        narratives.append(f"Low dimensionality (p/n={p_n_ratio:.2f}). Feature space is manageable.")
    
    # Events per candidate PARAMETER (classification). §A5.4's denominator, and
    # the sentence names what it counted so a reader can check the arithmetic.
    if task_type == 'classification':
        epv = _ss.events_per_parameter(minority_class_size, n_parameters)
        if epv is not None:
            if epv < 5:
                level = min(level, DataSufficiencyLevel.CRITICAL, key=lambda x: list(DataSufficiencyLevel).index(x))
            elif epv < _ss.CAUTION_EPV:
                level = min(level, DataSufficiencyLevel.SCARCE, key=lambda x: list(DataSufficiencyLevel).index(x))
            narratives.append(
                _ss.epv_sentence(epv, minority_class_size, n_parameters))


    # Observations per parameter heuristics for neural nets
    if n < p * 20:
        narratives.append("Neural networks may struggle: typically need 20+ samples per input feature.")
    
    narrative = " ".join(narratives)
    return level, narrative


def generate_warnings(profile: DatasetProfile) -> List[DataWarning]:
    """Generate warnings based on the dataset profile."""
    warnings = []
    
    # Sample size warning
    if profile.n_rows < 100:
        warnings.append(DataWarning(
            category="sample_size",
            level=WarningLevel.CRITICAL,
            short_message="Very small sample",
            detailed_message=f"With only {profile.n_rows:,} samples, model training is highly constrained. "
                           "Use strong regularization, cross-validation, and simple models. "
                           "Results may not generalize well.",
            affected_models=["Neural Networks", "Complex Ensembles"],
            suggested_actions=[
                "Consider collecting more data",
                "Use regularized linear models (Ridge, Lasso)",
                "Increase cross-validation folds",
                "Be cautious interpreting results"
            ]
        ))
    elif profile.n_rows < 500:
        warnings.append(DataWarning(
            category="sample_size",
            level=WarningLevel.WARNING,
            short_message="Small sample",
            detailed_message=f"With {profile.n_rows:,} samples, some models may overfit. "
                           "Regularization and validation are important.",
            affected_models=["Neural Networks", "Deep Trees"],
            suggested_actions=[
                "Use regularization",
                "Prefer simpler models",
                "Use cross-validation"
            ]
        ))
    
    # High dimensionality warning
    if profile.p_n_ratio > 0.5:
        warnings.append(DataWarning(
            category="dimensionality",
            level=WarningLevel.WARNING if profile.p_n_ratio <= 1.0 else WarningLevel.CRITICAL,
            short_message="High dimensionality",
            detailed_message=f"Feature-to-sample ratio is {profile.p_n_ratio:.2f} "
                           f"({profile.n_features} features, {profile.n_rows:,} samples). "
                           "This can cause overfitting and unstable estimates.",
            affected_models=["Unregularized Linear Models", "k-NN"],
            suggested_actions=[
                "Use regularized models (Ridge, Lasso, ElasticNet)",
                "Consider dimensionality reduction (PCA)",
                "Remove low-variance or redundant features"
            ]
        ))
    
    # Class imbalance warning
    if profile.target_profile and profile.target_profile.is_imbalanced:
        severity = profile.target_profile.imbalance_severity
        ratio = profile.target_profile.class_balance_ratio
        level = WarningLevel.CRITICAL if severity == "severe" else (
            WarningLevel.WARNING if severity == "moderate" else WarningLevel.CAUTION
        )
        warnings.append(DataWarning(
            category="imbalance",
            level=level,
            short_message=f"{severity.title()} imbalance",
            detailed_message=f"Class imbalance ratio is {ratio:.1f}:1. "
                           f"The minority class has only {profile.target_profile.minority_class_size:,} samples. "
                           "Accuracy can be misleading; use F1, PR-AUC, or balanced accuracy instead.",
            affected_models=["All classification models"],
            # `GUIDED-049`. This used to open with "Use class weights in
            # training" and "Consider SMOTE or other resampling". Van den
            # Goorbergh et al. (JAMIA 2022;29:1525) and Carriero et al. (Stat
            # Med 2025): rebalancing overestimates minority-class probability
            # without improving discrimination, and the apparent sensitivity
            # gain is reproducible by shifting the threshold. The remedy for a
            # rare outcome is penalization and sample size, because the problem
            # is small-sample overfitting rather than imbalance.
            #
            # The advice is not deleted, it is ROUTED — see
            # `ml/imbalance_advice.py`, which the Guided door reads with the
            # recorded purpose. What is removed here is the RECOMMENDATION,
            # because this list is read as one.
            suggested_actions=[
                "Focus on precision-recall metrics, not accuracy",
                "Report calibration alongside discrimination",
                "Adjust classification threshold based on costs",
                "Penalize the fit — the problem behind a rare outcome is "
                "small-sample overfitting, not imbalance",
            ]
        ))
    
    # Missing data warning
    if profile.n_features_high_missing > 0:
        level = WarningLevel.CRITICAL if profile.n_features_critical_missing > 0 else WarningLevel.WARNING
        warnings.append(DataWarning(
            category="missingness",
            level=level,
            short_message=f"{profile.n_features_high_missing} features with high missingness",
            detailed_message=f"{profile.n_features_high_missing} features have >10% missing values"
                           + (f", and {profile.n_features_critical_missing} have >50% missing" 
                              if profile.n_features_critical_missing > 0 else "") + 
                           ". Missing data can bias results if not handled properly.",
            affected_models=["Linear Models (need imputation)", "Neural Networks (need imputation)"],
            suggested_actions=[
                "Investigate if missingness is random (MAR) or informative (MNAR)",
                "Consider adding missingness indicators",
                "Use imputation (mean/median for simple, KNN/iterative for better)",
                "Tree models can handle missing values natively"
            ]
        ))
    
    # High cardinality warning
    if len(profile.high_cardinality_features) > 0:
        warnings.append(DataWarning(
            category="cardinality",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.high_cardinality_features)} high-cardinality features",
            detailed_message=f"Features {', '.join(profile.high_cardinality_features[:3])}"
                           + (f" and {len(profile.high_cardinality_features)-3} more" 
                              if len(profile.high_cardinality_features) > 3 else "") +
                           " have many unique values. One-hot encoding will create many columns.",
            affected_models=["Linear Models (with one-hot)", "Neural Networks"],
            suggested_actions=[
                "Consider target encoding",
                "Consider frequency encoding",
                "Group rare categories",
                "Tree models handle high cardinality better"
            ]
        ))
    
    # Outliers warning
    if len(profile.features_with_outliers) > 0:
        warnings.append(DataWarning(
            category="outliers",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.features_with_outliers)} features with outliers",
            # `AUDIT-012`. This sentence is `detail` on the Guided finding
            # (`turbotab/engine.py:312`), so it is the rendered instance of the
            # row. It used to name a consequence for the model and never the
            # values, which let one fence count stand for two findings with
            # opposite remedies.
            detailed_message=outlier_tier_sentence(
                len(profile.features_with_outliers), profile.outlier_tiers),
            affected_models=["Linear Regression (OLS)", "k-NN", "Neural Networks"],
            # The shelf is not shortened: all four options survive, and
            # `detailed_message` says which columns each one is right for.
            # "Investigate if outliers are errors or genuine" is kept and is
            # now answerable — the sentence above says which columns the app
            # could answer it for and which it could not.
            suggested_actions=[
                "Investigate if outliers are errors or genuine",
                "Consider robust models (Huber loss)",
                "Consider winsorizing or capping",
                "Tree models are robust to outliers"
            ]
        ))

    # Physiologic plausibility warning (NHANES reference)
    if profile.physio_plausibility_flags:
        warnings.append(DataWarning(
            category="physiologic_plausibility",
            level=WarningLevel.CAUTION,
            short_message=f"{len(profile.physio_plausibility_flags)} physiologic flags",
            detailed_message=(
                "Empirical plausibility checks found values outside the NHANES "
                "improbability band (p01\u2013p99), which is not a reference interval. "
                "These checks are based on population distributions, not clinical guidance."
            ),
            affected_models=["All Models"],
            suggested_actions=[
                "Verify units and data entry",
                "Review plausible ranges for affected features",
                "Consider unit harmonization or plausibility gating"
            ]
        ))
    
    return warnings


def compute_dataset_profile(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    outlier_method: str = "iqr"
) -> DatasetProfile:
    """
    Compute a comprehensive dataset profile.
    
    Args:
        df: The dataframe to profile
        target_col: Name of target column (optional)
        feature_cols: List of feature columns (optional, defaults to all non-target)
        task_type: 'regression' or 'classification' (optional, will infer if not provided)
    
    Returns:
        DatasetProfile with all diagnostics
    """
    from datetime import datetime
    
    n = len(df)
    if n == 0 or len(df.columns) == 0:
        raise ValueError("Cannot compute profile for empty DataFrame")
    
    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    p = len(feature_cols)
    
    # ── ONE counting rule for "numeric" and "categorical" ────────────────────
    # This split reached the report as "Numeric Features 25 / Categorical
    # Features 2" beside page 02's tiles saying 19 / 8 — a fourth type count for
    # one table — because `is_numeric_dtype` calls a bool column numeric and the
    # preprocessing pipeline does not. THE RULE THAT DECIDES IS THE PIPELINE'S:
    # pages/05 splits the columns with `data_processor.get_numeric_columns`, so
    # a bool column really is one-hot encoded. Page 02 was reconciled to that
    # rule; this is the same rule, read from the same function.
    from data_processor import get_numeric_columns as _get_numeric_columns

    _numeric_in_frame = set(_get_numeric_columns(df))
    numeric_cols = [c for c in feature_cols if c in _numeric_in_frame]
    categorical_cols = [c for c in feature_cols if c not in _numeric_in_frame]
    
    # Compute feature profiles
    feature_profiles = {}
    high_cardinality = []
    constant_features = []
    id_like_features = []
    features_with_outliers = []
    highly_skewed = []
    
    for col in feature_cols:
        fp = compute_feature_profile(df, col, n, outlier_method=outlier_method)
        feature_profiles[col] = fp
        
        if fp.is_high_cardinality:
            high_cardinality.append(col)
        if fp.is_constant:
            constant_features.append(col)
        if fp.is_id_like:
            id_like_features.append(col)
        if fp.has_outliers:
            features_with_outliers.append(col)
        if fp.skewness is not None and abs(fp.skewness) > 1.0:
            highly_skewed.append(col)
    
    # Missingness summary
    missing_counts = df[feature_cols].isna().sum()
    total_missing = missing_counts.sum()
    total_cells = n * p
    total_missing_rate = total_missing / total_cells if total_cells > 0 else 0.0
    n_features_with_missing = (missing_counts > 0).sum()
    n_features_high_missing = (missing_counts / n > 0.1).sum()
    n_features_critical_missing = (missing_counts / n > 0.5).sum()
    
    # Target profile
    target_profile = None
    minority_class_size = None
    if target_col is not None and target_col in df.columns:
        # Infer task type if not provided
        if task_type is None:
            target_unique = df[target_col].nunique()
            if target_unique <= 20 and target_unique < n * 0.05:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        target_profile = compute_target_profile(df, target_col, task_type, outlier_method=outlier_method)
        if task_type == 'classification' and target_profile.minority_class_size:
            minority_class_size = target_profile.minority_class_size
    
    # Candidate PARAMETERS, not columns. §A5.4, and it is the same count
    # `turbotab.resolution.candidate_parameters` charges in the Guided door —
    # both now call `ml.sample_size`, so the two doors cannot report different
    # numbers for the same frame (`AUDIT-020`).
    from ml import sample_size as _ss
    n_candidate_parameters = _ss.candidate_parameters(
        df, feature_cols)["total"]

    # ── The population every sufficiency and dimensionality claim describes ──
    # `n` counts every row of the frame handed in. A row with no outcome value
    # is in no analysis cohort — the split drops it and Table 1 is built on what
    # survives — so a sufficiency verdict or a p/n computed over those rows is
    # about a study that does not exist. Same rule and same words as page 02's
    # `_analysis_n`, so the two surfaces cannot report different populations.
    if target_col is not None and target_col in df.columns:
        n_analysis = int(df[target_col].notna().sum())
    else:
        n_analysis = n
    population_phrase = ("observations with a recorded outcome"
                         if n_analysis < n else "observations")

    # Compute data sufficiency
    p_n_ratio = p / n_analysis if n_analysis > 0 else float('inf')
    data_sufficiency, sufficiency_narrative = assess_data_sufficiency(
        n_analysis, p, task_type or 'regression', minority_class_size,
        n_parameters=n_candidate_parameters,
        population=population_phrase,
    )

    # Events per candidate parameter
    events_per_variable = None
    if task_type == 'classification':
        events_per_variable = _ss.events_per_parameter(
            minority_class_size, n_candidate_parameters)

    # Physiologic plausibility flags (NHANES reference only)
    reference_bundle = load_reference_bundle()
    nhanes_ref = reference_bundle["nhanes"]
    physio_flags = []
    for col in numeric_cols:
        var_key = match_variable_key(col, nhanes_ref)
        if not var_key:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        inferred_unit_info = infer_unit(col, col_data)
        improbability = get_improbability_band(nhanes_ref, var_key)
        if inferred_unit_info.get('conversion_factor') and improbability:
            improbable_low, improbable_high, improbable_unit = improbability
            converted = col_data * inferred_unit_info['conversion_factor']
            out_rate = ((converted < improbable_low) | (converted > improbable_high)).sum() / len(converted)
            if out_rate > 0.05:
                # `MISC-018`: p01–p99 is an improbability band, not a reference
                # interval. Same fact, same register as `ml/eda_actions.py`.
                physio_flags.append(
                    f"{col}: {out_rate:.1%} values outside the NHANES improbability band "
                    f"({improbable_low}-{improbable_high} {improbable_unit})"
                )

    # Create profile
    profile = DatasetProfile(
        n_rows=n,
        n_features=p,
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        p_n_ratio=p_n_ratio,
        total_missing_rate=total_missing_rate,
        n_features_with_missing=n_features_with_missing,
        n_features_high_missing=n_features_high_missing,
        n_features_critical_missing=n_features_critical_missing,
        n_candidate_parameters=n_candidate_parameters,
        events_per_variable=events_per_variable,
        n_analysis_rows=n_analysis,
        target_profile=target_profile,
        feature_profiles=feature_profiles,
        high_cardinality_features=high_cardinality,
        constant_features=constant_features,
        id_like_features=id_like_features,
        features_with_outliers=features_with_outliers,
        highly_skewed_features=highly_skewed,
        # `AUDIT-012`. Read HERE, because `generate_warnings` is given a
        # profile and never the frame — a correction that lived in the warning
        # would have had nothing to read.
        outlier_tiers=read_outlier_tiers(df, features_with_outliers),
        physio_plausibility_flags=physio_flags,
        physio_reference_version=nhanes_ref.get("version"),
        data_sufficiency=data_sufficiency,
        sufficiency_narrative=sufficiency_narrative,
        profile_timestamp=datetime.now().isoformat()
    )
    
    # Generate warnings
    profile.warnings = generate_warnings(profile)
    
    # Preliminary model family recommendations
    profile.recommended_families, profile.cautioned_families, profile.discouraged_families = \
        _assess_model_families(profile)
    
    return profile


def _assess_model_families(profile: DatasetProfile) -> Tuple[List[str], List[str], List[str]]:
    """
    Preliminary assessment of which model families are suitable.
    
    Returns:
        (recommended, cautioned, discouraged) lists of model family names
    """
    recommended = []
    cautioned = []
    discouraged = []
    
    n = profile.n_rows
    p = profile.n_features
    p_n_ratio = profile.p_n_ratio
    task_type = profile.target_profile.task_type if profile.target_profile else 'regression'
    
    # Always recommend regularized linear models as baseline
    recommended.append("Linear Models")
    
    # Tree-based models
    if n >= 50:
        recommended.append("Tree-Based Models")
    else:
        cautioned.append("Tree-Based Models")
    
    # Gradient Boosting
    if n >= 100:
        recommended.append("Gradient Boosting")
    elif n >= 50:
        cautioned.append("Gradient Boosting")
    else:
        discouraged.append("Gradient Boosting")
    
    # Neural Networks
    if n >= 1000 and n >= p * 20:
        recommended.append("Neural Networks")
    elif n >= 500 and n >= p * 10:
        cautioned.append("Neural Networks")
    else:
        discouraged.append("Neural Networks")
    
    # k-NN
    if p <= 20 and n >= 100 and n <= 10000:
        recommended.append("Distance-Based (k-NN)")
    elif n > 10000:
        cautioned.append("Distance-Based (k-NN)")  # Can be slow
    else:
        discouraged.append("Distance-Based (k-NN)")  # Curse of dimensionality
    
    # SVM
    if n <= 5000 and p <= 100:
        cautioned.append("Support Vector Machines")
    else:
        discouraged.append("Support Vector Machines")
    
    return recommended, cautioned, discouraged


def get_profile_summary_text(profile: DatasetProfile) -> str:
    """Generate a human-readable summary of the dataset profile."""
    lines = []
    
    # Basic stats
    lines.append(f"**Dataset Overview:** {profile.n_rows:,} samples × {profile.n_features} features")
    lines.append(f"- {profile.n_numeric} numeric, {profile.n_categorical} categorical features")
    
    # Data sufficiency
    lines.append(f"\n**Data Sufficiency:** {profile.data_sufficiency.value.title()}")
    lines.append(f"- {profile.sufficiency_narrative}")
    
    # Target info
    if profile.target_profile:
        tp = profile.target_profile
        if tp.task_type == 'regression':
            lines.append(f"\n**Target ({tp.name}):** Continuous (regression)")
            if tp.mean is not None:
                lines.append(f"- Range: {tp.min_val:.2f} to {tp.max_val:.2f}, Mean: {tp.mean:.2f}")
        else:
            lines.append(f"\n**Target ({tp.name}):** Categorical ({tp.n_classes} classes)")
            if tp.is_imbalanced:
                lines.append(f"- Imbalanced: {tp.imbalance_severity} ({tp.class_balance_ratio:.1f}:1 ratio)")
    
    # Warnings summary
    if profile.warnings:
        lines.append(f"\n**Warnings:** {len(profile.warnings)} issue(s) detected")
        for w in profile.warnings[:3]:
            lines.append(f"- {w.level.value.upper()}: {w.short_message}")
    
    return "\n".join(lines)
