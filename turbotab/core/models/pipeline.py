"""The model pipeline every family shares, and what the design stage says about it.

Step order (M1_CONTRACT §3): impute (when the answer is "impute") → energy adjustment (when one
is answered; under "none" the step takes total energy out of the model, audit ME-02) → one-hot for
categorical columns → scale (families that need it) → the model. Every step is fit inside the pipeline, so cross-validation refits it on each training fold
and nothing learned from the data ever sees a held-out row. Column names survive every step
(``set_output(transform="pandas")``), and so does the row-id index, which the lineage and the
leakage tests rely on.

The pipeline runs on RAW inputs: the columns as :func:`modeling_frame` reads them. Substitution
curves shift raw intakes and push them through the whole pipeline, energy adjustment included.

Missingness by mechanism (ROADMAP lockbox constitution §07, M2_CONTRACT §4), from the
missing-values answer:

* ``categorical == "missing_category"``: every categorical or two-valued predictor (a yes/no,
  a medication taken or not) keeps its blanks as a level of their own, :data:`MISSING_LEVEL`,
  encoded beside its other levels (``levels`` step, :class:`MissingLevelEncoder`). That is the
  honest reading of a column like ``meds_hbp``, blank where the question was not asked. Its
  blanks then drop no row under complete cases either (``stages.rows.cohort_inputs``).
* ``indicators``: under imputation, each numeric column with blanks in the fitting rows gains a
  ``missingindicator_<column>`` beside its filled values.
* Imputation is a pipeline step, so it is fit on each training fold, and the outcome is never
  among its inputs: the pipeline's inputs are the predictors (``DesignSpec.inputs``), and ``y``
  never reaches a transformer. That is prediction's rule (BLUEPRINT §12 ruling 4). Under inference
  with multiple imputation the coefficient table is pooled over completed tables imputed with the
  outcome (``turbotab.core.methods.imputation``); the pipeline's in-fold fill then serves only the
  cross-validated scores, which stay outcome-free.
* An energy-bearing nutrient's single fill is its in-fold line on total energy, not its median
  (:class:`~turbotab.core.methods.missing.EnergyAwareImputer`, audit WP7), so imputed rows keep the
  nutrient–energy relation the energy step depends on.
* Values below a detection limit (``detect``): a left-censored column's blanks are filled first,
  by half its smallest detected value or by their expected value below the limit
  (:class:`~turbotab.core.methods.missing.BelowDetectionFill`, audit ME-08), on the raw values.

Step order: detect → normalize → impute → energy adjustment → exposure form → levels → one-hot →
scale → model.

Omics values (AUDIT_REPORT §5 WP11): when the ``omics_scale`` finding's normalization is recorded,
the exposures among its columns are normalized first — log-CPM with TMM factors, or quotient
normalization and log2 (``turbotab.core.methods.omics``) — fit on each training fold like every
other step. It runs before imputation, so a missing value is filled on the normalized scale.

The exposure form (``set_exposure_form``; :class:`~turbotab.core.methods.exposure_form.ExposureForms`)
turns a numeric predictor into a restricted cubic spline basis or quintile indicators. It runs
after energy adjustment, so the curve or the fifths are of the energy-adjusted intake, and it is
fit in the pipeline, so knots and cut points come from the rows each fit sees (audit ME-17).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from turbotab.core.decisions import EnergyAdjustment, ProjectState, Purpose, Task, missing_strategy
from turbotab.core.methods.energy import METHOD_TABLE, RESIDUAL_METHODS
from turbotab.core.models.base import ModelFamily
from turbotab.core.models.steps import energy_step

PREDICTOR_ROLES = ("exposure", "covariate", "energy")
ADJUST_STEPS = ("detect", "normalize", "impute", "energy")  # the steps whose outputs form the lineage's "adjusted" lane
MANY_LEVELS = 20
MISSING_LEVEL = "Missing"
LEVEL_DTYPES = ("boolean", "categorical", "text")


def predictors_from_roles(roles: Mapping[str, str] | None, target: str | None,
                          drop: Sequence[str] = ()) -> list[str]:
    """Columns with role exposure, covariate or energy, in the order the roles list them.

    ``drop``: columns the missing-values answer left out of the predictors.
    """
    if not roles:
        return []
    gone = set(drop)
    return [c for c, r in roles.items() if r in PREDICTOR_ROLES and c != target and c not in gone]


def model_predictors(state: ProjectState) -> list[str]:
    """The predictors the models get under ``state``: by settled role, less the columns left out.
    A role that rode along unconfirmed puts no column in a model (BLUEPRINT §14.1; the fit asks
    first, ``readings.predictors_or_ask``)."""
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import settled_roles

    return predictors_from_roles(settled_roles(state), state.target, left_out(state))


def modeling_frame(store: Any, columns: Sequence[str], row_ids: Any, *,
                   outcome: str | None = None) -> pd.DataFrame:
    """The raw inputs, as every modeling stage and preview reads them.

    Booleans become 0/1 floats (missing stays missing); text stays object with ``NaN`` for
    missing, which is what scikit-learn's imputer and encoder recognize.

    ``outcome`` (the target) keeps the values the table spells: it is never a model input, and its
    event is named by one of those levels (``set_event``). A True/False outcome turned into 1.0 and
    0.0 matched no declared level, so the event was not coded and every caption read "`1.0` against
    `0.0`" (the methods gate, item D); declared False, the model would have estimated the odds of
    True under a sentence saying False was coded 1.
    """
    frame = store.materialize(list(columns), row_ids)
    if outcome is None or outcome not in frame.columns:
        return normalize_frame(frame)
    out = normalize_frame(frame.drop(columns=[outcome]))
    values = frame[outcome]
    if pd.api.types.is_bool_dtype(values) and not values.isna().any():
        # True and False as booleans (an estimator reads a bool array as two classes; an object
        # array of bools it cannot type).
        values = pd.Series(values.to_numpy(dtype=bool), index=frame.index, name=outcome)
    elif not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        values = values.astype(object).where(values.notna(), np.nan)
    out.insert(list(frame.columns).index(outcome), outcome, values)
    return out


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Plain number columns pass as they are, so a 20,000-gene matrix with two text covariates
    # converts two columns, not 20,000 (column by column that took most of a second).
    plain = {dtype for dtype in set(frame.dtypes)
             if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)}
    odd = [name for name, dtype in zip(frame.columns, frame.dtypes) if dtype not in plain]
    if not odd:
        return frame
    out = frame.copy(deep=False)
    for name in odd:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        elif series.dtype == object:
            present = series.dropna()
            if len(present) and present.map(lambda v: isinstance(v, (bool, np.bool_))).all():
                series = series.map(lambda v: np.nan if v is None or v is pd.NA else float(v))
                series = series.astype(float)
            else:
                series = series.astype(object).where(series.notna(), np.nan)
        elif not pd.api.types.is_numeric_dtype(series):
            series = series.astype(object).where(series.notna(), np.nan)
        out[name] = series
    return out


def is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def missing_as_level(state: ProjectState) -> bool:
    spec = getattr(state, "missing", None)
    return spec is not None and spec.categorical == "missing_category"


def takes_level(dtype: str | None, n_unique: int | None) -> bool:
    """A column whose blanks can be a level of their own: categorical, or two values at most."""
    return dtype in LEVEL_DTYPES or (n_unique is not None and int(n_unique) <= 2)


def level_columns(state: ProjectState, predictors: Sequence[str],
                  column_info: Mapping[str, Any]) -> list[str]:
    """The predictors whose blanks become :data:`MISSING_LEVEL` under ``state``'s missing answer:
    categorical or two-valued columns with a blank somewhere in the table.

    ``column_info``: column -> ``{dtype, n_unique, n_missing}`` as the ingest records them over
    every row (a mapping or a ColumnInfo). The cohort and the design both read it, so they agree
    on which columns these are, whichever rows each happens to hold.
    """
    if not missing_as_level(state):
        return []
    out = []
    for c in predictors:
        info = column_info.get(c)
        if info is None:
            continue
        get = info.get if isinstance(info, Mapping) else (lambda k, i=info: getattr(i, k, None))
        n_missing = get("n_missing")
        if takes_level(get("dtype"), get("n_unique")) and (n_missing is None or int(n_missing) > 0):
            out.append(c)
    return out


def frame_level_columns(state: ProjectState, frame: pd.DataFrame, predictors: Sequence[str]) -> list[str]:
    """:func:`level_columns` read from a frame of raw inputs, where no table summary is at hand."""
    if not missing_as_level(state):
        return []
    return [c for c in predictors if c in frame.columns and frame[c].isna().any()
            and (is_categorical(frame[c]) or frame[c].nunique(dropna=True) <= 2)]


def _level_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


class MissingLevelEncoder(TransformerMixin, BaseEstimator):
    """One-hot encoding in which a blank is a level of its own, :data:`MISSING_LEVEL`.

    Per column, fit learns the observed levels in sorted order; the first is the reference (as
    the one-hot step's ``drop="first"``), every other level gets a ``<column>_<level>`` indicator,
    and blanks get ``<column>_Missing`` when the fitting rows have any. A level never seen in fit
    reads as the reference, as the one-hot step's ``handle_unknown="ignore"`` does. A blank when
    the fitting rows had none has no level of its own to read as: it reads as the fitting rows'
    most frequent level, as the categorical imputer fills one (audit B23), not as the reference
    whatever that happens to be. Every other column passes through. Row-local once fit: a row's
    output depends on its own values.
    """

    def __init__(self, columns: Sequence[str] = ()):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MissingLevelEncoder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingLevelEncoder needs a pandas DataFrame with named columns.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.levels_: dict[str, list[str]] = {}
        self.has_missing_: dict[str, bool] = {}
        self.mode_: dict[str, str | None] = {}
        for c in self.columns:
            labels = X[c].dropna().map(_level_label)
            self.levels_[c] = sorted(labels.unique().tolist())
            self.has_missing_[c] = bool(X[c].isna().any())
            counts = labels.value_counts()
            # the most frequent level, ties to the first in sorted order (SimpleImputer's rule)
            self.mode_[c] = (sorted(counts.index[counts == counts.max()].tolist())[0]
                             if len(counts) else None)
        return self

    def _outputs(self, column: str) -> list[tuple[str, str | None]]:
        """(output name, level it marks; None marks the blanks)."""
        out = [(f"{column}_{level}", level) for level in self.levels_[column][1:]]
        if self.has_missing_[column]:
            out.append((f"{column}_{MISSING_LEVEL}", None))
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "levels_"):
            raise ValueError("MissingLevelEncoder is not fitted yet.")
        encoded = set(self.columns)
        parts: dict[str, Any] = {}
        for name in X.columns:
            if name not in encoded:
                parts[str(name)] = X[name]
                continue
            blank = X[name].isna().to_numpy()
            unseen = None if self.has_missing_[str(name)] else getattr(self, "mode_", {}).get(str(name))
            labels = np.asarray([(unseen if b else _level_label(v)) for v, b in zip(X[name], blank)],
                                dtype=object)
            for out, level in self._outputs(str(name)):
                parts[out] = (blank if level is None else (labels == level)).astype(float)
        return pd.DataFrame(parts, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names: list[str] = []
        encoded = set(self.columns)
        for name in self.feature_names_in_:
            names.extend([out for out, _ in self._outputs(name)] if name in encoded else [name])
        return np.asarray(names, dtype=object)


@dataclass
class DesignSpec:
    """Everything a pipeline is built from; plain data, so it pickles and hashes cleanly."""

    predictors: list[str]
    inputs: list[str]  # the raw columns the pipeline takes: predictors, plus a strata column
    categorical: list[str]
    numeric: list[str]
    energy: dict[str, Any] | None  # EnergyAdjustment as a dict
    impute: bool
    roles: dict[str, str] = field(default_factory=dict)
    levels: list[str] = field(default_factory=list)  # predictors whose blanks are a level, "Missing"
    indicators: bool = False  # imputed numbers gain a missing indicator
    # WP11: the omics normalization the pipeline runs first, ``{method, kind, columns}``, or None
    normalization: dict[str, Any] | None = None
    lenses: list[str] = field(default_factory=list)  # the declared lenses (what the steps say)
    # predictor -> {"form": "spline" | "quintiles", "knots": k}: the non-linear forms (WP12a)
    exposure_forms: dict[str, Any] = field(default_factory=dict)
    # WP7: the missing-values answer (MissingSpec as a dict), the energy-aware single fill
    # ``{energy, nutrients}``, and the below-detection fill ``{method, columns}``
    missing: dict[str, Any] | None = None
    energy_fill: dict[str, Any] | None = None
    censored: dict[str, Any] | None = None
    # numbers with exactly two values: one indicator either way, filled by their most frequent value
    two_valued: list[str] = field(default_factory=list)

    def multiple_imputation(self) -> bool:
        return bool(self.missing) and self.missing.get("strategy") == "multiple_imputation"

    def energy_adjustment(self) -> EnergyAdjustment | None:
        return EnergyAdjustment(**self.energy) if self.energy else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignSpec":
        return cls(**dict(data))


STATE = "state"  # design_spec(energy=STATE): use the state's own energy-adjustment slot


def input_columns(predictors: Sequence[str], adjustment: EnergyAdjustment | None) -> list[str]:
    """Predictors plus a strata column the residual method needs but the model does not see."""
    columns = list(predictors)
    if (adjustment is not None and adjustment.method in RESIDUAL_METHODS and adjustment.strata
            and adjustment.strata not in columns):
        columns.append(adjustment.strata)
    return columns


def design_spec(state: ProjectState, frame: pd.DataFrame, predictors: Sequence[str],
                energy: Any = STATE, column_info: Mapping[str, Any] | None = None) -> DesignSpec:
    """The spec for ``frame`` (raw inputs) under ``state``; ``energy`` overrides the state's slot.

    ``column_info`` (the table's column summaries) decides which predictors keep their blanks as
    a level (:func:`level_columns`); without it, ``frame`` does (:func:`frame_level_columns`).
    """
    adj = state.energy_adjustment if isinstance(energy, str) and energy == STATE else energy
    predictors = [c for c in predictors]
    wanted = set(input_columns(predictors, adj))
    inputs = [c for c in frame.columns if c in wanted]
    # Text is a category; so is a numeric column the user said holds codes, wherever the answer
    # is kept (BLUEPRINT §14.3, every confirmation is honored): ``set_categorical`` (audit MA-15:
    # RIDRETH3's codes 1, 2, 3, 4, 6, 7 are six groups, not one straight line), the fit's own
    # code-or-amount question (``confirm_reading`` / ``confirm_readings``), or a combining rule that
    # took the most frequent value. A later "amount" answer for the column stands over an earlier
    # declaration.
    from turbotab.core.readings import confirmation, energy_plan

    # Total energy the energy answer "none" leaves out, and the columns an energy method computes
    # with (amounts by that answer), never reach the one-hot step as codes: the energy step has
    # removed or replaced them by then (the fifth gate: a "codes" answer for `energy_kcal` under
    # "none" crashed the design stage). A codes answer for one of them is asked about first
    # (``readings.predictors_or_ask``), never ignored.
    left, amounts = energy_plan(state, inputs)
    declared = {c for c in inputs if confirmation(state, "code_or_count", c) == "code"} \
        - left - amounts
    # Text the user said holds amounts is read as numbers on the working table
    # (``stages.working.text_amounts``); one still text (its values below a detection limit wait
    # for their answer) is never one-hot encoded instead (the sixth gate: 175 indicators for a
    # BMI confirmed as an amount). The fit asks first (``readings.predictors_or_ask``).
    unread = [c for c in inputs if is_categorical(frame[c]) and c not in left | amounts
              and confirmation(state, "code_or_count", c) == "amount"]
    if unread:
        raise ValueError(f"`{unread[0]}` is recorded as amounts, but some of its values are not "
                         f"numbers yet (values below a detection limit, or commas that read two "
                         f"ways); answer how they read before the fit.")
    categorical = [c for c in inputs if is_categorical(frame[c]) or c in declared]
    numeric = [c for c in inputs if c not in categorical]
    # A number with exactly two values is one indicator either way (``readings.code_question``);
    # its single fill is its most frequent value, as a code's is, never a median between the two.
    two_valued = [c for c in numeric if int(frame[c].dropna().nunique()) == 2]
    present = [c for c in predictors if c in inputs]
    levels = (level_columns(state, present, column_info) if column_info is not None
              else frame_level_columns(state, frame, present))
    # Multiple imputation (inference) keeps the in-fold fill for the cross-validated scores.
    impute = missing_strategy(state) in ("impute", "multiple_imputation")
    from turbotab.core.methods.missing import energy_fill
    from turbotab.core.methods.omics import design_normalization

    forms = {str(c): f.model_dump() for c, f in (getattr(state, "exposure_forms", None) or {}).items()
             if c in present and c in numeric and f.form != "linear"}
    # The roles the spec copies (the energy step, the energy-aware fill, the normalization) are the
    # settled ones only (BLUEPRINT §14.1, the readings ledger).
    from turbotab.core.readings import settled_roles

    roles = {str(k): str(v) for k, v in settled_roles(state).items()}
    spec_missing = state.missing.model_dump(mode="json") if getattr(state, "missing", None) else None
    censored = None
    if spec_missing and spec_missing.get("below_detection") in ("half_minimum", "censoring_aware"):
        cols = [c for c in spec_missing.get("censored_columns") or [] if c in numeric]
        if cols:
            censored = {"method": spec_missing["below_detection"], "columns": cols}
    return DesignSpec(
        predictors=predictors,
        inputs=inputs,
        categorical=categorical,
        numeric=numeric,
        energy=adj.model_dump() if adj is not None else None,
        impute=impute,
        roles=roles,
        levels=levels,
        indicators=bool(impute and state.missing is not None and state.missing.indicators),
        normalization=design_normalization(state, inputs),
        lenses=[str(k) for k in (getattr(state, "lens", None) or [])],
        exposure_forms=forms,
        missing=spec_missing,
        energy_fill=(energy_fill(adj.model_dump() if adj is not None else None, present, roles,
                                 [c for c in numeric if c not in levels]) if impute else None),
        censored=censored,
        two_valued=two_valued,
    )


# ── building ────────────────────────────────────────────────────────────────


def shared_steps(spec: DesignSpec) -> list[tuple[str, Any]]:
    """The steps every family shares: impute → energy adjustment → exposure form → levels →
    one-hot.

    Columns whose blanks are a level skip the imputer and the one-hot step: the levels step
    encodes them, blanks included.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    levels = [c for c in spec.levels if c in spec.predictors and c in spec.inputs]
    steps: list[tuple[str, Any]] = []
    censored = getattr(spec, "censored", None)
    if censored and censored.get("columns"):
        from turbotab.core.methods.missing import BelowDetectionFill

        steps.append(("detect", BelowDetectionFill(list(censored["columns"]), str(censored["method"]))))
    if spec.normalization:
        from turbotab.core.methods.omics import normalizer

        step = normalizer(spec.normalization)
        if step is not None:
            steps.append(("normalize", step))
    if spec.impute:
        two = [c for c in getattr(spec, "two_valued", None) or [] if c not in levels]
        numeric = [c for c in spec.numeric if c not in levels]
        categorical = [c for c in spec.categorical if c not in levels]
        parts = []
        fill = getattr(spec, "energy_fill", None)
        nutrients = [c for c in (fill or {}).get("nutrients") or [] if c in numeric]
        if numeric and fill and nutrients and fill.get("energy") in numeric:
            from turbotab.core.methods.missing import EnergyAwareImputer

            parts.append(("numeric", EnergyAwareImputer(energy=str(fill["energy"]), nutrients=nutrients,
                                                        two_valued=[c for c in two if c in numeric],
                                                        keep_empty_features=True,
                                                        add_indicator=spec.indicators), numeric))
        elif numeric and any(c in numeric for c in two):
            from turbotab.core.methods.missing import MedianFill

            parts.append(("numeric", MedianFill(two_valued=[c for c in two if c in numeric],
                                                keep_empty_features=True,
                                                add_indicator=spec.indicators), numeric))
        elif numeric:
            parts.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True,
                                                   add_indicator=spec.indicators), numeric))
        if categorical:
            parts.append(("categorical", SimpleImputer(strategy="most_frequent",
                                                       keep_empty_features=True), categorical))
        if parts:
            steps.append(("impute", ColumnTransformer(parts, remainder="passthrough",
                                                      verbose_feature_names_out=False)))
    step = energy_step(spec.energy_adjustment(), spec.predictors, spec.roles)
    if step is not None:
        steps.append(("energy", step))
    if spec.exposure_forms:
        from turbotab.core.methods.exposure_form import ExposureForms, adjusted_forms

        # After the energy step a formed nutrient may carry the step's name (``protein_adj``).
        steps.append(("form", ExposureForms(adjusted_forms(spec.exposure_forms, spec.energy))))
    if levels:
        steps.append(("levels", MissingLevelEncoder(levels)))
    categorical = [c for c in spec.categorical if c in spec.predictors and c not in levels]
    if categorical:
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        steps.append(("onehot", ColumnTransformer([("onehot", encoder, categorical)],
                                                  remainder="passthrough",
                                                  verbose_feature_names_out=False)))
    return steps


def family_steps(spec: DesignSpec, family: ModelFamily) -> list[tuple[str, Any]]:
    """The shared steps plus what this family declares it needs.

    A family may add steps of its own (``preprocess(spec) -> [(name, step), ...]``, e.g. a spline
    basis); they run after the shared steps and before scaling. The lineage, the step list and the
    select_models preview all read them from the pipeline, so a new family needs nothing else.
    """
    from sklearn.preprocessing import StandardScaler

    steps = shared_steps(spec)
    extra = getattr(family, "preprocess", None)
    if extra is not None:
        steps.extend(extra(spec))
    if family.needs_scaling:
        steps.append(("scale", StandardScaler()))
    return steps


def build_pipeline(spec: DesignSpec, family: ModelFamily, task: Task, purpose: Purpose | None,
                   n_rows: int, n_features: int) -> Any:
    from sklearn.pipeline import Pipeline

    steps = family_steps(spec, family)
    # A family whose model step depends on the design (which columns it tests) builds from the spec.
    build_for = getattr(family, "build_for", None)
    model = (build_for(spec, task, purpose, n_rows, n_features) if build_for is not None
             else family.build(task, purpose, n_rows, n_features))
    steps.append(("model", model))
    return Pipeline(steps).set_output(transform="pandas")


def transformer(steps: list[tuple[str, Any]]) -> Any:
    """A pipeline of transformer steps (no model), pandas out; identity when there are none."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    if not steps:
        steps = [("identity", FunctionTransformer(feature_names_out="one-to-one"))]
    return Pipeline(steps).set_output(transform="pandas")


# ── describing ──────────────────────────────────────────────────────────────


def energy_detail(adj: EnergyAdjustment | None, energy: Sequence[str] = (),
                  purpose: Purpose | None = None) -> str | None:
    """One line on what the energy step does; ``energy`` names the total-energy columns "none"
    takes out of the model."""
    if adj is None:
        return None
    if adj.method == "none":
        gone = list(energy) or ([adj.energy_column] if adj.energy_column else [])
        if not gone:
            return None
        verb = "leaves" if len(gone) == 1 else "leave"
        return f"{', '.join(gone)} {verb} the model; nutrients enter as absolute intakes."
    nutrients = len(adj.nutrients)
    what = f"{nutrients} nutrient{'s' if nutrients != 1 else ''}"
    E = adj.energy_column
    if adj.method in RESIDUAL_METHODS:
        log = " on the log scale" if adj.log_transform else ""
        within = f" within each level of {adj.strata}" if adj.strata else ""
        where = (f"{E} stays in the model beside them" if adj.method == "residual"
                 else f"{E} leaves the outcome model")
        rows = "every analyzed row" if purpose == "inference" else "training rows"
        return f"Replaces {what} with their residual on {E}{within}{log}, fit on {rows}; {where}."
    if adj.method == "standard":
        return f"Keeps {what} as they are, with {E} in the model beside them."
    if adj.method == "density_multivariate":
        return f"Divides {what} by {E}; {E} stays in the model as its own term."
    if adj.method == "density":
        return f"Divides {what} by {E}; {E} leaves the model."
    if adj.method == "all_components":
        return (f"Splits {E} into kcal from {what}, each its own term, and kcal from everything "
                f"else; {E} leaves the model.")
    return f"Splits {E} into kcal from {what} and kcal from everything else; {E} leaves the model."


def describe_steps(spec: DesignSpec, family: ModelFamily, task: Task,
                   purpose: Purpose | None, n_matrix_columns: int | None = None) -> list[dict[str, str]]:
    """``[{key, label, detail}]`` for each step of this family's pipeline, in order."""
    out: list[dict[str, str]] = []
    for name, step in family_steps(spec, family):
        if name == "normalize":
            from turbotab.core.methods.omics import describe as describe_normalization

            label, detail = describe_normalization(spec.normalization or {}) or ("Normalize", "")
            out.append({"key": "normalize", "label": label, "detail": detail})
        elif name == "impute":
            out.append({"key": "impute", "label": "Fill missing values",
                        "detail": impute_detail(spec, step)})
        elif name == "detect":
            out.append({"key": "detect", "label": "Fill values below detection",
                        "detail": detect_detail(spec)})
        elif name == "form":
            from turbotab.core.methods.exposure_form import describe as describe_forms

            out.append({"key": "form", "label": "Exposure form",
                        "detail": describe_forms(step.forms)})
        elif name == "levels":
            cols = [c for c in spec.levels if c in spec.predictors]
            verb = "becomes" if len(cols) == 1 else "become"
            out.append({"key": "levels", "label": "Blanks as a level",
                        "detail": f"{', '.join(cols)} {verb} indicator columns with a blank as its "
                                  f"own level, {MISSING_LEVEL}; the first level is the reference."})
        elif name == "energy":
            adj = spec.energy_adjustment()
            energy = [c for c in spec.predictors if spec.roles.get(c) == "energy"]
            out.append({"key": "energy", "label": METHOD_TABLE[adj.method]["label"],
                        "detail": energy_detail(adj, energy, purpose) or ""})
        elif name == "onehot":
            cats = [c for c in spec.categorical if c in spec.predictors]
            verb = "becomes" if len(cats) == 1 else "become"
            out.append({"key": "onehot", "label": "One-hot encode",
                        "detail": f"{', '.join(cats)} {verb} indicator columns; the first level "
                                  f"is the reference."})
        elif name == "scale":
            width = f"all {n_matrix_columns} columns" if n_matrix_columns else "every column"
            detail = f"Centers and scales {width} on the training fold."
            if "metabolomics" in spec.lenses:
                # F17: a departure from the field's custom is named, with its justification.
                detail += (" This is autoscaling, to unit variance; Pareto scaling is customary in "
                           "metabolomics, and van den Berg et al. (2006) found autoscaling performed "
                           "better in their explorative analysis.")
            out.append({"key": "scale", "label": "Standardize", "detail": detail})
        else:
            custom = getattr(family, "describe_step", None)
            label, detail = (custom(name) if custom else None) or (
                name.replace("_", " ").capitalize(), "")
            out.append({"key": name, "label": label, "detail": detail})
    label, detail = family.describe(task, purpose)
    out.append({"key": "model", "label": label, "detail": detail})
    return out


def impute_detail(spec: DesignSpec, step: Any = None) -> str:
    """The impute step's line: what fills which blanks, learned where, and under multiple
    imputation what the step is for."""
    marked = "; each filled number also gets a missing indicator" if spec.indicators else ""
    fill = getattr(spec, "energy_fill", None) or {}
    nutrients = list(fill.get("nutrients") or [])
    numeric = getattr(step, "named_transformers_", {}).get("numeric") if step is not None else None
    if numeric is None and step is not None:
        numeric = next((t for n, t, _ in getattr(step, "transformers", []) if n == "numeric"), None)
    energy_aware = numeric is not None and type(numeric).__name__ == "EnergyAwareImputer"
    by_energy = (f"; {', '.join(nutrients[:4])}{' and others' if len(nutrients) > 4 else ''} from "
                 f"{'its' if len(nutrients) == 1 else 'their'} line on {fill.get('energy')}"
                 if energy_aware and nutrients else "")
    two = list(getattr(spec, "two_valued", None) or [])
    pairs = (f" and for numbers with two values ({', '.join(two[:3])}"
             f"{' and others' if len(two) > 3 else ''})" if two else "")
    text = (f"Median for numbers{by_energy}, most frequent value for categories{pairs}, learned "
            f"within each training fold without the outcome{marked}.")
    if spec.multiple_imputation():
        m = int((spec.missing or {}).get("m") or 20)
        text += (f" It serves the cross-validated scores; the coefficients are pooled over {m} "
                 f"multiple imputations with the outcome.")
    return text


def detect_detail(spec: DesignSpec) -> str:
    censored = getattr(spec, "censored", None) or {}
    cols = list(censored.get("columns") or [])
    named = f"{', '.join(cols[:3])}{f' and {len(cols) - 3:,} more' if len(cols) > 3 else ''}"
    if censored.get("method") == "half_minimum":
        return (f"Blanks in {named} are values below detection: each becomes half the column's "
                f"smallest detected value, learned within each training fold.")
    return (f"Blanks in {named} are values below detection: each becomes its expected value below "
            f"the smallest detected one under a censored-normal fit, learned within each training "
            f"fold without the outcome.")


def warnings_for(spec: DesignSpec, frame: pd.DataFrame, family_keys: Sequence[str],
                 families_by_key: Mapping[str, ModelFamily],
                 nested: Mapping[str, str] | None = None,
                 rows_word: str = "training rows") -> list[str]:
    """Plain statements about the design a reader should know before trusting the results.

    ``nested``: child -> parent among the predictors (``turbotab.core.methods.nesting``);
    ``rows_word`` names ``frame``'s rows (every analyzed row under inference).
    """
    from turbotab.core.methods.energy import nutrient_role
    from turbotab.core.methods.nesting import parts_of

    out: list[str] = []
    for c in spec.categorical:
        if c in spec.predictors:
            levels = int(frame[c].nunique(dropna=True))
            if levels > MANY_LEVELS:
                out.append(f"{c} has {levels} levels, so one-hot encoding adds {levels - 1} columns.")
    adj = spec.energy_adjustment()
    if adj is not None and adj.strata and adj.method not in RESIDUAL_METHODS:
        out.append(f"Strata apply to the residual method only; {adj.strata} does not change the "
                   f"{METHOD_TABLE[adj.method]['label'].lower()}.")
    from turbotab.core.methods.energy import fiber_beside_carbohydrate

    fiber = fiber_beside_carbohydrate([c for c in spec.predictors if spec.roles.get(c) == "exposure"])
    if fiber:
        out.append(fiber)
    by_role: dict[str, list[str]] = {}
    for c in spec.predictors:
        if spec.roles.get(c) != "exposure":
            continue
        try:
            role = nutrient_role(c)
        except ValueError:
            role = None
        if role:
            by_role.setdefault(role, []).append(c)
    from turbotab.core.methods.nesting import compositions

    shares = compositions(frame, [c for c in spec.predictors if spec.roles.get(c) == "exposure"
                                  and c in frame.columns])
    if shares:
        out.append(f"{', '.join(shares)} sum to 100% on every row: beside the intercept one of them "
                   f"is fixed by the others, so a linear model cannot estimate them all; leave one "
                   f"out as the reference.")
    nested = dict(nested or {})
    for parent, parts in parts_of(nested).items():
        if parent in spec.predictors:
            verb = "is a part" if len(parts) == 1 else "are parts"
            them = "it" if len(parts) == 1 else "them"
            # Audit ME-15: beside its parts, a total's coefficient is the remainder in none of them.
            out.append(f"{', '.join(parts)} {verb} of {parent}: {parent}'s coefficient is the "
                       f"remainder (holding {', '.join(parts)} fixed), not total {parent}; leave "
                       f"{them} out to estimate the total. A substitution through {parent} moves "
                       f"{them} in proportion, and never pairs {parent} with its own part.")
    for role, cols in by_role.items():
        loose = [c for c in cols if c not in nested and c not in parts_of(nested)]
        if len(cols) > 1 and len(loose) > 1:
            out.append(f"{', '.join(loose)} all read as {role}, and none is part of another. If some "
                       f"overlap, moving energy through one while the rest stay fixed is not a "
                       f"coherent substitution.")
    if not spec.impute:
        # Predictors only (as the cohort's complete cases): a strata column that is not one is the
        # energy step's input, where a blank is a level of its own. In spec.levels a blank is too.
        valued = [c for c in spec.predictors if c in spec.inputs and c not in spec.levels]
        incomplete = frame[valued].isna().any(axis=1)
        n_bad = int(incomplete.sum())
        if n_bad:
            cannot = [families_by_key[k].label for k in family_keys
                      if not families_by_key[k].handles_missing]
            if cannot:
                raise ValueError(
                    f"{n_bad:,} {rows_word} have a missing predictor value and no missing-values "
                    f"strategy was chosen; {', '.join(cannot)} cannot use them. Choose complete "
                    f"cases or imputation first.")
    return out


__all__ = [
    "ADJUST_STEPS", "DesignSpec", "MISSING_LEVEL", "MissingLevelEncoder", "PREDICTOR_ROLES",
    "build_pipeline", "describe_steps", "design_spec", "detect_detail", "energy_detail",
    "family_steps", "impute_detail",
    "frame_level_columns", "input_columns", "is_categorical", "level_columns", "missing_as_level",
    "model_predictors", "modeling_frame", "normalize_frame", "predictors_from_roles",
    "shared_steps", "takes_level", "transformer", "warnings_for",
]
