"""The model pipeline every family shares, and what the design stage says about it.

Step order (M1_CONTRACT §3): impute (when the answer is "impute") → energy adjustment (when the
method is not none) → one-hot for categorical columns → scale (families that need it) → the
model. Every step is fit inside the pipeline, so cross-validation refits it on each training fold
and nothing learned from the data ever sees a held-out row. Column names survive every step
(``set_output(transform="pandas")``), and so does the row-id index, which the lineage and the
leakage tests rely on.

The pipeline runs on RAW inputs: the columns as :func:`modeling_frame` reads them. Substitution
curves shift raw intakes and push them through the whole pipeline, energy adjustment included.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.decisions import EnergyAdjustment, ProjectState, Purpose, Task, missing_strategy
from turbotab.core.methods.energy import METHOD_TABLE
from turbotab.core.models.base import ModelFamily
from turbotab.core.models.steps import energy_step

PREDICTOR_ROLES = ("exposure", "covariate", "energy")
ADJUST_STEPS = ("impute", "energy")  # the steps whose outputs form the lineage's "adjusted" lane
MANY_LEVELS = 20


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
    """The predictors the models get under ``state``: by role, less the columns left out."""
    from turbotab.core.decisions import left_out

    return predictors_from_roles(state.roles, state.target, left_out(state))


def modeling_frame(store: Any, columns: Sequence[str], row_ids: Any) -> pd.DataFrame:
    """The raw inputs, as every modeling stage and preview reads them.

    Booleans become 0/1 floats (missing stays missing); text stays object with ``NaN`` for
    missing, which is what scikit-learn's imputer and encoder recognize.
    """
    frame = store.materialize(list(columns), row_ids)
    return normalize_frame(frame)


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
    if (adjustment is not None and adjustment.method == "residual" and adjustment.strata
            and adjustment.strata not in columns):
        columns.append(adjustment.strata)
    return columns


def design_spec(state: ProjectState, frame: pd.DataFrame, predictors: Sequence[str],
                energy: Any = STATE) -> DesignSpec:
    """The spec for ``frame`` (raw inputs) under ``state``; ``energy`` overrides the state's slot."""
    adj = state.energy_adjustment if isinstance(energy, str) and energy == STATE else energy
    predictors = [c for c in predictors]
    wanted = set(input_columns(predictors, adj))
    inputs = [c for c in frame.columns if c in wanted]
    categorical = [c for c in inputs if is_categorical(frame[c])]
    numeric = [c for c in inputs if c not in categorical]
    return DesignSpec(
        predictors=predictors,
        inputs=inputs,
        categorical=categorical,
        numeric=numeric,
        energy=adj.model_dump() if adj is not None else None,
        impute=missing_strategy(state) == "impute",
        roles={str(k): str(v) for k, v in (state.roles or {}).items()},
    )


# ── building ────────────────────────────────────────────────────────────────


def shared_steps(spec: DesignSpec) -> list[tuple[str, Any]]:
    """The steps every family shares: impute → energy adjustment → one-hot."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    steps: list[tuple[str, Any]] = []
    if spec.impute:
        parts = []
        if spec.numeric:
            parts.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
                          list(spec.numeric)))
        if spec.categorical:
            parts.append(("categorical", SimpleImputer(strategy="most_frequent",
                                                       keep_empty_features=True),
                          list(spec.categorical)))
        steps.append(("impute", ColumnTransformer(parts, remainder="passthrough",
                                                  verbose_feature_names_out=False)))
    step = energy_step(spec.energy_adjustment(), spec.predictors)
    if step is not None:
        steps.append(("energy", step))
    categorical = [c for c in spec.categorical if c in spec.predictors]
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
    steps.append(("model", family.build(task, purpose, n_rows, n_features)))
    return Pipeline(steps).set_output(transform="pandas")


def transformer(steps: list[tuple[str, Any]]) -> Any:
    """A pipeline of transformer steps (no model), pandas out; identity when there are none."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    if not steps:
        steps = [("identity", FunctionTransformer(feature_names_out="one-to-one"))]
    return Pipeline(steps).set_output(transform="pandas")


# ── describing ──────────────────────────────────────────────────────────────


def energy_detail(adj: EnergyAdjustment | None) -> str | None:
    if adj is None or adj.method == "none":
        return None
    nutrients = len(adj.nutrients)
    what = f"{nutrients} nutrient{'s' if nutrients != 1 else ''}"
    E = adj.energy_column
    if adj.method == "residual":
        log = " on the log scale" if adj.log_transform else ""
        within = f" within each level of {adj.strata}" if adj.strata else ""
        return f"Replaces {what} with their residual on {E}{within}{log}, fit on training rows."
    if adj.method == "standard":
        return f"Keeps {what} as they are, with {E} in the model beside them."
    if adj.method == "density_multivariate":
        return f"Divides {what} by {E}; {E} stays in the model as its own term."
    if adj.method == "density":
        return f"Divides {what} by {E}; {E} leaves the model."
    return f"Splits {E} into kcal from {what} and kcal from everything else; {E} leaves the model."


def describe_steps(spec: DesignSpec, family: ModelFamily, task: Task,
                   purpose: Purpose | None, n_matrix_columns: int | None = None) -> list[dict[str, str]]:
    """``[{key, label, detail}]`` for each step of this family's pipeline, in order."""
    out: list[dict[str, str]] = []
    for name, _ in family_steps(spec, family):
        if name == "impute":
            out.append({"key": "impute", "label": "Fill missing values",
                        "detail": "Median for numbers, most frequent value for categories, "
                                  "learned within each training fold."})
        elif name == "energy":
            adj = spec.energy_adjustment()
            out.append({"key": "energy", "label": METHOD_TABLE[adj.method]["label"],
                        "detail": energy_detail(adj) or ""})
        elif name == "onehot":
            cats = [c for c in spec.categorical if c in spec.predictors]
            verb = "becomes" if len(cats) == 1 else "become"
            out.append({"key": "onehot", "label": "One-hot encode",
                        "detail": f"{', '.join(cats)} {verb} indicator columns; the first level "
                                  f"is the reference."})
        elif name == "scale":
            width = f"all {n_matrix_columns} columns" if n_matrix_columns else "every column"
            out.append({"key": "scale", "label": "Standardize",
                        "detail": f"Centers and scales {width} on the training fold."})
        else:
            custom = getattr(family, "describe_step", None)
            label, detail = (custom(name) if custom else None) or (
                name.replace("_", " ").capitalize(), "")
            out.append({"key": name, "label": label, "detail": detail})
    label, detail = family.describe(task, purpose)
    out.append({"key": "model", "label": label, "detail": detail})
    return out


def warnings_for(spec: DesignSpec, frame: pd.DataFrame, family_keys: Sequence[str],
                 families_by_key: Mapping[str, ModelFamily],
                 nested: Mapping[str, str] | None = None) -> list[str]:
    """Plain statements about the design a reader should know before trusting the results.

    ``nested``: child -> parent among the predictors (``turbotab.core.methods.nesting``).
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
    if adj is not None and adj.strata and adj.method != "residual":
        out.append(f"Strata apply to the residual method only; {adj.strata} does not change the "
                   f"{METHOD_TABLE[adj.method]['label'].lower()}.")
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
            out.append(f"{', '.join(parts)} {verb} of {parent}: a substitution through {parent} "
                       f"moves {'it' if len(parts) == 1 else 'them'} in proportion, and never pairs "
                       f"{parent} with its own part.")
    for role, cols in by_role.items():
        loose = [c for c in cols if c not in nested and c not in parts_of(nested)]
        if len(cols) > 1 and len(loose) > 1:
            out.append(f"{', '.join(loose)} all read as {role}, and none is part of another. If some "
                       f"overlap, moving energy through one while the rest stay fixed is not a "
                       f"coherent substitution.")
    if not spec.impute:
        incomplete = frame[spec.inputs].isna().any(axis=1)
        n_bad = int(incomplete.sum())
        if n_bad:
            cannot = [families_by_key[k].label for k in family_keys
                      if not families_by_key[k].handles_missing]
            if cannot:
                raise ValueError(
                    f"{n_bad:,} training rows have a missing predictor value and no missing-values "
                    f"strategy was chosen; {', '.join(cannot)} cannot use them. Choose complete "
                    f"cases or imputation first.")
    return out


__all__ = [
    "ADJUST_STEPS", "DesignSpec", "PREDICTOR_ROLES", "build_pipeline", "describe_steps",
    "design_spec", "energy_detail", "family_steps", "input_columns", "is_categorical",
    "model_predictors", "modeling_frame", "normalize_frame", "predictors_from_roles",
    "shared_steps", "transformer",
    "warnings_for",
]
