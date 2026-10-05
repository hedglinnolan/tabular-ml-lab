"""Missing data by purpose (AUDIT_REPORT §5 WP7; closes ME-01, ME-08, and the minor E14, B23, F14).

**One rule** (BLUEPRINT §12 ruling 4), stated once here and quoted by the MISSING drawer, ROADMAP §07
and M2_CONTRACT §4 (:data:`RULE`):

* **Inference** — multiple imputation by chained equations with the outcome and total energy in the
  imputation model, m ≥ 20, pooled by Rubin's rules (``turbotab.core.methods.imputation``), ranked
  first. Moons et al. 2006 (*J Clin Epidemiol* 59:1092): "For all types of missing values,
  imputation of missing predictor values using the outcome is preferred over imputation without
  outcome"; Harrell (*Regression Modeling Strategies*, §3.8): "multiple imputation can and should
  use the response variable for imputing predictors". Complete cases stay on the menu with their
  assumption stated (:data:`COMPLETE_CASE_ASSUMPTION`) and a concern when they lose more than 10% of
  the rows. A single fill and the missing-indicator method (a missing indicator, or blanks as a
  level) are *blocked and recorded* for the inference table: kept only with the attestation the
  methods sentence carries. Groenwold et al. 2012 (*CMAJ* 184:1265): in nonrandomized studies "the
  missing-indicator method will almost always give biased results", while in randomized trials "the
  missing-indicator method is a valid method to handle missing baseline covariate data".
* **Prediction** — imputation inside each training fold without the outcome (unchanged), so the
  fitted pipeline imputes a new row the way it was developed. Sisk et al. 2023 (*Stat Methods Med
  Res* 32:1461): "the outcome should be used to impute development data when using multiple
  imputation and omitted under regression imputation. When missingness is allowed at deployment,
  omitting the outcome from the imputation model at the development was preferred." Multiple
  imputation with the outcome is refused under prediction: a held-out row's outcome would fill its
  own predictors.

**Energy-aware fill** (either purpose; audit D11): a single fill of an energy-bearing nutrient is its
in-fold least-squares line on total energy, not the nutrient's median, so an imputed row keeps the
nutrient–energy relation every later step (the residual method above all) depends on. With the
median, an imputed row's energy-adjusted value is the median minus the fitted line, which
correlates −1 with energy (the audit: −1.000 against +0.78 for observed rows' intakes).

**Values below a detection limit** (ME-08): a column the left-censoring finding names, or whose zeros
the user recoded as non-detections, holds blanks that are known to be small. Half the column's
minimum (MetaboAnalyst's default, customary) or a censoring-aware fill: under inference a
censored-normal (Tobit) draw within the multiple imputation, conditional on the outcome (Lubin et
al. 2004, *Environ Health Perspect* 112:1691); under prediction, in-fold and outcome-free, each
non-detect becomes its expected value below the limit under a censored-normal fit to the column. A
median fill there puts non-detections in the middle of the distribution; it is refused unless the
answer gives a reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# ── the rule and the sentences every surface shares ─────────────────────────

SISK = "Sisk et al. 2023"
MOONS = "Moons et al. 2006, via Harrell"
RULE = ("The outcome's place in the imputation model depends on the purpose. Under inference, "
        "missing predictors are multiply imputed with the outcome and total energy in the "
        "imputation model, and the analyses pooled by Rubin's rules (Moons et al. 2006, via "
        "Harrell). Under prediction, they are imputed inside each training fold without the "
        "outcome, so the fitted pipeline can impute a new row as it was developed (Sisk et al. "
        "2023).")
COMPLETE_CASE_ASSUMPTION = ("complete cases give unbiased coefficients only if whether a value is "
                            "missing does not depend on the outcome, given the predictors "
                            "(Groenwold et al. 2012; White & Carlin 2010)")
MI_ASSUMPTION = ("multiple imputation assumes the blanks are missing at random given the variables "
                 "in the imputation model")
SINGLE_FILL_CAUTION = ("a single fill treats each imputed value as measured, so the intervals are "
                       "too narrow, and a median fill biases a coefficient whenever who is missing "
                       "depends on another variable")
INDICATOR_CAUTION = ("the missing-indicator method almost always biases the estimates of a "
                     "nonrandomized study (Groenwold et al. 2012); it is valid for baseline "
                     "covariates in a randomized trial")
# CLINICAL_SURVEY_PACK, anti-pattern 1: "Flag when listwise deletion drops >10% of rows."
ROW_LOSS_SHARE = 0.10
MAX_COMPARED = 4  # predictors the row-loss concern names, largest standardized difference first

Purpose = Literal["prediction", "inference"]
Rung = Literal["recommended", "available", "block_and_record", "refused"]


@dataclass(frozen=True)
class MissingMethod:
    """One way to handle missing predictor values, with its two labels (north star 5): customary
    in the field (with a source), and sound for each purpose (with the reason)."""

    key: str
    label: str
    customary: str
    sound: Mapping[str, str]  # purpose -> why it is (or is not) sound
    rung: Mapping[str, Rung]  # purpose -> how the guidance treats it
    order: Mapping[str, int]  # purpose -> rank, soundest first
    decision: Mapping[str, Any]  # the set_missing fields that choose it

    def for_purpose(self, purpose: str) -> dict[str, Any]:
        p = purpose if purpose in self.order else "prediction"
        return {"key": self.key, "label": self.label, "customary": self.customary,
                "sound": self.sound[p], "rung": self.rung[p], "decision": dict(self.decision)}


METHODS: dict[str, MissingMethod] = {}


def register_method(method: MissingMethod) -> MissingMethod:
    METHODS[method.key] = method
    return method


register_method(MissingMethod(
    key="multiple_imputation", label="Multiple imputation (m = 20)",
    customary="Customary in epidemiology (Sterne et al. 2009, BMJ 338:b2393)",
    sound={"inference": "Sound: with the outcome and energy in the imputation model it is unbiased "
                        "under missing at random and its intervals carry the imputation's "
                        "uncertainty (Moons et al. 2006).",
           "prediction": "Not here: it uses the outcome, which a new row does not have; impute in "
                         "each fold without it (Sisk et al. 2023)."},
    rung={"inference": "recommended", "prediction": "refused"},
    order={"inference": 0, "prediction": 4},
    decision={"strategy": "multiple_imputation"}))
register_method(MissingMethod(
    key="complete_case", label="Complete cases",
    customary="Customary in epidemiology; most software's default",
    sound={"inference": f"Sound if {COMPLETE_CASE_ASSUMPTION}; it costs every incomplete row.",
           "prediction": "Valid but wasteful, and a new row with a blank cannot be scored."},
    rung={"inference": "available", "prediction": "available"},
    order={"inference": 1, "prediction": 2},
    decision={"strategy": "complete_case"}))
register_method(MissingMethod(
    key="impute", label="Single fill in each training fold",
    customary="Customary in machine-learning pipelines (scikit-learn's SimpleImputer)",
    sound={"inference": f"Unsound for an interval: {SINGLE_FILL_CAUTION}.",
           "prediction": "Sound: fit in each training fold without the outcome, it imputes a new "
                         "row as it was developed (Josse et al. 2019; Sisk et al. 2023)."},
    rung={"inference": "block_and_record", "prediction": "recommended"},
    order={"inference": 2, "prediction": 0},
    decision={"strategy": "impute"}))
register_method(MissingMethod(
    key="indicators", label="Single fill with missing indicators",
    customary="Customary in prediction from health records (Sperrin et al. 2020)",
    sound={"inference": f"Unsound for an estimate: {INDICATOR_CAUTION}.",
           "prediction": "Sound when a blank carries information, but harmful when missingness "
                         "depends on the outcome (Sisk et al. 2023)."},
    rung={"inference": "block_and_record", "prediction": "available"},
    order={"inference": 3, "prediction": 1},
    decision={"strategy": "impute", "indicators": True}))
register_method(MissingMethod(
    key="missing_category", label="Blanks as their own level",
    customary="Customary for questions not asked (a blank medication answer)",
    sound={"inference": f"The missing-indicator method for categories: {INDICATOR_CAUTION}; sound "
                        f"when a blank means not asked (then it is a value, not missing).",
           "prediction": "Sound: keeps the signal when a blank means not asked."},
    rung={"inference": "block_and_record", "prediction": "available"},
    order={"inference": 4, "prediction": 3},
    decision={"strategy": "impute", "categorical": "missing_category"}))


def methods_for(purpose: str | None) -> list[dict[str, Any]]:
    """The missing-data methods, soundest first for ``purpose`` (prediction when undeclared)."""
    p = purpose if purpose in ("prediction", "inference") else "prediction"
    return [m.for_purpose(p) for m in sorted(METHODS.values(), key=lambda m: m.order[p])]


BelowDetection = Literal["half_minimum", "censoring_aware", "as_missing", "qrilc"]
BELOW_DETECTION_LABELS = {
    "half_minimum": "Half the smallest detected value",
    "censoring_aware": "Censoring-aware",
    "as_missing": "As any other blank (the median fill)",
    "qrilc": "QRILC (quantile regression imputation of left-censored data)",
}


def below_detection_options(purpose: str | None) -> list[dict[str, Any]]:
    """The below-detection answers, soundest first for ``purpose``."""
    inference = purpose == "inference"
    aware = ("A censored-normal (Tobit) draw below the limit within the multiple imputation, given "
             "the outcome (Lubin et al. 2004)." if inference else
             "Each non-detect becomes its expected value below the limit under a censored-normal "
             "fit, in each training fold without the outcome.")
    return [
        {"key": "censoring_aware", "label": BELOW_DETECTION_LABELS["censoring_aware"],
         "customary": "Less common in metabolomics; the epidemiology of exposures below detection "
                      "(Lubin et al. 2004)",
         "sound": f"Sound for associations. {aware}", "rung": "recommended"},
        {"key": "half_minimum", "label": BELOW_DETECTION_LABELS["half_minimum"],
         "customary": "Customary: MetaboAnalyst's default",
         "sound": "Biased for associations as censoring grows, and collapses every non-detect to "
                  "one value (Lubin et al. 2004: biased unless 5–10% are below the limit).",
         "rung": "available"},
        {"key": "as_missing", "label": BELOW_DETECTION_LABELS["as_missing"],
         "customary": "Not customary for non-detects",
         "sound": "Unsound: the median places non-detections in the middle of the distribution.",
         "rung": "refused"},
    ]


# ── which columns are left-censored ─────────────────────────────────────────

LEFT_CENSORED = "pack::metabolomics::left_censored"
ZEROS = "pack::metabolomics::zeros_or_missing"


def censored_columns(findings: Any = None, state: Any = None) -> list[str]:
    """Columns whose blanks lie below a detection limit: those the left-censoring finding names,
    and those whose zeros the user recoded as non-detections (the ``zeros_as_nondetects`` repair).
    Zeros are never read as non-detections unless the user said so."""
    out: list[str] = []
    data = getattr(findings, "data", findings)
    items = (data or {}).get("findings") if isinstance(data, Mapping) else None
    for f in items or []:
        if str(f.get("id")).split("#")[0] == LEFT_CENSORED:
            cols = f.get("censored_columns") or f.get("affected_columns") or []
            out.extend(str(c) for c in cols)
    if state is not None:
        from turbotab.core.repairs import nondetect_columns

        out.extend(nondetect_columns(state))
    return list(dict.fromkeys(out))


# ── the pipeline steps ───────────────────────────────────────────────────────


def energy_fill(spec_energy: Mapping[str, Any] | None, predictors: Sequence[str],
                roles: Mapping[str, str], numeric: Sequence[str]) -> dict[str, Any] | None:
    """``{energy, nutrients}`` for an energy-aware single fill, or None: the nutrients the energy
    adjustment names (else the energy-bearing exposures) and the total-energy column, when both are
    numeric inputs."""
    from turbotab.core.stages.rows import energy_bearing

    numeric_set = set(numeric)
    energy = None
    if spec_energy and spec_energy.get("energy_column"):
        energy = str(spec_energy["energy_column"])
    if energy is None or energy not in numeric_set:
        energy = next((c for c in predictors if roles.get(c) == "energy" and c in numeric_set), None)
    if energy is None:
        return None
    named = [str(c) for c in (spec_energy or {}).get("nutrients") or []]
    nutrients = [c for c in named if c in numeric_set and c != energy]
    if not nutrients:
        nutrients = [c for c in predictors if roles.get(c) == "exposure" and c in numeric_set
                     and c != energy and energy_bearing(c)]
    return {"energy": energy, "nutrients": nutrients} if nutrients else None


class MedianFill(SimpleImputer):
    """SimpleImputer's median fill, except that a number with exactly two values (``two_valued``:
    a 1/2-coded sex, a 0/1 smoker) is filled by its most frequent value on the fitting rows, as a
    code is (the smallest of tied values, SimpleImputer's own rule), never by a median between the
    two (BLUEPRINT §14.3: such a column is one indicator either way, so its two readings must fill
    it alike). Everything else, the indicators included, is SimpleImputer's."""

    def __init__(self, *, two_valued: Sequence[str] = (), strategy: str = "median",
                 keep_empty_features: bool = True, add_indicator: bool = False,
                 missing_values: Any = np.nan, fill_value: Any = None, copy: bool = True):
        super().__init__(missing_values=missing_values, strategy=strategy, fill_value=fill_value,
                         copy=copy, add_indicator=add_indicator,
                         keep_empty_features=keep_empty_features)
        self.two_valued = two_valued

    def fit(self, X: Any, y: Any = None) -> "MedianFill":
        super().fit(X, y)
        two = set(self.two_valued or ())
        if not two:
            return self
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        for i, c in enumerate(self.feature_names_in_):
            if str(c) in two:
                present = pd.to_numeric(frame[c], errors="coerce").dropna()
                if len(present):
                    self.statistics_[i] = float(present.mode().iloc[0])
        return self


class EnergyAwareImputer(MedianFill):
    """A median fill (most frequent for categories is the caller's other part) in which each
    nutrient in ``nutrients`` is filled from its least-squares line on ``energy`` instead.

    Fit on the fitting rows only (in-fold): each nutrient's line is fit on the rows where both it
    and energy are observed; a row missing energy too reads energy at its median. A fill is kept
    within the nutrient's observed range on those rows. Every other column, the indicators and the
    medians (``statistics_``) are exactly :class:`MedianFill`'s.
    """

    def __init__(self, *, energy: str | None = None, nutrients: Sequence[str] = (),
                 two_valued: Sequence[str] = (),
                 strategy: str = "median", keep_empty_features: bool = True,
                 add_indicator: bool = False, missing_values: Any = np.nan, fill_value: Any = None,
                 copy: bool = True):
        super().__init__(two_valued=two_valued, missing_values=missing_values, strategy=strategy,
                         fill_value=fill_value, copy=copy, add_indicator=add_indicator,
                         keep_empty_features=keep_empty_features)
        self.energy = energy
        self.nutrients = nutrients

    def fit(self, X: Any, y: Any = None) -> "EnergyAwareImputer":
        super().fit(X, y)
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        self.lines_: dict[str, tuple[float, float, float, float]] = {}
        if self.energy is None or self.energy not in frame.columns:
            return self
        E = pd.to_numeric(frame[self.energy], errors="coerce").to_numpy(dtype=float)
        for n in self.nutrients:
            if n not in frame.columns:
                continue
            N = pd.to_numeric(frame[n], errors="coerce").to_numpy(dtype=float)
            both = np.isfinite(N) & np.isfinite(E)
            if both.sum() < 3 or np.ptp(E[both]) == 0:
                continue
            slope, intercept = np.polyfit(E[both], N[both], 1)
            self.lines_[n] = (float(intercept), float(slope), float(N[both].min()), float(N[both].max()))
        return self

    def transform(self, X: Any) -> Any:
        out = super().transform(X)
        if not getattr(self, "lines_", None):
            return out
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        names = [str(c) for c in self.get_feature_names_out()]
        values = np.array(out, dtype=float, copy=True)
        energy_at = names.index(self.energy) if self.energy in names else None
        if energy_at is not None:
            E = values[:, energy_at]
        else:
            E = pd.to_numeric(frame[self.energy], errors="coerce").to_numpy(dtype=float)
            median = float(self.statistics_[list(self.feature_names_in_).index(self.energy)])
            E = np.where(np.isfinite(E), E, median)
        for n, (a, b, lo, hi) in self.lines_.items():
            if n not in names:
                continue
            blank = pd.to_numeric(frame[n], errors="coerce").isna().to_numpy()
            if blank.any():
                values[blank, names.index(n)] = np.clip(a + b * E[blank], lo, hi)
        if isinstance(out, pd.DataFrame):
            return pd.DataFrame(values, index=out.index, columns=out.columns)
        return values


class BelowDetectionFill(TransformerMixin, BaseEstimator):
    """Fill each left-censored column's blanks, learned on the fitting rows (in-fold, outcome-free).

    ``half_minimum``: half the smallest detected value. ``censoring_aware``: the expected value below
    the limit (the smallest detected value) under a censored-normal fit to the column, on whichever
    of its own or the log scale the censored normal fits better (``imputation.censoring_of``):
    E[X | X < L] for the log-normal, μ − σφ(a)/Φ(a) for the normal. With one censored covariate in
    a linear model, filling it with its conditional expectation given what was observed leaves the
    slope unbiased (the regression-calibration argument: E[X·E[X|W]] = E[E[X|W]²]). Every other
    column passes through.
    """

    def __init__(self, columns: Sequence[str] = (), method: str = "half_minimum"):
        self.columns = columns
        self.method = method

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BelowDetectionFill":
        from turbotab.core.methods.imputation import tobit_fit

        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.fill_: dict[str, float] = {}
        self.limit_: dict[str, float] = {}
        for c in self.columns:
            if c not in X.columns:
                continue
            x = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            detected = x[np.isfinite(x)]
            if not len(detected):
                continue
            limit = float(detected.min())
            self.limit_[c] = limit
            if self.method == "half_minimum" or len(detected) < 3:
                self.fill_[c] = limit / 2 if limit > 0 else limit
                continue
            from turbotab.core.methods.imputation import log_scale_fits_better

            log = log_scale_fits_better(x, limit)
            z = np.log(x) if log else x
            L = math.log(limit) if log else limit
            blank = ~np.isfinite(x)
            if not blank.any():
                self.fill_[c] = limit / 2 if limit > 0 else limit
                continue
            theta, _ = tobit_fit(np.ones((len(x), 1)), np.where(blank, L, z), blank, L)
            mu, sigma = float(theta[0]), math.exp(float(theta[1]))
            self.fill_[c] = below_limit_mean(mu, sigma, L, log)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, v in getattr(self, "fill_", {}).items():
            if c in out.columns:
                col = pd.to_numeric(out[c], errors="coerce")
                out[c] = col.where(col.notna(), v)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        """Each column's operation for the lineage: filled below detection, or kept."""
        what = ("half the smallest detected value" if self.method == "half_minimum"
                else "expected value below detection")
        filled = set(getattr(self, "fill_", {}))
        return [{"output": c, "inputs": [c],
                 "operation": f"below detection: {what}" if c in filled else "kept"}
                for c in self.feature_names_in_]


def below_limit_mean(mu: float, sigma: float, limit: float, log: bool) -> float:
    """E[X | X < limit] when X (or log X, when ``log``; ``limit`` then on the log scale) is
    N(mu, σ²): μ − σφ(a)/Φ(a), or for the log-normal exp(μ + σ²/2) Φ(a − σ)/Φ(a), a = (limit − μ)/σ."""
    from scipy import stats

    a = (limit - mu) / sigma
    if not log:
        return float(mu - sigma * math.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a)))
    return float(math.exp(mu + sigma ** 2 / 2 + stats.norm.logcdf(a - sigma) - stats.norm.logcdf(a)))


# ── the inference table's missing data ───────────────────────────────────────


def _exit(label: str, decision: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"label": label, "decision": None if decision is None else
            {"kind": "set_missing", **{k: v for k, v in decision.items() if k != "kind"}}}


def missing_block(spec_missing: Mapping[str, Any] | None, purpose: str | None,
                  levels: Sequence[str] = ()) -> tuple[str, list[dict[str, Any]]] | None:
    """(reason, exits) when the inference table is blocked by the missing-values answer, else None.

    Under inference a single fill, a missing indicator or blanks as a level (``levels``: the
    columns that would carry one) are blocked unless the answer carries the recorded attestation.
    The ``set_missing`` validator refuses them at the answer; this holds the table when the purpose
    became inference after the answer."""
    if purpose != "inference" or not spec_missing or spec_missing.get("acknowledged"):
        return None
    strategy = spec_missing.get("strategy")
    indicator = bool(spec_missing.get("indicators")) or (
        spec_missing.get("categorical") == "missing_category" and bool(levels))
    if strategy != "impute" and not indicator:
        return None
    base = {k: v for k, v in spec_missing.items()}
    mi = {**base, "strategy": "multiple_imputation", "indicators": False, "categorical": "impute"}
    cc = {**base, "strategy": "complete_case", "indicators": False, "categorical": "impute"}
    exits = [_exit("Multiple imputation with the outcome and energy (m = 20)", mi),
             _exit("Complete cases, with their assumption stated", cc),
             _exit("Keep it, recorded as a limitation", {**base, "acknowledged": True})]
    if indicator:
        return (f"Under inference the missing-indicator method is blocked until it is recorded: "
                f"{INDICATOR_CAUTION}.", exits)
    return f"Under inference a single fill is blocked until it is recorded: {SINGLE_FILL_CAUTION}.", exits


def imputation_frame(spec: Any, X: pd.DataFrame, y: Any, task: str
                     ) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, Any], list[str]]:
    """What the chained equations run on: (data, columns to impute, kinds, censored, level columns).

    Every input column (predictors, total energy among them whatever the energy method, and a
    strata column) and the outcome's terms (``imputation.outcome_terms``). A column whose blanks
    are a level of their own keeps them as that level (a value, not missing). A left-censored
    column is filled by half its minimum before the chained equations under ``half_minimum``, and
    drawn below its limit within them under ``censoring_aware``.
    """
    from turbotab.core.methods.imputation import Censoring, censoring_of, column_kind, outcome_terms
    from turbotab.core.models.pipeline import MISSING_LEVEL

    inputs = [c for c in spec.inputs if c in X.columns]
    data = X[inputs].copy()
    levels = [c for c in getattr(spec, "levels", []) or [] if c in data.columns]
    for c in levels:
        data[c] = data[c].astype(object).where(data[c].notna(), MISSING_LEVEL)
    censored: dict[str, Censoring] = {}
    cens = getattr(spec, "censored", None) or {}
    for c in cens.get("columns") or []:
        if c not in data.columns:
            continue
        reading = censoring_of(data[c])
        if cens.get("method") == "half_minimum":
            col = pd.to_numeric(data[c], errors="coerce")
            data[c] = col.where(col.notna(), reading.limit / 2 if reading.limit > 0 else reading.limit)
        else:
            censored[c] = reading
    declared = set(getattr(spec, "categorical", []) or [])
    kinds = {c: ("censored" if c in censored else column_kind(data[c], c in declared))
             for c in data.columns}
    terms = outcome_terms(task, y)
    terms.index = data.index
    for c in terms.columns:
        data[c] = terms[c].to_numpy()
        kinds[c] = "numeric"
    impute = [c for c in inputs if c not in levels and data[c].isna().any()]
    return data, impute, kinds, censored, levels


def impute_for_inference(spec: Any, X: pd.DataFrame, y: Any, task: str, *, seed: int = 0,
                         progress: Any = None, cancelled: Any = None) -> Any:
    """The ``m`` completed copies of ``X`` the inference table is pooled over (``Imputations``,
    each frame holding ``X``'s columns, a level column's blanks left as they were)."""
    from turbotab.core.methods.imputation import M_DEFAULT, chained_equations

    data, impute, kinds, censored, levels = imputation_frame(spec, X, y, task)
    m = int((getattr(spec, "missing", None) or {}).get("m") or M_DEFAULT)
    out = chained_equations(data, impute=impute, kinds=kinds, censored=censored, m=m, seed=seed,
                            progress=progress, cancelled=cancelled)
    columns = list(X.columns)
    frames = []
    for f in out.frames:
        done = X.copy()
        for c in columns:
            if c in f.columns and c not in levels:
                done[c] = f[c].to_numpy() if not isinstance(f[c].dtype, pd.CategoricalDtype) else f[c]
        frames.append(done)
    out.frames = frames
    return out


def multiple_imputation_info(imputations: Any, spec: Any, n_rows: int) -> dict[str, Any]:
    """The ``inference.missing`` record of a pooled table."""
    from turbotab.core.methods.imputation import OUTCOME_PREFIX

    fill = getattr(spec, "energy_fill", None) or {}
    roles = getattr(spec, "roles", {}) or {}
    energy = fill.get("energy") or next((c for c in spec.inputs if roles.get(c) == "energy"), None)
    variables = [("the outcome" if v.startswith(OUTCOME_PREFIX) else v) for v in imputations.variables]
    return {
        "method": "multiple_imputation", "m": int(imputations.m), "iterations": int(imputations.iterations),
        "n_rows": int(n_rows), "n_incomplete_rows": int(imputations.n_incomplete_rows),
        "imputed": {str(k): int(v) for k, v in imputations.imputed.items()},
        "variables": list(dict.fromkeys(variables)), "outcome_in_model": True,
        "energy": energy, "censored": sorted(imputations.censored),
        "below_detection": (getattr(spec, "censored", None) or {}).get("method"),
        "assumption": MI_ASSUMPTION, "recorded": False, "n_dropped": None, "note":
            ("Cross-validated scores fill blanks in each training fold without the outcome "
             "(prediction's rule); only the coefficients use the multiple imputations."),
    }


def mi_concerns(imputations: Any, rows: Sequence[Mapping[str, Any]], n_rows: int) -> list[str]:
    """What a pooled table says about its imputation: a large fraction of missing information,
    and fewer imputations than White, Royston & Wood's (2011) rule of thumb."""
    out: list[str] = []
    high = [(str(r["feature"]), float(r["fmi"])) for r in rows
            if r.get("fmi") is not None and float(r["fmi"]) > 0.5 and not str(r["feature"]).startswith("(")]
    if high:
        named = ", ".join(f"`{n}` ({v:.2f})" for n, v in high[:3])
        out.append(f"The fraction of missing information exceeds 0.5 for {named}: those estimates "
                   f"lean heavily on the imputation model (van Buuren 2018 calls 0.5 high).")
    share = imputations.n_incomplete_rows / n_rows if n_rows else 0.0
    if 100 * share > imputations.m:
        out.append(f"{share:.0%} of the rows have an imputed value: White, Royston & Wood (2011) "
                   f"suggest at least as many imputations as that percentage, more than the m = "
                   f"{imputations.m} used; raise m so the pooled p-values reproduce.")
    return out


# ── what complete cases cost ─────────────────────────────────────────────────


def row_loss(kept: pd.DataFrame, dropped: pd.DataFrame, target: str | None,
             predictors: Sequence[str]) -> dict[str, Any] | None:
    """The rows complete cases dropped beside the rows they kept: the outcome (mean, or the share
    at the most common level for a category) and the predictors observed on the dropped rows,
    largest standardized difference first. None when nothing was dropped."""
    n_kept, n_dropped = int(len(kept)), int(len(dropped))
    if not n_dropped:
        return None
    total = n_kept + n_dropped

    def compare(column: str) -> dict[str, Any] | None:
        a, b = kept[column], dropped[column]
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
            a, b = a.astype(float), b.astype(float)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.dropna().astype(float), b.dropna().astype(float)
            if len(a) < 2 or len(b) < 2:
                return None
            pooled = math.sqrt((a.var() + b.var()) / 2) if (a.var() + b.var()) > 0 else 0.0
            smd = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
            return {"column": column, "kept": float(a.mean()), "dropped": float(b.mean()),
                    "smd": float(smd), "kind": "mean", "n_dropped_observed": int(len(b))}
        a, b = a.dropna().astype(str), b.dropna().astype(str)
        if not len(a) or not len(b):
            return None
        level = a.value_counts().index[0]
        pa, pb = float((a == level).mean()), float((b == level).mean())
        p = (pa + pb) / 2
        smd = (pb - pa) / math.sqrt(p * (1 - p)) if 0 < p < 1 else 0.0
        return {"column": column, "kept": pa, "dropped": pb, "smd": float(smd),
                "kind": f"share {level}", "n_dropped_observed": int(len(b))}

    outcome = compare(target) if target and target in kept.columns else None
    columns = [c for c in (compare(c) for c in predictors if c in kept.columns) if c is not None]
    columns.sort(key=lambda e: -abs(e["smd"]))
    return {"n_before": total, "n_kept": n_kept, "n_dropped": n_dropped,
            "share": round(n_dropped / total, 4), "outcome": outcome, "columns": columns[:20]}


def _num(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def row_loss_concern(loss: Mapping[str, Any] | None) -> str | None:
    """Under inference, the concern when complete cases drop more than :data:`ROW_LOSS_SHARE`."""
    if not loss or float(loss.get("share") or 0) <= ROW_LOSS_SHARE:
        return None
    parts = []
    for e in ([loss["outcome"]] if loss.get("outcome") else []) + list(loss.get("columns") or [])[:MAX_COMPARED]:
        what = "mean" if e["kind"] == "mean" else e["kind"].replace("share ", "share `") + "`"
        parts.append(f"`{e['column']}` {what} {_num(e['dropped'])} dropped vs {_num(e['kept'])} kept "
                     f"(SMD {e['smd']:+.2f})")
    compared = ("; ".join(parts) + ". ") if parts else ""
    return (f"Complete cases drop {loss['n_dropped']:,} of {loss['n_before']:,} rows "
            f"({loss['share']:.0%}), more than the 10% at which CLINICAL_SURVEY_PACK flags "
            f"listwise deletion. {compared}The kept rows estimate without bias only if whether a "
            f"value is missing does not depend on the outcome, given the predictors; multiple "
            f"imputation keeps every row.")


__all__ = [
    "BELOW_DETECTION_LABELS", "BelowDetection", "BelowDetectionFill", "COMPLETE_CASE_ASSUMPTION",
    "EnergyAwareImputer", "INDICATOR_CAUTION", "LEFT_CENSORED", "METHODS", "MI_ASSUMPTION",
    "MissingMethod", "ROW_LOSS_SHARE", "RULE", "SINGLE_FILL_CAUTION", "ZEROS", "below_detection_options",
    "below_limit_mean", "censored_columns", "energy_fill", "methods_for", "register_method",
    "row_loss", "row_loss_concern", "impute_for_inference", "imputation_frame", "mi_concerns",
    "missing_block", "multiple_imputation_info",
]
