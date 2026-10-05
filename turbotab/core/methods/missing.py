"""Missing data by purpose (AUDIT_REPORT §5 WP7; closes ME-01, ME-08, and the minor E14, B23, F14).

**One rule** (BLUEPRINT §12 ruling 4), stated once here and quoted by the MISSING drawer, ROADMAP §07
and M2_CONTRACT §4 (:data:`RULE`):

* **Inference** — multiple imputation with the outcome and total energy in the imputation model,
  pooled by Rubin's rules (``turbotab.core.methods.imputation``), ranked first. Since MS1–MS3
  (MODELING_SEQUENCE §0 ruling 5, §1.1) the imputation is compatible with the analysis model
  (``methods.smcfcs``: SMC-FCS for splines, logs, ratios and logistic or Cox outcomes; logged
  quantities on the log scale; the energy sources and the rest imputed and total energy derived;
  knots fixed on the observed values, :class:`FixedForms`), holds the survey design and the
  clustering (ruling 12), takes m = max(20, the percentage of rows with an imputed value) and
  reports its Monte Carlo error, and every estimate shown is pooled, the substitution curve
  included. Passive imputation with a declared nonlinear term and single-level imputation on
  clustered rows are blocked and recorded (:func:`missing_block`). The data's own imputed copies
  (NHANES DXA) are pooled as they come (:func:`supplied_copies`). Moons et al. 2006 (*J Clin
  Epidemiol* 59:1092): "For all types of missing values, imputation of missing predictor values
  using the outcome is preferred over imputation without outcome"; Harrell (*Regression Modeling Strategies*, §3.8): "multiple imputation can and should
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
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from turbotab.core.methods.exposure_form import ExposureForms

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


PASSIVE_CAUTION = ("passive imputation draws each value from a model linear in it and only then "
                   "derives the declared nonlinear terms, so the curvature and the nonlinearity "
                   "tests would be biased toward the null (Bartlett et al. 2015)")
SINGLE_LEVEL_CAUTION = ("imputing clustered rows one at a time gives a unit's time-invariant values "
                        "different draws on its different rows and erases the correlation within "
                        "units (Lüdtke, Robitzsch & Grund 2017)")
BARTLETT = "Bartlett et al. 2015"
WHITE_ROYSTON_WOOD = "White, Royston & Wood 2011"
MC_RULE = 0.10  # White, Royston & Wood (2011): Monte Carlo error under 10% of the standard error


def missing_block(spec_missing: Mapping[str, Any] | None, purpose: str | None,
                  levels: Sequence[str] = (), *, nonlinear: Sequence[str] = (),
                  passive: bool = False, clustered: bool = False
                  ) -> tuple[str, list[dict[str, Any]]] | None:
    """(reason, exits) when the inference table is blocked by the missing-values answer, else None.

    Under inference these are blocked unless the answer carries the recorded attestation
    (``acknowledged``), MODELING_SEQUENCE §4:

    * a single fill, a missing indicator or blanks as a level (``levels``: the columns that would
      carry one), whose ``set_missing`` refusal this holds when the purpose became inference later;
    * passive multiple imputation with a declared nonlinear term (``nonlinear``, the terms; the
      answer's ``imputation_model`` is ``"passive"``, or ``passive``: no compatible imputation
      exists for this outcome): "the form tests are biased toward the null";
    * single-level multiple imputation on clustered rows (``clustered``; the answer's
      ``imputation_levels`` is ``"single_level"``).
    """
    if purpose != "inference" or not spec_missing or spec_missing.get("acknowledged"):
        return None
    strategy = spec_missing.get("strategy")
    base = {k: v for k, v in spec_missing.items()}
    cc = {**base, "strategy": "complete_case", "indicators": False, "categorical": "impute"}
    keep = _exit("Keep it, recorded as a limitation", {**base, "acknowledged": True})
    if strategy == "multiple_imputation":
        chose_passive = spec_missing.get("imputation_model") == "passive"
        if nonlinear and (chose_passive or passive):
            terms = _terms_words(nonlinear)
            exits = ([_exit("Multiple imputation compatible with the analysis model (SMC-FCS)",
                            {**base, "imputation_model": "compatible"})] if chose_passive and not passive
                     else [])
            exits += [_exit("Complete cases, with their assumption stated", cc), keep]
            why = ("" if chose_passive else " No imputation compatible with this outcome's model "
                                            "is built here, so the copies would be passive.")
            return (f"Under inference passive multiple imputation with {terms} is blocked until it "
                    f"is recorded: {PASSIVE_CAUTION}.{why}", exits)
        if clustered and spec_missing.get("imputation_levels") == "single_level":
            exits = [_exit("Clustered multiple imputation (time-invariant values once per unit)",
                           {**base, "imputation_levels": "clustered"}),
                     _exit("Complete cases, with their assumption stated", cc), keep]
            return (f"Under inference single-level multiple imputation on clustered rows is blocked "
                    f"until it is recorded: {SINGLE_LEVEL_CAUTION}.", exits)
        return None
    indicator = bool(spec_missing.get("indicators")) or (
        spec_missing.get("categorical") == "missing_category" and bool(levels))
    if strategy != "impute" and not indicator:
        return None
    mi = {**base, "strategy": "multiple_imputation", "indicators": False, "categorical": "impute"}
    exits = [_exit("Multiple imputation with the outcome and energy (m = 20)", mi),
             _exit("Complete cases, with their assumption stated", cc), keep]
    if indicator:
        return (f"Under inference the missing-indicator method is blocked until it is recorded: "
                f"{INDICATOR_CAUTION}.", exits)
    return f"Under inference a single fill is blocked until it is recorded: {SINGLE_FILL_CAUTION}.", exits


def _terms_words(terms: Sequence[str]) -> str:
    from turbotab.core.voice import listing

    return listing(list(terms), limit=4, ticked=False)


def nonlinear_terms(spec: Any, task: str | None = None) -> list[str]:
    """The declared terms of the analysis model that are not linear in the variables they are
    imputed from (MODELING_SEQUENCE §2: "a declared nonlinear or derived term under MI implies an
    imputation model compatible with the analysis model"): each spline or quintile form, the log
    residual, and densities."""
    from turbotab.core.voice import tick

    out: list[str] = []
    for column, form in (getattr(spec, "exposure_forms", None) or {}).items():
        kind = form.get("form") if isinstance(form, Mapping) else getattr(form, "form", None)
        if kind == "spline":
            out.append(f"a restricted cubic spline of {tick(column)}")
        elif kind == "quintiles":
            out.append(f"quintiles of {tick(column)}")
    adj = spec.energy_adjustment() if hasattr(spec, "energy_adjustment") else None
    if adj is not None and adj.method in ("residual", "residual_energy_dropped") and adj.log_transform:
        out.append(f"the log residual of {_named(adj.nutrients)} on {tick(adj.energy_column)}")
    if adj is not None and adj.method in ("density", "density_multivariate"):
        out.append(f"{_named(adj.nutrients)} per {tick(adj.energy_column)}")
    return out


def _named(columns: Sequence[str]) -> str:
    from turbotab.core.voice import listing

    return listing(list(columns))


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


SUBSTANTIVE = {"regression": "linear", "binary": "logistic", "time_to_event": "cox"}


def logged_columns(spec: Any) -> list[str]:
    """The inputs the analysis takes the log of (MODELING_SEQUENCE §1.1: "logged quantities are
    imputed on the log scale"): the nutrients and total energy of a log residual, total energy and
    the nutrients of a density (a ratio is a log difference), and the columns a quotient
    normalization logs (a zero there becomes a blank)."""
    out: list[str] = []
    adj = spec.energy_adjustment() if hasattr(spec, "energy_adjustment") else None
    if adj is not None and ((adj.method in ("residual", "residual_energy_dropped") and adj.log_transform)
                            or adj.method in ("density", "density_multivariate")):
        out += [*adj.nutrients, *([adj.energy_column] if adj.energy_column else [])]
    norm = getattr(spec, "normalization", None) or {}
    if norm.get("method") not in (None, "log_cpm_tmm", "log_cpm"):
        out += list(norm.get("columns") or [])
    inputs = set(spec.inputs)
    return [c for c in dict.fromkeys(out) if c in inputs]


def energy_identity(spec: Any, X: pd.DataFrame, factors: Mapping[str, float] | None,
                    nested: Mapping[str, str] | None = None) -> tuple[str, dict[str, float]] | None:
    """(total energy, {source: kcal per unit}) when the imputation imputes the energy sources and
    "other" and derives total energy as their sum (MODELING_SEQUENCE §1.1), else None.

    ``factors``: each nutrient's kcal per unit **as the readings ledger holds it settled**
    (``readings.kcal_per_unit``: a recorded unit, or grams the registry's Atwater test reads;
    BLUEPRINT §14.1, never a name). The sources are the energy adjustment's nutrients with a settled
    factor, less any that is a part of another (``nested``); a nutrient without one is imputed by its
    own model and its energy stays in "other". It applies only when total energy or a source has a
    blank."""
    adj = spec.energy_adjustment() if hasattr(spec, "energy_adjustment") else None
    if adj is None or not adj.energy_column or adj.energy_column not in X.columns or not factors:
        return None
    E = adj.energy_column
    parts = set((nested or {}).keys())
    sources = {n: float(factors[n]) for n in adj.nutrients
               if n in factors and n in X.columns and n not in parts and n != E
               and pd.api.types.is_numeric_dtype(X[n]) and float(factors[n]) > 0}
    if not sources:
        return None
    if not (X[E].isna().any() or X[list(sources)].isna().any().any()):
        return None
    return E, sources


# ── knots and cut points, fixed across copies (MS1) ──────────────────────────


def fixed_forms(spec: Any, X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Each formed column's knots or quintile cut points, placed once on the observed values
    (MODELING_SEQUENCE §1 step 5: "knots at Harrell's percentiles of observed values, fixed across
    imputations"). The column the form receives (an energy-adjusted one, say) is computed from the
    raw inputs by the steps before the form, fit on every row with the values they need recorded;
    a row with a blank among them adds nothing."""
    from turbotab.core.methods.exposure_form import (DEFAULT_KNOTS, _form_of, adjusted_forms,
                                                     quantile_cuts, rcs_knots)
    from turbotab.core.models.pipeline import shared_steps, transformer

    forms = getattr(spec, "exposure_forms", None) or {}
    if not forms:
        return {}
    adjusted = adjusted_forms(forms, spec.energy)
    pre = [(n, s) for n, s in shared_steps(spec) if n in ("normalize", "energy")]
    inputs = [c for c in spec.inputs if c in X.columns]
    frame = X[inputs]
    if pre:
        frame = transformer(pre).fit(frame).transform(frame)
    out: dict[str, dict[str, Any]] = {}
    for column, form_spec in adjusted.items():
        form, k = _form_of(form_spec)
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if form == "spline":
            knots, notes = rcs_knots(values, k or DEFAULT_KNOTS)
            out[column] = {"form": "spline", "knots": [float(v) for v in knots], "notes": list(notes),
                           "n_observed": int(len(values))}
        elif form == "quintiles":
            out[column] = {"form": "quintiles", "cuts": [float(v) for v in quantile_cuts(values)],
                           "n_observed": int(len(values))}
    return out


class FixedForms(ExposureForms):
    """The exposure-form step with its knots and cut points fixed (``fixed``: column → ``{form,
    knots, notes}`` or ``{form, cuts}``, from :func:`fixed_forms`), so every completed copy holds the
    identical basis. A quintile's median (its trend score) is still each copy's own."""

    def __init__(self, forms: Mapping[str, Any] | None = None,
                 fixed: Mapping[str, Mapping[str, Any]] | None = None):
        super().__init__(forms)
        self.fixed = fixed

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FixedForms":
        from turbotab.core.methods.exposure_form import QUINTILES, quantile_group

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FixedForms needs a pandas DataFrame with named columns.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.knots_, self.knot_notes_, self.cuts_ = {}, {}, {}
        self.medians_, self.counts_ = {}, {}
        fixed = dict(self.fixed or {})
        for column, (form, _) in self._plan().items():
            if column not in X.columns:
                raise ValueError(f"`{column}` is not among the model's inputs at this step, so its "
                                 f"form cannot be applied.")
            if column not in fixed:
                raise ValueError(f"`{column}` has no knots or cut points placed on its observed values.")
            values = X[column].to_numpy(dtype=float, na_value=np.nan)
            values = values[np.isfinite(values)]
            if form == "spline":
                self.knots_[column] = np.asarray(fixed[column]["knots"], dtype=float)
                self.knot_notes_[column] = list(fixed[column].get("notes") or [])
                continue
            cuts = np.asarray(fixed[column]["cuts"], dtype=float)
            group = quantile_group(values, cuts)
            counts = np.bincount(group, minlength=QUINTILES)[:QUINTILES]
            self.cuts_[column] = cuts
            self.medians_[column] = np.array([float(np.median(values[group == g])) if counts[g]
                                              else float("nan") for g in range(QUINTILES)])
            self.counts_[column] = [int(c) for c in counts]
        return self


class CopyGuard(TransformerMixin, BaseEstimator):
    """The impute step of a completed copy's pipeline: there is nothing left to fill, so a blank
    that reaches it was made inside the copy (a zero a quotient normalization's log turned
    missing) and is refused, never filled by the median (MODELING_SEQUENCE §1.1: "a zero that a log
    would turn missing is sent to the detection-limit question, never to median fill").
    ``levels``: columns whose blanks are a level of their own (they pass)."""

    def __init__(self, levels: Sequence[str] = ()):
        self.levels = levels

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CopyGuard":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self._check(X)
        return self

    def _check(self, X: pd.DataFrame) -> None:
        from turbotab.core.methods.imputation import ImputationRefused

        keep = [c for c in X.columns if c not in set(self.levels or ())]
        blank = X[keep].isna()
        if blank.any().any():
            cols = [str(c) for c in blank.columns[blank.any()]]
            raise ImputationRefused(
                f"A completed copy holds a blank in {', '.join(f'`{c}`' for c in cols[:3])} after "
                f"its earlier steps (a zero a log turned missing): it goes to the detection-limit "
                f"question, never to a median fill inside the pooled table.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


def copy_template(template: Any, plan: Mapping[str, Any]) -> Any:
    """``template`` (an unfitted pipeline) as each completed copy is fit: its impute step a
    :class:`CopyGuard`, its form step's knots and cut points the observed ones (:class:`FixedForms`)."""
    from sklearn.base import clone

    pipe = clone(template)
    names = [n for n, _ in pipe.steps]
    if "impute" in names:
        pipe.set_params(impute=CopyGuard(list(plan.get("levels") or [])))
    fixed = plan.get("forms") or {}
    if "form" in names and fixed:
        step = pipe.named_steps["form"]
        pipe.set_params(form=FixedForms(dict(step.forms or {}), fixed))
    return pipe


# ── the plan: what the imputation model holds (MS1, MS2) ─────────────────────


def unit_level_columns(X: pd.DataFrame, units: np.ndarray, columns: Sequence[str]) -> list[str]:
    """The ``columns`` that are constant within every unit wherever they are recorded, with at least
    one unit recording them on two or more rows (ruling 12's time-invariant variables, read from
    the values; the record lists them, and a column that varies within any unit is row-level)."""
    out = []
    codes = pd.Series(np.asarray(units))
    for c in columns:
        col = X[c].reset_index(drop=True)
        rec = col.notna()
        if not rec.any():
            continue
        g = col[rec].groupby(codes[rec.to_numpy()].to_numpy())
        n = g.size()
        if not (n >= 2).any():
            continue
        if int(g.nunique().max()) <= 1:
            out.append(c)
    return out


def design_variables(survey_design: Any, index: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """The survey design's strata, PSU within its stratum and weight for each analysis row (by row
    id), as imputation-model predictors (Reiter, Raghunathan & Kinney 2006: "the safest course of
    action is to include design variables in the specification of imputation models"); rows the
    design cannot place get a level of their own and the median weight."""
    if survey_design is None:
        return None
    ids = np.asarray(index, dtype=np.int64)
    at = pd.Series(np.arange(len(survey_design.row_ids)), index=survey_design.row_ids)
    pos = at.reindex(ids).to_numpy()
    placed = np.isfinite(pos)
    p = np.where(placed, pos, 0).astype(np.int64)
    stratum = np.where(placed, survey_design.stratum[p], -1)
    psu = survey_design.psu[p]
    rank = pd.Series(psu).groupby(stratum).rank(method="dense").to_numpy()
    weight = np.where(placed, survey_design.weight[p], np.nan)
    weight = np.where(np.isfinite(weight), weight, np.nanmedian(weight) if np.isfinite(weight).any() else 1.0)
    frame = pd.DataFrame({
        "__stratum": [f"stratum {s}" if ok else "unplaced" for s, ok in zip(stratum, placed)],
        "__psu": [f"psu {int(r)}" if ok else "unplaced" for r, ok in zip(rank, placed)],
        "__weight": weight}, index=index)
    names = {"strata": survey_design.strata_column, "psu": survey_design.psu_column,
             "weight": survey_design.weight_column, "n_unplaced": int((~placed).sum())}
    if frame["__stratum"].nunique() < 2:
        frame = frame.drop(columns=["__stratum"])
    if frame["__psu"].nunique() < 2:
        frame = frame.drop(columns=["__psu"])
    return frame, names


def binary01(y: Any) -> np.ndarray:
    """A binary outcome as 0/1: as coded (numbers or booleans), else the level that sorts last is 1
    (the class order a fitted classifier holds)."""
    values = np.asarray(y)
    try:
        out = values.astype(float)
        if set(np.unique(out[np.isfinite(out)])) <= {0.0, 1.0}:
            return out
    except (TypeError, ValueError):
        pass
    levels = sorted(pd.unique(pd.Series(values).dropna()), key=str)
    return (pd.Series(values).astype(object) == levels[-1]).to_numpy(dtype=float)


def engine_substantive(spec: Any, task: str, y: Any, forms: Mapping[str, Any]) -> Any:
    """The analysis model as SMC-FCS reads it: the design is the pipeline's own shared steps
    (normalization, energy adjustment, the form with fixed knots, levels, one-hot), refit on the
    current completed rows once per iteration (the energy residual's line is re-estimated, as in
    each copy) and row-local once fit; the outcome model is the task's (linear, logistic, Cox)."""
    from sklearn.base import clone

    from turbotab.core.methods.smcfcs import Outcome, Substantive
    from turbotab.core.models.pipeline import shared_steps, transformer

    steps = []
    for name, step in shared_steps(spec):
        if name in ("impute", "detect"):
            continue
        if name == "form":
            step = FixedForms(dict(step.forms or {}), dict(forms))
        steps.append((name, step))
    template = transformer(steps)
    inputs = list(spec.inputs)
    holder: dict[str, Any] = {}

    def refresh(raw: pd.DataFrame) -> None:
        holder["T"] = clone(template).fit(raw[inputs])

    def design(raw: pd.DataFrame) -> np.ndarray:
        if "T" not in holder:
            refresh(raw)
        return np.asarray(holder["T"].transform(raw[inputs]), dtype=float)

    kind = SUBSTANTIVE[task]
    if kind == "cox":
        values = np.asarray(y)
        outcome = Outcome("cox", time=np.asarray(values["time"], dtype=float),
                          event=np.asarray(values["event"], dtype=float),
                          entry=np.asarray(values["entry"], dtype=float))
    elif kind == "logistic":
        outcome = Outcome("logistic", y=binary01(y))
    else:
        outcome = Outcome("linear", y=np.asarray(y, dtype=float))
    return Substantive(outcome=outcome, design=design, refresh=refresh)


@dataclass
class ImputationPlan:
    """What the imputation model holds and how it is drawn: the sampler's inputs and the record."""

    mode: str  # "fcs" | "smcfcs"
    compatible: bool
    substantive_kind: str | None
    nonlinear: list[str]
    data: pd.DataFrame
    variables: list[Any]
    identity: Any = None
    units: np.ndarray | None = None
    unit_column: str | None = None
    unit_level: list[str] = field(default_factory=list)
    design: dict[str, Any] | None = None
    logged: list[str] = field(default_factory=list)
    forms: dict[str, dict[str, Any]] = field(default_factory=dict)
    substantive: Any = None
    levels: list[str] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)  # each column's kind (declared codes kept)
    m: int = 20
    m_asked: int = 20
    n_incomplete: int = 0
    exclude: list[str] = field(default_factory=list)


OTHER = "__other_energy"


def imputation_plan(spec: Any, X: pd.DataFrame, y: Any, task: str, *, survey: Any = None,
                    clusters: Any = None, nested: Mapping[str, str] | None = None,
                    factors: Mapping[str, float] | None = None) -> ImputationPlan:
    """The imputation model for the inference table (MODELING_SEQUENCE §1.1, step 1):

    * **compatible with the analysis model**: SMC-FCS when the outcome is binary or time-to-event,
      or the analysis model holds a nonlinear term (:func:`nonlinear_terms`); chained equations
      with the outcome in the model for a linear model linear in its imputed variables, and for
      ordinal and multinomial outcomes (no SMC-FCS for them here: with a nonlinear term those
      copies are passive, which :func:`missing_block` holds). The answer's ``imputation_model``
      "passive" asks for the customary chained equations whatever the terms;
    * logged quantities on the log scale; the energy identity; knots fixed on observed values;
    * the survey design's strata, PSU and weight (``survey``: a ``SurveyDesign``);
    * with clustered rows (``clusters``) and unless the answer is ``"single_level"``, time-invariant
      variables once per unit and the unit means of the rest;
    * m by the rule (:func:`~turbotab.core.methods.imputation.m_rule`).
    """
    from turbotab.core.methods.imputation import M_DEFAULT, OUTCOME_PREFIX, m_rule
    from turbotab.core.methods.smcfcs import Identity, Variable

    answer = dict(getattr(spec, "missing", None) or {})
    data, impute, kinds, censored, levels = imputation_frame(spec, X, y, task)
    nonlinear = nonlinear_terms(spec, task)
    passive = answer.get("imputation_model") == "passive"
    substantive_kind = SUBSTANTIVE.get(task)
    smc = (not passive) and substantive_kind is not None and (
        substantive_kind in ("logistic", "cox") or bool(nonlinear))
    compatible = (not passive) and (smc or not nonlinear)
    logged = [c for c in logged_columns(spec) if c in data.columns]
    terms = [c for c in data.columns if c.startswith(OUTCOME_PREFIX)]
    if smc:
        data = data.drop(columns=terms)
    identity = energy_identity(spec, X, factors, nested)
    variables: list[Any] = []
    unit_level: list[str] = []
    units = None
    unit_column = None
    single = answer.get("imputation_levels") == "single_level"
    if clusters is not None and getattr(clusters, "clustered", False) and not single:
        units = np.asarray(clusters.codes, dtype=np.int64)
        unit_column = clusters.column
        unit_level = unit_level_columns(X, units, impute)
    for c in impute:
        if identity is not None and c == identity[0]:
            continue  # total energy is derived from its sources and other
        kind = kinds[c]
        log = c in logged and kind == "numeric"
        lower = 0.0 if (identity is not None and c in identity[1] and not log) else None
        variables.append(Variable(c, kind, log=log, lower=lower, unit_level=c in unit_level,
                                  censoring=censored.get(c)))
    ident = None
    if identity is not None:
        E, factors = identity
        parts = sum(float(f) * pd.to_numeric(data[s], errors="coerce") for s, f in factors.items())
        other = pd.to_numeric(data[E], errors="coerce") - parts
        data[OTHER] = other.to_numpy()
        recorded = other.dropna()
        log_other = (E in logged or any(s in logged for s in factors)) and bool(len(recorded)) \
            and bool((recorded > 0).all())
        variables.append(Variable(OTHER, "numeric", log=log_other, lower=None if log_other else 0.0))
        ident = Identity(energy=E, other=OTHER, factors=factors)
    design = None
    if survey is not None:
        found = design_variables(survey, X.index)
        if found is not None:
            frame, names = found
            for c in frame.columns:
                data[c] = frame[c].to_numpy()
            design = {**names, "columns": list(frame.columns)}
    forms = fixed_forms(spec, X)
    substantive = engine_substantive(spec, task, y, forms) if smc else None
    blank = np.zeros(len(X), dtype=bool)
    for c in impute:
        blank |= X[c].isna().to_numpy()
    n_incomplete = int(blank.sum())
    asked = int(answer.get("m") or M_DEFAULT)
    return ImputationPlan(
        mode="smcfcs" if smc else "fcs", compatible=compatible, substantive_kind=substantive_kind,
        nonlinear=nonlinear, data=data, variables=variables, identity=ident, units=units,
        unit_column=unit_column, unit_level=unit_level, design=design,
        # drawn on the log scale: total energy under the identity is derived from its parts
        logged=[c for c in logged if c in impute and not (identity is not None and c == identity[0])],
        forms=forms, substantive=substantive, levels=levels,
        kinds={c: k for c, k in kinds.items() if c in data.columns},
        m=m_rule(n_incomplete, len(X), asked), m_asked=asked, n_incomplete=n_incomplete)


def impute_for_inference(spec: Any, X: pd.DataFrame, y: Any, task: str, *, seed: int = 0,
                         progress: Any = None, cancelled: Any = None, survey: Any = None,
                         clusters: Any = None, nested: Mapping[str, str] | None = None,
                         factors: Mapping[str, float] | None = None) -> Any:
    """The completed copies of ``X`` the inference table is pooled over (``Imputations``, each frame
    holding ``X``'s columns, a level column's blanks left as they were), drawn as
    :func:`imputation_plan` says; ``Imputations.plan`` is the record's source."""
    from turbotab.core.methods.smcfcs import impute

    plan = imputation_plan(spec, X, y, task, survey=survey, clusters=clusters, nested=nested,
                           factors=factors)
    out = impute(plan.data, plan.variables, mode=plan.mode, substantive=plan.substantive,
                 identity=plan.identity, units=plan.units, m=plan.m, seed=seed,
                 progress=progress, cancelled=cancelled, kinds=plan.kinds)
    columns = list(X.columns)
    frames = []
    for f in out.frames:
        done = X.copy()
        for c in columns:
            if c in f.columns and c not in plan.levels:
                done[c] = f[c].to_numpy() if not isinstance(f[c].dtype, pd.CategoricalDtype) else f[c]
        frames.append(done)
    out.frames = frames
    out.plan = {
        "mode": plan.mode, "compatible": plan.compatible, "substantive": plan.substantive_kind,
        "nonlinear": list(plan.nonlinear), "logged": list(plan.logged),
        "identity": ({"energy": plan.identity.energy, "sources": list(plan.identity.factors),
                      "factors": dict(plan.identity.factors), "infeasible": out.identity_infeasible}
                     if plan.identity is not None else None),
        "design": plan.design, "unit": plan.unit_column if plan.units is not None else None,
        "unit_level": list(plan.unit_level), "forms": plan.forms, "levels": list(plan.levels),
        "m_asked": plan.m_asked, "n_incomplete": plan.n_incomplete, "n_rows": int(len(X)),
    }
    return out


def supplied_copies(frame: pd.DataFrame, implicate: str, unit: str | None, inputs: Sequence[str],
                    y: Any) -> Any:
    """Imputed copies the data carries (NHANES DXA: each participant ``m`` times, numbered by
    ``implicate``), as ``Imputations`` whose every copy holds its own predictors and its own outcome
    (``outcomes``), so each is analyzed as a completed dataset and Rubin's rules pool them (CDC,
    "Multiple Imputation Details": "analyze EACH OF THE FIVE datasets separately … and then
    combining the estimates and standard errors using the combining rules")."""
    from turbotab.core.methods.imputation import Imputations

    labels = frame[implicate]
    levels = sorted(pd.unique(labels.dropna()), key=lambda v: (str(type(v)), v))
    y = np.asarray(y)
    frames, outcomes = [], []
    for level in levels:
        rows = (labels == level).to_numpy()
        part = frame.loc[rows, list(inputs)]
        if unit is not None and unit in frame.columns:
            order = np.argsort(frame.loc[rows, unit].to_numpy(), kind="stable")
            part = part.iloc[order]
            outcomes.append(y[rows][order])
        else:
            outcomes.append(y[rows])
        frames.append(part)
    varies = [c for c in inputs if len(frames) > 1 and any(
        not np.array_equal(pd.isna(f[c].to_numpy()), pd.isna(frames[0][c].to_numpy())) or
        not frames[0][c].reset_index(drop=True).equals(f[c].reset_index(drop=True)) for f in frames[1:])]
    out = Imputations(frames=frames, m=len(frames), iterations=0, imputed={c: 0 for c in varies},
                      n_incomplete_rows=0, variables=list(inputs), kinds={}, method="supplied",
                      outcomes=outcomes)
    from turbotab.core.voice import number

    out.plan = {"mode": "supplied", "compatible": True, "implicate": implicate, "unit": unit,
                "copies": [number(v) for v in levels], "varies": varies, "forms": {}, "levels": [],
                "n_rows": int(len(frames[0])) if frames else 0}
    return out


# ── the record and its sentence ──────────────────────────────────────────────


def multiple_imputation_info(imputations: Any, spec: Any, n_rows: int,
                             rows: Sequence[Mapping[str, Any]] | None = None,
                             df_com: float | None = None, tests: int = 0) -> dict[str, Any]:
    """The ``inference.missing`` record of a pooled table: how the copies were drawn and what the
    imputation model held (``Imputations.plan``), m and its rule, the Monte Carlo error of the pooled
    ``rows`` (the largest as a share of its standard error), and the methods sentence."""
    from turbotab.core.methods.imputation import OUTCOME_PREFIX

    plan = dict(getattr(imputations, "plan", None) or {})
    fill = getattr(spec, "energy_fill", None) or {}
    roles = getattr(spec, "roles", {}) or {}
    energy = fill.get("energy") or next((c for c in spec.inputs if roles.get(c) == "energy"), None)
    smc = getattr(imputations, "method", "") == "smcfcs"
    variables = [("the outcome" if v.startswith(OUTCOME_PREFIX) else v) for v in imputations.variables
                 if not v.startswith("__") or v.startswith(OUTCOME_PREFIX)]
    if smc:
        variables.append("the outcome, through the analysis model")
    design_names = plan.get("design") or {}
    for key, column in (("strata", "__stratum"), ("psu", "__psu"), ("weight", "__weight")):
        if design_names.get(key) and column in (design_names.get("columns") or []):
            variables.append(str(design_names[key]))
    worst, worst_feature = None, None
    for r in rows or []:
        mc, se = r.get("mc_se"), r.get("se")
        name = str(r.get("feature"))
        if mc is None or not se or name.startswith("("):
            continue
        if worst is None or mc / se > worst:
            worst, worst_feature = float(mc / se), name
    design = plan.get("design") or None
    n_incomplete = int(imputations.n_incomplete_rows)
    record = {
        "method": "multiple_imputation", "m": int(imputations.m),
        "iterations": int(imputations.iterations), "n_rows": int(n_rows),
        "n_incomplete_rows": n_incomplete,
        "imputed": {str(k): int(v) for k, v in imputations.imputed.items()},
        "variables": list(dict.fromkeys(variables)), "outcome_in_model": True,
        "energy": energy, "censored": sorted(getattr(imputations, "censored", {}) or {}),
        "below_detection": (getattr(spec, "censored", None) or {}).get("method"),
        "assumption": MI_ASSUMPTION, "recorded": False, "n_dropped": None,
        "note": ("Cross-validated scores fill blanks in each training fold without the outcome "
                 "(prediction's rule); only the coefficients use the multiple imputations."),
        "model": getattr(imputations, "method", "chained_equations"),
        "compatible": bool(plan.get("compatible", True)),
        "substantive": plan.get("substantive"),
        "terms": list(plan.get("nonlinear") or []),
        "logged": list(plan.get("logged") or []),
        "identity_energy": (plan.get("identity") or {}).get("energy"),
        "identity_sources": list((plan.get("identity") or {}).get("sources") or []),
        "identity_infeasible": int((plan.get("identity") or {}).get("infeasible") or 0),
        "design_strata": (design or {}).get("strata") if design and "__stratum" in design.get("columns", []) else None,
        "design_psu": (design or {}).get("psu") if design and "__psu" in design.get("columns", []) else None,
        "design_weight": (design or {}).get("weight") if design else None,
        "df_com": float(df_com) if df_com is not None else None,
        "unit": plan.get("unit"), "unit_level": list(plan.get("unit_level") or []),
        "knots": {c: list(v["knots"]) for c, v in (plan.get("forms") or {}).items() if "knots" in v},
        "cuts": {c: list(v["cuts"]) for c, v in (plan.get("forms") or {}).items() if "cuts" in v},
        "m_asked": int(plan.get("m_asked") or imputations.m),
        "percent_incomplete": round(100.0 * n_incomplete / n_rows, 1) if n_rows else 0.0,
        "rejection_failures": int(getattr(imputations, "rejection_failures", 0) or 0),
        "mc_max_ratio": worst, "mc_feature": worst_feature,
        "copies": plan.get("copies"), "implicate": plan.get("implicate"),
        "tests_d1": int(tests),
    }
    if record["model"] == "supplied":
        record.update(outcome_in_model=False, n_incomplete_rows=None, percent_incomplete=None,
                      note=None, assumption=("the copies' imputation model, as the data's provider "
                                             "drew them"))
    record["sentence"] = mi_sentence(record)
    return record


def mi_sentence(record: Mapping[str, Any]) -> str:
    """The methods sentence of a pooled table's multiple imputation (STROBE-nut nut-13; Sterne et
    al. 2009's checklist: the imputation model, m, and how the results were combined)."""
    from turbotab.core.voice import finish, listing, plural, tick

    def num(v: float) -> str:
        return f"{v:.4g}"

    model = record.get("model")
    m = int(record.get("m") or 0)
    if model == "supplied":
        copies = list(record.get("copies") or [])
        numbered = f"numbered {copies[0]} to {copies[-1]} by " if len(copies) > 1 else "numbered by "
        text = (f"Each of the data's {m} imputed copies ({numbered}{tick(record.get('implicate'))}) "
                f"was analyzed as its own completed dataset with its own outcome, and the estimates "
                f"were pooled by Rubin's rules (NCHS's combining rules)")
    else:
        if model == "smcfcs":
            kind = {"linear": "linear", "logistic": "logistic", "cox": "Cox"}.get(
                str(record.get("substantive")), str(record.get("substantive")))
            terms = list(record.get("terms") or [])
            with_terms = f" with {listing(terms, ticked=False)}" if terms else ""
            how = (f"by substantive-model-compatible fully conditional specification (SMC-FCS; "
                   f"{BARTLETT}), compatible with the analysis model, a {kind} model{with_terms}")
        elif not record.get("compatible", True):
            how = (f"by chained equations with the outcome in the imputation model, with "
                   f"{listing(list(record.get('terms') or []), ticked=False)} derived in each "
                   f"copy (passive imputation), kept as a recorded limitation: {PASSIVE_CAUTION}")
        else:
            how = ("by chained equations with the outcome in the imputation model, compatible "
                   "with the analysis model, linear in the imputed variables")
        pct = record.get("percent_incomplete") or 0.0
        text = (f"Missing values were multiply imputed {how}; m = {m} imputations, at least 20 and "
                f"at least the {pct:g}% of rows with an imputed value ({WHITE_ROYSTON_WOOD})")
        logged = list(record.get("logged") or [])
        if logged:
            text += (f"; {listing(logged)} {plural(len(logged), 'was', 'were')} imputed on the log "
                     f"scale, as the analysis takes {plural(len(logged), 'its', 'their')} log")
        if record.get("identity_energy"):
            text += (f"; the energy sources ({listing(record.get('identity_sources') or [])}) and the "
                     f"rest of energy were imputed and total energy "
                     f"({tick(record['identity_energy'])}) derived as their sum, so the rest is "
                     f"never negative")
        for column, knots in (record.get("knots") or {}).items():
            text += (f"; the knots of {tick(column)} ({', '.join(num(k) for k in knots)}) were "
                     f"placed once on its observed values and held in every copy")
        for column, cuts in (record.get("cuts") or {}).items():
            text += (f"; the quintile cut points of {tick(column)} ({', '.join(num(c) for c in cuts)})"
                     f" were placed once on its observed values and held in every copy")
        design = [(w, record.get(k)) for w, k in (("strata", "design_strata"), ("PSU", "design_psu"),
                                                   ("weight", "design_weight")) if record.get(k)]
        if design:
            text += (f"; the survey {listing([f'{w} ({tick(c)})' for w, c in design], ticked=False)}"
                     f" were in the imputation model")
        if record.get("unit"):
            unit = tick(record["unit"])
            once = list(record.get("unit_level") or [])
            text += (f"; with rows clustered by {unit}, "
                     + (f"{listing(once)} {plural(len(once), 'was', 'were')} imputed once per {unit} "
                        f"and " if once else "")
                     + f"each row-level variable with the {unit} means of the others and its "
                       f"own mean over the {unit}'s other rows")
    pooled = "" if model == "supplied" else "; the estimates were pooled by Rubin's rules"
    if record.get("tests_d1"):
        pooled += ", multi-parameter tests by D1 (Li, Raghunathan & Rubin 1991)"
    if record.get("df_com"):
        pooled += f", on the design's {num(float(record['df_com']))} degrees of freedom"
    text += pooled
    ratio = record.get("mc_max_ratio")
    if ratio is not None:
        text += (f"; the largest Monte Carlo error, for {tick(record.get('mc_feature'))}, was "
                 f"{100 * ratio:.1f}% of its standard error")
    return finish(text)


def mi_concerns(imputations: Any, rows: Sequence[Mapping[str, Any]], n_rows: int) -> list[str]:
    """What a pooled table says about its imputation: a large fraction of missing information, fewer
    imputations than White, Royston & Wood's (2011) rule, a Monte Carlo error above 10% of a
    standard error, and SMC-FCS draws no candidate was kept for."""
    out: list[str] = []
    high = [(str(r["feature"]), float(r["fmi"])) for r in rows
            if r.get("fmi") is not None and float(r["fmi"]) > 0.5 and not str(r["feature"]).startswith("(")]
    if high:
        named = ", ".join(f"`{n}` ({v:.2f})" for n, v in high[:3])
        out.append(f"The fraction of missing information exceeds 0.5 for {named}: those estimates "
                   f"lean heavily on the imputation model (van Buuren 2018 calls 0.5 high).")
    share = imputations.n_incomplete_rows / n_rows if n_rows else 0.0
    if getattr(imputations, "method", "") != "supplied" and 100 * share > imputations.m + 1e-9:
        out.append(f"{share:.0%} of the rows have an imputed value: White, Royston & Wood (2011) "
                   f"suggest at least as many imputations as that percentage, more than the m = "
                   f"{imputations.m} used; raise m so the pooled p-values reproduce.")
    noisy = [(str(r["feature"]), float(r["mc_se"]) / float(r["se"])) for r in rows
             if r.get("mc_se") is not None and r.get("se") and not str(r["feature"]).startswith("(")
             and float(r["mc_se"]) / float(r["se"]) > MC_RULE]
    if noisy:
        named = ", ".join(f"`{n}` ({v:.0%})" for n, v in noisy[:3])
        out.append(f"The Monte Carlo error exceeds 10% of the standard error for {named} "
                   f"({WHITE_ROYSTON_WOOD}): another set of m imputations would move it; raise m.")
    failed = int(getattr(imputations, "rejection_failures", 0) or 0)
    if failed:
        out.append(f"SMC-FCS kept no candidate for {failed:,} draws within its limit of 1,000 "
                   f"candidates each; those values kept their last candidate, as R's smcfcs does.")
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


# ── the method contracts (BLUEPRINT §13) ─────────────────────────────────────
#
# Each method this module brings to the inference table enters through the one registry
# (``turbotab.core.contracts``): where it runs, what it may learn from, what it needs, how it is
# asked (its option labeled customary and sound per purpose, with the leash's rung), its
# storyboard, its sentence and its relations to other decisions (MODELING_SEQUENCE §2). The chain
# test fires every relation listed.

MI_CONTRACTS: tuple[str, ...] = ("multiple_imputation_compatible", "multiple_imputation_passive",
                                 "multiple_imputation_single_level", "imputed_copies_pooled")
_MI_SENTENCE = "turbotab.core.methods.missing:mi_sentence"
_MI_SOURCES = ("Bartlett, Seaman, White & Carpenter 2015 (SMC-FCS)",
               "White, Royston & Wood 2011 (m and Monte Carlo error)",
               "Barnard & Rubin 1999; Reiter 2007 (degrees of freedom)",
               "Lüdtke, Robitzsch & Grund 2017 (clustered imputation)")
# The ways forward the refusals of a blocked imputation name (``decisions._imputation_fits_the_analysis``)
# besides the sound one: the conflict relations below list them, in order.
_COMPLETE_CASES_EXIT = "Complete cases, with their assumption stated"
_KEEP_EXIT = "Keep it, recorded as a limitation"
_MI_SCOPE = ("Under inference the imputation model holds the outcome and reads every analyzed row "
             "(BLUEPRINT §12 rulings 3 and 4): a row's draws move when the outcome or another row "
             "changes, so it learns what the outcome model learns from; under prediction it is "
             "refused, as a new row has no outcome to impute with. It is the first step of each "
             "copy's pipeline (MODELING_SEQUENCE §1.1: the copies are drawn, then each copy's "
             "derived terms, scale scores and spline basis, then its fit).")


def _register_contracts() -> None:
    from turbotab.core.contracts import (CONTRACTS as REGISTRY, ContractOption, MethodContract,
                                        Relation, register_contract)

    if MI_CONTRACTS[0] in REGISTRY:
        return
    inference = ("inference",)
    here = "turbotab.core.methods.missing"
    not_here = "Not here: it uses the outcome, which a new row does not have"
    question = "Missing values under inference: how is the imputation model built?"

    def option(key: str, label: str, customary: str, sound: str, rung: str) -> ContractOption:
        return ContractOption(key, label, customary, {"inference": sound, "prediction": not_here},
                              {"inference": rung, "prediction": "refused"})

    register_contract(MethodContract(
        key="multiple_imputation_compatible",
        label="Multiple imputation compatible with the analysis model",
        slot="in_fold", scope="model", scope_note=_MI_SCOPE, run_order=0.5,
        needs=("the outcome", "the incomplete predictors",
               "the declared terms (forms, the energy model)",
               "the energy sources' settled kcal per unit", "the survey design",
               "the unit of clustering"),
        question=question,
        options=(option(
            "compatible", "Compatible with the analysis model (SMC-FCS where the model needs it)",
            "Chained equations with derived terms made in each copy (passive) are the field's habit "
            "(mice's and Stata's defaults)",
            "Sound: SMC-FCS where the model holds a spline, a log, a ratio or a logistic or Cox "
            "outcome, chained equations where it is linear in the imputed values (Bartlett et al. "
            "2015)", "recommended"),),
        storyboard=("Read the analysis model's terms and fix the knots on the observed values",
                    "Impute each incomplete column on the analysis scale (logs; the energy identity)",
                    "Draw each blank compatibly with the analysis model (SMC-FCS or chained "
                    "equations)",
                    "Fit the analysis pipeline on each completed copy",
                    "Pool every estimate by Rubin's rules, multi-parameter tests by D1"),
        relations=(
            Relation("implies", "pooled_estimates",
                     "every estimate shown is pooled: the coefficient table, the form tests (D1) "
                     "and the substitution curve", purposes=inference,
                     condition="multiple imputation under inference",
                     enforced_by="turbotab.core.stages.modeling:pooled_table",
                     id="mi.pools_every_estimate"),
            Relation("implies", "smcfcs",
                     "SMC-FCS with the analysis model as its substantive model", purposes=inference,
                     condition="a declared nonlinear or derived term (spline, quintiles, log "
                               "residual, density) or a logistic or Cox outcome",
                     enforced_by=f"{here}:engine_substantive",
                     id="mi.nonlinear_term_implies_compatible"),
            Relation("implies", "log_scale_imputation",
                     "it is imputed on the log scale, so no copy holds a non-positive value",
                     purposes=inference, condition="a column the analysis logs",
                     enforced_by=f"{here}:logged_columns", id="mi.logs_imply_log_scale"),
            Relation("implies", "energy_identity",
                     "the sources and the rest are imputed and total energy derived as their sum",
                     purposes=inference,
                     condition="energy sources with a settled kcal per unit and total energy in the "
                               "imputation model",
                     enforced_by=f"{here}:energy_identity", id="mi.energy_identity"),
            Relation("implies", "fixed_knots",
                     "knots and cut points placed once on the observed values, held in every copy",
                     purposes=inference, condition="a spline or quintile form on an imputed column",
                     enforced_by=f"{here}:fixed_forms", id="mi.knots_fixed"),
            Relation("invalidates", "imputations",
                     "the imputations, redrawn with the new analysis model", purposes=inference,
                     condition="a change to the declared form or the energy model",
                     enforced_by="turbotab.core.stages:build_graph",
                     id="mi.form_change_invalidates_imputations"),
            Relation("implies", "design_in_imputation",
                     "the strata, PSU and weight in the imputation model, and ν_com = the design "
                     "df in Rubin's rules and D1", purposes=inference,
                     condition="a survey design with a population estimand",
                     enforced_by=f"{here}:design_variables",
                     id="mi.survey_implies_design_variables_and_df"),
            Relation("implies", "clustered_imputation",
                     "time-invariant variables imputed once per unit, the rest with unit means",
                     purposes=inference, condition="rows a unit repeats",
                     enforced_by=f"{here}:unit_level_columns",
                     id="mi.clusters_imply_clustered_imputation"),
            Relation("precedes", "scales",
                     "a scale's items are imputed before it is scored in each copy, never the "
                     "score itself (MODELING_SEQUENCE §1.1)", purposes=inference,
                     condition="a declared scale with missing items",
                     enforced_by="turbotab.core.models.pipeline:shared_steps",
                     id="mi.items_before_the_score"),
        ),
        sources=_MI_SOURCES, decision="set_missing", stage="fit", place="6 · Missing data",
        sentence=_MI_SENTENCE,
        leash={"inference": "recommended", "prediction": "refused"}))

    register_contract(MethodContract(
        key="multiple_imputation_passive",
        label="Passive multiple imputation (chained equations, terms derived per copy)",
        slot="in_fold", scope="model", scope_note=_MI_SCOPE, run_order=0.5,
        needs=("the outcome", "the incomplete predictors"),
        question=question,
        options=(option(
            "passive", "Passive: terms derived in each completed copy",
            "Customary: mice's and Stata's ice defaults derive transformed and product terms in "
            "each completed copy",
            f"Unsound with a declared nonlinear term: {PASSIVE_CAUTION}", "block_and_record"),),
        storyboard=("Impute each column by chained equations with the outcome",
                    "Derive the declared terms in each completed copy", "Fit and pool"),
        relations=(
            Relation("conflicts", "nonlinear_term",
                     "blocked and recorded: the form tests are biased toward the null",
                     purposes=inference, rung="block_and_record",
                     condition="passive imputation and a declared nonlinear term, under inference",
                     exits=("Multiple imputation compatible with the analysis model (SMC-FCS)",
                            _COMPLETE_CASES_EXIT, _KEEP_EXIT),
                     enforced_by="turbotab.core.decisions:_imputation_fits_the_analysis",
                     id="mi.passive_conflicts_with_nonlinear_term"),
        ),
        sources=_MI_SOURCES, decision="set_missing", stage="fit", place="6 · Missing data",
        sentence=_MI_SENTENCE,
        leash={"inference": "block_and_record", "prediction": "refused"}))

    register_contract(MethodContract(
        key="multiple_imputation_single_level",
        label="Single-level multiple imputation on clustered rows",
        slot="in_fold", scope="model", scope_note=_MI_SCOPE, run_order=0.5,
        needs=("the outcome", "the incomplete predictors", "the unit of clustering"),
        question=question,
        options=(option(
            "single_level", "Single-level on clustered rows",
            "Customary: chained equations that ignore the clustering",
            f"Unsound on clustered rows: {SINGLE_LEVEL_CAUTION}", "block_and_record"),),
        storyboard=("Impute each row on its own, the unit left out", "Fit and pool"),
        relations=(
            Relation("conflicts", "clustered_rows",
                     "blocked and recorded: a unit's time-invariant values differ between its rows",
                     purposes=inference, rung="block_and_record",
                     condition="single-level imputation and rows a unit repeats, under inference",
                     exits=("Clustered multiple imputation (time-invariant values once per unit)",
                            _COMPLETE_CASES_EXIT, _KEEP_EXIT),
                     enforced_by="turbotab.core.decisions:_imputation_fits_the_analysis",
                     id="mi.single_level_conflicts_with_clusters"),
        ),
        sources=_MI_SOURCES, decision="set_missing", stage="fit", place="6 · Missing data",
        sentence=_MI_SENTENCE,
        leash={"inference": "block_and_record", "prediction": "refused"}))

    register_contract(MethodContract(
        key="imputed_copies_pooled",
        label="The data's own imputed copies, pooled by Rubin's rules",
        slot="reshape", scope="row_local", run_order=1.0,
        scope_note="The copies were drawn by the data's provider; nothing is learned here: each "
                   "row is kept as a record of its own copy, as it came, when the grain is read.",
        needs=("the column numbering the copies", "the unit the copies belong to"),
        question="The rows repeat as the data's own imputed copies: how are they analyzed?",
        options=(ContractOption(
            "supplied", "Each copy analyzed with its own outcome, pooled by Rubin's rules",
            "Customary in NHANES DXA analyses: each copy analyzed, the estimates combined (NCHS's "
            "combining rules)",
            {"inference": "Sound: each copy carries its own outcome; Rubin's rules carry the "
                          "imputation's uncertainty",
             "prediction": "The copies are kept as records; the concern is stated"},
            {"inference": "recommended", "prediction": "available"}),),
        storyboard=("Split the rows by their copy number", "Fit each copy with its own outcome",
                    "Pool by Rubin's rules"),
        relations=(
            Relation("implies", "rubins_rules",
                     "each copy analyzed with its own outcome and pooled by Rubin's rules",
                     purposes=inference, condition="imputed copies kept as records, under inference",
                     enforced_by=f"{here}:supplied_copies", id="copies.imply_rubins_rules"),
            Relation("conflicts", "combined_copies",
                     "blocked and recorded: imputed values treated as measured",
                     purposes=inference, rung="block_and_record",
                     condition="imputed copies combined into one row per unit, under inference",
                     exits=("Keep each copy as a record, pooled by Rubin's rules",
                            "Combine them, recorded: intervals too narrow"),
                     enforced_by="turbotab.core.structural:_imputed_copies_are_not_combined_unrecorded",
                     id="copies.conflict_with_combining"),
        ),
        sources=("NCHS, NHANES DXA multiple imputation data files (2008)", "Rubin 1987"),
        decision="set_repeat_kind", stage="fit", place="6 · Missing data", sentence=_MI_SENTENCE,
        leash={"inference": "recommended", "prediction": "available"}))


_register_contracts()


# The sub-answers of multiple imputation as the missing-values card offers them (§11.3: an option
# is never offered silently): each with its contract's labels and rung for the purpose.
_SUB_OPTIONS = (
    ("multiple_imputation_compatible", {"imputation_model": "compatible"}),
    ("multiple_imputation_passive", {"imputation_model": "passive"}),
    ("multiple_imputation_single_level", {"imputation_levels": "single_level"}),
)


def imputation_model_options(purpose: str | None) -> list[dict[str, Any]]:
    """Multiple imputation's sub-answers for ``purpose`` (prediction when undeclared), soundest
    first, each with its contract's customary and sound labels and its rung."""
    from turbotab.core.contracts import contract

    p = purpose if purpose in ("prediction", "inference") else "prediction"
    out = []
    for contract_key, decision in _SUB_OPTIONS:
        (labeled,) = contract(contract_key).options_for(p)
        out.append({"key": labeled["key"], "label": labeled["label"],
                    "customary": labeled["customary"], "sound": labeled["sound"],
                    "rung": labeled["rung"],
                    "decision": {"strategy": "multiple_imputation", **decision}})
    return out


__all__ = [
    "imputation_model_options", "MI_CONTRACTS", "CopyGuard", "FixedForms", "ImputationPlan",
    "PASSIVE_CAUTION", "SINGLE_LEVEL_CAUTION", "copy_template", "design_variables",
    "energy_identity", "fixed_forms", "imputation_plan", "logged_columns", "mi_sentence",
    "nonlinear_terms", "supplied_copies", "unit_level_columns",
    "BELOW_DETECTION_LABELS", "BelowDetection", "BelowDetectionFill", "COMPLETE_CASE_ASSUMPTION",
    "EnergyAwareImputer", "INDICATOR_CAUTION", "LEFT_CENSORED", "METHODS", "MI_ASSUMPTION",
    "MissingMethod", "ROW_LOSS_SHARE", "RULE", "SINGLE_FILL_CAUTION", "ZEROS", "below_detection_options",
    "below_limit_mean", "censored_columns", "energy_fill", "methods_for", "register_method",
    "row_loss", "row_loss_concern", "impute_for_inference", "imputation_frame", "mi_concerns",
    "missing_block", "multiple_imputation_info",
]
