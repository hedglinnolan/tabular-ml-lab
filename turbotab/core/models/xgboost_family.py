"""``xgboost``: gradient-boosted trees in the XGBoost library (RECIPES_AND_TUNING §2.2, §4.1;
WAVE_C6A_PLAN §3, package RT-5e).

The same kind of model as ``boosted_trees``, offered by name: the shelf reads boosted trees'
assessment less 1.0, at least 0.5 (``same_kind_as``; ``base.assessment``), never its own.

**The wrapper** (:class:`XGBoostRegressor`, :class:`XGBoostClassifier`) is a scikit-learn estimator
around xgboost's own ``XGBRegressor`` / ``XGBClassifier`` that gives the engine what it reads from
histogram boosting:

* **the early-stopping interface:** ``early_stopping`` and ``validation_fraction`` as parameters,
  and ``fit(X, y, X_val, y_val)``, so ``inner_cv.fit_pipeline`` hands it the stopping units it drew
  first (RECIPES F11, §4.3) and an imbalance correction defers to it (F12). Stopped, the booster
  keeps the rounds up to the best one only, so its predictions, its trees and its SHAP values are
  one model's;
* **a margin** ``decision_function`` (a yes/no outcome's log-odds of the class coded 1; one column
  per class for several), which ``Anatomy.raw_score`` reads;
* **labels encoded to 0…K−1**, as xgboost requires, and decoded on the way out;
* **safe column names:** xgboost refuses ``[``, ``]`` and ``<`` in a name, so each is replaced
  (:func:`safe_names`) and the original names are mapped back (``feature_names_in_``,
  ``original_names_``);
* **pinned threads:** ``n_jobs`` is xgboost's ``nthread``, 1 unless the plan says otherwise
  (``settings`` reads ``plan.threads``), so a fit is reproduced bit for bit at its recorded count.

**Settings** (:meth:`XGBoost.settings`): the least child weight is searched in rows (1 to 64) and
multiplied by the mean hessian at the base score on each fit's own outcome
(:func:`mean_hessian`), since it is a sum of hessians: at 5% prevalence an unscaled 64 would ask
for about 1,350 rows per leaf (RECIPES §4.1). The standard settings are XGBoost's defaults, so
the standard child weight is its unscaled 1 (``"default"``).

**Trees** (:func:`xgb_ensemble`): the fitted booster's JSON as an ``explain.TreeEnsemble`` on the
margin, with xgboost's split rule (a row goes left when its value is ``<`` the threshold) and
each split's learned default branch for blanks. **SHAP** (:meth:`XGBoost.tree_shap`) is
xgboost's own ``pred_contribs``.

**Loading.** xgboost is imported only when a model is fit or read, never when this module is. If
it cannot load (libomp missing on a launcher, say), the family stays registered, every other
family is unaffected, and a fit refuses with "XGBoost could not load" and the reason
(:class:`XGBoostUnavailable`).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Knob,
    Named,
    Situation,
    Source,
    assessment,
    register_family,
)
from turbotab.core.models.tuning import Dimension, TuningDecl

# xgboost refuses these characters in a feature name; each is replaced by its stand-in.
UNSAFE = {"[": "(", "]": ")", "<": "lt"}
STOP_ROUNDS = 2_000  # the most rounds under early stopping (RECIPES §4.1)
PATIENCE = 50  # rounds without improvement on the stopping rows before it stops
STANDARD_CHILD_WEIGHT = 1.0  # XGBoost's default min_child_weight, unscaled


class XGBoostUnavailable(RuntimeError):
    """xgboost could not be imported here: the family refuses to fit, saying why."""


_LOADED: dict[str, Any] = {}


def load_error() -> str | None:
    """Why xgboost cannot load here, or None when it can."""
    try:
        _xgb()
    except XGBoostUnavailable as e:
        return str(e)
    return None


def _xgb() -> Any:
    """The xgboost module, imported on first use; :class:`XGBoostUnavailable` when it cannot be."""
    if "module" in _LOADED:
        return _LOADED["module"]
    try:
        import xgboost
    except Exception as e:  # noqa: BLE001 - an ImportError, or an OSError from a missing libomp
        raise XGBoostUnavailable(f"XGBoost could not load: {type(e).__name__}: {e}") from e
    _LOADED["module"] = xgboost
    return xgboost


def safe_names(names: Any) -> list[str]:
    """Each name with ``[``, ``]`` and ``<`` replaced (:data:`UNSAFE`), kept distinct: a name that
    would repeat an earlier one gains ``~2``, ``~3``, … until it does not."""
    out: list[str] = []
    taken: set[str] = set()
    for name in (str(n) for n in names):
        safe = "".join(UNSAFE.get(ch, ch) for ch in name)
        candidate, k = safe, 1
        while candidate in taken:
            k += 1
            candidate = f"{safe}~{k}"
        taken.add(candidate)
        out.append(candidate)
    return out


def mean_hessian(y: Any, *, classify: bool) -> float:
    """The mean per-row hessian of xgboost's loss at its base score, on these rows' outcome.

    * A number (squared error): 1.
    * Two classes (``binary:logistic``, the base score the share p̄ of the class coded 1):
      p̄(1 − p̄).
    * K > 2 classes (``multi:softprob``, the base score each class's share p̄ₖ): the mean over the
      classes of 2·p̄ₖ(1 − p̄ₖ). xgboost's softmax hessian is 2p(1 − p) (its ``multiclass_obj``),
      which RECIPES §4.1's "mean of p̄ₖ(1−p̄ₖ)" leaves out; the factor keeps the child weight in
      rows. Each tree's root cover (its hessian sum) is n times these, which the tests check
      against xgboost's own covers.

    ``y`` is the fit's own outcome (its training rows): the scaling is never read from rows the
    fit does not train on."""
    if not classify:
        return 1.0
    if y is None:
        raise ValueError("The least child weight is scaled by the fit's own outcome, and none "
                         "was given.")
    y = np.asarray(y)
    if len(y) == 0:
        raise ValueError("The least child weight is scaled by the fit's own outcome, which has "
                         "no rows.")
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    if len(p) <= 2:
        return float(p[0] * (1.0 - p[0]))
    return float(np.mean(2.0 * p * (1.0 - p)))


def _as_matrix(X: Any) -> np.ndarray:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(X, dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# the wrapper
# ═════════════════════════════════════════════════════════════════════════════


class _XGBoostStep(BaseEstimator):
    """What the two wrappers share; see the module docstring. The parameters are xgboost's own
    names and defaults (1.x–3.x: learning rate 0.3, depth 6, child weight 1, every row and
    column, λ = 1, α = 0, 100 rounds), plus the early-stopping interface: ``early_stopping``
    (False: never; True: on the stopping rows handed to ``fit`` or, without them, on
    ``validation_fraction`` of the units drawn here; ``"auto"``: as True above 10,000 rows, as
    histogram boosting does), ``validation_fraction`` and ``early_stopping_rounds`` (the
    patience)."""

    _classify = False

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.3, max_depth: int = 6,
                 min_child_weight: float = 1.0, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, reg_lambda: float = 1.0, reg_alpha: float = 0.0,
                 booster: str = "gbtree", tree_method: str = "hist", n_jobs: int = 1,
                 random_state: int = 0, early_stopping: Any = False,
                 validation_fraction: float = 0.1, early_stopping_rounds: int = PATIENCE):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

    # ── the early-stopping interface ──

    def stopping_setting(self) -> tuple[Any, float | None]:
        """(flag, share), as ``inner_cv.stopping_setting`` reads it."""
        return self.early_stopping, self.validation_fraction

    def _stops(self, n_rows: int) -> bool:
        from turbotab.core.models.inner_cv import EARLY_STOPPING_ROWS

        flag = self.early_stopping
        return bool(flag is True or (flag == "auto" and n_rows > EARLY_STOPPING_ROWS))

    def _library_params(self, stops: bool) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators), "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth), "min_child_weight": float(self.min_child_weight),
            "subsample": float(self.subsample), "colsample_bytree": float(self.colsample_bytree),
            "reg_lambda": float(self.reg_lambda), "reg_alpha": float(self.reg_alpha),
            "booster": self.booster, "tree_method": self.tree_method, "n_jobs": int(self.n_jobs),
            "random_state": int(self.random_state), "missing": np.nan, "verbosity": 0,
            "early_stopping_rounds": int(self.early_stopping_rounds) if stops else None,
        }

    # ── names ──

    def _frame(self, X: Any) -> Any:
        """``X`` as the booster reads it: a frame under the safe names when the fit had names,
        else a float matrix."""
        import pandas as pd

        if isinstance(X, pd.DataFrame) and getattr(self, "feature_names_in_", None) is not None:
            given = [str(c) for c in X.columns]
            if given != list(self.feature_names_in_):
                raise ValueError("The columns are not the ones this model was fit on, in that "
                                 "order.")
        matrix = _as_matrix(X)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"This model was fit on {self.n_features_in_} columns, not "
                             f"{matrix.shape[1] if matrix.ndim == 2 else matrix.shape}.")
        if self.safe_names_ is None:
            return matrix
        return pd.DataFrame(matrix, columns=self.safe_names_)

    @property
    def original_names_(self) -> dict[str, str]:
        """Each safe name the booster holds, mapped back to its column's own name."""
        if self.safe_names_ is None:
            return {}
        return dict(zip(self.safe_names_, self.feature_names_in_))

    # ── the fit ──

    def _encode(self, y: Any) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def _library_model(self, xgb: Any, params: dict[str, Any]) -> Any:
        return xgb.XGBRegressor(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None, *,
            sample_weight: Any = None, sample_weight_val: Any = None) -> "_XGBoostStep":
        """Fit on ``X``, ``y``. ``X_val``, ``y_val``: the stopping rows (``inner_cv.fit_pipeline``
        draws them as whole units, first); passed only to a fit that stops early. Without them,
        a fit that stops early draws ``validation_fraction`` of the rows' units here, keyed by
        each row's contents (``inner_cv.validation_rows``), stratified for classes."""
        import pandas as pd

        from turbotab.core.models.inner_cv import row_keys, validation_rows

        xgb = _xgb()
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
            self.safe_names_ = safe_names(self.feature_names_in_)
        else:
            self.safe_names_ = None
            self.__dict__.pop("feature_names_in_", None)  # an earlier fit's names do not apply
        matrix = _as_matrix(X)
        self.n_features_in_ = int(matrix.shape[1])
        y_arr = np.asarray(y)
        target = self._encode(y_arr)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        val: tuple[Any, np.ndarray, Any] | None = None
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("Stopping rows need both X_val and y_val.")
            if self.early_stopping is False:
                raise ValueError("Stopping rows were passed to a fit that does not stop early.")
            val = (_as_matrix(X_val), self._encode(np.asarray(y_val)), sample_weight_val)
        elif self._stops(len(y_arr)):
            held = validation_rows(float(self.validation_fraction), keys=row_keys(X, y_arr),
                                   y=y_arr if self._classify else None,
                                   seed=int(self.random_state))
            val = (matrix[held], target[held], None if weights is None else weights[held])
            matrix, target = matrix[~held], target[~held]
            weights = None if weights is None else weights[~held]
        frame = (matrix if self.safe_names_ is None
                 else pd.DataFrame(matrix, columns=self.safe_names_))
        model = self._library_model(xgb, self._library_params(val is not None))
        if val is None:
            model.fit(frame, target, sample_weight=weights, verbose=False)
            booster = model.get_booster()
            self.best_iteration_ = None
        else:
            X_stop = (val[0] if self.safe_names_ is None
                      else pd.DataFrame(val[0], columns=self.safe_names_))
            stop_weights = None if val[2] is None else [np.asarray(val[2], dtype=float)]
            model.fit(frame, target, sample_weight=weights, eval_set=[(X_stop, val[1])],
                      sample_weight_eval_set=stop_weights, verbose=False)
            self.best_iteration_ = int(model.best_iteration)
            # The rounds after the best one are dropped: every reader sees one model.
            booster = model.get_booster()[: self.best_iteration_ + 1]
        self.booster_ = booster
        self.stopping_rows_ = 0 if val is None else int(len(val[1]))
        self.base_margin_ = _base_margin(booster)
        self.n_rounds_ = int(booster.num_boosted_rounds())
        return self

    def _margin(self, X: Any) -> np.ndarray:
        """The booster's raw score, (n,) or (n, K), as xgboost computes it (float32)."""
        return np.asarray(self.booster_.inplace_predict(self._frame(X), predict_type="margin",
                                                        missing=np.nan))

    def _value(self, X: Any) -> np.ndarray:
        return np.asarray(self.booster_.inplace_predict(self._frame(X), predict_type="value",
                                                        missing=np.nan))


class XGBoostRegressor(RegressorMixin, _XGBoostStep):
    """XGBoost on a number (``reg:squarederror``)."""

    def predict(self, X: Any) -> np.ndarray:
        return self._value(X).astype(float)


class XGBoostClassifier(ClassifierMixin, _XGBoostStep):
    """XGBoost on classes: two (``binary:logistic``, the class coded 1 being the second of
    ``classes_``) or more (``multi:softprob``), the labels encoded to 0…K−1 in ``classes_``'s
    sorted order."""

    _classify = True

    def _encode(self, y: Any) -> np.ndarray:
        y = np.asarray(y)
        codes = np.clip(np.searchsorted(self.classes_, y), 0, len(self.classes_) - 1)
        if not np.array_equal(self.classes_[codes], y):
            raise ValueError("The stopping rows hold a class the training rows do not.")
        return codes.astype(int)

    def _library_model(self, xgb: Any, params: dict[str, Any]) -> Any:
        return xgb.XGBClassifier(**params)

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None, *,
            sample_weight: Any = None, sample_weight_val: Any = None) -> "XGBoostClassifier":
        classes = np.unique(np.asarray(y))
        if len(classes) < 2:
            raise ValueError("XGBoost needs at least two classes among the rows it is fit on.")
        self.classes_ = classes  # read by _encode, for the stopping rows too
        return super().fit(X, y, X_val, y_val, sample_weight=sample_weight,
                           sample_weight_val=sample_weight_val)

    def decision_function(self, X: Any) -> np.ndarray:
        """The margin: (n,) log-odds of ``classes_[1]`` for two classes, (n, K) for more."""
        return self._margin(X).astype(float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self._value(X)
        if p.ndim == 1:  # as xgboost's own XGBClassifier: 1 − p in single precision
            p = np.column_stack([np.float32(1.0) - p, p])
        return p.astype(float)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _base_margin(booster: Any) -> np.ndarray:
    """The booster's base score on the margin, one per output: xgboost stores a logistic
    objective's as a probability (its logit is the margin) and the others' on the margin."""
    config = json.loads(booster.save_config())
    learner = config["learner"]
    text = str(learner["learner_model_param"]["base_score"]).strip("[]")
    held = [float(v) for v in text.split(",") if v.strip()]
    base = np.asarray(held, dtype=np.float32).astype(float)  # as xgboost holds it
    if learner["objective"]["name"] in ("binary:logistic", "reg:logistic"):
        base = np.log(base) - np.log1p(-base)
    return base


# ═════════════════════════════════════════════════════════════════════════════
# its trees and its SHAP values
# ═════════════════════════════════════════════════════════════════════════════

_NODE = np.dtype([("value", "f8"), ("count", "f8"), ("feature_idx", "i8"),
                  ("num_threshold", "f8"), ("missing_go_to_left", "u1"), ("left", "i8"),
                  ("right", "i8"), ("depth", "i8"), ("is_leaf", "u1")])


def _table(tree: dict[str, Any]) -> np.ndarray:
    """One tree of xgboost's JSON model as a node table (``explain.TreeEnsemble``'s ``tables``):
    a leaf's value is its ``split_conditions`` entry (the learning rate applied), a split's
    threshold likewise; ``count`` is the node's hessian sum (xgboost's cover, which its TreeSHAP
    weighs by); a blank goes left where ``default_left`` says.

    The JSON prints each single-precision number in its shortest form ("-0.23"), which read as a
    double is not the number xgboost holds; each is read back through single precision, so a
    threshold is xgboost's own and a row on it goes right, as in xgboost."""
    left = tree["left_children"]
    right = tree["right_children"]
    if any(int(t) != 0 for t in tree.get("split_type", [])):
        raise ValueError("a categorical split: the pipeline one-hot encodes categories first")
    condition = np.asarray(tree["split_conditions"], dtype=np.float32).astype(float)
    cover = np.asarray(tree["sum_hessian"], dtype=np.float32).astype(float)
    nodes = np.zeros(len(left), dtype=_NODE)
    queue = [(0, 0)]
    while queue:
        i, d = queue.pop()
        leaf = int(left[i]) == -1
        nodes[i] = (condition[i], cover[i],
                    0 if leaf else int(tree["split_indices"][i]),
                    0.0 if leaf else condition[i],
                    0 if leaf else int(bool(int(tree["default_left"][i]))),
                    0 if leaf else int(left[i]), 0 if leaf else int(right[i]), d, int(leaf))
        if not leaf:
            queue.extend([(int(left[i]), d + 1), (int(right[i]), d + 1)])
    return nodes


def xgb_ensemble(step: Any) -> Any:
    """A fitted :class:`XGBoostRegressor` / :class:`XGBoostClassifier` as an
    ``explain.TreeEnsemble`` on its margin: one output for a number or two classes, one per class
    for more; rule ``"lt"`` (a row goes left when its value is below the threshold, xgboost's
    ``<``). xgboost compares a value in single precision; a value of double precision just below
    a threshold that rounds up to it goes right in xgboost, so the engine's own TreeSHAP matches
    xgboost's exactly on single-precision values (the family's ``tree_shap`` is xgboost's own)."""
    from turbotab.core.models.explain import TreeEnsemble, leaf_paths

    model = json.loads(step.booster_.save_raw("json"))
    gbtree = model["learner"]["gradient_booster"]["model"]
    outputs = len(step.base_margin_)
    tables: list[list[np.ndarray]] = [[] for _ in range(outputs)]
    for tree, k in zip(gbtree["trees"], gbtree["tree_info"]):
        tables[int(k)].append(_table(tree))
    trees = [[leaf_paths(nodes) for nodes in per_output] for per_output in tables]
    return TreeEnsemble(base=np.asarray(step.base_margin_, dtype=float), trees=trees,
                        tables=tables, rule="lt", scale="margin")


def contributions(step: Any, Z: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """xgboost's own path-dependent TreeSHAP (``pred_contribs``) of every row of ``Z``:
    ``(phi (n, p, K), expected (K,))`` on the margin; None for no rows."""
    xgb = _xgb()
    matrix = _as_matrix(Z)
    if len(matrix) == 0:
        return None
    data = xgb.DMatrix(matrix, missing=np.nan, feature_names=step.safe_names_,
                       nthread=int(step.n_jobs))
    c = np.asarray(step.booster_.predict(data, pred_contribs=True), dtype=float)
    if c.ndim == 2:
        c = c[:, None, :]  # (n, 1, p + 1)
    phi = np.transpose(c[:, :, :-1], (0, 2, 1))  # (n, p, K)
    return phi, c[0, :, -1].copy()


# ═════════════════════════════════════════════════════════════════════════════
# the family
# ═════════════════════════════════════════════════════════════════════════════

_TUNING_NOTES = "Chen & Guestrin 2016"
_TUNABILITY = "Probst, Boulesteix & Bischl 2019"

TUNING = TuningDecl(
    "search",
    dimensions=(
        Dimension("learning_rate", "how big each correction step is", "learning rate", 0.01, 0.3,
                  "log", source=_TUNABILITY),
        Dimension("max_depth", "how many questions deep each tree goes", "maximum depth", 2, 10,
                  "int", source=_TUNABILITY),
        Dimension("min_child_weight", "how much evidence a split needs, in rows",
                  "minimum child weight", 1, 64, "log",
                  source=f"{_TUNABILITY} (child weight 2^[0, 7]); {_TUNING_NOTES}"),
        Dimension("subsample", "the share of rows each tree sees", "row subsampling", 0.5, 1.0,
                  "linear", source=_TUNABILITY),
        Dimension("colsample_bytree", "the share of columns each tree sees",
                  "column subsampling", 0.3, 1.0, "linear", source=_TUNABILITY),
        Dimension("reg_lambda", "how strongly each group's value is pulled toward zero",
                  "L2 regularization", 1e-3, 100.0, "log",
                  source=f"{_TUNABILITY} (λ 2^[−10, 10])"),
        Dimension("n_estimators", "how many trees", "boosting rounds", 25, 1000, "log_int",
                  source=_TUNABILITY, active="without_early_stopping"),
    ),
    standard={"learning_rate": 0.3, "max_depth": 6, "min_child_weight": "default",
              "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0, "n_estimators": 100},
    standard_source="XGBoost's defaults",
    fixed={"booster": "gbtree", "reg_alpha": 0.0, "tree_method": "hist"},
    # The standard settings never stop early (XGBoost's defaults run 100 rounds); Sobol
    # candidates stop early by the plan, on a tenth of each fit's units drawn first.
    early_stopping={"param": "n_estimators", "rounds": STOP_ROUNDS,
                    "patience_param": "early_stopping_rounds", "patience": PATIENCE,
                    "share": 0.1, "standard_rows": None},
    space_version="xgboost/1",
    structural=("booster", "objective"),
    reason="Its settings are searched inside every training fold, with XGBoost's defaults always "
           "among the candidates.",
)


class XGBoost(FamilyBase):
    key = "xgboost"
    label = "XGBoost"
    inductive_bias = ("Effects are step functions that can bend and interact; many shallow trees "
                      "each correct the last.")
    strengths = (
        "Finds curves and interactions without being told.",
        "Sends a blank down the branch each split learned for blanks in training.",
    )
    cautions = (
        "No coefficients: effects are read from curves.",
        "Overfits easily and needs many rows.",
        "Validated by cross-validation: Harrell's bootstrap overstates a near-interpolating "
        "learner's performance.",
    )
    needs_scaling = False
    handles_missing = True
    bootstrap_optimism = False  # near-interpolating, as boosted trees (Coley et al. 2023)
    identity = Identity(kind="estimator", library="xgboost",
                        estimator="XGBoostRegressor; XGBoostClassifier (wrapping xgboost's "
                                  "XGBRegressor and XGBClassifier, tree booster, hist)",
                        seed_policy="random_state from the tuning plan's seed (derive_seed of the "
                                    "split's seed), 0 without a plan; it seeds the row and column "
                                    "subsampling")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = True
    inference_decl = InferenceDecl(table="description_only")
    same_kind_as = ("boosted_trees", -1.0)  # RECIPES §2.2: boosted trees' assessment less 1.0
    invariances = ("monotone_per_column",)
    bias_terms = (Named(plain="Many shallow trees, each fit to what the trees before it missed, "
                              "with every leaf's value pulled toward zero.",
                        known_as="regularized gradient boosting",
                        source=Source("chen2016xgboost")),)
    curve_shape = "piecewise_constant"
    complexity = (
        Knob(setting="n_estimators", more_means="more_flexible"),  # rounds
        Knob(setting="learning_rate", more_means="more_flexible"),
        Knob(setting="max_depth", more_means="more_flexible"),
        Knob(setting="min_child_weight", more_means="simpler"),
        Knob(setting="reg_lambda", more_means="simpler"),
    )
    output = "margin"
    raw_scale = CLASS_SCALES
    attribution = "trees"
    architecture = ("trees",)
    review_lenses = ("shared",)
    sources = (Source("chen2016xgboost"),)
    tuning = TUNING
    consequence = ("The same kind of model as boosted trees, in the XGBoost library reviewers "
                   "often name.")

    def methods_label(self, task: Task | None) -> str:
        return "gradient-boosted trees (XGBoost)"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        """XGBoost's defaults, one thread, never stopping early."""
        if task == "regression":
            return XGBoostRegressor()
        return XGBoostClassifier()

    def settings(self, values: Any, *, task: str, n_units: int, n_rows: int, y: Any = None,
                 Z: Any = None, plan: Any = None) -> dict[str, Any]:
        """A candidate's values as the wrapper's parameters for one fit: the least child weight
        in rows times this fit's mean hessian at the base score (:func:`mean_hessian`), the
        standard ``"default"`` being XGBoost's unscaled 1; the plan's thread count and seed."""
        out = dict(values)
        weight = out.get("min_child_weight", "default")
        if isinstance(weight, str):
            if weight != "default":
                raise ValueError(f"min_child_weight {weight!r} is not a number of rows or "
                                 f"\"default\"")
            out["min_child_weight"] = STANDARD_CHILD_WEIGHT
        else:
            out["min_child_weight"] = float(weight) * mean_hessian(
                y, classify=task != "regression")
        if plan is not None:
            out["n_jobs"] = int(plan.threads)
            out["random_state"] = int(plan.seed)
        return out

    def trees(self, step: Any) -> Any:
        """Its fitted model step's trees, for the tree view and the engine's TreeSHAP (C10)."""
        return xgb_ensemble(step)

    def tree_shap(self, step: Any, Z: Any) -> Any:
        """xgboost's own TreeSHAP (``pred_contribs``) on the margin (C10)."""
        return contributions(step, Z)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        return ("XGBoost",
                "Gradient-boosted trees in the XGBoost library: 100 trees of depth up to 6 at "
                "learning rate 0.3 (its defaults); a blank follows the branch each split learned "
                "for blanks.")

    def assess(self, s: Situation) -> Assessment:
        """Boosted trees' assessment less 1.0, at least 0.5 (``base.assessment``): the same kind
        of model, offered by name."""
        return assessment(self, s)


XGBOOST = register_family(XGBoost())


__all__ = ["PATIENCE", "STOP_ROUNDS", "TUNING", "UNSAFE", "XGBOOST", "XGBoost",
           "XGBoostClassifier", "XGBoostRegressor", "XGBoostUnavailable", "contributions",
           "load_error", "mean_hessian", "safe_names", "xgb_ensemble"]
