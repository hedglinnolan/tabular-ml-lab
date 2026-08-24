"""
Base model wrapper interface.
All model wrappers should inherit from BaseModelWrapper.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator


class BaseModelWrapper(BaseEstimator, ABC):
    """Base class for all model wrappers.

    `MODELS-025`, `L56`. **Every subclass of this class failed to complete a
    training run**, and had done since `scikit-learn` 1.6 made the tags API
    required for an estimator used inside a `Pipeline`. The four affected
    registry keys are the wrapper-based ones — `glm`, `huber`, `rf` and `nn` —
    so four of twenty-two models sat on a shelf `PRODUCT_VISION.md` says is
    never shortened, were selectable, were ranked with an evidence-bearing
    concern, and **errored the moment a user fitted them.**

    The suite was green over it because **nothing trained one**: no test in
    `turbotab/` or `tests/` fitted `glm`, `huber` or `rf`, and none asserted
    that a run came back with no errored results. 2,449 passing tests said
    nothing about four unusable models. That absence is the defect's cause of
    survival, so the repair ships with the test that was missing.

    **Two additions, and the second is the one a naive fix forgets.**
    `BaseEstimator` supplies `__sklearn_tags__`, `get_params` and `set_params`
    — checked: all twenty-two registry models `clone()` cleanly. But sklearn
    decides *fitted* by looking for trailing-underscore attributes, and these
    wrappers record it as `self.is_fitted`, so inheritance alone moved the
    failure from `__sklearn_tags__` to `NotFittedError`. `__sklearn_is_fitted__`
    is the protocol's own way to say it, and it reads the flag the wrappers
    already keep rather than renaming anything.
    """

    def __sklearn_is_fitted__(self) -> bool:
        """Fitted-ness, in the protocol's vocabulary and the wrapper's state.

        Read from `is_fitted` rather than from a trailing-underscore attribute,
        because that flag is what every `fit` here already sets and what every
        `predict` here already checks. Renaming it would be a second source of
        truth for one fact.
        """
        return bool(getattr(self, "is_fitted", False))


    def __init__(self, name: str):
        """
        Initialize model wrapper.
        
        Args:
            name: Model name/identifier
        """
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classification).
        Returns None if not supported.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities or None
        """
        return None
    
    def get_model(self) -> Any:
        """Get the underlying model object."""
        return self.model

    # ── the wrapped estimator's coefficients, forwarded ──────────────────────
    #
    # `GUIDED-234`, `L56-C3`. Every wrapper here holds a fitted estimator in
    # `self.model` and forwarded none of its parameters, so `GLM (OLS/Logistic)`
    # and `GLM (Huber)` — the two models the registry names `interpretability_
    # tier="high"`, `supports_shap="linear"`, with notes reading *"Interpretable"*
    # — were the two the coefficient figure could not draw.
    #
    # `turbotab.figure_bundle._coefficients_for` decides whether §A4.7's forest
    # plot can be drawn by asking `hasattr(estimator, "coef_")` on the model
    # step of the fitted pipeline. It got `False`, skipped them, and a project
    # whose only fitted models were the two named for coefficients produced the
    # refusal written for tree ensembles.
    #
    # **A PROPERTY THAT RAISES IS THE POINT, NOT AN OVERSIGHT.** `hasattr`
    # answers `False` when the access raises `AttributeError`, so a wrapper
    # around a Random Forest or a neural net keeps saying *no coefficients* —
    # which is true — while the linear ones start saying yes. One accessor,
    # correct in both directions, and it stays correct for a wrapper added
    # later without anyone remembering to declare anything.
    #
    # **UNFITTED IS ALSO `False`, AND THAT IS THE HONEST ANSWER.** There are no
    # coefficients before a fit, and reporting some would be inventing them.

    @property
    def coef_(self) -> Any:
        """The wrapped estimator's coefficients. Absent until it has some."""
        model = self.model
        if model is None:
            raise AttributeError("coef_")
        return model.coef_

    @property
    def intercept_(self) -> Any:
        """The wrapped estimator's intercept, on the same contract as `coef_`.

        Forwarded WITH `coef_` rather than after it: a coefficient plot that
        cannot say where the line crosses is a plot of slopes, and
        `reporting_checklist` asks for *"full coefficients and intercept, or
        the model object"* in as many words.
        """
        model = self.model
        if model is None:
            raise AttributeError("intercept_")
        return model.intercept_

    # ── which classes the wrapped estimator learned, forwarded ───────────────
    #
    # `GUIDED-245`, `L64-A`. `coef_` above was added because a wrapper dropped
    # an attribute and the forest plot went missing. `classes_` is the same
    # defect one attribute over, and it costs more: `turbotab/training.py:648`
    # records WHICH CLASS a model's probabilities are about by reading
    # `classes_` off the fitted pipeline, and a wrapper that does not forward
    # it makes `positive_label` `None`. Driven on `clinical_risk.csv`: a
    # project fitted with only `glm` — or only `rf` — served **no ROC, no
    # calibration plot and no decision curve at all**, because
    # `predictions_for` refuses a run that cannot say which event it is about.
    # That refusal is correct (`GUIDED-093`: guessing `1` is right on a 0/1
    # target and silently wrong on `responder`/`non-responder`); the wrapper
    # lying about itself is not. `models/glm.py:25` holds a real
    # `LogisticRegression`, so the answer was one attribute away the whole time.
    #
    # **A SENTINEL, AND IT IS NOT DECORATION — BOTH NAIVE VERSIONS BREAK `nn`.**
    #
    #   A plain read-only property fails at `NNWeightedHuberWrapper.__init__`,
    #   not at its `fit`: `models/nn_whuber.py:320` assigns `self.classes_ =
    #   None` before any data exists, and assigning through a property with no
    #   setter raises. The nn wrapper would stop CONSTRUCTING.
    #
    #   "Defer to the instance attribute if it is not None" fails too, for the
    #   same line: `nn` assigns `None` deliberately and reads it back with
    #   `is not None` at `:636`. A None-check falls through to `self.model` — a
    #   torch module with no `classes_` — and raises.
    #
    # **AND THE ANSWER IS `__getattr__`, NOT A THIRD PROPERTY.** Python
    # consults `__getattr__` only when normal lookup FAILS, and an explicit
    # assignment lands in the instance `__dict__` — so `nn`'s deliberate `None`
    # is found by normal lookup and returned as itself, with no sentinel and no
    # setter needed. Both naive property versions break precisely because a
    # property intercepts a lookup that should have found the instance
    # attribute; `__getattr__` does not intercept it at all.
    #
    # A sentinel property WOULD also work, and it was built first. It was
    # removed because it is redundant beside the rule below — a property that
    # raises `AttributeError` falls through to `__getattr__` anyway, so the two
    # are one mechanism written twice, which is the two-engines failure this
    # codebase names everywhere else.
    #
    # **UNFITTED IS STILL `False`, by the same mechanism as `coef_`**: the
    # estimator raises `AttributeError` before it has seen data, `hasattr`
    # answers `False`, and a `clone()` — which carries no assigned value and no
    # fitted estimator — answers `False` too. A Random Forest REGRESSOR also
    # answers `False`, because `RandomForestRegressor` has no `classes_` and
    # the forwarding is one accessor rather than a declaration to keep in sync.

    # ── and the rule, rather than a third name ───────────────────────────────
    #
    # `L64-A5`. `coef_` was added at L56 because a wrapper dropped an
    # attribute. `intercept_` was added beside it. `classes_` was missing for
    # seven loops after that and cost a project its three clinical figures.
    # **Naming today's attributes is how this recurs**, so the general rule is
    # written down once: a trailing-underscore public name is sklearn's own
    # convention for *something a fit produced*, and a wrapper that holds a
    # fitted estimator answers for all of them.
    #
    # Narrow on purpose. Only `name_` — not `_private`, not `__dunder__`, not
    # `plain` — so this cannot swallow a genuine typo on a method or mask a
    # missing attribute of the wrapper's own. `model` is read out of
    # `__dict__` rather than through `self`, because `__getattr__` firing on
    # `model` before `__init__` sets it would recurse forever.
    #
    # The two properties above still win for the names they cover: they are on
    # the class, so normal lookup finds them and `__getattr__` is never asked.
    # They are kept for the contract they document, not because this needs them.

    def __getattr__(self, name: str) -> Any:
        if (name.startswith("_") or not name.endswith("_")
                or name.endswith("__")):
            raise AttributeError(name)
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        try:
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(name) from None

    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return False
