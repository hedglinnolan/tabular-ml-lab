"""WorkflowProvenance — Incremental, structured provenance for the full ML pipeline.

Each workflow page records its contribution as the user works. Downstream consumers
(NarrativeEngine, TRIPOD checker, consistency validator, Report Export) read from
this single structure instead of 100+ scattered session_state keys.

The InsightLedger continues handling coaching (observe → recommend → resolve).
WorkflowProvenance captures what happened, not what should happen.

Stored in st.session_state['workflow_provenance'].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Section dataclasses — one per workflow stage
# ---------------------------------------------------------------------------

@dataclass
class UploadProvenance:
    """Recorded when user saves data configuration on Upload & Audit page."""
    target_col: str = ""
    task_type: str = ""  # "regression" or "classification"
    feature_cols: List[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    data_source: str = ""
    timestamp: str = ""

    # Data cleaning actions (appended as user cleans)
    cleaning_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EDAProvenance:
    """Recorded as user runs EDA analyses."""
    analyses_run: List[str] = field(default_factory=list)
    table1_generated: bool = False
    key_findings: List[str] = field(default_factory=list)  # populated by insight ledger
    timestamp: str = ""


@dataclass
class FeatureEngineeringProvenance:
    """Recorded when feature engineering is applied."""
    transforms_applied: List[str] = field(default_factory=list)
    n_features_created: int = 0
    n_features_before: int = 0
    n_features_after: int = 0
    timestamp: str = ""


@dataclass
class FeatureSelectionProvenance:
    """Recorded when feature selection is applied."""
    method: str = ""  # "consensus", "manual", etc.
    n_features_before: int = 0
    n_features_after: int = 0
    features_kept: List[str] = field(default_factory=list)
    consensus_methods: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class ModelPreprocessingConfig:
    """Preprocessing config for a single model pipeline."""
    scaling: str = "none"
    encoding: str = ""
    outlier_treatment: str = "none"
    outlier_params: Dict[str, Any] = field(default_factory=dict)
    power_transform: str = "none"
    log_transform: bool = False
    imputation: str = ""
    use_pca: bool = False
    pca_n_components: Optional[Any] = None
    pca_mode: str = ""


@dataclass
class PreprocessingProvenance:
    """Recorded when preprocessing pipelines are built.

    Captures per-model configs — the core differentiator.
    """
    shared: Dict[str, str] = field(default_factory=dict)  # settings common to all models
    per_model: Dict[str, ModelPreprocessingConfig] = field(default_factory=dict)
    models_configured: List[str] = field(default_factory=list)
    timestamp: str = ""

    def configs_differ(self) -> bool:
        """True if models have different preprocessing configs."""
        if len(self.per_model) <= 1:
            return False
        configs = list(self.per_model.values())
        first = configs[0]
        return any(
            c.scaling != first.scaling
            or c.encoding != first.encoding
            or c.outlier_treatment != first.outlier_treatment
            or c.power_transform != first.power_transform
            or c.log_transform != first.log_transform
            or c.use_pca != first.use_pca
            for c in configs[1:]
        )


@dataclass
class TrainingProvenance:
    """Recorded when models are trained."""
    models_trained: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    primary_model: str = ""
    selection_criteria: str = ""  # "validation RMSE", "validation F1", etc.
    use_cv: bool = False
    cv_folds: Optional[int] = None
    use_hyperopt: bool = False
    class_weight_balanced: bool = False
    metrics_by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class SplitProvenance:
    """Recorded when data splitting is configured."""
    strategy: str = ""  # "random", "stratified", "chronological", "group"
    train_n: int = 0
    val_n: int = 0
    test_n: int = 0
    train_pct: float = 0.0
    val_pct: float = 0.0
    test_pct: float = 0.0
    random_seed: int = 42
    target_transform: str = "none"
    target_trim_enabled: bool = False
    target_trim_lower: float = 0.0
    target_trim_upper: float = 1.0
    timestamp: str = ""


@dataclass
class ExplainabilityProvenance:
    """Recorded when explainability analyses are run."""
    methods_used: List[str] = field(default_factory=list)
    models_explained: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class SensitivityProvenance:
    """Recorded when sensitivity analyses are run."""
    seed_stability: bool = False
    seed_stability_cv: Optional[float] = None
    feature_dropout: bool = False
    timestamp: str = ""


@dataclass
class StatisticalValidationProvenance:
    """Recorded as statistical tests are run."""
    tests_run: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Main provenance container
# ---------------------------------------------------------------------------

@dataclass
class WorkflowProvenance:
    """Single source of truth for what happened in the ML workflow.

    Each section is Optional — only populated when the user reaches that stage.
    Pages write their sections; consumers (NarrativeEngine, TRIPOD, Report Export)
    read the whole structure.
    """
    upload: Optional[UploadProvenance] = None
    eda: Optional[EDAProvenance] = None
    feature_engineering: Optional[FeatureEngineeringProvenance] = None
    feature_selection: Optional[FeatureSelectionProvenance] = None
    split: Optional[SplitProvenance] = None
    preprocessing: Optional[PreprocessingProvenance] = None
    training: Optional[TrainingProvenance] = None
    explainability: Optional[ExplainabilityProvenance] = None
    sensitivity: Optional[SensitivityProvenance] = None
    statistical_validation: Optional[StatisticalValidationProvenance] = None

    # Schema version for forward compatibility
    schema_version: int = 1

    # --- Writer methods (called by pages) ---

    def record_upload(
        self,
        target_col: str,
        task_type: str,
        feature_cols: List[str],
        n_samples: int,
        data_source: str = "",
    ) -> None:
        """Called by Upload & Audit when user saves configuration."""
        self.upload = UploadProvenance(
            target_col=target_col,
            task_type=task_type,
            feature_cols=list(feature_cols),
            n_samples=n_samples,
            n_features=len(feature_cols),
            data_source=data_source,
            timestamp=datetime.now().isoformat(),
        )
        # Reset downstream sections (config changed)
        self.feature_engineering = None
        self.feature_selection = None
        self.preprocessing = None
        self.training = None

    def record_cleaning(self, action: str, rows_before: int, rows_after: int,
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Called by Upload & Audit for each data cleaning action."""
        if self.upload is None:
            return
        self.upload.cleaning_actions.append({
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        })
        self.upload.n_samples = rows_after

    def record_eda_analysis(self, analysis_name: str) -> None:
        """Called by EDA page for each analysis run."""
        if self.eda is None:
            self.eda = EDAProvenance(timestamp=datetime.now().isoformat())
        if analysis_name not in self.eda.analyses_run:
            self.eda.analyses_run.append(analysis_name)

    def record_table1(self) -> None:
        """Called by EDA when Table 1 is generated."""
        if self.eda is None:
            self.eda = EDAProvenance(timestamp=datetime.now().isoformat())
        self.eda.table1_generated = True

    def record_feature_engineering(
        self,
        transforms: List[str],
        n_created: int,
        n_before: int,
        n_after: int,
    ) -> None:
        """Called by Feature Engineering when transforms are applied."""
        self.feature_engineering = FeatureEngineeringProvenance(
            transforms_applied=list(transforms),
            n_features_created=n_created,
            n_features_before=n_before,
            n_features_after=n_after,
            timestamp=datetime.now().isoformat(),
        )

    def record_feature_selection(
        self,
        method: str,
        n_before: int,
        n_after: int,
        features_kept: List[str],
        consensus_methods: Optional[List[str]] = None,
    ) -> None:
        """Called by Feature Selection when selection is applied."""
        self.feature_selection = FeatureSelectionProvenance(
            method=method,
            n_features_before=n_before,
            n_features_after=n_after,
            features_kept=list(features_kept),
            consensus_methods=list(consensus_methods or []),
            timestamp=datetime.now().isoformat(),
        )

    def record_split(
        self,
        strategy: str,
        train_n: int,
        val_n: int,
        test_n: int,
        random_seed: int = 42,
        target_transform: str = "none",
        target_trim_enabled: bool = False,
        target_trim_lower: float = 0.0,
        target_trim_upper: float = 1.0,
    ) -> None:
        """Called when data split is performed."""
        total = train_n + val_n + test_n
        self.split = SplitProvenance(
            strategy=strategy,
            train_n=train_n,
            val_n=val_n,
            test_n=test_n,
            train_pct=round(train_n / total * 100) if total else 0,
            val_pct=round(val_n / total * 100) if total else 0,
            test_pct=round(test_n / total * 100) if total else 0,
            random_seed=random_seed,
            target_transform=target_transform,
            target_trim_enabled=target_trim_enabled,
            target_trim_lower=target_trim_lower,
            target_trim_upper=target_trim_upper,
            timestamp=datetime.now().isoformat(),
        )

    def record_preprocessing(
        self,
        configs_by_model: Dict[str, Dict[str, Any]],
        imputation_method: str = "",
    ) -> None:
        """Called by Preprocess when pipelines are built.

        configs_by_model: {model_key: {scaling, encoding, outlier_treatment, ...}}
        """
        per_model = {}
        for model_key, cfg in configs_by_model.items():
            per_model[model_key] = ModelPreprocessingConfig(
                scaling=cfg.get("numeric_scaling", "none"),
                encoding=cfg.get("categorical_encoding", ""),
                outlier_treatment=cfg.get("numeric_outlier_treatment", "none"),
                outlier_params=cfg.get("numeric_outlier_params", {}),
                power_transform=cfg.get("numeric_power_transform", "none"),
                log_transform=cfg.get("numeric_log_transform", False),
                imputation=imputation_method or cfg.get("imputation", ""),
                use_pca=cfg.get("use_pca", False),
                pca_n_components=cfg.get("pca_n_components"),
                pca_mode=cfg.get("pca_mode", ""),
            )

        # Identify shared settings
        shared: Dict[str, str] = {}
        if per_model:
            first = next(iter(per_model.values()))
            if all(c.imputation == first.imputation for c in per_model.values()):
                shared["imputation"] = first.imputation
            if all(c.encoding == first.encoding for c in per_model.values()):
                shared["encoding"] = first.encoding

        self.preprocessing = PreprocessingProvenance(
            shared=shared,
            per_model=per_model,
            models_configured=list(configs_by_model.keys()),
            timestamp=datetime.now().isoformat(),
        )

    def record_training(
        self,
        models_trained: List[str],
        primary_model: str = "",
        selection_criteria: str = "",
        use_cv: bool = False,
        cv_folds: Optional[int] = None,
        use_hyperopt: bool = False,
        class_weight_balanced: bool = False,
        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Called by Train & Compare when training completes."""
        self.training = TrainingProvenance(
            models_trained=list(models_trained),
            hyperparameters=dict(hyperparameters or {}),
            primary_model=primary_model,
            selection_criteria=selection_criteria,
            use_cv=use_cv,
            cv_folds=cv_folds,
            use_hyperopt=use_hyperopt,
            class_weight_balanced=class_weight_balanced,
            metrics_by_model=dict(metrics_by_model or {}),
            timestamp=datetime.now().isoformat(),
        )

    def record_explainability(
        self,
        methods: List[str],
        models: List[str],
    ) -> None:
        """Called by Explainability when analyses are run."""
        self.explainability = ExplainabilityProvenance(
            methods_used=list(methods),
            models_explained=list(models),
            timestamp=datetime.now().isoformat(),
        )

    def record_sensitivity(
        self,
        seed_stability: bool = False,
        seed_stability_cv: Optional[float] = None,
        feature_dropout: bool = False,
    ) -> None:
        """Called by Sensitivity Analysis."""
        self.sensitivity = SensitivityProvenance(
            seed_stability=seed_stability,
            seed_stability_cv=seed_stability_cv,
            feature_dropout=feature_dropout,
            timestamp=datetime.now().isoformat(),
        )

    def record_statistical_test(
        self,
        test_name: str,
        variable: str = "",
        statistic: Optional[float] = None,
        p_value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called by Hypothesis Testing for each test run."""
        if self.statistical_validation is None:
            self.statistical_validation = StatisticalValidationProvenance(
                timestamp=datetime.now().isoformat(),
            )
        self.statistical_validation.tests_run.append({
            "test_name": test_name,
            "variable": variable,
            "statistic": statistic,
            "p_value": p_value,
            **(details or {}),
        })

    # --- Reader methods (for consumers) ---

    def get_completeness(self) -> Dict[str, bool]:
        """Returns which workflow stages have been recorded.

        Useful for TRIPOD compliance checking.
        """
        return {
            "upload": self.upload is not None,
            "eda": self.eda is not None,
            "feature_engineering": self.feature_engineering is not None,
            "feature_selection": self.feature_selection is not None,
            "split": self.split is not None,
            "preprocessing": self.preprocessing is not None,
            "training": self.training is not None,
            "explainability": self.explainability is not None,
            "sensitivity": self.sensitivity is not None,
            "statistical_validation": self.statistical_validation is not None,
        }

    def get_methods_context(self) -> Dict[str, Any]:
        """Returns a flat dict suitable for generate_methods_section().

        This replaces the 100+ scattered session_state reads in Report Export.
        When provenance is populated, methods generation reads from here.
        """
        ctx: Dict[str, Any] = {}

        if self.upload:
            ctx["target_name"] = self.upload.target_col
            ctx["task_type"] = self.upload.task_type
            ctx["feature_cols"] = self.upload.feature_cols
            ctx["n_features_original"] = self.upload.n_features
            ctx["n_total"] = self.upload.n_samples
            ctx["cleaning_actions"] = self.upload.cleaning_actions

        if self.feature_engineering:
            ctx["engineering_transforms"] = self.feature_engineering.transforms_applied
            ctx["n_engineered"] = self.feature_engineering.n_features_created

        if self.feature_selection:
            ctx["fs_method"] = self.feature_selection.method
            ctx["n_features_before_selection"] = self.feature_selection.n_features_before
            ctx["n_features_after_selection"] = self.feature_selection.n_features_after
            ctx["features_kept"] = self.feature_selection.features_kept

        if self.split:
            ctx["split_strategy"] = self.split.strategy
            ctx["n_train"] = self.split.train_n
            ctx["n_val"] = self.split.val_n
            ctx["n_test"] = self.split.test_n
            ctx["random_seed"] = self.split.random_seed
            ctx["target_transform"] = self.split.target_transform

        if self.preprocessing:
            ctx["models_configured"] = self.preprocessing.models_configured
            ctx["preprocessing_per_model"] = {
                mk: {
                    "scaling": cfg.scaling,
                    "encoding": cfg.encoding,
                    "outlier_treatment": cfg.outlier_treatment,
                    "outlier_params": cfg.outlier_params,
                    "power_transform": cfg.power_transform,
                    "log_transform": cfg.log_transform,
                    "imputation": cfg.imputation,
                    "use_pca": cfg.use_pca,
                    "pca_n_components": cfg.pca_n_components,
                    "pca_mode": cfg.pca_mode,
                }
                for mk, cfg in self.preprocessing.per_model.items()
            }
            ctx["preprocessing_differs"] = self.preprocessing.configs_differ()

        if self.training:
            ctx["models_trained"] = self.training.models_trained
            ctx["primary_model"] = self.training.primary_model
            # Provide task-appropriate default if selection_criteria is empty
            selection_criteria = self.training.selection_criteria
            if not selection_criteria and self.upload:
                task_type = self.upload.task_type
                if task_type == "regression":
                    selection_criteria = "validation RMSE"
                elif task_type == "classification":
                    selection_criteria = "validation F1"
            ctx["selection_criteria"] = selection_criteria
            ctx["use_cv"] = self.training.use_cv
            ctx["cv_folds"] = self.training.cv_folds
            ctx["use_hyperopt"] = self.training.use_hyperopt
            ctx["class_weight_balanced"] = self.training.class_weight_balanced
            ctx["hyperparameters"] = self.training.hyperparameters
            ctx["metrics_by_model"] = self.training.metrics_by_model

        if self.explainability:
            ctx["explainability_methods"] = self.explainability.methods_used
            ctx["models_explained"] = self.explainability.models_explained

        if self.sensitivity:
            ctx["seed_stability"] = self.sensitivity.seed_stability
            ctx["feature_dropout"] = self.sensitivity.feature_dropout

        if self.statistical_validation:
            ctx["statistical_tests"] = self.statistical_validation.tests_run

        return ctx

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence or debugging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowProvenance":
        """Reconstruct from serialized dict."""
        prov = cls()
        prov.schema_version = data.get("schema_version", 1)

        if data.get("upload"):
            prov.upload = UploadProvenance(**{
                k: v for k, v in data["upload"].items()
                if k in UploadProvenance.__dataclass_fields__
            })
        if data.get("eda"):
            prov.eda = EDAProvenance(**{
                k: v for k, v in data["eda"].items()
                if k in EDAProvenance.__dataclass_fields__
            })
        if data.get("feature_engineering"):
            prov.feature_engineering = FeatureEngineeringProvenance(**{
                k: v for k, v in data["feature_engineering"].items()
                if k in FeatureEngineeringProvenance.__dataclass_fields__
            })
        if data.get("feature_selection"):
            prov.feature_selection = FeatureSelectionProvenance(**{
                k: v for k, v in data["feature_selection"].items()
                if k in FeatureSelectionProvenance.__dataclass_fields__
            })
        if data.get("split"):
            prov.split = SplitProvenance(**{
                k: v for k, v in data["split"].items()
                if k in SplitProvenance.__dataclass_fields__
            })
        if data.get("preprocessing"):
            pp = data["preprocessing"]
            per_model = {}
            for mk, cfg_dict in pp.get("per_model", {}).items():
                per_model[mk] = ModelPreprocessingConfig(**{
                    k: v for k, v in cfg_dict.items()
                    if k in ModelPreprocessingConfig.__dataclass_fields__
                })
            prov.preprocessing = PreprocessingProvenance(
                shared=pp.get("shared", {}),
                per_model=per_model,
                models_configured=pp.get("models_configured", []),
                timestamp=pp.get("timestamp", ""),
            )
        if data.get("training"):
            prov.training = TrainingProvenance(**{
                k: v for k, v in data["training"].items()
                if k in TrainingProvenance.__dataclass_fields__
            })
        if data.get("explainability"):
            prov.explainability = ExplainabilityProvenance(**{
                k: v for k, v in data["explainability"].items()
                if k in ExplainabilityProvenance.__dataclass_fields__
            })
        if data.get("sensitivity"):
            prov.sensitivity = SensitivityProvenance(**{
                k: v for k, v in data["sensitivity"].items()
                if k in SensitivityProvenance.__dataclass_fields__
            })
        if data.get("statistical_validation"):
            prov.statistical_validation = StatisticalValidationProvenance(**{
                k: v for k, v in data["statistical_validation"].items()
                if k in StatisticalValidationProvenance.__dataclass_fields__
            })

        return prov


# ---------------------------------------------------------------------------
# Session-state accessor
# ---------------------------------------------------------------------------

def get_provenance() -> WorkflowProvenance:
    """Get or create WorkflowProvenance from Streamlit session state."""
    try:
        import streamlit as st
        if "workflow_provenance" not in st.session_state:
            st.session_state.workflow_provenance = WorkflowProvenance()
        return st.session_state.workflow_provenance
    except ImportError:
        # Not in Streamlit context — return a detached instance
        return WorkflowProvenance()
