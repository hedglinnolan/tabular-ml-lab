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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

    # A cohort run restricts EVERY downstream page to one group. It is the
    # single fact that makes the reported effect interpretable — the whole
    # point of running one group at a time is that the relationship may differ
    # between groups — and without it here, every exported artifact reports
    # the group's N as the study's and no reader can tell.
    cohort_column: str = ""
    cohort_value: str = ""
    cohort_n: int = 0
    study_n: int = 0
    # The OTHER groups this same analysis has already been run in, and how
    # many the split has. The app shows "report all k, not the one that
    # worked" on screen when two runs sit side by side; that caveat is a fact
    # about how many fits were made, and it has to leave with the export.
    cohort_runs_completed: List[str] = field(default_factory=list)
    cohort_runs_planned: int = 0
    # Predictors that are constant inside this group. They stay in the
    # predictor list and contribute nothing; a reader comparing two group
    # models needs to know the two did not see the same predictors.
    cohort_constant_features: List[str] = field(default_factory=list)

    @property
    def is_cohort_restricted(self) -> bool:
        return bool(self.cohort_column)

    def restriction_sentence(self) -> str:
        """One sentence a manuscript can use verbatim, or '' when unrestricted.

        Grows by one sentence for each fact that changes how the result should
        be read — other groups already analyzed, predictors flat in this group
        — and by nothing otherwise. A caveat about a comparison that was never
        made is the same class of error as a missing one.
        """
        if not self.is_cohort_restricted:
            return ""
        of_study = (f" of {self.study_n:,} in the full study"
                    if self.study_n and self.study_n > self.cohort_n else "")
        text = (f"This analysis was restricted to participants with "
                f"{self.cohort_column} = {self.cohort_value} "
                f"(n={self.cohort_n:,}{of_study}); the model was fitted and "
                f"evaluated in that group only, and results should not be read "
                f"as describing the whole study population.")
        others = [g for g in self.cohort_runs_completed if g != self.cohort_value]
        if others:
            k = len(others) + 1
            text += (f" The same analysis was also run separately in "
                     f"{self.cohort_column} = {', '.join(others)}; all {k} "
                     f"group-specific results should be reported together, "
                     f"and the difference between groups was not tested "
                     f"directly."
                     # WHERE those other results are. The sentence has always
                     # said they must be reported together; a reader holding
                     # only the manuscript had no way to find them, and this
                     # file is single-cohort by design. Naming the folder is
                     # what makes "reported together" an instruction rather
                     # than a hope.
                     f" Full results for the other group"
                     f"{'' if len(others) == 1 else 's'} are in `cohorts/` in "
                     f"the accompanying analysis bundle, with a comparison "
                     f"table in `cohort_comparison.csv`.")
        flat = list(self.cohort_constant_features)
        if flat:
            shown = ", ".join(flat[:6]) + (f" and {len(flat) - 6} more" if len(flat) > 6 else "")
            text += (f" Within this group {len(flat)} predictor"
                     f"{'' if len(flat) == 1 else 's'} ({shown}) "
                     f"{'was' if len(flat) == 1 else 'were'} constant and "
                     f"contributed nothing to the model.")
        return text


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
    #: The columns that were SCREENED, including every one selection dropped.
    #: `AUDIT-023` — §A5.4 sizes for the screened set, and applying a selection
    #: overwrites `data_config.feature_cols` in place, so without this list the
    #: number of candidate predictors is unrecoverable the moment the button is
    #: pressed. `n_features_before` carried the count already; the names are
    #: here because the EDA insight and the manuscript both name columns.
    candidates_screened: List[str] = field(default_factory=list)


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
    # "the held-out RMSE", "the held-out F1", etc. NEVER "validation <metric>":
    # nothing stores a per-model validation score, so that word named a split the
    # ranking never saw (`AUDIT-030`). `ml/holdout_selection.criterion_phrase`
    # composes it.
    selection_criteria: str = ""
    # Whether the primary model was chosen by comparing the trained models'
    # HELD-OUT scores — the thing that makes the reported number optimistic.
    # Recorded rather than inferred from the criterion string, and defaulting to
    # False, so a caller who selected some other way never has the caveat
    # attached to their manuscript on the strength of a substring match.
    selected_on_holdout: bool = False
    use_cv: bool = False
    cv_folds: Optional[int] = None
    # WHICH models a fold loop actually scored. `AUDIT-026`.
    #
    # `use_cv` above is the checkbox and was the only thing the Methods section
    # read, so a run that cross-validated nothing still asserted k-fold internal
    # validation. pages/06 skips CV for the neural network (:1455) and swallows
    # a CV exception (:1489), so "requested" and "ran" come apart routinely.
    #
    # `None` rather than `[]` as the default, and the distinction is load-
    # bearing: `None` is a record written before this field existed and the
    # answer is unknown; `[]` is a caller who looked and found nothing
    # cross-validated. Reporting the first as the second would assert a fact
    # about a run nobody recorded.
    cv_models_run: Optional[List[str]] = None
    # WHICH of those fold loops did not finish, `{model_key: [n_scored,
    # n_folds]}`, and empty when every loop that ran completed.
    #
    # `cv_models_run` is built from the truthiness of `cv_results`, and a fold
    # loop that lost a fold still produces a truthy dict — so on its own it
    # reports a complete k-fold design for a run that did not perform one.
    # ml/publication.py derives the third state in place from the per-model
    # results, but that generator is the FALLBACK; page 10 reaches
    # ml/narrative_engine.py first, and the narrative has only this record to
    # read. So the incompleteness travels with the provenance for the same
    # reason `hyperopt_trials` does.
    #
    # A list rather than a tuple because `asdict` -> JSON -> `from_dict`
    # round-trips one and silently changes the other into the first.
    cv_incomplete: Dict[str, List[int]] = field(default_factory=dict)
    use_hyperopt: bool = False
    # HOW MANY Optuna trials per tunable model, when the recorder knew.
    #
    # `use_hyperopt` above is the flag, and the flag alone was all the Methods
    # narrative ever had — so it could say a search happened but never what it
    # cost, and ml/publication.py filled the gap with the literal "30 trials
    # per model". That literal was true only while pages/06 hardcoded 30. The
    # count is a user control now, so it is recorded rather than assumed.
    #
    # `None` rather than `0`, and the distinction is the same one
    # `cv_models_run` draws: `None` is a record written before this field
    # existed, `0`/absence of the flag is a run that tuned nothing.
    hyperopt_trials: Optional[int] = None
    class_weight_balanced: bool = False
    metrics_by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    nn_config_source: str = ""  # "recommended", "custom", "recommended+modified"
    nn_config_reasoning: Dict[str, str] = field(default_factory=dict)
    nn_config_modifications: Dict[str, Any] = field(default_factory=dict)
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
class ExternalValidationProvenance:
    """Recorded when trained models are scored on an independent cohort.

    The page used to display external metrics and drop them, so the one result
    a reviewer weighs most heavily never reached the manuscript
    (`ml/publication.py`'s `external_validation` flag was never set by anything).
    The numbers themselves live in `session_state['external_validation_results']`;
    what belongs in the record is the account a Methods section has to make —
    which file, how many rows, which models, and what the import review had to
    repair before the file was believed.
    """
    dataset_name: str = ""
    n_rows: int = 0
    n_features: int = 0
    models_validated: List[str] = field(default_factory=list)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_bootstrap: int = 0
    # What the front door found and what the user applied — an external cohort
    # that needed repairs is a cohort the Methods section must say was repaired.
    import_repairs: List[str] = field(default_factory=list)
    structural_findings: str = ""
    records_key: str = ""
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
class CoachProvenance:
    """Model-shortlist rationale: what the coach advised and why.

    Model-selection rationale is a TRIPOD reporting item; recording it here
    lets the Methods draft cite the actual reasoning instead of the author
    reconstructing it."""
    headline: str = ""
    picks: List[Dict[str, Any]] = field(default_factory=list)
    probe_summary: str = ""
    timestamp: str = ""


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
    external_validation: Optional[ExternalValidationProvenance] = None
    sensitivity: Optional[SensitivityProvenance] = None
    statistical_validation: Optional[StatisticalValidationProvenance] = None
    coach: Optional[CoachProvenance] = None

    # Cleaning recorded before a configuration was ever saved. Import Doctor
    # repairs happen at the moment a file is committed, which is always before
    # record_upload exists to hold them; without this they were dropped and the
    # Methods section said nothing about a column whose sentinel codes had been
    # recoded before anything else saw the data.
    pending_cleaning_actions: List[Dict[str, Any]] = field(default_factory=list)

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
            timestamp=datetime.now(timezone.utc).isoformat(),
            # Repairs applied at import happen before this record exists. A
            # fresh UploadProvenance with cleaning_actions=[] used to erase them.
            cleaning_actions=list(self.pending_cleaning_actions),
        )
        self.pending_cleaning_actions = []
        self.record_cohort_restriction()
        # Reset downstream sections (config changed)
        self.feature_engineering = None
        self.feature_selection = None
        self.preprocessing = None
        self.training = None

    def record_cohort_restriction(self) -> None:
        """Copy the active cohort run onto the upload record.

        Called on every config save and whenever a run starts or clears, so the
        provenance never disagrees with what get_data() is actually returning.
        """
        if self.upload is None:
            return
        try:
            from utils.cohorts import active_cohort
            run = active_cohort()
        except Exception:
            run = None
        if run is None:
            self.upload.cohort_column = ""
            self.upload.cohort_value = ""
            self.upload.cohort_n = 0
            self.upload.study_n = 0
            self.upload.cohort_runs_completed = []
            self.upload.cohort_runs_planned = 0
            self.upload.cohort_constant_features = []
        else:
            self.upload.cohort_column = str(run.get("column", ""))
            self.upload.cohort_value = str(run.get("label", ""))
            self.upload.cohort_n = int(run.get("n_rows", 0))
            self.upload.study_n = int(run.get("n_total", 0))
            self.upload.cohort_runs_planned = int(run.get("of", 0) or 0)
            self.upload.cohort_constant_features = [
                str(c) for c in (run.get("dropped_features") or [])]
            try:
                from utils.cohorts import completed_runs
                done = completed_runs(str(run.get("column", "")))
            except Exception:
                done = []
            self.upload.cohort_runs_completed = [
                str(r.label) for r in done if r.label != run.get("label")]

    def record_cleaning(self, action: str, rows_before: int, rows_after: int,
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Called by Upload & Audit for each data cleaning action.

        Not every cleaning action removes rows. An Import Doctor repair recodes
        values in place and passes 0/0, which used to overwrite the recorded
        study N with zero — and, because repairs happen BEFORE a configuration
        is ever saved, used to be dropped entirely and then erased by the
        record_upload that followed. Both are handled here: the row count moves
        only when rows really moved, and a repair recorded before the upload
        record exists waits for it.
        """
        entry = {
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.upload is None:
            self.pending_cleaning_actions.append(entry)
            return
        self.upload.cleaning_actions.append(entry)
        if rows_after > 0:
            self.upload.n_samples = rows_after

    def record_eda_analysis(self, analysis_name: str) -> None:
        """Called by EDA page for each analysis run."""
        if self.eda is None:
            self.eda = EDAProvenance(timestamp=datetime.now(timezone.utc).isoformat())
        if analysis_name not in self.eda.analyses_run:
            self.eda.analyses_run.append(analysis_name)

    def record_table1(self) -> None:
        """Called by EDA when Table 1 is generated."""
        if self.eda is None:
            self.eda = EDAProvenance(timestamp=datetime.now(timezone.utc).isoformat())
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_feature_selection(
        self,
        method: str,
        n_before: int,
        n_after: int,
        features_kept: List[str],
        consensus_methods: Optional[List[str]] = None,
        candidates_screened: Optional[List[str]] = None,
    ) -> None:
        """Called by Feature Selection when selection is applied.

        **The screened set accumulates and never shrinks** (`AUDIT-023`).
        Screening 40 and keeping 8, then re-opening the page and keeping 5, is
        still a study that must be sized for 40 — §A5.4's ⚠ clause is about the
        degrees of freedom a data-driven choice consumes, and consuming them
        twice does not give any back. So this merges with whatever was recorded
        before rather than replacing it, and `n_features_before` is the size of
        the merged set rather than of this press alone.
        """
        prior = self.feature_selection
        merged: List[str] = []
        seen = set()
        for column in (list(getattr(prior, "candidates_screened", None) or [])
                       + list(candidates_screened or [])
                       + list(features_kept or [])):
            name = str(column)
            if name not in seen:
                merged.append(name)
                seen.add(name)
        n_before = max(int(n_before or 0),
                       int(getattr(prior, "n_features_before", 0) or 0),
                       len(merged))
        self.feature_selection = FeatureSelectionProvenance(
            method=method,
            n_features_before=n_before,
            n_features_after=n_after,
            features_kept=list(features_kept),
            consensus_methods=list(consensus_methods or []),
            timestamp=datetime.now(timezone.utc).isoformat(),
            candidates_screened=merged,
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
            timestamp=datetime.now(timezone.utc).isoformat(),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_training(
        self,
        models_trained: List[str],
        primary_model: str = "",
        selection_criteria: str = "",
        selected_on_holdout: bool = False,
        use_cv: bool = False,
        cv_folds: Optional[int] = None,
        cv_models_run: Optional[List[str]] = None,
        cv_incomplete: Optional[Dict[str, Sequence[int]]] = None,
        use_hyperopt: bool = False,
        hyperopt_trials: Optional[int] = None,
        class_weight_balanced: bool = False,
        hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
        nn_config_source: str = "",
        nn_config_reasoning: Optional[Dict[str, str]] = None,
        nn_config_modifications: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called by Train & Compare when training completes."""
        self.training = TrainingProvenance(
            models_trained=list(models_trained),
            hyperparameters=dict(hyperparameters or {}),
            primary_model=primary_model,
            selection_criteria=selection_criteria,
            selected_on_holdout=selected_on_holdout,
            use_cv=use_cv,
            cv_folds=cv_folds,
            cv_models_run=(list(cv_models_run)
                           if cv_models_run is not None else None),
            cv_incomplete={str(k): [int(n) for n in v]
                           for k, v in (cv_incomplete or {}).items()},
            use_hyperopt=use_hyperopt,
            hyperopt_trials=hyperopt_trials,
            class_weight_balanced=class_weight_balanced,
            metrics_by_model=dict(metrics_by_model or {}),
            nn_config_source=nn_config_source,
            nn_config_reasoning=dict(nn_config_reasoning or {}),
            nn_config_modifications=dict(nn_config_modifications or {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_external_validation(
        self,
        dataset_name: str,
        n_rows: int,
        n_features: int,
        models_validated: List[str],
        metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        n_bootstrap: int = 0,
        import_repairs: Optional[List[str]] = None,
        structural_findings: str = "",
        records_key: str = "",
    ) -> None:
        """Called by Explainability when an external cohort has been scored."""
        self.external_validation = ExternalValidationProvenance(
            dataset_name=str(dataset_name or ""),
            n_rows=int(n_rows),
            n_features=int(n_features),
            models_validated=list(models_validated),
            metrics=dict(metrics or {}),
            n_bootstrap=int(n_bootstrap),
            import_repairs=list(import_repairs or []),
            structural_findings=str(structural_findings or ""),
            records_key=str(records_key or ""),
            timestamp=datetime.now(timezone.utc).isoformat(),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def record_statistical_test(
        self,
        test_name: str,
        variable: str = "",
        statistic: Optional[float] = None,
        p_value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called by Hypothesis Testing for each test run.

        A test is recorded UNCORRECTED, because a multiple-comparison
        correction is a property of the family and the family is not complete
        until the last test has been run. `apply_multiplicity_correction` is
        the act that closes it.
        """
        if self.statistical_validation is None:
            self.statistical_validation = StatisticalValidationProvenance(
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        self.statistical_validation.tests_run.append({
            "test_name": test_name,
            "variable": variable,
            "statistic": statistic,
            "p_value": p_value,
            **(details or {}),
        })

    def apply_multiplicity_correction(
        self,
        method: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Correct the recorded family of tests, and record that it happened.

        `AUDIT-001`. The manuscript used to report how many tests reached a raw
        p < 0.05, which on a wide table is the count the literature names an
        anti-pattern. The correction itself is `statsmodels.multipletests`
        through `ml.multiplicity` — the same call `ml/feature_selection.py`
        already makes, not a second one.

        **It is a recorded ACT rather than something the draft does on the
        author's way past.** Benjamini-Hochberg over *"every test run in this
        session"* is a decision about what the family is, and the app does not
        get to make it silently. Until this is called, the draft says the tests
        are uncorrected and declines to count them.
        """
        from ml import multiplicity

        if self.statistical_validation is None:
            return {"tests": [], "n_tests": 0, "n_significant": 0,
                    "method": method, "alpha": float(alpha)}
        summary = multiplicity.adjust(
            self.statistical_validation.tests_run, method=method, alpha=alpha)
        self.statistical_validation.tests_run = summary["tests"]
        return summary

    # --- Reader methods (for consumers) ---

    def record_coach(self, headline: str, picks: List[Dict[str, Any]],
                     probe_summary: str = "") -> None:
        """Called by the Preprocess page when the Model Coach renders."""
        self.coach = CoachProvenance(
            headline=headline or "",
            picks=list(picks or []),
            probe_summary=probe_summary or "",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    #: Sections that are provenance but not reporting stages: the coach records
    #: advice given, and advice is not a TRIPOD-checkable step of the study.
    COMPLETENESS_EXCLUDED = ("coach",)

    def get_completeness(self) -> Dict[str, bool]:
        """Returns which workflow stages have been recorded.

        Useful for TRIPOD compliance checking. Derived from the record's own
        schema (like the reset's section list), so a new section cannot be
        forgotten here; exclusions are named, not implicit.
        """
        return {
            name: getattr(self, name) is not None
            for name in section_names()
            if name not in self.COMPLETENESS_EXCLUDED
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
            ctx["n_upload_total"] = self.upload.n_samples
            ctx["n_total"] = self.upload.n_samples
            ctx["cleaning_actions"] = self.upload.cleaning_actions

        if self.feature_engineering:
            ctx["engineering_transforms"] = self.feature_engineering.transforms_applied
            ctx["n_engineered"] = self.feature_engineering.n_features_created

        if self.feature_selection:
            ctx["fs_method"] = self.feature_selection.method
            ctx["fs_consensus_methods"] = self.feature_selection.consensus_methods
            ctx["n_features_before_selection"] = self.feature_selection.n_features_before
            ctx["n_features_after_selection"] = self.feature_selection.n_features_after
            ctx["features_kept"] = self.feature_selection.features_kept

        if self.split:
            analysis_total = self.split.train_n + self.split.val_n + self.split.test_n
            ctx["split_strategy"] = self.split.strategy
            ctx["n_train"] = self.split.train_n
            ctx["n_val"] = self.split.val_n
            ctx["n_test"] = self.split.test_n
            ctx["n_analysis_total"] = analysis_total
            if analysis_total:
                ctx["n_total"] = analysis_total
                if self.upload and self.upload.n_samples > analysis_total:
                    ctx["n_rows_removed_before_split"] = self.upload.n_samples - analysis_total
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
            # Provide task-appropriate default if selection_criteria is empty.
            #
            # `AUDIT-030`, and THESE TWO LINES ARE THE PART THE ROW MISSED. It
            # named `pages/06:1580` and `narrative_engine:587`; the same false
            # word was also the DEFAULT here, so a record that never got a
            # criterion written to it acquired the claim on the way out. A sweep
            # that stops at the sites the finding cited stops one surface early.
            from ml.holdout_selection import criterion_phrase

            selection_criteria = self.training.selection_criteria
            if not selection_criteria and self.upload:
                task_type = self.upload.task_type
                if task_type == "regression":
                    selection_criteria = criterion_phrase("RMSE")
                elif task_type == "classification":
                    selection_criteria = criterion_phrase("F1")
            ctx["selection_criteria"] = selection_criteria
            ctx["selected_on_holdout"] = self.training.selected_on_holdout
            ctx["use_cv"] = self.training.use_cv
            ctx["cv_folds"] = self.training.cv_folds
            # `AUDIT-026`. Carried onto the methods context so the narrative
            # reads what RAN rather than what was ticked. Stays `None` when the
            # record does not know.
            ctx["cv_models_run"] = (list(self.training.cv_models_run)
                                    if self.training.cv_models_run is not None
                                    else None)
            # Carried so the narrative can say a fold loop did not finish. Without
            # it the only signal is `cv_models_run`, which lists a model whose CV
            # lost a fold exactly as it lists one whose CV completed.
            ctx["cv_incomplete"] = dict(self.training.cv_incomplete or {})
            ctx["use_hyperopt"] = self.training.use_hyperopt
            # Carried so the narrative can name the search budget the run used
            # instead of describing the search without one. `None` when the
            # record does not know, and the sentence then omits the number.
            ctx["hyperopt_trials"] = self.training.hyperopt_trials
            ctx["class_weight_balanced"] = self.training.class_weight_balanced
            ctx["hyperparameters"] = self.training.hyperparameters
            ctx["metrics_by_model"] = self.training.metrics_by_model
            ctx["nn_config_source"] = self.training.nn_config_source
            ctx["nn_config_reasoning"] = self.training.nn_config_reasoning
            ctx["nn_config_modifications"] = self.training.nn_config_modifications

        if self.explainability:
            ctx["explainability_methods"] = self.explainability.methods_used
            ctx["models_explained"] = self.explainability.models_explained

        if self.external_validation:
            ev = self.external_validation
            ctx["external_validation"] = True
            ctx["external_validation_dataset"] = ev.dataset_name
            ctx["external_validation_n"] = ev.n_rows
            ctx["external_validation_models"] = list(ev.models_validated)
            ctx["external_validation_metrics"] = dict(ev.metrics)
            ctx["external_validation_bootstrap"] = ev.n_bootstrap
            ctx["external_validation_repairs"] = list(ev.import_repairs)

        if self.sensitivity:
            ctx["seed_stability"] = self.sensitivity.seed_stability
            ctx["feature_dropout"] = self.sensitivity.feature_dropout

        if self.coach:
            ctx["coach_headline"] = self.coach.headline
            ctx["coach_picks"] = self.coach.picks
            ctx["coach_probe_summary"] = self.coach.probe_summary

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
        if data.get("external_validation"):
            prov.external_validation = ExternalValidationProvenance(**{
                k: v for k, v in data["external_validation"].items()
                if k in ExternalValidationProvenance.__dataclass_fields__
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
        if data.get("coach"):
            prov.coach = CoachProvenance(**{
                k: v for k, v in data["coach"].items()
                if k in CoachProvenance.__dataclass_fields__
            })

        return prov


# ---------------------------------------------------------------------------
# Section registry — derived from the dataclass, never typed by hand
# ---------------------------------------------------------------------------

# Sections a downstream reset must KEEP. `upload` describes the data
# configuration itself, which the reset preserves by contract; every other
# section describes work computed FROM it and is therefore stale. Membership
# here is the only decision the resetter makes — anything not named is cleared,
# so a section added to WorkflowProvenance cannot be forgotten by it.
# `CONTRACT-034`/`STATE-047`: sensitivity and statistical_validation were absent
# from a hand-typed list in utils/session_state.py, so a Methods draft asserted
# hypothesis tests and seed-stability numbers whose results the same reset had
# just deleted, and get_completeness() reported both stages as done.
RESET_PRESERVED_SECTIONS: Tuple[str, ...] = ("upload",)

# Sections whose clearing follows the artifact they describe: the resetter can
# be asked to keep the engineered frame or the feature selection, and the record
# of that step must survive exactly when the step's output does.
_FLAGGED_SECTIONS: Dict[str, str] = {
    "feature_engineering": "clear_feature_engineering",
    "feature_selection": "clear_feature_selection",
}

_SECTION_NAMES_CACHE: Optional[Tuple[str, ...]] = None


def section_names() -> Tuple[str, ...]:
    """Every optional section field declared on WorkflowProvenance.

    Structural, not a list: a field is a section iff it is declared
    `Optional[<some *Provenance dataclass>]`. The two non-section fields
    (pending_cleaning_actions, schema_version) fail that test by construction.
    """
    global _SECTION_NAMES_CACHE
    if _SECTION_NAMES_CACHE is not None:
        return _SECTION_NAMES_CACHE

    import dataclasses as _dc
    import typing as _t

    try:
        hints = _t.get_type_hints(WorkflowProvenance)
    except Exception:
        hints = {}

    names: List[str] = []
    for name, fld in WorkflowProvenance.__dataclass_fields__.items():
        hint = hints.get(name)
        if hint is not None:
            args = [a for a in _t.get_args(hint) if a is not type(None)]
            if (_t.get_origin(hint) is _t.Union and len(args) == 1
                    and _dc.is_dataclass(args[0])):
                names.append(name)
            continue
        # No resolvable hint (a stringified annotation we could not evaluate):
        # fall back to the declaration's own text rather than dropping the
        # field, because a dropped field is a section that never gets cleared.
        text = fld.type if isinstance(fld.type, str) else str(fld.type)
        if text.startswith("Optional[") and text.endswith("Provenance]"):
            names.append(name)

    _SECTION_NAMES_CACHE = tuple(names)
    return _SECTION_NAMES_CACHE


def downstream_sections(clear_feature_engineering: bool = True,
                        clear_feature_selection: bool = True) -> Tuple[str, ...]:
    """Sections a downstream reset must null, given what else it is clearing."""
    flags = {"clear_feature_engineering": clear_feature_engineering,
             "clear_feature_selection": clear_feature_selection}
    out: List[str] = []
    for name in section_names():
        if name in RESET_PRESERVED_SECTIONS:
            continue
        flag = _FLAGGED_SECTIONS.get(name)
        if flag is not None and not flags[flag]:
            continue
        out.append(name)
    return tuple(out)


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


def cohort_restriction_sentence() -> str:
    """The sentence every exported artifact must carry, or '' when unrestricted.

    Read from provenance rather than from session_state directly, so an export
    can only state a restriction the pipeline actually recorded — which is the
    promise the manuscript draft header makes.
    """
    try:
        up = getattr(get_provenance(), "upload", None)
        return up.restriction_sentence() if up is not None else ""
    except Exception:
        return ""
