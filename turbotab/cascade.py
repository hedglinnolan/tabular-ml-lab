"""
turbotab.cascade — invalidation as a declared graph rather than a written-out list.

Today the same cascade exists three times: the production
`utils.session_state.reset_downstream_results`, a hand-rolled copy inline in
`pages/03_Feature_Engineering.py:1262-1282`, and re-implementations inside
tests that test themselves (`TRANSITION_PLAN.md` §03). They already disagree —
page 03 clears 19 keys and misses eleven, plus the ledger rollback and the
provenance clearing.

This module states the dependency graph once. Which keys go stale follows from
which stage changed, so a new result key is registered on its stage and every
call site gets it. `ARCHITECTURE.md` §02 is explicit that **invalidation is a
cascade, not a reset**: `reset_downstream_results(clear_feature_engineering=False)`
is a real call, so the graph has to express *partial* invalidation rather than
"clear everything below".

Headless: no Streamlit, no session state, no I/O. It computes a set of key
names; applying them is the caller's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class Stage:
    """One step of the analysis, and the results it owns.

    `produces` is the authority: a result key belongs to exactly one stage, and
    registering it here is what makes every call site invalidate it.
    """
    name: str
    produces: FrozenSet[str]
    depends_on: FrozenSet[str] = frozenset()
    # Sections of the provenance record this stage wrote, cleared with it —
    # a Methods section describing work that no longer exists is the app
    # asserting something false.
    provenance_sections: FrozenSet[str] = frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# The graph
#
# Read off the production cascade and page 03's copy, then verified against both
# by `tests/integration/test_cascade_dag_equivalence.py`. The stage a key sits
# on is a claim about what it is derived from.
# ─────────────────────────────────────────────────────────────────────────────

STAGES: Sequence[Stage] = (
    Stage(
        name="data",
        produces=frozenset(),          # the root: the table itself, not a result
    ),
    Stage(
        name="feature_engineering",
        depends_on=frozenset({"data"}),
        produces=frozenset({
            "df_engineered", "engineering_log",
        }),
        provenance_sections=frozenset({"feature_engineering"}),
    ),
    Stage(
        name="feature_selection",
        depends_on=frozenset({"feature_engineering"}),
        produces=frozenset({
            "feature_selection_results", "consensus_features",
        }),
        provenance_sections=frozenset({"feature_selection"}),
    ),
    Stage(
        name="preprocessing",
        depends_on=frozenset({"feature_selection"}),
        produces=frozenset({
            "preprocessing_pipeline", "preprocessing_config",
            "preprocessing_pipelines_by_model", "preprocessing_config_by_model",
            # The list of models that HAVE pipelines has to go with the
            # pipelines, or page 06 badges a model "tuned for this model" with
            # nothing left that tuned it.
            "preprocess_built_model_keys",
            # What the recipe DID to this data — the counts page 10 prints
            # beside the Methods' missing-data sentence. It describes one
            # pipeline built on one set of rows.
            "preprocessing_summary",
            # WHO the results are about (`STATE-037`). The row filter is a
            # preprocessing control (`pages/05_Preprocess.py:1098`), and
            # `get_data()` masks every page's frame by it, so a filter left
            # over from a superseded config keeps shrinking the study.
            "filtered_data",
        }),
        provenance_sections=frozenset({"preprocessing"}),
    ),
    Stage(
        name="split",
        depends_on=frozenset({"preprocessing"}),
        produces=frozenset({
            "X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
            "feature_names", "feature_names_by_model",
            "train_indices", "val_indices", "test_indices",
            "train_row_labels", "val_row_labels", "test_row_labels",
            "split_config", "target_transformer", "target_label_encoder",
            "y_train_original", "y_val_original", "y_test_original",
            "cv_strategy", "cv_groups_train",
            # The realized target trim of one split (`CONTRACT-021`) — the
            # thresholds actually applied, not the ones configured. Stale the
            # moment the split is redrawn.
            "split_trim_record",
        }),
        provenance_sections=frozenset({"split"}),
    ),
    Stage(
        name="training",
        depends_on=frozenset({"split"}),
        produces=frozenset({
            "trained_models", "model_results", "fitted_estimators",
            "fitted_preprocessing_pipelines", "cv_results",
        }),
        provenance_sections=frozenset({"training"}),
    ),
    Stage(
        name="explainability",
        depends_on=frozenset({"training"}),
        produces=frozenset({
            "permutation_importance", "partial_dependence",
            "explainability_robustness", "shap_results", "shap_matplotlib_figs",
            # Written by pages/07 beside `partial_dependence` and drawn by
            # pages/10. Kept apart from it once and survived a reset alone.
            "pdp_results",
            # External-cohort metrics, also pages/07. `ml/publication.py`
            # states that external validation was performed on the strength
            # of them, so they must not outlive the models they scored.
            "external_validation_results",
        }),
        provenance_sections=frozenset({"explainability"}),
    ),
    Stage(
        name="evaluation",
        depends_on=frozenset({"training"}),
        produces=frozenset({
            "bootstrap_results", "baseline_results", "calibration_results",
            "sensitivity_seed_results",
            # pages/08's feature-dropout sensitivity. `ml/publication.py`
            # asserts a sensitivity analysis was performed from these.
            "sensitivity_dropout_results", "sensitivity_dropout_baseline",
            # Agreement between predicted and observed values, per model pair.
            "bland_altman_results",
        }),
    ),
    Stage(
        # Descriptive work over the data values. Not downstream of the split —
        # it depends on the data alone — but stale the moment the data changes.
        name="analysis",
        depends_on=frozenset({"data"}),
        produces=frozenset({
            "eda_results", "eda_insights", "dataset_profile",
            # WHICH ROWS `dataset_profile` describes. A scope note that
            # outlives its profile labels the next one's numbers.
            "dataset_profile_scope",
            "hypothesis_test_results", "table1_df", "table1_metadata",
            "custom_table1_tests", "table1_custom_test_footnotes",
        }),
        provenance_sections=frozenset({"eda"}),
    ),
    Stage(
        name="coach",
        depends_on=frozenset({"data"}),
        produces=frozenset({
            # A probe verdict describes the data it was measured on. It must
            # never survive a data change and keep steering picks for a dataset
            # it never saw.
            "coach_probe_result", "_coach_applied",
        }),
        provenance_sections=frozenset({"coach"}),
    ),
    Stage(
        name="report",
        depends_on=frozenset({"training", "explainability", "evaluation"}),
        produces=frozenset({
            "report_data", "methods_section", "flow_diagram", "tripod_tracker",
            "latex_report", "report_best_model", "report_model_selection",
            "report_explain_selection", "report_include_results",
            "report_include_llm", "manuscript_context",
            # A stale `manuscript_export_context` WINS over rebuilding —
            # pages/10 only rebuilds when it is None — so it is the most
            # dangerous survivor here, not the least. The compiled PDF and
            # the manuscript's Table 1 are artifacts of one dead model.
            "manuscript_export_context", "compiled_pdf",
            "manuscript_table1_df", "manuscript_table1_metadata",
        }),
    ),
)

_BY_NAME: Dict[str, Stage] = {s.name: s for s in STAGES}


def stage(name: str) -> Stage:
    if name not in _BY_NAME:
        raise KeyError(f"No stage {name!r}. Known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def all_result_keys() -> Set[str]:
    return {k for s in STAGES for k in s.produces}


def descendants(root: str, *, inclusive: bool = True) -> List[str]:
    """Every stage reachable downstream of `root`, in graph order."""
    stage(root)
    out: List[str] = []
    frontier = {root}
    for s in STAGES:                      # STAGES is declared in dependency order
        if s.name == root:
            if inclusive:
                out.append(s.name)
            continue
        if s.depends_on & frontier:
            frontier.add(s.name)
            out.append(s.name)
    return out


def keys_to_clear(
    changed: str = "data",
    keep: Iterable[str] = (),
) -> Set[str]:
    """Which result keys go stale when `changed` changes.

    `keep` names stages whose results survive — this is what makes partial
    invalidation expressible. `reset_downstream_results(clear_feature_selection=False)`
    is `keep={"feature_selection"}`: everything built on the old feature set is
    stale, but the selection just made, and its record, must survive.

    Keeping a stage does **not** keep its descendants. Feature Selection applying
    a new selection invalidates the pipelines and models built on the old one;
    that is the whole point of the call.
    """
    keep = set(keep)
    unknown = keep - set(_BY_NAME)
    if unknown:
        raise KeyError(f"Unknown stage(s) to keep: {sorted(unknown)}")
    return {k for name in descendants(changed, inclusive=False)
            if name not in keep
            for k in stage(name).produces}


def provenance_sections_to_clear(changed: str = "data", keep: Iterable[str] = ()) -> Set[str]:
    keep = set(keep)
    return {sec for name in descendants(changed, inclusive=False)
            if name not in keep
            for sec in stage(name).provenance_sections}


# ─────────────────────────────────────────────────────────────────────────────
# The production call, expressed against the graph
# ─────────────────────────────────────────────────────────────────────────────

def keys_for_reset_downstream_results(
    clear_feature_engineering: bool = True,
    clear_feature_selection: bool = True,
) -> Set[str]:
    """The graph's answer to `utils.session_state.reset_downstream_results`.

    Pinned against the real function, flag combination by flag combination, in
    `tests/integration/test_cascade_dag_equivalence.py`. The two must agree
    before the DAG can replace anything.
    """
    keep: Set[str] = set()
    if not clear_feature_engineering:
        keep.add("feature_engineering")
    if not clear_feature_selection:
        keep.add("feature_selection")
    return keys_to_clear("data", keep=keep)


def missing_from(other_keys: Iterable[str],
                 changed: str = "data",
                 keep: Iterable[str] = ()) -> Set[str]:
    """What a hand-rolled cascade forgot, relative to the graph.

    Written for `pages/03`, whose inline copy clears nineteen keys. Naming the
    gap is more useful than asserting a number, because the answer is the
    to-do list for reconciling it.
    """
    return keys_to_clear(changed, keep=keep) - set(other_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Branch keys: which of the above belong to ONE COHORT rather than to the study
#
# A cohort run is "same question, different people" (`utils/cohorts.py`). When
# the analysis switches from the women to the men, the *decisions* — the
# engineering recipe, the preprocessing choices, the selection, the model picks
# — are carried across unchanged, and everything that was FITTED or MEASURED is
# rebuilt on the new rows. Persisting a cohort's work means archiving exactly
# the second group, and the boundary between the two is one rule:
#
#     A key is per-branch iff its value was COMPUTED FROM ROWS.
#
# That cuts across the stages rather than at a stage boundary — `preprocessing`
# owns both `preprocessing_config` (a choice) and `preprocessing_pipeline` (a
# fit) — so the set is derived from the graph minus a named list of decisions,
# not hand-maintained. A result key registered on a stage above is per-branch
# by default; making it shared is the exception that has to be argued for here.
# ─────────────────────────────────────────────────────────────────────────────

#: The session key holding archived branches, `Dict[Tuple[str, str], Snapshot]`.
#: Declared here, in the headless module, because the thing that must never
#: forget it is `utils.session_state.reset_downstream_results` — see
#: `SHARED_DECISION_KEYS` below for why the archive is dropped by default.
BRANCH_ARCHIVE_KEY = "cohort_branch_archive"

#: Results that are a CHOICE the researcher made, not a measurement of rows.
#: They survive a cohort switch, so they are never archived and never restored.
#:
#: `feature_selection_results` / `consensus_features` are the arguable pair.
#: Strictly the ranking was computed on whichever cohort's training rows were
#: active when Feature Selection ran, so it is per-branch. But the *selection*
#: it produced (`selected_features`) is shared by construction — same question,
#: same predictors — and making the ranking per-branch would leave a freshly
#: created branch with an empty page 04 inviting a re-selection that changes
#: the shared decision under every other branch. Shared, with page 04 saying
#: whose rows produced the ranking.
SHARED_DECISION_KEYS: FrozenSet[str] = frozenset({
    "preprocessing_config", "preprocessing_config_by_model",
    "feature_selection_results", "consensus_features",
})

#: State describing `df_engineered` that no stage `produces`. The production
#: reset writes these to `False` / `[]` rather than clearing them, which is why
#: the equivalence gate excludes them by name — but they describe one cohort's
#: engineered frame and travel with it.
_FE_FRAME_STATE: FrozenSet[str] = frozenset({
    "feature_engineering_applied", "engineered_feature_names",
})

#: Every key a cohort branch owns. Snapshot these, restore these, and the flat
#: session keys are the active branch's view of the study.
BRANCH_KEYS: FrozenSet[str] = frozenset(
    (all_result_keys() | _FE_FRAME_STATE) - SHARED_DECISION_KEYS
)

#: Pages whose insights, methodology entries and provenance describe one
#: cohort's rows. The ledger and the methodology log are keyed by page rather
#: than by session key, so the branch/shared split for them is stated here in
#: the same module, from the same rule.
#:
#: Pages 02–04 are excluded deliberately: under a branch model the EDA and the
#: selection are shared work over the study, and the sequential flow that makes
#: them look per-cohort is what the up-front cohort declaration retires.
BRANCH_PAGES: FrozenSet[str] = frozenset({
    "05_Preprocess", "06_Train_and_Compare", "07_Explainability",
    "08_Sensitivity_Analysis", "09_Hypothesis_Testing",
})

#: Provenance sections a branch does NOT own, because they record a decision
#: rather than a measurement. The branch's sections are
#: `workflow_provenance.downstream_sections()` minus these — derived there
#: rather than here, because that record's schema is the authority on which
#: sections exist and this module must stay headless.
SHARED_PROVENANCE_SECTIONS: FrozenSet[str] = frozenset({
    "feature_engineering", "feature_selection",
})
