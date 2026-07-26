"""
turbotab.readiness — the step-completion model, out of the stylesheet.

`utils/theme.py:685 render_sidebar_workflow` is not styling. It holds the only
implementation of the app's step-completion model — ten predicates over session
state — plus the quick/advanced split that decides which questions are optional.
`TRANSITION_PLAN.md` §02.4 says it plainly: *delete `theme.py` as "just styling"
during the cut and the step model goes with it.*

`ARCHITECTURE.md` §05 goes further — this is **the Router's readiness function**.
Before the Router can decide which question comes next, it has to be able to say
which steps are done, which are optional, and which cannot be asked yet. That is
this module.

Headless: predicates read a plain mapping, so the same function answers for
Streamlit's `session_state` and for an `AnalysisProject`. Both doors get the same
readiness or the two doors have forked.

The ten predicates are transcribed from `theme.py` and pinned against it by
`tests/integration/test_readiness_equivalence.py`, which reads the page's own
source rather than trusting this docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def _get(state: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return state[key] if key in state else default
    except (TypeError, KeyError):
        return default


def _nonempty(state: Mapping[str, Any], key: str) -> bool:
    v = _get(state, key)
    return bool(v) if v is not None else False


def _present(state: Mapping[str, Any], key: str) -> bool:
    return _get(state, key) is not None


# ─────────────────────────────────────────────────────────────────────────────
# The ten predicates
#
# One function each, named for the question it answers. `theme.py` computes
# these inline between two blocks of HTML; the logic is identical, and the
# equivalence test compares them expression by expression.
# ─────────────────────────────────────────────────────────────────────────────

def _data_uploaded(s: Mapping[str, Any]) -> bool:
    # theme.py calls `get_data() is not None`, which applies the active cohort
    # filter. A caller holding a project passes the working table under the same
    # key, so the predicate is the same question either way.
    return _present(s, "working_data")


def _data_configured(s: Mapping[str, Any]) -> bool:
    cfg = _get(s, "data_config")
    return cfg is not None and getattr(cfg, "target_col", None) is not None


def _features_selected(s: Mapping[str, Any]) -> bool:
    return _present(s, "feature_selection_results")


def _pipeline_built(s: Mapping[str, Any]) -> bool:
    return _present(s, "preprocessing_pipeline")


def _models_trained(s: Mapping[str, Any]) -> bool:
    return _nonempty(s, "trained_models")


def _explainability_run(s: Mapping[str, Any]) -> bool:
    return (len(_get(s, "permutation_importance", {}) or {}) > 0
            or len(_get(s, "shap_results", {}) or {}) > 0)


def _sensitivity_run(s: Mapping[str, Any]) -> bool:
    return _present(s, "sensitivity_seed_results")


def _report_generated(s: Mapping[str, Any]) -> bool:
    return _present(s, "report_data")


def _stat_validation_run(s: Mapping[str, Any]) -> bool:
    return (_present(s, "hypothesis_test_results")
            or len(_get(s, "custom_table1_tests", []) or []) > 0)


def _feature_engineering_applied(s: Mapping[str, Any]) -> bool:
    return bool(_get(s, "feature_engineering_applied", False))


@dataclass(frozen=True)
class Step:
    """One step of the workflow, and how to tell whether it is done."""
    key: str
    label: str
    page_id: str
    predicate: Callable[[Mapping[str, Any]], bool]
    # Quick workflow keeps the shortest defensible path to export; advanced
    # keeps the full sequence visible. This is the disclosure rule the Router
    # inherits — an "advanced" step is one the interview may decline to ask.
    core: bool


STEPS: Sequence[Step] = (
    Step("upload",       "Upload & Configure",     "01", _data_uploaded,               True),
    Step("eda",          "Explore (EDA)",          "02", _data_configured,             True),
    Step("features",     "Select Features",        "04", _features_selected,           True),
    Step("preprocess",   "Preprocess",             "05", _pipeline_built,              True),
    Step("train",        "Train Models",           "06", _models_trained,              True),
    Step("explain",      "Explain & Validate",     "07", _explainability_run,          True),
    Step("report",       "Export Report",          "10", _report_generated,            True),
    Step("engineering",  "Feature Engineering",    "03", _feature_engineering_applied, False),
    Step("sensitivity",  "Sensitivity Analysis",   "08", _sensitivity_run,             False),
    Step("stats",        "Statistical Validation", "09", _stat_validation_run,         False),
)

CORE_STEPS = tuple(s for s in STEPS if s.core)
ADVANCED_STEPS = tuple(s for s in STEPS if not s.core)


@dataclass
class Readiness:
    """Which steps are done, and what that implies for what may be asked next."""
    completed: Dict[str, bool] = field(default_factory=dict)
    mode: str = "quick"                       # quick | advanced

    @property
    def n_complete(self) -> int:
        return sum(1 for v in self.completed.values() if v)

    @property
    def n_total(self) -> int:
        return len(self.completed)

    @property
    def progress(self) -> float:
        return (self.n_complete / self.n_total) if self.n_total else 0.0

    def is_done(self, key: str) -> bool:
        return bool(self.completed.get(key, False))

    def visible_steps(self) -> List[Step]:
        """Which steps the workflow shows at this verbosity.

        Both modes show the core path; `quick` folds the advanced steps away
        rather than dropping them. Nothing is ever hidden outright — an
        invisible step is a question the user cannot discover they skipped.
        """
        return list(CORE_STEPS if self.mode == "quick" else STEPS)

    def next_step(self) -> Optional[Step]:
        """The first incomplete step on the core path.

        Deliberately *not* a Router. This says which step is next in sequence;
        the Router's job is to decide which **question** comes next, which is
        new construction (`ARCHITECTURE.md` §04) and is bound by the rule that
        only `high` confidence may pre-select. Readiness is an input to that
        decision, not the decision.
        """
        for s in CORE_STEPS:
            if not self.is_done(s.key):
                return s
        return None

    def blocked_steps(self) -> List[Step]:
        """Steps whose prerequisites are not met.

        A step is reachable once every earlier core step is done. Advanced steps
        hang off the core path and become reachable once the data is configured.
        """
        out: List[Step] = []
        core_done = True
        for s in CORE_STEPS:
            if not core_done:
                out.append(s)
            if not self.is_done(s.key):
                core_done = False
        if not self.is_done("eda"):
            out.extend(ADVANCED_STEPS)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"completed": dict(self.completed), "mode": self.mode,
                "n_complete": self.n_complete, "n_total": self.n_total,
                "progress": round(self.progress, 4),
                "next": (self.next_step().key if self.next_step() else None),
                "blocked": [s.key for s in self.blocked_steps()]}


def assess(state: Mapping[str, Any], mode: str = "quick") -> Readiness:
    """Run the ten predicates over a state mapping."""
    if mode not in ("quick", "advanced"):
        raise ValueError(f"workflow mode must be quick or advanced, got {mode!r}")
    return Readiness(completed={s.key: bool(s.predicate(state)) for s in STEPS},
                     mode=mode)
