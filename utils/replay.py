"""Replaying a study's DECISIONS onto a different set of people.

"Now run the same analysis on Male" carries an assumption the app has to honor:
the same analysis. Until now the switch cleared feature engineering and
preprocessing outright, which is the safe half of the rule and only half of it:

    DECISIONS REPLAY, FITS DO NOT.

The distinction is the whole point. "Create bmi_squared from bmi" is a decision
— a formula that means the same thing for anyone. "Scale by the mean, which is
28.4" is a FIT, and 28.4 came from the women; reusing it in the men's run leaks
one group into the other's results and produces a number nobody can reproduce.

So a recipe records what was DECIDED, in machine terms rather than the prose
the engineering log carries ("Polynomial degree 2 (full): +12 features" cannot
be replayed). Each step declares whether it is:

  pure     — a row-wise formula. Replaying it on other rows is exact, and this
             module does it directly.
  refit    — a decision with a fitted part (binning edges, PCA components).
             The PARAMETERS replay; the fit is redone on the new rows, using
             only training rows so the held-out set stays sealed.
  manual   — heavy or optional-dependency steps (UMAP, TDA). Recorded and
             reported so the researcher can redo them deliberately, never
             silently skipped.

Anything this module cannot replay exactly is SAID, not swallowed. A run that
quietly used fewer predictors than the run it is compared against is exactly
the silent divergence a cohort comparison exists to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_RECIPE_KEY = "fe_recipe"
_PENDING_KEY = "cohort_replay_pending"
# The widget-shaped decisions — preprocessing settings, model picks and
# hyperparameter choices — wait here until the page that owns each widget
# renders. Kept apart from the recipe because run_pending_replay() clears
# `_PENDING_KEY` the moment the engineered features are rebuilt, and that
# happens on Train & Compare several reruns before anyone opens Preprocess.
_DECISIONS_KEY = "cohort_decisions_pending"

# Keys that share the widget prefix but describe STATE rather than a choice.
_NOT_A_CHOICE = frozenset({"preprocess_built_model_keys"})

# The Preprocess page's radio option, verbatim. Carried settings are only
# read by the Advanced widgets; in Smart Defaults mode the page overwrites
# every one of them from the new group's profile before the build can see
# them, so a carry forces the mode. tests pin this string against the page.
ADVANCED_MODE_LABEL = "🔧 Advanced (full control)"

_PICK_PREFIX = "train_model_"

PURE = "pure"
REFIT = "refit"
MANUAL = "manual"


@dataclass
class Step:
    """One engineering decision, in terms that can be re-executed."""
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    produced: List[str] = field(default_factory=list)
    mode: str = PURE

    def describe(self) -> str:
        n = len(self.produced)
        made = f"{n} feature{'' if n == 1 else 's'}"
        return {
            "polynomial": f"polynomial degree {self.params.get('degree')} → {made}",
            "interaction": f"interaction {self.params.get('op')} → {made}",
            "math": f"{self.params.get('transform')} transform → {made}",
            "ratio": f"ratios → {made}",
            "binning": f"binning ({self.params.get('strategy')}, "
                       f"{self.params.get('n_bins')} bins) → {made}",
            "pca": f"PCA ({self.params.get('n_components')} components) → {made}",
            "umap": f"UMAP ({self.params.get('n_components')} components) → {made}",
            "tda": f"TDA (H{self.params.get('homology_dims')}) → {made}",
        }.get(self.kind, f"{self.kind} → {made}")


# ── recording ────────────────────────────────────────────────────────────

def record(kind: str, params: Dict[str, Any], produced: Sequence[str],
           mode: str = PURE) -> None:
    """Called by Feature Engineering each time it creates columns."""
    import streamlit as st
    steps = st.session_state.get(_RECIPE_KEY) or []
    steps.append(Step(kind=kind, params=dict(params),
                      produced=[str(c) for c in produced], mode=mode))
    st.session_state[_RECIPE_KEY] = steps


def recipe() -> List[Step]:
    import streamlit as st
    return [s for s in (st.session_state.get(_RECIPE_KEY) or [])
            if isinstance(s, Step)]


def clear_recipe() -> None:
    import streamlit as st
    st.session_state.pop(_RECIPE_KEY, None)


def unrecorded_features(engineered: Sequence[str]) -> List[str]:
    """Engineered columns no step in the recipe claims to have produced.

    The guard that matters. Instrumenting creation sites one by one is a game
    you lose the moment someone adds a ninth: the first version of this module
    recorded three of eight, and the other five vanished on a cohort switch
    under a green "Rebuilt N features" success. Comparing what EXISTS against
    what the recipe can rebuild catches every site, including ones not written
    yet, and turns a silent loss into a named one.
    """
    claimed = {c for step in recipe() for c in step.produced}
    return [c for c in engineered if c not in claimed]


# ── what a cohort switch hands to the next run ───────────────────────────

def _picked_models() -> List[str]:
    """The models ticked on Train & Compare, or the ones with pipelines.

    The checkboxes are what the researcher actually chose, and they exist on
    the run the switch button is pressed (they render above it). The built
    list is the durable fallback for a switch pressed somewhere else.
    """
    import streamlit as st
    ticked = sorted(k[len(_PICK_PREFIX):] for k, v in st.session_state.items()
                    if isinstance(k, str) and k.startswith(_PICK_PREFIX)
                    and v is True)
    if ticked:
        return ticked
    return sorted(str(k) for k in (st.session_state.get("preprocess_built_model_keys") or []))


def _hyperparam_widget_keys(model_keys: Sequence[str]) -> List[str]:
    """Every widget key Train & Compare renders for these models' settings.

    `{model}_{param}` for each schema entry, plus the `_none` checkbox the
    int-or-None controls carry. The registry is the only source of the names;
    if it cannot be imported there is nothing to capture, not a guess.
    """
    try:
        from ml.model_registry import get_registry
        registry = get_registry()
    except Exception:
        return []
    keys: List[str] = []
    for mk in model_keys:
        spec = registry.get(mk)
        schema = getattr(spec, "hyperparam_schema", None) or {}
        for name in schema:
            keys.append(f"{mk}_{name}")
            keys.append(f"{mk}_{name}_none")
    return keys


def stage_for_replay(reason: str = "") -> Optional[Dict[str, Any]]:
    """Capture the decisions BEFORE a reset wipes them.

    Preprocessing choices ride along as configuration, not as pipelines: the
    pipelines are fits and must be rebuilt on the new rows.

    Two things are staged, under two keys. The engineering recipe goes to
    `_PENDING_KEY` and is replayed by run_pending_replay(). Everything that
    lives in a widget — the per-model preprocessing settings, the mode and
    interpretability radios, the model picks, the hyperparameter controls —
    goes to `_DECISIONS_KEY`, to be claimed by the page that renders the
    widget. Writing those keys here would not survive: Streamlit drops a
    widget's value at the end of any run that does not render the widget,
    and the rerun this switch triggers lands on Train & Compare, which stops
    above its own widgets until the pipelines are rebuilt.
    """
    import streamlit as st
    steps = recipe()
    prep = st.session_state.get("preprocessing_config_by_model") or {}
    # Prefix-matching "preprocess_" also catches preprocess_built_model_keys,
    # which is not a choice — it is the list of models that HAVE pipelines.
    # Carrying it across revived the exact defect where page 06 badges a model
    # "Tuned for this model" with no pipeline behind it.
    widget_choices = {k: v for k, v in st.session_state.items()
                      if isinstance(k, str) and k.startswith("preprocess_")
                      and k not in _NOT_A_CHOICE}
    _stage_decisions(prep, widget_choices, reason)
    if not steps and not prep and not widget_choices:
        return None
    engineered = list(st.session_state.get("engineered_feature_names") or [])
    pending = {
        "steps": steps,
        # Carried so the next run can say WHICH features it could not rebuild,
        # by name, instead of quietly having fewer predictors than run 1.
        "engineered_before": engineered,
        "unrecorded": unrecorded_features(engineered),
        "preprocessing_config": dict(prep),
        "preprocess_widgets": dict(widget_choices),
        "reason": reason,
    }
    st.session_state[_PENDING_KEY] = pending
    return pending


def _stage_decisions(prep: Dict[str, Any], widget_choices: Dict[str, Any],
                     reason: str) -> Optional[Dict[str, Any]]:
    """Park the widget-shaped decisions for the pages that own them."""
    import copy
    import streamlit as st
    try:
        from utils.cohorts import active_cohort
        run = active_cohort()
    except Exception:
        run = None
    picks = _picked_models()
    # Keyed by model, then by widget key: a model key can itself contain an
    # underscore (`knn_clf`), so the widget key alone cannot name the model.
    hyper: Dict[str, Dict[str, Any]] = {}
    for mk in picks:
        got = {k: st.session_state[k] for k in _hyperparam_widget_keys([mk])
               if k in st.session_state}
        if got:
            hyper[mk] = got
    mode = st.session_state.get("preprocess_config_mode")
    imode = st.session_state.get("interpretability_mode")
    if not prep and not widget_choices and not picks and not hyper:
        st.session_state.pop(_DECISIONS_KEY, None)
        return None
    decisions: Dict[str, Any] = {
        "from_label": str(run["label"]) if run else "",
        "reason": reason,
        "preprocess": {
            # Deep-copied: the per-model sub-dicts are the same objects the
            # page holds, and a later build mutates them in place.
            "config_by_model": copy.deepcopy(dict(prep)),
            "widgets": {k: v for k, v in widget_choices.items()
                        if k != "preprocess_config_mode"},
            "mode": mode if isinstance(mode, str) else None,
            "interpretability_mode": imode if isinstance(imode, str) else None,
        },
        "models": {"picks": picks, "hyperparams": hyper},
    }
    st.session_state[_DECISIONS_KEY] = decisions
    return decisions


def pending() -> Optional[Dict[str, Any]]:
    import streamlit as st
    p = st.session_state.get(_PENDING_KEY)
    return p if isinstance(p, dict) else None


def clear_pending() -> None:
    import streamlit as st
    st.session_state.pop(_PENDING_KEY, None)


def restore_decisions() -> List[str]:
    """Put the DECISIONS back after the reset. Returns what was restored.

    Only the recipe is written here. The widget-shaped decisions stay parked
    under `_DECISIONS_KEY` until their page claims them — see
    claim_for_preprocess_page() and claim_for_train_page() — and the notes
    say so, because "your preprocessing choices" used to be appended to this
    list with nothing written back at all.
    """
    import streamlit as st
    notes: List[str] = []
    p = pending()
    if p:
        if p.get("steps"):
            st.session_state[_RECIPE_KEY] = list(p["steps"])
            notes.append(f"{len(p['steps'])} feature-engineering step(s)")
        # Harmless when the keys were culled (they are re-seeded on claim),
        # and the only route for a choice made outside a built config.
        for k, v in (p.get("preprocess_widgets") or {}).items():
            st.session_state.setdefault(k, v)
    notes.extend(_decision_notes(decisions_pending()))
    return notes


# ── the widget-shaped decisions, claimed by the page that renders them ──

def decisions_pending() -> Optional[Dict[str, Any]]:
    import streamlit as st
    d = st.session_state.get(_DECISIONS_KEY)
    return d if isinstance(d, dict) else None


def clear_decisions() -> None:
    import streamlit as st
    st.session_state.pop(_DECISIONS_KEY, None)


def _write_decisions(d: Optional[Dict[str, Any]]) -> None:
    """Store what is still unclaimed, or nothing once every part is taken."""
    import streamlit as st
    live = d and (d.get("preprocess") or (d.get("models") or {}).get("picks")
                  or (d.get("models") or {}).get("hyperparams"))
    if live:
        st.session_state[_DECISIONS_KEY] = d
    else:
        st.session_state.pop(_DECISIONS_KEY, None)


def _names(keys: Sequence[str]) -> str:
    return ", ".join(str(k).upper() for k in keys)


def _decision_notes(d: Optional[Dict[str, Any]]) -> List[str]:
    """What is parked, in the words the sidebar and the notes use."""
    if not d:
        return []
    out: List[str] = []
    prep = d.get("preprocess") or {}
    cfgs = prep.get("config_by_model") or {}
    if cfgs:
        out.append(f"preprocessing settings for {_names(sorted(cfgs))}")
    elif prep.get("widgets"):
        out.append("preprocessing settings")
    models = d.get("models") or {}
    if models.get("picks"):
        out.append(f"model picks ({_names(models['picks'])})")
    if models.get("hyperparams"):
        out.append(f"hyperparameter settings for "
                   f"{_names(sorted(models['hyperparams']))}")
    return out


def describe_pending_decisions() -> str:
    """One sentence for the sidebar while decisions wait to be applied.

    Empty once everything has been claimed, so the caption cannot outlive
    the thing it describes.
    """
    d = decisions_pending()
    notes = _decision_notes(d)
    if not d or not notes:
        return ""
    src = f"the {d['from_label']} run" if d.get("from_label") else "the previous run"
    where = []
    if d.get("preprocess") or (d.get("models") or {}).get("picks"):
        where.append("Preprocess")
    if (d.get("models") or {}).get("hyperparams"):
        where.append("Train & Compare")
    return (f"Carried from {src}: {'; '.join(notes)}. Applied when you open "
            f"{' and '.join(where)}; the pipelines themselves are refit on "
            f"this group's rows.")


def _preprocess_widget_seeds(model_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The Preprocess widget keys that reproduce one built model config.

    The mapping is not one-to-one: the config names some things differently
    from the widgets (`use_kmeans_features` is the `use_kmeans` checkbox,
    the outlier parameters are renamed on the way in, PCA's one number is a
    radio plus either a number input or a slider). Anything the build derives
    from the data — feature lists, unit factors, plausibility bounds, the
    output width, the override notes — has no widget and is left to the
    rebuild, which is where it belongs.
    """
    prefix = f"preprocess_{model_key}_"
    out: Dict[str, Any] = {}

    def put(suffix: str, value: Any) -> None:
        out[prefix + suffix] = value

    for key in ("numeric_imputation", "numeric_scaling", "categorical_imputation",
                "categorical_encoding", "plausibility_mode"):
        if isinstance(cfg.get(key), str):
            put(key, cfg[key])
    power = cfg.get("numeric_power_transform")
    log = bool(cfg.get("numeric_log_transform", False))
    if not isinstance(power, str) and log:
        power = "log1p"
    if isinstance(power, str):
        put("numeric_power_transform", power)
        put("numeric_log_transform", power == "log1p")
    for key in ("numeric_missing_indicators", "plausibility_gating",
                "unit_harmonization", "use_pca", "pca_whiten"):
        if key in cfg:
            put(key, bool(cfg[key]))
    treatment = cfg.get("numeric_outlier_treatment")
    if isinstance(treatment, str):
        put("numeric_outlier_treatment", treatment)
        params = cfg.get("numeric_outlier_params") or {}
        if treatment == "percentile":
            if "lower_q" in params:
                put("outlier_lower_q", float(params["lower_q"]))
            if "upper_q" in params:
                put("outlier_upper_q", float(params["upper_q"]))
        elif treatment == "mad" and "threshold" in params:
            put("outlier_mad_threshold", float(params["threshold"]))
    if cfg.get("use_pca"):
        n = cfg.get("pca_n_components")
        if isinstance(n, bool):
            n = None
        if isinstance(n, int) and n >= 1:
            put("pca_mode", "Fixed Components")
            put("pca_fixed_n", int(n))
            put("pca_n_components", int(n))
        elif isinstance(n, float) and 0.0 < n < 1.0:
            put("pca_mode", "Variance Threshold")
            put("pca_variance", float(n))
            put("pca_n_components", float(n))
    if "use_kmeans_features" in cfg:
        put("use_kmeans", bool(cfg["use_kmeans_features"]))
    if cfg.get("use_kmeans_features"):
        if "kmeans_n_clusters" in cfg:
            put("kmeans_n_clusters", int(cfg["kmeans_n_clusters"]))
        if "kmeans_add_distances" in cfg:
            put("kmeans_distances", bool(cfg["kmeans_add_distances"]))
        if "kmeans_add_onehot" in cfg:
            put("kmeans_onehot", bool(cfg["kmeans_add_onehot"]))
    return out


def claim_for_preprocess_page() -> Optional[Dict[str, Any]]:
    """Seed the Preprocess page's widgets from the parked decisions. Once.

    Must run before any of those widgets is instantiated on the same run —
    Streamlit refuses a write to an instantiated widget's key, and a write
    made on a run that never renders the widget is dropped at the end of it.
    Returns what was applied, or None when nothing was waiting.
    """
    import streamlit as st
    d = decisions_pending()
    if not d:
        return None
    prep = d.pop("preprocess", None) or {}
    models = d.get("models") or {}
    picks = list(models.pop("picks", None) or [])

    seeded: List[str] = []
    for mk, cfg in (prep.get("config_by_model") or {}).items():
        if not isinstance(cfg, dict):
            continue
        for k, v in _preprocess_widget_seeds(str(mk), cfg).items():
            st.session_state[k] = v
        seeded.append(str(mk))
    # A choice made outside a built config only fills gaps: the built config
    # is what the previous group's models actually used.
    for k, v in (prep.get("widgets") or {}).items():
        if k not in _NOT_A_CHOICE:
            st.session_state.setdefault(k, v)
    mode_forced = bool(seeded)
    if mode_forced:
        st.session_state["preprocess_config_mode"] = ADVANCED_MODE_LABEL
    elif prep.get("mode"):
        st.session_state["preprocess_config_mode"] = prep["mode"]
    if prep.get("interpretability_mode"):
        st.session_state["interpretability_mode"] = prep["interpretability_mode"]
    for mk in picks:
        st.session_state[f"{_PICK_PREFIX}{mk}"] = True
    if picks:
        # The coach re-applies its own top picks once per reset. The picks
        # are the researcher's; do not let it add to them.
        st.session_state["_coach_applied"] = True
    _write_decisions(d)
    if not seeded and not picks and not prep:
        return None
    return {"from_label": d.get("from_label", ""), "models": seeded,
            "picks": picks, "mode_forced": mode_forced}


def claim_for_train_page() -> Optional[Dict[str, Any]]:
    """Seed Train & Compare's picks and hyperparameter controls. Once.

    Call it only past the page's pipeline gate: the widgets must render on
    the same run, or the values are dropped again at the end of it.
    """
    import streamlit as st
    d = decisions_pending()
    if not d:
        return None
    models = d.pop("models", None) or {}
    picks = list(models.get("picks") or [])
    hyper = dict(models.get("hyperparams") or {})
    for mk in picks:
        st.session_state.setdefault(f"{_PICK_PREFIX}{mk}", True)
    for mk, values in hyper.items():
        for k, v in (values or {}).items():
            st.session_state[k] = v
    _write_decisions(d)
    if not picks and not hyper:
        return None
    return {"from_label": d.get("from_label", ""), "picks": picks,
            "hyperparams": sorted(hyper)}


# ── replaying ────────────────────────────────────────────────────────────

def replay_onto(df: pd.DataFrame, steps: Sequence[Step],
                train_mask: Optional[pd.Series] = None
                ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Re-execute `steps` on `df`. Returns (frame, created, could_not_replay).

    Stateful steps are refit on TRAINING rows only — the held-out set stays
    sealed through a replay exactly as it does through the original.
    """
    out = df.copy()
    created: List[str] = []
    skipped: List[str] = []
    if train_mask is None:
        train_mask = pd.Series(True, index=out.index)
    train_mask = train_mask.reindex(out.index, fill_value=True)

    for step in steps:
        try:
            if step.kind == "ratio":
                for num, den in step.params.get("pairs", []):
                    if num not in out.columns or den not in out.columns:
                        skipped.append(f"{step.describe()} (missing {num} or {den})")
                        continue
                    d = out[den]
                    if not (d != 0).all():
                        skipped.append(f"{num} / {den} — the denominator holds zeros "
                                       f"in this group")
                        continue
                    name = f"{num}_div_{den}"
                    out[name] = out[num] / d
                    created.append(name)

            elif step.kind == "math":
                fn = {"log": lambda v: np.log1p(v.clip(lower=0)),
                      "sqrt": lambda v: np.sqrt(v.clip(lower=0)),
                      "square": lambda v: v ** 2,
                      "cube": lambda v: v ** 3,
                      "reciprocal": lambda v: 1.0 / v.replace(0, np.nan)}.get(
                          step.params.get("transform"))
                if fn is None:
                    skipped.append(step.describe())
                    continue
                for src, name in zip(step.params.get("columns", []), step.produced):
                    if src not in out.columns:
                        skipped.append(f"{name} (needs {src})")
                        continue
                    out[name] = fn(pd.to_numeric(out[src], errors="coerce"))
                    created.append(name)

            elif step.kind == "interaction":
                a, b = step.params.get("left"), step.params.get("right")
                op = step.params.get("op", "*")
                if a not in out.columns or (op != "square" and b not in out.columns):
                    skipped.append(step.describe())
                    continue
                if op == "square":
                    name = step.produced[0] if step.produced else f"{a}_squared"
                    out[name] = pd.to_numeric(out[a], errors="coerce") ** 2
                    created.append(name)
                    continue
                name = step.produced[0] if step.produced else f"{a}_{op}_{b}"
                x, y = pd.to_numeric(out[a], errors="coerce"), pd.to_numeric(out[b], errors="coerce")
                out[name] = {"*": x * y, "+": x + y, "-": x - y,
                             "/": x / y.replace(0, np.nan)}.get(op, x * y)
                created.append(name)

            elif step.kind == "polynomial":
                from sklearn.preprocessing import PolynomialFeatures
                cols = [c for c in step.params.get("columns", []) if c in out.columns]
                if not cols:
                    skipped.append(step.describe())
                    continue
                poly = PolynomialFeatures(
                    degree=int(step.params.get("degree", 2)),
                    interaction_only=bool(step.params.get("interaction_only", False)),
                    include_bias=False)
                base = out[cols].apply(pd.to_numeric, errors="coerce")
                if base.isna().any().any():
                    # fillna(0.0) would invent a measurement of zero for people
                    # whose value is missing in THIS group — a fabricated
                    # observation, which is worse than a missing feature. The
                    # page itself refuses to run on missing data; so does this.
                    skipped.append(f"{step.describe()} — some of "
                                   f"{', '.join(cols[:3])} are missing in this "
                                   f"group, and filling them with 0 would "
                                   f"invent measurements")
                    continue
                arr = poly.fit_transform(base)          # deterministic, no fitted state
                names = list(poly.get_feature_names_out(cols))
                for i, nm in enumerate(names):
                    if nm in cols or nm in out.columns:
                        continue
                    out[nm] = arr[:, i]
                    created.append(nm)

            elif step.kind == "binning":
                from sklearn.preprocessing import KBinsDiscretizer
                cols = [c for c in step.params.get("columns", []) if c in out.columns]
                if not cols:
                    skipped.append(step.describe())
                    continue
                kb = KBinsDiscretizer(n_bins=int(step.params.get("n_bins", 5)),
                                      encode="ordinal",
                                      strategy=step.params.get("strategy", "quantile"))
                base = out[cols].apply(pd.to_numeric, errors="coerce")
                if base.isna().any().any():
                    skipped.append(f"{step.describe()} — some inputs are missing "
                                   f"in this group")
                    continue
                kb.fit(base.loc[train_mask])            # refit on THIS group's train rows
                binned = kb.transform(base)
                for i, c in enumerate(cols):
                    name = f"{c}_binned"
                    out[name] = binned[:, i]
                    created.append(name)

            elif step.kind == "pca":
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                cols = [c for c in step.params.get("columns", []) if c in out.columns]
                k = int(step.params.get("n_components", 2))
                if len(cols) < k or not cols:
                    skipped.append(f"{step.describe()} — this group has "
                                   f"{len(cols)} of the {k} inputs it needs")
                    continue
                base = out[cols].apply(pd.to_numeric, errors="coerce")
                if base.isna().any().any():
                    skipped.append(f"{step.describe()} — some inputs are missing "
                                   f"in this group")
                    continue
                sc = StandardScaler().fit(base.loc[train_mask])
                pca = PCA(n_components=k, random_state=int(step.params.get("seed", 42)))
                pca.fit(sc.transform(base.loc[train_mask]))
                comps = pca.transform(sc.transform(base))
                for i in range(k):
                    name = f"PCA_{i + 1}"
                    out[name] = comps[:, i]
                    created.append(name)

            else:
                # UMAP / TDA and anything else: recorded, never silently dropped.
                skipped.append(step.describe())
        except Exception as exc:
            skipped.append(f"{step.describe()} — {type(exc).__name__}: {exc}")

    return out, created, skipped


def run_pending_replay(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Execute a staged replay onto `df`, if one is waiting. Idempotent.

    Lives here rather than on Feature Engineering because the switch button is
    on Train & Compare and the promise it makes — "the features you engineered
    are rebuilt from their formulas on this group's rows" — has to hold for a
    researcher who presses Train straight away. While this only ran on page 03,
    walking to that page was the difference between comparing two runs with the
    same predictors and comparing two runs with different ones, silently.

    Returns None when there was nothing to do, otherwise a dict with the
    rebuilt frame and the three name lists the caller should report.
    """
    import streamlit as st
    p = pending()
    if not p or not p.get("steps") or st.session_state.get("df_engineered") is not None:
        return None
    from utils.test_lockbox import train_row_mask

    rebuilt, made, missed = replay_onto(df, p["steps"], train_row_mask(df.index))
    if made:
        st.session_state["df_engineered"] = rebuilt
        st.session_state["engineered_feature_names"] = list(made)
        st.session_state["feature_engineering_applied"] = True
        dc = st.session_state.get("data_config")
        base = list(st.session_state.get("pre_fe_feature_cols")
                    or getattr(dc, "feature_cols", []) or [])
        # Setting selected_features without this leaves Reset/Skip with nothing
        # to restore — reopening a defect this repo already fixed once.
        st.session_state.setdefault("pre_fe_feature_cols", list(base))
        st.session_state["selected_features"] = base + [c for c in made if c not in base]
        if dc is not None:
            dc.feature_cols = list(st.session_state["selected_features"])
        # Without a log the Methods section says feature engineering was
        # performed and lists none of it.
        st.session_state["engineering_log"] = [
            f"Replayed for this group: {st.session_state.get('_replay_step_desc', '')}".strip()
        ] + [s.describe() for s in p["steps"]]
    unrecorded = list(p.get("unrecorded") or [])
    clear_pending()
    return {
        "frame": rebuilt if made else df,
        "created": list(made),
        "skipped": list(missed),
        "unrecorded": unrecorded,
        "summary": replay_summary(made, missed, unrecorded),
    }


def render_replay_result(result: Optional[Dict[str, Any]]) -> None:
    """Say what the replay did. A rebuild the researcher cannot see is a guess."""
    if not result or not result.get("summary"):
        return
    import streamlit as st
    if result["skipped"] or result["unrecorded"]:
        st.warning(f"🔁 {result['summary']}")
    else:
        st.success(f"🔁 {result['summary']}")


def replay_summary(created: Sequence[str], skipped: Sequence[str],
                   unrecorded: Sequence[str] = ()) -> str:
    """One paragraph a researcher can read before trusting the comparison."""
    if not created and not skipped and not unrecorded:
        return ""
    parts = []
    if created:
        parts.append(f"Rebuilt {len(created)} engineered feature"
                     f"{'' if len(created) == 1 else 's'} on this group's rows, "
                     f"refitting anything that had to be fitted.")
    if unrecorded:
        parts.append(
            f"{len(unrecorded)} feature(s) from the previous run cannot be "
            f"rebuilt automatically and are NOT in this run: "
            f"{', '.join(unrecorded[:6])}. Recreate them on this page if you "
            f"want the two runs to be comparable.")
    if skipped:
        parts.append(f"{len(skipped)} step(s) could NOT be repeated here: "
                     + "; ".join(skipped[:4])
                     + ". Those predictors are missing from this run, so it is "
                       "not answering quite the same question as the other one "
                       "— redo them, or drop them from both runs.")
    return " ".join(parts)
