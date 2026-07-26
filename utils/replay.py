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

# Keys that share the widget prefix but describe STATE rather than a choice.
_NOT_A_CHOICE = frozenset({"preprocess_built_model_keys"})

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


# ── what a cohort switch hands to the next run ───────────────────────────

def stage_for_replay(reason: str = "") -> Optional[Dict[str, Any]]:
    """Capture the decisions BEFORE a reset wipes them.

    Preprocessing choices ride along as configuration, not as pipelines: the
    pipelines are fits and must be rebuilt on the new rows.
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
    if not steps and not prep and not widget_choices:
        return None
    pending = {
        "steps": steps,
        "preprocessing_config": dict(prep),
        "preprocess_widgets": dict(widget_choices),
        "reason": reason,
    }
    st.session_state[_PENDING_KEY] = pending
    return pending


def pending() -> Optional[Dict[str, Any]]:
    import streamlit as st
    p = st.session_state.get(_PENDING_KEY)
    return p if isinstance(p, dict) else None


def clear_pending() -> None:
    import streamlit as st
    st.session_state.pop(_PENDING_KEY, None)


def restore_decisions() -> List[str]:
    """Put the DECISIONS back after the reset. Returns what was restored."""
    import streamlit as st
    p = pending()
    if not p:
        return []
    notes: List[str] = []
    if p.get("steps"):
        st.session_state[_RECIPE_KEY] = list(p["steps"])
        notes.append(f"{len(p['steps'])} feature-engineering step(s)")
    for k, v in (p.get("preprocess_widgets") or {}).items():
        st.session_state.setdefault(k, v)
    if p.get("preprocessing_config"):
        notes.append("your preprocessing choices")
    return notes


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
                base = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
                base = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
                base = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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


def replay_summary(created: Sequence[str], skipped: Sequence[str]) -> str:
    """One paragraph a researcher can read before trusting the comparison."""
    if not created and not skipped:
        return ""
    parts = []
    if created:
        parts.append(f"Rebuilt {len(created)} engineered feature"
                     f"{'' if len(created) == 1 else 's'} on this group's rows, "
                     f"refitting anything that had to be fitted.")
    if skipped:
        parts.append(f"{len(skipped)} step(s) could NOT be repeated here: "
                     + "; ".join(skipped[:4])
                     + ". Those predictors are missing from this run, so it is "
                       "not answering quite the same question as the other one "
                       "— redo them, or drop them from both runs.")
    return " ".join(parts)
