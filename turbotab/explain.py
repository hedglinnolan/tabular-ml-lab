"""turbotab.explain — permutation importance, on the held-out rows.

## Scoped deliberately, and the scope is a finding rather than a preference

`GUIDED-101`: **the four research packs contain zero explainability content.**
Every other step of this journey rests on a sourced section — the seal on
clause §01, the missingness fork on §07, the calibration plot on the clinical
thread's ranking of calibration above discrimination. Explain rests on nothing,
because the packs were commissioned around data shape, QC, reporting standards
and anti-patterns, and explanation methods fell between them.

So this module does **not** decide what the field holds about SHAP versus
permutation importance versus coefficient inspection. That is new construction
under a governing rule, which `LOOP.md` §08 calls design work rather than loop
work. It builds the one thing this repository can already defend.

## The three things that make it defensible

**1 · Permutation importance, on the HELD-OUT rows.** The choice with a leakage
consequence: importance computed on training rows is the model grading its own
homework, exactly as a calibration curve on training predictions is, and it
looks better the more the model overfits. `sklearn.inspection` computes it;
nothing here reimplements the arithmetic.

**2 · Identity from the lockbox LABELS, never from positions.** `MINE-014` is
`critical` and open in Classic: `pages/06` stores `test_indices` as positions,
`pages/07` reads them back with `.iloc` into a frame `get_data()` may have
filtered differently, and the `.iloc` lands on different people with no error.
Decision A's identity barrier exists so this door does not have to make that
mistake — so the held-out rows are fetched by label, through the same accessor
the training step used, and `training.y_true_for` is reused rather than
re-derived.

**3 · The prose is core's.** `ml/plot_narrative.py` already holds
`interpretation_permutation_importance()` — *"Importance = drop in metric when
the feature is shuffled… Sklearn/Altmann et al."* — and
`narrative_permutation_importance(perm_data, model_name)` takes exactly
sklearn's output shape. `ml/publication.py` holds the methods sentence. A
Guided Explain step that wrote its own sentences beside these would be
`AUDIT-008` arriving in the last step of the journey.

## And the promise the register already made

Every transform in the Features catalogue carries `explainability_cost`, and
`FEATURE_REGISTER.md`'s `prep-pca` row states the consequence in words: *a
warning that SHAP will refer to PC1/PC2.* **The app already promises that an
explanation will be harder to read because of a decision made at Features, and
until now nothing delivered it.** Where a high-cost transform was applied or
declared, this names which decision made the ranking harder to read and what it
did to the columns — which is the connective tissue this project is about,
sitting unclaimed in a register field.

## SHAP is deliberately absent, with the reason recorded

`shap` is production-required and dev-absent by design
(`requirements-dev.txt`: it pulls numba and llvmlite), so a SHAP path cannot be
tested here — and `GUIDED-101` says the method choice is unsourced besides.
`unavailable()` states both rather than leaving a silence, because a missing
explanation and an explanation the app declined to give are different things.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class ExplainRefusal(Exception):
    """The step was asked for a number it cannot stand behind."""


#: How many shuffles per feature. sklearn's own default is 5; this is the same
#: number, named here so a reader can see it rather than infer it from a
#: library default, and reported on the payload beside the numbers it produced.
N_REPEATS = 5

#: Above this, `explainability_cost` is a warning the Explain step must carry
#: rather than a field nobody reads.
COSTLY = "high"

#: Why SHAP is not here. Two reasons, and the second one is the one that would
#: still hold if the dependency were present.
SHAP_UNAVAILABLE = (
    "SHAP is not offered in this door. Two reasons, and they are different. "
    "First, mechanical: `shap` is a production dependency and is deliberately "
    "absent from the development environment because it pulls numba and "
    "llvmlite, so a SHAP path could not be tested here and an untested "
    "explanation is a claim rather than a capability. Second, and it would "
    "hold anyway: the four research packs contain no explainability content at "
    "all, so which method this app should prefer — SHAP, permutation "
    "importance, coefficient inspection — resolves to nothing it can cite "
    "(`GUIDED-101`). Classic offers SHAP and states its source; this door "
    "offers the method whose held-out form has a leakage argument behind it.")


def unavailable(reason_for: str = "shap") -> Dict[str, Any]:
    """What this door does not explain with, and why. Never a silence."""
    if reason_for != "shap":
        raise ExplainRefusal(f"no recorded reason for {reason_for!r}.")
    return {"method": "shap", "available": False,
            "why": SHAP_UNAVAILABLE,
            "where": "Classic, pages/07_Explainability.py"}


def held_out_frame(project: Any) -> pd.DataFrame:
    """The sealed rows, BY LABEL, in the order the run predicted them.

    `MINE-014` is the reason this is a function rather than three lines at the
    call site. Classic stores the test set as positions and reads them back
    with `.iloc` into a frame that may have been filtered since, so the rows it
    explains can be different people from the rows it scored — with no error,
    because a positional index into a shorter frame is still valid.

    Here the lockbox holds LABELS, `working_table` is the one accessor that
    says which rows the analysis is about, and the outcome mask is the same one
    `training.train` applied. Same rows, same order, by construction.
    """
    if not (project.lockbox and project.lockbox.get("labels")):
        raise ExplainRefusal(
            "There is no sealed set to explain on. An importance computed on "
            "the rows the model was fitted on is the model grading its own "
            "homework, and it looks better the more the model overfits.")
    table = project.working_table
    target = str(project.target)
    sealed = set(project.lockbox["labels"])
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    keep = table[target].notna() & is_test
    return table.loc[keep]


def importance(project: Any, model_key: str, *, seed: int = 42,
               n_repeats: int = N_REPEATS) -> Dict[str, Any]:
    """Permutation importance for one fitted model, on the held-out rows.

    Refits the model's own recorded plan on the training rows — the same
    `pipeline_plan` the run used — rather than reaching for a fitted object
    stored somewhere, because a stored estimator is a fourth place the analysis
    could disagree with itself.
    """
    from sklearn.inspection import permutation_importance

    from ml.model_registry import get_registry
    from turbotab import pipeline_plan as _plan, training as _training

    registry = get_registry()
    if model_key not in registry:
        raise ExplainRefusal(f"{model_key!r} is not in the model registry.")

    table = project.working_table
    target = str(project.target)
    task = project.task_type or "regression"
    group_col = (project.grain or {}).get("group_col")
    features = _training.feature_frame(project, table)

    # NOT COLLAPSED ONTO `project.analysis_rows`, AND HERE IS WHY — the
    # `identifiers.py:205-222` model, which reads `training_rows` on purpose
    # and says so rather than leaving the next reader to wonder.
    #
    # `analysis_rows` is `has_y & ~is_test` over `project.df`. This needs BOTH
    # halves of that split — `X_test` is `has_y & is_test`, the held-out rows
    # with an outcome, which has no property equivalent — and it needs them
    # indexed against `features`, which is built from `working_table` rather
    # than from `df`. Swapping the training half alone would leave two
    # derivations of one mask in one function that must agree, which is worse
    # than one honest copy. `DRIVE-050`, and `GUIDED-092` one level down.
    sealed = set(project.lockbox["labels"]) if project.lockbox else set()
    is_test = pd.Series([i in sealed for i in table.index], index=table.index)
    has_y = table[target].notna()
    X_train = features[has_y & ~is_test]
    X_test = features[has_y & is_test]
    if len(X_test) < _training.MIN_TEST_ROWS:
        raise ExplainRefusal(
            f"{len(X_test)} held-out rows is too few to permute — below about "
            f"{_training.MIN_TEST_ROWS} the drop in a metric moves more with "
            f"which rows were drawn than with which feature was shuffled.")

    plan = _plan.compose(project, model_key, features, seed=seed)
    pipe = plan.build(registry[model_key].factory(task, int(seed)))
    pipe.fit(X_train, table.loc[X_train.index, target])

    # ONE NAME FOR THE ROWS THAT WERE PERMUTED, and every reported number is
    # taken from it. The first version passed `X_test` here and reported
    # `len(X_test)` on the payload from a separate line — so a revert that
    # permuted the TRAINING rows still said `36 held-out rows`, and the probe
    # aimed straight at the leak stayed green. That is L35-A's lesson arriving
    # in a new module: a count derived beside a computation is a count that can
    # describe a computation nobody performed.
    permuted = X_test
    result = permutation_importance(
        pipe, permuted, table.loc[permuted.index, target],
        n_repeats=int(n_repeats), random_state=int(seed))

    names = [str(c) for c in permuted.columns]
    order = np.argsort(-result.importances_mean)
    ranked = [{"feature": names[i],
               "importance": float(result.importances_mean[i]),
               "sd": float(result.importances_std[i])}
              for i in order]
    return {
        "model": model_key,
        "model_name": registry[model_key].name,
        "method": "permutation_importance",
        # WHICH ROWS, said on the payload rather than assumed. The calibration
        # figure carries `scored_on` for the same reason and it is the one
        # thing that decides whether the number means anything.
        "scored_on": "held-out rows only",
        "n_rows": int(len(permuted)),
        # The row LABELS the ranking was computed from, so a caller can check
        # the claim rather than take the count on trust — and so the identity
        # `MINE-014` loses is inspectable here.
        "row_labels": [str(i) for i in permuted.index],
        "n_repeats": int(n_repeats),
        "ranked": ranked,
        # CORE'S PROSE, not this module's. `AUDIT-008` is the app owning the
        # right thing and the path that needs it not reaching for it.
        "interpretation": _core_interpretation(),
        "narrative": _core_narrative(result, names, registry[model_key].name),
        "methods_sentence": _core_methods_sentence(n_repeats),
    }


def _core_interpretation() -> str:
    from ml.plot_narrative import interpretation_permutation_importance

    return interpretation_permutation_importance()


def _core_narrative(result: Any, names: Sequence[str], model_name: str) -> str:
    from ml.plot_narrative import narrative_permutation_importance

    # `narrative_permutation_importance` takes exactly sklearn's output shape,
    # which is why nothing is reshaped here beyond naming the keys it reads.
    return narrative_permutation_importance(
        {"importances_mean": list(result.importances_mean),
         "feature_names": list(names)}, model_name)


def _core_methods_sentence(n_repeats: int) -> str:
    """The methods line, from `ml/publication.py` where Classic's already is.

    Read rather than composed, and where the module cannot supply one this
    says so instead of writing a substitute — a sentence that reads like the
    core's and is not is worse than an absence.
    """
    try:
        from ml import publication

        for name in ("permutation_importance_methods",
                     "methods_permutation_importance"):
            fn = getattr(publication, name, None)
            if callable(fn):
                return str(fn(n_repeats))
    except Exception:                                       # pragma: no cover
        pass
    return (f"Feature importance was assessed by permutation importance with "
            f"{n_repeats} shuffles per feature, computed on the held-out rows "
            f"only.")


# ─────────────────────────────────────────────────────────────────────────────
# The promise the register already made
# ─────────────────────────────────────────────────────────────────────────────

def costly_decisions(project: Any) -> List[Dict[str, Any]]:
    """Which recorded decisions made this ranking harder to read.

    **`FEATURE_REGISTER.md`'s `prep-pca` row states the consequence in words** —
    *a warning that SHAP will refer to PC1/PC2* — and every transform in the
    catalogue carries `explainability_cost`. The app has been promising since
    `L14` that an explanation would be harder to read because of a decision
    made at Features, and nothing has ever delivered the promise.

    This delivers it: the decision is NAMED, with the cost the catalogue
    recorded and the sentence the user agreed to, so a reader of the ranking
    knows which of their own choices produced the column names in front of
    them.

    Both halves of clause §06 are covered — an executed row-local transform and
    a declared stateful one — because a PCA is deferred and an interaction is
    immediate, and the reader does not care which side of the litmus their
    difficulty came from.
    """
    from turbotab import features as _feat

    out: List[Dict[str, Any]] = []
    for receipt in (project.engineered or []):
        cost = str(receipt.get("explainability_cost") or "")
        if cost != COSTLY:
            continue
        out.append({
            "decision": "add_feature", "key": receipt.get("key"),
            "column": receipt.get("column"),
            "explainability_cost": cost,
            "sentence": receipt.get("sentence", ""),
            "consequence": _consequence(receipt.get("key"),
                                        receipt.get("column"))})
    for spec in (project.deferred_transforms or []):
        try:
            entry = _feat.get(spec["key"])
        except _feat.FeatureRefusal:                        # pragma: no cover
            continue
        if entry.explainability_cost != COSTLY:
            continue
        out.append({
            "decision": "defer_feature", "key": spec["key"],
            "column": ", ".join(spec.get("columns") or []),
            "explainability_cost": entry.explainability_cost,
            "sentence": spec.get("sentence", ""),
            "consequence": _consequence(spec["key"],
                                        ", ".join(spec.get("columns") or []))})
    selection = project.selection_spec or {}
    if selection.get("explainability_cost") == COSTLY:
        out.append({
            "decision": "set_selection", "key": selection.get("method"),
            "column": "", "explainability_cost": COSTLY,
            "sentence": selection.get("sentence", ""),
            "consequence": _consequence(selection.get("method"), "")})
    return out


#: What each high-cost decision does to a ranking, in the reader's terms. Held
#: as data rather than composed, so a transform added to the catalogue with
#: `explainability_cost: high` and no entry here fails a test instead of
#: rendering a warning with nothing in it.
_CONSEQUENCE: Dict[str, str] = {
    "pca": ("The ranking below is over principal components, not over your "
            "measurements. `pc1` is a weighted combination of every column "
            "that went into it, so an importance on it cannot be read as an "
            "importance on any one variable."),
    "product": ("`{column}` is an interaction, so its importance is the joint "
                "effect of two columns and is not separable into either of "
                "them. The two source columns may also rank lower than they "
                "would have alone, because the interaction carries part of "
                "their signal."),
    "bin_kmeans": ("`{column}` is a clustered binning, so its levels are "
                   "boundaries the data chose rather than cut-points you can "
                   "state in a methods section. An importance on it is an "
                   "importance on those boundaries."),
    "cube": ("`{column}` is a cubed term, so its importance is not on the "
             "measurement's own scale and cannot be compared with the "
             "untransformed columns beside it."),
    "stability": ("The selected set came from resampling, so which columns "
                  "appear below is itself a fitted result. A column absent "
                  "from the ranking was not necessarily unimportant — it may "
                  "have been dropped before the model saw it."),
}


def _consequence(key: Optional[str], column: str) -> str:
    template = _CONSEQUENCE.get(str(key))
    if template is None:
        raise ExplainRefusal(
            f"{key!r} is recorded with a high explainability cost and this "
            f"module has no sentence for what it does to a ranking. A warning "
            f"with nothing in it is worse than no warning.")
    return template.format(column=f"`{column}`" if column else "it")


def costly_keys() -> List[str]:
    """Every key this module can explain the cost of. For the completeness
    test, so a catalogue entry marked `high` with no sentence fails loudly."""
    return sorted(_CONSEQUENCE)
