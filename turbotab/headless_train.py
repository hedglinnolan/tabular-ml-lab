"""
CSV in, trained models out, with no Streamlit in the interpreter.

The L6 gate, as a script you can run:

    turbotab/.venv/Scripts/python turbotab/headless_train.py turbotab/sample_data/clinic_visits.csv --target outcome

It blocks `streamlit` on `sys.meta_path` *before importing anything else*, then
walks the whole path — read, diagnose, split, preprocess, fit, evaluate — using
only engine code. If any of it reaches for the host, the import raises here
rather than failing quietly in production later.

This is the first thing in the repository that trains a model without Streamlit,
and it is only possible because `pages/06:380-760` became `ml/splits.py`. The
split block was the last piece of the training path that existed solely inside
a Streamlit script.

Blocking is done with a `find_spec` finder. The `find_module`/`load_module`
protocol that `ARCHITECTURE.md` §01 used to print was removed from the import
system in Python 3.12 and blocks nothing on a current interpreter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


class _StreamlitBlocker:
    """Refuse to import Streamlit, loudly, from anywhere in the process."""

    def find_spec(self, name, path=None, target=None):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError(
                f"BLOCKED: {name} — this path is supposed to be headless. "
                "Something in the engine reached for the host.")
        return None


def install_blocker() -> None:
    if not any(isinstance(f, _StreamlitBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _StreamlitBlocker())


def run(csv_path: str, target: str | None = None, seed: int = 42) -> dict:
    """The whole path. Returns a small summary dict."""
    install_blocker()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from ml import import_doctor, triage
    from ml.eval import calculate_classification_metrics, calculate_regression_metrics
    from ml.splits import SplitSpec, make_split
    from models.registry_wrappers import RegistryModelWrapper
    from models.rf import RFWrapper

    df = pd.read_csv(csv_path)
    findings = import_doctor.diagnose(df)

    if target is None:
        target = df.columns[-1]
    detected = triage.detect_task_type(df, target)
    task_type = detected["detected"]

    # Features the model can actually consume without a full preprocess page:
    # numeric columns, plus low-cardinality text as one-hot.
    feature_cols, categorical = [], []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
        elif df[col].nunique(dropna=True) <= 12:
            feature_cols.append(col)
            categorical.append(col)
    numeric = [c for c in feature_cols if c not in categorical]

    split = make_split(df, feature_cols, target, task_type,
                       SplitSpec(random_state=seed))
    split.assert_disjoint()
    split.assert_identity_preserved(df)

    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), numeric),
         ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical)],
        remainder="drop",
    )
    def to_dense(a):
        return a.toarray() if hasattr(a, "toarray") else np.asarray(a)

    # Fit on training rows only, then apply. Never the other way round.
    Xtr = to_dense(pre.fit_transform(split.X_train))
    Xva = to_dense(pre.transform(split.X_val))
    Xte = to_dense(pre.transform(split.X_test))

    wrappers = {
        "random_forest": RFWrapper(n_estimators=60, task_type=task_type)
        if "task_type" in RFWrapper.__init__.__code__.co_varnames
        else RFWrapper(n_estimators=60),
        "ridge": RegistryModelWrapper(Ridge(alpha=1.0), name="ridge"),
    }
    if task_type == "classification":
        from sklearn.linear_model import LogisticRegression
        wrappers["logistic"] = RegistryModelWrapper(
            LogisticRegression(max_iter=1000), name="logistic")
        wrappers.pop("ridge", None)

    results = {}
    for name, w in wrappers.items():
        w.fit(Xtr, split.y_train, Xva, split.y_val)
        preds = w.predict(Xte)
        if task_type == "classification":
            proba = w.predict_proba(Xte) if w.supports_proba() else None
            results[name] = calculate_classification_metrics(
                split.y_test, preds, proba[:, 1] if proba is not None and proba.shape[1] == 2 else None)
        else:
            results[name] = calculate_regression_metrics(split.y_test, preds)

    assert "streamlit" not in sys.modules, "something imported Streamlit"

    return {
        "rows": int(len(df)),
        "findings": len(findings),
        "target": target,
        "task_type": task_type,
        "task_confidence": detected["confidence"],
        "strategy": split.strategy,
        "sizes": split.sizes,
        "cv_strategy": split.cv_strategy,
        "models": {k: {m: round(float(v), 4) for m, v in vals.items()
                       if isinstance(v, (int, float))}
                   for k, vals in results.items()},
        "streamlit_imported": "streamlit" in sys.modules,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--target", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = run(args.csv, args.target, args.seed)

    print(f"file            {args.csv}")
    print(f"rows            {summary['rows']}")
    print(f"findings        {summary['findings']} structural")
    print(f"target          {summary['target']} -> {summary['task_type']} "
          f"({summary['task_confidence']} confidence)")
    print(f"split           {summary['strategy']}  {summary['sizes']}  cv={summary['cv_strategy']}")
    for name, metrics in summary["models"].items():
        head = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:4])
        print(f"  {name:<15} {head}")
    print(f"streamlit       {'IMPORTED (gate failed)' if summary['streamlit_imported'] else 'never imported'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
