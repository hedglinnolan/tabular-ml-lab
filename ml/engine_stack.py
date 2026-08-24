"""What the estimator stack needs, named once, probed without importing it.

**`MODELS-026` / `TEST-087`.** `ml/model_registry.py` imports scikit-learn,
xgboost and lightgbm at module scope, so an interpreter missing any one of them
loses `GET /models` — and with it Train, Explain, the figures and the report —
to an unhandled `ModuleNotFoundError` that reaches Starlette as twenty-one
characters of *Internal Server Error*. Four human drives were spent on that,
and the cause was never in the code: the app was **measured** under `venv/` and
**served** from `turbotab/.venv`, which holds `fastapi` and `pandas` and none
of the estimators.

This module exists so three places can ask the same question and get the same
answer, without any of them importing the thing that explodes:

* `scripts/serve_turbotab.py` asks it **before binding the port**, so a server
  that cannot fit a model refuses to start instead of 500ing at the Train step;
* `turbotab/api.py` stamps it into `_SERVED_BUILD` at import, so `/dev/status`
  names the interpreter as well as the build;
* `ml/model_registry.py` reads the same names when it degrades.

**It uses `importlib.util.find_spec` and never imports.** That is deliberate on
two counts. Importing scikit-learn, xgboost and lightgbm costs seconds, and
this runs at API import — in every one of the two thousand test processes as
well as in the server. And the failure this is about is *absence*, which
`find_spec` sees exactly. A package that is installed and broken is a different
finding, and `get_registry()` is where it surfaces, with its own traceback.

**Naming the distribution as well as the module** because they differ where it
matters most: the import is `sklearn` and the thing you install is
`scikit-learn`, and a refusal that says *"pip install sklearn"* sends a person
to a stub package that exists to tell them they wanted a different name.
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List, Tuple

#: `(import name, distribution name, what it is for)`, in the order
#: `ml/model_registry.py` imports them — so `missing[0]` is the first absence
#: that would actually win, which is the sentence a traceback would have
#: produced. Run 4's sandbox had scikit-learn and no xgboost and its traceback
#: named xgboost at line 18; the host's serving interpreter had neither and
#: failed at line 6 on sklearn, twelve lines earlier. Both were real, and the
#: general statement is that the first absence wins.
ESTIMATOR_STACK: Tuple[Tuple[str, str, str], ...] = (
    ("sklearn", "scikit-learn", "every model on the shelf, and the pipeline "
                                "that fits them"),
    ("xgboost", "xgboost", "XGBoost, on the shelf for regression and "
                           "classification"),
    ("lightgbm", "lightgbm", "LightGBM, on the shelf for regression and "
                             "classification"),
)

#: Not part of the shelf, and asked about separately for that reason: `shap`
#: powers the explanation surface and `torch` is `TEST-038`'s deliberately
#: absent 1.1 GB. Neither takes `GET /models` down, and reporting them beside
#: the stack rather than inside it keeps *the shelf cannot be built* distinct
#: from *one explanation is unavailable*.
OPTIONAL_EXTRAS: Tuple[Tuple[str, str, str], ...] = (
    ("shap", "shap", "SHAP attributions on the Explain step"),
    ("torch", "torch", "the neural-network model (TEST-038: absent on "
                       "purpose)"),
)


def _installed(module: str) -> bool:
    """Whether `module` can be found, without importing it.

    `find_spec` raises rather than returning `None` when a PARENT package is
    missing, and returns `None` for a top-level one. Both mean absent here.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing() -> List[str]:
    """The estimator-stack modules this interpreter cannot find, in import
    order. Empty means the shelf can be built."""
    return [name for name, _dist, _why in ESTIMATOR_STACK if not _installed(name)]


def import_failure() -> Optional[Tuple[str, str]]:
    """`(module, error)` for the first stack member that will not import.

    **The other half of `missing()`, and the two are not interchangeable.**
    `find_spec` answers *is it there*, which is what `/dev/status` can afford
    at API import and is exactly the failure four drives hit. It says nothing
    about a package that is PRESENT and RAISES — a half-finished install, a
    shadowing file on `PYTHONPATH`, a binary wheel built for another
    architecture. `find_spec` finds all three and the import then dies anyway.

    The launcher can afford the real thing: it runs once, and the server it is
    about to start will import this stack on its first `/models` request
    regardless. So it does the import here, where the failure is a legible
    refusal in a terminal, instead of there, where it is twenty-one characters
    of Internal Server Error.

    Returns `None` when the whole stack imports.
    """
    import importlib

    for name, _dist, _why in ESTIMATOR_STACK:
        try:
            importlib.import_module(name)
        except BaseException as exc:              # noqa: BLE001 — any failure
            return name, f"{type(exc).__name__}: {exc}"
    return None


def report() -> Dict[str, Any]:
    """Everything `/dev/status` and the launcher both want, computed once."""
    absent = missing()
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "stack_ok": not absent,
        "missing": absent,
        "extras_missing": [name for name, _d, _w in OPTIONAL_EXTRAS
                           if not _installed(name)],
        "why": (None if not absent else
                f"This interpreter cannot import {absent[0]}, so the model "
                f"registry cannot be built and every model-shelf request will "
                f"fail. Nothing about the code is wrong; this is the "
                f"environment it is running in."),
        "fix": (None if not absent else
                f"Start the server from the environment that has them: "
                f"`make turbotab`, or "
                f"`venv/bin/python -m uvicorn turbotab.api:app --port 8777`. "
                f"If venv/ itself is incomplete, "
                f"`venv/bin/python -m pip install "
                f"{' '.join(_distribution(m) for m in absent)}`."),
    }


def _distribution(module: str) -> str:
    for name, dist, _why in ESTIMATOR_STACK + OPTIONAL_EXTRAS:
        if name == module:
            return dist
    return module


def sentence() -> str:
    """One line for a terminal or a banner. `''` when nothing is missing."""
    absent = missing()
    if not absent:
        return ""
    named = ", ".join(f"{m} (pip install {_distribution(m)})" for m in absent)
    return (f"{sys.executable} cannot import {named}. "
            f"The model shelf cannot be built in this interpreter.")
