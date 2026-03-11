"""
Compatibility shims for third-party libraries with sklearn version mismatches.

giotto-tda 0.6.2 (latest/final release) passes ``force_all_finite`` to
sklearn's ``check_array``, which was renamed to ``ensure_all_finite`` in
sklearn 1.8 with no deprecation shim.

Strategy: replace gtda's imported ``check_array`` reference with a thin
wrapper that translates the kwarg.  This is surgical — it only affects
gtda's calls, not the rest of the application.

Usage:
    from ml.compat import patch_gtda_for_sklearn
    patch_gtda_for_sklearn()  # call once before importing gtda
"""

import inspect
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_gtda_for_sklearn():
    """Patch giotto-tda's validation module to work with sklearn >= 1.8.

    Idempotent — safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    from sklearn.utils.validation import check_array as _real_check_array
    sig = inspect.signature(_real_check_array)

    if "force_all_finite" in sig.parameters:
        # sklearn still accepts the old name; nothing to do
        _PATCHED = True
        return

    try:
        import gtda.utils.validation as gtda_val
    except ImportError:
        _PATCHED = True
        return

    def _compat_check_array(array, *args, **kwargs):
        """Translate force_all_finite → ensure_all_finite for sklearn >= 1.8."""
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _real_check_array(array, *args, **kwargs)

    # Patch the reference in gtda.utils.validation — all gtda code that
    # calls check_array (directly or via _check_array_mod) goes through here.
    gtda_val.check_array = _compat_check_array

    # Also patch modules that may have already imported check_array or
    # validation helpers at module scope.
    import importlib
    for mod_name in [
        "gtda.homology.simplicial",
        "gtda.homology.cubical",
        "gtda.diagrams._utils",
        "gtda.metaestimators.collection_transformer",
    ]:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "check_array"):
                mod.check_array = _compat_check_array
        except (ImportError, AttributeError):
            pass

    _PATCHED = True
    logger.info(
        "Patched giotto-tda for sklearn >= 1.8 "
        "(force_all_finite → ensure_all_finite)"
    )
