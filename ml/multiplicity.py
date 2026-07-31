"""Multiple comparisons, in one place, over the family the user actually ran.

`AUDIT-001`. The generated manuscript reported *"N of M tests yielded
statistically significant results (p < 0.05)"* with no correction named and none
applied. `research/METABOLOMICS_PACK.md` §06.3 states the case that makes it
indefensible:

> Plotting raw p-values with a line at p = 0.05 on a 3,000-feature untargeted
> dataset **is an anti-pattern and would be flagged in review.**

and §10 lists *asterisks without the test or correction* among the figure
anti-patterns. The pack's own coaching gives the arithmetic:

> *"You have 3,000 features. At an uncorrected p < 0.05 you'd expect about 150
> 'significant' hits by chance alone, and you have 187 — which is to say, your
> uncorrected result is consistent with nothing happening."*

**This is a thin adapter over `statsmodels.stats.multitest.multipletests`, not a
second implementation.** `ml/feature_selection.py:186` already calls it; a
hand-rolled Benjamini–Hochberg beside that one is the two-engines failure, and
the whole reason this module is four functions long.

## What it deliberately does not do

**It does not correct a family nobody declared.** Benjamini–Hochberg over
*"every test the user happened to run in this session"* is a methodological
choice about what the family IS, and the app does not get to make it silently —
that is the same rule the packs run on: raise confidence, never replace the
user's answer. So the correction is an act that gets RECORDED
(`WorkflowProvenance.apply_multiplicity_correction`), and where it has not
happened the manuscript says so rather than quietly BH-ing on the author's
behalf.

The consequence is the branch in `NarrativeEngine._gen_statistical_validation`:
a corrected count where a correction was recorded, and where it was not, **no
count at all** — because a count of raw-p hits is the number the anti-pattern is
made of.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# The methods `multipletests` accepts that this app offers, with the words a
# manuscript should use for them. A method not in this table is passed through
# to statsmodels and reported by its own name — the table is for prose, not a
# gate on what may be run.
METHOD_LABELS: Dict[str, str] = {
    "fdr_bh": "Benjamini-Hochberg FDR",
    "fdr_by": "Benjamini-Yekutieli FDR",
    "bonferroni": "Bonferroni",
    "holm": "Holm-Bonferroni",
}

DEFAULT_METHOD = "fdr_bh"
DEFAULT_ALPHA = 0.05


def method_label(method: str) -> str:
    """The name to print. Unknown methods print as themselves rather than fail."""
    return METHOD_LABELS.get(str(method), str(method))


def expected_by_chance(n_tests: int, alpha: float = DEFAULT_ALPHA) -> float:
    """How many of `n_tests` would reach `alpha` with nothing going on.

    The number the pack's coaching is built around, and the one that makes an
    uncorrected count readable as what it is. `α × n` — deliberately arithmetic
    a reader can check in their head rather than a simulation.
    """
    return float(alpha) * max(int(n_tests), 0)


def family_wise_error_rate(n_tests: int, alpha: float = DEFAULT_ALPHA) -> float:
    """`1 − (1 − α)^n`: the chance of at least one false positive, uncorrected.

    Duplicated from nowhere. `pages/09_Hypothesis_Testing.py` computes this
    inline for its on-screen warning, and that inline copy is why the warning
    and the manuscript could disagree: the page told the user about multiplicity
    and the draft did not. Both read this now.
    """
    n = max(int(n_tests), 0)
    if not n:
        return 0.0
    return float(1.0 - (1.0 - float(alpha)) ** n)


def adjust(tests: Sequence[Dict[str, Any]], *, method: str = DEFAULT_METHOD,
           alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """Adjust a recorded family of tests, returning the family and its summary.

    `tests` are the dicts `WorkflowProvenance.record_statistical_test` writes:
    `test_name`, `variable`, `statistic`, `p_value`, plus whatever details the
    caller attached. Every test keeps its identity; what is added is `q_value`,
    `correction` and `correction_alpha`.

    A test with no `p_value` is carried through UNADJUSTED and counted in
    `n_without_p`, never treated as `p = 1`. The difference matters: a test that
    did not report a p is not a test that failed to reach significance, and
    padding the family with ones would shrink everybody else's q.
    """
    tests = [dict(t) for t in (tests or [])]
    with_p = [t for t in tests if isinstance(t.get("p_value"), (int, float))
              and t["p_value"] is not None]
    without_p = [t for t in tests if t not in with_p]

    if not with_p:
        return {"tests": tests, "method": method, "alpha": float(alpha),
                "n_tests": len(tests), "n_adjusted": 0,
                "n_without_p": len(without_p), "n_significant": 0,
                "expected_by_chance": 0.0}

    from statsmodels.stats.multitest import multipletests

    p_values = [float(t["p_value"]) for t in with_p]
    reject, q_values, _, _ = multipletests(p_values, alpha=float(alpha),
                                           method=method)
    for test, q, rejected in zip(with_p, q_values, reject):
        test["q_value"] = float(q)
        test["significant_after_correction"] = bool(rejected)
        test["correction"] = str(method)
        test["correction_alpha"] = float(alpha)
    for test in without_p:
        test.setdefault("q_value", None)
        test.setdefault("significant_after_correction", None)
        test["correction"] = str(method)
        test["correction_alpha"] = float(alpha)

    ordered = [t for t in tests]
    return {
        "tests": ordered,
        "method": str(method),
        "alpha": float(alpha),
        "n_tests": len(tests),
        "n_adjusted": len(with_p),
        "n_without_p": len(without_p),
        "n_significant": int(sum(1 for t in with_p
                                 if t["significant_after_correction"])),
        "expected_by_chance": expected_by_chance(len(with_p), alpha),
    }


def correction_of(tests: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The correction applied to this family, or `None` where none was.

    `None` is the honest answer for a mixed family too: if some tests carry a
    correction and some do not, the family as a whole has not been corrected,
    and reporting the method of the corrected subset would describe a family
    that does not exist.
    """
    if not tests:
        return None
    methods = {str(t.get("correction")) for t in tests}
    if len(methods) != 1:
        return None
    method = methods.pop()
    return None if method in ("None", "", "none") else method
