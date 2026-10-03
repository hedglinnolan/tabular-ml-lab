"""Two metabolomics readings that fired on null data (audit IN-17, IN-18).

**Run-order drift.** ``packs._acquisition_order`` counted features with a Pearson
|r(order, log1p x)| above 0.3 and fired when 15% did, with no test of significance. At n = 16
injections the null alone puts about a quarter of features past |r| = 0.3, so it reported drift
on drift-free data in 40 of 40 small studies. METABOLOMICS_PACK §01 item 7 specifies the reading
(METABOLOMICS_PACK.md:167-168): "Per feature, Spearman ρ of intensity vs injection order within
batch; % of features with |ρ| > 0.3 and an FDR-significant trend". That is what is computed here:
Spearman's ρ per feature, Benjamini–Hochberg across features at q = 0.05, and a feature counts
only when both hold. The finding states the share expected by chance beside the observed one.

**Redundancy.** ``packs._redundancy`` clustered features by Pearson r > 0.9 on raw intensities
with single linkage, so one sample concentrated 20-fold, being the largest value of every
feature, correlated 300 independent features with each other and they read as about 40
quantities. Here correlation is Spearman's (ranks: one extreme sample moves each feature's rank
correlation by about 3/n, not to 0.9), clusters are formed by average linkage (a group whose
members correlate at 0.9 on average, not a chain of single links), and each group is re-read
without each sample in turn: a group that falls apart when one sample is left out is reported as
resting on that sample. §01's r > 0.9 cut stays, and stays a convention.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

RHO = 0.3                 # §01 item 7's |ρ|
FDR = 0.05                # Benjamini–Hochberg level
DRIFT_SHARE = 0.15        # share of features drifting before it is worth a finding (legacy's)
MIN_INJECTIONS = 8        # below this no rank correlation over run order is read
R_REDUNDANT = 0.9         # §01's correlation cut (a convention; the claim says so)
MIN_OVERLAP = 20          # samples two features must share for their correlation to count
MIN_COLLAPSE = 0.05       # share of features that must collapse before it is worth a finding
MAX_CLUSTERED = 4000      # features clustered at most (average linkage keeps p² distances)


# ── run-order drift ──────────────────────────────────────────────────────────


def _ranks(x: np.ndarray) -> np.ndarray:
    """Column-wise average ranks, NaN kept NaN."""
    return pd.DataFrame(x).rank(axis=0, method="average").to_numpy(dtype=float)


def spearman_against(order: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spearman's ρ of each column of ``block`` with ``order``, its two-sided p-value (Student t on
    n − 2 degrees of freedom, scipy's approximation) and the pairs each used."""
    from scipy import stats

    rho = np.full(block.shape[1], np.nan)
    p = np.full(block.shape[1], np.nan)
    n_used = np.zeros(block.shape[1], dtype=int)
    for j in range(block.shape[1]):
        y = block[:, j]
        ok = np.isfinite(y) & np.isfinite(order)
        n = int(ok.sum())
        n_used[j] = n
        if n < MIN_INJECTIONS or np.unique(y[ok]).size < 3:
            continue
        r = stats.rankdata(order[ok])
        s = stats.rankdata(y[ok])
        c = float(np.corrcoef(r, s)[0, 1])
        if not math.isfinite(c):
            continue
        rho[j] = c
        t = c * math.sqrt((n - 2) / max(1e-300, 1 - c * c))
        p[j] = float(2 * stats.t.sf(abs(t), n - 2))
    return rho, p, n_used


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q), NaN kept NaN."""
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    q[ok] = out
    return q


def chance_share(n: int, rho: float = RHO) -> float:
    """P(|ρ| > rho) for one feature with no drift at ``n`` injections (the t approximation)."""
    from scipy import stats

    if n < 4:
        return 1.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return float(2 * stats.t.sf(t, n - 2))


def drift_reading(df: pd.DataFrame) -> dict[str, Any] | None:
    """The run-order column and §01's drift reading over every other numeric column, or None."""
    from turbotab import packs

    cols = packs._numeric(df)
    if len(cols) < 30:
        return None
    order_col = packs._permutation_column(df)
    if order_col is None:
        return None
    features = [c for c in cols if c != order_col]
    order = df[order_col].to_numpy(dtype=float)
    block = df[features].to_numpy(dtype=float)
    rho, p, used = spearman_against(order, block)
    q = benjamini_hochberg(p)
    read = np.isfinite(rho)
    beyond = read & (np.abs(rho) > RHO)
    drifting = beyond & (q < FDR)
    n_read = int(read.sum())
    n = int(len(df))
    return {"column": order_col, "n_injections": n, "n_features": n_read,
            "n_beyond": int(beyond.sum()), "n_drifting": int(drifting.sum()),
            "share_beyond": float(beyond.sum() / n_read) if n_read else 0.0,
            "share_drifting": float(drifting.sum() / n_read) if n_read else 0.0,
            "chance_beyond": chance_share(n),
            "drifting": [features[j] for j in np.flatnonzero(drifting)]}


def run_order_finding(df: pd.DataFrame) -> dict[str, Any] | None:
    """``pack::metabolomics::run_order``: a run-order column, and intensity that tracks it."""
    from turbotab.packs import METABOLOMICS, RUN_ORDER_EVIDENCE, _finding

    r = drift_reading(df)
    if r is None or r["share_drifting"] < DRIFT_SHARE:
        return None
    return _finding(
        "pack::metabolomics::run_order", "warning",
        "There is a run-order column, and intensity tracks it",
        (f"`{r['column']}` runs 1 to {r['n_injections']:,} with every position used exactly once. "
         f"{r['n_drifting']:,} of {r['n_features']:,} features ({r['share_drifting']:.0%}) have a "
         f"Spearman correlation with it beyond {RHO} that survives Benjamini–Hochberg at "
         f"{FDR:.0%}. With no drift, about {r['chance_beyond']:.0%} of features would pass "
         f"|ρ| > {RHO} by chance at {r['n_injections']:,} injections (here {r['share_beyond']:.0%} "
         f"do), and almost none would also survive the correction."),
        ("Instrument drift is often the largest single variance component in a metabolomics run, "
         "larger than the biology. Correction is not applied here: it alters every value in the "
         "table, so it is a decision rather than a default."),
        confidence="high", pack=METABOLOMICS, marker="offered", evidence=RUN_ORDER_EVIDENCE,
        columns=[r["column"]] + r["drifting"][:6],
        params={"run_order_column": r["column"], "n_tracking": r["n_drifting"],
                "share_tracking": round(r["share_drifting"], 3),
                "share_beyond_rho": round(r["share_beyond"], 3),
                "chance_share_beyond_rho": round(r["chance_beyond"], 3),
                "rule": "spearman_abs_rho_gt_0.3_and_bh_fdr_lt_0.05",
                "rho": RHO, "fdr": FDR, "n_features": r["n_features"]})


# ── redundancy ───────────────────────────────────────────────────────────────


def _spearman_matrix(frame: pd.DataFrame) -> np.ndarray:
    corr = frame.corr(method="spearman", min_periods=MIN_OVERLAP).to_numpy(dtype=float)
    np.fill_diagonal(corr, 1.0)
    return np.nan_to_num(corr, nan=0.0)


def average_linkage_groups(corr: np.ndarray, threshold: float = R_REDUNDANT) -> list[list[int]]:
    """Groups whose members correlate above ``threshold`` on average (UPGMA on 1 − ρ, cut at
    1 − threshold)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    p = corr.shape[0]
    if p < 2:
        return [[i] for i in range(p)]
    distance = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"),
                      t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for i, g in enumerate(labels):
        groups.setdefault(int(g), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _rests_on_one_sample(frame: pd.DataFrame, members: list[str]) -> int | None:
    """The row whose omission takes the group's average Spearman correlation below the cut, or
    None when it holds without any single row."""
    sub = frame[members]
    for i in range(len(sub)):
        corr = sub.drop(index=sub.index[i]).corr(method="spearman").to_numpy(dtype=float)
        mean = (np.nansum(corr) - len(members)) / (len(members) * (len(members) - 1))
        if mean <= R_REDUNDANT:
            return i
    return None


def redundancy_reading(df: pd.DataFrame) -> dict[str, Any] | None:
    """How many independent quantities the numeric block carries, or None when it is not an
    assay-wide block (or too wide to cluster here)."""
    from turbotab import packs

    if not packs._is_assay_wide(df):
        return None
    cols = packs._numeric(df)
    if len(cols) > MAX_CLUSTERED:
        return None
    frame = df[cols].reset_index(drop=True)
    groups = average_linkage_groups(_spearman_matrix(frame))
    multi = [g for g in groups if len(g) > 1]
    fragile = []
    for g in multi[:50]:
        row = _rests_on_one_sample(frame, [cols[i] for i in g])
        if row is not None:
            fragile.append({"columns": [cols[i] for i in g], "row": int(row)})
    observed = frame.notna().sum()
    return {"n_columns": len(cols), "effective": len(groups),
            "collapsed": len(cols) - len(groups),
            "groups": [[cols[i] for i in g] for g in multi], "fragile": fragile,
            "below_overlap": int((observed < MIN_OVERLAP).sum())}


def redundancy_finding(df: pd.DataFrame) -> dict[str, Any] | None:
    """``pack::metabolomics::redundancy``: the effective number of quantities, when it is
    materially smaller than the column count."""
    from turbotab.packs import METABOLOMICS, REDUNDANCY_CLAIMS, REDUNDANCY_EVIDENCE, _finding

    r = redundancy_reading(df)
    if r is None or r["collapsed"] < max(3, round(MIN_COLLAPSE * r["n_columns"])):
        return None
    largest = r["groups"][0]
    factor = r["n_columns"] / max(r["effective"], 1)
    fragile = r["fragile"]
    leaning = (f" {len(fragile):,} of the groups hold together only because of one sample: "
               f"without it their members no longer correlate at {R_REDUNDANT} on average."
               if fragile else " No group rests on a single sample: each holds with any one "
                               "sample left out.")
    unmeasured = (f" {r['below_overlap']:,} column{'s are' if r['below_overlap'] != 1 else ' is'} "
                  f"observed in fewer than {MIN_OVERLAP} samples and counted as independent."
                  if r["below_overlap"] else "")
    return _finding(
        "pack::metabolomics::redundancy", "warning",
        f"{r['n_columns']:,} numeric columns, but about {r['effective']:,} independent quantities",
        (f"{len(r['groups']):,} groups of columns move together, with an average Spearman "
         f"correlation above {R_REDUNDANT} inside each (average linkage); the largest holds "
         f"{len(largest):,}: `{'`, `'.join(largest[:4])}`"
         + ("" if len(largest) <= 4 else f" and {len(largest) - 4:,} more") + ". "
         f"A compound count read off this table's width overstates it by about {factor:.1f}×."
         + leaning + unmeasured
         + " This table carries no retention time, so co-elution, the other half of the "
           "research's rule, could only split these groups further."),
        ("Untargeted features are not independent: one compound leaves the source as several "
         "ions and isotopologues. The count enters the data-description sentence and the "
         "denominator of a multiple-testing correction. Nothing is collapsed here."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=REDUNDANCY_EVIDENCE, claims=REDUNDANCY_CLAIMS,
        columns=largest[:8],
        params={"n_columns": r["n_columns"], "effective_features": r["effective"],
                "n_collapsed": r["collapsed"], "n_groups": len(r["groups"]),
                "largest_group": len(largest), "overstatement_factor": round(factor, 2),
                "r_threshold": R_REDUNDANT, "correlation": "spearman", "linkage": "average",
                "min_overlapping_samples": MIN_OVERLAP,
                "n_columns_below_min_overlap": r["below_overlap"],
                "groups_resting_on_one_sample": fragile,
                "clustered_on": ["inter-feature rank correlation"],
                "not_clustered_on": ["retention time"],
                "groups": r["groups"], "largest_group_columns": largest},
        fix_label="", fix_kind="none")


__all__ = ["average_linkage_groups", "benjamini_hochberg", "chance_share", "drift_reading",
           "redundancy_finding", "redundancy_reading", "run_order_finding", "spearman_against"]
