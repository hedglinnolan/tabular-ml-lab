"""The genomics "what your numbers are" reading, where the pack's card is silent (audit IN-14,
IN-15, F16; GENOMICS_PACK §02).

``packs.data_type_card`` (shared with the legacy app, not edited) reads nine signatures. Three
gaps reached the Next app:

* **log2(TPM + 1) and voom log-CPM read as nothing.** §02's own coaching sends TPM/CPM to "a
  limma-style Gaussian workflow on log2(x+offset)" and has no row for that input. Here the
  matrix is back-transformed and tested: when 2^x − offset sums to one million in every sample
  (within 5%), it is log2 CPM or TPM with that offset; for a negative floor, offset 0 is tried,
  because voom's log-CPM is log2((count + 0.5) / (library size + 1) × 10^6) (limma, ``voom.R``:
  ``y <- t(log2(t(counts+0.5)/(lib.size+1)*1e6))``), whose back-transformed sums exceed one
  million by 0.5 × genes / library size. Within 25% it is a log of a composition-scaled CPM (TMM).
  Otherwise a non-integer matrix on a log scale (maximum ≤ 25) reads as log-scale expression whose
  normalization cannot be recovered, and the user is asked.
* **Shallow raw counts read as nothing.** §02's raw-count row requires a maximum "≫1e4"; a
  shallow library never reaches it. A matrix of non-negative whole numbers with zeros and unequal
  sample totals is read as raw counts at any depth, at medium confidence, and says why.
* **Single-cell matrices were silently modeled as bulk.** The card's out-of-scope reading reached
  no person. It is now a finding that names pseudoreplication: cells of one person are not
  independent replicates of that person, so a subject-level question needs one profile per
  person (pseudobulk) before any test; a cell-level prediction can use cells as units, and the
  finding says what it then cannot claim.

Every threshold is checked against the public GEO matrix GSE60450 (Fu et al. 2015; mouse
mammary gland, 12 samples) and its derived scales in the acceptance tests.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LOG_CEILING = 25.0          # §02's "max ~15–25" for log-scale expression
LOG_EXACT = 0.05            # back-transformed sample sums within 5% of one million
LOG_NEAR = 0.25             # … within 25%: a composition-scaled CPM (the pack's _LIBRARY_NEAR)
LIBRARY = 1e6
LOG_CPM = "log_cpm_or_tpm"
LOG_TMM = "log_scaled_cpm"
LOG_UNKNOWN = "log_expression"
SHALLOW_COUNTS = "raw_counts"

NAMES = {
    LOG_CPM: "log2 CPM or TPM",
    LOG_TMM: "log2 of a composition-scaled CPM (TMM or median-of-ratios)",
    LOG_UNKNOWN: "log-scale expression, normalization not recoverable",
    SHALLOW_COUNTS: "raw counts",
}

LOG_COACHING = (
    "These are log-scale expression values, the input a limma-style Gaussian workflow expects "
    "(GENOMICS_PACK §02: \"use a limma-style Gaussian workflow on log2(x+offset)\"). A count "
    "model is ruled out: the counts' variance cannot be recovered from them, so its p-values "
    "would be wrong. Do not log them again. limma-trend fits the mean–variance trend on these "
    "values; voom's precision weights need the raw counts, so bring those if you have them. "
    "Linear models here treat each value as a measurement on the log scale; a coefficient is a "
    "log2 fold difference.")
SHALLOW_COACHING = (
    "Raw counts, read from integrality and zeros rather than from depth: the largest value is "
    "below 10,000, so this is a shallow library or a targeted panel. Counts let a count model "
    "estimate measurement precision; at this depth many genes are near zero, so filtering "
    "low-count genes matters more than usual.")


def _back_sums(values: np.ndarray, offset: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.nansum(np.power(2.0, values) - offset, axis=1)


def log_signature(m: dict[str, Any], values: np.ndarray) -> dict[str, Any] | None:
    """The log-expression reading of a matrix the pack read as nothing, or None."""
    if m["all_integral"] or m["max"] > LOG_CEILING or m["min"] < -LOG_CEILING:
        return None
    offsets = (0.0,) if m["negatives"] else (1.0, 0.0)
    best = None
    for offset in offsets:
        sums = _back_sums(values, offset)
        deviation = float(np.nanmax(np.abs(sums - LIBRARY) / LIBRARY))
        if best is None or deviation < best[1]:
            best = (offset, deviation, sums)
    offset, deviation, sums = best
    if deviation <= LOG_EXACT:
        key = LOG_CPM
    elif deviation <= LOG_NEAR:
        key = LOG_TMM
    else:
        key = LOG_UNKNOWN
    return {"key": key, "offset": offset, "deviation": deviation,
            "back_sums": [float(np.nanmin(sums)), float(np.nanmax(sums))]}


QUANTUM_TOLERANCE = 0.02   # a value within this of a whole number of its sample's count quantum
QUANTUM_SHARE = 0.99       # of a sample's non-zero values that must be whole multiples
QUANTUM_SAMPLES = 0.9      # of samples that must read that way
# A count scaled by a library size times a composition factor sums near one million: TMM factors
# on the two public matrices the acceptance tests read run 0.64–1.22 (GSE60450, totals 0.82–1.56
# million) and 0.47–1.25 (GSE147507, 78 samples, totals 0.80–2.14 million). A quarter to four
# million keeps twice that margin either side; rounded non-expression data (servings a day to two
# decimals, which are "whole multiples" of 0.01) sum to tens, and are not counts (audit WP14 repair:
# a 130-item FFQ read as TMM-scaled CPM and was hinted genomics).
SCALED_SUM_RANGE = (0.25e6, 4.0e6)

# voom's log-CPM (limma, voom.R: ``y <- t(log2(t(counts+0.5)/(lib.size+1)*1e6))``) carries its
# counts exactly: within a sample, 2^y × (lib.size + 1) / 10^6 − 0.5 is a whole count. A sample's
# smallest value is its smallest count k0 (0 for any gene not seen), so lib.size + 1 =
# (k0 + 0.5) × 10^6 / 2^min, and every count follows. Read on the small counts (≤ 50), where the
# rounding of the values to four decimals moves a count by under 0.01. A library-size or
# back-sum threshold cannot do this: one shallow library moves voom's back-sum by
# 0.5 × genes / library size, 14–17% on GSE147507 (audit WP14 repair).
VOOM_MAX_COUNT = 50.0
VOOM_TOLERANCE = 0.05
VOOM_SHARE = 0.99
VOOM_MIN_VALUES = 20
VOOM_K0 = range(0, 6)


def scaled_counts(values: np.ndarray) -> dict[str, Any] | None:
    """Counts divided by a per-sample number (a library size, times a composition factor): each
    sample's values are whole multiples of its smallest non-zero value, one count. CPM and
    TMM-scaled CPM are; TPM and FPKM, divided by gene length too, and estimated counts are not.
    Returns the spread of the sample totals around one million, or None."""
    good, deviations = 0, []
    for row in values:
        nz = row[np.isfinite(row) & (row > 0)]
        if len(nz) < 20:
            continue
        # The quantum (one count) starts as the smallest value and is refined on ever larger
        # counts, so the rounding of the values (to a few decimals) does not compound into an
        # apparent fraction at a count of thousands.
        q = float(nz.min())
        for limit in (20.0, 500.0, np.inf):
            small = nz[nz / q <= limit]
            k = np.round(small / q)
            if not k.sum():
                break
            q = float(small.sum() / k.sum())
        r = nz / q
        if float(np.mean(np.abs(r - np.round(r)) <= QUANTUM_TOLERANCE)) >= QUANTUM_SHARE:
            good += 1
        deviations.append(abs(float(np.nansum(row)) - LIBRARY) / LIBRARY)
    if not deviations or good / len(deviations) < QUANTUM_SAMPLES or max(deviations) < 1e-3:
        return None
    totals = np.nansum(values, axis=1)
    if np.nanmin(totals) < SCALED_SUM_RANGE[0] or np.nanmax(totals) > SCALED_SUM_RANGE[1]:
        return None
    return {"max_deviation": max(deviations)}


def voom_reading(values: np.ndarray) -> dict[str, Any] | None:
    """The library sizes a voom log-CPM matrix implies, or None when the values are not voom's
    (see :data:`VOOM_MAX_COUNT`'s note). ``exact`` says whether each sample's recovered counts sum
    to its library size, as voom's default ``lib.size = colSums(counts)`` makes them."""
    libraries, exact, good, read = [], 0, 0, 0
    for row in values:
        x = row[np.isfinite(row)]
        if len(x) < VOOM_MIN_VALUES:
            continue
        read += 1
        low = float(x.min())
        with np.errstate(over="ignore"):
            ratio = np.power(2.0, x - low)
        for k0 in VOOM_K0:
            counts = (k0 + 0.5) * ratio - 0.5
            small = counts[counts <= VOOM_MAX_COUNT]
            if len(small) < VOOM_MIN_VALUES:
                break
            whole = np.abs(small - np.round(small)) <= VOOM_TOLERANCE
            if float(np.mean(whole)) >= VOOM_SHARE:
                good += 1
                library = (k0 + 0.5) * LIBRARY / 2.0 ** low - 1.0
                libraries.append(library)
                if abs(float(np.sum(np.round(counts))) - library) <= 0.01 * library:
                    exact += 1
                break
    if not read or good < QUANTUM_SAMPLES * read:
        return None
    return {"libraries": [float(min(libraries)), float(max(libraries))],
            "exact": exact == good, "n_samples": good}


def shallow_counts(m: dict[str, Any]) -> bool:
    """Non-negative whole numbers with zeros and unequal sample totals: counts at any depth."""
    return (m["all_integral"] and not m["negatives"] and m["max"] <= 1e4
            and m["pct_zeros"] > 0.0 and m["library_size"]["cv"] > 0.01)


def card(df: pd.DataFrame) -> dict[str, Any] | None:
    """The pack's card, with the readings it lacks filled in."""
    from turbotab import packs

    legacy = packs.data_type_card(df)
    if legacy is None:
        return None
    if legacy.get("out_of_scope"):
        return legacy
    lead = (legacy.get("classification") or {}).get("keys", [None])[0] if legacy.get("read") else None
    if legacy.get("read") and lead == packs.RAW_COUNTS:
        return legacy
    m = packs.read_matrix(df)
    if m is None:
        return legacy
    if shallow_counts(m) and lead in (None, packs.CPM_OR_TPM, packs.TMM_SCALED_CPM):
        return {"read": True, "extension": "shallow_counts",
                "classification": {"keys": [SHALLOW_COUNTS], "label": "Raw counts (shallow)",
                                   "confidence": "medium",
                                   "confidence_because": (
                                       "Every value is a whole number with zeros among them and the "
                                       "samples' totals differ, which is what counts look like at "
                                       "any depth; the largest is under 10,000, so depth does not "
                                       "confirm it."),
                                   "requires_input": False, "question": None,
                                   "coaching": SHALLOW_COACHING},
                "block": {k: m[k] for k in ("n_columns", "n_samples", "excluded", "n_excluded")},
                "measured": {k: m[k] for k in ("min", "max", "pct_zeros", "all_integral",
                                               "library_size")}}
    values = df[m["columns"]].to_numpy(dtype=float)
    if lead in (None, packs.ESTIMATED_COUNTS, packs.FPKM) and not m["all_integral"] \
            and not m["negatives"]:
        scaled = scaled_counts(values)
        if scaled is not None:
            # The pack's TMM row ("sums roughly but not exactly equal near 1e6", within 25%) misses
            # a public matrix whose composition factors run wide (GSE60450's TMM factors run
            # 0.64–1.22, so its TMM-scaled CPM sums 0.82–1.56 million), and its FPKM row then
            # reads it. Scaled counts say what it is.
            key = packs.TMM_SCALED_CPM
            return {"read": True, "extension": "scaled_counts",
                    "classification": {
                        "keys": [key], "label": packs.SIGNATURE_NAMES[key][:1].upper()
                        + packs.SIGNATURE_NAMES[key][1:], "confidence": "medium",
                        "confidence_because": (
                            f"Every sample's values are whole multiples of its smallest one, so "
                            f"these are counts divided by a per-sample number; the totals sit up "
                            f"to {scaled['max_deviation']:.0%} from one million, which is a "
                            f"library size times a composition factor such as TMM."),
                        "requires_input": False, "question": None,
                        "coaching": packs.COACHING[key]},
                    "block": {k: m[k] for k in ("n_columns", "n_samples", "excluded", "n_excluded")},
                    "measured": {k: m[k] for k in ("min", "max", "pct_zeros", "all_integral",
                                                   "negatives", "library_size")}}
    if legacy.get("read"):
        return legacy
    sig = log_signature(m, values)
    if sig is None:
        return legacy
    voom = None if m["all_integral"] else voom_reading(values)
    if voom is not None:
        lo, hi = voom["libraries"]
        sig = {**sig, "key": LOG_CPM, "offset": 0.0}
        label = "log2 CPM with a prior count (voom-style log-CPM)"
        because = (f"Within each sample, 2 to the power of each value, scaled by the library size "
                   f"its smallest value implies, is a whole count plus one half: limma's voom "
                   f"log-CPM, log2((count + 0.5) / (library size + 1) × 10^6), with libraries of "
                   f"{lo:,.0f} to {hi:,.0f} reads.")
        confidence, ask = "high", False
    elif sig["key"] == LOG_CPM:
        label = (f"log2(CPM or TPM + {sig['offset']:g})" if sig["offset"] else
                 "log2 CPM with a prior count (voom-style log-CPM)")
        because = (f"Raised to the power 2{' minus ' + format(sig['offset'], 'g') if sig['offset'] else ''}, "
                   f"every sample sums to one million (within {sig['deviation']:.1%}), so these are "
                   f"CPM or TPM on a log2 scale; which of the two cannot be recovered.")
        confidence, ask = "high", True
    elif sig["key"] == LOG_TMM:
        label = NAMES[LOG_TMM]
        because = (f"Raised to the power 2, the samples sum to near one million (within "
                   f"{sig['deviation']:.0%}) and not exactly, which is what a composition-aware "
                   f"rescaling leaves behind, on a log2 scale.")
        confidence, ask = "medium", False
    else:
        label = NAMES[LOG_UNKNOWN]
        because = ("The values are non-integer and top out below 25, a log scale, but no "
                   "back-transformation recovers a common library size, so which normalization "
                   "came first is not in the matrix.")
        confidence, ask = "low", True
    question = ("Which pipeline produced these values, and with what offset? CPM and TPM are "
                "indistinguishable once logged; you know, the matrix does not.") if ask else None
    return {"read": True, "extension": "log_expression",
            "classification": {"keys": [sig["key"]], "label": label, "confidence": confidence,
                               "confidence_because": because, "requires_input": ask,
                               "question": question, "coaching": LOG_COACHING,
                               "offset": sig["offset"], "back_sums": sig["back_sums"]},
            "block": {k: m[k] for k in ("n_columns", "n_samples", "excluded", "n_excluded")},
            "measured": {k: m[k] for k in ("min", "max", "median", "negatives", "pct_zeros",
                                           "all_integral", "library_size")}}


def _single_cell_finding(c: dict[str, Any]) -> dict[str, Any]:
    """Zimmerman, Espeland & Langefeld 2021 (Nat Commun 12:738): "Cells from the same individual
    share common genetic and environmental backgrounds and are not statistically independent;
    therefore, they are subsamples or pseudoreplicates." Squair et al. 2021 (Nat Commun 12:5692):
    "Methods that ignore this inevitable variation are biased and prone to false discoveries."
    The remedy is contested: Squair et al. favor pseudobulk; Zimmerman et al. find pseudobulk
    "conservative and underpowered relative to mixed models" with a random effect for individual.
    """
    from turbotab.packs import DISPUTED, GENOMICS, SETTLED, Claim, Evidence, _finding

    #: GENOMICS_PACK §11's anti-pattern registry: "Bulk pipeline on a single-cell matrix |
    #: SETTLED wrong | Zero-inflation, pseudoreplication".
    evidence = Evidence(status=SETTLED, source="research/GENOMICS_PACK.md#11 · Anti-pattern registry")
    crit = c.get("criteria") or {}
    return _finding(
        "pack::genomics::single_cell", "warning",
        "These look like single cells: cells of one person are not independent samples",
        (f"{crit.get('pct_zeros', 0):.0%} of the values are zero across "
         f"{crit.get('n_columns', 0):,} genes, with a median non-zero count of "
         f"{crit.get('median_nonzero', 0):g}: single-cell data, which the genomics pack is not "
         f"built for. Treating each cell as a sample is pseudoreplication when the question is "
         f"about people: thousands of cells from a handful of donors make p-values far too "
         f"small. For a subject-level question the person is the unit: sum each person's counts "
         f"per cell type (pseudobulk) and analyze one profile per person, or model the person as "
         f"a random effect; which of the two is better is disputed. For predicting a cell's type, "
         f"cells can be the unit, but the score then describes cells from these donors, not new "
         f"people."),
        ("Bulk models assume each row is an independent sample. Single cells are nested in "
         "donors, so a bulk test on cells counts the same person many times over."),
        confidence="high", pack=GENOMICS, marker="offered", evidence=evidence,
        claims=(
            Claim("pseudoreplication",
                  "Cells from one person are not independent samples of that person; methods that "
                  "treat them as independent are prone to false discoveries.", evidence),
            Claim("remedy",
                  "Whether to aggregate to one profile per person (pseudobulk) or to model the "
                  "person as a random effect is not settled.",
                  Evidence(status=DISPUTED,
                           source="research/GENOMICS_PACK.md#11 · Anti-pattern registry",
                           both_sides=(
                               "Squair et al. 2021 find pseudobulk methods avoid the false "
                               "discoveries of cell-level tests; Zimmerman et al. 2021 find "
                               "pseudobulk conservative and underpowered relative to mixed models "
                               "with a random effect for individual."))),
        ),
        columns=[],
        params={"out_of_scope": "single_cell", "criteria": crit,
                "concern": "pseudoreplication",
                "remedies": ["pseudobulk", "random effect for the person"],
                "by_purpose": {"inference": "one profile per person (pseudobulk) or the person as "
                                            "a random effect, before any test",
                               "prediction": "cells may be the unit; the score describes cells "
                                             "from these donors"}})


def findings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``pack::genomics::data_type`` (the pack's, or this module's where the pack is silent), or
    ``pack::genomics::single_cell``."""
    from turbotab import packs
    from turbotab.packs import DATA_TYPE_EVIDENCE, GENOMICS, _finding

    c = card(df)
    if c is None:
        return []
    if c.get("out_of_scope") == packs.SINGLE_CELL:
        return [_single_cell_finding(c)]
    if not c.get("extension"):
        found = packs._genomics_data_type(df)
        return [found] if found else []
    reading = c["classification"]
    key = reading["keys"][0]
    if c["extension"] == "scaled_counts":
        closed = [row["label"] for row in packs.capability_rows(key) if row["state"] == packs.DISABLED]
    else:
        closed = ([CLOSED_COUNT_MODEL, CLOSED_RELOG] if key != SHALLOW_COUNTS else [])
    return [_finding(
        "pack::genomics::data_type", "warning" if closed else "info",
        f"What your numbers are: {reading['label']}",
        (f"{c['block']['n_columns']:,} columns across {c['block']['n_samples']:,} samples. "
         f"{reading['confidence_because']}"),
        reading["coaching"] + (" " + reading["question"] if reading.get("question") else ""),
        confidence=reading["confidence"], pack=GENOMICS, marker="convention",
        evidence=DATA_TYPE_EVIDENCE,
        columns=list(c["block"]["excluded"]),
        params={"signatures": [key], "requires_input": reading["requires_input"], "lead": key,
                "n_features": c["block"]["n_columns"], "extension": c["extension"],
                "offset": reading.get("offset"), "back_transformed_sums": reading.get("back_sums"),
                "matrix_max": round(float(c["measured"]["max"]), 4),
                "negatives": bool(c["measured"].get("negatives", False)),
                "closed": closed})]


CLOSED_COUNT_MODEL = ("A count model of differential expression (DESeq2, edgeR) is ruled out: the "
                      "counts' variance cannot be recovered from log values.")
CLOSED_RELOG = "A further log transform is ruled out: these values are already on a log scale."

__all__ = ["LOG_CPM", "LOG_TMM", "LOG_UNKNOWN", "SCALED_SUM_RANGE", "card", "findings",
           "log_signature", "shallow_counts", "voom_reading"]
