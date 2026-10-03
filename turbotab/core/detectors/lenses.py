"""Lens hints and the lens contradiction check (audit IN-13).

``packs.suggest`` (shared with the legacy app, not edited) hinted genomics only for integer count
matrices and metabolomics for any table with at least 30 numeric columns, so six of nine genomics
fixtures, TPM and log2-TPM with Ensembl IDs, two survey fixtures, a 45-nutrient DR1T* table and a
38-lab clinical table were all hinted "metabolomics"; and ``packs.contradiction`` was called only
from the legacy app. A hint is never pre-selected (§01: "the user's answer is the answer"), but it
is where a user who does not know the vocabulary starts, so each hint here needs positive
evidence:

* **genomics**: a count matrix; or column labels drawn from a gene-identifier vocabulary
  (Ensembl, RefSeq, Affymetrix, Illumina, Agilent; GENOMICS_PACK §01); or a wide block (100+
  numeric columns) the data-type reading recognizes (CPM/TPM, log-expression, VST, microarray …);
* **metabolomics**: METABOLOMICS_PACK §01's cues: feature-name grammar (``mz_0001``,
  ``M123T456``, ``123.4567_5.67``, lipid shorthand), m/z or retention-time header tokens, or an
  assay-wide block beside quality-control rows or a run-order column;
* **survey** before any width rule: a response block (:mod:`.scales`);
* **dietary**: a numeric column the one recognizer reads as total energy intake
  (:func:`total_energy_column`; audit IN-08, closed when WP13 and WP14 met);
* **clinical** as before (two recognized clinical measurements), no longer withheld from a wide
  table that is not an assay.

The contradiction check runs on the stated lens in the findings stage and becomes a finding the
user answers (change the lens, or record that the answer stands); a "shape is an assay"
contradiction needs an assay hint behind it, so a wide survey or clinical table is not told it
is an assay panel.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import pandas as pd

from turbotab.core.detectors import scales

_FEATURE_GRAMMAR = re.compile(
    r"^(?:m/?z[_\s.-]?\d|M\d+(?:\.\d+)?T\d+(?:\.\d+)?$|\d+\.\d+[_@]\d+(?:\.\d+)?$|FT\d+$|"
    r"X\d+\.\d+$|(?:PC|PE|TG|DG|Cer|SM|LPC|LPE|PS|PI|PG|CE|FA)[\s(]\d+:\d+)", re.I)
_MZ_RT = re.compile(r"^(?:row[\s_]?)?(?:m/?z|mass|rt|retention[\s_]?time(?:[\s_]?\(?min\)?)?)$", re.I)
_QC = re.compile(r"(?:^|[_\s-])(?:qc|pqc|qcp|sqc|pool(?:ed)?|blank|blk)(?:$|[_\s\d-])", re.I)
_RUN_ORDER = re.compile(r"(?:^|[_.\s])(?:run|inj(?:ection)?|acq(?:uisition)?|sequence|seq)"
                        r"(?:[_.\s]?(?:order|index|no|number))?(?:$|[_.\s])|order$", re.I)
MIN_SHARE = 0.5          # of numeric column labels that must carry a vocabulary or grammar
MIN_WIDE = 100           # numeric columns for a data-type reading to hint genomics on its own
# Whole numbers are not yet counts of transcripts: a raw food-frequency questionnaire's 9-point
# codes, portions in a web recall, food groups in whole grams and minute-level accelerometer counts
# are all wide blocks of non-negative integers (audit WP13 gate repair: each was hinted genomics,
# "what a count matrix looks like", and drew a critical lens contradiction under its own dietary
# or survey lens). What sets expression counts apart is that features differ in abundance by
# orders of magnitude: the middle half of genes' mean counts spans 2.6 decades on GEO GSE60450 and
# 2.3 on GSE147507 (the acceptance tests' public matrices), while every non-assay table above
# spans under 0.1, its items being answered on one scale. TurboTab's own cut-off sits between.
ABUNDANCE_SPREAD = 0.5   # decades: interquartile range of log10(1 + each column's mean count)
# BLUEPRINT §14 ("lens hints only from positive evidence"). Spread alone is no count matrix: the gate
# hinted genomics for an FFQ in whole grams per food (tea in hundreds of grams, herbs in single
# grams: 1.4 decades) and for a Metabolon export of integer peak areas (2.3 decades), and drew a
# critical lens contradiction under the FFQ's own dietary lens. Two properties of read counts:
# a feature is never named in a unit of mass, volume or energy (``tea_g`` is grams eaten), and a
# sequencer records the small counts of scarcely expressed features: on the public matrices the
# tests read, 42% of GSE60450's cells, 47% of GSE147507's and 97% of GSE152075's are 10 or under,
# against none of the peak areas (minimum 1,766). TurboTab's own floor sits far below the counts.
MIN_SMALL_COUNTS = 0.05  # share of the block's cells at 10 or under
SMALL_COUNT = 10
# A name that ends in a unit of mass, volume or energy (``tea_g``, ``milk_ml``, ``sodium_mg``,
# ``snack_kcal``): an amount measured, read as a suffix so a gene label such as ``g_A1BG`` is not.
_UNIT_SUFFIX = re.compile(r"(?i)[_\s.(-](?:g|gm|gram|grams|mg|mcg|ug|µg|ml|l|dl|oz|kcal|kj|iu)\)?"
                          r"(?:[_\s.-]?(?:d|day|per_day))?$")


def _numeric(df: pd.DataFrame) -> list[str]:
    from turbotab import packs

    return packs._numeric(df)


def _share(labels: Sequence[str], test: Any) -> float:
    return sum(1 for c in labels if test(str(c))) / max(len(labels), 1)


def gene_vocabulary(df: pd.DataFrame) -> float:
    """The share of numeric column labels (or of a label column's values, for a table with genes
    in rows) drawn from a gene-identifier vocabulary."""
    from turbotab import packs

    numeric = _numeric(df)
    best = sum(packs.id_vocabulary(numeric)["matched"].values()) if numeric else 0.0
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame) or pd.api.types.is_numeric_dtype(s):
            continue
        values = s.dropna().astype(str).head(2000).tolist()
        if len(values) >= 20:
            best = max(best, sum(packs.id_vocabulary(values)["matched"].values()))
        break
    return float(best)


def metabolomics_evidence(df: pd.DataFrame) -> str | None:
    """Why the table reads as metabolomics (METABOLOMICS_PACK §01's cues), or None."""
    from turbotab import packs

    numeric = _numeric(df)
    if numeric and _share(numeric, _FEATURE_GRAMMAR.match) >= MIN_SHARE:
        return f"{len(numeric):,} measurement columns are named like LC–MS features (`{numeric[0]}`)"
    named = [str(c) for c in df.columns if _MZ_RT.match(str(c).strip())]
    if named:
        return f"the table has feature metadata columns ({', '.join(f'`{c}`' for c in named[:3])})"
    if not packs._is_assay_wide(df):
        return None
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame) or pd.api.types.is_numeric_dtype(s):
            continue
        hits = s.dropna().astype(str).str.contains(_QC).sum()
        if 2 <= hits < len(s):
            return f"`{c}` marks {int(hits):,} rows as quality-control or blank injections"
    # §01: "injection.order, inj_order, run_order, sequence, acq_order" — named as an order, and a
    # permutation of the rows. A patient number running 1..n is a permutation too, and no order.
    order = packs._permutation_column(df)
    if order and _RUN_ORDER.search(str(order)):
        return f"`{order}` is an injection order beside {len(numeric):,} measurement columns"
    return None


def abundance_spread(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """The interquartile range, in decades, of ``log10(1 + mean)`` over ``columns``."""
    import numpy as np

    means = df[list(columns)].apply(pd.to_numeric, errors="coerce").mean(axis=0).to_numpy(dtype=float)
    means = means[np.isfinite(means)]
    if len(means) < 4:
        return 0.0
    logged = np.log10(1.0 + np.clip(means, 0.0, None))
    q1, q3 = np.percentile(logged, [25, 75])
    return float(q3 - q1)


def count_signature(df: pd.DataFrame) -> str | None:
    """Why a wide block of non-negative whole numbers reads as expression counts, or None: its
    features' mean counts span orders of magnitude (:data:`ABUNDANCE_SPREAD`), as genes' do; a
    questionnaire, a recall's portions or minute counts sit on one scale."""
    from turbotab import packs

    import numpy as np

    block = packs.count_matrix(df)
    if block is None:
        return None
    columns = [c for c in block["columns"] if not _UNIT_SUFFIX.search(str(c))]
    if len(columns) < len(block["columns"]) / 2:
        return None  # amounts named in a unit (an FFQ's ``tea_g``), not counts of reads
    spread = abundance_spread(df, columns)
    if spread < ABUNDANCE_SPREAD:
        return None
    # A bounded read (the first 2,000 features on the first 2,000 rows) keeps a wide matrix cheap.
    cells = df[columns[:2000]].head(2000).apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float).ravel()
    cells = cells[np.isfinite(cells)]
    if not len(cells) or float(np.mean(cells <= SMALL_COUNT)) < MIN_SMALL_COUNTS:
        return None  # no small counts: integer intensities or peak areas, not reads
    return (f"{len(columns):,} columns hold non-negative whole numbers whose mean counts "
            f"span {spread:.1f} orders of magnitude across the middle half of them, as genes' do")


def total_energy_column(df: pd.DataFrame) -> str | None:
    """The first numeric column whose name reads as total energy intake, by the one recognizer
    (:func:`turbotab.core.recognizers.reads_as_total_energy`), or None.

    Audit IN-08: the pack's exact-alias matcher (``packs._reference_column``) left ``DR2TKCAL``,
    ``DRXTKCAL``, ``ENERC_KCAL``, ``TotalKcal``, ``total_energy`` and ``kcal_day`` with no dietary
    hint, though the roles recognizer found them. Every alias it matched (``energy``,
    ``calories``, ``kilocalories``, ``energy_kcal``, ``DR1TKCAL``, ``kcal``) still reads.
    """
    from turbotab.core.recognizers import energy_median_contradicts, reads_as_total_energy

    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame) or not pd.api.types.is_numeric_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            continue
        # A hint from positive evidence (BLUEPRINT §14): the name, and a median that is a day's
        # energy in kcal or kJ (an SF-36 ``energy`` score's 50 is none).
        if reads_as_total_energy(c) and energy_median_contradicts(
                pd.to_numeric(s, errors="coerce").median()) is None:
            return str(c)
    return None


def hints(df: pd.DataFrame | None) -> list[dict[str, str]]:
    """What the table hints at, survey first, each with its evidence."""
    from turbotab import packs
    from turbotab.core.detectors import genomics

    out: list[dict[str, str]] = []
    if df is None or df.empty:
        return out
    found = scales.blocks(df)
    if found:
        b = found[0]
        named = f", matching the {b.instrument}'s shape" if b.instrument else ""
        out.append({"lens": "survey",
                    "because": f"{len(b.columns):,} columns share one {b.points}-point response "
                               f"scale{named}"})
    numeric = _numeric(df)
    vocabulary = gene_vocabulary(df)
    genomic = None
    metabolomic = None if found else metabolomics_evidence(df)
    counts = count_signature(df) if not metabolomic else None
    if counts:
        genomic = counts
    elif vocabulary >= MIN_SHARE:
        genomic = f"{vocabulary:.0%} of the labels are gene identifiers"
        metabolomic = None
    elif len(numeric) >= MIN_WIDE and not found and not metabolomic:
        # The data-type reading hints genomics only where it is specific to expression data: a
        # library-size signature (CPM, TPM, scaled counts summing near a million, voom). "Log-scale,
        # normalization not recoverable" is any wide table of floats under 25, such as an MRI
        # thickness panel (audit WP14 repair), and hints nothing.
        card = genomics.card(df)
        keys = (card or {}).get("classification", {}).get("keys") or []
        if keys and keys[0] in ("raw_counts", genomics.SHALLOW_COUNTS) and not count_signature(df):
            keys = []  # whole numbers on one scale: a questionnaire, portions, minute counts
        if card and card.get("read") and keys and keys[0] != genomics.LOG_UNKNOWN:
            genomic = (f"{len(numeric):,} measurement columns read as "
                       f"{card['classification']['label']}, an expression matrix")
    if genomic:
        out.append({"lens": "genomics", "because": genomic})
    if metabolomic:
        out.append({"lens": "metabolomics", "because": metabolomic})
    energy = total_energy_column(df)
    if energy is not None:
        out.append({"lens": "dietary", "because": f"`{energy}` reads as total energy intake"})
    recognized = packs._clinical_columns(df)
    if len(recognized) >= 2 and not (genomic or metabolomic):
        out.append({"lens": "clinical",
                    "because": f"{len(recognized)} columns are recognized clinical measurements — "
                               + ", ".join(f"`{c}`" for c in recognized[:3])})
    return out


def contradiction_finding(df: pd.DataFrame, lens: Sequence[str]) -> dict[str, Any] | None:
    """The stated lens against the table (``packs.contradiction``), as a finding the user answers:
    change the lens, or dismiss it to record that the answer stands."""
    from turbotab import packs
    from turbotab.core.stages.finding_words import _finding

    try:
        found = packs.contradiction(df, list(lens))
    except Exception:  # noqa: BLE001 - a check that cannot read the table says nothing
        return None
    if not found:
        return None
    if found.get("kind") == "stated_lens_but_shape_is_an_assay":
        if not any(h["lens"] in ("genomics", "metabolomics") for h in hints(df)):
            return None
    if found.get("kind") == "stated_genomics_but_values_are_not_counts":
        from turbotab.core.detectors import genomics

        card = genomics.card(df)
        if card and card.get("read"):  # log-expression or shallow counts, which the pack misses
            return None
    suggests = ", ".join(str(s) for s in found.get("suggests") or [])
    message = str(found.get("message") or "")
    return _finding(
        "voice::lens_contradiction", "critical",
        "The lens you chose and the table disagree",
        "The lens and the table disagree; change the lens, or record that your answer stands.",
        message + (f" The table reads more like {suggests}." if suggests else ""),
        ("The lens changes what is looked for and what is assumed, such as whether a blank is "
         "below a detection limit. If the table is right and the lens is wrong, those assumptions "
         "are wrong; if the lens is right, dismissing this records that it stands."),
        [], "lens", "Change the lens",
        {"status": "CONVENTION", "source": "DOMAIN_PACKS.md#01 · The opening question"})


__all__ = ["ABUNDANCE_SPREAD", "abundance_spread", "contradiction_finding", "count_signature",
           "gene_vocabulary", "hints", "metabolomics_evidence", "total_energy_column"]
