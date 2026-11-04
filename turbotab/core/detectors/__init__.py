"""The detectors the Next app serves in place of the legacy ones it used to pass through (audit WP14).

The legacy detectors live in modules Classic or the legacy app also import (``ml.import_doctor``,
``turbotab.packs``, ``turbotab.clinical``), so they are wrapped here, never edited (BLUEPRINT §0).
Each replacement keeps the legacy finding's id and shape, so the repair registry, the voice and
the client read it unchanged:

=====================================  ===========================================  ==============
finding                                legacy reading                               here
=====================================  ===========================================  ==============
``sentinel_missing__<col>``            ``import_doctor.check_numeric_sentinels``    :mod:`.codes`
``pack::survey::ordinal_declared``,    ``packs.likert_block`` and                   :mod:`.scales`
``pack::survey::sentinel_codes``       ``survey.sentinel_codes_finding``
``pack::metabolomics::run_order``      ``packs._acquisition_order``                 :mod:`.assay`
``pack::metabolomics::redundancy``     ``packs._redundancy``                        :mod:`.assay`
``pack::clinical::impossible_vs_…``    ``clinical.impossible_vs_extreme_finding``   :mod:`.plausibility`
``pack::genomics::data_type``          ``packs._genomics_data_type``                :mod:`.genomics`
``pack::metabolomics::pooled_qc``      ``packs._pooled_qc``                         ``reference_rows``
``pack::metabolomics::left_censored``  ``packs._left_censored``, on the rows        ``packs``
                                       outside the served pooled-QC finding's
=====================================  ===========================================  ==============

:func:`pack_findings` runs every other pack detector exactly as ``packs.findings`` does (same
order, same failure handling), so a superseded detector is never computed twice. These are the
pack's readings; the findings stage speaks for some of them and adds others
(``stages.findings.speak_for``), and the lens preview counts the findings as it serves them.

Beside the findings: :mod:`.lenses` (the profile stage's lens hints and the lens contradiction
finding), :mod:`.orientation` (the oriented stage's reading, names before shape),
:mod:`.repeats` (the structure stage's repeats reading) and :mod:`.bins` (the histogram bins both
histogram implementations share).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from turbotab.core.detectors import codes, scales


def _superseding(units: Any = None, codings: Any = None, target: str | None = None
                 ) -> dict[str, Callable[[pd.DataFrame], list[dict[str, Any]]]]:
    """Legacy detector name -> the reading served in its place (a list: one may become two).
    ``units`` are the columns' recorded units (``set_column_unit``: an age in months);
    ``codings`` the sex columns' codings the user confirmed (BLUEPRINT §14.3); ``target`` the
    outcome, where one is chosen. The served pooled-QC reading is made once per table here, and
    the left-censoring reading sets aside the rows it names."""
    from turbotab import packs
    from turbotab.core.detectors import assay, genomics, plausibility

    def one(fn: Callable[[pd.DataFrame], dict[str, Any] | None]) -> Callable[[pd.DataFrame], list]:
        return lambda df: [f for f in [fn(df)] if f]

    read: list[tuple[pd.DataFrame, dict[str, Any] | None]] = []

    def pooled_qc(df: pd.DataFrame) -> dict[str, Any] | None:
        if not read or read[0][0] is not df:
            read[:] = [(df, _pooled_qc_reading(df))]
        return read[0][1]

    def on_the_participants(df: pd.DataFrame) -> dict[str, Any] | None:
        return packs._left_censored(df, qc_rows=_reference_rows(df, pooled_qc(df)), target=target)

    return {
        "_ordinal_declared": scales.findings,      # both survey findings, from one block reading
        "_survey_sentinel_codes": lambda df: [],
        "_acquisition_order": one(assay.run_order_finding),
        "_redundancy": one(assay.redundancy_finding),
        "_clinical_impossible_vs_extreme": one(
            lambda df: plausibility.impossible_vs_extreme_finding(df, units=units,
                                                                  codings=codings)),
        "_genomics_data_type": genomics.findings,
        # Audit RO-13 (WP18): the pooled-QC level of a Case/Control/QC label, read by variance,
        # with text that no longer says the rows are already out of the model.
        "_pooled_qc": one(pooled_qc),
        # INBOX:1032: the pack's own pooled-QC reading needs a two-level label, so on a
        # Case/Control/QC label the pack's left-censoring finding read the QC rows' blanks (age,
        # BMI, the outcome, blank there by design); here it reads the rows outside the ones the
        # served pooled-QC finding names, with the outcome set aside where one is handed.
        "_left_censored": one(on_the_participants),
    }


def _reference_rows(df: pd.DataFrame, found: dict[str, Any] | None) -> pd.Series:
    """The rows the served pooled-QC finding ``found`` names, as a mask on ``df``
    (``finding_words.pooled_qc_rows``, the rows the findings stage sets aside); none without it."""
    from turbotab.core.stages.finding_words import pooled_qc_rows

    at = np.zeros(len(df), dtype=bool)
    at[sorted(pooled_qc_rows(df, [found] if found else []))] = True
    return pd.Series(at, index=df.index)


def _pooled_qc_reading(df: pd.DataFrame) -> dict[str, Any] | None:
    from turbotab.core.reference_rows import pooled_qc_finding

    return pooled_qc_finding(df)


def pack_findings(df: pd.DataFrame, lens: Sequence[str], units: Any = None,
                  codings: Any = None, target: str | None = None) -> list[dict[str, Any]]:
    """``packs.findings``, with the superseded detectors read here instead. ``target`` is the
    outcome, where one is chosen: the left-censoring reading sets it aside. The findings stage
    hands none and reads that finding again with it
    (``finding_words.left_censoring_on_the_participants``)."""
    from turbotab import packs

    out: list[dict[str, Any]] = []
    if df is None or df.empty:
        return out
    replaced = _superseding(units, codings, target)
    for key in packs.normalize_quiet(lens):
        pack = packs.PACKS.get(key)
        if pack is None:
            continue
        for detector in pack.detectors:
            name = getattr(detector, "__name__", "?")
            try:
                found = replaced[name](df) if name in replaced else [detector(df)]
            except Exception:
                from turbotab import devchecks
                devchecks.swallowed(
                    f"packs.{key}::{name}", packs._last_exception(),
                    "this pack detector found nothing, and would have been indistinguishable from "
                    "one that legitimately found nothing")
                continue
            out.extend(f for f in found if f)
    # Two readings of one question: the naming census found no pooled-QC label (`pooled QC`
    # escapes its patterns) while the variance reading found the QC rows. The rows are there, so
    # "I couldn't find any pooled QC samples ... can't compute drift correction" is false beside
    # it, and after QC-RLSC ran (MS7 repair).
    ids = {str(f.get("id")) for f in out}
    if "pack::metabolomics::pooled_qc" in ids:
        out = [f for f in out if str(f.get("id")) != "pack::metabolomics::no_pooled_qc"]
    return out


def structural(findings: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The structural stream with its code findings read by :mod:`.codes`."""
    return codes.supersede(findings, frame)


#: The survey pack's one reframing, restated over this package's block reading: the legacy one
#: asks ``packs.likert_block``, which misses a floor-heavy PHQ-9 or a GAD-7.
SURVEY_WIDE_NOTE = ("These are the items of one instrument, not one quantity measured several "
                    "times. Items are combined by scoring the scale, which is a decision about the "
                    "instrument, not by reshaping the table.")


#: The genomics pack's own reframing note (``turbotab.packs``, GENOMICS ``reframings``), which it
#: applies only to a raw count matrix (``count_matrix``).
GENOMICS_WIDE_NOTE = ("These are different genes, not one gene measured several times. Reshaping "
                      "to long format would rebuild what a row is.")


def _expression_matrix(frame: pd.DataFrame) -> bool:
    """Whether the data-type card reads the table as an expression matrix (audit IN-14): CPM/TPM,
    FPKM, VST, log-expression, scaled or shallow counts, the matrices the pack's count-only
    reframing misses."""
    from turbotab.core.detectors import genomics

    try:
        found = genomics.card(frame)
    except Exception:  # noqa: BLE001 - a card that cannot read the table reframes nothing
        return False
    return bool(found and found.get("read") and not found.get("out_of_scope"))


def reframe(findings: list[dict[str, Any]], lens: Sequence[str], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """``packs.reframe``, with the wide-shape reframing read from this package: the survey lens's
    from :mod:`.scales`, and the genomics lens's from :mod:`.genomics`'s data-type card (the pack
    reframes a raw count matrix only, so a CPM, log-CPM or VST matrix was told its genes "look
    like repeated measures of one quantity")."""
    from turbotab import packs

    out = packs.reframe(findings, lens, frame)
    chosen = packs.normalize_quiet(lens)
    readings = []
    if "survey" in chosen:
        readings.append(("survey", SURVEY_WIDE_NOTE, lambda: bool(scales.blocks(frame))))
    if "genomics" in chosen:
        readings.append(("genomics", GENOMICS_WIDE_NOTE, lambda: _expression_matrix(frame)))
    for key, note, reads in readings:
        wide = [f for f in out if f.get("id") == "wide_repeated_measures"
                and key not in (f.get("reframed_by") or [])]
        if not wide or not reads():
            continue
        for f in wide:
            f["severity"] = "info"
            f["fix_kind"] = "none"
            f["fix_label"] = ""
            f["reframe_note"] = note
            f["title"] = "The wide shape is expected here"
            f["reframed_by"] = [*(f.get("reframed_by") or []), key]
    return out


def own_findings(frame: pd.DataFrame, lens: Sequence[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The app's own findings from this package, as ``(raw, finding)``: the lens contradiction."""
    from turbotab.core.detectors import lenses

    found = lenses.contradiction_finding(frame, lens)
    return [({"confidence": "high"}, found)] if found else []


__all__ = ["own_findings", "pack_findings", "reframe", "structural"]
