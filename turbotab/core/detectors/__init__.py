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
=====================================  ===========================================  ==============

:func:`pack_findings` runs every other pack detector exactly as ``packs.findings`` does (same
order, same failure handling), so a superseded detector is never computed twice.

Beside the findings: :mod:`.lenses` (the profile stage's lens hints and the lens contradiction
finding), :mod:`.orientation` (the oriented stage's reading, names before shape),
:mod:`.repeats` (the structure stage's repeats reading) and :mod:`.bins` (the histogram bins both
histogram implementations share).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import pandas as pd

from turbotab.core.detectors import codes, scales


def _superseding(units: Any = None, codings: Any = None
                 ) -> dict[str, Callable[[pd.DataFrame], list[dict[str, Any]]]]:
    """Legacy detector name -> the reading served in its place (a list: one may become two).
    ``units`` are the columns' recorded units (``set_column_unit``: an age in months);
    ``codings`` the sex columns' codings the user confirmed (BLUEPRINT §14.3)."""
    from turbotab.core.detectors import assay, genomics, plausibility

    def one(fn: Callable[[pd.DataFrame], dict[str, Any] | None]) -> Callable[[pd.DataFrame], list]:
        return lambda df: [f for f in [fn(df)] if f]

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
        "_pooled_qc": one(_pooled_qc_reading),
    }


def _pooled_qc_reading(df: pd.DataFrame) -> dict[str, Any] | None:
    from turbotab.core.reference_rows import pooled_qc_finding

    return pooled_qc_finding(df)


def pack_findings(df: pd.DataFrame, lens: Sequence[str], units: Any = None,
                  codings: Any = None) -> list[dict[str, Any]]:
    """``packs.findings``, with the superseded detectors read here instead."""
    from turbotab import packs

    out: list[dict[str, Any]] = []
    if df is None or df.empty:
        return out
    replaced = _superseding(units, codings)
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
