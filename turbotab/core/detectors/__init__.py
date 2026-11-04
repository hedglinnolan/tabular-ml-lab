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

Where the lens serves the pooled-QC finding, the impossible-values card (``pack::clinical::
impossible_…``) still counts every row, as its repairs act, and its detail counts the values on the
rows that finding names apart from the others'. Each of its columns read after a unit's conversion
carries its band in the unit the column is written in, the band its repairs compare the values to.

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


def _superseding(units: Any = None, codings: Any = None, target: str | None = None,
                 qc_served: bool = False
                 ) -> dict[str, Callable[[pd.DataFrame], list[dict[str, Any]]]]:
    """Legacy detector name -> the reading served in its place (a list: one may become two).
    ``units`` are the columns' recorded units (``set_column_unit``: an age in months);
    ``codings`` the sex columns' codings the user confirmed (BLUEPRINT §14.3); ``target`` the
    outcome, where one is chosen; ``qc_served`` whether the lens runs the pooled-QC reading (the
    metabolomics pack's), so that the findings page serves it. The served pooled-QC reading is
    made once per table here, the left-censoring reading sets aside the rows it names, and where
    the lens serves it the impossible-values card counts the values on those rows apart."""
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

    def bands(df: pd.DataFrame) -> dict[str, Any] | None:
        found = plausibility.impossible_vs_extreme_finding(df, units=units, codings=codings)
        if found is not None:
            _as_written(found, df)
            if qc_served:
                _impossible_apart(found, df, _reference_rows(df, pooled_qc(df)).to_numpy())
        return found

    return {
        "_ordinal_declared": scales.findings,      # both survey findings, from one block reading
        "_survey_sentinel_codes": lambda df: [],
        "_acquisition_order": one(assay.run_order_finding),
        "_redundancy": one(assay.redundancy_finding),
        # INBOX:1033, PP2 LC1: read over every row, as its repairs act, the pooled-QC rows' zeros
        # (an export's missing glucose) were counted among values "no living adult is expected to
        # show" beside a finding that says those rows are not participants; the detail now counts
        # them apart (``_impossible_apart``), and the repairs' consequences say both counts. Each
        # band is also given in the unit written (``_as_written``): the repairs compared the values
        # as written to the band in the card's unit, so under a conversion (a `cholesterol` in
        # mg/dL against 0.3–40 mmol/L) they blanked every value where the card counted 11.
        "_clinical_impossible_vs_extreme": one(bands),
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


def _as_written(found: dict[str, Any], df: pd.DataFrame) -> None:
    """Each column of the impossible-values card read after a unit's conversion, in place: its
    band in the unit the column is written in (``band_as_written``) and that unit
    (``unit_as_written``), the band and unit its repairs compare the values to and label their
    marks with (``repairs._offer_impossible``). The card reads a value as written times the
    conversion factor its unit reading gives (``plausibility._read``: ``ml.clinical_units.
    infer_unit`` on the column's values, read here the same way), so the band as written is the
    card's band divided by that factor: a `cholesterol` in mg/dL is read against 0.3–40 mmol/L,
    11.6–1544 mg/dL as written. A column read in the card's own unit (a factor of 1) carries
    neither, and its repairs read the card's band."""
    from ml.clinical_units import infer_unit

    for e in (found.get("params") or {}).get("columns") or []:
        if not isinstance(e, dict) or not e.get("impossible_band") \
                or e.get("column") not in df.columns:
            continue
        column = e["column"]
        present = pd.to_numeric(df[column], errors="coerce").dropna()
        unit = infer_unit(str(column), present) or {}
        factor = unit.get("conversion_factor")
        if not factor or float(factor) == 1.0:
            continue
        low, high = (float(v) for v in e["impossible_band"])
        e["band_as_written"] = [low / float(factor), high / float(factor)]
        if unit.get("inferred_unit"):
            e["unit_as_written"] = str(unit["inferred_unit"])


def _impossible_apart(found: dict[str, Any], df: pd.DataFrame, qc: np.ndarray) -> None:
    """The impossible-values card's detail, with each count it states of a column its repairs act
    on followed by how many of those values are on the non-QC rows and how many on the pooled-QC
    rows ``qc``, in place: the lead column (the age-unit card: the first two). Counted as the
    repairs count (``repairs._offer_impossible``: a value as written outside its column's band in
    the unit written, :func:`_as_written`), and said only where the two counts sum to the card's
    own: otherwise no split is said. Nothing is said where the QC rows hold none."""
    from turbotab.core.repairs import _outside

    params = found.get("params") or {}
    entries = [e for e in params.get("columns") or []
               if isinstance(e, dict) and e.get("impossible_band") and e.get("column") in df.columns]
    n_qc, n_other = int(qc.sum()), int((~qc).sum())
    detail = str(found.get("detail") or "")
    at = 0
    for e in entries[:2] if params.get("age_unit_question") else entries[:1]:
        column, n = str(e["column"]), int(e.get("n_impossible") or 0)
        band = e.get("band_as_written") or e["impossible_band"]
        outside = _outside(df[column], *[float(v) for v in band]).to_numpy()
        on_qc, on_other = int((outside & qc).sum()), int((outside & ~qc).sum())
        named = detail.find(f"`{column}`", at)
        mark = f"(a convention: {e.get('impossible_source')})."
        end = detail.find(mark, named) if named >= 0 else -1
        if not on_qc or on_qc + on_other != n or end < 0:
            continue
        rows = f"the {n_qc:,} pooled-QC {'row' if n_qc == 1 else 'rows'}"
        if on_other:
            said = (f"{on_other:,} of them {'is' if on_other == 1 else 'are'} on the {n_other:,} "
                    f"non-QC rows and {on_qc:,} on {rows}.")
        else:
            said = (f"{'It is' if on_qc == 1 else f'All {on_qc:,} are'} on {rows}, none on the "
                    f"{n_other:,} non-QC rows.")
        at = end + len(mark)
        detail = f"{detail[:at]} {said}{detail[at:]}"
        at += len(said) + 1
    found["detail"] = detail


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
    chosen = packs.normalize_quiet(lens)
    qc_served = any(getattr(d, "__name__", "") == "_pooled_qc"
                    for key in chosen for d in getattr(packs.PACKS.get(key), "detectors", ()))
    replaced = _superseding(units, codings, target, qc_served)
    for key in chosen:
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
