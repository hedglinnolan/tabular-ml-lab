"""The swarm's controlled vocabulary (SWARM_PLAN §2 and §4), fixed in this one file before any run.

An analyst tags each answer (``driver answer --marker``) and each want (``driver want --marker``)
with one marker name of :data:`MARKERS`, and names each want's kind from :data:`WANT_KINDS`. The
record keeps the tags (``record.Tag``, ``record.Want``) and the engine's readings of the same
markers (``record.Marker``), so a claim's region is read in one vocabulary on both sides.

**The markers** are SWARM_PLAN §4's, on its three axes (the regime, the measurement model, and the
purpose with its loss), each known **live** (read before a decision, so a promoted claim may key on
it), **declared** (quoted from the paper's methods or the NCHS documentation by the intake, §5) or
**post hoc** (known only after a run: it describes evidence and keys nothing):

* the regime, live: the rows, the rows analyzed, the units where rows repeat, the effective size,
  the events per parameter, the predictors by role and their ratio to the rows, the outcome's type
  and its event share, the missingness per column and the complete-case share, the design, the
  time order;
* the measurement model, live: the units as read with each reading's basis, the detection-limit
  shares, the batch and the run-order columns, imputed copies and repeated measurement, the omics
  scale reading; declared: the instrument, the assay or platform, the preprocessing, the units, the
  recall method;
* the purpose and its loss, live: the purpose, the primary score and why, the intended use and its
  thresholds, the estimand, the multiplicity family;
* post hoc: the signal and the result.

**The want kinds** are §2's four: an option, a check, a reading, an eligibility rule the engine
lacks.

The vocabulary's digest (:data:`DIGEST`, the SHA-256 of its canonical JSON) is kept in each run's
record (``record.Run.vocabulary``), so a record says which vocabulary its tags were taken under;
a change to this file after the first run changes the digest, and the records apart.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

Axis = Literal["regime", "measurement", "purpose"]
Timing = Literal["live", "declared", "post_hoc"]


@dataclass(frozen=True)
class MarkerName:
    """One marker: its name (what ``--marker`` takes), its axis, when it is known, and what it is,
    in SWARM_PLAN §4's words."""

    name: str
    axis: str
    timing: str
    means: str


MARKERS: tuple[MarkerName, ...] = (
    # The regime, live.
    MarkerName("rows", "regime", "live", "the rows"),
    MarkerName("rows_analyzed", "regime", "live", "the rows analyzed"),
    MarkerName("repeated_units", "regime", "live", "the units where rows repeat"),
    MarkerName("effective_size", "regime", "live", "the effective size"),
    MarkerName("events_per_parameter", "regime", "live", "the events per parameter"),
    MarkerName("predictors_by_role", "regime", "live", "the predictors by role"),
    MarkerName("predictors_per_row", "regime", "live", "the predictors' ratio to the rows"),
    MarkerName("outcome_type", "regime", "live", "the outcome's type"),
    MarkerName("event_share", "regime", "live", "the outcome's event share"),
    MarkerName("missingness", "regime", "live", "the missingness per column"),
    MarkerName("complete_case_share", "regime", "live", "the complete-case share"),
    MarkerName("design", "regime", "live", "the design"),
    MarkerName("time_order", "regime", "live", "the time order"),
    # The measurement model, live.
    MarkerName("units", "measurement", "live", "the units as read, with each reading's basis"),
    MarkerName("detection_limit_share", "measurement", "live", "the detection-limit shares"),
    MarkerName("batch_columns", "measurement", "live", "the batch columns"),
    MarkerName("run_order", "measurement", "live", "the run-order columns"),
    MarkerName("imputed_copies", "measurement", "live", "imputed copies"),
    MarkerName("repeated_measurement", "measurement", "live", "repeated measurement"),
    MarkerName("omics_scale", "measurement", "live", "the omics scale reading"),
    # The measurement model, declared (SWARM_PLAN §5: the intake quotes each).
    MarkerName("instrument", "measurement", "declared", "the instrument"),
    MarkerName("assay_platform", "measurement", "declared", "the assay or platform"),
    MarkerName("preprocessing", "measurement", "declared", "the preprocessing"),
    MarkerName("declared_units", "measurement", "declared", "the units, as declared"),
    MarkerName("recall_method", "measurement", "declared", "the recall method"),
    # The purpose and its loss, live.
    MarkerName("purpose", "purpose", "live", "the purpose"),
    MarkerName("primary_score", "purpose", "live", "the primary score and why"),
    MarkerName("intended_use", "purpose", "live", "the intended use"),
    MarkerName("thresholds", "purpose", "live", "the intended use's thresholds"),
    MarkerName("estimand", "purpose", "live", "the estimand"),
    MarkerName("multiplicity_family", "purpose", "live", "the multiplicity family"),
    # Post hoc: known only after a run.
    MarkerName("signal", "regime", "post_hoc", "the signal"),
    MarkerName("result", "purpose", "post_hoc", "the result"),
)
BY_NAME: dict[str, MarkerName] = {m.name: m for m in MARKERS}
NAMES: tuple[str, ...] = tuple(BY_NAME)
LIVE: tuple[str, ...] = tuple(m.name for m in MARKERS if m.timing == "live")
DECLARED: tuple[str, ...] = tuple(m.name for m in MARKERS if m.timing == "declared")
POST_HOC: tuple[str, ...] = tuple(m.name for m in MARKERS if m.timing == "post_hoc")

# SWARM_PLAN §2: what a want names the engine lacks.
WANT_KINDS: tuple[str, ...] = ("option", "check", "reading", "eligibility_rule")

VERSION = 1
DIGEST = hashlib.sha256(json.dumps(
    {"version": VERSION, "markers": [asdict(m) for m in MARKERS], "want_kinds": list(WANT_KINDS)},
    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def unknown(name: str, allowed: tuple[str, ...]) -> str:
    """Why ``name`` is not in the vocabulary, listing what is, in the words a refusal says."""
    return f"{name!r} is not in the vocabulary; it is one of {', '.join(allowed)}."


__all__ = ["Axis", "BY_NAME", "DECLARED", "DIGEST", "LIVE", "MARKERS", "MarkerName", "NAMES",
           "POST_HOC", "Timing", "VERSION", "WANT_KINDS", "unknown"]
