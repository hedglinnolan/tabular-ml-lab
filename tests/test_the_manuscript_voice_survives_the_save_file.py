"""Manuscript voice must survive a save/restore, not decay into coach voice.

`Insight.manuscript_text` exists for exactly one reason: coaching voice
addresses the ANALYST ("consider dropping…", "a reviewer would question…")
and a manuscript addresses the REVIEWER, stating facts about the study. When
`manuscript_text` is present, `discussion_points_for_manuscript()` prefers it
and falls back to a cleaned `finding` only when it is empty.

`Insight.to_dict()` enumerated its fields by hand and `manuscript_text` was not
among them. So the field was correct in memory and absent on disk: every
session save/restore — and every Record round-trip through a job queue —
silently stripped it, and the Discussion's Limitations came back in coach voice
in a manuscript the author was about to submit. The existing
`test_manuscript_text_preferred_over_finding` passes because it never
serializes.

`InsightLedger.from_list()` had the second half of the same defect: it used
`add()`, which skips duplicates, so a save file containing two entries with one
id silently kept the first and dropped the second.

Findings: RECORD-001, STATE-056.
"""
from __future__ import annotations

import json

import pytest

from utils.insight_ledger import Insight, InsightLedger

# Coaching register: second person, advisory verbs, addressed to the analyst.
COACH_VOICE = "Consider dropping `crp` — a reviewer would question it."
# Manuscript register: third person, past tense, a fact about the study.
MANUSCRIPT_VOICE = (
    "C-reactive protein was measured in 61% of participants; the remainder "
    "were excluded from the adjusted model."
)


def limitation_insight(insight_id: str = "eda_missing_crp") -> Insight:
    return Insight(
        id=insight_id,
        source_page="EDA",
        category="data_quality",
        severity="warning",
        finding=COACH_VOICE,
        implication="The adjusted model is fit on a subset.",
        manuscript_text=MANUSCRIPT_VOICE,
        acknowledged=True,
    )


# ── the field is on the wire at all ──────────────────────────────────────

def test_to_dict_carries_manuscript_text():
    d = limitation_insight().to_dict()
    assert "manuscript_text" in d, (
        "to_dict dropped manuscript_text — the field is correct in memory and "
        "absent on disk")
    assert d["manuscript_text"] == MANUSCRIPT_VOICE


def test_to_dict_covers_every_declared_field():
    """The class fix: a hand-written field list is what lost this one.

    Any field added to the dataclass from here on is serialized or this fails.
    """
    declared = set(Insight.__dataclass_fields__)
    written = set(limitation_insight().to_dict())
    assert declared == written, (
        f"to_dict and the dataclass disagree; missing from to_dict: "
        f"{sorted(declared - written)}; extra: {sorted(written - declared)}")


# ── the round trip the Record layer's whole contract rests on ────────────

def test_an_insight_round_trips_through_to_dict_and_from_dict():
    original = limitation_insight()
    restored = Insight.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored.manuscript_text == MANUSCRIPT_VOICE


def test_the_ledger_round_trips_through_json():
    """to_list -> JSON -> from_list -> to_list is identity.

    JSON is in the middle deliberately: `utils/session_manager.py` writes the
    ledger to `coaching.json`, so a field that survives a dict copy but not a
    serialization pass would still be lost.
    """
    ledger = InsightLedger()
    ledger.add(limitation_insight())
    ledger.add(Insight(
        id="eda_skew_alt", source_page="EDA", category="distribution",
        severity="info", finding="Consider a log transform for `bili`.",
        implication="Skewed predictor.",
        manuscript_text="Bilirubin was log-transformed before modeling.",
        acknowledged=True,
    ))

    before = ledger.to_list()
    after = InsightLedger.from_list(json.loads(json.dumps(before, default=str))).to_list()
    assert after == before


# ── what the reader actually sees ────────────────────────────────────────

def test_limitations_stay_in_manuscript_voice_after_a_save_and_restore():
    ledger = InsightLedger()
    ledger.add(limitation_insight())

    fresh = ledger.discussion_points_for_manuscript()["limitations"]
    assert MANUSCRIPT_VOICE in fresh

    restored = InsightLedger.from_list(
        json.loads(json.dumps(ledger.to_list(), default=str)))
    points = restored.discussion_points_for_manuscript()["limitations"]

    assert MANUSCRIPT_VOICE in points, (
        "the Discussion lost its manuscript-register phrasing across a "
        "save/restore")
    assert COACH_VOICE not in points, (
        "the Discussion fell back to the coaching-register finding — the "
        "manuscript now addresses the analyst instead of the reviewer")
    assert not any("consider" in p.lower() or "a reviewer would" in p.lower()
                   for p in points), (
        f"coaching voice reached the Limitations: {points}")


# ── the second half of STATE-056: duplicates are not silently dropped ────

def test_a_save_file_with_two_entries_for_one_id_keeps_the_later_one():
    first = limitation_insight()
    second = limitation_insight()
    second.manuscript_text = "The variable was excluded after the audit."

    restored = InsightLedger.from_list([first.to_dict(), second.to_dict()])

    assert len(restored) == 1
    assert restored.get("eda_missing_crp").manuscript_text == (
        "The variable was excluded after the audit."), (
        "from_list used add(), which skips duplicates, so the later entry in "
        "the save file was silently discarded")


def test_from_list_still_skips_a_malformed_entry_without_taking_the_rest():
    restored = InsightLedger.from_list([
        limitation_insight().to_dict(),
        {"id": "broken"},  # missing required fields
    ])
    assert len(restored) == 1
    assert restored.get("eda_missing_crp") is not None
