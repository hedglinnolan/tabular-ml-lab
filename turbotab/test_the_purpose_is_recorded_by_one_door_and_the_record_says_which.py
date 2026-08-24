"""`AUDIT-011` · the purpose is one door's record, and the seal does not hold it.

## The finding

`DOMAIN_SCIENCE.md` §01.3 — *the prediction/inference fork changes the correct
answer, not just the emphasis* — names five decisions whose advice **inverts**
on one answer: the missing-indicator method, values below the limit of
detection, repeated recalls, an instrument entering item-level or as a scale
score, and features versus compounds. The Guided door asks question 2.5 and
records the answer on `AnalysisProject.purpose`. **The Streamlit workflow has no
purpose field at all**, so none of those decisions can read it there.

## What was false in this repository, and where

`turbotab/resolution.py`'s statement of the module's second rule read:

    at the seal we know the target, the task, the grain, the eligibility and
    **the purpose**

Four of those five are preconditions — `seal_lockbox` refuses without the grain
and the eligibility, and neither the target nor the task can be absent by the
time it is reached. The purpose is not one of them. It is *asked and never
required*, so a project whose user walked past question 2.5 seals with
`purpose is None`, and the sentence listed it beside four things that cannot be
missing. Corrected to the four the seal actually holds, with the divergence
between the two doors stated separately and `AnalysisProject.purpose` named as
the authoritative record.

## What this file does and does not close

It closes the **claim**. It does not close `AUDIT-011`: recording the purpose
where the Streamlit workflow records its other answers means writing
`utils/session_state.py`, which is not this chunk's to write. The row stays
open on that half, and the assertions below are the shape of the test that will
still be true after it lands — the reader/writer counts move, and the second
one is what changes.

## Honesty about the third assertion

`test_the_seal_does_not_claim_to_know_the_purpose` reads a source docstring,
which is `AGENT_ONBOARD` §07 trap 2 — *a guard testing its own description* —
in every respect except one: the two assertions above it **drive** the behavior
that makes the description false. The description is checked only after the
behavior it describes has been observed. That is the `test_the_page_says_what_
the_record_says` shape, and it is stated here rather than left to be noticed.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import purpose as _purpose                       # noqa: E402
from turbotab import resolution as _resolution                 # noqa: E402
from turbotab.project import AnalysisProject                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. The seal is a journey step, so the claim runs against two
#: target shapes. Neither the seal's preconditions nor the purpose question
#: reads the target, so a difference here would itself be the finding.
TARGET_SHAPES = {
    "binary_numeric": ("leaky_sepsis.csv", "sepsis", "classification"),
    "multiclass_string": ("multiclass_stage.csv", "disease_stage",
                          "classification"),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = {
    "continuous": (
        "`seal_lockbox`'s refusals are on `grain` and `eligibility` and read "
        "neither the target nor the task, so a regression target exercises no "
        "branch the two above do not. Undriven, and named rather than assumed."),
    "a Streamlit-door project": (
        "There is no `AnalysisProject` on that side to drive — that absence "
        "IS `AUDIT-011`. The Streamlit half is measured below over source "
        "instead, with a positive control, because there is no object to "
        "interrogate."),
}

#: The three production sites that READ the Streamlit-side purpose. Listed so
#: the positive control below fails loudly if a rename makes the sweep vacuous.
EXPECTED_READERS = {
    "pages/06_Train_and_Compare.py",
    "ml/publication.py",
    "ml/narrative_engine.py",
}

#: Production trees. Test files are excluded — a fixture that hands
#: `model_purpose` to a function is not the workflow recording an answer, and
#: counting one would be `AGENT_ONBOARD` §07 trap 3 exactly.
_PRODUCTION = ("pages", "ml", "utils", "turbotab")


def _sealed(shape, answer_purpose=False):
    """A project sealed the way the Guided door seals one."""
    name, target, task = TARGET_SHAPES[shape]
    df = pd.read_csv(DATA / name)
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    if answer_purpose:
        p.set_purpose("prediction")
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    return p


def _model_purpose_lines():
    """Every production line mentioning `model_purpose`, split read vs write."""
    reads, writes = {}, {}
    for tree in _PRODUCTION:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if path.name.startswith("test_") or "test" in path.parts:
                continue
            rel = str(path.relative_to(ROOT))
            for i, line in enumerate(path.read_text(
                    encoding="utf-8", errors="ignore").splitlines(), 1):
                if "model_purpose" not in line:
                    continue
                if line.lstrip().startswith("#"):
                    continue
                # A WRITE puts the answer somewhere a reader will find it:
                # an assignment, or the key of a dict literal being built.
                if (re.search(r"model_purpose['\"]?\]?\s*=(?!=)", line)
                        or re.search(r"['\"]model_purpose['\"]\s*:", line)):
                    writes.setdefault(rel, []).append((i, line.strip()))
                elif "model_purpose" in line:
                    reads.setdefault(rel, []).append((i, line.strip()))
    return reads, writes


# ═══════════ 1 · the seal does not hold the purpose ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_a_project_seals_without_the_purpose_ever_being_answered(shape):
    """Driven: the seal succeeds with `purpose is None`.

    **Positive control first** (`GUIDED-045`): the purpose must be RECORDABLE,
    or "it is absent at the seal" is a statement about a field that does not
    exist rather than about a question nobody was made to answer.
    """
    answered = _sealed(shape, answer_purpose=True)
    assert answered.purpose == {"answer": "prediction"}, (
        f"{shape}: the Guided door did not record the purpose it was given — "
        f"the control for this sweep is broken, not the sweep")
    assert answered.barrier_raised

    walked_past = _sealed(shape, answer_purpose=False)
    assert walked_past.barrier_raised, (
        f"{shape}: the seal refused, so this says nothing about what it holds")
    assert walked_past.purpose is None, (
        f"{shape}: expected an unanswered purpose; got "
        f"{walked_past.purpose!r}")
    assert walked_past.lockbox["resolution"] is not None, (
        f"{shape}: `resolution.statement` did not run at this seal, so the "
        f"module whose claim is under test was never reached")


# ═══════════ 2 · the Streamlit door has no purpose to read ═══════════

def test_the_streamlit_side_reads_a_purpose_nothing_writes():
    """`AUDIT-011`'s own sentence, measured: readers exist, a writer does not.

    **Positive control first** (`GUIDED-045`): if the sweep found no readers,
    zero writers would be a fact about a spelling rather than about the app.
    """
    reads, writes = _model_purpose_lines()

    assert set(reads) >= EXPECTED_READERS, (
        f"the sweep lost its subject: expected {sorted(EXPECTED_READERS)} to "
        f"read `model_purpose` and found {sorted(reads)}. Zero writers means "
        f"nothing until the readers are found.")
    assert not writes, (
        f"`model_purpose` is now written at {sorted(writes)} — if the "
        f"Streamlit workflow records the purpose, AUDIT-011's other half has "
        f"landed and `resolution`'s docstring must stop calling "
        f"AnalysisProject.purpose the only record.")


def test_the_class_weighting_advisory_can_only_be_the_unanswered_one_there():
    """One of §01.3's five inversions, driven to the branch it cannot leave.

    `ml.imbalance_advice.advice` has exactly one production caller —
    `pages/06_Train_and_Compare.py`, passing `session_state['model_purpose']`,
    which nothing writes. So the branch that inverts is reachable in the
    function and unreachable through that door.
    """
    from ml import imbalance_advice as _imbalance

    unanswered = _imbalance.advice(None)
    inference = _imbalance.advice(_purpose.INFERENCE)

    # POSITIVE CONTROL — the two branches really are different sentences.
    assert unanswered["advisory"] != inference["advisory"], (
        "the purpose changes nothing in this advisory, so its absence costs "
        "nothing and this test is measuring the wrong function")
    assert _imbalance.INFERENCE_EXTRA in inference["advisory"]
    assert _imbalance.INFERENCE_EXTRA not in unanswered["advisory"], (
        "the unanswered branch already carries the inference sentence")
    assert "has not been recorded yet" in unanswered["advisory"], (
        "the unanswered branch no longer says the purpose is unrecorded")


# ═══════════ 3 · and the record says so ═══════════

def test_the_seal_does_not_claim_to_know_the_purpose():
    """The corrected claim. See this file's docstring on why it reads source.

    `resolution.py` rule 2 lists what is held at the seal in order to say what
    is NOT held. A fifth entry that can be `None` — proven `None` above, on two
    target shapes — makes the list assert something the seal does not have.
    """
    doc = _resolution.__doc__ or ""
    inventory = re.search(r"at the seal we know(.*?)(?:—|,? and we do not)",
                          doc, re.S)

    # POSITIVE CONTROL — the sentence under test still exists.
    assert inventory, (
        "`resolution.py` no longer states what is known at the seal; the "
        "claim was deleted rather than corrected, which AUDIT-028's model "
        "rules out")
    held = inventory.group(1)
    for precondition in ("target", "task", "grain", "eligibility"):
        assert precondition in held, (
            f"the seal's inventory dropped {precondition!r}: {held!r}")

    assert "purpose" not in held, (
        "`resolution.py` says the app knows the purpose at the seal. It does "
        "not: `seal_lockbox` refuses only on the grain and the eligibility, "
        "question 2.5 is asked and never required, and the Streamlit door has "
        "no purpose field at all. AUDIT-011. The inventory read: "
        f"{held!r}")


def test_the_record_names_which_door_is_authoritative():
    """`AUDIT-011`'s `act`, second option: say which door owns the answer.

    Driven half: `AnalysisProject` really is the only object holding it, and
    `purpose.CONSUMER` — the sentence the app shows a user about who reads the
    answer — is non-empty, so naming it as authoritative points somewhere.
    """
    p = _sealed("binary_numeric", answer_purpose=True)
    assert p.purpose == {"answer": "prediction"}
    assert p.to_dict()["purpose"] == {"answer": "prediction"}, (
        "the recorded purpose does not reach the project's own record, so no "
        "door is authoritative for it")

    assert _purpose.CONSUMER.strip(), "purpose.CONSUMER is empty"
    doc = _resolution.__doc__ or ""
    assert "AnalysisProject.purpose` is the authoritative record" in doc, (
        "the divergence between the two doors is not stated where the false "
        "claim was. AUDIT-011's act allows either recording the purpose on "
        "the Streamlit side or saying which door is authoritative; the first "
        "is utils/session_state.py's and this chunk does not own it.")
