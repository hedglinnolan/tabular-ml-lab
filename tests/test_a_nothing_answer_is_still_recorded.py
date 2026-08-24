""""Nothing to do here" is an answer, and the record has to carry it.

`DESIGN_LANGUAGE.md` §09, the recorded-absence rule: *the absence of a
restriction is a claim, and a claim needs a record.*

The rule was rediscovered three times before it was written down — at §03 for
the seal's basis, at §06 for feature selection, and at §04 for eligibility — and
each time the symptom was the same: **a step that concluded nothing and a step
nobody reached rendered identically.** This file makes the pattern executable so
it stops being rediscovered, and so a fourth step gets it by inheritance rather
than by somebody remembering.

The seal is the sharpest case and the reason it is a rule. `group_col: None` was
what a *verified cross-sectional* seal and a *failed detection* both produced, so
a consumer could not tell "we checked, and rows do not repeat" from "we could not
tell". Two different claims rendering as one.

**Deliberately about the RECORD, not about the UI.** A door may render the
nothing-answer however it likes; what it may not do is leave the decision list
unable to say the question was answered.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import (eligibility as E, engine, grain as G,          # noqa: E402
                      selection as S)
from turbotab.project import AnalysisProject                         # noqa: E402


def _study(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "glucose": rng.normal(95, 12, n),
        "outcome": rng.integers(0, 2, n),
    })


def _project() -> AnalysisProject:
    p = AnalysisProject.from_dataframe(_study(), "t")
    p.set_target("outcome", "classification", "high", [])
    return p


# Each entry: the question, the call that answers it with "nothing", the
# decision kind that must appear, and what the missing record would be
# indistinguishable from. Adding a step with a nothing-answer means adding a row
# here — that is the point of the table.
NOTHING_ANSWERS = [
    pytest.param(
        "eligibility",
        lambda p: (p.set_grain(G.ONE_ROW_PER_PERSON), p.set_eligibility(E.EVERYONE)),
        "set_eligibility",
        "the eligibility question never being asked",
        id="eligibility-everyone"),
    pytest.param(
        "feature selection",
        lambda p: p.set_selection(None),
        "set_selection",
        "the selection step never being reached",
        id="selection-every-column"),
    pytest.param(
        "the features step",
        lambda p: p.settle_features(skipped=True),
        "settle_features",
        "a features step nobody opened",
        id="features-skipped"),
]


@pytest.mark.parametrize("question,answer,kind,otherwise", NOTHING_ANSWERS)
def test_the_nothing_answer_leaves_a_record(question, answer, kind, otherwise):
    p = _project()
    before = len(p.decisions)
    answer(p)

    kinds = [d.kind for d in p.decisions]
    assert kind in kinds, (
        f"answering '{question}' with the nothing-option left no `{kind}` "
        f"decision, so it is indistinguishable from {otherwise}")
    assert len(p.decisions) > before

    said = next(d for d in p.decisions if d.kind == kind)
    assert said.text.strip(), (
        f"the `{kind}` decision carries no sentence. A record a reader cannot "
        "read is the absence again, one level down — the methods section needs "
        "'no exclusion criteria were applied', not an empty field.")


def test_a_verified_cross_sectional_seal_says_so_rather_than_leaving_a_field_empty():
    """§03's case, and the one the rule was extracted from.

    `group_col: None` is what BOTH a verified cross-sectional seal and an
    undetermined one produce. If the basis did not distinguish them, a consumer
    reading the lockbox could not tell "we checked, and rows do not repeat" from
    "we could not tell" — two different claims, one rendering.
    """
    verified = _project()
    verified.set_grain(G.ONE_ROW_PER_PERSON)
    unknown = _project()
    unknown.set_grain(G.NOT_SURE)

    assert verified.grain["group_col"] is None
    assert unknown.grain["group_col"] is None, (
        "this test no longer exercises the ambiguity it was written for")
    assert verified.grain["basis"] != unknown.grain["basis"], (
        "a verified cross-sectional seal and an undetermined one record the "
        "same thing, so the empty group column is doing the talking")


def test_the_nothing_answer_is_not_the_same_sentence_as_the_something_answer():
    """A record that says the same thing either way is a record of nothing.

    Cheap to get wrong: a single "eligibility settled" sentence would satisfy
    the check above and carry no information at all.
    """
    everyone = _project()
    everyone.set_grain(G.ONE_ROW_PER_PERSON)
    everyone.set_eligibility(E.EVERYONE)

    restricted = _project()
    restricted.set_grain(G.ONE_ROW_PER_PERSON)
    restricted.set_eligibility(E.RESTRICTED, column="age", minimum=30,
                               reason="The study is about adults over 30.")

    a = next(d for d in everyone.decisions if d.kind == "set_eligibility")
    b = next(d for d in restricted.decisions if d.kind == "set_eligibility")
    assert a.text != b.text, "both answers produce the same sentence"

    none_spec = _project()
    none_spec.set_selection(None)
    some_spec = _project()
    some_spec.set_selection(
        S.declare("mutual_info", "outcome", ["age", "glucose"], n_features=1))
    c = next(d for d in none_spec.decisions if d.kind == "set_selection")
    e = next(d for d in some_spec.decisions if d.kind == "set_selection")
    assert c.text != e.text, "both selection answers produce the same sentence"


def test_the_nothing_answer_survives_the_save_file():
    """The record is only as durable as the archive, and a dropped
    nothing-answer restores as "never asked" — which is the failure, not a
    lesser version of it."""
    from turbotab import archive
    p = _project()
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    p.set_selection(None)
    p.settle_features(skipped=True)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])

    back = archive.from_bytes(archive.to_bytes(p))
    assert back.eligibility is not None and back.eligibility["answer"] == E.EVERYONE
    assert back.features_settled is True
    kinds = [x.kind for x in back.decisions]
    for kind in ("set_eligibility", "set_selection", "settle_features"):
        assert kind in kinds, f"`{kind}` did not survive the archive"
