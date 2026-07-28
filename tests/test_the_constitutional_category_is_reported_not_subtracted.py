"""The L16 ruling on `GUIDED-018`, pinned.

> The prereg defines *irrelevant* as "absent from the decision inventory and
> cites no finding" — both conjuncts hold for grain, so the recorded numbers are
> correct and they stand permanently. Do not add a `grain::` key; that smuggles
> a new category into an old bucket. Instead name the category: report
> `constitutional` and `irrelevant_net = irrelevant − constitutional` alongside,
> **never instead**. The threshold keeps binding on literal `irrelevant`.

Three things have to stay true, and each of them is a way the ruling could be
quietly undone:

1. **`irrelevant_questions` does not move.** Labeling a question with a clause
   must not change the literal count. If it did, the category would have become
   the bucket.
2. **`constitutional` needs all three conjuncts.** A question that names a
   clause *and* cites a finding, or *and* settles an inventory key, is already
   counted where it belongs. Letting either count here would make the field a
   laundering mechanism: label anything, subtract it.
3. **Both readings are published.** A metric computed and not reported is a
   metric nobody can disagree with.

The fourth origin is the thing worth remembering. The harness assumed every
legitimate question comes from a finding; clause §02 introduced one asked
*because the app cannot know*, and §04's eligibility question is the second.
That is why the fix is a category and not a key — a denominator that gains an
entry per constitutional clause moves every loop.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab.measure import (DecisionRequirement, Measurement,   # noqa: E402
                              QuestionRecord)


def _m(*questions: QuestionRecord, required=("choose_target",)) -> Measurement:
    return Measurement(
        door="guided", dataset="fixture", n_rows=100, n_columns=4,
        required=[DecisionRequirement(key=k, reason="fixture") for k in required],
        questions=list(questions))


def _q(key, **kw) -> QuestionRecord:
    base = dict(key=key, label=key, door="guided", step="data")
    base.update(kw)
    return QuestionRecord(**base)


def test_labeling_a_question_with_a_clause_does_not_move_the_literal_count():
    """The whole ruling in one assertion.

    Same two questions, same inventory, one of them carrying a clause label.
    `irrelevant_questions` is identical either way, because the prereg's
    definition is about origin and inventory membership and a label changes
    neither.
    """
    unlabeled = _m(_q("choose_target", covers="choose_target"), _q("state_grain"))
    labeled = _m(_q("choose_target", covers="choose_target"),
                 _q("state_grain", clause="lockbox-02"))

    assert unlabeled.irrelevant_questions == 1
    assert labeled.irrelevant_questions == 1, (
        "labeling a question constitutional moved the literal count — the "
        "category became the bucket, which is what the ruling forbids")
    assert labeled.constitutional == 1 and unlabeled.constitutional == 0
    assert labeled.irrelevant_net == 0 and unlabeled.irrelevant_net == 1


def test_a_clause_label_alone_is_not_enough_to_be_constitutional():
    """All three conjuncts, each checked by removing it.

    Without this the field is a laundering mechanism: any question can name a
    clause and vanish from `irrelevant_net`.
    """
    # Names a clause, cites no finding, settles no key — the real case.
    assert _m(_q("state_grain", clause="lockbox-02")).constitutional == 1

    # Cites a finding: it already has an origin the harness recognizes.
    assert _m(_q("repair::x", clause="lockbox-02",
                 triggering_finding="x")).constitutional == 0, (
        "a findings-driven question was counted constitutional, so it would be "
        "subtracted twice over")

    # Settles an inventory key: already in the denominator.
    assert _m(_q("choose_target", clause="lockbox-02",
                 covers="choose_target")).constitutional == 0, (
        "a question that covers a required decision was counted constitutional")

    # Skipped, and a pull affordance: neither was asked.
    assert _m(_q("state_grain", clause="lockbox-02",
                 skipped=True)).constitutional == 0
    assert _m(_q("look::x", clause="lockbox-02", mode="pull")).constitutional == 0


def test_both_readings_are_published_together():
    """A metric computed and not reported is a metric nobody can disagree with.

    The pair travels in `to_dict`, which is what the recorded result and the
    printed table both read, so the literal count cannot become the only figure
    anybody sees — nor the net one.
    """
    metrics = _m(_q("choose_target", covers="choose_target"),
                 _q("state_grain", clause="lockbox-02")).to_dict()["metrics"]
    assert metrics["irrelevant_questions"] == 1
    assert metrics["constitutional"] == 1
    assert metrics["irrelevant_net"] == 0


def test_the_net_reading_is_floored_at_zero_like_the_literal_one():
    """Symmetry with `irrelevant_questions`, which is floored for a stated
    reason: asking FEWER questions than the data requires is a different
    failure — silence — and `coverage` catches it. A negative net would let a
    door bank credit for that."""
    m = _m(_q("state_grain", clause="lockbox-02"),
           required=("choose_target", "repair::a", "repair::b"))
    assert m.irrelevant_questions == 0
    assert m.constitutional == 1
    assert m.irrelevant_net == 0, "the net reading went negative"
