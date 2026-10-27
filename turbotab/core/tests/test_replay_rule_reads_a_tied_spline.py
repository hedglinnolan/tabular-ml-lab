"""A fresh draw's replay reads Harrell's rule on a spline whose values are too tied for its knots
as the size it reads (the verifier's T2-C-V2; the orchestrator's ruling (e) on the swarm's tail).

The declaration-time tie check (``exposure_form._knots_place_on_the_analyzed_rows``) ran inside
``complete_spec``, the rule's own reading, so the replay's :func:`replay.rule_forms`, which reads
the rule through ``complete_spec`` and takes any refusal as a size not read, recorded no size on a
draw whose values cannot place the knots, and its sentence said "the rule reads no effective
sample size" where the rule read 1,000. The check now runs where a spline is declared (the
``set_exposure_form`` and ``set_forms`` completions), so the rule's reading states the size it
reads, and the rebuild goes on to the fit, which refuses the spline in ``spline_too_tied``'s own
words (``test_spline_tie_refusal``).

The construction is ``test_spline_tie_refusal``'s WIDE column: 1,000 analyzed rows, a numeric
outcome recorded on each, so the effective sample size is 1,000 (RMS §2.4.6: the number of rows
for a continuous outcome) and Harrell's rule gives k = 5 (from 100); its type-7 quantiles place
one distinct knot, so the declaration is refused. The rule reads no size where the rows are not
read: there the replay still says so and refuses.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turbotab.core import decisions as d
from turbotab.core.export import replay
from turbotab.core.methods import exposure_form as ef
from turbotab.core.stages import build_graph
from turbotab.core.tests.test_spline_tie_refusal import WIDE, _answer_ctx, _Store

SIZE = 1000.0  # every analyzed row records the numeric outcome
K = 5  # Harrell's rule from an effective sample size of 100


def _declared(said: float, knots: int) -> dict:
    return {"form": "spline", "knots": knots, "knots_rule": "harrell", "n_effective": said}


def test_the_rule_states_the_size_it_reads_on_a_column_too_tied_for_its_knots():
    ctx, _ = _answer_ctx(WIDE)
    assert len(WIDE) == SIZE
    out = ef.complete_spec("x", _declared(SIZE, K), ctx)
    assert (out["knots"], out["n_effective"], out["knots_rule"]) == (K, SIZE, "harrell")


def test_the_declaration_on_that_column_is_still_refused():
    ctx, _ = _answer_ctx(WIDE)
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, ctx)
    assert refused.value.code == "spline_too_tied"
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_forms", "forms": {"x": {"form": "spline", "knots": 4}}}, ctx)
    assert refused.value.code == "spline_too_tied"


class _Service:
    """What :func:`replay.rule_forms` reads of the server: the log, the engine's graph and the
    decision context of the rows as the log ends."""

    def __init__(self, records: list, ctx: SimpleNamespace):
        self._records, self._ctx = records, ctx
        self.engine = SimpleNamespace(graph=build_graph())

    def log(self, pid: str) -> SimpleNamespace:
        return SimpleNamespace(records=lambda: list(self._records))

    def decision_context(self, pid: str) -> SimpleNamespace:
        return self._ctx


def _read(tmp_path, said: float, knots: int, *, rows: str = "all") -> replay.RuleRead:
    """The spline of `x` recorded with k = ``knots`` on ``said``, read again on the WIDE draw."""
    ctx, frame = _answer_ctx(WIDE, rows=rows)
    state = ctx["state"]
    spec = d.ExposureFormSpec(**_declared(said, knots), scale=ef.transform_signature(state, "x"),
                              unit=ef.scale_words(state, "x"))
    state = state.model_copy(update={"exposure_forms": {"x": spec}})
    log = d.DecisionLog(tmp_path / "decisions.jsonl")
    log.append(d.SetExposureForm(column="x", **spec.model_dump()))
    n = len(frame)
    context = SimpleNamespace(state=state, artifact=lambda stage: None, sealed=None,
                              store=lambda: _Store(frame), task="regression",
                              analyzed=(lambda: np.arange(n)) if rows == "all" else (lambda: None))
    (read,) = replay.rule_forms(_Service(log.records(), context), "p")
    return read


def test_the_replay_reads_that_size_and_keeps_the_record_that_states_it(tmp_path):
    read = _read(tmp_path, SIZE, K)
    assert (read.read, read.k, read.said) == (SIZE, K, SIZE)
    assert replay.rule_verdict(read) == ("kept", None)


def test_a_record_of_another_size_is_replayed_at_the_size_the_rule_reads(tmp_path):
    """Recorded on 80 rows (k = 4): on this draw the rule reads 1,000 (k = 5), and says so."""
    read = _read(tmp_path, 80.0, 4)
    verdict, said = replay.rule_verdict(read)
    assert verdict == "refilled"
    assert said == (
        "Record 1 (set_exposure_form) gives `x` a spline whose k is Harrell's rule on the "
        "analyzed rows: k = 4 on an effective sample size of 80 on the recorded draw, and on "
        "this draw the rule reads 1,000 (k = 5) on the rows as the log ends. Its k was left to "
        "the rule, so the record is replayed at the rule's k on this draw, stating k = 5 on an "
        "effective sample size of 1,000 in place of k = 4 on an effective sample size of 80.")


def test_where_the_rows_are_not_read_the_replay_still_says_no_size_is_read(tmp_path):
    read = _read(tmp_path, SIZE, K, rows="none")
    assert read.read is None
    verdict, said = replay.rule_verdict(read)
    assert verdict == "refused"
    assert "on this draw the rule reads no effective sample size on the rows as the log ends" in said
