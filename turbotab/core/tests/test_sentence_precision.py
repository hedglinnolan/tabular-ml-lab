"""W1-R round 2, R-f 20 · a number in a sentence is printed as the results table prints it.

The walkthrough read "CV MSE of 1033.261" and "held-out MSE 1133.621" beside a table that shows
1,033 and 1,134 (``turbotab/frontend/src/components/stage/format.ts``, ``fmtNum``: three
significant digits, a whole number with thousands separators from 1,000, a true minus). The rule is
written out here as the table's code states it, independently of the engine's formatter, and the
sentences are checked against it: the selection's BBC-CV sentence, the opening's record and the
reported result.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

from turbotab.core import voice
from turbotab.core.decisions import OpenSeal, ProjectState
from turbotab.core.models.selection import OutOfFold, selection_optimism


def js_string(x: float) -> str:
    """JavaScript's ``String(x)`` for a finite nonzero number, as ECMAScript's Number::toString
    (radix 10) writes it out: the shortest digits ``s`` that read back as ``x`` (the digits Python's
    ``repr`` gives), ``k`` of them, with ``x = 0.s × 10^n``; in decimal notation while
    ``-6 < n <= 21``, else as ``d.ddde±e`` with ``e = n − 1``."""
    _sign, digits, exponent = Decimal(repr(abs(x))).as_tuple()
    n = len(digits) + exponent
    s = "".join(map(str, digits)).rstrip("0")
    k = len(s)
    if k <= n <= 21:
        body = s + "0" * (n - k)
    elif 0 < n <= 21:
        body = f"{s[:n]}.{s[n:]}"
    elif -6 < n <= 0:
        body = "0." + "0" * -n + s
    else:
        e = n - 1
        body = (s if k == 1 else f"{s[0]}.{s[1:]}") + f"e{'+' if e >= 0 else '-'}{abs(e)}"
    return ("-" if x < 0 else "") + body


def table(v: float) -> str:
    """``fmtNum(v)``: ``Math.round(v).toLocaleString("en-US")`` from 1,000, else
    ``String(+v.toPrecision(3))`` (a number rounded to three significant digits, a tie away from
    zero, printed as JavaScript prints a number, :func:`js_string`); its first minus a true one, as
    ``replace("-", "−")`` makes it."""
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        text = f"{int(math.floor(v + 0.5)):,}"
    else:
        exact = Decimal(v)
        rounded = float(exact.quantize(Decimal(1).scaleb(exact.adjusted() - 2),
                                       rounding=ROUND_HALF_UP))
        text = js_string(rounded)
    return text.replace("-", "−", 1)


@pytest.mark.parametrize("value, shown", [(1033.261, "1,033"), (1133.621, "1,134"),
                                          (0.15897, "0.159"), (33.66, "33.7"), (0.0712, "0.0712"),
                                          (-0.0123456, "−0.0123"), (12345.6, "12,346"), (0.0, "0"),
                                          (1.5, "1.5"), (999.6, "1000")])
def test_the_table_rule_written_out(value, shown):
    """The reference itself, on values whose table form is read off the rule by hand."""
    assert table(value) == shown
    assert voice.table_number(value) == shown


# W1-R round 3 (SE-m3): below 1e-4 Python's ``repr`` writes an exponent where JavaScript writes
# decimals down to 1e-6, so "5e−05" stood where the table shows 0.00005. Each value's table form
# here is what ``fmtNum`` printed under node 25.9.0 (an independent engine), byte for byte; the
# last minus of a negative's exponent stays a hyphen, as the table's one ``replace`` leaves it.
@pytest.mark.parametrize("value, shown", [
    (5e-05, "0.00005"), (1.23e-05, "0.0000123"), (1e-06, "0.000001"), (9.99e-07, "9.99e−7"),
    (1.23e-07, "1.23e−7"), (-1.23e-07, "−1.23e-7"), (7e-07, "7e−7"), (-5e-05, "−0.00005"),
    (9.99e-06, "0.00000999"), (0.000123456, "0.000123"), (-0.00009996, "−0.0001"),
    (9.995e-07, "9.99e−7"), (1e-21, "1e−21"), (2.675, "2.67"), (123.456, "123"),
    (-999.95, "−1000"), (0.1 + 0.2, "0.3"), (-2 / 3, "−0.667"), (999.4999, "999"),
    (0.0012345, "0.00123")])
def test_a_small_score_is_printed_as_the_table_prints_it(value, shown):
    assert table(value) == shown
    assert voice.table_number(value) == shown


class _Summary:
    def __init__(self, estimate: float):
        self.estimate = estimate

    def summary(self, task):
        return {"mse": {"estimate": self.estimate}}


def test_the_selection_sentence_prints_its_scores_as_the_table_does():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.normal(100, 35, n)
    preds = {"a": y + rng.normal(0, 32, n), "b": y + rng.normal(0, 34, n)}
    oof = OutOfFold("regression", np.zeros((n, 1)), y,
                    [(0, np.ones(n, dtype=bool), np.ones(n, dtype=bool))])
    for k, v in preds.items():
        oof.predictions[k] = v
    results = {k: _Summary(float(np.mean((y - v) ** 2))) for k, v in preds.items()}
    got = selection_optimism("regression", "mse", results, oof, metric_label="MSE",
                             replicates=200, seed=3)
    cv = results[got["best"]].estimate
    assert cv > 1000  # the walkthrough's scale
    text = got["text"]
    assert f"CV MSE of {table(cv)} " in text
    assert (f"corrected for the choice, {table(got['corrected'])} (95% interval "
            f"{table(got['corrected_low'])} to {table(got['corrected_high'])})") in text
    assert f"{cv:.3f}" not in text


def test_the_opening_records_its_held_out_score_as_the_table_does():
    opening = OpenSeal(family="linear", target="glucose", metric="mse",
                       scores={"linear": {"mse": 1133.621, "r2": 0.15897}}, n_holdout=4370)
    said = voice.sentence_for(opening, ProjectState(target="glucose"))
    assert ": held-out MSE `1,134`" in said and "1133.621" not in said
