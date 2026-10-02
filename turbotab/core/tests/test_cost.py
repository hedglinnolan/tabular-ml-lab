"""Tier B: what a fit will take, stated before it runs (M2_CONTRACT §12.6)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from turbotab.core.models import cost
from turbotab.core.models.cost import NOTEWORTHY_SECONDS
from turbotab.core.models.previews import _say_the_cost
from turbotab.core.teaching import COMPOSED_BUDGETS


def test_a_duration_is_worded_as_an_estimate_and_names_what_makes_it_large():
    assert cost.duration(0.4) == "under a second"
    assert cost.duration(1.4) == "about 1 second"
    assert cost.duration(23) == "about 25 seconds"
    assert cost.duration(310.7) == "about 5 minutes"
    assert cost.duration(2 * 3600) == "about 2 hours"
    assert cost.say(310.7, 400, 20_004) == "about 5 minutes at `20,004` columns"
    assert cost.say(90, 800_000, 29) == "about 2 minutes on `800,000` rows"
    assert cost.say(4, 480, 12) == "about 4 seconds"
    # a short fit is not explained by its width: nobody asked why it takes a second
    assert cost.say(0.3, 72, 394) == "under a second" and cost.say(2, 60, 497) == "about 2 seconds"


def test_the_timing_sample_stays_small_and_takes_a_small_table_whole():
    assert cost.sample_shape(480, 12) == (480, 12)  # timed whole: the scaling is exact
    assert cost.sample_shape(400, 20_004) == (200, 1_000)
    rows, columns = cost.sample_shape(800_000, 29)
    assert rows == cost.SAMPLE_ROWS and columns == 29
    assert cost.sample_shape(30, 50_000)[0] == 30  # never more rows than the table has


def _ctx(families: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(artifact=lambda stage: {"families": families} if stage == "shelf" else None,
                           read={})


SHELF = [
    {"key": "linear", "label": "Linear model", "estimate_seconds": 1.5},
    {"key": "elastic_net", "label": "Elastic net", "estimate_seconds": 233.9},
    {"key": "boosted_trees", "label": "Boosted trees", "estimate_seconds": 61.9},
]


def test_the_models_card_states_a_long_fit_and_stays_quiet_on_a_short_one():
    ctx = _ctx(SHELF)
    _say_the_cost(SimpleNamespace(models=["elastic_net", "boosted_trees"]), ctx, 400, 20_002)
    note = ctx.read["note"]
    assert note == "Fitting these takes about 5 minutes at `20,002` columns, most of it elastic net."
    assert len(note.split()) <= COMPOSED_BUDGETS["preview_note"]
    one = _ctx(SHELF)
    _say_the_cost(SimpleNamespace(models=["elastic_net"]), one, 400, 20_002)
    assert one.read["note"] == "Fitting it takes about 4 minutes at `20,002` columns."
    quick = _ctx(SHELF)
    _say_the_cost(SimpleNamespace(models=["linear"]), quick, 400, 20_002)
    assert 1.5 < NOTEWORTHY_SECONDS and "note" not in quick.read  # a cost no one asks about
    untimed = _ctx([{**f, "estimate_seconds": None} for f in SHELF])
    _say_the_cost(SimpleNamespace(models=["elastic_net"]), untimed, 400, 20_002)
    assert "note" not in untimed.read  # an estimate that could not be measured is not invented


def test_a_whole_table_timing_scales_by_the_folds_alone(monkeypatch):
    """A table inside the timing sample is timed whole: the estimate is one fit times the folds."""
    import pandas as pd

    from turbotab.core.decisions import ProjectState
    from turbotab.core.models import get_family

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
    frame["y"] = frame["x1"] + rng.normal(size=120)

    class Store:
        def materialize(self, columns, ids):
            return frame.loc[np.asarray(ids), list(columns)]

    monkeypatch.setattr(cost, "time_one_fit", lambda pipeline, X, y: 0.5)
    state = ProjectState(target="y", task="regression", purpose="prediction",
                         roles={"x1": "exposure", "x2": "covariate"})
    out = cost.estimate_fits(Store(), state, "regression", np.arange(120),
                             [get_family("linear")], folds=5)
    assert out["linear"].seconds == 2.5 and out["linear"].text == "about 2 seconds"
