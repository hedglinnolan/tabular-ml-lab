"""Row previews (Tier A, leakage): once the split exists, no preview reads a held-out row."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import fact_previews, row_previews  # noqa: F401 - registers the builders
from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, PreviewContext, plan, words
from turbotab.core.datastore import DataStore, ingest
from turbotab.core.decisions import ExclusionRule, ProjectState, parse_decision
from turbotab.core.graph import Bundle
from turbotab.core.stages.rows import compute_cohort, draw_split, split_inputs

DIETARY = Path(__file__).resolve().parents[3] / "turbotab" / "sample_data" / "dietary_recalls.csv"
ROLES = {"participant_id": "identifier", "recall_number": "time", "age": "covariate", "sex": "covariate",
         "bmi": "covariate", "energy_kcal": "energy", "protein_g": "exposure", "fat_g": "exposure",
         "carbohydrate_g": "exposure", "sodium_mg": "exposure"}


class SpyStore:
    """A DataStore that remembers every row id it was asked to read."""

    def __init__(self, store: DataStore):
        self._store = store
        self.read: set[int] = set()
        self.read_everything = False

    def materialize(self, columns=None, row_ids=None):
        if row_ids is None:
            self.read_everything = True
        else:
            self.read.update(int(i) for i in np.asarray(row_ids))
        return self._store.materialize(columns, row_ids)

    def sample(self, *a, **k):  # pragma: no cover - previews never sample unseen rows
        raise AssertionError("a preview sampled rows directly")

    def __getattr__(self, name):
        return getattr(self._store, name)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    dest = tmp_path_factory.mktemp("previews") / "dietary.parquet"
    ingest(DIETARY, dest)
    with DataStore(dest, 2 << 30) as s:
        yield s


def _sealed_project(store: DataStore):
    rule = ExclusionRule(column="energy_kcal", low=500, high=5000, reason="implausible intake")
    state = ProjectState(lens=["dietary"], target="hba1c", task="regression", roles=ROLES,
                         exclusions=[rule], missing="complete_case", split={"holdout": 0.25, "seed": 0})
    ingest_info = {"columns": [c.to_dict() for c in store.info().columns]}
    measured: list = []
    steps, kept, preds = compute_cohort(store, state, ingest_info, measured=measured)
    cohort = Bundle(data={"steps": steps, "n_final": len(kept), "predictors": preds},
                    frames={"rows": pd.DataFrame({"row_id": kept}),
                            "measured": pd.DataFrame({"row_id": measured[0]})})
    inputs = split_inputs(state, measured[0], store, "regression")
    frame, info = draw_split(kept, holdout=0.25, seed=0, folds=5, universe=measured[0], **inputs)
    sealed = info.pop("sealed")
    split = Bundle(data=info, frames={"assignment": frame, "sealed": pd.DataFrame({"row_id": sealed})})
    return state, cohort, split, set(sealed.tolist())


@pytest.mark.parametrize("decision", [
    {"kind": "set_exclusions", "rules": []},
    {"kind": "set_exclusions", "rules": [{"column": "energy_kcal", "low": 800, "high": 4500, "reason": "tighter"}]},
    {"kind": "set_missing", "strategy": "impute"},
    {"kind": "set_missing", "strategy": "complete_case"},
    {"kind": "set_roles", "roles": {**ROLES, "sodium_mg": "excluded", "bmi": "exposure"}},
])
def test_no_preview_reads_a_sealed_row(store, decision):
    state, cohort, split, sealed = _sealed_project(store)
    assert sealed, "the fixture must hold rows out"
    spy = SpyStore(store)
    training = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    ctx = PreviewContext(project_id="p", state=state, datastore=spy,
                         artifact={"cohort": cohort, "split": split}.get,
                         training_row_ids=training, cohort_row_ids=cohort.frames["rows"]["row_id"].to_numpy(),
                         sealed_row_ids=np.array(sorted(sealed)))
    result = plan(parse_decision(decision), ctx, basis="b")
    assert result.views, result.note
    assert not spy.read_everything, "a preview read the whole table after the split"
    assert not (spy.read & sealed), f"{len(spy.read & sealed)} held-out rows were read"
    for view in result.views:
        assert words(view.title) <= TITLE_WORDS and words(view.caption) <= CAPTION_WORDS


@pytest.mark.parametrize("decision", [
    {"kind": "set_exclusions", "rules": []},
    {"kind": "set_missing", "strategy": "impute"},
    {"kind": "set_target", "column": "hba1c"},
])
def test_no_preview_reads_a_sealed_row_while_the_split_recomputes(store, decision):
    """A changed exclusion makes the split stale (it re-runs): the newest split's sealed rows
    still hold — they are drawn over every row with the outcome measured — so no preview may read
    them in the meantime, though no split is fresh to ask (the reviewers' instrumented drive)."""
    state, cohort, split, sealed = _sealed_project(store)
    spy = SpyStore(store)
    ctx = PreviewContext(project_id="p", state=state, datastore=spy,
                         artifact={"cohort": None, "split": None}.get,  # both recomputing
                         training_row_ids=None, cohort_row_ids=None,
                         sealed_row_ids=np.array(sorted(sealed)))
    result = plan(parse_decision(decision), ctx, basis="b")
    assert result.views, result.note
    assert not spy.read_everything, "a preview read the whole table while the split recomputed"
    assert not (spy.read & sealed), f"{len(spy.read & sealed)} held-out rows were read"
    assert spy.read, "the preview must have read something to make this test mean anything"


def test_relaxing_a_rule_brings_rows_back_in_the_preview(store):
    state, cohort, split, sealed = _sealed_project(store)
    ctx = PreviewContext(project_id="p", state=state, datastore=SpyStore(store),
                         artifact={"cohort": cohort, "split": split}.get, training_row_ids=None,
                         cohort_row_ids=None, sealed_row_ids=np.array(sorted(sealed)))
    result = plan(parse_decision({"kind": "set_exclusions", "rules": []}), ctx, basis="b")
    flow = result.views[0]
    assert flow.kind == "row_flow"
    assert flow.before[0].n == flow.after[0].n == 600 - len(sealed)
    assert flow.after[-1].n > flow.before[-1].n  # the excluded rows come back
