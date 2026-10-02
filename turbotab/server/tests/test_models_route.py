"""Tier B: ``GET /api/models`` lists the registry."""
from __future__ import annotations


def test_models_lists_every_family_with_what_it_assumes(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert [f["key"] for f in body] == ["linear", "elastic_net", "boosted_trees", "featurewise",
                                        "proportional_odds", "mixed", "gee", "cox"]
    tasks = {
        # an ordered outcome stays on their shelf, as unordered classes, ranked lower (WP12a)
        "linear": {"regression", "binary", "multiclass", "ordinal"},
        "elastic_net": {"regression", "binary", "multiclass", "ordinal"},
        "boosted_trees": {"regression", "binary", "multiclass", "ordinal"},
        "featurewise": {"regression", "binary"},  # WP11
        "proportional_odds": {"ordinal"},  # WP12a: the ordinal family models order alone
        # WP12b: rows that repeat within a unit, and a time-to-event outcome
        "mixed": {"regression"},
        "gee": {"regression", "binary"},
        "cox": {"time_to_event"},
    }
    for family in body:
        assert set(family) == {"key", "label", "tasks", "inductive_bias", "strengths", "cautions",
                               "needs_scaling", "handles_missing", "purposes", "predicts"}
        assert set(family["tasks"]) == tasks[family["key"]]
        assert 0 < len(family["inductive_bias"].split()) <= 20
        assert family["strengths"] and family["cautions"]
    predicting = [f for f in body if f["predicts"]]
    assert [f["key"] for f in predicting] == ["linear", "elastic_net", "boosted_trees",
                                              "proportional_odds", "mixed", "gee", "cox"]
    for family in predicting:
        assert set(family["purposes"]) == {"prediction", "inference"}
    trees = body[2]
    assert trees["handles_missing"] and not trees["needs_scaling"]
    # WP11: the feature-wise tests serve inference only, and make no predictions.
    featurewise = body[3]
    assert featurewise["purposes"] == ["inference"] and not featurewise["predicts"]
    assert set(featurewise["tasks"]) == {"regression", "binary"}
