"""Tier B: ``GET /api/models`` lists the registry."""
from __future__ import annotations


def test_models_lists_every_family_with_what_it_assumes(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert [f["key"] for f in body] == ["linear", "elastic_net", "boosted_trees", "featurewise",
                                        "proportional_odds"]
    for family in body:
        assert set(family) == {"key", "label", "tasks", "inductive_bias", "strengths", "cautions",
                               "needs_scaling", "handles_missing", "purposes", "predicts"}
        if family["key"] == "proportional_odds":  # WP12a: the ordinal family models order alone
            assert family["tasks"] == ["ordinal"]
        elif family["key"] != "featurewise":  # an ordered outcome stays on their shelf, as
            # unordered classes, ranked lower
            assert set(family["tasks"]) == {"regression", "binary", "multiclass", "ordinal"}
        assert 0 < len(family["inductive_bias"].split()) <= 20
        assert family["strengths"] and family["cautions"]
    predicting = [f for f in body if f["predicts"]]
    assert [f["key"] for f in predicting] == ["linear", "elastic_net", "boosted_trees",
                                              "proportional_odds"]
    for family in predicting:
        assert set(family["purposes"]) == {"prediction", "inference"}
    trees = body[2]
    assert trees["handles_missing"] and not trees["needs_scaling"]
    # WP11: the feature-wise tests serve inference only, and make no predictions.
    featurewise = body[3]
    assert featurewise["purposes"] == ["inference"] and not featurewise["predicts"]
    assert set(featurewise["tasks"]) == {"regression", "binary"}
