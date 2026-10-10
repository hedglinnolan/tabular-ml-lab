"""Tier B: ``GET /api/models`` lists the registry."""
from __future__ import annotations


def test_models_lists_every_family_with_what_it_assumes(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert [f["key"] for f in body] == ["linear", "elastic_net", "ridge", "huber", "boosted_trees",
                                        "random_forest", "xgboost", "featurewise",
                                        "proportional_odds", "mixed", "gee", "cox",
                                        "screened_elastic_net"]
    tasks = {
        # an ordered outcome stays on their shelf, as unordered classes, ranked lower (WP12a)
        "linear": {"regression", "binary", "multiclass", "ordinal"},
        "elastic_net": {"regression", "binary", "multiclass", "ordinal"},
        "ridge": {"regression", "binary", "multiclass", "ordinal"},  # RT-5b
        "huber": {"regression"},  # RT-5c: robust linear regression, a number only
        "boosted_trees": {"regression", "binary", "multiclass", "ordinal"},
        "random_forest": {"regression", "binary", "multiclass", "ordinal"},  # RT-5d
        "xgboost": {"regression", "binary", "multiclass", "ordinal"},  # RT-5e
        "featurewise": {"regression", "binary"},  # WP11
        "proportional_odds": {"ordinal"},  # WP12a: the ordinal family models order alone
        # WP12b: rows that repeat within a unit, and a time-to-event outcome
        "mixed": {"regression"},
        "gee": {"regression", "binary"},
        "cox": {"time_to_event"},
        "screened_elastic_net": {"regression", "binary"},  # MS7: in-fold screening at p ≫ n
    }
    for family in body:
        assert set(family) == {"key", "label", "tasks", "inductive_bias", "strengths", "cautions",
                               "needs_scaling", "handles_missing", "purposes", "predicts",
                               # MC-1: the model-family contract's user-facing declarations
                               "flexible", "bootstrap_optimism", "invariances", "inference_table",
                               "bias_terms", "reads"}
        assert set(family["tasks"]) == tasks[family["key"]]
        assert 0 < len(family["inductive_bias"].split()) <= 20
        assert family["strengths"] and family["cautions"]
    predicting = [f for f in body if f["predicts"]]
    assert [f["key"] for f in predicting] == ["linear", "elastic_net", "ridge", "huber",
                                              "boosted_trees", "random_forest", "xgboost",
                                              "proportional_odds", "mixed", "gee", "cox",
                                              "screened_elastic_net"]
    for family in predicting:
        # MS7: screening chooses features from the outcome, so it serves prediction only; RECIPES
        # §5 refuses robust linear regression under inference.
        prediction_only = {"screened_elastic_net", "huber"}
        expected = {"prediction"} if family["key"] in prediction_only else {"prediction", "inference"}
        assert set(family["purposes"]) == expected
    by_key = {f["key"]: f for f in body}
    trees = by_key["boosted_trees"]
    assert trees["handles_missing"] and not trees["needs_scaling"]
    # WP11: the feature-wise tests serve inference only, and make no predictions.
    featurewise = by_key["featurewise"]
    assert featurewise["purposes"] == ["inference"] and not featurewise["predicts"]
    assert set(featurewise["tasks"]) == {"regression", "binary"}
