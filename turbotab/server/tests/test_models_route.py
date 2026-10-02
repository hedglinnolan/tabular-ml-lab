"""Tier B: ``GET /api/models`` lists the registry."""
from __future__ import annotations


def test_models_lists_every_family_with_what_it_assumes(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert [f["key"] for f in body] == ["linear", "elastic_net", "boosted_trees", "mixed", "gee",
                                        "cox"]
    tasks = {
        "linear": {"regression", "binary", "multiclass"},
        "elastic_net": {"regression", "binary", "multiclass"},
        "boosted_trees": {"regression", "binary", "multiclass"},
        # WP12: rows that repeat within a unit, and a time-to-event outcome
        "mixed": {"regression"},
        "gee": {"regression", "binary"},
        "cox": {"time_to_event"},
    }
    for family in body:
        assert set(family) == {"key", "label", "tasks", "inductive_bias", "strengths", "cautions",
                               "needs_scaling", "handles_missing"}
        assert set(family["tasks"]) == tasks[family["key"]]
        assert 0 < len(family["inductive_bias"].split()) <= 20
        assert family["strengths"] and family["cautions"]
    trees = body[2]
    assert trees["handles_missing"] and not trees["needs_scaling"]
