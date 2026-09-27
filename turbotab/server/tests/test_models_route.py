"""Tier B: ``GET /api/models`` lists the registry."""
from __future__ import annotations


def test_models_lists_every_family_with_what_it_assumes(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    body = response.json()
    assert [f["key"] for f in body] == ["linear", "elastic_net", "boosted_trees"]
    for family in body:
        assert set(family) == {"key", "label", "tasks", "inductive_bias", "strengths", "cautions",
                               "needs_scaling", "handles_missing"}
        assert set(family["tasks"]) == {"regression", "binary", "multiclass"}
        assert 0 < len(family["inductive_bias"].split()) <= 20
        assert family["strengths"] and family["cautions"]
    trees = body[2]
    assert trees["handles_missing"] and not trees["needs_scaling"]
