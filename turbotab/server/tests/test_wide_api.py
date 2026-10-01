"""Wide tables over HTTP (M2_CONTRACT §5): the roles search on ``/columns`` and the visible
columns of ``/table``. Tier B: one test per route family."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.server.tests.conftest import open_by_path, wait_for

N_GENES = 256


@pytest.fixture(scope="module")
def wide(client, tmp_path_factory):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f"gene_{j:03d}": rng.poisson(4, size=30) for j in range(N_GENES)})
    frame.insert(0, "age", rng.integers(20, 80, size=30))
    frame["dose, mg"] = rng.normal(size=30).round(3)   # a name with a comma in it
    frame["outcome"] = rng.normal(size=30).round(3)
    path = tmp_path_factory.mktemp("wide") / "wide.csv"
    frame.to_csv(path, index=False)
    pid = open_by_path(client, path)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    return pid, frame


def test_columns_are_searched_and_paged_for_the_roles_list(client, wide):
    pid, frame = wide
    everything = client.get(f"/api/projects/{pid}/columns")
    assert everything.status_code == 200 and "x-total-count" not in everything.headers
    assert [c["name"] for c in everything.json()] == list(frame.columns)

    found = client.get(f"/api/projects/{pid}/columns", params={"query": "GENE_01"})
    expected = [c for c in frame.columns if "gene_01" in c]
    assert [c["name"] for c in found.json()] == expected
    assert found.headers["x-total-count"] == str(len(expected))

    page = client.get(f"/api/projects/{pid}/columns",
                      params={"query": "gene 2", "offset": 3, "limit": 4})
    matches = [c for c in frame.columns if "gene" in c and "2" in c]
    assert [c["name"] for c in page.json()] == matches[3:7]
    assert page.headers["x-total-count"] == str(len(matches))
    summary = page.json()[0]
    column = frame[summary["name"]]
    assert summary["n"] == 30 and summary["max"] == int(column.max())

    named = client.get(f"/api/projects/{pid}/columns", params={"names": "outcome,dose, mg"})
    assert [c["name"] for c in named.json()] == ["outcome", "dose, mg"]
    none = client.get(f"/api/projects/{pid}/columns", params={"query": "no such column"})
    assert none.json() == [] and none.headers["x-total-count"] == "0"
    assert client.get(f"/api/projects/{pid}/columns", params={"names": "nope"}).status_code == 404


def test_a_table_window_reads_only_the_visible_columns(client, wide):
    pid, frame = wide
    visible = ["gene_200", "dose, mg", "age"]
    response = client.get(f"/api/projects/{pid}/table",
                          params={"offset": 5, "limit": 3, "columns": ",".join(visible)})
    assert response.status_code == 200
    window = response.json()
    assert window["columns"] == visible and window["total_rows"] == 30
    assert window["rows"] == frame.loc[5:7, visible].values.tolist()
