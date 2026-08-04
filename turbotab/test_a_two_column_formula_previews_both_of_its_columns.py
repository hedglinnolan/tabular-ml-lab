"""`GUIDED-171` — the feature-engineering before/after showed only the FIRST
operand.

Driven on `clinical_labs.csv` before this loop, `GET /project/<id>/feature/
preview?transform=ratio&columns=weight_kg,height_cm` returned, verbatim:

    {"sentence": "The ratio `weight_kg / height_cm` was computed row by row; …",
     "rows": [{"label": 0, "before": 95.8, "after": 0.5682}, …]}

`ratio` declares `n_inputs=2`. **The preview showed one.** 95.8 is `weight_kg`;
the 168.6 of `height_cm` that the division actually used appears nowhere, so a
user looking at the surface where they consent to the transform cannot see what
it consumed. The `after` is arithmetically correct and unexplainable.

And the structured payload was **poorer than the sentence beside it**: the
prose named both columns and the machine-readable form named none — no
`inputs`, no second value, nothing. Trap #7 again, one surface over from
`GUIDED-157`.

## Why nothing caught it

`/features` and `/feature/preview` are outside every field-level gate.

- `test_every_field_the_server_composes_has_a_reader.py::NOT_SWEPT` lists
  *"/features, /recipes, /preprocess — the Features and Preprocess steps, which
  the Explore-step drive does not open"*. The reason is true: that sweep's
  fixture stops at Explore.
- `test_the_three_unswept_payloads_are_swept.py` then drove the journey through
  the seal and **enumerated** all three. But it has no equivalent of L42-B's
  `test_every_unread_family_names_a_reader_or_a_row` — it *prints* its unread
  counts and gates nothing. So an unread field in `/features` is a number in a
  captured stdout block and not a failure.

That is the gap, and it is one surface wide rather than one field wide. It is
**reported, not closed here**: making the late sweep gate its dispositions is a
sweep-wide change with its own disposition table to write, and doing it in the
same loop as the defect it would have caught is the pattern
`AGENT_ONBOARD.md` §08.2 warns about from the other direction.

## Fixtures

`GUIDED-097`: `clinical_labs.csv` on a **classification** target and
`clinic_visits.csv` on a **regression** target. Not covered: a **multiclass**
target, and the `deferred` half of the catalogue — a deferred transform's
preview returns `rows: []` on purpose (clause §06: there is no single set of
values to show before the fold), so the operand table this file is about does
not exist there.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import features as F                                    # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

#: Two target shapes, and a two-operand formula on each. The arithmetic is
#: named here so the assertion can re-derive the `after` from the operands the
#: preview showed — which is what makes it a claim that those are the values
#: the computation used, rather than that two numbers were rendered.
CASES = {
    "classification": ("clinical_labs", "readmitted", "ratio",
                       ["weight_kg", "height_cm"], lambda a, b: a / b),
    "regression": ("clinic_visits", "hba1c", "product",
                   ["age", "glucose"], lambda a, b: a * b),
}


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _preview(fixture, target, transform, columns):
    client = _client()
    with open(DATA / f"{fixture}.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": (f"{fixture}.csv", handle, "text/csv")}).json()["id"]
    ok = client.post(f"/project/{pid}/decision",
                     json={"kind": "set_target", "payload": {"column": target}})
    assert ok.status_code == 200, ok.text[:300]

    served = client.get(f"/project/{pid}/features")
    assert served.status_code == 200, served.text[:300]
    catalogue = served.json()
    entry = next(t for t in catalogue["row_local"] + catalogue["deferred"]
                 if t["key"] == transform)

    r = client.get(f"/project/{pid}/feature/preview",
                   params={"transform": transform, "columns": ",".join(columns)})
    assert r.status_code == 200, r.text[:300]
    from turbotab.api import STORE
    return client, pid, entry, r.json(), STORE.get(pid).df


@pytest.mark.parametrize("shape", sorted(CASES))
def test_a_two_column_formula_previews_both_of_its_columns(shape):
    """**The defect, as the count it produced.**

    The transform declares how many columns it consumes and the preview has to
    show that many. Each shown value is then checked against the cell it claims
    to be, and the `after` is re-derived from the two of them — so this is a
    claim that the preview shows *what the computation used*, not that it shows
    two numbers.
    """
    fixture, target, transform, columns, arith = CASES[shape]
    _client_, _pid, entry, body, df = _preview(fixture, target, transform,
                                               columns)
    needs = entry["n_inputs"]
    assert needs == 2, f"{transform} no longer takes two columns: {entry}"
    assert body["rows"], "a preview with no rows is a description"

    for row in body["rows"]:
        shown = row.get("operands")
        assert shown is not None and len(shown) == needs, (
            f"{transform} consumes {needs} columns and the preview shows "
            f"{0 if shown is None else len(shown)} per row, so a user cannot "
            f"see what it was computed from: {row}")
        # Each operand IS the cell it stands for.
        for column, value in zip(columns, shown):
            assert value == pytest.approx(float(df.loc[row["label"], column])), (
                f"the preview shows {value} for `{column}` on row "
                f"{row['label']} and the table holds "
                f"{df.loc[row['label'], column]}")
        # And the two of them produce the `after` that is on screen beside them.
        assert row["after"] == pytest.approx(arith(*[float(v) for v in shown]),
                                             rel=1e-3), (
            f"the operands shown do not produce the result shown: "
            f"{shown} -> {row['after']}")


@pytest.mark.parametrize("shape", sorted(CASES))
def test_the_payload_names_the_columns_the_computation_consumed(shape):
    """The structured half. The sentence named both columns from the start;
    the payload named none, and the payload is what everything downstream
    reads."""
    fixture, target, transform, columns, _arith = CASES[shape]
    _client_, _pid, entry, body, _df = _preview(fixture, target, transform,
                                                columns)
    assert body.get("inputs") == columns[:entry["n_inputs"]], (
        f"the preview does not say which columns it consumed: "
        f"{body.get('inputs')!r}")
    for row in body["rows"]:
        assert len(row["operands"]) == len(body["inputs"]), (
            "the header and the values disagree about how many operands there "
            "are, which is worse than showing one")


def test_a_one_column_transform_still_says_exactly_what_it_always_said():
    """The shelf is never shortened. `before` is a shipped field with shipped
    readers, and it stays — as `operands[0]`, computed once, not twice."""
    df = pd.DataFrame({"chol": [120.0, 180.0, 240.0, 90.0, 200.0, 160.0, 210.0]})
    pv = F.preview(df, "log", ["chol"])
    assert pv["inputs"] == ["chol"]
    for row in pv["rows"]:
        assert row["operands"] == [row["before"]]
        assert row["before"] == pytest.approx(float(df.loc[row["label"], "chol"]))


def test_the_operands_are_the_slice_the_computation_actually_reads():
    """Trap #3's rule pointed at a payload: the names the preview publishes
    have to resolve to what ran, not to what the caller asked for.

    `_require_columns` validates `columns[:n_inputs]` and `_compute` reads
    `columns[:n_inputs]`, so a caller passing a third column gets two operands
    — and the preview must report two, or it names a column the arithmetic
    never touched.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "b": [2.0, 2.0, 4.0, 4.0, 5.0, 6.0],
                       "c": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]})
    pv = F.preview(df, "ratio", ["a", "b", "c"])
    assert F.get("ratio").n_inputs == 2
    assert pv["inputs"] == ["a", "b"], (
        f"the preview claims to consume {pv['inputs']}, and `c` reaches no "
        f"arithmetic in this transform")
    for row in pv["rows"]:
        assert len(row["operands"]) == 2


def test_a_deferred_transform_still_shows_no_values_and_says_why():
    """The half this fix deliberately does not touch, asserted so the change
    above cannot leak into it. Clause §06: a preview of a transform fitted
    in-fold has no single set of values to show, and inventing an operand
    table for it would be showing the researcher a picture of their held-out
    data."""
    df = pd.DataFrame({"age": [31.0, 44.0, 52.0, 61.0, 27.0, 38.0, 49.0]})
    pv = F.preview(df, "bin_quantile", ["age"], {"n_bins": 3})
    assert pv["rows"] == []
    assert pv["preview_not_applied"] is True
    assert "inputs" not in pv, (
        "the deferred preview grew an operand list it renders nowhere — a "
        "field with no consumer, which is the trap this loop is avoiding")


# ── the page ─────────────────────────────────────────────────────────────────

def _page_function(name: str) -> str:
    """One top-level function's source, out of `web/index.html`.

    By indentation: every function in the controller opens at two spaces and
    closes on a line that is exactly `  }`. The caller checks what it got.
    """
    lines = PAGE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(f"  function {name}("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "  }")
    return "\n".join(lines[start:end + 1])


def test_the_preview_table_renders_a_column_for_every_operand():
    """The consumer, run — the table behind `data-feat-preview`.

    `featPreviewHTML` built a fixed three-column table, `row | before | after`,
    so even a payload carrying both operands would still have rendered one.
    The server half and the page half are the same defect at two layers and
    neither is the fix on its own.
    """
    import shutil
    import subprocess
    import tempfile
    import json as _json

    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    _client_, _pid, _entry, body, _df = _preview(
        "clinical_labs", "readmitted", "ratio", ["weight_kg", "height_cm"])

    renderer = _page_function("featPreviewHTML")
    assert "data-feat-preview" in PAGE.read_text(encoding="utf-8"), (
        "the control this renders for is gone from the page")
    assert "feat-prevbox" in renderer and renderer.rstrip().endswith("}"), (
        "featPreviewHTML was not lifted out of the page intact:\n" + renderer)

    program = "\n".join([
        _page_function("esc"), _page_function("num"), renderer,
        "process.stdout.write(featPreviewHTML(" + _json.dumps(body) + "));"])
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as handle:
        handle.write(program)
        path = handle.name
    try:
        run = subprocess.run([shutil.which("node"), path],
                             capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    assert run.returncode == 0, run.stderr[-2000:]
    out = run.stdout

    for column in body["inputs"]:
        assert f"<th>{column}</th>" in out, (
            f"the preview table has no column for `{column}`, which the "
            f"formula consumes:\n{out}")
    first = body["rows"][0]
    for value in first["operands"]:
        assert f"<td>{value:g}</td>" in out or str(value) in out, (
            f"row {first['label']} was computed from {first['operands']} and "
            f"the table shows {value} nowhere:\n{out}")
    # The header row now carries one cell per operand plus `row` and the new
    # column — the count is the claim, so it is counted.
    header = out[out.index("<tr>"):out.index("</tr>")]
    assert header.count("<th>") == len(body["inputs"]) + 2, (
        f"the header has {header.count('<th>')} cells for "
        f"{len(body['inputs'])} operands: {header}")
