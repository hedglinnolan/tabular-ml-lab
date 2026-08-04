"""`GUIDED-191` / `GUIDED-166` — one provenance mechanism, every writer on it.

## What L48 left, and why it could not be closed by adding one more writer

L48 gave `set_impossible_missing` a provenance record and said so in
`missingness.PROVENANCE_MIXED`'s own comment: the remainder of a column's
blanks had to be called *"not recorded as made here"* rather than *"blank in
the file"*, **because `coerce_numeric` turns unparseable text into `NaN` and
files no `made_blanks`.** A hedge with a named cause is a to-do.

The cause was structural rather than one missed writer. `ml/import_doctor.py`
is frozen (`TRANSITION_PLAN.md` §05) and `apply_fix` returns
`(frame, prose_string)` — there is no payload channel, so no repair beneath it
can file anything. What every writer DOES share is the moment it installs a new
working table, and that is where the record is taken now:
`project.Project._install` is the one door, `missingness.blanks_made` is the one
recorder, and nothing else composes a provenance record.

## The writers, enumerated by driving them rather than by reading them

Twelve passes install a working table (every `self._history.append` /
`self.df = …` pair in `project.py`, plus `set_orientation`, which had no undo
entry). Driven across all seventeen fixtures in `sample_data`:

| pass | blanks? | measured |
|---|---|---|
| `apply` · `recode_missing` | **yes** | 20 of 20 applications, 189 cells |
| `apply` · `coerce_numeric` | **yes** | 2 of 8 applications, 55 cells |
| `apply` · `melt_repeated` | **yes** | 4,388 cells, index rebuilt → unattributable |
| `add_feature` · log/log1p/sqrt/ratio/bin_fixed | **yes** | 136/136/136/28/4 cells |
| `set_impossible_missing` | **yes** | 4 cells on `clinical_labs.csv` `sbp` |
| `turn_the_table_around` | **yes** | coerces per column at a 90% threshold |
| `combine_rows` | index rebuilt | 600 → 200 rows; unattributable |
| `apply` · `read_as_binary` | **no** | 20 applications, 0 cells — driven, not assumed |
| `apply` · `normalize_categories` / `drop_columns` | **no** | 0 cells |
| `remove_feature`, `eligibility_criterion`, `trim_training_rows` | **no** | remove rows/columns |
| `missingness::*` (explicit category, indicator) | **no** | they fill, or add a column |

Two candidates named in the brief are dismissed **with a reason, not silently**:

* `ml/import_doctor.py` `reinfer_types` (~L274-293) converts a text column only
  when **every** non-null value parses (`parsed.notna().mean() < 1.0: continue`),
  so it cannot produce a blank. Asserted below rather than argued.
* `ml/join_doctor.py` is reached only from `utils/combine*.py` — the Streamlit
  multi-file import path. No `turbotab` module imports it, so it is not a
  writer of this app's working table.

## `GUIDED-097` — two fixtures of different target shape

`clinical_labs.csv` / `readmitted` (binary classification) and
`clinical_longitudinal.csv` / `hba1c` (continuous). **The shape not covered is a
multiclass target**: `multiclass_stage.csv` exists, and it has no repair that
blanks a cell, so it can drive the enumeration but not the provenance sentence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                              # noqa: E402
from turbotab import engine                                           # noqa: E402
from turbotab import missingness as M                                 # noqa: E402
from turbotab import pageharness as H                                 # noqa: E402
from turbotab import project as PROJ                                  # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _project(client, fixture: str, lens: str, target: str) -> str:
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": [lens]}})
    answer = client.post(f"/project/{pid}/decision",
                         json={"kind": "set_target",
                               "payload": {"column": target}})
    assert answer.status_code == 200, answer.text
    return pid


def _survey_row(client, pid: str, column: str):
    rows = client.get(f"/project/{pid}/preprocess").json()["columns"]
    return next((r for r in rows if r["column"] == column), None)


def _finding(client, pid: str, fix_kind: str, column: str = ""):
    findings = client.get(f"/project/{pid}").json()["findings"]
    for f in findings:
        if f.get("fix_kind") != fix_kind:
            continue
        if column and column not in (f.get("affected_columns") or []):
            continue
        return f
    return None


# ═══════════ (a) coerce_numeric · the writer the hedge was named after ══════

def test_coerce_numeric_files_the_cells_it_blanked_and_the_survey_says_so(client):
    """**The load-bearing drive.** Blank count before, blank count after, and
    what the missingness survey then says about where those blanks came from.

    `clinical_labs.csv` `troponin` holds 41 qualitative results — `positive`
    and `negative` — beside 247 numeric ones. Reading the column as numeric
    turns exactly those 41 into blanks, which is a real and defensible repair
    and is also the app deleting a measurement that was recorded. Before this
    change the count reached the user in one prose clause and nothing
    machine-readable carried it, so two cards later the app asked *"could a
    blank in `troponin` mean something?"* about 41 blanks it had written.
    """
    pid = _project(client, "clinical_labs.csv", "clinical", "readmitted")

    frame = pd.read_csv(DATA / "clinical_labs.csv")
    n_before = int(frame["troponin"].isna().sum())
    assert n_before == 0, (
        "`troponin` already carries blanks, so this cannot show that the "
        "repair created them")
    before_row = _survey_row(client, pid, "troponin")
    assert before_row is None, "`troponin` is in the survey before any repair"

    finding = _finding(client, pid, "coerce_numeric", "troponin")
    assert finding is not None, "no coerce_numeric repair offered on `troponin`"
    answer = client.post(f"/project/{pid}/decision",
                         json={"kind": "apply", "subject": finding["id"]})
    assert answer.status_code == 200, answer.text

    after = _survey_row(client, pid, "troponin")
    assert after is not None and after["n_missing"] == 41, (
        f"the repair blanked {after and after['n_missing']} cells, not 41")

    prov = after["provenance"]
    assert prov is not None, (
        "`troponin` holds 41 blanks this app wrote and the survey row says "
        "nothing about where they came from — GUIDED-191 exactly")
    assert prov["n_created_by_the_app"] == 41
    assert prov["n_blank_in_the_file"] == 0
    assert len(prov["rows"]) == 41 and prov["rows_known"] is True, (
        "the count and the row list disagree, so one of them is derived")
    # WHICH PASS. `kind` is `apply` for all nine repairs, so the decision kind
    # alone cannot answer "what blanked this cell".
    assert prov["by"] and prov["by"][0]["pass"] == "coerce_numeric", (
        f"the record does not name the pass: {prov['by']}")
    assert "coerce_numeric" in prov["sentence"], (
        f"the sentence does not say which pass made the blanks: "
        f"{prov['sentence']!r}")


def test_the_recorded_rows_are_the_cells_that_actually_went_blank(client):
    """The row list is checked against the table, not against its own length.

    Trap #3: a count and a list that agree with each other prove nothing if
    both come from the same derivation. These labels are compared with the
    cells that are actually blank in the frame the user now has.
    """
    pid = _project(client, "clinical_labs.csv", "clinical", "readmitted")
    finding = _finding(client, pid, "coerce_numeric", "troponin")
    client.post(f"/project/{pid}/decision",
                json={"kind": "apply", "subject": finding["id"]})

    prov = _survey_row(client, pid, "troponin")["provenance"]
    column = api._project(pid).working_table["troponin"]
    actually_blank = {PROJ._label(i) for i in column.index[column.isna()]}
    assert set(prov["rows"]) == actually_blank, (
        "the recorded rows are not the rows that are blank in the table")

    original = pd.read_csv(DATA / "clinical_labs.csv")["troponin"]
    unparseable = set(original[original.isin(["positive", "negative"])].index)
    assert set(prov["rows"]) == {PROJ._label(i) for i in unparseable}, (
        "the recorded rows are not the cells the coercion could not read")


# ═══════════ (b) ONE mechanism, asserted as one ═════════════════════════════

def test_every_table_installing_pass_goes_through_the_one_door(client):
    """**The one-mechanism claim, enforced rather than described.**

    `Project._install` is the only place `self.df` is assigned and the only
    place `_history` grows, so the provenance record cannot be forgotten by a
    thirteenth writer — there is nowhere else to write the table from.

    This reads the source because the claim IS about the file: it is that no
    other assignment exists. The behavior half is every other test here.
    """
    source = (Path(PROJ.__file__)).read_text(encoding="utf-8")
    body = [ln for ln in source.splitlines()
            if ln.strip().startswith("self.df = ")
            or ".append((" in ln and "_history" in ln]
    # `_install` itself, and `revert_last_fix`, which restores a frame off the
    # stack rather than installing a new one — its blanks were recorded by
    # whichever pass first made them and re-filing them here would double-count.
    assert len(body) == 3, (
        "a pass assigns `self.df` outside `_install`, so it can change the "
        f"working table and file no record of it:\n" + "\n".join(body))


def test_the_reader_accepts_exactly_one_payload_shape(client):
    """No legacy branch beside the new one. Two accepted shapes is two
    mechanisms wearing one function name, and the stale one goes quiet rather
    than red — the survey still renders, just without the blanks."""
    legacy = [{"kind": "set_impossible_missing", "text": "4 were set to missing.",
               "payload": {"column": "sbp", "made_blanks": True,
                           "n_set": 4, "rows": [1, 2, 3, 4]}}]
    assert M.blanks_the_app_made(legacy) == {}, (
        "the reader still honors L48's single-column payload, so a writer "
        "could file in the old shape and nothing would say it had drifted")


# ═══════════ (c) the writers that are NOT blank writers, driven ═════════════

def test_read_as_binary_is_not_a_blank_writer(client):
    """Dismissed by driving it, not by reading it. `apply_read_as_binary`'s
    docstring says *"blanks stay blank"*; this is the assertion behind it."""
    frame = pd.read_csv(DATA / "clinic_visits.csv")
    findings = [f for f in engine.diagnose(frame)
                if f.fix_kind == "read_as_binary"]
    assert findings, "no read_as_binary repair on this fixture"
    for finding in findings:
        after, _desc = engine.apply_fix(frame, finding)
        record = M.blanks_made(frame, after, pass_name=finding.fix_kind)
        assert record == {}, f"{finding.id} blanked cells: {record}"


def test_reinfer_types_cannot_blank_a_cell(client):
    """`ml/import_doctor.py` ~L274-293, the read-time coercion the brief named
    as a candidate. It converts only where EVERY non-null value parses, so a
    blank is not reachable — and it is frozen, so the useful thing is a test
    that says why it needs no change."""
    from ml.import_doctor import reinfer_types
    frame = pd.DataFrame({
        "clean": ["1", "2", "3"],            # converts
        "mixed": ["1", "two", "3"],          # must be left as text
        "gappy": ["1", None, "3"],           # converts, blank stays blank
    })
    after = reinfer_types(frame)
    assert M.blanks_made(frame, after, pass_name="reinfer_types") == {}
    assert after["mixed"].tolist() == ["1", "two", "3"], (
        "a column that does not fully parse was converted, which is how this "
        "would become a blank writer")


def test_join_doctor_is_not_reachable_from_this_apps_working_table(client):
    """The other frozen module. It is the multi-file import path's, reached
    from `utils/combine*.py` and `pages/`, and nothing in `turbotab/` imports
    it — so its blanks are not this table's blanks."""
    root = Path(PROJ.__file__).resolve().parent
    importers = [p.name for p in root.glob("*.py")
                 if "join_doctor" in p.read_text(encoding="utf-8")
                 and not p.name.startswith("test_")]
    assert importers == ["grain.py"], (
        f"a turbotab module now reaches join_doctor: {importers}")
    assert "import" not in [
        ln.strip()[:6] for ln in (root / "grain.py").read_text(
            encoding="utf-8").splitlines() if "join_doctor" in ln], (
        "grain.py imports join_doctor rather than only naming it in prose")


# ═══════════ (d) the hedge, kept where it is still earned ═══════════════════

def test_a_melt_names_the_blanks_in_the_column_it_created(client):
    """The reshape case, and the answer is NOT a refusal — which is worth an
    assertion, because the refusal branch is the tempting one to reach for.

    `melt_repeated` rebuilds the index, and the 36 blanks it makes are in a
    `value` column that did not exist a moment earlier. Every one of them was
    written by this pass by definition, the labels name real rows of the frame
    the user now has, and `n_blank_in_the_file` is 0 rather than withheld — a
    column that came from nowhere brought nothing with it.
    """
    frame = pd.read_csv(DATA / "survey_instrument.csv")
    finding = next(f for f in engine.diagnose(frame)
                   if f.fix_kind == "melt_repeated")
    after, _desc = engine.apply_fix(frame, finding)
    record = M.blanks_made(frame, after, pass_name=finding.fix_kind)

    assert record.get(M.MADE_BLANKS) is True, (
        "the melt created a `value` column holding blanks and filed nothing")
    created = {b["column"]: b for b in record[M.BLANKS_MADE]}
    assert "value" in created and created["value"]["n"] == 36
    assert created["value"]["new_column"] is True
    assert len(created["value"]["rows"]) == 36

    reading = M.provenance("value", 36, M.blanks_the_app_made(
        [{"kind": "apply", "text": "Reshaped.", "payload": record}])["value"])
    assert reading["n_created_by_the_app"] == 36
    assert reading["n_blank_in_the_file"] == 0
    assert "melt_repeated" in reading["sentence"]


def test_a_reshaped_column_that_survives_the_pass_refuses_to_decompose(client):
    """**Where the hedge is still earned**, and it is the honest heir to the
    old field name. `n_blank_in_the_file` is `None`, never a number.

    A column that exists on both sides of a pass that renumbered the rows has
    no cell that can be compared with the cell it replaced. A net difference of
    blank counts would BE a number here, and it would be the wrong one the
    moment a pass both fills and blanks — trap #9 at the field layer.
    """
    before = pd.DataFrame({"lab": [1.0, None, 3.0, None, 5.0],
                           "who": list("abcde")})
    after = before.iloc[1:].reset_index(drop=True)   # dropped AND renumbered
    record = M.blanks_made(before, after, pass_name="drop_rows")
    opaque = record.get(M.BLANKS_UNATTRIBUTABLE) or []
    assert [o["column"] for o in opaque] == ["lab"], (
        f"every label now names a different row and the record claimed it "
        f"could still compare cells: {record}")
    assert opaque[0]["because"] == M.ROWS_RESHAPED

    reading = M.provenance("lab", 2, M.blanks_the_app_made(
        [{"kind": "apply", "text": "Dropped rows.", "payload": record}])["lab"])
    assert reading["n_blank_in_the_file"] is None, (
        "a split was asserted over a frame whose rows were rebuilt underneath "
        "it — this is the case the old field name existed for")
    assert reading["n_created_by_the_app"] == 0
    assert "not something the record can say" in reading["sentence"]


def test_an_ordinary_row_exclusion_stays_attributable(client):
    """The other side of the same rule, and the reason it is not simply
    *"index changed → refuse"*. Dropping rows keeps the surviving labels
    meaning the same rows, and treating that as opaque would erase the
    provenance of every column the first time a user set an eligibility
    criterion — the most ordinary thing this app does.

    Confirmed by content rather than by label, because `.reset_index(drop=True)`
    also produces a subset of a `RangeIndex` while renumbering everything.
    """
    before = pd.DataFrame({"lab": [1.0, None, 3.0, None, 5.0],
                           "who": list("abcde")})
    kept = before.iloc[1:]                       # dropped, labels preserved
    assert M.blanks_made(before, kept, pass_name="eligibility_criterion") == {}, (
        "an exclusion that blanked nothing reported an unattributable column")


def test_a_duplicated_row_label_is_a_count_without_rows(client):
    """The one case where a count is exact and the rows genuinely are not
    available: a label that picks out more than one row names none of them.
    `rows_known` says so rather than shipping a short list that a consumer
    would read as the whole set."""
    before = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[7, 7, 8])
    after = before.assign(b=[None, 2.0, None])
    record = M.blanks_made(before, after, pass_name="add_feature::ratio")
    block = record[M.BLANKS_MADE][0]
    assert block["column"] == "b" and block["n"] == 2
    assert block["rows_known"] is False and block["rows"] == []


# ═══════════ (e) both doors, and the page that renders it ═══════════════════

#: `(file, lens, target, pass, column)`. **Two target shapes, and each is
#: driven by a pass that actually blanks a cell ON THAT FIXTURE** — a skip here
#: would make the two-shape claim vacuous, which is `GUIDED-097`'s whole point.
#: `clinical_longitudinal.csv` has no `coerce_numeric` repair, so the
#: regression shape rides the impossibility pass, which also proves L48's
#: writer still agrees across both doors now that it goes through `_install`
#: instead of computing its own row list.
SHAPES = [
    pytest.param("clinical_labs.csv", "clinical", "readmitted",
                 "coerce_numeric", "troponin", id="classification-target"),
    pytest.param("clinical_longitudinal.csv", "clinical", "hba1c",
                 "set_impossible_missing", "sbp", id="regression-target"),
]


def _blank_the_column(client, pid: str, pass_name: str, column: str) -> None:
    if pass_name == "set_impossible_missing":
        answer = client.post(
            f"/project/{pid}/decision",
            json={"kind": "set_impossible_missing", "subject": column,
                  "payload": {"column": column}})
    else:
        finding = _finding(client, pid, pass_name, column)
        assert finding is not None, (
            f"no {pass_name} repair offered on `{column}`, so nothing below is "
            f"being driven")
        answer = client.post(f"/project/{pid}/decision",
                             json={"kind": "apply", "subject": finding["id"]})
    assert answer.status_code == 200, answer.text


@pytest.mark.parametrize("fixture,lens,target,pass_name,column", SHAPES)
def test_the_explore_card_and_the_survey_agree_after_a_repair(
        client, fixture, lens, target, pass_name, column):
    """`GUIDED-097`: two target shapes, two different blank-writing passes.

    The Explore card and the Preprocess survey read one `provenance`, and they
    must still agree once the blanks come from a repair under the frozen
    module rather than from the impossibility pass.
    """
    from ml import missingness_plan as MP

    pid = _project(client, fixture, lens, target)
    _blank_the_column(client, pid, pass_name, column)

    row = _survey_row(client, pid, column)
    assert row is not None and row["provenance"] is not None, (
        f"`{column}` gained blanks from `{pass_name}` and the survey row says "
        f"nothing about where they came from")
    assert pass_name in row["provenance"]["sentence"], (
        "the survey does not say which pass made the blanks")

    project = api._project(pid)
    cards = MP.missingness_cards(project.working_table, threshold=0.0,
                                 provenance=project.blank_provenance())
    card = next(c for c in cards if c["column"] == column)
    assert card["provenance"] == row["provenance"], (
        "the Explore card and the Preprocess survey describe the same blanks "
        "differently")


def test_the_page_file_wires_the_provenance_renderer_before_the_question():
    """Trap #6, and it had already fired: the server has composed this sentence
    since L48 and no surface on the page read it, so the mechanism question was
    asked over blanks the app made with the explanation sitting on the wire.

    **The name says `wires`, not `renders`, and that is trap #3b honored rather
    than described.** Nothing below observes a render; this asserts the
    renderer exists, is called, and is called BEFORE the mechanism question —
    an ordering claim that is genuinely about the file. The render itself is
    `test_the_page_actually_renders_the_provenance_sentence`, which runs the
    controller, because a grep cannot tell a page that reads a field from a
    page that mentions it.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert "prepProvHTML" in page, (
        "nothing on the page renders `provenance`, so the survey row's "
        "explanation of where its blanks came from is invisible")
    assert "prepProvHTML(col) + prepMechHTML(col)" in page, (
        "the provenance renderer exists and is not called before the "
        "mechanism question, which is the one place it has to be")
    assert "p.rows_known === false" in page, (
        "the page renders a short row list beside a larger count as though it "
        "were the whole set")


@pytest.mark.skipif(not H.available(), reason="no JS engine on this machine")
def test_the_page_actually_renders_the_provenance_sentence(client):
    """**The driven half.** The page's real controller, under node, against the
    real server's real `/preprocess` payload — and what is asserted is that the
    server's sentence comes out in the rendered HTML of the column whose blanks
    the app made.

    `GUIDED-037`'s lesson: a text search over `index.html` cannot tell a page
    that reads a field from a page that merely mentions it. This one ran the
    code.
    """
    pid = _project(client, "clinical_labs.csv", "clinical", "readmitted")
    _blank_the_column(client, pid, "coerce_numeric", "troponin")
    prep = client.get(f"/project/{pid}/preprocess").json()
    routes = {
        f"/project/{pid}": client.get(f"/project/{pid}").json(),
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/preprocess": prep,
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/evidence/plausibility": {"columns": []},
    }
    out = H.run("__emit({html: __harness.html('prepCols')});",
                routes=routes, search=f"?project={pid}")
    rendered = out["html"] or ""

    served = next(r for r in prep["columns"] if r["column"] == "troponin")
    sentence = served["provenance"]["sentence"]
    assert 'data-prep-prov="troponin"' in rendered, (
        "the provenance block never reached the DOM, so the mechanism question "
        "is on screen over 41 blanks the app made and the explanation is not")
    # The SERVER'S sentence, not a rendering of the numbers in it. A page that
    # recomposed the sentence would be the second composer GUIDED-098 cost a
    # loop, and it would pass an assertion that only looked for "41".
    assert sentence[:80] in rendered.replace("&#x27;", "'").replace("&amp;", "&"), (
        f"the page rendered something other than the server's sentence:\n"
        f"  served:   {sentence[:120]!r}\n  rendered: {rendered[:400]!r}")
    assert "coerce_numeric" in rendered, (
        "the rendered note does not say which pass made the blanks")
