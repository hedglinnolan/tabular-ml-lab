"""`GUIDED-157` — the bulk binary repair recorded WHICH COLUMNS and never WHICH
VALUE BECAME 1.

The record the app kept for a bulk `read_as_binary`, driven on
`clinical_labs.csv` before this loop, verbatim:

    1 feature (`sex`) was read as binary.
    payload: {"fix_kind": "read_as_binary", "label": "read as binary",
              "findings": ["binary_text__sex"], "columns": ["sex"],
              "declined": [...], "n_selected": 1, "n_offered": 2}

`sex` holds `M` and `F`. Nothing in that sentence and nothing in that payload
says which of them is now the 1, so *"is the coefficient on `M` or on `F`"* has
**no answer anywhere in the record** — and the direction of every effect
estimate for that variable follows from it. On the product owner's NHANES
export the column is `gender ∈ {female, male}` and the question is the same one.

**No number in this defect is wrong.** A reported number is made
uninterpretable, which is trap #7's shape — the machine-readable form lossier
than the sentence — with the additional turn that the sentence did not carry it
either, so there was nothing for the payload to be lossier *than*.

## The shape this copies

`GUIDED-165`, at L47: one ambiguous kind became two declared kinds, each with a
machine-readable payload beside the sentence a person reads, because *"I
repaired this"* and *"I left it alone"* had to be distinguishable without
string-matching prose. Here the record already had its own kind; what it lacked
was the payload. So `engine.fix_encoding` reports what the transform does,
`api.apply_bulk` records it, and `repairs.sentence` states it.

## What is asserted, and against what

**The record is checked against the FRAME, never against itself.** A payload
saying ``{"M": 1, "F": 0}`` beside a rewrite that did the opposite would satisfy
any test that reads the receipt — and would be a worse defect than the one being
fixed, because it would assert something false rather than say nothing. So every
mapping claim here is re-derived from the rows: the positions that held `M`
before are read out of the column after, and they are required to hold exactly
the value the record says they hold.

**Two fixtures of different target shape** (`GUIDED-097`): `clinical_labs.csv`
on a **classification** target and `metabolomics_untargeted.csv` on a
**regression** target. The shape not covered is a **multiclass** target — no
fixture in `turbotab/sample_data/` pairs one with a `read_as_binary` group of
two or more members (`multiclass_stage.csv` has exactly one binary-text column,
`sex`, and `repairs.MIN_GROUP` is 2).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import engine, repairs as R                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

#: The two target shapes, and the fixture that carries a bulk binary group under
#: each. `GUIDED-097`'s rule: one fixture is one `float()` that happened to
#: succeed.
FIXTURES = {
    "classification": ("clinical_labs", "readmitted"),
    "regression": ("metabolomics_untargeted", "bmi"),
}


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _frame(pid):
    from turbotab.api import STORE
    return STORE.get(pid).df


def _project(fixture, target):
    """Upload, choose the target, and hand back the group as served."""
    client = _client()
    with open(DATA / f"{fixture}.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": (f"{fixture}.csv", handle, "text/csv")}).json()["id"]
    ok = client.post(f"/project/{pid}/decision",
                     json={"kind": "set_target", "payload": {"column": target}})
    assert ok.status_code == 200, ok.text[:300]
    group = client.get(f"/project/{pid}/repair_group/read_as_binary")
    assert group.status_code == 200, (
        f"{fixture} serves no read_as_binary group, so this fixture cannot see "
        f"the defect: {group.text[:200]}")
    return client, pid, group.json()


# ── the record says which value became 1, and the frame agrees ───────────────

@pytest.mark.parametrize("shape", sorted(FIXTURES))
def test_the_bulk_record_says_which_original_value_became_one(shape):
    """**The load-bearing assertion**, and it is re-derived from the rows.

    For every column the bulk repair applied, the recorded mapping is checked
    against the column as it now stands: the rows that held the level the
    record calls the 1 are read back and required to hold 1. A payload agreeing
    with itself is the receipt-only test this project already paid for once.
    """
    fixture, target = FIXTURES[shape]
    client, pid, group = _project(fixture, target)
    members = group["members"]
    assert len(members) >= 2, f"{fixture} has no group to apply, only {members}"

    before = _frame(pid).copy()
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": "read_as_binary",
        "payload": {"findings": [m["id"] for m in members]}})
    assert r.status_code == 200, r.text[:400]
    after = _frame(pid)

    said = next(d for d in client.get(f"/project/{pid}").json()["decisions"]
                if d["kind"] == "apply_bulk")
    encodings = said["payload"].get("encodings") or {}
    applied = said["payload"]["columns"]
    assert applied, "nothing was applied, so there is no mapping to check"

    for column in applied:
        enc = encodings.get(column)
        assert enc, (
            f"the record says `{column}` was read as binary and carries no "
            f"mapping for it, so which of its two values is now the 1 has no "
            f"answer anywhere in the record — GUIDED-157 exactly")
        assert enc["positive_values"] and enc["negative_values"]
        assert set(enc["mapping"].values()) == {0, 1}

        # THE FRAME, not the receipt. The rows that held each spelling are
        # located in the ORIGINAL column and read out of the REPAIRED one.
        for spelling, expected in enc["mapping"].items():
            where = before[column].astype(str) == spelling
            assert where.any(), (
                f"the record claims `{column}` held {spelling!r} and no row "
                f"did — the mapping names a level this table does not have")
            got = set(after.loc[where, column].dropna().tolist())
            assert got == {expected}, (
                f"the record says {spelling!r} in `{column}` became "
                f"{expected}, and those rows now hold {sorted(got)}. The "
                f"record and the rewrite disagree, which is worse than the "
                f"record saying nothing")

        # AND THE SENTENCE A PERSON READS. The payload is what the draft and
        # the export consume; the sentence is what the methods section carries,
        # and GUIDED-157 was both of them silent at once.
        for spelling in enc["positive_values"] + enc["negative_values"]:
            assert f"`{spelling}`" in said["text"], (
                f"the sentence does not name {spelling!r}, so a reader of the "
                f"methods section cannot tell which level is the 1:\n"
                f"  {said['text']}")
        assert "= 1" in said["text"] and "= 0" in said["text"]


@pytest.mark.parametrize("shape", sorted(FIXTURES))
def test_the_recorded_direction_is_the_engines_and_says_whether_it_was_known(shape):
    """The second fact the mapping carries, and it is not the same fact.

    `ml.binary_text.KNOWN_PAIRS` recognizes `yes`/`no` and `true`/`false`; for
    `M`/`F` the plan takes sorted order and **declares that it did**. *"The
    engine recognized this pair"* and *"the engine picked deterministically and
    said so"* are two different claims about the same 1, and a record carrying
    only the 1 cannot tell them apart. Checked against the plan rather than
    against a hard-coded expectation, so this cannot drift from the module that
    decides it.
    """
    from ml import binary_text as BT

    fixture, target = FIXTURES[shape]
    client, pid, group = _project(fixture, target)
    frame = _frame(pid)
    for member in group["members"]:
        enc = member["encoding"]
        assert enc, f"{member['id']} carries no encoding"
        plan = BT.read_as_binary_plan(frame[enc["column"]])
        assert enc["positive_known"] == bool(plan["positive_known"])
        assert BT._normalize(enc["positive"]) == plan["positive"]
        assert BT._normalize(enc["negative"]) == plan["negative"]


def test_a_repair_with_no_mapping_records_none_rather_than_an_invented_one():
    """Trap 9 at the record layer: return nothing rather than a wrong value.

    `coerce_numeric` rewrites a column and has no 1 and no 0 in it. An encoding
    manufactured for it would be the record asserting a direction the transform
    never chose, which is the defect this loop is closing pointed backwards.
    """
    from ml.import_doctor import ShapeFinding

    df = pd.DataFrame({"weight": ["72 kg", "81 kg", "66 kg", "70 kg", "75 kg"],
                       "sex": ["M", "F", "M", "F", "M"]})
    numeric = ShapeFinding(
        id="numeric_as_text__weight", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="coerce_numeric", confidence="high",
        params={"column": "weight"}, affected_columns=["weight"])
    assert engine.fix_encoding(df, numeric) is None

    # The positive control, so the None above is a refusal and not a broken
    # helper: the same call on the kind that DOES have a mapping returns one.
    binary = ShapeFinding(
        id="binary_text__sex", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="read_as_binary", confidence="medium",
        params={"column": "sex"}, affected_columns=["sex"])
    assert (engine.fix_encoding(df, binary) or {}).get("mapping") == {"M": 1,
                                                                     "F": 0}

    # And a column that is no longer binary gets nothing rather than a stale
    # mapping read off the finding's own params.
    already = pd.DataFrame({"sex": [1, 0, 1, 0, 1]})
    assert engine.fix_encoding(already, binary) is None


def test_the_mapping_names_every_spelling_that_maps_to_a_side():
    """`Male` and `male` are one level and two strings.

    `ml.binary_text` compares on a normalized token, so a column written both
    ways has one level with two spellings — and a record naming the first one
    it happened to see would be a claim that is right about half the rows.
    Written against a frame rather than a fixture file because no shipped
    fixture has this shape and the ones that come close have four levels, not
    two: `clinic_visits.csv`'s `sex` holds `Male`, `male`, `M`, `Female`,
    `female`, `F`, which is why the engine does not read it as binary at all.
    """
    from ml.import_doctor import ShapeFinding

    df = pd.DataFrame({"gender": ["Male", "male", "Female", "female", "MALE",
                                  "Female", "male"]})
    finding = ShapeFinding(
        id="binary_text__gender", severity="warning",
        title="x", detail="x", why_it_matters="x",
        fix_label="x", fix_kind="read_as_binary", confidence="medium",
        params={"column": "gender"}, affected_columns=["gender"])
    enc = engine.fix_encoding(df, finding)
    assert enc is not None
    assert set(enc["positive_values"]) == {"Male", "male", "MALE"}
    assert set(enc["negative_values"]) == {"Female", "female"}
    assert enc["mapping"] == {"Male": 1, "male": 1, "MALE": 1,
                              "Female": 0, "female": 0}
    assert enc["n_positive"] == 4 and enc["n_negative"] == 3

    # And the sentence carries all of them, for the same reason.
    said = R.sentence("read as binary", ["gender"], (), {"gender": enc})
    for spelling in ("Male", "male", "MALE", "Female", "female"):
        assert f"`{spelling}`" in said, said


def test_a_kind_with_no_mapping_keeps_the_sentence_it_always_had():
    """The other repairs are not given punctuation for a distinction they do
    not have. `read_as_binary` is the only kind with a per-column direction."""
    assert (R.sentence("read as numbers", ["a", "b"])
            == "2 features (`a`, `b`) were read as numbers.")
    assert (R.sentence("read as numbers", ["a", "b"], (), {})
            == "2 features (`a`, `b`) were read as numbers.")


def test_every_member_carries_its_own_mapping_and_not_only_the_worked_example():
    """The card's worked example is member[0]'s **by design**.

    `api.repair_group`'s own docstring says the example is the group's first
    member rather than a representative chosen by any cleverness. That is the
    right call and it is also why the mapping had to go on every member: a user
    selecting three columns could see the direction of one of them.
    """
    client, pid, group = _project("metabolomics_untargeted", "bmi")
    assert len(group["members"]) >= 3, group["members"]
    example_id = group["example"]["finding_id"]
    assert example_id == group["members"][0]["id"]

    for member in group["members"]:
        enc = member["encoding"]
        assert enc, (
            f"{member['id']} carries no mapping. It is member "
            f"{group['members'].index(member)} of {len(group['members'])} and "
            f"the worked example only covers {example_id}")
        assert enc["column"] == member["columns"][0]
    # The group-level index, for a consumer that does not walk the members.
    assert set(group["encodings"]) == {m["encoding"]["column"]
                                       for m in group["members"]}


# ── the page ─────────────────────────────────────────────────────────────────
#
# `pageharness.run` cannot reach this surface, and that is a property of the
# surface rather than of the harness. The panel is written into
# `[data-rg-body="<kind>"]`, which the handler locates with
# `document.querySelector` — the shim answers `null` there by design, so the
# whole `data-rg-show` → panel path returns early under the harness and the
# panel never exists to be read.
#
# So the renderer is lifted out of the page and run **as itself**, over the real
# `/repair_group` response, with the three helpers it calls lifted with it. That
# is weaker than a full drive in one way and stronger than a text search in
# every way that matters: the assertion is on rendered HTML produced by the
# page's own source, so a page that names `encoding` without rendering it fails.

def _page_function(name: str) -> str:
    """One top-level function's source, out of `web/index.html`.

    By indentation, not by brace counting: every function in the controller
    opens at two spaces and closes on a line that is exactly `  }`, and a brace
    scanner would have to know about the string and regex literals in between.
    The caller checks what it got.
    """
    lines = PAGE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith(f"  function {name}("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "  }")
    return "\n".join(lines[start:end + 1])


def test_the_picker_says_which_value_becomes_one_for_every_feature_offered():
    """The consumer, run. Each `data-rg-pick` button carries its own direction.

    The mapping reaching the picker matters at least as much as the mapping
    reaching the record: the record is written **after** the user consented,
    and the picker is where the consent happens. The worked example above it is
    member[0]'s by design, so without this the user selects `site` and `batch`
    having been shown the direction of `sex`.

    Trap #1 — `member.encoding` is a field the server composes, and a field
    nothing renders is a capability shipped without its consumer.
    """
    import json as _json
    import shutil
    import subprocess
    import tempfile

    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    _client_, pid, group = _project("clinical_labs", "readmitted")
    renderer = _page_function("repairGroupPanel")
    # The positive control for the lift: an extraction that silently caught the
    # wrong lines would make every assertion below vacuous.
    assert "data-rg-pick" in renderer and renderer.rstrip().endswith("}"), (
        "repairGroupPanel was not lifted out of the page intact:\n" + renderer)

    program = "\n".join([
        _page_function("esc"), _page_function("atControlSlot"),
        _page_function("miniTable"), "var PICKED = {};", renderer,
        "process.stdout.write(repairGroupPanel(" + _json.dumps(group) + "));"])
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

    picks = [b for b in PH.elements(out, "button") if "data-rg-pick" in b]
    assert len(picks) == len(group["members"]), (
        f"the picker rendered {len(picks)} buttons for "
        f"{len(group['members'])} members")
    for member in group["members"]:
        enc = member["encoding"]
        assert f"{enc['positive']} = 1" in out and f"{enc['negative']} = 0" in out, (
            f"the picker offers `{enc['column']}` and never says that "
            f"{enc['positive']!r} becomes 1, so a user selecting it is "
            f"consenting to a direction the page does not state.\n"
            f"rendered:\n{out}")
