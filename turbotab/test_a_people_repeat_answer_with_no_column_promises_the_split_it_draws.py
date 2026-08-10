"""`DRIVE-024` — the app said whole people would be held out, and held out rows.

Answering *"Yes, people repeat"* without naming the column that identifies the
person is a legitimate state: `set_grain` accepts it, and `grain.seal_basis`
correctly returns `undetermined` rather than `grouped`. **The engine was honest
about it and two composed sentences were not.**

`api._disclosures` served `grain.answer_disclosure(answer, group_col)`, which
formatted one sentence with `group_col or ""`:

> Recorded: people repeat, identified by ``. **Whole people will be held out
> rather than individual rows, so nobody appears on both sides of the split.**

Completing the chain and sealing then drew the held-out rows **by row**, and
`seal_disclosure` said so on the same page:

> … drawn BY ROW because the data's shape is unknown … the same person is on
> both sides and held-out performance will read better than the model is.

Two sentences, one screen, opposite claims — and the false one is the one shown
at the moment of the decision. `project.set_grain`'s own comment on the escape
hatch names the class in advance: *a promise the split did not keep is worse
than the wrong confident answer the option exists to avoid.*

The transcript had the same defect one field over, composing *"people repeat,
identified by 'None'"* — a recorded sentence asserting a column that does not
exist, which travels into the methods section.

## Why it surfaced now

It was **unreachable through the Guided door until L58**. `state_grain` sat in
`HANDLED_QUESTION_KEYS` claiming a card nothing had built (`DRIVE-017`), so no
human could press *"Yes, people repeat"* at all. Opening that door is what makes
this reachable in one press, which is why it is fixed in the same loop.
"""
from __future__ import annotations

from pathlib import Path

DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, fixture: str) -> str:
    with (DATA / fixture).open("rb") as handle:
        return client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]


def _decide(client, pid, kind, payload):
    resp = client.post(f"/project/{pid}/decision",
                       json={"kind": kind, "payload": payload})
    assert resp.status_code == 200, (kind, resp.status_code, resp.text[:300])
    return resp.json()


def test_the_disclosure_describes_the_split_the_seal_actually_draws(capsys):
    """The load-bearing claim, and it is asserted against the DRAWN split.

    Not against the sentence's own wording: the two disclosures are compared to
    the basis the lockbox recorded, so a rewrite that made them agree with each
    other and not with the split would still fail here.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    _decide(client, pid, "set_target", {"column": "age"})

    # The state under test, established from the record rather than assumed.
    after = _decide(client, pid, "set_grain", {"answer": "people_repeat"})
    assert after["grain"]["group_col"] is None
    assert after["grain"]["basis"] == "undetermined", (
        "this fixture no longer produces the state this claim is about")
    said = after["disclosures"]["grain"]

    # Complete the chain and seal, so the promise can be checked against the
    # split rather than against another sentence.
    _decide(client, pid, "set_repeat_kind", {"kind": "time_points"})
    _decide(client, pid, "set_unit_of_analysis", {"unit": "record"})
    _decide(client, pid, "set_temporal_prediction", {"temporal": False})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    sealed = _decide(client, pid, "seal", {})

    lockbox = sealed["lockbox"]
    assert lockbox["group_col"] is None
    assert lockbox["exploratory"] is True, (
        "an undetermined basis that is not labeled exploratory is a different "
        "defect and this claim is not about it")

    # THE ASSERTION. The split is by row; the sentence shown at the decision
    # must not promise otherwise.
    assert "Whole people will be held out" not in said, (
        f"the grain disclosure promises a grouped split and the seal drew "
        f"{lockbox['seal_basis']!r}:\n  {said}")
    assert "BY ROW" in said, (
        f"the disclosure does not say how the rows are actually drawn:\n  {said}")
    # And it does not render an empty name where a column would go.
    assert "identified by ``" not in said and "identified by `None`" not in said

    # The other half: with a column named, the promise is TRUE and must stay.
    pid2 = _project(client, "clinical_longitudinal.csv")
    _decide(client, pid2, "set_target", {"column": "age"})
    grouped = _decide(client, pid2, "set_grain",
                      {"answer": "people_repeat", "group_col": "subject_id"})
    assert grouped["grain"]["basis"] == "grouped"
    assert "Whole people will be held out" in grouped["disclosures"]["grain"], (
        "the fix removed the true promise along with the false one")
    assert "`subject_id`" in grouped["disclosures"]["grain"]

    with capsys.disabled():
        print(f"\n  no column  basis={after['grain']['basis']!r} "
              f"sealed={lockbox['seal_basis']!r}")
        print(f"  named      basis={grouped['grain']['basis']!r}")


def test_the_transcript_does_not_name_a_column_called_none(capsys):
    """The record's own half.

    `Decision.text` is quoted verbatim by the page's acknowledgment row and by
    the methods section, so a sentence naming `'None'` is a false claim in two
    places downstream of one composition.
    """
    client = _client()
    pid = _project(client, "clinical_longitudinal.csv")
    _decide(client, pid, "set_target", {"column": "age"})
    record = _decide(client, pid, "set_grain", {"answer": "people_repeat"})

    text = [d["text"] for d in record["decisions"] if d["kind"] == "set_grain"][-1]
    assert "identified by 'None'" not in text, text
    assert "None" not in text, text
    assert "no column identifying the person" in text, text

    # With a column, the name is the column's and is still there.
    pid2 = _project(client, "clinical_longitudinal.csv")
    _decide(client, pid2, "set_target", {"column": "age"})
    rec2 = _decide(client, pid2, "set_grain",
                   {"answer": "people_repeat", "group_col": "subject_id"})
    text2 = [d["text"] for d in rec2["decisions"] if d["kind"] == "set_grain"][-1]
    assert "identified by 'subject_id'" in text2, text2

    with capsys.disabled():
        print(f"\n  no column: {text}")


def test_every_grain_answer_records_a_sentence_with_no_empty_name(capsys):
    """The sweep, so this closes on the class rather than on the instance.

    All four answers, each with and without a grouping column — eight
    combinations, and the count is reported including the ones that pass. A
    sweep that terminated at the answer it was pointed at would be §08 check 5's
    own failure.
    """
    from turbotab import grain

    client = _client()
    rows = []
    for answer in grain.ANSWERS:
        for group_col in (None, "subject_id"):
            pid = _project(client, "clinical_longitudinal.csv")
            _decide(client, pid, "set_target", {"column": "age"})
            payload = {"answer": answer}
            if group_col:
                payload["group_col"] = group_col
            resp = client.post(f"/project/{pid}/decision",
                               json={"kind": "set_grain", "payload": payload})
            if resp.status_code != 200:
                # `one_row_per_person` on a repeating table is a 409 by design,
                # and that refusal is a different claim's subject. Counted and
                # named rather than dropped.
                rows.append((answer, group_col, resp.status_code, "", ""))
                continue
            body = resp.json()
            said = body["disclosures"]["grain"]
            text = [d["text"] for d in body["decisions"]
                    if d["kind"] == "set_grain"][-1]
            rows.append((answer, group_col, 200, said, text))

    with capsys.disabled():
        print("\n  ── every grain answer, with and without a column ──")
        for answer, group_col, status, said, text in rows:
            print(f"  {answer:<22} col={str(group_col):<12} {status}")
        print(f"  {len(rows)} combinations, "
              f"{sum(1 for r in rows if r[2] == 200)} recorded, "
              f"{sum(1 for r in rows if r[2] != 200)} refused")

    for answer, group_col, status, said, text in rows:
        if status != 200:
            continue
        assert "``" not in said, (answer, group_col, said)
        assert "'None'" not in text and "`None`" not in said, (answer, group_col)
    assert len(rows) == 8
