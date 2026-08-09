"""L48-C — `GUIDED-180`, scoped to the half that can be driven.

Eighteen live decision kinds had no `ACTION_CONTRACT` row, and **two of them —
`apply_bulk` and `route_missingness_bulk` — rewrite the working table**, so all
three of `devchecks.py`'s contract guards were inert on the highest-blast-radius
paths in the app. `ACTION_CONTRACT.get(kind)` returns `None` for an unlisted
kind and every consumer then returns `[]`.

**Filling all eighteen is not one loop**, and that ruling stands: each row is a
claim about whether a kind touches the table and how many things it makes stale,
and a wrong claim turns a guard into a false alarm, which is how a guard gets
switched off. So this loop does three things instead.

## 1 · The two that mutate, DERIVED BY DRIVING

Not reasoned. The input was changed, the table's content hash and the stale list
were watched, and what moved is what the row says. The numbers are in
`ACTION_CONTRACT`'s own comment and re-derived here.

## 2 · `unlisted` becomes a state instead of a hole

`.get(kind)` returning `None` means **unchecked**, and *unchecked* and
*declared as touching nothing* were rendering as one value. That is the seal's
`group_col: None` exactly, and `DESIGN_LANGUAGE.md` §09's recorded-absence rule
is the pattern — `undetermined` being first-class is the precedent.

Three states now: `declared`, `unclassified` (a decision, with the loop it is
due at, reported on every transition and written to `unclassified.jsonl`), and
`undispositioned` — a kind in neither table, which **is** a violation, because
it means a kind was added and nobody decided.

**Why `unclassified` is not a violation**, stated because the opposite is the
obvious move: `test_the_same_drive_with_no_planted_bug_is_clean` drives
`set_repeat_kind` and `set_unit_of_analysis`, both unclassified, and asserts
zero violations. That test is right — *"a check that fires on a correct drive
gets switched off within a day"* — and a drive using an unclassified kind is a
correct drive. The report is the answer; the alarm is not.

## 3 · The other sixteen are a list with deadlines, not sixteen guesses

`devchecks.UNCLASSIFIED`, and this file gates that it stays complete.

## What is NOT covered

- **`stale` for `apply_bulk` is declared `None`.** Three drives added zero, and
  none of them changed the COLUMN SET — which is the case that would stale a
  recorded selection. Declaring 0 from three runs that could not have produced
  anything else would be a measurement dressed as a contract.
- **The sixteen themselves.** Named, dated, unmeasured.
- **`EFFECTS`' twenty-nine missing sentences** — `GUIDED-181`, deliberately
  untouched, and the same argument applies one layer up.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "sample_data"


def _live_kinds() -> set:
    """Every decision kind `api.py` handles, derived rather than listed."""
    from turbotab import api

    source = Path(api.__file__).read_text(encoding="utf-8")
    live = set(re.findall(r'decision\.kind == "([a-z_]+)"', source))
    for group in re.findall(r'decision\.kind in \(([^)]*)\)', source):
        live |= {k.strip().strip('"\'') for k in group.split(",")}
    # The generic tail at the bottom of the handler, which accepts these with
    # no branch of their own.
    live |= {"dismiss", "undismiss", "flag", "unflag", "note", "defer"}
    return {k for k in live if k and k.replace("_", "").isalpha()}


def _client_and_project(fixture: str, target: str):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


def _snapshot(client, pid) -> Dict[str, object]:
    """The three quantities the contract guards actually read.

    The fingerprint is a content hash of values, labels and dtypes — the same
    reading `a_deferred_transform_leaves_the_table_byte_identical` takes, so
    *"nothing was touched"* is a comparison here too and not a claim.
    """
    from turbotab import api

    project = api.STORE.get(pid)
    served = client.get(f"/project/{pid}").json()
    frame: pd.DataFrame = project.df
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        + str(list(frame.columns)).encode()
        + str(list(frame.dtypes)).encode()).hexdigest()
    return {"fingerprint": digest,
            "stale": len(served.get("stale_downstream") or []),
            "decisions": len(served.get("decisions") or []),
            "columns": tuple(frame.columns)}


# ── 1 · the two that mutate ─────────────────────────────────────────────────

@pytest.mark.parametrize("group", ["recode_missing", "coerce_numeric"])
def test_apply_bulk_matches_the_contract_it_was_just_given(group):
    """Driven, on both groups `clinic_visits.csv` offers.

    `touches_table=True` is the load-bearing half: this is one of the two kinds
    that rewrite the working table with all three guards inert, and the guard
    that watches for a deferred action touching the table cannot fire on a kind
    it has no row for.
    """
    from turbotab import devchecks

    client, pid = _client_and_project("clinic_visits.csv", "outcome")
    got = client.get(f"/project/{pid}/repair_group/{group}")
    # `AUDIT-039`. THE FIXTURE IS SHIPPED AND THE PRECONDITION IS A FACT
    # ABOUT IT, so a skip here stands down over exactly the regression the
    # test exists to catch — and pytest counts a skip as not-a-failure.
    assert got.status_code == 200 and got.json().get("members"), (
        f"clinic_visits.csv no longer offers a {group} repair group "
        f"(status {got.status_code}). The contract this test checks is about "
        f"what `apply_bulk` writes for that group; no group means the contract "
        f"is unchecked, which is a finding rather than a reason to stop")
    ids = [m["id"] for m in got.json()["members"]]

    before = _snapshot(client, pid)
    posted = client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": group, "payload": {"findings": ids}})
    assert posted.status_code == 200, posted.json()
    after = _snapshot(client, pid)

    spec = devchecks.ACTION_CONTRACT["apply_bulk"]
    assert spec.touches_table is True
    assert before["fingerprint"] != after["fingerprint"], (
        f"apply_bulk over {len(ids)} finding(s) is declared to touch the "
        f"working table and the content hash did not move")
    assert spec.records is True
    assert after["decisions"] - before["decisions"] == 1, (
        "DRIVE-002's design is ONE decision covering N features, and the "
        f"record grew by {after['decisions'] - before['decisions']}")
    assert spec.stale is None and spec.because, (
        "`stale` is None and the reason is what makes it a recorded absence "
        "rather than a shrug")
    # AND THE REASON IS TRUE OF THIS RUN. The declaration says 0 was observed on
    # groups that do not change the column set; if this group DID change it,
    # the reason is wrong and the row needs re-deriving rather than trusting.
    assert before["columns"] == after["columns"], (
        f"{group} changed the column set, which is the case `apply_bulk`'s "
        f"`because` says was never driven — re-derive the row")


@pytest.mark.parametrize("fixture,target,branch,strategy,moves", [
    ("clinic_visits.csv", "outcome", "numeric", "indicator", True),
    ("clinic_visits.csv", "outcome", "categorical", "explicit_category", True),
    ("metabolomics_untargeted.csv", "bmi", "numeric", "impute_median", False),
], ids=["row-local numeric", "row-local categorical", "deferred numeric"])
def test_route_missingness_bulk_matches_the_contract_it_was_just_given(
        fixture, target, branch, strategy, moves):
    """Both sides of clause §06's litmus, and two fixtures of different target
    shape (`GUIDED-097`): `outcome` is classification, `bmi` is regression.

    `touches_table=None` is the *right* answer here rather than an evasion, and
    the third case is what proves it: a fitted strategy leaves the frame
    byte-identical while a row-local one rewrites it, on the same kind.
    """
    from turbotab import devchecks

    client, pid = _client_and_project(fixture, target)
    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    columns = [c["column"] for c in cards if c["dtype_route"] == branch][:3]
    if not columns:
        pytest.skip(f"{fixture} has no {branch} column with missing values")

    before = _snapshot(client, pid)
    posted = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness_bulk", "subject": branch,
        "payload": {"columns": columns, "branch": branch,
                    "mechanism": "not_informative", "strategy": strategy}})
    assert posted.status_code == 200, posted.json()
    after = _snapshot(client, pid)

    spec = devchecks.ACTION_CONTRACT["route_missingness_bulk"]
    assert spec.touches_table is None and spec.because
    assert (before["fingerprint"] != after["fingerprint"]) is moves, (
        f"{strategy} over {len(columns)} {branch} column(s): the table "
        f"{'did not move' if moves else 'moved'}, which is the opposite of "
        f"what clause §06 says about it")
    assert after["stale"] - before["stale"] == spec.stale == 1, (
        f"the cascade fired {after['stale'] - before['stale']} time(s) and the "
        f"contract says {spec.stale}")
    assert after["decisions"] - before["decisions"] == 1


# ── 2 · unlisted is a state ─────────────────────────────────────────────────

def test_a_kind_in_neither_table_is_a_violation():
    """The hole, closed. A kind nobody dispositioned no longer passes silently."""
    from turbotab import devchecks

    assert devchecks.classification("apply_bulk") == "declared"
    assert devchecks.classification("set_lens") == "unclassified"
    assert devchecks.classification("a_kind_nobody_wrote") == "undispositioned"

    quiet = devchecks.every_decision_kind_has_a_disposition("set_lens", {}, {})
    assert quiet == [], (
        "a DECLARED unclassified kind raises a violation. A drive that uses "
        "one is a correct drive, and a check that fires on a correct drive "
        "gets switched off within a day")
    loud = devchecks.every_decision_kind_has_a_disposition(
        "a_kind_nobody_wrote", {}, {})
    assert len(loud) == 1
    assert loud[0].check == "a_decision_kind_has_no_disposition"
    assert "a_kind_nobody_wrote" in loud[0].message
    # And it runs on every transition rather than only when called by hand.
    source = Path(devchecks.__file__).read_text(encoding="utf-8")
    assert "_guard(every_decision_kind_has_a_disposition" in source, (
        "the check exists and `check_transition` does not call it — a "
        "capability without its consumer, in the file that watches for them")


def test_an_unclassified_transition_is_written_down(tmp_path, monkeypatch):
    """`note_unclassified` records, and the index says so.

    Not a violation, and not silence either. Without this, *"no violations"*
    over a drive of unclassified kinds reads as *"everything was checked"*.
    """
    from turbotab import devchecks

    monkeypatch.setattr(devchecks, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(devchecks, "_SESSION", None)
    monkeypatch.setattr(devchecks, "enabled", lambda: True)

    devchecks.note_unclassified("set_lens", {"kind": "set_lens"})
    devchecks.note_unclassified("apply_bulk", {"kind": "apply_bulk"})
    session = devchecks.session()
    assert [row["kind"] for row in session.unclassified] == ["set_lens"], (
        "either the unclassified kind was not recorded, or a DECLARED kind "
        "was recorded as unclassified")
    assert session.unclassified[0]["due"] == devchecks.UNCLASSIFIED["set_lens"]
    assert (tmp_path / session.started_at / "unclassified.jsonl").exists()

    index = devchecks.write_index()
    text = Path(index).read_text(encoding="utf-8") if index else ""
    assert "no contract watching" in text and "`set_lens`" in text, (
        f"the drive index does not say which transitions ran unwatched: "
        f"{text[:400]!r}")


# ── 3 · the list stays complete ─────────────────────────────────────────────

def test_every_live_kind_is_declared_or_dated(capsys):
    """The gate. Every kind in exactly one table, and every deadline stated."""
    from turbotab import devchecks

    live = _live_kinds()
    declared = sorted(k for k in live if k in devchecks.ACTION_CONTRACT)
    dated = sorted(k for k in live if k in devchecks.UNCLASSIFIED)
    neither = sorted(live - set(devchecks.ACTION_CONTRACT)
                     - set(devchecks.UNCLASSIFIED))
    both = sorted(set(devchecks.ACTION_CONTRACT) & set(devchecks.UNCLASSIFIED))

    with capsys.disabled():
        print("\n  ── L48-C · GUIDED-180, the half that could be driven ──")
        print(f"  live decision kinds                 {len(live)}")
        print(f"    with a contract row               {len(declared)}")
        print(f"    declared unclassified, with a due loop  {len(dated)}")
        for kind in dated:
            print(f"        {kind:<28} {devchecks.UNCLASSIFIED[kind]}")
        print(f"    in neither table                  {len(neither)}")
        print(f"  contract rows whose `None` names its reason  "
              f"{sum(1 for s in devchecks.ACTION_CONTRACT.values() if s.because)}")
        print("  NOT done: the sixteen above, and GUIDED-181's 29 missing")
        print("  EFFECTS sentences. Named and dated, not guessed.")

    assert not both, f"{both} are in both tables, which is two answers"
    assert not neither, (
        f"{neither} are live decision kinds in neither `ACTION_CONTRACT` nor "
        f"`UNCLASSIFIED`. All three contract guards skip them silently. Add a "
        f"row you have DRIVEN, or a line in `UNCLASSIFIED` with the loop it is "
        f"due at — a guess in the contract is worse than a dated hole")
    for kind, due in devchecks.UNCLASSIFIED.items():
        assert re.match(r"^L\d+ — .+", due), (
            f"{kind}'s entry is {due!r}; an unclassified kind carries the loop "
            f"it is due at, or the list is a backlog nobody reads")
    # AND EVERY DATA-DEPENDENT ROW SAYS WHY. A bare `None` is the shrug this
    # part exists to remove.
    for kind, spec in devchecks.ACTION_CONTRACT.items():
        if spec.touches_table is None or spec.stale is None:
            assert spec.because, (
                f"{kind} declares a `None` with no reason, so *unchecked* and "
                f"*depends on the data* are the same object again")
