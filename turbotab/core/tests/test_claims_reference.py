"""The claims reference and the claims' history stay what the registry generates (CLAIMS_REGISTRY.md
§5, §6; R0_PLAN.md R0a acceptance 5).

* The committed ``CLAIMS_REFERENCE.md`` is regenerated from the registry and must be identical, and
  ``turbotab/core/claims/history.jsonl`` must hold the version of every claim now in force.
* ``--check`` exits 1 when either is stale and 0 when both are current, and writes nothing.
* A claim whose content hash changes keeps its old version in the history: the generator appends
  the new version after it, and an old manifest's hash still resolves to the words it ran under.
* An id the history holds stays registered (§1: a retired id stays, marked retracted): where the
  registry drops one, the generator writes nothing and it and ``--check`` exit 1.
* Proposed claims are listed under "Ruled, not built" (§6).
"""
from __future__ import annotations

import json

from turbotab.core import claims
from turbotab.core.reference import claims as reference
from turbotab.core.claims import Fires
from turbotab.core.tests.test_claims_fixtures import registry, rule, teaching

REPO = reference.REPO


def test_the_committed_reference_and_history_are_what_the_registry_generates():
    reg = claims.registry()
    committed = reference.OUT.read_text(encoding="utf-8") if reference.OUT.is_file() else ""
    assert committed == reference.render(reg), (
        f"{reference.OUT.relative_to(REPO)} is stale: run `{reference.COMMAND}`")
    assert reference.pending(reg, reference.read_history(reference.HISTORY)) == [], (
        f"{reference.HISTORY.relative_to(REPO)} is stale: run `{reference.COMMAND}`")
    assert reference.main(["--check"]) == 0


def test_an_empty_registry_renders_its_headers_and_says_no_claim_is_registered():
    text = reference.render(claims.Registry())
    assert text.startswith("# TurboTab v2 claims reference\n")
    assert "No claim is registered yet." in text
    for heading in ("## Built", "## Ruled, not built", "## Retracted"):
        assert f"\n{heading}\n" in text


def test_a_proposed_claim_is_listed_as_ruled_not_built_and_a_built_one_as_built(tmp_path):
    proposed = teaching(id="planted.ruled", card="A ruled sentence.", ruled="R0_PLAN.md, planted",
                        tests=(Fires("planted/test_rule.py::test_not_yet", "construction"),))
    text = reference.render(registry(tmp_path, rule(), proposed))
    built, rest = text.split("\n## Ruled, not built\n", 1)
    ruled, retracted = rest.split("\n## Retracted\n", 1)
    assert "`planted.few_clusters`" in built and "`planted.few_clusters`" not in ruled
    assert "`planted.ruled`" in ruled and "R0_PLAN.md, planted" in ruled
    assert "`planted.ruled`" not in built
    assert "No claim is registered yet." not in text
    assert "Fewer than 10 clusters: intervals can still be too narrow." in built


def test_the_count_agrees_in_number_with_the_claims_it_counts(tmp_path):
    one = reference.render(registry(tmp_path, rule()))
    assert "It holds 1 claim: 1 built, 0 ruled and not built, 0 retracted." in one
    two = reference.render(registry(tmp_path, rule(), teaching()))
    assert "It holds 2 claims: 2 built, 0 ruled and not built, 0 retracted." in two


def _paths(tmp_path):
    out, history = tmp_path / "CLAIMS_REFERENCE.md", tmp_path / "history.jsonl"
    return out, history, ["--out", str(out), "--history", str(history)]


def test_check_exits_1_on_a_stale_reference_and_0_on_a_current_one_and_writes_nothing(tmp_path):
    reg = registry(tmp_path, rule())
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0  # writes both
    before = (out.read_bytes(), history.read_bytes())
    assert reference.main(["--check", *args], registry=reg) == 0
    out.write_text(out.read_text(encoding="utf-8") + "hand edit\n", encoding="utf-8")
    stale = out.read_bytes()
    assert reference.main(["--check", *args], registry=reg) == 1
    assert (out.read_bytes(), history.read_bytes()) == (stale, before[1])


def test_check_exits_1_when_the_history_lacks_a_version_in_force_and_writes_nothing(tmp_path):
    reg = registry(tmp_path, rule())
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=reg) == 0
    history.write_text("", encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 1
    assert history.read_text(encoding="utf-8") == ""


def test_a_changed_hash_keeps_the_old_version_and_appends_the_new(tmp_path):
    out, history, args = _paths(tmp_path)
    old = rule()
    new = rule(card="Fewer than {few} clusters: the intervals can be too narrow.", version=2)
    assert reference.main(args, registry=registry(tmp_path, old)) == 0
    first = reference.read_history(history)
    assert [(h["id"], h["hash"]) for h in first] == [(old.id, claims.content_hash(old))]

    assert reference.main(["--check", *args], registry=registry(tmp_path, new)) == 1
    assert reference.main(args, registry=registry(tmp_path, new)) == 0
    lines = reference.read_history(history)
    assert [h["hash"] for h in lines] == [claims.content_hash(old), claims.content_hash(new)]
    assert lines[0] == first[0], "the old version is kept as it was written"
    assert lines[0]["claim"] == claims.content(old)
    assert lines[0]["claim"]["card"] == old.card and lines[0]["version"] == 1
    assert reference.main(["--check", *args], registry=registry(tmp_path, new)) == 0


def test_generating_twice_appends_nothing_the_second_time(tmp_path):
    out, history, args = _paths(tmp_path)
    reg = registry(tmp_path, rule(), teaching())
    assert reference.main(args, registry=reg) == 0
    once = history.read_bytes()
    assert reference.main(args, registry=reg) == 0
    assert history.read_bytes() == once
    assert len(once.decode("utf-8").splitlines()) == 2


def test_a_status_change_alone_makes_the_reference_stale_but_not_the_history(tmp_path):
    """Status is derived, not content: writing a claim's missing test moves the reference only."""
    out, history, args = _paths(tmp_path)
    late = Fires("planted/test_late.py::test_written_later", "construction")
    reg = registry(tmp_path, rule(tests=(*rule().tests, late)))
    assert reference.main(args, registry=reg) == 0
    (reg.root / "planted" / "test_late.py").write_text("def test_written_later():\n    pass\n",
                                                       encoding="utf-8")
    assert reference.main(["--check", *args], registry=reg) == 1
    assert reference.pending(reg, reference.read_history(history)) == []


def test_each_history_line_is_one_json_object_naming_its_claim(tmp_path):
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule())) == 0
    (line,) = history.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert set(record) == {"id", "version", "hash", "claim"}
    assert record["id"] == "planted.few_clusters" and record["claim"]["id"] == record["id"]


def test_a_claim_deleted_from_the_registry_fails_the_check_and_the_generator(tmp_path, capsys):
    """An id is never dropped (§1): once the history holds it, the registry must still register
    it, so the generator writes nothing and both it and ``--check`` exit 1."""
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule(), teaching())) == 0
    before = (out.read_bytes(), history.read_bytes())
    capsys.readouterr()
    without = registry(tmp_path, rule())
    assert reference.main(["--check", *args], registry=without) == 1
    assert reference.main(args, registry=without) == 1
    assert (out.read_bytes(), history.read_bytes()) == before
    err = capsys.readouterr().err
    assert "planted.your_call" in err and "retract" in err
    out.write_text(reference.render(without), encoding="utf-8")  # a reference written regardless
    assert reference.main(["--check", *args], registry=without) == 1


def test_a_retracted_claim_stays_and_passes_the_check(tmp_path):
    out, history, args = _paths(tmp_path)
    assert reference.main(args, registry=registry(tmp_path, rule(), teaching())) == 0
    retracted = registry(tmp_path, rule(), teaching(retracted="2026-10-16: a planted retraction"))
    assert reference.main(args, registry=retracted) == 0
    assert reference.main(["--check", *args], registry=retracted) == 0
    assert [h["id"] for h in reference.read_history(history)] == [
        "planted.few_clusters", "planted.your_call", "planted.your_call"]
    assert "`planted.your_call`" in out.read_text(encoding="utf-8").split("\n## Retracted\n")[1]
