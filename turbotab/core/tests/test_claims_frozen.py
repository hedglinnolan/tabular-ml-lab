"""A sentence renders from the status the generator froze, never from the source (CLAIMS_REGISTRY.md
§6; R0_PLAN.md R0a).

The shipped app holds no tests: the image leaves out ``**/tests`` and ``turbotab/test_*.py``
(``turbotab/deploy/Dockerfile.dockerignore``), and the release zip every ``tests`` directory
(``.github/workflows/release.yml``). Derived there, every claim that names a test would be proposed
and refuse to render. So a status is derived only where the source is complete (by the generator,
by ``--check``, by the meta-tests, and in strict mode), and the generator freezes each claim's id,
status and content hash in ``turbotab/core/claims/status.json``, which rendering reads.

* A built claim renders from a planted copy of the repository that leaves its tests out as the
  image does, and no status is derived there.
* A claim the frozen file lacks, or whose content hash differs from the frozen one, does not render
  and the refusal says to run the generator; with no frozen file, or one that does not read,
  nothing renders.
* Strict mode, on in the tests, also derives the status and refuses where the frozen one differs,
  so a stale file fails a test before it ships.

Each planted root is written here, so the expected status is the construction's.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from turbotab.core import claims
from turbotab.core.claims import Fires, NotRenderable, Registry
from turbotab.core.reference import claims as reference
from turbotab.core.tests.test_claims_fixtures import (FIRES, SILENT, planted_root, registry, rule,
                                                       teaching)

ID = "planted.few_clusters"
CARD = "Fewer than 10 clusters: intervals can still be too narrow."
METHODS = "With fewer than 10 clusters, the concern is named."
TEACHING_CARD = "Your call: here is what each choice does."
INSIDE = {"outcome": "binary"}
FROZEN = ("turbotab", "core", "claims", "status.json")  # the app's copy, under the repository root
RUN_THE_GENERATOR = re.escape("run python -m turbotab.core.reference.claims")


def _generate(tmp_path: Path, *planted: claims.Claim) -> Path:
    """A complete planted repository with the generator run on it, as CI builds from."""
    source = tmp_path / "source"
    source.mkdir()
    reg = Registry(planted, root=planted_root(source))
    assert reference.main(["--out", str(tmp_path / "CLAIMS_REFERENCE.md"),
                           "--history", str(tmp_path / "history.jsonl")], registry=reg) == 0
    return source


def _ship(tmp_path: Path, source: Path) -> Path:
    """What the image copies of ``source``: all but its ``tests`` directories and ``test_*.py``."""
    shipped = tmp_path / "shipped"
    shutil.copytree(source, shipped, ignore=shutil.ignore_patterns("tests", "test_*.py"))
    assert list(source.rglob("test_*.py")) and not list(shipped.rglob("test_*.py"))
    return shipped


def _no_derivation(monkeypatch) -> None:
    """From here on, reading the source for a status fails the test."""
    def derive(*_):
        raise AssertionError("rendering derived a status from the source")
    monkeypatch.setattr(Registry, "resolves", derive)
    monkeypatch.setattr(Registry, "test_exists", derive)


def test_a_built_claim_renders_where_its_tests_do_not_ship(tmp_path, monkeypatch):
    shipped = _ship(tmp_path, _generate(tmp_path, rule()))
    app = Registry([rule()], root=shipped)
    assert app.status(ID) == "proposed", "derived where no test ships, the claim is proposed"
    _no_derivation(monkeypatch)
    for ctx in (INSIDE, None):
        assert app.card(ID, ctx, strict=False) == CARD
        assert app.methods(ID, ctx, strict=False) == METHODS
    monkeypatch.delenv("TURBOTAB_CLAIMS_STRICT")  # the shipped app does not set it
    assert app.card(ID, INSIDE) == CARD


def test_a_claim_the_frozen_file_lacks_does_not_render_there(tmp_path, monkeypatch):
    shipped = _ship(tmp_path, _generate(tmp_path, teaching()))
    app = Registry([rule(), teaching()], root=shipped)
    _no_derivation(monkeypatch)
    assert app.card("planted.your_call", strict=False) == TEACHING_CARD
    with pytest.raises(NotRenderable, match=rf"^{ID} is not in the frozen status file .*: "
                                            rf"{RUN_THE_GENERATOR}"):
        app.card(ID, INSIDE, strict=False)


def test_a_claim_whose_content_changed_since_it_was_frozen_does_not_render(tmp_path, monkeypatch):
    shipped = _ship(tmp_path, _generate(tmp_path, rule()))
    changed = rule(card="Fewer than {few} clusters: a changed sentence.")
    assert claims.content_hash(changed) != claims.content_hash(rule())
    _no_derivation(monkeypatch)
    assert Registry([rule()], root=shipped).card(ID, INSIDE, strict=False) == CARD
    with pytest.raises(NotRenderable, match=rf"^{ID} has changed since .* froze its status: "
                                            rf"{RUN_THE_GENERATOR}"):
        Registry([changed], root=shipped).card(ID, INSIDE, strict=False)


@pytest.mark.parametrize("text", [None, "not json\n",
                                  '{"claims": [{"id": "planted.few_clusters"}]}'])
def test_with_no_frozen_file_or_one_that_does_not_read_nothing_renders(tmp_path, text):
    reg = registry(tmp_path, rule(), freeze=False)
    if text is not None:
        reg.frozen_path.parent.mkdir(parents=True)
        reg.frozen_path.write_text(text, encoding="utf-8")
    for strict in (True, False):
        with pytest.raises(NotRenderable, match=RUN_THE_GENERATOR):
            reg.card(ID, INSIDE, strict=strict)


def test_strict_mode_refuses_a_frozen_built_the_source_no_longer_builds(tmp_path):
    reg = registry(tmp_path, rule())
    assert reg.card(ID, INSIDE, strict=True) == CARD  # the frozen status and the source's agree
    (reg.root / "planted" / "test_rule.py").unlink()
    with pytest.raises(NotRenderable, match=rf"^{ID}: .* froze it as built, but the source now "
                                            rf"makes it proposed \(strict mode\): "
                                            rf"{RUN_THE_GENERATOR}"):
        reg.card(ID, INSIDE, strict=True)
    assert reg.card(ID, INSIDE, strict=False) == CARD  # as the shipped app, with no tests, renders


def test_strict_mode_refuses_a_frozen_proposed_the_source_now_builds(tmp_path):
    late = Fires("planted/test_late.py::test_written_later", "construction")
    reg = registry(tmp_path, rule(tests=(FIRES, SILENT, late)))
    with pytest.raises(NotRenderable, match=rf"^{ID} is proposed"):
        reg.card(ID, INSIDE, strict=True)  # the frozen status and the source's agree
    (reg.root / "planted" / "test_late.py").write_text("def test_written_later():\n    pass\n",
                                                       encoding="utf-8")
    with pytest.raises(NotRenderable, match="froze it as proposed, but the source now makes it "
                                            "built"):
        reg.card(ID, INSIDE, strict=True)
    with pytest.raises(NotRenderable, match=rf"^{ID} is proposed"):
        reg.card(ID, INSIDE, strict=False)


def test_the_generator_freezes_each_claim_s_id_derived_status_and_content_hash(tmp_path):
    proposed = teaching(id="planted.ruled", card="A ruled sentence.",
                        tests=(Fires("planted/test_rule.py::test_not_yet", "construction"),))
    retracted = teaching(retracted="2026-10-16: a planted retraction")
    source = _generate(tmp_path, rule(), proposed, retracted)
    frozen = json.loads(source.joinpath(*FROZEN).read_text(encoding="utf-8"))
    assert frozen["claims"] == [
        {"id": ID, "status": "built", "hash": claims.content_hash(rule())},
        {"id": "planted.ruled", "status": "proposed", "hash": claims.content_hash(proposed)},
        {"id": "planted.your_call", "status": "retracted", "hash": claims.content_hash(retracted)},
    ]


def test_the_default_registry_reads_the_frozen_file_beside_the_claims_package():
    reg = claims.registry()
    assert reg.frozen_path == claims.REPO.joinpath(*FROZEN)
    assert reg.frozen_path.parent == Path(claims.__file__).resolve().parent
    assert reg.frozen_path.is_file()


def test_a_frozen_file_rewritten_within_one_timestamp_tick_is_read_afresh(tmp_path):
    """A file system's timestamps can be coarser than two writes; the same size and mtime with
    another hash in it is still read as written."""
    reg = registry(tmp_path, rule())
    path, written = reg.frozen_path, reg.frozen_path.read_text(encoding="utf-8")
    st = path.stat()
    assert reg.card(ID, INSIDE, strict=False) == CARD
    other = written.replace(claims.content_hash(rule()), "0" * 64)
    assert other != written and len(other) == len(written)
    for text, renders in ((other, False), (written, True)):
        path.write_text(text, encoding="utf-8")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert path.stat().st_mtime_ns == st.st_mtime_ns and path.stat().st_size == st.st_size
        if renders:
            assert reg.card(ID, INSIDE, strict=False) == CARD
        else:
            with pytest.raises(NotRenderable, match="has changed since"):
                reg.card(ID, INSIDE, strict=False)
