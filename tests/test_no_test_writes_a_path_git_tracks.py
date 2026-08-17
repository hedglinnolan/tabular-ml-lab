"""`TEST-098`'s class, filed rather than left as four repaired instances:
**a test that writes a path git tracks.**

## Why this class needed a guard and not four fixes

Four tests wrote inside the checkout during a run and three wrote a tracked
path. **Every one of them restored.** After each, `git status` is clean and
`git diff` is empty — the generator rewrites byte-identically, the research doc
is put back and re-checksummed, the mtime is replaced, the stray CSV is
unlinked. The tree never drifts; only the disk moves, mid-run. That is why two
full suite sweeps never saw it, and it is why the detector here is a poller's
answer rather than `git status`'s.

The consequence is measured, not argued. The generator runs for **0.421 s**;
six concurrent readers doing exactly what the six cross-file readers of those
fixtures do logged **56 corrupted observations in 261 reads** — 55
`EmptyDataError`, and **one silent 59-row frame where 60 rows are committed**.
The silent short read is the one to fear: it does not raise, it computes
statistics on a partial matrix. The bare-marker window is **0.666 s** over
66,925 polls and its realized consequence is a false RED in two other files.

**And this is realizable at HEAD without any xdist adoption.**
`tests/test_a_fixed_row_names_a_test_that_actually_runs.py` spawns
`pytest -n auto --dist load` — per-test distribution, deliberately not
`loadfile`, measured and commented in that file — over every node a `FIXED` row
names. `GUIDED-212` names the bare-marker node and `TEST-084`/`TEST-097` name
the build-answered node, so the writers **and their readers** are already in
that set.

## What this asserts

1. No tracked test writes into the checkout — including through a subprocess.
2. The detector can find each of the four shapes it was built from. An absence
   claim whose instrument is untested is `GUIDED-045`'s case, and three of the
   four shapes are invisible to the pattern sweep that originally found them.
3. Destinations the resolver cannot compute are **counted**, so the blind spot
   cannot quietly grow into the answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import repo_write_guard as guard                                # noqa: E402

#: Tests permitted to write inside the checkout, each with the reason.
#:
#: EMPTY, AND THAT IS THE REPORTED RESULT rather than an omission — all four
#: known writers were repaired rather than excused. It is a dict rather than a
#: set for the reason `turbotab/rankings.py` gives about scopes: an exemption
#: with no argument is a classification nobody can revisit.
EXEMPT: dict = {}

#: What the resolver could not compute a destination for, at the commit this
#: was written. Not a budget to spend — the list is printed on failure so a new
#: entry has to be looked at rather than absorbed.
UNRESOLVED_CEILING = 32


def _relative(sites):
    return sorted(f"{s.key} {s.call}" for s in sites)


def test_no_tracked_test_writes_a_path_git_tracks(capsys):
    """The class, over every tracked test module.

    A repo write here is not automatically a bug in the assertion the test is
    making — it is a bug in *where* the test makes it. Every one of the four
    kept every assertion it had; three got stronger in the move.
    """
    repo, unresolved, spawns, n_files = guard.sweep()

    # THE INSTRUMENT'S OWN CONTROL, before its silence is quoted. A sweep that
    # enumerated nothing reports "no writers" in exactly the same words as a
    # clean corpus.
    assert n_files >= 250, (
        f"only {n_files} tracked test modules found; the enumeration is "
        f"looking in the wrong place and its silence means nothing")

    offenders = [s for s in repo if s.key not in EXEMPT]
    assert not offenders, (
        "these tests write inside the checkout:\n  "
        + "\n  ".join(repr(s) for s in offenders)
        + "\n\nEvery one of them will restore, and `git status` will be clean "
          "afterwards — that is the class, not a defense. Move the write to "
          "`tmp_path`, or add the site to EXEMPT with the reason it cannot be.")

    with capsys.disabled():
        print(f"\n  {n_files} tracked test modules · {len(repo)} repo "
              f"write(s) · {len(EXEMPT)} exempt · {len(spawns)} spawn(s) into "
              f"an in-repo script, followed · {len(unresolved)} unresolved "
              f"destination(s)")


def test_the_unresolved_destinations_are_counted_rather_than_ignored():
    """The blind spot, named and bounded.

    A destination built from a fixture or a loop variable resolves to nothing,
    and *nothing* is indistinguishable from *not a repo write* unless somebody
    counts it. `AGENT_ONBOARD.md` §07 trap 5b: a sweep returning zero is a
    claim.
    """
    _, unresolved, _, _ = guard.sweep()
    assert len(unresolved) <= UNRESOLVED_CEILING, (
        f"{len(unresolved)} write destinations no longer resolve, against "
        f"{UNRESOLVED_CEILING} when this was written. Each one is a write the "
        f"guard above cannot see:\n  "
        + "\n  ".join(_relative(unresolved)))


def test_a_spawn_into_an_in_repo_script_is_followed():
    """The half a pattern sweep structurally cannot do.

    `turbotab/test_the_fixture_constants_match_the_fixtures.py` returned
    **zero** hits across twenty write-shaped patterns, because it does not
    write — it spawns a script that rewrites twelve tracked files. If following
    stops working, the guard keeps passing and loses its most expensive case.
    """
    _, _, spawns, _ = guard.sweep()
    assert spawns, (
        "no tracked test spawns an in-repo script any more, so the follow-"
        "through is exercised by nothing and its correctness is untested")


# ── the positive control: the four shapes, planted ──────────────────────────
#
# Planted rather than taken from git history. History would work today and
# would stop working the moment these four are committed fixed — a control that
# decays into a tautology is `AGENT_ONBOARD.md` §07 trap #2 with a delay on it.
#
# Each plant is a real shape from a real writer, at a REAL module path, so the
# `__file__`-derived constants resolve exactly as they do in production.

_HOST = "turbotab/test_the_app_says_which_build_answered.py"

PLANTS = {
    "receiver write on a tracked file": '''
from pathlib import Path
RESEARCH = Path(__file__).resolve().parents[1] / "docs" / "turbotab" / "research"
def test_x():
    victim = RESEARCH / "METABOLOMICS_PACK.md"
    victim.write_text("bare", encoding="utf-8")
''',
    "metadata write on a tracked file": '''
import os
from pathlib import Path
PAGE = Path(__file__).resolve().parent / "web" / "index.html"
def test_x():
    os.utime(PAGE, (1, 2))
''',
    "arg0 write into a tracked directory": '''
from pathlib import Path
FIXTURES = Path(__file__).resolve().parent / "sample_data"
def test_x(frame, name):
    frame.to_csv(FIXTURES / name, index=False)
''',
    "subprocess into an in-repo writer": '''
import subprocess, sys
from pathlib import Path
DATA = Path(__file__).resolve().parent / "sample_data"
GENERATOR = DATA / "make_genomics_siblings.py"
def test_x():
    subprocess.run([sys.executable, str(GENERATOR)], cwd=str(DATA.parent))
''',
}


@pytest.mark.parametrize("shape", sorted(PLANTS), ids=sorted(PLANTS))
def test_the_detector_finds_each_shape_it_was_built_from(shape):
    """`GUIDED-045`. The sweep finds what it must before its silence counts."""
    repo, _, _ = guard.analyze(_HOST, source=PLANTS[shape])
    assert repo, (
        f"the detector did not flag the {shape!r} plant, so its silence over "
        f"the real corpus says nothing about that shape")
    assert all(guard._inside_repo(s.dest) for s in repo)


def test_the_detector_does_not_fire_on_a_tmp_path_write():
    """The other half of the polarity: a clean file must come back clean.

    A detector that flagged everything would satisfy the control above and be
    useless. This is the shape all four writers were moved TO.
    """
    clean = '''
import shutil, subprocess, sys
from pathlib import Path
DATA = Path(__file__).resolve().parent / "sample_data"
GENERATOR = DATA / "make_genomics_siblings.py"
def test_x(tmp_path):
    sandbox = tmp_path / "sample_data"
    sandbox.mkdir()
    shutil.copy2(GENERATOR, sandbox / GENERATOR.name)
    (tmp_path / "note.md").write_text("fine")
    subprocess.run([sys.executable, str(sandbox / GENERATOR.name)])
'''
    repo, _, _ = guard.analyze(_HOST, source=clean)
    assert not repo, (
        f"a test that reads a tracked file and writes only into `tmp_path` was "
        f"reported as a repo writer: {[repr(s) for s in repo]}")
