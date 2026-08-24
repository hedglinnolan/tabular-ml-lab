"""L51-A1 — `GUIDED-212`. The guarantee that held for one marker in nine.

`[verify-at-build]` is described as **structurally forbidden from shipping as a
constant** in three places: `AGENT_ONBOARD.md` §00, `ROADMAP.md` condition 6 and
`LOOP.md` §04. Gate 6 in `docs/turbotab/tools/evidence.py` built its held-out
set from `_NUMBER.findall(m.group(1))` — the marker's **payload** — so a marker
with nothing after the colon contributed no numbers and was never checked
against anything.

**Eight of the nine markers in the corpus were bare.** One number was held out
of the code. Three documents called it structural.

## What changed, and the shape of the fix

A bare marker is a gate **failure** now, because an uncheckable marker is the
recorded-absence rule's case rather than a pass: *nobody said which number* and
*there is no number* were rendering as one value. Two sentinels make the second
sayable — `[verify-at-build: no number]` for a qualitative claim, and
`[verify-at-build: legend]` for the line that defines the marker.

Then the eight got their numbers, each read from the section it sits in.

## And the gate's reader was wrong in three ways, which only surfaced here

Holding out a second number ran the literal scan over code it had never really
read. `_code_only` replaces three regexes with `tokenize`:

1. They stripped single-line strings and comments only, so a number inside a
   **docstring** read as code.
2. Patched to eat triple quotes, they **mis-aligned** on a nested apostrophe and
   reported a line that did not contain the number.
3. On Python 3.12+ an f-string tokenizes into START / MIDDLE / END rather than
   one STRING, so prose **between the braces** leaked through as well.

A regex approximating Python is a second implementation of Python to keep in
sync — this project's most-repeated defect, one level down, inside the gate that
enforces the rule against it.

## The limit this does not remove, stated

Gate 6 holds out a **bare number**, so a generic one collides. Held out, `40`
matched six unrelated literals — a minimum-N guard, a truncation width, a
mean-string-length threshold. That is the gate's own docstring predicting
itself: *a bare-number scan widened too far cries wolf, and a gate that cries
wolf is one somebody switches off.* The `40` was resolved by reading §11 item
12, whose own remedy is to read software defaults from the user's installed
version rather than hard-code one — so there is no number to hold, and L50-F2
already ships all three as refusals. **It was not resolved by loosening the
gate**, and if a future marker names a number this generic the answer is the
same: read the section, not the scanner.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "docs" / "turbotab" / "tools" / "evidence.py"
RESEARCH = ROOT / "docs" / "turbotab" / "research"

sys.path.insert(0, str(TOOL.parent))
import evidence as E  # noqa: E402


def _check(cwd=ROOT, tool=TOOL):
    return subprocess.run([sys.executable, str(tool), "check"],
                          capture_output=True, text=True, cwd=str(cwd))


def _mirror(root: Path) -> Path:
    """A tree the gate can be pointed at, whose research corpus is a COPY.

    `TEST-098`. The planted-marker test below used to strip a marker out of the
    real `docs/turbotab/research/METABOLOMICS_PACK.md` and put it back in a
    `finally` — a git-tracked file mutated for a measured **0.666 s** over
    66,925 polls, during which two other files (three nodes) read the corrupted
    corpus and went falsely **RED**. A false red is the tolerable direction and
    it is still a false result in a two-hour sweep.

    `evidence.py` derives `ROOT` from its own `__file__` and accepts no root
    override, so **the copy carries the tool**: a real copy of `evidence.py`
    four levels down a mirrored tree makes `parents[3]` the mirror. Everything
    else is symlinked, so the gate reads the same package, the same emitters
    and the same modules it always does — the corpus is the only thing that
    moves, and it moves inside `tmp_path`.

    Symlinks and not copies deliberately: `evidence.py` imports the `turbotab`
    package to walk it, and that package imports `ml/` and `utils/` from the
    root beside it. Mirroring by copy would be a second checkout to keep in
    sync, which is this project's most-repeated defect one level down.
    """
    def link_siblings(real: Path, into: Path, carve: str) -> None:
        into.mkdir(parents=True, exist_ok=True)
        for entry in real.iterdir():
            if entry.name != carve:
                (into / entry.name).symlink_to(entry)

    link_siblings(ROOT, root, "docs")
    link_siblings(ROOT / "docs", root / "docs", "turbotab")
    link_siblings(ROOT / "docs" / "turbotab", root / "docs" / "turbotab",
                  "research")
    shutil.copytree(RESEARCH, root / "docs" / "turbotab" / "research")

    # The tool itself must be a REAL FILE at the mirrored depth: `.resolve()`
    # follows symlinks, so a symlinked tool would compute the real root and
    # read the real corpus — the mirror would be silently inert.
    tools = root / "docs" / "turbotab" / "tools"
    tools.unlink()
    link_siblings(TOOL.parent, tools, TOOL.name)
    shutil.copy2(TOOL, tools / TOOL.name)
    return tools / TOOL.name


def test_the_gate_passes_and_says_how_much_it_is_holding():
    """The positive control. A gate reporting nothing held is not a gate."""
    out = _check()
    assert out.returncode == 0, out.stdout[-2000:]
    match = re.search(r"(\d+) \[verify-at-build\] number\(s\) held out", out.stdout)
    assert match, out.stdout
    assert int(match.group(1)) >= 5, (
        f"only {match.group(1)} numbers are held out of the code. Before L51 it "
        f"was ONE, because eight of nine markers were bare — a drop back "
        f"toward that is the guarantee quietly un-holding")
    for phrase in ("declared to carry no number", "legend(s)"):
        assert phrase in out.stdout, (
            f"the summary does not report {phrase!r}, so a reader cannot tell "
            f"a marker that names nothing from one that names a number")


def test_every_marker_in_the_corpus_says_what_it_holds():
    """No bare markers left, asserted over the corpus rather than a list."""
    # THE POSITIVE CONTROL, and `GUIDED-045` is why it is written down: an
    # all-absence assertion passes hardest on an empty corpus, so *no bare
    # markers* is the same output for a clean corpus, a corpus with no markers
    # at all, and a glob that matched nothing.
    files = sorted(RESEARCH.glob("*.md"))
    assert len(files) >= 5, f"only {len(files)} research files found in {RESEARCH}"
    markers = sum(len(E._VERIFY.findall(p.read_text(encoding="utf-8")))
                  for p in files)
    assert markers >= 9, (
        f"only {markers} [verify-at-build] markers are in the corpus. The "
        f"assertion below is an absence claim and would pass on a corpus with "
        f"none; this is what makes it mean something")

    bare = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for m in E._VERIFY.finditer(line):
                payload = m.group(1).strip().lower()
                if payload in (E._NO_NUMBER, E._LEGEND):
                    continue
                if not E._NUMBER.findall(m.group(1)):
                    bare.append(f"{path.name}:{lineno}")
    assert not bare, (
        f"these markers name no number and declare nothing: {bare}. Each "
        f"guarantees exactly nothing, which is what `GUIDED-212` was")


def test_a_bare_marker_fails_the_gate(tmp_path):
    """The load-bearing half, planted rather than argued.

    A copy of the corpus with one marker stripped back to bare must turn the
    gate red. Without this the test above is satisfied by a corpus that happens
    to be clean, which is what it looked like for a dozen loops.

    **The copy is now a real copy.** It used to be the tracked file itself,
    edited and restored — see `_mirror`, and `TEST-098`. `tmp_path` was already
    on this signature and unused; the isolation was requested and then not
    used, so the fix was half-written for as long as the defect existed.
    """
    tool = _mirror(tmp_path)
    victim = tmp_path / "docs" / "turbotab" / "research" / "METABOLOMICS_PACK.md"
    original = victim.read_text(encoding="utf-8")
    marker = "**[verify-at-build: 50% and the SD-vs-MAD default]**"
    assert marker in original, "the marker this plant uses has moved"

    # THE MIRROR'S OWN CONTROL, and it is the whole reason the red below means
    # anything: an inert mirror — one whose tool resolved back to the real
    # corpus, or whose symlinks did not resolve — would go red for its own
    # reasons and read exactly like a caught plant.
    clean = _check(cwd=tmp_path, tool=tool)
    assert clean.returncode == 0, (
        f"the UNMODIFIED mirror does not pass the gate, so a red below would "
        f"be the mirror rather than the plant:\n{clean.stdout[-2000:]}"
        f"\n{clean.stderr[-800:]}")

    victim.write_text(original.replace(marker, "**[verify-at-build]**"),
                      encoding="utf-8")
    out = _check(cwd=tmp_path, tool=tool)
    assert out.returncode != 0, (
        "a bare [verify-at-build] passed the gate. That is `GUIDED-212` "
        "exactly, and it is what three documents said could not happen")
    assert "bare [verify-at-build]" in out.stdout, out.stdout[-1200:]
    assert "METABOLOMICS_PACK.md" in out.stdout, (
        f"the gate went red without naming the file the marker is in, so this "
        f"cannot tell the plant from an unrelated failure:\n{out.stdout[-1200:]}")

    # The real corpus was never touched, so there is nothing to restore — and
    # this is the assertion that says so rather than the comment.
    assert (RESEARCH / "METABOLOMICS_PACK.md").read_text(encoding="utf-8") \
        == original, "the plant reached the tracked corpus"


def test_the_declared_sentinels_are_not_a_way_round_it():
    """`no number` and `legend` are dispositions, not exemptions.

    Both are counted and reported separately, so a corpus that answered every
    marker with `no number` would show five held out becoming zero — visible in
    one line rather than hidden in a pass.
    """
    counted = 0
    for path in sorted(RESEARCH.glob("*.md")):
        for m in E._VERIFY.finditer(path.read_text(encoding="utf-8")):
            if m.group(1).strip().lower() in (E._NO_NUMBER, E._LEGEND):
                counted += 1
    assert counted <= 5, (
        f"{counted} markers declare they hold no number. Two are legends and "
        f"the rest are qualitative claims; a corpus where most markers declare "
        f"nothing has answered the gate rather than the question")


def test_the_reader_does_not_read_prose_as_code():
    """`_code_only`, and the three ways its predecessor was wrong.

    Asserted on constructed source rather than on the package, because the
    package's own text changes and this is a claim about the reader.
    """
    source = (
        'X = 1\n'
        'def f():\n'
        '    """A docstring mentioning 40 percent."""\n'
        '    # a comment mentioning 40\n'
        '    s = "a string mentioning 40"\n'
        "    t = f\"an f-string mentioning 40 and {X + 40}\"\n"
        '    return 41\n')
    code = E._code_only(source)
    assert "41" in code, "the reader dropped a real literal"
    # The f-string's INTERPOLATION is code and its prose is not — so exactly
    # one 40 survives, the one inside the braces.
    assert code.count("40") == 1, (
        f"the reader kept {code.count('40')} of the four prose 40s. A number "
        f"in a docstring, a comment, a string or an f-string's text is "
        f"discussion, not a constant:\n{code}")
    assert len(code.split("\n")) == len(source.split("\n")) + 1, (
        "the reader is not line-aligned, so a failure cannot name the line — "
        "which is how the first patched version reported a line that did not "
        "contain the number at all")


@pytest.mark.parametrize("marker,expect", [
    ("[verify-at-build: 0.37]", ["0.37"]),
    ("[verify-at-build: 50% and 75%]", ["50", "75"]),
    ("[verify-at-build: no number]", []),
    ("[verify-at-build: legend]", []),
])
def test_the_payload_forms_parse_as_they_are_documented(marker, expect):
    """The four forms the corpus now uses, each doing what its name says."""
    m = E._VERIFY.search(marker)
    assert m, marker
    payload = m.group(1).strip().lower()
    if payload in (E._NO_NUMBER, E._LEGEND):
        assert expect == []
    else:
        assert E._NUMBER.findall(m.group(1)) == expect
