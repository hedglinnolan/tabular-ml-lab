"""The app is written in American English. Keep it that way.

This is a real defect class, not pedantry: a researcher reading "analyse" and
"much commoner" beside "analyze" and "color" in the same product is reading
text that was clearly not proofread, and that undermines trust in the numbers
next to it. Mixed spelling also breaks Ctrl-F and any string assertion.

docs/audit/ is exempt: those files quote what the app printed at the time, as
evidence. Rewriting a quoted transcript falsifies the record.
"""
from __future__ import annotations

import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXEMPT_PREFIXES = (os.path.join("docs", "audit"),)

# British -> American. Bare stems, matched as substrings so inflections are
# caught too ("analyse", "analysed", "analysing", "unanalysed").
BRITISH = {
    "analys": "analyz (analyse/analysed/analysing)",
    "commoner": "more common",
    "labelled": "labeled",
    "labelling": "labeling",
    "modelled": "modeled",
    "modelling": "modeling",
    "summaris": "summariz",
    "standardis": "standardiz",
    "normalis": "normaliz",
    "generalis": "generaliz",
    "regularis": "regulariz",
    "recognis": "recogniz",
    "canonicalis": "canonicaliz",
    "memoris": "memoriz",
    "capitalis": "capitaliz",
    "artefact": "artifact",
    "colour": "color",
    "behaviour": "behavior",
    "defence": "defense",
    "honour": "honor",
    "judgement": "judgment",
    "favour": "favor",
    "whilst": "while",
    "amongst": "among",
}
# "analysis"/"analyses" are correct American English and contain "analys".
ALLOWED = re.compile(r"analysis|analyses|analyst", re.IGNORECASE)


def _listed(*args):
    out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", *args],
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p.endswith((".py", ".md"))]


def _source_files():
    """This repository's prose, asked of git rather than of a directory walk.

    **`TEST-106`, and the point is which question is being asked.** The walk
    enumerated *everything on disk that is not on a skip list*, so the skip
    list had to name every directory that is not source — and it named
    `.worktrees` while the harness puts its own worktrees in
    `.claude/worktrees/`. The gate therefore walked into a full nested checkout
    of this repository and failed on quoted historical prose that is exempt at
    its real path. **Adding `.claude` to the skip list is the same fix that
    already failed**: the list is a claim about what exists, and it decays the
    moment anything new appears beside it.

    `git ls-files` answers *what is this repository's source* directly, and
    every directory the skip list named is already gitignored — so the list
    stops being needed rather than getting one more entry.

    **Measured, because a gate must not get smaller quietly.** Walk: 600 files.
    Tracked plus untracked-but-not-ignored, same `docs/audit` exemption: 591.
    Tracked files LOST: **zero** — the walk was a strict superset. The 9 files
    dropped are `turbotab/sessions/*/{README.md,index.md,replay.py}`, generated
    session-replay material under a gitignored directory, which is not this
    repository's prose. And `--others --exclude-standard` is why this is
    stronger rather than merely different: a `.md` a contributor has written
    and not yet staged is still checked, which is the one thing the walk did
    that a bare `ls-files` would have lost.
    """
    for rel in _listed() + _listed("--others", "--exclude-standard"):
        # `git ls-files` emits forward slashes on every platform, but
        # `EXEMPT_PREFIXES` is built with `os.path.join` — so on Windows the
        # comparison below was 'docs/audit' == 'docs\audit', never true. The
        # exemption silently stopped applying, `docs/audit`'s quoted historical
        # prose was scanned, and the gate failed for a reason unrelated to the
        # code under test. A gate that is red for the wrong reason is worse
        # than one that is absent: it gets classified as a known failure, and
        # then it hides the real offence it was written to catch. It did
        # exactly that once, on the commit that introduced this comment.
        rel = rel.replace("/", os.sep)
        rel_dir = os.path.dirname(rel) or "."
        if any(rel_dir == p or rel_dir.startswith(p + os.sep)
               for p in EXEMPT_PREFIXES):
            continue
        yield os.path.join(ROOT, rel)


def test_the_enumeration_covers_the_source_and_nothing_ignored():
    """`TEST-106`'s control, because the fix moves what the gate READS.

    A change to an enumeration is a change to coverage, and an all-absence
    assertion like `test_no_british_spellings` reports the same clean nothing
    for a corrected repository and for a gate that stopped looking. So the
    population is asserted before its silence is quoted.
    """
    files = sorted(os.path.relpath(p, ROOT) for p in _source_files())
    assert len(files) >= 400, (
        f"the gate is reading {len(files)} files; it was 591 when the "
        f"enumeration moved to `git ls-files` and 600 under the walk it "
        f"replaced. A gate that quietly got smaller reports the same green")

    # The prose that matters most is in it. Named files rather than a count,
    # because a count is satisfied by any 400 files.
    for must in (os.path.join("docs", "turbotab", "LOOP.md"),
                 os.path.join("docs", "turbotab", "prompts",
                              "AGENT_ONBOARD.md"),
                 os.path.join("turbotab", "project.py")):
        assert must in files, f"{must} is not being checked"

    # AND NOTHING GIT IGNORES, which is the defect this replaced: the walk
    # descended into `.claude/worktrees/`, a full nested checkout, and failed
    # on `docs/audit/` prose at a path where the exemption no longer matched.
    ignored = subprocess.run(
        ["git", "-C", ROOT, "check-ignore", "--stdin"],
        input="\n".join(files), capture_output=True, text=True)
    assert not ignored.stdout.strip(), (
        f"the gate is reading files git ignores, which is how it read a "
        f"nested checkout of this repository:\n  "
        + "\n  ".join(ignored.stdout.strip().split("\n")[:10]))

    # The exemption still bites. `docs/audit/` quotes what the app printed at
    # the time; rewriting a quoted transcript falsifies the record.
    assert not [f for f in files if f.startswith(EXEMPT_PREFIXES[0])]
    assert _listed(), "git lists no source at all; the enumeration is broken"


def _offences_in(line):
    stripped = ALLOWED.sub("", line)
    return [b for b in BRITISH if b in stripped.lower()]


def test_the_detector_fires_on_the_spellings_it_is_about():
    """The other half of the polarity, and it had none.

    `test_no_british_spellings` is an absence claim over a filtered population.
    The population is now controlled above; this controls the MATCHER, so the
    two together mean the green is about the prose rather than about the gate.
    """
    for line, expected in (("we analysed the data", "analys"),
                           ("the colour of the label", "colour"),
                           ("modelling the behaviour", "modelling")):
        assert expected in _offences_in(line), (
            f"the matcher did not flag {expected!r} in {line!r}, so its "
            f"silence over the corpus says nothing")
    # And it does NOT fire on the American forms, or the gate is unusable.
    assert not _offences_in("we analyzed the color of the behavior")
    # `analysis` and `analyses` are correct American English and contain
    # `analys`; the exemption is what keeps this gate from being noise.
    assert not _offences_in("the analysis and the analyses of the analyst")


def test_no_british_spellings():
    offences = []
    for path in _source_files():
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = ALLOWED.sub("", line)
                for brit, amer in BRITISH.items():
                    if brit in stripped.lower():
                        offences.append(
                            f"{os.path.relpath(path, ROOT)}:{lineno} "
                            f"'{brit}' -> use '{amer}'")
    assert not offences, (
        "British spellings found; this app is written in American English:\n  "
        + "\n  ".join(offences[:25])
        + (f"\n  ... and {len(offences) - 25} more" if len(offences) > 25 else "")
    )
