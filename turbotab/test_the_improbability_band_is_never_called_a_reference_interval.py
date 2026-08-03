"""`MISC-018`, reopened by `MISC-019` and closed properly. L44-D.

**The row was `FIXED` and the fix reached the code and not the copy.** L42
renamed `get_reference_interval` → `get_improbability_band`, renamed the
predicate, updated six call sites and their locals, and rewrote the module
docstring. It left the phrase in **five user-facing strings**, the sharpest
being an EDA table header reading::

    'Reference Interval (NHANES p01–p99)'

which names the central 95% and prints the central 98% beside it. The label
was the defect and the parenthesis was the proof.

**Why the term matters**, from `research/CLINICAL_SURVEY_PACK.md` §A1.2: a
reference interval is a **defined quantity** — the central 95% of a measured
healthy reference population, per CLSI EP28-A3c. The p01/p99 pair is the
central 98% of *whatever this dataset contains*, healthy or not. Calling it a
reference interval tells a clinician the app compared their values against a
healthy population it never had.

**This is `MISC-019`'s class in the work of the loop that named it.** A row
whose item says a *word* must stop being used, closed by a fix that changed
the *identifiers*. `ledger.py check` was satisfied: the row named a test, and
the test passed — over the surface it covered.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: `physiology_reference` is the one module allowed to say the phrase, because
#: saying what the band is NOT is its job. Excluded structurally rather than by
#: a list that grows: it is the module that owns the definition.
OWNER = "ml/physiology_reference.py"

#: This file too — it quotes the false label to guard against it. `TEST-043`:
#: a guard that searches the tree has its own source inside the search space,
#: and the exclusion is structural rather than a silencer.
SELF = pathlib.Path(__file__).relative_to(ROOT).as_posix()

SHIPPED = sorted(
    p for p in [*ROOT.glob("ml/*.py"), *ROOT.glob("turbotab/*.py"),
                *ROOT.glob("pages/*.py"), *ROOT.glob("utils/*.py"),
                ROOT / "turbotab" / "web" / "index.html"]
    if p.exists() and not p.name.startswith("test_")
)

#: The band being CALLED one. Not any mention — a sentence that says the band
#: *is not* a reference interval is the honest disclosure and must stay
#: sayable, or the fix becomes a word ban and the next writer routes around it.
#: Any use of the phrase. The FIRST version tried to match only the
#: label-shaped uses — `the|a|an|NHANES` then the phrase — and missed the
#: sharpest string that actually shipped, `'Reference Interval (NHANES
#: p01–p99)'`, because a quote preceded it. A detector tuned to the shapes you
#: remember misses the shape you wrote, which is trap 5 in a regex.
#:
#: So: match the phrase, and subtract the honest disclosure. The negative
#: control below is what keeps that from becoming a word ban.
MENTIONS = re.compile(r"reference\s+intervals?\b", re.I)

DENIES_IT = re.compile(
    r"(?:is\s+)?not\s+an?\s+reference\s+interval"
    r"|NOT\s+A\s+REFERENCE\s+INTERVAL"
    r"|reference\s+interval\s+is\s+(?:a\s+defined|the\s+central)", re.I)


def _lines(path: pathlib.Path):
    """Lines the app could SAY, skipping comment-only ones.

    A comment recording what a line used to say — `# Was 'Reference Interval
    (NHANES p01-p99)'` — is documentation of the fix, not the app making the
    claim. Without this the guard flags the note explaining its own subject,
    which is `TEST-043`'s class and the third time this loop family has hit
    it. The rule is structural: a line whose first non-space character starts
    a comment is not a user-facing string.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith(("#", "<!--", "*", "/*")):
            continue
        yield i, line


def test_no_shipped_surface_calls_the_band_a_reference_interval():
    """The standing check, over the whole shipped surface rather than the
    call sites the rename touched."""
    offenders = []
    for path in SHIPPED:
        rel = path.relative_to(ROOT).as_posix()
        if rel in (OWNER, SELF):
            continue
        for number, line in _lines(path):
            if DENIES_IT.search(line):
                continue                       # the honest disclosure
            if MENTIONS.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()[:90]}")
    assert not offenders, (
        f"these call the p01/p99 improbability band a reference interval: "
        f"{offenders}. A reference interval is the central 95% of a measured "
        f"HEALTHY population (CLSI EP28-A3c); p01/p99 is the central 98% of "
        f"whatever this dataset holds. Say `improbability band`.")


def test_the_detector_sees_the_strings_that_actually_shipped():
    """The positive control. Every assertion above is an absence claim, and
    these five are quoted from the diff that removed them."""
    shipped = [
        "'Reference Interval (NHANES p01–p99)': f\"{lo}-{hi}\",",
        'f"{col}: {rate:.1%} values outside NHANES reference interval "',
        '"Review units and validate values against NHANES reference intervals."',
        '"Empirical plausibility checks found values outside NHANES reference intervals. "',
        '" outside the NHANES reference interval (" + num(b.low, 2) + "–" +',
    ]
    for line in shipped:
        assert MENTIONS.search(line), (
            f"the detector misses a string that actually shipped: {line!r} — "
            f"so its silence over the tree means nothing")


def test_the_detector_leaves_the_honest_disclosure_alone():
    """The negative control, and it is what stops this becoming a word ban.

    The app must be able to say *this is not a reference interval*, because
    that sentence is the correction. A guard that forbids the phrase outright
    forces the next writer to paraphrase around it and the meaning goes.
    """
    honest = [
        "That band is not a reference interval — a reference interval is "
        "the central 95% of a healthy reference population",
        "improbability band (p01–p99), which is not a reference interval. ",
        "#   NOT a reference interval, which is the central 95% (`MISC-018`).",
    ]
    for line in honest:
        assert DENIES_IT.search(line), (
            f"the disclosure detector misses an honest correction: {line!r}")


def test_the_owner_module_still_defines_what_it_is_not():
    """The correction has to survive somewhere, or removing the false label
    leaves the user with a number and no way to read it.

    §A1.2's distinction is the whole content of `MISC-018`, and a fix that
    deleted every mention would take the explanation with the error.
    """
    owner = (ROOT / OWNER).read_text(encoding="utf-8")
    assert "not a reference interval" in owner.lower() or \
           "NOT A REFERENCE INTERVAL" in owner, (
        "physiology_reference no longer says what the band is not")
    assert "central 95" in owner, (
        "the module no longer states the quantity a reference interval IS, "
        "which is what makes the distinction checkable")
    assert "get_improbability_band" in owner
    # The old name survives EXACTLY ONCE, in the rename note that records what
    # it used to be called — trap 8's rule that the record cite something real.
    # Anything more is the name coming back.
    assert owner.count("get_reference_interval") == 1, (
        f"`get_reference_interval` appears {owner.count('get_reference_interval')} "
        f"times in core; one is the rename note, more is the name returning")


def test_the_eda_header_names_the_band_and_its_percentiles():
    """The sharpest instance, pinned by content rather than by absence.

    The header printed `p01–p99` beside the words `Reference Interval`. Both
    halves matter: the percentiles are what make the label checkably wrong, so
    the corrected header keeps them.
    """
    source = (ROOT / "ml" / "eda_actions.py").read_text(encoding="utf-8")
    assert "'Improbability band (NHANES p01–p99)'" in source, (
        "the EDA table header no longer names the band and its percentiles")
