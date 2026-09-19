"""The noun is "analyses"; "analyzes" is only ever the verb.

The American-English gate (tests/test_american_spelling.py) rewrites the
British "-yse" verb to "-yze" and cannot tell the verb from the noun, so an
over-correction slipped through it: the generated Methods opened with "All
analyzes were conducted using the Tabular ML Lab", the cohort report carried a
"## Cohort analyzes" heading, and twelve more sites said the same. The app
emits manuscripts; a misspelling in the first sentence of the Methods is the
first thing a reviewer sees.

The gate below is narrow on purpose. It looks for "analyzes" where only a noun
can stand — after a determiner or a qualifying adjective — and leaves the verb
alone: "EDA now analyzes the engineered data" and "the tab analyzes" are
correct and must stay.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# A word that can only precede a noun here. "now analyzes", "tab analyzes" and
# a sentence-initial "Analyzes patterns" do not match.
NOUN_CONTEXT = re.compile(
    r"\b(all|cohort|separate|eda|explainability|which|both|the|these|those|our|"
    r"no|further|additional|sensitivity|subgroup|secondary|primary|multiple|"
    r"several|two|three|per-cohort|planned)\s+analyzes\b",
    re.IGNORECASE,
)

SCANNED_PREFIXES = ("ml/", "utils/", "pages/", "app.py", "turbotab/")


def _source_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", *SCANNED_PREFIXES],
                         capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.split("\0") if p.endswith(".py")]


def test_the_matcher_fires_on_the_sentence_that_shipped_and_not_on_the_verb():
    """Positive and negative controls, so an empty result below means clean
    rather than blind."""
    assert NOUN_CONTEXT.search("All analyzes were conducted using the Tabular ML Lab")
    assert NOUN_CONTEXT.search("## Cohort analyzes")
    assert NOUN_CONTEXT.search("separate analyzes is a lot")
    assert not NOUN_CONTEXT.search("EDA now analyzes the engineered data.")
    assert not NOUN_CONTEXT.search("The model the feature-dropout tab analyzes, and")
    assert not NOUN_CONTEXT.search("**What this is:** Analyzes patterns in missing data")


def test_no_source_string_uses_analyzes_as_a_noun():
    files = _source_files()
    assert len(files) > 50, f"only {len(files)} files scanned; the enumeration is wrong"
    offenders = []
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NOUN_CONTEXT.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:90]}")
    assert not offenders, "\"analyzes\" used as a noun (the noun is \"analyses\"):\n" + "\n".join(offenders)
