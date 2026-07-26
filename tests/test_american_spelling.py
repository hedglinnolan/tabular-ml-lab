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

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv",
             "build", "dist"}
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


def _source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, ROOT)
        if any(rel_dir == p or rel_dir.startswith(p + os.sep)
               for p in EXEMPT_PREFIXES):
            continue
        for name in filenames:
            if name.endswith((".py", ".md")):
                yield os.path.join(dirpath, name)


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
