"""One statement of how a column name is matched against a known vocabulary.

**The remedy for a class this project has now fixed five times.** Each time it
was the same defect and each time it was repaired in place:

| Registry | Where | How it failed |
|---|---|---|
| `CLINICAL_VARIABLES` | `ml/clinical_units.py` | substring, so `hba1c_proxy` inherited HbA1c's units |
| the NHANES reference | `ml/physiology_reference.py` | substring, so a proxy inherited a licence to propose deleting entries |
| `theory_anchors` | `utils/theory_anchors.py` | substring fallback, so a reworded finding silently lost its theory link |
| `_KNOWN_UNITS` | `ml/import_doctor.py` | the bare letters `l` and `s`, so `Bachelors` mixed units (`IMPORT-267`) |
| `entity_id_patterns` and two siblings | `ml/triage.py` | substring, so `subjective_wellbeing` is a patient identifier |

Three were fixed individually with the same repair — **exact key or declared
alias; an unknown name yields silence** — and the fourth and fifth arrived
anyway, because the remedy lived in three separate modules and nothing carried
it across. `FEATURE_PARITY.md` calls that principle-locality: *a principle
written in one place and applied in another is the same silence as a capability
with no row.*

So it is written once here, and
`tests/test_a_name_registry_matches_exactly_or_says_nothing.py` enumerates the
sites that must use it and fails on a sixth.

## Why silence, and not a looser match

A name that is not in the registry gets **no answer**, not a guessed one. That
is deliberate and it is stated in `ml/physiology_reference.py` in as many words:
*silence is a gap, and a wrong bound is a claim.* A gap is visible to the user
and costs a question; a wrong claim is invisible and licenses an action — a
deleted entry, a stratified split on a wellbeing score, a units warning on a
column of education levels.

## Why this does not become a sixth name list

It holds no names. A registry is passed in by whoever owns the vocabulary; this
module owns only the **matching rule**, which is the part that kept being got
wrong. `turbotab/grain.py` states the other half and it still binds: *name lists
cannot close a question about the shape of the data and must not be tuned as
though they could.* Exact matching makes a name list honest about its own
coverage; it does not make one sufficient.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence

# Everything that is punctuation between words in a column name. `subject_id`,
# `subject-id`, `Subject ID` and `SubjectID` are one name spelled four ways, and
# a registry that distinguishes them is a registry that will miss three.
_SEPARATORS = re.compile(r"[\s_\-.·/\\()\[\]]+")


def normalize(name: object) -> str:
    """One name, in the form a registry is keyed on.

    Case and separators only. Nothing is stemmed, nothing is truncated, and no
    plural is folded — every one of those is a substring match wearing a
    normalizer's clothes, and each would reintroduce the defect this module
    exists to close.
    """
    return _SEPARATORS.sub("", str(name).strip().lower())


def build(registry: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """A lookup from every normalized spelling to its canonical key.

    Raises on a collision rather than resolving one, because two canonical keys
    claiming one spelling is a question the data cannot answer and a silent
    winner is exactly the confidently-wrong outcome the remedy is for.
    """
    lookup: Dict[str, str] = {}
    for canonical, aliases in registry.items():
        for spelling in [canonical, *(aliases or ())]:
            key = normalize(spelling)
            if not key:
                continue
            if key in lookup and lookup[key] != canonical:
                raise ValueError(
                    f"{spelling!r} is claimed by both {lookup[key]!r} and "
                    f"{canonical!r}. Two canonical keys for one spelling is a "
                    f"question the data cannot settle, and picking one quietly "
                    f"is the failure this registry exists to prevent.")
            lookup[key] = canonical
    return lookup


def match(name: object, lookup: Mapping[str, str]) -> Optional[str]:
    """The canonical key for this name, or `None`.

    `None` is an answer. It means *this vocabulary does not describe this
    column*, which is a thing a caller can report, ask about, or ignore — and
    all three are better than a neighbor's answer.
    """
    return lookup.get(normalize(name))


def any_match(names: Iterable[object], lookup: Mapping[str, str]) -> bool:
    return any(match(n, lookup) is not None for n in names)
