"""The Explore stack's budget — what is pushed, what is collapsed, and the
arithmetic that has to be exact. `GUIDED-149`.

## Two numbers that answer different questions

`PRODUCT_VISION.md` §08 asks *"how many noticings may a step raise at once?"* and
answers **two**, from the prototype. §04 separately describes the Explore section
as *"a ranked stack of findings, each carrying its plot, its plain-language
consequence, and its downstream action."* Those are not the same object and
conflating them is what makes the budget look like it has already been decided.

* An **interruption** arrives at a decision point and takes the user off what
  they were doing. Two is right for that, and it is the Clippy problem.
* A **stack** is the section opening already answered. It interrupts nothing —
  it *is* the content — and capping it at two on `clinical_labs.csv` would
  collapse eleven of thirteen findings, which is the domain content three loops
  made reachable being deleted from view by a number borrowed from a different
  question.

So the stack is governed by ranking, bounding and disclosure. **Never by
truncation.**

## Three rules, all ruled elsewhere

1. **Nothing that gates a decision is ever collapsed.** `ROADMAP.md` Decision B,
   third clause: a question of consequence *"is always pushed, never offered —
   a blocker that only offers is not gating"*, and its refinement, *"blockers
   rank first. A blocker third in a list of nine is a blocker in name only."*
   `NEVER_COLLAPSED` is that clause as a set, and `stack()` puts those findings
   at the head of the pushed list **regardless of the rank the engine gave
   them** — see `_order` for why that is a contract rather than a duplicated
   sort.
2. **The shelf is never shortened.** `PRODUCT_VISION.md`: judgment renders as
   order and prominence, never as absence. Everything served stays reachable;
   the collapsed group is disclosure, not deletion, and it travels in the same
   payload so opening it needs no round trip.
3. **The remainder is counted and typed, never merely hidden.**
   `DESIGN_LANGUAGE.md` §09's recorded-absence rule and `LOOP.md` §10's
   no-silent-caps rule are the same rule at two scales: *"7 more — 3 warnings,
   4 cautions"* is an answer a reader can act on, and a fade-out is a truncation
   wearing an affordance's clothes.

Rule 3 has a second half that is easy to miss and is the recorded-absence rule
proper: **when nothing is collapsed the stack still says so.** A reader cannot
otherwise tell *"this is everything"* from *"this is the top few"*, and those are
two different claims rendering as one — the exact shape §09 built the rule
around, where `group_col: None` meant both *we checked and rows do not repeat*
and *we could not tell*. `complete` and its sentence cost one muted line.

## The bound is five, and the reason is a measurement

Not a preference and not the prototype's two. Across the sixteen tables in
`turbotab/sample_data/`, driven through the API under the lens each companion
file names, the Explore stack runs

    1 · 3 · 3 · 3 · 3 · 4 · 4 · 5 · 5 · 5 · 5 · 6 · 6 · 6 · 8 · 13

— **median five.** A bound at the median collapses something on three of the
sixteen and nothing on the other thirteen, so it fires on the tail rather than on
the typical table. That is the property being bought, and `PRODUCT_VISION.md`
already states it about a different surface: *"a resolution statement that fires
on every dataset is wallpaper."* A collapse that fires on every dataset is the
same thing — it stops being judgment and becomes a cap.

**Stated as a parameter rather than shipped as a constant**, because the
population it was measured on is this repository's fixtures, which are synthetic.
`DESIGN_LANGUAGE.md` §05 is explicit about what that is worth: *treat a lesson
learned on synthetic data as a hypothesis until a real dataset has seen it.* The
number is therefore an argument, `prototypes/explore-stack.html` is where it gets
looked at, and every caller can pass a different one.

**What deliberately has no second parameter.** A remainder of exactly one is
collapsed like any other, and the affordance then hides one card and costs one
row — which is mildly silly and is not wrong. The alternative is a rule that says
*collapse unless the overflow is one*, and §05 has already paid for that shape:
the scroll rule was revised to a size-dependent condition, was correct at one
dataset size and wrong at the next, and *"prefer a rule with no free parameter
over a rule with a tuned one"* is the lesson written down. So the one-card
remainder is shown to the product owner in the prototype rather than special-cased
here, because that is a ruling somebody makes after looking.

## What this module does not do

It does not sort. `engine.rank_findings` is the one function that produces the
ranked list the app presents, and a second ordering rule here would be the two
copies this project keeps paying for. `stack()` reads `rank` and re-asserts
exactly one clause of the constitution on top of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: The two finding streams the Explore stack presents. `structure` is not here:
#: it renders in its own card at the Data step, where the repair groups filter
#: it, and mixing the two lists would put a question the user already answered
#: back in front of them.
EXPLORE_SOURCES = ("profile", "pack")

#: Severities that gate a decision and are therefore never collapsed and always
#: first. `critical` is the engine's word and `blocker` is `ml.router`'s; both
#: are here because `ml/router.py:77` ranks them identically and a set that knew
#: only one of them would be a rule that depends on which module produced the
#: finding.
NEVER_COLLAPSED = frozenset({"critical", "blocker"})

#: How many findings the stack pushes, over and above whatever `NEVER_COLLAPSED`
#: puts there. See the module docstring for the measurement.
BOUND = 5

#: The sentence the number ships with, because a magic constant is a decision
#: nobody can argue with. Quoted by the prototype and by the ledger row.
BOUND_BECAUSE = (
    "Five is the median Explore stack across the sixteen tables in "
    "turbotab/sample_data/ (1·3·3·3·3·4·4·5·5·5·5·6·6·6·8·13). A bound at the "
    "median collapses something on three of them and nothing on thirteen, so it "
    "fires on the tail rather than on the typical table — a collapse that fires "
    "on every dataset has stopped being judgment and become a cap. It is not the "
    "prototype's two: two is the interruption budget, and a stack is not an "
    "interruption."
)

#: What each stream's chip says. The profile speaks about the table and the pack
#: speaks about the field — L42's reason for grouping them, kept as a LABEL now
#: that ranking owns the order (`ROADMAP.md` Decision B: *ordering is part of the
#: gate*). Composed here rather than in the page for the same reason
#: `cohort_findings.render_shape` is: a page that decided for itself what a
#: finding's provenance is called would hold a second copy of the rule.
_SOURCE_LABEL = {
    "profile": "this table",
    "structure": "file shape",
}

#: Plural forms for the typed remainder. `info` is uncountable here because
#: "3 infos" is not a phrase, and the alternative — inventing a synonym — would
#: put a word on screen that no other surface uses for the same severity.
_PLURAL = {"critical": "criticals", "blocker": "blockers",
           "warning": "warnings", "caution": "cautions", "info": "info"}


class StackError(AssertionError):
    """The partition did not account for everything it was given.

    Raised rather than returned, because a stack whose arithmetic is off by one
    is the app asserting something false about its own contents — and the
    affordance would state the wrong number in the user's face.
    """


def source_label(finding: Dict[str, Any]) -> str:
    """What this finding's provenance chip says.

    A pack names its lens, because *which* lens is the thing the user answered a
    question to get. Anything else names its stream.
    """
    source = str(finding.get("source") or "")
    if source == "pack":
        pack = str(finding.get("pack") or "").strip()
        return f"{pack} lens" if pack else "a lens"
    return _SOURCE_LABEL.get(source, source or "unattributed")


def _severity(finding: Dict[str, Any]) -> str:
    return str(finding.get("severity") or "")


def gates_a_decision(finding: Dict[str, Any]) -> bool:
    """Whether this finding is one Decision B forbids collapsing."""
    return _severity(finding) in NEVER_COLLAPSED


def _rank(finding: Dict[str, Any], fallback: int) -> int:
    """The engine's rank, or this finding's arrival position.

    A finding with no rank is ranked by where it arrived rather than dropped or
    pushed to the end: the caller handed us a list it believes is ordered, and
    inventing a position for it would be this module sorting, which the docstring
    says it does not do.
    """
    value = finding.get("rank")
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def _order(findings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank order, with the one clause the constitution makes absolute on top.

    **This is not a second sort and the distinction matters.** The order inside
    each group is the engine's, untouched. What is re-asserted is only
    *blockers rank first* — because that clause is a property of the surface's
    contract rather than of any one ranking function, and today the two
    disagree: `engine.SEVERITY_RANK` has no `blocker` key, so a
    `blocker`-severity finding sorts to 99 and ranks **last** while
    `ml/router.py:77` ranks it 0. Nothing emits that severity into a finding
    today (`GUIDED-151` is the row), so this costs nothing now and cannot be
    wrong later.
    """
    indexed = list(enumerate(findings))
    return [f for _, f in sorted(
        indexed, key=lambda pair: (0 if gates_a_decision(pair[1]) else 1,
                                   _rank(pair[1], pair[0]), pair[0]))]


def explore_findings(findings: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The findings the Explore stack presents, in the order it presents them."""
    return _order([f for f in findings
                   if str(f.get("source") or "") in EXPLORE_SOURCES])


def _tally(findings: Sequence[Dict[str, Any]], key) -> List[Dict[str, Any]]:
    """Counts in first-appearance order, so the tally reads down the stack.

    Order matters here: the remainder is ranked, so its highest severity is its
    first entry, and a tally sorted alphabetically would put `caution` before
    `warning` and bury the answer to *should I look?*
    """
    out: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for finding in findings:
        label = key(finding)
        if label not in seen:
            seen[label] = {"key": label, "n": 0}
            out.append(seen[label])
        seen[label]["n"] += 1
    return out


def _severity_word(key: str, n: int) -> str:
    return key if n == 1 else _PLURAL.get(key, key + "s")


def _source_phrase(label: str) -> str:
    """The provenance label as it reads inside a sentence.

    The chip says `clinical lens` because a chip is a name; the tally says
    *from the clinical lens* because a tally is prose, and *"2 clinical lens"*
    is not a sentence the app is allowed to write.
    """
    if label.endswith(" lens") or label == "a lens":
        return f"from the {label}" if label != "a lens" else "from a lens"
    if label == "this table":
        return "about this table"
    return f"about the {label}"


def stack(findings: Sequence[Dict[str, Any]],
          bound: Optional[int] = None) -> Dict[str, Any]:
    """Partition the Explore findings into what is pushed and what is collapsed.

    Returns ids rather than findings. The page already resolves an id against
    `P.findings` (`findingById`), and a payload carrying every finding twice
    would be two copies of one object on one wire — the shape trap #7 is about,
    one layer down.

    The invariants, asserted here rather than trusted, because both are the kind
    of thing that silently stops being true:

    * `len(pushed) + len(collapsed) == served` — every finding is in exactly one
      place, and the number in the affordance is the number behind it.
    * no finding in `collapsed` gates a decision.
    """
    limit = BOUND if bound is None else int(bound)
    if limit < 0:
        raise StackError(f"a bound of {limit} is not a number of cards")

    ordered = explore_findings(findings)
    pushed: List[Dict[str, Any]] = []
    collapsed: List[Dict[str, Any]] = []
    budget = limit
    for finding in ordered:
        if gates_a_decision(finding):
            pushed.append(finding)
        elif budget > 0:
            budget -= 1
            pushed.append(finding)
        else:
            collapsed.append(finding)

    if len(pushed) + len(collapsed) != len(ordered):
        raise StackError(
            f"{len(pushed)} pushed + {len(collapsed)} collapsed does not "
            f"account for {len(ordered)} served")
    gated = [f.get("id") for f in collapsed if gates_a_decision(f)]
    if gated:
        raise StackError(
            f"{gated} gate a decision and are inside the collapsed group; "
            f"a blocker that only offers is not gating")

    by_severity = _tally(collapsed, _severity)
    by_source = _tally(collapsed, source_label)

    n = len(collapsed)
    if collapsed:
        headline = f"{n} more — " + ", ".join(
            f"{e['n']} {_severity_word(e['key'], e['n'])}" for e in by_severity)
        detail = " · ".join(
            f"{e['n']} {_source_phrase(e['key'])}" for e in by_source)
        # NEVER A BARE VERB (§05.1 rule 2). "Show fewer" names the mechanism;
        # this names what changes and what does not.
        opened = f"Fold {'that one' if n == 1 else f'those {n}'} back"
        title = (f"Adds {n} more finding{'' if n == 1 else 's'} to this list. "
                 f"{'It ranked' if n == 1 else 'They ranked'} below the top "
                 f"{limit}; nothing is out of the record either way.")
    else:
        # THE RECORDED-ABSENCE RULE (`DESIGN_LANGUAGE.md` §09). Without this the
        # reader cannot tell "this is everything" from "this is the top few",
        # which is two claims rendering as one — the shape the rule exists for.
        headline = (f"All {len(ordered)} shown." if ordered
                    else "Nothing stood out in the profile or under the lens.")
        detail = ""
        opened = ""
        title = ""

    return {
        "bound": limit,
        "bound_because": BOUND_BECAUSE,
        "served": len(ordered),
        "pushed": [f.get("id") for f in pushed],
        "collapsed": [f.get("id") for f in collapsed],
        # How many are pushed BECAUSE they gate a decision rather than because
        # they fit — so a reader of the payload can tell a stack of eight at
        # bound five from a bound somebody quietly raised.
        "outside_bound": sum(1 for f in pushed if gates_a_decision(f)),
        "complete": not collapsed,
        "remainder": {
            "n": len(collapsed),
            "by_severity": [dict(e, label=_severity_word(e["key"], e["n"]))
                            for e in by_severity],
            "by_source": [dict(e, label=_source_phrase(e["key"]))
                          for e in by_source],
        },
        # The sentence the affordance QUOTES. §05.1 rule 3: the page reads back
        # what the record holds rather than composing its own, so an interface
        # that disagreed with the server would show a visible disagreement
        # instead of a reassuring number it made up.
        "affordance": headline,
        "affordance_detail": detail,
        "affordance_open": opened,
        "affordance_title": title,
    }
