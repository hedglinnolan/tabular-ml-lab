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
`turbotab/sample_data/`, driven through the API under the lens each fixture's
companion names — and with no lens where there is no companion, which is a state
a real project reaches, since the lens question is answerable with *none of
these* — the Explore stack runs

    1 · 3 · 3 · 3 · 3 · 4 · 4 · 5 · 5 · 5 · 5 · 6 · 6 · 6 · 8 · 13

— **median five.** `docs/turbotab/prototypes/capture_explore_stack.py` re-derives
that list every time it runs, and the prototype prints it, so the number behind
this constant is checkable rather than quoted. A bound at the median collapses
something on **one** of the sixteen and nothing on the other fifteen, so it fires
on the tail rather than on the typical table. That is the property being bought,
and `PRODUCT_VISION.md` already states it about a different surface: *"a
resolution statement that fires on every dataset is wallpaper."* A collapse that
fires on every dataset is the same thing — it stops being judgment and becomes a
cap.

**One, and L45 measured three. `MIN_COLLAPSE` moved it.** Two of the three tables
that used to collapse were collapsing exactly one card, and an overflow of one is
now shown — so A2 changed the number A1 was justified by, in the direction that
strengthens A1's argument and away from where the measurement says much. At one
in sixteen the median is doing very little work, and if this fixture population
is ever taken seriously as a population, that is the thing to revisit. Recorded
rather than quietly restated, because the number was load-bearing.

**Stated as a parameter rather than shipped as a constant**, because the
population it was measured on is this repository's fixtures, which are synthetic.
`DESIGN_LANGUAGE.md` §05 is explicit about what that is worth: *treat a lesson
learned on synthetic data as a hypothesis until a real dataset has seen it.* The
number is therefore an argument, `prototypes/explore-stack.html` is where it gets
looked at, and every caller can pass a different one.

**The one-card remainder is not collapsed, and the exception has to state its
reason here because the next reader meets §05's lesson before they meet this.**
`MIN_COLLAPSE` is 2: an overflow of one is pushed.

L45 shipped without it and argued the other way, from §05 — *"prefer a rule with
no free parameter over a rule with a tuned one"*, the lesson the scroll rule
bought. **The product owner overruled it and the distinction is the whole of
why.** The scroll rule's parameter was tuned to **dataset size**: it was correct
when a section held two or three cards and wrong on a table with nine findings,
and nothing in the interface could tell which case it was in. **An overflow of one
is not tuned to anything.** It is the point where the affordance costs exactly one
row to hide exactly one row, so the control returns nothing for the space it
occupies — a derivation from what the affordance *is*, and it does not move when
the data does. An exception with no stated reason is how a recorded lesson gets
quietly reversed, so this paragraph is the reason and it sits where the constant
does.

**And it is one of three claims this loop sent to the literature.**
`research/INTERACTION_PACK.md` §03 is the check; if it comes back saying a
one-card affordance is fine, this is cheap to reverse.

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

#: The smallest remainder worth collapsing. **Not a tuned parameter, and the
#: module docstring says why at length**: below this the affordance costs the row
#: it saves, so the control returns nothing for the space it occupies. That is a
#: property of the affordance rather than of any dataset, which is what separates
#: it from the size-dependent scroll condition §05 retired.
MIN_COLLAPSE = 2

#: The sentence the number ships with, because a magic constant is a decision
#: nobody can argue with. Quoted by the prototype and by the ledger row.
BOUND_BECAUSE = (
    "Five, and the argument is that it FIRES ON THE TAIL rather than on the "
    "typical table - a collapse that fires on every dataset has stopped being "
    "judgment and become a cap. Across the sixteen tables in "
    "turbotab/sample_data/ the Explore stack runs "
    "1-3-3-3-3-4-4-5-5-5-5-6-6-6-8-13 and a bound of five collapses something on "
    "ONE of them. On the real NHANES export the product owner drove - 21,849 "
    "rows, 29 columns, nine pooled cycles - it collapses NOTHING: 14 findings, 5 "
    "pushed, 0 collapsed. "
    "IT IS NO LONGER JUSTIFIED BY THE MEDIAN, and that is a ruling rather than a "
    "restatement. The median was the L45 argument; L46's MIN_COLLAPSE moved the "
    "count it rested on from three of sixteen to one, and at one in sixteen a "
    "median is not doing work. What survives is the tail argument, which "
    "one-in-sixteen strengthens. It is not the prototype's two either: two is the "
    "interruption budget, and a stack is not an interruption. "
    "AND THE FIXTURE POPULATION HAS BEEN SUPERSEDED AS EVIDENCE. "
    "DESIGN_LANGUAGE.md section 05 says a lesson learned on synthetic data is a "
    "hypothesis until a real dataset has seen it. One has now seen it. The "
    "sixteen stay re-derivable by "
    "docs/turbotab/prototypes/capture_explore_stack.py rather than quoted, and "
    "test_the_prototype_cannot_disagree_with_the_build is what stops the "
    "prototype from showing an older version of this sentence."
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


def spent_ids(decisions: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Findings the user has cleared, mapped to how — and which cost no budget.

    `GUIDED-154`, ruled by the product owner at L46. A dismissed card collapses
    to a one-line *"Dismissed: X. Still in the record, out of your way."* note
    and a deferred one goes grey with its destination named; neither is asking
    for attention any more, so neither should be holding a slot that keeps a live
    finding behind an affordance. **The stack keeps its full budget of live
    findings.**

    Last decision wins for dismiss/undismiss, and a deferral has no inverse — the
    same reading the page's `statusOf` does. **They are two readers of one record
    and that is deliberate rather than sloppy**, because they answer different
    questions: `statusOf` asks *how does this card look* and needs `flagged` too;
    this asks *does this finding still cost budget*, which is a partition rule
    and belongs on the server with the rest of them.
    `test_the_promoted_card_says_why_it_is_there` asserts the two agree, which is
    the check that keeps the pair honest.
    """
    state: Dict[str, Optional[str]] = {}
    for decision in decisions or ():
        subject = str((decision or {}).get("subject") or "")
        kind = str((decision or {}).get("kind") or "")
        if not subject:
            continue
        if kind in ("dismiss", "defer"):
            state[subject] = kind
        elif kind == "undismiss":
            state[subject] = None
    return {k: v for k, v in state.items() if v}


def deferred_noticings(findings: Sequence[Dict[str, Any]],
                       decisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the user set aside, grouped by the step it comes back at.
    `GUIDED-153`.

    **This exists because a deferred pack finding had nowhere to come back to.**
    `ml.router.plan` re-presents a deferred finding as a repair question, and
    `_is_repairable` admits only findings that carry a `fix_kind` — a pack
    finding carries `none`, by design, because §A1.1's rule is *detect, propose,
    require explicit confirmation* and none of the eighteen detectors proposes a
    repair. So the deferral was recorded, the dock listed it, and the step it
    named never showed it. `PRODUCT_VISION.md` §04: *deferred items resurface,
    pre-checked and attributed, at the step they target. That closes the loop
    between what the app noticed and what the user decided, which is the whole
    point.* The loop stayed open.

    **A noticing is not a question and this does not make it one.** It comes back
    as the card it already was, attributed to where it was set aside — which is
    `test_a_finding_with_no_repair_still_offers_something`'s precedent, one
    disposition over: something is always offerable, and for a finding with no
    repair that something is the finding itself, in front of the step that can
    act on it.

    Keyed by step so the page asks one question — *what comes back here* — and
    holds no rule about which findings those are.
    """
    cleared = spent_ids(decisions)
    by_step: Dict[str, List[Dict[str, Any]]] = {}
    for finding in findings:
        fid = str(finding.get("id") or "")
        if cleared.get(fid) != "defer":
            continue
        step = str(finding.get("defer_target") or "")
        label = str(finding.get("defer_target_label") or "")
        if not step:
            # A deferral with no destination is the defect this closes, so it is
            # reported rather than filed under a step nobody chose.
            by_step.setdefault("__unrouted__", []).append(
                {"id": fid, "title": finding.get("title") or "", "from": "explore",
                 "why": "This was set aside and nothing recorded where it comes "
                        "back, so it cannot be shown at a step. That is a defect "
                        "rather than a state."})
            continue
        by_step.setdefault(step, []).append({
            "id": fid,
            "title": finding.get("title") or "",
            "label": label,
            # WHERE IT CAME FROM, because "attributed" is half of §04's promise
            # and a card arriving with no history is the promotion problem in a
            # different costume.
            "why": f"You set this aside at Explore. {label} is the step that can "
                   f"act on it.",
        })
    return by_step


def stack(findings: Sequence[Dict[str, Any]],
          bound: Optional[int] = None,
          spent: Sequence[str] = ()) -> Dict[str, Any]:
    """Partition the Explore findings into what is pushed and what is collapsed.

    Returns ids rather than findings. The page already resolves an id against
    `P.findings` (`findingById`), and a payload carrying every finding twice
    would be two copies of one object on one wire — the shape trap #7 is about,
    one layer down.

    `spent` is the ids the user has dismissed or deferred. **They stay exactly
    where they are and stop consuming budget**, which is the whole of
    `GUIDED-154`'s ruling. Two things it deliberately is not:

    * **Not a promotion of the spent card.** A finding cleared while the group
      was open does not climb into the pushed list — spentness frees a slot, it
      does not grant one. Otherwise dismissing a collapsed card would pull it up,
      which is the opposite of what dismissing means.
    * **Not a shortening.** A spent finding is still `pushed` and still rendered;
      the shelf is never shortened, and `.gone`'s undo note is the record saying
      so out loud.

    The invariants, asserted here rather than trusted, because all three are the
    kind of thing that silently stops being true:

    * `len(pushed) + len(collapsed) == served` — every finding is in exactly one
      place, and the number in the affordance is the number behind it.
    * `live + spent_pushed + collapsed == served` — the same ledger with the
      pushed side split, which is the form the arithmetic takes once a dismissal
      can move the line.
    * no finding in `collapsed` gates a decision.
    """
    limit = BOUND if bound is None else int(bound)
    if limit < 0:
        raise StackError(f"a bound of {limit} is not a number of cards")
    # A mapping id → how it was cleared, or a bare sequence of ids when the
    # caller does not know. The kinds are only ever used to make the promoted
    # card's sentence say the true verb.
    kinds: Dict[str, str] = (dict(spent) if isinstance(spent, dict)
                             else {str(i): "" for i in (spent or ())})
    cleared = set(kinds)

    ordered = explore_findings(findings)

    def partition(free_spent: bool):
        pushed_, collapsed_, budget = [], [], limit
        for finding in ordered:
            if gates_a_decision(finding):
                pushed_.append(finding)
            elif budget > 0:
                pushed_.append(finding)
                if not (free_spent and str(finding.get("id")) in cleared):
                    budget -= 1
            else:
                collapsed_.append(finding)
        # THE ONE-CARD REMAINDER IS NOT COLLAPSED (`MIN_COLLAPSE`, and the module
        # docstring carries the reason at length). Applied inside the partition
        # so both the real one and the counterfactual below obey it — otherwise
        # `promoted` would be computed against a stack that does not exist.
        if 0 < len(collapsed_) < MIN_COLLAPSE:
            pushed_ = pushed_ + collapsed_
            collapsed_ = []
        return pushed_, collapsed_

    pushed, collapsed = partition(True)
    # WHICH CARDS AROSE BECAUSE A SLOT OPENED, derived by asking what the
    # partition would have been if nothing had been cleared. Derived rather than
    # tracked through the loop because the difference IS the definition: a
    # promoted card is one that would otherwise be behind the affordance.
    if cleared:
        would_be, _ = partition(False)
        was = {str(f.get("id")) for f in would_be}
        promoted = [str(f.get("id")) for f in pushed if str(f.get("id")) not in was]
    else:
        promoted = []
    # THE VERB THE MARKER USES, taken from what actually happened rather than
    # generalized. `dismiss` and `defer` are different decisions — one lets a
    # finding go, the other says *bring this back at Preprocess* — and a marker
    # that called both "cleared" when only one occurred would be the interface
    # rounding a recorded decision off.
    freed = {kinds.get(str(f.get("id")), "")
             for f in pushed if str(f.get("id")) in cleared}
    verb = ("dismissed" if freed == {"dismiss"}
            else "deferred" if freed == {"defer"} else "cleared")
    promoted_because = f"Moved up when you {verb} a card above."

    if len(pushed) + len(collapsed) != len(ordered):
        raise StackError(
            f"{len(pushed)} pushed + {len(collapsed)} collapsed does not "
            f"account for {len(ordered)} served")
    spent_pushed = [f for f in pushed if str(f.get("id")) in cleared]
    live = [f for f in pushed if str(f.get("id")) not in cleared]
    if len(live) + len(spent_pushed) + len(collapsed) != len(ordered):
        raise StackError(
            f"{len(live)} live + {len(spent_pushed)} cleared + "
            f"{len(collapsed)} collapsed does not account for {len(ordered)}")
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
        # `GUIDED-154`. Which cards are on screen because a slot opened, and the
        # sentence each one carries. **A state, not a motion** — §05.2's list
        # stays closed at four, the app has no mechanism for animating a change
        # of content, and a fifth slot pulls in `GUIDED-073`. §09's
        # recorded-absence rule from the other side: an object appearing without
        # explanation is as unexplained as one vanishing without it.
        "promoted": promoted,
        "promoted_because": promoted_because if promoted else "",
        # The pushed side split, so a reader of the payload can check the ledger
        # after a dismissal without recomputing which ids were cleared.
        "live": [f.get("id") for f in live],
        "cleared": [f.get("id") for f in spent_pushed],
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
