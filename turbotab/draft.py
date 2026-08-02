"""Read-as-draft: the decisions so far, as methods-section prose.

`DESIGN_LANGUAGE.md` §10 names this a *faded worked example* and calls it
pedagogy rather than decoration: the draft accumulating in the right panel shows
the user what their decisions look like in publishable prose **while they still
have time to change them**. It teaches on their own data, which is layer 3 of
the education model, and it costs no navigation.

Two rules it must not break.

**The app never speaks in the user's name** (§06.5). Every interpretive claim —
why the cohort was restricted, what a limitation means for the conclusion — is
left as a visible `[AUTHOR REQUIRED]` gap. The draft states what was done, never
what it meant.

**No internal placeholder ever renders.** The coach ledger used to print the
literal string "not built yet" beside a deferred item (GUIDED-007). A section
with nothing in it yet says what it is waiting for, in the app's own voice, or
it does not render at all.

This module holds no state: it is a fold over the project's decisions, which is
the same fold the frontend does for dispositions. Two readers of one record
cannot drift.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

AUTHOR_GAP = "[AUTHOR REQUIRED]"

# Steps in the order a methods section reads them. A decision lands in the
# section its step owns; anything unrouted lands in Data preparation rather than
# disappearing.
SECTIONS: Sequence[Dict[str, str]] = (
    {"key": "data", "title": "Data preparation",
     "waiting": "Nothing has been decided about the table yet."},
    {"key": "target", "title": "Outcome and analysis population",
     "waiting": "The outcome has not been chosen yet."},
    {"key": "explore", "title": "Exploratory analysis",
     "waiting": "No exploratory decision has been recorded yet."},
    # THE PREPROCESSING PLAN, AND IT IS ONLY SAFE TO EXPORT NOW.
    #
    # `GUIDED-089`'s note recorded that this module had no reference to
    # missingness or preprocess, so the recorded methods sentence was never
    # exported — and until `L35-B` that was FORTUNATE, because the sentence
    # disagreed with the fit: the record said a blank was left and the pipeline
    # filled it with the median. `pipeline_plan.py` composes the fit from the
    # record now, and the sentence and the pipeline are one object, so the same
    # string is true of both. That is why this section exists in this loop and
    # could not have existed in the last one.
    {"key": "preprocess", "title": "Missing data and preprocessing",
     "waiting": ("No missingness or preprocessing decision has been recorded "
                 "yet.")},
    {"key": "features", "title": "Feature handling",
     "waiting": "No feature or selection decision has been recorded yet."},
    {"key": "limitations", "title": "Limitations",
     "waiting": "Nothing has been accepted over an objection yet."},
)

_KIND_SECTION = {
    "note": "data",
    "apply": "data",
    "revert": "data",
    "set_target": "target",
    "set_task_type": "target",
    "set_grain": "target",
    "set_eligibility": "target",
    "set_repeat_kind": "target",
    "set_unit_of_analysis": "target",
    "set_aggregation": "data",
    "seal_lockbox": "target",
    "dismiss": "explore",
    "defer": "explore",
    "flag": "explore",
    "unflag": "explore",
    "acknowledge_blocker": "limitations",
    "resolve_blocker": "data",
    "select_models": "target",
    "set_model_recipe": "target",
    # `L36-C`. Each of these carries a sentence composed once by the record and
    # now fitted by `pipeline_plan`, so the manuscript quotes the same string
    # the transcript shows and the pipeline performs.
    "route_missingness": "preprocess",
    "route_missingness_bulk": "preprocess",
    "settle_preprocess": "preprocess",
    "add_feature": "features",
    "remove_feature": "features",
    "defer_feature": "features",
    "set_selection": "features",
    "settle_features": "features",
    "trim_training_rows": "limitations",
    # The comparison caveat is a LIMITATION, not a method note. Per-model
    # preparation is the right default and it makes a between-model difference
    # ambiguous between the model and its pipeline — which is precisely the
    # kind of thing a reader needs stated rather than inferred, so it lands
    # where a reader looks for what the study cannot conclude.
    "set_preparation_mode": "limitations",
}

# Decisions that are bookkeeping rather than method. A draft that reports every
# click is a log, not a manuscript.
_NOT_METHOD = frozenset({"flag", "unflag", "undismiss"})


def _sentence_for(d: Dict[str, Any]) -> Optional[str]:
    """One decision as one methods sentence, or None if it is not method.

    The sentence quotes the record. Where the record cannot supply the *reason*
    — and it never can, because a reason is an interpretive claim — the gap is
    left visible rather than filled with a plausible one.
    """
    kind = d.get("kind")
    text = (d.get("text") or "").strip()
    subject = d.get("subject") or ""
    payload = d.get("payload") or {}

    if kind in _NOT_METHOD:
        return None

    if kind == "note":
        return text or None

    if kind == "set_target":
        return text or None

    if kind == "set_task_type":
        return text or None

    if kind == "set_preparation_mode":
        # The CAVEAT is the deliverable, not the answer. Under `uniform` there
        # is nothing to caveat, so the sentence is the plain statement; under
        # `per_model` the caveat travels automatically, because a comparison
        # whose asymmetry is disclosed only on screen is a comparison the
        # reader of the paper cannot interpret.
        caveat = (payload.get("caveat") or "").strip()
        return (f"{text} {caveat}".strip() if caveat else (text or None))

    if kind == "apply":
        return text or None

    if kind == "revert":
        return text or None

    if kind == "dismiss":
        return (f"The finding '{subject}' was reviewed and not acted on. "
                f"{AUTHOR_GAP} — state why it does not affect the analysis.")

    if kind == "defer":
        step = payload.get("target_step") or "a later step"
        return (f"'{subject}' was carried forward to the {step} step for a "
                f"decision there.")

    if kind == "acknowledge_blocker":
        return (f"{text} {AUTHOR_GAP} — state the effect of this on the "
                f"interpretation of the results.")

    if kind == "route_missingness":
        # THE RECORD'S OWN SENTENCE, quoted rather than rewritten. It is the
        # same string `pipeline_plan` carries on the fitted step — asserted as
        # IDENTITY there — so the transcript, the fit and the manuscript are
        # one claim rather than three that agree today (`GUIDED-089`).
        #
        # The STABILITY ASSUMPTION travels with it where §07 recorded one,
        # because it is a methods assumption rather than a warning: it may not
        # hold across sites and a reader has to be able to see it.
        assumption = (payload.get("assumption") or "").strip()
        line = text or None
        if line and assumption:
            line = f"{line} {assumption}"
        if line and payload.get("acknowledged_signal_loss"):
            line += (f" This was accepted over the app's objection and is "
                     f"recorded as a stated limitation.")
        return line

    if kind == "settle_preprocess":
        # The receipt's headline, plus the count of columns nobody answered
        # for. A methods section that reported only the answered ones would be
        # describing a plan more complete than the one that ran.
        outstanding = (payload.get("outstanding") or "").strip()
        return f"{text} {outstanding}".strip() if outstanding else (text or None)

    if kind == "seal_lockbox":
        # `GUIDED-102`. The seal's own sentence, plus what a holdout this size
        # can resolve. `PRODUCT_VISION.md` §04: *it is a recorded decision, and
        # it belongs in the manuscript* — a reader who is told 11 rows were
        # held out and not told what 11 rows can distinguish has been given the
        # number without the thing the number means.
        #
        # It travels WHETHER OR NOT the card was pushed, and that asymmetry is
        # deliberate: `push` decides what interrupts a user mid-journey, and a
        # methods section interrupts nobody. Suppressing the line on the
        # comfortable studies would make its presence a verdict, which is the
        # reading `turbotab/resolution.py` exists to avoid.
        res = payload.get("resolution") or {}
        line = res.get("sentence")
        if not line:
            return text or None
        # The record's sentence first — it states the BASIS, which is §03's
        # requirement and is not derivable from the arithmetic.
        return f"{text} {line}".strip() if text else line

    if kind == "set_selection":
        # `L36-A` made this a fitted decision rather than a recorded one, so
        # the manuscript may quote it. The scope the RECORD carries is what is
        # quoted; where the run could not honor it the run says so, per model,
        # and that divergence belongs to the results rather than the methods.
        return text or None

    return text or None


def draft(project_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The draft, as sections of prose plus the gaps still open.

    Takes the project's own `to_dict()` output so this stays a pure fold and can
    be tested without a server.
    """
    decisions: Iterable[Dict[str, Any]] = project_dict.get("decisions") or []
    buckets: Dict[str, List[Dict[str, Any]]] = {s["key"]: [] for s in SECTIONS}

    for d in decisions:
        sentence = _sentence_for(d)
        if not sentence:
            continue
        key = _KIND_SECTION.get(d.get("kind"), "data")
        buckets[key].append({
            "text": sentence,
            "kind": d.get("kind"),
            "subject": d.get("subject") or "",
            "at": d.get("at"),
            "has_gap": AUTHOR_GAP in sentence,
        })

    # The outcome, stated once, in the register a methods section uses.
    target = project_dict.get("target")
    if target:
        task = project_dict.get("task_type") or "prediction"
        confidence = project_dict.get("task_confidence")
        line = (f"The outcome was `{target}`, modeled as a {task} problem")
        line += (f" (read from the column at {confidence} confidence)."
                 if confidence else ".")
        buckets["target"].insert(0, {"text": line, "kind": "derived",
                                     "subject": target, "at": None,
                                     "has_gap": False})
        buckets["target"].append({
            "text": (f"{AUTHOR_GAP} — state the clinical or scientific question "
                     f"`{target}` answers, and why this population."),
            "kind": "derived", "subject": target, "at": None, "has_gap": True,
        })

    sections = []
    for spec in SECTIONS:
        items = buckets[spec["key"]]
        sections.append({
            "key": spec["key"],
            "title": spec["title"],
            "sentences": items,
            # What an empty section says. Never an internal placeholder: the
            # section states what it is waiting for, in the app's own voice.
            "waiting_for": None if items else spec["waiting"],
        })

    n_gaps = sum(1 for s in sections for i in s["sentences"] if i["has_gap"])
    n_sentences = sum(len(s["sentences"]) for s in sections)

    return {
        "sections": sections,
        "n_sentences": n_sentences,
        "n_gaps": n_gaps,
        "gap_marker": AUTHOR_GAP,
        # Said on the panel, not in a tooltip: a draft that does not announce
        # itself as a draft is a draft pretending to be a manuscript.
        "standfirst": (
            "This is what your decisions say so far, written the way a methods "
            "section says it. It updates as you decide. Every "
            f"{AUTHOR_GAP} is a claim only you can make — the app does not "
            "write in your name."
        ),
        "is_empty": n_sentences == 0,
    }
