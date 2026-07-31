"""turbotab.figures — the figure spec, as a spine.

`DOMAIN_SCIENCE.md` §02. The app already has seven geometries — scatter, bar,
histogram, box, heatmap, area, violin — and nineteen EDA actions rendering into
them, and **none of them knows what field it is looking at.** The research says
that is the wrong axis to build on:

> Every pack specified its signature figures as a **checklist**, and the
> checklist items are overwhelmingly about **annotation rather than geometry**.

Eight signature figures across four packs, and the publication-grade delta is
almost never the shape. It is the % variance in the axis label, the risk
histogram under the curve, the numbers-at-risk table, the anchors verbatim in
the legend, the log-scale x-axis, the rug that shows the upturn is eleven
people. The differentiator is not that TurboTab can draw a calibration curve —
every library can. It is that TurboTab draws the risk distribution under it,
annotates the six numbers a reviewer wants, refuses to truncate the tail, and
writes the caption naming the test, the correction and the n.

**So this module is a caption-and-annotation engine wrapped around a plotting
library, and not a plotting library with captions bolted on.** It computes no
geometry it can borrow: `ml/calibration.py` and `ml/macro_shape.py` already hold
the mathematics, and a second implementation of either would be the
two-engines failure inside the figure layer.

## The five fields, and which one has no analogue today

``layers`` is the geometry, and it is the least interesting field here.
``annotations`` are the numbers that must appear, each naming where it came
from. ``checklist`` is the publication-grade list, evaluated **pass/fail against
this render** rather than stated as advice — a checklist nobody scores is a
style guide. ``caption`` is generated and names the test, the correction and the
n.

``companions`` is the one with no analogue in the current app, and it is
load-bearing:

> A PLS-DA scores plot's companion is its permutation plot, and **a confirmatory
> figure with a missing companion is not rendered into the results bundle.**

That single rule kills the circular-figure family in §01.6 — the family whose
members all have the shape *a figure that looks like evidence and is not*. It is
not a warning beside the figure and it is not a caption caveat, both of which
the reader can skip. It is admissibility: `admissible()` returns False and the
bundle does not contain it.

## `tier`, and why the enum is two values rather than a severity

EXPLORATORY and CONFIRMATORY are the metabolomics pack's two-tier logic
generalized: Tier 1 never sees the labels and cannot establish a difference;
Tier 2 sees them and can therefore be fooled. **The signature failure of the
field is presenting a Tier-2 figure with Tier-1 credibility**, and the tier is
what makes that checkable rather than a matter of taste. Only CONFIRMATORY
figures have companions to be missing.

## What is deliberately not here

Two figures are implemented, and the count is the point. A third built now
would be built against an abstraction that has survived one shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

EXPLORATORY = "EXPLORATORY"
CONFIRMATORY = "CONFIRMATORY"
TIERS = (EXPLORATORY, CONFIRMATORY)


class FigureError(Exception):
    """A figure the app cannot honestly draw or describe."""


@dataclass(frozen=True)
class Annotation:
    """One number that must appear on the figure, and where it came from.

    `source` is the computation, not a citation — `ml.calibration` rather than a
    paper. The paper is the figure's `evidence`. Two different questions: *where
    did this number come from* and *does the field agree it belongs here*.
    """
    key: str
    label: str
    source: str
    required: bool = True


@dataclass(frozen=True)
class ChecklistItem:
    """One publication-grade requirement, scored against a render.

    `check` takes the rendered figure's payload and returns True when the item
    is satisfied. A checklist whose items are prose is a style guide; the
    callable is what makes it a score.

    `because` is what the reviewer would say. It is shown when the item fails,
    which is the only moment it is worth reading.
    """
    id: str
    text: str
    because: str
    check: Callable[[Dict[str, Any]], bool]


@dataclass(frozen=True)
class FigureSpec:
    """One figure, as the packs specify them.

    `when_applicable` takes the project's state and returns whether this figure
    has anything to say about it. It is the same shape as a pack detector's
    trigger and for the same reason: a figure offered on data it does not
    describe is guard #2 broken one layer out.
    """
    id: str
    title: str
    when_applicable: Callable[[Dict[str, Any]], bool]
    layers: Tuple[str, ...]
    annotations: Tuple[Annotation, ...]
    checklist: Tuple[ChecklistItem, ...]
    caption: Callable[[Dict[str, Any]], str]
    tier: str
    # Figures that must accompany this one for it to be admissible. Empty on an
    # EXPLORATORY figure by construction — see `__post_init__`.
    companions: Tuple[str, ...] = ()
    # Where the field stands on this figure's requirements, and where that was
    # read. Same primitive as a pack prior (`GUIDED-047`).
    evidence: Optional[Any] = None
    # WHETHER THIS FIGURE'S CONTENT MAY BECOME MODEL INPUT (`GUIDED-052`).
    # See `promotable` below — the rule is re-executability, not label-blindness.
    promotable: bool = False
    promotable_because: str = ""
    compute: Optional[Callable[..., Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise FigureError(
                f"{self.id}: tier must be one of {list(TIERS)}. The two-tier "
                f"logic is what makes 'a Tier-2 figure with Tier-1 "
                f"credibility' checkable rather than a matter of taste.")
        if self.tier == EXPLORATORY and self.companions:
            raise FigureError(
                f"{self.id}: an EXPLORATORY figure has no companions. "
                f"Companions exist because a confirmatory claim needs its "
                f"validation beside it; requiring one of a figure that makes "
                f"no claim would be ceremony.")
        if not self.evidence:
            raise FigureError(
                f"{self.id}: a figure states where the field stands on its "
                f"requirements. Its checklist is a set of claims about what a "
                f"reviewer expects, and an unbadged claim is the uniform "
                f"confidence `DOMAIN_SCIENCE.md` §01.1 exists to end.")
        if self.promotable and not self.promotable_because:
            raise FigureError(
                f"{self.id}: promotable is a claim that the app can re-run "
                f"this computation inside every fold, and it names why. "
                f"`True` with no argument is the claim without the evidence.")

    def admissible(self, present: Sequence[str]) -> Tuple[bool, List[str]]:
        """Whether this figure may enter the results bundle, and what is missing.

        **The rule with no analogue in the current app.** A CONFIRMATORY figure
        whose companion is absent is not admitted — not warned about, not
        caption-caveated, not rendered greyed out. Both of those are things a
        reader skips, and the circular-figure family survives precisely by being
        skippable.
        """
        missing = [c for c in self.companions if c not in set(present)]
        return (not missing), missing

    def score(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every checklist item, pass or fail, against THIS render.

        Not against the spec, and not against intent. `GUIDED-045`'s axis
        applied to a checklist: an item that is satisfied by the figure's
        existence rather than by its content has a pass set broader than its
        claim.
        """
        out = []
        for item in self.checklist:
            try:
                ok = bool(item.check(payload))
            except Exception as exc:                       # pragma: no cover
                # Reported, never swallowed: a checklist item that raises is an
                # item nobody scored, and silence would read as a pass.
                ok = False
                out.append({"id": item.id, "text": item.text, "passed": False,
                            "because": f"the check could not run: {exc}"})
                continue
            out.append({"id": item.id, "text": item.text, "passed": ok,
                        "because": "" if ok else item.because})
        return out

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "id": self.id, "title": self.title, "tier": self.tier,
            "layers": list(self.layers),
            "annotations": [{"key": a.key, "label": a.label,
                             "source": a.source, "required": a.required}
                            for a in self.annotations],
            "checklist": [{"id": c.id, "text": c.text} for c in self.checklist],
            "companions": list(self.companions),
            "promotable": self.promotable,
            "promotable_because": self.promotable_because,
        }
        if self.evidence is not None:
            base.update(self.evidence.to_dict())
        return base


# ─────────────────────────────────────────────────────────────────────────────
# The registry
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY: Dict[str, FigureSpec] = {}


def register(spec: FigureSpec) -> FigureSpec:
    if spec.id in REGISTRY:
        raise FigureError(
            f"{spec.id} is already registered. Two specs under one id is the "
            f"shadowing `recipes.register_operation` refuses for the same "
            f"reason: the resolved one is whichever imported last.")
    REGISTRY[spec.id] = spec
    return spec


def applicable(state: Dict[str, Any]) -> List[FigureSpec]:
    """Every figure that has something to say about this project."""
    return [s for s in REGISTRY.values() if _safely(s.when_applicable, state)]


def _safely(fn: Callable[[Dict[str, Any]], bool], state: Dict[str, Any]) -> bool:
    try:
        return bool(fn(state))
    except Exception:                                      # pragma: no cover
        return False


def bundle(rendered: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The results bundle: what is admitted, what is held back, and why.

    `rendered` maps figure id to its computed payload. The return separates the
    two rather than filtering silently — a figure held back for a missing
    companion is a *finding about the analysis*, and dropping it without a word
    would be the app being quiet about the thing the companion rule exists to
    make loud.
    """
    present = list(rendered)
    admitted, held = [], []
    for figure_id, payload in rendered.items():
        spec = REGISTRY.get(figure_id)
        if spec is None:                                   # pragma: no cover
            continue
        ok, missing = spec.admissible(present)
        row = {"id": figure_id, "title": spec.title, "tier": spec.tier,
               "checklist": spec.score(payload),
               "caption": spec.caption(payload),
               "promotable": spec.promotable,
               "promotable_because": spec.promotable_because}
        if ok:
            admitted.append(row)
        else:
            row["missing_companions"] = missing
            row["why_held"] = (
                f"{spec.title} is a {spec.tier.lower()} figure and its "
                f"companion{'' if len(missing) == 1 else 's'} "
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                f"not in this bundle. A confirmatory figure without its "
                f"validation beside it is the shape every circular-figure "
                f"defect takes, so it is held rather than captioned with a "
                f"caveat a reader can skip.")
            held.append(row)
    return {"admitted": admitted, "held": held,
            "n_admitted": len(admitted), "n_held": len(held)}
