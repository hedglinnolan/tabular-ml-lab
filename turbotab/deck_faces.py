"""L54-B3 — the deck's three faces, two of them specified and not built.

`GUIDED-178`. The product owner's specification, given in conversation on
2026-08-07: **one component spanning three steps** — a per-model card that shows
the pipeline at Preprocess, becomes the model's own shape at Train, and reorders
into the comparison at Compare. His standing thesis is that *the steps are not
the product, the connective tissue between them is*, and this is that thesis as
a single object rather than as an argument.

**Face 1 is built** (`turbotab/web/index.html`, the deck region). Faces 2 and 3
are recorded here the way `turbotab/figure_specs.py` records a `Pending`: the
specification written down, the blocker named, and **nothing asserted that is
not built.** Two, never three, when testing an abstraction.

## The one correction to his framing, and why it is constitutional

He said the place transforms **when you scroll**. `DESIGN_LANGUAGE.md` §05 rule 5 is **"Nothing else moves"** and forbids ambient animation, and a
scroll-driven transform is hover theatrics at page scale — the page reacting to
the pointer rather than to the work.

**So the face changes because the analysis advanced, not because you scrolled.**
Scrolling takes you to where that already happened. That keeps it cause and
effect, which is what §05 is titled, and it is the only change made to his
design. It is his design and he can overrule it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Face:
    """One face of the deck. `built` is the whole of the honesty here."""

    key: str
    step: str
    title: str
    specified_in: str
    built: bool
    needs: str = ""
    blocked_by: str = ""


PREPROCESS = Face(
    key="preprocess",
    step="Preprocess",
    title="What this model needs from your data, and why",
    specified_in="the product owner, 2026-08-07; DESIGN_LANGUAGE.md#05.2",
    built=True,
)

TRAIN = Face(
    key="train",
    step="Train",
    title="What this model is, and what you are choosing",
    specified_in="the product owner, 2026-08-07",
    built=False,
    needs=(
        "A conceptual diagram that CHANGES WITH THE HYPERPARAMETERS rather "
        "than decorating them — `max_depth=3` draws three levels — and the "
        "search space shown BEFORE it is searched, which is this project's "
        "preregistration ethos applied to tuning. Each hyperparameter labeled "
        "with what it trades. None of that exists: the app has no per-model "
        "diagram vocabulary and no representation of a search space, and the "
        "tuning surface itself is unbuilt."),
    blocked_by=(
        "no tuning surface, and no diagram vocabulary. AND THE DOMAIN LENS "
        "MEETS IT HERE: under a survey lens with an ordinal target this face "
        "should carry a cumulative-link card, and it cannot, because the lens "
        "does not reach the model shelf at all — `turbotab/models.py` and "
        "`ml/model_registry.py` contain zero references to it. That is L54-C, "
        "which was not built this loop."),
)

COMPARE = Face(
    key="compare",
    step="Compare",
    title="What that choice bought",
    specified_in="the product owner, 2026-08-07; PRODUCT_VISION.md#the-shelf",
    built=False,
    needs=(
        "**The deck reorders, and the reorder IS the comparison** — judgment "
        "rendered as order rather than as absence, so no card is ever removed. "
        "The reorder sweeps in document order, which is `Propagate`'s shape, "
        "so the ranking is watched rather than discovered. Each card keeps its "
        "identity through the move, which is exactly what FLIP is for."),
    blocked_by=(
        "not the mechanism — L54-B0 built that, and the cards are retained "
        "and moved rather than reprinted. What is missing is a completed "
        "training run to rank BY, and the motion itself. `TEST-066` is the "
        "second obstacle: the harness's DOM appends a copy instead of moving "
        "an attached node, so a reorder written the browser-native way is "
        "unverifiable here."),
)

FACES: Tuple[Face, ...] = (PREPROCESS, TRAIN, COMPARE)


def built() -> Tuple[Face, ...]:
    return tuple(f for f in FACES if f.built)


def pending() -> Tuple[Face, ...]:
    """The faces that are specified and not built.

    Every one carries `needs` and `blocked_by`, because a `Pending` whose
    blocker is unnamed is a wish rather than a specification.
    """
    return tuple(f for f in FACES if not f.built)
