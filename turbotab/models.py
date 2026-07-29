"""turbotab.models — the shelf, ordered and never shortened.

`PRODUCT_VISION.md`, *"the shelf is never shortened"*:

> The shape of the data changes how models are **ranked**. It never changes
> which models are **available.**

Silently withholding a model is the app deciding in the user's name, which
`DESIGN_LANGUAGE.md` §06 forbids. Offering it at the bottom with the reason
stated is the app doing its job. **Ordering and prominence carry the judgment**
— which is exactly what Classic's Train page failed at: `ml.model_coach` was
already bucketing into good / ok / poor with an evidence-bearing clause per
model, and the layout rendered taxonomy instead. The bucketing was never the
missing piece. The rendering was.

So this module adds almost no intelligence. It calls `model_coach.model_viability`
— which already returns `{model_key: (verdict, clause)}` where the clause cites
*this dataset's* numbers — and turns it into an order, with every model still on
the shelf and every concern stated in full.

**Where this sits on the three-rung ladder.** Model choice is the bottom rung —
*rank and state the concern* — and the ladder's own test is why: the top rung is
not about severity but about whether a competent researcher could have a reason.
There are many analyses in which a tree ensemble at p ≫ n is the right call
anyway. So nothing here refuses, and nothing here blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# The three buckets, in the words the coach already uses on screen. The mapping
# from the engine's verdict is one place rather than one per renderer.
RECOMMENDED = "recommended"
WORTH_TRYING = "worth_trying"
NOT_RECOMMENDED = "not_recommended"

_BUCKET_FROM_VERDICT = {"good": RECOMMENDED, "ok": WORTH_TRYING,
                        "poor": NOT_RECOMMENDED}

BUCKET_LABEL = {
    RECOMMENDED: "Recommended",
    WORTH_TRYING: "Worth trying",
    NOT_RECOMMENDED: "Not recommended for this data",
}

# Rendered above the third group so the ordering cannot be misread as a
# shortlist. The list is not filtered and the app says so, because a user who
# believes a model is unavailable will not think to look for it.
SHELF_DISCLOSURE = (
    "Every model is available. This order is about your data, not about which "
    "models are any good — a model low on this list is one whose concern "
    "applies to a table this shape, and you may have a reason it does not "
    "apply to yours. Select whatever you intend to train.")

_BUCKET_ORDER = {RECOMMENDED: 0, WORTH_TRYING: 1, NOT_RECOMMENDED: 2}


class ModelSelectionError(Exception):
    """The shelf was asked for something it cannot honestly provide."""


@dataclass(frozen=True)
class ShelfEntry:
    """One model, with its verdict and the concern in full."""
    key: str
    name: str
    group: str
    bucket: str
    # The engine's own clause, citing this dataset's numbers. Never summarized
    # and never truncated: a concern shortened to fit a badge is the failure
    # this module exists to correct.
    concern: str
    requires_scaled_numeric: bool
    recommended_for_high_dim: bool
    interpretability: str

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "group": self.group,
                "bucket": self.bucket, "bucket_label": BUCKET_LABEL[self.bucket],
                "concern": self.concern,
                "requires_scaled_numeric": self.requires_scaled_numeric,
                "recommended_for_high_dim": self.recommended_for_high_dim,
                "interpretability": self.interpretability}


def shelf(profile: Any, task_type: str, probe: Any = None) -> List[ShelfEntry]:
    """Every model the registry offers, ordered by fit and never filtered.

    `profile` is a `ml.dataset_profile` profile object — the same one the coach
    reads — so the clauses quote real numbers rather than adjectives.
    """
    from ml.model_registry import get_registry
    from ml import model_coach

    registry = get_registry()
    try:
        verdicts = model_coach.model_viability(profile, probe)
    except Exception:
        # A profile the coach cannot read must not empty the shelf. Silence
        # about ORDER is recoverable; an empty list is the app withholding
        # every model, which is the thing this module forbids.
        verdicts = {}

    supports = ("supports_classification" if task_type == "classification"
                else "supports_regression")
    out: List[ShelfEntry] = []
    for key, spec in registry.items():
        caps = spec.capabilities
        if not getattr(caps, supports, False):
            # NOT a judgment and NOT a shortening: a classifier cannot fit a
            # continuous outcome at all, so offering it would be offering
            # something that raises rather than something that fits poorly.
            # The ladder's top rung — no legitimate use exists.
            continue
        verdict, clause = verdicts.get(key, ("ok", ""))
        out.append(ShelfEntry(
            key=key, name=spec.name, group=spec.group,
            bucket=_BUCKET_FROM_VERDICT.get(verdict, WORTH_TRYING),
            concern=clause or "No specific concern for a table this shape.",
            requires_scaled_numeric=bool(caps.requires_scaled_numeric),
            recommended_for_high_dim=bool(caps.recommended_for_high_dim),
            interpretability=str(caps.interpretability_tier)))

    out.sort(key=lambda e: (_BUCKET_ORDER[e.bucket], e.group, e.name))
    return out


def grouped(entries: Sequence[ShelfEntry]) -> List[Dict[str, Any]]:
    """The shelf as three labeled groups, in order, none of them omitted.

    An empty group is RETURNED EMPTY rather than dropped: "no model is
    recommended for this data" is a real and informative state, and a renderer
    that only sees two groups cannot say it.
    """
    return [{"bucket": b, "label": BUCKET_LABEL[b],
             "models": [e.to_dict() for e in entries if e.bucket == b]}
            for b in (RECOMMENDED, WORTH_TRYING, NOT_RECOMMENDED)]


def validate_selection(entries: Sequence[ShelfEntry],
                       chosen: Sequence[str]) -> List[str]:
    """Accept any model on the shelf, in the order the shelf presents them.

    The only refusals are structural — an unknown key, or an empty selection —
    and neither is a judgment about fit. Selecting three `not_recommended`
    models is a legitimate thing to do and this returns them.
    """
    known = {e.key: e for e in entries}
    unknown = [k for k in chosen if k not in known]
    if unknown:
        raise ModelSelectionError(
            f"{unknown} are not models this task can use. Available: "
            f"{', '.join(sorted(known))}.")
    if not chosen:
        raise ModelSelectionError(
            "Choose at least one model. Preprocessing is configured per model, "
            "so there is nothing to configure until you say what you intend to "
            "train.")
    order = [e.key for e in entries]
    return sorted(set(chosen), key=order.index)


def selection_note(entries: Sequence[ShelfEntry],
                   chosen: Sequence[str]) -> Optional[str]:
    """What the record says when a low-ranked model is chosen.

    Not a warning and not a confirmation step — the choice is already made and
    already legitimate. This is the sentence the methods section carries, so
    that the concern travels with the result rather than being an on-screen
    caveat the reader never sees.
    """
    by_key = {e.key: e for e in entries}
    poor = [by_key[k] for k in chosen
            if k in by_key and by_key[k].bucket == NOT_RECOMMENDED]
    if not poor:
        return None
    lines = "; ".join(f"{e.name} — {e.concern}" for e in poor)
    return (f"{len(poor)} of the selected model(s) carry a stated concern for a "
            f"table this shape: {lines}. Selected deliberately; the concern is "
            f"recorded so it can be reported rather than discovered.")
