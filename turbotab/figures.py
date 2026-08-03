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
    # WHERE IT STANDS ON EACH REQUIREMENT SEPARATELY (`GUIDED-064`). A figure's
    # checklist is a set of claims and the field does not hold them all at one
    # status: the volcano's q-on-the-y-axis rule is SETTLED and the |log2FC|
    # cut beside it is a stated convention; the diverging bar is the field
    # standard and how it treats the neutral midpoint is disputed. One badge
    # over both is a machine-readable form coarser than the caption.
    claims: Tuple[Any, ...] = ()
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
            from turbotab.packs import _badge_payload
            base.update(_badge_payload(self.evidence, self.claims))
        return base


# ─────────────────────────────────────────────────────────────────────────────
# The registry
# ─────────────────────────────────────────────────────────────────────────────

class _SelfPopulating(dict):
    """A registry that cannot be observed empty. `TEST-041`.

    **The defect this removes.** `REGISTRY` was a plain dict filled as an
    *import side effect* of `turbotab.figure_specs`, whose module body is a
    run of `register(...)` calls. So a reader's answer depended on whether
    something else had imported the populator first — and
    `test_the_companion_rule_reaches_the_document` asserted a count over it in
    its first test, which takes no fixture. Alone the file said
    ``assert 0 >= 4``; inside the full suite it was green. **The full-suite
    green was the false one**: a different file's import made the assertion
    true, not the code under test.

    That is the third face of one property. `TEST-030` is ordering in
    `tests/workflow`; `TEST-040` was its load-dependent twin, where a bounded
    poll reported *the app had not answered yet* as *the app answered wrong*.
    **A count that moves with something other than the code is not a count.**

    L43 fixed the four readers and added a static guard that every registry
    reader imports a populator. That is a convention with a check, and the
    check is static where the property is behavioral. This is the structure:
    the first read *is* the population, so no reader can observe an unfilled
    registry and no new reader has to remember anything.

    **Why a `dict` subclass rather than a `registry()` accessor.** Callers
    read it as a mapping — ``.values()``, ``.get(id)``, ``id in REGISTRY``,
    ``len()`` — in three shipped modules and a dozen test files. An accessor
    would rewrite every one of those call sites for no behavioral gain; this
    keeps them and removes the hazard underneath.
    """

    #: Set BEFORE the import rather than after, and the reason is narrower
    #: than it looks.
    #:
    #: `register()` writes into this dict while `figure_specs`'s module body
    #: runs, and it asks `spec.id in REGISTRY` first — which populates. So the
    #: obvious story is that a flag set afterwards would re-enter `_populate`
    #: and recurse. **That story is false and a revert probe caught it**:
    #: moving the assignment after the import comes back
    #: `GREEN — NOT LOAD-BEARING`, because Python's `sys.modules` already
    #: returns the partially-initialized module on re-entry and the nested
    #: `import` is a no-op. Measured on a cold interpreter with the
    #: recursion limit at 200: 17 specs, no error.
    #:
    #: It stays before the import anyway, and the honest reason is that this
    #: class should not depend on the import system's re-entrancy behavior to
    #: terminate. `test_a_read_during_population_does_not_loop` pins the
    #: property directly rather than through that coincidence.
    _populated = False

    def _populate(self) -> None:
        if self._populated:
            return
        self._populated = True
        import turbotab.figure_specs                    # noqa: F401

    # Every read triggers it; writes do not, because a write is `register`
    # doing the populating.
    def __getitem__(self, key):
        self._populate()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._populate()
        return dict.__contains__(self, key)

    def __len__(self):
        self._populate()
        return dict.__len__(self)

    def __iter__(self):
        self._populate()
        return dict.__iter__(self)

    def get(self, key, default=None):
        self._populate()
        return dict.get(self, key, default)

    def keys(self):
        self._populate()
        return dict.keys(self)

    def values(self):
        self._populate()
        return dict.values(self)

    def items(self):
        self._populate()
        return dict.items(self)


REGISTRY: Dict[str, FigureSpec] = _SelfPopulating()


def register(spec: FigureSpec) -> FigureSpec:
    if spec.id in REGISTRY:
        raise FigureError(
            f"{spec.id} is already registered. Two specs under one id is the "
            f"shadowing `recipes.register_operation` refuses for the same "
            f"reason: the resolved one is whichever imported last.")
    REGISTRY[spec.id] = spec
    return spec


@dataclass(frozen=True)
class Pending:
    """A figure that is SPECIFIED and not built, in a form something resolves.

    **`GUIDED-060`.** Two of the four prevalence refusals offered a figure that
    did not exist — `distribution_against_ai` and
    `distribution_against_ear_and_rda` — and the AI case is the flagship, the
    reason the nutrition pack was built first. *Every refusal offers what it can
    draw* is the stated principle, and its reason is that a refusal offering
    nothing is indistinguishable from a missing feature. **An offer naming an
    unbuilt figure is that same failure arriving one layer later**, at the worst
    possible moment, and the test that was supposed to catch it asserted the
    offer strings were non-empty.

    This project already resolves two kinds of reference — a prior's source
    through `evidence.py` and a `FIXED` row's test through `ledger.py check`.
    An offer's draw target is the third, and it is one because a pending figure
    is a first-class record rather than a string nobody follows:

    * `specified_in` is a research citation in `Evidence`'s own form, so the
      same resolver checks it.
    * `needs` is what has to exist first, in one sentence a user can read.
    * `blocked_by` is the ledger row, so the record and the backlog agree.

    A pending entry is NOT a promise of a date. It is the difference between
    *"the app will not draw this"* and *"the app cannot draw this yet, here is
    what is missing"*, and only the second is honest when the second is true.
    """
    id: str
    title: str
    specified_in: str
    needs: str
    blocked_by: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": PENDING_STATUS,
                "specified_in": self.specified_in, "needs": self.needs,
                "blocked_by": self.blocked_by}


REGISTERED_STATUS = "registered"
PENDING_STATUS = "pending"

#: Filled by the same import as `REGISTRY` and read the same way, so it
#: carries the same hazard and the same fix.
PENDING: Dict[str, Pending] = _SelfPopulating()


def register_pending(entry: Pending) -> Pending:
    if entry.id in REGISTRY:
        raise FigureError(
            f"{entry.id} is registered and is not pending. A figure cannot be "
            f"both built and unbuilt, and the offer that names it would "
            f"resolve to two different answers.")
    if entry.id in PENDING:                                # pragma: no cover
        raise FigureError(f"{entry.id} is already pending.")
    PENDING[entry.id] = entry
    return entry


def resolve(figure_id: str) -> Dict[str, Any]:
    """What an offer's `draw` target actually is. Raises where it is neither.

    **The resolution `GUIDED-060` asked for.** An id in neither table is not a
    pending figure — it is a typo or a figure somebody imagined, and a refusal
    offering one is worse than a refusal offering nothing, because it reads as
    a feature.
    """
    spec = REGISTRY.get(figure_id)
    if spec is not None:
        return {"id": figure_id, "title": spec.title,
                "status": REGISTERED_STATUS, "tier": spec.tier}
    entry = PENDING.get(figure_id)
    if entry is not None:
        return entry.to_dict()
    raise FigureError(
        f"{figure_id!r} is neither a registered figure nor a declared pending "
        f"one. An offer naming it would promise the user a picture nobody can "
        f"draw, which is the failure the offer exists to prevent arriving one "
        f"layer later.")


def resolve_offer(offer: Dict[str, Any]) -> Dict[str, Any]:
    """An offer with its draw target resolved, ready to be rendered or refused."""
    resolved = resolve(str((offer or {}).get("draw") or ""))
    out = dict(offer or {})
    out["resolved"] = resolved
    out["pending"] = resolved["status"] == PENDING_STATUS
    return out


def applicable(state: Dict[str, Any]) -> List[FigureSpec]:
    """Every figure that has something to say about this project."""
    return [s for s in REGISTRY.values() if _safely(s.when_applicable, state)]


def _safely(fn: Callable[[Dict[str, Any]], bool], state: Dict[str, Any]) -> bool:
    try:
        return bool(fn(state))
    except Exception:                                      # pragma: no cover
        return False


NOT_ESTIMABLE = "not estimable"

_ABSENT = (
    "The figure does not carry a value for this. A number is not shown because "
    "there is not one, rather than because it failed to render.")


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                         for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{v:.1%}" if isinstance(v, float) else str(v)
                         for v in value)
    return str(value)


def annotation_rows(spec: FigureSpec,
                    payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Every annotation the spec requires, rendered — the ABSENCE included.

    **A missing number renders as `not estimable` with its reason, never as a
    blank cell.** `calibration_render` discovered this and owned it alone: the
    weak-calibration fit is undefined for one outcome class, constant
    predictions or complete separation — which is what a very good model on a
    small sample produces — and a blank beside five real numbers reads as a
    rendering fault rather than as the app declining to state a quantity it does
    not have.

    That is the governing rule's *silent* branch made visible, and it belongs to
    every figure rather than to the one that met it first. A figure that has
    computed its own rows keeps them: `payload["annotation_box"]` wins, because
    a figure knows why ITS number is missing better than this does.

    The checklist item still fails when a number is absent, and it should. The
    figure is not publication-grade without it. Failing the checklist and
    rendering honestly are different jobs.
    """
    computed = {row.get("key"): row
                for row in (payload.get("annotation_box") or [])}
    reasons = payload.get("not_estimable_because") or {}
    rows: List[Dict[str, str]] = []
    for annotation in spec.annotations:
        if annotation.key in computed:
            rows.append(dict(computed[annotation.key]))
            continue
        value = payload.get(annotation.key)
        if value is None or value == [] or value == {}:
            rows.append({"key": annotation.key, "label": annotation.label,
                         "value": NOT_ESTIMABLE,
                         "why": reasons.get(annotation.key) or _ABSENT,
                         "required": annotation.required})
        else:
            rows.append({"key": annotation.key, "label": annotation.label,
                         "value": _render_value(value), "why": "",
                         "required": annotation.required})
    return rows


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
               # THE ANNOTATION BOX TRAVELS WITH THE FIGURE. The research's
               # whole finding about this layer is that the publication-grade
               # delta is annotation rather than geometry, so a bundle carrying
               # the caption and not the numbers would ship the half that is
               # easy.
               "annotations": annotation_rows(spec, payload),
               "caption": spec.caption(payload),
               "promotable": spec.promotable,
               "promotable_because": spec.promotable_because}
        if spec.evidence is not None:
            from turbotab.packs import _badge_payload
            row.update(_badge_payload(spec.evidence, spec.claims))
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
