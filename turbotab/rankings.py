"""turbotab.rankings — every order a user picks from, enumerated.

`GUIDED-088` established the property, and it is easy to state and easy to
forget:

> **A ranking is a parameter estimated from data.** Choosing a model is a
> decision; a decision informed by the held-out rows is the thing the seal
> exists to prevent. Nobody had thought of an *order* as something estimated.
> It is one.

`GUIDED-092` is why this module exists rather than a third inline mask. The
guard written for that class asserted the instance that was already correct, so
it passed against the reverted shelf and enumerated nothing — which means a
ranking added next loop was not covered either. **A class guarded by one
example is guarded by nothing.**

So the surfaces are held here as data, and
`test_every_ranking_is_computed_on_the_training_rows.py` iterates them. Adding a
ranking means adding a row here; a row declared `TRAINING_ROWS` that can see a
held-out value fails the probe, and a probe with no row (or a row with no probe)
fails the completeness check.

## The two scopes, and why the second one exists

`TRAINING_ROWS` is the requirement. `WHOLE_TABLE` is an **exemption**, and it is
not a shrug: it carries the reason in `because` and the ledger row tracking it
in `tracked_by`, because the alternative to an explicit exemption is a surface
that quietly does not have the property while sitting in a file named for it.

The exemptions all share one shape, and it is a real design question rather than
a defect anybody has ruled on: **a description that answers "is this data
corrupted?" is entitled to every row; a description that informs a modeling
CHOICE is not.** `PRODUCT_VISION.md` §04b already draws exactly that line for
the eligibility question — *the app may show what is needed to answer "is this
data corrupted?" and not what is needed to answer "where should I cut?"* — and
nothing has applied it to the Explore surfaces. `GUIDED-096` carries it.

## What this cannot see, stated rather than found later

The completeness sweep below is over **call sites of the known primitives**. A
brand-new ranking function, written from scratch and never routed through
`models.shelf`, `selection.evidence`, `engine.rank_findings` or
`recipes.worth_asking`, is invisible to it. That is a real hole and the honest
mitigation is the sweep's own failure mode: a new *call* to any existing
primitive from an undeclared file fails, which is how a ranking normally gets
added in this codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: The requirement. The surface must be computable from `project.training_rows`
#: alone, so replacing every held-out value changes nothing it serves.
TRAINING_ROWS = "training_rows"

#: The exemption. Stated, reasoned, and tracked — never assumed.
WHOLE_TABLE = "whole_table"


@dataclass(frozen=True)
class Surface:
    """One order a user picks from."""

    key: str
    #: What the user does with it. Not what it is called — what it decides.
    decides: str
    #: Where a reader finds it: a route, or the method that composes one.
    served_by: str
    #: The primitive underneath, so the sweep below and this row agree.
    computes: str
    scope: str
    #: Why this scope. For `WHOLE_TABLE` it must name the question, not restate
    #: the behavior.
    because: str
    #: The ledger row tracking an exemption. Required when the scope is one.
    tracked_by: str = ""

    def __post_init__(self) -> None:
        if self.scope not in (TRAINING_ROWS, WHOLE_TABLE):
            raise ValueError(
                f"{self.key}: scope must be {TRAINING_ROWS!r} or "
                f"{WHOLE_TABLE!r}.")
        if len(self.because) <= 60:
            raise ValueError(
                f"{self.key}: `because` must state the reasoning. A scope with "
                f"no reason is a classification nobody can check.")
        if self.scope == WHOLE_TABLE and not self.tracked_by:
            raise ValueError(
                f"{self.key}: an exemption with no ledger row is an exemption "
                f"nobody will revisit.")


SURFACES: Tuple[Surface, ...] = (
    Surface(
        key="model_shelf",
        decides="which model the user trains",
        served_by="GET /project/{id}/models · AnalysisProject.model_shelf",
        computes="turbotab.models.shelf",
        scope=TRAINING_ROWS,
        because=("`GUIDED-088`, the instance this whole enumeration came from. "
                 "`select_models` states the requirement in its own refusal: "
                 "the shape it reads must be the shape the models will "
                 "actually be fitted on."),
    ),
    Surface(
        key="selection_evidence",
        decides="which features the user asks selection to keep",
        served_by="GET /project/{id}/selection/evidence",
        computes="turbotab.selection.evidence",
        scope=TRAINING_ROWS,
        because=("The feature ranking and the selection evidence are one "
                 "object, and it is the sharpest case in the constitution: the "
                 "selected SET encodes test signal even though no held-out "
                 "value is copied anywhere."),
    ),
    Surface(
        key="recipe_lattice",
        decides="which per-model variant question is put to the user at all",
        served_by="GET /project/{id}/recipes",
        computes="turbotab.recipes.worth_asking",
        scope=TRAINING_ROWS,
        because=("The divergence statistic measures how the columns would be "
                 "rescaled relative to one another IN THE FIT, so it is a "
                 "claim about the training rows by construction — and it read "
                 "the whole table until `GUIDED-092`."),
    ),
    Surface(
        key="ranked_findings",
        decides="which noticing the user acts on first, and in what order",
        served_by="GET /project/{id}/findings · api._recompute",
        computes="turbotab.engine.rank_findings",
        scope=WHOLE_TABLE,
        because=("Genuinely undecided, and NOT quietly. This list is both a "
                 "data-quality report and a ranking a user picks from. Masking "
                 "it would make the app blind to corruption in 15-30% of the "
                 "file — an impossible value in a held-out row is still an "
                 "impossible value, and repairing it is row-local and leaks "
                 "nothing. Not masking it means the shape claims a user routes "
                 "on were computed with the sealed rows in view. The line that "
                 "resolves it already exists in `PRODUCT_VISION.md` §04b and "
                 "has never been applied here."),
        tracked_by="GUIDED-096",
    ),
    Surface(
        key="missingness_survey",
        decides=("which column the user routes next, and which strategies are "
                 "offered for it"),
        served_by="GET /project/{id}/preprocess · AnalysisProject.missingness_survey",
        computes="turbotab.missingness.survey",
        scope=WHOLE_TABLE,
        because=("The same undecided question as `ranked_findings`, and the "
                 "case that makes it concrete: *how many blanks are in this "
                 "column* is a fact about the file the user needs whole — the "
                 "held-out rows will be scored and their blanks have to be "
                 "handled — while *which fill is right* is a choice that the "
                 "held-out rows should not inform."),
        tracked_by="GUIDED-096",
    ),
)

_BY_KEY: Dict[str, Surface] = {s.key: s for s in SURFACES}


def surface(key: str) -> Surface:
    if key not in _BY_KEY:
        raise KeyError(
            f"{key!r} is not an enumerated ranking surface. Known: "
            f"{', '.join(sorted(_BY_KEY))}.")
    return _BY_KEY[key]


def keys() -> Tuple[str, ...]:
    return tuple(s.key for s in SURFACES)


def training_scoped() -> Tuple[Surface, ...]:
    return tuple(s for s in SURFACES if s.scope == TRAINING_ROWS)


def exemptions() -> Tuple[Surface, ...]:
    return tuple(s for s in SURFACES if s.scope == WHOLE_TABLE)


# ─────────────────────────────────────────────────────────────────────────────
# The completeness sweep's declaration
# ─────────────────────────────────────────────────────────────────────────────
#
# Every module, outside the test tree, that is allowed to CALL a ranking
# primitive. A call from anywhere else is a ranking somebody added without
# declaring which rows it may see, and it fails the sweep rather than the next
# seal probe — which is the whole of `GUIDED-092`.
#
# Keyed by the attribute the call is spelled with, because that is what an AST
# walk can see without resolving imports. Definitions are not calls and are not
# listed.
CALL_SITES: Dict[str, Tuple[str, ...]] = {
    "shelf": ("turbotab/project.py",),
    "evidence": ("turbotab/api.py",),
    "rank_findings": ("turbotab/api.py",),
    "worth_asking": ("turbotab/api.py",),
    "survey": ("turbotab/api.py", "turbotab/project.py"),
}
