"""`AUDIT-013` — the Preprocess page's scaling comes from the recipe table.

`pages/05_Preprocess.py` read `spec.capabilities.requires_scaled_numeric` off the
model registry at four sites (`:512`, `:849`, `:895`, `:1140` when the row was
filed) and hard-coded `"standard"` from it. The declared capability is not the
decision — `turbotab/recipes.py` holds the decision, as a precedence lattice of
`Default` rows, and `caps:requires_scaled_numeric` is one selector in it that a pack
may override. `turbotab/packs.py:5182` registers exactly that override:
`scale → pareto` for the metabolomics lens. Reading the flag meant the override
could not reach the Classic door, while the page's sentences went on attributing the
choice to the model's declared requirement — *"Enabled standard scaling (model
requires scaling)"*, *"scaling enabled (standard); appropriate for this model"*.
The governing rule: the app may be silent and it may refuse, but it must never
assert something false.

**The refusal matters as much as the routing.** `ml/pipeline.py`'s scaler branch
chain has no `else`, so a variant it does not know applies *no* scaling while the
interface says otherwise. Routing `pareto` straight through would have converted a
display defect into a fit defect. `ml.pipeline.scaling_from_recipe` therefore falls
back and **states the departure**, and `test_a_variant_the_builder_cannot_construct_is_refused_out_loud`
drives that sentence onto the page.

The load-bearing test here is **driven**: it registers a pack-style row into
`turbotab.recipes` and then renders `pages/05_Preprocess.py`, asserting the row
reached a person. It imports nothing the fix added, so a total revert leaves it
answering and only the content differs (`AGENT_ONBOARD.md` §08.1).

`GUIDED-097`: the driven claims run against two fixtures of **different target
shape** — a continuous float outcome (`glucose`) and a binary 0/1 outcome
(`condition`). **The shape not covered is a non-numeric outcome** — a string-labeled
or multi-class target, for which `tests/integration/conftest.py` has no fixture.

`GUIDED-045`: every absence assertion is preceded by a positive control that the
surface swept was rendered and non-empty.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit.testing.v1 import AppTest
from turbotab import recipes as _recipes

from tests.integration.conftest import (
    build_classification_dataframe,
    build_test_dataframe,
    inject_data_state,
)

TARGET_SHAPE_NOT_COVERED = "non-numeric outcome (string labels / multi-class)"

# A model whose registry spec declares `requires_scaled_numeric=True`, so the
# `caps:requires_scaled_numeric` row is the one that wins for it.
SCALED_MODEL = "ridge"


@pytest.fixture
def pack_row():
    """Register a pack-style override and put the table back afterwards.

    `_DEFAULTS` is module-global and `register_default` appends, so this has to
    be snapshot/restore rather than a delete — the same reason
    `turbotab.recipes.allowed_origins` filters by origin instead of unloading.
    """
    state = _recipes.snapshot()

    def _register(variant: str):
        # A pack that introduces a NEW variant has to widen the operation
        # first — `register_default` rejects a variant the operation does not
        # list. This mirrors `turbotab/packs.py:5170-5189` rather than
        # inventing a shortcut, so the fixture stands for a row a pack can
        # actually produce (`AGENT_ONBOARD.md` §07 trap 3).
        core = _recipes.operation("scale")
        if variant not in core.variants:
            _recipes.register_operation(
                _recipes.Operation(
                    key="scale", label=core.label,
                    variants=tuple(core.variants) + (variant,),
                    determinacy=core.determinacy, scope=core.scope,
                    because=core.because, applies_to=core.applies_to,
                    origin="metabolomics_pack",
                    pushed_alternatives=tuple(core.pushed_alternatives)),
                replace_existing=True)
        _recipes.register_default(_recipes.Default(
            operation="scale", variant=variant,
            selector="caps:requires_scaled_numeric",
            origin="metabolomics_pack",
            reason=(
                f"The field convention in this lens is {variant} scaling; a "
                f"defensible compromise rather than a fact, and the core row "
                f"is offered beside it. (Registered by a test fixture standing "
                f"for turbotab/packs.py's metabolomics rows.)"
            ),
        ))

    try:
        yield _register
    finally:
        _recipes.restore(state)


def _ss(at, key, default=None):
    """`AppTest.session_state` has no `.get` — it reads `get` as a key."""
    try:
        return at.session_state[key]
    except KeyError:
        return default


def _rendered_text(at):
    parts = []
    for attr in ("markdown", "caption", "info", "warning", "error", "success"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _preprocess(df, target, task):
    at = AppTest.from_file("pages/05_Preprocess.py", default_timeout=180)
    inject_data_state(at, df, target_col=target, task_type=task)
    at.session_state[f"train_model_{SCALED_MODEL}"] = True
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


TARGET_SHAPES = [
    pytest.param(build_test_dataframe, "glucose", "regression", id="continuous-float"),
    pytest.param(
        build_classification_dataframe, "condition", "classification", id="binary-0-1"
    ),
]


def test_the_table_agrees_with_the_registry_before_any_pack_is_loaded():
    """Positive control for the whole file: core alone still says `standard`.

    If this ever fails, every assertion below is measuring the wrong thing —
    the two sources have diverged for a reason that has nothing to do with a
    pack.

    **The precondition in the name is ESTABLISHED here, not assumed.** The
    first version took *before any pack is loaded* as given, and
    `turbotab/recipes._OPERATIONS` is a module-level dict shared by the whole
    process — so any earlier test that registered a pack's `scale` left this
    one measuring that pack. It did: in a full run this asserted
    `'pareto' == 'standard'` while passing alone and passing within
    `tests/integration`, which is the signature of cross-test state rather than
    of a broken claim (`TEST-063`). pytest guarantees no ordering, so a test
    whose premise is *nothing has happened yet* has to make that true — and
    then put back what it found, so establishing the precondition does not
    become the next leak.
    """
    from ml.model_registry import get_registry

    from turbotab import recipes as _R

    was = _R.operation("scale")
    core = _R.Operation(
        key="scale", label=was.label,
        variants=("standard", "robust", "minmax", "none"),
        determinacy=was.determinacy, scope=was.scope,
        because=was.because, applies_to=was.applies_to, origin="core")
    _R.register_operation(core, replace_existing=True)
    try:
        registry = get_registry()
        assert registry, "the model registry is empty; nothing was swept"
        spec = registry[SCALED_MODEL]
        assert spec.capabilities.requires_scaled_numeric is True, (
            f"{SCALED_MODEL} no longer declares requires_scaled_numeric; pick "
            f"a model that does, or this file tests nothing")
        resolved = _recipes.resolve(SCALED_MODEL, "scale", registry)
        assert resolved.variant == "standard", (
            f"core alone resolves {SCALED_MODEL}/scale to "
            f"{resolved.variant!r}. Every assertion in this file about what a "
            f"pack CHANGES is measured against this baseline")
        assert resolved.selector == "caps:requires_scaled_numeric", (
            f"the winning row for {SCALED_MODEL}/scale is "
            f"{resolved.selector!r}, not the capability selector a pack "
            f"overrides")
    finally:
        _R.register_operation(was, replace_existing=True)


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_a_pack_row_reaches_the_preprocess_page(pack_row, builder, target, task):
    """A buildable pack override changes what the page says it will build."""
    pack_row("robust")  # a variant ml/pipeline.py CAN construct

    at = _preprocess(builder(), target, task)
    text = _rendered_text(at)

    # GUIDED-045 positive control: the summary that carries the claim rendered.
    assert text.strip(), "Preprocess rendered no text at all"
    assert "Pipeline Summary" in text, (
        "the pipeline summary did not render, so the scaling claim below was "
        f"never on the page; text was:\n{text[:1500]}"
    )

    applied = _ss(at, f"preprocess_{SCALED_MODEL}_numeric_scaling")
    assert applied == "robust", (
        f"a pack registered scale→robust against caps:requires_scaled_numeric, "
        f"which is the row that decides scaling for {SCALED_MODEL}; the "
        f"Preprocess page built {applied!r} instead. The page is reading "
        f"capabilities.requires_scaled_numeric off the registry rather than "
        f"resolving through turbotab.recipes, so the precedence lattice — and "
        f"any pack row in it — cannot reach this door."
    )
    assert "Scale: robust" in text, (
        f"the page told the user what it was not going to build. session state "
        f"says {applied!r}; the summary card says otherwise. text:\n{text[:1500]}"
    )


@pytest.mark.parametrize("builder,target,task", TARGET_SHAPES)
def test_a_variant_the_builder_cannot_construct_is_refused_out_loud(
    pack_row, builder, target, task
):
    """`pareto` is the real pack row, and `ml/pipeline.py` cannot build it.

    The scaler branch chain in `build_preprocessing_pipeline` has no `else`, so
    an unknown variant silently applies nothing. The app may refuse; it may not
    quietly substitute and say nothing.
    """
    pack_row("pareto")  # the variant turbotab/packs.py:5182 actually registers

    at = _preprocess(builder(), target, task)
    text = _rendered_text(at)
    assert text.strip(), "Preprocess rendered no text at all"

    from ml.pipeline import SUPPORTED_NUMERIC_SCALINGS  # noqa: F401  (documented below)

    # The refusal has to be visible, and it has to name both sides: what was
    # asked for and what was done instead.
    lowered = text.lower()
    assert "pareto" in lowered, (
        "a pack asked for pareto scaling and the page never mentioned it — the "
        "override was dropped in silence, which is the defect the row names "
        f"one degree worse. text:\n{text[:1500]}"
    )
    assert "departure from the table" in lowered, (
        "the page substituted a scaler the recipe table did not ask for and "
        f"did not say it was a departure. text:\n{text[:1500]}"
    )

    applied = _ss(at, f"preprocess_{SCALED_MODEL}_numeric_scaling")
    assert applied in SUPPORTED_NUMERIC_SCALINGS, (
        f"the page put {applied!r} into the pipeline config; "
        f"ml/pipeline.py builds only {SUPPORTED_NUMERIC_SCALINGS} and its "
        f"scaler branch has no else, so this fits with NO scaling while the "
        f"interface claims a scaler"
    )


def test_departure_is_stated_exactly_when_the_table_is_not_followed():
    """The invariant `scaling_from_recipe` promises, over the whole registry.

    Phrased as an assertion rather than a bare import so that a revert reads as
    a claim about the code and not as an `ImportError` at collection time.
    """
    from ml import pipeline as _pipeline
    from ml.model_registry import get_registry

    assert hasattr(_pipeline, "scaling_from_recipe"), (
        "ml/pipeline.py does not expose scaling_from_recipe, so nothing on the "
        "Preprocess page can ask the recipe table what this model's scaling is"
    )

    registry = get_registry()
    assert registry, "the model registry is empty; nothing was swept"
    seen_required = 0
    for key in sorted(registry):
        d = _pipeline.scaling_from_recipe(key, registry)
        assert d["consulted"] is True, f"{key} is in the registry but the table refused it"
        if not d["required"]:
            assert d["applied"] is None, (
                f"{key}: the table does not require scaling, so this function "
                f"must return nothing rather than a guess (§07 trap 9); it "
                f"returned {d['applied']!r}"
            )
            continue
        seen_required += 1
        assert (d["departure"] != "") == (d["applied"] != d["table_variant"]), (
            f"{key}: departure={d['departure']!r} but applied={d['applied']!r} "
            f"vs table={d['table_variant']!r} — the sentence and the fact disagree"
        )
        assert d["applied"] in _pipeline.SUPPORTED_NUMERIC_SCALINGS, (
            f"{key}: {d['applied']!r} is not a scaler this builder constructs"
        )
    assert seen_required, (
        "no model in the registry requires scaling, so the branch under test "
        "never ran"
    )
