"""Per-model preprocessing defaults, as data — and provably extensible.

`turbotab/recipes.py` is model × operation → variant + reason, resolved at
runtime. The reason it is a table rather than a chain of `if model in TREES`
branches is `DOMAIN_PACKS.md` §02: a pack supplies detectors, reference data,
conventions and prose and **never interface**, which only holds if the defaults
it adjusts are a lookup rather than code.

That claim is untestable against a pack that does not exist yet, and *"an
extension point with no test is a claim, not a capability."* So this file
installs a **fake pack** — one added operation, one overridden variant — and
asserts both resolve. Sample-level normalization is the honest example: PQN is
not in the generic catalogue, no amount of overriding variants would produce it,
and it is exactly the sort of thing a metabolomics pack would bring.

The other half is the routing split. Model-determined is a FACT and may be
pre-selected with a rendered skip; data-determined is a CHOICE and stays asked.
The split lives on the OPERATION, because it is a property of the question and
not of the answer, and this file asserts that rather than trusting it.

**Structure, not prose substrings.** The model-determined layer is checked
against `ModelCapabilities` for every model in the registry, not against a
hand-written expected dict — a second copy of the registry is the thing this
module was built to avoid, and a test containing one would drift the same way.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.model_registry import get_registry                            # noqa: E402
from turbotab import packs as P                                       # noqa: E402
from turbotab import recipes as R                                     # noqa: E402


@pytest.fixture
def table():
    """Every test gets the CORE table, however badly it mangles it — and
    however badly anything that ran earlier in the session did.

    It used to snapshot and restore, which gives back *whatever was there*
    rather than core. That was indistinguishable from core for as long as
    nothing in the suite loaded a real pack before this file ran. `GUIDED-099`
    is what changed: `packs.load` is process-global and never unloads, so once
    any project selects the metabolomics lens, an unscoped `resolve("ridge",
    "scale")` answers `pareto` — and every assertion here about "core" was
    really about core-plus-whatever-ran-first. Which is the failure this file's
    own `test_the_pack_is_gone_when_the_test_is` was written to prevent, one
    level out from where it was looking.

    So: reset to core, run, put back exactly what was there — the table AND the
    load bookkeeping, because restoring one without the other leaves them
    disagreeing about which rows are registered.
    """
    state = R.snapshot()
    loaded = P.loaded_for_test()
    R._install_core()
    P.unload_for_test()
    try:
        yield R
    finally:
        R.restore(state)
        P.restore_loaded_for_test(loaded)


# ─────────────────────────────────────────────────────────────────────────────
# The extension point, proved before a pack exists
# ─────────────────────────────────────────────────────────────────────────────

PQN = R.Operation(
    key="pqn", label="Probabilistic quotient normalization",
    variants=("none", "median_fold"),
    determinacy=R.DATA_DETERMINED, scope=R.STATEFUL,
    because=("Stateful: the reference spectrum is a median across the rows the "
             "normalizer saw, so it is fitted inside the training folds like "
             "any other column statistic."),
    applies_to="numeric", origin="fake_metabolomics_pack",
    pushed_alternatives=(("median_fold", "none"), ("none", "median_fold")))


def install_fake_pack() -> None:
    """One added operation and one overridden variant. The whole contract.

    Deliberately BOTH, because they fail differently: a table that supports
    overriding variants and not adding operations looks extensible right up to
    the first pack that needs an operation the core catalogue never imagined,
    and then needs the schema changed.
    """
    R.register_operation(PQN)
    R.register_default(R.Default(
        operation="pqn", variant="median_fold", selector="*",
        reason=("Dilution varies between samples for reasons that are not "
                "biological, and every downstream comparison inherits it."),
        origin="fake_metabolomics_pack"))
    # The override: robust scaling for one exact model, beating the core rule
    # that gives every `requires_scaled_numeric` model standard scaling.
    R.register_default(R.Default(
        operation="scale", variant="robust", selector="ridge",
        reason=("Feature intensities in this assay are right-tailed by "
                "construction, so a standard deviation is dominated by the "
                "tail and the median-and-IQR pair is the stabler summary."),
        origin="fake_metabolomics_pack"))


def test_a_pack_can_add_an_operation_and_it_resolves(table):
    """The half that is not merely configuration.

    Asserted on the resolved object — the variant, the origin AND the reason —
    because a pack-added operation that resolves to a variant with no traceable
    origin is an operation the record cannot attribute.
    """
    assert "pqn" not in {o.key for o in R.operations()}, "core already has pqn"
    install_fake_pack()

    resolved = R.resolve("rf", "pqn")
    assert resolved.variant == "median_fold"
    assert resolved.origin == "fake_metabolomics_pack", (
        "the resolved default does not carry the pack it came from, so the "
        "record cannot say which pack asserted it")
    assert resolved.determinacy == R.DATA_DETERMINED, (
        "a pack-added operation lost its routing classification on the way "
        "through resolution, which would make it silently pre-selectable")

    # And it appears in the full recipe, not merely on request — an operation
    # that only resolves when named is an operation no interface will render.
    assert "pqn" in {r.operation for r in R.recipe("rf")}


def test_a_pack_can_override_a_variant_and_specificity_decides(table):
    """The other half, with the resolution rule that makes it safe.

    An override is only meaningful if it BEATS the core rule it contradicts.
    Asserted against a model the core rule genuinely covers — `ridge` requires
    scaled numeric input, so core says `standard` — otherwise the test proves
    only that a rule with no competition wins.
    """
    before = R.resolve("ridge", "scale")
    assert before.variant == "standard" and before.selector.startswith("caps:"), (
        "the core rule this test overrides is not the rule it thinks it is")

    install_fake_pack()
    after = R.resolve("ridge", "scale")
    assert after.variant == "robust", (
        "the pack's exact-model rule lost to core's capability rule. Most "
        "specific must win, or a pack can never correct a general default.")
    assert after.origin == "fake_metabolomics_pack"

    # And the override is SCOPED: another model covered by the same core rule
    # is untouched, which is what makes an override an override rather than a
    # global reconfiguration.
    others = [k for k, s in get_registry().items()
              if s.capabilities.requires_scaled_numeric and k != "ridge"]
    assert others, "no other scaled-numeric model exists; the check is vacuous"
    assert R.resolve(others[0], "scale").variant == "standard", (
        f"overriding ridge also changed {others[0]}; the selector is not "
        "scoping anything")


def test_the_pack_is_gone_when_the_test_is(table):
    """The snapshot/restore contract itself, because everything above rests on it.

    A pack that leaks between tests makes every later assertion about "core" an
    assertion about core-plus-whatever-ran-first, and the failure then appears
    in a file that never mentioned packs. Asserted here rather than trusted,
    because the `table` fixture is the only thing keeping the other tests
    honest.
    """
    saved = R.snapshot()
    install_fake_pack()
    assert "pqn" in {o.key for o in R.operations()}
    assert R.resolve("ridge", "scale").variant == "robust"

    R.restore(saved)
    assert "pqn" not in {o.key for o in R.operations()}
    assert R.resolve("ridge", "scale").variant == "standard", (
        "restoring the table left the pack's override in place; every later "
        "test would be measuring a table this file mangled")


def test_a_pack_cannot_shadow_a_core_operation_silently(table):
    """`DOMAIN_PACKS.md` §05 in its quietest form.

    A pack that redefines `scale` without saying so changes behavior nobody can
    see. Shadowing is allowed — deliberately, with a flag — because a pack that
    genuinely needs different variants should not have to invent a new key.
    """
    clash = R.Operation(
        key="scale", label="Assay scaling", variants=("pareto", "none"),
        determinacy=R.MODEL_DETERMINED, scope=R.STATEFUL,
        because=("Stateful: the square root of the standard deviation is a "
                 "property of the column across the rows it saw."),
        origin="fake_metabolomics_pack")
    with pytest.raises(R.RecipeError):
        R.register_operation(clash)
    # L53. THE ORIGINAL IS RESTORED, and it was not before. `_OPERATIONS` is a
    # module-level dict, so `replace_existing=True` here rewrote `scale` for the
    # REST OF THE PROCESS: every later test in the same run resolved `scale` to
    # ("pareto", "none") instead of the core's ("standard", "robust", "minmax",
    # "none"). It went unnoticed for as long as nothing downstream asked, and
    # L53-C added a test that does — `test_the_table_agrees_with_the_registry`
    # went red with `assert 'pareto' == 'standard'` in the full suite while
    # passing alone and passing within `tests/integration`, which is the
    # signature of exactly this.
    #
    # THE APP IS NOT AFFECTED and that was checked rather than assumed: driving
    # a metabolomics lens on one project and then reading `/recipes` on another
    # leaves the variants at the core's four, so no pack rewrites `scale` at
    # request time and there is no cross-project bleed in the server. This is a
    # test-isolation defect and it is stated as one.
    original = R.operation("scale")
    try:
        R.register_operation(clash, replace_existing=True)
        assert R.operation("scale").variants == ("pareto", "none")
    finally:
        R.register_operation(original, replace_existing=True)
    assert R.operation("scale").variants == original.variants, (
        "the core `scale` operation was not restored, so every test after this "
        "one in the same process resolves a fake pack's variants")


def test_an_operation_must_answer_the_litmus_and_a_default_must_state_a_reason():
    """A pack does not get to skip clause §06, or to skip the rendered skip.

    The reason on a default is not documentation — it is the text the user sees
    where the question would have been. A default without one cannot be
    pre-selected honestly, so it is refused at registration rather than
    rendering as a blank.
    """
    with pytest.raises(R.RecipeError):
        R.Operation(key="x", label="X", variants=("a",), determinacy="maybe",
                    scope=R.STATEFUL, because="x" * 60)
    with pytest.raises(R.RecipeError):
        R.Operation(key="x", label="X", variants=("a",),
                    determinacy=R.DATA_DETERMINED, scope="whenever",
                    because="x" * 60)
    with pytest.raises(R.RecipeError):
        R.Operation(key="x", label="X", variants=("a",),
                    determinacy=R.DATA_DETERMINED, scope=R.ROW_LOCAL,
                    because="because")


def test_a_default_for_an_unregistered_operation_or_variant_is_refused(table):
    with pytest.raises(R.RecipeError):
        R.register_default(R.Default(
            operation="nonesuch", variant="a",
            reason="x" * 60))
    with pytest.raises(R.RecipeError):
        R.register_default(R.Default(
            operation="scale", variant="pareto", reason="x" * 60))
    with pytest.raises(R.RecipeError):
        R.register_default(R.Default(
            operation="scale", variant="none", reason="because"))


# ─────────────────────────────────────────────────────────────────────────────
# The routing split
# ─────────────────────────────────────────────────────────────────────────────

def test_every_model_resolves_every_operation():
    """No model falls through to nothing.

    A model with no default for an operation forces the interface to invent
    one, and an invented default has no reason to show — which is the rendered
    skip failing open into a silent decision.
    """
    reg = get_registry()
    missing = []
    for key in reg:
        for op in R.operations():
            try:
                R.resolve(key, op.key)
            except R.RecipeError as exc:
                missing.append(f"{key}/{op.key}: {exc}")
    assert not missing, "\n".join(missing)


def test_the_model_determined_layer_agrees_with_the_registry():
    """Checked against `ModelCapabilities`, never against a second list.

    `requires_scaled_numeric` is already declared on every spec. Hand-listing
    which models need scaling would be a second copy of a fact that exists, and
    this project keeps finding second copies to have drifted. So the assertion
    is the JOIN: scale is `none` exactly where the capability is false.
    """
    wrong = []
    for key, spec in get_registry().items():
        got = R.resolve(key, "scale").variant
        want_none = not spec.capabilities.requires_scaled_numeric
        if want_none != (got == "none"):
            wrong.append(f"{key}: requires_scaled_numeric="
                         f"{spec.capabilities.requires_scaled_numeric} but "
                         f"scale={got}")
    assert not wrong, "\n".join(wrong)


def test_a_fact_may_be_preselected_and_a_choice_may_not():
    """The routing constitution, on the object rather than in a comment.

    Scaling is model-determined whatever variant you pick; a power transform is
    data-determined whatever model you pick. That is why the determinacy lives
    on the operation — and why `may_be_preselected` must be readable from the
    resolved row, so a renderer cannot decide it locally.
    """
    facts = {r.operation for r in R.recipe("ridge") if r.may_be_preselected}
    choices = {r.operation for r in R.recipe("ridge") if not r.may_be_preselected}
    assert "scale" in facts and "encode" in facts
    assert "power" in choices and "outliers" in choices

    # And the classification does not depend on the model, which is the claim
    # "determinacy is a property of the question" in checkable form.
    for key in ("rf", "ridge", "knn"):
        if key not in get_registry():
            continue
        assert {r.operation for r in R.recipe(key) if r.may_be_preselected} == facts


def test_every_preselectable_default_states_a_reason_long_enough_to_read():
    """A rendered skip shows the reason where the question would have been.

    Asserted on every model × preselectable operation in the registry, because
    the one that renders blank will be the one nobody thought to check.
    """
    thin = [f"{k}/{r.operation}" for k in get_registry()
            for r in R.recipe(k)
            if r.may_be_preselected and len(r.reason) < 60]
    assert not thin, (
        f"{thin} would render a skip with nothing under it. A pre-selected "
        "fact with no visible reason is a decision made in the user's name.")


# ─────────────────────────────────────────────────────────────────────────────
# Ask only when the choice changes the answer
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_columns(n: int = 400) -> pd.DataFrame:
    """Different scales, same shape. σ/IQR ≈ 1/1.349 in every column."""
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(100, 15, n),
        "c": rng.normal(-4, 0.02, n),
        "d": rng.normal(1e4, 900, n),
    })


def heavy_tailed_columns(n: int = 400) -> pd.DataFrame:
    """Two well-behaved columns and two with tails, which is the hard case.

    Uniformly heavy tails would inflate σ/IQR everywhere and the RATIO would
    stay near-constant — the two scalings would still differ by one global
    factor. It is the MIXTURE that makes standard and robust reweight the
    features against one another.
    """
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(100, 15, n),
        "c": rng.standard_t(1.6, n) * 30,
        "d": rng.standard_t(1.4, n) * 5,
    })


def test_on_well_behaved_data_the_scaling_choice_is_not_worth_asking():
    df = gaussian_columns()
    r = R.resolve("ridge", "scale")
    ask, evidence = R.worth_asking(df, list(df.columns), r)
    assert not ask, (
        f"asked about standard vs robust on Gaussian columns: {evidence}")
    assert evidence is not None and evidence.statistic < R.SCALE_THRESHOLD


def test_on_heavy_tailed_data_the_same_choice_is_worth_asking():
    """The other side of the same mechanism, and the reason it is not a constant.

    A suppressor that suppresses everywhere is indistinguishable from a
    hardcoded default, so the pair of tests is the claim — one fixture where
    asking is ceremony, one where it is essential, same code path.
    """
    df = heavy_tailed_columns()
    r = R.resolve("ridge", "scale")
    ask, evidence = R.worth_asking(df, list(df.columns), r)
    assert ask, f"did not ask on heavy-tailed columns: {evidence}"
    assert evidence.statistic > R.SCALE_THRESHOLD
    assert evidence.threshold == R.SCALE_THRESHOLD, (
        "the evidence does not carry the threshold it was judged against, so a "
        "reader cannot disagree with a number nobody calibrated")


def test_an_alternative_nothing_can_compare_is_raised_rather_than_suppressed(table):
    """The failure mode the whole mechanism must avoid.

    Not knowing whether a choice matters is not evidence that it does not. The
    fake pack pushes `none` against `median_fold` and teaches no comparison, so
    the alternative is raised — and the returned evidence is `None`, which a
    caller must never read as "measured and found identical".
    """
    install_fake_pack()
    df = gaussian_columns()
    r = R.resolve("rf", "pqn")
    raise_it, evidence = R.worth_asking(df, list(df.columns), r)
    assert raise_it and evidence is None, (
        "an alternative with no comparison was suppressed. Silence about "
        "whether a choice matters is not a finding that it does not.")


def test_a_variant_with_no_pushed_alternative_is_an_absence_not_a_suppression():
    """The distinction the suppression count depends on.

    `power` resolves to `none` and pushes nothing against it — there is no
    question here to suppress. Counting that as a suppression would report the
    mechanism working hardest on exactly the rows it never looked at. The
    boolean is False and the evidence is `None`, and the caller tells the two
    apart by the evidence rather than by the boolean.
    """
    df = gaussian_columns()
    raise_it, evidence = R.worth_asking(
        df, list(df.columns), R.resolve("ridge", "power"))
    assert raise_it is False and evidence is None, (
        "a row with no pushed alternative produced evidence, so the "
        "suppression count would include questions that never existed")


def test_a_pack_can_teach_the_mechanism_about_its_own_operation(table):
    """Extensibility again, on the half that is easiest to forget.

    A pack that adds an operation and cannot teach the divergence test gets its
    operation asked unconditionally forever — correct, but the mechanism stops
    applying to exactly the operations a domain expert knows most about.
    """
    install_fake_pack()
    df = gaussian_columns()
    r = R.resolve("rf", "pqn")
    assert R.worth_asking(df, list(df.columns), r)[0] is True

    R.register_divergence("pqn", lambda d, c, a, b: R.Divergence(
        operation="pqn", a=a, b=b, material=False, statistic=0.0,
        threshold=1.0, evidence="Dilution is constant across these samples."))
    ask, evidence = R.worth_asking(df, list(df.columns), r)
    assert not ask and evidence is not None and not evidence.material
