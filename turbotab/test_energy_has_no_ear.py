"""`DRIVE-034` — a SETTLED badge on a method used outside its domain.

A human asked the prevalence widget for `kcal / modeled usual intake / against
the EAR` and the app answered *"Prevalence of inadequacy for `kcal` is computed
by the EAR cut-point method"* — **with a SETTLED badge**.

## Settled by reading the pack, which is what the row required

`research/NUTRITION_PACK.md` §07, in the paragraph headed **"Exceptions that
must be hard-coded"**, carrying `[SETTLED]`:

> **Energy** has no EAR-style cut-point (use EER).

and §11's conflation list repeats it:

> The EAR cut-point applies to usual intake only, requires an EAR (not an AI),
> is not the RDA, and fails for iron in menstruating women.

So this is the row's **first** outcome — the pack excludes energy, the widget
must refuse, and the refusal is the fix. Not the second (the pack is silent) and
not the third (the pack permits it).

## Why all four existing refusals fell through

`prevalence_of_inadequacy` had four, and each asks a different question: is the
subject a nutrient, is the reference an AI, is the reference the RDA, is the
basis a usual-intake distribution. **Energy passes every one.** It *is* a
nutrient the pack recognizes, it is not AI-only, and the drive asked for the EAR
against modeled usual intake — so the settled tail answered.

That is `GUIDED-170`'s shape one axis over: a complete set of refusals, none of
which asked this question. `GUIDED-170` was `SEQN`, where the missing axis was
*is this a nutrient at all*; here it is *does this nutrient have an EAR at all*.

## Why it is terminal rather than a routing rule

Iron in menstruating women routes to the probability approach because a
requirement distribution exists and is skewed. Energy's requirement standard is
the **EER**, which is an estimate for an individual rather than a distribution a
population's intakes sit below — so there is no cut-point and no probability
approach either. There is no method to route to.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PACK = (Path(__file__).resolve().parents[1] / "docs" / "turbotab" /
        "research" / "NUTRITION_PACK.md")


def _refuse(nutrient, **kw):
    from turbotab import nutrition

    kw.setdefault("basis", "usual_intake_modeled")
    kw.setdefault("reference_kind", "EAR")
    return nutrition.prevalence_of_inadequacy(nutrient, **kw)


def test_the_pack_says_energy_has_no_ear(capsys):
    """The citation, asserted — so the refusal cannot outlive its source.

    If the pack is ever revised to permit this, the fix is not to quietly delete
    a branch: it is for this test to fail and someone to read the new text.
    """
    text = PACK.read_text(encoding="utf-8")
    # The sentence wraps in the source, so the whitespace is normalized rather
    # than the assertion loosened to a fragment that could match elsewhere.
    flat = " ".join(text.split())
    assert "**Energy** has no EAR-style cut-point (use EER)" in flat, (
        "NUTRITION_PACK.md no longer carries the sentence this refusal rests "
        "on — re-read §07 before changing the code")
    # And it is in the hard-coded-exceptions paragraph, not an aside.
    para = flat[flat.index("Exceptions that must be hard-coded"):][:600]
    assert "Energy" in para and "EER" in para
    with capsys.disabled():
        print("\n  pack §07: Energy has no EAR-style cut-point (use EER)")


@pytest.mark.parametrize("name", [
    "kcal", "energy", "calories", "kilocalories", "energy_kcal",
    "DR1TKCAL", "DR2TKCAL", "kj", "kilojoules",
])
def test_every_spelling_of_energy_is_refused(name, capsys):
    """The drive used `kcal`; a registry with one spelling would be a refusal
    that fires on the case that was reported and nothing else."""
    from turbotab import nutrition

    with pytest.raises(nutrition.PrevalenceRefusal) as caught:
        _refuse(name)
    said = str(caught.value)
    assert "no Estimated Average Requirement" in said, said
    assert "EER" in said, said
    with capsys.disabled():
        print(f"\n  {name:<14} refused")


def test_the_refusal_is_terminal_rather_than_a_routing_rule(capsys):
    """Iron routes; energy does not, and the sentences must differ.

    Iron in menstruating women has a requirement distribution that is skewed, so
    another method applies. Energy has no requirement distribution at all, so a
    refusal that offered the probability approach would be a second wrong
    answer.
    """
    from turbotab import nutrition

    routed = nutrition.prevalence_of_inadequacy(
        "iron", basis="usual_intake_modeled", reference_kind="EAR",
        stratum="menstruating")
    assert routed["method"] == "probability_approach"

    with pytest.raises(nutrition.PrevalenceRefusal) as caught:
        _refuse("kcal")
    said = str(caught.value)
    assert "probability approach either" in said, (
        f"the energy refusal does not rule out the other method: {said}")
    with capsys.disabled():
        print(f"\n  iron → probability approach · energy → no method at all")


def test_it_offers_the_distribution_and_forbids_the_claim(capsys):
    """A refusal that offers nothing is indistinguishable from a missing
    feature. This one draws the column and names what it will not label."""
    from turbotab import nutrition

    with pytest.raises(nutrition.PrevalenceRefusal) as caught:
        _refuse("kcal")
    offer = getattr(caught.value, "offer", None)
    assert offer, "the refusal offers nothing"
    assert offer["draw"] == "per_nutrient_distribution"
    assert offer["forbidden"] == "prevalence_of_inadequacy_for_energy"
    assert "EER is per person" in offer["caption_note"]
    with capsys.disabled():
        print(f"\n  offers {offer['draw']}, forbids {offer['forbidden']}")


def test_a_nutrient_that_does_have_an_ear_still_answers(capsys):
    """The positive control, and the reason this is not "refuse everything".

    Protein has an EAR and a symmetric requirement distribution. If the widget
    stopped answering for it, the refusal would have eaten the capability rather
    than corrected it — `PRODUCT_VISION`'s shelf-is-never-shortened rule.
    """
    got = _refuse("protein")
    assert got["method"] == "cut_point"
    assert got["reference_kind"] == "EAR"
    with capsys.disabled():
        print(f"\n  protein still answers: {got['method']}")


def test_the_badge_is_not_spent_on_the_refused_case(capsys):
    """The reason this outranked an unbadged wrong answer.

    SETTLED is the apparatus that keeps a settled fact and a disputed one from
    reading alike. The refusal still carries the badge — it IS a settled claim,
    that energy has no EAR — and what must never carry it is the computed
    prevalence.
    """
    from turbotab import nutrition

    with pytest.raises(nutrition.PrevalenceRefusal) as caught:
        _refuse("kcal")
    evidence = getattr(caught.value, "evidence", None)
    assert evidence is not None, "the refusal carries no evidence badge"
    as_dict = evidence.to_dict()
    assert as_dict["evidence_status"] == "SETTLED"
    assert "NUTRITION_PACK.md" in as_dict["source"]
    with capsys.disabled():
        print(f"\n  refusal badged {as_dict['evidence_status']} "
              f"from {as_dict['source'][:46]}…")
