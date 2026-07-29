"""Guard #2, executable: **a pack must not fire on non-matching data.**

`DOMAIN_PACKS.md` §03, and the one that decides whether the whole idea is safe:

> A pack that fires on the wrong data **asserts something false in the one place
> the app has promised it never will** — and it does so authoritatively, which
> makes it harder for the user to catch than an ordinary bug.

So this file runs **every pack against every fixture** and asserts the question
count is unchanged everywhere the pack does not belong. The discrimination
matrix it produces is the deliverable; the individual `must surface` assertions
beside it are what stop a pack passing the guard by doing nothing at all.

**Both halves are necessary and they fail differently.** A pack that never fires
passes guard #2 perfectly and is worthless. A pack that fires everywhere is
worse than worthless. Neither half alone is a test.

The metric is **questions added**, not findings changed. A pack that reframes an
existing finding without asking anything new has not violated guard #2 — it has
done exactly what §02 says a pack is for, which is to change the answers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import packs as P                                       # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

# Every fixture, its target, and the lenses it legitimately matches. A pack not
# listed for a fixture must add nothing to it.
FIXTURES = {
    "metabolomics_untargeted": ("responder", {P.METABOLOMICS}),
    "dietary_recalls": ("hba1c", {P.DIETARY, P.CLINICAL}),
    "clinical_longitudinal": ("progressed", {P.CLINICAL}),
    "survey_instrument": ("sought_support", {P.SURVEY}),
    "genomics_expression": ("condition", {P.GENOMICS}),
    "clinic_visits": ("outcome", {P.CLINICAL}),
}

REAL_PACKS = [P.METABOLOMICS, P.GENOMICS, P.DIETARY, P.CLINICAL, P.SURVEY]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def _questions(df: pd.DataFrame, target: str, lens) -> int:
    """Pushed questions the interview asks at the data step under this lens.

    Pull affordances are excluded, as everywhere else: a pull affordance is
    offered and costs nothing to ignore, so counting one would read as the
    interview becoming more talkative when it is doing the opposite.
    """
    from turbotab import engine
    structural = engine.diagnose(df, target=target)
    ranked = engine.rank_findings(structural, None)
    ranked = P.reframe(ranked, lens or [], df)
    ranked = ranked + P.findings(df, lens or [])

    block = P.likert_block(df) if (lens and P.SURVEY in lens) else None
    plan = router.plan(ranked, target=target, detection=None, step="data",
                       deferred={}, answered=["state_lens"],
                       recommendations=[], signals=None, missing_columns=[],
                       lens_block=block)
    router.audit(plan)
    return sum(1 for q in plan if q.mode == "push" and q.status == "asked")


# ── the guard ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("pack", REAL_PACKS)
def test_a_pack_adds_no_question_to_a_fixture_it_does_not_match(pack, fixture):
    """The matrix, one cell at a time.

    Parametrized both ways rather than looped, so a failure names the pair —
    `[survey-genomics_expression]` in the output is the whole diagnosis, and a
    single test over a nested loop would say only that something somewhere
    fired.

    Discharges `lockbox-01`: the lens comes first, and it earns that position
    only if it is safe — a lens that fires on the wrong data is worse than no
    lens at all.
    """
    target, matches = FIXTURES[fixture]
    if pack in matches:
        pytest.skip(f"{pack} legitimately matches {fixture}")
    df = load(fixture)
    baseline = _questions(df, target, [])
    with_pack = _questions(df, target, [pack])
    assert with_pack == baseline, (
        f"the {pack} pack added {with_pack - baseline} question(s) to "
        f"{fixture}, which it does not match. A pack that fires on the wrong "
        f"data asserts something false authoritatively.")


def test_every_pack_installed_at_once_adds_nothing_to_the_control():
    """`clinic_visits.csv` is the control, and this is its whole assertion.

    Not just each pack alone — ALL FIVE AT ONCE. A pack can be individually
    quiet and collectively noisy: two packs whose priors disagree, or two
    detectors whose preconditions are each nearly met, are exactly the
    combination a per-pack loop never constructs.

    Discharges `lockbox-01`: the lens is first in the sequence, so a pack that
    fires on a table it does not describe corrupts everything after it.
    """
    df = load("clinic_visits")
    baseline = _questions(df, "outcome", [])
    everything = _questions(df, "outcome", REAL_PACKS)
    assert everything == baseline, (
        f"every pack installed together added {everything - baseline} "
        f"question(s) to the generic fixture")


def test_no_pack_adds_a_question_to_a_fixture_it_does_match_either():
    """The stricter reading, and it holds for four of the five.

    Guard #1 says a pack supplies findings and defaults and may not invent a
    card type. `reverse_coding` is the one deliberate exception — it needs a
    codebook the app does not have, so it cannot be a stated fact — and it is
    named here rather than left as an unexplained plus-one, because an
    exception nobody has written down is indistinguishable from a leak.
    """
    added, removed = {}, {}
    for fixture, (target, matches) in FIXTURES.items():
        df = load(fixture)
        baseline = _questions(df, target, [])
        for pack in sorted(matches):
            n = _questions(df, target, [pack]) - baseline
            if n > 0:
                added[(pack, fixture)] = n
            elif n < 0:
                removed[(pack, fixture)] = -n
    assert added == {(P.SURVEY, "survey_instrument"): 1}, added

    # Questions REMOVED are the thesis, not a leak. `DOMAIN_PACKS.md` opens on
    # it: domain knowledge should make the interview SHORTER, and the naive
    # reading of "support more fields" is "ask more questions", which is the
    # failure the whole product exists to escape. The genomics lens rereads ten
    # `critical` sentinel findings as counts, so ten repair questions stop
    # being asked.
    assert removed == {(P.GENOMICS, "genomics_expression"): 10}, removed


# ── the other half: a pack that never fires is worthless ─────────────────────

def test_the_metabolomics_pack_reads_its_own_fixture():
    found = {f["id"]: f for f in P.findings(load("metabolomics_untargeted"),
                                            [P.METABOLOMICS])}
    assert "pack::metabolomics::left_censored" in found
    assert "pack::metabolomics::run_order" in found
    assert "pack::metabolomics::pooled_qc" in found

    censored = found["pack::metabolomics::left_censored"]
    assert censored["marker"] == "derived", (
        "detection is derived — a detection limit is one instrument threshold "
        "and which features fall below it is decided by where they sit")
    assert censored["params"]["rho"] < -0.9
    assert censored["params"]["suggested_method"] == "half_minimum"

    # Correction is OFFERED and never automatic: it alters every value.
    assert found["pack::metabolomics::run_order"]["marker"] == "offered"
    assert found["pack::metabolomics::run_order"]["params"]["run_order_column"] \
        == "run_order"

    qc = found["pack::metabolomics::pooled_qc"]
    assert qc["params"]["n_qc"] == 8
    assert qc["params"]["qc_value"] == "pooled_qc"
    assert qc["marker"] == "derived", (
        "they are not participants; modeling them is an error with no "
        "legitimate reading")


def test_the_dietary_pack_reads_its_own_fixture():
    found = {f["id"]: f for f in P.findings(load("dietary_recalls"), [P.DIETARY])}
    assert "pack::dietary::compositional" in found
    assert "pack::dietary::implausible_intake" in found
    assert "pack::dietary::energy_adjustment" in found

    comp = found["pack::dietary::compositional"]
    assert set(comp["params"]["columns"]) == {
        "protein_pct_kcal", "fat_pct_kcal",
        "carbohydrate_pct_kcal", "alcohol_pct_kcal"}
    assert comp["params"]["total"] == 100.0
    assert comp["params"]["gates"] == "collinearity_figure", (
        "this gates the collinearity figure rather than adding a step")

    intake = found["pack::dietary::implausible_intake"]
    assert intake["marker"] == "offered", (
        "an exclusion changes N, so it is a criterion the user states")
    assert intake["params"]["n_flagged"] == 20
    assert intake["params"]["offers"] == "eligibility_criterion"
    assert intake["fix_kind"] == "none", "it must never be applied"


def test_the_survey_pack_reads_its_own_fixture_and_refuses_to_guess():
    df = load("survey_instrument")
    found = {f["id"]: f for f in P.findings(df, [P.SURVEY])}
    assert "pack::survey::ordinal_declared" in found
    ordinal = found["pack::survey::ordinal_declared"]
    assert ordinal["params"]["scale"] == [1, 2, 3, 4, 5]
    assert len(ordinal["params"]["columns"]) == 40
    assert ordinal["params"]["encoding"] == "declared"

    # AND IT DOES NOT NAME THE REVERSE-CODED ITEMS. It could: they correlate at
    # about -0.56 with the rest and an inference would get all eight right on
    # this file. That is exactly why it must not — two subscales measuring
    # opposing constructs produce identical evidence.
    from turbotab.sample_data.make_fixtures import REVERSE_CODED
    blob = repr(found)
    for item in REVERSE_CODED:
        assert f"reverse" not in blob.lower() or item not in ordinal["params"].get(
            "reverse_coded", []), "the pack inferred reverse-coding"
    assert "reverse_coded" not in ordinal["params"]
    reverse = P.priors([P.SURVEY], "reverse_coding", df)
    assert [r["variant"] for r in reverse] == [None], (
        "the pack named the reverse-coded items; it must ask for them")
    assert reverse[0]["marker"] == "offered"
    # Scoped to the instrument's own items, not to the table (`GUIDED-027`).
    assert set(reverse[0]["columns"]) == set(ordinal["params"]["columns"])


def test_the_genomics_pack_recognizes_the_shape_and_asserts_no_normalization():
    """The thin pack's whole job, and the one place declining IS the answer."""
    found = {f["id"]: f for f in P.findings(load("genomics_expression"),
                                            [P.GENOMICS])}
    assert "pack::genomics::counts_p_over_n" in found
    counts = found["pack::genomics::counts_p_over_n"]
    # 496, not 495: `age` is a non-negative whole number and is
    # shape-indistinguishable from a count. The detector reads shape and says
    # so, which is the honest reading — inferring that a column called `age` is
    # not a gene would be the name list constitution §02 forbids, arriving in a
    # different pack.
    assert counts["params"]["n_features"] == 496
    assert counts["params"]["p_over_n"] > 8.0
    assert counts["params"]["model_prior"] == "regularized_first"

    # THE ASSERTION THAT MATTERS. CPM, TPM and VST are not interchangeable, and
    # a pack that guesses is the confidently-wrong failure DOMAIN_PACKS exists
    # to prevent. The key is PRESENT and its variant is None — an absent key
    # would be indistinguishable from a pack that never considered the question.
    assert counts["params"]["normalization_default"] is None
    prior = P.priors([P.GENOMICS], "normalization",
                     load("genomics_expression"))
    assert len(prior) == 1, "the considered refusal must be RECORDED"
    assert prior[0]["variant"] is None
    assert prior[0]["marker"] == "offered"
    assert prior[0]["scope"] == P.DATASET

    # AND NOTHING REACHED THE RECIPE TABLE. The recipe table is canonical for
    # variant preferences, so a genomics pack that registered a normalization
    # default would be asserting one — the absence here IS the position.
    assert P.recipe_origins([P.GENOMICS]) == []


def test_the_clinical_pack_is_one_prior_and_no_findings():
    """Deliberately thin: physiologic bounds and unit harmonization already
    exist in the core. The pack adds ONE prior, and it points the OPPOSITE way
    from the metabolomics one."""
    assert P.PACKS[P.CLINICAL].detectors == ()
    for fixture in FIXTURES:
        assert P.findings(load(fixture), [P.CLINICAL]) == []

    clinical = P.priors([P.CLINICAL], "missingness_direction")[0]
    metabolomics = P.priors([P.METABOLOMICS], "missingness_direction")[0]
    assert clinical["mechanism"] == "not_ordered"
    assert metabolomics["mechanism"] == "below_detection_limit"
    assert clinical["mechanism"] != metabolomics["mechanism"]

    # And the disagreement SURVIVES being asked for. A lens that is both
    # clinical and metabolomic gets both priors, named, rather than one of them
    # silently winning.
    both = P.priors([P.CLINICAL, P.METABOLOMICS], "missingness_direction")
    assert {p["pack"] for p in both} == {P.CLINICAL, P.METABOLOMICS}

    # The clinical prior is `offered`, so it informs the mechanism question and
    # never removes it. Only `derived` reaches the skip rule.
    assert clinical["marker"] == "offered"


# ── reframing changes the answer, never the question ─────────────────────────

def test_the_wide_shape_is_reframed_and_never_deleted():
    """A pack that DELETED this finding would delete it on the control too,
    where `bp_1`/`bp_2`/`bp_3` is exactly what it is for."""
    from turbotab import engine
    df = load("metabolomics_untargeted")
    raw = engine.rank_findings(engine.diagnose(df, target="responder"), None)
    before = next(f for f in raw if f["id"] == "wide_repeated_measures")
    assert before["fix_kind"] == "melt_repeated"

    after = P.reframe(raw, [P.METABOLOMICS], df)
    seen = next(f for f in after if f["id"] == "wide_repeated_measures")
    assert seen["fix_kind"] == "none", "the offer is withdrawn, not the finding"
    assert seen["severity"] == "info"
    assert P.METABOLOMICS in seen["reframed_by"]
    assert "different analytes" in seen["reframe_note"]

    # The input is not mutated: two callers reading one finding must not see
    # each other's edits.
    assert before["fix_kind"] == "melt_repeated"

    # AND ON THE CONTROL, WITH EVERY PACK, IT IS UNTOUCHED.
    control = load("clinic_visits")
    raw_control = engine.rank_findings(engine.diagnose(control, target="outcome"), None)
    kept = next(f for f in P.reframe(raw_control, REAL_PACKS, control)
                if f["id"] == "wide_repeated_measures")
    assert kept["fix_kind"] == "melt_repeated", (
        "a pack suppressed the finding on the one fixture where it is correct")
    assert "reframe_note" not in kept


def test_the_genomics_lens_rereads_the_sentinel_criticals_as_counts():
    from turbotab import engine
    df = load("genomics_expression")
    raw = engine.rank_findings(engine.diagnose(df, target="condition"), None)
    sentinels = [f for f in raw if f["id"].startswith("sentinel_missing__gene_")]
    assert len(sentinels) >= 5
    assert all(f["severity"] == "critical" for f in sentinels)

    after = P.reframe(raw, [P.GENOMICS], df)
    reread = [f for f in after if f["id"].startswith("sentinel_missing__gene_")]
    assert all(f["severity"] == "info" for f in reread)
    assert all(f["fix_kind"] == "none" for f in reread)
    assert all("count" in f["reframe_note"] for f in reread)


# ── the lens answer itself ───────────────────────────────────────────────────

def test_not_sure_is_a_recorded_answer_and_an_empty_selection_is_not():
    assert P.normalize(["other"]) == ["other"]
    assert "no domain-specific defaults" in P.methods_sentence(["other"])

    with pytest.raises(P.PackError):
        P.normalize([])
    with pytest.raises(P.PackError):
        P.normalize(["other", "dietary"])
    with pytest.raises(P.PackError):
        P.normalize(["astrology"])


def test_the_answer_is_ordered_stably_and_not_by_click_order():
    assert P.normalize(["clinical", "dietary"]) == \
        P.normalize(["dietary", "clinical"]) == ["dietary", "clinical"]


def test_the_methods_sentence_names_every_lens_the_reader_must_check():
    sentence = P.methods_sentence(["dietary", "clinical"])
    assert "dietary intake" in sentence
    assert "clinical measurements and labs" in sentence
    assert "overridden" in sentence, (
        "a lens the manuscript cannot see is a lens the reader cannot check, "
        "and one it cannot see was overturnable is worse")


def test_no_lens_at_all_leaves_every_finding_exactly_as_the_engine_left_it():
    """The app is fully functional with no lens. A design in which an unlisted
    field degrades the experience has built a tool for five disciplines."""
    from turbotab import engine
    for fixture, (target, _) in FIXTURES.items():
        df = load(fixture)
        raw = engine.rank_findings(engine.diagnose(df, target=target), None)
        assert P.reframe(raw, [], df) == raw
        assert P.reframe(raw, [P.OTHER], df) == raw
        assert P.findings(df, [P.OTHER]) == []


def test_detection_is_a_suggestion_and_a_contradiction_detector_never_the_answer():
    """Constitution §02's demotion, applied one level out."""
    metab = load("metabolomics_untargeted")
    hints = {h["lens"] for h in P.suggest(metab)["hints"]}
    assert P.METABOLOMICS in hints
    assert P.suggest(load("genomics_expression"))["hints"][0]["lens"] == P.GENOMICS
    assert P.SURVEY in {h["lens"] for h in P.suggest(load("survey_instrument"))["hints"]}
    assert P.DIETARY in {h["lens"] for h in P.suggest(load("dietary_recalls"))["hints"]}

    # A suggestion is never an answer: nothing is recorded by suggesting.
    assert P.findings(metab, []) == []

    # The contradiction detector: 400 columns across 80 rows described as a
    # clinical panel is a disagreement worth raising. Escalate on evidence that
    # a reading is wrong, never on the size of the consequence.
    clash = P.contradiction(metab, [P.CLINICAL])
    assert clash is not None
    assert clash["kind"] == "stated_lens_but_shape_is_an_assay"
    assert clash["n_numeric"] > 300

    # And it stays quiet when the answer fits the shape.
    assert P.contradiction(metab, [P.METABOLOMICS]) is None
    assert P.contradiction(load("clinic_visits"), [P.CLINICAL]) is None


# ── the priors layer, wired · GUIDED-024 / 025 / 027 ─────────────────────────

def test_a_derived_column_prior_turns_a_mechanism_question_into_a_stated_fact():
    """`GUIDED-024`: metabolomics, the flagship pack, changed zero questions.

    `DOMAIN_PACKS.md` §02's table is explicit about the intended result — with a
    metabolomics lens, *"how should missing values be filled?"* leaves the
    question list. On `metabolomics_untargeted.csv` that is 308 columns with
    blanks, so the layer being inert was not a rounding error.

    Discharges `lockbox-01`: the lens comes first, and this is what it buys.
    """
    df = load("metabolomics_untargeted")
    with_blanks = [str(c) for c in df.columns if df[c].isna().any()]
    assert len(with_blanks) > 300

    def plan_for(lens):
        pri = {c: P.prior_for_column(lens, "missingness_direction", c, df)
               for c in with_blanks} if lens else {}
        plan = router.plan([], target="responder", detection=None,
                           step="preprocess", deferred={},
                           answered=["choose_models", "choose_preparation_mode"],
                           recommendations=[], signals=None,
                           missing_columns=with_blanks, missingness_priors=pri)
        router.audit(plan)
        return ([q for q in plan if q.mode == "push" and q.status == "asked"],
                [q for q in plan if q.status == "skipped"])

    asked, skipped = plan_for([])
    assert len(asked) == len(with_blanks) and not skipped

    asked, skipped = plan_for([P.METABOLOMICS])
    assert len(skipped) > 300, "the derived prior did not reach the question"
    assert len(asked) < 5

    # Every skip carries the PACK'S OWN reason and the reopen affordance —
    # `audit()` refuses a skip without one, and a skip whose reason was written
    # at the router would be a second copy of the pack's sentence.
    reason = P.priors([P.METABOLOMICS], "missingness_direction", df)[0]["reason"]
    assert all(q.skip_reason and reason in q.skip_reason for q in skipped)
    assert all(q.confidence == "high" for q in skipped)


def test_an_offered_prior_informs_the_question_and_never_removes_it():
    """Only `derived` reaches the skip rule. The clinical pack's prior is
    `offered` — it supplies the reading, not the answer."""
    df = load("metabolomics_untargeted")
    with_blanks = [str(c) for c in df.columns if df[c].isna().any()]
    pri = {c: P.prior_for_column([P.CLINICAL], "missingness_direction", c, df)
           for c in with_blanks}
    plan = router.plan([], target="responder", detection=None, step="preprocess",
                       deferred={}, answered=["choose_models",
                                              "choose_preparation_mode"],
                       recommendations=[], signals=None,
                       missing_columns=with_blanks, missingness_priors=pri)
    router.audit(plan)
    assert not [q for q in plan if q.status == "skipped"]


def test_two_packs_disagreeing_about_a_column_leaves_the_question_asked():
    """The reason `priors()` returns a list, made operational.

    `age` and `bmi` on the metabolomics fixture are blank on the QC rows AND are
    clinical reference variables, so both packs have something to say about
    them and they disagree. A skip taken on whichever prior happens to be
    `derived` would resolve that silently, which is the one thing the list
    return exists to prevent.
    """
    df = load("metabolomics_untargeted")
    with_blanks = [str(c) for c in df.columns if df[c].isna().any()]
    both = [P.METABOLOMICS, P.CLINICAL]
    contested = [c for c in with_blanks
                 if len(P.prior_for_column(both, "missingness_direction", c, df)) > 1]
    assert contested, "the fixture no longer produces a contested column"

    pri = {c: P.prior_for_column(both, "missingness_direction", c, df)
           for c in with_blanks}
    plan = router.plan([], target="responder", detection=None, step="preprocess",
                       deferred={}, answered=["choose_models",
                                              "choose_preparation_mode"],
                       recommendations=[], signals=None,
                       missing_columns=with_blanks, missingness_priors=pri)
    router.audit(plan)
    asked = {q.key for q in plan if q.status == "asked" and q.kind == "missingness"}
    for col in contested:
        assert f"missingness::{col}" in asked, (
            f"{col} has two packs disagreeing about it and the question was "
            f"skipped anyway")


def test_a_column_prior_is_withheld_where_its_detector_did_not_fire():
    """`GUIDED-027`. The scoping information already existed; it simply was not
    used to scope the prior it justifies."""
    df = load("dietary_recalls")
    # No left-censoring here, so the metabolomics prior has nothing to be about
    # and is withheld rather than applied to the table.
    assert P.priors([P.METABOLOMICS], "missingness_direction", df) == []
    # Unscoped, the declaration is still readable — a caller reading the
    # catalogue rather than a dataset wants that.
    assert len(P.priors([P.METABOLOMICS], "missingness_direction")) == 1

    metab = load("metabolomics_untargeted")
    scoped = P.priors([P.METABOLOMICS], "missingness_direction", metab)[0]
    assert scoped["scope"] == P.COLUMNS
    assert len(scoped["columns"]) > 300
    # And it does NOT claim the columns its detector never named.
    assert "run_order" not in scoped["columns"]


def test_a_pack_variant_preference_lives_in_the_recipe_table(monkeypatch):
    """`GUIDED-025`: the recipe table is canonical, and the packs now reach it.

    Two mechanisms that never met — `register_operation` / `register_default`
    with specificity resolution and origin tracking, proven by the fake-pack
    test and called by no real pack, beside a prior dict nothing resolved.
    """
    from turbotab import recipes as R
    state = R.snapshot()
    P.unload_for_test()
    try:
        assert R.resolve("ridge", "scale").variant == "standard"
        assert R.resolve("ridge", "power").variant == "none"

        P.load([P.METABOLOMICS])

        scaled = R.resolve("ridge", "scale")
        assert scaled.variant == "pareto"
        assert scaled.origin == "metabolomics_pack"
        assert "Pareto" in scaled.reason
        # Stated AS convention, never in the user's name.
        assert "The field convention here" in scaled.reason
        assert "you should" not in scaled.reason.lower()

        powered = R.resolve("ridge", "power")
        assert powered.variant == "log1p"
        assert powered.origin == "metabolomics_pack"

        # A tree still gets no scaling: the pack overrode the capability rule,
        # not the `*` rule, so specificity still decides.
        assert R.resolve("extratrees_reg", "scale").variant == "none"
    finally:
        R.restore(state)
        P.unload_for_test()


def test_loading_a_pack_twice_registers_it_once():
    from turbotab import recipes as R
    state = R.snapshot()
    P.unload_for_test()
    try:
        assert P.load([P.METABOLOMICS]) == [P.METABOLOMICS]
        assert P.load([P.METABOLOMICS]) == []
        assert len(P.recipe_origins([P.METABOLOMICS])) == 2
    finally:
        R.restore(state)
        P.unload_for_test()


# ── reframing as an extension point, and the gate that gates · GUIDED-028 ────

def test_a_pack_can_contribute_a_reframing_without_editing_reframe():
    """The extension point, proved the way this project proves extension points.

    `reframe()` covered two hardcoded finding ids, so a pack could not
    contribute a reframing at all. The fake pack here is the same shape as the
    one in `test_the_recipe_table_is_a_table.py`, and for the same reason: a
    mechanism is not extensible until something outside it extends it.
    """
    import dataclasses
    df = load("clinic_visits")
    raw = [f for f in _diagnose(df, "outcome") if f["id"] == "constant_columns"]
    assert raw, "the control fixture no longer has a constant column"
    assert raw[0]["fix_kind"] == "drop_columns"

    fake = dataclasses.replace(
        P.PACKS[P.CLINICAL],
        reframings=(P.Reframing(
            matches=lambda f, d: f.get("id") == "constant_columns",
            title=lambda f, d: "A constant column is expected here",
            note=lambda f, d: "The fake pack read this one differently."),))
    original = P.PACKS[P.CLINICAL]
    P.PACKS[P.CLINICAL] = fake
    try:
        out = P.reframe(_diagnose(df, "outcome"), [P.CLINICAL], df)
    finally:
        P.PACKS[P.CLINICAL] = original

    seen = next(f for f in out if f["id"] == "constant_columns")
    assert seen["title"] == "A constant column is expected here"
    assert seen["reframed_by"] == [P.CLINICAL]
    assert seen["fix_kind"] == "none", "the offer is withdrawn"
    assert seen["severity"] == "info"

    # And the pack is gone when the test is. A reframing that leaked between
    # tests would make every later assertion about the control meaningless.
    after = P.reframe(_diagnose(df, "outcome"), [P.CLINICAL], df)
    assert next(f for f in after
                if f["id"] == "constant_columns")["fix_kind"] == "drop_columns"


def _diagnose(df, target):
    from turbotab import engine
    return engine.rank_findings(engine.diagnose(df, target=target), None)


def test_the_compositional_gate_reaches_the_figure_it_says_it_gates():
    """`params["gates"] = "collinearity_figure"` was a field a test asserted the
    existence of and nothing read.

    Correlation between parts of a whole is negatively biased BY CONSTRUCTION,
    so a matrix drawn over them is not a figure with a caveat — it is a figure
    that cannot be read. The gate annotates rather than withholds, for the same
    reason reframing annotates rather than deletes: the other columns'
    correlations are real.
    """
    from fastapi.testclient import TestClient
    from turbotab import api

    client = TestClient(api.app)
    fixture = DATA / "dietary_recalls.csv"
    with open(fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": ("dietary_recalls.csv", fh, "text/csv")}).json()["id"]

    ungated = client.get(f"/project/{pid}/evidence/correlations").json()
    assert not ungated.get("gated")

    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["dietary"]}})
    gated = client.get(f"/project/{pid}/evidence/correlations").json()
    assert gated.get("gated") is True
    assert len(gated["gates"]) == 1
    assert set(gated["gates"][0]["columns"]) == {
        "protein_pct_kcal", "fat_pct_kcal",
        "carbohydrate_pct_kcal", "alcohol_pct_kcal"}
    assert gated["gates"][0]["draw"] == "log_ratio"
    assert "parts of a whole" in gated["gates"][0]["reason"]

    # The matrix itself is still served. A user who asked to see it gets it.
    assert ungated.keys() <= gated.keys()


def test_the_gate_is_on_the_chip_before_the_figure_is_opened():
    """A caveat discovered after looking is a caveat applied to a reading the
    user has already taken."""
    from fastapi.testclient import TestClient
    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("dietary_recalls.csv", fh, "text/csv")}).json()["id"]
    for kind, payload in [("set_lens", {"lens": ["dietary"]}),
                          ("set_target", {"column": "hba1c"})]:
        client.post(f"/project/{pid}/decision",
                    json={"kind": kind, "payload": payload})

    iv = client.get(f"/project/{pid}/interview?step=explore").json()
    chips = {q["key"]: q for q in iv["questions"] if q["mode"] == "pull"}
    if "look::r8_collinearity" not in chips:
        pytest.skip("the correlation affordance is not offered on this record")
    chip = chips["look::r8_collinearity"]
    assert chip.get("gated") is True
    assert chip["gate"]["packs"] == ["dietary"]
    assert chip["gate"]["draw"] == "log_ratio"


def test_no_lens_gates_nothing():
    """The app is fully functional with no lens, and a gate is a pack's claim."""
    from fastapi.testclient import TestClient
    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "dietary_recalls.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("dietary_recalls.csv", fh, "text/csv")}).json()["id"]
    assert not client.get(
        f"/project/{pid}/evidence/correlations").json().get("gated")


# ── the lens reaches the diagnosis the app presents ──────────────────────────

def test_the_lens_reaches_every_finding_list_the_app_presents():
    """`OPENING_SEQUENCE.md` orders the lens BEFORE the diagnosis.

    The detectors in `ml/import_doctor.py` take a frame and nothing else, are
    field-blind by construction, and are frozen. So the lens is a parameter of
    `rank_findings` — the one function that produces the finding list the app
    presents — rather than of the detector pass underneath it.

    **And it must be, rather than acting at generation.** Reframing annotates
    and never deletes: a user who reads *"these are different analytes"* and
    still wants to reshape can. `apply` and `preview` re-run `diagnose()` and
    need the real `fix_kind` to execute the repair, so a lens that erased the
    reading at generation would take that route away and turn the annotation
    into a deletion by another name.

    This is the executable half. `_recompute` is the only path from a frame to
    the findings a user sees, and it must pass the lens.

    Discharges `lockbox-01`: the lens comes first.
    """
    import inspect
    from turbotab import api

    source = inspect.getsource(api._recompute)
    assert "rank_findings" in source
    assert "lens=project.lens" in source, (
        "`_recompute` builds the presented finding list without the lens. The "
        "sequence says the lens comes before the diagnosis; this is where that "
        "is true or is not.")

    # And nothing else in the API builds one. `diagnose()` is still called
    # directly by `apply` and `preview`, and those want the RAW reading.
    api_source = inspect.getsource(api)
    presented = [line for line in api_source.splitlines()
                 if "rank_findings(" in line and "def " not in line]
    assert len(presented) == 1, presented


def test_the_raw_reading_survives_for_the_user_who_wants_it_anyway():
    """The reason the lens acts on presentation rather than on generation.

    A reframed finding is annotated, not withdrawn. `apply` re-diagnoses and
    gets the real `fix_kind`, so a user who reads the metabolomics reading and
    decides to reshape the table anyway still can — which is what makes
    "annotates, never deletes" true of the whole app rather than of one list.
    """
    from turbotab import engine
    df = load("metabolomics_untargeted")

    raw = engine.diagnose(df, target="responder")
    wide = next(f for f in raw if f.id == "wide_repeated_measures")
    assert wide.fix_kind == "melt_repeated", (
        "the raw reading lost its repair, so a user who overrules the lens has "
        "nowhere to go")

    presented = engine.rank_findings(raw, None, lens=[P.METABOLOMICS], df=df)
    shown = next(f for f in presented if f["id"] == "wide_repeated_measures")
    assert shown["fix_kind"] == "none"
    assert P.METABOLOMICS in shown["reframed_by"]
