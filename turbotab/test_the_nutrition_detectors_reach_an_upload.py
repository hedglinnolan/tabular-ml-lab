"""`GUIDED-058` — the nutrition module's detectors reach the live pack path.

`packs.findings(df, lens)` was already wired: `engine.py:710` and
`project.py:815` both call it. `turbotab/nutrition.py` was imported by its own
tests and by nothing else, so four detectors and a refusal sat one registration
away from a user and nobody could reach them.

## The contract decision, which is a deliverable beside the wire

`Pack.detectors` is `Tuple[Callable[[pd.DataFrame], Optional[Dict]], ...]`.
`atwater_finding` matched it as written. `design_findings` returned a `List`
and did not. **Split, not widened.** Widening the contract to admit a list is
one line in `packs.findings` and costs the type its meaning — every caller then
handles two shapes, `prior_columns`'s `f["id"] == detector` lookup gets
ambiguous, and the widening would rest on one example, which is the mistake
`GUIDED-056` refused to make with the figure `tier` enum one layer down.

Splitting cost one thing and revealed that it was not a cost: the early
`return out` that suppressed the lonely-PSU check after a partial-design
finding turned out to be expressible as a guard, because `lonely_psu` needs
both strata and PSU and the partial case is exactly their absence. Control flow
that is expressible as preconditions on each part is evidence the parts were
independent.

## What the wire found that no unit test could

`design_findings` fired `pack::dietary::partial_design` on
`clinic_visits.csv` — *"There is a survey weight in this table and no strata or
PSU column"* — about a column holding `107 kg`. **A body weight called a
sampling weight is the dietary pack asserting something false on a clinical
table, authoritatively**, which is guard #2's whole subject. It was invisible
because the module's own tests used the NHANES names.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import nutrition as N                                   # noqa: E402
from turbotab import packs as P                                       # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ── the wire ────────────────────────────────────────────────────────────────

def test_the_detectors_are_on_the_live_path_and_not_only_importable():
    """Through `packs.findings`, which `engine` and `project` already call."""
    ids = {f["id"] for f in P.findings(load("dietary_recalls"), [P.DIETARY])}
    assert "pack::dietary::atwater" in ids

    ids = {f["id"] for f in P.findings(load("nhanes_dietary"), [P.DIETARY])}
    assert {"pack::dietary::survey_weights",
            "pack::dietary::lonely_psu"} <= ids, sorted(ids)

    ids = {f["id"] for f in P.findings(load("nhanes_partial_design"),
                                       [P.DIETARY])}
    assert {"pack::dietary::survey_weights",
            "pack::dietary::partial_design"} <= ids, sorted(ids)


def test_a_project_sees_them_through_its_own_accessor():
    """`AnalysisProject.pack_findings` is what the interview reads."""
    from turbotab.project import AnalysisProject

    project = AnalysisProject.from_dataframe(load("nhanes_dietary"), "nhanes")
    project.lens = [P.DIETARY]
    ids = {f["id"] for f in project.pack_findings()}
    assert "pack::dietary::lonely_psu" in ids
    project.lens = [P.OTHER]
    assert not [f for f in project.pack_findings()
                if f["id"].startswith("pack::dietary::")]


def test_the_atwater_finding_has_one_id_rather_than_one_per_verdict():
    """A varying id cannot be bound to anything — `LooksFor` names an id and
    `prior_columns` looks a detector up by it. The verdict is a parameter,
    which it already was, and the title is where a reader meets it."""
    seen = set()
    for name in ("dietary_recalls", "nhanes_dietary"):
        finding = N.atwater_finding(load(name))
        if finding:
            seen.add(finding["id"])
            assert finding["params"]["verdict"], "the verdict was dropped"
    assert seen == {"pack::dietary::atwater"}, seen


# ── guard #1 · a pack may not invent a card type ────────────────────────────

def test_a_reporting_pack_still_cannot_add_a_question(monkeypatch):
    """`fix_kind="none"` is what makes guard #2 structural rather than
    aspirational: `router._is_repairable` reads it as the engine refusing to
    guess, so a reporting finding is a report and not a fork. Every one of the
    four new findings carries it."""
    from ml import router

    produced = []
    for name in ("dietary_recalls", "nhanes_dietary", "nhanes_partial_design"):
        produced += P.findings(load(name), [P.DIETARY])
    assert len(produced) >= 5, "the fixtures stopped producing findings"
    for finding in produced:
        assert finding["fix_kind"] == "none", finding["id"]
        assert not finding["fix_label"], finding["id"]
        assert not router._is_repairable(finding), finding["id"]


# ── guard #2 · a pack must not fire on non-matching data ────────────────────

GENERIC = ("clinic_visits", "clinical_longitudinal", "survey_instrument",
           "genomics_expression", "metabolomics_untargeted",
           "longitudinal_visits", "leaky_sepsis", "wide_assay")


@pytest.mark.parametrize("fixture", GENERIC)
def test_the_nutrition_detectors_are_silent_on_every_table_they_do_not_describe(
        fixture):
    """The four run directly, so a silence produced by the router rather than
    by the detector cannot be mistaken for the detector being right."""
    df = load(fixture)
    fired = [f["id"] for f in
             [N.atwater_finding(df), N.survey_weights_finding(df),
              N.partial_design_finding(df), N.lonely_psu_finding(df)] if f]
    assert not fired, (
        f"the nutrition detectors fired {fired} on {fixture}, which is not a "
        f"dietary table. A pack that fires on the wrong data asserts something "
        f"false authoritatively.")


def test_a_body_weight_is_not_a_sampling_weight():
    """The defect the wire found, as its own assertion.

    `clinic_visits.csv` has a column called `weight` holding `107 kg`. The
    research's §01 generic pattern `weight|wt|pweight` matches it, and taking
    that as sufficient produced a `warning` telling a clinician their body-mass
    column left their standard errors too narrow.
    """
    df = load("clinic_visits")
    design = N.survey_design(df)
    assert design["any_weight"] is False
    assert design["generic_weights"] == []
    assert "weight" in dict(design["rejected_weights"])
    assert N.partial_design_finding(df) is None


# ── GUIDED-070 · the recoverable half of that fix's cost ────────────────────
#
# The fix was right and it took a real capability with it: a genuine sampling
# weight called `weight` was missed even in a table carrying `SDMVSTRA` and
# `SDMVPSU`. That loss lived in a report and a comment and in no ledger row.
# Most of it is recoverable, because the corroboration the name lacks is in the
# columns beside it — nobody names a column `SDMVSTRA` by accident.


def _survey_frame(n: int = 60, seed: int = 3, weight_name: str = "weight",
                  **columns) -> pd.DataFrame:
    """A generically-named sampling weight beside real design variables."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        weight_name: rng.gamma(4, 6000, n),
        "DR1TKCAL": rng.normal(2000, 300, n),
        "SDMVSTRA": rng.integers(100, 105, n),
        "SDMVPSU": rng.integers(1, 3, n),
    })
    for key, value in columns.items():
        frame[key] = value
    return frame


def test_a_generic_weight_beside_recognized_design_variables_counts_again():
    """`GUIDED-070`. The recovered case, asserted end to end: the design
    finding this table supports comes back."""
    frame = _survey_frame()
    frame.loc[frame.index[:6], "SDMVSTRA"] = 999
    frame.loc[frame.index[:6], "SDMVPSU"] = 1      # a stratum with one PSU

    design = N.survey_design(frame)
    assert design["generic_weights"] == ["weight"]
    assert design["any_weight"] is True
    assert design["rejected_weights"] == []
    assert set(design["recognized_design"]) == {"SDMVSTRA", "SDMVPSU"}

    ids = {f["id"] for f in N.design_findings(frame)}
    assert "pack::dietary::lonely_psu" in ids, sorted(ids)


def test_the_same_weight_with_a_strata_column_and_no_psu_is_partial_again():
    frame = _survey_frame().drop(columns=["SDMVPSU"])
    finding = N.partial_design_finding(frame)
    assert finding is not None
    assert finding["params"]["missing"] == ["PSU"]
    assert finding["affected_columns"] == ["weight"]


def test_the_corroboration_is_exact_names_and_never_the_generic_fallbacks():
    """`GUIDED-069`'s unfixed half must not become this one's second signal.

    A table with `weight` and `cluster` would otherwise be a survey design
    assembled from two guesses — the original defect arriving through the back
    door, with one more step in it.
    """
    frame = _survey_frame().drop(columns=["SDMVSTRA", "SDMVPSU"])
    frame["strata"] = 1
    frame["cluster"] = 2
    design = N.survey_design(frame)
    assert design["recognized_design"] == []
    assert design["generic_weights"] == []
    assert design["any_weight"] is False
    assert N.design_findings(frame) == []


def test_a_body_weight_beside_real_design_variables_is_still_rejected():
    """The third signal, and the reason it is a reading rather than a lookup.

    An NHANES-derived analysis file can carry `SDMVSTRA` and a body-mass column
    somebody renamed `weight`. Design variables say the TABLE is a complex
    survey; they say nothing about which column is the weight. The values do.
    """
    rng = np.random.default_rng(11)
    frame = _survey_frame()
    frame["weight"] = rng.normal(78, 12, len(frame))       # kilograms
    design = N.survey_design(frame)
    assert design["generic_weights"] == []
    assert design["any_weight"] is False
    reason = dict(design["rejected_weights"])["weight"]
    assert "recognizes it as weight" in reason
    assert "plausible for that measurement" in reason


def test_what_is_still_lost_is_smaller_and_is_stated():
    """A generically-named weight in a table with no exactly-named design
    variable. The app says nothing, and the rejection records why — silence
    over a false assertion, with the absence inspectable rather than absent."""
    df = pd.DataFrame({"pweight": [1000.0, 2500.0, 1800.0] * 10,
                       "DR1TKCAL": [2000, 2200, 1900] * 10})
    design = N.survey_design(df)
    assert design["generic_weights"] == []
    assert design["any_weight"] is False
    assert N.design_findings(df) == []
    assert "no design variable this pack recognizes by its exact name" in \
        dict(design["rejected_weights"])["pweight"]


def test_the_exact_names_are_the_ones_the_research_lists():
    """§01: *"Flag `WTINT2YR`, `WTMEC2YR`, `WTDRD1`, `WTDR2D`, `SDMVPSU`,
    `SDMVSTRA`, `SDDSRVYR`."* The corroboration set is that list and not a
    convenience subset of it."""
    assert set(N.EXACT_DESIGN_NAMES) == {
        "WTDRD1", "WTDR2D", "WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU",
        "SDDSRVYR"}


def test_every_generic_fixture_gains_no_question_from_the_dietary_pack():
    """Guard #2's own metric — questions added, not findings changed — over
    every fixture the dietary lens does not match."""
    from ml import router
    from turbotab import engine

    def questions(df, target, lens):
        ranked = engine.rank_findings(engine.diagnose(df, target=target), None)
        ranked = P.reframe(ranked, lens, df) + P.findings(df, lens)
        plan = router.plan(ranked, target=target, detection=None, step="data",
                           deferred={}, answered=["state_lens"],
                           recommendations=[], signals=None,
                           missing_columns=[])
        router.audit(plan)
        return sum(1 for q in plan if q.mode == "push" and q.status == "asked")

    for fixture, target in (("clinic_visits", "outcome"),
                            ("survey_instrument", "sought_support"),
                            ("genomics_expression", "condition"),
                            ("metabolomics_untargeted", "responder")):
        df = load(fixture)
        assert questions(df, target, [P.DIETARY]) == questions(df, target, []), (
            f"the dietary pack added a question to {fixture}")


# ── guard #3 · every pre-selected default states its reason ─────────────────

def test_every_finding_the_pack_now_emits_states_its_reason_and_its_badge():
    """Guard #3 reaches findings and not only priors. `marker` governs the
    treatment — `derived` is pre-selected with its reason shown — so a finding
    without a reason raises confidence without earning it."""
    produced = []
    for name in ("dietary_recalls", "nhanes_dietary", "nhanes_partial_design"):
        produced += P.findings(load(name), [P.DIETARY])
    for finding in produced:
        assert finding["marker"] in ("derived", "convention", "offered")
        assert len(finding["why_it_matters"]) > 60, finding["id"]
        assert len(finding["detail"]) > 60, finding["id"]
        badge = finding["evidence"]
        assert badge["evidence_status"] in P.EVIDENCE_STATUSES
        if badge["evidence_status"] == P.DISPUTED:
            assert finding["marker"] == "offered", finding["id"]


def test_the_dietary_priors_still_state_theirs():
    """Unchanged by the wire, asserted because the wire is where a prior's
    scope would quietly break: two of the three are column-scoped and read
    their columns from a detector that now runs in a different place."""
    for prior in P.PACKS[P.DIETARY].priors:
        assert len(prior.reason) > 40, prior.question
        assert prior.evidence is not None
        if prior.scope == P.COLUMNS:
            assert prior.detector, prior.question


def test_a_column_scoped_prior_still_finds_its_detectors_columns():
    """`prior_columns` matches `f["id"] == detector` over `packs.findings`,
    which now has four more detectors in it. `None` — not `[]` — where the
    detector did not fire is `GUIDED-027`'s whole distinction."""
    df = load("dietary_recalls")
    columns = P.prior_columns(P.DIETARY, "pack::dietary::compositional", df)
    assert columns and all(c in df.columns for c in columns)
    assert P.prior_columns(P.DIETARY, "pack::dietary::compositional",
                           load("clinic_visits")) is None
