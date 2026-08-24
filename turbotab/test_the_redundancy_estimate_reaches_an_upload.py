"""L51 — `METABOLOMICS_PACK.md` §01's redundancy estimate, half built on purpose.

> Untargeted features are not independent. Ionization produces adducts
> (`[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[M-H]-`, `[M+HCOO]-`),
> isotopologues (Δm/z 1.00336), dimers, and in-source fragments. Cluster features
> by near-identical RT (±0.05–0.1 min) **and** high inter-feature correlation
> (r > 0.9), and report an estimated *effective* feature count. If 5,000 features
> collapse to ~1,200 clusters, the user's "5,000 metabolites" claim is wrong by
> ~4×.

## What was checked before anything was built, and what it changed

`metabolomics_untargeted.csv` is 80 × 400 with columns `mz_0001`…`mz_0392`.
**`mz_` is an ordinal index and not a mass** — `make_fixtures.py` writes
`frame[f"mz_{j + 1:04d}"]`, so the digits are `j + 1` and nothing else. There is
no m/z anywhere in the file, no retention time anywhere in the file, and no other
table in this repository carries either.

So of §01's two clustering criteria, **the correlation half is computable and the
retention-time half has no input at all** — and the Δm/z 1.00336 isotopologue
test that the same paragraph names is not computable either, for the same reason:
it needs the mass this column name does not carry. The correlation half is built,
and `test_the_retention_time_criterion_is_not_built` is the other half as a
failing test naming exactly what the data would have to supply. **No RT is
faked**, because a fabricated column would make a half-built diagnostic look
whole, which is `AGENT_ONBOARD.md` §07 trap #3.

## `GUIDED-097` — two fixtures of different shape

`metabolomics_redundant.csv` carries 100 compounds under 404 feature names.
`metabolomics_untargeted.csv` carries 392 features drawn independently, whose
largest off-diagonal correlation anywhere in the matrix is **0.87** — so the
detector must be silent on it, and that silence is asserted here with the
measured number beside it rather than as an absence nobody checked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Redundant and independent, both real files.
REDUNDANCY_FIXTURES = {
    "100 compounds under 404 names": ("metabolomics_redundant", "responder"),
    "392 independent features": ("metabolomics_untargeted", "responder"),
}

#: NOT COVERED, said out loud.
#:
#: A TABLE CARRYING m/z AND RT. The whole reason the retention-time half is a
#: failing test rather than a feature. Nothing in this repository has either.
#:
#: THE ISOTOPOLOGUE MASS TEST. §01 names Δm/z 1.00336 in the same paragraph, and
#: it needs the mass for the same reason the RT window needs the retention time.
#: Covered by the same failing test and by nothing else.
#:
#: THE ION-MODE SPLIT. §01's positive/negative merge is a separate diagnostic and
#: neither fixture carries an ion-mode column.
#:
#: IN-SOURCE FRAGMENTS THAT ARE NOT PROPORTIONAL TO THE PARENT. Every product in
#: the sibling fixture is proportional by construction, which is the easy case: a
#: fragment whose yield varies with matrix would correlate less and would split.
SHAPES_NOT_COVERED = [
    "a table carrying m/z and retention time — nothing in this repository has "
    "either, which is what the xfail below is about",
    "the Δm/z 1.00336 isotopologue mass test, which needs the same missing m/z",
    "the positive/negative ion-mode split, which neither fixture carries",
    "in-source fragments whose yield is not proportional to the parent",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ═══════════ 1 · IT FIRES, AND ITS ANSWER MATCHES THE FIXTURE'S TRUTH ═══════

def test_the_estimate_recovers_the_number_of_compounds_the_fixture_was_built_from():
    """**The assertion that is not the detector marking its own work.**

    `metabolomics_redundant.csv` was built from exactly 100 parent compounds,
    each given 2–4 ionization products, and the generator records that in its
    companion. So there is a ground truth here that the detector never sees, and
    the check is against it: **100 multi-member groups**, one per compound, with
    sizes in 3–5.

    That is what separates this from trap #3. A test asserting only that the
    effective count is smaller than the column count would pass on any
    correlated matrix and would prove that correlation exists.
    """
    finding = P._redundancy(load("metabolomics_redundant"))
    assert finding is not None
    params = finding["params"]

    assert params["n_groups"] == 100, (
        "one group per compound the fixture was built from, and no more")
    sizes = sorted(len(g) for g in params["groups"])
    assert (sizes[0], sizes[-1]) == (3, 5), sizes
    assert sum(sizes) == 403

    assert params["n_columns"] == 408
    assert params["effective_features"] == 105
    assert params["overstatement_factor"] == 3.89
    assert params["r_threshold"] == 0.9


def test_the_five_columns_it_does_not_group_are_each_accounted_for():
    """105 rather than 100, and every one of the five is a real reason.

    Four are not features at all — `run_order`, `age`, `bmi` and `responder` sit
    in the numeric block, which is what the finding means by *numeric columns*
    rather than *metabolites*. The fifth is `mz_0015`, a faint product observed
    in **5 of 80 samples**: no pair involving it clears the 20-sample overlap
    floor, so no correlation is computable and it is counted as independent.

    That fifth one is the reason this finding does not claim to be a one-sided
    bound. The retention-time criterion could only raise the effective count;
    this could only lower it.
    """
    df = load("metabolomics_redundant")
    finding = P._redundancy(df)
    grouped = {c for group in finding["params"]["groups"] for c in group}
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    singles = sorted(set(numeric) - grouped)
    assert singles == ["age", "bmi", "mz_0015", "responder", "run_order"]

    assert int(df["mz_0015"].notna().sum()) == 5
    assert finding["params"]["n_columns_below_min_overlap"] == 1
    assert finding["params"]["min_overlapping_samples"] == 20


def test_the_finding_does_not_claim_a_one_sided_bound():
    """Both corrections are named, and they point opposite ways.

    A payload that said `"lower bound"` and a sentence that named two directions
    would be `GUIDED-064`'s defect in the direction that flatters the app: the
    machine-readable form asserting more than the prose beside it. So the payload
    carries what is actually true of each half and no summary word.
    """
    finding = P._redundancy(load("metabolomics_redundant"))
    params = finding["params"]
    assert params["retention_time_column"] is None
    assert params["co_elution_would"] == "split groups, never merge them"
    assert "lower bound" not in str(params)
    assert "upper bound" not in str(params)
    detail = finding["detail"]
    assert "no retention time" in detail
    assert "could only lower it" in detail


def test_the_pack_is_silent_where_the_features_are_independent():
    """`GUIDED-097`'s second arm, with the measurement beside it.

    `metabolomics_untargeted.csv`'s features are drawn independently — the only
    thing they share is the instrument-drift ramp — and the largest off-diagonal
    correlation in the whole 392 × 392 matrix is 0.87. The detector must say
    nothing, and the number is asserted here so that a future change making the
    fixture correlated cannot turn this silence into a passing test about
    nothing.
    """
    df = load("metabolomics_untargeted")
    feats = [c for c in df.columns if c.startswith("mz_")]
    corr = df[feats].corr(min_periods=20).to_numpy()
    np.fill_diagonal(corr, 0.0)
    assert float(np.nanmax(corr)) == pytest.approx(0.871, abs=0.005)
    assert P._redundancy(df) is None


@pytest.mark.parametrize("fixture", ("genomics_expression", "survey_instrument",
                                     "wide_assay", "genomics_microarray"))
def test_the_estimate_does_not_fire_on_another_pack_s_wide_table(fixture):
    """Guard #2. Three of these are wide numeric blocks that clear the assay
    precondition, so the silence is a real discrimination rather than a width
    check declining early."""
    assert P._redundancy(load(fixture)) is None


def test_it_reports_and_never_collapses():
    """`offered`, and the absence of a repair is the content. Merging features
    changes what is analyzed, and which member of a group stands for the
    compound is a question about chemistry that the correlation cannot answer."""
    finding = P._redundancy(load("metabolomics_redundant"))
    assert finding["fix_kind"] == "none"
    assert finding["fix_label"] == ""
    assert finding["marker"] == "offered"


def test_the_thresholds_are_badged_as_the_conventions_they_are():
    """`GUIDED-064`. The finding says two things the field holds differently:
    that untargeted features are not independent, which is SETTLED, and that the
    cut points are r > 0.9 and ±0.05–0.1 min, which §01 states without a citation
    behind either — a convention written down. One badge over both would be the
    machine-readable form coarser than the sentence."""
    finding = P._redundancy(load("metabolomics_redundant"))
    claims = {c["key"]: c for c in finding["evidence"]["claims"]}
    assert claims["not_independent"]["evidence_status"] == "SETTLED"
    assert claims["cut_points"]["evidence_status"] == "CONVENTION"
    assert finding["evidence"]["weakest_status"] == "CONVENTION"
    for claim in claims.values():
        assert claim["source"].startswith(
            "research/METABOLOMICS_PACK.md#Redundancy detection")


# ═══════════ 2 · THE HALF THAT IS NOT BUILT ═══════════

#: The three pairs that ARE one compound each — same retention time to two
#: decimals, which is well inside §01's ±0.05–0.1 min window.
CO_ELUTING = (("M180.0634T3.21", "M202.0453T3.22"),
              ("M244.0932T6.40", "M266.0751T6.44"),
              ("M310.1401T8.07", "M332.1220T8.10"))

#: The three that are two compounds each — correlated just as hard and **four
#: minutes apart**, which no ionization event can produce. Nothing but the
#: retention time separates these from the three above.
FAR_APART = (("M341.1088T1.04", "M355.1245T5.11"),
             ("M412.2010T2.30", "M426.2167T6.35"),
             ("M501.2544T0.88", "M515.2701T4.93"))


def _co_elution_frame() -> pd.DataFrame:
    """Nine compounds, twelve of whose features pair up, and RT in the names.

    The naming is XCMS/CAMERA's `M<mz>T<rt>` convention, which is what an
    untargeted feature table exported from R actually carries — `M180.0634T3.21`
    is m/z 180.0634 at 3.21 minutes. It is used here because it is a **real**
    convention a real file would have, so the failing test below names an input
    somebody could supply rather than one somebody would have to invent.

    Six pairs, correlated at ~0.99 apiece. Three co-elute and are one compound;
    three are four minutes apart and are two. A diagnostic clustering on
    correlation alone cannot tell them apart, and that is what the xfail says.
    """
    rng = np.random.default_rng(3)
    n = 60
    frame = {}
    for parent, product in CO_ELUTING + FAR_APART:
        base = np.exp(rng.normal(10.0, 0.6, size=n))
        frame[parent] = base
        frame[product] = base * float(rng.uniform(0.2, 0.5)) * np.exp(
            rng.normal(0.0, 0.05, size=n))
    # Padding so the block clears the assay precondition and the collapse floor.
    # Uncorrelated with everything and with each other.
    for i in range(36):
        frame[f"M{600 + i}.0000T{2 + i * 0.05:.2f}"] = np.exp(
            rng.normal(9.0, 0.7, size=n))
    return pd.DataFrame(frame)


def test_correlation_alone_cannot_separate_co_elution_from_coincidence():
    """The positive control for the xfail below, and it must PASS.

    Without it, a strict xfail that started passing for the wrong reason — the
    detector declining on this frame, say — would read as the feature having
    been built. This asserts the frame really does carry the ambiguity: all six
    pairs group, and the retention times in the names say only three of them
    should.
    """
    finding = P._redundancy(_co_elution_frame())
    assert finding is not None
    groups = sorted(sorted(g) for g in finding["params"]["groups"])
    assert groups == sorted(sorted(p) for p in CO_ELUTING + FAR_APART)
    assert finding["params"]["n_columns"] == 48
    assert finding["params"]["effective_features"] == 42


@pytest.mark.xfail(strict=True, reason=(
    "THE RETENTION-TIME HALF IS NOT BUILT. `METABOLOMICS_PACK.md` §01 clusters "
    "on near-identical RT (±0.05–0.1 min) AND correlation (r > 0.9); only the "
    "correlation half has an input. WHAT THE DATA WOULD NEED: a retention time "
    "per feature — as a `M<mz>T<rt>` column name (XCMS/CAMERA), as a separate "
    "feature-metadata table, or as an `rt`/`retention_time` row. No table in "
    "this repository carries one, and none is fabricated, so this is the "
    "capability with a failing test rather than a green suite over half a "
    "diagnostic. `M341.1088T1.04` and `M355.1245T5.11` correlate at 0.99 and "
    "elute four minutes apart; they are two compounds and this groups them."))
def test_the_retention_time_criterion_is_not_built():
    finding = P._redundancy(_co_elution_frame())
    assert finding["params"]["retention_time_column"] is not None, (
        "nothing reads a retention time out of this frame")
    groups = sorted(sorted(g) for g in finding["params"]["groups"])
    assert groups == sorted(sorted(p) for p in CO_ELUTING), (
        "the three co-eluting pairs are one compound each and the three "
        "four-minutes-apart pairs are two each; requiring co-elution is what "
        "separates them, and 45 rather than 42 is the effective count")


# ═══════════ 3 · AND IT REACHES A PERSON ═══════════

def test_the_redundancy_estimate_reaches_a_person_and_carries_its_badge():
    """**Trap #1.** The assertions above prove the detector and prove nothing
    about the app. Driven through the real API and then through the page's real
    controller in node, with the headline number read back off the DOM —
    `GUIDED-142` is why the page half is here and not only the API half.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client = TestClient(api.app)
    with open(DATA / "metabolomics_redundant.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("metabolomics_redundant.csv", handle,
                     "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [P.METABOLOMICS]}),
                          ("set_target", {"column": "responder"})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, (kind, ok.text[:300])

    project = client.get(f"/project/{pid}").json()
    served = [f for f in project["findings"] if f["source"] == "pack"]
    reached = {f["id"] for f in served}
    assert "pack::metabolomics::redundancy" in reached, sorted(reached)

    redundancy = next(f for f in served
                      if f["id"] == "pack::metabolomics::redundancy")
    # THE BADGE SURVIVED THE BOUNDARY, at claim granularity. `DRIVE-001`'s class
    # is a status computed on the server and dropped on the wire, and the
    # per-claim statuses are the part a flat serializer loses first.
    assert redundancy["evidence"]["weakest_status"] == "CONVENTION"
    assert {c["key"] for c in redundancy["evidence"]["claims"]} == {
        "not_independent", "cut_points"}

    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }
    out = PH.run(
        "var shut = (__harness.html('profList') || '');\n"
        "__harness.dispatch('click', __harness.target("
        "{'data-stack-more':'1','aria-expanded':'false'}));\n"
        "__emit({shut: shut.slice(0, 90000),"
        " open: ((__harness.html('profList') || '') +"
        "        (__harness.html('profRest') || '')).slice(0, 200000)});",
        routes=routes, search=f"?project={pid}")
    html = out["open"]
    assert out["shut"], "the Explore findings list rendered nothing at all"

    missing = [f["id"] for f in served if f["title"][:28] not in html]
    assert not missing, (
        f"the metabolomics pack computes {missing} and the page never shows "
        f"them, pushed or collapsed")

    # THE NUMBER ITSELF, not just the card. `GUIDED-207`: the server composes
    # the sentence and the page renders it, so the sentence a reader acts on has
    # to arrive intact — "105 independent quantities" is the whole finding, and
    # a card that showed the title with the count stripped would pass a
    # title-only assertion.
    assert "105 independent quantities" in html
