"""`AUDIT-008` — the class beneath `AUDIT-016` and `AUDIT-036`, swept.

`AUDIT-008` names the shape the anti-pattern audit keeps finding: **the core
already holds the correct capability and the path that needs it does not read
it.** At the caption layer the shape is narrower and easier to state — *a
caption describes the figure that was DRAWN* — and it has now produced four
numbered rows:

| row | the caption said | the figure did |
|---|---|---|
| `AUDIT-016` | the curve is a loess estimate with a pointwise 95% band | 10 equal-width bins, no interval key |
| `AUDIT-036` | the loadings and reliability appear below | neither is computed anywhere |
| `AUDIT-008` here | items ordered by hierarchical clustering, with a dendrogram layer | `frame.corr().columns`, no linkage |
| `AUDIT-008` here | rows ordered by domain | the shipped caller fills no domain |

`test_a_caption_describes_the_figure_that_was_drawn.py` holds the first two.
This file holds the class, and the two instances the class found once it was
pointed one surface over from the two rows that were filed — `§08` check 5.

## Why the sweep matches phrases rather than words

Both corrections KEPT the word. §B5.4 asks for hierarchical clustering and
§A4.7 asks for grouping by domain, so a caption that deletes the word leaves a
reader unable to tell that the figure falls short of the standard — the same
argument `AUDIT-016` settled for *loess*. So `_CLAIMS_ORDERING` lists the
**assertive** forms only, and `test_the_detector_does_not_fire_on_the_disclosure`
is the control proving the difference is real rather than intended.

## CLASS

`CAPTION-CLAIM`, extended from *a smoothing method* to *any procedure*: a
caption may name a procedure only where a payload value records that the
procedure was applied. `layers` is included because `FigureSpec.to_dict` puts
it on the wire, so a layer nobody draws is the same assertion one field over.

`GUIDED-045` — every absence assertion here is preceded by the positive
control for the thing being swept.
`GUIDED-097` — two fixtures of different shape per claim; the shapes not
covered are named in `SHAPES_NOT_COVERED`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import figure_specs as F
from turbotab import figures

#: `GUIDED-097`. The forest caption is driven on both target shapes it can be
#: drawn for — a binary outcome, whose coefficients are a RATIO measure on a
#: log axis, and a continuous outcome, whose coefficients are a DIFFERENCE on a
#: linear axis — crossed with the two domain shapes, because the ordering
#: sentence and the axis sentence are composed by the same lambda. The survey
#: matrix takes an item block rather than an outcome, so its two shapes are the
#: block's: a 5-point Likert block and a dichotomized one.
SHAPES_NOT_COVERED = [
    "a multi-level categorical predictor with reference rows — "
    "`n_reference_rows` inserts a sentence ahead of the ordering sentence "
    "and no fixture here supplies a `reference` coefficient",
    "survival / time-to-event — no task type exists, so no forest of hazard "
    "ratios can be built to caption",
    "the four instability figures — they are registered but absent from "
    "`figure_bundle.SOURCES`, so no project reaches their captions and "
    "nothing here constructs them",
]

_ITEMS = [f"item_{i:02d}" for i in range(1, 21)]


def _likert_block() -> pd.DataFrame:
    return pd.read_csv("turbotab/sample_data/survey_instrument.csv")[_ITEMS]


def _binary_block() -> pd.DataFrame:
    block = _likert_block()
    return (block >= block.median()).astype(int)


SURVEY_SHAPES = {"5-point Likert block": _likert_block,
                 "dichotomized block": _binary_block}

#: THE SHIPPED SHAPE, and it is the whole point of the forest half of this
#: file. `figure_bundle._coefficients_for` builds exactly
#: `{"name", "estimate", "low", "high"}` per coefficient — there is no `group`
#: key on the production path, so `grouped_by_domain` is False for every
#: forest plot any project can draw.
_AS_THE_BUNDLE_BUILDS_THEM = [
    {"name": "age", "estimate": 0.41, "low": None, "high": None},
    {"name": "creatinine", "estimate": -0.22, "low": None, "high": None},
    {"name": "systolic_bp", "estimate": 0.07, "low": None, "high": None},
]

#: The same three with a domain filled, which is what §A4.7 asks for and what
#: `forest_payload` supports. Nothing in the app produces this today; it is
#: here so the assertion is about the payload rather than about a constant.
_WITH_A_DOMAIN = [dict(row, group=group) for row, group in
                  zip(_AS_THE_BUNDLE_BUILDS_THEM,
                      ("demographics", "labs", "vitals"))]

FOREST_SHAPES = {
    "binary outcome, ratio measure, no domain": (
        _AS_THE_BUNDLE_BUILDS_THEM, True),
    "continuous outcome, difference measure, no domain": (
        _AS_THE_BUNDLE_BUILDS_THEM, False),
    "binary outcome, ratio measure, domain supplied": (_WITH_A_DOMAIN, True),
    "continuous outcome, difference measure, domain supplied": (
        _WITH_A_DOMAIN, False),
}


def _forest(shape: str):
    coefficients, ratio = FOREST_SHAPES[shape]
    payload = F.forest_payload(coefficients, ratio_measure=ratio)
    return payload, figures.REGISTRY["forest"].caption(payload)


def _calibration_binary(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-x))
    return (rng.random(n) < p).astype(int), p


def _calibration_multiclass(n=400, seed=11):
    """Three classes reduced the way `calibration_classification` reduces them
    — it takes the last column of a k-column matrix when k != 2."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p3 = np.column_stack([1.0 / (1.0 + np.exp(-(x - 1.0))),
                          np.full(n, 0.3),
                          1.0 / (1.0 + np.exp(-(x + 0.5)))])
    p3 = p3 / p3.sum(axis=1, keepdims=True)
    return (rng.random(n) < p3[:, -1]).astype(int), p3


CALIBRATION_SHAPES = {"binary outcome": _calibration_binary,
                      "multiclass reduced one-vs-rest": _calibration_multiclass}

#: `AUDIT-016`'s own sentence, as the assertive forms it took. Kept here beside
#: the ordering claims because the two rows are one class and a detector that
#: lives in only one file is a class that comes back under a fifth number.
_CLAIMS_SMOOTHING = (
    "the flexible (loess) estimate", "the loess estimate", "the loess curve",
    "the spline curve", "the smoothed curve", "is the flexible",
    "fitted by loess", "estimated by loess", "loess-smoothed",
)


@pytest.mark.parametrize("shape", sorted(CALIBRATION_SHAPES))
def test_the_calibration_caption_does_not_assert_the_curve_is_smoothed(shape):
    """`AUDIT-016`, asserted against the FALSE claim rather than against the
    correction that replaced it.

    The sibling test in `test_a_caption_describes_the_figure_that_was_drawn.py`
    checks the bin count first, so a revert makes it red for the *absence of
    the new sentence* before it ever reaches the *presence of the old one*.
    This one checks only the presence, so the red names the sentence the row
    was filed for.
    """
    y, p = CALIBRATION_SHAPES[shape]()
    payload = F.calibration_render(y, p)
    caption = figures.REGISTRY["calibration"].caption(payload).lower()

    assert payload["curve"]["predicted"], (
        "the fixture produced no curve and proves nothing")
    claimed = [phrase for phrase in _CLAIMS_SMOOTHING if phrase in caption]
    assert not claimed, (
        f"{shape}: the caption asserts the curve is a smoothed estimate "
        f"{claimed}; it is {len(payload['curve']['predicted'])} equal-width "
        f"bins and the payload carries no band, interval or CI key")


# ═══════════ THE ITEM-CORRELATION ORDERING ═══════════

@pytest.mark.parametrize("shape", sorted(SURVEY_SHAPES))
def test_the_correlation_caption_does_not_claim_an_ordering_nothing_computes(shape):
    """`AUDIT-008`. Nothing here computes a linkage, so no caption may say one
    ordered the items.

    Driven rather than asserted about: the payload's own column order is
    compared against the frame's, because *the items were reordered* is a
    consequence and not a spelling.
    """
    block = SURVEY_SHAPES[shape]()
    payload = F.item_correlations_payload(block)
    caption = figures.REGISTRY["item_correlations"].caption(payload)

    assert payload["columns"], "the fixture produced no items and proves nothing"
    assert payload["columns"] == [str(c) for c in block.columns], (
        f"{shape}: the payload's item order now differs from the frame's, so "
        f"something DID reorder them — rewrite this test rather than delete it")

    for phrase in ("items ordered by hierarchical clustering",
                   "ordered by hierarchical clustering",
                   "clustered ordering"):
        assert phrase not in caption.lower(), (
            f"{shape}: the caption says {phrase!r}; the items are in the "
            f"instrument's own column order and no linkage is computed")


@pytest.mark.parametrize("shape", sorted(SURVEY_SHAPES))
def test_the_correlation_caption_still_says_which_ordering_the_field_asks_for(shape):
    """The shelf is never shortened. §B5.4 asks for the clustered ordering and
    a dendrogram; deleting the words would leave a reader unable to see that
    this matrix is not presented the way the field presents one."""
    caption = figures.REGISTRY["item_correlations"].caption(
        F.item_correlations_payload(SURVEY_SHAPES[shape]()))

    assert "hierarchical clustering" in caption, (
        f"{shape}: the caption no longer names the ordering §B5.4 asks for — "
        f"the row was about a FALSE claim, not about the word")
    assert "does not compute" in caption, (
        f"{shape}: the caption names the clustering without saying the app "
        f"does not do it, which is the false claim coming back")


def test_the_correlation_figure_does_not_declare_a_layer_nobody_draws():
    """`layers` reaches the client through `FigureSpec.to_dict`, so a declared
    `dendrogram` is the same assertion as the caption's, one field over."""
    spec = figures.REGISTRY["item_correlations"]
    assert spec.layers, "the spec declares no layers and this proves nothing"
    assert "dendrogram" not in spec.layers, (
        f"item_correlations declares a dendrogram layer and nothing computes "
        f"a linkage to draw one: {spec.layers}")
    assert spec.to_dict()["layers"], (
        "the wire form carries no layers, so this control checks nothing")
    assert "dendrogram" not in spec.to_dict()["layers"], (
        "the dendrogram is off the spec but still on the wire")


# ═══════════ THE FOREST ORDERING ═══════════

#: The ASSERTIVE forms of §A4.7's grouping claim — phrases rather than words,
#: for the reason `_CLAIMS_ORDERING` is: the shipped caption KEEPS §A4.7's
#: sentence and states that the app does not satisfy it, so a detector over the
#: word *domain* would fire on the disclosure and the next loop would delete the
#: disclosure to make the suite pass.
#: `test_the_grouping_detector_separates_the_claim_from_the_disclosure` below is
#: the control that proves the difference is real rather than intended.
_ASSERTS_THE_GROUPING = (
    "rows are ordered by domain",
    "ordered by domain rather than",
    "grouped by domain, in the order",
)


@pytest.mark.parametrize("shape", sorted(FOREST_SHAPES))
def test_the_forest_caption_claims_a_domain_grouping_only_where_one_exists(shape):
    """`AUDIT-008`. §A4.7 asks for predictors *grouped by domain and ordered
    meaningfully, not by significance*. The second half has always been true.
    The first is true only where the caller filled `group`, and the shipped
    caller does not — so the caption must read the payload rather than assert
    the better of the two.

    ── WHY THE CAPTION IS READ FIRST, AND WHY NOTHING HERE SUBSCRIPTS ──
    This test was written with `payload["grouped_by_domain"]` as its first
    assertion, and `grouped_by_domain` is a key **the fix itself added**. Under
    a revert of that fix all four parametrizations died on `KeyError` at that
    line — the `TypeError`/`AttributeError` family, red for the wrong reason —
    before ever reaching the claim they exist to check, so `AUDIT-008` could
    not be closed on them (`L52`, §08.1). The claim is now driven through the
    CAPTION, a surface whose SHAPE did not change: a reverted app still returns
    a string and only its CONTENT differs. The payload key is still checked,
    last, and through `.get`. The failure messages use `.get` and locals for
    the same reason — an f-string is evaluated only when the assert fails, so a
    bare subscript in one destroys the probe at the moment it is needed.
    """
    payload, caption = _forest(shape)
    lowered = caption.lower()

    # `GUIDED-045`: the fixture produced something before any absence is read
    # off it.
    assert payload["rows"], "the fixture produced no rows and proves nothing"
    with_a_domain = [row for row in payload["rows"] if row.get("group")]
    grouped = bool(with_a_domain)

    # THE CLAIM, and it is one assertion carrying both directions: the caption
    # asserts the grouping exactly where the rows have one.
    asserted = [phrase for phrase in _ASSERTS_THE_GROUPING if phrase in lowered]
    assert bool(asserted) is grouped, (
        f"{shape}: the caption "
        f"{'asserts' if asserted else 'never asserts'} that the rows are "
        f"grouped by domain {asserted} while {len(with_a_domain)} of "
        f"{len(payload['rows'])} rows carry one — §A4.7's grouping is a claim "
        f"about the payload, not a house sentence")

    if not grouped:
        # `AUDIT-028`'s third clause: where nothing true can be said, the
        # silence is STATED. And the shelf is never shortened — the caption
        # still names the grouping §A4.7 asks for, as the absent thing.
        assert "the rows are ungrouped" in lowered, (
            f"{shape}: the caption neither claims the grouping nor states its "
            f"absence, which is the blank the house form forbids")
        assert "grouped by domain" in lowered, (
            f"{shape}: the caption dropped §A4.7's grouping altogether — the "
            f"row was about a FALSE claim, not about the words")

    assert "not re-sorted by significance" in lowered, (
        f"{shape}: the true half of §A4.7's sentence was dropped along with "
        f"the false half — that is deletion, not correction")

    # LAST, AND THROUGH `.get`. The caption above is the claim; this is the
    # record behind it, and a payload that stopped recording which of the two
    # happened is the caption asserting again rather than reporting.
    assert payload.get("grouped_by_domain") is grouped, (
        f"{shape}: the payload records grouped_by_domain="
        f"{payload.get('grouped_by_domain')!r} and the rows say {grouped}")


def test_the_grouping_detector_separates_the_claim_from_the_disclosure():
    """`GUIDED-045`'s control for the four parametrizations above, run on the
    sentence `AUDIT-008` was filed for and on the one that replaced it.

    A detector reporting *no assertive claim* reports the same nothing for an
    honest caption, a mistyped phrase, and a phrase the caption was rewritten
    around.
    """
    before = ("Rows are ordered by domain rather than by significance. "
              "**These are the model's coefficients, not causal effects**.")
    assert [p for p in _ASSERTS_THE_GROUPING if p in before.lower()], (
        "the detector does not fire on the sentence `AUDIT-008` was filed "
        "for, so its silence on the shipped caption means nothing")

    shipped = figures.REGISTRY["forest"].caption(
        F.forest_payload(_AS_THE_BUNDLE_BUILDS_THEM)).lower()
    assert "grouped by domain" in shipped, (
        "the shipped caption no longer names §A4.7's grouping at all, so this "
        "control is checking nothing")
    assert not [p for p in _ASSERTS_THE_GROUPING if p in shipped], (
        f"the detector fires on the shipped disclosure — it is banning the "
        f"words rather than the claim, and the next loop will delete the "
        f"disclosure to satisfy it: {shipped}")


def test_the_forest_payload_records_the_ordering_the_caption_reads():
    """THE POSITIVE CONTROL for
    `test_the_forest_caption_claims_a_domain_grouping_only_where_one_exists`.

    Both of its branches are about a caption reading `ordering`. If the
    caption stopped reading it and hard-coded the sentence again, they would
    keep passing while the payload and the caption drifted apart. This changes
    the payload and requires the caption to move with it.
    """
    payload = F.forest_payload(_AS_THE_BUNDLE_BUILDS_THEM)
    caption = figures.REGISTRY["forest"].caption(payload)
    assert payload["ordering"] in caption, (
        "the caption does not contain the payload's own `ordering` value, so "
        "it is asserting an ordering rather than reporting one")

    moved = dict(payload, ordering="sorted by the analyst's own hand")
    assert "sorted by the analyst's own hand" in (
        figures.REGISTRY["forest"].caption(moved)), (
        "changing the payload's ordering did not change the caption — the "
        "caption is not reading the payload")


# ═══════════ THE SWEEP · CAPTION-CLAIM OVER ORDERING ═══════════

#: An ASSERTIVE ordering claim, paired with the payload predicate that would
#: make it keepable. The disclosure forms — *the field's recommendation is to
#: order the items by hierarchical clustering*, *§A4.7 asks for predictors
#: grouped by domain* — are deliberately not matched, and
#: `test_the_detector_does_not_fire_on_the_disclosure` proves it.
_CLAIMS_ORDERING = (
    ("ordered by hierarchical clustering",
     lambda p: bool(p.get("ordering_applied_clustering"))),
    ("items ordered by clustering",
     lambda p: bool(p.get("ordering_applied_clustering"))),
    ("rows are ordered by domain",
     lambda p: bool(p.get("grouped_by_domain"))),
    ("grouped by domain, in the order",
     lambda p: bool(p.get("grouped_by_domain"))),
    ("ordered by net agreement",
     lambda p: p.get("sort") == "net_agreement_descending"),
    ("ordered by effect size",
     lambda p: p.get("sorted_by_significance") is True),
)


def _constructible():
    """Every registered figure whose payload this file can build."""
    rng = np.random.default_rng(19)
    n = 300
    x = rng.normal(size=n)
    proba = 1.0 / (1.0 + np.exp(-x))
    y = (rng.random(n) < proba).astype(int)
    block = _likert_block()
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    spline_frame = pd.DataFrame({"intake": rng.normal(size=n)})
    spline_frame["outcome"] = 0.4 * spline_frame["intake"] + rng.normal(size=n)
    return {
        "calibration": F.calibration_render(y, proba),
        "pca_scores": F.pca_scores_payload(frame),
        "shrinkage": F.shrinkage_payload(
            {"single_day": list(rng.normal(2000, 500, n)),
             "mean_of_days": list(rng.normal(2000, 300, n)),
             "usual_intake": list(rng.normal(2000, 200, n))},
            nutrient="energy", unit="kcal", n_days=2),
        "dose_response_spline": F.spline_payload(
            spline_frame, exposure="intake", outcome="outcome"),
        "diverging_stacked_bar": F.diverging_bar_payload(
            block, columns=_ITEMS[:5], scale=[1, 2, 3, 4, 5]),
        "decision_curve": F.decision_curve_payload(y, {"model": proba}),
        "roc": F.roc_payload(y, {"model": proba}),
        "forest": F.forest_payload(_AS_THE_BUNDLE_BUILDS_THEM),
        "scree": F.scree_payload(block, n_simulations=8, seed=1),
        "item_correlations": F.item_correlations_payload(block),
        "floor_ceiling": F.floor_ceiling_payload(block),
        "item_panel": F.item_panel_payload(block),
    }


def test_no_caption_asserts_an_ordering_its_payload_did_not_record():
    """THE SWEEP. `AUDIT-008` as a class rather than as two more instances."""
    constructible = _constructible()

    checked, offenders, kept = 0, [], []
    for figure_id, payload in constructible.items():
        spec = figures.REGISTRY.get(figure_id)
        assert spec is not None, (
            f"{figure_id} is not in the registry, so this sweep is checking a "
            f"figure no bundle can admit")
        caption = spec.caption(payload).lower()
        checked += 1
        for phrase, keepable in _CLAIMS_ORDERING:
            if phrase in caption:
                (kept if keepable(payload) else offenders).append(
                    (figure_id, phrase))

    assert checked == len(constructible)
    # GUIDED-045: the sweep found SOMETHING before it is allowed to report
    # finding nothing wrong. `diverging_stacked_bar` sorts by net agreement and
    # says so, so at least one assertive ordering claim must be standing.
    assert kept, (
        "no caption in the registry names an ordering at all, so a clean "
        "result here is the sweep matching nothing rather than the captions "
        "being honest")
    assert not offenders, (
        f"these captions assert an ordering the payload did not record: "
        f"{offenders}")

    not_constructed = sorted(set(figures.REGISTRY) - set(constructible))
    assert len(not_constructed) + checked == len(figures.REGISTRY)
    assert checked >= 12, (
        f"only {checked} of {len(figures.REGISTRY)} registered figures were "
        f"swept; the rest are {not_constructed} — the four instability "
        f"figures need a bootstrap result and volcano refuses autoscaled data")


def test_the_detector_fires_on_the_claim_it_exists_to_catch():
    """GUIDED-045's positive control for the sweep above.

    A sweep reporting no offenders reports the same nothing for honest
    captions, a mistyped phrase and a predicate that stopped being reachable.
    """
    before = ("Inter-item correlations among 20 items, items ordered by "
              "hierarchical clustering, palette fixed from -1 to +1.")
    hits = [phrase for phrase, keepable in _CLAIMS_ORDERING
            if phrase in before.lower() and not keepable({})]
    assert hits, (
        "the detector does not fire on the sentence `AUDIT-008` was filed "
        "for, so a clean sweep means nothing")


def test_the_detector_does_not_fire_on_the_disclosure():
    """The other half of the control, and the reason the phrases are phrases.

    Both corrections KEPT the method's name and said the app does not apply
    it. A detector that banned the words would force the next loop to delete
    the disclosure to make the suite pass — the deletion `AUDIT-016` refused.
    """
    disclosure = figures.REGISTRY["item_correlations"].caption(
        F.item_correlations_payload(_likert_block())).lower()
    assert "hierarchical clustering" in disclosure, (
        "the disclosure under test no longer names the method, so this "
        "control is checking nothing")

    hits = [phrase for phrase, keepable in _CLAIMS_ORDERING
            if phrase in disclosure and not keepable(
                F.item_correlations_payload(_likert_block()))]
    assert not hits, (
        f"the detector fires on the shipped disclosure {hits} — it is banning "
        f"the word rather than the claim, and the next loop will delete the "
        f"disclosure to satisfy it")
