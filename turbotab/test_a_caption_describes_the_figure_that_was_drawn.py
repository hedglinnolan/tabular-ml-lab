"""`AUDIT-016` and `AUDIT-036` — a caption is a claim about how a figure was made.

Two rows, one class. A reader checks the method against the caption, and a
caption is the one place in a figure where the app speaks in its own voice
about what it did.

**`AUDIT-016` · the calibration caption said the curve was a loess estimate
with a pointwise 95% band.** It is ten equal-width bins joined by straight
lines and the payload carries no band, no CI and no interval key of any kind.
`§A4.3` names the smooth curve with a shaded 95% pointwise band as item 3 of
what makes a calibration plot publication-grade, and names *10-decile binned
calibration plots* in its anti-pattern list — so the caption claimed the
figure met the standard by which the same section judges it wanting.

**`AUDIT-036` · the item-correlation and scree captions told the reader the
loadings and reliability appeared 'below'.** No loading matrix and no
reliability coefficient exists in this repository. `§B6` asks that reliability
be reported alongside the model; not computing it is permitted silence, and
directing the reader to a quantity the document does not contain is not.

## The fix is a correction, not a deletion, and that is the load-bearing part

`PRODUCT_VISION.md`, *the shelf is never shortened*. The word *loess* still
appears in the calibration caption and *reliability* still appears in the
correlation note — both now saying what the figure is **not** and where the
missing quantity has to come from. Deleting the words would have removed the
disclosure §A4.3 and §B6 exist to require, and left a reader with no way to
know the figure is the anti-pattern rather than the standard. So the
assertions below are about the **direction of the claim**, never about whether
a word appears.

The naive form of this test — *the caption must not contain 'loess'* — is
written out and refused in `test_the_disclosure_is_not_satisfied_by_deletion`,
because it is the form a later loop would reach for.

## CLASS

`CAPTION-CLAIM`: a caption asserting a method or a quantity the payload cannot
supply. `test_no_caption_claims_a_smoothing_method_its_payload_does_not_carry`
is the sweep over the registry; the counts it reports are in the loop report.

`GUIDED-097` — THE FIXTURE RULE. Two target shapes per claim, and the shapes
not covered are named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import figure_specs as F
from turbotab import figures

#: `GUIDED-097`. The calibration figure is classification-only, so the two
#: shapes are the two the figure can actually be drawn for: a genuinely binary
#: outcome, and a multiclass outcome reduced one-vs-rest — which is what
#: `ml.calibration.calibration_classification` does with a k-column probability
#: matrix, and it is the shape nothing had driven this caption against.
#:
#: The survey figures take an item block rather than an outcome, so their
#: second shape is the block's own: a 5-point Likert block and a binary
#: (agree/disagree) block, which differ in whether the correlation matrix is
#: near-singular.
SHAPES_NOT_COVERED = [
    "survival / time-to-event — no task type exists and no calibration "
    "figure is registered for one",
    "a continuous outcome — `calibration_regression` exists in ml/ but no "
    "FigureSpec is registered over it, so there is no caption to check",
    "a smoothed correlation matrix — `smoothing_note` fires only on a "
    "non-positive-definite matrix and no fixture produces one",
]


def _binary(n=400, seed=7):
    """A real two-class outcome with predictions that are not perfect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.random(n) < p).astype(int)
    return y, p


def _multiclass_one_vs_rest(n=400, seed=11):
    """Three classes, reduced the way the shipped code reduces them.

    `calibration_classification` takes the last column of a k-column
    probability matrix when k != 2, so this is the array pair the figure
    actually receives from a three-class run.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p3 = np.column_stack([
        1.0 / (1.0 + np.exp(-(x - 1.0))),
        np.full(n, 0.3),
        1.0 / (1.0 + np.exp(-(x + 0.5)))])
    p3 = p3 / p3.sum(axis=1, keepdims=True)
    y = (rng.random(n) < p3[:, -1]).astype(int)
    return y, p3


CALIBRATION_SHAPES = {
    "binary outcome": _binary,
    "multiclass reduced one-vs-rest": _multiclass_one_vs_rest,
}

_ITEMS = [f"item_{i:02d}" for i in range(1, 21)]


def _likert_block():
    return pd.read_csv("turbotab/sample_data/survey_instrument.csv")[_ITEMS]


def _binary_block():
    """The same items dichotomized — a different block shape over the same
    figure, so the caption is not verified against one matrix."""
    block = _likert_block()
    return (block >= block.median()).astype(int)


SURVEY_SHAPES = {"5-point Likert block": _likert_block,
                 "dichotomized block": _binary_block}


# ═══════════ AUDIT-016 · THE CALIBRATION CAPTION ═══════════

@pytest.mark.parametrize("shape", sorted(CALIBRATION_SHAPES))
def test_the_calibration_caption_does_not_claim_a_curve_the_payload_is_not(shape):
    """`AUDIT-016`. The claim direction, driven on a real render.

    The payload carries `curve.predicted` — the mean predicted risk within
    each of `n_bins` equal-width bins — and nothing else. So the caption may
    say the curve is binned and must not say it is smoothed, and it may name
    the smooth curve only as the thing this figure is not.
    """
    y, p = CALIBRATION_SHAPES[shape]()
    payload = F.calibration_render(y, p)
    caption = figures.REGISTRY["calibration"].caption(payload)

    points = len(payload["curve"]["predicted"])
    assert points > 0, "the fixture produced no curve and proves nothing"
    assert f"{points} equal-width bins" in caption, (
        f"{shape}: the caption does not say the curve is {points} bins; it is")

    lowered = caption.lower()
    for phrase in ("the flexible (loess) estimate",
                   "the loess estimate",
                   "is the flexible",
                   "smoothed curve"):
        assert phrase not in lowered, (
            f"{shape}: the caption claims the curve is a smoothed estimate "
            f"({phrase!r}); it is {points} equal-width bins")


@pytest.mark.parametrize("shape", sorted(CALIBRATION_SHAPES))
def test_the_calibration_caption_does_not_promise_a_band_the_payload_has_no_key_for(shape):
    """`AUDIT-016`'s second half. An interval is a quantity, not a style.

    Driven as a property of the payload rather than as a fixed string: any key
    carrying a band or an interval would make the promise keepable, and the
    assertion is that the caption promises one only if such a key exists.
    """
    y, p = CALIBRATION_SHAPES[shape]()
    payload = F.calibration_render(y, p)
    caption = figures.REGISTRY["calibration"].caption(payload)

    interval_keys = [k for k in payload
                     if "band" in k or "interval" in k or k.endswith("_ci")]
    assert not interval_keys, (
        f"{shape}: the payload now carries {interval_keys}, so this test is "
        f"asserting the wrong thing — rewrite it rather than delete it")

    assert "it carries no interval" in caption, (
        f"{shape}: the caption does not state that the curve carries no "
        f"interval, and the payload has no key that could supply one")
    assert "with a pointwise 95% band that" in caption, (
        f"{shape}: 'pointwise 95% band' must appear only as the thing this "
        f"figure is not — the phrase is §A4.3's item 3 and dropping it "
        f"removes the disclosure rather than the false claim")


def test_the_disclosure_is_not_satisfied_by_deletion():
    """**The shelf is never shortened**, as an assertion.

    The obvious repair for `AUDIT-016` is to delete the word *loess* from the
    caption. That closes the row by removing the claim, and it makes the
    figure worse: §A4.3 marks the binned plot an anti-pattern and the reader's
    only way to know that is the caption saying so. This test fails if a later
    loop reaches for the deletion.
    """
    y, p = _binary()
    caption = figures.REGISTRY["calibration"].caption(F.calibration_render(y, p))

    assert "loess" in caption, (
        "the caption no longer names the smooth curve at all. The row was "
        "about a FALSE claim, not about the word — removing the word removes "
        "§A4.3's disclosure that this figure is the anti-pattern")
    assert "publication-grade" in caption, (
        "the caption no longer says which standard this figure falls short of")


# ═══════════ AUDIT-036 · THE SURVEY CAPTIONS ═══════════

#: BOTH captions, because the same string reaches both. `_correlations`
#: composes `method_note` once; `item_correlations` appends it directly and
#: `scree` carries it through `correlation_note`. A fix applied to one would
#: have left the other asserting the same false thing.
SURVEY_CAPTIONS = {
    "item_correlations": lambda block: (
        figures.REGISTRY["item_correlations"].caption(
            F.item_correlations_payload(block))),
    "scree": lambda block: (
        figures.REGISTRY["scree"].caption(
            F.scree_payload(block, n_simulations=8, seed=1))),
}


@pytest.mark.parametrize("shape", sorted(SURVEY_SHAPES))
@pytest.mark.parametrize("figure_id", sorted(SURVEY_CAPTIONS))
def test_no_survey_caption_points_the_reader_at_a_quantity_below(figure_id, shape):
    """`AUDIT-036`. Nothing in this repository computes a factor loading or a
    reliability coefficient, so no caption may direct the reader to one."""
    caption = SURVEY_CAPTIONS[figure_id](SURVEY_SHAPES[shape]())

    for phrase in ("loadings and reliability below",
                   "reliability below", "loadings below",
                   "the loadings and reliability"):
        assert phrase not in caption.lower(), (
            f"{figure_id} / {shape}: the caption says {phrase!r}; this app "
            f"produces neither a loading matrix nor a reliability "
            f"coefficient")


@pytest.mark.parametrize("shape", sorted(SURVEY_SHAPES))
@pytest.mark.parametrize("figure_id", sorted(SURVEY_CAPTIONS))
def test_the_survey_captions_say_where_the_missing_quantity_has_to_come_from(
        figure_id, shape):
    """`AUDIT-036` again, and the same shelf rule as the calibration pair.

    §B6 asks that reliability be reported alongside the model. The app cannot
    compute it, so the honest form is to say so — deleting the sentence would
    leave the researcher unaware there is a quantity they still owe.
    """
    caption = SURVEY_CAPTIONS[figure_id](SURVEY_SHAPES[shape]())

    assert "no factor loadings and no reliability coefficient" in caption, (
        f"{figure_id} / {shape}: the caption no longer states that the app "
        f"computes neither, which is §B6's requirement met by disclosure")
    assert "has to come from elsewhere" in caption, (
        f"{figure_id} / {shape}: the caption does not tell the researcher "
        f"the reliability has to come from somewhere else")


def test_nothing_in_the_repository_computes_the_quantity_the_captions_disclaim():
    """THE POSITIVE CONTROL for the two tests above.

    Their claim is *the app produces no reliability coefficient*. If one were
    ever added, the captions' disclaimer would become the false sentence and
    these tests would keep passing — an absence assertion that has quietly
    become wrong. This checks the premise instead of assuming it.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"def .*(cronbach|omega_total|internal_consistency)",
                         re.IGNORECASE)

    # AND THIS CONTROL NEEDS ONE OF ITS OWN, which is the joke `GUIDED-045`
    # keeps making: a sweep that reports no hits reports the same nothing for a
    # clean repository, a mistyped folder name, and a regex that stopped
    # compiling to anything useful. Both halves are checked before the sweep
    # is allowed to mean anything.
    assert pattern.search("def cronbach_alpha(items):\n"), (
        "the pattern does not match the definition it exists to find")
    assert not pattern.search('CAPTION = "no Cronbach alpha is computed"\n'), (
        "the pattern matches prose about the quantity, so a hit would not "
        "mean the quantity is computed")

    hits = []
    scanned = 0
    for folder in ("turbotab", "ml", "utils", "pages"):
        for path in (root / folder).rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            scanned += 1
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(path.relative_to(root)))
    assert scanned >= 100, (
        f"only {scanned} non-test modules were read across turbotab/, ml/, "
        f"utils/ and pages/ — this is an absence claim and it would pass "
        f"loudest on a glob that matched nothing")
    assert not hits, (
        f"a reliability coefficient is computed in {hits} — the captions now "
        f"disclaim a quantity the app has, and they are the false sentence")


# ═══════════ THE CLASS · CAPTION-CLAIM ═══════════

#: A caption saying the curve IS one of these is claiming a smoothing method.
#: Listed as the possessive/copular forms the anti-pattern actually takes, so
#: a caption naming the method as the standard it falls short of — which
#: §A4.3 requires — is not caught by it.
_CLAIMS_SMOOTHING = (
    "the flexible (loess) estimate", "the loess estimate",
    "the loess curve", "the spline curve", "the smoothed curve",
    "fitted by loess", "estimated by loess", "loess-smoothed",
)


def test_no_caption_claims_a_smoothing_method_its_payload_does_not_carry():
    """THE SWEEP, and it is the class rather than the two instances.

    Every registered figure whose payload this test can construct, checked for
    the `AUDIT-016` shape. It reports what it could not construct as a count
    rather than passing over it in silence — an absence assertion over a set
    the sweep could not enumerate is a claim about nothing.
    """
    y, p = _binary()
    block = _likert_block()
    constructible = {
        "calibration": F.calibration_render(y, p),
        "item_correlations": F.item_correlations_payload(block),
        "scree": F.scree_payload(block, n_simulations=8, seed=1),
        "floor_ceiling": F.floor_ceiling_payload(block),
        "item_panel": F.item_panel_payload(block),
    }

    checked, offenders = 0, []
    for figure_id, payload in constructible.items():
        spec = figures.REGISTRY.get(figure_id)
        assert spec is not None, (
            f"{figure_id} is not in the registry, so this sweep is checking "
            f"a figure no bundle can admit")
        caption = spec.caption(payload).lower()
        checked += 1
        for phrase in _CLAIMS_SMOOTHING:
            if phrase in caption:
                offenders.append((figure_id, phrase))

    assert checked == len(constructible)
    assert not offenders, (
        f"these captions claim a smoothing method: {offenders}")

    # WHAT WAS NOT CONSTRUCTED, counted rather than omitted. Reported so the
    # sweep's coverage is a number instead of an impression.
    not_constructed = sorted(set(figures.REGISTRY) - set(constructible))
    assert len(not_constructed) + checked == len(figures.REGISTRY)
    assert checked >= 5, (
        f"only {checked} of {len(figures.REGISTRY)} registered figures were "
        f"swept; the rest need a fitted model or a project and are "
        f"{not_constructed}")
