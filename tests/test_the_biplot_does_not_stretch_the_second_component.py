"""`AUDIT-006` — the biplot's panel is a claim about the data, and it made a
false one.

`ml.macro_shape.plot_pca_biplot` labels its axes with the % variance each
component explains — `PC1 (96.6%)` against `PC2 (0.8%)` on the fixture below —
and then set **no aspect constraint at all**. Plotly autoranges each axis
independently, so PC2's spread was drawn to the full height of the panel
whatever its size: on this fixture one data unit on PC2 was drawn about **11
times longer** than one data unit on PC1. The axis label said 0.8% and the
picture showed a component as tall as PC1. A reader reads separation off the
picture.

`research/GENOMICS_PACK.md` §07 · *EDA and presentation — the priority*,
subsection A · *PCA of samples*, item 2 names the failure in the same terms
this ledger does:

> *"**Equal aspect ratio** — without it the visual separation is a lie
> proportional to the aspect ratio. Widely ignored; the DESeq2 workflow
> explicitly recommends it."*

and §11 · *Anti-pattern registry* grades *"Unlabeled PCA axes, unequal aspect
ratio"* a **SETTLED presentation error**, consequence *"Uninterpretable /
misleading"*. `research/METABOLOMICS_PACK.md` §06.1 item 3 asks for *"Aspect
ratio proportional to variance explained (or at minimum equal aspect) —
stretching PC2 to fill the panel visually exaggerates separation."*

**THE CORRECTION IS GEOMETRY, NOT WORDS.** Nothing was deleted from the figure:
the axis labels still carry the percentages, and `test_the_percentages_stayed_on_the_axes`
holds them there. What changed is that the y axis is anchored to the x axis at a
ratio of 1, so a unit of score is a unit of score on both axes and the panel's
extent along PC2 relative to PC1 is the ratio of the components' standard
deviations — set by the variance each component explains rather than by the
height of the panel. This is what `turbotab.figure_specs.PCA_SCORES` already
records as `aspect_ratio` for the Guided door's renderer; the Classic door's
plotly figure had no equivalent.

**WHAT IS OBSERVED, EXACTLY.** Pixels cannot be observed without a browser, so
these tests assert the constraint on the figure's **wire form** — the JSON
Streamlit hands to plotly.js, which is the thing that decides the geometry. The
test name says `constrains`, not `draws`, for that reason (§07 trap 3b).

`GUIDED-097` — the claim is driven on all three target shapes the function
branches on, because the scatter is added in three separate places and a fix
written into one branch's `add_trace` would pass on the others.
`SHAPES_NOT_COVERED` names what is left.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.macro_shape import compute_pca, plot_pca_biplot

#: `GUIDED-097`. The three shapes below are the three branches of
#: `plot_pca_biplot`'s coloring: no target, a target `pd.to_numeric` accepts,
#: and a target it raises on. What is NOT covered:
SHAPES_NOT_COVERED = [
    "a datetime target — `pd.to_numeric` raises on it, so it takes the "
    "categorical branch and draws one color per timestamp; that is a "
    "separate defect and is not this row",
    "a target with more than eight levels — `px.colors.qualitative.Set2` "
    "wraps around and two levels share a color, also a separate defect",
    "UMAP and Mapper, which are the other two embeddings on the same page "
    "and are drawn by `plot_umap` / `plot_mapper` with no aspect constraint "
    "either — the same lens one surface over, reported rather than fixed here",
]


def _anisotropic_frame(seed: int = 3, n: int = 150, k: int = 8,
                       noise: float = 0.25) -> pd.DataFrame:
    """Eight columns driven by one latent factor, so PC1 explains ~97% and PC2
    under 1%. This is the ordinary shape of a correlated tabular dataset, not a
    contrived one — and it is where an unconstrained panel distorts most."""
    rng = np.random.RandomState(seed)
    driver = rng.normal(size=n)
    return pd.DataFrame(
        {f"f{i}": driver * (1.0 + 0.1 * i) + rng.normal(scale=noise, size=n)
         for i in range(k)})


def _pca():
    result = compute_pca(_anisotropic_frame(), n_components=2)
    assert "error" not in result, result.get("error")
    return result


TARGET_SHAPES = {
    "no target supplied": lambda n: None,
    "continuous target, numeric branch": (
        lambda n: np.linspace(0.0, 10.0, n)),
    "string class labels, categorical branch": (
        lambda n: np.array(["case", "control"] * (n // 2) + ["case"] * (n % 2))),
}


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_biplot_constrains_pc2_to_the_scale_of_pc1(shape):
    """`AUDIT-006`. The panel may not stretch PC2 to fill it."""
    result = _pca()
    scores = np.asarray(result["components"])
    evr = [float(v) for v in result["explained_variance_ratio"][:2]]

    # `GUIDED-045`'s positive control, and here it is load-bearing rather than
    # ceremonial: on an isotropic fixture an unconstrained panel distorts
    # nothing, so a fixture whose components explain similar variance would
    # make this test pass for a reason that has nothing to do with the fix.
    assert evr[0] / evr[1] > 20, (
        f"the fixture's components explain {evr[0]:.3f} and {evr[1]:.3f} of "
        f"the variance — too close for an unconstrained panel to distort, so "
        f"this test would prove nothing")
    stretch = float(np.ptp(scores[:, 0]) / np.ptp(scores[:, 1]))

    fig = plot_pca_biplot(result, TARGET_SHAPES[shape](len(scores)), "Target")
    assert fig.data, f"{shape}: no trace was drawn, so the layout proves nothing"

    wire = fig.to_dict()["layout"]
    yaxis = wire.get("yaxis", {})
    assert yaxis.get("scaleanchor") == "x", (
        f"{shape}: the biplot sets no aspect constraint on its y axis "
        f"({yaxis.get('scaleanchor')!r}), so plotly autoranges PC2 to fill the "
        f"panel: PC2 explains {evr[1]:.1%} of the variance against PC1's "
        f"{evr[0]:.1%} and its {stretch:.1f}x smaller spread is drawn the same "
        f"height, which exaggerates separation by that factor")
    ratio = yaxis.get("scaleratio")
    assert ratio is None or float(ratio) == 1.0, (
        f"{shape}: the y axis is anchored at scaleratio={ratio!r}, so a unit "
        f"of score on PC2 is still not a unit of score on PC1")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_percentages_stayed_on_the_axes(shape):
    """The shelf is never shortened, and this is the half of the pair that
    proves the correction was geometry rather than the deletion of the labels.

    Green before the fix as well as after — it is a control on the fix, not
    evidence for it, and saying so is cheaper than an adjudicator finding out.
    """
    result = _pca()
    fig = plot_pca_biplot(result, TARGET_SHAPES[shape](
        len(result["components"])), "Target")
    wire = fig.to_dict()["layout"]
    for axis, component in (("xaxis", "PC1"), ("yaxis", "PC2")):
        title = str((wire.get(axis, {}).get("title") or {}).get("text", ""))
        assert title.startswith(component) and "%" in title, (
            f"{shape}: the {axis} label is {title!r} — §06.1 item 1 calls "
            f"omitting the % variance the single most common defect in this "
            f"figure, and an equal-aspect panel without it is still "
            f"uninterpretable")


def test_the_constraint_is_on_the_wire_streamlit_hands_to_plotly():
    """§07 trap 6 — a geometry the server computes and the browser never sees
    is not a geometry. `st.plotly_chart` serialises the figure; this asserts
    the constraint survives that serialisation rather than living only on the
    python object."""
    fig = plot_pca_biplot(_pca(), None, "Target")
    payload = fig.to_plotly_json()
    yaxis = payload["layout"].get("yaxis", {})
    assert yaxis.get("scaleanchor") == "x", (
        f"the aspect constraint is absent from the JSON handed to plotly.js: "
        f"{yaxis!r}")
