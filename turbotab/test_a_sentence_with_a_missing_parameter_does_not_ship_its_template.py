"""`GUIDED-175` — a sentence with a missing parameter does not ship its template.

`turbotab/features.py:462-465` was `try: return t.sentence.format(**fields)` /
`except KeyError: return t.sentence`, so a transform whose parameter had not
been supplied returned its TEMPLATE. The product owner drove it and read, on
screen, *"`{a}` will be grouped into {n_bins} clustered bins."* — and that
string is the decision sentence, which is the manuscript's methods line at a
different level of formality.

**The class: a fourth branch nobody authorized.** `AGENT_ONBOARD.md` §00 gives
three — the app may assert truly, it may be **silent**, and it may **refuse**.
Template syntax on screen is none of them: not false, not silent, not a
refusal, but noise where a sentence was promised. The old handler is also the
project's silent-degradation shape — an `except` that keeps going with a worse
answer of the right type, which is why no test noticed for as long as it shipped.

**The second half, and it is the half that gets missed:** a `KeyError` on
`n_bins` discarded the substitution of `{a}` as well, so the column the user
HAD chosen was thrown away with the parameter they had not.
`test_the_column_the_code_could_fill_is_not_discarded_with_the_parameter_it_could_not`
is that half, on its own.

**The option taken is (a): refuse, and say which parameter is outstanding.**
Three reasons. The refusal is the branch the governing rule already authorizes,
and it is already wired at all four `_sentence` call sites — `api.preview_feature`
(`api.py:2105`) and `project.defer_feature` (`project.py:646`) each catch
`FeatureRefusal` today, so nothing new has to be rendered anywhere. This module
already answers a missing parameter this way for the other half of the same
split pair (`_compute` refuses `bin_fixed` without `edges` and
`ordinal_declared` without `order`), so the deferred half returning a template
was the inconsistency. And a sentence is a decision's claim about itself:
`declare` writes one into the record, so a composed-anyway sentence is a
decision the user never made.

**(b) compose without the clause was rejected** — for `pca` the missing
`{n_components}` is the sentence's grammatical subject, and dropping it leaves
"principal components will be computed", which asserts an unstated number;
for the three binning entries "grouped into equal-sized bins" reads as *the app
chose for you*, on top of `pipeline_plan.py:764`'s undisclosed `n_bins=4`
default. **(c) return a sentence naming the outstanding parameter was rejected
as the RETURN value and kept as the refusal's content** — the returned string
is recorded and exported, and a methods line reading "the number of bins is
outstanding" is a decision recorded before it was made.

Every assertion here is over a **returned** value, and the no-placeholder claim
is a regex over the result rather than a comparison against one known template,
so a new catalogue entry with a new parameter is covered the day it is added.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import features as F                                       # noqa: E402
from turbotab.project import AnalysisProject, ProjectError               # noqa: E402

# What `str.format` would have consumed. Derived from the RESULT, never from a
# list of the placeholders this catalogue happens to use today — the defect is
# "an unsubstituted field reached the user", not "`{n_bins}` reached the user".
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")

# One legitimate value per parameter the catalogue declares in `needs`. Keyed by
# the parameter NAME, so a new transform that needs `n_bins` is covered and one
# that needs something else fails loudly in `_params_for` rather than silently
# skipping.
_PARAM_VALUES = {
    "n_bins": 4,
    "n_components": 2,
    "edges": [50.0, 70.0, 90.0, 200.0],
    "order": ["mild", "moderate", "severe"],
}

_ALL_KEYS = sorted(F.CATALOGUE)
_PARAMETERIZED = sorted(k for k in F.CATALOGUE if F.get(k).needs)
_PARAMETERIZED_DEFERRED = sorted(k for k in _PARAMETERIZED if F.get(k).defers)


def study(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "weight_kg": rng.normal(78, 12, n),
        "height_m": rng.normal(1.7, 0.1, n),
        "severity": (["mild", "moderate", "severe"] * n)[:n],
        "outcome": rng.integers(0, 2, n),
    })


def _columns_for(t: F.Transform) -> list:
    if t.key == "ordinal_declared":            # the one entry needing strings
        return ["severity"]
    return ["weight_kg", "height_m"][:t.n_inputs]


def _params_for(t: F.Transform) -> dict:
    return {name: _PARAM_VALUES[name] for name in t.needs}


def _sentences_from_every_call_site(key: str, params: dict) -> dict:
    """One entry, through every door that composes a sentence for it.

    The four `_sentence` call sites in `features.py` are the row-local preview
    (347), the `apply` receipt (389), the `declare` spec (413) and the deferred
    preview (483). A row-local key reaches 347 and 389; a deferred key reaches
    483 and 413. Across the catalogue the union is all four, which is what
    makes this a check on the composer rather than on one caller.
    """
    t = F.get(key)
    cols = _columns_for(t)
    out = {"preview": F.preview(study(), key, cols, params)["sentence"]}
    if t.defers:
        out["declare"] = F.declare(key, cols, params)["sentence"]
    else:
        out["apply"] = F.apply(study(), key, cols, params)["receipt"]["sentence"]
    return out


# ── the sweep is not vacuous, and its ids are addressable ────────────────────

def test_the_sweep_covers_every_transform_and_every_parameter_the_catalogue_has():
    """A parametrized sweep over an empty list is a green test over nothing.

    `TEST-045`: the ids must be ASCII, or the revert harness cannot address the
    case it needs to turn red.
    """
    assert len(_ALL_KEYS) >= 15, f"only {len(_ALL_KEYS)} catalogue entries swept"
    assert _PARAMETERIZED_DEFERRED, (
        "no deferred transform declares a parameter, so the missing-parameter "
        "sweep below would pass without exercising anything")
    for key in _ALL_KEYS:
        assert key.isascii(), f"parametrize id {key!r} is not ASCII (TEST-045)"
        for name in F.get(key).needs:
            assert name in _PARAM_VALUES, (
                f"{key} needs {name!r} and this test has no value for it, so it "
                f"would be swept with the parameter missing and pass for the "
                f"wrong reason")


# ── half one: no unsubstituted placeholder reaches a returned sentence ───────

@pytest.mark.parametrize("key", _ALL_KEYS, ids=_ALL_KEYS)
def test_no_returned_sentence_carries_an_unsubstituted_placeholder(key):
    """Regex over the RESULT. Not a comparison against a known template."""
    for door, said in _sentences_from_every_call_site(key, _params_for(F.get(key))).items():
        found = _PLACEHOLDER.search(said)
        assert found is None, (
            f"{key} via {door} shipped {found.group(0)!r} to the user: {said!r}")
        assert said != F.get(key).sentence, (
            f"{key} via {door} returned its template verbatim: {said!r}")


def _what_the_user_would_see(call) -> tuple:
    """The string that reaches a person, whichever branch the app took.

    A returned sentence and a refusal are both *what the user sees*; the point
    of `GUIDED-175` is that template syntax is neither. Collapsing the two here
    is deliberate — it makes the assertion below a claim about the CLASS rather
    than about option (a), so a later loop that switches to (b) or (c) keeps
    this guard and does not have to rewrite it.
    """
    try:
        return ("sentence", str(call()))
    except F.FeatureRefusal as exc:
        return ("refusal", str(exc))


@pytest.mark.parametrize("key", _PARAMETERIZED_DEFERRED, ids=_PARAMETERIZED_DEFERRED)
def test_nothing_a_missing_parameter_produces_carries_template_syntax(key):
    """`GUIDED-175` itself: the fourth branch is closed at every door.

    All three honest options pass this. Only the template fails it, which is
    what makes it the revert anchor for this finding.
    """
    t = F.get(key)
    cols = _columns_for(t)

    for door, call in (("declare", lambda: F.declare(key, cols)["sentence"]),
                       ("preview",
                        lambda: F.preview(study(), key, cols)["sentence"])):
        kind, seen = _what_the_user_would_see(call)
        found = _PLACEHOLDER.search(seen)
        assert found is None, (
            f"{key} via {door} put {found.group(0)!r} in front of the user "
            f"as a {kind}: {seen!r}")
        assert seen != t.sentence, (
            f"{key} via {door} handed over its raw template as a {kind}")
        assert cols[0] in seen, (
            f"{key} via {door} lost the column {cols[0]!r} that WAS known: "
            f"{seen!r}")


@pytest.mark.parametrize("key", _PARAMETERIZED_DEFERRED, ids=_PARAMETERIZED_DEFERRED)
def test_a_transform_missing_its_parameter_refuses_rather_than_returning_a_sentence(key):
    """`GUIDED-175` driven: no params at all, which is what the page sends.

    `web/index.html:5807` fetches the preview with no `params` and `:5825`
    records with `params: {}`, so this is not a hypothetical call shape — it is
    the only one the product can make today.
    """
    t = F.get(key)
    cols = _columns_for(t)

    for door, call in (("declare", lambda: F.declare(key, cols)),
                       ("preview", lambda: F.preview(study(), key, cols))):
        with pytest.raises(F.FeatureRefusal) as caught:
            call()
        message = str(caught.value)
        found = _PLACEHOLDER.search(message)
        assert found is None, (
            f"{key} via {door} refused with {found.group(0)!r} still in the "
            f"refusal: {message!r}")
        assert t.sentence not in message, (
            f"{key} via {door} put its template inside the refusal instead of "
            f"withholding it: {message!r}")
        for name in t.needs:
            assert name in message, (
                f"{key} via {door} refused without saying which parameter is "
                f"outstanding: {message!r}")


# ── half two: what IS known survives the field that is not ───────────────────

@pytest.mark.parametrize("key", _PARAMETERIZED_DEFERRED, ids=_PARAMETERIZED_DEFERRED)
def test_the_column_the_code_could_fill_is_not_discarded_with_the_parameter_it_could_not(key):
    """The half that gets missed.

    The old `except KeyError` threw away the WHOLE substitution, so `{a}` — a
    column the user had already chosen from the dropdown — was lost because
    `n_bins` was absent. Partial knowledge is not partial ignorance.
    """
    t = F.get(key)
    cols = _columns_for(t)

    with pytest.raises(F.FeatureRefusal) as caught:
        F.declare(key, cols)
    message = str(caught.value)
    assert cols[0] in message, (
        f"{key} lost the column {cols[0]!r} the user had supplied when it could "
        f"not fill {t.needs}: {message!r}")

    # And at the composer: with the column known and the parameter not, the
    # column is recognized as filled. Only the genuinely absent field is
    # reported, which is the same statement one level down.
    assert F._unfilled(t.sentence, {"a": cols[0]}) == list(t.needs), (
        f"{key}: with the column supplied, the outstanding set should be "
        f"exactly {list(t.needs)}")
    assert F._unfilled(t.sentence, {"a": cols[0], **_params_for(t)}) == [], (
        f"{key}: nothing is outstanding once the parameter is supplied")


def test_the_unfilled_check_reads_the_catalogues_own_templates():
    """The new logic, against real entries rather than a manufactured one."""
    kmeans = F.get("bin_kmeans").sentence
    assert F._unfilled(kmeans, {"a": "weight_kg"}) == ["n_bins"]
    assert F._unfilled(kmeans, {"a": "weight_kg", "n_bins": 5}) == []
    assert F._unfilled(kmeans, {"n_bins": 5}) == ["a"]
    # A field supplied as blank is unfilled too: "The ratio `weight_kg` /  was
    # computed" is the same defect with the braces removed.
    assert F._unfilled(F.get("ratio").sentence,
                       {"a": "weight_kg", "b": ""}) == ["b"]
    assert F._unfilled(F.get("ratio").sentence,
                       {"a": "weight_kg", "b": "height_m"}) == []
    assert F._unfilled(F.get("pca").sentence, {"a": "weight_kg"}) == ["n_components"]


# ── the happy path is unchanged ──────────────────────────────────────────────

def test_the_happy_path_still_reads_as_the_full_methods_sentence():
    """The sentence the product owner should have seen, fixed as a literal."""
    spec = F.declare("bin_kmeans", ["weight_kg"], {"n_bins": 5})
    assert spec["sentence"] == (
        "`weight_kg` will be grouped into 5 clustered bins, fitted within each "
        "training fold.")

    receipt = F.apply(study(), "ratio", ["weight_kg", "height_m"])["receipt"]
    assert receipt["sentence"] == (
        "The ratio `weight_kg / height_m` was computed row by row; rows where "
        "`height_m` is zero are undefined and become missing.")


@pytest.mark.parametrize("key", _ALL_KEYS, ids=_ALL_KEYS)
def test_every_sentence_names_the_column_it_is_about(key):
    """A composed sentence that does not name its subject is under-specified.

    `pca` is the exception and says so: it names components rather than the
    column, which is why `_deferred_names` gives it `pc1`/`pc2` and not the
    column's name.
    """
    t = F.get(key)
    cols = _columns_for(t)
    said = _sentences_from_every_call_site(key, _params_for(t))["preview"]
    if key == "pca":
        assert said.startswith("2 principal components")
        return
    for col in cols:
        assert col in said, f"{key} composed {said!r} without naming {col!r}"


# ── the refusal reaches the step that records the decision ───────────────────

def test_the_refusal_reaches_the_recording_step_and_nothing_is_written():
    """The consumer, not the module in isolation.

    `project.defer_feature` is what the page's "Record it" button reaches. The
    assertions observe the consequence the name claims: the refusal arrives as
    a `ProjectError`, the deferred list stays empty, and NO decision is
    recorded — because a decision sentence carrying `{n_bins}` is what reaches
    the manuscript.
    """
    p = AnalysisProject.from_dataframe(study(), "t")

    with pytest.raises(ProjectError) as caught:
        p.defer_feature("bin_kmeans", ["weight_kg"])
    assert _PLACEHOLDER.search(str(caught.value)) is None
    assert "n_bins" in str(caught.value)
    assert p.deferred_transforms == [], (
        "a transform that could not be described was recorded anyway")
    assert p.decisions == [] or all(
        d.kind != "defer_feature" for d in p.decisions), (
        "a decision was written for a transform the app refused to describe")

    decision = p.defer_feature("bin_kmeans", ["weight_kg"], {"n_bins": 5})
    assert _PLACEHOLDER.search(decision.text) is None
    assert decision.text == p.deferred_transforms[-1]["sentence"], (
        "the recorded decision and the spec disagree about the sentence")
    assert "5 clustered bins" in decision.text
