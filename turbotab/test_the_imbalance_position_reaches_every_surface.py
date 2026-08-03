"""L43-B · §A5.2, the flagship warning, checked against shipped code.

`research/CLINICAL_SURVEY_PACK.md` §A5.2 is the starred one, and its position
is **[SETTLED]**:

> *"Random undersampling, random oversampling and SMOTE all led to poor
> calibration — strong overestimation of the probability of belonging to the
> minority class — without improving discrimination. Any apparent gain in
> sensitivity/specificity was reproducible simply by shifting the
> classification threshold."*
> (van den Goorbergh et al., *JAMIA* 2022;29:1525; replicated for ML methods by
> Carriero et al., *Stat Med* 2025.)

> *"Rare outcomes do create a real problem — but it is small-sample
> overfitting, not imbalance per se, and the remedy is penalization (ridge,
> LASSO, Firth's correction for separation) and adequate sample size, not
> resampling."*

**`GUIDED-049` already filed this and is marked `FIXED`.** `ml/imbalance_advice`
holds the correct position, with the citation, the badge, and the routing by
purpose. What this file found is that the fix reached **three** call sites and
**six** others were never routed:

| surface | what it said |
|---|---|
| `ml/publication.py` | the removed sentence, **verbatim**, in the second methods generator |
| `pages/06` toggle | *"Enable class weighting (recommended)"*, default **on**, over *"Without correction, models will favor the majority class"* |
| `pages/06` diagnostics | *"consider class weights"* |
| `ml/model_coach.py` | *"Enable class weighting"* in the headline and three more places |
| `ml/eda_actions.py` | *"consider class weighting"* |
| `pages/02_EDA.py` | `recommended_action="Use class weighting or stratified sampling"` |

The `publication.py` one is the serious one and it is the same defect
`GUIDED-049` was filed against, arriving by a second route:
`pages/10_Report_Export.py` falls through to `generate_methods_section` whenever
the provenance singleton is empty, and the app went on endorsing rebalancing in
**the artifact that is the product**.

**The shelf is not shortened.** The toggle stays — rebalancing is defensible
for a classifier read at a fixed operating point, and this app cannot yet tell
one from a risk model. What changed is that it is *offered with the citation*
rather than recommended, and no longer defaults on. `imbalance_advice`'s own
docstring settled that trade-off before this loop; the surfaces just never
asked it.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from ml import imbalance_advice as IA

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Every shipped surface that may talk about imbalance. Derived at import from
#: the tree rather than listed, so a NEW surface is swept the day it lands —
#: the adjudicator's stated check on the L42 sweep was whether the enumeration
#: was derived, and the same standard applies here.
SHIPPED = sorted(
    p for p in [*ROOT.glob("ml/*.py"), *ROOT.glob("pages/*.py"),
                *ROOT.glob("turbotab/*.py"), *ROOT.glob("utils/*.py")]
    if not p.name.startswith("test_")
)

#: Phrases that RECOMMEND the contraindicated step. Not "mentions class
#: weighting" — `imbalance_advice` itself has to name the thing it is
#: contraindicating, and so does any honest disclosure. What is forbidden is
#: the imperative and the endorsement.
RECOMMENDING = (
    re.compile(r"\benable class weight", re.I),
    re.compile(r"\bconsider class weight", re.I),
    re.compile(r"\buse class weight", re.I),
    re.compile(r"\bapply(?:ing)? SMOTE\b", re.I),
    re.compile(r"\bconsider SMOTE\b", re.I),
    re.compile(r"class weighting \(recommended\)", re.I),
    re.compile(r"to address class imbalance", re.I),
    re.compile(r"without correction, models will", re.I),
)

#: `imbalance_advice` is the one module allowed to state the contraindicated
#: forms, because stating them is its job.
OWNER = "ml/imbalance_advice.py"


def _lookup_keys(path: pathlib.Path) -> set:
    """Strings that are MATCHED ON rather than said.

    `turbotab/actions.py` is a routing table: `_mark("Use class weights in
    training", …)` registers what to do *when some other advisor emits that
    phrase*. The phrase is a key. Reading it as advice is the same mistake as
    reading a `case` label as a statement, and it is the exact false positive
    that makes a guard get deleted — `GUIDED-140`'s ruling puts the bar at
    zero of them.

    So this is a structural exclusion, not an exemption list: the first
    positional argument of `_mark`/`_op` is a key by construction, and nothing
    has to be named for it to apply. What the user actually reads from that
    table — the `label` and the `why` — is still swept.
    """
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:                                      # pragma: no cover
        return set()
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") not in ("_mark", "_op"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
    return keys


def _prose(path: pathlib.Path) -> str:
    """Source with comments stripped, so a comment recording what a line USED
    to say is not read as the line still saying it.

    Line-based and deliberately crude: a `#` inside a string literal loses a
    tail. That direction is safe here — it can only *hide* text from the
    detector on a line that already contains a `#`, and every offender found
    was a plain user-facing string. `GUIDED-140` records why a real
    comment/string tokenizer is its own piece of work.
    """
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("  # ")[0])
    return "\n".join(out)


def test_every_surface_this_check_reads_actually_parses():
    """**This check reads source as TEXT, and a file that does not parse is
    still text.**

    L43-B shipped an `IndentationError` in `ml/eda_actions.py` — my own edit,
    committed green. The five pre-commit gates do not parse that file, and
    `_prose()` reads it with `read_text`, so nothing in the loop that touched
    it noticed. `tests/` caught it three commits later at collection.

    A guard that reads a file as a string owes an assertion that the string is
    a program. Cheap, and it turns a class of edit mistake into a fast failure
    at the place the edit was made.
    """
    import ast

    broken, parsed = {}, 0
    for path in SHIPPED:
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            parsed += 1
        except SyntaxError as exc:
            broken[str(path.relative_to(ROOT))] = f"line {exc.lineno}: {exc.msg}"

    # THE POSITIVE CONTROL, and the standing absence check demanded it —
    # correctly. Everything below is *nothing was broken*, which passes
    # hardest on an empty sweep. `SHIPPED` globs four directories; if the
    # glob ever stops matching, this fails here instead of reporting a clean
    # tree it never read.
    assert parsed > 40, (
        f"only {parsed} shipped files parsed, and there were well over a "
        f"hundred when this was written — the sweep has stopped sweeping, so "
        f"its silence means nothing")
    assert any(p.name == "imbalance_advice.py" for p in SHIPPED), (
        "the module that owns the position is not in the swept set")

    assert not broken, (
        f"these do not parse: {broken}. Every check in this file reads them as "
        f"text, so its verdict over them means nothing.")


def test_no_shipped_surface_recommends_rebalancing():
    """The standing check, and it is the whole of §A5.2 as a behavior.

    A tool that ships SMOTE in the default pipeline "is broadcasting that it
    does not know the clinical prediction literature", and the research file
    calls that an *inverted* embarrassment risk — **this is a differentiator**.
    Recommending `class_weight='balanced'` is the same claim in a milder form:
    it distorts predicted probabilities the same way, without improving
    discrimination.
    """
    offenders = {}
    for path in SHIPPED:
        rel = str(path.relative_to(ROOT))
        if rel == OWNER:
            continue
        prose = _prose(path)
        keys = _lookup_keys(path)
        hits = sorted({m.group(0) for rx in RECOMMENDING
                       for m in rx.finditer(prose)
                       if not any(m.group(0).lower() in k.lower() for k in keys)})
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        f"these recommend the step §A5.2 says damages calibration without "
        f"improving discrimination: {offenders}. The position, the citation "
        f"and the routing by purpose already live in `{OWNER}` — "
        f"`GUIDED-049` put them there. Route the surface through it rather "
        f"than restating the advice.")


def test_the_detector_would_see_the_sentences_it_was_written_for():
    """The positive control. Every assertion above is an absence claim, so a
    detector that matched nothing would report a clean tree it never read.

    These are the exact strings that shipped, quoted from the diff.
    """
    was_shipped = [
        "To address class imbalance, class_weight='balanced' was applied",
        "Enable class weighting (recommended)",
        "Without correction, models will favor the majority class.",
        "Enable class weighting and judge models by AUROC/F1",
        "Class imbalance detected (ratio: 0.30) - consider class weighting",
        "Use class weighting or stratified sampling",
        "consider class weights, or collect more minority samples.",
    ]
    for sentence in was_shipped:
        assert any(rx.search(sentence) for rx in RECOMMENDING), (
            f"the detector does not match a sentence that actually shipped: "
            f"{sentence!r} — so its silence over the tree means nothing")


def test_the_detector_does_not_fire_on_an_honest_disclosure():
    """The negative control, and it is the one that makes the check usable.

    A guard that cannot tell *"enable class weighting"* from *"rebalancing is
    contraindicated"* forces every honest disclosure to avoid the words, which
    is how a guard gets deleted. `GUIDED-140`'s ruling names zero false
    positives as the acceptance bar for exactly this reason.
    """
    honest = [
        IA.CONTRAINDICATED.format(citation=IA.CITATION),
        IA.UNANSWERED.format(citation=IA.CITATION),
        IA.manuscript_sentence("prediction"),
        IA.manuscript_sentence(None),
        "Rebalancing is contraindicated for a risk model",
        "Sets class_weight='balanced' for supported models.",
    ]
    for sentence in honest:
        hit = [rx.pattern for rx in RECOMMENDING if rx.search(sentence)]
        assert not hit, (
            f"the detector fires on an honest disclosure: {sentence[:70]!r} "
            f"matched {hit}. A guard that cannot tell a warning from a "
            f"recommendation is one the next person deletes.")


# ═══════════ THE SURFACES, ONE AT A TIME ═══════════

def test_the_second_methods_generator_no_longer_endorses_it(monkeypatch):
    """`ml/publication.py` — the serious one.

    This is character-for-character the sentence `GUIDED-049` removed from
    `ml/narrative_engine.py`, still shipping from the generator
    `pages/10_Report_Export.py` falls through to when the provenance singleton
    is empty. Unconditional, approving, no limitation, no citation — the app
    endorsing a contraindicated step in the artifact that is the product.

    **Driven through a controlled log, and a revert probe is why.** The first
    version of this test called the generator with the flag in
    `manuscript_context` and asserted the sentence was absent — and the
    sentence was absent, because `logged_steps` comes from a module-level
    singleton and the branch never ran at all. Reverting the fix came back
    `GREEN — NOT LOAD-BEARING`: the test asserted the absence of a string that
    was missing for an unrelated reason. So the log source is replaced and the
    branch is genuinely entered.
    """
    from ml import publication

    log = {"Model Training": [
        {"step": "Model Training", "action": "Trained models",
         "details": {"class_weight_balanced": True}}]}
    monkeypatch.setattr(publication, "generate_methods_from_log", lambda: log)

    text = publication.generate_methods_section(
        data_config={}, preprocessing_config={}, model_configs={},
        split_config={}, n_total=300, n_train=200, n_val=0, n_test=100,
        feature_names=["a", "b"], target_name="y", task_type="classification",
        metrics_used=["roc_auc"],
        manuscript_context={"model_purpose": "prediction"},
    )

    # THE BRANCH RAN. Without this the assertions below are the same vacuum
    # the probe caught: they would pass on an empty string.
    assert "class_weight='balanced'" in text, (
        "the class-weighting branch did not run, so nothing below this line "
        "is a claim about it")

    assert "To address class imbalance" not in text, (
        "the endorsing sentence GUIDED-049 removed is still generated here")
    assert IA.CITATION in text, "the sentence ships without its citation"
    assert "limitation" in text, "the sentence ships without the limitation"


def test_the_flag_this_reads_is_the_one_production_writes():
    """Trap #3's rule: when a test hands a collaborator a key that stands for
    a real thing, assert the key resolves in the real producer.

    The test above supplies `details['class_weight_balanced']`. If production
    wrote a different key the test would be guarding a shape nothing emits —
    which is exactly how the companion-admissibility check looked enforced for
    six loops.
    """
    page = (ROOT / "pages" / "06_Train_and_Compare.py").read_text(
        encoding="utf-8")
    assert "'class_weight_balanced':" in page, (
        "pages/06 no longer writes `class_weight_balanced` into its "
        "methodology log, so the generator branch this file tests is fed by "
        "nothing and the test above stands for no production shape")
    assert "log_methodology" in page or "class_weight_balanced=" in page, (
        "the flag is written but not into a methodology log entry")


def test_the_manuscript_sentence_says_what_was_done_without_approving_it():
    """And the replacement is not merely subtractive.

    The reader has to know rebalancing was applied — removing the sentence
    outright would be the app going silent about a thing that changes how its
    numbers should be read. `imbalance_advice.manuscript_sentence` says it,
    with the limitation and the citation.
    """
    said = IA.manuscript_sentence("prediction")
    assert "class_weight='balanced') was applied" in said, (
        "the sentence no longer says what was done")
    assert "limitation" in said, "the sentence no longer carries the limitation"
    assert IA.CITATION in said, "the sentence no longer carries the citation"
    assert "To address" not in said, "the sentence is endorsing again"

    inference = IA.manuscript_sentence("inference")
    assert "intercept" in inference, (
        "under an association objective the intercept consequence is the "
        "specific harm, and §A5.2 names it")


def test_the_coach_names_penalization_as_the_remedy_not_reweighting():
    """`ml/model_coach.py` — four sites, one on the *headline* for an
    imbalanced profile.

    §A5.2's nuance is the load-bearing part: a rare outcome IS a real problem,
    and the remedy is penalization and adequate sample size. Deleting the
    advice without saying what to do instead would leave a user with a real
    problem and no move.
    """
    source = _prose(ROOT / "ml" / "model_coach.py")
    assert "class weighting" not in source.lower(), (
        "the coach still recommends reweighting")
    assert "penaliz" in source.lower(), (
        "the coach removed the wrong advice and offered nothing in its place, "
        "which leaves a real problem unanswered")


@pytest.mark.parametrize("path,forbidden", [
    ("pages/06_Train_and_Compare.py", "(recommended)"),
    ("pages/02_EDA.py", "Use class weighting or stratified sampling"),
    ("ml/eda_actions.py", "consider class weighting"),
])
def test_each_streamlit_surface_stops_recommending_it(path, forbidden):
    """Classic is never retired, so a defect present there stays a defect —
    `resolved-in-core` is not closure. All three are Streamlit surfaces and all
    three shipped the recommendation."""
    assert forbidden not in _prose(ROOT / path), (
        f"{path} still recommends rebalancing: {forbidden!r}")


def test_the_train_toggle_is_offered_rather_than_recommended():
    """The one that actually *applies* the correction, and the one where the
    shelf could most easily have been shortened.

    `imbalance_advice`'s docstring settled this before this loop: the
    capability is not deleted, because for a classifier read at a fixed
    operating point rebalancing is a defensible way to move the point. It is
    offered with the citation shown, never recommended, and never written into
    the manuscript without the qualification.
    """
    source = _prose(ROOT / "pages" / "06_Train_and_Compare.py")
    assert "use_class_weight" in source, (
        "the toggle is gone — that is the shelf being shortened, and "
        "`PRODUCT_VISION.md` rules that judgment renders as a stated basis, "
        "never as absence")
    assert 'value=False,\n            key="use_class_weight"' in source, (
        "the toggle still defaults ON; §A5.2 is the flagship warning and a "
        "default-on rebalance is the inverted embarrassment risk it names")
    assert "imbalance_advice" in source, (
        "the page states its own advice instead of asking the module that "
        "owns the position — which is how six surfaces drifted from one fix")


def test_the_position_is_never_that_rebalancing_is_recommended():
    """`advice()` returns `recommended: False` under every purpose, including
    the unrecorded one. Pinned because it is the field every caller reads, and
    a future purpose that flipped it would silently re-endorse the step
    everywhere at once."""
    for purpose in ("prediction", "inference", None, "", "exploration"):
        got = IA.advice(purpose)
        assert got["recommended"] is False, purpose
        assert IA.CITATION in got["advisory"], purpose
        assert got["evidence_status"] == "SETTLED", purpose
        assert got["instead"], (
            f"{purpose}: contraindicated with nothing offered instead is a "
            f"refusal, and §A5.2 has a positive answer to give")
