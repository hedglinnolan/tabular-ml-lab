"""L53-A — two modules that were finished and had no consumer.

`ml/sample_size_claim.py` (`AUDIT-022`) and `ml/candidate_predictors.py`
(`AUDIT-023`) were written at L51 and never imported by anything. **That was not
trap #1**, and the adjudicator's correction is worth carrying: trap #1 is
dangerous when the ROW READS CLOSED, and both rows read `OPEN`, so the ledger
was telling the truth. What existed was finished logic for two open rows,
blocked from its wiring site by a fan-out partitioned by row instead of by fix
site.

## `AUDIT-022` — a bullet that says the same thing whatever the number

The Strengths list printed `f"Sample size of {analysis_total:,} observations"`
behind `if analysis_total > 0`. **Forty rows and forty thousand earned the same
bullet**, under a heading asserting the item is a methodological strength — and
the limitations half, drawn from the EDA ledger, could print *"Sample size may
be insufficient (40 rows…)"* in the same document. The strengths half consulted
nothing.

**Corrected, not deleted.** The count is still in the document either way. What
changed is that the bullet now states **which check was run** and appears under
Strengths only where that check was favorable; where it was not, the same
sentence — composed once, in the module — moves to Limitations.

## `AUDIT-023` — the denominator was the survivors

`regime.n_features` counts the columns currently in `feature_cols`, which after
a selection on page 04 is **what survived**, and the sentences around it call
them *candidate predictors*. §A5.4's ⚠ clause is explicit that a predictor
counts toward sample size even when it is later dropped, because it was looked
at. A 40-candidate study that kept 8 reported a five-times-better ratio than it
earned, under the word *candidate*.

## What these tests do NOT cover

**Neither page is driven end to end here.** Streamlit page drives live in
`tests/integration/conftest.py`'s `AppTest` harness and both of these sites sit
deep inside report composition that needs a fitted run. What is asserted is the
composed string and the module contract that produces it — a **composition
test**, not a page drive, which is the distinction `AUDIT-025`'s L52 sibling
recorded and it is repeated here rather than blurred. The consumer half is
asserted structurally: the import exists and the page calls it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ── the consumer half ────────────────────────────────────────────────────────

@pytest.mark.parametrize("module,page", [
    ("ml.sample_size_claim", "pages/10_Report_Export.py"),
    ("ml.candidate_predictors", "pages/02_EDA.py"),
])
def test_the_module_has_a_consumer_outside_its_own_tests(module, page):
    """A capability ships with its consumer. These shipped without one."""
    leaf = module.split(".")[-1]
    importers = []
    for path in sorted(ROOT.rglob("*.py")):
        if ".worktrees" in str(path) or "venv" in str(path):
            continue
        if path.name.startswith("test_") or path.name == f"{leaf}.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(rf"(from\s+{re.escape(module)}\s+import|import\s+{re.escape(module)}\b)", text):
            importers.append(str(path.relative_to(ROOT)))
    assert importers, (
        f"{module} is imported by nothing outside its own tests. It was "
        f"written for an OPEN row and left unwired for a loop, which is the "
        f"cheapest kind of unfinished work to miss")
    assert page in importers, (
        f"{module} has importers {importers} but not {page}, which is the "
        f"site the row names")


# ── `AUDIT-022` ──────────────────────────────────────────────────────────────

def test_the_same_count_is_not_a_strength_at_every_size():
    """The row's own sentence: forty rows and forty thousand differed in nothing."""
    from ml.sample_size_claim import sample_size_claim

    small = sample_size_claim(40, sufficiency="critical", n_candidate_predictors=8)
    large = sample_size_claim(40_000, sufficiency="abundant", n_candidate_predictors=8)
    assert small is not None and large is not None
    assert large.is_strength, "an abundant sample earns the strengths heading"
    assert not small.is_strength, (
        "40 observations rated `critical` is printed as a methodological "
        "STRENGTH, which is the bullet this row was filed for")
    assert small.text != large.text, "the two studies get the identical sentence"
    assert "40" in small.text, (
        "the count vanished from the unfavorable branch. The shelf is never "
        "shortened — it moves to Limitations, it does not disappear")


def test_an_unevaluated_sufficiency_is_not_read_as_a_pass():
    """No verdict is not a good verdict. Ambiguity is never a strength."""
    from ml.sample_size_claim import sample_size_claim

    for absent in (None, "", "   "):
        claim = sample_size_claim(5_000, sufficiency=absent)
        assert claim is not None and not claim.is_strength, (
            f"sufficiency={absent!r} produced a strengths bullet, so a study "
            f"where the check never ran reads as one where it passed")
        assert "5,000" in claim.text


def test_the_page_prints_the_count_under_one_heading_or_the_other():
    """Corrected, not deleted — asserted on the page's own control flow.

    The bullet appears under Strengths when the claim is one and under
    Limitations when it is not. A version that only appended to `strength_items`
    would silently drop the count from every unfavorable study.
    """
    src = _source("pages/10_Report_Export.py")
    assert "strength_items.append(_claim.text)" in src
    assert "limitation_items.append(_claim.text)" in src, (
        "the count is printed only when it is a strength, so an insufficient "
        "study loses its sample size from the document entirely")
    # AND THE SENTENCE IS QUOTED RATHER THAN RE-COMPOSED. A page that rebuilt
    # the wording would be a second copy of the claim.
    assert not re.search(r'f"Sample size of \{analysis_total', src), (
        "the page still composes its own sample-size sentence beside the "
        "module's, which is two authors for one claim")


# ── `AUDIT-023` ──────────────────────────────────────────────────────────────

def test_the_denominator_is_what_was_screened_not_what_survived():
    """§A5.4's ⚠ clause, which is the whole row."""
    from ml.candidate_predictors import candidate_count, candidate_phrase

    class _Record:
        candidates_screened = [f"x{i}" for i in range(40)]
        n_features_before = 40

    class _Prov:
        feature_selection = _Record()

    kept = [f"x{i}" for i in range(8)]
    count = candidate_count(kept, _Prov())
    assert count.screened == 40, (
        f"the count is {count.screened}; 40 columns were screened and 8 kept, "
        f"and a predictor counts toward sample size even when it is dropped")
    assert count.retained == 8 and count.dropped == 32
    phrase = candidate_phrase(count)
    assert "40 candidate predictors" in phrase and "8" in phrase, phrase


def test_nothing_dropped_does_not_advertise_a_screening_that_never_ran():
    """The negative control, and it is the one that stops this being noise."""
    from ml.candidate_predictors import candidate_count, candidate_phrase

    count = candidate_count([f"x{i}" for i in range(8)], None)
    assert count.screened == count.retained == 8
    phrase = candidate_phrase(count)
    assert phrase == "8 candidate predictors", phrase
    assert "retained" not in phrase, (
        "with no selection recorded the phrase advertises a screening step, "
        "which sends a reader looking for a stage that did not happen")


def test_the_eda_sentence_and_the_ratio_beside_it_count_the_same_thing():
    """A first draft moved the RATIO to the screened set and left the FINDING
    on `regime.n_features`, so one sentence would have read *'8 features,
    5.00:1 samples per feature'* where 5.00 was 40/8. Two numbers in one
    sentence describing different denominators is the defect one level down
    from the one being fixed.
    """
    src = _source("pages/02_EDA.py")
    block = src[src.index("def _auto_generate_insights"):]
    block = block[:block.index("_suff_ratio_str") + 2000]
    assert "_cands.screened" in block and "max(_cands.screened, 1)" in block, (
        "the sufficiency ratio is not computed over the screened set")
    assert "{regime.n_features} features, {_suff_ratio_str}" not in block, (
        "the finding still counts kept features beside a ratio computed over "
        "screened ones")
    assert "candidate predictors" in block


def test_the_manuscript_sentences_quote_the_module_rather_than_rebuild_it():
    """One author for one claim. `_cand_text` is the module's phrase."""
    src = _source("pages/02_EDA.py")
    for marker in ("from ml.candidate_predictors import candidate_count",
                   "from ml.candidate_predictors import candidate_phrase",
                   "_cand_text = _cand_phrase(_cands)"):
        assert marker in src, marker
    tree = ast.parse(src)
    assert tree is not None
    # THE POSITIVE CONTROL for the absence check below: the two manuscript
    # sentences this row names are present, so `n_features` being gone from
    # them means something.
    assert src.count("_cand_text") >= 3, (
        "the module's phrase reaches fewer than both manuscript sentences")
    for sentence in ("the sample size was small relative to the number of candidate",
                     "the modest ratio of observations to candidate predictors"):
        assert sentence in src, sentence
        tail = src[src.index(sentence):src.index(sentence) + 400]
        assert "regime.n_features" not in tail, (
            f"this sentence still reports the kept count as candidates: "
            f"{tail[:200]!r}")
