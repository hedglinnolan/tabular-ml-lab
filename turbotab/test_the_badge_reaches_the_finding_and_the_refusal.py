"""`GUIDED-059` — the badge reaches the claim, not only the prior.

`DOMAIN_SCIENCE.md` §01.1 asks for the epistemic status of **every claim the app
makes**. Before this, `Prior` enforced a badge and nothing else did:

* `packs._finding()` emitted `source="pack"`, a marker, and no status and no
  citation — so every finding four nutrition detectors and eight pack detectors
  produced went out unbadged.
* The four `PrevalenceRefusal` cases carried a message and an offer and no badge
  at all, which made **the pack's highest-stakes statements its only unbadged
  ones.**
* `ATWATER_EVIDENCE` and `DESIGN_EVIDENCE` were constructed and attached to
  nothing. The only reference to either was a test asserting the constants were
  well-formed — a guard testing its own description rather than the app, which
  is the class `README.md` counts and this was the seventh instance of.
* `docs/turbotab/tools/evidence.py` walked `PACKS[*].priors` and scanned one
  file, so `nutrition.py` was outside both the badge walk and the
  `[verify-at-build]` literal scan, and the gate printed a green tick on the
  commit that introduced all of the above.

The tests below are ordered by what a revert would break first, and every one of
them asserts against a **produced** finding or a **raised** refusal rather than
against a constant's shape. That distinction is the whole finding.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import nutrition as N                                  # noqa: E402
from turbotab import packs as P                                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "docs" / "turbotab" / "tools"


def _resolves(source: str) -> bool:
    filename, _, section = (source or "").partition("#")
    path = ROOT / "docs" / "turbotab" / filename
    if not filename or not section or not path.exists():
        return False
    headings = {m.group(1).strip() for m in
                re.finditer(r"^#{1,6}\s+(.*?)\s*$",
                            path.read_text(encoding="utf-8"), re.M)}
    return section in headings


def _nhanes(n: int = 220) -> pd.DataFrame:
    """An NHANES-shaped dietary table with a real Atwater violation in it."""
    rng = np.random.default_rng(11)
    protein = rng.uniform(40, 120, n)
    carb = rng.uniform(150, 350, n)
    fat = rng.uniform(40, 110, n)
    kcal = 4 * protein + 4 * carb + 9 * fat
    # Half the rows in kilojoules — the multi-source merge, which is the one
    # finding this pack raises at `critical`.
    kcal[: n // 2] *= 4.184
    return pd.DataFrame({
        "DR1TKCAL": kcal, "DR1TPROT": protein, "DR1TCARB": carb,
        "DR1TTFAT": fat,
        "WTDRD1": rng.uniform(1000, 90000, n),
        "WTMEC2YR": rng.uniform(1000, 90000, n),
        "SDMVSTRA": rng.integers(1, 6, n),
        "SDMVPSU": rng.integers(1, 3, n),
    })


# ── the findings ────────────────────────────────────────────────────────────

def test_every_finding_the_nutrition_pack_produces_carries_a_resolvable_badge():
    """The claim, asserted against what the detectors actually emitted.

    Not against `ATWATER_EVIDENCE` being well-formed — that was true the whole
    time it reached nothing.
    """
    df = _nhanes()
    produced = [f for f in [N.atwater_finding(df)] if f] + N.design_findings(df)
    assert len(produced) >= 2, "the fixture stopped producing findings"
    for finding in produced:
        badge = finding.get("evidence")
        assert badge, f"{finding['id']} is unbadged"
        assert badge["evidence_status"] in P.EVIDENCE_STATUSES, finding["id"]
        assert _resolves(badge["source"]), (finding["id"], badge["source"])


def test_the_badge_never_overwrites_the_layer_the_finding_came_from():
    """`source` on a finding names the LAYER — `structure`, `profile`, `pack` —
    and `ml.router._defer_target` routes on it. `Evidence.to_dict()` emits a
    `source` too, meaning the citation. Spreading one into the other would have
    quietly repurposed a field the router reads, so the badge is nested."""
    df = _nhanes()
    finding = N.atwater_finding(df)
    assert finding["source"] == "pack"
    assert finding["evidence"]["source"].startswith("research/")


def test_every_pack_detector_badges_what_it_finds():
    """Wider than nutrition: every detector on every pack, over the fixtures
    that make each one fire. A sweep that stopped at the module the finding was
    filed against would be `LOOP.md` §06.5 — a sweep terminating where the
    sweeper's attention ended."""
    fixtures = sorted((Path(__file__).parent / "sample_data").glob("*.csv"))
    assert fixtures, "no fixtures; the sweep asserts nothing"
    seen = 0
    for path in fixtures:
        try:
            df = pd.read_csv(path)
        except Exception:                                  # pragma: no cover
            continue
        for finding in P.findings(df, list(P.LENS_KEYS)):
            seen += 1
            badge = finding.get("evidence")
            assert badge, f"{path.name}: {finding['id']} is unbadged"
            assert _resolves(badge["source"]), (finding["id"], badge["source"])
    assert seen, "no pack detector fired on any fixture; the sweep is empty"


def test_a_finding_cannot_be_made_without_a_badge():
    """Structural, in the same shape guard #2 is: a property of the data model
    rather than of anybody's restraint."""
    with pytest.raises(P.PackError, match="states where the field stands"):
        P._finding("pack::test::x", "info", "t", "d", "w",
                   confidence="high", pack=P.DIETARY, marker="derived")


def test_a_derived_finding_cannot_rest_on_a_convention():
    """The compatibility table reaches findings too, not only priors."""
    with pytest.raises(P.PackError, match="disagree"):
        P._finding(
            "pack::test::x", "info", "t", "d", "w",
            confidence="high", pack=P.DIETARY, marker="derived",
            evidence=P.Evidence(
                status=P.CONVENTION_STATUS,
                source="research/NUTRITION_PACK.md#02 · Implausible intake exclusions"))


# ── the refusals ────────────────────────────────────────────────────────────

REFUSAL_CASES = [
    ("fiber", N.USUAL_INTAKE, "AI"),
    ("calcium", N.USUAL_INTAKE, "RDA"),
    ("calcium", N.SINGLE_DAY, "EAR"),
    ("calcium", N.NAIVE_MEAN, "EAR"),
]


@pytest.mark.parametrize("nutrient,basis,kind", REFUSAL_CASES)
def test_every_refusal_says_where_the_field_stands(nutrient, basis, kind):
    """*"Nobody can compute this, not the app and not you with a spreadsheet"*
    is the strongest sentence this pack says, and it was the only kind of
    sentence going out with no citation."""
    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy(nutrient, basis=basis, reference_kind=kind)
    refusal = caught.value
    assert isinstance(refusal.evidence, P.Evidence)
    payload = refusal.to_dict()
    assert payload["refused"] is True
    assert payload["evidence_status"] in P.EVIDENCE_STATUSES
    assert _resolves(payload["source"]), payload["source"]
    assert payload["reason"] == str(refusal)
    assert payload["offer"], "the refusal serializes without its offer"


def test_a_refusal_cannot_be_raised_without_a_badge():
    with pytest.raises(P.PackError, match="states where the field stands"):
        raise P.PackRefusal("a refusal with no citation behind it")


# ── the gate ────────────────────────────────────────────────────────────────

def _tool():
    sys.path.insert(0, str(TOOLS))
    import importlib
    return importlib.import_module("evidence")


def test_the_gate_opens_the_files_the_claims_live_in():
    """The file-scope half. `PACKS = turbotab/packs.py` meant `nutrition.py` and
    `figure_specs.py` were outside the scan entirely, and the module docstring
    named the gate as the guarantor of numbers it had never read."""
    emitters = {p.name for p in _tool().call_sites()[1]}
    assert {"packs.py", "nutrition.py", "figure_specs.py"} <= emitters, emitters


def test_no_finding_or_refusal_anywhere_is_emitted_without_a_badge():
    """The static half, which catches the detector that never fires on a
    fixture — precisely the one whose missing badge nobody would notice."""
    problems, _, n_calls = _tool().call_sites()
    assert n_calls >= 15, f"only {n_calls} call sites found; the walk is wrong"
    assert problems == [], "\n".join(problems)


def test_the_gate_exits_zero_and_still_states_its_own_limit():
    tool = _tool()
    assert tool.check() == 0
    text = (TOOLS / "evidence.py").read_text(encoding="utf-8")
    assert "does not check the claim is faithful to it" in text
    # And the limit this loop added, which is the more useful one: a threshold
    # the research never mentions is marked nowhere and is invisible here.
    assert "invented" in text and "GUIDED-061" in text
