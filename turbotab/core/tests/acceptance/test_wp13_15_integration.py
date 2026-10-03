"""The seams between WP13, WP14 and WP15, as only the merged tree shows them (AUDIT_REPORT §5).

Each package's acceptance tests pass on its own branch. Four things were true of none of the three
branches alone and are held here:

1. **IN-08's dietary hint.** The audit's evidence: ``DR2TKCAL``, ``DRXTKCAL``, ``total_energy``,
   ``kcal_day``, ``TotalKcal`` and ``ENERC_KCAL`` "produce no dietary hint" because the pack matched
   energy by exact alias. WP13 fixed the findings for those names and left the hint to WP14's lens
   work (its deviation); WP14 kept the pack's matcher for the dietary hint. Merged, the hint reads
   the one recognizer. Replayed on H-skeptic/h10.py's tables, draw for draw.
2. **The survey-code finding's evidence.** WP14's finding took ``SENTINEL_EVIDENCE`` from the
   legacy ``turbotab.survey``, which WP15 keeps out of ``turbotab/core`` because it badges the
   polychoric claim SETTLED (ledger row 96). The evidence is restated; it must equal the legacy
   definition (read from its source, not imported) and the ledger's verdict on it (row 8).
3. **An energy-related outcome read by the one tokenizer.** WP15 read outcome names with its own
   tokenizer, which keeps ``BMIChange`` and ``T2DIncident`` whole; WP13's splits a run of capitals
   from the word after it. NUTRITION_PACK §04, Diagnostic: "Detect whether the outcome is itself
   energy-related (weight, BMI, adiposity, diabetes) — if so, escalate the mediation/collider
   warning." The energy column here is ``ENERC_KCAL``, which only WP13's recognizer reads, so the
   finding comes through WP13's path and carries WP15's wording.
4. **Under the genomics lens a normalized expression matrix is not "repeated measures".** With
   WP14's evidence-based hints the word-budget harness checks the genomics sample matrices under
   the genomics lens, where the pack's wide-shape reframing applies only to a raw count matrix: a
   CPM, FPKM, TMM-CPM, VST or estimated-count matrix was told its genes "look like repeated
   measures of one quantity". The fixtures are one 60-sample count matrix put through the named
   transform (``sample_data/make_genomics_siblings.py``): every gene column is a different gene.

Expected values never come from the code under test: the audit's lists and generators, the
legacy source text (parsed, not imported), the claims ledger, the pack's quoted rule, and the
fixtures' own documentation.
"""
from __future__ import annotations

import ast
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState
from turbotab.core.stages.data import profile_stage
from turbotab.core.stages.findings import findings_stage
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

REPO = Path(__file__).resolve().parents[4]


def _write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = folder / name
    frame.to_csv(path, index=False)
    return path


def _hints(path: Path) -> list[dict]:
    return Ingested(path, Path(tempfile.mkdtemp())).run(profile_stage, ProjectState())["lens_hints"]


def _findings(path: Path, lens: list[str], target: str | None = None) -> list[dict]:
    t = Ingested(path, Path(tempfile.mkdtemp()))
    return t.run(findings_stage, ProjectState(lens=lens, target=target))["findings"]


# ── 1 · IN-08: the dietary hint reads the one recognizer ─────────────────────

# H-skeptic/h10.py's names, in its order; the first and ``energy`` the pack already hinted.
H10_NAMES = ["DR1TKCAL", "DR2TKCAL", "DRXTKCAL", "total_energy", "kcal_day", "TotalKcal",
             "ENERC_KCAL", "energy", "energy_kj"]


def _h10_tables():
    """H-skeptic/h10.py's nine tables (``default_rng(10)``, n = 300), draw for draw: the same
    energy under each name (``energy_kj`` × 4.184), beside protein, carbohydrate and fat."""
    rng = np.random.default_rng(10)
    n = 300
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    E = (4 * P + 4 * C + 9 * F).round(0)
    E[:12] = rng.uniform(5100, 6500, 12).round()
    for name in H10_NAMES:
        yield name, pd.DataFrame({name: E if name != "energy_kj" else E * 4.184,
                                  "protein_g": P.round(1), "carb_g": C.round(1),
                                  "fat_g": F.round(1), "age": rng.integers(20, 70, n)})


def test_1a_every_total_energy_name_the_audit_lists_hints_the_dietary_lens(tmp_path):
    """IN-08's six names, h10's ``energy_kj`` and the two names the pack already read: each table
    is hinted dietary, the hint naming its energy column."""
    for name, frame in _h10_tables():
        hints = _hints(_write(frame, tmp_path, f"{name}.csv"))
        dietary = [h for h in hints if h["lens"] == "dietary"]
        assert dietary, (name, hints)
        assert f"`{name}`" in dietary[0]["because"], (name, dietary)


def test_1b_every_alias_the_pack_matched_still_hints_the_dietary_lens(tmp_path):
    """The pack's matcher read the physiology reference's ``kcal`` key and its aliases (the data
    file ``ml/physiology_reference`` loads); a table under any of them is still hinted."""
    from ml.physiology_reference import load_reference_bundle

    spec = load_reference_bundle()["nhanes"]
    spec = spec.get("variables", spec)["kcal"]
    names = ["kcal", *spec["aliases"]]
    assert {"energy", "calories", "dr1tkcal"} <= {n.lower() for n in names}
    rng = np.random.default_rng(0)
    for name in names:
        frame = pd.DataFrame({name: rng.normal(2100, 450, 200).round(),
                              "protein_g": rng.normal(80, 20, 200).round(1)})
        hints = _hints(_write(frame, tmp_path, f"{name}.csv"))
        assert any(h["lens"] == "dietary" for h in hints), (name, hints)


# Energy that is not total energy intake: an expenditure, a resting rate, one macronutrient's own
# energy (the methods gate's ``alc_kcal``), a share and a per-kilogram rate.
NOT_INTAKE = ["energy_expenditure_kcal", "REE_kcal", "alc_kcal", "pct_kcal_fat", "kcal_per_kg"]


@pytest.mark.parametrize("name", NOT_INTAKE)
def test_1c_energy_that_is_not_intake_hints_no_diet(tmp_path, name):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({name: rng.normal(30, 5, 200).round(1),
                          "steps_per_day": rng.integers(2000, 15000, 200),
                          "age": rng.integers(20, 70, 200)})
    hints = _hints(_write(frame, tmp_path, "t.csv"))
    assert not [h for h in hints if h["lens"] == "dietary"], hints


# ── 2 · the survey-code finding's evidence equals the legacy definition ──────


def _legacy_sentinel_evidence() -> tuple[str, str]:
    """``SENTINEL_EVIDENCE`` as ``turbotab/survey.py`` defines it, read from its source text."""
    tree = ast.parse((REPO / "turbotab" / "survey.py").read_text("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SENTINEL_EVIDENCE" for t in node.targets):
            kw = {k.arg: k.value for k in node.value.keywords}
            return kw["status"].id, kw["source"].value
    raise AssertionError("turbotab/survey.py no longer defines SENTINEL_EVIDENCE")


def _five_with_nines() -> pd.DataFrame:
    """H-skeptic/h1.py's 1–5 block with 3% of answers replaced by 9 (seed 11, n = 300), draw for
    draw: the 6- and 7-point blocks are drawn first and discarded."""
    rng = np.random.default_rng(11)
    n = 300
    for i in range(10):
        rng.choice(range(1, 7), n, p=np.array([1, 2, 3, 3, 2, 1.2]) / 12.2)
    for i in range(10):
        rng.choice(range(1, 8), n, p=np.array([10, 18, 22, 22, 17, 7, 4]) / 100)
    five = pd.DataFrame({f"q{i+1}": rng.choice(range(1, 6), n) for i in range(10)})
    return five.mask(rng.random(five.shape) < 0.03, 9)


def test_2_the_restated_sentinel_evidence_is_the_legacy_one_and_the_ledgers(tmp_path):
    from turbotab.core.detectors import scales

    status, source = _legacy_sentinel_evidence()
    assert status == "SETTLED"
    heading = source.split("#", 1)[1]
    pack = (REPO / "docs" / "turbotab" / source.split("#", 1)[0]).read_text("utf-8")
    assert re.search(rf"^#+ {re.escape(heading)}\s*$", pack, re.M), heading
    ledger = (REPO / "docs/turbotab-next/audit/claims-ledger.md").read_text("utf-8")
    row8 = next(line for line in ledger.splitlines() if line.startswith("| 8 |"))
    assert "sentinel" in row8 and "| SETTLED | CONSISTENT |" in row8, row8

    frame = _five_with_nines()
    raw = scales.sentinel_finding(frame, scales.blocks(frame))
    claims = {c["key"]: c for c in raw["evidence"]["claims"]}
    for key in ("must_recode", "never_auto_recode"):
        assert (claims[key]["evidence_status"], claims[key]["source"]) == (status, source), key
    assert (raw["evidence"]["evidence_status"], raw["evidence"]["source"]) == (status, source)
    # Served, the finding carries its weakest claim's badge, as the legacy finding did:
    # "'Don't know' is not automatically the same as missing" is DISPUTED.
    served = next(f for f in _findings(_write(frame, tmp_path, "five.csv"), ["survey"])
                  if f["id"] == "pack::survey::sentinel_codes")
    assert served["evidence"] == {"status": "DISPUTED", "source": source}


# ── 3 · an energy-related outcome, read by the one tokenizer ─────────────────

NUT04 = "research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature"


def _with_outcome(outcome: str, values: np.ndarray) -> pd.DataFrame:
    """h10's ``ENERC_KCAL`` table (the seventh, draw for draw) with an outcome beside it."""
    frame = dict(_h10_tables())["ENERC_KCAL"]
    return frame.assign(**{outcome: values})


@pytest.mark.parametrize("outcome,kind", [("BMIChange", "BMI"), ("T2DIncident", "diabetes"),
                                          ("BodyWeightKg", "body weight")])
def test_3_a_camel_case_energy_related_outcome_marks_the_energy_finding_disputed(
        tmp_path, outcome, kind):
    rng = np.random.default_rng(3)
    values = rng.integers(0, 2, 300) if kind == "diabetes" else rng.normal(0, 1, 300).round(2)
    path = _write(_with_outcome(outcome, values), tmp_path, "t.csv")
    found = {f["id"]: f for f in _findings(path, ["dietary"], outcome)}
    energy = found["pack::dietary::energy_adjustment"]
    assert energy["evidence"] == {"status": "DISPUTED", "source": NUT04}
    assert f"`{outcome}` reads as {kind}" in energy["why_it_matters"]


def test_3_an_outcome_that_is_not_energy_related_keeps_the_convention(tmp_path):
    rng = np.random.default_rng(3)
    path = _write(_with_outcome("LDLChange", rng.normal(0, 1, 300).round(2)), tmp_path, "t.csv")
    found = {f["id"]: f for f in _findings(path, ["dietary"], "LDLChange")}
    energy = found["pack::dietary::energy_adjustment"]
    assert energy["evidence"]["status"] == "CONVENTION"
    assert "reads as" not in energy["why_it_matters"]


# ── 4 · under the genomics lens, genes are not repeated measures ─────────────

# make_genomics_siblings.py's derived matrices, each documented as the counts matrix "put through
# the transform its name claims" (genomics_cpm.csv.md: "rows are samples (60), gene columns 495").
EXPRESSION = ["genomics_cpm.csv", "genomics_estimated_counts.csv", "genomics_fpkm.csv",
              "genomics_tmm_cpm.csv", "genomics_vst.csv", "genomics_expression.csv"]


@pytest.mark.parametrize("name", EXPRESSION)
def test_4a_an_expression_matrix_under_the_genomics_lens_is_not_repeated_measures(name):
    found = {f["id"]: f for f in _findings(SAMPLES / name, ["genomics"])}
    wide = found.get("wide_repeated_measures")
    assert wide is not None, "the engine reads the gene columns as a numbered series"
    assert wide["title"] == "The wide shape is expected here."
    assert "repeated measures" not in wide["summary"]
    assert wide["severity"] == "info" and wide["lever_label"] is None


def test_4b_real_repeated_measures_are_still_named_under_the_genomics_lens():
    """clinic_visits.md: "``bp_1``/``bp_2``/``bp_3`` is a genuine wide repeated-measures family".
    The reframing reads the table, not the lens alone."""
    found = {f["id"]: f for f in _findings(SAMPLES / "clinic_visits.csv", ["genomics"])}
    wide = found["wide_repeated_measures"]
    assert "`bp_1`, `bp_2` and `bp_3` look like repeated measures" in wide["summary"]
