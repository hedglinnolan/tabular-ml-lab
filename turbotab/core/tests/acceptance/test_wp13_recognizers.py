"""WP13 acceptance tests: recognizers read whole tokens, corroborated (AUDIT_REPORT §5).

Closes IN-01, IN-02, IN-05, IN-06, IN-07, IN-08, IN-10 and IN-19, and the methods gate's
``reads_as_total_energy("alc_kcal")`` (a substring hit of the IN-01 kind). The six tests of §5:

1. **Must not be nutrients:** ``fatty_fish_g``, ``fat_mass_kg``, ``body_fat_pct``,
   ``fatigue_score``, ``c_reactive_protein``, ``total_protein``, ``prothrombin_time``,
   ``fibrinogen``, ``fib4_score``, ``bicarbonate``, ``carbamazepine``, ``carbonated_drinks_g``,
   ``alcohol_use_disorder``, ``lipid_lowering_meds``. **Must be recognized:** ``sfa_g``, ``mufa_g``,
   ``pufa_g``, ``DR1TSFAT``, ``DR1TMFAT``, ``DR1TPFAT``, ``DR1TSUGR``, ``DR2TKCAL``, ``DRXTKCAL``,
   ``ENERC_KCAL``, ``PROCNT``, ``CHOAVL``; on the 45-nutrient DR1T* table no nutrient is proposed
   "covariate".
2. **Must stay predictors:** on an RCT table, ``treatment``, ``arm``, ``diet_group``,
   ``condition``, ``phenotype``, ``fasting``; ``steps_per_day``, ``drinks_per_week``,
   ``baseline_glucose``; ``birth_weight`` in grams. Batch is proposed as "acquisition/batch", a
   covariate under inference.
3. **Identifiers:** one recognizer agrees with itself on the 46-name table; ``eid``, ``patid``,
   ``ptid``, ``HHID``, ``IDNO``, ``USUBJID`` are identifiers; ``site_id``, ``household_id`` are
   clusters; no float or measurement column is ever suggested as a unit ID.
4. **Units:** dietary choline (mg/day) is not "mg/dL"; any unit not read from an explicit suffix
   requires a recorded decision before a sentence carries it. Source check: CLINICAL_SURVEY_PACK,
   "TurboTab will not guess".
5. **Energy units:** a kJ "energy" column with only sodium beside it is read as kJ by the
   magnitude prior; the implausible-intake count after conversion matches the kcal screens.
6. **Weights:** with WTSAF2YR and fasting analytes present, the least-common-denominator rule names
   the fasting weight. Source check: NHANES weighting tutorial.

Every fixture replays the audit's own generator draw for draw where one exists (``repro.tar.gz``:
``H-skeptic/h3.py``, ``h4.py``, ``h5b.py``, ``h10.py``, ``h10b.py``, ``h16.py``, ``H/idnames.py``,
``C-skeptic/c4.py``, ``D-skeptic/s10_kj.py``). Expected values never come from the code under test:
they are the audit's lists, the codebooks' own labels (quoted where used), the Atwater factors
(FAO: 4/4/9/7 kcal/g), and counts made with NumPy on the fixtures' own values.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import PREDICTOR_ROLES, ProjectState
from turbotab.core.stages.rows import roles_stage
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

REPO = Path(__file__).resolve().parents[4]
KCAL_PER_KJ = 4.184  # FAO: 1 kcal = 4.184 kJ (NUTRITION_PACK §01, SETTLED)
ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}  # FAO general factors


def _roles(path: Path, *, lens: list[str], target: str | None, purpose: str | None = None) -> dict:
    """The roles stage on ``path``, as a worker runs it: column → proposal."""
    t = Ingested(path, Path(tempfile.mkdtemp()))
    out = t.run(roles_stage, ProjectState(lens=lens, target=target, purpose=purpose))
    return {p["column"]: p for p in out["columns"]}


def _write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = folder / name
    frame.to_csv(path, index=False)
    return path


# ── 1 · nutrients ────────────────────────────────────────────────────────────

MUST_NOT = ["fatty_fish_g", "fat_mass_kg", "body_fat_pct", "fatigue_score", "c_reactive_protein",
            "total_protein", "prothrombin_time", "fibrinogen", "fib4_score", "bicarbonate",
            "carbamazepine", "carbonated_drinks_g", "alcohol_use_disorder", "lipid_lowering_meds"]
# The audit's further probes (IN-01 evidence) and the methods gate's ``alc_kcal``.
ALSO_NOT = ["fat_free_mass", "liver_fat", "visceral_fat_area", "serum_protein_g_dl",
            "carbon_monoxide", "carboxyhemoglobin"]

# Each codebook name with what its documentation says it is. NHANES DR1TOT_L codebook SAS labels:
# DR1TSFAT "Total saturated fatty acids (gm)", DR1TMFAT "Total monounsaturated fatty acids (gm)",
# DR1TPFAT "Total polyunsaturated fatty acids (gm)", DR1TSUGR "Total sugars (gm)". USDA SR27
# NUTR_DEF (INFOODS tagnames): "~203~^~g~^~PROCNT~^~Protein~"; CHOAVL is FAO/INFOODS's available
# carbohydrate. SFA, MUFA and PUFA are fatty acids: fat, at fat's 9 kcal/g.
MUST_BE = {
    "sfa_g": ("fat", "sfa"), "mufa_g": ("fat", "mufa"), "pufa_g": ("fat", "pufa"),
    "DR1TSFAT": ("fat", "sfa"), "DR1TMFAT": ("fat", "mufa"), "DR1TPFAT": ("fat", "pufa"),
    "DR1TSUGR": ("carbohydrate", "sugar"), "PROCNT": ("protein", None),
    "CHOAVL": ("carbohydrate", None),
}
# Total energy: NHANES DR1TOT_L "DR1TKCAL - Energy (kcal)", the day-2 file's DR2TKCAL and the
# 1999–2002 DRXTKCAL (NUTRITION_PACK §01), NUTR_DEF "~208~^~kcal~^~ENERC_KCAL~^~Energy~".
ENERGY_NAMES = ["DR2TKCAL", "DRXTKCAL", "ENERC_KCAL"]


def test_1a_names_that_only_share_letters_with_a_nutrient_carry_no_energy():
    """IN-01: substrings made body fat, CRP, fibrinogen and fatty fish energy-bearing nutrients.
    Expected: the audit's list, none of which is a nutrient intake. ``alc_kcal`` is alcohol's own
    energy, never total energy (the methods gate)."""
    from turbotab.core.methods.energy import energy_factor, nutrient_role, reads_as_total_energy
    from turbotab.core.recognizers import is_nutrient
    from turbotab.core.stages.proposals import energy_bearing

    for name in MUST_NOT + ALSO_NOT:
        assert nutrient_role(name) is None, name
        assert energy_factor(name).factor is None, name
        assert not energy_bearing(name), name
        assert not is_nutrient(name), name
        assert not reads_as_total_energy(name), name
    assert not reads_as_total_energy("alc_kcal")
    assert nutrient_role("alc_kcal") == "alcohol"  # alcohol's energy, already in kcal
    assert energy_factor("alc_kcal").factor == 1.0
    # Whole words, not letters: a food group in grams carries no Atwater factor (fatty fish is
    # about 2 kcal/g, not fat's 9), while the same word as a nutrient does. Nor is the food said
    # to carry no energy: it carries some, by no general factor.
    assert energy_factor("fat_g").factor == ATWATER["fat"]
    assert "carries no energy" not in energy_factor("fatty_fish_g").reason
    assert "a food" in energy_factor("fatty_fish_g").reason


def test_1b_codebook_and_tagname_nutrients_are_recognized_with_their_factors():
    """IN-08: SFA, MUFA, PUFA and the NHANES fat and sugar codes got no Atwater factor and a false
    "carries no energy"; DR2TKCAL, DRXTKCAL and INFOODS tags went unread. Expected: the codebook
    labels above and FAO's factors."""
    from turbotab.core.methods.energy import energy_factor, unit_of
    from turbotab.core.methods.nesting import nested_components
    from turbotab.core.recognizers import read_nutrient, reads_as_total_energy

    for name, (macro, part) in MUST_BE.items():
        reading = read_nutrient(name)
        assert reading is not None and (reading.macro, reading.part) == (macro, part), name
        factor = energy_factor(name)
        assert factor.factor == ATWATER[macro] and factor.declared, (name, factor)
        assert "carries no energy" not in factor.reason
    assert unit_of("DR1TSFAT") == "grams"  # the codebook's "(gm)"
    for name in ENERGY_NAMES + ["DR1TKCAL", "energy_kcal", "TotalKcal", "total_energy", "kcal_day"]:
        assert reads_as_total_energy(name), name
    # Shared with nesting (audit IN-08: the vocabularies had diverged): the NHANES parts nest in
    # the NHANES total, on data where each part is at most the total on every row.
    rng = np.random.default_rng(13)
    total = rng.gamma(9, 9, 200)
    parts = rng.dirichlet([4, 4, 3], 200) * (total[:, None] * 0.9)
    frame = pd.DataFrame({"DR1TTFAT": total, "DR1TSFAT": parts[:, 0], "DR1TMFAT": parts[:, 1],
                          "DR1TPFAT": parts[:, 2]})
    assert nested_components(frame) == {"DR1TSFAT": "DR1TTFAT", "DR1TMFAT": "DR1TTFAT",
                                        "DR1TPFAT": "DR1TTFAT"}


def _diet_and_body_composition(folder: Path) -> Path:
    """H-skeptic/h3.py's diet-plus-body-composition table (``default_rng(7)``, n = 300), draw for
    draw, with the rest of §5's must-not names appended from a stream of their own."""
    rng = np.random.default_rng(7)
    n = 300
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    df = pd.DataFrame({
        "id": np.arange(n), "age": rng.integers(30, 70, n), "sex": rng.choice(["F", "M"], n),
        "energy_kcal": (4 * P + 4 * C + 9 * F).round(0), "protein_g": P.round(1),
        "carbohydrate_g": C.round(1), "fat_g": F.round(1),
        "fat_mass_kg": rng.normal(25, 8, n).round(1), "crp_mg_l": rng.lognormal(0.5, 0.8, n).round(2),
        "c_reactive_protein": rng.lognormal(0.5, 0.8, n).round(2),
        "fibrinogen": rng.normal(320, 60, n).round(0), "hba1c": rng.normal(5.6, 0.5, n).round(1)})
    extra = np.random.default_rng(1307)
    for name in MUST_NOT:
        if name not in df.columns:
            df[name] = extra.gamma(4, 10, n).round(2)
    return _write(df, folder, "diet_bc.csv")


def test_1c_on_a_diet_and_body_composition_table_only_intakes_are_adjusted(tmp_path):
    """IN-01: ``fat_mass_kg``, ``c_reactive_protein`` and ``fibrinogen`` were proposed "exposure
    (high): A nutrient that carries energy" and the residual method regressed CRP on kcal. Expected:
    the energy-bearing nutrients are exactly the three gram macronutrients the fixture built
    energy from (energy_kcal = 4P + 4C + 9F)."""
    from turbotab.core.stages.proposals import build_proposals

    path = _diet_and_body_composition(tmp_path)
    roles = _roles(path, lens=["dietary"], target="hba1c")
    nutrient_reasons = ("A nutrient that carries energy: an exposure under the dietary lens.",
                        "A nutrient intake: an exposure under the dietary lens.")
    for name in MUST_NOT + ["crp_mg_l"]:
        assert roles[name]["reason"] not in nutrient_reasons, (name, roles[name])
        assert "energy" not in roles[name]["reason"].lower(), (name, roles[name])
    # A food group in grams is an intake, said as one, not a nutrient.
    assert roles["fatty_fish_g"]["proposed"] == "exposure"
    assert roles["fatty_fish_g"]["reason"].startswith("An intake by its unit, not a nutrient")
    assert roles["protein_g"]["reason"] == nutrient_reasons[0]
    assert roles["energy_kcal"]["proposed"] == "energy"
    frame = pd.read_csv(path)
    columns = [{"name": c, "dtype": "numeric" if frame[c].dtype.kind in "if" else "categorical",
                "n_unique": int(frame[c].nunique()), "n_missing": 0} for c in frame.columns]
    proposed = {c: p["proposed"] for c, p in roles.items()}
    out = build_proposals(frame, columns, lens=["dietary"], target="hba1c", roles=proposed)
    assert out["energy"]["nutrients"] == ["protein_g", "carbohydrate_g", "fat_g"]
    assert out["energy"]["energy_column"] == "energy_kcal"
    # The exposures left unadjusted are listed with a true reason (IN-01: "show excluded columns
    # with their reason"): a food is not said to carry no energy.
    left = {e["column"]: e["reason"] for e in out["energy"]["not_adjusted"]}
    assert left["fatty_fish_g"] == "a food, not a nutrient: no energy factor unless one is declared"


def _dr1t_table(folder: Path) -> Path:
    """H-skeptic/h10b.py's 45-nutrient DR1T* table (``default_rng(1)``, n = 200), draw for draw."""
    codes = ("KCAL PROT CARB SUGR FIBE TFAT SFAT MFAT PFAT CHOL ATOC ATOA RET VARA ACAR BCAR CRYP "
             "LYCO LZ VB1 VB2 NIAC VB6 FOLA FA FF FDFE CHL VB12 B12A VC VD VK CALC PHOS MAGN IRON "
             "ZINC COPP SODI POTA SELE CAFF THEO ALCO MOIS").split()
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"SEQN": np.arange(n),
                       **{f"DR1T{c}": rng.lognormal(3, .5, n).round(2) for c in codes},
                       "LBXGLU": rng.normal(100, 15, n)})
    return _write(df, folder, "dr1t.csv")


def test_1d_no_nhanes_nutrient_is_proposed_a_covariate(tmp_path):
    """IN-08: on the 45-nutrient DR1T* table 24 nutrients were proposed "covariate (low)".
    Expected: every one of the 45 nutrient codes in the DR1TOT_L codebook is an exposure, and
    DR1TKCAL ("Energy (kcal)") is the energy column."""
    roles = _roles(_dr1t_table(tmp_path), lens=["dietary"], target="LBXGLU")
    nutrients = [c for c in roles if c.startswith("DR1T") and c != "DR1TKCAL"]
    assert len(nutrients) == 45
    assert [c for c in nutrients if roles[c]["proposed"] == "covariate"] == []
    assert {roles[c]["proposed"] for c in nutrients} == {"exposure"}
    assert roles["DR1TKCAL"]["proposed"] == "energy" and roles["SEQN"]["proposed"] == "identifier"


@pytest.mark.parametrize("energy_name", ["DR2TKCAL", "DRXTKCAL", "ENERC_KCAL", "TotalKcal",
                                         "total_energy", "kcal_day"])
def test_1e_energy_the_pack_could_not_name_still_raises_its_findings(tmp_path, energy_name):
    """IN-08: these names produced no energy-adjustment and no implausible-intake finding, because
    the pack matched energy by exact alias. H-skeptic/h10.py's table (``default_rng(10)``): 12 rows
    planted at 5,100–6,500 kcal. Expected count: NumPy, rows outside 500–5,000 kcal."""
    from turbotab.core.stages.findings import findings_stage

    rng = np.random.default_rng(10)
    n = 300
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    E = (4 * P + 4 * C + 9 * F).round(0)
    E[:12] = rng.uniform(5100, 6500, 12).round()
    df = pd.DataFrame({energy_name: E, "protein_g": P.round(1), "carb_g": C.round(1),
                       "fat_g": F.round(1), "age": rng.integers(20, 70, n)})
    path = _write(df, tmp_path, "energy.csv")
    t = Ingested(path, tmp_path)
    found = {f["id"]: f for f in t.run(findings_stage, ProjectState(lens=["dietary"]))["findings"]}
    assert "pack::dietary::energy_adjustment" in found
    implausible = found["pack::dietary::implausible_intake"]
    expected = int(((E < 500) | (E > 5000)).sum())
    assert expected == 12
    assert implausible["title"].startswith(f"{expected:,} records report")
    assert implausible["affected_columns"] == [energy_name]


# ── 2 · roles: what must stay a predictor ────────────────────────────────────

LENSES = (["clinical"], ["dietary"], ["metabolomics"], ["genomics"], [])


def _rct(folder: Path) -> Path:
    """H-skeptic/h4.py's RCT table (``default_rng(3)``, n = 200), draw for draw."""
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({
        "participant_id": np.arange(n), "treatment": rng.choice(["placebo", "fish_oil"], n),
        "arm": rng.integers(0, 2, n), "diet_group": rng.choice(["med", "control"], n),
        "condition": rng.choice(["case", "control"], n), "phenotype": rng.choice(["lean", "obese"], n),
        "fasting": rng.integers(0, 2, n), "group": rng.choice(["A", "B"], n),
        "plate_reader_od": rng.normal(1.2, 0.3, n).round(3), "site": rng.integers(1, 5, n),
        "age": rng.integers(30, 70, n), "sex": rng.choice(["F", "M"], n),
        "ldl_change": rng.normal(-5, 10, n).round(1)})
    return _write(df, folder, "rct.csv")


def test_2a_a_trials_arms_groups_and_fasting_stay_predictors_under_every_lens(tmp_path):
    """IN-02: under every lens these became "design: An acquisition column (batch, plate or run
    order), not biology" and left the model. Expected: §5's list, each a predictor role (exposure,
    covariate or energy); the trial's arms are exposures, what the study compares."""
    path = _rct(tmp_path)
    for lens in LENSES:
        roles = _roles(path, lens=lens, target="ldl_change")
        for name in ("treatment", "arm", "diet_group", "condition", "phenotype", "fasting"):
            assert roles[name]["proposed"] in PREDICTOR_ROLES, (lens, name, roles[name])
            assert "acquisition" not in roles[name]["reason"].lower(), (lens, name)
        for name in ("treatment", "arm", "diet_group", "condition", "phenotype", "group"):
            assert roles[name]["proposed"] == "exposure", (lens, name, roles[name])
        assert roles["fasting"]["proposed"] == "covariate"
        # An ELISA readout is a measurement, not a plate.
        assert roles["plate_reader_od"]["proposed"] in PREDICTOR_ROLES
        assert roles["plate_reader_od"]["kind"] is None
        assert roles["site"]["proposed"] == "cluster"  # test 3: a site groups participants


def test_2b_rates_and_birth_weight_stay_predictors(tmp_path):
    """IN-10: ``steps_per_day``, ``drinks_per_week``, ``baseline_glucose`` went to "time" and
    ``birth_weight`` in grams to "design: a sampling weight". H-skeptic/h16.py's table
    (``default_rng(6)``, n = 300). Expected: §5's list, each a predictor role."""
    rng = np.random.default_rng(6)
    n = 300
    df = pd.DataFrame({
        "steps_per_day": rng.normal(7000, 2500, n).round(), "servings_fruit_per_day": rng.poisson(2, n),
        "drinks_per_week": rng.poisson(3, n), "baseline_glucose": rng.normal(100, 12, n).round(),
        "glucose_followup": rng.normal(98, 12, n).round(), "birth_weight": rng.normal(3300, 450, n).round(),
        "met": rng.normal(25, 6, n).round(1), "methionine_umol": rng.normal(25, 6, n).round(1),
        "hba1c": rng.normal(5.6, .5, n).round(1)})
    path = _write(df, tmp_path, "tokens.csv")
    for lens in (["clinical"], ["dietary"], []):
        roles = _roles(path, lens=lens, target="hba1c")
        for name in ("steps_per_day", "servings_fruit_per_day", "drinks_per_week",
                     "baseline_glucose", "glucose_followup", "birth_weight"):
            assert roles[name]["proposed"] in PREDICTOR_ROLES, (lens, name, roles[name])
        assert "sampling weight" not in roles["birth_weight"]["reason"]
    # A survey weight is still read as one when the table says it is a survey (NHANES's names).
    from turbotab.core.recognizers import reads_as_survey_weight

    assert reads_as_survey_weight("WTMEC2YR") and reads_as_survey_weight("survey_weight")
    assert reads_as_survey_weight("weight", median=34_000, design_in_table=True)
    assert not reads_as_survey_weight("birth_weight", median=3_300, design_in_table=True)
    assert not reads_as_survey_weight("weight", median=34_000, design_in_table=False)


def _omics(folder: Path) -> Path:
    rng = np.random.default_rng(23)
    n = 120
    df = pd.DataFrame({"sample_id": [f"S{i:03d}" for i in range(n)], "batch": rng.integers(1, 5, n),
                       "run_order": rng.permutation(np.arange(1, n + 1)), "plate_id": rng.integers(1, 3, n),
                       "group": rng.choice(["case", "control"], n), "age": rng.integers(30, 70, n),
                       **{f"m{i:02d}": rng.lognormal(2, .4, n).round(3) for i in range(12)}})
    return _write(df, folder, "omics.csv")


def test_2c_batch_is_an_acquisition_column_whose_role_follows_the_purpose(tmp_path):
    """IN-02: batch and run order became "design" and were dropped whatever the purpose, although
    the pack's own default is to model the batch (METABOLOMICS_PACK; Nygaard et al. 2016 block for
    batch in the model). Expected (§5): "acquisition/batch", a covariate under inference; left out
    under prediction, where new samples come from new batches, with the reason stated."""
    path = _omics(tmp_path)
    for purpose, role in (("inference", "covariate"), (None, "covariate"), ("prediction", "excluded")):
        roles = _roles(path, lens=["metabolomics"], target="group", purpose=purpose)
        for name in ("batch", "run_order", "plate_id"):
            p = roles[name]
            assert p["kind"] == "acquisition" and p["proposed"] == role, (purpose, name, p)
            assert p["reason"].startswith("Acquisition/batch:"), p
        assert roles["sample_id"]["proposed"] == "identifier"
    roles = _roles(path, lens=["metabolomics"], target="age", purpose="inference")
    assert roles["group"]["proposed"] == "exposure"


# ── 3 · identifiers ──────────────────────────────────────────────────────────

# H/idnames.py's 46 real-world names, in its order.
NAMES_46 = ["eid", "f.eid", "MRN", "mrn", "patid", "ptid", "pt_id", "Participant", "ParticipantID",
            "participantId", "subjid", "SUBJID", "USUBJID", "ResponseId", "record", "Subject", "id",
            "ID", "SEQN", "seqn", "pid", "hhid", "HHID", "person_number", "studyno", "Study Number",
            "case_no", "Sample Name", "SampleID", "sample", "batch_id", "site_id", "plate_id",
            "visit_id", "household_id", "family_id", "lipid", "uric_acid", "fatty_acid_id",
            "RIDAGEYR", "rid", "RID", "idno", "IDNO", "barcode", "accession"]
# What the cohorts' own documentation says these name: UK Biobank's encoded participant identifier
# (EID), CPRD's "unique CPRD patient identifier is [patid]", MESA's "Participant Identification
# Number" (idno), CDISC's unique subject identifier (USUBJID), a participant ID (ptid), and the
# Health and Retirement Study's household identifier (HHID, which "uniquely identifies an original
# household"). A site and a household group participants.
IDENTIFIERS = ["eid", "patid", "ptid", "HHID", "IDNO", "USUBJID"]
PEOPLE = ["eid", "patid", "ptid", "IDNO", "USUBJID", "MRN", "Participant", "SEQN", "subjid"]
CLUSTERS = ["site_id", "household_id", "HHID"]
NOT_IDS = ["lipid", "uric_acid", "RIDAGEYR", "sample"]  # RIDAGEYR is NHANES age in years


def test_3a_one_recognizer_agrees_with_itself_on_the_46_names():
    """IN-06: ``rows._id_like``, ``finding_words.is_identifier_name`` and the lockbox's ``_id_kind``
    disagreed on 11 of the 46 names, and none read ``eid``, ``patid``, ``ptid``, ``HHID`` or
    ``IDNO``. Every reader in ``turbotab/core`` now asks one recognizer, so they agree on all 46,
    and the names the cohorts document read as identifiers."""
    from turbotab.core.recognizers import acquisition_kind, id_kind, is_identifier
    from turbotab.core.stages.finding_words import is_identifier_name
    from turbotab.core.stages.rows import _id_like, propose_roles
    from turbotab.core.stages.working import person_identifiers

    assert len(NAMES_46) == 46
    for name in NAMES_46:
        kind = id_kind(name)
        assert _id_like(name) == is_identifier_name(name) == (kind is not None), name
        assert (person_identifiers([name], None) == [name]) == (kind == "subject"), name
        # The roles proposal reads the same: a whole-number column, unique on every row, is an
        # identifier exactly when the name reads as one. A batch or plate number is the one
        # exception, by design: an acquisition column, whose role follows the purpose (test 2c).
        summary = {"name": name, "dtype": "integer", "n": 100, "n_missing": 0, "n_unique": 100}
        proposed = propose_roles([summary], lens=[], target=None, n_rows=100)[0]
        if acquisition_kind(name) is not None:
            assert proposed["kind"] == "acquisition", (name, proposed)
            continue
        assert (proposed["proposed"] == "identifier") == (kind is not None), (name, proposed)
    for name in IDENTIFIERS:
        assert is_identifier(name), name
    for name in PEOPLE:
        assert id_kind(name) == "subject", name
    for name in CLUSTERS:
        assert id_kind(name) == "cluster", name
    for name in NOT_IDS:
        assert not is_identifier(name), name


def _ids(folder: Path) -> Path:
    """H-skeptic/h5b.py's table (``default_rng(1)``, n = 400), draw for draw."""
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "participant_id": np.arange(1000, 1000 + n), "site_id": rng.integers(1, 6, n),
        "household_id": rng.integers(1, 200, n),
        "eid": rng.permutation(np.arange(1000000, 1000000 + n)),
        "MRN": rng.permutation(np.arange(5000, 5000 + n)), "patid": np.arange(n) * 7 + 3,
        "Participant": np.arange(1, n + 1), "age": rng.integers(30, 70, n),
        "ldl": rng.normal(130, 30, n).round()})
    return _write(df, folder, "ids.csv")


def test_3b_sites_and_households_are_clusters_and_a_clean_study_seals_cleanly(tmp_path):
    """IN-06: ``site_id`` and ``household_id`` were proposed "identifier (high)", so "one row each"
    abandoned the seal on a clean study (``site_id`` repeats), and "repeated" grouped by 5 sites.
    Expected: clusters, never identifiers; the seal over the person identifiers is clean."""
    from turbotab.core.decisions import GrainSpec
    from turbotab.core.seal import decide_basis

    path = _ids(tmp_path)
    roles = _roles(path, lens=["clinical"], target="ldl")
    assert roles["site_id"]["proposed"] == "cluster" and roles["household_id"]["proposed"] == "cluster"
    for name in ("participant_id", "eid", "MRN", "patid", "Participant"):
        assert roles[name]["proposed"] == "identifier", (name, roles[name])
    proposed = {c: p["proposed"] for c, p in roles.items()}
    ids = [c for c, r in proposed.items() if r == "identifier"]
    frame = pd.read_csv(path)
    state = ProjectState(grain=GrainSpec(grain="one_row_per_unit", id_column="participant_id"),
                         roles=proposed)
    basis, column = decide_basis(state, frame[ids], ids)
    assert basis.state == "one_row_per_unit" and not basis.exploratory and column is None


def test_3c_no_measurement_is_suggested_as_a_unit(tmp_path):
    """IN-06 (I13): the grain question offered ``length_of_stay_days``, ``age`` and
    ``sodium_mmol_l`` on ``clinical_risk.csv`` as the unit. Expected: no suggestion is a float
    column or a column whose name says what it measures; an identifier-named column holding
    fractional values is no identifier either."""
    from turbotab.core.stages.working import structure_stage

    t = Ingested(SAMPLES / "clinical_risk.csv", tmp_path)
    reading = t.run(structure_stage, ProjectState(lens=["clinical"], target="readmit_30d"))["grain"]
    frame = pd.read_csv(SAMPLES / "clinical_risk.csv")
    for name in reading["suggested"]:
        assert frame[name].dtype.kind != "f", name
        assert name not in ("length_of_stay_days", "age", "sodium_mmol_l", "charlson_index"), name

    # A roster-shaped float column and a fractional "subject_id" (a measurement mislabeled).
    rng = np.random.default_rng(31)
    n = 240
    people = np.repeat(np.arange(60), 4)
    df = pd.DataFrame({"SUBJ": people, "visit_weight_kg": np.repeat(rng.normal(80, 9, 60).round(1), 4),
                       "subject_id": rng.normal(50, 10, n).round(2), "glucose": rng.normal(100, 10, n)})
    path = _write(df, tmp_path, "roster.csv")
    t = Ingested(path, Path(tempfile.mkdtemp()))
    suggested = t.run(structure_stage, ProjectState(lens=["clinical"], target="glucose"))["grain"]["suggested"]
    assert suggested == ["SUBJ"]
    roles = _roles(path, lens=["clinical"], target="glucose")
    assert roles["subject_id"]["proposed"] != "identifier"


def test_3d_a_measurement_named_as_the_unit_never_makes_a_clean_seal(tmp_path):
    """IN-06 (I13): naming ``length_of_stay_days`` the unit gave a clean "grouped" seal over 19
    "units". The answer stands (the draw keeps its values together) but the basis is exploratory
    and says why; a person's identifier still seals cleanly."""
    from turbotab.core.decisions import GrainSpec
    from turbotab.core.seal import decide_basis

    frame = pd.read_csv(SAMPLES / "clinical_risk.csv")
    los = ProjectState(grain=GrainSpec(grain="repeated", id_column="length_of_stay_days"))
    basis, column = decide_basis(los, frame[["length_of_stay_days"]], [])
    assert basis.state == "grouped" and column == "length_of_stay_days"
    assert basis.exploratory and "reads as a measurement" in basis.sentence
    people = pd.DataFrame({"participant_id": np.repeat(np.arange(50), 3)})
    clean = ProjectState(grain=GrainSpec(grain="repeated", id_column="participant_id"))
    basis, column = decide_basis(clean, people, ["participant_id"])
    assert basis.state == "grouped" and not basis.exploratory and column == "participant_id"


# ── 4 · the outcome's unit ───────────────────────────────────────────────────

# C-skeptic/c4.py's names (``default_rng(0)``), each with the unit the auditor generated it in.
UNIT_CASES = [
    ("gestational_age", "weeks", lambda r: r.normal(39, 1.5, 500)),
    ("birth_weight", "g", lambda r: r.normal(3300, 450, 500)),
    ("dietary_choline", "mg/day", lambda r: r.normal(330, 90, 500)),
    ("dietary_cholesterol", "mg/day", lambda r: r.normal(290, 110, 500)),
    ("cholesterol_intake", "mg/day", lambda r: r.normal(290, 110, 500)),
    ("glucosinolate_intake", "mg/day", lambda r: r.lognormal(2.5, 0.6, 500)),
    ("urine_creatinine", "mg/dL", lambda r: r.normal(120, 50, 500).clip(10)),
    ("creatine_kinase", "U/L", lambda r: r.lognormal(4.7, 0.5, 500)),
    ("recreational_activity_min", "min", lambda r: r.lognormal(4, 0.8, 500)),
    ("telomere_length_bp", "bp", lambda r: r.normal(6500, 900, 500)),
    ("sleep_hr", "hours", lambda r: r.normal(7, 1, 500)),
    ("height", "cm (toddlers)", lambda r: r.normal(88, 5, 500)),
    ("weight", "kg (bariatric)", lambda r: r.normal(130, 20, 500)),
    ("weight", "lb (children)", lambda r: r.normal(60, 15, 500)),
    ("hba1c_mmol", "mmol/mol", lambda r: r.normal(40, 6, 500)),
]
# Names that spell their unit out in full: the only units stated without a decision.
SPELLED = {"glucose_mg_dl": "mg/dL", "ldl_mmol_l": "mmol/L", "sbp_mmhg": "mmHg", "weight_kg": "kg",
           "choline_mg_day": "mg/day", "energy_kcal": "kcal"}


def test_4a_no_unit_is_stated_that_the_name_does_not_spell_out(tmp_path):
    """IN-05: the unit was guessed from the name and the values and written into the record:
    dietary choline in mg/day became "mg/dL". Expected: for every one of the auditor's names the
    target stage states no unit and the record's sentence carries none; choline is never proposed
    as "mg/dL"; a spelled-out suffix is stated."""
    from turbotab.core import voice
    from turbotab.core.stages.target import target_info_stage

    rng = np.random.default_rng(0)
    for i, (name, truth, make) in enumerate(UNIT_CASES):
        one = pd.DataFrame({name: make(rng)})
        path = _write(one, tmp_path, f"{i:02d}_{name}.csv")
        info = Ingested(path, Path(tempfile.mkdtemp())).run(
            target_info_stage, ProjectState(target=name, task="regression"))
        assert info["unit"] is None and info["unit_source"] is None, (name, truth, info["unit"])
        sentence = voice.sentence_for(d.SetTarget(column=name), None, {"frame": one})
        assert sentence == f"`{name}` was chosen as the outcome.", sentence
        if name in ("dietary_choline", "dietary_cholesterol", "cholesterol_intake"):
            assert info["proposed_unit"] is None, (name, info["proposed_unit"])
    for name, unit in SPELLED.items():
        one = pd.DataFrame({name: rng.normal(100, 10, 200)})
        path = _write(one, tmp_path, f"{name}.csv")
        info = Ingested(path, Path(tempfile.mkdtemp())).run(
            target_info_stage, ProjectState(target=name, task="regression"))
        assert (info["unit"], info["unit_source"]) == (unit, "name"), name


def test_4b_a_proposed_unit_reaches_a_sentence_only_once_recorded(tmp_path):
    """The pack's reading is proposed for the user's decision; once ``set_outcome_unit`` records a
    unit, the target stage, the record and the substitution estimand state it. Source check: the
    pack's own rule, quoted from CLINICAL_SURVEY_PACK §A1.1."""
    from turbotab.core import voice
    from turbotab.core.stages.target import target_info_stage

    pack = (REPO / "docs/turbotab/research/CLINICAL_SURVEY_PACK.md").read_text("utf-8")
    flat = " ".join(" ".join(line.lstrip("> ") for line in pack.splitlines()).split())
    assert ("Please confirm units per analyte against the source data dictionary — TurboTab will "
            "not guess.") in flat
    assert "Detect, propose, require explicit confirmation." in flat

    rng = np.random.default_rng(42)
    path = _write(pd.DataFrame({"glucose": rng.normal(100, 15, 300)}), tmp_path, "glucose.csv")
    unanswered = ProjectState(target="glucose", task="regression")
    info = Ingested(path, tmp_path).run(target_info_stage, unanswered)
    assert info["unit"] is None and info["proposed_unit"] == "mg/dL"
    assert info["unit_candidates"] == ["mg/dL", "mmol/L"]
    recorded = unanswered.model_copy(update={"outcome_unit": "mmol/L"})
    info = Ingested(path, Path(tempfile.mkdtemp())).run(target_info_stage, recorded)
    assert (info["unit"], info["unit_source"]) == ("mmol/L", "decision")
    assert voice.sentence_for(d.SetOutcomeUnit(column="glucose", unit="mmol/L"), unanswered, {}) == \
        "The unit of `glucose` was recorded as mmol/L."
    # The decision answers for the outcome only, and stands only while its column is the outcome.
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_outcome_unit", "column": "age", "unit": "years"},
                   {"state": unanswered, "target": "glucose"})
    assert refused.value.code == "not_the_target" and refused.value.exits
    log = d.DecisionLog(tmp_path / "log.jsonl")
    log.append(d.SetTarget(column="glucose"))
    log.append(d.SetOutcomeUnit(column="glucose", unit="mmol/L"))
    assert log.state().outcome_unit == "mmol/L"
    log.append(d.SetTarget(column="age"))
    assert log.state().outcome_unit is None


def _choline_diet(n: int = 400, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    return pd.DataFrame({"protein_g": P.round(1), "carbohydrate_g": C.round(1), "fat_g": F.round(1),
                         "energy_kcal": (4 * P + 4 * C + 9 * F).round(0),
                         "dietary_choline": (2.2 * P + rng.normal(150, 40, n)).round(1)})


def test_4c_the_substitution_estimand_states_only_a_recorded_unit(tmp_path):
    """IN-05: the substitution estimand read "predicted X (in mg/dL)" for dietary choline, a
    mg/day intake. Expected: no unit before a decision; the recorded one after."""
    from turbotab.core.decisions import SubstitutionSpec
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _choline_diet()
    paths = mf.ingest_frame(frame, tmp_path)
    roles = {"protein_g": "exposure", "carbohydrate_g": "exposure", "fat_g": "exposure",
             "energy_kcal": "energy"}
    estimands = {}
    for unit in (None, "mg/day"):
        st = mf.state(roles=roles, target="dietary_choline", models=["linear"], purpose="inference",
                      substitution=SubstitutionSpec(donor="protein_g", recipient="fat_g"),
                      outcome_unit=unit)
        split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
        ti = mf.target_info("regression", "dietary_choline")
        design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
        out = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
        estimands[unit] = out["estimand"]
    assert "predicted dietary_choline when" in estimands[None]
    assert "(in" not in estimands[None] and "mg/dL" not in estimands[None]
    assert "predicted dietary_choline (in mg/day)" in estimands["mg/day"]


# ── 5 · energy in kilojoules ─────────────────────────────────────────────────


def _kj_with_sodium(planted: bool) -> pd.DataFrame:
    """D-skeptic/s10_kj.py's "energy_only" table (``default_rng(5)``, n = 800), draw for draw:
    energy in kJ with sodium, sex and glucose beside it, no macronutrient. ``planted`` replaces 40
    rows with 300 kcal and 40 with 5,600 kcal (as kJ), from a stream of their own."""
    rng = np.random.default_rng(5)
    n = 800
    kcal = np.clip(rng.normal(2100, 550, n), 600, None)
    if planted:
        extra = np.random.default_rng(55)
        at = extra.choice(n, 80, replace=False)
        kcal[at[:40]] = 300.0
        kcal[at[40:]] = 5_600.0
    return pd.DataFrame({"energy": kcal * KCAL_PER_KJ, "sodium_mg": rng.normal(3300, 900, n),
                         "sex": rng.choice(["male", "female"], n), "glucose": rng.normal(100, 15, n)})


@pytest.mark.parametrize("planted", [False, True])
def test_5_a_kj_energy_beside_sodium_is_read_as_kj_and_counted_as_the_screens_count(tmp_path,
                                                                                     planted):
    """IN-07: with no macronutrients the unit fell back to kcal, so the screens would remove
    764–989 of 800 rows and the pack's finding said nearly every row was "above 5000". Expected:
    the magnitude prior (NUTRITION_PACK §01: "energy 1,600–2,600 kcal (7,000–11,000 → kJ)") reads
    kJ; the finding's count, and the 500–5,000 kcal screen's, equal NumPy's count of rows whose
    value ÷ 4.184 falls outside 500–5,000."""
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.stages.proposals import build_proposals

    frame = _kj_with_sodium(planted)
    kcal = frame["energy"] / KCAL_PER_KJ
    expected = int(((kcal < 500) | (kcal > 5000)).sum())
    assert expected == (80 if planted else 0)
    raw = int(((frame["energy"] < 500) | (frame["energy"] > 5000)).sum())
    assert raw > 700  # what the pack counted on the raw kJ column

    columns = [{"name": c, "dtype": "categorical" if frame[c].dtype == object else "numeric",
                "n_unique": int(frame[c].nunique()), "n_missing": 0} for c in frame.columns]
    out = build_proposals(frame, columns, lens=["dietary"], target=None)
    assert out["energy_unit"]["unit"] == "kj" and out["energy_unit"]["basis"] == "magnitude"
    screens = {e["key"]: e["affected"] for e in out["exclusions"]}
    assert screens["sex_neutral_500_5000"] == expected

    path = _write(frame, tmp_path, "kj.csv")
    found = Ingested(path, tmp_path).run(findings_stage, ProjectState(lens=["dietary"]))["findings"]
    implausible = [f for f in found if f["id"] == "pack::dietary::implausible_intake"]
    assert len(implausible) == 1
    f = implausible[0]
    assert "kJ" in f["detail"] and "2,092 kJ (500 kcal)" in f["detail"]
    if expected:
        assert f["title"] == f"{expected:,} records report an implausible daily intake."
        assert f["summary"].startswith(f"`{expected}` of `800` rows report `energy` below `2,092` kJ")
    else:
        assert f["title"] == "No record reports an implausible daily intake, read in kJ."
    coach = out["coach"].get("exclusions")
    if expected:
        assert "kJ" in coach["text"] and "kcal" not in coach["text"]


def test_5b_the_atwater_reconstruction_still_reads_kj_and_the_counts_agree():
    """H-skeptic/h9.py's table (``default_rng(9)``, n = 300): energy in kJ beside protein,
    carbohydrate and fat in grams; and the shipped ``nhanes_kilojoules.csv``, whose finding said
    "118 records … above 5000" beside "The energy column is in kilojoules". Expected: NumPy's
    count in kcal, by the reconstruction."""
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.stages.proposals import energy_unit_reading

    rng = np.random.default_rng(9)
    n = 300
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(250, 60, n).clip(50)
    F = rng.normal(75, 20, n).clip(15)
    h9 = pd.DataFrame({"energy": ((4 * P + 4 * C + 9 * F) * KCAL_PER_KJ).round(0),
                       "protein": P.round(1), "carbohydrate": C.round(1), "fat": F.round(1),
                       "age": rng.integers(20, 70, n)})
    shipped = pd.read_csv(SAMPLES / "nhanes_kilojoules.csv")
    for frame, energy in ((h9, "energy"), (shipped, "DR1TKCAL")):
        reading = energy_unit_reading(frame, energy)
        assert (reading["unit"], reading["basis"]) == ("kj", "atwater"), reading
        kcal = frame[energy] / KCAL_PER_KJ
        expected = int(((kcal < 500) | (kcal > 5000)).sum())
        with tempfile.TemporaryDirectory() as folder:
            path = _write(frame, Path(folder), "t.csv")
            found = Ingested(path, Path(folder)).run(findings_stage, ProjectState(lens=["dietary"]))
        f = next(x for x in found["findings"] if x["id"] == "pack::dietary::implausible_intake")
        said = 0 if f["title"].startswith("No record") else int(f["title"].split()[0].replace(",", ""))
        assert said == expected, (energy, f["title"], expected)


# ── 6 · NHANES weights: the least common denominator ─────────────────────────


def _nhanes_fasting(folder: Path, with_fasting_weight: bool = True) -> Path:
    rng = np.random.default_rng(66)
    n = 300
    df = pd.DataFrame({"SEQN": np.arange(n), "DR1TKCAL": rng.normal(2000, 400, n).round(0),
                       "LBXGLU": rng.normal(100, 12, n).round(0), "LBXTR": rng.normal(120, 40, n).round(0),
                       "WTDRD1": rng.uniform(5000, 60000, n).round(1),
                       "WTMEC2YR": rng.uniform(5000, 60000, n).round(1),
                       "SDMVSTRA": rng.integers(1, 15, n), "SDMVPSU": rng.integers(1, 3, n)})
    if with_fasting_weight:
        df["WTSAF2YR"] = rng.uniform(5000, 120000, n).round(1)
    return _write(df, folder, f"nhanes_{int(with_fasting_weight)}.csv")


def test_6_fasting_analytes_name_the_fasting_subsample_weight(tmp_path):
    """IN-19: a table with DR1TKCAL, LBXGLU, LBXTR, WTDRD1, WTMEC2YR and WTSAF2YR was told "Use the
    dietary weights, not the examination weight" (SETTLED), and WTSAF2YR was never mentioned.

    Source check, NHANES Tutorials, Weighting Module (wwwn.cdc.gov/nchs/nhanes/tutorials/
    weighting.aspx), read 2026-10-03: "You must use the weight of the smallest subpopulation that
    includes all the variables you want to include in your analysis." "A good rule of thumb is to
    use 'the least common denominator' where the variable that was collected on the smallest number
    of respondents is the 'least common denominator.'" On fasting triglycerides: "You would use the
    fasting subsample weights (wtsaf4yr)." Expected: WTSAF2YR, with the rule quoted and badged a
    convention; without the fasting weight in the table, the finding says it is missing; without
    fasting analytes, the dietary weight stands."""
    from turbotab.core.recognizers import NHANES_LCD_QUOTE
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.survey import offered

    tutorial = ("A good rule of thumb is to use 'the least common denominator' where the variable "
                "that was collected on the smallest number of respondents is the 'least common "
                "denominator.'")
    assert tutorial in NHANES_LCD_QUOTE

    def weights_finding(path: Path) -> dict:
        found = Ingested(path, Path(tempfile.mkdtemp())).run(
            findings_stage, ProjectState(lens=["dietary"]))["findings"]
        return next(f for f in found if f["id"] == "pack::dietary::survey_weights")

    f = weights_finding(_nhanes_fasting(tmp_path))
    assert f["title"] == "Use the fasting subsample weight, `WTSAF2YR`."
    assert "`LBXGLU` and `LBXTR` were measured on the morning fasting subsample" in f["detail"]
    assert tutorial in f["detail"]
    assert f["evidence"]["status"] == "CONVENTION"
    assert "WTSAF2YR" in f["summary"] and "WTDRD1" in f["summary"]

    missing = weights_finding(_nhanes_fasting(tmp_path, with_fasting_weight=False))
    assert missing["title"] == "The fasting subsample weight is not in this table."
    assert "`WTSAF2YR`" in missing["detail"]

    frame = pd.read_csv(_nhanes_fasting(tmp_path)).drop(columns=["LBXGLU", "LBXTR"])
    frame["LBXGH"] = 5.6  # glycohemoglobin: measured on the examined sample, not the fasting one
    dietary = weights_finding(_write(frame, tmp_path, "no_fasting.csv"))
    assert dietary["title"] == "Use the dietary weights, not the examination weight."

    # The survey question offers the rule's weight first (WP10's options; audit IN-19).
    roles = {"SEQN": "identifier", "DR1TKCAL": "energy", "LBXTR": "covariate",
             "WTDRD1": "design", "WTMEC2YR": "design", "WTSAF2YR": "design",
             "SDMVSTRA": "design", "SDMVPSU": "design"}
    state = ProjectState(lens=["dietary"], target="LBXGLU", roles=roles, purpose="inference")
    assert offered(state)[0]["decision"]["weight"] == "WTSAF2YR"
    no_lab = state.model_copy(update={"target": "DR1TKCAL",
                                      "roles": {k: v for k, v in roles.items() if k != "LBXTR"}})
    assert offered(no_lab)[0]["decision"]["weight"] == "WTDRD1"
