"""Column recognizers: one reading of a column's name, shared by every stage (audit WP13).

The audit found the app reading names as substrings: ``fat`` matched fat mass and fatigue, ``prot``
matched C-reactive protein, ``fib`` matched fibrinogen, ``carb`` matched bicarbonate (IN-01); three
identifier recognizers disagreed on 11 of 46 real names (IN-06); ``_per_day`` sent a rate to "time"
and a birth weight in grams became a sampling weight (IN-10). This module replaces those readings
with one set of rules that every caller in ``turbotab/core`` shares:

* **Whole tokens.** A name is split into words at separators and at a lower-to-upper case change
  (``TotalKcal`` → ``total``, ``kcal``); a vocabulary word must equal a whole word, never a part of
  one. ``fatigue``, ``fatty``, ``fibrinogen``, ``bicarbonate`` and ``prothrombin`` share letters
  with a nutrient and no word.
* **Corroboration.** NUTRITION_PACK §01: "match on three signals jointly, never names alone". A
  nutrient word is read as an intake only when nothing else in the name says it is something else:
  a body compartment (``body_fat_pct``, ``fat_mass_kg``, ``liver_fat``), a specimen (``serum``,
  ``plasma``, ``urine``), a concentration unit (``_g_dl``, ``_mmol_l``), a score, a medication, a
  behavior (``alcohol_use_disorder``) or a food (``fatty_fish_g``: a food group in grams carries no
  Atwater factor unless the user declares one). The kJ prior on magnitude is
  :func:`energy_unit_by_magnitude`.
* **Codebooks over guesses.** The NHANES total-nutrient variables are read from the codebook's own
  suffixes, and food-composition columns from their INFOODS tagnames, so ``DR1TSFAT`` is saturated
  fat and ``PROCNT`` is protein without any guessing.

Sources quoted here:

* NHANES 2021–2023 DR1TOT_L codebook ("Dietary Interview – Total Nutrient Intakes, First Day"),
  wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1TOT_L.htm: ``DR1TKCAL`` "Energy (kcal)",
  ``DR1TPROT`` "Protein (gm)", ``DR1TCARB`` "Carbohydrate (gm)", ``DR1TSUGR`` "Total sugars (gm)",
  ``DR1TFIBE`` "Dietary fiber (gm)", ``DR1TTFAT`` "Total fat (gm)", ``DR1TSFAT`` "Total saturated
  fatty acids (gm)", ``DR1TMFAT`` "Total monounsaturated fatty acids (gm)", ``DR1TPFAT`` "Total
  polyunsaturated fatty acids (gm)", ``DR1TALCO`` "Alcohol (gm)", ``DR1TNUMF`` "Number of
  foods/beverages reported" (not a nutrient), and the individual fatty acids ``DR1TS040`` …
  ``DR1TP226``. NUTRITION_PACK §01 names the day-2 (``DR2T*``) and 1999–2002 (``DRXT*``) files.
* USDA National Nutrient Database for Standard Reference, Release 27, ``NUTR_DEF.txt`` (the INFOODS
  tagnames): ``~203~^~g~^~PROCNT~^~Protein~``, ``~204~^~g~^~FAT~^~Total lipid (fat)~``,
  ``~205~^~g~^~CHOCDF~^~Carbohydrate, by difference~``, ``~208~^~kcal~^~ENERC_KCAL~^~Energy~``,
  ``~268~^~kJ~^~ENERC_KJ~^~Energy~``, ``~221~^~g~^~ALC~^~Alcohol, ethyl~``,
  ``~269~^~g~^~SUGAR~^~Sugars, total~``, ``~291~^~g~^~FIBTG~^~Fiber, total dietary~``,
  ``~606~^~g~^~FASAT~``, ``~645~^~g~^~FAMS~``, ``~646~^~g~^~FAPU~``, ``~605~^~g~^~FATRN~``,
  ``~209~^~g~^~STARCH~``, ``~601~^~mg~^~CHOLE~^~Cholesterol~``. ``CHOAVL`` (carbohydrate,
  available) is the FAO/INFOODS tagname for available carbohydrate summed from its components; its
  definition was read from FAO/INFOODS search excerpts, as the FAO PDFs could not be parsed here.
* Identifiers: UK Biobank releases each participant under "a unique, project-specific encoded
  identifier" (the EID, column ``eid``); CPRD's "unique CPRD patient identifier is [patid]";
  MESA's ``idno`` is the "MESA Participant Identification Number"; CDISC SDTM's ``USUBJID`` is the
  unique subject identifier; the Health and Retirement Study's ``HHID`` "uniquely identifies an
  original household" and, with ``PN`` (person number), a person, so ``HHID`` is read as a
  household: a cluster of people unless it is unique on every row (a household-level table).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

# ── words ─────────────────────────────────────────────────────────────────────

_CASE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokens(name: Any) -> list[str]:
    """The words of a column name, lowercased: split at separators and at a lower-to-upper case
    change (``TotalKcal``, ``ParticipantID``), never inside a run of capitals or after a digit
    (``DR1TKCAL``, ``WTSAF2YR`` and ``USUBJID`` stay one word)."""
    spaced = _CASE.sub("_", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def _joined(name: Any) -> str:
    return "".join(tokens(name))


# ── units a name declares by its suffix ─────────────────────────────────────

# A concentration names a specimen measure, never an intake (``serum_protein_g_dl``).
_CONCENTRATION = re.compile(
    r"(?:^|_)(?:g_?dl|mg_?dl|ug_?dl|mcg_?dl|g_?l|mg_?l|ug_?l|ng_?ml|pg_?ml|ug_?ml|mmol_?l|umol_?l|"
    r"nmol_?l|pmol_?l|meq_?l|mmol_?mol|iu_?l|u_?l|mg_?ml)$")


def _norm(name: Any) -> str:
    return "_".join(tokens(name))


def concentration_unit(name: Any) -> bool:
    """The name ends in a concentration unit (``_g_dl``, ``_mmol_l``): a specimen, not an intake."""
    return bool(_CONCENTRATION.search(_norm(name)))


# ── nutrients ─────────────────────────────────────────────────────────────────

Macro = Literal["protein", "carbohydrate", "fat", "alcohol", "fiber"]
MACROS: tuple[str, ...] = ("protein", "carbohydrate", "fat", "alcohol", "fiber")

# The NHANES total-nutrient suffixes (DR1TOT_L codebook), read after the file prefix: what each one
# is, the energy-bearing macronutrient it is a part of, and the part.
_NHANES_PREFIX = re.compile(r"^(?:DR1T|DR2T|DRXT|DR1I|DR2I|DRXI)(?P<code>[A-Z0-9]+)$")
_NHANES_CODES: dict[str, tuple[str, str | None, str | None]] = {
    "KCAL": ("energy", None, None),
    "PROT": ("protein", "protein", None),
    "CARB": ("carbohydrate", "carbohydrate", None),
    "SUGR": ("sugars", "carbohydrate", "sugar"),
    "FIBE": ("fiber", "fiber", None),
    "TFAT": ("fat", "fat", None),
    "SFAT": ("saturated fat", "fat", "sfa"),
    "MFAT": ("monounsaturated fat", "fat", "mufa"),
    "PFAT": ("polyunsaturated fat", "fat", "pufa"),
    "ALCO": ("alcohol", "alcohol", None),
    "CHOL": ("cholesterol", None, None), "ATOC": ("vitamin E", None, None),
    "ATOA": ("added vitamin E", None, None), "RET": ("retinol", None, None),
    "VARA": ("vitamin A", None, None), "ACAR": ("alpha-carotene", None, None),
    "BCAR": ("beta-carotene", None, None), "CRYP": ("beta-cryptoxanthin", None, None),
    "LYCO": ("lycopene", None, None), "LZ": ("lutein and zeaxanthin", None, None),
    "VB1": ("thiamin", None, None), "VB2": ("riboflavin", None, None),
    "NIAC": ("niacin", None, None), "VB6": ("vitamin B6", None, None),
    "FOLA": ("folate", None, None), "FA": ("folic acid", None, None),
    "FF": ("food folate", None, None), "FDFE": ("folate DFE", None, None),
    "CHL": ("choline", None, None), "VB12": ("vitamin B12", None, None),
    "B12A": ("added vitamin B12", None, None), "VC": ("vitamin C", None, None),
    "VD": ("vitamin D", None, None), "VK": ("vitamin K", None, None),
    "CALC": ("calcium", None, None), "PHOS": ("phosphorus", None, None),
    "MAGN": ("magnesium", None, None), "IRON": ("iron", None, None),
    "ZINC": ("zinc", None, None), "COPP": ("copper", None, None),
    "SODI": ("sodium", None, None), "POTA": ("potassium", None, None),
    "SELE": ("selenium", None, None), "CAFF": ("caffeine", None, None),
    "THEO": ("theobromine", None, None), "MOIS": ("moisture", None, None),
}
_NHANES_FATTY_ACID = re.compile(r"^(?P<kind>[SMP])\d{3}$")  # DR1TS160, DR1TM181, DR1TP205
_FATTY_ACID_PART = {"S": "sfa", "M": "mufa", "P": "pufa"}

# INFOODS tagnames (USDA SR27 NUTR_DEF; FAO/INFOODS), matched as whole names.
_INFOODS: dict[str, tuple[str, str | None, str | None]] = {
    "procnt": ("protein", "protein", None),
    "chocdf": ("carbohydrate", "carbohydrate", None),
    "choavl": ("carbohydrate", "carbohydrate", None),
    "choavldf": ("carbohydrate", "carbohydrate", None),
    "fat": ("fat", "fat", None),
    "fatce": ("fat", "fat", None),
    "fasat": ("saturated fat", "fat", "sfa"),
    "fams": ("monounsaturated fat", "fat", "mufa"),
    "fapu": ("polyunsaturated fat", "fat", "pufa"),
    "fatrn": ("trans fat", "fat", "trans"),
    "fibtg": ("fiber", "fiber", None),
    "sugar": ("sugars", "carbohydrate", "sugar"),
    "starch": ("starch", "carbohydrate", "starch"),
    "alc": ("alcohol", "alcohol", None),
    "chole": ("cholesterol", None, None),
}
_INFOODS_ENERGY = {"enerc", "enerc_kcal", "enerc_kj", "ener_kcal", "ener_kj"}

# Whole words naming an energy-bearing macronutrient.
_MACRO_WORDS: dict[str, str] = {
    **dict.fromkeys(("protein", "proteins", "prot", "procnt"), "protein"),
    **dict.fromkeys(("carbohydrate", "carbohydrates", "carb", "carbs", "cho", "chocdf", "choavl"),
                    "carbohydrate"),
    **dict.fromkeys(("fat", "fats", "tfat", "lipid", "lipids"), "fat"),
    **dict.fromkeys(("alcohol", "alco", "etoh", "ethanol", "alc"), "alcohol"),
    **dict.fromkeys(("fiber", "fibre", "fibers", "fibres", "fibe", "fibtg"), "fiber"),
}
# Words that name a part of one macronutrient on their own (``sfa_g``, ``sugar``).
_PART_WORDS: dict[str, tuple[str, str]] = {
    **dict.fromkeys(("sfa", "sfat", "fasat", "saturated", "satfat"), ("fat", "sfa")),
    **dict.fromkeys(("mufa", "mfat", "fams", "monounsaturated"), ("fat", "mufa")),
    **dict.fromkeys(("pufa", "pfat", "fapu", "polyunsaturated"), ("fat", "pufa")),
    **dict.fromkeys(("tfa", "fatrn", "transfat"), ("fat", "trans")),
    **dict.fromkeys(("sugar", "sugars", "sugr", "sucrose", "fructose", "lactose"),
                    ("carbohydrate", "sugar")),
    **dict.fromkeys(("starch",), ("carbohydrate", "starch")),
}
# Words that name a part only beside their macronutrient (``fat_sat``, ``poly_fat_g``).
_QUALIFIED_PARTS: dict[str, dict[str, str]] = {
    "fat": {"sat": "sfa", "mono": "mufa", "mon": "mufa", "poly": "pufa", "trans": "trans"},
    "protein": {"animal": "animal", "plant": "plant", "vegetable": "plant", "dairy": "dairy"},
}

# What else a nutrient word can be, by macronutrient: body composition, a specimen, a lab test, a
# behavior. Any of these words in the name and the nutrient reading is withdrawn.
_SPECIMEN = {"serum", "plasma", "blood", "urine", "urinary", "rbc", "erythrocyte", "csf", "saliva",
             "salivary", "tissue", "adipose", "breastmilk", "fecal", "faecal", "stool"}
_NOT_AN_INTAKE = {"score", "scale", "index", "test", "meds", "med", "medication", "medications",
                  "drug", "drugs", "rx", "therapy", "lowering", "id", "ids", "code", "flag"}
_MACRO_DENY: dict[str, set[str]] = {
    "fat": {"mass", "body", "bodyfat", "free", "lean", "liver", "hepatic", "visceral",
            "subcutaneous", "trunk", "android", "gynoid", "area", "volume", "fraction",
            "distribution", "pct", "percent", "low", "reduced", "fatty"},
    "protein": {"reactive", "c", "crp", "binding", "kinase", "creatinine", "electrophoresis"},
    "carbohydrate": {"antigen", "deficient", "transferrin"},
    "alcohol": {"use", "disorder", "dependence", "abuse", "audit", "drinker", "drinkers",
                "drinking", "drinks", "drink", "frequency", "freq", "status", "history", "ever",
                "never", "current", "former", "binge", "problem", "problems", "withdrawal",
                "days"},
    "fiber": {"muscle", "nerve", "optic", "type"},
}
# Food words: a food or a food group in grams is an intake but not a nutrient, so it carries no
# Atwater factor (``fatty_fish_g``: fish is about 2 kcal/g, not fat's 9).
_FOODS = {"fish", "meat", "dairy", "milk", "cheese", "yogurt", "yoghurt", "butter", "oil", "oils",
          "nut", "nuts", "egg", "eggs", "food", "foods", "serving", "servings", "portion",
          "portions", "drink", "drinks", "beverage", "beverages", "snack", "snacks", "dressing",
          "dessert", "desserts", "sweets", "cereal", "cereals", "bread", "juice", "soda", "fruit",
          "fruits", "vegetable", "vegetables", "veg", "grain", "grains", "legume", "legumes",
          "dish", "meal", "meals", "fried", "bar", "bars", "carbonated"}

# Non-energy nutrients by whole word (NUTRITION_PACK §01's name list, as words).
_OTHER_NUTRIENTS = {
    "sodium", "potassium", "calcium", "iron", "zinc", "magnesium", "phosphorus", "selenium",
    "copper", "iodine", "manganese", "chromium", "fluoride", "molybdenum", "cholesterol", "chol",
    "caffeine", "theobromine", "folate", "folic", "niacin", "thiamin", "thiamine", "riboflavin",
    "retinol", "carotene", "carotenoids", "cryptoxanthin", "lycopene", "lutein", "zeaxanthin",
    "tocopherol", "vitamin", "vitamins", "vit", "vita", "vitc", "vitd", "vite", "vitk", "b12", "b6",
    "choline", "moisture", "omega", "epa", "dha", "dpa", "flavonoids", "polyphenols",
    "isoflavones", "betaine",
}
_OTHER_DENY = {"hdl", "ldl", "vldl", "nonhdl", "deficiency", "status"}

# A name with these words beside "protein" names the serum test, not the intake: NHANES LBXSTP is
# "Total protein (g/dL)" while dietary protein is ``DR1TPROT`` "Protein (gm)". An intake unit
# (``total_protein_g``) restores the dietary reading.
_LAB_QUALIFIERS = {"protein": {"total"}, "cholesterol": {"total"}}


@dataclass(frozen=True)
class NutrientReading:
    """What a column's name says it is as an intake."""

    column: str
    nutrient: str  # "protein", "saturated fat", "sodium", …
    macro: str | None  # the energy-bearing macronutrient it is (or is a part of), else None
    part: str | None  # sfa | mufa | pufa | trans | sugar | starch | animal | plant | dairy
    source: Literal["nhanes", "infoods", "name"]


# The unit each codebook states (DR1TOT_L "(gm)", "(mg)", "(mcg)"; NUTR_DEF's units column).
_NHANES_GRAMS = {"PROT", "CARB", "SUGR", "FIBE", "TFAT", "SFAT", "MFAT", "PFAT", "ALCO", "MOIS"}
_NHANES_MG = {"CHOL", "ATOC", "ATOA", "VB1", "VB2", "NIAC", "VB6", "CHL", "VC", "CALC", "PHOS",
              "MAGN", "IRON", "ZINC", "COPP", "SODI", "POTA", "CAFF", "THEO"}
_NHANES_MCG = {"RET", "VARA", "ACAR", "BCAR", "CRYP", "LYCO", "LZ", "FOLA", "FA", "FF", "FDFE",
               "VB12", "B12A", "VD", "VK", "SELE"}
_INFOODS_UNITS = {"enerc_kcal": "kcal", "ener_kcal": "kcal", "enerc_kj": "kj", "ener_kj": "kj",
                  "chole": "milligrams"}


def codebook_unit(name: Any) -> str | None:
    """The unit a codebook variable's documentation states (``grams``, ``milligrams``,
    ``micrograms``, ``kcal``, ``kj``), or None for a name that is no codebook variable."""
    raw = str(name)
    code = _NHANES_PREFIX.match(raw.upper()) if re.fullmatch(r"[A-Za-z0-9]+", raw) else None
    if code is not None:
        c = code.group("code")
        if c == "KCAL":
            return "kcal"
        if c in _NHANES_GRAMS or _NHANES_FATTY_ACID.match(c):
            return "grams"
        if c in _NHANES_MG:
            return "milligrams"
        if c in _NHANES_MCG:
            return "micrograms"
        return None
    # A tagname is written as one (``FAT``, ``PROCNT``, ``ENERC_KCAL``): a lowercase ``fat`` is a
    # plain word whose unit nothing states.
    if raw != raw.upper():
        return None
    joined = "_".join(tokens(raw))
    if joined in _INFOODS_UNITS:
        return _INFOODS_UNITS[joined]
    if joined in _INFOODS:
        return "grams"
    return None


class AmbiguousNutrient(ValueError):
    """The name names two energy-bearing macronutrients (``carb_fat_ratio``)."""


def _intake_unit(name: Any) -> bool:
    from turbotab.core.methods.energy import unit_of

    return unit_of(name) in ("grams", "kcal", "kj", "density", "milligrams", "micrograms")


def read_nutrient(name: Any) -> NutrientReading | None:
    """The intake a column's name declares, or None. Raises :class:`AmbiguousNutrient` when the
    name names two energy-bearing macronutrients at once."""
    raw = str(name)
    code = _NHANES_PREFIX.match(raw.upper()) if re.fullmatch(r"[A-Za-z0-9]+", raw) else None
    if code is not None:
        c = code.group("code")
        if c in _NHANES_CODES:
            what, macro, part = _NHANES_CODES[c]
            return None if what == "energy" else NutrientReading(raw, what, macro, part, "nhanes")
        fa = _NHANES_FATTY_ACID.match(c)
        if fa:
            part = _FATTY_ACID_PART[fa.group("kind")]
            return NutrientReading(raw, f"fatty acid {c}", "fat", part, "nhanes")
        return None  # DR1TNUMF and other non-nutrient codebook variables
    words = tokens(raw)
    if not words:
        return None
    joined = "_".join(words)
    if joined in _INFOODS:
        what, macro, part = _INFOODS[joined]
        return NutrientReading(raw, what, macro, part, "infoods")
    present = set(words)
    if present & (_SPECIMEN | _NOT_AN_INTAKE) or concentration_unit(raw):
        return None
    unit = _intake_unit(raw)
    found: dict[str, str | None] = {}
    for w in words:
        if w in _PART_WORDS:
            macro, part = _PART_WORDS[w]
            found.setdefault(macro, part)
            if found[macro] is None:
                found[macro] = part
        elif w in _MACRO_WORDS:
            found.setdefault(_MACRO_WORDS[w], None)
    # "fatty acids" is fat: the pair names the nutrient, and "fatty" is then no food word.
    if "fatty" in present and present & {"acid", "acids"}:
        found.setdefault("fat", None)
        present = present - {"fatty"}
    for macro in list(found):
        denied = present & _MACRO_DENY.get(macro, set())
        if macro == "fat" and _density(raw):
            denied -= {"pct", "percent"}  # ``fat_pct_kcal``: a share of energy, not body fat %
        lab = present & _LAB_QUALIFIERS.get(macro, set())
        if denied or present & _FOODS or (lab and not unit):
            del found[macro]
    if len(found) > 1:
        raise AmbiguousNutrient(f"{raw} matches more than one nutrient ({' and '.join(found)})")
    if found:
        macro, part = next(iter(found.items()))
        if part is None:
            for w in words:
                part = _QUALIFIED_PARTS.get(macro, {}).get(w) or part
        return NutrientReading(raw, macro if part is None else f"{macro} ({part})", macro, part,
                               "name")
    if present & _FOODS:
        return None
    other = present & _OTHER_NUTRIENTS
    if other and not present & _OTHER_DENY:
        if present & _LAB_QUALIFIERS["cholesterol"] and "cholesterol" in other and not unit:
            return None
        return NutrientReading(raw, sorted(other)[0], None, None, "name")
    return None


def _density(name: Any) -> bool:
    from turbotab.core.methods.energy import unit_of

    return unit_of(name) == "density"


def nutrient_role(name: Any) -> str | None:
    """The energy-bearing macronutrient a column's name declares (protein, carbohydrate, fat,
    alcohol or fiber), or None. Raises ``ValueError`` when it names two."""
    reading = read_nutrient(name)
    return reading.macro if reading is not None else None


def is_nutrient(name: Any) -> bool:
    """The name declares a nutrient intake, energy-bearing or not (two macronutrients included)."""
    try:
        return read_nutrient(name) is not None
    except AmbiguousNutrient:
        return True


# ── total energy ──────────────────────────────────────────────────────────────

_ENERGY_WORDS = {"energy", "kcal", "kcals", "kj", "calories", "calorie", "kilocalories",
                 "kilocalorie", "kilojoules", "kilojoule", "enerc", "ener"}
# An expenditure, a requirement, a basal rate or a goal is energy, but not intake.
_NOT_INTAKE_ENERGY = {"expenditure", "expend", "expended", "tee", "ree", "bmr", "rmr", "pal",
                      "burn", "burned", "burnt", "basal", "resting", "requirement",
                      "requirements", "eer", "goal", "goals", "target", "need", "needs",
                      "balance", "density", "dense", "share", "pct", "percent", "ratio", "from",
                      "drink", "drinks", "bar", "bars", "adjusted", "residual"}
_PER_DENOMINATOR = {"kg", "g", "1000", "1000kcal", "mj", "bw", "kgbw", "m2"}


def reads_as_total_energy(name: Any) -> bool:
    """Whether a column's name reads as total energy intake (``energy_kcal``, ``DR2TKCAL``,
    ``TotalKcal``, ``ENERC_KCAL``, ``energy_kj``). Not a macronutrient's own energy (``alc_kcal``,
    ``protein_kcal``, ``kcal_from_fat``), not a share of energy, and not an expenditure, a
    requirement or a goal."""
    raw = str(name)
    code = _NHANES_PREFIX.match(raw.upper()) if re.fullmatch(r"[A-Za-z0-9]+", raw) else None
    if code is not None:
        return code.group("code") == "KCAL"
    words = tokens(raw)
    if "_".join(words) in _INFOODS_ENERGY:
        return True
    present = set(words)
    if not present & _ENERGY_WORDS or _density(raw):
        return False
    if present & (_NOT_INTAKE_ENERGY | _SPECIMEN):
        return False
    for i, w in enumerate(words[:-1]):
        if w == "per" and words[i + 1] in _PER_DENOMINATOR:
            return False
    try:
        return read_nutrient(raw) is None
    except AmbiguousNutrient:
        return False


def energy_unit(name: Any) -> Literal["kcal", "kj"] | None:
    """The energy unit the name states: ``kcal`` (``DR1TKCAL``, ``calories``), ``kj``, or None."""
    words = set(tokens(name))
    if words & {"kj", "kilojoules", "kilojoule"} or _norm(name).endswith("_kj"):
        return "kj"
    raw = str(name).upper()
    if (words & {"kcal", "kcals", "calories", "calorie", "kilocalories", "kilocalorie"}
            or raw.endswith("KCAL")):
        return "kcal"
    return None


# NUTRITION_PACK §01, "Median-magnitude plausibility priors (adults/day) as a second signal:
# energy 1,600–2,600 kcal (7,000–11,000 → kJ)". A median inside the kJ band and above any
# plausible daily intake in kcal is read as kilojoules; one inside the kcal band as kcal.
KCAL_PRIOR = (1_600.0, 2_600.0)
KJ_PRIOR = (7_000.0, 11_000.0)
ENERGY_PRIOR_SOURCE = ("NUTRITION_PACK §01, median-magnitude plausibility priors: energy "
                       "1,600–2,600 kcal (7,000–11,000 → kJ)")


def energy_unit_by_magnitude(values: Any) -> Literal["kcal", "kj"] | None:
    """The unit the median daily energy points to, by the pack's magnitude prior, or None when the
    median sits in neither band (a child's intake, a subsample, a weekly total: not guessed)."""
    try:
        import numpy as np
        import pandas as pd

        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
    except Exception:  # noqa: BLE001 - nothing to judge by
        return None
    if not len(x):
        return None
    median = float(np.median(x))
    if KJ_PRIOR[0] <= median <= KJ_PRIOR[1]:
        return "kj"
    if KCAL_PRIOR[0] <= median <= KCAL_PRIOR[1]:
        return "kcal"
    return None


# ── identifiers ───────────────────────────────────────────────────────────────

IdKind = Literal["subject", "record", "cluster", "visit"]

# A person: the lockbox's subject words (utils/test_lockbox.py ``_PERSON_TOKENS``) with the
# cohort-specific spellings it missed (audit IN-06): UK Biobank ``eid``, CPRD ``patid``, ADNI-style
# ``ptid``, MESA ``idno``, CDISC ``usubjid``.
_SUBJECT_WORDS = {
    "subject", "subj", "subjid", "usubjid", "subjectid", "participant", "participantid",
    "patient", "patientid", "patid", "ptid", "person", "personid", "respondent", "respondentid",
    "mrn", "seqn", "pid", "sid", "individual", "idind", "enrollmentid", "eid", "idno",
    "memberid", "personnumber",
}
# Words that make an identifier only beside an identifier word (``pt_id``, ``member_id``).
_ID_TAILS = {"id", "ids", "uid", "uuid", "guid", "no", "num", "number", "nr", "code", "key",
             "name"}
_SUBJECT_WITH_TAIL = {"pt", "pat", "member"}
# Words an identifier's name may carry beside its kind word (``unique_subject_id``, ``f.eid``).
_ID_FILLERS = {"unique", "encoded", "anon", "anonymized", "anonymised", "pseudo", "research",
               "random", "study", "f", "original", "new", "old", "global", "local", "sequence"}
# A cluster contains people; grouping by one keeps people whole but is not a person.
_CLUSTER_WORDS = {"site", "sites", "center", "centre", "clinic", "hospital", "practice", "household",
                  "hh", "family", "school", "village", "cluster", "ward", "community", "district",
                  "batch", "plate", "chip", "array", "lane", "flowcell", "run", "cohort"}
_CLUSTER_JOINED = {"hhid", "hhno", "householdid", "familyid", "siteid", "centerid", "centreid",
                   "clinicid", "batchid", "plateid"}
# Finer than a person: a visit, an encounter, a replicate. Never the person's grouping.
_VISIT_WORDS = {"visit", "visits", "timepoint", "occasion", "session", "encounter", "event",
                "round", "wave", "cycle", "followup", "replicate", "aliquot", "draw", "measurement",
                "observation", "obs", "reading"}
# A row or sample: named as such.
_RECORD_JOINED = {"id", "ids", "uid", "uuid", "guid", "record", "recordid", "recordno",
                  "sampleid", "samplename", "sampleno", "specimenid", "responseid", "rowid",
                  "studyno", "studynumber", "studyid", "caseno", "casenumber", "caseid"}


def id_kind(name: Any) -> IdKind | None:
    """What an identifier-named column names: a ``subject`` (a person), a ``cluster`` of people
    (site, household, batch), a ``visit`` (finer than a person), a ``record`` (a row or sample), or
    None when the name does not read as an identifier.

    The kind word must be the whole name, or stand only beside identifier words (``id``, ``no``,
    ``unique``…): ``patient_id`` is a subject and ``patient_age`` is not an identifier at all."""
    words = tokens(name)
    if not words:
        return None
    joined = "".join(words)
    tail = words[-1] in _ID_TAILS or words[0] in {"id", "ids"}
    core = {w for w in words if w not in _ID_TAILS and w not in _ID_FILLERS}
    if joined in _CLUSTER_JOINED:
        return "cluster"
    if joined in _SUBJECT_WORDS:
        return "subject"
    subject = core & (_SUBJECT_WORDS | (_SUBJECT_WITH_TAIL if tail else set()))
    if subject and core <= _SUBJECT_WORDS | _SUBJECT_WITH_TAIL | _VISIT_WORDS:
        return "visit" if core & _VISIT_WORDS else "subject"
    if core and core <= _CLUSTER_WORDS and (tail or len(words) == 1):
        return "cluster"
    if core and core <= _VISIT_WORDS and tail:
        return "visit"
    if joined in _RECORD_JOINED:
        return "record"
    if tail and len(words) > 1 and words[-1] in {"id", "ids", "uid", "uuid", "guid"}:
        return "record"
    if tail and core and core <= {"case", "record", "sample", "specimen", "response", "row"}:
        return "record"
    return None


def is_identifier(name: Any) -> bool:
    """The name reads as an identifier of any kind (a person, a cluster, a visit or a record)."""
    return id_kind(name) is not None


def names_a_person(name: Any) -> bool:
    """The name reads as a person's identifier: what may state the grain or group the seal."""
    return id_kind(name) == "subject"


# ── acquisition columns (batch, plate, run order) ─────────────────────────────

# METABOLOMICS_PACK §01's acquisition columns (``packs.DESIGN_COLUMNS``), matched as whole names:
# the name ends in the acquisition word, optionally followed by an identifier word, so
# ``extraction_batch`` and ``plate_id`` are acquisition columns and ``plate_reader_od`` (a reading)
# is not.
_ACQUISITION_PHRASES: dict[tuple[str, ...], str] = {
    ("batch",): "batch", ("plate",): "plate", ("well",): "well", ("plex",): "plex",
    ("polarity",): "polarity", ("run",): "run", ("chip",): "batch", ("lane",): "batch",
    ("flowcell",): "batch",
    ("run", "order"): "run_order", ("injection", "order"): "run_order",
    ("inj", "order"): "run_order", ("acq", "order"): "run_order",
    ("acquisition", "order"): "run_order", ("injection",): "run_order",
    ("well", "position"): "well", ("plate", "position"): "plate",
    ("tmt", "channel"): "plex",
}


def acquisition_kind(name: Any) -> str | None:
    """``batch``, ``plate``, ``well``, ``run_order``, ``run``, ``plex`` or ``polarity`` when the
    column records how samples were acquired; None otherwise."""
    words = tokens(name)
    while words and words[-1] in _ID_TAILS:
        words = words[:-1]
    for size in (2, 1):
        if len(words) >= size and tuple(words[-size:]) in _ACQUISITION_PHRASES:
            return _ACQUISITION_PHRASES[tuple(words[-size:])]
    return None


# ── study design: arms and groups ─────────────────────────────────────────────

# What a trial or an omics study compares (METABOLOMICS_PACK §01's group/class column): these are
# exposures, never acquisition columns (audit IN-02).
_STUDY_GROUP_WORDS = {"treatment", "treatments", "arm", "arms", "group", "groups", "condition",
                      "phenotype", "intervention", "allocation", "randomization", "randomisation",
                      "class", "exposure", "diet", "dose"}


def study_group(name: Any) -> bool:
    """The name says the column is a study's arm or comparison group (``treatment``, ``arm``,
    ``diet_group``, ``condition``, ``phenotype``)."""
    words = tokens(name)
    return bool(words) and words[-1] in _STUDY_GROUP_WORDS and not is_identifier(name)


# ── time and rates ────────────────────────────────────────────────────────────

_TIME_WORDS = {"date", "time", "datetime", "timestamp", "year", "years", "yr", "yrs", "cycle",
               "visit", "wave", "day", "days", "month", "months", "week", "weeks", "period",
               "recall", "round", "occasion", "followup", "baseline", "timepoint", "session",
               "hour", "hours", "dt"}
_TIME_FILLERS = {"of", "at", "no", "num", "number", "index", "begin", "start", "end", "stop",
                 "exam", "interview", "collection", "sample", "first", "last", "follow", "up",
                 "survey", "calendar", "study", "entry", "exit", "to", "event", "survival",
                 "censor", "censoring", "censored", "death", "fu", "since", "elapsed"}
_RATE_UNITS = {"day", "days", "d", "week", "weeks", "wk", "month", "months", "year", "years", "yr",
               "hour", "hr", "h", "min", "night"}

# Words that say the column measures something: a covariate, an analyte, an amount or a count.
_MEASURE_WORDS = {
    "age", "sex", "gender", "bmi", "weight", "height", "waist", "hip", "glucose", "insulin",
    "hba1c", "a1c", "cholesterol", "ldl", "hdl", "triglycerides", "triglyceride", "tg", "crp",
    "creatinine", "albumin", "hemoglobin", "haemoglobin", "sbp", "dbp", "bp", "systolic",
    "diastolic", "pressure", "sodium", "potassium", "score", "index", "count", "counts", "total",
    "level", "levels", "length", "duration", "rate", "ratio", "admissions", "steps", "drinks",
    "servings", "minutes", "intake", "dose", "concentration", "stay", "los", "charlson",
}


def is_rate(name: Any) -> bool:
    """``steps_per_day``, ``drinks_per_week``: an amount per unit of time, not a time."""
    words = tokens(name)
    return any(w == "per" and i + 1 < len(words) and words[i + 1] in _RATE_UNITS
               for i, w in enumerate(words))


def reads_as_time(name: Any) -> bool:
    """The name says *when* a row was measured (``visit``, ``recall_date``, ``cycle_begin_year``):
    a time word and nothing that measures something. ``baseline_glucose`` is a glucose and
    ``steps_per_day`` a rate (audit IN-10)."""
    words = tokens(name)
    present = set(words)
    if not present & _TIME_WORDS or is_rate(name):
        return False
    if present & _MEASURE_WORDS or is_nutrient(name):
        return False
    return present <= (_TIME_WORDS | _TIME_FILLERS | {w for w in present if w.isdigit()})


# ── measurements (never a unit's identifier) ──────────────────────────────────

_UNIT_WORDS = {"days", "years", "yrs", "months", "weeks", "hours", "hrs", "mins", "kg", "g",
               "mg", "mcg", "ug", "cm", "mm", "lb", "lbs", "kcal", "kj", "mmhg", "pct", "percent",
               "l", "dl", "ml", "bpm", "12mo", "mo"}


def reads_as_measurement(name: Any, *, dtype: str | None = None,
                         values: Any = None) -> str | None:
    """Why a column is a measurement and not a unit's identifier, or None.

    Fractional values make a measurement whatever the name (a ``subject_id`` holding 49.73 names
    no one). Otherwise a column the name reads as an identifier is not a measurement; any other is
    when its dtype is ``numeric`` with no values given, its name carries a unit
    (``length_of_stay_days``, ``sodium_mmol_l``), or a word that measures something (``age``,
    ``score``, a nutrient or an analyte)."""
    if values is not None:
        try:
            import numpy as np
            import pandas as pd

            x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) and not np.all(np.isfinite(x) & (x == np.floor(x))):
                return "its values have fractional parts"
        except Exception:  # noqa: BLE001 - text values: judged by name only
            pass
    if is_identifier(name):
        return None
    if values is None and dtype == "numeric":
        return "it holds fractional numbers"
    words = tokens(name)
    present = set(words)
    if present & _UNIT_WORDS or concentration_unit(name):
        return "its name carries a unit"
    if present & _MEASURE_WORDS or is_nutrient(name) or is_rate(name):
        return "its name says what it measures"
    return None


# ── survey weights ────────────────────────────────────────────────────────────

_SURVEY_WEIGHT_WORDS = {"pweight", "sampweight", "sampwt", "survwt", "svywt", "wgt"}
_SURVEY_CONTEXT = {"survey", "sampling", "sample", "design", "svy", "probability", "inverse"}
_BODY_WORDS = {"birth", "body", "bw", "baby", "infant", "newborn", "fetal", "gestational",
               "maternal", "pregnancy", "gain", "loss", "change", "kg", "lb", "lbs", "g", "grams",
               "gram", "oz", "ideal", "target", "current", "usual", "self", "reported"}
NHANES_WEIGHT_PREFIX = ("WTDR", "WTMEC", "WTINT", "WTSA", "WTSB", "WTSOG", "WTSAF", "WTSH",
                        "WTSCD", "WTSPO", "WTSVOC", "WTSHM", "WTFSM", "WTSSB")


def reads_as_survey_weight(name: Any, *, median: float | None = None,
                           design_in_table: bool = False) -> bool:
    """A sampling weight, never a body or birth weight (audit IN-10).

    An NHANES weight by its exact prefix (``WTDRD1``, ``WTMEC2YR``, ``WTSAF2YR``); a name in survey
    vocabulary (``survey_weight``, ``pweight``, ``sampling_wt``); or a bare ``weight``/``wt`` whose
    values are far above any body weight in a table that names its design exactly
    (``SDMVSTRA``…), the corroboration ``turbotab.nutrition.survey_design`` requires. A birth,
    body or gain word, or a body unit (g, kg, lb), is never a sampling weight."""
    raw = str(name)
    if re.fullmatch(r"[A-Za-z0-9]+", raw) and raw.upper().startswith(NHANES_WEIGHT_PREFIX):
        return True
    words = tokens(raw)
    present = set(words)
    if present & _BODY_WORDS:
        return False
    if present & _SURVEY_WEIGHT_WORDS:
        return True
    if present & {"weight", "weights", "wt", "wts"} and present & _SURVEY_CONTEXT:
        return True
    if set(words) <= {"weight", "weights", "wt", "final", "w"} and design_in_table:
        return median is not None and math.isfinite(float(median)) and float(median) > 1_000
    return False


# ── NHANES weights: the least common denominator ──────────────────────────────

# The NHANES weighting tutorial (wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx): "You must use
# the weight of the smallest subpopulation that includes all the variables you want to include in
# your analysis." "A good rule of thumb is to use 'the least common denominator' where the
# variable that was collected on the smallest number of respondents is the 'least common
# denominator.'" On fasting triglycerides: "You would use the fasting subsample weights
# (wtsaf4yr)."
NHANES_WEIGHTING_SOURCE = ("NHANES Tutorials, Weighting Module "
                           "(wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)")
NHANES_LCD_QUOTE = ("\"A good rule of thumb is to use 'the least common denominator' where the "
                    "variable that was collected on the smallest number of respondents is the "
                    "'least common denominator.'\"")

# The weights from the widest sample to the smallest: interview, examination, dietary day 1, both
# dietary days, then the morning fasting subsample (which is drawn from the examined sample).
WEIGHT_TIERS: tuple[tuple[str, str, int], ...] = (
    (r"^WTINT(2YR|4YR)?$", "interview", 0),
    (r"^WTMEC(2YR|4YR)?$", "examination", 1),
    (r"^WTDRD1$", "dietary day 1", 2),
    (r"^WTDR2D$", "dietary days 1 and 2", 3),
    (r"^WTSAF(2YR|4YR)?$", "fasting subsample", 4),
)
# Laboratory variables measured on the morning fasting subsample (NHANES GLU, TRIGLY and INS
# files: fasting glucose LBXGLU/LBDGLUSI, triglycerides LBXTR/LBDTRSI, LDL LBDLDL/LBDLDLSI, insulin
# LBXIN/LBDINSI), which the tutorial says take the fasting weight.
FASTING_ANALYTES = re.compile(r"^(LBXGLU|LBDGLUSI|LBXTR|LBDTRSI|LBDLDL|LBDLDLSI|LBDLDLM|LBDLDLN|"
                              r"LBXIN|LBDINSI|LBXAPB|LBDAPBSI)$")
_DIETARY_VARIABLE = re.compile(r"^(DR1T|DR2T|DR1I|DR2I|DR1_|DR2_|DRXT|DBQ|DBD)")


def weight_tier(name: Any) -> tuple[str, int] | None:
    """``(the sample the weight describes, its rank from widest to smallest)``, or None."""
    raw = str(name).upper()
    for pattern, label, rank in WEIGHT_TIERS:
        if re.match(pattern, raw):
            return label, rank
    return None


def least_common_denominator(columns: Sequence[str]) -> dict[str, Any] | None:
    """The NHANES weight the least-common-denominator rule names for these columns, or None when
    the table carries no NHANES weight.

    The weight follows the smallest sample whose variables are present: a fasting analyte with
    ``WTSAF2YR`` in the table names the fasting weight; dietary variables name the dietary weight;
    otherwise the examination weight. ``components`` says which variables set it."""
    weights = {}
    for c in columns:
        tier = weight_tier(c)
        if tier is not None:
            weights.setdefault(tier[1], (str(c), tier[0]))
    if not weights:
        return None
    fasting = [str(c) for c in columns if FASTING_ANALYTES.match(str(c).upper())]
    dietary = [str(c) for c in columns if _DIETARY_VARIABLE.match(str(c).upper())]
    wanted, because = None, []
    if fasting:
        wanted, because = 4, fasting
    elif dietary:
        wanted, because = 2, dietary
    if wanted is None:
        rank = 1 if 1 in weights else min(weights)
        name, label = weights[rank]
        return {"use": name, "sample": label, "because": [], "not": [w for r, (w, _) in
                                                               sorted(weights.items()) if r != rank],
                "missing": None}
    # Both dietary days are a smaller sample than day 1: prefer it when the table has it and both
    # days' variables are present.
    if wanted == 2 and 3 in weights and any(str(c).upper().startswith(("DR2T", "DR2I", "DR2_"))
                                            for c in columns):
        wanted = 3
    if wanted in weights:
        name, label = weights[wanted]
        return {"use": name, "sample": label, "because": because,
                "not": [w for r, (w, _) in sorted(weights.items()) if r != wanted], "missing": None}
    label = {4: "fasting subsample", 2: "dietary day 1", 3: "dietary days 1 and 2"}[wanted]
    expected = {4: "WTSAF2YR", 2: "WTDRD1", 3: "WTDR2D"}[wanted]
    return {"use": None, "sample": label, "because": because,
            "not": [w for _, (w, _) in sorted(weights.items())], "missing": expected}


__all__ = [
    "AmbiguousNutrient", "ENERGY_PRIOR_SOURCE", "codebook_unit", "FASTING_ANALYTES", "IdKind", "KCAL_PRIOR",
    "KJ_PRIOR", "MACROS", "NHANES_LCD_QUOTE", "NHANES_WEIGHTING_SOURCE", "NutrientReading",
    "acquisition_kind", "concentration_unit", "energy_unit", "energy_unit_by_magnitude",
    "id_kind", "is_identifier", "is_nutrient", "is_rate", "least_common_denominator",
    "names_a_person", "nutrient_role", "read_nutrient", "reads_as_measurement",
    "reads_as_survey_weight", "reads_as_time", "reads_as_total_energy", "study_group", "tokens",
    "weight_tier",
]
