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
_KJ = re.compile(r"(?<![A-Za-z])kJ(?![a-z])")


def tokens(name: Any) -> list[str]:
    """The words of a column name, lowercased: split at separators and at a lower-to-upper case
    change (``TotalKcal``, ``ParticipantID``), never inside a run of capitals or after a digit
    (``DR1TKCAL``, ``WTSAF2YR`` and ``USUBJID`` stay one word)."""
    # The kilojoule's own spelling, ``kJ``, is one word (``Energy_kJ``), not a case change.
    spaced = _CASE.sub("_", _KJ.sub("KJ", str(name)))
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


# The unit a name spells, read as a whole expression rather than by its first unit word (audit
# WP13 gate repair): ``protein_g_kg`` is grams per kg of body weight, ``fat_g_1000kcal`` and
# ``fibre_g_MJ`` are grams per unit of energy (a density), ``alcohol_g_week`` a weekly amount and
# ``glucose_mg_dl`` a concentration; only ``protein_g`` or ``protein_g_day`` is a day's amount in
# grams. The legacy reading took any ``g`` word for grams, so all four of the first were offered
# as "a nutrient that carries energy", and density was offered on a value already per energy.
_UNIT_NUMERATORS = {"g": "grams", "gm": "grams", "gram": "grams", "grams": "grams",
                    "mg": "milligrams", "mcg": "micrograms", "ug": "micrograms", "iu": "IU",
                    "kcal": "kcal", "kcals": "kcal", "kj": "kj"}
_PER_DAY = {"d", "day", "days", "daily", "perday", "24h", "24hr"}
_PER_PERIOD = {"wk", "wks", "week", "weeks", "weekly", "mo", "month", "months", "monthly", "yr",
               "yrs", "year", "years", "yearly", "annual"}
_PER_BODY = {"kg", "kgbw", "bw", "kgbm", "lbm", "ffm", "m2", "bsa"}
_PER_ENERGY = {"kcal", "kj", "mj", "1000kcal", "100kcal", "1000kj", "kcal1000", "energy"}
_PER_VOLUME = {"l", "dl", "ml", "100ml"}
UNIT_KINDS = ("density", "concentration", "per_body", "per_period")


def amount_unit(name: Any) -> str | None:
    """The unit the name's own unit expression spells, read whole: ``grams``, ``milligrams``,
    ``micrograms``, ``IU``, ``kcal`` or ``kj`` for an amount (alone or per day); ``density`` for an
    amount per unit of energy; ``per_body`` per kg of body weight (or m² of body surface);
    ``per_period`` per week, month or year; ``concentration`` per volume. None when the name spells
    no unit (a lone ``kcal`` is a name, not a unit)."""
    words = tokens(name)
    if len(words) < 2:
        return None
    for i, w in enumerate(words):
        if w not in _UNIT_NUMERATORS:
            continue
        rest = [x for x in words[i + 1:] if x != "per"]
        kinds = set()
        for j, x in enumerate(rest):
            nxt = rest[j + 1] if j + 1 < len(rest) else None
            if x in ("100", "1000") and nxt in ("kcal", "kj"):
                kinds.add("density")
            elif x in _PER_ENERGY:
                kinds.add("density")
            elif x in _PER_VOLUME:
                kinds.add("concentration")
            elif x in _PER_BODY:
                kinds.add("per_body")
            elif x in _PER_PERIOD:
                kinds.add("per_period")
        for kind in UNIT_KINDS:
            if kind in kinds:
                return kind
        return _UNIT_NUMERATORS[w]
    return None


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

# The Framingham Heart Study food-frequency nutrient file (dbGaP phd001373; FHS Coding Manual,
# "Food Frequency Questionnaire Data for Willett Purple Form (88)"): "nutrient fields starting with
# NUT_", e.g. "NUT_CALOR DERIVED FIELD: CALORIES, (kcal)", "NUT_PROT … PROTEIN, (gm)", "NUT_CARBO …
# CARBOHYDRATES, (gm)", "NUT_ALCO … ALCOHOL, (gm)", "NUT_SATFAT … SATURATED FAT, (gm)",
# "NUT_AFAT … ANIMAL FAT, (gm)", "NUT_VFAT … VEGETABLE FAT, (gm)", "NUT_DTFIB … DIETARY FIBER,
# (gm)". The file carries no total fat: its fat is in parts (animal and vegetable; saturated,
# monounsaturated, polyunsaturated). ``AOFIB`` (AOAC fiber) and ``CRUDE`` (crude fiber) measure the
# same fiber as ``DTFIB`` by other methods, so they are read as nutrients that carry no energy of
# their own here, never as a second fiber source (the energy would be counted twice).
_FHS_PREFIX = re.compile(r"^NUT_(?P<code>[A-Z0-9]+)$", re.I)
_FHS_CODES: dict[str, tuple[str, str | None, str | None, str | None]] = {
    # code: (what it is, macro, part, unit as the manual states it)
    "CALOR": ("energy", None, None, "kcal"),
    "PROT": ("protein", "protein", None, "grams"),
    "APROT": ("animal protein", "protein", "animal", "grams"),
    "CARBO": ("carbohydrate", "carbohydrate", None, "grams"),
    "ALCO": ("alcohol", "alcohol", None, "grams"),
    "SATFAT": ("saturated fat", "fat", "sfa", "grams"),
    "MONFAT": ("monounsaturated fat", "fat", "mufa", "grams"),
    "POLY": ("polyunsaturated fat", "fat", "pufa", "grams"),
    "AFAT": ("animal fat", "fat", "animal", "grams"),
    "VFAT": ("vegetable fat", "fat", "plant", "grams"),
    "DTFIB": ("dietary fiber", "fiber", None, "grams"),
    "AOFIB": ("AOAC fiber", None, None, "grams"),
    "CRUDE": ("crude fiber", None, None, "grams"),
    "SUCR": ("sucrose", "carbohydrate", "sugar", "grams"),
    "FRUCT": ("fructose", "carbohydrate", "sugar", "grams"),
    "LACT": ("lactose", "carbohydrate", "sugar", "grams"),
    "CHOL": ("cholesterol", None, None, "milligrams"),
    "SODIUM": ("sodium", None, None, "milligrams"), "K": ("potassium", None, None, "milligrams"),
    "CALC": ("calcium", None, None, "milligrams"), "IRON": ("iron", None, None, "milligrams"),
    "MAGN": ("magnesium", None, None, "milligrams"), "ZN": ("zinc", None, None, "milligrams"),
    "CU": ("copper", None, None, "milligrams"), "PH": ("phosphorus", None, None, "milligrams"),
    "CAFF": ("caffeine", None, None, "milligrams"), "VITC": ("vitamin C", None, None, "milligrams"),
    "FOLATE": ("folate", None, None, "micrograms"), "SE": ("selenium", None, None, "micrograms"),
    "VITK": ("vitamin K", None, None, "micrograms"), "B12": ("vitamin B12", None, None, "micrograms"),
}


def _fhs_code(name: Any) -> tuple[str, str | None, str | None, str | None] | None:
    """The Framingham FFQ nutrient variable ``name`` is (``NUT_PROT``…), or None."""
    m = _FHS_PREFIX.match(str(name))
    return _FHS_CODES.get(m.group("code").upper()) if m else None

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
             "salivary", "tissue", "adipose", "breastmilk", "fecal", "faecal", "stool", "breath"}
# Words that make the column about something other than an amount eaten: a score or a test, a
# medication, a diagnosis or a lab panel (``lipid_disorder``, ``lipid_panel_done``), a behavior or
# a supplement's use (``protein_supplement_use``, ``alcohol_abstainer``), an enzyme
# (``alcohol_dehydrogenase``), a body scan (``dxa_fat_g``), a parenteral infusion
# (``lipid_emulsion_ml``), a vitamin class (``fat_soluble_vit_d``) or a clinical syndrome
# (``protein_energy_wasting``). Names are one signal of three (NUTRITION_PACK §01); the values are
# read too (:func:`intake_check`).
_NOT_AN_INTAKE = {"score", "scale", "index", "test", "meds", "med", "medication", "medications",
                  "drug", "drugs", "rx", "therapy", "lowering", "id", "ids", "code", "flag",
                  "disorder", "disorders", "disease", "diseases", "panel", "done", "profile",
                  "clinic", "allergy", "allergic", "intolerance", "intolerant", "supplement",
                  "supplements", "supplementation", "supp", "suppl", "use", "user", "users",
                  "usage", "abstainer", "abstainers", "abstinence", "abstinent", "dehydrogenase",
                  "enzyme", "oxidase", "synthase", "dxa", "dexa", "scan", "mri", "emulsion",
                  "infusion", "intralipid", "parenteral", "tpn", "soluble", "wasting",
                  "restriction", "restricted", "oxidation"}
_MACRO_DENY: dict[str, set[str]] = {
    "fat": {"mass", "body", "bodyfat", "free", "lean", "liver", "hepatic", "visceral",
            "subcutaneous", "trunk", "android", "gynoid", "area", "volume", "fraction",
            "distribution", "pct", "percent", "low", "reduced", "fatty"},
    "protein": {"reactive", "c", "crp", "binding", "kinase", "creatinine", "electrophoresis"},
    "carbohydrate": {"antigen", "deficient", "transferrin"},
    "alcohol": {"use", "disorder", "dependence", "abuse", "audit", "drinker", "drinkers",
                "drinking", "drinks", "drink", "frequency", "freq", "status", "history", "ever",
                "never", "current", "former", "binge", "problem", "problems", "withdrawal",
                "days", "units", "unit", "glasses"},
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
_LAB_QUALIFIERS = {"protein": {"total", "tot"}, "cholesterol": {"total", "tot"}}
# A diet named by what it restricts or favors (``low_carb_diet``, ``LowFatDiet``,
# ``high_protein_diet``) is a dietary pattern or a trial arm, not an amount of the nutrient.
_DIET_PATTERN_TAILS = {"diet", "diets", "pattern", "group", "arm"}
_SUPPLEMENT = {"supplement", "supplements", "supplementation", "supp", "suppl"}


@dataclass(frozen=True)
class NutrientReading:
    """What a column's name says it is as an intake."""

    column: str
    nutrient: str  # "protein", "saturated fat", "sodium", …
    macro: str | None  # the energy-bearing macronutrient it is (or is a part of), else None
    part: str | None  # sfa | mufa | pufa | trans | sugar | starch | animal | plant | dairy
    source: Literal["nhanes", "infoods", "fhs", "name"]


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
    fhs = _fhs_code(raw)
    if fhs is not None:
        return fhs[3]
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
    fhs = _fhs_code(raw)
    if fhs is not None:
        what, macro, part, _unit = fhs
        return None if what == "energy" else NutrientReading(raw, what, macro, part, "fhs")
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
    unit = _intake_unit(raw)
    # A supplement's amount in an intake unit is an intake (``protein_supplement_g``); without one
    # the name says the supplement is taken (``protein_supplement_use``).
    denied = _NOT_AN_INTAKE - (_SUPPLEMENT if unit else set())
    if present & (_SPECIMEN | denied) or concentration_unit(raw):
        return None
    if len(words) > 1 and words[-1] in _DIET_PATTERN_TAILS:
        return None
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


# ── corroboration by the values ───────────────────────────────────────────────

# NUTRITION_PACK §01: "match on three signals jointly, never names alone". A name read as an
# energy-bearing macronutrient is corroborated by a second signal and contradicted by values that
# no intake shows. The FAO general Atwater factors (kcal/g) turn a median amount into energy, and
# NUTRITION_PACK §02's loosest screen in circulation (sex-neutral 500–5,000 kcal a day) bounds a
# day's energy: a column whose median alone would carry more is no day's intake in grams (a DXA
# total fat of 25,000 g would be 225,000 kcal).
ATWATER_KCAL_PER_G = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0, "fiber": 2.0}
MAX_DAY_KCAL = 5_000.0
# With no unit and no codebook, the name is one signal; under an energy column the second is that
# the column rises with total energy, as an energy source must on average (audit IN-01: "require a
# nutrient to correlate positively with energy before it becomes a default adjustment target").
# One-sided Fisher z test of r > 0 at 1%: an unrelated column passes 1 time in 100.
CORRELATION_Z = 2.326


@dataclass(frozen=True)
class IntakeCheck:
    """Whether a column the name reads as an energy-bearing nutrient reads as a day's intake by its
    values too, and why (one sentence's clause, naming the signal that decided).

    ``corroborated`` is False when the values contradict the name (the reading is withdrawn).
    ``by_values`` is True only when the values themselves agree: a codebook variable whose
    documentation states what it is, or a column that rises with total energy. A name (with or
    without a unit) that nothing in the values confirms stands as the name's reading only, and the
    proposal says so (audit WP13 gate repair: such names were proposed "high" with a reason that
    hid the doubt)."""

    corroborated: bool
    why: str
    r: float | None = None
    by_values: bool = False


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3g}"


# A day's protein, fat or carbohydrate is never under 1% of a day's energy: the adult acceptable
# ranges are 10-35%, 20-35% and 45-65% (IOM Dietary Reference Intakes for Macronutrients, 2005),
# and even a ketogenic diet keeps carbohydrate near 5%. A column whose median carries less is no
# day's amount of the nutrient in grams (``protein_g_kg`` at a median of 1.0 is 0.2%): TurboTab's
# own floor, set a fifth of the way below the lowest diet in practice. Without an energy column the
# floor is read against the loosest screen's lowest day (500 kcal).
MIN_ENERGY_SHARE = 0.01
MIN_DAY_KCAL = 500.0
_FLOORED = ("protein", "carbohydrate", "fat")


def _rises_with(x: Any, energy: Any) -> tuple[float, bool, int]:
    """``(r, passes, n)``: Pearson's r with total energy and whether the one-sided Fisher z test of
    r > 0 passes at 1%."""
    import numpy as np
    import pandas as pd

    e = pd.to_numeric(pd.Series(energy), errors="coerce").reindex(x.index)
    both = pd.DataFrame({"x": x, "e": e}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(both)
    r = float(both["x"].corr(both["e"])) if n >= 4 else float("nan")
    if not math.isfinite(r):
        return r, False, n
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(max(n - 3, 1))
    return r, z >= CORRELATION_Z, n


def intake_check(name: Any, values: Any, *, energy: Any = None) -> IntakeCheck | None:
    """The values' verdict on a column :func:`read_nutrient` reads as an energy-bearing
    macronutrient amount, or None when the name reads as none (nothing to corroborate), or as an
    amount per body weight, per energy, per week or per volume (no day's energy to check).

    Contradicted, whatever the name: two values only (a yes/no, ``lipid_disorder``), negative
    values (no amount eaten is), a median that would carry more than a day's energy as that
    macronutrient (``dxa_fat_g`` at 25,000 g), or, for protein, fat and carbohydrate, less than 1%
    of the day's energy (:data:`MIN_ENERGY_SHARE`); with the name alone and an energy column, a
    column that does not rise with it. Corroborated by the values (``by_values``): a codebook
    variable (``DR1TFAT``, ``PROCNT``, ``NUT_PROT``), or a column that rises with ``energy``
    (one-sided test of r > 0 at 1%). A name with an intake unit and nothing else stands, said as
    the name's reading only."""
    import numpy as np
    import pandas as pd

    try:
        reading = read_nutrient(name)
    except AmbiguousNutrient:
        return None
    if reading is None or reading.macro is None:
        return None
    from turbotab.core.methods.energy import unit_of

    unit = unit_of(name)
    if unit in UNIT_KINDS:
        return None
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    present = x[np.isfinite(x.to_numpy(dtype=float))]
    if len(present) < 3:
        return IntakeCheck(True, "too few values to judge, so the name decides")
    distinct = np.unique(present.to_numpy(dtype=float))
    if len(distinct) <= 2:
        shown = " and ".join(f"`{_fmt(v)}`" for v in distinct)
        return IntakeCheck(False, f"it holds only {shown}, a yes/no, not an amount eaten")
    if float(present.min()) < 0:
        return IntakeCheck(False, "it holds negative values, which no amount eaten has")
    median = float(present.median())
    if unit == "kcal":
        day_kcal = median
    elif unit in ("grams", "unmarked"):
        day_kcal = median * ATWATER_KCAL_PER_G[reading.macro]
    else:
        day_kcal = None
    if day_kcal is not None and day_kcal > MAX_DAY_KCAL:
        return IntakeCheck(False, f"its median, `{_fmt(median)}`, would carry `{_fmt(day_kcal)}` "
                                  f"kcal a day as {reading.macro}, more than any day's intake")
    if day_kcal is not None and reading.macro in _FLOORED and reading.part is None:
        e = pd.to_numeric(pd.Series(energy), errors="coerce") if energy is not None else None
        e_median = float(e.median()) if e is not None and e.notna().any() else float("nan")
        # Against an energy column read in kcal or kJ, whichever is the more lenient floor; else
        # against the loosest screen's lowest day.
        day = e_median if math.isfinite(e_median) and e_median > 0 else MIN_DAY_KCAL
        share = day_kcal * (4.184 if math.isfinite(e_median) else 1.0) / day
        if share < MIN_ENERGY_SHARE:
            return IntakeCheck(False, f"its median, `{_fmt(median)}`, would carry under 1% of a "
                                      f"day's energy as {reading.macro}, so it is no day's amount "
                                      f"in grams")
    if reading.source in ("nhanes", "infoods", "fhs"):
        return IntakeCheck(True, "a codebook variable whose documentation states the unit",
                           by_values=True)
    if energy is not None:
        r, rises, n = _rises_with(x, energy)
        if rises:
            said = ("its name and unit agree, and it rises with total energy"
                    if unit in ("grams", "kcal", "kj") else "it rises with total energy")
            return IntakeCheck(True, f"{said} (r = {r:.2f})", r, by_values=True)
        if unit in ("grams", "kcal", "kj"):
            shown = f"r = {r:.2f}" if math.isfinite(r) else "it cannot be compared"
            return IntakeCheck(True, f"only its name and unit say it is an intake: it does not "
                                     f"rise with total energy ({shown})",
                               r if math.isfinite(r) else None)
        if not math.isfinite(r):
            return IntakeCheck(False, "only its name says it is an intake, and it cannot be "
                                      "compared with total energy")
        return IntakeCheck(False, f"only its name says it is an intake: it has no unit and does "
                                  f"not rise with total energy (r = {r:.2f})", r)
    if unit in ("grams", "kcal", "kj"):
        return IntakeCheck(True, "only its name and unit say it is an intake; there is no total "
                                 "energy to check it against")
    return IntakeCheck(True, "only its name says it is an intake; there is no energy column to "
                             "check it against")


# ── total energy ──────────────────────────────────────────────────────────────

_ENERGY_WORDS = {"energy", "kcal", "kcals", "kj", "calories", "calorie", "kilocalories",
                 "kilocalorie", "kilojoules", "kilojoule", "enerc", "ener", "tei"}
# The words a total-energy-intake name may carry beside its energy word: what the total is
# (``total``, ``daily``, ``intake``), where it was reported (``dietary``, ``ffq``, ``recall``),
# a summary (``mean``, ``usual``), an occasion (``baseline``, ``day1``, ``w2``) and a unit. Any
# other word says the energy is something else (audit WP13 repair): spent (``exercise_kcal``,
# ``activity_energy_kcal``, ``PAEE_kj``, ``steps_kcal``, ``met_kcal``), felt
# (``sf36_energy_fatigue``, ``phq9_energy``, ``little_energy``, ``energy_level_score``), a
# requirement, a goal, a share or a food's. An allow-list, so a word nobody listed is not read as
# intake by default.
_ENERGY_COMPANIONS = {
    "total", "tot", "daily", "day", "days", "d", "per", "intake", "intakes", "ei", "dietary",
    "diet", "food", "foods", "reported", "report", "self", "mean", "avg", "average", "usual",
    "habitual", "estimated", "est", "ffq", "recall", "recalls", "24h", "24hr", "baseline", "bl",
    "followup", "fu", "en", "consumed", "consumption", "value", "raw", "all", "weekly", "week",
    "wk",
}
_OCCASION = re.compile(r"^(?:\d+|(?:day|d|r|recall|w|wave|wk|week|v|visit|y|yr|year|t|time|m|"
                       r"month|bl|fu|tp|p|period)\d+|\d+(?:d|day|days|h|hr|hrs|y|yr|yrs|m|mo|w|"
                       r"wk))$")
# An expenditure, a requirement, a basal rate or a goal is energy, but not intake.
_NOT_INTAKE_ENERGY = {"expenditure", "expend", "expended", "tee", "ree", "bmr", "rmr", "pal",
                      "burn", "burned", "burnt", "basal", "resting", "requirement",
                      "requirements", "eer", "goal", "goals", "target", "need", "needs",
                      "balance", "density", "dense", "share", "pct", "percent", "ratio", "from",
                      "drink", "drinks", "bar", "bars", "adjusted", "residual"}
_PER_DENOMINATOR = {"kg", "g", "1000", "1000kcal", "mj", "bw", "kgbw", "m2"}


def reads_as_total_energy(name: Any) -> bool:
    """Whether a column's name reads as total energy intake (``energy_kcal``, ``DR2TKCAL``,
    ``TotalKcal``, ``ENERC_KCAL``, ``energy_kj``, ``energy_intake``). Not a macronutrient's own
    energy (``alc_kcal``, ``protein_kcal``, ``kcal_from_fat``), not a share of energy, and not an
    expenditure, a requirement, a goal or a questionnaire item: every word beside the energy word
    must be one a total intake's name carries (:data:`_ENERGY_COMPANIONS`)."""
    raw = str(name)
    fhs = _fhs_code(raw)
    if fhs is not None:
        return fhs[0] == "energy"
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
    if "per" in words and not any(w == "per" and i + 1 < len(words) and words[i + 1] in
                                  {"day", "d", "week", "wk"} for i, w in enumerate(words)):
        return False
    if any(w not in _ENERGY_WORDS and w not in _ENERGY_COMPANIONS and not _OCCASION.match(w)
           for w in words):
        return False
    try:
        return read_nutrient(raw) is None
    except AmbiguousNutrient:
        return False


# The values' check on a total-energy name: a population's median day of energy lies inside
# NUTRITION_PACK §02's loosest screen (sex-neutral 500–5,000 kcal a day) read in kcal at its floor
# and in kJ at its ceiling (5,000 × 4.184 = 20,920 kJ). A median of 50 (an SF-36 vitality score)
# or of 2 (a PHQ-9 item) is no day's energy in either unit.
ENERGY_MEDIAN_RANGE = (500.0, 5_000.0 * 4.184)


def energy_median_contradicts(median: Any) -> str | None:
    """Why a total-energy column's median says it is not a day's energy intake, or None."""
    try:
        m = float(median)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(m):
        return None
    lo, hi = ENERGY_MEDIAN_RANGE
    if m < lo:
        return (f"its median, `{_fmt(m)}`, is below {lo:,.0f}, no day's energy intake in kcal or "
                f"kJ")
    if m > hi:
        return (f"its median, `{_fmt(m)}`, is above {hi:,.0f}, no day's energy intake in kcal or "
                f"kJ")
    return None


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


# ── total energy, corroborated by the macronutrients ─────────────────────────

# A total-energy name is one signal (NUTRITION_PACK §01: "match on three signals jointly, never
# names alone"); the values are another. Total energy intake is, by construction, the energy its
# macronutrients carry: NUTRITION_PACK §01's Atwater reconstruction E = 4P + 4C + 9F + 7A, which
# on NHANES 2017–2018 day-1 recalls agrees with DR1TKCAL to within the general factors' error. An
# energy *expenditure* exported beside a diet record (Fitabase's Fitbit dictionary:
# dailyActivity_merged "Calories … Total estimated energy expenditure (in kilocalories)"; ActiLife's
# "Kcals") says kcal in its name and does not follow what people ate (audit WP13 gate repair:
# ``Calories`` and ``Kcals`` were read as total energy intake, r −0.03 to −0.08 with the
# macronutrients, over the real intake column). TurboTab's own cut-offs: a column that tracks the
# reconstruction at r >= 0.7 reads as total energy by its values; one under 0.3 does not, whatever
# its name; in between, the name decides and the proposal says the values agree only loosely.
ENERGY_CORROBORATE_R = 0.7
ENERGY_CONTRADICT_R = 0.3
_RECONSTRUCT = ("protein", "carbohydrate", "fat", "alcohol")
MIN_RECONSTRUCTED_ROWS = 10


def macro_totals(frame: Any, exclude: Iterable[str] = ()) -> dict[str, str]:
    """``{macronutrient: column}`` for the protein, carbohydrate, fat and alcohol totals the names
    read as a day's amount in grams (a codebook's, a ``_g`` suffix, or unmarked), whose values are
    amounts (numeric, never negative, more than two values). A part (saturated fat, sugars) is no
    total."""
    import numpy as np
    import pandas as pd

    skip = set(exclude)
    from turbotab.core.methods.energy import unit_of

    out: dict[str, str] = {}
    for c in frame.columns:
        if c in skip:
            continue
        try:
            reading = read_nutrient(c)
        except AmbiguousNutrient:
            continue
        if reading is None or reading.macro not in _RECONSTRUCT or reading.part is not None:
            continue
        if reading.macro in out or unit_of(c) not in ("grams", "unmarked"):
            continue
        x = pd.to_numeric(frame[c], errors="coerce")
        v = x[np.isfinite(x.to_numpy(dtype=float))]
        if len(v) < 3 or float(v.min()) < 0 or v.nunique() <= 2:
            continue
        out[reading.macro] = str(c)
    return out


def energy_against_macros(frame: Any, column: str,
                          macros: Mapping[str, str] | None = None) -> IntakeCheck | None:
    """What the values say about ``column`` as total energy intake, read against the energy its
    macronutrients carry (:func:`macro_totals`, FAO factors 4/4/9/7), or None when there is nothing
    to read it against (fewer than two of protein, carbohydrate and fat, or too few rows).

    ``corroborated`` False: it does not follow the macronutrients (r < 0.3), whatever its name.
    ``by_values`` True: it follows them (r >= 0.7). Otherwise the name decides and ``why`` says the
    values agree only loosely."""
    import numpy as np
    import pandas as pd

    if column not in frame.columns:
        return None
    macros = dict(macros if macros is not None else macro_totals(frame, exclude=[column]))
    macros = {k: v for k, v in macros.items() if v != column and v in frame.columns}
    if len([k for k in macros if k != "alcohol"]) < 2:
        return None
    parts = [pd.to_numeric(frame[c], errors="coerce") * ATWATER_KCAL_PER_G[m]
             for m, c in macros.items()]
    reconstructed = sum(p.fillna(0.0) if m == "alcohol" else p
                        for p, m in zip(parts, macros))
    e = pd.to_numeric(frame[column], errors="coerce")
    both = pd.DataFrame({"e": e, "r": reconstructed}).replace([np.inf, -np.inf], np.nan).dropna()
    both = both[(both["r"] > 0) & (both["e"] > 0)]
    if len(both) < MIN_RECONSTRUCTED_ROWS or both["e"].nunique() < 3:
        return None
    r = float(np.corrcoef(both["e"], both["r"])[0, 1])
    if not math.isfinite(r):
        return None
    listed = ", ".join(f"`{c}`" for c in macros.values())
    if r >= ENERGY_CORROBORATE_R:
        return IntakeCheck(True, f"its values follow the energy {listed} carry (r = {r:.2f})", r,
                           by_values=True)
    if r < ENERGY_CONTRADICT_R:
        return IntakeCheck(False, f"its values do not follow the energy {listed} carry "
                                  f"(r = {r:.2f}), as a total energy intake must", r)
    return IntakeCheck(True, f"its values follow the energy {listed} carry only loosely "
                             f"(r = {r:.2f})", r)


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
    # ``arm_id``, ``group_id``, ``diet_id``: the label of what a study compares, not a record's
    # name (audit WP13 repair: an RCT's arm coded 1/2 was proposed "identifier", "Names each
    # unit; 2 units"); :func:`study_group` reads it.
    if tail and core and core <= _STUDY_GROUP_WORDS:
        return None
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


# The only words that may stand before an acquisition word: what was acquired and how
# (``extraction_batch``, ``lcms_run_order``, ``elisa_plate``, ``sample_well``). Any other word
# makes the phrase something else: ``insulin_injection`` and ``steroid_injection`` are a
# treatment, ``sleep_well`` and ``eat_well`` an answer, ``potato_chip`` a food, ``morning_run`` an
# activity (audit WP13 repair). The acquisition columns are matched as whole names (IN-02).
_ACQUISITION_QUALIFIERS = {
    "sample", "samples", "extraction", "extract", "analysis", "analytical", "assay", "lab",
    "laboratory", "lc", "ms", "gc", "nmr", "lcms", "gcms", "uplc", "hplc", "elisa", "pcr", "qpcr",
    "array", "microarray", "seq", "sequencing", "library", "lib", "prep", "preparation",
    "processing", "measurement", "acquisition", "acq", "instrument", "machine", "plate", "well",
    "batch", "run", "storage", "box", "kit", "lot", "reagent", "shipment", "tmt", "itraq", "dna",
    "rna", "metabolomics", "lipidomics", "proteomics", "omics", "pos", "neg", "positive",
    "negative", "ion", "mode", "injection", "inj", "chip", "lane", "flowcell", "ms1", "ms2",
    "original", "raw", "qc",
}


def acquisition_kind(name: Any) -> str | None:
    """``batch``, ``plate``, ``well``, ``run_order``, ``run``, ``plex`` or ``polarity`` when the
    column records how samples were acquired; None otherwise. The name is the acquisition phrase,
    optionally after words naming what was acquired (:data:`_ACQUISITION_QUALIFIERS`) and before an
    identifier word or a number."""
    words = tokens(name)
    while words and (words[-1] in _ID_TAILS or words[-1].isdigit()):
        words = words[:-1]
    for size in (2, 1):
        if len(words) >= size and tuple(words[-size:]) in _ACQUISITION_PHRASES:
            before = words[:-size]
            if all(w in _ACQUISITION_QUALIFIERS or w.isdigit() for w in before):
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
    ``diet_group``, ``condition``, ``phenotype``, ``arm_id``)."""
    words = tokens(name)
    while len(words) > 1 and words[-1] in _ID_TAILS:
        words = words[:-1]
    return bool(words) and words[-1] in _STUDY_GROUP_WORDS and not is_identifier(name)


def has_id_tail(name: Any) -> bool:
    """The name ends in an identifier word (``arm_id``, ``group_no``)."""
    words = tokens(name)
    return len(words) > 1 and words[-1] in _ID_TAILS


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

# Words that are a sampling weight on their own (``pweight``, ACS ``PERWT``, ``finalwgt``), and
# words that make a weight word a sampling weight: the survey's own vocabulary.
_SURVEY_WEIGHT_WORDS = {"pweight", "sampweight", "sampwt", "survwt", "svywt", "svyweight",
                        "surveyweight", "samplingweight", "designweight", "perwt", "pwgtp", "wgtp"}
_SURVEY_CONTEXT = {"survey", "sampling", "design", "svy", "probability", "inverse", "raking",
                   "poststratified", "poststrat", "household"}
# Words that make a weight a sampling weight in a survey and something weighed elsewhere: a
# laboratory's ``sample_wt`` is the tissue mass extracted, a rodent study's ``final_weight`` and
# ``base_weight`` are body weights. Beside them a weight word is a sampling weight only when the
# table names its survey design (audit WP13 gate repair).
_AMBIGUOUS_CONTEXT = {"sample", "samp", "smp", "final", "fin", "base", "person"}
_WEIGHT_WORDS = {"weight", "weights", "wt", "wts", "wgt", "wght"}
_BODY_WORDS = {"birth", "body", "bw", "baby", "infant", "newborn", "fetal", "gestational",
               "maternal", "pregnancy", "gain", "loss", "change", "kg", "lb", "lbs", "g", "grams",
               "gram", "oz", "ideal", "target", "current", "usual", "self", "reported",
               # A mass unit or a specimen makes a weight something weighed (audit WP13 repair:
               # ``sample_weight_mg``, a tissue mass, was proposed "A sampling weight").
               "mg", "ug", "mcg", "ng", "milligram", "milligrams", "microgram", "micrograms",
               "tissue", "biopsy", "specimen", "wet", "dry", "organ", "tumor", "tumour", "liver",
               "brain", "muscle", "fresh", "net", "gross"}
NHANES_WEIGHT_PREFIX = ("WTDR", "WTMEC", "WTINT", "WTSA", "WTSB", "WTSOG", "WTSAF", "WTSH",
                        "WTSCD", "WTSPO", "WTSVOC", "WTSHM", "WTFSM", "WTSSB")


def reads_as_survey_weight(name: Any, *, median: float | None = None,
                           design_in_table: bool = False) -> bool:
    """A sampling weight, never a body or birth weight (audit IN-10). The one weight reader every
    stage shares (the roles proposal and the survey question alike).

    An NHANES weight by its exact prefix (``WTDRD1``, ``WTMEC2YR``, ``WTSAF2YR``); a word that is a
    sampling weight on its own (``pweight``, ``finalwgt``, ``PERWT``) or a weight word beside the
    survey's vocabulary (``survey_weight``, ``sampling_wt``, ``final_weight``). Read only beside a
    survey design the table names (strata or primary sampling units, ``design_in_table``): a weight
    beside the ambiguous ``sample`` (``sample_wt``), a word ending in a weight word (BRFSS
    ``_LLCPWT``), or a bare ``weight``/``wt`` whose values are far above any body weight. A birth,
    body or gain word, a mass unit or a specimen is never a sampling weight."""
    raw = str(name)
    if re.fullmatch(r"[A-Za-z0-9]+", raw) and raw.upper().startswith(NHANES_WEIGHT_PREFIX):
        return True
    words = tokens(raw)
    present = set(words)
    if present & _BODY_WORDS:
        return False
    if present & _SURVEY_WEIGHT_WORDS:
        return True
    if present & _WEIGHT_WORDS and present & _SURVEY_CONTEXT:
        return True
    if not design_in_table:
        return False
    if present & _WEIGHT_WORDS and present & _AMBIGUOUS_CONTEXT:
        return True
    if set(words) <= {"weight", "weights", "wt", "final", "w"}:
        return median is not None and math.isfinite(float(median)) and float(median) > 1_000
    # BRFSS's final weight ``_LLCPWT``, ``finalwgt``, ``wtfinal``: one word ending or starting in
    # a weight word, never a body's (``bodyweight``), with values a weight can take (above 1).
    word = words[0] if len(words) == 1 else ""
    if len(word) <= 3 or any(b in word for b in ("body", "birth", "bw")) \
            or word.endswith(("kg", "lb", "lbs", "gm")):
        return False
    from turbotab.core.methods.dietary_caveats import energy_related

    if energy_related(raw) is not None:
        return False  # NHANES's BMXWT is the body weight, beside the survey's own design
    if not (word.endswith(("wt", "wgt", "wght", "weight")) or word.startswith(("wt", "wgt"))):
        return False
    return median is None or (math.isfinite(float(median)) and float(median) > 1)


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
# dietary days, the morning fasting subsample (drawn from the examined sample), and the oral
# glucose tolerance test subsample (drawn from the fasting one). Each name as the CDC codebooks
# spell it, read 2026-10-03: the 2-year and 4-year names (DR1TOT_L "WTDRD1 - Dietary day one sample
# weight", "WTDR2D - Dietary two-day sample weight"; GLU_L "WTSAF2YR - Fasting Subsample 2 Year MEC
# Weight"; DRXTOT_B "WTDR4YR - Dietary day one 4-Year sample weight"), the 2017-March 2020
# pre-pandemic names (P_DEMO "WTINTPRP - Full sample interview weight", "WTMECPRP - Full sample MEC
# exam weight"; P_DR1TOT "WTDRD1PP - Dietary day one sample weight", "WTDR2DPP - Dietary two-day
# sample weight"; P_GLU "WTSAFPRP - Fasting Subsample Weight") and the OGTT subsample's (OGTT_I
# "WTSOG2YR - OGTT Subsample MEC Weight": "Specific sample weights for this subsample are included
# in this data file and should be used when analyzing these data.").
WEIGHT_TIERS: tuple[tuple[str, str, int], ...] = (
    (r"^WTINT(2YR|4YR|PRP)?$", "interview", 0),
    (r"^WTMEC(2YR|4YR|PRP)?$", "examination", 1),
    (r"^(WTDRD1(PP)?|WTDR4YR)$", "dietary day 1", 2),
    (r"^WTDR2D(PP)?$", "dietary days 1 and 2", 3),
    (r"^WTSAF(2YR|4YR|PRP)?$", "fasting subsample", 4),
    (r"^WTSOG(2YR|4YR|PRP)?$", "oral glucose tolerance test subsample", 5),
)
# Laboratory variables measured on the morning fasting subsample (NHANES GLU, TRIGLY and INS
# files: fasting glucose LBXGLU/LBDGLUSI, triglycerides LBXTR/LBDTRSI, LDL LBDLDL/LBDLDLSI, insulin
# LBXIN/LBDINSI), which the tutorial says take the fasting weight.
FASTING_ANALYTES = re.compile(r"^(LBXGLU|LBDGLUSI|LBXTR|LBDTRSI|LBDLDL|LBDLDLSI|LBDLDLM|LBDLDLN|"
                              r"LBXIN|LBDINSI|LBXAPB|LBDAPBSI)$")
# The OGTT file's analytes (OGTT_I: "LBXGLT - Two Hour Glucose (OGTT) (mg/dL)", and its SI twin).
OGTT_ANALYTES = re.compile(r"^(LBXGLT|LBDGLTSI)$")
_DIETARY_VARIABLE = re.compile(r"^(DR1T|DR2T|DR1I|DR2I|DR1_|DR2_|DRXT|DBQ|DBD)")
_SUBSAMPLE = {4: "fasting subsample", 5: "oral glucose tolerance test subsample"}


def weight_tier(name: Any) -> tuple[str, int] | None:
    """``(the sample the weight describes, its rank from widest to smallest)``, or None."""
    raw = str(name).upper()
    for pattern, label, rank in WEIGHT_TIERS:
        if re.match(pattern, raw):
            return label, rank
    return None


def other_subsample_weight(name: Any) -> bool:
    """An NHANES subsample weight whose sample TurboTab does not know (``WTSA2YR``, ``WTSB2YR``,
    ``WTSH2YR``, ``WTSVOC2Y``…): the weight of whichever analytes that file holds."""
    raw = str(name).upper()
    return (bool(re.fullmatch(r"[A-Z0-9]+", raw)) and raw.startswith(NHANES_WEIGHT_PREFIX)
            and weight_tier(raw) is None)


def _expected(rank: int, prepandemic: bool) -> str:
    if prepandemic:
        return {5: "WTSOG2YR", 4: "WTSAFPRP", 3: "WTDR2DPP", 2: "WTDRD1PP"}[rank]
    return {5: "WTSOG2YR", 4: "WTSAF2YR", 3: "WTDR2D", 2: "WTDRD1"}[rank]


# A subsample's variables are recorded only on its rows: in an NHANES merge, a fasting analyte is
# present exactly where WTSAF2YR is positive, whatever the variable is called (``LBDLDNSI``,
# ``LBDLDMSI`` from TRIGLY_J, a glucose renamed ``fasting_glucose`` on import). The values
# corroborate where the name list cannot (audit WP13 gate repair: those names were absent from
# FASTING_ANALYTES, and the dietary weight was named, SETTLED). TurboTab's own tolerance: at most
# 1% of a variable's recorded rows may lie outside the subsample (a merge's stray rows), and the
# variable must be missing on at least 5% of the table, or it is no subsample's.
SUBSAMPLE_SLACK = 0.01
SUBSAMPLE_MIN_MISSING = 0.05
SUBSAMPLE_MIN_ROWS = 10


def subsample_variables(frame: Any, weight: str, *, skip: Iterable[str] = ()) -> list[str]:
    """The variables recorded only where ``weight`` is positive: those measured on its subsample."""
    import pandas as pd

    if weight not in frame.columns:
        return []
    w = pd.to_numeric(frame[weight], errors="coerce")
    inside = w.notna() & (w > 0)
    n = len(frame)
    if not n or int(inside.sum()) >= n:
        return []
    skipped = set(skip) | {weight}
    out = []
    for c in frame.columns:
        if c in skipped or weight_tier(c) is not None or other_subsample_weight(c):
            continue
        present = frame[c].notna()
        k = int(present.sum())
        if k < SUBSAMPLE_MIN_ROWS or k > (1 - SUBSAMPLE_MIN_MISSING) * n:
            continue
        if int((present & ~inside).sum()) <= SUBSAMPLE_SLACK * k:
            out.append(str(c))
    return out


def least_common_denominator(columns: Sequence[str], frame: Any = None) -> dict[str, Any] | None:
    """The NHANES weight the least-common-denominator rule names for these columns, or None when
    the table carries no NHANES weight.

    The weight follows the smallest sample whose variables are present: an OGTT analyte names the
    OGTT subsample weight; a fasting analyte names the fasting weight; dietary variables name the
    dietary weight; otherwise the examination weight. ``because`` says which variables set it.
    With ``frame``, a variable recorded only where a subsample weight is positive is that
    subsample's whatever its name (:func:`subsample_variables`). ``other_subsamples`` lists
    subsample weights whose analytes TurboTab cannot tell (``WTSA2YR``…) and ``unconfirmed`` the
    known subsample weights (``WTSAF2YR``) beside which no variable reads as measured on the
    subsample: with either, the rule cannot be applied from what the table shows, and the choice is
    the user's (``settled`` False)."""
    weights = {}
    for c in columns:
        tier = weight_tier(c)
        if tier is not None:
            weights.setdefault(tier[1], (str(c), tier[0]))
    others = [str(c) for c in columns if other_subsample_weight(c)]
    if not weights and not others:
        return None
    prepandemic = any(str(w).upper().endswith(("PRP", "PP")) for w, _ in weights.values())
    present = set(map(str, columns))
    by_values: dict[int, list[str]] = {}
    other_found: dict[str, list[str]] = {}
    if frame is not None:
        for rank in (5, 4):
            if rank in weights:
                found = [c for c in subsample_variables(frame, weights[rank][0]) if c in present]
                if found:
                    by_values[rank] = found
        for w in others:
            found = [c for c in subsample_variables(frame, w) if c in present]
            if found:
                other_found[w] = found
    ogtt = list(dict.fromkeys([*(str(c) for c in columns if OGTT_ANALYTES.match(str(c).upper())),
                               *by_values.get(5, [])]))
    fasting = list(dict.fromkeys([*(str(c) for c in columns
                                    if FASTING_ANALYTES.match(str(c).upper())),
                                  *[c for c in by_values.get(4, []) if c not in ogtt]]))
    dietary = [str(c) for c in columns if _DIETARY_VARIABLE.match(str(c).upper())]
    # A subsample weight beside which nothing reads as its variable (by name or by values): the rule
    # cannot tell whether the analysis uses it.
    unconfirmed = [weights[r][0] for r in (4, 5) if r in weights
                   and not (fasting if r == 4 else ogtt)]
    # An unknown subsample weight whose variables the values found is no longer unknown.
    unknown = [w for w in others if w not in other_found]
    wanted, because = None, []
    if ogtt:
        wanted, because = 5, ogtt
    elif fasting:
        wanted, because = 4, fasting
    elif dietary:
        wanted, because = 2, dietary
    base = {"other_subsamples": unknown, "unconfirmed": unconfirmed,
            "settled": not unknown and not unconfirmed and not other_found}
    if other_found and wanted not in (4, 5):
        # A subsample weight the names cannot place, with variables recorded only on its rows: it is
        # that subsample's weight, the smallest sample here (the one with the fewest such rows).
        name = min(other_found, key=lambda w: _positive_rows(frame, w))
        return {"use": name, "sample": "subsample", "because": other_found[name],
                "not": [w for _, (w, _) in sorted(weights.items())]
                + [w for w in others if w != name], "missing": None, **base,
                "settled": not unknown and not unconfirmed}
    if wanted is None:
        if not weights:
            return {"use": None, "sample": None, "because": [], "not": [], "missing": None, **base}
        rank = 1 if 1 in weights else min(weights)
        name, label = weights[rank]
        return {"use": name, "sample": label, "because": [], "not": [w for r, (w, _) in
                                                               sorted(weights.items()) if r != rank],
                "missing": None, **base}
    # Both dietary days are a smaller sample than day 1: prefer it when the table has it and both
    # days' variables are present.
    if wanted == 2 and 3 in weights and any(str(c).upper().startswith(("DR2T", "DR2I", "DR2_"))
                                            for c in columns):
        wanted = 3
    if wanted in weights:
        name, label = weights[wanted]
        return {"use": name, "sample": label, "because": because,
                "not": [w for r, (w, _) in sorted(weights.items()) if r != wanted], "missing": None,
                **base}
    label = {5: _SUBSAMPLE[5], 4: _SUBSAMPLE[4], 2: "dietary day 1", 3: "dietary days 1 and 2"}[wanted]
    return {"use": None, "sample": label, "because": because,
            "not": [w for _, (w, _) in sorted(weights.items())],
            "missing": _expected(wanted, prepandemic), **base}


def _positive_rows(frame: Any, weight: str) -> int:
    import pandas as pd

    w = pd.to_numeric(frame[weight], errors="coerce")
    return int((w > 0).sum())


__all__ = [
    "AmbiguousNutrient", "ENERGY_PRIOR_SOURCE", "codebook_unit", "FASTING_ANALYTES", "IdKind", "KCAL_PRIOR",
    "KJ_PRIOR", "MACROS", "NHANES_LCD_QUOTE", "NHANES_WEIGHTING_SOURCE", "NutrientReading",
    "ATWATER_KCAL_PER_G", "ENERGY_MEDIAN_RANGE", "IntakeCheck", "OGTT_ANALYTES",
    "acquisition_kind", "concentration_unit", "energy_median_contradicts", "energy_unit",
    "energy_unit_by_magnitude", "has_id_tail", "id_kind", "intake_check", "other_subsample_weight", "is_identifier", "is_nutrient", "is_rate", "least_common_denominator",
    "names_a_person", "nutrient_role", "read_nutrient", "reads_as_measurement",
    "reads_as_survey_weight", "reads_as_time", "reads_as_total_energy", "study_group", "tokens",
    "weight_tier",
]
