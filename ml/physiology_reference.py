"""
Physiologic plausibility reference framework.

Empirical plausibility intervals are derived from NHANES-like population distributions
and are distinct from clinical guideline thresholds, which are informational only.

**Two tiers, not one.** An improbability band says a value is unusual for the
reference population; an impossibility band says it cannot describe a living
person. A diastolic blood pressure of 1e-15 is not an outlier at the low end of
a distribution, it is an entry error, and treating both as the same violation
set means the advisory drowns the repair. So each variable may also carry
`floor` / `ceiling` — hard bounds, wider than the improbability band by
construction, outside which the value is a data-entry artifact rather than a
rare patient.

The two tiers get different treatments downstream: outside the improbability
band is advisory and stays advisory; outside the impossibility band gets a
repair proposal (set the entry to missing) with the affected rows shown.

**Neither tier is a REFERENCE INTERVAL, and the word is not used for them here
any more (`MISC-018`).** A reference interval is the central 95% of a healthy
reference population — the 2.5th to 97.5th percentile, CLSI EP28-A3c, minimum
reference sample n=120. `p01`/`p99` is the central 98% and is a different
quantity; for `bp_sys` it gives 90-200 mmHg where the published adult reference
interval is 90-120. Calling it one asserted a clinical term of art this file
does not hold, to both doors.

Finding: GUIDED-004.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from ml.clinical_units import CLINICAL_VARIABLES


DEFAULT_NHANES_REFERENCE: Dict[str, Any] = {
    "version": "nhanes_reference_demo_v3_aliases",
    "source": "NHANES (reference population, demo defaults)",
    # `aliases` is how a column earns this variable's bounds. Matching is exact
    # against the key or one of these, after case and separators are stripped —
    # never by substring, which would let `hba1c_proxy` inherit HbA1c's floor
    # and its licence to propose deleting entries. A name that is not here gets
    # no bounds and no flags: silence is a gap, and a wrong bound is a claim.
    "variables": {
        # p01/p99  — the improbability band: unusual for this population.
        #   NOT a reference interval, which is the central 95% (`MISC-018`).
        # floor/ceiling — the impossibility band: incompatible with a living
        #   person, or with the instrument that produced the number. Deliberately
        #   generous: this tier proposes a repair, so a false positive here costs
        #   more than a missed one, and the improbability band still catches
        #   the merely improbable.
        "glucose":      {"unit": "mg/dL", "p01": 70,  "p99": 200,  "floor": 10,   "ceiling": 2000,
                         "aliases": ["blood_glucose", "serum_glucose", "plasma_glucose",
                                     "fasting_glucose", "glucose_mgdl", "glu", "lbxglu"]},
        "bmi":          {"unit": "kg/m²", "p01": 15,  "p99": 50,   "floor": 8,    "ceiling": 200,
                         "aliases": ["body_mass_index", "bmxbmi"]},
        "hba1c":        {"unit": "%",     "p01": 4.0, "p99": 15.0, "floor": 2.0,  "ceiling": 30.0,
                         "aliases": ["a1c", "hemoglobin_a1c", "glycated_hemoglobin",
                                     "hgba1c", "lbxgh"]},
        "cholesterol":  {"unit": "mmol/L", "p01": 2.0, "p99": 10.0, "floor": 0.3, "ceiling": 40.0,
                         "aliases": ["total_cholesterol", "chol", "tc", "lbxtc"]},
        "triglyceride": {"unit": "mg/dL", "p01": 50,  "p99": 500,  "floor": 5,    "ceiling": 10000,
                         "aliases": ["triglycerides", "trig", "tg", "lbxtr"]},
        "weight":       {"unit": "kg",    "p01": 35,  "p99": 200,  "floor": 0.4,  "ceiling": 650,
                         "aliases": ["body_weight", "wt", "weight_kg", "bmxwt"]},
        "height":       {"unit": "cm",    "p01": 140, "p99": 210,  "floor": 20,   "ceiling": 280,
                         "aliases": ["standing_height", "ht", "height_cm", "bmxht"]},
        "waist":        {"unit": "cm",    "p01": 55,  "p99": 150,  "floor": 20,   "ceiling": 350,
                         "aliases": ["waist_circumference", "waist_cm", "bmxwaist"]},
        "bp_sys":       {"unit": "mmHg",  "p01": 90,  "p99": 200,  "floor": 40,   "ceiling": 300,
                         "aliases": ["systolic", "systolic_bp", "sbp", "bp_systolic",
                                     "bpxsy1"]},
        "bp_di":        {"unit": "mmHg",  "p01": 50,  "p99": 120,  "floor": 15,   "ceiling": 220,
                         "aliases": ["diastolic", "diastolic_bp", "dbp", "bp_diastolic",
                                     "bpxdi1"]},
        "kcal":         {"unit": "kcal",  "p01": 800, "p99": 4500, "floor": 100,  "ceiling": 30000,
                         "aliases": ["energy", "calories", "kilocalories", "energy_kcal",
                                     "dr1tkcal"]},
    },
}


def _build_clinical_guidelines() -> Dict[str, Any]:
    guidelines: Dict[str, Any] = {}
    for var_name, var_config in CLINICAL_VARIABLES.items():
        if "thresholds" in var_config:
            guidelines[var_name] = {
                "canonical_unit": var_config.get("canonical_unit"),
                "thresholds_by_unit": var_config.get("thresholds", {}),
                "fasting_note": var_config.get("fasting_note", False),
                "source": "Clinical guidelines (informational only)"
            }
    return guidelines


@lru_cache(maxsize=1)
def load_nhanes_reference(reference_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NHANES-based empirical plausibility reference.
    Attempts a dynamic pull when NHANES_REFERENCE_URL is set.
    Falls back to bundled defaults if unavailable.
    """
    url = reference_url or os.getenv("NHANES_REFERENCE_URL")
    if url:
        try:
            with urlopen(url, timeout=5) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if _validate_reference(data):
                    return data
        except (URLError, HTTPError, ValueError):
            pass
    return DEFAULT_NHANES_REFERENCE


@lru_cache(maxsize=4)
def load_reference_bundle(reference_url: Optional[str] = None) -> Dict[str, Any]:
    """Return both empirical NHANES reference and clinical guideline overlays. Cached per process."""
    return {
        "nhanes": load_nhanes_reference(reference_url),
        "clinical": _build_clinical_guidelines()
    }


def _validate_reference(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if "variables" not in data:
        return False
    if not isinstance(data["variables"], dict) or len(data["variables"]) == 0:
        return False
    for _, payload in data["variables"].items():
        if not isinstance(payload, dict):
            return False
        if "unit" not in payload or "p01" not in payload or "p99" not in payload:
            return False
    return True


def _normalize_name(name: str) -> str:
    """A column or alias reduced to comparable form: lowercase, no separators."""
    return re.sub(r"[^0-9a-z]+", "", str(name).lower())


def match_variable_key(col_name: str, reference: Dict[str, Any]) -> Optional[str]:
    """The reference variable this column *is*, or None.

    **Exact key or declared alias. Nothing else.**

    This used to match by substring — `if key in col_lower` — which answered an
    allowlist question with an accident. `hba1c_proxy`, `bp_sys_delta` and
    `weight_change` inherited their parent variable's improbability bands and,
    once the impossibility band landed, its licence to propose deleting entries.
    L9b contained that with a closed list of modifier words, which was the wrong
    shape for the same reason: it catches `hba1c_proxy` because `proxy` is
    listed and misses `hba1c_v2`, `hba1c_imputed` and `hba1c_lab2` because they
    are not. A denylist cannot answer "is this that variable?".

    So an unrecognized name gets **no bounds and no flags**. That is the
    governing rule's *may be silent* branch, taken deliberately: saying nothing
    about `hba1c_lab2` is a gap, and applying HbA1c's floor to it is the app
    asserting something it does not know. Aliases are how a real column earns
    its bounds — declared in the reference, per variable, and readable.
    """
    target = _normalize_name(col_name)
    if not target:
        return None
    variables = reference.get("variables", {})
    for key, payload in variables.items():
        names = [key]
        if isinstance(payload, dict):
            names += list(payload.get("aliases") or [])
        if any(_normalize_name(n) == target for n in names):
            return key
    return None


def get_improbability_band(reference: Dict[str, Any],
                           var_key: str) -> Optional[Tuple[float, float, str]]:
    """The `p01`/`p99` pair, which is the **improbability** band.

    **`MISC-018`. This was called `get_reference_interval` and it never
    returned one.** A reference interval is a defined quantity — the central
    95% of a healthy reference population, the 2.5th to 97.5th percentile, per
    CLSI EP28-A3c, which also sets a minimum reference sample of n=120
    (`research/CLINICAL_SURVEY_PACK.md` §A1.2). What this returns is the
    central **98%**, from a percentile pair, and for `bp_sys` that is
    `90–200 mmHg` where §A1.2's own table gives the typical adult reference
    interval as `90–120`. The upper bound was wrong by a **category**, not by
    calibration.

    The old name asserted a clinical term of art to both doors, and
    `get_impossibility_band`'s docstring below still had the right sentence
    with the wrong noun in it. **The code knew what it held; the name did
    not** — which is the governing rule failing in an identifier.

    **Renamed rather than aliased.** An alias would leave the false name
    importable and let a future reader trust it, and `GUIDED-120`'s and
    `GUIDED-124`'s precedent is that a shared-core defect is corrected once in
    core rather than papered over per door.

    **What this is NOT.** It is not a licence to call the band a reference
    interval anywhere else, and it is not a substitute for one. Whether a real
    `p025`/`p975` pair is worth carrying is a separate decision under D4: it is
    reference data and must be read from primary sources, because a wrong
    reference interval is worse than none — a clinician reads that name and
    believes it.
    """
    var_data = reference.get("variables", {}).get(var_key)
    if not var_data:
        return None
    return float(var_data["p01"]), float(var_data["p99"]), var_data["unit"]


def get_impossibility_band(reference: Dict[str, Any],
                           var_key: str) -> Optional[Tuple[float, float, str]]:
    """Hard floor and ceiling for a variable, or None when none is published.

    Distinct from `get_improbability_band` in what it licenses. A value outside
    the improbability band is *improbable* and stays advisory. A value outside
    this band is *impossible* — no living person, or no working instrument —
    and earns a repair proposal.

    Returns None rather than falling back to the improbability band. A missing
    band means the tier is unknown for that variable, and answering "unknown"
    with the weaker bound would silently promote improbable values to
    impossible ones and propose deleting real data.
    """
    var_data = reference.get("variables", {}).get(var_key)
    if not var_data:
        return None
    if "floor" not in var_data or "ceiling" not in var_data:
        return None
    floor, ceiling = float(var_data["floor"]), float(var_data["ceiling"])
    if not floor < ceiling:
        return None
    return floor, ceiling, var_data["unit"]


def impossibility_contains_improbability(reference: Dict[str, Any],
                                         var_key: str) -> bool:
    """The tiers must nest: impossible ⊃ improbable, never the other way.

    An impossibility band narrower than the improbability band would call
    ordinary values impossible and propose deleting them. Checked rather than
    assumed — `test_the_impossibility_band_contains_the_improbability_band`
    runs this over every variable in the bundled reference.

    **`MISC-018`: renamed with the thing it compares.** It was
    `band_is_wider_than_interval`, and the *interval* in that name was the
    percentile pair above, which is not an interval. A predicate that names its
    operands wrongly is the same defect as a getter that does, one call deep.
    """
    improbable = get_improbability_band(reference, var_key)
    band = get_impossibility_band(reference, var_key)
    if improbable is None or band is None:
        return True
    low, high, _ = improbable
    floor, ceiling, _ = band
    return floor <= low and ceiling >= high

