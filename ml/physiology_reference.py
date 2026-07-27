"""
Physiologic plausibility reference framework.

Empirical plausibility intervals are derived from NHANES-like population distributions
and are distinct from clinical guideline thresholds, which are informational only.

**Two tiers, not one.** A reference interval says a value is unusual for the
reference population; an impossibility band says it cannot describe a living
person. A diastolic blood pressure of 1e-15 is not an outlier at the low end of
a distribution, it is an entry error, and treating both as the same violation
set means the advisory drowns the repair. So each variable may also carry
`floor` / `ceiling` — hard bounds, wider than the reference interval by
construction, outside which the value is a data-entry artifact rather than a
rare patient.

The two tiers get different treatments downstream: outside the reference
interval is advisory and stays advisory; outside the impossibility band gets a
repair proposal (set the entry to missing) with the affected rows shown.

Finding: GUIDED-004.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from ml.clinical_units import CLINICAL_VARIABLES


DEFAULT_NHANES_REFERENCE: Dict[str, Any] = {
    "version": "nhanes_reference_demo_v2_impossibility",
    "source": "NHANES (reference population, demo defaults)",
    "variables": {
        # p01/p99  — the reference interval: unusual for this population.
        # floor/ceiling — the impossibility band: incompatible with a living
        #   person, or with the instrument that produced the number. Deliberately
        #   generous: this tier proposes a repair, so a false positive here costs
        #   more than a missed one, and the reference interval still catches the
        #   merely improbable.
        "glucose":      {"unit": "mg/dL", "p01": 70,  "p99": 200,  "floor": 10,   "ceiling": 2000},
        "bmi":          {"unit": "kg/m²", "p01": 15,  "p99": 50,   "floor": 8,    "ceiling": 200},
        "hba1c":        {"unit": "%",     "p01": 4.0, "p99": 15.0, "floor": 2.0,  "ceiling": 30.0},
        "cholesterol":  {"unit": "mmol/L", "p01": 2.0, "p99": 10.0, "floor": 0.3, "ceiling": 40.0},
        "triglyceride": {"unit": "mg/dL", "p01": 50,  "p99": 500,  "floor": 5,    "ceiling": 10000},
        "weight":       {"unit": "kg",    "p01": 35,  "p99": 200,  "floor": 0.4,  "ceiling": 650},
        "height":       {"unit": "cm",    "p01": 140, "p99": 210,  "floor": 20,   "ceiling": 280},
        "waist":        {"unit": "cm",    "p01": 55,  "p99": 150,  "floor": 20,   "ceiling": 350},
        "bp_sys":       {"unit": "mmHg",  "p01": 90,  "p99": 200,  "floor": 40,   "ceiling": 300},
        "bp_di":        {"unit": "mmHg",  "p01": 50,  "p99": 120,  "floor": 15,   "ceiling": 220},
        "kcal":         {"unit": "kcal",  "p01": 800, "p99": 4500, "floor": 100,  "ceiling": 30000},
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


def match_variable_key(col_name: str, reference: Dict[str, Any]) -> Optional[str]:
    col_lower = col_name.lower()
    for key in reference.get("variables", {}).keys():
        if key in col_lower:
            return key
    return None


def get_reference_interval(reference: Dict[str, Any], var_key: str) -> Optional[Tuple[float, float, str]]:
    var_data = reference.get("variables", {}).get(var_key)
    if not var_data:
        return None
    return float(var_data["p01"]), float(var_data["p99"]), var_data["unit"]


def get_impossibility_band(reference: Dict[str, Any],
                           var_key: str) -> Optional[Tuple[float, float, str]]:
    """Hard floor and ceiling for a variable, or None when none is published.

    Distinct from `get_reference_interval` in what it licenses. A value outside
    the reference interval is *improbable* and stays advisory. A value outside
    this band is *impossible* — no living person, or no working instrument —
    and earns a repair proposal.

    Returns None rather than falling back to the reference interval. A missing
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


def band_is_wider_than_interval(reference: Dict[str, Any], var_key: str) -> bool:
    """The tiers must nest: impossible ⊃ improbable, never the other way.

    A band narrower than the reference interval would call ordinary values
    impossible and propose deleting them. Checked rather than assumed —
    `test_the_impossibility_band_contains_the_reference_interval` runs this over
    every variable in the bundled reference.
    """
    interval = get_reference_interval(reference, var_key)
    band = get_impossibility_band(reference, var_key)
    if interval is None or band is None:
        return True
    ref_low, ref_high, _ = interval
    floor, ceiling, _ = band
    return floor <= ref_low and ceiling >= ref_high

