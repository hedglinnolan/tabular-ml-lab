"""Columns that can group the participants, read by their structure (the routing gate's leash note).

The grouping question (``turbotab/core/estimand.py``, audit RO-08) used to be asked only of a column
the recognizer named a site, centre, household or batch. The routing verifier found that names miss
groupings every week: ``study_site``, an integer ``region``, ``trial_site``, ``recruitment_centre``,
``gp_practice`` and ``hosp`` were not recognized, and ``facility_id``, ``physician_id`` and
``doctor_id`` were read as row identifiers, so the question was skipped and the intervals were HC3
where CR2 belonged (MODELING_SEQUENCE §2: repeated units or clusters imply cluster-aware intervals
under inference). A grouping's own structure is what makes clustering matter, so under inference the
question is asked of **any column that can structurally group rows**:

* its values repeat: more than :data:`GROUPING_LEVELS` distinct values, and at least
  :data:`GROUPING_ROWS` rows on the median value and on average (several rows per level);
* they are labels: text, or whole numbers (a value with decimals is a measurement);
* the packs read no characteristic or measured quantity in its name (an age, a sex, an income, a
  blood pressure, minutes of activity: :mod:`turbotab.core.covariate_guesses`);
* and, for whole numbers whose name reads as no grouping, the counts do not fall away from the
  middle of the values as a measured quantity's distribution does (:func:`profile`: Spearman's ρ
  between each value's count and its distance from the median, at most
  :data:`MEASUREMENT_PROFILE`); under an assay's lens the numbers are its measured features (a
  gene's counts, a metabolite's intensities), so only a grouping's name asks about one.

Names never settle anything here (BLUEPRINT §14): a name that reads as a grouping adds the
question, and one that reads as a measured quantity scopes it out, but whether rows sharing a value
belong together is only ever the user's answer (the ``cluster`` reading,
``readings.cluster_reading``). The question shows a guess for each column: it **groups the
participants** (a name that reads as a grouping or an identifier, codes with digits, whole numbers
whose counts follow no order), or it is **a category with many labels** (words, such as country of
birth). A sex, an education level, or any category of ten values or fewer never triggers it.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

GROUPING_LEVELS = 10  # more than ten values (three to ten are a category or a small study's units)
GROUPING_ROWS = 2.0  # rows per value, on the median value and on average
MEASUREMENT_PROFILE = -0.5  # counts falling away from the middle: a measurement's distribution
READ_AT_MOST = 300  # columns read by their values per table (wide assays: named ones beyond)
STRUCTURAL_VERSION = 1

# Words that name a group of participants, whatever stands beside them (``study_site``,
# ``recruitment_centre``, ``gp_practice``, ``facility_id``): a name may only add the question.
GROUP_WORDS = {
    "site", "sites", "center", "centre", "centres", "centers", "clinic", "clinics", "hospital",
    "hospitals", "hosp", "practice", "practices", "gp", "household", "hh", "family", "school",
    "schools", "village", "cluster", "ward", "community", "district", "region", "regions",
    "county", "province", "state", "country", "area", "facility", "facilities", "physician",
    "doctor", "provider", "clinician", "surgeon", "interviewer", "recruiter", "team", "batch",
    "plate", "neighborhood", "neighbourhood", "zip", "postcode", "tract", "municipality",
}


# A place a person comes from is their characteristic, not the group the study sampled them in
# (``country_of_birth``, ``region_of_origin``): such a name adds nothing.
_ORIGIN_WORDS = {"birth", "born", "origin", "nativity", "ancestry"}


def named_grouping(name: Any) -> bool:
    """The name reads as a group of participants, or as an identifier that may be one."""
    from turbotab.core.recognizers import id_kind, tokens

    words = set(tokens(name))
    if words & _ORIGIN_WORDS:
        return False
    return bool(words & GROUP_WORDS) or id_kind(name) in ("cluster", "record")


def measured_name(name: Any) -> bool:
    """The packs read a measured quantity in the name (a characteristic, a clinical measurement,
    a body measure, a lifestyle amount, a nutrient): its whole numbers are no group's labels."""
    from turbotab.core.covariate_guesses import read_name

    found = read_name(name)
    return found is not None and found.cls != "medication"


def profile(values: Any) -> float | None:
    """Spearman's ρ between each distinct value's count and its distance from the values' median:
    near −1 when the counts fall away from the middle, as a measured quantity's distribution does
    (a bell or a skewed hump); near 0 for labels, whose counts follow no order. None when fewer
    than four values."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if s.nunique() < 4:
        return None
    counts = s.value_counts()
    centre = float(np.median(s.to_numpy(dtype=float)))
    distance = np.abs(counts.index.to_numpy(dtype=float) - centre)
    if np.ptp(counts.to_numpy()) == 0:
        return 0.0  # every value equally often: labels of groups of one size
    rho = stats.spearmanr(counts.to_numpy(dtype=float), distance).statistic
    return None if rho is None or not np.isfinite(rho) else float(rho)


def structural_facts(store: Any, columns: Sequence[Mapping[str, Any]], *,
                     target: str | None, assay: bool = False) -> list[dict[str, Any]]:
    """Each column that structurally can group rows, with what its values say (the roles stage's
    ``groupings``; the grouping question filters them by the state, :func:`candidates`).

    ``assay``: the lens is an assay's (metabolomics, genomics). Its numbers are the measured
    features (counts or intensities, amounts by that answer, as ``readings.unsettled_codes`` reads
    them), so a numeric column is read only when its name reads as a grouping."""
    import pandas as pd

    pre = []
    for c in columns:
        name = str(c["name"])
        if name == target or name.startswith("__"):
            continue
        dtype = str(c.get("dtype") or "")
        if dtype not in ("text", "categorical", "integer", "numeric"):
            continue
        if assay and dtype in ("integer", "numeric") and not named_grouping(name):
            continue
        n_present = int(c.get("n") or 0)
        k = int(c.get("n_unique") or 0)
        if k <= GROUPING_LEVELS or not n_present or n_present / k < GROUPING_ROWS:
            continue
        pre.append((name, dtype))
    if not pre:
        return []
    named = [n for n, _ in pre if named_grouping(n)]
    read = list(dict.fromkeys([*named, *(n for n, _ in pre)]))[:max(READ_AT_MOST, len(named))]
    dtypes = dict(pre)
    numeric = [n for n in read if dtypes[n] in ("integer", "numeric")]
    whole = store.whole_numbers(numeric) if numeric else {}
    keep = [n for n in read if dtypes[n] in ("text", "categorical")
            or (whole.get(n) or {}).get("whole")]
    if not keep:
        return []
    frame = store.materialize(keep)
    out = []
    for name in keep:
        s = frame[name].dropna()
        if s.empty:
            continue
        counts = s.astype(str).value_counts() if dtypes[name] in ("text", "categorical") \
            else s.value_counts()
        k, n = int(len(counts)), int(counts.sum())
        median_rows = float(counts.median())
        if k <= GROUPING_LEVELS or n / k < GROUPING_ROWS or median_rows < GROUPING_ROWS:
            continue
        text = dtypes[name] in ("text", "categorical")
        digits = (float(s.astype(str).str.contains(r"\d").mean()) if text else None)
        out.append({
            "column": name, "levels": k, "rows": n, "median_rows": median_rows,
            "kind": ("codes" if digits is not None and digits >= 0.5 else "words") if text
            else "whole_numbers",
            "profile": None if text else profile(pd.to_numeric(s, errors="coerce")),
            "named": named_grouping(name),
        })
    return out


def guess_of(fact: Mapping[str, Any]) -> tuple[str, str] | None:
    """The guess the question shows for one structural candidate (``yes``: it groups the
    participants; ``no``: a category with many labels) and its evidence, or None when the column
    is no candidate at all (whole numbers read as a measurement)."""
    column = str(fact["column"])
    levels, median = int(fact["levels"]), float(fact["median_rows"])
    shape = f"`{levels:,}` values, `{median:g}` rows on the median one"
    if measured_name(column):
        return None  # the packs read a characteristic or a measured quantity (age, sex, income)
    if fact.get("named"):
        return "yes", f"named like a group of participants or an identifier, and {shape}"
    kind = fact.get("kind")
    if kind == "codes":
        return "yes", f"codes written with digits that repeat: {shape}"
    if kind == "words":
        return "no", f"words that repeat, as a category with many labels does: {shape}"
    rho = fact.get("profile")
    if rho is not None and rho <= MEASUREMENT_PROFILE:
        return None  # the counts fall away from the middle: a measured quantity
    said = f"their counts follow no order (ρ = {rho:.2f})" if rho is not None else "few values"
    return "yes", f"whole numbers that repeat as labels do, {said}: {shape}"


def candidates(state: Any, roles: Any) -> list[dict[str, Any]]:
    """Under inference, the structural candidates the grouping question asks about, each with its
    guess: every column the roles stage read as able to group rows, less the outcome, the grain's
    unit, a column whose cluster reading is already settled (the user's own word, an identifier or
    cluster role the user confirmed), and a column whose role says it is no group's label (energy,
    a design column, a time, a flag, a column left out, a value-corroborated characteristic or
    nutrient, a confirmed amount)."""
    from turbotab.core.readings import cluster_reading, confirmation, proposals_of

    if getattr(state, "purpose", None) != "inference":
        return []
    data = getattr(roles, "data", roles)
    facts = (data or {}).get("groupings") if isinstance(data, Mapping) else None
    if not facts:
        return []
    target = getattr(state, "target", None)
    grain = getattr(state, "grain", None)
    unit = getattr(grain, "id_column", None) if grain is not None else None
    recorded = getattr(state, "roles", None) or {}
    proposed = {str(p.get("column")): p for p in proposals_of(roles)}
    out = []
    for fact in facts:
        column = str(fact["column"])
        if column in (target, unit):
            continue
        role = recorded.get(column) or (proposed.get(column) or {}).get("proposed")
        if role in ("energy", "design", "time", "flag", "excluded"):
            continue
        p = proposed.get(column) or {}
        if p.get("confidence") == "high" and p.get("proposed") in (
                "covariate", "exposure", "energy", "identifier"):
            continue  # its values settled it as a characteristic, a nutrient or the row's name
        if confirmation(state, "code_or_count", column) == "amount":
            continue
        if role in ("identifier", "cluster") and cluster_reading(state, column).settled:
            continue  # whether its rows belong together is already answered
        if (getattr(state, "reading_confirmations", None) or {}).get(f"cluster:{column}"):
            continue
        found = guess_of(fact)
        if found is None:
            continue
        out.append({"column": column, "guess": found[0], "why": found[1], "structural": True})
    return out


__all__ = ["GROUPING_LEVELS", "GROUPING_ROWS", "GROUP_WORDS", "MEASUREMENT_PROFILE",
           "candidates", "guess_of", "measured_name", "named_grouping", "profile",
           "structural_facts"]
