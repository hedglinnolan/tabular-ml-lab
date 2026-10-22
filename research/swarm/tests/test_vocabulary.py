"""The controlled vocabulary (SWARM_PLAN §2, §4; ``vocabulary.py``): every marker §4 lists, on its
axis and with when it is known, read off §4's own lists here by hand; the four want kinds of §2; and
the digest a record keeps, rebuilt here from the vocabulary's own content."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from research.swarm import vocabulary as V

# SWARM_PLAN §4, "Markers in the pilot", each list in its order, as the names the vocabulary
# gives them.
PLAN = {
    ("regime", "live"): ["rows", "rows_analyzed", "repeated_units", "effective_size",
                         "events_per_parameter", "predictors_by_role", "predictors_per_row",
                         "outcome_type", "event_share", "missingness", "complete_case_share",
                         "design", "time_order"],
    ("measurement", "live"): ["units", "detection_limit_share", "batch_columns", "run_order",
                              "imputed_copies", "repeated_measurement", "omics_scale"],
    ("measurement", "declared"): ["instrument", "assay_platform", "preprocessing",
                                  "declared_units", "recall_method"],
    ("purpose", "live"): ["purpose", "primary_score", "intended_use", "thresholds", "estimand",
                          "multiplicity_family"],
    ("post_hoc", "post_hoc"): ["signal", "result"],
}


def test_every_marker_of_section_4_is_in_the_vocabulary_once_and_nothing_else():
    names = [m.name for m in V.MARKERS]
    assert len(names) == len(set(names))
    assert sorted(names) == sorted(n for group in PLAN.values() for n in group)
    for (axis, timing), group in PLAN.items():
        for name in group:
            m = V.BY_NAME[name]
            assert m.timing == timing, name
            if axis != "post_hoc":
                assert m.axis == axis, name
    assert set(V.LIVE) | set(V.DECLARED) | set(V.POST_HOC) == set(names)


def test_the_want_kinds_are_section_2s_four():
    assert V.WANT_KINDS == ("option", "check", "reading", "eligibility_rule")


def test_the_digest_is_of_the_vocabulary_itself():
    body = {"version": V.VERSION, "markers": [asdict(m) for m in V.MARKERS],
            "want_kinds": list(V.WANT_KINDS)}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert V.DIGEST == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_an_unknown_name_is_said_with_what_the_vocabulary_holds():
    said = V.unknown("hunch", V.WANT_KINDS)
    assert said == ("'hunch' is not in the vocabulary; it is one of option, check, reading, "
                    "eligibility_rule.")
