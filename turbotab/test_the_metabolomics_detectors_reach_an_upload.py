"""L50-D — `METABOLOMICS_PACK.md` §01's three families, and the upload that
reaches them.

The metabolomics pack held **three detectors** against a §01 that specifies
three diagnostic families in full, with its regex library written out. What was
missing was not depth in what existed — `_left_censored` reads a rank
correlation and `_pooled_qc` reads variance, and both are good — it was that
**sample-role detection covered one of six families, design detection covered
one column, and value-state detection did not exist.**

## Hardest-first, and what the order bought

`LOOP.md` §02: judge hardest by what is most likely to break the abstraction.
`_repeated_subjects` went first, because it is the only detector here whose
answer **already exists somewhere else in the app** — the grain question holds
it, in the lockbox, with its own contradiction detector and its own two exits.
A pack detector's contract is `df -> Optional[dict]` and cannot see a project,
so the question was whether the abstraction could express *"report this and
route it"* without either asking twice or inventing a second reading. It could:
the finding takes its column FROM `grain.suggestion`, so the two cannot name
different columns, and `fix_kind="none"` makes "asks nothing" structural.

**The abstraction did not bend.** Thirteen detectors, one contract, no new
finding shape, no widening of `Pack.detectors`. What it did was surface four
defects, and all four were found by driving fixtures rather than by reading
code:

- **`str.capitalize()` shipped a false identifier.** The duplicate-id finding
  composed its sentence with `"; ".join(said).capitalize()`, which upper-cases
  the first character and **lower-cases every other one**, so sample `S040`
  went out as `s040`. True about the count, false about the id, in the one
  field a user would search their run list with.
- **The intensity block ate the defect.** `metabolite_columns` drops any column
  with at most two distinct values, to keep a 0/1 outcome out of a range
  reading. An all-zero feature and a constant feature have exactly ONE, so the
  rule dropped every column `_empty_blocks` is for, before it ran.
- **`already_transformed` told a survey researcher their Likert responses had
  been log-transformed.** §01's range readings are about abundances; a block of
  41 items scored 1–5 has a maximum of 5 and a range of 5×.
- **`repeated_subjects` claimed a subject id it had not identified.** The
  lockbox OFFERS roster-shaped columns and asserts nothing about them; this
  sentence says *"subject IDs repeat"*. Without a name to corroborate the shape
  it made that claim about `genomics_expression.csv`.

The first three are `AGENT_ONBOARD.md` §07 trap #3's shape from the production
side — the assertion right, the input to it wrong — and none is visible in the
assertion.

## `GUIDED-097` — the fixture rule

Five metabolomics tables of deliberately different shape. Four are DERIVED from
the first by `sample_data/make_metabolomics_siblings.py`, each through an
operation a real export performs, each with a companion. `SHAPES_NOT_COVERED`
names what none of them reaches.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import grain as G
from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"
PACK = (Path(__file__).resolve().parents[1] / "docs" / "turbotab" /
        "research" / "METABOLOMICS_PACK.md")

#: `GUIDED-097`. Five tables, and the preconditions of the thirteen detectors
#: are in several places exact negations — a table cannot both have pooled QCs
#: and lack them — so this is a tuple by necessity rather than for breadth.
METABOLOMICS_FIXTURES = {
    "the untargeted run as acquired": "metabolomics_untargeted",
    "the export with the QCs and the run order removed": "metabolomics_no_qc",
    "an analyst's file, already log2, with a paired design":
        "metabolomics_paired_logged",
    "the same run with non-detections written as zeros":
        "metabolomics_mzmine_zeros",
    "the two polarities merged badly": "metabolomics_merged_modes",
}

#: NOT COVERED, said out loud. A sweep that reports only what it covered has not
#: reported its coverage.
#:
#: A TARGETED PANEL. §00 forks the pack into seven sub-domains and this
#: repository has tables for exactly one of them, untargeted LC-MS. The
#: divergence §00 calls out is not cosmetic — *"misapplied QC-RSD filtering on a
#: targeted panel deletes validated analytes"* — and a targeted table would be
#: named metabolites, concentration units, LLOQ/ULOQ columns and 100–600
#: features. Nothing here is one, and one could not be DERIVED from the
#: untargeted fixture without inventing analyte identities, so the shape is
#: named rather than manufactured.
#:
#: A COMPRESSED RANGE WITH A MAXIMUM ABOVE 40. §01's dynamic-range reading has
#: a case the log-signature reading does not cover — a panel reported in µM
#: spanning 10 to 500 — and no fixture here has it, which is why the range is a
#: SIGNAL of `already_transformed` rather than a detector of its own. Built the
#: other way it was a capability nothing could reach.
#:
#: DILUTION QC, BLANKS, SYSTEM SUITABILITY, CALIBRANTS AND PROTEOMICS
#: REFERENCES. Five of the six role families have no rows in any shipped table:
#: every fixture here derives from one run whose only non-biological injections
#: are pooled QCs. The library is exercised against constructed frames below and
#: the five families are named here, because a role library verified only on the
#: family that was already detected is the trap this file is about.
#:
#: A LITERAL DUPLICATE COLUMN LABEL. `read_csv` renames a repeated header to
#: `name.1`, so a duplicate feature id cannot ARRIVE as one from any CSV. The
#: renaming signature is what a real file carries and is covered by
#: `metabolomics_merged_modes.csv`; the literal form is reachable only from a
#: constructed frame and is tested as one.
#:
#: AN ACQUISITION TIMESTAMP. `acquisition_timestamp` derives run order from a
#: parsed datetime, which §01 states as an imperative. No shipped table carries
#: one — the untargeted fixture was generated with an integer `run_order` — so
#: the derivation is exercised on a constructed frame and named here.
#:
#: FEATURES IN ROWS. Every table here is samples-in-rows. §01's orientation
#: cascade and `orientation.py` own that fork, and `packs.contradiction` already
#: declines to speak on a feature-major frame; nothing in this file re-tests it.
SHAPES_NOT_COVERED = [
    "a targeted panel — named analytes, concentration units, LLOQ/ULOQ "
    "columns; §00's sub-domain fork has seven branches and this repository has "
    "tables for one",
    "a compressed dynamic range with a maximum above 40 — the case the "
    "log-signature reading does not cover",
    "dilution QC, blank, system-suitability, calibrant and proteomics-reference "
    "rows — five of the six role families, constructed here and in no fixture",
    "a literal duplicate column label — `read_csv` mangles it, so no CSV can "
    "carry one",
    "an acquisition timestamp column — the run-order derivation runs on a "
    "constructed frame",
    "features in rows — `orientation.py` owns that fork and is tested there",
    "a PREFIX polarity convention (`pos_mz_0001`) — the marker is read as the "
    "last token of a feature name, because reading it anywhere calls "
    "`negative_control_probe` a negative-mode feature",
]

#: The ten detectors L50-D added, by their short ids.
NEW_DETECTORS = (
    "sample_roles", "no_pooled_qc", "acquisition_design", "no_run_order",
    "repeated_subjects", "zeros_or_missing", "already_transformed",
    "duplicate_ids", "empty_blocks", "ion_modes",
)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def fired(name: str) -> dict:
    return {f["id"].split("::")[-1]: f
            for f in P.findings(load(name), [P.METABOLOMICS])}


# ── the census · every detector fires on a named fixture ─────────────────────

#: Which fixture each new detector is expected to fire on, and which it must
#: stay quiet on. Derived from the companions rather than from a run: the
#: companions are the stated expectation, and a table copied out of a passing
#: run is a test asserting what the code does.
EXPECTED = {
    "sample_roles": ("metabolomics_untargeted", "metabolomics_no_qc"),
    "no_pooled_qc": ("metabolomics_no_qc", "metabolomics_untargeted"),
    "acquisition_design": ("metabolomics_untargeted", "metabolomics_no_qc"),
    "no_run_order": ("metabolomics_no_qc", "metabolomics_untargeted"),
    "repeated_subjects": ("metabolomics_paired_logged",
                          "metabolomics_untargeted"),
    "already_transformed": ("metabolomics_paired_logged",
                            "metabolomics_untargeted"),
    "zeros_or_missing": ("metabolomics_mzmine_zeros",
                         "metabolomics_untargeted"),
    "duplicate_ids": ("metabolomics_merged_modes", "metabolomics_untargeted"),
    "empty_blocks": ("metabolomics_merged_modes", "metabolomics_untargeted"),
    "ion_modes": ("metabolomics_merged_modes", "metabolomics_untargeted"),
}


@pytest.mark.parametrize("detector", sorted(EXPECTED))
def test_each_detector_fires_where_its_companion_says_and_not_where_it_says_not(
        detector):
    """Both halves, and the second is the one that catches a detector that
    fires on everything. `test_a_pack_does_not_fire_on_the_wrong_data` measures
    questions added and a reporting pack adds none by construction, so a
    finding that is merely WRONG passes it — this is where wrongness is
    measured."""
    fires_on, silent_on = EXPECTED[detector]
    assert detector in fired(fires_on), (
        f"{detector} does not fire on {fires_on}, which its companion says it "
        f"must")
    assert detector not in fired(silent_on), (
        f"{detector} fires on {silent_on}, which its companion says it must "
        f"not")


def test_the_five_fixtures_produce_the_counts_their_companions_state():
    """The whole census in one assertion, so a detector that quietly stops
    firing on a fixture it was not parametrized against still fails."""
    counts = {name: len(P.findings(load(name), [P.METABOLOMICS]))
              for name in METABOLOMICS_FIXTURES.values()}
    assert counts == {
        "metabolomics_untargeted": 5,
        "metabolomics_no_qc": 3,
        "metabolomics_paired_logged": 7,
        "metabolomics_mzmine_zeros": 5,
        "metabolomics_merged_modes": 8,
    }, counts


def test_no_detector_offers_a_repair():
    """Guard #1 — a pack supplies findings and defaults and may not invent a
    card type — as a property of the payload rather than of any one detector.
    `router._is_repairable` reads `fix_kind`, so this is what makes "a
    reporting pack cannot add a question" structural."""
    from ml import router

    produced = []
    for name in METABOLOMICS_FIXTURES.values():
        produced += P.findings(load(name), [P.METABOLOMICS])
    assert len(produced) >= 20, "the fixtures stopped producing findings"
    for finding in produced:
        assert finding["fix_kind"] == "none", finding["id"]
        assert not finding["fix_label"], finding["id"]
        assert not router._is_repairable(finding), finding["id"]


def test_every_new_finding_states_its_reason_and_its_badge():
    """`marker` governs the treatment, so a finding without a reason raises
    confidence without earning it. And a DISPUTED claim may never pre-select."""
    seen = set()
    for name in METABOLOMICS_FIXTURES.values():
        for finding in P.findings(load(name), [P.METABOLOMICS]):
            short = finding["id"].split("::")[-1]
            if short not in NEW_DETECTORS:
                continue
            seen.add(short)
            assert finding["marker"] in ("derived", "convention", "offered")
            assert len(finding["why_it_matters"]) > 60, finding["id"]
            assert len(finding["detail"]) > 60, finding["id"]
            badge = finding["evidence"]
            assert badge["evidence_status"] in P.EVIDENCE_STATUSES
            assert badge["source"].startswith("research/METABOLOMICS_PACK.md#")
            if badge["evidence_status"] == P.DISPUTED:
                assert finding["marker"] == "offered", finding["id"]
    assert seen == set(NEW_DETECTORS), sorted(set(NEW_DETECTORS) - seen)


def test_every_detector_has_a_declared_defer_destination():
    """`GUIDED-153`: a pack finding routed by a DEFAULT is the record saying
    'deferred to the step where it belongs' with nothing having decided where
    that is. Ten new ids, ten declarations, and `defer_destination` raises
    rather than defaulting."""
    from ml import router

    for name in METABOLOMICS_FIXTURES.values():
        for finding in P.findings(load(name), [P.METABOLOMICS]):
            step, label = router.defer_destination(finding)
            assert step in ("data", "explore", "preprocess", "features",
                            "train"), (finding["id"], step)
            assert label


# ── D1 · the role library ────────────────────────────────────────────────────

def test_the_regex_library_is_the_packs_and_not_a_recollection_of_it():
    """**Every pattern this file matches on appears verbatim in §01.**

    `evidence.py` resolves that a source names a real section; it explicitly
    cannot tell whether the claim is faithful to it. For a transcribed regex
    library that gap is the whole risk — a library assembled from memory looks
    exactly like a transcribed one — so the transcription is checked against the
    file rather than asserted in a docstring.
    """
    text = PACK.read_text(encoding="utf-8")
    section = text[text.index("### Sample-role detection"):
                   text.index("### Run order, batch, and design")]
    missing = [pattern for _, patterns in P.ROLE_PATTERNS
               for pattern in patterns if f"`{pattern}`" not in section]
    assert not missing, (
        f"these patterns are not in §01's library: {missing}. A pattern the "
        f"pack does not contain is a claim with no record behind it.")
    # And the other direction, with the one departure named rather than left as
    # an unexplained gap: §01 lists `pool` under BOTH pooled QC and proteomics,
    # so as written the families are not disjoint and every pooled QC in a
    # metabolomics run would be reported as a reference channel.
    listed = set(re.findall(r"`([^`]+)`", section))
    ours = {p for _, patterns in P.ROLE_PATTERNS for p in patterns}
    unused = listed - ours - {"type", "sample_type", "role", "class", "group"}
    assert unused == set(), unused
    assert "pool" in dict(P.ROLE_PATTERNS)[P.POOLED_QC]
    assert "pool" not in dict(P.ROLE_PATTERNS)[P.PROTEOMICS_REFERENCE]


def test_the_absent_case_quotes_the_pack_word_for_word():
    """§01's coaching sentence is the valuable half of this family, and a
    paraphrase of it is a different claim: the quoted one names two citations
    and says the omission cannot be repaired later. Asserted against the
    research file so a later edit to either side fails here."""
    text = PACK.read_text(encoding="utf-8")
    quoted = " ".join(P.NO_POOLED_QC_COACHING.split())
    inline = " ".join(text.replace("> ", " ").replace("*", "").split())
    assert quoted in inline, (
        "the no-QC coaching is no longer §01's sentence verbatim")
    finding = fired("metabolomics_no_qc")["no_pooled_qc"]
    for phrase in ("QC-RSD", "D-ratio", "drift correction",
                   "cannot be reconstructed later"):
        assert phrase in finding["detail"], phrase
    # AND THE PAYLOAD SAYS EVERYTHING THE PROSE DOES. Trap #7 is the
    # machine-readable form being the lossier of the two, and a refusal whose
    # structured half drops what it refuses is exactly that.
    assert finding["params"]["cannot_compute"] == ["QC-RSD", "D-ratio",
                                                   "drift correction"]
    assert finding["params"]["reconstructable"] is False
    assert finding["params"]["scanned_columns"]


@pytest.mark.parametrize("family,name", [
    (P.POOLED_QC, "QC_01"), (P.POOLED_QC, "Plasma_pool"),
    (P.POOLED_QC, "SQC_3"), (P.POOLED_QC, "PQC-2"),
    (P.DILUTION_QC, "dQC_A"), (P.DILUTION_QC, "QC_4x"),
    (P.DILUTION_QC, "RQC_1"), (P.DILUTION_QC, "DIL_02"),
    (P.BLANK, "extraction_blank_1"), (P.BLANK, "BLK03"),
    (P.BLANK, "B01"), (P.BLANK, "solvent_A"), (P.BLANK, "water"),
    (P.SYSTEM_SUITABILITY, "SST_1"), (P.SYSTEM_SUITABILITY, "equil_05"),
    (P.SYSTEM_SUITABILITY, "wash"), (P.SYSTEM_SUITABILITY, "sys_suit_2"),
    (P.CALIBRANT, "CAL3"), (P.CALIBRANT, "STD_high"),
    (P.CALIBRANT, "NIST_SRM1950"), (P.CALIBRANT, "LTR_04"),
    (P.CALIBRANT, "ISTD_mix"),
    (P.PROTEOMICS_REFERENCE, "QC_HeLa_01"), (P.PROTEOMICS_REFERENCE, "iRT_pep"),
    (P.PROTEOMICS_REFERENCE, "bridge_channel_1"),
])
def test_all_six_role_families_are_reachable(family, name):
    """Five of the six have no rows in any shipped fixture — every table here
    derives from one run whose only non-biological injections are pooled QCs —
    so the library is exercised against constructed names and the gap is in
    `SHAPES_NOT_COVERED`."""
    frame = _assay_frame(["S001", "S002", "S003", name])
    census = P.sample_roles(frame)
    assert census["families"][family]["n"] == 1, (
        f"{name!r} was read as {census['present']}, not {family}")
    assert census["n_biological"] == 3


@pytest.mark.parametrize("name", [
    "HISTIDINE_free", "condition_A", "STDEV_sample", "wastewater_02",
    "Sample_ISOLEUCINE", "CALCIUM_high", "BASELINE_1", "ANALYSIS_04",
    "Whirlpool_site", "understandard_x",
])
def test_a_bare_word_pattern_does_not_match_inside_a_longer_word(name):
    """**The boundary rule, which the pack leaves implicit and which is the
    difference between a library and a hazard.**

    Applied as unanchored substrings, §01's bare words are catastrophic: `IS`
    matches `HISTIDINE`, `cond` matches `condition`, `STD` matches `STDEV`,
    `water` matches `wastewater`, `CAL\\d` matches `CALCIUM_1`, `^B\\d+` is
    anchored and correctly does not match `BASELINE_1`. A pack that reports a
    calibrant because a sample is named after an element asserts something
    false in the one place the app has promised it will not.
    """
    frame = _assay_frame(["S001", "S002", "S003", name])
    census = P.sample_roles(frame)
    assert census["present"] == [], (
        f"{name!r} was read as {census['present']} — a bare-word pattern "
        f"matched inside a longer word")
    assert census["n_biological"] == 4


def test_a_role_is_read_from_the_run_type_column_and_not_only_the_name():
    """§01 applies the library to sample names **and** to a metadata column
    named `type`, `sample_type`, `role`, `class` or `group`. A table whose
    sample names are opaque barcodes and whose run type is spelled out is the
    common vendor shape, and the name-only reading is blind to it."""
    frame = _assay_frame(["A7F21", "A7F22", "A7F23", "A7F24"])
    frame["Sample Type"] = ["Sample", "Sample", "Pooled QC", "Blank"]
    census = P.sample_roles(frame)
    assert census["families"][P.POOLED_QC]["n"] == 1
    assert census["families"][P.BLANK]["n"] == 1
    assert "Sample Type" in census["role_columns"]
    assert census["n_biological"] == 2


def test_the_scan_does_not_reach_a_column_the_pack_did_not_name():
    """The restriction to five column names is not a convenience — it is what
    stops `^B\\d+` reading a `batch` column of `B1`/`B2` as extraction blanks.
    Driven, this is what the untargeted fixture would have produced."""
    frame = _assay_frame(["S001", "S002", "S003", "S004"])
    frame["batch"] = ["B1", "B1", "B2", "B2"]
    census = P.sample_roles(frame)
    assert census["present"] == []
    assert "batch" not in census["scanned_columns"]


def test_the_name_reading_and_the_variance_reading_corroborate_each_other():
    """Two independent instruments on one fixture, and the finding says when
    they agree. `_pooled_qc` reads variance and can see exactly one family;
    this reads names and sees six. A disagreement between them is worth
    seeing, so the agreement is recorded rather than assumed."""
    finding = fired("metabolomics_untargeted")["sample_roles"]
    assert finding["params"]["corroborated_by"] == "pack::metabolomics::pooled_qc"
    assert finding["params"]["families"] == {P.POOLED_QC: 8}
    assert finding["params"]["n_biological"] == 72
    variance = fired("metabolomics_untargeted")["pooled_qc"]
    assert variance["params"]["n_qc"] == finding["params"]["families"][P.POOLED_QC]


# ── D2 · run order, batch and design ─────────────────────────────────────────

def test_the_run_order_reading_has_exactly_one_implementation():
    """`_no_run_order` asserts *"there is no run order in this file"* and
    `_acquisition_order` asserts there is one. Two readings of "is there a run
    order" is the arrangement in which the app says both sentences about one
    table, so the permutation reading was factored out and both call it."""
    import inspect

    source = inspect.getsource(P._acquisition_order)
    assert "_permutation_column(df)" in source
    assert "np.arange" not in source, (
        "the permutation reading is back in the detector, which is where the "
        "two answers could diverge")
    untargeted = load("metabolomics_untargeted")
    assert P._permutation_column(untargeted) == "run_order"
    assert P._no_run_order(untargeted) is None
    no_order = load("metabolomics_no_qc")
    assert P._permutation_column(no_order) is None
    assert P._acquisition_order(no_order) is None
    assert P._no_run_order(no_order) is not None


def test_the_loud_absence_names_what_stops_being_possible():
    """§01 says *"say so loudly"*, which is met by naming the three
    diagnostics rather than by an exclamation mark — and the payload carries
    the same three, because that is what everything downstream reads."""
    finding = fired("metabolomics_no_qc")["no_run_order"]
    assert finding["params"]["cannot_compute"] == ["drift", "QC-RLSC",
                                                   "run-order PCA overlay"]
    for phrase in ("drift", "QC-RLSC", "run-order"):
        assert phrase in finding["why_it_matters"]
    assert finding["params"]["n_columns_examined"] == 398


def test_run_order_is_derived_from_an_acquisition_timestamp():
    """§01's one imperative in this family: *"If an acquisition timestamp
    exists, derive run order from it."* No shipped table carries one — named in
    `SHAPES_NOT_COVERED` — so the derivation runs on a constructed frame.

    **And the derivation is checked, not merely reported.** The rank of the
    timestamps has to equal the injection order the frame was built with; a
    test that asserted only *"a timestamp was found"* would pass on a reader
    that returned the rows in file order.
    """
    frame = _assay_frame([f"S{i:03d}" for i in range(40)])
    shuffled = np.array([(i * 17) % 40 for i in range(40)])
    frame["AcquisitionDateTime"] = [
        str(pd.Timestamp("2026-03-01 08:00") + pd.Timedelta(minutes=float(9 * int(k))))
        for k in shuffled]
    read = P.acquisition_timestamp(frame)
    assert read is not None and read["column"] == "AcquisitionDateTime"
    assert list(read["order"]) == list(shuffled + 1), (
        "run order was reported but not derived from the timestamps")

    assert P._no_run_order(frame) is None, (
        "a table with a derivable run order must not be told it has none")
    finding = P._acquisition_design(frame)
    assert finding["params"]["run_order_source"] == "derived_from_timestamp"
    assert "derived from it" in finding["detail"]


def test_the_design_inventory_names_the_columns_it_found_by_role():
    finding = fired("metabolomics_untargeted")["acquisition_design"]
    by_role = finding["params"]["by_role"]
    assert by_role["run_order"] == ["run_order"]
    assert by_role["batch"] == ["batch"]
    assert sorted(by_role["confounder"]) == ["age", "bmi", "sex"]
    assert finding["params"]["run_order_source"] == "named"
    assert finding["severity"] == "info", (
        "an inventory is not a complaint")


def test_a_confounder_name_does_not_match_inside_a_longer_word():
    """`age` as a substring matches `image_id` and `percentage`. A design
    reader that calls an image id a confounder has asserted something false
    about the study, in the record that becomes the methods section."""
    frame = _assay_frame([f"S{i:03d}" for i in range(6)])
    frame["image_id"] = range(6)
    frame["percentage_complete"] = np.linspace(0, 1, 6)
    frame["age_years"] = [40, 41, 42, 43, 44, 45]
    found = P.design_columns(frame)
    assert found["confounder"] == ["age_years"], found["confounder"]


# ── D2 · the grain question through a second door ────────────────────────────

def test_the_subject_finding_routes_to_the_grain_answer_and_asks_nothing():
    """§01 calls repeated subject ids *"routinely missed"*, and the app does
    not miss them — it asks about them at question 3, in the lockbox. So this
    finding **routes** and does not re-ask.

    Trap #3b: the word *routes* in a test name is a claim about a consequence,
    so it is observed here rather than implied. Three observations —
    the id the finding names resolves in the lockbox's own candidate list, the
    lockbox accepts it, and accepting it MOVES the seal basis. A finding
    pointing at a question that cannot take its answer is a route to nowhere.
    """
    from turbotab.project import AnalysisProject

    frame = load("metabolomics_paired_logged")
    finding = fired("metabolomics_paired_logged")["repeated_subjects"]
    column = finding["params"]["group_column"]

    assert finding["params"]["routes_to"] == "set_grain"
    assert finding["params"]["asks_nothing"] is True
    assert finding["fix_kind"] == "none"

    # THE STAND-IN RESOLVES IN THE REAL REGISTRY (trap #3). The column is not
    # chosen here and is not chosen by the detector either — it comes out of
    # `grain.suggestion`, which is what the question itself offers.
    assert column in G.suggestion(frame)["columns"]
    assert finding["params"]["grain_answer"] in G.ANSWERS

    project = AnalysisProject.from_dataframe(frame, "paired")
    project.set_grain(G.PEOPLE_REPEAT, group_col=column)
    assert project.grain["group_col"] == column
    assert project.grain["n_groups"] == finding["params"]["n_subjects"] == 36
    # CHANGE THE ANSWER AND SEE IF ANYTHING DOWNSTREAM MOVES — §07's flip for
    # the case where the question is not import-shaped.
    grouped = project.grain["basis"]
    other = AnalysisProject.from_dataframe(frame, "unpaired")
    other.set_grain(G.ONE_ROW_PER_PERSON, acknowledged_contradiction=True)
    assert grouped != other.grain["basis"], (
        f"both grain answers produce basis {grouped!r}, so routing to the "
        f"question changes nothing")


def test_the_subject_claim_needs_a_name_and_not_only_a_shape():
    """The lockbox OFFERS roster-shaped columns and asserts nothing about them.
    This sentence says *"subject IDs repeat"*, which is a claim that the column
    IS one. Driven without the corroboration it made that claim about
    `genomics_expression.csv` — *"60 samples from 28 subjects"* — on a column
    whose values merely repeat."""
    for name in ("genomics_expression", "survey_instrument",
                 "clinical_longitudinal"):
        frame = load(name)
        assert P._repeated_subjects(frame) is None, (
            f"the subject claim was made about {name} on shape alone")
    # And the positive control: the same reading, with a name behind it.
    assert P._repeated_subjects(load("metabolomics_paired_logged")) is not None


# ── D3 · value states ────────────────────────────────────────────────────────

def test_the_zero_finding_asks_and_never_defaults():
    """§01: *"The pack must ask: do zeros here mean 'not detected' or 'true
    zero'? Defaulting wrong corrupts every downstream step."* A pack may not
    add a question, so the honest form is a report that says nothing has been
    decided — and it names the four tools that disagree, in the payload as well
    as the prose."""
    finding = fired("metabolomics_mzmine_zeros")["zeros_or_missing"]
    assert finding["marker"] == "offered"
    assert finding["params"]["not_defaulted"] is True
    assert finding["params"]["n_zeros"] == 4316
    assert finding["params"]["blanks_and_zeros_coexist"] is False
    vendors = {v["tool"] for v in finding["params"]["vendor_conventions"]}
    assert vendors == {"XCMS", "MZmine", "MaxQuant", "Progenesis"}


def test_a_single_zero_is_not_a_vendor_convention():
    """`wide_assay.csv` produced *"1 zeros across 1 features"* — true,
    ungrammatical, and an interruption about nothing. An export that writes
    zeros for non-detections writes a lot of them."""
    assert P._zeros_or_missing(load("wide_assay")) is None
    assert P._zeros_or_missing(load("metabolomics_untargeted")) is None


def test_the_zero_filled_export_loses_the_left_censoring_evidence():
    """**The concrete cost of the vendor disagreement, measured.**

    `metabolomics_mzmine_zeros.csv` is the same run written by a tool that
    puts `0` where XCMS puts a small number. The left-censoring reading is a
    rank correlation against the MISSING rate, and there is no longer one — so
    a generic tool sees a complete table and the evidence that these are
    non-detections is gone. This is why §01 asks rather than defaults.
    """
    assert "left_censored" in fired("metabolomics_untargeted")
    assert "left_censored" not in fired("metabolomics_mzmine_zeros")
    assert "zeros_or_missing" in fired("metabolomics_mzmine_zeros")


def test_the_transform_warning_is_hard_and_its_marker_moves_with_its_evidence():
    """§01: *"Warn hard; a second log transform is a silent catastrophe."*

    The marker moves and the id does not. A negative value in an abundance
    table is derived — an ion count below zero is not a measurement. A
    compressed maximum has an innocent reading, so where that is the only
    signal the app offers it. `atwater_finding`'s rule: the verdict is a
    parameter, because `LooksFor` and `prior_columns` both bind to an id.
    """
    logged = fired("metabolomics_paired_logged")["already_transformed"]
    assert logged["severity"] == "critical"
    assert logged["marker"] == "offered"
    assert "compressed_max" in logged["params"]["signals"]
    assert logged["params"]["n_negative"] == 0

    negatives = P._already_transformed(load("wide_assay"))
    assert negatives is not None
    assert negatives["id"] == logged["id"], "the id must not vary"
    assert negatives["marker"] == "derived"
    assert "negative_values" in negatives["params"]["signals"]


def test_a_likert_block_is_not_a_log_transformed_intensity_block():
    """**The false statement this detector made before `_INTEGRAL_SHARE_MAX`
    existed.** §01's range readings are about abundances. A block of 41 items
    scored 1–5 has a maximum of 5, a positive minimum and a range of 5×, and
    the app told a survey researcher their responses had been log-transformed.
    """
    for name in ("survey_instrument", "survey_sentinels"):
        assert P._already_transformed(load(name)) is None, name
    read = P.transformation_signals(load("survey_instrument"))
    assert read["reads_as_abundances"] is False
    assert read["integral_share"] == 1.0
    # The positive control, so the guard is not passing by never firing.
    assert P.transformation_signals(
        load("metabolomics_paired_logged"))["reads_as_abundances"] is True


def test_the_compressed_range_is_a_signal_rather_than_a_second_card():
    """§01's two bullets are one reading: *"a max below ~40 with a positive min
    and low dynamic range"* and *"a ratio below 10² means something has already
    been done"*. Built as a separate detector the second had **no fixture in
    this repository able to fire it**, which is a capability with no consumer —
    so it composes a different sentence inside the same finding instead."""
    frame = _assay_frame([f"S{i:03d}" for i in range(20)], features=40)
    rng = np.random.default_rng(7)
    block = [c for c in frame.columns if c.startswith("mz_")]
    frame[block] = rng.uniform(60.0, 3000.0, size=(20, len(block)))
    read = P.transformation_signals(frame)
    assert read["signals"] == ["compressed_range"], read["signals"]
    finding = P._already_transformed(frame)
    assert finding["id"] == "pack::metabolomics::already_transformed"
    assert finding["marker"] == "offered"
    assert "so something has been done to them" in finding["detail"]
    assert "can't say what" in finding["detail"]


def test_the_degenerate_columns_survive_the_block_filter():
    """**The defect the merged fixture found.** `metabolite_columns` drops any
    column with at most two distinct values, to keep a 0/1 outcome out of a
    range reading. An all-zero feature and a constant feature have exactly ONE,
    so the rule dropped every column this detector is for before it ran — and
    the finding reported one empty sample on a table with eight all-zero
    features while saying nothing false."""
    frame = load("metabolomics_merged_modes")
    assert len(P.metabolite_columns(frame)) == 385
    assert len(P.metabolite_columns(frame, keep_degenerate=True)) == 398
    # A genuine 0/1 indicator is still excluded by both.
    assert "responder" not in P.metabolite_columns(frame, keep_degenerate=True)

    finding = fired("metabolomics_merged_modes")["empty_blocks"]
    assert finding["params"]["empty_features_total"] == 8
    assert finding["params"]["constant_features_total"] == 5
    assert finding["params"]["empty_sample_names"] == ["S080"]


def test_a_constant_column_is_not_evidence_that_a_sample_was_measured():
    """A gap filler writes the same constant into every row of a feature it
    could not detect, **including the row of an injection that failed
    outright**. Counting that constant as a value reports the failed injection
    as fine, which is the app asserting something false about the one row a
    person most needs to see."""
    finding = fired("metabolomics_merged_modes")["empty_blocks"]
    frame = load("metabolomics_merged_modes")
    failed = frame.index[frame["sample_id"] == "S080"][0]
    gap_filled = finding["params"]["constant_features"]
    assert (frame.loc[failed, gap_filled] != 0).all(), (
        "the fixture no longer exercises the case")
    assert finding["params"]["empty_sample_rows"] == [int(failed)]


def test_the_duplicate_reading_covers_the_renaming_a_reader_performs():
    """`read_csv` renames a repeated header to `name.1`, so a duplicate feature
    id **never arrives as a duplicate** — it arrives silently renamed, and
    every count of how many metabolites were measured is that many too high.
    Both forms are read; only one of them can come from a CSV."""
    finding = fired("metabolomics_merged_modes")["duplicate_ids"]
    assert finding["params"]["mangled_features_total"] == 6
    assert finding["params"]["literal_features_total"] == 0
    assert finding["params"]["duplicate_samples"] == ["S040"]
    # The identifier is not lower-cased on its way to a person. `capitalize()`
    # shipped it as `s040`, which is false about the one field a user would
    # search their run list with.
    assert "'S040'" in finding["detail"]

    literal = _assay_frame([f"S{i:03d}" for i in range(8)], features=40)
    literal.columns = list(literal.columns[:-1]) + [literal.columns[3]]
    read = P.duplicate_ids(literal)
    assert read["literal_features"] == [literal.columns[3]]


def test_both_polarities_are_reported_as_a_convention_and_not_a_fact():
    """§01 says the merge strategy affects the results and does not say which
    to use, so neither does the app. The badge is the difference between
    telling a user what is normally done and telling them what is true."""
    finding = fired("metabolomics_merged_modes")["ion_modes"]
    assert finding["evidence"]["evidence_status"] == P.CONVENTION_STATUS
    assert finding["marker"] == "offered"
    assert finding["params"]["n_features_by_mode"] == {"negative": 196,
                                                       "positive": 202}
    assert "convention" in finding["why_it_matters"]

    # And from a `polarity` column rather than from feature names, which is
    # what an unmerged export carries.
    frame = _assay_frame([f"S{i:03d}" for i in range(10)])
    frame["polarity"] = ["pos"] * 5 + ["neg"] * 5
    assert P.ion_modes(frame)["modes"] == ["negative", "positive"]
    assert P._ion_modes(frame) is not None


def test_a_negative_control_gene_is_not_a_polarity_marker():
    """`neg` unbounded matches `negative_control_gene`, which is a name and not
    an acquisition mode."""
    frame = _assay_frame([f"S{i:03d}" for i in range(10)])
    frame = frame.rename(columns={"mz_000": "negative_control_probe",
                                  "mz_001": "position_marker"})
    assert P.ion_modes(frame)["modes"] == []


# ── GUIDED-209 · a list that is cut says its bound ───────────────────────────

def test_every_list_a_new_finding_serves_states_its_bound():
    """`GUIDED-209`: any list this loop serves states its bound or is not cut.

    Derived rather than hand-listed — every list-valued key in every new
    finding's payload, checked for the `_shown`/`_total`/`_bound` triple — so a
    list added next loop is unclassified until somebody disposes of it.
    """
    #: Keys that are complete enumerations rather than cuts. Each is bounded by
    #: something small and fixed: the six role families, the four vendors, the
    #: two polarities, the columns a detector scanned.
    NOT_A_CAP = {
        "absent_families", "scanned_columns", "cannot_compute",
        "vendor_conventions", "modes", "signals", "matches",
        "polarity_columns", "candidate_columns", "columns",
    }
    unclassified = []
    for name in METABOLOMICS_FIXTURES.values():
        for finding in P.findings(load(name), [P.METABOLOMICS]):
            if finding["id"].split("::")[-1] not in NEW_DETECTORS:
                continue
            params = finding["params"]
            for key, value in params.items():
                if not isinstance(value, list) or key in NOT_A_CAP:
                    continue
                if key.endswith(("_shown", "_total", "_bound")):
                    continue
                if all(f"{key}_{s}" in params
                       for s in ("shown", "total", "bound")):
                    assert params[f"{key}_shown"] == len(value)
                    assert params[f"{key}_total"] >= params[f"{key}_shown"]
                    continue
                unclassified.append(f"{finding['id']}::{key}")
    assert not unclassified, (
        f"these served lists neither state a bound nor are declared complete: "
        f"{sorted(set(unclassified))}")


# ── the wire · a project, an upload, and the page ────────────────────────────

def test_a_project_sees_them_through_its_own_accessor():
    """`AnalysisProject.pack_findings` is what the interview reads, and the
    lens gates it: with a different lens answered, nothing metabolomic."""
    from turbotab.project import AnalysisProject

    project = AnalysisProject.from_dataframe(
        load("metabolomics_merged_modes"), "merged")
    project.lens = [P.METABOLOMICS]
    ids = {f["id"].split("::")[-1] for f in project.pack_findings()}
    assert {"ion_modes", "duplicate_ids", "empty_blocks"} <= ids, sorted(ids)
    project.lens = [P.OTHER]
    assert not [f for f in project.pack_findings()
                if f["id"].startswith("pack::metabolomics::")]


def test_every_new_finding_reaches_a_person_and_carries_its_badge():
    """**Trap #6, on the door that has already paid for it at six surfaces.**

    `test_the_clinical_detectors_reach_an_upload` proves the RENDERER reaches
    pack findings, on one metabolomics fixture with three of them. This drives
    the two siblings that carry the ten new ones, because a renderer that
    reaches three findings and truncates at eight would pass that test and hide
    most of this part.

    The stack is bounded (`GUIDED-149`), so *"reaches a person"* means pushed
    OR behind an affordance that states its count — and the affordance is
    pressed rather than assumed.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client = TestClient(api.app)
    for fixture, target in (("metabolomics_merged_modes.csv", "responder"),
                            ("metabolomics_paired_logged.csv", "responder")):
        with open(DATA / fixture, "rb") as handle:
            pid = client.post("/project", files={
                "file": (fixture, handle, "text/csv")}).json()["id"]
        for kind, payload in (("set_lens", {"lens": [P.METABOLOMICS]}),
                              ("set_target", {"column": target})):
            ok = client.post(f"/project/{pid}/decision",
                             json={"kind": kind, "payload": payload})
            assert ok.status_code == 200, (fixture, kind, ok.text[:300])

        project = client.get(f"/project/{pid}").json()
        served = [f for f in project["findings"] if f["source"] == "pack"]
        new = [f for f in served if f["id"].split("::")[-1] in NEW_DETECTORS]
        assert new, f"{fixture} served none of the new findings"

        routes = {
            f"/project/{pid}": project,
            f"/project/{pid}/interview?step=data":
                client.get(f"/project/{pid}/interview?step=data").json(),
            f"/project/{pid}/interview?step=explore":
                client.get(f"/project/{pid}/interview?step=explore").json(),
            f"/project/{pid}/evidence/missingness": {"cards": []},
            f"/project/{pid}/capabilities":
                client.get(f"/project/{pid}/capabilities").json(),
        }
        out = PH.run(
            "var shut = (__harness.html('profList') || '');\n"
            "__harness.dispatch('click', __harness.target("
            "{'data-stack-more':'1','aria-expanded':'false'}));\n"
            "__emit({shut: shut.slice(0, 90000),"
            " open: ((__harness.html('profList') || '') +"
            "        (__harness.html('profRest') || '')).slice(0, 200000),"
            " more: (__harness.html('profMore') || '')});",
            routes=routes, search=f"?project={pid}")
        html = out["open"]
        assert out["shut"], f"{fixture}: the findings list rendered nothing"

        missing = [f["id"] for f in new if f["title"][:28] not in html]
        assert not missing, (
            f"{fixture}: the pack computes {missing} and the page never shows "
            f"them, pushed or collapsed.")

        stack = project["explore_stack"]
        behind = [f["id"] for f in new if f["id"] in stack["collapsed"]]
        if behind:
            assert str(stack["remainder"]["n"]) in out["more"], (
                f"{fixture}: {len(behind)} findings sit behind an affordance "
                f"that does not state its count: {out['more'][:200]}")

        statuses = set(re.findall(r'class="badge (\w+)"', html))
        expected = {f["evidence"]["evidence_status"].lower() for f in new}
        assert expected <= statuses, (
            f"{fixture}: these badge statuses are on the wire and not on the "
            f"page: {sorted(expected - statuses)}")


# ── helpers ──────────────────────────────────────────────────────────────────

def _assay_frame(names, features: int = 40, seed: int = 3) -> pd.DataFrame:
    """A minimal assay-shaped table: sample names and enough features that the
    `_is_assay_wide` precondition every §01 detector carries is met."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"sample_id": list(names)})
    for index in range(features):
        frame[f"mz_{index:03d}"] = rng.lognormal(8, 1.5, len(frame))
    return frame
