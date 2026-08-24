"""L51 — `GENOMICS_PACK.md` §01's four gene-ID diagnostics, and the upload that
reaches them.

§01 asks for four readings off one identifier classification:

> **Diagnostic:** classify vocabulary; report version suffixes present (joins
> against unversioned annotation fail silently, dropping genes), duplicate IDs
> after symbol mapping (many-to-one), mixed vocabularies, and **Excel
> corruption** — date-like strings (`1-Mar`, `2-Sep`, `44621`).

and one prohibition, which is the only sentence in §01 set in bold:

> **Never auto-repair — report and stop.**

That is `DOMAIN_SCIENCE.md` §01.2's **hard-stop class** — high-confidence
detection, irreversible-if-wrong action, and no signal in the data that resolves
the ambiguity — and the app already models it: a finding with a proposed action
and **no pre-selection**. `clinical.mixed_units_finding` is the shape that was
copied, down to `hard_stop` in the payload, and no component was invented.

## The one place the data genuinely cannot decide, and what was done about it

**A five-digit integer is both an Excel serial and an Entrez gene ID.** `44621`
is 2022-03-01 and it is also a perfectly ordinary NCBI Gene identifier. Nothing
in the string separates them, which is the §01.2 litmus — *can the data
distinguish the causes of what I just detected?* — arriving one level earlier
than usual, on the **detection** rather than on the action.

The resolution is the company the integer keeps, and it is two conditions read
off the rest of the identifier set: no integer identifier falls outside the
serial window (one that does proves the table uses Entrez), and the set is at
least half HGNC symbols (which is what Excel destroys). Where either fails the
reading is withheld and the integers stay Entrez.
`test_the_serial_reading_is_withheld_when_the_table_uses_entrez_ids` is that
branch, and it is the assertion this file exists for — the other four are
comparatively easy.

## `GUIDED-097` — two fixtures of different shape

`genomics_gene_ids.csv` carries all four defects; `genomics_microarray.csv` is a
clean single vocabulary of Affymetrix probe sets and must stay silent on all
four. The second is the more useful arm, and the microarray file is a sharper
negative than `genomics_expression.csv` because its identifiers *are* a
vocabulary — a detector that fired on anything it recognized would pass against
`gene_0001` and fail here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"

#: The four diagnostics, and the detector that emits each.
DIAGNOSTICS = {
    "Excel corruption": P._gene_id_excel_corruption,
    "version suffixes": P._gene_id_versions,
    "duplicates after collapsing versions": P._gene_id_duplicates,
    "mixed vocabularies": P._gene_id_mixed_vocabulary,
}

#: `GUIDED-097`. The dirty table and the clean one, both real files.
GENE_ID_FIXTURES = {
    "all four defects": ("genomics_gene_ids", "condition"),
    "one clean probe vocabulary": ("genomics_microarray", "condition"),
}

#: The tables §01's identifier diagnostics describe nothing about. Silence here
#: is guard #2: a pack firing on the wrong data asserts something false
#: authoritatively.
NOT_A_GENE_ID_SET = ("metabolomics_untargeted", "survey_instrument",
                     "clinical_labs", "nhanes_dietary", "genomics_expression")

#: NOT COVERED, said out loud, because a sweep that reports only what it covered
#: has not reported its coverage.
#:
#: A FILE CORRUPTED ONLY INTO SERIALS. With no date strings left and no symbol
#: majority, nothing in the numbers says the column names were ever gene
#: symbols. The reading is withheld and the corruption is invisible — see the
#: Entrez test below for the deliberate half of that trade.
#:
#: MANY-TO-ONE THAT APPEARS ONLY AFTER MAPPING. §01 says *"duplicate IDs after
#: symbol mapping"*, and this app has no mapping table: two distinct Ensembl
#: accessions that resolve to one HGNC symbol are invisible here. The finding
#: says which half it reports rather than implying the whole class.
#:
#: GENES IN ROWS. §01 is written for the field convention; this app's tables are
#: samples in rows, so the identifiers arrive as a header. A file with an
#: identifier COLUMN is handled by the orientation reading that runs first.
#:
#: REFSEQ. `NM_000546.6` is versioned exactly like an Ensembl accession and the
#: grammar covers it, but no fixture in this repository uses RefSeq, so the
#: RefSeq branch of `_gene_id_versions` is exercised by a constructed frame
#: below rather than by a file.
SHAPES_NOT_COVERED = [
    "a file corrupted only into serials, with no date strings and no symbol "
    "majority left to license the reading",
    "many-to-one that appears only after mapping accessions to symbols, which "
    "needs a mapping table the app does not have",
    "genes in rows — the identifiers here are column names",
    "RefSeq accessions, covered by a constructed frame rather than a fixture",
    "an identifier repeated VERBATIM in the source file — pandas renames the "
    "second to `TP53.1` before the app sees it, the dot puts it outside every "
    "grammar, and `_gene_id_duplicates` therefore cannot see the one duplicate "
    "that is most obviously a duplicate",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ═══════════ 1 · ALL FOUR FIRE, WITH THE NUMBERS ═══════════

def test_all_four_diagnostics_fire_on_the_sibling_and_agree_with_its_companion():
    """The counts `genomics_gene_ids.csv.md` states, asserted against the code.

    A companion is load-bearing — it is what turns a drive into a comparison
    against a stated expectation rather than a guess — and a load-bearing
    document nothing checks is `FEATURE_PARITY.md`'s expiring guarantee.
    """
    df = load("genomics_gene_ids")
    assert df.shape == (60, 186)

    corruption = P._gene_id_excel_corruption(df)
    assert corruption["params"]["n_date_strings"] == 12
    assert corruption["params"]["n_serial_numbers"] == 2
    assert set(corruption["params"]["serial_numbers"]) == {"44805", "44627"}
    # DERIVED, NOT TYPED. `44627` is `7-Mar-2022` counted from Excel's
    # 1899-12-30 epoch, which is what `MARCH7` becomes. The fixture computes it;
    # this recomputes it, so the two would have to drift together to stay green.
    import datetime as dt
    assert (dt.date(2022, 3, 7) - dt.date(1899, 12, 30)).days == 44627
    assert (dt.date(2022, 9, 1) - dt.date(1899, 12, 30)).days == 44805

    versions = P._gene_id_versions(df)
    assert versions["params"]["n_versioned"] == 70
    assert versions["params"]["vocabularies"] == ["Ensembl accessions"]

    duplicates = P._gene_id_duplicates(df)
    assert duplicates["params"]["n_duplicate_bases"] == 3
    assert duplicates["params"]["n_duplicate_columns"] == 6

    mixed = P._gene_id_mixed_vocabulary(df)
    assert mixed["params"]["vocabularies"] == {"HGNC symbols": 97,
                                               "Ensembl accessions": 70}
    # The corrupted 14 are NOT counted as symbols. A wreck is not a member of
    # the vocabulary it used to belong to, and counting it as one would inflate
    # the very share that licenses the serial reading.
    assert mixed["params"]["n_classified"] == 167


@pytest.mark.parametrize("fixture", NOT_A_GENE_ID_SET)
def test_no_gene_id_diagnostic_fires_on_a_table_it_does_not_describe(fixture):
    """`genomics_expression.csv` is in this list on purpose and it is the
    interesting one: it IS a gene matrix, and its identifiers belong to no
    vocabulary at all. A classifier that guessed rather than declined would have
    something to say about `gene_0001`, and it would be inventing it."""
    df = load(fixture)
    fired = [name for name, detector in DIAGNOSTICS.items()
             if detector(df) is not None]
    assert fired == [], f"{fixture}: {fired}"


def test_a_single_clean_vocabulary_is_not_a_mixed_one():
    """`GUIDED-097`'s second arm, and the sharper negative.

    `genomics_microarray.csv` names its 496 columns `1000000_at` — a real
    identifier vocabulary, recognized, with no versions, no collisions and no
    corruption. It is the fixture that separates *"the classifier declines to
    guess"* from *"the classifier fires on anything it recognizes."*
    """
    df = load("genomics_microarray")
    reading = P.read_gene_ids(df)
    assert reading is not None, (
        "the probe-set vocabulary is recognized; the point is that recognizing "
        "it produces no finding")
    assert set(reading.vocabularies) == {"Affymetrix probe sets"}
    # 495 probe sets. The 496th numeric column is `age`, which belongs to no
    # grammar and is counted as unclassified rather than quietly folded in.
    assert reading.recognized == 495
    assert reading.unclassified == ("age",)
    for name, detector in DIAGNOSTICS.items():
        assert detector(df) is None, name


# ═══════════ 2 · THE HARD STOP ═══════════

def test_the_corruption_finding_proposes_nothing_and_preselects_nothing():
    """§01's one bolded sentence — *"Never auto-repair — report and stop"* —
    as a property of the payload rather than of anybody's restraint.

    `router._is_repairable` reads `fix_kind`, so `"none"` is what makes this
    structural: a reporting finding cannot become a question the interview asks,
    whatever it reports. Copied from `clinical.mixed_units_finding` down to the
    `hard_stop` key, which is `GUIDED-064`'s rule — *the machine-readable form
    must not be lossier than the sentence* — and *never auto-repair* is the
    whole content of this finding.
    """
    finding = P._gene_id_excel_corruption(load("genomics_gene_ids"))
    assert finding["fix_kind"] == "none"
    assert finding["fix_label"] == ""
    assert finding["params"]["hard_stop"] == "never_auto_repair_gene_symbols"
    assert finding["params"]["hard_stop_because"]
    assert finding["severity"] == "critical"
    # `offered`, never `derived`. The CLAIM is settled and the finding carries
    # it at `critical`; the ACTION is forbidden, and `derived` is the marker
    # that licenses pre-selecting one.
    assert finding["marker"] == "offered"
    assert "will not" in finding["detail"]


@pytest.mark.parametrize("shape", sorted(GENE_ID_FIXTURES))
def test_no_gene_id_detector_offers_a_repair(shape):
    """All four, on both fixtures. §01 gives no repair for any of them, and
    stripping a version suffix is the one that looks safest and is not:
    two versions of one accession collapse onto each other, which is the finding
    beside it."""
    name, _target = GENE_ID_FIXTURES[shape]
    df = load(name)
    for label, detector in DIAGNOSTICS.items():
        found = detector(df)
        if found is None:
            continue
        assert found["fix_kind"] == "none", f"{label} offers {found['fix_kind']}"
        assert found["fix_label"] == ""


# ═══════════ 3 · THE AMBIGUITY THE DATA DOES NOT RESOLVE ═══════════

def _symbol_frame(extra: dict | None = None) -> pd.DataFrame:
    """Forty HGNC symbols, two Excel serials, and whatever is added."""
    rng = np.random.default_rng(7)
    names = ["TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "NRAS", "HRAS", "MYC",
             "PTEN", "RB1", "VHL", "APC", "ATM", "ATR", "BRAF", "AKT1",
             "MTOR", "TSC1", "TSC2", "SMAD4", "NOTCH1", "JAG1", "WNT1",
             "GATA1", "GATA3", "RUNX1", "CEBPA", "IKZF1", "PAX5", "IRF4",
             "BCL2", "BCL6", "BAX", "CASP3", "CASP8", "FAS", "TNF", "IL6",
             "STAT1", "STAT3", "JAK1", "JAK2", "ACTB", "GAPDH", "B2M"]
    frame = {n: rng.integers(0, 500, size=60) for n in names}
    frame["44621"] = rng.integers(0, 500, size=60)      # 2022-03-01
    frame["44622"] = rng.integers(0, 500, size=60)      # 2022-03-02
    frame.update(extra or {})
    return pd.DataFrame(frame)


def test_a_serial_among_gene_symbols_is_read_as_a_corrupted_date():
    """The positive control. Without it the test below passes by doing
    nothing, which is the *check nothing triggers* failure."""
    finding = P._gene_id_excel_corruption(_symbol_frame())
    assert finding is not None
    assert set(finding["params"]["serial_numbers"]) == {"44621", "44622"}


def test_the_serial_reading_is_withheld_when_the_table_uses_entrez_ids():
    """**The assertion this file exists for.**

    One column is added: `7157`, the real Entrez ID for TP53, which is outside
    the Excel serial window. That single column proves the table's integers are
    gene identifiers, and the reading of `44621` as a date must be withdrawn —
    not softened, not hedged, withdrawn. `DOMAIN_SCIENCE.md` §01.2's litmus
    applied to detection: the data cannot distinguish the causes, so the app
    declines.

    The cost is stated rather than hidden: a table that mixes Entrez IDs with
    symbols and IS corrupted gets no corruption finding from its serials. That
    is in `SHAPES_NOT_COVERED`, and it is the right side of the trade — a
    missing finding is recoverable and a wrong one asserted authoritatively is
    the failure this pack's guard #2 exists to prevent.
    """
    rng = np.random.default_rng(11)
    with_entrez = _symbol_frame({"7157": rng.integers(0, 500, size=60)})
    assert P._gene_id_excel_corruption(with_entrez) is None

    # AND THE INTEGERS ARE STILL CLASSIFIED, rather than dropped on the floor.
    reading = P.read_gene_ids(with_entrez)
    assert reading.excel_serials == ()
    assert set(reading.vocabularies["Entrez gene IDs"]) == {
        "44621", "44622", "7157"}


def test_a_date_string_needs_no_such_licence():
    """`1-Mar` is not ambiguous the way `44621` is — nothing else in a gene
    matrix is spelled that way — so the date-string reading survives the Entrez
    column that killed the serial reading. Two readings of different strength,
    and collapsing them would have made the stronger one hostage to the
    weaker."""
    rng = np.random.default_rng(13)
    frame = _symbol_frame({"7157": rng.integers(0, 500, size=60),
                           "1-Mar": rng.integers(0, 500, size=60),
                           "2-Sep": rng.integers(0, 500, size=60)})
    finding = P._gene_id_excel_corruption(frame)
    assert finding is not None
    assert set(finding["params"]["date_strings"]) == {"1-Mar", "2-Sep"}
    assert finding["params"]["serial_numbers"] == []


def test_a_refseq_accession_carries_a_version_the_same_way():
    """The RefSeq branch, which no fixture reaches. Named in
    `SHAPES_NOT_COVERED` and constructed here rather than left as an untested
    line in a grammar table."""
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        f"NM_{500000 + i:06d}.{(i % 5) + 1}": rng.integers(0, 500, size=60)
        for i in range(40)})
    finding = P._gene_id_versions(frame)
    assert finding is not None
    assert finding["params"]["n_versioned"] == 40
    assert finding["params"]["vocabularies"] == ["RefSeq accessions"]


def test_a_repeated_column_name_is_not_read_as_a_version():
    """`pandas` renames a repeated header to `TP53.1`, and a `.1` is exactly
    what a version suffix looks like.

    **The mechanism, stated after a revert probe corrected the first version of
    this docstring.** It is not that the version reading is scoped to Ensembl
    and RefSeq — widening `_VERSIONED_VOCABULARIES` to include symbols leaves
    this test green, and the probe reported `GREEN — NOT LOAD-BEARING`, which is
    the harness doing exactly its job. It is that the version reading runs over
    the CLASSIFIED buckets and `TP53.1` is in none of them: the HGNC grammar
    admits no dot, so a mangled header is unclassified and never reaches the
    suffix test. The two guards together are what hold, and neither alone does.

    **And the honest consequence, which is a gap rather than a win.** A column
    name literally repeated in the source file IS a duplicate gene identifier —
    the many-to-one §01 is about — and pandas has renamed it before the app sees
    it. `_gene_id_duplicates` cannot see it either. Named in
    `SHAPES_NOT_COVERED`, because a guard whose side effect is a blind spot
    should say so where somebody reading the guard will find it.
    """
    rng = np.random.default_rng(19)
    frame = _symbol_frame()
    frame["TP53.1"] = rng.integers(0, 500, size=60)
    reading = P.read_gene_ids(frame)
    assert reading.versioned == ()
    assert P._gene_id_versions(frame) is None
    assert "TP53.1" in reading.unclassified
    assert not any("TP53.1" in members
                   for members in reading.vocabularies.values())
    # The blind spot, asserted so it is a recorded limit rather than a surprise.
    assert reading.duplicate_bases == {}


# ═══════════ 4 · AND IT REACHES A PERSON ═══════════

def test_the_four_diagnostics_reach_a_person_and_carry_their_badges():
    """**Trap #1, checked the only way it can be.** A capability is gratifying
    to build and fully verifiable in isolation; the four assertions above prove
    the detectors and prove nothing about the app.

    Driven through the real API and then through the page's real controller in
    node: a file, a lens answer, a target, the findings the project serves, and
    the titles read back off the rendered DOM. `GUIDED-142` is the reason the
    page half is here rather than only the API half — five packs and eighteen
    detectors were correct on the wire and rendered nowhere for two loops.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client = TestClient(api.app)
    with open(DATA / "genomics_gene_ids.csv", "rb") as handle:
        pid = client.post("/project", files={
            "file": ("genomics_gene_ids.csv", handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [P.GENOMICS]}),
                          ("set_target", {"column": "condition"})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, (kind, ok.text[:300])

    project = client.get(f"/project/{pid}").json()
    served = [f for f in project["findings"] if f["source"] == "pack"]
    reached = {f["id"] for f in served}
    # THE FOUR ARE PRESENT — not "the pack serves exactly these five".
    #
    # This was an equality against a closed set, and it went red the hour the
    # gene-ID work was merged with the data-type work built in a sibling
    # worktree: `pack::genomics::data_type` is a legitimate fifth reading and
    # the assertion called it a failure. A closed-set assertion on a pack that
    # is being filled out asserts that no further detector may ever exist,
    # which is the opposite of what this file is checking — and it would have
    # broken every future genomics detector, one loop at a time.
    #
    # The subset assertion says what the test means. The count is asserted as a
    # FLOOR beside it so an emptied pack still fails.
    assert reached >= {
        "pack::genomics::gene_id_excel_corruption",
        "pack::genomics::gene_id_versions",
        "pack::genomics::gene_id_duplicates",
        "pack::genomics::gene_id_mixed_vocabulary"}, sorted(reached)
    assert len(reached) >= 4, sorted(reached)

    for finding in served:
        badge = finding["evidence"]
        assert badge["source"].startswith("research/GENOMICS_PACK.md#")
        assert badge["evidence_status"] in ("SETTLED", "CONVENTION", "DISPUTED")

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
        "        (__harness.html('profRest') || '')).slice(0, 200000)});",
        routes=routes, search=f"?project={pid}")
    html = out["open"]
    assert out["shut"], "the Explore findings list rendered nothing at all"

    missing = [f["id"] for f in served if f["title"][:28] not in html]
    assert not missing, (
        f"the genomics pack computes {missing} and the page never shows them, "
        f"pushed or collapsed")

    # AND THE CORRUPTED IDENTIFIERS THEMSELVES REACH THE PAGE. The count is the
    # finding's headline and the names are what a person acts on — a card that
    # said "14 identifiers" and showed none of them would leave the user with
    # a number and no file to open.
    assert "8-Mar" in html or "3-Mar" in html, (
        "the corrupted identifiers are in `affected_columns` and none of them "
        "is on the page")
