"""The intake's version 2 (SWARM_PLAN §5, §10 package 5): declared measurement markers, the glucose
table rebuilt from the NCHS files, and eligibility by code.

* **Declared markers.** Each passage is found on its page by a second reading of the HTML, made
  here independently of the intake's (tags struck by a regular expression, entities read by
  ``html.unescape``); a passage changed by a word, or credited to another page, is refused, and so
  is a page the dataset did not download; ``--check`` finds a passage tampered with after the
  write. Each marker is the record's (``record.DeclaredMarker``), through the assembler's
  ``dataset_of``. What a value says a page does not name, the page's text does not hold.
* **The glucose table** against pandas' own SAS reader: one row per participant of each cycle's
  demographics file, the cycle column from the file each row came from, every kept variable equal
  to its file's, a file a cycle lacks blank there; today's export's 21 predictors and its outcome
  are the variables :data:`intake.GLUCOSE_TODAY` names, on the cycles both hold; no glycemic
  measure of the same draw is kept; each cycle's glucose file carries its weight.
* **Eligibility by code.** The card's codes are the codebooks' (read here from the HTML tables by
  a regular expression), its blank counts pandas'; a level rule written from the card's codes
  excludes, in the engine, exactly the rows its codes name; the helpers refuse codebooks that
  disagree, a level a codebook lacks, two refused codes.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import intake as I  # noqa: E402

PILOT = list(I.DECLARING)
SAS_ZERO = 1e-70  # pandas' SAS reader reads a SAS zero as 5.4e-79; TurboTab's reads it as 0


def text_of(path: Path) -> str:
    """A page's text read a second way: scripts and styles struck, every tag a space, entities
    unescaped, whitespace collapsed."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(html.unescape(raw).split())


def pages_on_disk(id_: str) -> dict[str, Path]:
    """Each codebook page of the dataset's manifest that is on disk, by its URL."""
    manifest = I.load(id_)["manifest"]
    return {e["url"]: I.REPO / e["file"] for e in manifest["files"]
            if e["file"].endswith(".htm") and (I.REPO / e["file"]).is_file()}


def need_pages(id_: str) -> dict[str, Path]:
    pages = pages_on_disk(id_)
    if not pages:
        pytest.skip(f"{id_}: its pages are not on disk (research/data/ is not tracked)")
    return pages


def sas(path: Path) -> pd.DataFrame:
    """A transport file by pandas' own reader, its SAS zero read as 0."""
    frame = pd.read_sas(path, format="xport")
    numbers = frame.select_dtypes("number")
    frame[numbers.columns] = numbers.mask(numbers.abs() < SAS_ZERO, 0.0)
    return frame


# ── the passages and their pages ─────────────────────────────────────────────


def test_page_text_reads_a_pages_text_as_a_reader_sees_it():
    page = ("<html><head><style>p {color: red}</style><script>var x = '<b>no</b>';</script>"
            "</head><body><p>Glucose&nbsp;(mg/dL)</p><p>A &amp; B&#8211;C\n   D</p><!-- c -->"
            "</body></html>")
    assert I.page_text(page) == "Glucose (mg/dL) A & B–C D"
    assert I.page_text(page.encode()) == I.page_text(page)
    assert I.quoted("A & B–C\n D", I.page_text(page))
    assert not I.quoted("var x", I.page_text(page)) and not I.quoted("color", I.page_text(page))


PAGE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.htm"
OTHER = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_GLU.htm"
PAGES = {PAGE: "Glucose was measured on the Cobas C311. It is in mg/dL.",
         OTHER: "No adjustments are needed."}


def test_a_declared_marker_holds_its_passages_each_on_its_page():
    m = I.declared("instrument", "the Cobas C311", [
        (PAGE, "Glucose was measured on the Cobas C311."), ([PAGE, OTHER], "e")], PAGES)
    assert m == {"name": "instrument", "status": "declared", "value": "the Cobas C311",
                 "quote": "Glucose was measured on the Cobas C311.\ne\ne",
                 "source": f"{PAGE}\n{PAGE}\n{OTHER}"}
    from research.swarm.record import DeclaredMarker
    assert DeclaredMarker.model_validate(m).quote.split(I.PASSAGES)[0].endswith("C311.")


@pytest.mark.parametrize("name, passages, message", [
    ("instrument", [(PAGE, "Glucose was measured on the Cobas C501.")], "is not on"),  # a word
    ("instrument", [(OTHER, "Glucose was measured on the Cobas C311.")], "is not on"),  # a page
    ("instrument", [(PAGE.replace("GLU_I", "GHB_I"), "Glucose")], "not a page this dataset"),
    ("instrument", [(PAGE, "Glucose was\nmeasured")], "is not on"),  # a break splits passages
    ("instrument", [], "holds its value and a passage"),
    ("units", [(PAGE, "mg/dL")], "not a declared marker"),  # a live marker's name
])
def test_a_passage_not_on_its_page_is_refused(name, passages, message):
    with pytest.raises(I.QuoteRefused, match=message):
        I.declared(name, "a value", passages, PAGES)


def test_a_marker_not_stated_names_the_pages_read_and_only_pages_downloaded():
    m = I.not_stated("recall_method", [PAGE, OTHER], PAGES)
    assert m == {"name": "recall_method", "status": "not_stated", "source": f"{PAGE}\n{OTHER}"}
    with pytest.raises(I.QuoteRefused, match="not a page this dataset"):
        I.not_stated("recall_method", [PAGE.replace("GLU", "GHB")], PAGES)


@pytest.mark.parametrize("id_", PILOT)
def test_every_committed_passage_is_on_its_page_read_a_second_way(id_):
    from research.swarm.vocabulary import DECLARED
    pages = need_pages(id_)
    markers = I.load(id_)["declared"]
    assert [m["name"] for m in markers] == list(DECLARED)
    for m in markers:
        assert m["status"] == "declared" and m["value"]
        for url, quote in zip(m["source"].split("\n"), m["quote"].split("\n"), strict=True):
            assert quote in text_of(pages[url]), (m["name"], url, quote)


@pytest.mark.parametrize("id_", PILOT)
def test_the_committed_markers_are_the_records_through_the_assembler(id_):
    from research.swarm import assemble as A
    from research.swarm.record import Dataset
    meta = {"dataset": I.dataset_meta(id_)}
    provenance = {"inputs": [{"role": "table", "name": "t", "sha256": "0" * 64}]}
    dataset = Dataset.model_validate(A.dataset_of(meta, provenance, {}, {}))
    assert [m.model_dump(exclude_none=True) for m in dataset.declared] == I.load(id_)["declared"]
    assert all(m.checked is None for m in dataset.declared)  # left to the referee


def test_the_check_finds_a_passage_changed_after_the_write_and_passes_the_committed(monkeypatch,
                                                                                    tmp_path):
    need_pages("nhanes_glucose_predict")
    for id_ in PILOT:
        assert I.marker_problems(I.load(id_)["declared"], I.load(id_)["manifest"]) == []
    shutil.copytree(I.DATASETS / "nhanes_glucose_predict", tmp_path / "nhanes_glucose_predict")
    path = tmp_path / "nhanes_glucose_predict" / "declared.json"
    markers = json.loads(path.read_text())
    markers[0]["quote"] = markers[0]["quote"].replace("C311", "C411", 1)
    path.write_text(json.dumps(markers))
    monkeypatch.setattr(I, "DATASETS", tmp_path)
    found = I.check("nhanes_glucose_predict")
    assert len(found) == 1 and "declared: instrument: the passage" in found[0]
    assert "C411" in found[0]
    path.unlink()
    assert I.check("nhanes_glucose_predict") == ["no declared.json"]


def test_markers_that_are_not_the_records_are_refused():
    manifest = {"dataset": "x", "files": []}
    twice = [{"name": "units_x", "status": "declared"}]
    assert I.marker_problems(twice, manifest)[0].startswith("the declared markers are not")
    bare = [{"name": "instrument", "status": "declared", "value": "v"}]  # no passage
    assert any("holds its value, the passage" in e for e in I.marker_problems(bare, manifest))
    one = {"name": "instrument", "status": "not_stated", "source": PAGE}
    assert any("listed once" in e for e in I.marker_problems([one, one], manifest))
    assert I.marker_problems([one], manifest) == [f"instrument: {PAGE} is not a page on disk"]


NOT_NAMED = [  # (dataset, page stem, what a marker's value says the page does not name)
    ("nhanes_glucose_predict", "GLU_G", ("Cobas", "C501", "C311", "Hitachi")),
    ("nhanes_glucose_predict", "BPX_G", ("Omron", "oscillometric", "sphygmomanometer")),
    ("nhanes_glucose_predict", "BPX_H", ("Omron", "oscillometric", "sphygmomanometer")),
    ("nhanes_glucose_predict", "BPX_I", ("Omron", "oscillometric", "sphygmomanometer")),
    ("sullivan2021", "GLU_G", ("Cobas", "C501", "C311", "Hitachi")),
    ("bang2009", "BPX_C", ("Omron", "oscillometric", "sphygmomanometer")),
    ("bang2009", "LAB10AM", ("Cobas", "Hitachi", "Mira", "Beckman", "Synchron", "analyzer")),
    ("bang2009", "L10AM_B", ("Cobas", "Hitachi", "Mira", "Beckman", "Synchron", "analyzer")),
    ("bang2009", "L10AM_C", ("Cobas", "Hitachi", "Mira", "Beckman", "Synchron", "analyzer")),
    ("bang2009", "MCQ", ("household interview", "home interview", "in the home",
                         "Mode of Administration")),
]


@pytest.mark.parametrize("id_, stem, words", NOT_NAMED, ids=[f"{d}:{s}" for d, s, _ in NOT_NAMED])
def test_what_a_value_says_a_page_does_not_name_the_page_does_not_hold(id_, stem, words):
    pages = need_pages(id_)
    page = next(p for url, p in pages.items() if url.endswith(f"/{stem}.htm"))
    text = text_of(page).lower()
    assert [w for w in words if w.lower() in text] == []


def test_the_pre_pandemic_page_names_an_analyzer_only_for_2015_to_2018():
    """The glucose instrument marker says no page names the analyzer of 2019–March 2020: no
    sentence of P_GLU's page that names an analyzer names 2019 or 2020, and the one that dates
    the C311 dates it 2017-2018 (the other, "Some of the samples were collected using the Cobas
    C501.", follows a sentence about 2015-2016's samples)."""
    pages = need_pages("nhanes_glucose_predict")
    text = text_of(pages[I.glucose_url("GLU", I.GLUCOSE_TEST)])
    naming = [s for s in re.split(r"(?<=\.) ", text) if re.search(r"C311|C501|Cobas", s)]
    assert len(naming) == 2 and not any(re.search(r"2019|2020", s) for s in naming), naming
    assert "same instrument used in 2017-2018" in naming[0] and "2015-2016" in naming[0]


# ── the glucose table, against pandas' own SAS reader ────────────────────────


def need_glucose() -> pd.DataFrame:
    files = [I.NHANES / c / f"{s}.xpt" for c, spec in I.glucose_spec()["cycles"].items()
             for s in spec["files"]]
    if not (I.GLUCOSE_TABLE.is_file() and all(p.is_file() for p in files)):
        pytest.skip("the glucose files are not on disk (research/data/ is not tracked)")
    return pd.read_parquet(I.GLUCOSE_TABLE)


def test_the_glucose_table_is_every_demo_participant_with_its_cycle():
    table = need_glucose()
    for cycle, (year, _) in I.GLUCOSE_CYCLES.items():
        demo = sas(I.NHANES / cycle / f"{I.glucose_stems('DEMO')[cycle]}.xpt")
        rows = table[table[I.GLUCOSE_CYCLE] == year]
        assert len(rows) == len(demo) and set(rows["SEQN"]) == set(demo["SEQN"]), cycle
    assert len(table) == table["SEQN"].nunique()
    assert sorted(table[I.GLUCOSE_CYCLE].unique()) == [2011, 2013, 2015, 2017]
    assert I.load(I.GLUCOSE_ID)["manifest"]["table"]["rows"] == len(table)


def test_every_kept_variable_is_its_files_by_seqn_and_blank_where_its_cycle_lacks_it():
    table = need_glucose().set_index("SEQN")
    for name, spec in I.GLUCOSE_FILES.items():
        stems = I.glucose_stems(name)
        for cycle, (year, _) in I.GLUCOSE_CYCLES.items():
            rows = table[table[I.GLUCOSE_CYCLE] == year]
            if cycle not in stems:
                assert rows[spec["vars"][1:]].isna().all().all(), (name, cycle)
                continue
            raw = sas(I.NHANES / cycle / f"{stems[cycle]}.xpt").set_index("SEQN")
            for v in spec["vars"][1:]:
                got = rows[v]
                if v not in raw.columns:
                    assert got.isna().all(), (name, cycle, v)
                    continue
                want = raw[v].reindex(rows.index)
                np.testing.assert_allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float),
                                           rtol=0, atol=1e-9, equal_nan=True,
                                           err_msg=f"{cycle} {v}")
    measured = table.groupby(I.GLUCOSE_CYCLE)[I.GLUCOSE_OUTCOME].count()
    for cycle, (year, _) in I.GLUCOSE_CYCLES.items():
        glu = sas(I.NHANES / cycle / f"{I.glucose_stems('GLU')[cycle]}.xpt")
        assert measured[year] == glu["LBXGLU"].notna().sum() > 0


def test_todays_export_is_the_variables_glucose_today_names_on_the_cycles_both_hold():
    """turbotab/core/tests/fixtures/nhanes.csv.gz holds 2011, 2013 and 2015 too: each of its
    columns equals the variable named for it, participant by participant (an imputed value
    aside), and gender, meds_hbp and meds_chol are the codes' words."""
    table = need_glucose().set_index("SEQN")
    export = pd.read_csv(REPO / "turbotab" / "core" / "tests" / "fixtures" / "nhanes.csv.gz")
    export = export[export["cycle_begin_year"].isin([2011, 2013, 2015])].set_index("SEQN")
    ours = table.loc[export.index]
    assert len(export) > 6000
    assert set(I.GLUCOSE_TODAY) == {c for c in export.columns if not c.startswith("imputed_")}
    assert len(I.GLUCOSE_TODAY) == 22  # the outcome and the 21 predictors
    for column, variables in I.GLUCOSE_TODAY.items():
        if column in ("gender", "meds_hbp", "meds_chol"):
            continue
        imputed = export.get(f"imputed_{column}")
        keep = ~imputed.astype(str).str.lower().eq("true") if imputed is not None else slice(None)
        np.testing.assert_allclose(ours.loc[keep, variables[0]].to_numpy(dtype=float),
                                   export.loc[keep, column].to_numpy(dtype=float), rtol=0,
                                   atol=1e-6, equal_nan=True, err_msg=column)
    assert (ours["RIAGENDR"].map({1.0: "male", 2.0: "female"}) == export["gender"]).all()
    for column, variable in (("meds_hbp", "BPQ050A"), ("meds_chol", "BPQ100D")):
        words = export[column].map({True: 1.0, False: 2.0, "TRUE": 1.0, "FALSE": 2.0})
        said = words.notna()
        assert (ours.loc[said, variable] == words[said]).all() and said.sum() > 1000, column


def test_no_glycemic_measure_of_the_same_draw_is_kept_though_the_files_carry_some():
    table = need_glucose()
    glycemic = {"LBXIN", "LBDINSI", "LBXGH", "LBXGLT", "LBDGLTSI", "LBDGLUSI", "LBXSGL"}
    assert glycemic.isdisjoint(table.columns)
    carried = sas(I.NHANES / "2011-2012" / "GLU_G.xpt").columns
    assert "LBXIN" in carried  # GLU_G carries insulin; the table does not keep it


def test_each_cycles_glucose_file_carries_its_weight_and_not_the_others():
    need_glucose()
    weights = set(I.GLUCOSE_WEIGHTS.values())
    for cycle, weight in I.GLUCOSE_WEIGHTS.items():
        columns = set(sas(I.NHANES / cycle / f"{I.glucose_stems('GLU')[cycle]}.xpt").columns)
        assert weight in columns and not (weights - {weight}) & columns, cycle


def test_the_glucose_card_fixes_the_test_cycle_by_its_cut_point():
    table = need_glucose()
    card = I.load(I.GLUCOSE_ID)["card"]
    held = card["held_out"]
    assert held["column"] == I.GLUCOSE_CYCLE and held["at"] == 2017
    n = int(((table[I.GLUCOSE_CYCLE] >= 2017) & table["LBXGLU"].notna()).sum())
    assert f"({n:,} of them with a fasting glucose)" in held["meaning"]
    assert "2017–March 2020" in held["meaning"] and "research/swarm/benchmark.py" in held["meaning"]
    assert card["outcome"]["column"] == "LBXGLU" and card["purpose"] == "Predict"
    assert ("without diagnosed diabetes and not taking a glucose-lowering medication"
            in card["population"])
    assert "analyst's eligibility decisions" in card["population"]


def test_the_glucose_cards_documentation_quotes_only_what_its_pages_hold():
    need_glucose()
    manifest = I.load(I.GLUCOSE_ID)["manifest"]
    pages = I.manifest_pages(manifest)
    assert I.glucose_documentation(pages) == I.load(I.GLUCOSE_ID)["card"]["documentation"]
    url = I.glucose_url("BPXO", I.GLUCOSE_TEST)
    with pytest.raises(I.QuoteRefused, match="Auscultatory BP measurement"):
        I.glucose_documentation({**pages, url: pages[url].replace("Auscultatory", "Manual")})


def test_the_card_validator_reads_held_out_and_eligibility_both_ways():
    card = dict(I.load(I.GLUCOSE_ID)["card"])
    assert I.validate_card(card, I.GLUCOSE_ID) == []
    for bad in ({"column": "c", "at": "2017", "meaning": "m"}, {"column": "c", "at": True,
                                                                 "meaning": "m"},
                {"column": "c", "at": 2017}, ["c", 2017]):
        assert "held_out is not {column, at: a number, meaning}" in I.validate_card(
            {**card, "held_out": bad}, I.GLUCOSE_ID)
    assert "eligibility is not a list of sentences" in I.validate_card(
        {**card, "eligibility": ["a", ""]}, I.GLUCOSE_ID)


def test_the_glucose_population_reads_each_cycles_top_code_from_the_table():
    rows = pd.DataFrame({I.GLUCOSE_CYCLE: [y for y, _ in I.GLUCOSE_CYCLES.values()],
                         "RIDAGEYR": [80.0] * 4})
    text = I.glucose_population(rows)
    assert "the 4 survey cycles 2011–2012 to 2017–March 2020" in text
    with pytest.raises(SystemExit, match="not the top code 80"):
        I.glucose_population(rows.assign(RIDAGEYR=[80.0, 80.0, 79.0, 80.0]))


def frames_of(**by: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return by


def test_the_glucose_table_on_constructed_files_names_each_gap(monkeypatch):
    """Two cycles: the second has no BPX file and its GLU lacks W1; the cycle column is the file's
    cycle, whatever SDDSRVYR says, and a record with no DEMO participant is refused."""
    files = {"DEMO": {"about": "d", "vars": ["SEQN", "SDDSRVYR"]},
             "GLU": {"about": "g", "vars": ["SEQN", "LBXGLU", "W1"]},
             "BPX": {"about": "b", "cycles": ["2011-2012"], "vars": ["SEQN", "BPXSY1"]}}
    cycles = {"2011-2012": (2011, "{}_G"), "2017-2020": (2017, "P_{}")}
    frames = {"2011-2012/DEMO_G": pd.DataFrame({"SEQN": [1.0, 2.0], "SDDSRVYR": [9.0, 9.0]}),
              "2017-2020/P_DEMO": pd.DataFrame({"SEQN": [3.0], "SDDSRVYR": [9.0]}),
              "2011-2012/GLU_G": pd.DataFrame({"SEQN": [2.0], "LBXGLU": [130.0], "W1": [5.0]}),
              "2017-2020/P_GLU": pd.DataFrame({"SEQN": [3.0], "LBXGLU": [99.0]}),
              "2011-2012/BPX_G": pd.DataFrame({"SEQN": [1.0], "BPXSY1": [120.0]})}
    monkeypatch.setattr(I, "GLUCOSE_FILES", files)
    monkeypatch.setattr(I, "GLUCOSE_CYCLES", cycles)
    monkeypatch.setattr(I, "read_xpt", lambda path: frames[f"{path.parent.name}/{path.stem}"])
    table, absent = I.glucose_table()
    assert table["SEQN"].tolist() == [1.0, 2.0, 3.0]
    assert table[I.GLUCOSE_CYCLE].tolist() == [2011, 2011, 2017]
    np.testing.assert_array_equal(table["LBXGLU"], [np.nan, 130.0, 99.0])
    np.testing.assert_array_equal(table["BPXSY1"], [120.0, np.nan, np.nan])
    assert absent == {"GLU": {"W1": ["2017-2020"]}, "BPX": {"BPXSY1": ["2017-2020"]}}
    gaps = next(s for s in I.glucose_prepared(absent) if s.startswith("Blank where"))
    assert gaps == ("Blank where a cycle does not carry the variable: W1 in 2017–March 2020 "
                    "(P_GLU does not carry it); every BPX variable in 2017–March 2020, which has "
                    "no BPX file.")
    frames["2017-2020/P_GLU"] = pd.DataFrame({"SEQN": [3.0, 4.0], "LBXGLU": [99.0, 1.0]})
    with pytest.raises(SystemExit, match="GLU: 1 records whose SEQN the base table lacks"):
        I.glucose_table()


def test_the_glucose_files_are_the_cdcs_four_cycles_with_the_pre_pandemic_p_files():
    spec = I.glucose_spec()
    assert list(spec["cycles"]) == ["2011-2012", "2013-2014", "2015-2016", "2017-2020"]
    assert spec["cycles"]["2017-2020"]["year"] == 2017
    assert all(f.startswith("P_") for f in spec["cycles"]["2017-2020"]["files"])
    assert "P_BPXO" in spec["cycles"]["2017-2020"]["files"]
    assert "BPX_I" in spec["cycles"]["2015-2016"]["files"]
    assert spec["url"].startswith("https://wwwn.cdc.gov/") and "?" not in spec["url"]
    manifest = I.load(I.GLUCOSE_ID)["manifest"]
    assert manifest["license"] == "public domain"
    for e in manifest["files"]:
        assert e["url"].startswith("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"), e
        assert len(e["sha256"]) == 64 and e["bytes"] > 0 and e["downloaded"]
    named = {v["variable"] for v in manifest["variables"]}
    assert named == {v for spec_ in I.GLUCOSE_FILES.values() for v in spec_["vars"][1:]}
    assert manifest["cycle_column"] == {"column": I.GLUCOSE_CYCLE, "values": {
        c: y for c, (y, _) in I.GLUCOSE_CYCLES.items()}}


def test_a_file_on_disk_keeps_the_download_date_another_manifest_recorded_for_its_bytes():
    """The glucose and Sullivan datasets share 2011–2016's DEMO, GLU, BMX…: the glucose manifest's
    entry of a shared file is Sullivan's (same bytes, same date), and a record of other bytes is
    not taken."""
    sullivan = {e["file"]: e for e in I.load("sullivan2021")["manifest"]["files"]}
    known = {e["file"]: e for e in I.on_record(None)["files"]}
    shared = I.rel(I.NHANES / "2011-2012" / "GLU_G.xpt")
    assert known[shared] == sullivan[shared]
    stale = {"files": [{**sullivan[shared], "downloaded": None, "sha256": "0" * 64}]}
    assert {e["file"]: e for e in I.on_record(stale)["files"]}[shared]["sha256"] == "0" * 64


# ── eligibility by code ──────────────────────────────────────────────────────


def html_codes(path: Path, variable: str) -> dict[str, str]:
    """A variable's codes read from its codebook table by a regular expression: the rows of the
    values table after its ``id="<variable>"`` heading, before the next heading."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    start = re.search(rf'id="{variable}"', raw, flags=re.I)
    section = raw[start.end():].split('class="vartitle"')[0]
    out = {}
    for row in re.findall(r"(?s)<tr>(.*?)</tr>", section):
        cells = [" ".join(html.unescape(c).split())
                 for c in re.findall(r"(?s)<td[^>]*>(.*?)</td>", row)]
        if len(cells) >= 2 and cells[0] not in (".", "") and cells[1] != "Range of Values":
            out[cells[0]] = cells[1]
    return out


CODED = [("bang2009", "DIQ", "DIQ010"), ("bang2009", "DIQ", "DIQ050"),
         ("bang2009", "DIQ", "DIQ070"), ("bang2009", "DEMO", "RIDEXPRG"),
         (I.GLUCOSE_ID, "DIQ", "DIQ010"), (I.GLUCOSE_ID, "DIQ", "DIQ050"),
         (I.GLUCOSE_ID, "DIQ", "DIQ070"), (I.GLUCOSE_ID, "DEMO", "RIDEXPRG")]


@pytest.mark.parametrize("id_, file, variable", CODED, ids=[f"{d}:{v}" for d, _, v in CODED])
def test_the_cards_level_rules_name_the_codebooks_codes_and_pandas_blank_counts(id_, file,
                                                                              variable):
    """Each cycle's codebook, read here by a regular expression: the excluded level 1, the
    refused and don't-know codes and the other levels the card names; and the blanks among the
    rows with a fasting glucose, counted by pandas on the table."""
    need_pages(id_)
    stems = I.bang_stems(file) if id_ == "bang2009" else I.glucose_stems(file)
    codes = [html_codes(I.NHANES / c / f"{s}.htm", variable) for c, s in stems.items()]
    assert all(c == codes[0] for c in codes) and codes[0]["1"].startswith("Yes")
    card = I.load(id_)["card"]
    line = next(s for s in card["eligibility"] if f"a level rule on {variable} " in s)
    assert f"excluding level 1 ({codes[0]['1']})" in line
    for code, said in codes[0].items():
        if said == "Refused":
            assert f"naming {code} as its refused code" in line
        elif said.lower() == "don't know":
            assert f"{code} as its don't-know code" in line
        elif code != "1":
            assert f"{code} ({said})" in line
    table = pd.read_parquet(I.BANG_TABLE if id_ == "bang2009" else I.GLUCOSE_TABLE)
    measured = table["LBXGLU"].notna()
    blank = int(table.loc[measured, variable].isna().sum())
    among = int(measured.sum())
    assert (f"{variable} is blank in {blank:,} of the {among:,} rows" in line if blank else
            f"recorded in every one of the {among:,} rows" in line)


def test_a_level_rule_from_the_cards_codes_excludes_in_the_engine_what_its_codes_name():
    """DIQ010's rule as the card writes it (exclude 1, refused 7, don't know 9) on the Bang table:
    without missing kept, the engine leaves out 1, 7, 9 and the blanks; with it, only 1."""
    from turbotab.core.decisions import LevelRule
    from turbotab.core.stages.rows import rule_keep
    if not I.BANG_TABLE.is_file():
        pytest.skip("the Bang table is not on disk")
    table = pd.read_parquet(I.BANG_TABLE)
    rule = dict(column="DIQ010", exclude=["1"], refused=["7"], dont_know=["9"], reason="known")
    strict = rule_keep(table, LevelRule(**rule)).to_numpy()
    kept = rule_keep(table, LevelRule(**rule, missing="keep")).to_numpy()
    d = table["DIQ010"]
    np.testing.assert_array_equal(strict, d.isin([2.0, 3.0]).to_numpy())
    np.testing.assert_array_equal(kept, (d != 1.0).to_numpy())
    assert (d == 7).sum() + (d == 9).sum() > 0 and d.isna().sum() > 0  # the two differ here


def test_the_eligibility_helpers_refuse_codebooks_that_disagree_and_a_missing_level():
    from turbotab.core.codebook import Entry
    yes = Entry("V", codes={"1": "Yes", "2": "No", "7": "Refused", "9": "Don't know"})
    other = Entry("V", codes={"1": "Yes", "2": "No", "9": "Don't know"})
    assert I.answer_codes({"a": {"V": yes}, "b": {"V": yes}}, "V") == yes.codes
    with pytest.raises(SystemExit, match="the codebooks of b give other codes than a's"):
        I.answer_codes({"a": {"V": yes}, "b": {"V": other}}, "V")
    with pytest.raises(SystemExit, match="the b codebook documents no codes"):
        I.answer_codes({"a": {"V": yes}, "b": {}}, "V")
    assert I.unconfirmed(yes.codes) == ("7", "9")
    assert I.unconfirmed({"1": "Yes", "9": "Don’t Know"}) == (None, "9")
    with pytest.raises(SystemExit, match="two codes are refused"):
        I.unconfirmed({"7": "Refused", "77": "Refused"})
    with pytest.raises(SystemExit, match="its codebook has no level 3"):
        I.level_line("w", "V", "3", yes.codes, "b.")
    line = I.level_line("Not X", "V", "1", yes.codes, I.blanks_line("V", 0, 10, "here"))
    assert line.startswith("Not X: a level rule on V excluding level 1 (Yes), naming 7 as its "
                           "refused code and 9 as its don't-know code; its other levels are 2 "
                           "(No). V is recorded in every one of the 10 rows here.")
    assert I.blanks_line("V", 4, 10, "here", (4, "C")) == ("V is blank in 4 of the 10 rows "
                                                          "here, all of them in C.")
    assert I.blanks_line("V", 4, 10, "here", (0, "C")).endswith(", none of them in C.")


def test_the_bang_card_states_its_eligibility_and_the_tables_columns_stay():
    card = I.load("bang2009")["card"]
    text = " ".join(card["eligibility"])
    for variable in ("RIDAGEYR", "RIDEXPRG", "DIQ010", "DIQ050", "DIQ070"):
        assert f"rule on {variable}" in text
    assert I.BANG_MISSING_RULE in text and I.BANG_MISSING_RULE in (REPO / I.BANG_NOTES).read_text()
    assert "3 (Borderline) is known diabetes" in text
    if I.BANG_TABLE.is_file():
        columns = pd.read_parquet(I.BANG_TABLE).columns
        assert {"DIQ010", "DIQ050", "DIQ070", "RIDAGEYR", "RIDEXPRG", "LBXGLU"} <= set(columns)


def test_the_bang_card_is_refused_when_the_notes_no_longer_quote_the_missing_rule(monkeypatch,
                                                                                tmp_path):
    if not I.BANG_TABLE.is_file():
        pytest.skip("the Bang table is not on disk")
    notes = tmp_path / "bang2009.md"
    notes.write_text((REPO / I.BANG_NOTES).read_text().replace(I.BANG_MISSING_RULE, "(gone)"))
    monkeypatch.setattr(I, "BANG_NOTES", str(notes))
    with pytest.raises(SystemExit, match="no longer quotes the paper's rule"):
        I.bang_eligibility(pd.read_parquet(I.BANG_TABLE))
