"""The swarm's intake (``research/swarm/intake.py``): manifests, question cards and answers.

* A manifest's SHA-256 is hashlib's on the bytes the server sent (a local server serves bytes the
  test makes), and the fetcher refuses a page served for a transport file (NHANES's old URLs
  answer 200 with an HTML "Page Not Found"), leaving nothing on disk.
* The committed manifests match the bytes on disk where the data are present, and the Sullivan
  files' SHA-256 equal those of the replication's own manifest, an independent download.
* Every card and answers file validates; the validators fire on each broken field.
* No card states a number its answers state, beyond the pinned coincidences; the scan fires on
  one written in.
* The fixtures' truths are the generators': the clinical model's log-odds are the ones
  make_fixtures.py computes, its C-statistic agrees with a Mann-Whitney count, and the survey's
  outcome does not move when its ages are changed.
* The tables take no analytic decision (checked against pandas' own SAS reader where the NHANES
  files are present).
"""
from __future__ import annotations

import copy
import hashlib
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.swarm import decoys  # noqa: E402
from research.swarm import intake as I  # noqa: E402

IDS = list(I.INTAKES)
# The numbers a card may share with its answers, each a coincidence of two different facts
# (intake.SHARED_NUMBERS says which): pinned here, so a new one needs this test changed.
PINNED_SHARED = {"bang2009": {1.0}, "sullivan2021": {10.0, 20.0}}
XPT = I.XPT_MAGIC + b"!!!!!!!000000000000000000000000000000  " + bytes(range(256)) * 40
CODEBOOK = (b"<html><head><title>LAB10AM Codebook</title></head><body><p>First Published: "
            b"June 2002</p><p>Last Revised: September 2009</p></body></html>")
NOT_FOUND = (b"<!DOCTYPE html><html><head><title>Page Not Found | CDC</title></head><body>"
             b"Page Not Found: this page has moved</body></html>")
LOCAL = {"pause": 0, "log": lambda s: None, "hosts": ("127.0.0.1",)}  # the test's own server


def committed(id_: str) -> dict:
    return I.load(id_)


# ── downloading ──────────────────────────────────────────────────────────────


@pytest.fixture
def server():
    """A local server of the bytes a test puts in ``pages`` (path → bytes), counting requests."""
    pages: dict[str, bytes] = {}
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = pages.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain" if self.path.endswith(".xpt")
                             else "text/html")
            self.send_header("Last-Modified", "Sun, 22 May 2014 00:00:00 GMT")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(http.server.ThreadingHTTPServer):
        def server_bind(self):  # HTTPServer's own looks the host's name up, which can stall
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    httpd = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}" + "/{year}/{file}.{ext}"
    try:
        yield {"pages": pages, "hits": hits, "spec": {"url": url, "cycles": {
            "1999-2000": {"year": 1999, "files": ["LAB10AM"]}}}}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_manifests_sha256_is_hashlibs_on_the_bytes_the_server_sent(server, tmp_path):
    server["pages"].update({"/1999/LAB10AM.xpt": XPT, "/1999/LAB10AM.htm": CODEBOOK})
    files = I.download(server["spec"], folder=tmp_path, **LOCAL)
    xpt, htm = files
    assert xpt["sha256"] == hashlib.sha256(XPT).hexdigest()
    assert htm["sha256"] == hashlib.sha256(CODEBOOK).hexdigest()
    assert xpt["bytes"] == len(XPT) and htm["bytes"] == len(CODEBOOK)
    assert (tmp_path / "1999-2000" / "LAB10AM.xpt").read_bytes() == XPT
    assert xpt["release"] == {"first_published": "June 2002", "last_revised": "September 2009"}
    assert xpt["codebook"] == htm["url"] and htm["url"].endswith("/1999/LAB10AM.htm")
    assert xpt["downloaded"] == time.strftime("%Y-%m-%d")
    assert xpt["last_modified_header"] == "Sun, 22 May 2014 00:00:00 GMT"
    assert not list(tmp_path.rglob("*.part"))


def test_a_page_served_for_a_transport_file_is_refused_and_nothing_is_written(server, tmp_path):
    server["pages"].update({"/1999/LAB10AM.xpt": NOT_FOUND, "/1999/LAB10AM.htm": CODEBOOK})
    with pytest.raises(I.FetchRefused, match="not a SAS transport file"):
        I.download(server["spec"], folder=tmp_path, **LOCAL)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


def test_a_not_found_page_served_for_a_codebook_is_refused(server, tmp_path):
    server["pages"].update({"/1999/LAB10AM.xpt": XPT, "/1999/LAB10AM.htm": NOT_FOUND})
    with pytest.raises(I.FetchRefused, match="not an NHANES codebook page"):
        I.download(server["spec"], folder=tmp_path, **LOCAL)
    assert not (tmp_path / "1999-2000" / "LAB10AM.htm").exists()
    assert I.refusal(CODEBOOK, "htm") is None and I.refusal(XPT, "xpt") is None


def test_a_url_off_the_cdcs_host_or_with_a_query_is_refused_before_any_request(server, tmp_path):
    """The intake's own hosts are wwwn.cdc.gov alone: the local server is refused under them, a
    query string is refused under any, and neither sends a request."""
    server["pages"].update({"/1999/LAB10AM.xpt": XPT, "/1999/LAB10AM.htm": CODEBOOK})
    assert I.HOSTS == ("wwwn.cdc.gov",)
    with pytest.raises(I.FetchRefused, match="only wwwn.cdc.gov"):
        I.download(server["spec"], folder=tmp_path, pause=0, log=lambda s: None)
    queried = {**server["spec"], "url": server["spec"]["url"] + "?email=x"}
    with pytest.raises(I.FetchRefused, match="no query string"):
        I.download(queried, folder=tmp_path, **LOCAL)
    assert server["hits"] == [] and not [p for p in tmp_path.rglob("*") if p.is_file()]
    for spec in (I.bang_spec(), json.loads((I.SULLIVAN / "files.json").read_text())):
        assert spec["url"].startswith("https://wwwn.cdc.gov/") and "?" not in spec["url"]


def test_a_file_on_disk_is_not_fetched_again_and_a_changed_one_is_caught(server, tmp_path):
    server["pages"].update({"/1999/LAB10AM.xpt": XPT, "/1999/LAB10AM.htm": CODEBOOK})
    first = I.download(server["spec"], folder=tmp_path, **LOCAL)
    manifest = {"files": first}
    before = len(server["hits"])
    again = I.download(server["spec"], manifest, folder=tmp_path, **LOCAL)
    assert len(server["hits"]) == before  # nothing fetched
    assert again == first  # the same entries, the download date kept
    assert I.changed(manifest) == []
    path = tmp_path / "1999-2000" / "LAB10AM.xpt"
    path.write_bytes(XPT[:-1] + b"?")
    assert I.changed(manifest) == [first[0]["file"]]
    redone = I.download(server["spec"], manifest, folder=tmp_path, **LOCAL)
    assert redone[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert redone[0]["downloaded"] is None  # these bytes are not the ones downloaded then


# ── the committed manifests ──────────────────────────────────────────────────


@pytest.mark.parametrize("id_", IDS)
def test_a_committed_manifest_matches_the_bytes_on_disk(id_):
    manifest = committed(id_)["manifest"]
    entries = [*manifest["files"], manifest["table"]]
    present = [e for e in entries if (REPO / e["file"]).is_file()]
    if not present:
        pytest.skip(f"{id_}: no data file on disk (research/data/ is not tracked)")
    for e in present:
        path = REPO / e["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e["sha256"], e["file"]
        assert path.stat().st_size == e["bytes"], e["file"]


def test_the_sullivan_files_hash_as_the_replications_own_download_did():
    """Two downloads of the same CDC files, a day's work apart, by two intakes: every file the
    replication's manifest lists is in this one with the same SHA-256 and size."""
    ours = {e["file"]: e for e in committed("sullivan2021")["manifest"]["files"]}
    theirs = json.loads((I.SULLIVAN / "manifest.json").read_text())["files"]
    assert len(theirs) > 0 and set(e["file"] for e in theirs) == set(ours)
    for e in theirs:
        assert (ours[e["file"]]["sha256"], ours[e["file"]]["bytes"]) == (e["sha256"], e["bytes"])


def test_every_downloaded_file_is_from_the_cdc_with_its_codebook_dates():
    for id_ in ("bang2009", "sullivan2021"):
        manifest = committed(id_)["manifest"]
        assert manifest["license"] == "public domain"
        for e in manifest["files"]:
            assert e["url"].startswith("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"), e
            assert "?" not in e["url"] and "@" not in e["url"]
            assert len(e["sha256"]) == 64 and e["bytes"] > 0 and e["downloaded"]
            if e["file"].endswith(".xpt"):
                assert set(e["release"]) == {"first_published", "last_revised"}
                assert e["release"]["first_published"], e["file"]


# ── cards and answers ────────────────────────────────────────────────────────


@pytest.mark.parametrize("id_", IDS)
def test_each_card_and_answers_file_validates(id_):
    files = committed(id_)
    assert I.validate_card(files["card"], id_) == []
    assert I.validate_answers(files["answers"], id_) == []
    assert files["card"]["table"]["path"] == files["manifest"]["table"]["file"]


def test_the_six_pilot_datasets_have_the_purposes_swarm_plan_names():
    purposes = {i: committed(i)["card"]["purpose"] for i in IDS}
    kinds = {i: committed(i)["answers"]["kind"] for i in IDS}
    assert purposes == {"bang2009": "Predict", "sullivan2021": "Estimate",
                        "nhanes_glucose_predict": "Predict",
                        "clinical_readmission_predict": "Predict",
                        "survey_support_estimate": "Estimate", I.DECOY_ID: "Predict"}
    assert kinds == {"bang2009": "replication", "sullivan2021": "replication",
                     "nhanes_glucose_predict": "no_known_answer",
                     "clinical_readmission_predict": "known_answer",
                     "survey_support_estimate": "known_answer", I.DECOY_ID: "decoy"}


@pytest.mark.parametrize("id_", IDS)
def test_no_card_states_a_number_its_answers_state(id_):
    files = committed(id_)
    shared = I.shared_numbers(files["card"], files["answers"])
    assert shared <= PINNED_SHARED.get(id_, set()), sorted(shared)


def test_the_shared_numbers_the_intake_allows_are_the_pinned_ones():
    assert {k: set(v) for k, v in I.SHARED_NUMBERS.items()} == PINNED_SHARED
    assert all(I._text(reason) for v in I.SHARED_NUMBERS.values() for reason in v.values())


def test_the_number_scan_fires_on_an_answer_number_written_into_a_card():
    files = committed("bang2009")
    card = copy.deepcopy(files["card"])
    assert I.problems("bang2009", card, files["answers"]) == []
    card["question"] = card["question"].rstrip("?") + ", with an AUC near 0.79?"
    assert 0.79 in I.shared_numbers(card, files["answers"])
    assert any("numbers an answer states" in p for p in I.problems("bang2009", card,
                                                                     files["answers"]))
    sul = committed("sullivan2021")
    card = copy.deepcopy(sul["card"])
    card["population"] += " The analytic sample is 16,913 people."
    assert 16913.0 in I.shared_numbers(card, sul["answers"])
    card = copy.deepcopy(sul["card"])
    card["prepared"] = card["prepared"] + ["β is −2.0 per step."]
    assert 2.0 in I.shared_numbers(card, sul["answers"])


def test_the_number_scan_reads_numbers_not_names_or_identifiers():
    card = copy.deepcopy(committed("bang2009")["card"])
    assert I.card_numbers({"x": "DBQ700, item_05, WTSAF2YR and log10"}) == set()
    assert I.card_numbers({"x": "from 1999–2004, 21,849 rows, 7.0 mmol/L"}) == {
        1999.0, 2004.0, 21849.0, 7.0}
    # the paper's DOI holds 11, a number Table 3 prints; the identifier is not scanned
    assert "11" in card["replication"]["paper"]["doi"]
    assert 11.0 not in I.card_numbers(card)
    assert 11.0 in I.answer_numbers(committed("bang2009")["answers"])


BROKEN = [
    ("question", lambda c: c.pop("question"), "missing question"),
    ("unknown", lambda c: c.update(answer="x"), "unknown field answer"),
    ("purpose", lambda c: c.update(purpose="Describe"), "purpose"),
    ("design", lambda c: c.update(design="audit"), "design"),
    ("no mark", lambda c: c.update(question="Which predictors matter."), "question mark"),
    ("outcome", lambda c: c.update(outcome={"column": "x"}), "outcome is not"),
    ("exposure", lambda c: c.update(exposure={"column": "x", "meaning": "y"}),
     "a Predict card names an exposure"),
    ("paper", lambda c: c["replication"].pop("paper"), "names no paper"),
    ("paper field", lambda c: c["replication"]["paper"].update(auc="0.79"), "the paper holds"),
    ("absolute", lambda c: c["table"].update(path="/tmp/x.parquet"), "table path"),
    ("suffix", lambda c: c["table"].update(path="research/data/x.csv"), "table path"),
    ("prepared", lambda c: c.update(prepared="stacked"), "prepared is not"),
    ("manifest", lambda c: c.update(manifest="manifest.json"), "manifest"),
]


@pytest.mark.parametrize("label, breaks, expect", BROKEN, ids=[b[0] for b in BROKEN])
def test_the_card_validator_fires_on_each_broken_field(label, breaks, expect):
    card = copy.deepcopy(committed("bang2009")["card"])
    assert I.validate_card(card, "bang2009") == []
    breaks(card)
    errors = I.validate_card(card, "bang2009")
    assert any(expect in e for e in errors), errors


def test_an_estimate_card_needs_its_exposure_and_ours_no_paper():
    card = copy.deepcopy(committed("survey_support_estimate")["card"])
    assert I.validate_card(card, "survey_support_estimate") == []
    card.pop("exposure")
    assert any("names no exposure" in e for e in I.validate_card(card))
    card = copy.deepcopy(committed("survey_support_estimate")["card"])
    card["replication"] = committed("bang2009")["card"]["replication"]
    assert any("names a replication" in e for e in I.validate_card(card))
    assert any("is not its folder's" in e for e in I.validate_card(card, "bang2009"))


def test_the_answers_validator_fires_on_each_broken_field():
    decoy = copy.deepcopy(committed(I.DECOY_ID)["answers"])
    assert I.validate_answers(decoy, I.DECOY_ID) == []
    decoy["real_findings"] = ["age"]
    assert any("real_findings" in e for e in I.validate_answers(decoy))
    known = copy.deepcopy(committed("clinical_readmission_predict")["answers"])
    assert I.validate_answers(known) == []
    known["fixture_key"] = "nowhere.csv"
    assert any("FIXTURE_TRUTHS" in e for e in I.validate_answers(known))
    rep = copy.deepcopy(committed("bang2009")["answers"])
    rep["truths"] = []
    assert any("states no truth" in e for e in I.validate_answers(rep))
    rep = copy.deepcopy(committed("bang2009")["answers"])
    rep["truths"][0].pop("source")
    assert any("truth 0" in e for e in I.validate_answers(rep))
    rep["kind"] = "guess"
    assert any("kind" in e for e in I.validate_answers(rep))


@pytest.mark.parametrize("id_", IDS)
def test_a_cards_columns_are_in_its_table(id_):
    card = committed(id_)["card"]
    path = REPO / card["table"]["path"]
    if not path.is_file():
        pytest.skip(f"{id_}: its table is not on disk (research/data/ is not tracked)")
    columns = list(pd.read_parquet(path).columns if path.suffix == ".parquet"
                   else pd.read_csv(path, nrows=5).columns)
    assert card["outcome"]["column"] in columns
    if "exposure" in card:
        assert card["exposure"]["column"] in columns


def test_the_decoys_card_does_not_say_what_it_is():
    """Its card is shaped as the generated fixtures' are, and no word or path marks it."""
    decoy, fixture = committed(I.DECOY_ID)["card"], committed("clinical_readmission_predict")["card"]
    text = json.dumps(decoy).lower()
    for word in ("decoy", "null", "noise", "random", "signal", "independent", "seed"):
        assert word not in text, word
    assert set(decoy) == set(fixture) and decoy["source"] == fixture["source"]
    assert decoy["prepared"] == fixture["prepared"] == []


def test_the_decoys_table_is_its_seeds():
    entry = committed(I.DECOY_ID)["manifest"]
    assert entry["generator"]["seed"] == I.DECOY_SEED
    data = decoys.generate(I.DECOY_SEED, entry["generator"]["rows"]).to_csv(index=False).encode()
    assert hashlib.sha256(data).hexdigest() == entry["table"]["sha256"]


# ── the fixtures' truths ─────────────────────────────────────────────────────


def test_the_clinical_model_is_the_log_odds_the_generator_computes(monkeypatch):
    """make_fixtures.py::clinical_risk draws readmit_30d with probability 1 / (1 + exp(-z)): the
    one np.exp call it makes receives -z, which must equal the answers' model on its rows."""
    from turbotab.sample_data import make_fixtures
    seen = []
    exp = np.exp

    def spy(x, *args, **kwargs):
        seen.append(np.array(x, dtype=float))
        return exp(x, *args, **kwargs)

    monkeypatch.setattr(np, "exp", spy)
    frame = make_fixtures.clinical_risk()
    monkeypatch.setattr(np, "exp", exp)
    assert len(seen) == 1
    model = next(t["value"] for t in committed("clinical_readmission_predict")["answers"]["truths"]
                 if t["name"] == "the outcome's generating model")
    np.testing.assert_allclose(-seen[0], I.linear_predictor(model, frame), rtol=0, atol=1e-12)
    tracked = pd.read_csv(I.SAMPLE / "clinical_risk.csv")
    pd.testing.assert_frame_equal(frame, tracked, check_dtype=False)
    in_model = {t["column"] for t in model["terms"]}
    assert in_model.isdisjoint(model["no_signal"])
    assert in_model | set(model["no_signal"]) == set(tracked.columns) - {"readmit_30d"}


def test_the_clinical_c_statistic_is_a_mann_whitney_count():
    from scipy.stats import mannwhitneyu
    answers = committed("clinical_readmission_predict")["answers"]
    truth = {t["name"]: t["value"] for t in answers["truths"]}
    frame = pd.read_csv(I.SAMPLE / "clinical_risk.csv")
    z = I.linear_predictor(truth["the outcome's generating model"], frame)
    y = frame["readmit_30d"].to_numpy()
    u = mannwhitneyu(z[y == 1], z[y == 0]).statistic  # ties count a half, as the C-statistic's
    assert truth["the C-statistic of ranking every row by its true risk"] == pytest.approx(
        u / ((y == 1).sum() * (y == 0).sum()), abs=1e-12)


def test_the_surveys_outcome_does_not_move_when_its_ages_do(monkeypatch):
    """make_fixtures.py::survey with every age drawn reversed (the same draws, so every later
    draw is the same): sought_support is unchanged, so it is not a function of age."""
    from turbotab.sample_data import make_fixtures
    original = make_fixtures.survey()
    base = np.random.RandomState

    class ReversedAges(base):
        def randint(self, low, high=None, size=None, *args, **kwargs):
            out = super().randint(low, high, size, *args, **kwargs)
            return out[::-1].copy() if (low, high) == (18, 82) else out

    monkeypatch.setattr(np.random, "RandomState", ReversedAges)
    changed = make_fixtures.survey()
    monkeypatch.setattr(np.random, "RandomState", base)
    assert not changed["age"].equals(original["age"])
    assert np.array_equal(changed["age"].to_numpy(), original["age"].to_numpy()[::-1])
    pd.testing.assert_series_equal(changed["sought_support"], original["sought_support"])
    tracked = pd.read_csv(I.SAMPLE / "survey_instrument.csv")
    assert np.array_equal(tracked["sought_support"], original["sought_support"])
    assert np.array_equal(tracked["age"], original["age"])


# ── the tables take no analytic decision ─────────────────────────────────────


def nhanes_present(stems: dict[str, tuple[str, ...]]) -> bool:
    """Whether each cycle's files (``stems[cycle]``) are on disk."""
    return all((I.NHANES / c / f"{s}.xpt").is_file() for c, names in stems.items() for s in names)


def test_the_bang_table_keeps_every_participant_and_names_the_outcome_by_the_cut():
    """Against pandas' own SAS reader: one row per DEMO participant of the three cycles, the
    children, the pregnant and the known diabetics among them (no exclusion), and diabetes_fpg
    1 exactly where pandas' LBXGLU is at or above the cut."""
    stems = {c: (f"DEMO{s}", I.bang_stems("LAB10AM")[c], f"DIQ{s}")
             for c, (_, s) in I.BANG_CYCLES.items()}
    if not (nhanes_present(stems) and I.BANG_TABLE.is_file()):
        pytest.skip("the Bang 2009 files are not on disk")
    table = pd.read_parquet(I.BANG_TABLE)
    demo = pd.concat([pd.read_sas(I.NHANES / c / f"{d}.xpt", format="xport")
                      for c, (d, _, _) in stems.items()])
    glu = pd.concat([pd.read_sas(I.NHANES / c / f"{g}.xpt", format="xport")[["SEQN", "LBXGLU"]]
                     for c, (_, g, _) in stems.items()])
    diq = pd.concat([pd.read_sas(I.NHANES / c / f"{q}.xpt", format="xport")[["SEQN", "DIQ010"]]
                     for c, (_, _, q) in stems.items()])
    assert len(table) == len(demo) and set(table["SEQN"]) == set(demo["SEQN"])
    assert (table["RIDAGEYR"] < I.BANG_MIN_AGE).any()
    assert (table["RIDEXPRG"] == 1).any()
    assert (table["DIQ010"] == 1).sum() == (diq["DIQ010"] == 1).sum() > 0
    want = table[["SEQN"]].merge(glu, on="SEQN", how="left")["LBXGLU"]
    expected = np.where(want.isna(), np.nan, (want >= I.BANG_FPG).astype(float))
    np.testing.assert_array_equal(table[I.BANG_OUTCOME].to_numpy(), expected)
    assert np.nansum(expected) > 0 and (expected == 0).sum() > 0


def test_the_sullivan_table_is_the_preps_files_joined_to_demo():
    plan = I.sullivan_plan()
    folder = REPO / plan["table"]["join_folder"]
    if not (I.SULLIVAN_TABLE.is_file() and (REPO / plan["table"]["main"]).is_file()):
        pytest.skip("the Sullivan prep's files are not on disk")
    table = pd.read_parquet(I.SULLIVAN_TABLE)
    demo = pd.read_parquet(REPO / plan["table"]["main"])
    pd.testing.assert_frame_equal(table[list(demo.columns)], demo)  # DEMO's rows, in its order
    indexed = table.set_index("SEQN")
    for j in plan["table"]["joins"]:
        part = pd.read_parquet(folder / f"{j}.parquet").set_index("SEQN")
        columns = list(part.columns)
        pd.testing.assert_frame_equal(indexed.loc[part.index, columns], part, check_dtype=False)
        assert indexed.drop(index=part.index)[columns].isna().all().all(), j
    assert set(table.columns) == set(demo.columns).union(
        *(pd.read_parquet(folder / f"{j}.parquet").columns for j in plan["table"]["joins"]))


def test_every_column_the_sullivan_prep_derives_has_its_sentence_in_the_card():
    """The card's prepared steps describe the prep's derived columns one by one: a column the prep
    adds without a sentence (or a sentence for one it dropped) fails here."""
    path = REPO / I.sullivan_plan()["table"]["join_folder"] / "derived.parquet"
    if not path.is_file():
        pytest.skip("the Sullivan prep's files are not on disk")
    derived = set(pd.read_parquet(path).columns) - {"SEQN"}
    said = {s.split()[1].rstrip(":,") for s in committed("sullivan2021")["card"]["prepared"]
            if s.startswith("Derived ")}
    assert said == derived
