"""The swarm's intake (``research/swarm/intake.py``): manifests, question cards and answers.

* A manifest's SHA-256 is hashlib's on the bytes the server sent (a local server serves bytes the
  test makes), and the fetcher refuses a page served for a transport file (NHANES's old URLs
  answer 200 with an HTML "Page Not Found"), leaving nothing on disk.
* The committed manifests match the bytes on disk where the data are present, and the Sullivan
  files' SHA-256 equal those of the replication's own manifest, an independent download.
* Every card and answers file validates; the validators fire on each broken field.
* No card states a number its answers state, in digits or in words, to the card's own decimal
  places, beyond the coincidences pinned by their form ("1" pinned does not excuse "1.0" or
  "one"); an answer is compared as stated, times 100 and divided by 100 (0.74194 is "74%", a
  sensitivity of 82 is "0.82"). The scan reads ".79", "210cases", "2.6x", "74%" and
  "twenty-five" whole, fires on an answer number written in at any precision that rounds to it,
  and stays silent on names (DBQ700, item_05, log10, item_one), words that hold a number word
  (someone, often, weight) and the paper's DOI.
* The table builders' refusals (a repeated or stray SEQN, a column clash, a variable no cycle
  carries, the NHANES export's changed bytes or repeated participant, a join other than a left
  join by SEQN) each fire on a frame built to trip them and stay silent on a clean one.
* The NHANES card reads each cycle's age top code and the values the export flags as imputed
  from the frame; the survey card states the instrument's whole reverse-coding key.
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
import re
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
# (intake.SHARED_NUMBERS says which), by the form the card writes them in: pinned here, so a new
# one needs this test changed. A whole 0 or 1 (a binary column's coding) matches every answer
# within 0.5 of it, and every one up to 50, or from 50 to 150, divided by 100.
PINNED_SHARED = {"bang2009": {"0", "1", "one"},
                 "sullivan2021": {"0", "zero", "1", "one", "two", "five", "10", "20", "24"},
                 "nhanes_glucose_predict": {"one"},
                 "clinical_readmission_predict": {"0", "1", "one"},
                 "inpatient_readmission_predict": {"0", "1", "one"},
                 "walking_bp_estimate": {"0", "1", "one"},
                 "plasma_panel_estimate": {"0", "1", "one"}}
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
    """SWARM_PLAN §5's six, the decoy, and the version-1 dry run's two fixtures, which its
    committed records still name until the dry run is made again on the hardened fixtures (the
    hardened Predict, the Estimate and the measurement fixture: items 4 to 6)."""
    purposes = {i: committed(i)["card"]["purpose"] for i in IDS}
    kinds = {i: committed(i)["answers"]["kind"] for i in IDS}
    assert purposes == {"bang2009": "Predict", "sullivan2021": "Estimate",
                        "nhanes_glucose_predict": "Predict",
                        "clinical_readmission_predict": "Predict",
                        "survey_support_estimate": "Estimate", I.DECOY_ID: "Predict",
                        "inpatient_readmission_predict": "Predict",
                        "walking_bp_estimate": "Estimate", "plasma_panel_estimate": "Estimate"}
    assert kinds == {"bang2009": "replication", "sullivan2021": "replication",
                     "nhanes_glucose_predict": "no_known_answer",
                     "clinical_readmission_predict": "known_answer",
                     "survey_support_estimate": "known_answer", I.DECOY_ID: "decoy",
                     "inpatient_readmission_predict": "known_answer",
                     "walking_bp_estimate": "known_answer", "plasma_panel_estimate": "known_answer"}


@pytest.mark.parametrize("id_", IDS)
def test_no_card_states_a_number_its_answers_state(id_):
    """Beyond the pinned coincidences, each of which still occurs (a pin no card needs would
    excuse that number's leak unseen)."""
    files = committed(id_)
    shared = I.shared_numbers(files["card"], files["answers"])
    assert shared == PINNED_SHARED.get(id_, set()), sorted(shared)


def test_the_shared_numbers_the_intake_allows_are_the_pinned_ones():
    assert {k: set(v) for k, v in I.SHARED_NUMBERS.items()} == PINNED_SHARED
    assert all(I._text(reason) for v in I.SHARED_NUMBERS.values() for reason in v.values())


def test_the_number_scan_fires_on_an_answer_number_written_into_a_card():
    files = committed("bang2009")
    card = copy.deepcopy(files["card"])
    assert I.problems("bang2009", card, files["answers"]) == []
    card["question"] = card["question"].rstrip("?") + ", with an AUC near 0.79?"
    assert "0.79" in I.shared_numbers(card, files["answers"])
    assert any("numbers an answer states" in p for p in I.problems("bang2009", card,
                                                                     files["answers"]))
    sul = committed("sullivan2021")
    card = copy.deepcopy(sul["card"])
    card["population"] += " The analytic sample is 16,913 people."
    assert "16913" in I.shared_numbers(card, sul["answers"])
    card = copy.deepcopy(sul["card"])
    card["prepared"] = card["prepared"] + ["β is −2.0 per step."]
    assert "2.0" in I.shared_numbers(card, sul["answers"])


def test_the_number_scan_reads_numbers_not_names_or_identifiers():
    card = copy.deepcopy(committed("bang2009")["card"])
    assert I.card_numbers({"x": "DBQ700, item_05, WTSAF2YR and log10"}) == set()
    assert I.card_numbers({"x": "Someone often weighs item_one; none alone is done."}) == set()
    assert I.card_numbers({"x": "from 1999–2004, 21,849 rows, 7.0 mmol/L"}) == {
        1999.0, 2004.0, 21849.0, 7.0}
    # the paper's DOI holds 11, a number Table 3 prints; the identifier is not scanned
    assert "11" in card["replication"]["paper"]["doi"]
    assert 11.0 not in I.card_numbers(card)
    assert 11.0 in I.answer_numbers(committed("bang2009")["answers"])


# (text, its numbers with the decimal places each is written to and its form): a leading point,
# a unit, sign or word against the number, the decimal part read whole, and numbers in words.
WRITTEN = [("AUC .79", {(0.79, 2, "0.79")}), ("p = .06", {(0.06, 2, "0.06")}),
           ("210cases", {(210.0, 0, "210")}), ("126mg", {(126.0, 0, "126")}),
           ("OR 2.6x", {(2.6, 1, "2.6")}), ("C = 0.742", {(0.742, 3, "0.742")}),
           ("21,849.5 rows", {(21849.5, 1, "21849.5")}), ("2.60", {(2.6, 2, "2.60")}),
           ("end.", set()), ("1,2345", {(1.0, 0, "1"), (2345.0, 0, "2345")}), ("v1.2.3", set()),
           ("−2.0", {(2.0, 1, "2.0")}), ("74%", {(74.0, 0, "74")}),
           ("74.2 percent", {(74.2, 1, "74.2")}), ("five or more", {(5.0, 0, "five")}),
           ("Twenty-five", {(25.0, 0, "twenty-five")}),
           ("ninety nine", {(99.0, 0, "ninety-nine")}),
           ("seventeen, zero", {(17.0, 0, "seventeen"), (0.0, 0, "zero")}),
           ("forty percent", {(40.0, 0, "forty")}), ("someone often weighs", set()),
           ("item_one", set())]


@pytest.mark.parametrize("text, readings", WRITTEN, ids=[w[0] for w in WRITTEN])
def test_the_number_scan_reads_each_way_a_number_is_written(text, readings):
    assert I.card_readings({"x": text}) == readings


# (dataset, a sentence written into its card's question, whether it states an answer's number to
# the sentence's own places: as stated, times 100 or divided by 100). Each answer is the dataset's
# answers.json's: Bang's AUC 0.79, P 0.06, 210 undiagnosed, OR 2.6, the activity CI's upper bound
# 1.0, P 0.004, Table 3's sensitivity of 82 and the high-risk cut of 5; the clinical C-statistic
# 0.74194...; the Sullivan sample 16,913 and its glucose beta -2.0. The pinned forms ("1" and "0"
# for Bang's coding, "one" and "two" for prose) stay silent, and no other form of their values
# does.
LEAKS = [
    ("bang2009", "with an AUC near .79", True),
    ("bang2009", "with p = .06 for activity", True),
    ("bang2009", "from 210cases", True),
    ("bang2009", "with an OR of 2.6x for men", True),
    ("bang2009", "with an AUC near .78", False),  # 0.79 is not .78 to two places
    ("bang2009", "with an OR of 2.7x for men", False),
    ("clinical_readmission_predict", "where C = 0.742", True),
    ("clinical_readmission_predict", "where C = 0.74", True),
    ("clinical_readmission_predict", "where C = .7", True),
    ("clinical_readmission_predict", "where C = 0.743", False),  # 0.74194 is 0.742 to three
    ("clinical_readmission_predict", "where C = 0.75", False),
    ("sullivan2021", "in 16,913 people", True),
    ("sullivan2021", "in 16,914 people", False),
    # a proportion written as a percent, and a percent as a proportion
    ("clinical_readmission_predict", "with a C-statistic of 74%", True),
    ("clinical_readmission_predict", "with a C-statistic of 74.2%", True),
    ("clinical_readmission_predict", "with a C-statistic of seventy-four percent", True),
    ("clinical_readmission_predict", "with a C-statistic of 76%", False),
    ("clinical_readmission_predict", "with a C-statistic of 74.5%", False),  # 74.194 is 74.2
    ("bang2009", "with a sensitivity of 0.82 at the cut", True),
    ("bang2009", "with a sensitivity of 0.84 at the cut", False),
    # a number in words
    ("bang2009", "with high risk at a score of five or more", True),
    ("bang2009", "with high risk at a score of twelve or more", False),
    # a pin excuses its own form alone
    ("bang2009", "with an activity odds ratio's upper bound of 1.0", True),
    ("bang2009", "with points of 0.0 for some", True),
    ("bang2009", "with an upper bound of 1.00", True),
    ("bang2009", "coded 1 or 0", False),
    ("bang2009", "with one row per participant", False),
    ("sullivan2021", "with a beta of -2 per step", True),
    ("sullivan2021", "with two recall days", False),
]


@pytest.mark.parametrize("id_, text, leaks", LEAKS, ids=[f"{d}:{t}" for d, t, _ in LEAKS])
def test_the_number_scan_fires_on_an_answer_number_at_the_cards_own_places(id_, text, leaks):
    files = committed(id_)
    card = copy.deepcopy(files["card"])
    assert I.problems(id_, card, files["answers"]) == []
    card["question"] = card["question"].rstrip("?") + f", {text}?"
    fired = any("numbers an answer states" in p for p in I.problems(id_, card, files["answers"]))
    assert fired is leaks


def test_the_number_scan_stays_silent_on_names_and_the_doi_whose_digits_an_answer_holds():
    """Answers built to hold every digit run the names and the DOI carry: none of it is read."""
    card = copy.deepcopy(committed("bang2009")["card"])
    doi = card["replication"]["paper"]["doi"]
    digits = [float(d) for d in re.findall(r"\d+", doi)] + [700.0, 5.0, 2.0, 10.0]
    answers = {"truths": [{"name": "x", "source": "y", "value": digits}]}
    card["population"] += " DBQ700, item_05, WTSAF2YR, log10 and item_one are names."
    # the outcome's coding 0 alone: 2, 3, 5, 10 and 11 divided by 100 round to it
    assert I.shared_numbers(card, answers) == {"0"}
    card["population"] += " 700 is not."
    assert "700" in I.shared_numbers(card, answers)


# (an answer, a card's text, whether the text states the answer to its own places): the answer as
# stated, times 100 and divided by 100, and no other scale.
SCALES = [
    (0.5, "50%", True), (0.5, "50.0 percent", True), (0.5, "fifty percent", True),
    (0.5, "51%", False), (0.5, "fifty-one percent", False), (0.5, "5", False),
    (50, "0.5", True), (50, "0.50", True), (50, "0.49", False), (50, "5.0", False),
    (12, "twelve", True), (12, "0.12", True), (12, "1200", True), (12, "thirteen", False),
    (12, "1.2", False), (12, "120", False),
]


@pytest.mark.parametrize("answer, text, leaks", SCALES, ids=[f"{a}:{t}" for a, t, _ in SCALES])
def test_the_number_scan_compares_an_answer_as_stated_as_a_percent_and_as_a_proportion(
        answer, text, leaks):
    answers = {"truths": [{"name": "x", "source": "y", "value": answer}]}
    assert bool(I.shared_numbers({"x": text}, answers)) is leaks


def test_a_pinned_number_excuses_its_own_form_and_the_refusal_names_the_other():
    """Bang's coding "1" is pinned; "1.0" (the activity CI's upper bound, to its printed place)
    is refused and named as the card writes it, and so is Sullivan's "2" beside its pinned
    "two"."""
    files = committed("bang2009")
    card = copy.deepcopy(files["card"])
    assert {"1", "0"} <= I.shared_numbers(card, files["answers"])
    assert I.problems("bang2009", card, files["answers"]) == []
    card["population"] += " The upper bound is 1.0."
    assert I.problems("bang2009", card, files["answers"]) == [
        "the card states numbers an answer states: 1.0"]
    sul = committed("sullivan2021")
    card = copy.deepcopy(sul["card"])
    assert "two" in I.shared_numbers(card, sul["answers"])
    card["population"] += " The beta is -2 per step."
    assert I.problems("sullivan2021", card, sul["answers"]) == [
        "the card states numbers an answer states: 2"]


def test_rounds_to_counts_a_tie_either_way_and_nothing_past_half_a_unit():
    assert I.rounds_to(0.745, 0.74, 2) and I.rounds_to(0.745, 0.75, 2)  # half even, half up
    assert not I.rounds_to(0.7451, 0.74, 2) and not I.rounds_to(0.7349, 0.74, 2)
    assert I.rounds_to(8.1, 8.0, 0) and not I.rounds_to(8.6, 8.0, 0)
    assert I.rounds_to(5258.0, 5258.0, 0) and not I.rounds_to(5258.0, 5259.0, 0)


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


# ── the table builders' refusals, on constructed frames ──────────────────────

BASE = pd.DataFrame({"SEQN": [1, 2, 3], "age": [30, 40, 50]})
LAB = pd.DataFrame({"SEQN": [3, 1], "glu": [99.0, 130.0]})


def test_merge_on_seqn_joins_a_clean_part_to_every_base_row_in_its_order():
    out = I.merge_on_seqn(BASE, {"LAB": LAB, "BP": pd.DataFrame({"SEQN": [2], "sys": [120]})})
    expected = pd.DataFrame({"SEQN": [1, 2, 3], "age": [30, 40, 50],
                             "glu": [130.0, np.nan, 99.0], "sys": [np.nan, 120.0, np.nan]})
    pd.testing.assert_frame_equal(out, expected)


MERGE_REFUSALS = [
    ("base repeats", BASE.iloc[[0, 0, 1]], {"LAB": LAB}, "the base table repeats a SEQN"),
    ("part repeats", BASE, {"LAB": pd.DataFrame({"SEQN": [1, 1], "glu": [1.0, 2.0]})},
     "LAB: SEQN repeats"),
    ("stray", BASE, {"LAB": pd.DataFrame({"SEQN": [1, 4], "glu": [1.0, 2.0]})},
     "LAB: 1 records whose SEQN the base table lacks"),
    ("clash with base", BASE, {"LAB": pd.DataFrame({"SEQN": [1], "age": [31]})},
     r"LAB: columns already in the table: \['age'\]"),
    ("clash between parts", BASE, {"A": LAB, "B": LAB},
     r"B: columns already in the table: \['glu'\]"),
]


@pytest.mark.parametrize("label, base, parts, message", MERGE_REFUSALS,
                         ids=[m[0] for m in MERGE_REFUSALS])
def test_merge_on_seqn_refuses_each_join_that_would_drop_repeat_or_rename(label, base, parts,
                                                                          message):
    with pytest.raises(SystemExit, match=message):
        I.merge_on_seqn(base, parts)


def xpt_folder(monkeypatch, frames: dict[str, pd.DataFrame], files: dict) -> None:
    """``I.read_xpt`` serves ``frames`` by ``<cycle>/<stem>``, and ``I.BANG_FILES`` is ``files``."""
    monkeypatch.setattr(I, "BANG_FILES", files)
    monkeypatch.setattr(I, "read_xpt", lambda path: frames[f"{path.parent.name}/{path.stem}"])


T = {"T": {"about": "a test file", "vars": ["SEQN", "A", "B"]}}
C1, C2, C3 = I.BANG_CYCLES  # 1999-2000 (T), 2001-2002 (T_B), 2003-2004 (T_C)


def cycles_of(**frames: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f"{c}/T{s}": f for c, s, f in zip((C1, C2, C3), ("", "_B", "_C"),
                                               (frames["one"], frames["two"], frames["three"]))}


def test_bang_stack_stacks_the_kept_variables_and_names_a_cycle_that_lacks_one(monkeypatch):
    one = pd.DataFrame({"SEQN": [1], "A": [1.0], "B": [2.0], "X": [9]})
    two = pd.DataFrame({"SEQN": [2], "A": [3.0]})  # no B
    three = pd.DataFrame({"SEQN": [3], "B": [6.0], "A": [5.0]})
    xpt_folder(monkeypatch, cycles_of(one=one, two=two, three=three), T)
    stacked, absent = I.bang_stack("T")
    expected = pd.DataFrame({"SEQN": [1, 2, 3], "A": [1.0, 3.0, 5.0], "B": [2.0, np.nan, 6.0]})
    pd.testing.assert_frame_equal(stacked, expected)
    assert absent == {"B": [C2]}


def test_bang_stack_refuses_a_variable_no_cycle_carries_or_a_cycle_without_seqn(monkeypatch):
    no_b = [pd.DataFrame({"SEQN": [i], "A": [1.0]}) for i in (1, 2, 3)]
    xpt_folder(monkeypatch, cycles_of(one=no_b[0], two=no_b[1], three=no_b[2]), T)
    with pytest.raises(SystemExit, match=r"T: no cycle carries \['B'\]"):
        I.bang_stack("T")
    full = [pd.DataFrame({"SEQN": [i], "A": [1.0], "B": [2.0]}) for i in (1, 2, 3)]
    xpt_folder(monkeypatch, cycles_of(one=full[0], two=full[1].drop(columns="SEQN"),
                                      three=full[2]), T)
    with pytest.raises(SystemExit, match=r"T: no cycle carries \['SEQN'\]"):
        I.bang_stack("T")


DEMO_LAB = {"DEMO": {"about": "demographics", "vars": ["SEQN", "RIDAGEYR"]},
            "LAB10AM": {"about": "glucose", "vars": ["SEQN", "LBXGLU"],
                        "stems": {C1: "LAB10AM", C2: "L10AM_B", C3: "L10AM_C"}}}


def demo_lab(lab_three: pd.DataFrame) -> dict[str, pd.DataFrame]:
    demo = {c: pd.DataFrame({"SEQN": [2 * i + 1, 2 * i + 2], "RIDAGEYR": [15.0, 60.0]})
            for i, c in enumerate((C1, C2, C3))}
    lab = {C1: pd.DataFrame({"SEQN": [1, 2], "LBXGLU": [126.0, 125.9]}),
           C2: pd.DataFrame({"SEQN": [3], "LBXGLU": [np.nan]}), C3: lab_three}
    stems = DEMO_LAB["LAB10AM"]["stems"]
    return {**{f"{c}/DEMO{s}": demo[c] for c, s in zip((C1, C2, C3), ("", "_B", "_C"))},
            **{f"{c}/{stems[c]}": lab[c] for c in (C1, C2, C3)}}


def test_bang_table_keeps_every_demo_row_and_cuts_the_outcome_at_126(monkeypatch):
    """By construction: glucose 126 is 1 (the cut is "126 mg/dL or more"), 125.9 is 0, a blank
    glucose and a participant with no glucose record are blank, and the child stays."""
    frames = demo_lab(pd.DataFrame({"SEQN": [6], "LBXGLU": [200.0]}))
    xpt_folder(monkeypatch, frames, DEMO_LAB)
    table, absent = I.bang_table()
    assert table["SEQN"].tolist() == [1, 2, 3, 4, 5, 6] and absent == {}
    np.testing.assert_array_equal(table[I.BANG_OUTCOME].to_numpy(),
                                  [1.0, 0.0, np.nan, np.nan, np.nan, 1.0])
    assert I.BANG_FPG == 126 and (table["RIDAGEYR"] < I.BANG_MIN_AGE).sum() == 3


def test_bang_table_refuses_a_glucose_record_whose_participant_demo_lacks(monkeypatch):
    frames = demo_lab(pd.DataFrame({"SEQN": [6, 99], "LBXGLU": [200.0, 90.0]}))
    xpt_folder(monkeypatch, frames, DEMO_LAB)
    with pytest.raises(SystemExit, match="LAB10AM: 1 records whose SEQN the base table lacks"):
        I.bang_table()


def nhanes_bytes() -> bytes:
    import gzip
    return gzip.decompress(I.NHANES_GZ.read_bytes())


def test_the_nhanes_export_is_read_only_as_the_bytes_stage_harness_pins():
    from turbotab.core.tests.stage_harness import NHANES_SHA256
    data = nhanes_bytes()
    assert hashlib.sha256(data).hexdigest() == NHANES_SHA256
    assert len(I.nhanes_export(data, NHANES_SHA256)) == data.count(b"\n") - 1  # one per line
    flipped = bytearray(data)
    flipped[len(data) // 2] ^= 1  # one bit of one byte
    with pytest.raises(SystemExit, match="not the bytes stage_harness pins"):
        I.nhanes_export(bytes(flipped), NHANES_SHA256)


@pytest.mark.parametrize("csv, refused", [
    (b"SEQN,glucose\n1,90\n2,100\n", False),
    (b"SEQN,glucose\n1,90\n1,100\n", True),  # a participant twice
    (b"SEQN,glucose\n1,90\n2,\n", True),  # a row without a glucose
])
def test_the_nhanes_export_is_refused_unless_one_row_per_participant_each_with_a_glucose(
        csv, refused):
    pinned = hashlib.sha256(csv).hexdigest()
    if refused:
        with pytest.raises(SystemExit, match="one row per participant, each with a glucose"):
            I.nhanes_export(csv, pinned)
    else:
        assert I.nhanes_export(csv, pinned)["SEQN"].tolist() == [1, 2]


def test_the_nhanes_intake_writes_nothing_from_an_export_with_a_changed_byte(monkeypatch, tmp_path):
    import gzip
    data = bytearray(nhanes_bytes())
    data[-2] ^= 1
    gz = tmp_path / "nhanes.csv.gz"
    gz.write_bytes(gzip.compress(bytes(data)))
    monkeypatch.setattr(I, "NHANES_GZ", gz)
    monkeypatch.setattr(I, "TABLES", tmp_path / "tables")
    with pytest.raises(SystemExit, match="not the bytes stage_harness pins"):
        I.intake_nhanes_glucose_predict(None)
    assert not (tmp_path / "tables").exists()


def test_the_sullivan_table_is_refused_on_any_join_but_a_left_join_by_seqn(tmp_path):
    BASE.to_parquet(tmp_path / "DEMO.parquet", index=False)
    LAB.to_parquet(tmp_path / "LAB.parquet", index=False)
    plan = {"table": {"main": str(tmp_path / "DEMO.parquet"), "join_folder": str(tmp_path),
                      "joins": ["LAB"], "on": "SEQN", "how": "left"}}
    out = I.sullivan_table(None, plan)
    assert out["SEQN"].tolist() == [1, 2, 3] and out["glu"].isna().tolist() == [False, True, False]
    for on, how in (("SEQN", "inner"), ("SEQN", "outer"), ("RESPID", "left")):
        bad = {"table": {**plan["table"], "on": on, "how": how}}
        with pytest.raises(SystemExit, match="merge_on_seqn does not"):
            I.sullivan_table(None, bad)


# ── the NHANES and survey cards' sentences, from the frame ───────────────────


def ages(by_cycle: dict[int, list[float]]) -> pd.DataFrame:
    rows = [(y, a) for y, values in by_cycle.items() for a in values]
    return pd.DataFrame({"cycle_begin_year": [y for y, _ in rows], "age": [a for _, a in rows]})


def test_the_nhanes_population_reads_each_cycles_top_code_from_the_frame():
    """NHANES's demographics codebooks: "Individuals 85 and over are topcoded at 85 years of age"
    in DEMO_D (2005–2006), "80 and over ... at 80" in DEMO_E (2007–2008)."""
    text = I.nhanes_population(ages({2005: [18, 85], 2007: [19, 80], 2009: [30, 80]}))
    assert "aged 18 or older" in text and "the 3 survey cycles 2005–2006 to 2009–2010" in text
    assert ("85 stands for 85 or older in 2005–2006 and 80 stands for 80 or older in the cycles "
            "2007–2008 to 2009–2010") in text
    for wrong in ({2005: [18, 84], 2007: [19, 80]}, {2005: [18, 85], 2007: [19, 85]}):
        with pytest.raises(SystemExit, match="not the top code"):
            I.nhanes_population(ages(wrong))


def test_the_nhanes_prepared_steps_count_the_flagged_values_and_the_blanks():
    frame = pd.DataFrame({"age": [18.0, 40.0, 70.0], "glucose": [90.0, 100.0, 130.0],
                          "weight": [70.0, 80.0, 90.0], "bp_sys": [120.0, np.nan, 110.0],
                          "meds": [1.0, np.nan, np.nan],
                          "imputed_weight": [True, False, True],
                          "imputed_bp_sys": [False, False, True]})
    imputed, rows = I.nhanes_prepared(frame)
    assert ("weight on the 2 rows where imputed_weight is true and bp_sys on the 1 row where "
            "imputed_bp_sys is true; blanks remain in bp_sys (1 row)") in imputed
    assert "an age of 18 or more" in rows
    assert "No column has a blank but bp_sys (1 row) and meds (2 rows)." in rows
    assert I.nhanes_prepared(frame.drop(columns=["imputed_weight", "imputed_bp_sys"])) == [rows]
    with pytest.raises(SystemExit, match="flags no column"):
        I.nhanes_prepared(frame.assign(imputed_height=False))


def test_the_nhanes_card_states_the_exports_top_codes_and_flagged_values():
    card = committed("nhanes_glucose_predict")["card"]
    frame = pd.read_csv(I.NHANES_GZ)
    assert "aged 18 or older" in card["population"] and " to 85" not in card["population"]
    assert ("85 stands for 85 or older in the cycles 2001–2002 to 2005–2006 and 80 stands for 80 "
            "or older in the cycles 2007–2008 to 2017–2018") in card["population"]
    imputed = card["prepared"][0]
    for c in [c for c in frame.columns if c.startswith("imputed_")]:
        assert f"on the {int(frame[c].sum()):,} rows where {c} is true" in imputed
    assert "none of these measures has a blank" in imputed
    assert card["prepared"] == I.nhanes_prepared(frame)


def test_the_survey_card_states_the_instruments_whole_key():
    """The key from survey_instrument.md's own table (8 reverse-coded items), not the generator."""
    doc = (I.SAMPLE / "survey_instrument.md").read_text()
    row = next(line for line in doc.splitlines() if line.startswith("| Reverse-coded items"))
    numbers = re.findall(r"`(?:item_)?(\d\d)`", row)
    key = [f"item_{n}" for n in numbers]
    assert len(key) == 8 and key[0] == "item_05"
    text = " ".join(committed("survey_support_estimate")["card"]["documentation"])
    assert f"its reverse-coded items are {', '.join(key[:-1])} and {key[-1]}" in text
    assert "item_01 to item_40 are one 40-item instrument, each item answered from 1 to 5" in text


def test_the_survey_card_asks_only_what_the_drivers_commands_can_record():
    """DRV-INT V2 (the verifier, round 2): no Router question, proposals card or control posts
    ``set_scales`` (the engine's own account, ``catalog.py``), and the driver composes no option
    that records it, so the card asks about age and having sought support and names no scale
    scored from the items: which columns enter beside age is the analyst's decision."""
    import inspect

    from research.swarm import options as opts
    from turbotab.core.reference.catalog import NOT_ASKED

    assert "No Router question, proposals card or control posts `set_scales`" in NOT_ASKED["scales"]
    assert "set_scales" not in inspect.getsource(opts)
    card = committed("survey_support_estimate")["card"]
    assert card["question"] == "Is age associated with having sought support?"
    for text in (card["question"], *card["documentation"]):
        assert "scale" not in text.lower(), text


def test_the_survey_key_is_refused_where_it_disagrees_with_the_codebook():
    """The response range from survey_instrument.md's own table ("Response levels | 1, 2, 3, 4,
    5"), the key from make_fixtures.py: a reverse-coded item the instrument does not hold, or a
    range an answer lies outside, is refused."""
    from turbotab.sample_data.make_fixtures import REVERSE_CODED
    doc = (I.SAMPLE / "survey_instrument.md").read_text()
    row = next(line for line in doc.splitlines() if line.startswith("| Response levels"))
    levels = [int(n) for n in re.findall(r"\d", row.split("|")[2])]
    assert I.SURVEY_RESPONSES == (min(levels), max(levels)) == (1, 5)
    frame = pd.read_csv(I.SAMPLE / "survey_instrument.csv")
    assert len(I.survey_key(REVERSE_CODED, frame, *I.SURVEY_RESPONSES)) == 1
    with pytest.raises(SystemExit, match="not the instrument's"):
        I.survey_key((*REVERSE_CODED, "item_41"), frame, *I.SURVEY_RESPONSES)
    with pytest.raises(SystemExit, match="answered outside 1 to 4"):
        I.survey_key(REVERSE_CODED, frame, 1, 4)
