"""The product designer's NHANES drive, as tests.

Eight findings came out of one hour with the Guided door on a real NHANES
extract (GUIDED-001 … GUIDED-008), plus the §09 reskin of the built steps. Each
is pinned here against the thing that was actually wrong, so the fix cannot
regress into the shape the drive found.

The frontend assertions read `web/index.html` rather than driving a browser.
That is a real limit and it is the honest one: it can prove the treatment is
present and exclusive, and it cannot prove it renders. What can be checked
end-to-end goes over HTTP against the real engine.

Run:  turbotab/.venv/Scripts/python -m pytest turbotab/test_guided_drive.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from turbotab import draft, engine
from turbotab.api import app

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = (REPO_ROOT / "turbotab" / "web" / "index.html").read_text(encoding="utf-8")
# Everything after the prototype stylesheet: the markup, the added styles, and
# the controller. The prototype block itself is asserted verbatim elsewhere.
BODY = PAGE[PAGE.index("</style>"):]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def nhanes_like(n: int = 160) -> pd.DataFrame:
    """A table with each of the drive's artifacts in it."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "seqn": np.arange(1, n + 1),
        "age": rng.integers(20, 80, n).astype(float),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        "bp_di": np.round(rng.normal(78, 9, n), 1),
        "glucose": np.round(rng.normal(101, 18, n), 1),
        # True/False text with blanks — the GUIDED-001 column.
        "meds_chol": rng.choice(["True", "False", ""], n, p=[0.4, 0.35, 0.25]),
        "diabetes": rng.integers(0, 2, n),
    })
    df.loc[3, "bp_di"] = 1.5e-15        # not a patient: an entry error
    df.loc[11, "bp_di"] = 301.0
    df["leaky_score"] = df["diabetes"] * 5.0 + rng.normal(0, 0.02, n)
    df.loc[rng.choice(n, 60, replace=False), "bmi"] = np.nan
    return df


@pytest.fixture(scope="module")
def raw() -> bytes:
    return nhanes_like().to_csv(index=False).encode()


@pytest.fixture()
def project(client, raw) -> str:
    pid = client.post("/project",
                      files={"file": ("nhanes.csv", raw, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "diabetes"}})
    return pid


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-001 — the binary reading outranks numeric coercion
# ═══════════════════════════════════════════════════════════════════════════

def test_a_true_false_column_is_read_as_binary_not_coerced_to_numbers(client, project):
    body = client.get(f"/project/{project}/findings").json()
    by_col = {}
    for f in body["findings"]:
        for c in f.get("affected_columns") or []:
            by_col.setdefault(c, []).append(f)

    proposals = by_col.get("meds_chol", [])
    kinds = {f["fix_kind"] for f in proposals}
    assert "read_as_binary" in kinds, (
        "the doctor still reaches for numeric coercion on a True/False column")
    assert "coerce_numeric" not in kinds, (
        "both proposals are shown for one column, which makes the user settle "
        "the engine's own disagreement")


def test_the_binary_proposal_says_which_level_is_one(client, project):
    body = client.get(f"/project/{project}/findings").json()
    f = next(x for x in body["findings"] if x["fix_kind"] == "read_as_binary")
    assert "True = 1" in f["fix_label"] and "False = 0" in f["fix_label"]
    assert "binary" in f["title"].lower()


def test_the_binary_repair_leaves_the_blanks_blank(client, project):
    """The missingness question is separate and belongs to Preprocess.

    Filling it here would answer it by accident — the drive's whole point about
    this column is that its blanks are the interesting part.
    """
    body = client.get(f"/project/{project}/findings").json()
    fid = next(x["id"] for x in body["findings"] if x["fix_kind"] == "read_as_binary")
    before = client.get(f"/project/{project}").json()
    pv = client.get(f"/project/{project}/finding/{fid}/preview").json()
    assert pv["applicable"] and pv["changed_cells"] > 0, (
        "the preview reports no visible change for True -> 1")
    applied = client.post(f"/project/{project}/decision",
                          json={"kind": "apply", "subject": fid}).json()
    assert fid in applied["applied_fixes"]
    assert applied["n_rows"] == before["n_rows"]


def test_an_unknown_binary_pair_does_not_pretend_to_know_the_positive_level():
    from ml.binary_text import binary_text_finding
    s = pd.Series(["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha"])
    f = binary_text_finding("arm", s)
    assert f is not None
    assert f.confidence == "medium", "an arbitrary 0/1 assignment is not high confidence"
    assert "your call" in f.why_it_matters


def test_a_three_level_column_is_not_called_binary():
    from ml.binary_text import read_as_binary_plan
    s = pd.Series(["yes", "no", "unknown", "yes", "no", "unknown", "yes"])
    assert read_as_binary_plan(s) is None


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-002 — missingness names its features, routes by dtype, states timing
# ═══════════════════════════════════════════════════════════════════════════

def test_missingness_cards_name_the_features(client, project):
    cards = client.get(f"/project/{project}/evidence/missingness").json()["cards"]
    assert cards, "the drive's table has two high-missingness columns"
    for c in cards:
        assert c["column"], "a card with no column is the '2 features' bug again"
        assert f"`{c['column']}`" in c["question"], (
            "the question does not name its own column in mono")
        assert c["n_missing"] > 0 and 0 < c["share"] <= 1


def test_missingness_routes_by_dtype(client, project):
    cards = {c["column"]: c for c in
             client.get(f"/project/{project}/evidence/missingness").json()["cards"]}
    assert cards["bmi"]["dtype_route"] == "numeric"
    assert cards["meds_chol"]["dtype_route"] == "binary"

    numeric_keys = {o["key"] for o in cards["bmi"]["options"]}
    assert {"impute_median", "impute_mean"} <= numeric_keys
    binary_keys = {o["key"] for o in cards["meds_chol"]["options"]}
    assert "indicator" in binary_keys, (
        "a binary column must be asked whether its missingness is informative")


def test_a_categorical_column_is_offered_an_explicit_missing_level():
    from ml.missingness_plan import missingness_cards
    df = pd.DataFrame({"site": ["a", "b", None, None, "c", None, "b", "a", None, "c"]})
    card = missingness_cards(df)[0]
    assert card["dtype_route"] == "categorical"
    keys = {o["key"] for o in card["options"]}
    # The card speaks the RECORD's vocabulary since `GUIDED-090`: one table
    # decides what both doors offer, so the option key IS the declaration key.
    assert "explicit_category" in keys and "impute_mode" in keys
    assert "leave" in keys, (
        "the categorical branch permits leaving the blanks and the card does "
        "not offer it — judgment renders as ranking, never as absence")


def test_every_option_states_when_it_happens(client, project):
    """The action-timing ruling, applied everywhere.

    Structural repairs execute now; statistical transforms are recorded and
    fitted inside the per-model pipeline on training folds. Every option carries
    which it is, and the decision sentence says so in methods prose.
    """
    # `TIMING_MIXED` is the third, added at `DRIVE-008` — two options are
    # genuinely both, the indicator landing now and the fill running in the
    # fold. It is admitted here rather than the option being forced into one of
    # the two, because "the frontend cannot invent a third timing" is a rule
    # about the INTERFACE inventing one; the engine naming a compound it really
    # performs is the clause being stated more precisely, not less.
    from ml.missingness_plan import (TIMING_IMMEDIATE, TIMING_IN_PIPELINE,
                                     TIMING_MIXED, TIMING_RECORDED_ONLY)
    cards = client.get(f"/project/{project}/evidence/missingness").json()["cards"]
    for c in cards:
        for o in c["options"]:
            assert o["timing"] in (TIMING_IMMEDIATE, TIMING_IN_PIPELINE,
                                   TIMING_MIXED, TIMING_RECORDED_ONLY)
            assert o["timing_prose"]
            assert o["decision_sentence"].endswith("."), (
                "a decision sentence that is not a sentence cannot appear in a "
                "methods section")


def test_the_timing_is_methods_prose_not_a_ui_lecture(client, project):
    """"will be imputed with the training-fold median" — never hidden, never a
    lecture about how the software works."""
    cards = client.get(f"/project/{project}/evidence/missingness").json()["cards"]
    sentences = [o["decision_sentence"] for c in cards for o in c["options"]]
    # "training fold", not "training-fold": since `GUIDED-090` the card quotes
    # the RECORD's sentence rather than writing its own, and the record's
    # phrasing is "within each training fold". The claim is about the timing
    # being in the prose, which is unchanged; the hyphen was never the claim.
    assert any("training fold" in s for s in sentences), (
        "no sentence states that the statistic is fitted on training folds")
    banned = ("pipeline", "the app ", "the tool ", "click", "button", "the UI")
    for s in sentences:
        low = s.lower()
        assert not any(b in low for b in banned), (
            f"the decision sentence explains the software rather than the "
            f"method: {s!r}")


def test_a_numeric_option_can_show_its_before_and_after(client, project):
    pv = client.get(f"/project/{project}/evidence/imputation/bmi"
                    f"?strategy=impute_median").json()
    assert pv["n_filled"] > 0
    assert pv["before"]["n"] < pv["after"]["n"]
    assert pv["after"]["std"] < pv["before"]["std"], (
        "imputing at the median must shrink the spread; the preview is not "
        "describing the transform it names")
    assert "training fold" in pv["note"]


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-003 / GUIDED-005 — the evidence is on the table
# ═══════════════════════════════════════════════════════════════════════════

def test_the_plausibility_card_shows_the_entries_it_counted(client, project):
    rep = client.get(f"/project/{project}/evidence/plausibility").json()
    blocks = rep["impossible"] + rep["improbable"]
    assert blocks, "the fixture has two impossible pressures in it"
    for b in blocks:
        assert b["entries"], f"{b['column']} reports a count with no entries"
        for e in b["entries"]:
            assert e["row"] is not None and e["value"] is not None
            assert e["side"] in ("below", "above")


def test_a_shape_claim_can_fetch_the_shape(client, project):
    h = client.get(f"/project/{project}/evidence/histogram/bmi").json()
    assert sum(h["counts"]) == h["n"] > 0
    assert len(h["edges"]) == len(h["counts"]) + 1
    assert h["skew"] is not None


def test_a_constant_column_refuses_a_histogram_rather_than_drawing_one(client, raw):
    df = pd.DataFrame({"flat": [3.0] * 40, "y": list(range(40))})
    pid = client.post("/project", files={"file": ("f.csv", df.to_csv(index=False).encode(),
                                                 "text/csv")}).json()["id"]
    r = client.get(f"/project/{pid}/evidence/histogram/flat")
    assert r.status_code == 404
    assert "constant" in r.json()["detail"]


def test_the_gallery_and_the_matrix_are_gated_on_feature_count():
    from ml.card_evidence import (MAX_FEATURES_FOR_GALLERY, correlation_matrix,
                                  histogram_gallery)
    rng = np.random.default_rng(1)
    wide = pd.DataFrame(rng.normal(size=(60, MAX_FEATURES_FOR_GALLERY + 5)))
    wide.columns = [f"f{i}" for i in range(wide.shape[1])]

    gallery = histogram_gallery(wide)
    matrix = correlation_matrix(wide)
    assert gallery["available"] is False and matrix["available"] is False
    assert str(MAX_FEATURES_FOR_GALLERY) in gallery["reason"]
    assert gallery["plots"] == [] and matrix["matrix"] == []

    narrow = wide.iloc[:, :8]
    assert histogram_gallery(narrow)["available"] is True
    assert correlation_matrix(narrow)["available"] is True


def test_the_matrix_reports_the_pair_the_blocker_is_about(client, project):
    corr = client.get(f"/project/{project}/evidence/correlations").json()
    pairs = {frozenset((p["a"], p["b"])) for p in corr["strong_pairs"]}
    assert frozenset(("diabetes", "leaky_score")) in pairs


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-004 — impossible is not a kind of outlier
# ═══════════════════════════════════════════════════════════════════════════

def test_the_impossibility_band_contains_the_improbability_band():
    """The tiers must nest. An impossibility band inside the improbability one
    would call ordinary values impossible and propose deleting them.

    **`MISC-018` renamed this test with what it checks.** It was
    `..._contains_the_reference_interval`, and the p01/p99 pair it compares
    against is not a reference interval — the central 98%, where CLSI EP28-A3c
    defines the interval as the central 95%.
    """
    from ml.physiology_reference import (impossibility_contains_improbability,
                                         load_nhanes_reference)
    ref = load_nhanes_reference()
    for key in ref["variables"]:
        assert impossibility_contains_improbability(ref, key), (
            f"{key}'s impossibility band is narrower than its improbability band")


def test_a_variable_with_no_published_band_returns_none_not_the_interval():
    from ml.physiology_reference import get_impossibility_band
    ref = {"variables": {"widget": {"unit": "u", "p01": 1, "p99": 9}}}
    assert get_impossibility_band(ref, "widget") is None, (
        "falling back to the reference interval would promote improbable values "
        "to impossible ones and propose deleting real data")


def test_a_diastolic_of_zero_is_impossible_and_a_high_one_is_improbable(client, project):
    rep = client.get(f"/project/{project}/evidence/plausibility").json()
    impossible = {b["column"]: b for b in rep["impossible"]}
    assert "bp_di" in impossible
    rows = {e["row"] for e in impossible["bp_di"]["entries"]}
    assert 3 in rows, "the ~0 diastolic pressure was not called impossible"
    assert impossible["bp_di"]["tier"] == "impossible"

    improbable = {b["column"]: b for b in rep["improbable"]}
    for b in improbable.values():
        assert b["tier"] == "improbable"
        assert not b.get("whole_column_suspect")


def test_the_two_tiers_do_not_double_count_the_same_entry(client, project):
    rep = client.get(f"/project/{project}/evidence/plausibility").json()
    for col in {b["column"] for b in rep["impossible"]}:
        bad = next(b for b in rep["impossible"] if b["column"] == col)
        mild = next((b for b in rep["improbable"] if b["column"] == col), None)
        if mild is None:
            continue
        assert not ({e["row"] for e in bad["entries"]} &
                    {e["row"] for e in mild["entries"]})


def test_a_column_that_is_mostly_impossible_is_read_as_a_unit_problem():
    """The predicate escalates on evidence, not on how much a repair would cost.

    A glucose column recorded in mmol/L reads as entirely impossible against
    mg/dL bounds. It is not: multiplying by 18 — a unit glucose is actually
    recorded in — puts the whole column inside its reference interval, so the
    reading is wrong and no entry is.

    (`hba1c_proxy`, which used to be this test's fixture, no longer reaches the
    predicate at all: `match_variable_key` matches exact keys and declared
    aliases only, so a column merely named like a variable now gets no bounds.
    That is asserted in
    `tests/test_doubt_the_reading_not_the_data.py::test_an_unknown_suffix_yields_silence_rather_than_inherited_bounds`.)
    """
    from ml.card_evidence import READING_UNITS, plausibility_report
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"glucose": rng.normal(5.4, 0.6, 80)})
    rep = plausibility_report(df)
    block = next(b for b in rep["impossible"] if b["column"] == "glucose")

    assert block["reading"] == READING_UNITS
    assert block["whole_column_suspect"] is True
    assert any(e.startswith("rescued-by:") for e in block["reading_evidence"])
    assert rep["n_impossible"] == 0, (
        "a suspect column inflated the count of entries that earn a repair")
    assert block["entries"], "the values are still shown; silence is the other lie"


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-006 — no live-styled dead controls
# ═══════════════════════════════════════════════════════════════════════════

def test_every_pull_chip_says_whether_it_runs(client, project):
    plan = client.get(f"/project/{project}/interview?step=explore").json()
    pulls = [q for q in plan["questions"] if q["mode"] == "pull"]
    assert pulls, "the palette is empty; the audit would be vacuous"
    for q in pulls:
        assert "built" in q, f"{q['key']} does not say whether it is wired"
        if q["built"]:
            assert q["endpoint"], f"{q['key']} claims to be built with no endpoint"
        else:
            assert q["not_built_reason"], (
                f"{q['key']} is dark with no reason on it")


def test_every_endpoint_a_chip_claims_actually_answers(client, project):
    from turbotab.api import PULL_CAPABILITIES
    for key, cap in PULL_CAPABILITIES.items():
        if not cap.get("built"):
            continue
        r = client.get(f"/project/{project}/evidence/{cap['endpoint']}")
        assert r.status_code == 200, (
            f"{key} is registered as built and its endpoint returns "
            f"{r.status_code}")


def test_the_page_disables_the_chips_it_marks_not_built():
    assert "notbuilt" in BODY
    assert "b.disabled = true" in BODY, (
        "a not-built chip is styled dark but still clickable")
    assert 'aria-disabled' in BODY


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-007 — both panels expand; nothing internal renders
# ═══════════════════════════════════════════════════════════════════════════

def test_both_docks_are_expandable():
    for toggle, panel in (("ledgerToggle", "ledgerPanel"),
                          ("draftToggle", "draftPanel")):
        assert f'id="{toggle}"' in BODY and f'id="{panel}"' in BODY
        assert f'aria-controls="{panel}"' in BODY


def test_no_internal_placeholder_string_renders():
    """A coach item printed the literal engineering to-do in production UI.

    **The positive control is the load-bearing half** (`GUIDED-045`). Every
    assertion below is of the form *"this string does not appear"*, and an
    absence assertion over a file gets **monotonically easier as the file loses
    content**: `pageprobe.py` found this test green against a page emptied to
    `<body></body>`. A guard that passes hardest when there is nothing to guard
    is not measuring what its name says.

    So it first asserts the page is THERE. Deleting the controller now fails
    this test, which is the property an absence claim cannot otherwise have.
    """
    assert len(BODY) > 20_000, (
        f"the page is {len(BODY)} characters; every absence assertion below "
        f"would pass on an empty file, so they are checked against a page that "
        f"exists first")
    assert "askedQuestions" in BODY and "data-answer-key" in BODY, (
        "the page no longer contains the interview it is being checked for "
        "placeholders in")

    for ghost in ("not built yet", "TODO", "FIXME", "TBD", "lorem ipsum",
                  "XXX", "coming soon", "stub", "dummy"):
        assert ghost.lower() not in BODY.lower(), (
            f"the internal string {ghost!r} can reach a user")
    # `placeholder=` is a real HTML attribute and the word appears in a comment
    # about this very defect; what must not happen is the word rendering as
    # text, so the check is scoped to the markup rather than the controller.
    markup = BODY[:BODY.index("<script>")]
    assert not re.search(r">[^<]*\bplaceholder\b[^<]*<", markup, re.IGNORECASE), (
        "the word 'placeholder' renders as text")


def test_the_draft_is_prose_with_the_gaps_left_open(client, project):
    client.post(f"/project/{project}/decision",
                json={"kind": "defer", "subject": "missing__bmi",
                      "payload": {"target_step": "preprocess"}})
    d = client.get(f"/project/{project}/draft").json()

    assert d["n_sentences"] > 0 and not d["is_empty"]
    assert d["gap_marker"] == "[AUTHOR REQUIRED]"
    assert d["n_gaps"] >= 1, (
        "a draft with no authored gaps is the app writing in the user's name")
    titles = [s["title"] for s in d["sections"]]
    assert "Data preparation" in titles and "Limitations" in titles
    for s in d["sections"]:
        for item in s["sentences"]:
            assert item["text"].strip()


def test_an_empty_section_says_what_it_is_waiting_for():
    empty = draft.draft({"decisions": [], "target": None})
    assert empty["is_empty"] is True
    for s in empty["sections"]:
        assert s["waiting_for"], f"{s['title']} renders empty and silent"
        assert "not built" not in s["waiting_for"].lower()


def test_the_draft_never_states_a_reason_on_the_users_behalf(client, project):
    client.post(f"/project/{project}/decision",
                json={"kind": "dismiss", "subject": "profile_missingness_1"})
    d = client.get(f"/project/{project}/draft").json()
    dismissals = [i for s in d["sections"] for i in s["sentences"]
                  if i["kind"] == "dismiss"]
    assert dismissals, "the dismissal is missing from the draft entirely"
    for item in dismissals:
        assert item["has_gap"], (
            "the draft supplied a reason for dismissing a finding; only the "
            "author can say why it does not affect the analysis")


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-008 — a deferral names its destination
# ═══════════════════════════════════════════════════════════════════════════

def test_every_finding_carries_the_step_its_deferral_goes_to(client, project):
    body = client.get(f"/project/{project}/findings").json()
    assert body["findings"]
    for f in body["findings"]:
        assert f["defer_target"], f"{f['id']} has no deferral destination"
        assert f["defer_target_label"], f"{f['id']} has no words for its destination"


def test_a_structural_repair_and_a_profile_finding_defer_to_different_steps(client, project):
    body = client.get(f"/project/{project}/findings").json()
    structural = next(f for f in body["findings"] if f["source"] == "structure")
    profile = next(f for f in body["findings"] if f["source"] == "profile")
    assert structural["defer_target"] == "explore"
    assert profile["defer_target"] == "preprocess"


def test_a_missingness_card_names_preprocess(client, project):
    cards = client.get(f"/project/{project}/evidence/missingness").json()["cards"]
    for c in cards:
        assert c["target_step"] == "preprocess"
        assert c["target_step_label"] == "Preprocess"


def test_the_ledger_shows_the_destination_rather_than_a_placeholder():
    assert "comes back at " in BODY
    assert "Decide at " in BODY, (
        "the deferral affordance still says 'later' instead of where")


def test_the_recorded_deferral_carries_the_step_the_button_named(client, project):
    body = client.post(f"/project/{project}/decision",
                       json={"kind": "defer", "subject": "missing__bmi",
                             "payload": {"target_step": "preprocess"}}).json()
    d = next(x for x in body["decisions"] if x["kind"] == "defer")
    assert d["payload"]["target_step"] == "preprocess"


# ═══════════════════════════════════════════════════════════════════════════
# §09 — the blocker's costume, and its exclusivity
# ═══════════════════════════════════════════════════════════════════════════

def test_the_blocker_wears_the_reserved_signal_word_and_shape():
    assert "blocker-word" in BODY and "font-variant:small-caps" in BODY
    assert ">Blocker<" in BODY, "the signal word does not render"
    assert "blocker-glyph" in BODY
    # The notched square: one path, used by blockers alone.
    assert BODY.count("blocker-glyph") >= 2      # the CSS rule and the SVG
    assert "var(--stop)" in BODY


def test_the_stop_token_exists_in_both_themes():
    style = PAGE[PAGE.index("§09 QUESTION GRAMMAR"):]
    for selector in ('prefers-color-scheme: dark',
                     ':root[data-theme="light"]',
                     ':root[data-theme="dark"]'):
        block = style[style.index(selector):]
        assert "--stop:" in block[:400], (
            f"--stop is not redefined under {selector}; the viewer's toggle "
            "would not win")


def test_nothing_but_the_blocker_borrows_the_blocker_treatment():
    """Exclusivity is what makes the shape semantic. Habituation starts at the
    second exposure, so the treatment survives only if nothing else wears it.

    **Exclusivity is two claims and this asserted one** (`GUIDED-045`): that
    nothing else wears the treatment, and — silently assumed — that the blocker
    does. Only the first was checked, and it is an absence assertion, so
    deleting every `--stop` rule in the sheet made the test pass *more*
    easily. `pageprobe.py` found it green against an empty page: a stylesheet
    with no blocker treatment at all satisfies "nothing else borrows it"
    perfectly.
    """
    worn = [l for l in BODY.splitlines()
            if "var(--stop)" in l and "blocker" in l]
    assert worn, (
        "nothing in the page wears the blocker treatment, so 'nothing ELSE "
        "wears it' is true of a page with no blocker — which is the reading "
        "this test is supposed to make impossible")

    for token in ("var(--stop)", "var(--stop-tint)", "var(--stop-line)"):
        for line in BODY.splitlines():
            if token not in line or line.strip().startswith("*"):
                continue
            allowed = ("--stop", "blocker", "attest", "att-", ".bad", "ev-body",
                       "stopbtn", "on-stop")
            assert any(a in line for a in allowed), (
                f"{token} appears outside the blocker treatment: {line.strip()}")


def test_the_notched_glyph_is_defined_once():
    paths = re.findall(r'class="blocker-glyph"', BODY)
    assert len(paths) == 1, (
        "the reserved glyph is emitted from more than one place; a shape used "
        "twice is not a reserved shape")


def test_a_blocker_reaches_a_terminal_state_by_acknowledgment(client, project):
    plan = client.get(f"/project/{project}/interview?step=explore").json()
    assert plan["unresolved_blockers"], "the fixture no longer raises a blocker"
    key = plan["unresolved_blockers"][0]
    column = key.split("::")[-1]

    refused = client.post(f"/project/{project}/decision",
                          json={"kind": "acknowledge_blocker", "subject": key})
    assert refused.status_code == 400, "silence was accepted as an acknowledgment"

    client.post(f"/project/{project}/decision",
                json={"kind": "acknowledge_blocker", "subject": key,
                      "text": f"I am keeping {column} although it may leak the outcome."})
    again = client.get(f"/project/{project}/interview?step=explore").json()
    assert again["unresolved_blockers"] == [], (
        "the blocker re-fires on the same facts after acknowledgment — a flag "
        "that cannot be satisfied teaches contempt for all flags")
    assert again["acknowledgment_required"] is None


def test_a_blocker_reaches_a_terminal_state_by_resolution(client, raw):
    pid = client.post("/project",
                      files={"file": ("nhanes.csv", raw, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "diabetes"}})
    plan = client.get(f"/project/{pid}/interview?step=explore").json()
    key = plan["unresolved_blockers"][0]
    column = key.split("::")[-1]

    body = client.post(f"/project/{pid}/decision",
                       json={"kind": "resolve_blocker", "subject": key,
                             "payload": {"column": column}}).json()
    assert column not in [c["name"] for c in body["columns"]]
    again = client.get(f"/project/{pid}/interview?step=explore").json()
    assert again["unresolved_blockers"] == []


def test_the_attestation_is_object_specific(client, project):
    """A generic PROCEED habituates. The sentence names the column.

    **It is composed by the SERVER now** (`GUIDED-076`). This asserted the
    sentence appeared in `index.html`, which was true and was the defect: the
    page wrote a leakage sentence for every consequence, so a second one would
    have asked the user to type a sentence about leakage. The page renders
    `exit.typed`; the router builds it.
    """
    from ml import router

    class _Signals:
        leakage_candidate_cols = ["glucose"]
        leakage_flags = []

    served = router.blockers(_Signals(), step="explore")[0].to_dict()
    typed = [x.get("typed") for x in served["exits"] if x.get("typed")]
    assert typed and "although it may leak the outcome" in typed[0], (
        "the server no longer composes an object-specific attestation")
    assert "glucose" in typed[0], "the sentence does not name the column"

    # And the page renders what it was given rather than writing its own.
    assert "exitTyped" in BODY and "q.exits" in BODY
    assert "although it may leak the outcome" not in BODY, (
        "the page composes the leakage sentence again")
    assert "Pasting is fine" in BODY, (
        "blocking paste harms accessibility and adds nothing")


def test_the_acknowledgment_artifact_is_never_green(client, project):
    assert ".attested" in BODY
    attested = BODY[BODY.index(".attested{"):]
    head = attested[:attested.index("}")]
    assert "var(--stop)" in head and "var(--ok)" not in head, (
        "an accepted blocker is styled as a recorded decision; the review "
        "cannot then tell accepted from resolved")


# ═══════════════════════════════════════════════════════════════════════════
# §09 — FACT skips and CHOICE buttons
# ═══════════════════════════════════════════════════════════════════════════

def test_a_rendered_skip_is_muted_provenance_not_a_recorded_decision():
    assert ".skip{" in BODY and ".skip .clause" in BODY
    skip_rules = BODY[BODY.index(".skips{"):BODY.index(".choice-acts{")]
    assert "var(--ok)" not in skip_rules, (
        "a skip is styled green; green means a human recorded it, and nobody "
        "recorded this")
    assert "var(--mono)" in skip_rules, "the provenance clause is not in mono"
    assert "Ask me anyway" in BODY


def test_choice_buttons_are_outcome_labeled_and_symmetric():
    assert "Keep as is" in BODY, "the decline option is not outcome-labeled"
    assert "btn-primary" not in BODY[BODY.index("function previewHTML"):
                                     BODY.index("CONSEQUENCE — the blocker band")], (
        "the apply button is styled as a primary action against a de-emphasized "
        "decline; declining must be as easy and as dignified")
    cbtn = BODY[BODY.index(".cbtn{"):]
    assert "border:1.5px solid var(--line)" in cbtn[:cbtn.index("}")], (
        "the two outcomes do not carry the same weight")


def test_the_apply_button_names_the_operation(client, project):
    body = client.get(f"/project/{project}/findings").json()
    fid = next(x["id"] for x in body["findings"] if x["fix_kind"] == "read_as_binary")
    pv = client.get(f"/project/{project}/finding/{fid}/preview").json()
    assert pv["fix_label"] and pv["fix_label"].lower().startswith("read ")


# ═══════════════════════════════════════════════════════════════════════════
# The preview's count and its highlighting must mean the same thing
# ═══════════════════════════════════════════════════════════════════════════

def test_the_change_count_and_the_highlighted_cells_agree():
    """Found while building the binary reading, described by no finding.

    The count came from a value comparison and the highlighting from a text
    comparison, so `"1200"` -> `1200` reported eight changed cells over an
    unmarked table, and `True` -> `1` reported none over a marked one.
    """
    from ml import import_doctor

    df = pd.DataFrame({"id": range(8),
                       "dose": ["1200", "2400", "3100", "4050",
                                "5900", "6300", "7700", "8100"]})
    coerce = next(f for f in import_doctor.diagnose(df)
                  if f.fix_kind == "coerce_numeric")
    pv = engine.preview_fix(df, coerce)
    marked = sum(1 for r in pv["sample"]["rows"] for c in r["changed"] if c)
    assert pv["changed_cells"] == 0 and marked == 0
    assert any(s["key"].startswith("dtype of") for s in pv["stats"]), (
        "a type-only change must still be reported, in the statistics")

    df2 = pd.DataFrame({"id": range(8),
                        "meds": [True, False, True, False, True, True, False, True]})
    binary = next(f for f in engine.diagnose(df2) if f.fix_kind == "read_as_binary")
    pv2 = engine.preview_fix(df2, binary)
    marked2 = sum(1 for r in pv2["sample"]["rows"] for c in r["changed"] if c)
    assert pv2["changed_cells"] == 8 and marked2 == 8
