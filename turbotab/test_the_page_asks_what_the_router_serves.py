"""`DRIVE-001` — a question can be built, tested, measured and invisible.

The product owner uploaded NHANES and saw no domain question. That was **exactly
correct behavior for the code as written**: `GET /project/{id}/lens` existed, the
Router served `state_lens` through `/interview`, five packs and a reframe layer
and a priors layer sat behind it, the 313 → 7 result was measured over HTTP — and
`grep lens turbotab/web/index.html` returned nothing.

Five loops of work, unreachable by a human.

## The class, and why nothing caught it

Every question in the page had been **hand-written**, so a new question rendered
nowhere and nothing said so. And `test_guided_drive.py` states its own limit
honestly — its frontend assertions read `index.html` as text and cannot prove it
renders. That honesty is exactly how the lens got through: the limit was
declared, and then relied on.

This file closes the half that is checkable without a browser: **every question
key the Router can serve is either handled by a dedicated section or answerable
through the generic channel.** A question that is servable and unanswerable is
the same dead end in a new place, which is what the lens was.

## What it still cannot prove

That a card is *visible*. Reading `index.html` as text proves a handler exists,
not that a human sees it. So this is a necessary check and not a sufficient one,
and it is worth saying rather than implying: the remaining gap needs a browser,
and the honest form of that is the driver.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import bulk as B, engine, missingness as MISS           # noqa: E402
from turbotab import packs as P, repeats as R                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "turbotab" / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _js_array(name: str) -> list:
    """One exported JS array of string literals, read from the page."""
    text = _page()
    start = text.index(f"var {name} = [")
    end = text.index("];", start)
    return re.findall(r'"([^"]+)"', text[start:end])


def _js_array_at_import(name: str) -> list:
    """`_js_array` before the module body has finished — same reader, named so
    the import-time use is obviously deliberate."""
    return _js_array(name)


def _js_object_keys(name: str) -> list:
    text = _page()
    start = text.index(f"var {name} = {{")
    depth, i = 0, text.index("{", start)
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return re.findall(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*:", text[start:i],
                      re.MULTILINE)


# Keys whose rendering belongs to a section this page has always had, matched by
# prefix because they are per-column or per-finding and their count is data.
#
# READ OUT OF THE PAGE, not restated here. This tuple used to be a second copy,
# and `renderAsked` filtered on the exact-key list alone — so every `repair::`
# question was rendered twice, once as its finding card and once as a generic
# card with dead buttons, while this test happily reported full coverage
# (`GUIDED-040`). A list written in one place and applied in another is the same
# silence as a capability with no row.
HANDLED_PREFIXES = tuple(_js_array_at_import("HANDLED_QUESTION_PREFIXES"))


def _every_key_the_router_can_serve() -> set:
    """Drive `plan()` over every fixture, every step, and both grain branches.

    Enumerated by RUNNING the Router rather than by reading a list of keys,
    because a list is the thing that goes stale — and a key nobody enumerated is
    exactly the failure being checked.
    """
    keys: set = set()
    fixtures = {
        "clinic_visits": "outcome",
        "metabolomics_untargeted": "responder",
        "dietary_recalls": "hba1c",
        "clinical_longitudinal": "progressed",
        "survey_instrument": "sought_support",
        "genomics_expression": "condition",
    }
    for name, target in fixtures.items():
        df = pd.read_csv(DATA / f"{name}.csv")
        ranked = engine.rank_findings(engine.diagnose(df, target=target), None)
        rows = MISS.survey(df, target)
        columns = [r["column"] for r in rows]
        groups = [g.to_dict() for g in B.group_columns(rows)]
        block = P.likert_block(df)
        reading = R.read(df, None)
        for step in router.STEPS:
            for answered in ([], ["state_lens", "choose_target", "state_grain",
                                  "state_repeat_kind", "state_unit_of_analysis",
                                  "choose_models", "choose_preparation_mode"]):
                for repeats in (None,
                                {"reading": R.REPEATS, "sentence": reading["sentence"],
                                 "confidence": "medium", "kind": R.REPEATS,
                                 "unit": R.UNIT_PERSON,
                                 "menu": R.menu(R.REPEATS)},
                                {"reading": R.TIME_POINTS, "sentence": "…",
                                 "confidence": "high", "kind": R.TIME_POINTS,
                                 "unit": R.UNIT_RECORD, "menu": R.menu(R.TIME_POINTS)}):
                    plan = router.plan(
                        ranked, target=target, detection=None, step=step,
                        deferred={}, answered=answered, recommendations=[],
                        signals=None, missing_columns=columns,
                        lens_block=block, repeats=repeats,
                        missingness_groups=groups)
                    router.audit(plan)
                    for q in plan:
                        if q.mode == "push":
                            keys.add(q.key)
    return keys


def test_the_page_renders_every_question_the_router_can_serve():
    """The check `DRIVE-001` says was missing.

    A key in neither list is a question the interview asks and the interface
    cannot answer — which is what the lens was for five loops.
    """
    handled = set(_js_array("HANDLED_QUESTION_KEYS"))
    answerable = set(_js_object_keys("ANSWERABLE"))
    served = _every_key_the_router_can_serve()
    assert served, "the enumeration found nothing; the driver is wrong"

    orphans = sorted(
        k for k in served
        if k not in handled and k not in answerable
        and not k.startswith(HANDLED_PREFIXES))
    assert not orphans, (
        "the Router can serve these and the page can neither render nor answer "
        "them:\n  " + "\n  ".join(orphans)
        + "\n\nAdd the key to ANSWERABLE in turbotab/web/index.html with the "
          "decision it records, or to HANDLED_QUESTION_KEYS if a dedicated "
          "section already asks it. A question the interview asks and the "
          "interface cannot answer is DRIVE-001.")


def test_the_lens_is_the_case_this_check_was_written_for():
    """Named specifically, because the general check would pass again the day
    somebody removed the lens from the plan."""
    served = _every_key_the_router_can_serve()
    assert "state_lens" in served, "the Router no longer serves the lens"
    assert "state_lens" in set(_js_object_keys("ANSWERABLE"))
    page = _page()
    assert "set_lens" in page, "the page cannot record a lens answer"


def test_the_lens_options_carry_values_and_not_only_labels():
    """The other half of why it was unanswerable.

    The lens's labels are prose — *"Metabolomics or proteomics"* — and its values
    are keys. A page rendering labels had nothing to submit, so `option_values`
    travels beside `options` and the page reads it.
    """
    plan = router.plan([], target=None, detection=None, step="data",
                       deferred={}, answered=[], recommendations=[],
                       signals=None, missing_columns=[])
    lens = next(q.to_dict() for q in plan if q.key == "state_lens")
    assert lens["multi_select"] is True
    assert lens["option_values"][:2] == ["metabolomics", "genomics"]
    assert lens["options"][0] == "Metabolomics or proteomics"
    assert len(lens["options"]) == len(lens["option_values"])
    assert "other" in lens["option_values"], (
        "'Something else, or not sure' must be reachable; it is first-class")

    page = _page()
    assert "data-answer-value=" in page, "the page submits labels, not values"
    # THIS LINE USED TO READ
    #     assert "option_values" in page or "q.options" in page
    # and it passed for five loops on the wrong half of the disjunction. The
    # page said `q.options` twice, never read `option_values`, and posted the
    # label — `GUIDED-037`, the interview unable to start at question 1.
    #
    # `FEATURE_PARITY.md`: *a substring of a message is a wildcard wearing an
    # assertion's clothes.* A grep cannot tell a page that READS a field from
    # one that merely names it, so the real check is behavioral and lives in
    # `test_answering_the_lens_changes_the_recorded_lens.py`, which runs the
    # page's own controller and reads the record back. What stays here is the
    # narrow structural half a text search CAN carry.
    assert "q.option_values" in page, (
        "the page never reads the values array, so every option submits its "
        "prose label — see test_answering_the_lens_changes_the_recorded_lens")


@pytest.mark.parametrize("key", [
    "state_repeat_kind", "state_unit_of_analysis", "state_aggregation",
    "state_temporal_prediction", "state_reverse_coding",
])
def test_every_generic_question_can_be_answered(key):
    """The chain and the survey question ride the same channel, so they are
    subject to the same check rather than to a second one."""
    answerable = _js_object_keys("ANSWERABLE")
    assert key in answerable


def test_a_generic_question_states_its_effect_before_it_is_taken():
    """`DRIVE-003`. Every kind the generic channel can submit has an effect
    sentence, so no control reaches a user saying only its own name."""
    page = _page()
    effects = _js_object_keys("EFFECTS")
    start = page.index("var ANSWERABLE = {")
    end = page.index("};", start)
    kinds = re.findall(r'kind:\s*"([a-z_]+)"', page[start:end])
    assert kinds, "the answer map has no kinds"
    missing = sorted(k for k in kinds if k not in effects)
    assert not missing, (
        f"these actions can be taken and cannot state their effect: {missing}. "
        f"A control whose effect cannot be stated in one sentence should not "
        f"exist.")


def test_every_action_acknowledges_itself_after_it_is_taken():
    """`DRIVE-004`, the third moment.

    An answered question leaves the Router's plan entirely, so the card VANISHES.
    A control that disappears when pressed has acknowledged nothing — the user is
    left inferring from an absence, which is the one thing §09 reserves green for.
    """
    page = _page()
    ack = _js_object_keys("ACK_LABEL")
    start = page.index("var ANSWERABLE = {")
    kinds = set(re.findall(r'kind:\s*"([a-z_]+)"', page[start:page.index("};", start)]))
    missing = sorted(kinds - set(ack))
    assert not missing, (
        f"these actions can be taken and say nothing afterward: {missing}. The "
        f"card leaves the plan when answered, so with no acknowledgment row it "
        f"simply disappears.")


def test_the_acknowledgment_quotes_the_record_rather_than_composing_a_sentence():
    """Why it is a quotation and not a past-tense rewrite.

    The acknowledgment and the transcript are then the same string BY
    CONSTRUCTION, so the interface cannot report an effect the record does not
    carry. `EFFECTS` promises, this reports, and a promise the server did not keep
    surfaces as a visible disagreement rather than as a reassuring sentence the
    page made up.
    """
    page = _page()
    body = page[page.index("function ackRows()"):]
    body = body[:body.index("\n  }")]
    assert "esc(d.text)" in body, (
        "the acknowledgment composes its own sentence; it must quote the "
        "decision the server recorded")
    assert "EFFECTS" not in body and "effectOf" not in body, (
        "the acknowledgment is reading the promise back to the user instead of "
        "the record, so an unkept promise would read as kept")


def test_the_quoted_record_is_never_empty(tmp_path):
    """The load-bearing half, over HTTP: a row quoting an empty string is a
    collapsed row that says nothing, which is the vanishing card with extra
    steps."""
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    with open(DATA / "clinical_longitudinal.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("clinical_longitudinal.csv", fh, "text/csv")}).json()["id"]

    def decide(project, kind, payload):
        r = client.post(f"/project/{project}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text)

    # Two drives, because the chain REFUSES the union of these answers and it is
    # right to: `set_temporal_prediction` does not apply once the rows are
    # recorded as repeated measurements of one quantity, and `set_aggregation`
    # does not apply once they are recorded as time points. The branches are
    # mutually exclusive by construction, so a single drive cannot reach both
    # sentences and a test that tried would be testing a state the app forbids.
    decide(pid, "set_target", {"column": "progressed"})
    decide(pid, "set_lens", {"lens": ["clinical"]})
    decide(pid, "set_grain", {"answer": "people_repeat", "group_col": "subject_id"})
    decide(pid, "set_repeat_kind", {"kind": "repeats"})
    decide(pid, "set_unit_of_analysis", {"unit": "person"})
    decide(pid, "set_aggregation", {"method": "mean"})

    with open(DATA / "clinical_longitudinal.csv", "rb") as fh:
        pid2 = client.post("/project", files={
            "file": ("clinical_longitudinal.csv", fh, "text/csv")}).json()["id"]
    decide(pid2, "set_target", {"column": "progressed"})
    decide(pid2, "set_grain", {"answer": "people_repeat", "group_col": "subject_id"})
    decide(pid2, "set_repeat_kind", {"kind": "time_points"})
    decide(pid2, "set_unit_of_analysis", {"unit": "record"})
    decide(pid2, "set_temporal_prediction", {"temporal": True})

    # A third, for the two kinds neither longitudinal branch reaches: reverse
    # coding belongs to a survey and `set_selection` to the Features step.
    with open(DATA / "survey_instrument.csv", "rb") as fh:
        pid3 = client.post("/project", files={
            "file": ("survey_instrument.csv", fh, "text/csv")}).json()["id"]
    decide(pid3, "set_target", {"column": "sought_support"})
    decide(pid3, "set_reverse_coding", {"columns": ["item_03"]})
    decide(pid3, "set_selection", {})          # "every column goes to the models"

    # A fourth, for the one kind no upright table can reach: question 1.5 fires
    # only on a feature-major assay export, so the drive that exercises it has
    # to be driven on one. Built by turning a shipped fixture around rather than
    # written by hand.
    from turbotab.test_a_transposed_assay_table_is_turned_around_before_diagnosis \
        import _transposed_bytes
    pid4 = client.post("/project", files={
        "file": ("t.csv", _transposed_bytes(), "text/csv")}).json()["id"]
    decide(pid4, "set_lens", {"lens": ["metabolomics"]})
    decide(pid4, "set_orientation", {"answer": "rows_are_features"})

    labeled = set(_js_object_keys("ACK_LABEL"))
    seen = set()
    for project in (pid, pid2, pid3, pid4):
        for d in client.get(f"/project/{project}").json()["decisions"]:
            if d["kind"] not in labeled:
                continue
            seen.add(d["kind"])
            assert d["text"].strip(), f"{d['kind']} recorded an empty sentence"
            # A statement of fact, not a restatement of the control's name.
            assert len(d["text"]) > 25, f"{d['kind']}: {d['text']!r}"
    assert seen == labeled, (
        f"unexercised: {sorted(labeled - seen)}. Every kind the page can label must have a sentence a drive actually produced.")


# ── one thing reading the page as text CAN prove ─────────────────────────────

def test_the_pages_javascript_parses():
    """`test_guided_drive.py` says its frontend assertions read `index.html` as
    text and cannot prove it renders. True — and this is the one thing that
    reading it can prove, which is worth having: a syntax error kills the ENTIRE
    controller, so every question goes invisible at once rather than one at a
    time.

    Skipped where no JS engine is installed, because the claim is about the file
    and a machine without an engine cannot check it. `test_engine_is_headless`
    is the precedent — a check that cannot run says so rather than passing.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        pytest.skip("no JS engine on this machine")

    page = _page()
    script = page[page.index("<script>") + len("<script>"):page.rindex("</script>")]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        out = subprocess.run([node, "--check", path], capture_output=True,
                             text=True, timeout=60)
    finally:
        os.unlink(path)
    assert out.returncode == 0, (
        "the page's JavaScript does not parse, so nothing in the interview "
        "renders:\n" + (out.stderr or out.stdout)[-1200:])
