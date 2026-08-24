"""`sibling-of: AUDIT-025` — an EDA nudge must name a page that has the control.

`AUDIT-025` is the Theory Reference telling the user that *"The Feature Selection page
offers VIF-based filtering as one of its selection methods"* when no such method exists.
That sentence lives in `pages/11_Theory_Reference.py`, which the `classic` chunk does not
own, so the row itself is blocked. `AGENT_ONBOARD.md` §08 check 5 asks what the same lens
finds one surface over, and it finds this: `ACTION_NEXT_STEPS['plausibility_check']` in
`pages/02_EDA.py` read

    "Review flagged implausible values. Apply target trimming or filter rows in
     Upload & Audit."

`pages/01_Upload_and_Audit.py` contains **zero** occurrences of `trim` and **zero** of
`plausib`. Neither control is there. Target trimming is `pages/06_Train_and_Compare.py`
("Enable target trimming before split", run before the split); plausibility filtering is
`pages/05_Preprocess.py`, inside the per-model block that `:580` skips while Smart
Defaults is selected. Same governing-rule failure as `AUDIT-025`, different surface.

**THIS IS A COMPOSITION TEST, NOT A PAGE DRIVE.** It reads the `ACTION_NEXT_STEPS` literal
out of the shipped `pages/02_EDA.py` with `ast` — the real source, not a transcribed copy
(`tests/test_eda_ledger_bridge.py` reproduces the page's mapping in the test file, which
this deliberately does not do). But the string is rendered only after a user opens Deep
Dive Diagnostics and runs the plausibility action, and **that click path is not driven
here**, so this file certifies what the server would compose and not what a person saw
(`AGENT_ONBOARD.md` §07 trap 6). The page-drive version is the honest next step.

`GUIDED-045`: every absence assertion below is preceded by a positive control — the dict
parsed non-empty, the entry exists, and the control being sought is present on the page
the corrected sentence names.
"""
from __future__ import annotations

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EDA_PAGE = os.path.join(ROOT, "pages", "02_EDA.py")
UPLOAD_PAGE = os.path.join(ROOT, "pages", "01_Upload_and_Audit.py")
PREPROCESS_PAGE = os.path.join(ROOT, "pages", "05_Preprocess.py")
TRAIN_PAGE = os.path.join(ROOT, "pages", "06_Train_and_Compare.py")


def _source(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _action_next_steps():
    """Pull the real ACTION_NEXT_STEPS literal out of the shipped page."""
    tree = ast.parse(_source(EDA_PAGE))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "ACTION_NEXT_STEPS":
                    return ast.literal_eval(node.value)
    raise AssertionError("ACTION_NEXT_STEPS is no longer a module-level literal in "
                         "pages/02_EDA.py — this test can no longer read the real dict")


def test_the_controls_the_corrected_nudge_names_are_where_it_says_they_are():
    """Positive control for the whole file: the destinations are real."""
    train = _source(TRAIN_PAGE).lower()
    assert "enable target trimming before split" in train, (
        "target trimming is not a control on pages/06_Train_and_Compare.py, so the "
        "corrected nudge names a page that does not have it"
    )
    preprocess = _source(PREPROCESS_PAGE).lower()
    assert "plausibility_gating" in preprocess, (
        "plausibility filtering is not a control on pages/05_Preprocess.py, so the "
        "corrected nudge names a page that does not have it"
    )


def test_upload_and_audit_has_neither_control_which_is_this_rows_warrant():
    """The premise the correction rests on, asserted rather than remembered."""
    upload = _source(UPLOAD_PAGE).lower()
    assert upload.strip(), "pages/01_Upload_and_Audit.py read empty — nothing was swept"
    assert "trim" not in upload, (
        "pages/01_Upload_and_Audit.py now mentions trimming; if it has gained a "
        "target-trimming control, the nudge in pages/02_EDA.py should be re-read "
        "rather than left saying the control is not there"
    )
    assert "plausib" not in upload, (
        "pages/01_Upload_and_Audit.py now mentions plausibility; the nudge in "
        "pages/02_EDA.py asserts the control is not there and must be re-read"
    )


def test_no_eda_nudge_sends_implausible_values_to_upload_and_audit():
    """The corrected claim itself."""
    nudges = _action_next_steps()
    # GUIDED-045 positive control.
    assert nudges, "ACTION_NEXT_STEPS parsed empty — nothing was swept"
    assert "plausibility_check" in nudges, (
        f"the entry under test is gone; entries present: {sorted(nudges)}"
    )
    text = nudges["plausibility_check"]

    assert not re.search(r"trimming[^.]*Upload & Audit", text, re.I), (
        "pages/02_EDA.py sends the user to Upload & Audit to apply target trimming. "
        "That page has no such control (zero occurrences of 'trim'); trimming is on "
        "Train & Compare. Same defect as AUDIT-025 — a page naming a capability on a "
        "page that does not have it."
    )
    assert not re.search(r"filter rows in Upload & Audit", text, re.I), (
        "pages/02_EDA.py sends the user to Upload & Audit to filter rows on "
        "implausible values. That page has no such control (zero occurrences of "
        "'plausib'); plausibility filtering is on Preprocess."
    )
    assert "Train & Compare" in text, (
        "the wrong destination was removed without naming the right one — "
        "AUDIT-028's model is a corrected claim, not a shorter one"
    )
    assert "Preprocess" in text, (
        "the plausibility-filter destination is not named, so the user is told "
        "where the control is not and never where it is"
    )


def test_the_two_nudges_about_trimming_agree_with_each_other():
    """`influence_diagnostics` already said "the Train page"; they must not conflict."""
    nudges = _action_next_steps()
    trimming = {k: v for k, v in nudges.items() if "trim" in v.lower()}
    assert trimming, "no nudge mentions trimming — nothing was swept"
    for action_id, text in trimming.items():
        assert not re.search(r"trimming[^.]*Upload & Audit", text, re.I), (
            f"ACTION_NEXT_STEPS['{action_id}'] sends trimming to Upload & Audit while "
            f"another entry sends it to the Train page; one of them is false and the "
            f"user is told both"
        )
