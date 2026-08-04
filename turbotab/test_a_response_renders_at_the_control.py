"""L47-C — a press answers where it was pressed. `GUIDED-167` · `173` · `161` · `162`.

The NHANES drive produced **five presses with no visible response, from five
different causes — and two of them were the app answering correctly, out of
sight.** A user cannot tell *the control is not wired* from *the app refused and
did not tell me*, and that drive produced both.

## The ruling this file is the gate for

> **A response to a press renders at the control** — adjacent to the button
> pressed, not merely somewhere in the same card.

*In the card* is what the feature receipt already did: `#featDecided` is six
levels up from the Add button and then a sibling hop, past the per-row preview
and past every other transform row. It was in the card and still off-screen.

**The answer is not to move the viewport.** `DESIGN_LANGUAGE.md` §05 —
`DRIVE-006` deleted the nudge, `test_the_page_never_moves_the_viewport.py` pins
its absence, and that test must stay green. Three reasons the placement rule is
right rather than convenient are in the page's own comment at `atControlSlot`;
the shortest is that **it has no free parameter**, which is precisely how the
middle nudge rule failed.

## The four, and what each one actually was

- **`GUIDED-167` was a hidden container before it was a placement problem.**
  `#upErr` and `#refusal` both live inside `<div class="sub" id="sub-upload">`,
  and `renderData()` runs `$("sub-upload").classList.add("is-hidden")` on **every
  render**, with `.sub.is-hidden{display:none}`. So from the first successful
  upload the app's error sink is inside a `display:none` subtree for the rest of
  the session — twenty `setErr` call sites and every `showRefusal` write. The
  canonical nodes stay (they are correct *during* upload, when that subtree is
  on screen); what is new is that the same sentence also lands at the control.
- **`GUIDED-173`** — the receipt now renders **in place of** the control, which
  is `prepColHTML`'s and `selBuildHTML`'s `answered-card` shape verbatim.
- **`GUIDED-161`** — `earmarkRow` had no receipt slot while `offerRow` two
  functions up already had one, and `deferrals()` returned early on
  `d.kind !== "defer"` so the dock's *"Anything you set aside waits here"* was
  unkept for every earmark.
- **`GUIDED-162` was a caption, not a missing handler.** There is no apply path
  for an offer key anywhere: `data-offer-key` appears exactly twice in the page,
  emitted and read, and the `apply` decision is keyed to a structural finding's
  `fix_kind` — a different mechanism, on a branch `openPanel` reaches only when
  `fix_kind` is absent. **The shelf is not shortened to make the sentence true**:
  the preview and the button stay, and the caption stops naming a control that
  is not there.

## What a subtree assertion cannot see, said here rather than found later

This drives the *pressed control's own slot*. It cannot see whether that slot is
on screen — nothing without layout can, and `pageharness.py` says so in its own
docstring. It is strictly weaker than visibility and strictly stronger than
"somewhere in the page", which is what every check before it could ask.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "sample_data"


def _driven(fixture="clinical_labs.csv", lens="clinical", target="readmitted"):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    if lens:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": [lens]}})
    if target:
        client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


def _routes(client, pid):
    """Every route one full render fetches.

    Nineteen, not four. A harness that stubs four gets a controller that throws
    on the rest and a sweep that reports the throw as a finding.
    """
    got = client.get(f"/project/{pid}").json()
    out = {f"/project/{pid}": got}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "capabilities", "features",
                 "recipes", "preprocess", "figures", "draft", "manuscript",
                 "models", "training", "instability", "explain", "sensitivity",
                 "evidence/plausibility", "evidence/missingness"):
        resp = client.get(f"/project/{pid}/{path}")
        out[f"/project/{pid}/{path}"] = resp.json() if resp.status_code == 200 else {}
    return out


_SLOT = re.compile(r'<span class="at-ctl" id="ac-([^"]+)"></span>')


def test_the_page_defines_the_slot_and_the_rule_it_serves():
    """The machinery exists and is addressable, before anything is driven."""
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    for name in ("function atControlSlot(", "function atControl(",
                 'AT_CONTROL = t.getAttribute("data-ac")'):
        assert name in page, f"{name} is gone; the placement rule has no mechanism"
    assert page.count('data-ac="') >= 1 or 'data-ac=' in page, (
        "no control identifies its own receipt slot")


@pytest.mark.parametrize("fixture,lens,target", [
    ("clinical_labs.csv", "clinical", "readmitted"),
    ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL"),
], ids=["clinical binary target", "dietary continuous target"])
def test_a_refusal_lands_beside_the_control_that_caused_it(fixture, lens, target):
    """`GUIDED-167`, driven — the class, not one instance.

    Two fixtures of different target shape (`GUIDED-097`). **The press is a
    `data-dismiss`** whose server call is made to fail, because that is a control
    whose entire error path went to `setErr` and therefore into the hidden
    subtree, and because a finding card is the one action row every fixture
    renders without further setup.

    This docstring named the feature-add control for one loop while the body
    dispatched `data-dismiss` — trap #3b, in the file whose subject is the trap
    next door. `test_a_test_that_names_a_control_presses_it.py` is the detector
    that would have caught it, and it did.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _driven(fixture, lens, target)
    routes = _routes(client, pid)
    # The server refuses this press. A refusal is exactly the case the drive
    # found invisible, and `__status` is the harness's own way to serve one.
    routes[f"POST /project/{pid}/decision"] = {
        "__status": 400, "body": {"detail": "This is the refusal under test."}}

    out = PH.run(
        "var slot = (__harness.html('profList') || '').match("
        "  /id=\"ac-(find-[^\"]+)\"/);\n"
        "if (!slot) { __emit({no_control: true}); } else {\n"
        "  __harness.dispatch('click', __harness.target("
        "    {'data-dismiss': slot[1].slice(5), 'data-ac': slot[1]}));\n"
        # The POST is async and the failure lands in a `.catch`. Emitting
        # straight after the dispatch reads the DOM before the response has
        # arrived, which reports "nothing rendered" for a render that is one
        # microtask away — a false negative the sweep in Part D must also avoid.
        "  for (var i = 0; i < 8; i++) "
        "    await new Promise(function(r){ setTimeout(r, 0); });\n"
        "  __emit({key: slot[1], at: __harness.html('ac-' + slot[1]),"
        "          canonical: __harness.el('upErr') ?"
        "                     __harness.el('upErr').textContent : null});\n}",
        routes=routes, search=f"?project={pid}")

    if out.get("no_control"):
        pytest.skip(f"{fixture} rendered no finding card to press")
    assert out["at"], (
        f"the refusal rendered nothing at the control. It went to `#upErr`, "
        f"which lives inside `#sub-upload` and is `display:none` from the first "
        f"render onward. Canonical sink held: {out['canonical']!r}")
    assert "refusal under test" in out["at"], (
        f"something rendered at the control and it is not the server's reason: "
        f"{out['at'][:200]!r}")
    # AND THE CANONICAL SINK STILL CARRIES IT. One string, two places — the
    # upload step is the moment `#upErr` is the visible one.
    assert out["canonical"] and "refusal under test" in out["canonical"], (
        "the message stopped reaching `#upErr`, which is the sink that IS "
        "visible during upload")


def test_the_feature_receipt_renders_where_the_control_was():
    """`GUIDED-173`. The `answered-card` substitution, at the row's own position."""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _driven()
    routes = _routes(client, pid)
    feats = routes[f"/project/{pid}/features"]
    row = (feats.get("row_local") or feats.get("deferred") or [None])[0]
    if not row:
        pytest.skip("no transform in the catalogue on this fixture")

    # The server reports the transform as engineered, which is what a successful
    # add produces. The row must then render its receipt in place.
    landed = dict(feats)
    landed["engineered"] = list(feats.get("engineered") or []) + [
        {"key": row["key"], "column": "made_up_col",
         "sentence": "A receipt sentence the server composed."}]
    routes[f"/project/{pid}/features"] = landed

    out = PH.run("__emit({build: __harness.html('featBuild'),"
                 "        decided: __harness.html('featDecided')});",
                 routes=routes, search=f"?project={pid}")
    assert "answered-card" in (out["build"] or ""), (
        "the catalogue row still renders its control after the transform "
        "landed, so the only receipt is the one six levels away")
    assert "A receipt sentence the server composed." in (out["build"] or ""), (
        "the receipt at the control does not quote the server's sentence")
    assert "In the table now." in (out["build"] or "")


def test_an_earmark_reaches_the_dock_that_promises_to_hold_it():
    """`GUIDED-161`. The dock says *"Anything you set aside waits here."*"""
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _driven()
    project = client.get(f"/project/{pid}").json()
    project = dict(project, decisions=list(project["decisions"]) + [
        {"id": "e1", "kind": "earmark", "subject": "verify_units",
         "text": "Verify the units against the source — earmarked.",
         "at": "2026-08-04T00:00:00+00:00",
         "payload": {"key": "verify_units", "target_step": "you",
                     "label": "Verify the units", "for_a_person": True}},
        {"id": "e2", "kind": "earmark", "subject": "winsorize",
         "text": "Winsorize the tails — earmarked.",
         "at": "2026-08-04T00:00:00+00:00",
         "payload": {"key": "winsorize", "target_step": "preprocess",
                     "label": "Winsorize the tails", "for_a_person": False}}])
    routes = _routes(client, pid)
    routes[f"/project/{pid}"] = project

    out = PH.run("__emit({list: __harness.html('ledgerList'),"
                 "        count: __harness.el('ledgerCt') ?"
                 "               __harness.el('ledgerCt').textContent : null});",
                 routes=routes, search=f"?project={pid}")
    assert out["count"] == "2", (
        f"the coach ledger counts {out['count']} with two earmarks recorded, so "
        f"its own copy — 'Anything you set aside waits here' — is unkept")
    assert "Verify the units" in (out["list"] or "")
    assert "Winsorize the tails" in (out["list"] or "")
    # AND `you` IS NOT RENDERED AS A STEP. An earmark for a person never comes
    # back; saying "comes back at you" would be the dock asserting something
    # false about the app's own reach.
    assert "yours to do" in (out["list"] or ""), (
        "an earmark for a person is labeled as a step it will return at")
    assert "comes back at you" not in (out["list"] or "")


def test_the_offer_caption_does_not_name_a_control_that_does_not_exist():
    """`GUIDED-162`. Deletion or build — and it was neither: the caption was wrong.

    There is no apply path for an offer key anywhere in the app.
    `data-offer-key` appears exactly twice in the page — emitted on the button,
    read to build the preview URL — and the `apply` decision is keyed to a
    structural finding's `fix_kind`, on a branch `openPanel` reaches only when
    `fix_kind` is absent or `"none"`. So the caption named a control that cannot
    exist, and the shelf is not shortened to make the sentence true.

    **The stated limit that used to be here was wrong and is now driven.**
    L47 recorded, and `GUIDED-182` filed, that *no fixture produces a previewable
    offer at all*. That check read `previewable` off the project payload, where
    findings carry `suggested_actions` and no offers — the offers live behind
    `GET /project/<id>/finding/<fid>/offers`, which it never called. Called:
    **35 options across the shipped fixtures, 29 of them previewable**, and the
    preview endpoint answers 200 with a real before/after. The branch is
    reachable and it is driven in
    `test_the_offer_preview_branch_is_reachable_and_says_so` below.
    """
    from turbotab import api

    # THE DOCSTRING'S OWN PREMISE, ASSERTED. "`data-offer-key` appears exactly
    # twice" is the whole argument for "there is no apply path", and for one
    # loop it was a sentence in prose with no record behind it — the same trap
    # #3b as the sibling test above, one notch weaker. Counted here so the
    # premise fails loudly if an apply path is ever added without this row
    # being revisited.
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert page.count("data-offer-key") == 2, (
        f"`data-offer-key` appears {page.count('data-offer-key')} times, not "
        f"twice. This finding's premise is that there is no apply path for an "
        f"offer key; a third mention means there may now be one, and the "
        f"caption below has to be revisited rather than kept")

    for defers in (True, False):
        said = api.offer_caption(defers)
        assert "pressing apply" not in said, (
            f"defers={defers}: the caption still names an apply control, and "
            f"there is no apply path for an offer key: {said!r}")
        assert said.startswith("preview, not applied"), said
    assert "earmark it to record it" in api.offer_caption(False), (
        "the non-deferring branch names no way to take the offer, so the panel "
        "is a dead end rather than a preview of something")
    assert "training fold" in api.offer_caption(True)


@pytest.mark.parametrize("fixture,target", [
    ("clinic_visits.csv", "outcome"),
    ("metabolomics_untargeted.csv", "bmi"),
], ids=["classification target", "regression target"])
def test_the_offer_preview_branch_is_reachable_and_says_so(fixture, target):
    """`GUIDED-182`, and it is NOT a defect — the row was mine and it was wrong.

    The claim was that the offer-preview branch is unreachable from any shipped
    fixture, which made `GUIDED-162`'s caption uncheckable end to end and was
    offered as the reason four loops of sweeps missed it. The claim came from
    reading `previewable` off `GET /project/<id>` — where findings carry
    `suggested_actions` and no offers at all — instead of from
    `GET /project/<id>/finding/<fid>/offers`, which is where `actions.offers`
    runs. **A grep answering the wrong question, trap #5, in a row I filed.**

    Driven here on two fixtures of different target shape (`GUIDED-097`): every
    previewable option the fixture produces is previewed over HTTP, and the
    caption under test is the one the endpoint actually serves.

    NOT COVERED: whether the preview renders on screen — the page path is
    `data-offer-preview` → `offerPreviewHTML` into the row's own
    `data-offer-pv` slot, and that slot's existence is what `GUIDED-161` was
    about; visibility needs layout.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})

    previewed = 0
    for finding in client.get(f"/project/{pid}").json()["findings"]:
        offers = client.get(f"/project/{pid}/finding/{finding['id']}/offers")
        if offers.status_code != 200:
            continue
        for option in offers.json().get("options") or []:
            if not option.get("previewable"):
                continue
            got = client.get(f"/project/{pid}/finding/{finding['id']}"
                             f"/offer/{option['key']}/preview")
            assert got.status_code == 200, (
                f"{finding['id']}/{option['key']} is offered as previewable and "
                f"the preview refuses it ({got.status_code}): "
                f"{str(got.json())[:200]}")
            body = got.json()
            assert body["applied"] is False, (
                "a preview reports itself as applied, which is the one thing "
                "the panel promises it is not")
            assert body["label_note"] == api.offer_caption(bool(body["defers"])), (
                f"the served caption is not the composed one: "
                f"{body['label_note']!r}")
            previewed += 1

    assert previewed >= 3, (
        f"only {previewed} previewable offers were reachable on {fixture}. "
        f"`GUIDED-182` claimed zero and the number is what refutes it, so a "
        f"drop to zero means the row was right after all and this test is now "
        f"asserting a fiction")


def test_the_viewport_still_never_moves():
    """The guard this fix is not allowed to buy its way past.

    Asserted here as well as in its own file, because *"render it at the
    control"* is one bad afternoon away from *"scroll to the control"*, and the
    standing grep covers `window.scrollTo` and `nudge(` only — a plain
    `el.scrollTop = …` is an ordinary property write on the harness shim and
    would pass silently.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    script = page[page.index("<script>"):page.rindex("</script>")]
    for spelling in ("window.scrollTo", "function nudge(", "scrollIntoView",
                     ".scrollTop =", ".scrollLeft =", "scrollBy("):
        assert spelling not in script, (
            f"the controller can move the viewport, by `{spelling}`. The "
            f"standing guard greps two spellings; this is the third, fourth "
            f"and fifth.")
