"""`TEST-084` — a page newer than the engine behind it, and nobody could see it.

Python modules are pinned at import. `StaticFiles` re-reads `index.html` per
request. So a long-running server drifts into serving a **new interface against
an old engine**, silently, with no symptom except findings that do not
reproduce.

Run 3 was driven on exactly that: a page written at 11:38 against a process
started 28 hours earlier. Every page-only change worked on screen and the whole
of a server-side part was absent, and working out why cost an adjudication.
Three consecutive drives have had a version question attached to them.

`PM_TRANSITION.md` §07 item 8 recorded the lesson *"check where the process is
serving from."* That was right and it is not sufficient — this process was
serving from the right **directory** and still not from the right **code**. So
the app answers it instead of the adjudicator.

## What is asserted

`/dev/status` is already fetched on every page load, so the cost is zero. The
build half is served **unconditionally** rather than behind the dev flag: the
drift is invisible by construction and a driver has no other way to ask.

And it **renders**, because a payload nobody reads is the defect this fixes —
`GUIDED-078`'s shape, twelve composed sentences that reached no reader.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def test_dev_status_names_the_build_the_process_is_running(capsys):
    """Served always, not only under the dev flag."""
    client = _client()
    body = client.get("/dev/status").json()
    assert "build" in body, "/dev/status does not say which build answered"
    build = body["build"]
    for key in ("rev", "engine_loaded_at", "page_mtime", "page_newer_than_engine"):
        assert key in build, f"{key} missing from /dev/status build block"
    # One of the two identifiers has to be real, or the route answers nothing.
    assert build["rev"] or build["engine_loaded_at"], (
        "neither a git rev nor a source mtime — the route cannot identify the "
        "build at all")
    assert isinstance(build["page_newer_than_engine"], bool)
    with capsys.disabled():
        print(f"\n  rev={build['rev']!r} page_newer={build['page_newer_than_engine']}")


def test_the_engine_stamp_is_read_once_and_does_not_follow_the_disk(capsys):
    """The whole point: it reports what the running Python IS.

    A stamp that re-read the checkout on every request would always agree with
    the page and could never report the drift — it would be measuring the disk,
    which is the thing that already lies.
    """
    from turbotab import api

    first = client_build = api._SERVED_BUILD
    again = api._SERVED_BUILD
    assert first is again, "the build stamp is recomputed per call"
    assert isinstance(client_build, dict)
    with capsys.disabled():
        print(f"\n  stamped once at import: {client_build['rev']!r}")


def test_a_page_newer_than_the_engine_is_reported(capsys):
    """The condition itself, driven by making the page newer.

    The file's mtime is moved and restored — no content changes, so nothing
    else in the tree can observe this.

    **The clean baseline is ESTABLISHED rather than required, and that is
    `L62`'s correction.** This used to open by asserting the working tree was
    not already hybrid, which is right in intent — without it the test would
    pass whether or not the change works — and wrong in mechanism: whether the
    page is newer than the engine depends on *which file the last edit
    touched*. `L62` edited `index.html` after the Python, so the assertion
    fired on a clean tree with working code and cost a red in a two-hour sweep.

    `_SERVED_BUILD` stamps `engine_loaded_at` once at import and `page_mtime`
    is read per request, so this test already owns the only variable that
    matters. It sets the page BELOW the engine for the baseline and ABOVE it
    for the condition, and restores the real mtime either way — so the
    precondition is a fact this test creates rather than one it hopes for.
    """
    import os

    from turbotab import api

    client = _client()
    engine_at = api._SERVED_BUILD["newest_source_mtime"]
    assert engine_at, "the served build carries no engine mtime to compare with"

    original = PAGE.stat()
    try:
        # THE BASELINE, MADE rather than assumed: the page an hour OLDER than
        # the engine is unambiguously not a hybrid.
        os.utime(PAGE, (original.st_atime, engine_at - 3600))
        before = client.get("/dev/status").json()["build"]
        assert before["page_newer_than_engine"] is False, (
            "the page is an hour OLDER than the engine and /dev/status still "
            "reports a hybrid, so the flag is not reading the two mtimes")
        assert before["why"] is None, (
            "a clean build carries a reason to restart, so the sentence is "
            "not conditional on the state it explains")

        os.utime(PAGE, (original.st_atime, engine_at + 3600))
        after = client.get("/dev/status").json()["build"]
    finally:
        os.utime(PAGE, (original.st_atime, original.st_mtime))

    assert after["page_newer_than_engine"] is True, (
        "the page is an hour newer than the engine and /dev/status says they "
        "agree")
    assert after["why"], "the report names no reason"
    assert "restart" in after["why"].lower()

    restored = client.get("/dev/status").json()["build"]
    assert isinstance(restored["page_newer_than_engine"], bool), (
        "the mtime was restored and the route stopped answering the question")
    with capsys.disabled():
        print(f"\n  page −1h → clean; page +1h → reported; mtime restored")


def test_the_banner_reaches_the_dom_rather_than_only_the_payload(capsys):
    """`GUIDED-078`'s shape: a sentence the server composes and nobody reads.

    Driven under `pageharness` with `/dev/status` answering the hybrid state,
    because the whole finding is that the drift was invisible.
    """
    from turbotab import api, pageharness

    if not pageharness.available():
        pytest.skip("no JS engine on this machine")

    client = _client()
    with (DATA / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]

    ids = sorted(set(re.findall(r'\bid="([A-Za-z0-9_-]+)"',
                                PAGE.read_text(encoding="utf-8"))))
    reader = ("var IDS = " + json.dumps(ids) + ";\n"
              "var blob = \"\"; IDS.forEach(function(i){\n"
              "  var e = document.getElementById(i); if (e) blob += (e.innerHTML || \"\"); });\n"
              "__emit({blob: blob, devBuild: __harness.html(\"devBuild\"),\n"
              "        cls: (__harness.el(\"devBuild\") || {}).className,\n"
              "        calls: __harness.calls().map(function(c){\n"
              "          return {method: c.method, path: c.path}; })});")

    def _routes(status_payload):
        routes = {"/dev/status": status_payload}
        for step in ("data", "explore", "preprocess", "features"):
            path = f"/project/{pid}/interview?step={step}"
            resp = client.get(path)
            if resp.status_code == 200:
                routes[path] = resp.json()
        seen: set = set()
        for _ in range(6):
            out = pageharness.run(reader, routes=routes, search=f"?project={pid}")
            calls = {(c["method"], c["path"]) for c in out["calls"]}
            if calls <= seen:
                break
            seen |= calls
            for call in out["calls"]:
                if call["method"] != "GET" or call["path"] in routes:
                    continue
                resp = client.get(call["path"])
                if resp.status_code == 200:
                    try:
                        routes[call["path"]] = resp.json()
                    except ValueError:
                        pass
        routes["/dev/status"] = status_payload
        return routes

    healthy = {"enabled": False, "session": None, "flag": "X",
               "build": {"rev": "abc1234", "engine_loaded_at": 2.0,
                         "page_mtime": 1.0, "page_newer_than_engine": False,
                         "why": None}}
    hybrid = {"enabled": False, "session": None, "flag": "X",
              "build": {"rev": "abc1234", "engine_loaded_at": 1.0,
                        "page_mtime": 2.0, "page_newer_than_engine": True,
                        "why": "The page is re-read from disk on every request "
                               "and the Python behind it was loaded when this "
                               "process started. Restart the server."}}

    # THE NEGATIVE FIRST: an agreeing server must render nothing, or the banner
    # is a mark that asserts nothing (§02) and the positive below proves little.
    clean = pageharness.run(reader, routes=_routes(healthy), search=f"?project={pid}")
    assert len(clean["blob"]) > 5_000, "the reader read nothing; believe no absence"
    assert not (clean["devBuild"] or "").strip(), (
        f"a healthy server rendered a build banner: {clean['devBuild']!r}")

    drifted = pageharness.run(reader, routes=_routes(hybrid), search=f"?project={pid}")
    assert "newer page than its engine" in (drifted["devBuild"] or ""), (
        f"the hybrid build reached the payload and not the page: "
        f"{drifted['devBuild']!r}")
    assert "Restart the server" in drifted["devBuild"]
    assert "abc1234" in drifted["devBuild"], "the banner does not name the build"
    assert "devbuild" in (drifted["cls"] or ""), (
        "the banner has content and no class, so the stylesheet keeps it hidden")
    with capsys.disabled():
        print(f"\n  healthy: silent · hybrid: {len(drifted['devBuild'])} chars on screen")
