"""Server mode's sign-in and per-user workspaces (V2 definition of done §4; turbotab.server.auth,
turbotab.server.tenancy, turbotab.server.users). Local mode has neither and is tested in test_api.
"""
from __future__ import annotations

import io
import ipaddress
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import anyio
import pytest
from fastapi.testclient import TestClient

from turbotab.core.config import Settings
from turbotab.server import users
from turbotab.server.app import create_app
from turbotab.server import auth as signin
from turbotab.server.auth import (COOKIE, SECURE_COOKIE, AuthConfig, AuthConfigError,
                                  parse_networks, safe_next)
from turbotab.server.tenancy import user_home
from turbotab.server.users import Account, InvalidUsername, check_username, hash_password

REPO = Path(__file__).resolve().parents[3]
PASSWORD = "correct horse battery staple"
FAST = 2**10  # a cheap scrypt cost for test accounts; the hash records the cost it was made with
CSV = b"id,energy_kcal,protein_g,outcome\n" + b"".join(
    f"{i},{1800 + 13 * i},{50 + i % 17},{i % 2}\n".encode() for i in range(40))
VALUES = {"pid": "p0123456789", "jid": "j1", "stage": "ingest", "name": "energy_kcal", "fid": "f1"}


def write_users(path: Path, names: list[str], password: str = PASSWORD) -> Path:
    users.write_accounts(path, [Account(n, hash_password(password, n=FAST)) for n in names])
    return path


def server_app(home: Path, **config) -> tuple:
    config.setdefault("users_file", write_users(home / "users.toml", ["alice", "bob"]))
    app = create_app(Settings(home=home, mode="server", workers=1, memory_budget_bytes=2 << 30),
                     frontend_dist=home / "no-frontend-here", auth=AuthConfig(**config))
    app.state.auth.users.recheck = 0.0  # see a rewritten users file at once
    return app


def sign_in(client: TestClient, user: str, password: str = PASSWORD) -> str:
    """Sign in and return the session id; the client's own cookie jar is left empty, so each
    request says whose it is (:func:`as_user`)."""
    response = client.post("/login", data={"username": user, "password": password, "next": "/"},
                           follow_redirects=False)
    assert response.status_code == 303, response.text
    token = re.search(r"(?:__Host-)?turbotab_session=([^;]+)", response.headers["set-cookie"]).group(1)
    client.cookies.clear()
    return token


def as_user(token: str) -> dict[str, str]:
    return {"cookie": f"{COOKIE}={token}"}


def api_routes(app) -> list[tuple[str, str]]:
    """(method, path template) for every /api route the app mounts, from its own OpenAPI document."""
    paths = app.openapi()["paths"]
    return [(method.upper(), path) for path, item in sorted(paths.items()) if path.startswith("/api")
            for method in sorted(item)]


def fill(template: str, **values: str) -> str:
    merged = {**VALUES, **values}
    return re.sub(r"\{(\w+)\}", lambda m: merged[m.group(1)], template)


def call(client: TestClient, method: str, url: str, headers: dict[str, str] | None = None):
    body = {"files": {"file": ("t.csv", CSV, "text/csv")}} if url.endswith("/upload") else (
        {"json": {}} if method == "POST" else {})
    return client.request(method, url, headers=headers or {}, follow_redirects=False, **body)


def wait_fresh(client: TestClient, pid: str, token: str, stage: str = "ingest") -> dict:
    end = time.monotonic() + 120
    while True:
        view = client.get(f"/api/projects/{pid}", headers=as_user(token)).json()
        status = view["stages"][stage]["status"]
        if status == "fresh":
            return view
        assert status != "error", view["stages"][stage]
        assert time.monotonic() < end, f"{stage} never became fresh: {status}"
        time.sleep(0.05)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A server-mode site with alice and bob, alice holding one uploaded project."""
    home = tmp_path_factory.mktemp("site")
    app = server_app(home, stream_recheck=0.2)
    with TestClient(app, base_url="http://turbotab.example") as client:
        alice = sign_in(client, "alice")
        response = client.post("/api/projects/upload", headers=as_user(alice),
                               files={"file": ("intake.csv", CSV, "text/csv")})
        assert response.status_code == 200, response.text
        pid = response.json()["id"]
        wait_fresh(client, pid, alice)
        yield {"app": app, "home": home, "client": client, "alice": alice, "pid": pid}


# ── every route needs a session ─────────────────────────────────────────────


def test_every_api_route_answers_401_without_a_session(site):
    client = site["client"]
    routes = api_routes(site["app"])
    families = {path.split("/")[-1] for _, path in routes}
    # the families the requirement names, so the sweep cannot quietly lose one
    assert {"events", "upload", "export", "preview", "health", "projects"} <= families
    assert len(routes) >= 30
    for method, template in routes:
        for headers in ({}, as_user("not-a-session"), {"cookie": f"{SECURE_COOKIE}=forged"}):
            response = call(client, method, fill(template), headers)
            assert response.status_code == 401, (method, template, response.status_code)
            assert response.json()["error"]["code"] == "unauthenticated"
    # positive control: the same routes answer a signed-in user
    assert client.get("/api/health", headers=as_user(site["alice"])).status_code == 200


def test_a_page_without_a_session_goes_to_the_sign_in_page_and_comes_back(site):
    client = site["client"]
    for path in ("/", "/projects/p0123456789?tab=rows", "/assets/app.js"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303
        location = response.headers["location"]
        assert location.startswith("/login?next=") and unquote(location.split("=", 1)[1]) == path
    page = client.get("/login")
    assert page.status_code == 200 and 'action="/login"' in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    back = client.post("/login", data={"username": "alice", "password": PASSWORD,
                                       "next": "/projects/p0123456789?tab=rows"}, follow_redirects=False)
    assert back.headers["location"] == "/projects/p0123456789?tab=rows"
    client.cookies.clear()
    assert client.get("/healthz").json() == {"ok": True}  # liveness needs no session
    assert client.get("/login/inter-latin.woff2").status_code == 200
    assert client.get("/login/users.toml").status_code == 404


@pytest.mark.parametrize("raw", ["//evil.example/x", "https://evil.example", "/\\evil.example",
                                 "evil", "/login", "/logout?x=1", "/x\r\nSet-Cookie: a=b", None, ""])
def test_the_page_to_return_to_is_always_on_this_server(raw):
    assert safe_next(raw) == "/"


def test_no_page_but_turbotabs_own_can_sign_in_sign_out_or_change_anything(tmp_path):
    """Login CSRF: a page elsewhere that posts the sign-in form signs the visitor in to the
    attacker's account, and the visitor's next upload lands in the attacker's workspace. The
    sign-in routes are open to everyone, so the check comes before them, not after."""
    app = server_app(tmp_path)
    client = TestClient(app, base_url="http://turbotab.univ.example")
    form = {"username": "bob", "password": PASSWORD, "next": "/"}
    attacker = {"sec-fetch-site": "cross-site", "origin": "https://attacker.example"}
    sibling = {"sec-fetch-site": "same-site", "origin": "http://people.univ.example"}
    old_browser = {"origin": "https://attacker.example"}  # no Sec-Fetch-Site: Origin against Host
    framed = {"origin": "null"}                            # a sandboxed frame, a file:// page
    for headers in (attacker, sibling, old_browser, framed):
        refused = client.post("/login", data=form, headers=headers, follow_redirects=False)
        assert (refused.status_code, refused.json()["error"]["code"]) == (403, "cross_site"), headers
        assert "set-cookie" not in refused.headers
    alice = sign_in(client, "alice")
    for headers in (attacker, sibling, old_browser, framed):
        out = client.post("/logout", headers={**as_user(alice), **headers}, follow_redirects=False)
        assert out.status_code == 403, headers
        upload = client.post("/api/projects/upload", headers={**as_user(alice), **headers},
                             files={"file": ("x.csv", CSV, "text/csv")})
        assert upload.status_code == 403, headers
    assert client.get("/", headers=as_user(alice), follow_redirects=False).status_code == 200
    assert not (user_home(tmp_path, "alice") / "projects").exists()  # nothing was uploaded

    # TurboTab's own pages, an old browser on them, and clients that name no page get through.
    own = {"sec-fetch-site": "same-origin", "origin": "http://turbotab.univ.example"}
    for headers in (own, {"origin": "http://turbotab.univ.example"}, {}):
        signed = client.post("/login", data=form, headers=headers, follow_redirects=False)
        assert signed.status_code == 303 and "set-cookie" in signed.headers, headers
    out = client.post("/logout", headers={**as_user(alice), **own}, follow_redirects=False)
    assert out.status_code == 303
    assert client.get("/", headers=as_user(alice), follow_redirects=False).status_code == 303

    # proxy mode has no sign-in form, and the same rule
    proxied = server_app(tmp_path / "p", mode="proxy",
                         trusted_proxies=parse_networks("10.1.2.3"))
    via_proxy = TestClient(proxied, base_url="http://turbotab.univ.example", client=("10.1.2.3", 1))
    named = {"x-forwarded-user": "alice"}
    for headers in (attacker, sibling):
        assert via_proxy.post("/api/projects/upload", headers={**named, **headers},
                              files={"file": ("x.csv", CSV, "text/csv")}).status_code == 403


def test_the_health_says_who_is_signed_in_and_how(site):
    body = site["client"].get("/api/health", headers=as_user(site["alice"])).json()
    assert (body["mode"], body["user"], body["auth"]) == ("server", "alice", "password")


# ── per-user workspaces ─────────────────────────────────────────────────────


def test_no_user_can_reach_another_users_project_on_any_route(site):
    client, pid, home = site["client"], site["pid"], site["home"]
    bob = sign_in(client, "bob")
    alice = site["alice"]
    pid_routes = [(m, t) for m, t in api_routes(site["app"]) if "{pid}" in t]
    assert len(pid_routes) >= 20
    for method, template in pid_routes:
        response = call(client, method, fill(template, pid=pid), as_user(bob))
        assert response.status_code == 404, (method, template, response.status_code, response.text)
        assert response.json()["error"]["code"] == "unknown_project", (method, template)
    # positive control: the same id answers its owner
    assert client.get(f"/api/projects/{pid}", headers=as_user(alice)).json()["summary"]["id"] == pid
    assert pid in [p["id"] for p in client.get("/api/projects", headers=as_user(alice)).json()]
    assert pid not in [p["id"] for p in client.get("/api/projects", headers=as_user(bob)).json()]
    # on disk: alice's workspace holds it, bob's does not
    assert (user_home(home, "alice") / "projects" / pid / "project.json").is_file()
    assert not (user_home(home, "bob") / "projects" / pid).exists()
    assert not (home / "projects").exists()  # nothing in the shared root


def test_a_job_id_is_answered_only_in_its_owners_workspace(site):
    """Job ids are another key a request can name: bob, asking about alice's job under his own
    project, is told it does not exist (each workspace's event bus knows only its own jobs)."""
    client, app = site["client"], site["app"]
    bob = sign_in(client, "bob")
    created = client.post("/api/projects/upload", headers=as_user(bob),
                          files={"file": ("bob.csv", CSV, "text/csv")})
    bob_pid = created.json()["id"]
    alice_jobs = list(app.state.tenants.for_user("alice").bus._owners)  # alice's announced jobs
    assert alice_jobs, "alice's ingest announced a job"
    for jid in alice_jobs:
        response = client.get(f"/api/projects/{bob_pid}/jobs/{jid}", headers=as_user(bob))
        assert (response.status_code, response.json()["error"]["code"]) == (404, "unknown_job")
        cancel = client.post(f"/api/projects/{bob_pid}/jobs/{jid}/cancel", headers=as_user(bob))
        assert cancel.status_code == 404
    owner = client.get(f"/api/projects/{site['pid']}/jobs/{alice_jobs[0]}",
                       headers=as_user(site["alice"]))
    assert owner.status_code == 200  # positive control


def test_an_event_stream_needs_a_session_and_ends_with_it(site):
    client, pid = site["client"], site["pid"]
    token = sign_in(client, "alice")
    assert client.get(f"/api/projects/{pid}/events").status_code == 401
    bob = sign_in(client, "bob")
    assert client.get(f"/api/projects/{pid}/events", headers=as_user(bob)).status_code == 404

    async def run() -> tuple[int, list[str], float]:
        status, events, ended = 0, [], anyio.Event()
        subscribed = anyio.Event()

        async def receive() -> dict:
            await anyio.sleep_forever()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                text = message.get("body", b"").decode()
                events.extend(re.findall(r"^event: (\w+)", text, re.M))
                if "resync" in events:
                    subscribed.set()
                if not message.get("more_body", False):
                    ended.set()

        path = f"/api/projects/{pid}/events"
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                 "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
                 "query_string": b"", "client": ("127.0.0.1", 50000), "server": ("turbotab.example", 80),
                 "headers": [(b"host", b"turbotab.example"),
                             (b"cookie", f"{COOKIE}={token}".encode())]}
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.app, scope, receive, send)
            with anyio.fail_after(30):
                await subscribed.wait()
            started = time.monotonic()
            await anyio.to_thread.run_sync(
                lambda: client.post("/logout", headers=as_user(token), follow_redirects=False))
            with anyio.fail_after(30):
                await ended.wait()
            took = time.monotonic() - started
            tg.cancel_scope.cancel()
        return status, events, took

    status, events, took = client.portal.call(run)
    assert status == 200 and events[0] == "resync"
    assert took < 15  # it rechecks every 0.2 s here; without the check it never ends
    assert client.get("/api/health", headers=as_user(token)).status_code == 401


# ── sessions ─────────────────────────────────────────────────────────────────


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_session_expires_when_idle_and_at_its_limit_however_busy(tmp_path):
    clock = Clock()
    app = server_app(tmp_path, idle_seconds=60, max_seconds=300, clock=clock)
    client = TestClient(app, base_url="http://turbotab.example")  # pages only: no engine needed

    def alive(token: str) -> bool:
        response = client.get("/", headers=as_user(token), follow_redirects=False)
        assert response.status_code in (200, 303)
        return response.status_code == 200

    idle = sign_in(client, "alice")
    clock.now += 59
    assert alive(idle)          # a request resets the idle clock
    clock.now += 59
    assert alive(idle)
    clock.now += 61
    assert not alive(idle)      # 61 s without a request
    clock.now -= 61
    assert not alive(idle)      # and an ended session stays ended

    busy = sign_in(client, "alice")
    for _ in range(5):
        clock.now += 50
        assert alive(busy)      # 250 s, never idle
    clock.now += 50
    assert not alive(busy)      # 300 s after signing in, however busy

    gone = sign_in(client, "alice")
    assert alive(gone)
    client.post("/logout", headers=as_user(gone), follow_redirects=False)
    assert not alive(gone)      # signing out ends it on the server, not only in the browser


def test_a_new_password_or_a_removed_account_ends_its_sessions(tmp_path):
    app = server_app(tmp_path)
    client = TestClient(app, base_url="http://turbotab.example")
    alice, bob = sign_in(client, "alice"), sign_in(client, "bob")
    users_file = tmp_path / "users.toml"
    write_users(users_file, ["alice"], password="another long password")  # bob removed, alice new
    for token in (alice, bob):
        assert client.get("/", headers=as_user(token), follow_redirects=False).status_code == 303
    assert sign_in(client, "alice", "another long password")


def test_a_users_file_that_cannot_be_read_lets_no_one_in(tmp_path):
    app = server_app(tmp_path)
    client = TestClient(app, base_url="http://turbotab.example")
    alice = sign_in(client, "alice")
    (tmp_path / "users.toml").write_text("[users.alice\nhash = ", "utf-8")  # a broken hand edit
    assert client.get("/", headers=as_user(alice), follow_redirects=False).status_code == 303
    refused = client.post("/login", data={"username": "alice", "password": PASSWORD})
    assert refused.status_code == 401
    assert users.main(["--file", str(tmp_path / "users.toml"), "list"]) == 1  # says so, no trace


def test_the_cookie_is_httponly_strict_and_secure_when_it_should_be(tmp_path):
    def cookie(base_url: str = "http://turbotab.example", headers: dict | None = None,
               client_addr: tuple = ("testclient", 50000), **config) -> str:
        app = server_app(tmp_path, **config)
        client = TestClient(app, base_url=base_url, client=client_addr)
        response = client.post("/login", data={"username": "alice", "password": PASSWORD},
                               headers=headers or {}, follow_redirects=False)
        return response.headers["set-cookie"]

    plain = cookie()
    assert plain.startswith(f"{COOKIE}=") and "HttpOnly" in plain and "SameSite=Strict" in plain
    assert "Secure" not in plain and "Path=/" in plain
    assert len(re.match(rf"{COOKIE}=([^;]+)", plain).group(1)) >= 43  # 32 random bytes, base64
    for secure in (cookie(secure_cookies=True), cookie("https://turbotab.example"),
                   cookie(headers={"x-forwarded-proto": "https"}, client_addr=("10.0.0.2", 1),
                          trusted_proxies=(ipaddress.ip_network("10.0.0.0/8"),))):
        assert secure.startswith(f"{SECURE_COOKIE}=") and "; Secure" in secure, secure
        assert "HttpOnly" in secure and "SameSite=Strict" in secure
    # a forwarded scheme from a peer that is not a trusted proxy is not believed
    assert "Secure" not in cookie(headers={"x-forwarded-proto": "https"})


# ── sign-in attempts ─────────────────────────────────────────────────────────


def test_failed_sign_ins_are_limited_per_username_and_per_address(tmp_path):
    clock = Clock()
    app = server_app(tmp_path, user_attempts=3, ip_attempts=5, attempt_window=600, clock=clock)
    home_net = TestClient(app, base_url="http://turbotab.example", client=("198.51.100.1", 1))
    other = TestClient(app, base_url="http://turbotab.example", client=("198.51.100.2", 1))

    def attempt(client: TestClient, user: str, password: str) -> int:
        return client.post("/login", data={"username": user, "password": password},
                           follow_redirects=False).status_code

    assert [attempt(home_net, "alice", "wrong guess here") for _ in range(3)] == [401] * 3
    refused = home_net.post("/login", data={"username": "alice", "password": PASSWORD},
                            follow_redirects=False)
    assert refused.status_code == 429 and int(refused.headers["retry-after"]) > 0
    assert "Too many sign-in attempts" in refused.text
    assert attempt(other, "alice", PASSWORD) == 429   # the account waits wherever it is tried from
    assert attempt(home_net, "bob", PASSWORD) == 303   # another account from here is not blocked
    clock.now += 601
    assert attempt(home_net, "alice", PASSWORD) == 303  # the window passed

    # per address: names that do not exist count too, so guessing across names stops
    assert [attempt(other, f"nobody{i}", "wrong guess here") for i in range(5)] == [401] * 5
    assert attempt(other, "bob", PASSWORD) == 429
    assert attempt(home_net, "bob", PASSWORD) == 303


def test_sign_ins_sent_at_once_are_held_to_the_limits(tmp_path, monkeypatch):
    """Counted only after the hash, every attempt in flight passed the check: a burst of 30
    tried 30 passwords against a limit of 5. Each attempt now counts as it starts."""
    app = server_app(tmp_path, user_attempts=5, ip_attempts=8)
    auth = app.state.auth
    checked: list[str] = []
    lock = threading.Lock()

    def slow_and_wrong(password: str, *_: object) -> bool:  # scrypt's time, never a match
        with lock:
            checked.append(password)
        time.sleep(0.2)
        return False

    monkeypatch.setattr(signin, "verify_password", slow_and_wrong)
    monkeypatch.setattr(signin, "burn_time", slow_and_wrong)

    def burst(scope: dict, names: list[str]) -> list[str]:
        notes: list[str] = [""] * len(names)

        def one(i: int) -> None:
            notes[i] = auth.sign_in(scope, names[i], f"guess-{i}")[1]
        threads = [threading.Thread(target=one, args=(i,)) for i in range(len(names))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return notes

    notes = burst({"client": ("198.51.100.7", 1), "headers": []}, ["alice"] * 30)
    assert len(checked) == 5, f"{len(checked)} passwords were checked against a limit of 5"
    assert sum("Too many" in n for n in notes) == 25
    checked.clear()
    burst({"client": ("198.51.100.8", 1), "headers": []}, [f"nobody{i}" for i in range(30)])
    assert len(checked) == 8  # per address, across names that do not exist

    # an address's own successful sign-ins are not failures and never use up its limit
    monkeypatch.undo()
    client = TestClient(app, base_url="http://turbotab.example", client=("198.51.100.9", 1))
    for _ in range(12):
        assert client.post("/login", data={"username": "bob", "password": PASSWORD},
                           follow_redirects=False).status_code == 303


def test_an_unknown_name_and_a_wrong_password_get_the_same_answer(tmp_path):
    client = TestClient(server_app(tmp_path), base_url="http://turbotab.example")
    unknown = client.post("/login", data={"username": "mallory", "password": "x" * 12})
    wrong = client.post("/login", data={"username": "alice", "password": "x" * 12})
    assert unknown.status_code == wrong.status_code == 401
    note = re.compile(r'<p class="note" role="alert">(.*?)</p>')
    assert note.search(unknown.text).group(1) == note.search(wrong.text).group(1)


# ── the institution's proxy ─────────────────────────────────────────────────


def test_the_proxy_header_counts_only_from_a_trusted_peer(tmp_path):
    app = server_app(tmp_path, mode="proxy", trusted_proxies=(ipaddress.ip_network("10.1.2.0/24"),),
                     proxy_header="X-Remote-User")
    proxy = TestClient(app, base_url="http://turbotab.example", client=("10.1.2.3", 1))
    stranger = TestClient(app, base_url="http://turbotab.example", client=("203.0.113.9", 1))
    named = {"x-remote-user": "alice"}

    assert proxy.get("/", headers=named, follow_redirects=False).status_code == 200
    assert proxy.get("/", headers={"x-remote-user": "Alice@Univ.edu"}).status_code == 200
    for headers in (named, {**named, "x-forwarded-for": "10.1.2.3"}, {}):
        assert stranger.get("/api/models", headers=headers).status_code == 401
        page = stranger.get("/", headers=headers, follow_redirects=False)
        assert page.status_code == 401 and "institution" in page.text
    assert proxy.get("/api/models").status_code == 401  # the proxy named nobody
    # a copy the browser sent beside the proxy's own: neither is believed
    two = [("x-remote-user", "mallory"), ("x-remote-user", "alice")]
    assert proxy.get("/api/models", headers=two).status_code == 401
    for bad in ("../../etc", "a/b", "..", ".hidden", "con"):
        assert proxy.get("/api/models", headers={"x-remote-user": bad}).status_code == 401
    assert not (tmp_path / "users").exists()  # no name the proxy sent made a folder
    assert proxy.post("/login", data={"username": "alice", "password": PASSWORD}).status_code == 404


def test_the_client_address_is_read_from_a_trusted_proxy_only(tmp_path):
    app = server_app(tmp_path, trusted_proxies=(ipaddress.ip_network("10.0.0.0/8"),))
    auth = app.state.auth

    def address(peer: str, *forwarded: str) -> str:
        headers = [(b"x-forwarded-for", f.encode()) for f in forwarded]
        return auth.client_address({"client": (peer, 1), "headers": headers})

    assert address("10.0.0.2", "203.0.113.5") == "203.0.113.5"
    assert address("10.0.0.2", "198.51.100.9, 203.0.113.5, 10.0.0.7") == "203.0.113.5"  # nearest
    assert address("10.0.0.2", "198.51.100.9", "203.0.113.5") == "203.0.113.5"  # two headers
    assert address("203.0.113.5", "10.0.0.9") == "203.0.113.5"  # a stranger cannot name another


def test_proxy_mode_refuses_to_start_without_a_narrow_list_of_trusted_proxies(tmp_path, monkeypatch):
    """In proxy mode a trusted network is everyone in it: each of them can name any user."""
    with pytest.raises(AuthConfigError, match="TURBOTAB_TRUSTED_PROXIES"):
        AuthConfig(mode="proxy")
    settings = Settings(home=tmp_path, mode="server", workers=1, memory_budget_bytes=1 << 30)
    with pytest.raises(AuthConfigError):
        AuthConfig.from_env(settings, {"TURBOTAB_AUTH": "proxy"})
    with pytest.raises(AuthConfigError, match="not an IP"):
        AuthConfig.from_env(settings, {"TURBOTAB_AUTH": "proxy", "TURBOTAB_TRUSTED_PROXIES": "proxy"})
    config = AuthConfig.from_env(settings, {"TURBOTAB_AUTH": "proxy",
                                            "TURBOTAB_TRUSTED_PROXIES": "127.0.0.1, 10.0.0.0/24"})
    assert [str(n) for n in config.trusted_proxies] == ["127.0.0.1/32", "10.0.0.0/24"]
    for broad in ("0.0.0.0/0", "::/0", "172.16.0.0/12", "172.17.0.0/16", "10.0.0.0/23",
                  "127.0.0.1, 10.0.0.0/8", "fd00::/64", "fd00::/119"):
        with pytest.raises(AuthConfigError, match="at most 256 addresses"):
            AuthConfig(mode="proxy", trusted_proxies=parse_networks(broad))
        AuthConfig(mode="password", trusted_proxies=parse_networks(broad))  # forwarded-for only
    AuthConfig(mode="proxy", trusted_proxies=parse_networks("fd00::/120, 192.0.2.7"))

    from turbotab.server.__main__ import main

    monkeypatch.setenv("TURBOTAB_HOME", str(tmp_path))
    monkeypatch.setenv("TURBOTAB_AUTH", "proxy")
    for trusted in (None, "0.0.0.0/0"):
        if trusted is None:
            monkeypatch.delenv("TURBOTAB_TRUSTED_PROXIES", raising=False)
        else:
            monkeypatch.setenv("TURBOTAB_TRUSTED_PROXIES", trusted)
        with pytest.raises(SystemExit) as stopped:
            main(["--mode", "server"])
        assert stopped.value.code == 2, trusted


# ── usernames and the accounts command ─────────────────────────────────────


@pytest.mark.parametrize("name", ["../bob", "..", ".", "a/b", "a\\b", "/etc", "", ".hidden", "bob.",
                                  "a..b", "Alice", "con", "nul.txt", "x" * 65, "al ice", "al\x00ice",
                                  "~root", "café"])
def test_a_username_that_could_name_a_path_is_refused(name, tmp_path):
    with pytest.raises(InvalidUsername):
        check_username(name)
    with pytest.raises(InvalidUsername):
        user_home(tmp_path, name)


@pytest.mark.parametrize("name", ["alice", "jdoe@univ.edu", "a.b-c_d", "7", "x" * 64])
def test_an_ordinary_username_is_accepted(name, tmp_path):
    assert check_username(name) == name
    assert user_home(tmp_path, name).parent == tmp_path / "users"


def test_the_users_command_adds_lists_changes_and_removes(tmp_path, monkeypatch, capsys):
    path = tmp_path / "conf" / "users.toml"

    def run(*args: str, stdin: str = "") -> int:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        return users.main(["--file", str(path), *args])

    monkeypatch.setattr(users, "SCRYPT_N", FAST)  # the command's default cost, cheap for the test
    assert run("add", "carol", "--password-stdin", stdin="a sufficiently long one\n") == 0
    assert run("add", "dave", "--password-stdin", stdin="another long password\n") == 0
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    accounts = users.read_accounts(path)
    assert sorted(accounts) == ["carol", "dave"]
    assert users.verify_password("a sufficiently long one", accounts["carol"].hash)
    assert not users.verify_password("a sufficiently long two", accounts["carol"].hash)
    assert accounts["carol"].hash != accounts["dave"].hash and accounts["carol"].hash.startswith("scrypt$")

    capsys.readouterr()
    assert run("list") == 0
    assert [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()] == ["carol", "dave"]

    assert run("add", "carol", "--password-stdin", stdin="a sufficiently long one\n") == 1
    with pytest.raises(SystemExit):
        run("add", "erin", "--password-stdin", stdin="short\n")
    assert run("add", "../erin", "--password-stdin", stdin="a sufficiently long one\n") == 2
    assert sorted(users.read_accounts(path)) == ["carol", "dave"]

    assert run("passwd", "carol", "--password-stdin", stdin="a brand new passphrase\n") == 0
    carol = users.read_accounts(path)["carol"]
    assert users.verify_password("a brand new passphrase", carol.hash)
    assert carol.created == accounts["carol"].created
    assert run("passwd", "nobody", "--password-stdin", stdin="a brand new passphrase\n") == 1

    assert run("remove", "dave") == 0
    assert sorted(users.read_accounts(path)) == ["carol"]
    assert run("remove", "dave") == 1

    # the module runs as a command (python -m turbotab.server.users)
    done = subprocess.run([sys.executable, "-m", "turbotab.server.users", "--file", str(path), "list"],
                          capture_output=True, text=True, cwd=REPO, timeout=60)
    assert done.returncode == 0 and done.stdout.startswith("carol\t"), done.stderr


def test_a_hash_records_its_cost_and_a_damaged_one_never_verifies():
    stored = hash_password("a long enough password", n=FAST)
    assert stored.startswith(f"scrypt$n={FAST},r=8,p=1$")
    assert users.verify_password("a long enough password", stored)
    for damaged in ("", "scrypt$", stored.replace("scrypt", "md5"), stored[:-4], "plain text",
                    stored.replace(f"n={FAST}", "n=1000"),           # not a power of two
                    stored.replace(f"n={FAST}", f"n={2**30}")):      # would take 256 GiB

        assert not users.verify_password("a long enough password", damaged)
    salts = {hash_password("same password!", n=FAST).split("$")[2] for _ in range(3)}
    assert len(salts) == 3  # a salt per hash


def test_local_mode_has_no_sign_in(client):
    body = client.get("/api/health").json()
    assert (body["mode"], body["user"], body["auth"]) == ("local", None, "none")
    assert client.get("/api/projects").status_code == 200
    page = client.get("/login")
    assert page.status_code == 200 and "Sign in to TurboTab" not in page.text  # the app's own route


def test_the_compose_example_and_the_image_agree_on_paths():
    """The deploy files name the same port, data folder and users file as the server reads."""
    compose = (REPO / "turbotab" / "deploy" / "docker-compose.example.yml").read_text("utf-8")
    dockerfile = (REPO / "turbotab" / "deploy" / "Dockerfile").read_text("utf-8")
    for needle in ("TURBOTAB_HOME=/data", "TURBOTAB_USERS=/etc/turbotab/users.toml", "8787"):
        assert needle in dockerfile.replace('"', ""), needle
    assert "/etc/turbotab:ro" in compose and ":/data" in compose


def test_the_compose_example_trusts_only_the_address_it_means(tmp_path):
    """The example trusted 172.16.0.0/12, every Docker network and the gateway every host process
    comes in through, beside "TURBOTAB_AUTH: password  # or: proxy". In proxy mode anyone on the
    host could then name the user. Its password setup trusts the gateway alone, and its single
    sign-on setup publishes no port and trusts only the SSO container."""
    yaml = pytest.importorskip("yaml")
    text = (REPO / "turbotab" / "deploy" / "docker-compose.example.yml").read_text("utf-8")
    compose = yaml.safe_load(text)
    ipam = compose["networks"]["turbotab"]["ipam"]["config"][0]
    subnet, gateway = ipaddress.ip_network(ipam["subnet"]), ipaddress.ip_address(ipam["gateway"])
    turbotab = compose["services"]["turbotab"]
    env = turbotab["environment"]
    assert env["TURBOTAB_AUTH"] == "password" and "or: proxy" not in text
    assert turbotab["ports"] == ["127.0.0.1:8787:8787"] and turbotab["networks"] == ["turbotab"]
    assert parse_networks(env["TURBOTAB_TRUSTED_PROXIES"]) == (ipaddress.ip_network(gateway),)

    # the single sign-on changes, shown at the end of the file as indented comments
    tail = text.split("Single sign-on instead of passwords", 1)[1]
    sso = yaml.safe_load("\n".join(line[4:] for line in tail.splitlines() if line.startswith("#   ")))
    proxied, proxy = sso["services"]["turbotab"], sso["services"]["sso"]
    assert "ports" not in proxied and proxied["environment"]["TURBOTAB_AUTH"] == "proxy"
    address = ipaddress.ip_address(proxy["networks"]["turbotab"]["ipv4_address"])
    assert address in subnet and address != gateway
    trusted = parse_networks(proxied["environment"]["TURBOTAB_TRUSTED_PROXIES"])
    assert trusted == (ipaddress.ip_network(address),)

    # run that way, the SSO container names the user and a host process through the gateway cannot
    app = server_app(tmp_path, mode="proxy", trusted_proxies=trusted)
    named = {"x-forwarded-user": "alice"}
    with TestClient(app, base_url="http://turbotab", client=(str(address), 1)) as client:
        assert client.get("/api/health", headers=named).json()["user"] == "alice"
    host = TestClient(app, base_url="http://turbotab", client=(str(gateway), 1))
    assert host.get("/api/health", headers=named).status_code == 401
