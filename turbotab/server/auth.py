"""Server mode's sign-in (V2 definition of done §4; local mode has none and is unchanged).

Two ways to know who is asking, chosen by ``TURBOTAB_AUTH``:

* ``password`` (the default): accounts in the users file (``turbotab.server.users``), a sign-in
  page at ``/login``, and a session cookie: a random 256-bit id that names a session kept in this
  process (only its SHA-256 is stored), sent ``HttpOnly`` and ``SameSite=Strict``, and ``Secure``
  when ``TURBOTAB_SECURE_COOKIES`` is set or the request came over https. A session ends after
  ``TURBOTAB_SESSION_IDLE_MINUTES`` without a request (default 120), after
  ``TURBOTAB_SESSION_MAX_HOURS`` however busy (default 12), at sign-out, and when its account is
  removed or given a new password. Failed sign-ins are limited per username and per client address.
* ``proxy``: an institution's single sign-on in front of TurboTab names the user in a header
  (``TURBOTAB_PROXY_HEADER``, default ``X-Forwarded-User``). The header is read only from a peer
  in ``TURBOTAB_TRUSTED_PROXIES`` (addresses or networks); from anyone else it is ignored. The
  server refuses to start in this mode without that list.

``TURBOTAB_TRUSTED_PROXIES`` also says whose ``X-Forwarded-For`` and ``X-Forwarded-Proto`` to
believe in password mode, behind a TLS reverse proxy: the client address the limits count, and
whether the request was https.

:class:`AuthGate` stands in front of every route: in server mode every ``/api`` route needs a
signed-in user (the event stream, uploads, export downloads and previews included), a page
request without one is sent to ``/login``, and a route that names a project answers 404 unless the
project is in the user's own workspace (``turbotab.server.tenancy``). Statistics never come here
(BLUEPRINT §1), and nothing in ``turbotab/core`` knows about users.
"""
from __future__ import annotations

import base64
import hashlib
import html
import ipaddress
import os
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import parse_qs, quote, urlsplit

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from turbotab.core.config import Settings
from turbotab.server.errors import ApiError
from turbotab.server.tenancy import owns
from turbotab.server.users import (
    InvalidUsername,
    UsersFile,
    burn_time,
    check_username,
    default_path,
    verify_password,
)

AuthMode = Literal["password", "proxy"]
AUTH_MODES: tuple[str, ...] = ("password", "proxy")
Network = ipaddress.IPv4Network | ipaddress.IPv6Network

COOKIE = "turbotab_session"
SECURE_COOKIE = "__Host-turbotab_session"  # the __Host- prefix: Secure, Path=/, no Domain
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
MAX_FORM_BYTES = 8192
FONTS = Path(__file__).resolve().parents[2] / "static" / "fonts"
LOGIN_FONTS = ("inter-latin.woff2",)
PROJECT_PATH = re.compile(r"^/api/projects/([^/]+)(?:/|$)")


class AuthConfigError(ValueError):
    """A server-mode setting that would leave the server unsafe or unusable; it does not start."""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def parse_networks(raw: str) -> tuple[Network, ...]:
    networks = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            raise AuthConfigError(
                f"TURBOTAB_TRUSTED_PROXIES: {part!r} is not an IP address or network "
                "(for example 127.0.0.1 or 10.0.0.0/8)") from None
    return tuple(networks)


def _number(env: dict[str, str] | os._Environ[str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise AuthConfigError(f"{name} must be a number, not {raw!r}") from None
    if value <= 0:
        raise AuthConfigError(f"{name} must be positive, not {raw!r}")
    return value


@dataclass(frozen=True)
class AuthConfig:
    mode: AuthMode = "password"
    users_file: Path | None = None
    trusted_proxies: tuple[Network, ...] = ()
    proxy_header: str = "x-forwarded-user"
    secure_cookies: bool = False
    idle_seconds: float = 120 * 60
    max_seconds: float = 12 * 3600
    user_attempts: int = 5       # failed sign-ins per username per window
    ip_attempts: int = 20        # failed sign-ins per client address per window
    attempt_window: float = 15 * 60
    stream_recheck: float = 15.0  # how often an open event stream checks that its session lives
    clock: Callable[[], float] = field(default=time.monotonic, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in AUTH_MODES:
            raise AuthConfigError(f"TURBOTAB_AUTH must be 'password' or 'proxy', not {self.mode!r}")
        if self.mode == "proxy" and not self.trusted_proxies:
            raise AuthConfigError(
                "TURBOTAB_AUTH=proxy needs TURBOTAB_TRUSTED_PROXIES: the addresses of the proxy "
                "that signs users in. Without it any client could name itself in the header.")
        object.__setattr__(self, "proxy_header", self.proxy_header.strip().lower())
        if not self.proxy_header:
            raise AuthConfigError("TURBOTAB_PROXY_HEADER must name a header")

    @classmethod
    def from_env(cls, settings: Settings, environ: dict[str, str] | None = None) -> "AuthConfig":
        env = os.environ if environ is None else environ
        mode = (env.get("TURBOTAB_AUTH") or "password").strip().lower()
        users = (env.get("TURBOTAB_USERS") or "").strip()
        return cls(
            mode=mode,  # type: ignore[arg-type]
            users_file=Path(users).expanduser().absolute() if users else settings.home / "users.toml",
            trusted_proxies=parse_networks(env.get("TURBOTAB_TRUSTED_PROXIES") or ""),
            proxy_header=env.get("TURBOTAB_PROXY_HEADER") or "x-forwarded-user",
            secure_cookies=_truthy(env.get("TURBOTAB_SECURE_COOKIES")),
            idle_seconds=_number(env, "TURBOTAB_SESSION_IDLE_MINUTES", 120) * 60,
            max_seconds=_number(env, "TURBOTAB_SESSION_MAX_HOURS", 12) * 3600,
        )


# ── sessions and attempts ────────────────────────────────────────────────────


@dataclass
class Session:
    user: str
    credential: str  # the account's hash at sign-in: a new password or a removal ends the session
    created: float
    seen: float


class SessionStore:
    """Sessions by the SHA-256 of their id; the id itself is never kept."""

    def __init__(self, idle: float, max_age: float, clock: Callable[[], float]):
        self.idle = idle
        self.max_age = max_age
        self.clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()

    def _expired(self, session: Session, now: float) -> bool:
        return now - session.seen >= self.idle or now - session.created >= self.max_age

    def create(self, user: str, credential: str) -> str:
        token = secrets.token_urlsafe(32)  # 256 bits
        now = self.clock()
        with self._lock:
            for key in [k for k, s in self._sessions.items() if self._expired(s, now)]:
                del self._sessions[key]
            self._sessions[self._key(token)] = Session(user, credential, now, now)
        return token

    def get(self, token: str, *, touch: bool = True) -> Session | None:
        now = self.clock()
        key = self._key(token)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if self._expired(session, now):
                del self._sessions[key]
                return None
            if touch:
                session.seen = now
            return session

    def end(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(self._key(token), None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class Attempts:
    """Failed sign-ins per key in a sliding window; at ``limit`` the key waits."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float], max_keys: int = 10_000):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.max_keys = max_keys
        self._lock = threading.Lock()
        self._fails: OrderedDict[str, deque[float]] = OrderedDict()

    def _recent(self, key: str, now: float) -> deque[float]:
        fails = self._fails.get(key)
        if fails is None:
            return deque()
        while fails and now - fails[0] >= self.window:
            fails.popleft()
        if not fails:
            del self._fails[key]
        return fails

    def wait(self, key: str) -> float:
        """Seconds until ``key`` may try again; 0 when it may now."""
        now = self.clock()
        with self._lock:
            fails = self._recent(key, now)
            if len(fails) < self.limit:
                return 0.0
            return max(0.0, self.window - (now - fails[-self.limit]))

    def fail(self, key: str) -> None:
        now = self.clock()
        with self._lock:
            fails = self._recent(key, now)
            fails.append(now)
            self._fails[key] = fails
            self._fails.move_to_end(key)
            while len(self._fails) > self.max_keys:
                self._fails.popitem(last=False)

    def clear(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


# ── who is asking ────────────────────────────────────────────────────────────


def _headers(scope: Scope, wanted: bytes) -> list[str]:
    return [value.decode("latin-1").strip() for name, value in scope.get("headers") or ()
            if name == wanted]


def _header(scope: Scope, wanted: bytes) -> str | None:
    values = _headers(scope, wanted)
    return values[0] if values else None


def _cookies(scope: Scope) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in scope.get("headers") or ():
        if name != b"cookie":
            continue
        for part in value.decode("latin-1").split(";"):
            key, sep, val = part.strip().partition("=")
            if sep and key not in out:
                out[key] = val.strip()
    return out


class Auth:
    def __init__(self, settings: Settings, config: AuthConfig):
        self.settings = settings
        self.config = config
        self.users = UsersFile(config.users_file or default_path())
        self.sessions = SessionStore(config.idle_seconds, config.max_seconds, config.clock)
        self.by_user = Attempts(config.user_attempts, config.attempt_window, config.clock)
        self.by_address = Attempts(config.ip_attempts, config.attempt_window, config.clock)

    # addresses and https, believing forwarded headers only from a trusted proxy
    def trusted(self, address: str | None) -> bool:
        try:
            ip = ipaddress.ip_address(address or "")
        except ValueError:
            return False
        return any(ip in network for network in self.config.trusted_proxies)

    def peer(self, scope: Scope) -> str | None:
        client = scope.get("client")
        return str(client[0]) if client else None

    def client_address(self, scope: Scope) -> str:
        peer = self.peer(scope) or "unknown"
        if not self.trusted(peer):
            return peer
        joined = ",".join(_headers(scope, b"x-forwarded-for"))
        hops = [h.strip() for h in joined.split(",") if h.strip()]
        for hop in reversed(hops):  # the nearest hop a trusted proxy did not add itself
            if not self.trusted(hop):
                return hop
        return hops[0] if hops else peer

    def https(self, scope: Scope) -> bool:
        if scope.get("scheme") in ("https", "wss"):
            return True
        if self.trusted(self.peer(scope)):
            proto = (_header(scope, b"x-forwarded-proto") or "").split(",")[0].strip().lower()
            return proto == "https"
        return False

    def secure(self, scope: Scope) -> bool:
        return self.config.secure_cookies or self.https(scope)

    # identity
    def tokens(self, scope: Scope) -> list[str]:
        cookies = _cookies(scope)
        return [t for t in (cookies.get(SECURE_COOKIE), cookies.get(COOKIE)) if t]

    def session_user(self, token: str | None, *, touch: bool = True) -> str | None:
        if not token:
            return None
        session = self.sessions.get(token, touch=touch)
        if session is None:
            return None
        account = self.users.get(session.user)
        if account is None or account.hash != session.credential:
            self.sessions.end(token)  # removed, or given a new password, since signing in
            return None
        return session.user

    def proxy_user(self, scope: Scope) -> str | None:
        if not self.trusted(self.peer(scope)):
            return None  # the header means nothing from anyone but the proxy
        values = _headers(scope, self.config.proxy_header.encode("latin-1"))
        if len(values) != 1 or not values[0]:
            return None  # none, or a copy the browser sent beside the proxy's: whose is whose?
        try:
            return check_username(values[0].lower())
        except InvalidUsername:
            return None

    def identify(self, scope: Scope) -> tuple[str | None, str | None]:
        """(user, session token): the token only in password mode."""
        if self.config.mode == "proxy":
            return self.proxy_user(scope), None
        for token in self.tokens(scope):
            user = self.session_user(token)
            if user is not None:
                return user, token
        return None, None

    # signing in
    def sign_in(self, scope: Scope, username: str, password: str) -> tuple[str | None, str, float]:
        """(token or None, a message for the page, seconds to wait when refused for attempts)."""
        address = self.client_address(scope)
        name = username.strip().lower()
        wait = max(self.by_address.wait(address), self.by_user.wait(name) if name else 0.0)
        if wait > 0:
            minutes = max(1, int(-(-wait // 60)))
            return None, (f"Too many sign-in attempts for this account or from this address. Try "
                          f"again in {minutes} minute{'s' if minutes != 1 else ''}."), wait
        try:
            check_username(name)
            account = self.users.get(name)
        except InvalidUsername:
            account = None
        if account is None:
            burn_time(password)
            ok = False
        else:
            ok = verify_password(password, account.hash)
        if not ok or account is None:
            self.by_address.fail(address)
            if name:
                self.by_user.fail(name)
            return None, "That username and password do not match an account on this server.", 0.0
        self.by_user.clear(name)
        return self.sessions.create(name, account.hash), "", 0.0

    def cookie(self, scope: Scope, token: str) -> str:
        if self.secure(scope):
            return f"{SECURE_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Secure"
        return f"{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"

    def clear_cookies(self, response: Response) -> None:
        response.headers.append(
            "set-cookie", f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict")
        response.headers.append(
            "set-cookie", f"{SECURE_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict; Secure")


def current_user(request: Request) -> str:
    """The signed-in user the gate let through (server mode)."""
    user = request.scope.get("turbotab.user")
    if not user:  # the gate stands in front of every route, so this is a wiring fault
        raise ApiError(401, "unauthenticated", "Sign in to use TurboTab on this server.")
    return user


# ── the gate ─────────────────────────────────────────────────────────────────

UNAUTHENTICATED = ApiError(401, "unauthenticated", "Sign in to use TurboTab on this server.",
                           [{"label": "Sign in", "decision": None}])
PUBLIC_PATHS = frozenset({"/login", "/logout", "/healthz"})


class AuthGate:
    def __init__(self, app: ASGIApp, auth: Auth):
        self.app = app
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or "/"
        if path in PUBLIC_PATHS or path.startswith("/login/"):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":  # TurboTab has none; refuse any it does not know
            await send({"type": "websocket.close", "code": 1008})
            return
        method = scope.get("method", "GET")
        if method not in SAFE_METHODS and (_header(scope, b"sec-fetch-site") or "").lower() == "cross-site":
            body = ApiError(403, "cross_site", "This request came from a page on another website. "
                            "TurboTab takes changes only from its own pages.").body()
            await JSONResponse(body, status_code=403)(scope, receive, send)
            return
        user, token = self.auth.identify(scope)
        if user is None:
            await self._refuse(scope, receive, send, path, method)
            return
        scope["turbotab.user"] = user
        found = PROJECT_PATH.match(path)
        if found and found.group(1) != "upload" and not owns(self.auth.settings.home, user, found.group(1)):
            pid = found.group(1)
            body = ApiError(404, "unknown_project", f"There is no project {pid!r}.").body()
            await JSONResponse(body, status_code=404)(scope, receive, send)
            return
        if token and path.endswith("/events"):
            await self._watched(token, scope, receive, send)
            return
        await self.app(scope, receive, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send, path: str, method: str) -> None:
        api = path == "/api" or path.startswith("/api/")
        if api or method not in ("GET", "HEAD"):
            response: Response = JSONResponse(UNAUTHENTICATED.body(), status_code=401)
        elif self.auth.config.mode == "proxy":
            response = HTMLResponse(page(
                "Your sign-in did not reach TurboTab",
                "<p>This TurboTab signs people in through your institution, and the request "
                "arrived without a user name from it. Open TurboTab through your institution's "
                "address for it, or ask its administrator.</p>"), status_code=401)
        else:
            query = scope.get("query_string", b"").decode("latin-1")
            target = path + (f"?{query}" if query else "")
            response = RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
        response.headers["Cache-Control"] = "no-store"
        await response(scope, receive, send)

    async def _watched(self, token: str, scope: Scope, receive: Receive, send: Send) -> None:
        """An event stream that ends when its session does (sign-out, expiry, a new password)."""
        started = finished = False

        async def tracked(message: Message) -> None:
            nonlocal started, finished
            if message["type"] == "http.response.start":
                started = True
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                finished = True
            await send(message)

        async with anyio.create_task_group() as tg:
            async def watch() -> None:
                while True:
                    await anyio.sleep(self.auth.config.stream_recheck)
                    if self.auth.session_user(token, touch=False) is None:
                        tg.cancel_scope.cancel()
                        return

            tg.start_soon(watch)
            try:
                await self.app(scope, receive, tracked)
            finally:
                tg.cancel_scope.cancel()
        if started and not finished:
            try:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except Exception:  # noqa: BLE001 - the client may already be gone
                pass


# ── the pages ────────────────────────────────────────────────────────────────

# DESIGN_LANGUAGE §02 (tokens; light, dark, and the viewer's choice in both directions) and §03
# (the app speaks serif, the user acts sans). The page is outside the app's bundle, so it carries
# the few tokens it uses itself.
_STYLE = """
:root{color-scheme:light;--ground:#f7f8f6;--surface:#fff;--ink:#1c2b29;--muted:#5b6b68;
--line:#dce3e0;--accent:#0e7368;--accent-ink:#0a5a51;--on-accent:#fff;--surface-2:#eff3f1;
--serif:"Charter","Iowan Old Style","Source Serif Pro",Georgia,serif;
--sans:"Inter","Seravek","Avenir Next","Segoe UI",system-ui,-apple-system,sans-serif;
--mono:"JetBrains Mono",ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
--ground:#111917;--surface:#19231f;--ink:#e7eeeb;--muted:#9baba7;--line:#2a3733;
--accent:#45bfaf;--accent-ink:#6fd4c6;--on-accent:#08201c;--surface-2:#202c28}}
:root[data-theme=dark]{color-scheme:dark;--ground:#111917;--surface:#19231f;--ink:#e7eeeb;
--muted:#9baba7;--line:#2a3733;--accent:#45bfaf;--accent-ink:#6fd4c6;--on-accent:#08201c;
--surface-2:#202c28}
@font-face{font-family:"Inter";font-weight:300 900;font-display:swap;
src:url("/login/inter-latin.woff2") format("woff2")}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px 16px;
background:var(--ground);color:var(--ink);font:14px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
main{width:100%;max-width:380px}
.brand{font:700 16px var(--serif);margin:0 0 18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:26px 26px 22px}
h1{font:400 19px/1.35 var(--serif);margin:0 0 6px;text-wrap:balance}
.say{font:15px/1.5 var(--serif);color:var(--muted);margin:0 0 20px}
.note{font:15px/1.5 var(--serif);margin:0 0 18px;padding:2px 0 2px 12px;border-left:2px solid var(--line)}
label{display:block;font-weight:600;font-size:13px;margin:0 0 5px}
input{width:100%;font:inherit;font-size:14px;color:inherit;background:var(--surface);
border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin:0 0 14px}
button{width:100%;font:600 13.5px var(--sans);border:0;border-radius:8px;padding:10px 12px;
background:var(--accent);color:var(--on-accent);cursor:pointer;margin-top:4px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.foot{font-size:12.5px;color:var(--muted);margin:14px 2px 0}
.v{font-family:var(--mono);font-size:.82em;background:var(--surface-2);border:1px solid var(--line);
border-radius:5px;padding:0 5px;white-space:nowrap}
"""
_THEME = ('try{var t=localStorage.getItem("turbotab.theme");'
          'if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}')
_THEME_HASH = base64.b64encode(hashlib.sha256(_THEME.encode()).digest()).decode()
CSP = (f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{_THEME_HASH}'; "
       "font-src 'self'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; "
       "base-uri 'none'")
PAGE_HEADERS = {"Cache-Control": "no-store", "Content-Security-Policy": CSP,
                "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin",
                "X-Content-Type-Options": "nosniff"}


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)} · TurboTab</title><script>{_THEME}</script>"
            f"<style>{_STYLE}</style></head><body><main><p class=\"brand\">TurboTab</p>"
            f'<div class="card">{body}</div></main></body></html>')


def login_page(*, next_path: str, username: str = "", note: str = "", no_accounts: bool = False) -> str:
    note_html = f'<p class="note" role="alert">{html.escape(note)}</p>' if note else ""
    if no_accounts:
        foot = ('This server has no accounts yet. Its administrator adds one with '
                '<span class="v">python -m turbotab.server.users add &lt;name&gt;</span>.')
    else:
        foot = "Accounts are made by this server's administrator."
    return page("Sign in", f"""
<h1>Sign in to TurboTab</h1>
<p class="say">Your projects live in your own workspace on this server. No other account can open them.</p>
{note_html}
<form method="post" action="/login">
<input type="hidden" name="next" value="{html.escape(next_path)}">
<label for="username">Username</label>
<input id="username" name="username" autocomplete="username" autocapitalize="none" spellcheck="false" required value="{html.escape(username)}"{'' if username else ' autofocus'}>
<label for="password">Password</label>
<input id="password" name="password" type="password" autocomplete="current-password" required{' autofocus' if username else ''}>
<button type="submit">Sign in</button>
</form>
<p class="foot">{foot}</p>""")


def safe_next(raw: str | None) -> str:
    """A path on this server to return to after signing in; anything else is ``/``."""
    if not raw or not raw.startswith("/") or raw.startswith("//") or any(c in raw for c in "\\\r\n\t"):
        return "/"
    if urlsplit(raw).netloc or raw.split("?")[0] in ("/login", "/logout"):
        return "/"
    return raw


async def _form(request: Request) -> dict[str, str]:
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_FORM_BYTES:
            raise ApiError(413, "form_too_large", "That sign-in form is larger than any real one.")
    try:
        fields = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True, max_num_fields=8)
    except ValueError:
        raise ApiError(400, "bad_form", "That is not the sign-in form.") from None
    return {k: v[0] for k, v in fields.items() if v}


def install(app: FastAPI, settings: Settings, config: AuthConfig) -> Auth:
    """The sign-in routes and the gate in front of every route (server mode)."""
    auth = Auth(settings, config)
    app.state.auth = auth

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> Response:
        """Liveness for a container or a load balancer: no session needed, nothing disclosed."""
        return JSONResponse({"ok": True}, headers={"Cache-Control": "no-store"})

    @app.get("/login/{name}", include_in_schema=False)
    def login_font(name: str) -> Response:
        if name not in LOGIN_FONTS or not (FONTS / name).is_file():
            raise ApiError(404, "not_found", "No such file.")
        return FileResponse(FONTS / name, headers={"Cache-Control": "max-age=86400"})

    @app.get("/login", include_in_schema=False)
    def login_form(request: Request, next: str | None = None) -> Response:  # noqa: A002
        target = safe_next(next)
        if config.mode == "proxy":
            return RedirectResponse(target, status_code=303)
        user, _ = auth.identify(request.scope)
        if user is not None:
            return RedirectResponse(target, status_code=303)
        note = "You have signed out." if request.query_params.get("signed_out") else ""
        return HTMLResponse(login_page(next_path=target, note=note,
                                       no_accounts=not auth.users.accounts()), headers=PAGE_HEADERS)

    @app.post("/login", include_in_schema=False)
    async def login_submit(request: Request) -> Response:
        if config.mode == "proxy":
            raise ApiError(404, "not_found", "This server signs people in through its proxy.")
        form = await _form(request)
        target = safe_next(form.get("next"))
        username = form.get("username", "")[:128]
        token, note, wait = await run_in_threadpool(
            auth.sign_in, request.scope, username, form.get("password", "")[:1024])
        if token is None:
            status = 429 if wait else 401
            headers = dict(PAGE_HEADERS)
            if wait:
                headers["Retry-After"] = str(int(-(-wait // 1)))
            return HTMLResponse(login_page(next_path=target, username=username.strip().lower(),
                                           note=note, no_accounts=not auth.users.accounts()),
                                status_code=status, headers=headers)
        response = RedirectResponse(target, status_code=303, headers={"Cache-Control": "no-store"})
        response.headers.append("set-cookie", auth.cookie(request.scope, token))
        return response

    @app.post("/logout", include_in_schema=False)
    def logout(request: Request) -> Response:
        if config.mode == "proxy":
            return RedirectResponse("/", status_code=303)
        for token in auth.tokens(request.scope):
            auth.sessions.end(token)
        response = RedirectResponse("/login?signed_out=1", status_code=303,
                                    headers={"Cache-Control": "no-store"})
        auth.clear_cookies(response)
        return response

    app.add_middleware(AuthGate, auth=auth)
    return auth


def describe(auth: Auth) -> str:
    """One line for the console at start-up."""
    config = auth.config
    if config.mode == "proxy":
        nets = ", ".join(str(n) for n in config.trusted_proxies)
        return f"sign-in: through the proxy at {nets} (header {config.proxy_header})"
    count = len(auth.users.accounts())
    line = f"sign-in: password, {count} account{'s' if count != 1 else ''} in {auth.users.path}"
    if not count:
        line += " (add one: python -m turbotab.server.users add <name>)"
    return line
