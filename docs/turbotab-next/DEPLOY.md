# Running TurboTab v2

Two ways, one app (V2_DEFINITION_OF_DONE §4, "Runs where researchers are"):

- **On your own computer**, with one double-click or one command. Nothing leaves the machine.
- **On a university server**, in Docker, with accounts and a workspace per user, when a laptop
  cannot hold the data.

Classic (the Streamlit app) has its own launchers, `Dockerfile` and `UNIVERSITY_DEPLOYMENT.md` at
the repository root. They are not these.

## On your own computer

| | macOS | Windows |
|---|---|---|
| Double-click | `turbotab/deploy/Start TurboTab.command` | `turbotab\deploy\Start TurboTab.bat` |
| One command | `bash turbotab/deploy/turbotab.sh` | `powershell -ExecutionPolicy Bypass -File turbotab\deploy\turbotab.ps1` |

With Python 3.12 or newer already installed, `python3 turbotab/deploy/launch.py` does the same on
any system. The first time, macOS may refuse a downloaded file: right-click it, choose **Open**,
then **Open** again. Windows may say it protected your PC: choose **More info**, then **Run
anyway**.

The first start sets TurboTab up, once:

1. It finds Python 3.12 or newer. If there is none, it fetches one with
   [uv](https://docs.astral.sh/uv/) into `~/.turbotab/tools`.
2. It makes a Python environment in `~/.turbotab/env` and installs
   `turbotab/server/requirements.txt` into it, at the versions in `turbotab/server/constraints.txt`
   (the ones the tests ran on). That takes a few minutes. It installs again only when either file
   changes.
3. It builds the interface into `turbotab/frontend/dist` if that folder is missing and Node.js
   20.19 or newer is installed. Without Node.js it says how to get it and starts anyway, and the
   page says how to build the interface.

After that, a start takes a few seconds. TurboTab opens in the browser at
`http://localhost:8787/`, or on another free port if 8787 is taken. Starting it again while it runs
opens the running one. To stop it, press Ctrl+C: the server and its job workers shut down
cleanly and free the port. Closing the window stops it too.

Options: `--port N`, `--no-open` (no browser), `--smoke` (start, check health, upload a small CSV,
stop; exits 0 when all of it worked). Settings: `TURBOTAB_HOME` (default `~/.turbotab`: projects,
uploads, the environment), `TURBOTAB_VENV`, `TURBOTAB_PYTHON`, `TURBOTAB_PORT`, and the server's
own `TURBOTAB_WORKERS` and `TURBOTAB_MEMORY_BUDGET`. To uninstall, delete `~/.turbotab` and this
folder.

On a Mac, the boosted-tree model families (LightGBM, XGBoost) need the OpenMP runtime. Without it
they are listed as unavailable, and everything else works. Install it with
`brew install libomp`.

## On a university server (Docker)

```sh
# 1. Build the image, from the repository root.
docker build -f turbotab/deploy/Dockerfile -t turbotab:2 .

# 2. Make accounts. The image runs as user 10001, which must be able to read the users file.
mkdir -p turbotab-config && sudo chown 10001:10001 turbotab-config
docker run --rm -it -v "$PWD/turbotab-config:/etc/turbotab" turbotab:2 \
    python -m turbotab.server.users add alice          # asks for the password twice
#   ... users passwd alice | users remove alice | users list | add bob --password-stdin

# 3. Run it, behind your TLS reverse proxy.
cp turbotab/deploy/docker-compose.example.yml docker-compose.turbotab.yml   # then adapt it
docker compose -f docker-compose.turbotab.yml up -d
```

The image has two stages: Node builds the interface, then a slim Python runs the server as a
non-root user. It has no R; the R reference implementations are only for tests. `/data` (a
volume) holds every user's workspace. `/etc/turbotab/users.toml` (mounted read-only) holds the
accounts. Mount the folder rather than the file, so that a users file rewritten on the host takes
effect without a restart. The container's health check calls `/healthz`, which needs no session
and reports nothing but "ok".

**Accounts and sessions.** Each account is a username with a `hashlib.scrypt` hash (N = 2^17,
r = 8, p = 1) and its own salt. A username is lowercase letters, digits and `. _ @ -`, so it can
never name a path. Signing in gives a session cookie: a random 256-bit id, `HttpOnly`,
`SameSite=Strict`, and `Secure` behind TLS. Behind TLS the cookie is named
`__Host-turbotab_session`, and TurboTab reads no other name. A page on a sibling subdomain can set a
plain-named cookie for the whole domain and so sign visitors in to its own account, but it cannot
set a `__Host-` cookie. TurboTab knows it is behind TLS from `TURBOTAB_SECURE_COOKIES=1` or from
the trusted proxy's `X-Forwarded-Proto`. Sessions are held in the server's memory. A session
ends:

- after 2 hours without a request (`TURBOTAB_SESSION_IDLE_MINUTES`);
- 12 hours after sign-in, however busy (`TURBOTAB_SESSION_MAX_HOURS`);
- at **Sign out** in the header;
- when its account is removed or given a new password;
- when the server restarts.

After 5 failed sign-ins for one account, or 20 from one address, within 15 minutes, further
attempts wait. Each attempt counts as it starts, so a burst sent at once is held to the same
limits. Passwords are checked two at a time, in threads of their own, so a burst of sign-ins never
slows the people already signed in. When 32 sign-ins are already waiting, the next one is told the
server is busy (503) and to try again in a few seconds, and that attempt does not count. TurboTab takes a change (a sign-in, a sign-out, an upload, a run) only from its own pages.
A browser request that a page on another site or on a sibling subdomain sent is refused.

**Workspaces.** Each user works in `/data/users/<name>/`, and no user can list, open or change
another user's project. Every route that names a project checks that the project belongs to the
signed-in user. Every `/api` route needs a session, including the event stream, uploads, previews
and export downloads. An open event stream ends when its session does. The job workers
(`TURBOTAB_WORKERS`) are shared by all users. Sharing projects between users is not in v2.

**TLS.** TurboTab speaks plain HTTP. Publish its port on `127.0.0.1` only, as the example does, and
let the reverse proxy terminate TLS. Set `TURBOTAB_TRUSTED_PROXIES` to the proxy's address as the
container sees it. The access log's client column shows that address. The example gives TurboTab
a network of its own, `172.31.87.0/24`, so a proxy on the host arrives from its gateway,
`172.31.87.1`, which the example trusts. TurboTab then believes the proxy's `X-Forwarded-For` (the
address the sign-in limits count) and `X-Forwarded-Proto` (https), and nobody else's. Every other
process on the host also arrives from the gateway. In password mode that lets them choose the
address the per-address limit counts and say a request was https, and nothing more: the
per-account limit still holds. Keep the `Host` line below. A browser too old to send `Sec-Fetch-Site` is checked by comparing its
`Origin` with the `Host`. An nginx example:

```nginx
location / {
    proxy_pass http://127.0.0.1:8787;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_buffering off;        # the event stream
    proxy_read_timeout 1h;
    client_max_body_size 0;     # uploads stream to disk; set a limit here if you want one
}
```

**Single sign-on.** Set `TURBOTAB_AUTH=proxy` to let the institution's SSO proxy (Shibboleth, CAS,
Keycloak, oauth2-proxy and the like) sign people in. It names the user in a header,
`TURBOTAB_PROXY_HEADER` (default `X-Forwarded-User`). TurboTab reads that header only on requests
whose peer address is in `TURBOTAB_TRUSTED_PROXIES`, and it refuses to start in proxy mode without
that list. Whatever can connect from a trusted address can sign in as anyone, so list the proxy's
exact address, not a range other machines share. TurboTab refuses to start in proxy mode if a
trusted network holds more than 256 addresses (`0.0.0.0/0`, a campus range, a Docker network).

A proxy on the same host that reaches TurboTab through a published port cannot be told apart from
any other process or container on that host: they all arrive from Docker's gateway. So in proxy
mode, publish no port for TurboTab. Run the SSO proxy as a container on TurboTab's network with a
fixed address, list that address alone, and let the proxy face the users. The end of
`docker-compose.example.yml` shows the changes. Without Docker, a proxy on `127.0.0.1` has the same
problem: every local user's programs arrive from `127.0.0.1` too, so use that only on a host no one
else signs in to.

The proxy must set the header on every request and overwrite any copy a browser sent. Names are
lowercased, and a name that could form a path is refused. There is no password and no sign-out in
TurboTab in this mode; both belong to the SSO.

**Without Docker:** `TURBOTAB_HOME=/srv/turbotab TURBOTAB_USERS=/etc/turbotab/users.toml python -m
turbotab.server --mode server --host 127.0.0.1 --port 8787`, in an environment with
`turbotab/server/requirements.txt` installed (`pip install -c turbotab/server/constraints.txt -r
turbotab/server/requirements.txt`), behind the same proxy. The constraints file's header says how
to update the versions.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `TURBOTAB_MODE` | `local` (`server` in the image) | server mode signs users in and keeps a workspace per user |
| `TURBOTAB_HOME` | `~/.turbotab` (`/data` in the image) | where workspaces live |
| `TURBOTAB_USERS` | `$TURBOTAB_HOME/users.toml` (`/etc/turbotab/users.toml`) | the accounts file |
| `TURBOTAB_AUTH` | `password` | or `proxy` (single sign-on) |
| `TURBOTAB_TRUSTED_PROXIES` | none | addresses or networks whose forwarded headers are believed; required for `proxy`, where none may hold more than 256 addresses |
| `TURBOTAB_PROXY_HEADER` | `X-Forwarded-User` | the header naming the user in `proxy` mode |
| `TURBOTAB_SECURE_COOKIES` | off | `1`: the session cookie is sent over https only (it is anyway when the request is https) |
| `TURBOTAB_SESSION_IDLE_MINUTES` | 120 | a session ends after this long without a request |
| `TURBOTAB_SESSION_MAX_HOURS` | 12 | a session ends this long after sign-in |
| `TURBOTAB_WORKERS` | the least of: cores minus one, GB of RAM / 4, and 4 | job worker processes, shared by all users |
| `TURBOTAB_MEMORY_BUDGET` | half the memory free at start | the most memory the data a fit materializes may take |

## Checked by CI

CI has two tiers: `.github/workflows/v2.yml` (fast) and `.github/workflows/v2-full.yml` (full).
Every Python library installs at the version in `turbotab/server/constraints.txt` (its header
says how to update the file). `turbotab/core/tests/test_ci_tiers.py` checks the split below.

**Fast**, on every push to `turbotab-next` or `ci/**` and every pull request to `main`, in under
ten minutes. It runs `npm run check` (type checks, lint, vitest) and the core and server tests
except the acceptance suite (`turbotab/core/tests/acceptance`) and the few tests marked `slow`
(over 35 seconds each), in parallel shards. The guarantee tests stay in it: no held-out row
reaches a fitted step, the cross-validated scores equal scikit-learn's own, and no held-out score
is served before the seal is opened. It builds the image, makes an account, and
checks that requests without a session get 401. It then signs in over HTTP, checks
`/api/health`, uploads a CSV and waits until it is read. It also starts the launcher on macOS and
Windows (Python 3.12), twice each: once to set up, check health and an upload, and stop cleanly;
once more to show that the second start skips the setup. On Windows it then runs the numerics
check (the audit's §3.4): the prediction results of every model family reproduce the committed
references to 1e-9, and the elastic net chooses the same penalty as on Linux.

**Full**, on every push to `turbotab-next` or to a branch named `ci/full*`, on every pull request
to `main`, and nightly at 06:23 UTC on `turbotab-next`: the whole core and server suite,
acceptance tests included (50 minutes on CI), and the mock browser tests (Playwright). GitHub
reads a schedule, and shows "Run workflow" in the Actions tab, only for workflows on the default
branch (`main`), so the nightly and "Run workflow" start once `v2-full.yml` is on `main`.
