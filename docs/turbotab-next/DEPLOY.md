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
   `turbotab/server/requirements.txt` into it. That takes a few minutes. It installs again only
   when that file changes.
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
`SameSite=Strict`, and `Secure` behind TLS. Sessions are held in the server's memory. A session
ends:

- after 2 hours without a request (`TURBOTAB_SESSION_IDLE_MINUTES`);
- 12 hours after sign-in, however busy (`TURBOTAB_SESSION_MAX_HOURS`);
- at **Sign out** in the header;
- when its account is removed or given a new password;
- when the server restarts.

After 5 failed sign-ins for one account, or 20 from one address, within 15 minutes, further
attempts wait.

**Workspaces.** Each user works in `/data/users/<name>/`, and no user can list, open or change
another user's project. Every route that names a project checks that the project belongs to the
signed-in user. Every `/api` route needs a session, including the event stream, uploads, previews
and export downloads. An open event stream ends when its session does. The job workers
(`TURBOTAB_WORKERS`) are shared by all users. Sharing projects between users is not in v2.

**TLS.** TurboTab speaks plain HTTP. Publish its port on `127.0.0.1` only, as the example does, and
let the reverse proxy terminate TLS. Set `TURBOTAB_TRUSTED_PROXIES` to the proxy's address as the
container sees it. The access log's client column shows that address (often Docker's bridge
gateway, such as 172.17.0.1). TurboTab then believes the proxy's `X-Forwarded-For` (the address
the sign-in limits count) and `X-Forwarded-Proto` (https), and nobody else's. An nginx example:

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
that list. List the proxy's exact address, not a range other machines share. The proxy must set
the header on every request and overwrite any copy a browser sent. Names are lowercased, and a
name that could form a path is refused. There is no password and no sign-out in TurboTab in this
mode; both belong to the SSO.

**Without Docker:** `TURBOTAB_HOME=/srv/turbotab TURBOTAB_USERS=/etc/turbotab/users.toml python -m
turbotab.server --mode server --host 127.0.0.1 --port 8787`, in an environment with
`turbotab/server/requirements.txt` installed, behind the same proxy.

### Settings

| Variable | Default | Meaning |
|---|---|---|
| `TURBOTAB_MODE` | `local` (`server` in the image) | server mode signs users in and keeps a workspace per user |
| `TURBOTAB_HOME` | `~/.turbotab` (`/data` in the image) | where workspaces live |
| `TURBOTAB_USERS` | `$TURBOTAB_HOME/users.toml` (`/etc/turbotab/users.toml`) | the accounts file |
| `TURBOTAB_AUTH` | `password` | or `proxy` (single sign-on) |
| `TURBOTAB_TRUSTED_PROXIES` | none | addresses or networks whose forwarded headers are believed; required for `proxy` |
| `TURBOTAB_PROXY_HEADER` | `X-Forwarded-User` | the header naming the user in `proxy` mode |
| `TURBOTAB_SECURE_COOKIES` | off | `1`: the session cookie is sent over https only (it is anyway when the request is https) |
| `TURBOTAB_SESSION_IDLE_MINUTES` | 120 | a session ends after this long without a request |
| `TURBOTAB_SESSION_MAX_HOURS` | 12 | a session ends this long after sign-in |
| `TURBOTAB_WORKERS` | the least of: cores minus one, GB of RAM / 4, and 4 | job worker processes, shared by all users |
| `TURBOTAB_MEMORY_BUDGET` | half the memory free at start | the most memory the data a fit materializes may take |

## Checked by CI

`.github/workflows/v2.yml` runs on every push to `turbotab-next` and every pull request to `main`.
It runs the core and server tests and `npm run check`. It builds the image, makes an account, and
checks that requests without a session get 401. It then signs in over HTTP, checks
`/api/health`, uploads a CSV and waits until it is read. It also starts the launcher on macOS and
Windows (Python 3.12), twice each: once to set up, check health and an upload, and stop cleanly;
once more to show that the second start skips the setup.
