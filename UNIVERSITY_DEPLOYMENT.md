# University Deployment Guide

Deploy Tabular ML Lab on your institutional infrastructure with Docker.

## Quick Start (5 minutes)

```bash
git clone https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab
cp .env.example .env
docker compose up --build
```

The app is live at **http://localhost:8501**.

To include a local Ollama LLM sidecar:
```bash
docker compose --profile ollama up --build
# In another terminal, pull a model:
docker exec -it tabular-ml-lab-ollama-1 ollama pull qwen3.5:9b
```

## Who can reach the app

**The app has no logins of its own, and there is nothing in it to switch on.** Anyone who can reach the port can use it. That is the whole security model, so the only question is who you let reach the port.

### On a closed network

If the app sits on a lab network, a VPN, or a machine only your group can reach, that network *is* the access control. Nothing further to do.

### Behind your university's single sign-on (recommended for anything wider)

Your institution already runs a central login system — Shibboleth, CAS, Azure AD, Keycloak, OAuth2 Proxy. Put it in front of the container, like a doorman in front of an unlocked room: it checks the person at the door, and only then passes the request through. **All traffic** — pages, WebSockets, API calls — is gated before it reaches the app.

> **Why not a login screen inside the app?** Streamlit gives every page its own URL. A login gate on the landing page protects only that page; anyone can navigate straight to `/EDA` and skip it. A proxy gates every request whatever the URL, which is why this is the standard pattern rather than a shortcut.

There is **no app-side configuration** for this. You do not set anything in `.env`; you configure your proxy and point it at the container. Then remove the `ports:` mapping from the `app` service in `docker-compose.yml` and use `expose: ["8501"]` instead, so the app is reachable *only* through the proxy.

> **If you deployed an earlier version:** `.env.example` used to offer `AUTH_MODE=proxy` and a set of `AUTH_*` headers. Nothing in the app ever read them — setting them looked like protection and provided none. They have been removed rather than left to mislead. If you set them, your app was never gated by them; check that a proxy is actually in front of it.

**nginx example** (behind Shibboleth):
```nginx
location / {
    auth_request /shibboleth;
    proxy_pass http://localhost:8501;
    proxy_set_header X-Remote-User $upstream_http_x_remote_user;
    proxy_set_header X-Remote-Email $upstream_http_x_remote_email;
    proxy_set_header X-Remote-Name $upstream_http_x_remote_name;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    client_max_body_size 2000m;   # nginx's default is 1m — see below
    proxy_read_timeout 3600s;     # a fit or a SHAP run is minutes, not seconds
    proxy_send_timeout 3600s;
}
```

**Apache example** (behind CAS/Shibboleth):
```apache
# Server/vhost scope — ProxyTimeout is not permitted inside <Location>.
# Apache otherwise inherits Timeout (60s) and kills long analyses.
ProxyTimeout 3600

# Streamlit talks to the browser over a WebSocket, and Apache does not tunnel
# one through ProxyPass. Without this (and `a2enmod proxy_wstunnel`) the page
# loads and then nothing responds.
RewriteEngine On
RewriteCond %{HTTP:Upgrade} =websocket [NC]
RewriteRule /(.*) ws://localhost:8501/$1 [P,L]

<Location />
    AuthType shibboleth
    ShibRequestSetting requireSession 1
    Require valid-user
    ProxyPass http://localhost:8501/
    ProxyPassReverse http://localhost:8501/
    RequestHeader set X-Remote-User %{REMOTE_USER}e
    RequestHeader set X-Remote-Email %{mail}e
    RequestHeader set X-Remote-Name %{displayName}e

    LimitRequestBody 2097152000   # 2000 MB; overrides a site policy, see below
</Location>
```

**OAuth2 Proxy example** (for KeyCloak, Azure AD, or any OIDC provider):

If your institution doesn't already run an auth proxy, [OAuth2 Proxy](https://oauth2-proxy.github.io/oauth2-proxy/) is a lightweight container that handles OIDC authentication and can be added to your Docker Compose stack:

```yaml
# Add to docker-compose.yml
auth:
  image: quay.io/oauth2-proxy/oauth2-proxy:v7.7.1
  ports:
    - "4180:4180"
  environment:
    OAUTH2_PROXY_PROVIDER: keycloak-oidc  # or google, azure, oidc
    OAUTH2_PROXY_OIDC_ISSUER_URL: https://keycloak.your-university.edu/realms/your-realm
    OAUTH2_PROXY_CLIENT_ID: tabular-ml-lab
    OAUTH2_PROXY_CLIENT_SECRET: your-client-secret
    OAUTH2_PROXY_COOKIE_SECRET: $(python -c "import secrets; print(secrets.token_hex(32))")
    OAUTH2_PROXY_UPSTREAMS: http://app:8501
    OAUTH2_PROXY_HTTP_ADDRESS: 0.0.0.0:4180
    OAUTH2_PROXY_PROXY_WEBSOCKETS: "true"
    OAUTH2_PROXY_SET_XAUTHREQUEST: "true"
    OAUTH2_PROXY_EMAIL_DOMAINS: your-university.edu
```

Then remove the `ports` mapping from the `app` service (use `expose: ["8501"]` instead) so Streamlit is only reachable through the auth proxy.

The app never sees passwords or tokens, and does not show who is signed in — the proxy has already decided that before the request arrives. The `X-Remote-*` headers in the examples above are conventional and harmless to set; nothing in the app currently reads them.

### Proxy settings that decide whether the app is usable

A proxy that authenticates correctly can still make this app unusable, because
four proxy defaults are wrong for it. Each fails in a way that looks like an app
bug, which is why they are spelled out here rather than left to the reader.

**Request body size.** nginx rejects any request body over **1 MB** unless told
otherwise. Without `client_max_body_size`, the proxy answers an upload with its
own 413 before Streamlit sees a byte of the file — a ceiling 2,000x below what
the app admits, and stricter than a researcher running the app on their own
laptop. Apache is the opposite: `LimitRequestBody` defaults to unlimited, so the
line in the recipe above is not raising a default, it is overriding a site-wide
policy that many institutional configs do set. Whatever the proxy allows must be
at least `server.maxUploadSize` in `.streamlit/config.toml` (2000 MB as
shipped).

**Read timeout.** nginx gives up on a silent upstream after 60 s; Apache inherits
`Timeout`, also 60 s. A model fit, a SHAP run or a bootstrap on a wide frame runs
for minutes, and during that time the connection carries nothing — so the proxy
closes it and the browser shows a 504 while the container is still computing
perfectly happily. `proxy_read_timeout` / `ProxyTimeout` at an hour costs nothing
while nothing is running, and it is the difference between "the app crashes on
big datasets" and the app working.

**WebSockets.** Every interaction after page load travels over a WebSocket at
`/_stcore/stream`. nginx needs the `Upgrade`/`Connection` headers shown above;
Apache needs `mod_proxy_wstunnel` plus the rewrite shown above; OAuth2 Proxy
needs `OAUTH2_PROXY_PROXY_WEBSOCKETS: "true"`, which the example above sets.
Miss it and the page renders once and then never responds to anything.

**Session stickiness.** If you run more than one replica behind a load balancer,
turn on sticky sessions. A Streamlit session lives in one container's memory; a
reconnect routed to a different replica finds nothing, and the uploaded frame,
the fitted models and the audit trail are gone. Stickiness is also what makes
the raised `server.disconnectedSessionTTL` in `.streamlit/config.toml` (30
minutes, so a VPN blip does not destroy a session) worth anything at all.

## LLM Backend

The AI interpretation feature is optional. Without it, the app works fully — it just won't generate plain-language analysis summaries.

### Option 1: Local Ollama (bundled)

```bash
docker compose --profile ollama up --build
docker exec -it tabular-ml-lab-ollama-1 ollama pull qwen3.5:9b
```

Model recommendation by server RAM:

| Server RAM | Model | Pull command |
|------------|-------|-------------|
| 8 GB | `qwen3.5:1.5b` | `ollama pull qwen3.5:1.5b` |
| 16 GB | `qwen3.5:9b` | `ollama pull qwen3.5:9b` |
| 32 GB+ / GPU | `qwen3.5:32b` | `ollama pull qwen3.5:32b` |

### Option 2: External Ollama or vLLM

If your institution already runs an Ollama or vLLM server, point the app to it:

```bash
# .env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://your-ollama-server:11434

# Or for vLLM (OpenAI-compatible endpoint):
LLM_BACKEND=openai
OPENAI_API_KEY=not-needed
VLLM_BASE_URL=http://your-vllm-server:8000/v1
```

### Option 3: Cloud API

```bash
# .env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...

# Or:
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Option 4: Disabled

```bash
# .env
LLM_BACKEND=disabled
```

## Hardware

### The largest file this deployment can take is set in three places

Three limits decide it, and the smallest one wins silently. Raising one of them
alone can change nothing at all, which is what caught earlier deployments: the
app capped uploads at 50 MB, so adding RAM to the container moved nothing, and a
site behind an unconfigured nginx had a **stricter** ceiling than a researcher
running the app on a laptop. Set all three, and keep them consistent.

| Limit | Where | As shipped |
|-------|-------|-----------|
| Upload ceiling | `server.maxUploadSize` in `.streamlit/config.toml` | 2000 MB |
| Proxy body size | `client_max_body_size` (nginx) / `LimitRequestBody` (Apache) | 2000 MB in the recipes above; **1 MB** if you omit it from nginx |
| Container memory | `APP_MEMORY_LIMIT` in `.env` | 4 GB |

Give the container as much memory as you can spare. `APP_MEMORY_LIMIT=4g` is
enough for ordinary spreadsheet-sized studies; 8-16 GB is a reasonable
departmental default if you expect omics files.

### What that memory actually buys

Uploads are admitted on the parsed frame's **shape**, not on the file's size
(`utils/admission.py`) — bytes of text do not predict memory, because the same
matrix written at two different decimal precisions differs several-fold on disk
and not at all in RAM. The gate measures the parsed frame's real dtypes and
budgets a safety factor of 4 over that, for the copies made on the way to a
loaded frame. A **float64 matrix works out at about 32 bytes per cell**; a table
of text or categorical answers costs several times more per cell, and is
measured rather than assumed.

That ceiling is the **smaller** of two readings: the container's cgroup limit
less what the cgroup has already spent, and the host's own free memory as
`/proc/meminfo` reports it. Both matter, and which one binds depends on the
machine. On a dedicated host `APP_MEMORY_LIMIT` is the number that decides; on a
busy shared server the host's free memory can bind first, and the app will
refuse a frame that would have fitted inside its own limit — because the pages
to back it genuinely are not there.

Headroom is always *less* than `APP_MEMORY_LIMIT`: the interpreter and the
loaded libraries are several hundred megabytes and come out of the same budget,
which is why usage is subtracted rather than the limit being used whole. As a
rough figure, a 4 GB container admits float64 frames on the order of a hundred
million cells; 16 GB roughly quadruples that.

### Width is a separate limit, and a harder one

A frame wider than **2,000 columns** is admitted with a warning rather than
refused, and the warning is not a formality. The EDA and explainability pages
still do uncapped O(p²) work — 20,000 columns is 200 million column pairs, and
no amount of RAM makes that finish. Raising `APP_MEMORY_LIMIT` will let a very
wide frame *load*; it will not make the analysis pages return.

So if your study is a full expression matrix, subset to the features you
actually intend to model before uploading. This app is a good fit for cohort
tables and for targeted panels of hundreds to a couple of thousand features. It
is not, today, a viable workbench for a 20,000-gene matrix, and we would rather
say so here than have you find out on page 3.

Everything compute-intensive (SHAP, bootstrap confidence intervals, hyperparameter
search) is controlled by sliders inside the app, per analysis. There is no
server-side profile to set — a `COMPUTE_PROFILE` variable used to be documented
here and nothing read it.

## Troubleshooting

### App won't start
```bash
docker compose logs app
```
Common causes: port 8501 already in use, insufficient memory, missing .env file.

### People can reach the app without signing in
The app does not check anyone — that is the proxy's job. From a machine that has
NOT signed in, try to reach the app directly:
```bash
curl -I http://your-server:8501
```
If that answers, the app is reachable without going through your login system.
Remove the `ports:` mapping from the `app` service so only the proxy can reach it.

### Ollama model not loading
```bash
docker exec -it tabular-ml-lab-ollama-1 ollama list
```
If empty, pull a model: `docker exec -it tabular-ml-lab-ollama-1 ollama pull qwen3.5:9b`

If the Ollama container isn't running, check that you used `--profile ollama`:
```bash
docker compose --profile ollama up
```

### GPU not detected by Ollama
Ensure the NVIDIA Container Toolkit is installed:
```bash
nvidia-smi  # Should show your GPU
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi  # Should work in Docker
```

### Uploads fail with a 413, or fail before the app says anything
That refusal came from the proxy, not the app — the app's own refusal names the
file, its shape and the memory it needed. Check `client_max_body_size` (nginx)
or `LimitRequestBody` (Apache) against `server.maxUploadSize`; nginx's 1 MB
default is the usual culprit. See "Proxy settings that decide whether the app is
usable" above.

### A long analysis ends in a 504 while the container is still busy
The proxy closed an idle connection. Raise `proxy_read_timeout` (nginx) or
`ProxyTimeout` (Apache); both default to 60 s, and a fit or a SHAP run on a wide
frame is minutes.

### Out of memory during training
Give the container more memory in `.env`:
```bash
APP_MEMORY_LIMIT=8g
```
This is the limit the upload gate reads, via the container's cgroup rather than
the host's total RAM, so it is the one that matters even on a large server.
If the server cannot spare it, reduce the SHAP sample count and bootstrap
iterations on the Explainability and Train & Compare pages — and if the frame is
wide rather than tall, subset the columns instead: memory is not what is hurting
you (see "Width is a separate limit" above).

## Updating

The deployment files live on `main` alongside the app, so updating is an ordinary pull:

```bash
cd tabular-ml-lab
git pull origin main
docker compose up --build
```

Your `.env` file and any local configuration are preserved (they're in `.gitignore`).
