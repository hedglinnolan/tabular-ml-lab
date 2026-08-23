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
}
```

**Apache example** (behind CAS/Shibboleth):
```apache
<Location />
    AuthType shibboleth
    ShibRequestSetting requireSession 1
    Require valid-user
    ProxyPass http://localhost:8501/
    ProxyPassReverse http://localhost:8501/
    RequestHeader set X-Remote-User %{REMOTE_USER}e
    RequestHeader set X-Remote-Email %{mail}e
    RequestHeader set X-Remote-Name %{displayName}e
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

Give the container as much memory as you can spare — `APP_MEMORY_LIMIT` in `.env`.
4 GB is enough for ordinary spreadsheet-sized studies; raise it for wide omics
files, where a single frame can be tens of thousands of columns.

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

### Out of memory during training
Give the container more memory in `.env`:
```bash
APP_MEMORY_LIMIT=8g
```
If the server cannot spare it, reduce the SHAP sample count and bootstrap
iterations on the Explainability and Train & Compare pages.

## Updating

The deployment files live on `main` alongside the app, so updating is an ordinary pull:

```bash
cd tabular-ml-lab
git pull origin main
docker compose up --build
```

Your `.env` file and any local configuration are preserved (they're in `.gitignore`).
