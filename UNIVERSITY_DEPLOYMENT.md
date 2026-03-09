# University Deployment Guide

This guide helps universities deploy Tabular ML Lab on their own infrastructure with Docker containerization, KeyCloak OIDC authentication, and institutional LLM integration.

---

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+
- KeyCloak instance with OIDC client (admin access required to create one)
- Institutional LLM endpoint — vLLM (recommended) or Ollama
- Git access to this repository

---

## Quick Start for IT Administrators

### Step 1: Clone and Configure

```bash
# Clone this branch
git clone -b university-docker https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Create environment configuration
cp .env.example .env
nano .env
```

### Step 2: Configure Environment

Edit `.env` file:

```bash
# LLM Backend — vLLM is recommended (OpenAI-compatible API)
VLLM_URL=http://your-vllm-server.university.edu:8000
VLLM_MODEL=          # Leave blank to auto-detect loaded model

# Ollama (legacy fallback — only needed if using Ollama instead of vLLM)
OLLAMA_URL=

# Authentication
AUTH_ENABLED=true

# Compute Profile — Adjust based on your hardware
# Options: standard, high_performance, enterprise
COMPUTE_PROFILE=high_performance

# Streamlit (defaults work for most setups)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Step 3: Configure KeyCloak OIDC Authentication

#### 3a. Create a KeyCloak OIDC Client

In the KeyCloak admin console:

1. **Select your realm** (or create one)
2. Navigate to **Clients → Create Client**
3. Set:
   - **Client type:** OpenID Connect
   - **Client ID:** `tabular-ml-lab` (or any name)
4. Under **Settings:**
   - **Valid Redirect URIs:** `https://YOUR_APP_URL/oauth2callback`
   - **Web Origins:** `https://YOUR_APP_URL` (or `*` for testing)
5. Under **Credentials:**
   - Copy the **Client Secret**
6. Note the **OIDC Discovery URL:**
   ```
   https://YOUR_KEYCLOAK_DOMAIN/auth/realms/YOUR_REALM/.well-known/openid-configuration
   ```

#### 3b. Configure Streamlit Secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml
```

Fill in the values from KeyCloak:

```toml
[auth]
redirect_uri = "https://ml-lab.university.edu/oauth2callback"
cookie_secret = "your-random-secret-string"

[auth.keycloak]
client_id = "tabular-ml-lab"
client_secret = "your-client-secret-from-keycloak"
server_metadata_url = "https://keycloak.university.edu/auth/realms/myrealm/.well-known/openid-configuration"
```

Generate the cookie secret:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

> **⚠️ Never commit secrets.toml to version control.** It is already in `.gitignore`.

### Step 4: Choose Compute Profile

See [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for detailed performance guidance.

**Quick recommendations:**
- **Consumer GPUs** (GTX 1080 Ti, RTX 3060): `COMPUTE_PROFILE=standard`
- **Professional GPUs** (A6000, A100): `COMPUTE_PROFILE=high_performance`
- **Multi-GPU Cluster**: `COMPUTE_PROFILE=enterprise`

### Step 5: Build and Deploy

```bash
# Build Docker image
docker build -t tabular-ml-lab:latest .

# Run with Docker Compose (recommended)
docker-compose up -d

# Or run standalone container
docker run -d \
  --name tabular-ml-lab \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  tabular-ml-lab:latest

# Verify deployment
docker logs tabular-ml-lab
curl http://localhost:8501/_stcore/health
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│  Student/Faculty Browser                               │
│  (clicks "Sign in with KeyCloak")                      │
└────────────────────────────────────────────────────────┘
              │                          ▲
              │  OIDC redirect           │  ID token
              ▼                          │
┌────────────────────────────────────────────────────────┐
│  KeyCloak (Identity Broker)                            │
│  - Federates with Entra ID / SAML2 / LDAP / etc.      │
│  - Issues OIDC tokens to Streamlit                     │
└────────────────────────────────────────────────────────┘
              │                          ▲
              │  SAML2 / OIDC            │  Assertion
              ▼                          │
┌────────────────────────────────────────────────────────┐
│  Institutional Identity Provider                       │
│  (Microsoft Entra ID, Shibboleth, Okta, etc.)          │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Docker Container (Tabular ML Lab)                     │
│  - st.login("keycloak") handles OIDC flow natively     │
│  - Session-isolated analysis workflows                 │
│  - User identity available for audit logging           │
└────────────────────────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────┐
│  LLM Backend (vLLM recommended)                        │
│  - OpenAI-compatible API                               │
│  - No external API calls — fully on-premises           │
└────────────────────────────────────────────────────────┘
```

---

## LLM Backend Configuration

### Option A: vLLM (Recommended)

vLLM provides an OpenAI-compatible API and typically hosts more capable models than Ollama.

```bash
VLLM_URL=http://your-vllm-server.university.edu:8000
VLLM_MODEL=          # Auto-detects the loaded model if blank
```

The app connects to `{VLLM_URL}/v1/chat/completions`. No API key is required by default.

**Verify connectivity:**
```bash
curl http://your-vllm-server:8000/v1/models
```

### Option B: Ollama (Legacy)

If your institution runs Ollama:

```bash
OLLAMA_URL=http://ollama.cs.university.edu:11434
VLLM_URL=             # Leave blank to disable vLLM
```

Users can switch between backends in the sidebar LLM Settings panel.

**To start the bundled Ollama (for testing only):**
```bash
docker-compose --profile ollama up -d
docker exec -it ollama-backend ollama pull llama3.1:8b
```

---

## Deployment Options

### Option A: Docker Compose (Recommended)

Best for: Coordinating app + services, standard deployments

```bash
docker-compose up -d
```

### Option B: Docker (Simplest)

Best for: Single-server, custom orchestration

```bash
docker run -d -p 8501:8501 --env-file .env \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  tabular-ml-lab
```

### Option C: Kubernetes (Production)

Best for: Large-scale, high availability, auto-scaling

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete Kubernetes manifests.

Mount `secrets.toml` as a Kubernetes Secret:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: tabular-ml-secrets
type: Opaque
stringData:
  secrets.toml: |
    [auth]
    redirect_uri = "https://ml-lab.university.edu/oauth2callback"
    cookie_secret = "..."
    [auth.keycloak]
    client_id = "..."
    client_secret = "..."
    server_metadata_url = "..."
```

---

## Security Considerations

✅ **Already implemented:**
- Non-root container user
- OIDC authentication (no credentials stored in app)
- Session-only data storage (no persistence by default)
- Health check endpoint for monitoring

⚠️ **Your responsibility:**
- HTTPS termination (TLS certificate on load balancer or reverse proxy)
- Keep `secrets.toml` secure (restrict file permissions, don't commit to git)
- Firewall rules (restrict container network access)
- Regular Docker image updates
- KeyCloak client configuration review

---

## Testing Your Deployment

### 1. Health Check

```bash
curl http://localhost:8501/_stcore/health
# Expected: {"status": "ok"}
```

### 2. Authentication Test

1. Open `https://ml-lab.university.edu` in a browser
2. You should see the "Sign in with KeyCloak" button
3. Click it — you'll be redirected to KeyCloak → your IdP login
4. After login, you should see the app with your name in the sidebar

### 3. LLM Connectivity

```bash
# vLLM
curl http://your-vllm-server:8000/v1/models

# Ollama (if using)
curl http://your-ollama-server:11434/api/tags
```

### 4. Full Integration Test

1. Access the app through your deployment URL
2. Sign in with institutional credentials
3. Upload a CSV file
4. Run through the EDA page
5. Try LLM interpretation (sidebar → LLM Settings → Interpret with AI)

---

## Troubleshooting

### "Sign in with KeyCloak" doesn't work

**Possible causes:**
- `secrets.toml` not mounted or not readable
- `redirect_uri` doesn't match what's registered in KeyCloak
- `server_metadata_url` is unreachable from the container
- `authlib` not installed (check `pip list | grep Authlib` in container)

**Debug:**
```bash
# Check secrets are mounted
docker exec tabular-ml-lab cat /app/.streamlit/secrets.toml

# Check OIDC discovery is reachable from container
docker exec tabular-ml-lab curl -s https://YOUR_KEYCLOAK/auth/realms/YOUR_REALM/.well-known/openid-configuration

# Check app logs
docker logs tabular-ml-lab | grep -i auth
```

### Redirect URI Mismatch

KeyCloak returns "Invalid parameter: redirect_uri":
- The `redirect_uri` in `secrets.toml` must **exactly** match what's registered in KeyCloak's Valid Redirect URIs
- Must be `https://YOUR_APP_URL/oauth2callback` (with the `/oauth2callback` path)
- Check for trailing slashes or http vs https mismatch

### LLM Interpretation Not Working

**vLLM:**
```bash
# Test from inside container
docker exec tabular-ml-lab curl http://your-vllm-server:8000/v1/models
```

**Ollama:**
```bash
docker exec tabular-ml-lab curl http://your-ollama-server:11434/api/tags
```

**Common issues:**
- Firewall blocking the port
- DNS not resolving from inside Docker network
- No model loaded on the vLLM server

### Slow Performance

1. Check compute profile: `docker exec tabular-ml-lab env | grep COMPUTE`
2. Consider lowering profile: `COMPUTE_PROFILE=standard`
3. Monitor resources: `docker stats tabular-ml-lab`

**See [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for benchmarking guide.**

---

## Compute Profile Optimization

**Set in `.env`:**
```bash
COMPUTE_PROFILE=enterprise  # For powerful clusters
```

| Profile | PDP Samples | SHAP Evals | Optuna Trials | Best For |
|---------|-------------|------------|---------------|----------|
| standard | 2,000 | 50 | 30 | Consumer GPUs, laptops |
| high_performance | 10,000 | 200 | 50 | A6000, A100 single GPU |
| enterprise | 50,000 | 500 | 100 | Multi-GPU clusters |

**Full details:** [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md)

---

## Monitoring

### Resource Usage

```bash
docker stats tabular-ml-lab
docker logs -f tabular-ml-lab
```

### User Activity

With OIDC, user identity is available in the app. Enable access logging on your HTTPS proxy to track usage.

---

## Updating the Application

```bash
cd tabular-ml-lab
git pull origin university-docker
docker build -t tabular-ml-lab:latest .
docker-compose down && docker-compose up -d
```

---

## Scaling for Large Classes

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

### Resource Limits (Kubernetes)

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

---

## Differences from Public Version

This university-docker branch includes:
- ✅ Docker/Kubernetes deployment configs
- ✅ KeyCloak OIDC authentication (any IdP: Entra ID, Shibboleth, Okta, etc.)
- ✅ vLLM + Ollama institutional LLM integration
- ✅ Configurable compute profiles
- ❌ Removed Anthropic/Claude API (on-premises only)

The main branch at https://github.com/hedglinnolan/tabular-ml-lab includes all features for personal/research use.

---

## Support

- GitHub Issues: https://github.com/hedglinnolan/tabular-ml-lab/issues
- [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) — Detailed deployment configs
- [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) — Performance tuning

---

## License

MIT License — Free for educational and research use. See [LICENSE](LICENSE) for full terms.
