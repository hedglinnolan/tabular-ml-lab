# University Deployment Guide

This guide helps universities deploy Tabular ML Lab on their own infrastructure with Docker containerization, flexible authentication, and institutional LLM integration.

**Designed to work with YOUR infrastructure** — supports multiple authentication providers (OIDC, SAML, reverse proxy) and LLM backends (vLLM, Ollama, OpenAI).

---

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+
- Authentication infrastructure (choose one):
  - **Option A:** OIDC provider (KeyCloak, Azure AD, Google Workspace, Okta, etc.)
  - **Option B:** SAML provider
  - **Option C:** Reverse proxy with existing auth (nginx/Apache + LDAP/AD)
- LLM backend (choose one):
  - **Option A (Recommended):** vLLM endpoint (OpenAI-compatible API)
  - **Option B:** Ollama endpoint
  - **Option C:** OpenAI API key (for testing/small deployments)
- Git access to this repository

---

## Quick Start

```bash
# Clone this branch
git clone -b university-docker https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Configure based on your infrastructure (see sections below)
cp .env.example .env
nano .env

# Start the application
docker-compose up -d
```

Access at `http://localhost:8501` (or your configured domain).

---

## Configuration Options

### 1. Authentication Setup

Choose the authentication method that matches your institution's infrastructure:

#### Option A: OIDC Providers (KeyCloak, Azure AD, Google, Okta)

**Supported providers:** KeyCloak, Azure AD, Google Workspace, Okta, Auth0, or any OIDC-compliant provider.

**Step 1:** Create OIDC client in your provider's admin console.

**Step 2:** Configure `.streamlit/secrets.toml`:

```toml
# KeyCloak example
[connections.keycloak]
provider = "keycloak"
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
discovery_url = "https://your-keycloak.edu/realms/YOUR_REALM/.well-known/openid-configuration"
redirect_uri = "https://your-app.edu/_stcore/login/keycloak"

# Azure AD example
[connections.azure]
provider = "azure"
client_id = "YOUR_APP_ID"
client_secret = "YOUR_CLIENT_SECRET"
tenant_id = "YOUR_TENANT_ID"
redirect_uri = "https://your-app.edu/_stcore/login/azure"

# Google Workspace example
[connections.google]
provider = "google"
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
redirect_uri = "https://your-app.edu/_stcore/login/google"
```

**Step 3:** Set environment variable:

```bash
# In .env
AUTH_ENABLED=true
AUTH_PROVIDER=keycloak  # or 'azure', 'google', 'okta'
```

#### Option B: SAML Provider

**Step 1:** Configure `.streamlit/secrets.toml`:

```toml
[connections.saml]
provider = "saml"
entity_id = "https://your-app.edu"
sso_url = "https://your-idp.edu/saml/sso"
x509cert = "YOUR_IDP_CERTIFICATE"
```

**Step 2:** Set environment variable:

```bash
# In .env
AUTH_ENABLED=true
AUTH_PROVIDER=saml
```

#### Option C: Reverse Proxy (Legacy)

If your institution already has nginx/Apache handling authentication:

**Step 1:** Configure reverse proxy to pass `X-Remote-User` header or redirect with `?user=<username>`.

**Step 2:** Set environment variables:

```bash
# In .env
AUTH_ENABLED=true
AUTH_PROVIDER=reverseproxy
```

**Note:** This method is legacy and less secure than OIDC/SAML. Upgrade to OIDC if possible.

#### Option D: Disable Authentication (Development Only)

```bash
# In .env
AUTH_ENABLED=false
```

**⚠️ WARNING:** Only use this for local development. NEVER in production.

---

### 2. LLM Backend Setup

Choose the LLM backend that matches your institution's infrastructure:

#### Option A: vLLM (Recommended for Universities)

**Best for:** Institutions with GPU infrastructure (A100, A6000, H100, etc.).

**Configuration:**

```bash
# In .env
VLLM_URL=http://your-vllm-server.edu:8000
VLLM_MODEL=              # Leave blank to auto-detect loaded model
```

**Advantages:**
- OpenAI-compatible API
- Institutional control over models
- High throughput on GPUs
- Auto-detects loaded model (no manual config needed)

**In docker-compose.yml:** Point `vllm` service to your institutional endpoint, or run vLLM locally (see vLLM section below).

#### Option B: Ollama

**Best for:** Smaller institutions or CPU-only deployments.

**Configuration:**

```bash
# In .env
OLLAMA_URL=http://your-ollama-server.edu:11434
```

**In docker-compose.yml:** Enable Ollama profile:

```bash
docker-compose --profile ollama up -d
```

**Advantages:**
- Easy to deploy on CPU
- Good for moderate usage
- No GPU required

**Disadvantages:**
- Slower inference than vLLM on GPUs
- Limited batch processing

#### Option C: OpenAI API (Testing Only)

**Best for:** Small-scale testing or pilot deployments.

**Configuration:** Users configure their own API key via the app sidebar (🤖 LLM Settings).

**⚠️ Limitations:**
- **Cost:** Pay-per-token pricing
- **Privacy:** Data sent to OpenAI (not on-premises)
- **NOT recommended for institutional deployment**

---

### 3. Compute Profile

Adjust compute limits based on your hardware:

```bash
# In .env
COMPUTE_PROFILE=standard  # Options: standard, high_performance, enterprise
```

| Profile | PDP Samples | SHAP Evals | Optuna Trials | Hardware |
|---------|-------------|------------|---------------|----------|
| `standard` | 2,000 | 50 | 30 | Laptop/workstation |
| `high_performance` | 10,000 | 200 | 50 | Beefy server |
| `enterprise` | 50,000 | 500 | 100 | A6000+ GPUs, 2TB RAM |

See **[COMPUTE_PROFILES.md](COMPUTE_PROFILES.md)** for details.

---

## Running vLLM Locally (Optional)

If you don't have an institutional vLLM endpoint, you can run it via Docker:

**Step 1:** Uncomment vLLM service in `docker-compose.yml`:

```yaml
vllm:
  image: vllm/vllm-openai:latest
  container_name: vllm-backend
  ports:
    - "8000:8000"
  environment:
    - VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
  volumes:
    - vllm-models:/root/.cache/huggingface
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  command:
    - --model=meta-llama/Llama-3.1-8B-Instruct
    - --port=8000
    - --host=0.0.0.0
```

**Step 2:** Start with GPU support:

```bash
docker-compose up -d
```

**Requires:** NVIDIA GPU + nvidia-docker runtime.

---

## Production Deployment Checklist

- [ ] **HTTPS enabled** (TLS certificate via Let's Encrypt or institutional CA)
- [ ] **Authentication configured** (OIDC/SAML preferred over reverse proxy)
- [ ] **LLM backend tested** (can reach vLLM/Ollama endpoint)
- [ ] **Compute profile set** (matches available hardware)
- [ ] **Secrets secured** (`.streamlit/secrets.toml` has restricted file permissions)
- [ ] **Firewall rules** (only necessary ports exposed)
- [ ] **Backup strategy** (container volumes backed up)
- [ ] **Monitoring** (health checks configured)
- [ ] **Resource limits** (Docker memory/CPU limits set)

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `false` | Enable authentication |
| `AUTH_PROVIDER` | `reverseproxy` | Auth provider: `keycloak`, `azure`, `google`, `saml`, `reverseproxy` |
| `VLLM_URL` | `http://vllm:8000` | vLLM endpoint (OpenAI-compatible) |
| `VLLM_MODEL` | _(auto-detect)_ | Model name (leave blank to auto-detect) |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint |
| `COMPUTE_PROFILE` | `standard` | Compute limits: `standard`, `high_performance`, `enterprise` |
| `STREAMLIT_SERVER_PORT` | `8501` | App port |

---

## Troubleshooting

### Authentication Issues

**Problem:** "Authentication Required" page loops.

**Solution:**
1. Check `.streamlit/secrets.toml` exists and has correct provider config
2. Verify `AUTH_PROVIDER` env var matches provider in `secrets.toml`
3. Check redirect URI matches in both provider admin console and `secrets.toml`

### LLM Backend Issues

**Problem:** "Could not get interpretation from vLLM" error.

**Solution:**
1. Test vLLM endpoint directly:
   ```bash
   curl http://your-vllm-server:8000/v1/models
   ```
2. Check network connectivity from container:
   ```bash
   docker exec tabular-ml-lab curl http://vllm:8000/v1/models
   ```
3. Verify `VLLM_URL` in `.env` is correct

**Problem:** Ollama backend not responding.

**Solution:**
1. Start Ollama with profile:
   ```bash
   docker-compose --profile ollama up -d
   ```
2. Pull a model:
   ```bash
   docker exec ollama-backend ollama pull llama3.1:8b
   ```

### Compute Performance Issues

**Problem:** Explainability analysis times out or runs very slowly.

**Solution:**
1. Lower compute profile in `.env`:
   ```bash
   COMPUTE_PROFILE=standard
   ```
2. Increase Docker memory limit (edit `docker-compose.yml`):
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16GB
   ```

---

## Security Best Practices

1. **Never commit secrets.toml to version control**
   - Already in `.gitignore`, but double-check
   - Use secret management service (Vault, AWS Secrets Manager) for production

2. **Set restrictive file permissions:**
   ```bash
   chmod 600 .streamlit/secrets.toml
   ```

3. **Run behind HTTPS reverse proxy:**
   ```
   User → HTTPS → nginx/Apache → HTTP → Docker Container
   ```

4. **Enable authentication:**
   - NEVER run in production with `AUTH_ENABLED=false`
   - OIDC/SAML preferred over reverse proxy

5. **Restrict Docker network:**
   - Use custom Docker networks (already configured in `docker-compose.yml`)
   - Don't expose ports directly to internet

6. **Regular updates:**
   ```bash
   git pull origin university-docker
   docker-compose build --no-cache
   docker-compose up -d
   ```

---

## Scaling Considerations

### Vertical Scaling (Single Instance)

- Increase Docker memory/CPU limits
- Use `high_performance` or `enterprise` compute profile
- Deploy on GPU-equipped server for vLLM

### Horizontal Scaling (Multiple Instances)

- **Load balancing:** nginx/HAProxy distributes traffic
- **Session persistence:** Use sticky sessions (IP hash)
- **Shared LLM backend:** Point all instances to same vLLM/Ollama endpoint
- **Stateless design:** App already session-isolated (no persistent storage)

**Example nginx config:**

```nginx
upstream tabular_ml {
    ip_hash;  # Sticky sessions
    server instance1:8501;
    server instance2:8501;
    server instance3:8501;
}

server {
    listen 443 ssl;
    server_name tabular.university.edu;
    
    location / {
        proxy_pass http://tabular_ml;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Support

**Documentation Issues:** Open issue at [GitHub repository](https://github.com/hedglinnolan/tabular-ml-lab)  
**Feature Requests:** Create issue with `enhancement` label  
**Security Issues:** Email maintainer directly (see repository)

---

## License

MIT License - See LICENSE file for details
