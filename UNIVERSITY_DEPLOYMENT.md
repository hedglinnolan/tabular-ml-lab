# University Deployment Guide

This guide helps universities deploy Tabular ML Lab on their own infrastructure with Docker containerization, Active Directory authentication, and institutional Ollama integration.

---

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+
- Reverse proxy with AD/SAML authentication (nginx or Apache)
- Institutional Ollama endpoint (or deploy your own)
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
# Ollama Backend - Point to your institutional endpoint
OLLAMA_URL=http://your-ollama-server.university.edu:11434

# Authentication
AUTH_ENABLED=true
AUTH_HEADER=X-Remote-User  # Header from reverse proxy

# Compute Profile - Adjust based on your hardware
# Options: standard, high_performance, enterprise
COMPUTE_PROFILE=high_performance

# Streamlit (defaults work for most setups)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Step 3: Choose Compute Profile

See [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for detailed performance guidance.

**Quick recommendations:**
- **Consumer GPUs** (GTX 1080 Ti, RTX 3060): `COMPUTE_PROFILE=standard`
- **Professional GPUs** (A6000, A100): `COMPUTE_PROFILE=high_performance`
- **Multi-GPU Cluster**: `COMPUTE_PROFILE=enterprise`

### Step 4: Build and Deploy

```bash
# Build Docker image
docker build -t tabular-ml-lab:latest .

# Run container
docker run -d \
  --name tabular-ml-lab \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  tabular-ml-lab:latest

# Verify deployment
docker logs tabular-ml-lab
curl http://localhost:8501/_stcore/health
```

### Step 5: Configure Reverse Proxy

The app expects authentication to be handled by an upstream reverse proxy (nginx or Apache with AD integration).

**Complete examples in [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md):**
- nginx with mod_auth_ldap
- Apache with mod_authnz_ldap
- Kubernetes with OAuth2 Proxy

**Key requirement:** Proxy must pass authenticated username via HTTP header (default: `X-Remote-User`)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│  Student/Faculty Browser                               │
│  (logs in with university credentials)                 │
└────────────────────────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────┐
│  Reverse Proxy (nginx/Apache)                          │
│  - Active Directory / LDAP authentication              │
│  - Passes X-Remote-User header to backend              │
└────────────────────────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────┐
│  Docker Container (Tabular ML Lab)                     │
│  - Verifies authentication header                      │
│  - Session-isolated analysis workflows                 │
│  - No persistent data storage                          │
└────────────────────────────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────┐
│  Institutional Ollama Backend                          │
│  - Provides LLM interpretation (optional)              │
│  - No external API calls                               │
└────────────────────────────────────────────────────────┘
```

---

## Deployment Options

### Option A: Docker (Simplest)

Best for: Single-server deployments, testing, small classes

```bash
docker run -d -p 8501:8501 --env-file .env tabular-ml-lab
```

### Option B: Docker Compose (Recommended)

Best for: Coordinating multiple services (app + Ollama)

```bash
docker-compose up -d
```

Includes local Ollama instance for testing. Edit `docker-compose.yml` to point to institutional Ollama.

### Option C: Kubernetes (Production)

Best for: Large-scale deployments, high availability, auto-scaling

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete Kubernetes manifests.

---

## Authentication Setup

### Method 1: Reverse Proxy Header (Recommended)

Your existing web infrastructure handles authentication:

1. User hits `https://ml-lab.university.edu`
2. Nginx/Apache redirects to university login
3. After authentication, proxy sets `X-Remote-User: username` header
4. Request forwarded to Docker container with header

**Pros:**
- Leverages existing SSO infrastructure
- No code changes needed
- Works with any identity provider

**Configuration examples in DOCKER_DEPLOYMENT.md**

### Method 2: Direct LDAP (Alternative)

App queries AD directly - requires more configuration.

**Not recommended** - use reverse proxy method instead.

---

## Ollama Backend

### Connect to Institutional Ollama

If your university runs a centralized Ollama service:

```bash
OLLAMA_URL=http://ollama.cs.university.edu:11434
```

### Deploy Your Own

If you don't have institutional Ollama:

```bash
# Use included docker-compose setup
docker-compose up -d

# Pull models
docker exec -it ollama-backend ollama pull llama3.1:8b
docker exec -it ollama-backend ollama pull mistral:7b
```

**Models recommended:**
- `llama3.1:8b` - Best balance of quality and speed
- `mistral:7b` - Alternative, slightly faster
- `gemma2:9b` - Google's open model

---

## Compute Profile Optimization

The app automatically adjusts performance based on your hardware.

**Set in `.env`:**
```bash
COMPUTE_PROFILE=enterprise  # For powerful clusters
```

**Performance comparison:**

| Profile | PDP Samples | SHAP Evals | Optuna Trials | Best For |
|---------|-------------|------------|---------------|----------|
| standard | 2,000 | 50 | 30 | Consumer GPUs, laptops |
| high_performance | 10,000 | 200 | 50 | A6000, A100 single GPU |
| enterprise | 50,000 | 500 | 100 | Multi-GPU clusters |

**Full details:** [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md)

---

## Security Considerations

✅ **Already implemented:**
- Non-root container user
- Session-only data storage (no persistence)
- Authentication gate before app loads
- Health check endpoint for monitoring

⚠️ **Your responsibility:**
- HTTPS on reverse proxy
- Firewall rules (restrict container access)
- Regular Docker image updates
- AD/LDAP credentials security
- Audit logs enabled on reverse proxy

---

## Testing Your Deployment

### 1. Health Check

```bash
curl http://localhost:8501/_stcore/health
# Expected: {"status": "ok"}
```

### 2. Authentication Test

```bash
# Without auth header (should fail)
curl http://localhost:8501

# With auth header (should work)
curl -H "X-Remote-User: testuser" http://localhost:8501
```

### 3. Ollama Connectivity

```bash
# From inside container
docker exec -it tabular-ml-lab curl http://your-ollama-server:11434/api/tags

# Should return JSON list of available models
```

### 4. Full Integration Test

1. Access through reverse proxy: `https://ml-lab.university.edu`
2. Log in with university credentials
3. Upload a CSV file
4. Run through EDA page
5. Try LLM interpretation (if Ollama configured)

---

## Troubleshooting

### "Authentication Required" Error

**Possible causes:**
- Reverse proxy not passing `X-Remote-User` header
- Header name mismatch (check `AUTH_HEADER` in `.env`)
- Proxy authentication not working

**Debug:**
```bash
# Check what headers the app sees
docker exec -it tabular-ml-lab env | grep AUTH
```

### LLM Interpretation Not Working

**Possible causes:**
- Ollama URL incorrect
- Firewall blocking port 11434
- Model not pulled

**Debug:**
```bash
# Test Ollama directly
curl http://your-ollama-server:11434/api/tags

# Check app logs
docker logs tabular-ml-lab | grep -i ollama
```

### Slow Performance

**If analysis is taking too long:**

1. Check compute profile: `docker exec -it tabular-ml-lab env | grep COMPUTE`
2. Consider lowering profile: `COMPUTE_PROFILE=standard`
3. Monitor resources: `docker stats tabular-ml-lab`

**See [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for benchmarking guide.**

---

## Monitoring

### Resource Usage

```bash
# Real-time monitoring
docker stats tabular-ml-lab

# Memory usage over time
docker stats --no-stream tabular-ml-lab

# Logs
docker logs -f tabular-ml-lab
```

### User Activity

Enable reverse proxy access logs:

```nginx
# nginx example
access_log /var/log/nginx/ml-lab-access.log combined;
```

Track:
- Login frequency
- Page views per session
- Analysis completion rates

---

## Updating the Application

```bash
# Pull latest changes
cd tabular-ml-lab
git pull origin university-docker

# Rebuild image
docker build -t tabular-ml-lab:latest .

# Stop old container
docker stop tabular-ml-lab && docker rm tabular-ml-lab

# Start new container
docker run -d --name tabular-ml-lab --restart unless-stopped \
  -p 8501:8501 --env-file .env tabular-ml-lab:latest
```

**For Kubernetes:** Update deployment image tag and roll out.

---

## Scaling for Large Classes

### Horizontal Scaling (Multiple Containers)

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

### Load Balancing

Use nginx upstream:

```nginx
upstream tabular_ml {
    server app1:8501;
    server app2:8501;
    server app3:8501;
}
```

### Resource Limits

```yaml
# Kubernetes example
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

---

## Support

**For deployment questions:**
1. Check [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for detailed configs
2. Review [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for performance tuning
3. Check Docker logs: `docker logs tabular-ml-lab`
4. Open an issue on GitHub: https://github.com/hedglinnolan/tabular-ml-lab/issues

**For application bugs:**
- GitHub Issues: https://github.com/hedglinnolan/tabular-ml-lab/issues
- Include logs and reproduction steps

---

## Example Use Cases

✅ **Statistics courses** - Students analyze datasets without coding  
✅ **PhD research** - Publication-ready outputs with TRIPOD checklists  
✅ **Capstone projects** - Guided ML workflow ensures methodology quality  
✅ **Faculty research** - Bootstrap CIs, SHAP, calibration analysis  

---

## License

MIT License - Free for educational and research use

See [LICENSE](LICENSE) for full terms.

---

## Differences from Public Version

This university-docker branch includes:
- ✅ Docker/Kubernetes deployment configs
- ✅ Reverse proxy authentication
- ✅ Institutional Ollama integration
- ✅ Configurable compute profiles
- ❌ Removed Anthropic/Claude API (on-premises only)

The main branch at https://github.com/hedglinnolan/tabular-ml-lab includes all features for personal/research use.

---

## Questions?

**Common questions answered in:**
- [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) - Detailed deployment configs
- [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) - Performance tuning
- [README.md](README.md) - Feature overview

**Still stuck?** Open a GitHub issue with:
- Deployment method (Docker/K8s)
- Hardware specs
- Error messages and logs
- What you've tried
