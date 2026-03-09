# Docker Deployment Guide - University Edition

**Comprehensive technical reference for deploying Tabular ML Lab at your institution.**

**Flexibility:** This guide covers multiple authentication methods (OIDC, SAML, reverse proxy) and LLM backends (vLLM, Ollama, OpenAI). Choose what works for YOUR infrastructure.

**Primary audience:** University IT administrators and DevOps engineers  
**For quickstart:** See `UNIVERSITY_DEPLOYMENT.md`

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Build & Deploy](#build--deploy)
4. [Configuration](#configuration)
5. [Authentication Options](#authentication-options)
6. [LLM Backend Options](#llm-backend-options)
7. [Reverse Proxy Setup](#reverse-proxy-setup)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

---

## Architecture

### High-Level Diagram

```
User Browser (institutional credentials)
         ↓
HTTPS Reverse Proxy (nginx/Apache/Traefik)
         ↓
Authentication Layer (choose one):
  • OIDC Provider (KeyCloak, Azure AD, Google, Okta)
  • SAML Provider
  • Reverse Proxy Auth (legacy)
         ↓
Docker Container (Tabular ML Lab)
         ↓
LLM Backend (choose one):
  • vLLM (institutional GPU cluster)
  • Ollama (institutional CPU/GPU server)
  • OpenAI API (testing only)
```

### Key Design Principles

- **Stateless:** No persistent storage, all data in-memory (session-isolated)
- **Flexible:** Adapts to your existing auth + LLM infrastructure
- **Secure:** On-premises deployment, no external API calls (except OpenAI if chosen)
- **Scalable:** Horizontal scaling via load balancer with sticky sessions

---

## Prerequisites

### Software

- **Docker Engine** 20.10+ (`docker --version`)
- **Docker Compose** 1.29+ _(optional but recommended)_
- **Git** (`git --version`)

### Hardware (Recommended)

| Profile | CPU | RAM | Disk | Use Case |
|---------|-----|-----|------|----------|
| **Standard** | 4 cores | 8GB | 15GB | Development, small datasets |
| **High Performance** | 8 cores | 16GB | 20GB | Production, moderate usage |
| **Enterprise** | 16+ cores | 32GB+ | 30GB | Large institution, GPU-accelerated |

**GPU:** Optional, improves ML performance (NVIDIA CUDA-compatible recommended)

### Network

**Inbound:**
- Port 8501 from reverse proxy/load balancer

**Outbound:**
- Auth provider (OIDC/SAML endpoint, HTTPS/443)
- LLM backend (vLLM/Ollama, typically HTTP/8000 or 11434)
- _(No internet required after build if using institutional backends)_

### Access & Credentials

- **Git:** Access to this repository
- **Auth:** Admin access to create OIDC client (or existing reverse proxy setup)
- **LLM:** Endpoint URL + credentials for institutional vLLM/Ollama

---

## Build & Deploy

### Option A: Docker Run (Simple)

```bash
# Clone repository
git clone -b university-docker https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Configure (see Configuration section below)
cp .env.example .env
nano .env

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml

# Build image
docker build -t tabular-ml-lab:latest .

# Run container
docker run -d \
  --name tabular-ml-lab \
  --restart unless-stopped \
  -p 8501:8501 \
  --memory="16g" \
  --cpus="8.0" \
  --env-file .env \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  tabular-ml-lab:latest

# Verify
docker logs -f tabular-ml-lab
curl http://localhost:8501/_stcore/health
```

### Option B: Docker Compose (Recommended)

**Edit `docker-compose.yml`** to match your infrastructure, then:

```bash
docker-compose up -d
docker-compose logs -f app
```

### Option C: Kubernetes

See [Kubernetes Deployment](#kubernetes-deployment) section.

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `AUTH_ENABLED` | `false` | `true`, `false` | Enable authentication |
| `AUTH_PROVIDER` | `reverseproxy` | `keycloak`, `azure`, `google`, `saml`, `reverseproxy` | Auth method |
| `VLLM_URL` | `http://vllm:8000` | Any HTTP(S) URL | vLLM endpoint (if using vLLM) |
| `VLLM_MODEL` | _(auto-detect)_ | Model name | vLLM model (blank = auto) |
| `OLLAMA_URL` | `http://ollama:11434` | Any HTTP(S) URL | Ollama endpoint (if using Ollama) |
| `COMPUTE_PROFILE` | `standard` | `standard`, `high_performance`, `enterprise` | Compute limits |
| `STREAMLIT_SERVER_PORT` | `8501` | Any port | App HTTP port |

**Example `.env` for vLLM + KeyCloak:**

```bash
AUTH_ENABLED=true
AUTH_PROVIDER=keycloak
VLLM_URL=http://ml-cluster.university.edu:8000
VLLM_MODEL=
COMPUTE_PROFILE=high_performance
```

**Example `.env` for Ollama + Azure AD:**

```bash
AUTH_ENABLED=true
AUTH_PROVIDER=azure
OLLAMA_URL=http://ollama.university.edu:11434
COMPUTE_PROFILE=standard
```

**Example `.env` for development (no auth, OpenAI):**

```bash
AUTH_ENABLED=false
COMPUTE_PROFILE=standard
# Users configure OpenAI API key in app sidebar
```

### Secrets File (`.streamlit/secrets.toml`)

**Only required if using OIDC or SAML authentication.** Not needed for reverse proxy auth or disabled auth.

**Format depends on AUTH_PROVIDER:**

#### KeyCloak OIDC

```toml
[connections.keycloak]
provider = "keycloak"
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
discovery_url = "https://keycloak.university.edu/realms/YOUR_REALM/.well-known/openid-configuration"
redirect_uri = "https://tabular.university.edu/_stcore/login/keycloak"
```

#### Azure AD (Microsoft Entra ID)

```toml
[connections.azure]
provider = "azure"
client_id = "YOUR_APP_ID"
client_secret = "YOUR_CLIENT_SECRET"
tenant_id = "YOUR_TENANT_ID"
redirect_uri = "https://tabular.university.edu/_stcore/login/azure"
```

#### Google Workspace

```toml
[connections.google]
provider = "google"
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
redirect_uri = "https://tabular.university.edu/_stcore/login/google"
```

#### SAML

```toml
[connections.saml]
provider = "saml"
entity_id = "https://tabular.university.edu"
sso_url = "https://idp.university.edu/saml/sso"
x509cert = "YOUR_IDP_CERTIFICATE"
```

**⚠️ Security:** Set restrictive permissions:

```bash
chmod 600 .streamlit/secrets.toml
chown 1000:1000 .streamlit/secrets.toml  # Match container user
```

---

## Authentication Options

### Option 1: OIDC Providers (Recommended)

**Supported:** KeyCloak, Azure AD, Google Workspace, Okta, Auth0, or any OIDC-compliant provider.

**Advantages:**
- Native Streamlit integration (st.login())
- Secure, modern standard
- Easy to configure

**Setup:**

1. **Create OIDC client in your provider:**
   - Client type: OpenID Connect, Confidential
   - Redirect URI: `https://YOUR_APP_URL/_stcore/login/PROVIDER`
   - Note Client ID + Secret

2. **Configure secrets.toml** (see examples above)

3. **Set environment variables:**
   ```bash
   AUTH_ENABLED=true
   AUTH_PROVIDER=keycloak  # or azure, google, okta
   ```

4. **Restart container**

**Testing:**
- Navigate to app URL
- Should redirect to provider login
- After login, should return to app

### Option 2: SAML

**Best for:** Institutions with existing SAML identity provider (Shibboleth, SimpleSAMLphp, etc.)

**Setup similar to OIDC:**

1. Register app as SAML service provider in your IdP
2. Configure `.streamlit/secrets.toml` with SAML settings
3. Set `AUTH_PROVIDER=saml` in `.env`

### Option 3: Reverse Proxy Auth (Legacy)

**Best for:** Institutions with existing nginx/Apache authentication layer.

**How it works:**
- Reverse proxy (nginx/Apache) handles authentication (LDAP, AD, etc.)
- Proxy passes authenticated username via HTTP header (e.g., `X-Remote-User`)
- App trusts the header

**Setup:**

1. **Configure reverse proxy** to authenticate and set header (see [Reverse Proxy Setup](#reverse-proxy-setup))

2. **Set environment variables:**
   ```bash
   AUTH_ENABLED=true
   AUTH_PROVIDER=reverseproxy
   ```

3. **No secrets.toml needed**

**⚠️ Security Warning:** Only use if reverse proxy is properly configured to prevent header spoofing.

### Option 4: Disabled (Development Only)

```bash
AUTH_ENABLED=false
```

**⚠️ WARNING:** NEVER use in production. Anyone can access the app.

---

## LLM Backend Options

Users can select LLM backend in the app sidebar (🤖 LLM Settings). Configure what's available via environment variables.

### Option 1: vLLM (Recommended for GPU Clusters)

**Best for:** Institutions with GPU infrastructure (A100, A6000, H100, etc.)

**Advantages:**
- OpenAI-compatible API (easy integration)
- High throughput on GPUs
- Auto-detects loaded model (no manual config)
- Institutional control

**Configuration:**

```bash
# In .env
VLLM_URL=http://ml-cluster.university.edu:8000
VLLM_MODEL=  # Leave blank to auto-detect
```

**Test vLLM endpoint:**

```bash
curl http://ml-cluster.university.edu:8000/v1/models
```

Expected: JSON list of models

**In app:** Users select "vLLM (Institutional)" in sidebar, no API key needed.

### Option 2: Ollama

**Best for:** CPU-only deployments or smaller institutions

**Advantages:**
- Easy to deploy
- Good CPU performance
- No GPU required

**Disadvantages:**
- Slower than vLLM on GPUs
- Limited batch processing

**Configuration:**

```bash
# In .env
OLLAMA_URL=http://ollama.university.edu:11434
```

**Test Ollama endpoint:**

```bash
curl http://ollama.university.edu:11434/api/tags
```

Expected: JSON list of models

**In app:** Users select "Ollama" in sidebar, configure model name (e.g., `llama3.1:8b`)

**Optional: Run Ollama in same Docker Compose:**

```yaml
# In docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-backend
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    restart: unless-stopped
    # For GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

volumes:
  ollama-data:
```

Start with: `docker-compose --profile ollama up -d`

### Option 3: OpenAI API (Testing Only)

**Best for:** Small-scale testing, pilot deployments

**Disadvantages:**
- **Cost:** Pay-per-token
- **Privacy:** Data sent to OpenAI (not on-premises)
- **NOT recommended for production**

**Configuration:**

No environment variables needed. Users configure their own API key in app sidebar.

**In app:** Users select "OpenAI API" in sidebar, enter API key.

---

## Reverse Proxy Setup

### nginx Configuration

**`/etc/nginx/sites-available/tabular-ml-lab.conf`:**

```nginx
upstream tabular_ml {
    server localhost:8501;
    # For multiple instances (horizontal scaling):
    # server host1:8501;
    # server host2:8501;
    # ip_hash;  # Sticky sessions required
}

server {
    listen 443 ssl http2;
    server_name tabular.university.edu;

    # TLS certificate
    ssl_certificate /etc/ssl/certs/university.edu.crt;
    ssl_certificate_key /etc/ssl/private/university.edu.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Logging
    access_log /var/log/nginx/tabular-ml-access.log;
    error_log /var/log/nginx/tabular-ml-error.log;

    # Streamlit requires WebSocket support
    location / {
        proxy_pass http://tabular_ml;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (CRITICAL for Streamlit)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    # Health check endpoint (bypass auth if needed)
    location /_stcore/health {
        proxy_pass http://tabular_ml;
        access_log off;
    }
}

# HTTP → HTTPS redirect
server {
    listen 80;
    server_name tabular.university.edu;
    return 301 https://$server_name$request_uri;
}
```

**Enable:**

```bash
sudo ln -s /etc/nginx/sites-available/tabular-ml-lab.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Apache Configuration

**`/etc/apache2/sites-available/tabular-ml-lab.conf`:**

```apache
<VirtualHost *:443>
    ServerName tabular.university.edu

    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/university.edu.crt
    SSLCertificateKeyFile /etc/ssl/private/university.edu.key

    # WebSocket proxy for Streamlit (CRITICAL)
    ProxyPass / http://localhost:8501/
    ProxyPassReverse / http://localhost:8501/
    
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*)           ws://localhost:8501/$1 [P,L]

    # Security headers
    Header always set Strict-Transport-Security "max-age=31536000"
    Header always set X-Frame-Options "SAMEORIGIN"
    Header always set X-Content-Type-Options "nosniff"

    ErrorLog ${APACHE_LOG_DIR}/tabular-ml-error.log
    CustomLog ${APACHE_LOG_DIR}/tabular-ml-access.log combined
</VirtualHost>

<VirtualHost *:80>
    ServerName tabular.university.edu
    Redirect permanent / https://tabular.university.edu/
</VirtualHost>
```

**Enable:**

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite headers ssl
sudo a2ensite tabular-ml-lab
sudo systemctl reload apache2
```

### Reverse Proxy Authentication (Legacy)

If using `AUTH_PROVIDER=reverseproxy`, configure your proxy to pass authenticated username:

**nginx with LDAP:**

```nginx
location / {
    auth_ldap "Please log in";
    auth_ldap_servers ldap_server;
    
    # Pass authenticated username to app
    proxy_set_header X-Remote-User $remote_user;
    
    proxy_pass http://tabular_ml;
    # ... rest of proxy config
}
```

**Apache with Kerberos:**

```apache
<Location />
    AuthType Kerberos
    AuthName "University Login"
    KrbAuthRealms UNIVERSITY.EDU
    Krb5Keytab /etc/apache2/http.keytab
    Require valid-user
    
    # Pass authenticated username
    RequestHeader set X-Remote-User %{REMOTE_USER}e
</Location>
```

---

## Kubernetes Deployment

### Deployment Manifest

**`k8s/deployment.yaml`:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tabular-ml-lab
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tabular-ml-lab
  template:
    metadata:
      labels:
        app: tabular-ml-lab
    spec:
      containers:
      - name: app
        image: tabular-ml-lab:latest
        ports:
        - containerPort: 8501
        env:
        - name: AUTH_ENABLED
          value: "true"
        - name: AUTH_PROVIDER
          value: "keycloak"
        - name: VLLM_URL
          valueFrom:
            configMapKeyRef:
              name: tabular-ml-config
              key: vllm_url
        - name: COMPUTE_PROFILE
          value: "high_performance"
        volumeMounts:
        - name: secrets
          mountPath: /app/.streamlit/secrets.toml
          subPath: secrets.toml
          readOnly: true
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: secrets
        secret:
          secretName: tabular-ml-secrets
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tabular-ml-config
data:
  vllm_url: "http://vllm-service.default.svc.cluster.local:8000"
---
apiVersion: v1
kind: Secret
metadata:
  name: tabular-ml-secrets
type: Opaque
stringData:
  secrets.toml: |
    [connections.keycloak]
    provider = "keycloak"
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    discovery_url = "https://keycloak.university.edu/realms/YOUR_REALM/.well-known/openid-configuration"
    redirect_uri = "https://tabular.university.edu/_stcore/login/keycloak"
```

### Service & Ingress

**`k8s/service.yaml`:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tabular-ml-lab
spec:
  type: ClusterIP
  ports:
  - port: 8501
    targetPort: 8501
  selector:
    app: tabular-ml-lab
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tabular-ml-lab
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/websocket-services: "tabular-ml-lab"
spec:
  tls:
  - hosts:
    - tabular.university.edu
    secretName: tabular-ml-tls
  rules:
  - host: tabular.university.edu
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tabular-ml-lab
            port:
              number: 8501
```

**Deploy:**

```bash
kubectl apply -f k8s/
kubectl get pods -l app=tabular-ml-lab
kubectl logs -f deployment/tabular-ml-lab
```

---

## Monitoring & Logging

### Health Check Endpoint

**URL:** `http://localhost:8501/_stcore/health`

**Response:**
```json
{"status": "ok"}
```

**Use for:**
- Load balancer health checks
- Kubernetes liveness/readiness probes
- Monitoring (Nagios, Prometheus, etc.)

### Logging

**View logs:**

```bash
# Docker
docker logs -f tabular-ml-lab

# Docker Compose
docker-compose logs -f app

# Kubernetes
kubectl logs -f deployment/tabular-ml-lab
```

**Configure log rotation:**

```yaml
# In docker-compose.yml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

**Send to syslog:**

```yaml
services:
  app:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://syslog.university.edu:514"
        tag: "tabular-ml-lab"
```

---

## Troubleshooting

### Authentication Issues

**Problem:** Login redirect loop, or "Authentication Error" after successful login.

**Solutions:**

1. **Verify redirect URI matches:**
   - In `.streamlit/secrets.toml`
   - In OIDC/SAML provider admin console

2. **Check provider configuration:**
   - Client type: Confidential (not Public)
   - Valid Redirect URIs includes your app URL
   - Web Origins set to `*` or your domain

3. **Test OIDC discovery endpoint:**
   ```bash
   curl https://keycloak.university.edu/realms/YOUR_REALM/.well-known/openid-configuration
   ```

4. **Check container can reach auth provider:**
   ```bash
   docker exec tabular-ml-lab curl -I https://keycloak.university.edu
   ```

5. **Review logs:**
   ```bash
   docker logs tabular-ml-lab | grep -i auth
   ```

### LLM Backend Connection Issues

**Problem:** "Could not get interpretation from vLLM/Ollama" error.

**Solutions:**

1. **Test backend directly:**
   ```bash
   # vLLM
   curl http://vllm-server:8000/v1/models
   
   # Ollama
   curl http://ollama-server:11434/api/tags
   ```

2. **Test from container:**
   ```bash
   docker exec tabular-ml-lab curl http://vllm-server:8000/v1/models
   ```

3. **Verify environment variable:**
   ```bash
   docker exec tabular-ml-lab env | grep -E 'VLLM|OLLAMA'
   ```

4. **Check network connectivity:**
   ```bash
   docker exec tabular-ml-lab ping vllm-server
   docker exec tabular-ml-lab traceroute vllm-server
   ```

### Performance Issues

**Problem:** Slow analysis, timeouts, high memory usage.

**Solutions:**

1. **Lower compute profile:**
   ```bash
   # In .env
   COMPUTE_PROFILE=standard
   ```

2. **Increase memory:**
   ```bash
   docker update --memory="32g" tabular-ml-lab
   ```

3. **Monitor resources:**
   ```bash
   docker stats tabular-ml-lab
   ```

### WebSocket Connection Fails

**Problem:** App doesn't update in real-time, "WebSocket connection failed" in browser console.

**Solutions:**

1. **Verify reverse proxy supports WebSockets** (see configs above)

2. **Check nginx/Apache logs:**
   ```bash
   tail -f /var/log/nginx/error.log
   ```

3. **Test WebSocket directly:**
   ```bash
   wscat -c ws://localhost:8501/_stcore/stream
   ```

4. **Increase proxy timeout:**
   ```nginx
   proxy_read_timeout 86400;  # 24 hours
   ```

---

## Security Best Practices

### Secrets Management

1. **Never commit secrets to Git:**
   - `.streamlit/secrets.toml` is in `.gitignore`
   - Verify: `git status --ignored`

2. **Restrict file permissions:**
   ```bash
   chmod 600 .streamlit/secrets.toml
   ```

3. **Use secrets management service in production:**
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault
   - Kubernetes Secrets with encryption at rest

### Network Security

1. **Don't expose container port directly:**
   - Always use reverse proxy
   - Only expose 8501 to proxy, not internet

2. **Firewall rules:**
   ```bash
   # Only allow reverse proxy → container
   iptables -A INPUT -p tcp --dport 8501 -s PROXY_IP -j ACCEPT
   iptables -A INPUT -p tcp --dport 8501 -j DROP
   ```

3. **Internal DNS for LLM backend:**
   - Don't expose vLLM/Ollama to public internet

### Container Security

1. **Non-root user:** Already configured (UID 1000, `appuser`)

2. **Read-only filesystem (optional):**
   ```bash
   docker run --read-only --tmpfs /tmp --tmpfs /app/.streamlit \
     tabular-ml-lab:latest
   ```

3. **Drop capabilities:**
   ```bash
   docker run --cap-drop=ALL tabular-ml-lab:latest
   ```

4. **No new privileges:**
   ```bash
   docker run --security-opt=no-new-privileges tabular-ml-lab:latest
   ```

---

## Scaling

### Vertical Scaling

**Increase resources for single instance:**

```bash
docker update --memory="32g" --cpus="16.0" tabular-ml-lab
```

**Use higher compute profile:**

```bash
# In .env
COMPUTE_PROFILE=enterprise
```

### Horizontal Scaling

**Run multiple instances behind load balancer:**

1. **Enable sticky sessions** (IP hash) in nginx/Apache

2. **Scale Docker Compose:**
   ```bash
   docker-compose up -d --scale app=3
   ```

3. **Scale Kubernetes:**
   ```bash
   kubectl scale deployment/tabular-ml-lab --replicas=5
   ```

4. **Share LLM backend:** All instances point to same vLLM/Ollama endpoint

**Session isolation:** App is already stateless, safe to scale horizontally.

---

## Backup & Recovery

### What to Backup

- `.streamlit/secrets.toml`
- `.env`
- Custom `docker-compose.yml` (if modified)
- Reverse proxy configs

**No user data to backup** - app is stateless (all data in-memory).

### Automated Backup Script

```bash
#!/bin/bash
BACKUP_DIR=/backup/tabular-ml-lab
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR
cp .streamlit/secrets.toml $BACKUP_DIR/secrets.toml.$DATE
cp .env $BACKUP_DIR/.env.$DATE

# Encrypt backup
tar czf $BACKUP_DIR/config-$DATE.tar.gz $BACKUP_DIR/*.$DATE
openssl enc -aes-256-cbc -in $BACKUP_DIR/config-$DATE.tar.gz \
  -out $BACKUP_DIR/config-$DATE.tar.gz.enc

# Cleanup old backups
find $BACKUP_DIR -name "*.enc" -mtime +30 -delete
```

---

## Support

**Repository Issues:** [GitHub Issues](https://github.com/hedglinnolan/tabular-ml-lab/issues)  
**Documentation:** [GitHub Wiki](https://github.com/hedglinnolan/tabular-ml-lab/wiki)  
**Security Issues:** Email maintainer directly

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-09  
**Branch:** `university-docker`
