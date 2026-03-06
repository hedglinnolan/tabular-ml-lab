# Docker Deployment Guide - Enterprise Edition

This guide covers deploying Tabular ML Lab in an enterprise environment with:
- Docker containerization
- Reverse proxy authentication (Microsoft Active Directory)
- Institutional Ollama backend integration

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+
- Reverse proxy with AD authentication (nginx or Apache)
- Access to institutional Ollama endpoint
- Git

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/YOUR-INSTITUTION/tabular-ml-lab.git
cd tabular-ml-lab

# Copy and configure environment
cp .env.example .env
nano .env
```

### 2. Configure Environment Variables

Edit `.env`:

```bash
# Point to your institution's Ollama backend
OLLAMA_URL=http://your-ollama-server:11434

# Enable authentication (default: true)
AUTH_ENABLED=true

# Header name from reverse proxy (default: X-Remote-User)
AUTH_HEADER=X-Remote-User

# Streamlit settings (defaults usually work)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 3. Build and Deploy

```bash
# Build the Docker image
docker build -t tabular-ml-lab:latest .

# Run the container
docker run -d \
  --name tabular-ml-lab \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  tabular-ml-lab:latest
```

### 4. Verify Deployment

```bash
# Check container status
docker ps | grep tabular-ml-lab

# View logs
docker logs tabular-ml-lab

# Health check
curl http://localhost:8501/_stcore/health
```

## Reverse Proxy Configuration

### Nginx with AD Authentication

```nginx
server {
    listen 443 ssl;
    server_name ml-lab.your-institution.edu;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # AD authentication (requires nginx-auth-ldap module)
    auth_ldap "Institutional Login";
    auth_ldap_servers ad_server;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Pass authenticated username to app
        proxy_set_header X-Remote-User $remote_user;
        
        # Streamlit-specific
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}

# LDAP server config
ldap_server ad_server {
    url ldap://your-ad-server.edu:389/DC=your-institution,DC=edu?sAMAccountName?sub?(objectClass=person);
    binddn "CN=nginx-service,OU=Service Accounts,DC=your-institution,DC=edu";
    binddn_passwd "service-account-password";
    group_attribute member;
    group_attribute_is_dn on;
    require valid_user;
}
```

### Apache with mod_authnz_ldap

```apache
<VirtualHost *:443>
    ServerName ml-lab.your-institution.edu
    
    SSLEngine on
    SSLCertificateFile /path/to/cert.pem
    SSLCertificateKeyFile /path/to/key.pem
    
    # AD authentication
    <Location />
        AuthType Basic
        AuthName "Institutional Login"
        AuthBasicProvider ldap
        AuthLDAPURL "ldap://your-ad-server.edu:389/DC=your-institution,DC=edu?sAMAccountName?sub?(objectClass=user)"
        AuthLDAPBindDN "CN=apache-service,OU=Service Accounts,DC=your-institution,DC=edu"
        AuthLDAPBindPassword "service-account-password"
        Require valid-user
        
        # Pass username to backend
        RequestHeader set X-Remote-User %{REMOTE_USER}s
    </Location>
    
    # Proxy to Streamlit
    ProxyPreserveHost On
    ProxyPass / http://localhost:8501/
    ProxyPassReverse / http://localhost:8501/
    
    # WebSocket support for Streamlit
    RewriteEngine on
    RewriteCond %{HTTP:Upgrade} websocket [NC]
    RewriteCond %{HTTP:Connection} upgrade [NC]
    RewriteRule ^/?(.*) "ws://localhost:8501/$1" [P,L]
</VirtualHost>
```

## Docker Compose (Alternative)

For orchestration with multiple services:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## Kubernetes Deployment

Example Kubernetes manifests:

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tabular-ml-lab
  namespace: ml-tools
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
        image: your-registry.edu/tabular-ml-lab:latest
        ports:
        - containerPort: 8501
        env:
        - name: OLLAMA_URL
          value: "http://ollama-service.ml-backend:11434"
        - name: AUTH_ENABLED
          value: "true"
        - name: AUTH_HEADER
          value: "X-Remote-User"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tabular-ml-lab
  namespace: ml-tools
spec:
  selector:
    app: tabular-ml-lab
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: ClusterIP
```

### Ingress (with OAuth2 Proxy for AD)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tabular-ml-lab
  namespace: ml-tools
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/auth-url: "https://oauth2.your-institution.edu/oauth2/auth"
    nginx.ingress.kubernetes.io/auth-signin: "https://oauth2.your-institution.edu/oauth2/start?rd=$escaped_request_uri"
    nginx.ingress.kubernetes.io/auth-response-headers: "X-Auth-Request-User,X-Auth-Request-Email"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - ml-lab.your-institution.edu
    secretName: tabular-ml-lab-tls
  rules:
  - host: ml-lab.your-institution.edu
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tabular-ml-lab
            port:
              number: 80
```

## Connecting to Institutional Ollama

The app uses the `OLLAMA_URL` environment variable to connect to your institution's Ollama backend.

**Supported models** (check with your IT department):
- `llama3.1:8b` - Default, good balance
- `llama3.2:3b` - Faster, lighter
- `mistral:7b` - Alternative architecture
- `gemma2:9b` - Google's model

**Testing connectivity:**

```bash
# From inside the container
docker exec -it tabular-ml-lab curl http://your-ollama-server:11434/api/tags

# Expected response: JSON list of available models
```

## Troubleshooting

### Authentication Issues

**Problem:** Users see "Authentication Required" error

**Solutions:**
1. Verify reverse proxy is passing `X-Remote-User` header:
   ```bash
   curl -H "X-Remote-User: testuser" http://localhost:8501
   ```
2. Check `AUTH_HEADER` matches your proxy configuration
3. Verify proxy authentication is working independently
4. Check container logs: `docker logs tabular-ml-lab`

### Ollama Connection Issues

**Problem:** "Could not get interpretation" or LLM errors

**Solutions:**
1. Test Ollama endpoint directly:
   ```bash
   curl http://your-ollama-server:11434/api/tags
   ```
2. Verify `OLLAMA_URL` in `.env` is correct
3. Check network connectivity from container to Ollama server
4. Verify firewall rules allow traffic on port 11434

### Container Won't Start

**Problem:** Container exits immediately

**Solutions:**
1. Check logs: `docker logs tabular-ml-lab`
2. Verify all environment variables are set correctly
3. Ensure port 8501 is not already in use:
   ```bash
   netstat -tulpn | grep 8501
   ```
4. Try running interactively to see errors:
   ```bash
   docker run -it --env-file .env tabular-ml-lab:latest
   ```

### Memory Issues

**Problem:** Container crashes or kills jobs (OOM)

**Solutions:**
1. Increase Docker memory limit:
   ```bash
   docker run --memory=8g --memory-swap=8g ...
   ```
2. For Kubernetes, increase resource limits in deployment manifest
3. Monitor usage: `docker stats tabular-ml-lab`

## Security Best Practices

1. **Run as non-root user** (already configured in Dockerfile)
2. **Use secrets management** for sensitive environment variables
3. **Enable TLS** on reverse proxy
4. **Restrict network access** - only allow proxy → container traffic
5. **Regular updates** - rebuild image monthly to patch vulnerabilities
6. **Audit authentication logs** from reverse proxy
7. **Use private registry** for institutional image storage

## Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild image
docker build -t tabular-ml-lab:latest .

# Stop old container
docker stop tabular-ml-lab && docker rm tabular-ml-lab

# Start new container
docker run -d \
  --name tabular-ml-lab \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  tabular-ml-lab:latest
```

## Support

For deployment issues:
- Check container logs: `docker logs tabular-ml-lab`
- Verify health endpoint: `curl http://localhost:8501/_stcore/health`
- Review reverse proxy logs for authentication issues
- Contact your institution's IT support for infrastructure questions

For application bugs or feature requests:
- Open an issue on GitHub
- Include logs and reproduction steps
