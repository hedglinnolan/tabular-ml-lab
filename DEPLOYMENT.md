# Deployment Guide

For local development setup, see the [Quick Start in README.md](README.md#quick-start) or [QUICKSTART.md](QUICKSTART.md) for detailed instructions with `uv`, preflight checks, and troubleshooting.

This document covers remote and institutional deployment options.

## Streamlit Cloud (Personal / Demo)

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with GitHub, click "New app"
4. Select your repository, branch `main`, main file `app.py`
5. Deploy

Your app will be live at `https://your-app-name.streamlit.app`.

## Docker (Self-Hosted)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t tabular-ml-lab .
docker run -p 8501:8501 tabular-ml-lab
```

## University / Institutional Deployment

For on-premises hosting or a managed LLM backend, see
[`UNIVERSITY_DEPLOYMENT.md`](UNIVERSITY_DEPLOYMENT.md). The `Dockerfile` and
`docker-compose.yml` live here on `main` — there is no separate deployment branch.

- Docker Compose setup that works on departmental servers, institutional cloud or VMs
- Optional Ollama sidecar for local LLM interpretation, so no data leaves your network
- Access control is your institution's single sign-on placed in front of the
  container; the app itself has no logins

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Port number |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind address |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint (if using local LLM) |
