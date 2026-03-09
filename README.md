# 🔬 Tabular ML Lab - University Docker Edition

**Publication-grade machine learning for institutional research.**

A guided, interactive research workbench for faculty and students working with tabular data. Upload your CSV and follow a defensible ML workflow — from exploratory analysis to journal-ready methods sections.

> 🎓 **Flexible Deployment** - Adapts to your institution's infrastructure (KeyCloak, Azure AD, Google, vLLM, Ollama). See [deployment guide](UNIVERSITY_DEPLOYMENT.md).

---

## University Features

✅ **Docker containerized** - Deploy on any institutional infrastructure  
✅ **Flexible authentication** - KeyCloak, Azure AD, Google, SAML, or reverse proxy  
✅ **Multiple LLM backends** - vLLM, Ollama, or OpenAI API  
✅ **On-premises deployment** - All data stays within your network  
✅ **Compute profiles** - Adapts to available hardware (standard → enterprise)  

---

## Quick Start

### For Researchers (Faculty & Students)

1. Navigate to your institution's deployment URL
2. Authenticate with your institutional credentials
3. Upload your dataset and follow the guided workflow

### For IT Administrators

**See [UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md) for complete deployment guide.**

**TL;DR:**

```bash
# Clone repository
git clone -b university-docker https://github.com/hedglinnolan/tabular-ml-lab.git
cd tabular-ml-lab

# Configure authentication (choose one):
# - OIDC (KeyCloak, Azure AD, Google): Edit .streamlit/secrets.toml
# - SAML: Edit .streamlit/secrets.toml
# - Reverse proxy: Configure your proxy to pass X-Remote-User header

# Configure LLM backend (choose one):
# - vLLM: Set VLLM_URL in .env
# - Ollama: Set OLLAMA_URL in .env
# - OpenAI: Users configure API key in app

# Deploy
docker-compose up -d
```

**Detailed instructions:** [UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md)  
**Technical reference:** [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)

---

## 9-Step Research Workflow

| Step | Page | What it does |
|------|------|-------------|
| 1 | **Upload & Audit** | Load CSVs/Excel, data quality checks, missing data analysis |
| 2 | **EDA** | Distributions, correlations, Table 1, statistical summaries |
| 3 | **Feature Selection** | LASSO, RFE-CV, stability selection, consensus ranking |
| 4 | **Preprocess** | Model-specific pipelines: imputation, scaling, encoding |
| 5 | **Train & Compare** | 18 models with bootstrap confidence intervals |
| 6 | **Explainability** | SHAP, permutation importance, calibration curves |
| 7 | **Sensitivity Analysis** | Seed robustness, feature dropout testing |
| 8 | **Hypothesis Testing** | t-tests, ANOVA, chi-square, correlation |
| 9 | **Report Export** | Auto-generated methods section + TRIPOD checklist |

---

## 18 Models Included

**Linear:** Ridge, Lasso, ElasticNet, Logistic, GLM, Huber  
**Trees:** Random Forest, ExtraTrees, HistGradientBoosting  
**Distance:** KNN (regression & classification)  
**Margin:** SVM (SVR / SVC)  
**Probabilistic:** Gaussian Naive Bayes, LDA  
**Neural:** PyTorch MLP (configurable)  
**Baselines:** Mean/majority + simple linear/logistic  

---

## Publication-Ready Outputs

- **Table 1** with p-values and standardized mean differences
- **Bootstrap 95% CIs** (BCa, 1000 resamples) on all metrics
- **Calibration analysis** with reliability diagrams
- **Decision curve analysis** for clinical utility
- **Subgroup forest plots**
- **TRIPOD checklist** auto-tracked
- **Auto-generated methods section**
- **Journal-quality figures** (300+ DPI)

---

## AI-Powered Interpretation

Connect to your institution's LLM backend for plain-language result interpretation:

**Supported backends:**
- **vLLM** (recommended for GPU clusters) - OpenAI-compatible API, auto-detects model
- **Ollama** (good for CPU/smaller deployments) - Easy to deploy, no GPU required
- **OpenAI API** (testing only) - Not recommended for production (cost + privacy)

Click **"🔬 Interpret with AI"** on any results page to get expert-level analysis.

---

## Configuration Options

### Authentication

Choose what works for YOUR institution:

| Method | Best For | Setup Difficulty |
|--------|----------|------------------|
| **OIDC** (KeyCloak, Azure AD, Google) | Modern SSO infrastructure | Easy (recommended) |
| **SAML** | Legacy enterprise SSO | Medium |
| **Reverse Proxy** | Existing auth layer | Medium (legacy) |

See [UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md) for configuration examples.

### LLM Backend

| Backend | Best For | Setup |
|---------|----------|-------|
| **vLLM** | GPU clusters (A100, A6000, H100) | Set `VLLM_URL` in .env |
| **Ollama** | CPU/single GPU servers | Set `OLLAMA_URL` in .env |
| **OpenAI** | Testing, pilot deployments | Users configure API key |

### Compute Profiles

Adjust performance limits based on hardware:

| Profile | Hardware | PDP Samples | SHAP Evals | Optuna Trials |
|---------|----------|-------------|------------|---------------|
| **standard** | Laptop/workstation | 2,000 | 50 | 30 |
| **high_performance** | Server, single GPU | 10,000 | 200 | 50 |
| **enterprise** | GPU cluster, 2TB RAM | 50,000 | 500 | 100 |

Set via `COMPUTE_PROFILE=enterprise` in .env. See [COMPUTE_PROFILES.md](COMPUTE_PROFILES.md) for details.

---

## Technical Stack

- **Frontend:** Streamlit 1.42+ (native OIDC/SAML support)
- **Authentication:** KeyCloak, Azure AD, Google, SAML, or reverse proxy
- **ML:** scikit-learn 1.3+, PyTorch 2.0+
- **Explainability:** SHAP 0.42+
- **Visualization:** Plotly 5.14+, Kaleido 0.2+
- **Optimization:** Optuna 3.0+
- **LLM:** vLLM, Ollama, or OpenAI API
- **Containerization:** Docker 20.10+

---

## Architecture

```
User Browser (institutional credentials)
         ↓
HTTPS Reverse Proxy (nginx/Apache/Traefik)
         ↓
Authentication Layer (OIDC/SAML/Reverse Proxy)
         ↓
Docker Container (Tabular ML Lab)
  • Stateless (no persistent storage)
  • Session-isolated
  • Non-root user
         ↓
LLM Backend (vLLM/Ollama/OpenAI)
```

---

## Security & Privacy

✅ **No persistent storage** - All analysis in-memory, sessions isolated  
✅ **On-premises deployment** - No data leaves institutional network  
✅ **Flexible authentication** - Integrates with existing SSO  
✅ **Non-root container** - Runs as unprivileged user  
✅ **No external APIs** - (unless using OpenAI for testing)  
✅ **HTTPS enforced** - All traffic encrypted (via reverse proxy)  

---

## Requirements

**Runtime:**
- Docker Engine 20.10+
- 16GB RAM minimum (32GB+ recommended)
- 8 CPU cores minimum

**Infrastructure:**
- Authentication provider (OIDC/SAML or reverse proxy with auth)
- Institutional LLM endpoint (vLLM/Ollama) OR OpenAI API key
- HTTPS certificate (Let's Encrypt or institutional CA)

**Recommended Hardware:**
- **Development:** 4 cores, 8GB RAM, no GPU
- **Production:** 8+ cores, 16GB+ RAM, GPU optional
- **Enterprise:** 16+ cores, 32GB+ RAM, GPU cluster (A6000+)

---

## Deployment

### Docker Run

```bash
docker build -t tabular-ml-lab .
docker run -d \
  --name tabular-ml-lab \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  tabular-ml-lab
```

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Kubernetes

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for K8s manifests.

---

## Documentation

- **[UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md)** - Complete deployment guide with all configuration options
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Technical reference: nginx/Apache configs, K8s, monitoring
- **[COMPUTE_PROFILES.md](COMPUTE_PROFILES.md)** - Hardware-specific performance tuning

---

## Support

**Deployment Issues:** See [UNIVERSITY_DEPLOYMENT.md](UNIVERSITY_DEPLOYMENT.md) troubleshooting section  
**Feature Requests:** [GitHub Issues](https://github.com/hedglinnolan/tabular-ml-lab/issues)  
**Security Concerns:** Email maintainer directly  

---

## Example Configurations

### Small University (CPU-only, KeyCloak)

```bash
# .env
AUTH_ENABLED=true
AUTH_PROVIDER=keycloak
OLLAMA_URL=http://ollama.university.edu:11434
COMPUTE_PROFILE=standard
```

### Large University (GPU cluster, Azure AD)

```bash
# .env
AUTH_ENABLED=true
AUTH_PROVIDER=azure
VLLM_URL=http://ml-cluster.university.edu:8000
COMPUTE_PROFILE=enterprise
```

### Testing/Development (No auth, OpenAI)

```bash
# .env
AUTH_ENABLED=false
COMPUTE_PROFILE=standard
# Users configure OpenAI key in app
```

---

## Scaling

**Vertical:** Increase Docker memory/CPU limits, use higher compute profile  
**Horizontal:** Run multiple instances behind load balancer (sticky sessions required)

**Session isolation:** App is stateless, safe to scale horizontally.

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

Originally developed for West Point by Capt Nolan Hedglin, Department of Mathematical Sciences.

This is the generic university deployment branch with flexible authentication and LLM backend support.

**Public `main` branch:** [hedglinnolan/tabular-ml-lab](https://github.com/hedglinnolan/tabular-ml-lab)  
**University Docker branch:** You are here (`university-docker`)
