# 🔬 Tabular ML Lab - Enterprise Edition

**Publication-grade machine learning for institutional research.**

An interactive research workbench for scientists working with tabular data. Upload your CSV and follow a guided, defensible ML workflow — from exploratory analysis to journal-ready methods sections.

---

## Enterprise Features

✅ **Docker containerized** - Deploy on institutional clusters  
✅ **Active Directory integration** - SSO via reverse proxy  
✅ **Institutional Ollama backend** - Connect to your LLM infrastructure  
✅ **No external dependencies** - Runs entirely on-premises  
✅ **Multi-user ready** - Session isolation, no data persistence between users  

---

## Quick Start

### For Administrators

See **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** for complete deployment instructions.

**TL;DR:**
```bash
git clone https://github.com/YOUR-INSTITUTION/tabular-ml-lab.git
cd tabular-ml-lab
cp .env.example .env
# Edit .env with your Ollama URL and auth settings
docker build -t tabular-ml-lab .
docker run -d -p 8501:8501 --env-file .env tabular-ml-lab
```

### For Researchers

Access the app through your institutional portal. Authentication is handled automatically via your .edu credentials.

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

Connect to your institution's Ollama backend for plain-language result interpretation.

**Supported:**
- Ollama (institutional backend, no API key needed)
- OpenAI (optional, requires API key)

---

## Technical Stack

- **Frontend:** Streamlit 1.28+
- **ML:** scikit-learn 1.3+, PyTorch 2.0+
- **Explainability:** SHAP 0.42+
- **Visualization:** Plotly 5.14+
- **Optimization:** Optuna 3.0+
- **Authentication:** Reverse proxy (nginx/Apache + AD)
- **Containerization:** Docker 20.10+

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Browser (.edu credentials)                            │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Reverse Proxy (nginx/Apache + AD)                          │
│  - Handles authentication                                   │
│  - Passes X-Remote-User header                              │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Docker Container (Tabular ML Lab)                          │
│  - Verifies authentication header                           │
│  - Runs analysis workflows                                  │
│  - Session-isolated (no persistent data)                    │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Institutional Ollama Backend                                │
│  - Provides LLM interpretation                              │
│  - No external API calls                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Security & Privacy

✅ **No data persistence** - All analysis happens in-memory, sessions isolated  
✅ **On-premises deployment** - No data leaves institutional network  
✅ **AD authentication** - Leverages existing identity infrastructure  
✅ **Non-root container** - Runs as unprivileged user  
✅ **No external APIs** - Ollama backend runs on institutional servers  

---

## Requirements

**Runtime:**
- Docker Engine 20.10+
- 8GB RAM minimum (16GB recommended for large datasets)
- 4 CPU cores minimum

**Infrastructure:**
- Reverse proxy with AD/LDAP authentication (nginx or Apache)
- Institutional Ollama endpoint
- HTTPS certificate (for production)

---

## Support

**Deployment:** See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)  
**Issues:** Open a ticket with your institution's IT support  
**Feature Requests:** Contact your institutional administrator  

---

## License

MIT License - See LICENSE file for details

---

## Differences from Public Version

This enterprise edition:
- ❌ Removes Anthropic/Claude API support
- ✅ Adds reverse proxy authentication
- ✅ Adds Docker/Kubernetes deployment configs
- ✅ Configurable Ollama backend endpoint
- ✅ Enterprise-focused documentation

The public version is available at: https://github.com/hedglinnolan/tabular-ml-lab
