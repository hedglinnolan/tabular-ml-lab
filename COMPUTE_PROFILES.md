# Computational Resource Profiles

Tabular ML Lab automatically adjusts performance limits based on available hardware. This ensures optimal performance without overloading limited resources.

## Profile Selection

Set via environment variable:
```bash
COMPUTE_PROFILE=enterprise
```

---

## Available Profiles

### 🏠 STANDARD (Default)
**Hardware:** GTX 1080 Ti, RTX 3060, consumer GPUs (≤16GB VRAM)

| Setting | Value | Reason |
|---------|-------|--------|
| PDP max samples | 2,000 | Prevents stalling on large datasets |
| SHAP KernelExplainer evals | 50 | KernelExplainer is O(n²) — very slow |
| SHAP background samples | 50 | Limited GPU memory |
| Optuna trials | 30 | Reasonable search, ~5-10 min per model |
| Bootstrap resamples | 1,000 | Standard statistical practice |
| Seed sensitivity samples | 10 | Adequate robustness check |

**Best for:** Personal workstations, laptops, development

---

### 🚀 HIGH_PERFORMANCE
**Hardware:** A6000 (48GB), A100 (40/80GB), single professional GPU

| Setting | Value | Change vs Standard |
|---------|-------|-------------------|
| PDP max samples | 10,000 | **5x** more detailed PDPs |
| SHAP KernelExplainer evals | 200 | **4x** more accurate SHAP values |
| SHAP background samples | 100 | **2x** larger background |
| Optuna trials | 50 | **67%** more hyperparameter search |
| Bootstrap resamples | 1,000 | (no change, 1000 is standard) |
| Seed sensitivity samples | 20 | **2x** more robustness tests |

**Best for:** Research labs, single-server deployments

---

### 🏢 ENTERPRISE
**Hardware:** Multi-GPU clusters, A6000 arrays, H100s, 2TB+ RAM

| Setting | Value | Change vs Standard |
|---------|-------|-------------------|
| PDP max samples | 50,000 | **25x** — near full-dataset PDPs |
| SHAP KernelExplainer evals | 500 | **10x** — publication-grade SHAP |
| SHAP background samples | 200 | **4x** larger background |
| Optuna trials | 100 | **3.3x** — exhaustive search |
| Bootstrap resamples | 2,000 | **2x** — tighter confidence intervals |
| Seed sensitivity samples | 50 | **5x** — comprehensive robustness |

**Best for:** Institutional clusters, West Point deployment, large-scale research

---

## Performance Impact

### PDP (Partial Dependence Plots)
- **STANDARD (2k samples):** ~2-5 seconds per feature
- **HIGH_PERFORMANCE (10k):** ~10-20 seconds per feature
- **ENTERPRISE (50k):** ~30-60 seconds per feature

**Why it matters:** PDPs show how predictions change with feature values. More samples = smoother, more accurate curves. Critical for publication-quality figures.

---

### SHAP (Feature Importance)
- **STANDARD (50 evals):** ~10-30 seconds for KernelExplainer
- **HIGH_PERFORMANCE (200):** ~1-2 minutes
- **ENTERPRISE (500):** ~3-5 minutes

**Why it matters:** KernelExplainer is O(n²) slow but works on any model. More evaluations = more accurate Shapley values. TreeExplainer and LinearExplainer are unaffected (already fast).

---

### Optuna Hyperparameter Optimization
- **STANDARD (30 trials):** ~5-10 minutes per model
- **HIGH_PERFORMANCE (50):** ~8-15 minutes
- **ENTERPRISE (100):** ~15-25 minutes

**Why it matters:** More trials = better chance of finding optimal hyperparameters. Diminishing returns after ~50 trials for most models.

---

### Bootstrap Confidence Intervals
- **STANDARD (1000 resamples):** ~10-20 seconds per metric
- **ENTERPRISE (2000):** ~20-40 seconds

**Why it matters:** 1000 resamples is the statistical standard for 95% CIs. 2000 gives slightly tighter intervals but minimal practical benefit.

---

## Setting the Profile

### Docker Deployment

**In `.env` file:**
```bash
COMPUTE_PROFILE=enterprise
```

**In `docker-compose.yml`:**
```yaml
services:
  app:
    environment:
      - COMPUTE_PROFILE=enterprise
```

**In Dockerfile (default):**
```dockerfile
ENV COMPUTE_PROFILE=enterprise
```

### Kubernetes

```yaml
spec:
  containers:
  - name: app
    env:
    - name: COMPUTE_PROFILE
      value: "enterprise"
```

### Local Development

```bash
export COMPUTE_PROFILE=high_performance
streamlit run app.py
```

---

## Benchmarking Your Hardware

Run this test to see actual performance on your cluster:

```python
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from shap import KernelExplainer

# Generate test data
X = np.random.randn(10000, 20)
y = np.random.randint(0, 2, 10000)

# Train model
model = RandomForestClassifier(n_estimators=100).fit(X[:5000], y[:5000])

# Benchmark SHAP
start = time.time()
explainer = KernelExplainer(model.predict_proba, X[:100])
shap_values = explainer.shap_values(X[5000:5010])
elapsed = time.time() - start

print(f"SHAP time: {elapsed:.1f}s")
print(f"Recommended profile: ", end="")
if elapsed < 5:
    print("ENTERPRISE (fast)")
elif elapsed < 15:
    print("HIGH_PERFORMANCE (moderate)")
else:
    print("STANDARD (slow)")
```

---

## Profile Recommendations by Use Case

| Use Case | Profile | Why |
|----------|---------|-----|
| Course/teaching | STANDARD | Fast enough, predictable runtime |
| PhD research | HIGH_PERFORMANCE | Balance speed vs quality |
| Journal submissions | ENTERPRISE | Publication-grade precision |
| Exploratory analysis | STANDARD | Quick iterations |
| Final analysis | ENTERPRISE | Best quality for papers |
| Production inference | N/A | Profiles only affect training |

---

## Neural Network Limitation

**All profiles:** Seed sensitivity is **disabled** for PyTorch neural networks due to sklearn's `clone()` incompatibility.

This is a framework limitation, not a compute limitation. The app will show a warning if you attempt seed sensitivity on NN models.

**Workaround:** Test robustness by manually changing the random seed in the sidebar and retraining the NN.

---

## Custom Profile

You can create a custom profile by setting individual limits:

```python
# In your deployment config
import os
os.environ["COMPUTE_PROFILE"] = "high_performance"  # Start with this

# Then override specific limits (advanced users only)
from utils.compute_config import ComputeProfile
ComputeProfile.HIGH_PERFORMANCE["pdp_max_samples"] = 25000
```

**Not recommended** unless you're profiling performance issues.

---

## Monitoring Performance

Add this to your deployment to log compute profile at startup:

```bash
docker logs tabular-ml-lab 2>&1 | grep "COMPUTE_PROFILE"
```

The app logs the active profile when pages load:
```
2026-03-06 22:00:00 [INFO] Using ENTERPRISE compute profile
2026-03-06 22:00:05 [INFO] PDP subsampling: 50000 max samples
```

---

## Questions?

- **"Will ENTERPRISE crash my server?"** No. Limits are maximums — actual usage depends on dataset size.
- **"Can I change profile without rebuilding Docker?"** Yes! Update `.env` and restart the container.
- **"Does profile affect accuracy?"** Only at the margins. STANDARD is publication-quality. ENTERPRISE is just more precise.
- **"What about GPU memory?"** Profile affects CPU-bound operations (SHAP, PDP). GPU memory is auto-managed by PyTorch.

---

For deployment instructions, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).
