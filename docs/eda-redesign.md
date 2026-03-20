# EDA Page Redesign — Comprehensive Design Document

**Branch:** `feature/eda-improvements`
**Date:** 2026-03-20
**Status:** APPROVED for implementation
**Scope:** Full Option C — EDA page redesign + Insight Ledger across all pages

---

## Part I: Design Philosophy

### The Inversion
The current EDA page leads with *coaching* (verdict, decision hub, reviewer risks) and buries *exploration* behind nested expanders. This inverts the natural workflow.

**Current:** System tells you what it thinks → you dig for the actual data
**Proposed:** You see and touch the data immediately → system coaches you contextually as you go

### Principles
1. **Data first, coaching second.** Show the shape of the data within 2 seconds of landing.
2. **Interactivity over static walls.** Click, filter, hover, select — not scroll through pre-rendered dumps.
3. **Progressive disclosure.** Surface → Shape → Relationships → Topology → Diagnostics → Publication.
4. **No dead ends.** Every insight suggests a next action. Every chart connects to what it means for modeling.
5. **Adaptive layout.** A 5-feature clinical dataset and a 500-feature genomics dataset get different UIs, not the same UI paginated.
6. **Insight provenance.** Every observation during EDA should be traceable forward into modeling decisions.

---

## Part II: Adaptive Layout — Dataset Regime Detection

The page detects dataset shape on load and adapts component selection, not just pagination.

### Regime Definitions

```python
@dataclass
class DatasetRegime:
    """Computed once on page load, drives all layout decisions."""
    n_rows: int
    n_features: int
    n_numeric: int
    n_categorical: int
    
    @property
    def feature_regime(self) -> str:
        if self.n_features <= 15:
            return "narrow"       # show everything
        elif self.n_features <= 50:
            return "medium"       # paginated gallery, full corr matrix
        elif self.n_features <= 200:
            return "wide"         # top-N pairs, feature search, auto-group
        else:
            return "ultra_wide"   # summary-of-summaries, drill-down only

    @property
    def row_regime(self) -> str:
        if self.n_rows < 100:
            return "tiny"         # show all points, warn about sample size
        elif self.n_rows < 10_000:
            return "standard"     # normal plotting
        elif self.n_rows < 100_000:
            return "large"        # sample for scatters, hexbin option
        else:
            return "massive"      # always sample, aggregate views primary
```

### Component Adaptation by Regime

| Component | narrow (≤15) | medium (16-50) | wide (51-200) | ultra_wide (200+) |
|-----------|-------------|----------------|---------------|-------------------|
| **Data Snapshot table** | All columns | All columns, horizontal scroll | Column selector, show 20 at a time | Column groups + search |
| **Distribution gallery** | All at once, 3×N grid | Paginated 3×3 | Paginated 3×3 + type filter pills | Summary stats table + distribution-of-distributions (skew histogram, missing rate histogram). Search to drill into any single feature |
| **Correlation matrix** | Full heatmap | Full heatmap + threshold filter | Top-N correlated pairs list + expandable full matrix | Pairs list only. Hierarchical clustering dendrogram as alternative |
| **Target relationship** | All features shown | Paginated | Paginated + "top correlated with target" auto-sort | Top-10 by MI/correlation, rest searchable |
| **Macro Shape (PCA/UMAP/TDA)** | Skipped (not needed) | PCA biplot offered | PCA + UMAP default | PCA + UMAP + TDA recommended |
| **Outlier heatmap** | Full matrix | Full matrix | Clustered/sorted matrix | Top-N outlier-heavy features only |
| **Missing data** | Bar chart | Bar chart + pattern matrix if >5% | Bar chart + co-missingness clusters | Bar chart of % + top co-missing pairs |

| Component | tiny (<100 rows) | standard | large (10K-100K) | massive (100K+) |
|-----------|-----------------|----------|-------------------|-----------------|
| **Scatter plots** | All points, no sampling | All points | Sampled (5K) + density contours | Hexbin / 2D histogram only |
| **Distributions** | Histogram + rug plot | Histogram + KDE | Histogram + KDE | Histogram only (KDE expensive) |
| **Sample size warning** | Prominent banner | Hidden | Hidden | Hidden |
| **Correlation computation** | Full | Full | Sampled if >50 features | Always sampled |

---

## Part III: Page Architecture

### Wireframe (top-to-bottom scroll order)

```
┌─────────────────────────────────────────────────────────┐
│ Section 0: AT-A-GLANCE HEADER                    ~80px  │
│ [Rows: 1,234] [Features: 42] [Num: 38 Cat: 4]         │
│ [Missing: 2.1%] [Target: regression] [●  Adequate]     │
│ ⚠️ Alert ribbon (only if blockers exist)                │
├─────────────────────────────────────────────────────────┤
│ Section 1: DATA SNAPSHOT                        ~300px  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Interactive st.dataframe — sort, filter, search     │ │
│ │ Column headers show type icons                      │ │
│ │ Click column → popover with stats + sparkline       │ │
│ └─────────────────────────────────────────────────────┘ │
│ [# 38 numeric] [Aa 4 categorical] [📅 0 datetime]      │
│                    (type filter pills)                   │
├─────────────────────────────────────────────────────────┤
│ Section 2: SHAPE OF THE DATA                    ~600px  │
│                                                         │
│ Target Distribution (prominent, full width)             │
│ ┌──────────────────────┬──────────────────────────────┐ │
│ │ Histogram + KDE      │ Box plot + basic stats       │ │
│ └──────────────────────┴──────────────────────────────┘ │
│                                                         │
│ Feature Distribution Gallery (paginated 3×3)            │
│ [All] [Numeric] [Categorical] [High Missing] [Outliers] │
│ ┌────────┐ ┌────────┐ ┌────────┐                       │
│ │ feat_1 │ │ feat_2 │ │ feat_3 │                       │
│ ├────────┤ ├────────┤ ├────────┤                       │
│ │ feat_4 │ │ feat_5 │ │ feat_6 │                       │
│ ├────────┤ ├────────┤ ├────────┤                       │
│ │ feat_7 │ │ feat_8 │ │ feat_9 │                       │
│ └────────┘ └────────┘ └────────┘                       │
│              [< Page 1 of 5 >]                          │
│                                                         │
│ Outlier Overview                                        │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Heatmap: features × methods, cells = % flagged      │ │
│ │ Method selector (IQR/MAD/Z-score/percentile)        │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Missing Data (conditional — only if missingness > 0%)   │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Bar chart: missing % per column                     │ │
│ │ [Expand: co-missingness pattern matrix] if >5%      │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Section 3: RELATIONSHIPS                        ~500px  │
│                                                         │
│ Correlation Matrix (or top-N pairs for wide datasets)   │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Plotly heatmap, hover for values                    │ │
│ │ [Pearson] [Spearman] [Mutual Info]    Threshold: 0.8│ │
│ │ Click cell → bivariate scatter opens below          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Target Relationship Gallery (paginated)                 │
│ ┌────────┐ ┌────────┐ ┌────────┐                       │
│ │ feat×T │ │ feat×T │ │ feat×T │  regression: scatter  │
│ │ LOWESS │ │ LOWESS │ │ LOWESS │  classif: violin      │
│ └────────┘ └────────┘ └────────┘                       │
│              [< Page 1 of 5 >]                          │
│                                                         │
│ Feature Explorer (interactive)                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ [Feature X ▼] vs [Feature Y ▼]  Color: [Target ▼]  │ │
│ │ ┌─────────────────────────────────┐                 │ │
│ │ │                                 │                 │ │
│ │ │        scatter plot             │                 │ │
│ │ │                                 │                 │ │
│ │ └─────────────────────────────────┘                 │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Suggested Interactions (auto-detected)                  │
│ [💡 BMI × Age] [💡 Glucose × Insulin] [💡 ...]          │
│ (click chip → populates Feature Explorer above)         │
├─────────────────────────────────────────────────────────┤
│ Section 4: MACRO SHAPE (≥16 features only)      ~400px  │
│                                                         │
│ Variance Profile (always first)                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Scree plot: cumulative explained variance by PC     │ │
│ │ "3 components explain 87% of variance"              │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Embedding Views                                         │
│ [PCA Biplot] [UMAP] [Persistence Diagram] [Mapper]     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │                                                     │ │
│ │    Active embedding visualization                   │ │
│ │    Colored by target / density / cluster            │ │
│ │    Hover shows original feature values              │ │
│ │                                                     │ │
│ └─────────────────────────────────────────────────────┘ │
│ Coaching annotation: what this view reveals/hides       │
├─────────────────────────────────────────────────────────┤
│ Section 5: COACHING LAYER                       ~300px  │
│                                                         │
│ Insight Ledger Summary                                  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 🚨 2 blockers · ⚠️ 4 warnings · 💡 3 opportunities │ │
│ │                                                     │ │
│ │ BLOCKER: Potential target leakage in [col_x]        │ │
│ │   → Action: Review in Feature Selection  [Go →]     │ │
│ │                                                     │ │
│ │ WARNING: Features A × B correlated at 0.97          │ │
│ │   → Action: Drop one in Feature Selection [Go →]    │ │
│ │                                                     │ │
│ │ INFO: Target is right-skewed (skew=2.4)             │ │
│ │   → Action: Consider log transform in Preprocess    │ │
│ │                                                     │ │
│ │ OPPORTUNITY: UMAP reveals 3 distinct clusters       │ │
│ │   → Action: Consider cluster-based features in FE   │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                         │
│ Reviewer Risk Flags (collapsed by default)              │
│ Downstream Recommendations (what to do on next pages)   │
├─────────────────────────────────────────────────────────┤
│ Section 6: DEEP DIVE DIAGNOSTICS                ~varies │
│                                                         │
│ [Model Readiness] [Feature Quality] [Advanced]  (tabs)  │
│                                                         │
│ Each tab: 2-3 analyses with Run buttons                 │
│ Results inline with narrative + LLM interpretation      │
│ Insights auto-written to ledger when analyses complete  │
├─────────────────────────────────────────────────────────┤
│ Section 7: TABLE 1 — PUBLICATION SUMMARY       (collapsed) │
│ (unchanged functionality, bottom of page)               │
└─────────────────────────────────────────────────────────┘
```

### Inline Coaching Annotations

Throughout Sections 2-4, contextual coaching appears *next to the relevant chart* when something notable is detected. These are NOT in Section 5 — they're inline.

Examples:
- Next to a skewed histogram: `ℹ️ Skew = 3.2 — log transform often helps. [Add to ledger]`
- Next to a high-correlation cell: `⚠️ r = 0.97 — likely redundant. [Add to ledger]`
- Next to class balance bar: `⚠️ 85/15 split — stratified sampling recommended. [Add to ledger]`
- Next to UMAP showing clusters: `💡 3 apparent clusters — consider as feature. [Add to ledger]`

The `[Add to ledger]` button lets users explicitly promote an observation into the Insight Ledger, which then carries it forward to downstream pages. Auto-detected blockers (leakage, severe imbalance) are added automatically.

---

## Part IV: Insight Ledger — Cross-App Information Architecture

### The Problem

Currently there are 5 separate, ad hoc mechanisms carrying information forward:
1. `eda_insights[]` — flat list of {finding, implication, category}
2. `feature_engineering_hints` — {skewed_features, high_corr_pairs, has_missing}
3. `dataset_profile` — stored in session_state
4. `eda_decision_hub` — {verdict, downstream_plan, top_recommendations}
5. `methodology_log` — append-only action log

Most downstream pages read only 1-2 of these. No coherent narrative thread. No traceability.

### The Solution: Unified Insight Ledger

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import datetime


@dataclass
class Insight:
    """A single trackable observation from any page in the app."""
    id: str                                     # deterministic, e.g. "eda_skew_BMI"
    source_page: str                            # "02_EDA", "05_Preprocess", etc.
    category: Literal[
        "data_quality",       # missing data, duplicates, implausible values
        "distribution",       # skewness, outliers, target shape
        "relationship",       # correlations, interactions, leakage
        "topology",           # PCA structure, UMAP clusters, persistence features
        "sufficiency",        # sample size, events per variable
        "methodology",        # split strategy, model assumptions
    ]
    severity: Literal[
        "blocker",            # must resolve before modeling is defensible
        "warning",            # should address, or explicitly justify not addressing
        "info",               # worth knowing, no action required
        "opportunity",        # something to exploit (cluster structure, interaction)
    ]
    finding: str                                # what was observed (plain English)
    implication: str                            # what it means for modeling
    affected_features: List[str] = field(default_factory=list)
    recommended_action: str = ""                # what to do about it
    action_page: str = ""                       # which page handles the action
    auto_generated: bool = True                 # system-detected vs user-promoted
    resolved: bool = False
    resolved_by: str = ""                       # "Applied log transform to BMI"
    resolved_on_page: str = ""                  # "05_Preprocess"
    resolved_at: Optional[str] = None           # ISO timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InsightLedger:
    """Central insight registry across all pages."""
    insights: List[Insight] = field(default_factory=list)
    
    def add(self, insight: Insight) -> bool:
        """Add insight if not duplicate. Returns True if added."""
        if any(i.id == insight.id for i in self.insights):
            return False
        self.insights.append(insight)
        return True
    
    def resolve(self, insight_id: str, resolved_by: str, resolved_on_page: str):
        """Mark an insight as resolved."""
        for i in self.insights:
            if i.id == insight_id:
                i.resolved = True
                i.resolved_by = resolved_by
                i.resolved_on_page = resolved_on_page
                i.resolved_at = datetime.now().isoformat()
                return
    
    def get_unresolved(self, severity: Optional[str] = None, 
                       category: Optional[str] = None,
                       action_page: Optional[str] = None) -> List[Insight]:
        """Get unresolved insights, optionally filtered."""
        results = [i for i in self.insights if not i.resolved]
        if severity:
            results = [i for i in results if i.severity == severity]
        if category:
            results = [i for i in results if i.category == category]
        if action_page:
            results = [i for i in results if i.action_page == action_page]
        return results
    
    def get_resolved(self) -> List[Insight]:
        """Get all resolved insights."""
        return [i for i in self.insights if i.resolved]
    
    def get_for_features(self, features: List[str]) -> List[Insight]:
        """Get insights affecting specific features."""
        return [i for i in self.insights 
                if any(f in i.affected_features for f in features)]
    
    def summary(self) -> dict:
        """Summary counts for display."""
        unresolved = self.get_unresolved()
        return {
            'total': len(self.insights),
            'unresolved': len(unresolved),
            'blockers': len([i for i in unresolved if i.severity == 'blocker']),
            'warnings': len([i for i in unresolved if i.severity == 'warning']),
            'info': len([i for i in unresolved if i.severity == 'info']),
            'opportunities': len([i for i in unresolved if i.severity == 'opportunity']),
            'resolved': len(self.get_resolved()),
        }
    
    def narrative_for_report(self) -> str:
        """Generate narrative for the Report Export page."""
        s = self.summary()
        resolved = self.get_resolved()
        unresolved = self.get_unresolved()
        
        lines = [f"EDA and preprocessing identified {s['total']} insights. "
                 f"{s['resolved']} were resolved during the workflow."]
        
        if resolved:
            lines.append("\nResolved insights:")
            for i in resolved:
                lines.append(f"- {i.finding} → {i.resolved_by} ({i.resolved_on_page})")
        
        if unresolved:
            lines.append("\nAccepted/unresolved insights:")
            for i in unresolved:
                lines.append(f"- [{i.severity.upper()}] {i.finding}: {i.implication}")
        
        return "\n".join(lines)
```

### Ledger Integration Per Page

| Page | Role | What it does with the ledger |
|------|------|------------------------------|
| **01 Upload & Audit** | Writer | Writes data quality insights (missing cols, detected task type, cohort structure) |
| **02 EDA** | Primary writer | Writes distribution, relationship, topology, sufficiency insights. User can promote inline annotations. |
| **03 Feature Engineering** | Reader + Resolver | Reads distribution insights (skew → suggest transforms). Resolves "skewed feature" when transform applied. Writes new insights about engineered features. |
| **04 Feature Selection** | Reader + Resolver | Reads relationship insights (collinearity → suggest drops). Resolves "redundant pair" when feature dropped. |
| **05 Preprocess** | Reader + Resolver | Reads distribution insights (outliers → suggest winsorization). Resolves when preprocessing applied. |
| **06 Train & Compare** | Reader | Shows unresolved blockers as warnings. Reads methodology insights for model selection coaching. |
| **07 Explainability** | Writer | Writes insights about feature importance, unexpected model behavior. |
| **08 Sensitivity** | Writer | Writes insights about model robustness. |
| **09 Hypothesis Testing** | Reader | Reads sufficiency insights. Writes statistical test results. |
| **10 Report Export** | Reader | Calls `ledger.narrative_for_report()` to auto-generate provenance narrative. Shows full insight timeline. |

### Migration from Current System

The `feature_engineering_hints`, `eda_decision_hub`, and `eda_insights` session state keys become **computed views** of the ledger, maintained for backward compatibility during migration:

```python
# Backward compat — computed from ledger
st.session_state['feature_engineering_hints'] = {
    'skewed_features': [
        {'name': f, 'skewness': ...} 
        for i in ledger.get_unresolved(category='distribution')
        for f in i.affected_features
        if 'skew' in i.finding.lower()
    ],
    'high_corr_pairs': [...],  # from relationship insights
    'has_missing': ledger.summary()['total'] > 0  # approx
}
```

### Session State Changes

```python
# In init_session_state(), ADD:
'insight_ledger': InsightLedger(),

# KEEP (backward compat, computed from ledger):
'eda_insights': [],
'feature_engineering_hints': {},
'eda_decision_hub': {},

# KEEP (separate concern — action log, not insight tracking):
'methodology_log': [],

# In reset_data_dependent_state(), ADD:
st.session_state.insight_ledger = InsightLedger()
```

---

## Part V: Component Specifications

### Section 0 — At-a-Glance Header

```python
# Compact metric bar — always visible, no expanders
cols = st.columns([1, 1, 1, 1, 1, 1])
# Rows | Features | Numeric | Categorical | Missing % | Sufficiency badge

# Alert ribbon — only if blockers exist
blockers = ledger.get_unresolved(severity='blocker')
if blockers:
    st.error(f"🚨 {len(blockers)} blocker(s) detected — resolve before modeling")
```

**Height budget:** ~80px. No narrative. Numbers only.

### Section 1 — Data Snapshot

```python
# Interactive dataframe with column config
st.dataframe(
    df.head(100),
    use_container_width=True,  # or width="stretch" in 1.54
    column_config={col: st.column_config.NumberColumn(col, format="%.2f") 
                   for col in numeric_cols},
    height=400,
)

# Type filter pills
selected_type = st.pills(
    "Column types",
    ["All", f"# {n_numeric} Numeric", f"Aa {n_cat} Categorical", ...],
    default="All"
)

# Column inspector — st.popover per selected column
selected_col = st.selectbox("Inspect column", df.columns)
with st.popover(f"📊 {selected_col}"):
    # dtype, unique count, missing %, min/max/mean, sparkline histogram
    ...
```

### Section 2 — Shape of the Data

**Target Distribution:** Full-width, two columns (histogram+KDE | box+stats).

**Feature Distribution Gallery:**
```python
# Pagination
page_size = 9  # 3×3
feature_page = st.pills("Filter", ["All", "Numeric", "Categorical", "High Missing", "Outlier-heavy"])
filtered_features = apply_filter(feature_cols, feature_page)
total_pages = ceil(len(filtered_features) / page_size)
current_page = st.number_input("Page", 1, total_pages, 1)
page_features = filtered_features[(current_page-1)*page_size : current_page*page_size]

for row_start in range(0, len(page_features), 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        if row_start + j < len(page_features):
            feat = page_features[row_start + j]
            with col:
                fig = px.histogram(df, x=feat, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
                # Inline annotation if notable
                skew = df[feat].skew()
                if abs(skew) > 1.5:
                    st.caption(f"ℹ️ Skew = {skew:.1f}")
```

**Outlier Heatmap:**
```python
# Compute outlier % for each feature × each method
# Single Plotly heatmap, features on Y, methods on X
# Color scale: white (0%) → yellow (5%) → red (20%+)
```

**Missing Data:** Conditional render. Bar chart default. Pattern matrix expandable if >5% columns have missing.

### Section 3 — Relationships

**Correlation Matrix:**
```python
method = st.pills("Method", ["Pearson", "Spearman", "Mutual Info"], default="Pearson")
threshold = st.slider("Highlight threshold", 0.0, 1.0, 0.8)

if regime.feature_regime in ("narrow", "medium"):
    # Full heatmap
    fig = go.Figure(data=go.Heatmap(z=corr_matrix, ...))
else:
    # Top-N pairs list
    pairs = get_top_correlated_pairs(corr_matrix, n=20)
    table(pairs_df)
```

**Feature Explorer:**
```python
col1, col2, col3 = st.columns(3)
with col1:
    feat_x = st.selectbox("X axis", feature_cols)
with col2:
    feat_y = st.selectbox("Y axis", [f for f in feature_cols if f != feat_x])
with col3:
    color_by = st.selectbox("Color", ["None", target_col] + feature_cols)

fig = px.scatter(df_sampled, x=feat_x, y=feat_y, color=color_by, ...)
st.plotly_chart(fig, use_container_width=True)
```

**Interaction Detector:**
```python
# Computed in background: top 5 MI-based feature interactions
suggested_pairs = compute_top_interactions(df, feature_cols, target_col, top_n=5)
selected = st.pills("Suggested interactions", 
                     [f"{a} × {b}" for a, b in suggested_pairs])
# Selection populates Feature Explorer dropdowns
```

### Section 4 — Macro Shape

**Only rendered when `regime.feature_regime != "narrow"`.**

**Variance Profile (always first):**
```python
from sklearn.decomposition import PCA
pca = PCA().fit(df_scaled)
cumvar = np.cumsum(pca.explained_variance_ratio_)
fig = go.Figure()
fig.add_trace(go.Scatter(y=cumvar, mode='lines+markers'))
# Annotate: "N components explain X% of variance"
n_90 = np.searchsorted(cumvar, 0.9) + 1
st.plotly_chart(fig)
st.caption(f"{n_90} components explain 90% of variance in {len(feature_cols)} features")
```

**Embedding Tabs:**
```python
view = st.pills("View", ["PCA Biplot", "UMAP", "Persistence Diagram", "Mapper Graph"])

if view == "PCA Biplot":
    # 2D scatter of PC1 vs PC2 with loading arrows
    ...
elif view == "UMAP":
    # umap-learn embedding, colored by target
    from umap import UMAP
    embedding = UMAP(n_components=2, random_state=42).fit_transform(df_scaled)
    ...
elif view == "Persistence Diagram":
    # giotto-tda VietorisRipsPersistence
    from gtda.homology import VietorisRipsPersistence
    from gtda.plotting import plot_diagram
    ...
elif view == "Mapper Graph":
    # KeplerMapper with PCA lens
    # Interactive network visualization
    ...
```

**Per-view coaching annotation:** Each view includes a brief explanation of what it reveals and what it hides, plus any auto-detected insights (e.g., "UMAP reveals 3 apparent clusters" → opportunity insight to ledger).

### Section 5 — Coaching Layer

```python
st.header("Insight Summary")
s = ledger.summary()

# Compact severity bar
cols = st.columns(4)
cols[0].metric("🚨 Blockers", s['blockers'])
cols[1].metric("⚠️ Warnings", s['warnings'])
cols[2].metric("ℹ️ Info", s['info'])
cols[3].metric("💡 Opportunities", s['opportunities'])

# Insight list — grouped by severity
for insight in ledger.get_unresolved():
    icon = {"blocker": "🚨", "warning": "⚠️", "info": "ℹ️", "opportunity": "💡"}[insight.severity]
    with st.container(border=True):
        st.markdown(f"{icon} **{insight.finding}**")
        st.caption(f"→ {insight.implication}")
        if insight.recommended_action:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**Action:** {insight.recommended_action}")
            if insight.action_page:
                col2.button(f"Go to {insight.action_page} →", 
                           key=f"goto_{insight.id}")

# Reviewer risks (collapsed)
with st.expander("Reviewer-facing risks"):
    ...

# Downstream plan
with st.expander("Recommended next steps"):
    ...
```

### Section 6 — Deep Dive Diagnostics

```python
tab_ready, tab_quality, tab_advanced = st.tabs(
    ["Model Readiness", "Feature Quality", "Advanced"]
)

with tab_ready:
    # Linearity scatter, residual analysis, normality, VIF
    # Each with Run button → results inline → insights auto-added to ledger
    ...

with tab_quality:
    # Plausibility check, leakage scan, missingness deep dive
    ...

with tab_advanced:
    # Interaction analysis, dose-response, data sufficiency, quick baselines
    ...
```

### Section 7 — Table 1

Unchanged. Collapsed expander at bottom.

---

## Part VI: Implementation Plan

### Phase 1: Foundation (Insight Ledger + Regime Detection)
- [ ] Create `utils/insight_ledger.py` with `Insight` and `InsightLedger` classes
- [ ] Add `insight_ledger` to session state in `init_session_state()`
- [ ] Add `DatasetRegime` class to `ml/eda_recommender.py` or new `ml/regime.py`
- [ ] Add backward-compat shim for `feature_engineering_hints` / `eda_insights`
- [ ] Update `reset_data_dependent_state()` to clear ledger

### Phase 2: EDA Page Restructure
- [ ] Rewrite `02_EDA.py` with new section order
- [ ] Section 0: At-a-Glance Header
- [ ] Section 1: Data Snapshot (interactive table + type pills + column inspector)
- [ ] Section 2: Shape of the Data (target dist, gallery, outlier heatmap, missing)
- [ ] Section 3: Relationships (correlation matrix, target gallery, feature explorer, interaction chips)
- [ ] Section 4: Macro Shape (PCA, UMAP, persistence, mapper)
- [ ] Section 5: Coaching Layer (ledger summary, reviewer risks, downstream plan)
- [ ] Section 6: Deep Dive Diagnostics (tabbed, intent-based)
- [ ] Section 7: Table 1 (moved to bottom, unchanged)

### Phase 3: Inline Coaching + Auto-Insights
- [ ] Distribution gallery: auto-detect skew, bimodality → inline annotations + ledger writes
- [ ] Correlation matrix: auto-detect high pairs → inline annotations + ledger writes
- [ ] Target analysis: auto-detect imbalance, outliers → inline annotations + ledger writes
- [ ] Macro Shape: auto-detect clusters, effective dimensionality → inline annotations + ledger writes
- [ ] Deep Dive: existing analyses write to ledger on completion

### Phase 4: Cross-App Ledger Integration
- [ ] 03 Feature Engineering: read ledger, resolve skew/correlation insights when transforms applied
- [ ] 04 Feature Selection: read ledger, resolve redundancy insights when features dropped
- [ ] 05 Preprocess: read ledger, resolve outlier/scaling insights when preprocessing applied
- [ ] 06 Train & Compare: show unresolved blockers as warnings
- [ ] 07 Explainability: write importance insights to ledger
- [ ] 09 Hypothesis Testing: read sufficiency insights
- [ ] 10 Report Export: use `ledger.narrative_for_report()` for provenance section

### Phase 5: Polish + Adaptive Behavior
- [ ] Test with narrow (≤15), medium (16-50), wide (51-200) datasets
- [ ] Test with tiny (<100 rows) and large (>10K) datasets
- [ ] Visual refinement (consistent spacing, chart themes, responsive behavior)
- [ ] Performance profiling (UMAP/TDA compute times, caching strategy)
- [ ] Full-page wireframe screenshot review after each phase

---

## Part VII: Files Modified

### New files:
- `utils/insight_ledger.py` — Insight + InsightLedger classes
- `ml/regime.py` — DatasetRegime detection
- `ml/macro_shape.py` — PCA, UMAP, persistence, mapper computations

### Modified files:
- `pages/02_EDA.py` — full rewrite
- `utils/session_state.py` — add ledger to state, update reset
- `utils/storyline.py` — bridge add_insight() to ledger
- `pages/03_Feature_Engineering.py` — read ledger, resolve insights
- `pages/04_Feature_Selection.py` — read ledger, resolve insights
- `pages/05_Preprocess.py` — read ledger, resolve insights
- `pages/06_Train_and_Compare.py` — read ledger for warnings
- `pages/07_Explainability.py` — write to ledger
- `pages/10_Report_Export.py` — use ledger narrative
- `ml/eda_actions.py` — return structured insights for ledger

### Preserved (no changes):
- `ml/eda_recommender.py` — still computes signals, recommendations still used
- `ml/dataset_profile.py` — still computes profile, still cached
- `ml/plot_narrative.py` — still generates narratives for deep dive results
- `utils/llm_ui.py` — still provides LLM interpretation buttons
- `utils/theme.py` — CSS unchanged
- `pages/08_Sensitivity_Analysis.py` — minimal ledger integration
- `pages/09_Hypothesis_Testing.py` — minimal ledger integration
