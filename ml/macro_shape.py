"""
Macro Shape Analysis — PCA, UMAP, Persistence Diagrams, Mapper.

Provides high-dimensional data structure visualization for the EDA page.
Only used when DatasetRegime.show_macro_shape is True (≥16 features).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

@st.cache_data
def compute_pca(
    _df_numeric: pd.DataFrame,
    n_components: Optional[int] = None,
    max_features: int = 200,
) -> Dict[str, Any]:
    """Compute PCA on numeric features.
    
    Returns dict with:
        - components: transformed data (n_samples × n_components)
        - explained_variance_ratio: per-component
        - cumulative_variance: cumulative explained
        - loadings: component loadings (n_components × n_features)
        - feature_names: original feature names
        - n_components_90: components needed for 90% variance
    """
    df = _df_numeric.dropna()
    if len(df) < 10 or len(df.columns) < 2:
        return {"error": "Insufficient data for PCA"}

    # Limit features if ultra-wide
    cols = list(df.columns[:max_features])
    X = df[cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n = n_components or min(len(cols), len(df), 50)
    pca = PCA(n_components=n, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cumvar, 0.9)) + 1
    
    return {
        "components": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
        "loadings": pca.components_,
        "feature_names": cols,
        "n_components_90": min(n_90, n),
        "total_variance_explained": float(cumvar[-1]) if len(cumvar) > 0 else 0,
    }


def plot_scree(pca_result: Dict[str, Any]) -> go.Figure:
    """Scree plot showing cumulative explained variance."""
    cumvar = pca_result["cumulative_variance"]
    n_90 = pca_result["n_components_90"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumvar) + 1)),
        y=cumvar,
        mode="lines+markers",
        name="Cumulative Variance",
        marker=dict(size=6),
        line=dict(color="#667eea"),
    ))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#94a3b8",
                  annotation_text="90% threshold")
    fig.add_vline(x=n_90, line_dash="dot", line_color="#22c55e",
                  annotation_text=f"{n_90} components")
    fig.update_layout(
        title="Cumulative Explained Variance",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Variance Explained",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=350,
    )
    return fig


def plot_pca_biplot(
    pca_result: Dict[str, Any],
    color_values: Optional[np.ndarray] = None,
    color_label: str = "Target",
    max_arrows: int = 10,
) -> go.Figure:
    """2D PCA biplot with optional coloring and loading arrows."""
    components = pca_result["components"]
    loadings = pca_result["loadings"]
    features = pca_result["feature_names"]
    evr = pca_result["explained_variance_ratio"]
    
    fig = go.Figure()
    
    # Scatter of samples
    if color_values is not None:
        fig.add_trace(go.Scatter(
            x=components[:, 0], y=components[:, 1],
            mode="markers",
            marker=dict(
                color=color_values, colorscale="Viridis",
                size=4, opacity=0.6, colorbar=dict(title=color_label),
            ),
            name="Samples",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=components[:, 0], y=components[:, 1],
            mode="markers",
            marker=dict(color="#667eea", size=4, opacity=0.5),
            name="Samples",
        ))
    
    # Loading arrows (top N by magnitude)
    loading_magnitudes = np.sqrt(loadings[0] ** 2 + loadings[1] ** 2)
    top_idx = np.argsort(loading_magnitudes)[-max_arrows:]
    
    scale = max(abs(components[:, 0]).max(), abs(components[:, 1]).max()) * 0.8
    for idx in top_idx:
        x_end = loadings[0, idx] * scale
        y_end = loadings[1, idx] * scale
        fig.add_annotation(
            x=x_end, y=y_end, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor="#ef4444",
        )
        fig.add_annotation(
            x=x_end * 1.1, y=y_end * 1.1,
            text=features[idx], showarrow=False,
            font=dict(size=10, color="#ef4444"),
        )
    
    fig.update_layout(
        title=f"PCA Biplot (PC1: {evr[0]:.1%}, PC2: {evr[1]:.1%})",
        xaxis_title=f"PC1 ({evr[0]:.1%})",
        yaxis_title=f"PC2 ({evr[1]:.1%})",
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

@st.cache_data
def compute_umap(
    _df_numeric: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_samples: int = 10_000,
    max_features: int = 200,
) -> Dict[str, Any]:
    """Compute UMAP embedding on numeric features."""
    df = _df_numeric.dropna()
    if len(df) < 15 or len(df.columns) < 2:
        return {"error": "Insufficient data for UMAP"}
    
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    cols = list(df.columns[:max_features])
    X = df[cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        embedding = reducer.fit_transform(X_scaled)
        return {
            "embedding": embedding,
            "sample_indices": df.index.tolist(),
            "n_samples": len(df),
            "n_features": len(cols),
        }
    except ImportError:
        return {"error": "umap-learn not installed"}
    except Exception as e:
        return {"error": f"UMAP failed: {str(e)[:100]}"}


def plot_umap(
    umap_result: Dict[str, Any],
    color_values: Optional[np.ndarray] = None,
    color_label: str = "Target",
) -> go.Figure:
    """Plot UMAP embedding."""
    embedding = umap_result["embedding"]
    
    fig = go.Figure()
    if color_values is not None:
        # Align color values to sampled indices
        fig.add_trace(go.Scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            mode="markers",
            marker=dict(
                color=color_values, colorscale="Viridis",
                size=4, opacity=0.6, colorbar=dict(title=color_label),
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            mode="markers",
            marker=dict(color="#667eea", size=4, opacity=0.5),
        ))
    
    n = umap_result["n_samples"]
    p = umap_result["n_features"]
    fig.update_layout(
        title=f"UMAP Embedding ({n:,} samples, {p} features)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Persistence Diagrams (TDA)
# ---------------------------------------------------------------------------

@st.cache_data
def compute_persistence(
    _df_numeric: pd.DataFrame,
    max_samples: int = 2_000,
    max_features: int = 50,
    homology_dimensions: Tuple[int, ...] = (0, 1),
) -> Dict[str, Any]:
    """Compute persistent homology via giotto-tda."""
    df = _df_numeric.dropna()
    if len(df) < 20 or len(df.columns) < 2:
        return {"error": "Insufficient data for persistence computation"}
    
    # Sample aggressively — TDA is O(n³) in worst case
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    cols = list(df.columns[:max_features])
    X = df[cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        from gtda.homology import VietorisRipsPersistence
        
        vr = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            max_edge_length=np.inf,
            n_jobs=-1,
        )
        diagrams = vr.fit_transform(X_scaled[np.newaxis, :, :])
        
        # Parse diagram: (birth, death, dimension)
        diag = diagrams[0]
        features_by_dim = {}
        for dim in homology_dimensions:
            mask = diag[:, 2] == dim
            pts = diag[mask, :2]
            # Remove infinite deaths for H0
            finite_mask = np.isfinite(pts[:, 1])
            pts_finite = pts[finite_mask]
            persistence = pts_finite[:, 1] - pts_finite[:, 0]
            features_by_dim[dim] = {
                "points": pts_finite,
                "persistence": persistence,
                "n_features": len(pts_finite),
                "max_persistence": float(persistence.max()) if len(persistence) > 0 else 0,
                "mean_persistence": float(persistence.mean()) if len(persistence) > 0 else 0,
            }
        
        return {
            "diagram": diag,
            "features_by_dim": features_by_dim,
            "n_samples": len(df),
            "n_features": len(cols),
            "homology_dimensions": homology_dimensions,
        }
    except ImportError:
        return {"error": "giotto-tda not installed"}
    except Exception as e:
        return {"error": f"Persistence computation failed: {str(e)[:100]}"}


def plot_persistence_diagram(result: Dict[str, Any]) -> go.Figure:
    """Plot persistence diagram (birth vs death)."""
    diag = result["diagram"]
    dims = result["homology_dimensions"]
    
    colors = {0: "#667eea", 1: "#ef4444", 2: "#22c55e"}
    dim_names = {0: "H₀ (components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}
    
    fig = go.Figure()
    
    # Diagonal reference line
    finite_mask = np.isfinite(diag[:, 1])
    if finite_mask.any():
        max_val = diag[finite_mask, :2].max() * 1.1
    else:
        max_val = 1.0
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="#94a3b8"),
        showlegend=False,
    ))
    
    for dim in dims:
        mask = (diag[:, 2] == dim) & np.isfinite(diag[:, 1])
        pts = diag[mask, :2]
        if len(pts) == 0:
            continue
        persistence = pts[:, 1] - pts[:, 0]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode="markers",
            name=dim_names.get(dim, f"H{dim}"),
            marker=dict(
                color=colors.get(dim, "#94a3b8"),
                size=np.clip(persistence * 20, 4, 15),
                opacity=0.7,
            ),
            hovertemplate="Birth: %{x:.3f}<br>Death: %{y:.3f}<br>Persistence: %{customdata:.3f}",
            customdata=persistence,
        ))
    
    fig.update_layout(
        title="Persistence Diagram",
        xaxis_title="Birth",
        yaxis_title="Death",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_persistence_barcode(result: Dict[str, Any]) -> go.Figure:
    """Plot persistence barcode (horizontal bars)."""
    dims = result["homology_dimensions"]
    colors = {0: "#667eea", 1: "#ef4444", 2: "#22c55e"}
    dim_names = {0: "H₀ (components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}
    
    fig = go.Figure()
    y_offset = 0
    
    for dim in dims:
        info = result["features_by_dim"].get(dim, {})
        pts = info.get("points", np.array([]))
        if len(pts) == 0:
            continue
        
        # Sort by persistence (longest first)
        persistence = pts[:, 1] - pts[:, 0]
        order = np.argsort(-persistence)
        pts = pts[order][:30]  # Show top 30 per dimension
        
        for i, (birth, death) in enumerate(pts):
            fig.add_trace(go.Scatter(
                x=[birth, death], y=[y_offset + i, y_offset + i],
                mode="lines",
                line=dict(color=colors.get(dim, "#94a3b8"), width=3),
                name=dim_names.get(dim, f"H{dim}") if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=f"Birth: {birth:.3f}, Death: {death:.3f}, Persistence: {death - birth:.3f}",
            ))
        y_offset += len(pts) + 2
    
    fig.update_layout(
        title="Persistence Barcode",
        xaxis_title="Scale",
        yaxis_visible=False,
        template="plotly_white",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Mapper Graph
# ---------------------------------------------------------------------------

@st.cache_data
def compute_mapper(
    _df_numeric: pd.DataFrame,
    n_cubes: int = 10,
    overlap: float = 0.3,
    max_samples: int = 3_000,
    max_features: int = 100,
) -> Dict[str, Any]:
    """Compute Mapper graph using PCA as lens function.
    
    Returns graph as nodes + edges for Plotly network visualization.
    """
    df = _df_numeric.dropna()
    if len(df) < 30 or len(df.columns) < 2:
        return {"error": "Insufficient data for Mapper"}
    
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    cols = list(df.columns[:max_features])
    X = df[cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        # Use PCA projection as lens
        pca = PCA(n_components=2, random_state=42)
        lens = pca.fit_transform(X_scaled)
        
        # Manual Mapper implementation (avoid KeplerMapper dependency)
        # 1. Cover the lens space with overlapping cubes
        nodes = {}  # node_id -> list of sample indices
        node_id = 0
        
        for dim in range(2):
            vals = lens[:, dim]
            min_val, max_val = vals.min(), vals.max()
            cube_width = (max_val - min_val) / n_cubes
            step = cube_width * (1 - overlap)
            
            if dim == 0:
                # First dimension: create initial bins
                for i in range(n_cubes + 1):
                    low = min_val + i * step
                    high = low + cube_width
                    mask = (vals >= low) & (vals <= high)
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        nodes[node_id] = indices.tolist()
                        node_id += 1
        
        # 2. Cluster within each node using simple DBSCAN-like approach
        from sklearn.cluster import DBSCAN
        
        refined_nodes = {}
        refined_id = 0
        for nid, indices in nodes.items():
            if len(indices) < 3:
                refined_nodes[refined_id] = indices
                refined_id += 1
                continue
            
            sub_X = X_scaled[indices]
            eps = np.median(np.std(sub_X, axis=0)) * 1.5
            if eps < 1e-10:
                eps = 0.5
            
            clustering = DBSCAN(eps=eps, min_samples=2).fit(sub_X)
            labels = clustering.labels_
            
            for label in set(labels):
                if label == -1:
                    continue
                cluster_indices = [indices[i] for i, l in enumerate(labels) if l == label]
                if cluster_indices:
                    refined_nodes[refined_id] = cluster_indices
                    refined_id += 1
        
        # 3. Build edges: nodes sharing samples are connected
        edges = []
        node_ids = list(refined_nodes.keys())
        sample_to_nodes = {}
        for nid, indices in refined_nodes.items():
            for idx in indices:
                sample_to_nodes.setdefault(idx, []).append(nid)
        
        edge_set = set()
        for idx, nids in sample_to_nodes.items():
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    edge = (min(nids[i], nids[j]), max(nids[i], nids[j]))
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edges.append(edge)
        
        # 4. Compute node positions (mean of lens values)
        node_positions = {}
        node_sizes = {}
        for nid, indices in refined_nodes.items():
            node_positions[nid] = lens[indices].mean(axis=0)
            node_sizes[nid] = len(indices)
        
        return {
            "nodes": refined_nodes,
            "edges": edges,
            "node_positions": node_positions,
            "node_sizes": node_sizes,
            "n_nodes": len(refined_nodes),
            "n_edges": len(edges),
            "n_samples": len(df),
        }
    except Exception as e:
        return {"error": f"Mapper computation failed: {str(e)[:200]}"}


def plot_mapper(
    mapper_result: Dict[str, Any],
    color_values: Optional[np.ndarray] = None,
    color_label: str = "Target",
) -> go.Figure:
    """Plot Mapper graph as network visualization."""
    nodes = mapper_result["nodes"]
    edges = mapper_result["edges"]
    positions = mapper_result["node_positions"]
    sizes = mapper_result["node_sizes"]
    
    fig = go.Figure()
    
    # Edges
    for n1, n2 in edges:
        if n1 in positions and n2 in positions:
            p1, p2 = positions[n1], positions[n2]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                mode="lines",
                line=dict(color="#94a3b8", width=1),
                showlegend=False,
                hoverinfo="skip",
            ))
    
    # Nodes
    node_x = [positions[n][0] for n in positions]
    node_y = [positions[n][1] for n in positions]
    node_size = [min(max(sizes[n] * 0.5, 5), 30) for n in positions]
    
    if color_values is not None:
        # Mean target value per node
        node_colors = []
        for nid in positions:
            indices = nodes[nid]
            valid_indices = [i for i in indices if i < len(color_values)]
            if valid_indices:
                node_colors.append(float(np.mean(color_values[valid_indices])))
            else:
                node_colors.append(0)
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_colors,
                colorscale="Viridis",
                colorbar=dict(title=color_label),
                line=dict(width=1, color="white"),
            ),
            hovertemplate="Samples: %{customdata}<br>Mean target: %{marker.color:.3f}",
            customdata=[sizes[n] for n in positions],
        ))
    else:
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color="#667eea",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="Samples: %{customdata}",
            customdata=[sizes[n] for n in positions],
        ))
    
    n_nodes = mapper_result["n_nodes"]
    n_edges = mapper_result["n_edges"]
    fig.update_layout(
        title=f"Mapper Graph ({n_nodes} nodes, {n_edges} edges)",
        xaxis_title="PCA Lens 1",
        yaxis_title="PCA Lens 2",
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    return fig
