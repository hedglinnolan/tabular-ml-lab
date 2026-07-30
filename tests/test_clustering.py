"""
k-means cluster exploration (ml/clustering.py).

The tests that matter here are the refusals. k-means returns k non-empty
clusters on uniform noise exactly as readily as on real subgroups, so the
value of this module is entirely in whether it declines to call the first one
structure. Anything that weakens the permutation baseline, the recommendation
threshold, or the dominance check should fail here rather than reach a user
who is about to name a patient phenotype.
"""
import numpy as np
import pandas as pd
import pytest

from ml import clustering as clus


# ── Fixtures ─────────────────────────────────────────────────────────

def _blobs(n=600, seed=0):
    """Three well-separated groups plus one pure-noise column."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [6, 6, 0], [0, 6, 6]])
    grp = rng.integers(0, 3, n)
    X = centers[grp] + rng.normal(0, 1.0, (n, 3))
    df = pd.DataFrame(X, columns=["ldl", "hdl", "crp"])
    df["noise"] = rng.normal(0, 1, n)
    return df, grp


def _noise(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0, 1, (n, 5)), columns=[f"v{i}" for i in range(5)])


def _prep(df, features=None, **kw):
    return clus.prepare_cluster_matrix(df, features or list(df.columns), data_id=id(df), **kw)


def _sweep(prep, k_max=5):
    return clus.sweep_k(prep["X"], tuple(range(2, k_max + 1)), prep["variable_spans"])


# ── The refusals ─────────────────────────────────────────────────────

class TestRefusesStructureThatIsNotThere:

    def test_uniform_noise_gets_no_recommendation(self):
        prep = _prep(_noise())
        assert _sweep(prep)["recommended_k"] is None

    def test_single_gaussian_blob_gets_no_recommendation(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(0, 1, (600, 6)), columns=[f"g{i}" for i in range(6)])
        assert _sweep(_prep(df))["recommended_k"] is None

    def test_noise_silhouette_does_not_beat_its_own_shuffled_baseline(self):
        rows = _sweep(_prep(_noise()))["table"]
        for row in rows:
            assert row["excess"] < clus.MIN_MEANINGFUL_EXCESS, (
                f"k={row['k']} claimed a {row['excess']:.3f} margin over noise"
            )

    def test_a_hairs_breadth_margin_is_not_promoted(self):
        """Both gates, not either: absolute size AND width against the spread."""
        assert clus.MIN_MEANINGFUL_EXCESS > 0
        assert clus.MIN_EXCESS_SD_MULTIPLE >= 1.0


class TestFindsStructureThatIsThere:

    def test_three_blobs_recommend_three(self):
        df, _ = _blobs()
        assert _sweep(_prep(df))["recommended_k"] == 3

    def test_recovered_labels_match_the_planted_groups(self):
        from sklearn.metrics import adjusted_rand_score
        df, truth = _blobs()
        prep = _prep(df)
        fit = clus.fit_clusters(prep["X"], 3)
        assert adjusted_rand_score(truth, fit["labels"]) > 0.95

    def test_real_structure_is_seed_stable(self):
        df, _ = _blobs()
        stability = clus.seed_stability(_prep(df)["X"], 3)
        assert stability["verdict"] == "stable"
        assert stability["mean_ari"] > 0.9

    def test_profile_ranks_the_separating_columns_above_the_noise_column(self):
        df, _ = _blobs()
        prep = _prep(df)
        fit = clus.fit_clusters(prep["X"], 3)
        profile = clus.cluster_profile(
            df, fit["labels"], prep["row_index"], prep["numeric_cols"], data_id=1
        )
        assert profile["ranked_features"][-1] == "noise"


# ── The permutation baseline ─────────────────────────────────────────

class TestPermutationBaseline:

    def test_one_hot_columns_move_together(self):
        """Shuffling dummy columns independently yields invalid rows.

        Each categorical variable must occupy one span, so a shuffled row is
        still a valid encoding. Otherwise the baseline scores badly for a
        reason that has nothing to do with structure, and every categorical
        dataset looks clustered.
        """
        rng = np.random.default_rng(5)
        n = 300
        df = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "site": rng.choice(["A", "B", "C", "D"], n),
        })
        prep = _prep(df)
        spans = prep["variable_spans"]
        widths = [end - start for start, end in spans]
        assert 1 in widths, "the numeric column should be its own span"
        assert max(widths) == prep["encoded_levels"]["site"] > 1, (
            "the one-hot block for 'site' is not held together as one span"
        )
        # Spans must tile the matrix without gaps or overlap.
        assert sorted(spans) == list(spans)
        assert spans[0][0] == 0
        assert spans[-1][1] == prep["X"].shape[1]
        for (_, prev_end), (start, _) in zip(spans, spans[1:]):
            assert prev_end == start

    def test_shuffling_preserves_every_column_marginal(self):
        """The null destroys relationships, not distributions."""
        prep = _prep(_noise(n=200))
        X = prep["X"]
        rng = np.random.default_rng(0)
        shuffled = X.copy()
        for start, end in prep["variable_spans"]:
            shuffled[:, start:end] = shuffled[rng.permutation(X.shape[0]), start:end]
        for j in range(X.shape[1]):
            assert np.allclose(np.sort(X[:, j]), np.sort(shuffled[:, j]))

    def test_baseline_is_reported_for_every_k(self):
        rows = _sweep(_prep(_noise(n=300)))["table"]
        for row in rows:
            assert np.isfinite(row["null_silhouette"])
            assert np.isfinite(row["p_value"])


# ── Matrix preparation ───────────────────────────────────────────────

class TestPreparation:

    def test_drops_constant_sparse_and_identifier_columns(self):
        rng = np.random.default_rng(7)
        n = 300
        df = pd.DataFrame({
            "good": rng.normal(0, 1, n),
            "also_good": rng.normal(0, 1, n),
            "constant": 1.0,
            "mostly_missing": np.where(rng.random(n) < 0.8, np.nan, 1.0),
            "row_id": [f"r{i}" for i in range(n)],
        })
        prep = _prep(df)
        dropped = " ".join(prep["dropped"])
        assert "constant" in dropped
        assert "mostly_missing" in dropped
        assert "row_id" in dropped
        assert prep["numeric_cols"] == ["good", "also_good"]

    def test_scales_numeric_columns(self):
        """Unscaled data clusters on unit choices; this must not be optional."""
        rng = np.random.default_rng(8)
        n = 400
        df = pd.DataFrame({
            "grams": rng.normal(0, 1, n),
            "micrograms": rng.normal(0, 1, n) * 1_000_000,
        })
        X = _prep(df)["X"]
        assert np.allclose(X.std(axis=0), 1.0, atol=0.05)

    def test_transforms_heavy_skew_before_scaling(self):
        rng = np.random.default_rng(9)
        n = 500
        df = pd.DataFrame({
            "crp": rng.lognormal(1.0, 1.8, n),
            "age": rng.normal(60, 10, n),
        })
        prep = _prep(df)
        assert prep["skew_transformed"] == ["crp"]
        crp = prep["X"][:, prep["numeric_cols"].index("crp")]
        assert abs(pd.Series(crp).skew()) < clus.SKEW_TRANSFORM_ABOVE

    def test_one_hot_block_is_not_standardized(self):
        """Standardizing dummies turns a 1% level into a ~10 SD spike.

        That is the mechanism behind spurious rare-category "subtypes": the
        handful of rows carrying the level get pushed far from everyone else
        and k-means gives them a cluster.
        """
        rng = np.random.default_rng(10)
        n = 1000
        df = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "site": rng.choice(["A", "B", "rare"], n, p=[0.6, 0.39, 0.01]),
        })
        prep = _prep(df)
        start, end = prep["variable_spans"][-1]
        dummies = prep["X"][:, start:end]
        assert dummies.max() <= 1.0 + 1e-9, (
            "one-hot values exceed 1 — the block was standardized"
        )
        assert set(np.unique(dummies)).issubset({0.0, 1.0})

    def test_categorical_weight_scales_the_block(self):
        rng = np.random.default_rng(11)
        n = 300
        df = pd.DataFrame({"num": rng.normal(0, 1, n), "site": rng.choice(["A", "B"], n)})
        light = _prep(df, categorical_weight=0.5)
        heavy = _prep(df, categorical_weight=2.0)
        s, e = light["variable_spans"][-1]
        assert heavy["X"][:, s:e].max() == pytest.approx(4 * light["X"][:, s:e].max())

    def test_subsamples_large_frames_and_says_so(self, monkeypatch):
        monkeypatch.setattr(clus, "MAX_FIT_ROWS", 200)
        rng = np.random.default_rng(12)
        df = pd.DataFrame(rng.normal(0, 1, (900, 3)), columns=list("abc"))
        prep = clus.prepare_cluster_matrix(df, list(df.columns), max_rows=200, data_id="sub")
        assert prep["sampled"] is True
        assert prep["n_rows"] == 200
        assert prep["n_source_rows"] == 900
        assert len(prep["row_index"]) == 200

    def test_errors_instead_of_raising_when_nothing_usable_survives(self):
        df = pd.DataFrame({"constant": [1.0] * 50, "also": [2.0] * 50})
        assert "error" in _prep(df)


# ── Power and dominance ──────────────────────────────────────────────

class TestPowerCap:

    @pytest.mark.parametrize("n,expected", [(45, 2), (150, 5), (240, 8), (10_000, 8)])
    def test_k_is_capped_by_observations_per_subgroup(self, n, expected):
        assert clus.max_supported_k(n) == expected

    def test_never_returns_less_than_two(self):
        assert clus.max_supported_k(5) == 2


class TestDominance:

    def test_flags_a_partition_that_is_one_column_relabeled(self):
        rng = np.random.default_rng(13)
        n = 600
        flag = rng.integers(0, 2, n)
        df = pd.DataFrame({
            "marker": flag * 15.0 + rng.normal(0, 0.3, n),
            "n1": rng.normal(0, 1, n),
            "n2": rng.normal(0, 1, n),
            "n3": rng.normal(0, 1, n),
        })
        prep = _prep(df)
        fit = clus.fit_clusters(prep["X"], 2)
        dom = clus.feature_dominance(df, fit["labels"], prep["row_index"], list(df.columns), data_id=1)
        assert dom["dominant"] == "marker"

    def test_does_not_flag_genuine_multivariate_structure(self):
        """Several columns scoring high is the opposite finding."""
        df, _ = _blobs()
        prep = _prep(df)
        fit = clus.fit_clusters(prep["X"], 3)
        dom = clus.feature_dominance(df, fit["labels"], prep["row_index"], list(df.columns), data_id=2)
        assert dom["dominant"] is None

    def test_ignores_near_unique_categorical_columns(self):
        """A row identifier determines any partition and explains nothing."""
        rng = np.random.default_rng(14)
        n = 300
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "row_id": [f"r{i}" for i in range(n)],
        })
        prep = _prep(df, features=["a", "b"])
        fit = clus.fit_clusters(prep["X"], 3)
        dom = clus.feature_dominance(df, fit["labels"], prep["row_index"], list(df.columns), data_id=3)
        assert "row_id" not in [r["Feature"] for r in dom["table"]]


# ── Target association: the one non-circular check ───────────────────

class TestTargetAssociation:

    def test_regression_target_tracking_the_groups_shows_a_large_effect(self):
        rng = np.random.default_rng(15)
        df, truth = _blobs()
        df["outcome"] = 10 + 5 * truth + rng.normal(0, 1, len(df))
        prep = _prep(df, features=["ldl", "hdl", "crp", "noise"])
        fit = clus.fit_clusters(prep["X"], 3)
        assoc = clus.target_association(
            df, fit["labels"], prep["row_index"], "outcome", "regression", data_id=1
        )
        assert assoc["kind"] == "regression"
        assert assoc["effect"] > 0.5
        assert assoc["p_value"] < 0.01

    def test_unrelated_target_shows_a_small_effect(self):
        rng = np.random.default_rng(16)
        df, _ = _blobs()
        df["outcome"] = rng.normal(0, 1, len(df))
        prep = _prep(df, features=["ldl", "hdl", "crp", "noise"])
        fit = clus.fit_clusters(prep["X"], 3)
        assoc = clus.target_association(
            df, fit["labels"], prep["row_index"], "outcome", "regression", data_id=2
        )
        assert assoc["effect"] < 0.05

    def test_classification_target_column_names_are_all_strings(self):
        """A mixed-type column index does not round-trip through Arrow."""
        rng = np.random.default_rng(17)
        df, truth = _blobs(n=400)
        df["cls"] = (truth > 0).astype(int)
        prep = _prep(df, features=["ldl", "hdl", "crp"])
        fit = clus.fit_clusters(prep["X"], 3)
        assoc = clus.target_association(
            df, fit["labels"], prep["row_index"], "cls", "classification", data_id=3
        )
        assert assoc["kind"] == "classification"
        assert all(isinstance(c, str) for c in assoc["table"].columns)


# ── Plots build ──────────────────────────────────────────────────────

class TestPlots:

    def test_every_plot_builds(self):
        df, _ = _blobs(n=300)
        prep = _prep(df)
        sweep = _sweep(prep, k_max=4)
        fit = clus.fit_clusters(prep["X"], 3)
        proj = clus.project_for_display(prep["X"])
        profile = clus.cluster_profile(
            df, fit["labels"], prep["row_index"], prep["numeric_cols"], data_id=1
        )
        assert clus.plot_k_sweep(sweep) is not None
        assert clus.plot_cluster_scatter(proj["coords"], fit["labels"], proj["explained"]) is not None
        assert clus.plot_silhouette_knife(
            fit["silhouette_samples"], fit["labels"][fit["silhouette_index"]], fit["silhouette"]
        ) is not None
        assert clus.plot_cluster_profile(profile["centroids_z"]) is not None

    def test_cluster_colors_are_stable_and_distinct(self):
        colors = clus.cluster_colors(5)
        assert len(colors) == 5
        assert len(set(colors)) == 5
        assert clus.cluster_colors(5)[:3] == clus.cluster_colors(3)

    def test_silhouette_reading_bands(self):
        assert clus.silhouette_reading(0.7) == "reasonable separation"
        assert clus.silhouette_reading(0.3) == "weak, overlapping structure"
        assert clus.silhouette_reading(0.05) == "no substantial structure"
        assert clus.silhouette_reading(float("nan")) == "not computable"
