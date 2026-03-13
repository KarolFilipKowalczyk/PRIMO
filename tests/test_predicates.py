"""
Tests for primo.predicates — I and Phi predicate classification.

Uses hand-crafted trajectories with known expected classifications,
plus spot-checks on individual measurement functions.
"""

import pytest
import numpy as np
import networkx as nx

from primo.predicates import (
    classify_I, classify_Phi, classify_rule, classify_all_rules,
    classification_cells, measure_I_scores, measure_Phi_scores,
)
from primo.trajectories import (
    embed_trajectory, embed_laplacian, embed_random_projection,
    embed_degree_profile, subspace_cosine, compression_ratio,
    spectral_dimension_estimate, curvature_homogeneity,
    law_residual_score, EMBEDDING_FUNCTIONS,
)
from primo.rules import ALL_RULES, make_initial_graphs, run_trajectory


# ══════════════════════════════════════════════════════════════════════
# Embedding tests
# ══════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    def test_laplacian_shape(self):
        G = nx.complete_graph(10)
        X = embed_laplacian(G, d=5)
        assert X.shape == (10, 5)

    def test_random_proj_shape(self):
        G = nx.complete_graph(10)
        X = embed_random_projection(G, d=5)
        assert X.shape == (10, 5)

    def test_degree_profile_shape(self):
        G = nx.complete_graph(10)
        X = embed_degree_profile(G, d=5)
        assert X.shape == (10, 5)

    def test_single_node_returns_zeros(self):
        G = nx.Graph()
        G.add_node(0)
        for fn in EMBEDDING_FUNCTIONS.values():
            X = fn(G, d=5)
            assert X.shape == (1, 5)
            assert np.allclose(X, 0)

    def test_embedding_deterministic(self):
        G = nx.complete_graph(8)
        X1 = embed_laplacian(G, d=3)
        X2 = embed_laplacian(G, d=3)
        assert np.allclose(X1, X2)


# ══════════════════════════════════════════════════════════════════════
# Subspace cosine
# ══════════════════════════════════════════════════════════════════════

class TestSubspaceCosine:
    def test_identical_subspaces(self):
        X = np.random.RandomState(0).randn(10, 3)
        sc = subspace_cosine(X, X)
        assert sc > 0.99

    def test_dissimilar_subspaces(self):
        """Random independent subspaces should have lower similarity."""
        X1 = np.random.RandomState(0).randn(20, 3)
        X2 = np.random.RandomState(99).randn(20, 3)
        sc = subspace_cosine(X1, X2)
        # Not identical, but not necessarily zero for random matrices
        assert sc < 0.99

    def test_different_sizes_padded(self):
        X1 = np.random.RandomState(0).randn(5, 3)
        X2 = np.random.RandomState(1).randn(8, 3)
        sc = subspace_cosine(X1, X2)
        assert 0.0 <= sc <= 1.0

    def test_range_is_bounded(self):
        """Subspace cosine should be in [0, 1]."""
        X1 = np.random.RandomState(0).randn(10, 3)
        X2 = np.random.RandomState(1).randn(10, 3)
        sc = subspace_cosine(X1, X2)
        assert 0.0 <= sc <= 1.0


# ══════════════════════════════════════════════════════════════════════
# Compression ratio
# ══════════════════════════════════════════════════════════════════════

class TestCompressionRatio:
    def test_constant_trajectory_is_compressible(self):
        """Repeating the same graph should yield very low ratio."""
        G = nx.complete_graph(5)
        traj = [G.copy() for _ in range(20)]
        cr = compression_ratio(traj, method="edge_list")
        assert cr < 0.3

    def test_random_trajectory_less_compressible(self):
        """Random graphs should compress less."""
        np.random.seed(42)
        traj = [nx.erdos_renyi_graph(10, 0.3) for _ in range(20)]
        cr = compression_ratio(traj, method="edge_list")
        # Should be notably higher than constant trajectory
        assert cr > 0.05

    def test_all_methods_return_valid_ratio(self):
        G = nx.complete_graph(5)
        traj = [G.copy() for _ in range(10)]
        for method in ["edge_list", "adjacency", "canonical"]:
            cr = compression_ratio(traj, method=method)
            assert 0.0 < cr <= 1.0


# ══════════════════════════════════════════════════════════════════════
# Spectral dimension
# ══════════════════════════════════════════════════════════════════════

class TestSpectralDimension:
    def test_small_graph_returns_none(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert spectral_dimension_estimate(G) is None

    def test_grid_has_finite_ds(self):
        G = nx.grid_2d_graph(5, 5)
        G = nx.convert_node_labels_to_integers(G)
        ds = spectral_dimension_estimate(G)
        assert ds is not None
        assert 0 < ds < 10

    def test_path_has_ds_near_one(self):
        G = nx.path_graph(50)
        ds = spectral_dimension_estimate(G)
        assert ds is not None
        assert 0.5 < ds < 2.0


# ══════════════════════════════════════════════════════════════════════
# Curvature homogeneity
# ══════════════════════════════════════════════════════════════════════

class TestCurvatureHomogeneity:
    def test_too_few_edges(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        assert curvature_homogeneity(G) == float('inf')

    def test_complete_graph_is_homogeneous(self):
        G = nx.complete_graph(6)
        ch = curvature_homogeneity(G)
        assert ch < 0.1  # very homogeneous


# ══════════════════════════════════════════════════════════════════════
# Law residual
# ══════════════════════════════════════════════════════════════════════

class TestLawResidual:
    def test_linear_growth(self):
        """Linearly growing edge count should fit well."""
        traj = [nx.path_graph(n) for n in range(3, 20)]
        from primo.trajectories import total_edges
        resid, model = law_residual_score(traj, total_edges)
        assert resid < 0.05
        assert model == "linear"

    def test_constant_trajectory(self):
        G = nx.complete_graph(5)
        traj = [G.copy() for _ in range(10)]
        from primo.trajectories import total_edges
        resid, model = law_residual_score(traj, total_edges)
        assert resid == 0.0
        assert model == "constant"

    def test_insufficient_data(self):
        G = nx.complete_graph(3)
        traj = [G.copy() for _ in range(2)]
        from primo.trajectories import total_edges
        resid, model = law_residual_score(traj, total_edges)
        assert resid == float('inf')
        assert model == "insufficient_data"


# ══════════════════════════════════════════════════════════════════════
# I-predicate classification
# ══════════════════════════════════════════════════════════════════════

class TestClassifyI:
    def test_converging_trajectory_is_I_positive(self):
        """Line growth should be I-positive (stable embedding convergence)."""
        np.random.seed(42)
        G0 = nx.complete_graph(2)
        traj = run_trajectory(ALL_RULES['line_growth'], G0, T=30)
        I_pos, detail = classify_I(traj)
        assert I_pos is True
        assert detail["gate"] == "passed"

    def test_random_trajectory_is_I_negative(self):
        """ER random should be I-negative (no convergence)."""
        np.random.seed(42)
        G0 = nx.erdos_renyi_graph(10, 0.3)
        traj = run_trajectory(ALL_RULES['er_random'], G0, T=30)
        I_pos, detail = classify_I(traj)
        assert I_pos is False

    def test_do_nothing_is_I_negative(self):
        np.random.seed(42)
        G0 = nx.complete_graph(3)
        traj = run_trajectory(ALL_RULES['do_nothing'], G0, T=30)
        I_pos, _ = classify_I(traj)
        assert I_pos is False

    def test_compression_gate_detail(self):
        np.random.seed(42)
        G0 = nx.complete_graph(3)
        traj = run_trajectory(ALL_RULES['line_growth'], G0, T=30)
        _, detail = classify_I(traj)
        assert "compression" in detail
        assert detail["compression"] < 0.85


# ══════════════════════════════════════════════════════════════════════
# Phi-predicate classification
# ══════════════════════════════════════════════════════════════════════

class TestClassifyPhi:
    def test_grid_growth_is_Phi_positive(self):
        """Grid growth produces stable spectral dimension."""
        np.random.seed(42)
        G0 = nx.complete_graph(3)
        traj = run_trajectory(ALL_RULES['grid_growth'], G0, T=30)
        Phi_pos, detail = classify_Phi(traj)
        assert Phi_pos is True

    def test_er_random_is_Phi_negative(self):
        np.random.seed(42)
        G0 = nx.erdos_renyi_graph(10, 0.3)
        traj = run_trajectory(ALL_RULES['er_random'], G0, T=30)
        Phi_pos, _ = classify_Phi(traj)
        assert not Phi_pos

    def test_phi_detail_has_required_keys(self):
        np.random.seed(42)
        G0 = nx.complete_graph(3)
        traj = run_trajectory(ALL_RULES['grid_growth'], G0, T=30)
        _, detail = classify_Phi(traj)
        assert "ds_mean" in detail
        assert "ds_std" in detail
        assert "best_law_resid" in detail
        assert "curv_homogeneity" in detail


# ══════════════════════════════════════════════════════════════════════
# Rule-level classification
# ══════════════════════════════════════════════════════════════════════

class TestClassifyRule:
    def test_line_growth_classification(self):
        result = classify_rule(ALL_RULES['line_growth'])
        assert result["I_positive"] is True
        assert result["Phi_positive"] is True
        assert result["I_count"] == 4
        assert result["Phi_count"] == 4

    def test_do_nothing_classification(self):
        result = classify_rule(ALL_RULES['do_nothing'])
        assert result["I_positive"] is False
        assert result["Phi_positive"] is False

    def test_result_has_seed_details(self):
        result = classify_rule(ALL_RULES['line_growth'])
        assert "seed_results" in result
        assert set(result["seed_results"].keys()) == {'K1', 'K2', 'K3', 'P3'}


# ══════════════════════════════════════════════════════════════════════
# Classification cells
# ══════════════════════════════════════════════════════════════════════

class TestClassificationCells:
    def test_all_rules_assigned(self):
        cls = {
            "a": (True, True),
            "b": (True, False),
            "c": (False, True),
            "d": (False, False),
        }
        cells = classification_cells(cls)
        assert cells["(I+, Φ+)"] == ["a"]
        assert cells["(I+, Φ-)"] == ["b"]
        assert cells["(I-, Φ+)"] == ["c"]
        assert cells["(I-, Φ-)"] == ["d"]

    def test_empty_classifications(self):
        cells = classification_cells({})
        for v in cells.values():
            assert v == []
