"""
PRIMO Trajectories — generation, tensor conversion, and checkpointing.

This module sits between rules.py (which defines rewrite rules) and
predicates.py (which computes I/Φ scores). It handles:

- Generating trajectory sets: run each rule from each seed graph
- Converting graph trajectories to embedding matrices (numpy arrays)
- Padding/truncating to fixed sizes for batched computation
- Checkpointing trajectory results to disk for resumption

Usage:
    from primo.trajectories import generate_trajectories, embed_trajectory
    trajs = generate_trajectories(rules_dict, seeds_dict, T=30)
    embeddings = embed_trajectory(traj, embed_fn, d=5)
"""

import json
import zlib
import hashlib
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

from primo.config import (
    T_DEFAULT, N_MAX, MASTER_SEED, EMBEDDING_DIM,
    EMBEDDING_PROJECTION_SEED, CHECKPOINT_INTERVAL, DATA_DIR,
)
from primo.backend import get_backend


# ══════════════════════════════════════════════════════════════════════
# TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_trajectories(rules, seeds, T=None, n_max=None, seed=None):
    """Generate trajectories for all (rule, seed) combinations.

    Parameters
    ----------
    rules : dict
        name → callable rule function.
    seeds : dict
        name → nx.Graph initial graph.
    T : int or None
        Steps per trajectory. Default: config.T_DEFAULT.
    n_max : int or None
        Safety cap on node count. Default: config.N_MAX.
    seed : int or None
        Random seed reset before each trajectory. Default: config.MASTER_SEED.

    Returns
    -------
    dict
        (rule_name, seed_name) → list of nx.Graph
    """
    T = T if T is not None else T_DEFAULT
    n_max = n_max or N_MAX
    seed = seed if seed is not None else MASTER_SEED

    cached = {}
    for rule_name, rule_fn in rules.items():
        for seed_name, G0 in seeds.items():
            np.random.seed(seed)
            cached[(rule_name, seed_name)] = _run_trajectory(
                rule_fn, G0, T, n_max)
    return cached


def _run_trajectory(rule_fn, G0, T, n_max):
    """Run a single rule from G0 for T steps with safety cap."""
    traj = [G0.copy()]
    G = G0.copy()
    for _ in range(T):
        try:
            G = rule_fn(G)
            if G.number_of_nodes() > n_max:
                break
        except Exception:
            break
        traj.append(G.copy())
    return traj


# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════

def embed_laplacian(G, d=None):
    """Laplacian spectral embedding: smallest non-trivial eigenvectors."""
    d = d or EMBEDDING_DIM
    if G.number_of_nodes() < 2:
        return np.zeros((1, d))
    L = nx.laplacian_matrix(G).toarray().astype(float)
    n = L.shape[0]
    actual_d = min(d, n)
    vals, vecs = np.linalg.eigh(L)
    return vecs[:, 1:actual_d + 1] if actual_d > 1 else vecs[:, :1]


def embed_random_projection(G, d=None):
    """Random projection embedding: A @ R where R is a fixed random matrix."""
    d = d or EMBEDDING_DIM
    if G.number_of_nodes() < 2:
        return np.zeros((1, d))
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = A.shape[0]
    R = np.random.RandomState(EMBEDDING_PROJECTION_SEED).randn(n, min(d, n))
    return A @ R[:n, :min(d, n)]


def embed_degree_profile(G, d=None):
    """Degree-profile embedding: local structural features per node.

    Features: [normalized_degree, clustering_coeff, avg_neighbor_degree,
               2hop_size, eccentricity_proxy], padded/truncated to d.
    """
    d = d or EMBEDDING_DIM
    if G.number_of_nodes() < 2:
        return np.zeros((1, d))
    n = G.number_of_nodes()
    X = np.zeros((n, d))
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    clustering = nx.clustering(G)

    for v in nodes:
        i = node_idx[v]
        deg = degrees[v]
        X[i, 0] = deg / max(max_deg, 1)
        if d > 1:
            X[i, 1] = clustering[v]
        if d > 2:
            nbrs = list(G.neighbors(v))
            if nbrs:
                X[i, 2] = np.mean([degrees[u] for u in nbrs]) / max(max_deg, 1)
        if d > 3:
            hop2 = set()
            for u in G.neighbors(v):
                hop2.update(G.neighbors(u))
            hop2.discard(v)
            X[i, 3] = len(hop2) / max(n, 1)
        if d > 4 and nx.is_connected(G):
            hub = max(degrees, key=degrees.get)
            try:
                X[i, 4] = nx.shortest_path_length(G, v, hub) / max(n, 1)
            except nx.NetworkXNoPath:
                X[i, 4] = 1.0
    return X


# Embedding registry
EMBEDDING_FUNCTIONS = {
    "laplacian": embed_laplacian,
    "random_proj": embed_random_projection,
    "degree_prof": embed_degree_profile,
}


def get_embedding_fn(name):
    """Look up an embedding function by name."""
    if name not in EMBEDDING_FUNCTIONS:
        raise ValueError(
            f"Unknown embedding: {name}. "
            f"Available: {list(EMBEDDING_FUNCTIONS.keys())}")
    return EMBEDDING_FUNCTIONS[name]


# ══════════════════════════════════════════════════════════════════════
# TENSOR CONVERSION — graph trajectories to embedding sequences
# ══════════════════════════════════════════════════════════════════════

def embed_trajectory(traj, embed_fn, d=None):
    """Compute embeddings for each graph in a trajectory.

    Parameters
    ----------
    traj : list of nx.Graph
    embed_fn : callable
        Graph → numpy array (n_nodes × d).
    d : int or None
        Embedding dimension. Default: config.EMBEDDING_DIM.

    Returns
    -------
    list of numpy arrays (or None for failed embeddings)
    """
    d = d or EMBEDDING_DIM
    embeddings = []
    for G in traj:
        try:
            embeddings.append(embed_fn(G, d=d))
        except Exception:
            embeddings.append(None)
    return embeddings


def pad_embedding(X, n_max, d=None):
    """Pad an embedding matrix to (n_max, d) with zeros.

    Parameters
    ----------
    X : numpy array (n, d_actual)
    n_max : int
        Target number of rows.
    d : int or None
        Target number of columns. Default: config.EMBEDDING_DIM.

    Returns
    -------
    numpy array (n_max, d)
    """
    d = d or EMBEDDING_DIM
    if X is None:
        return np.zeros((n_max, d))
    n, d_actual = X.shape
    result = np.zeros((n_max, d))
    n_copy = min(n, n_max)
    d_copy = min(d_actual, d)
    result[:n_copy, :d_copy] = X[:n_copy, :d_copy]
    return result


def trajectory_to_tensor(traj, embed_fn, n_max=None, d=None):
    """Convert a graph trajectory to a 3D numpy array.

    Parameters
    ----------
    traj : list of nx.Graph
    embed_fn : callable
    n_max : int or None
        Pad node dimension to this. Default: config.N_MAX.
    d : int or None
        Embedding dimension. Default: config.EMBEDDING_DIM.

    Returns
    -------
    numpy array (T+1, n_max, d)
        Padded embedding tensor.
    mask : numpy array (T+1,)
        Boolean mask: True where embedding was successfully computed.
    """
    n_max = n_max or N_MAX
    d = d or EMBEDDING_DIM
    embeddings = embed_trajectory(traj, embed_fn, d)

    T = len(embeddings)
    tensor = np.zeros((T, n_max, d))
    mask = np.zeros(T, dtype=bool)

    for t, emb in enumerate(embeddings):
        if emb is not None:
            tensor[t] = pad_embedding(emb, n_max, d)
            mask[t] = True

    return tensor, mask


# ══════════════════════════════════════════════════════════════════════
# SUBSPACE SIMILARITY (Grassmannian distance)
# ══════════════════════════════════════════════════════════════════════

def subspace_cosine(X1, X2, backend=None):
    """Compute subspace cosine similarity between two embedding matrices.

    Uses SVD to find principal subspaces, then computes mean of
    principal angles cosines (canonical correlations).

    Parameters
    ----------
    X1, X2 : numpy arrays (n1 × d1), (n2 × d2)
    backend : Backend or None

    Returns
    -------
    float in [0, 1]
    """
    # Align sizes
    if X1.shape[0] != X2.shape[0]:
        n = max(X1.shape[0], X2.shape[0])
        X1p = np.zeros((n, X1.shape[1]))
        X1p[:X1.shape[0]] = X1
        X2p = np.zeros((n, X2.shape[1]))
        X2p[:X2.shape[0]] = X2
        X1, X2 = X1p, X2p

    if X1.size == 0 or X2.size == 0:
        return 0.0

    B = backend or get_backend("cpu")
    U1, _, _ = B.svd(B.to_device(X1), full_matrices=False)
    U2, _, _ = B.svd(B.to_device(X2), full_matrices=False)

    U1 = B.to_numpy(U1)
    U2 = B.to_numpy(U2)

    k = min(U1.shape[1], U2.shape[1])
    if k == 0:
        return 0.0
    S = np.linalg.svd(U1[:, :k].T @ U2[:, :k], compute_uv=False)
    return float(np.mean(S))


def subspace_cosine_gpu(X1, X2, backend):
    """GPU-accelerated subspace cosine (keeps data on device)."""
    if X1.shape[0] != X2.shape[0]:
        n = max(X1.shape[0], X2.shape[0])
        X1p = backend.zeros((n, X1.shape[1]))
        X1p[:X1.shape[0]] = backend.to_device(X1)
        X2p = backend.zeros((n, X2.shape[1]))
        X2p[:X2.shape[0]] = backend.to_device(X2)
        X1_d, X2_d = X1p, X2p
    else:
        X1_d = backend.to_device(X1)
        X2_d = backend.to_device(X2)

    U1, _, _ = backend.svd(X1_d, full_matrices=False)
    U2, _, _ = backend.svd(X2_d, full_matrices=False)

    k = min(U1.shape[1], U2.shape[1])
    if k == 0:
        return 0.0
    S = backend.svd(backend.matmul(U1[:, :k].T, U2[:, :k]),
                    compute_uv=False)
    return float(backend.to_numpy(backend.mean(S)))


# ══════════════════════════════════════════════════════════════════════
# COMPRESSION RATIOS (for I-predicate gate)
# ══════════════════════════════════════════════════════════════════════

def compression_ratio(traj, method="edge_list"):
    """Compute compression ratio of trajectory serialization.

    Parameters
    ----------
    traj : list of nx.Graph
    method : str
        "edge_list" (default), "adjacency", or "canonical".

    Returns
    -------
    float
        len(compressed) / len(raw). Lower = more compressible.
    """
    serializers = {
        "edge_list": _serialize_edge_list,
        "adjacency": _serialize_adjacency,
        "canonical": _serialize_canonical,
    }
    serialize = serializers.get(method, _serialize_edge_list)
    raw = serialize(traj)
    if len(raw) == 0:
        return 1.0
    compressed = zlib.compress(raw, 9)
    return len(compressed) / len(raw)


def _serialize_edge_list(traj):
    data = []
    for G in traj:
        edges = sorted((int(u), int(v)) for u, v in G.edges())
        data.append(edges)
    return json.dumps(data).encode()


def _serialize_adjacency(traj):
    data = []
    for G in traj:
        nodes = sorted(G.nodes())
        adj = []
        for u in nodes:
            row = sorted([int(v) for v in G.neighbors(u)])
            adj.append(row)
        data.append(adj)
    return json.dumps(data).encode()


def _serialize_canonical(traj):
    data = []
    for G in traj:
        edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
        n = G.number_of_nodes()
        data.append(f"{n}:{edges}")
    return "|".join(data).encode()


# ══════════════════════════════════════════════════════════════════════
# SPECTRAL DIMENSION (for Φ-predicate)
# ══════════════════════════════════════════════════════════════════════

def spectral_dimension_estimate(G):
    """Estimate spectral dimension from Laplacian eigenvalue scaling.

    Returns None for graphs with < 4 nodes or < 3 positive eigenvalues.
    """
    if G.number_of_nodes() < 4:
        return None
    L = nx.laplacian_matrix(G).toarray().astype(float)
    evals = sorted(np.linalg.eigvalsh(L))
    pos_evals = [e for e in evals if e > 1e-8]
    if len(pos_evals) < 3:
        return None
    N = np.arange(1, len(pos_evals) + 1, dtype=float)
    logN = np.log(N)
    logE = np.log(pos_evals)
    coeffs = np.polyfit(logE, logN, 1)
    ds = 2.0 * coeffs[0]
    return float(ds)


# ══════════════════════════════════════════════════════════════════════
# CURVATURE (for Φ-predicate)
# ══════════════════════════════════════════════════════════════════════

def curvature_homogeneity(G):
    """Coefficient of variation of Jaccard curvature across edges."""
    if G.number_of_edges() < 3:
        return float('inf')
    jaccards = []
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        union = nu | nv
        inter = nu & nv
        if len(union) > 0:
            jaccards.append(len(inter) / len(union))
        else:
            jaccards.append(0.0)
    arr = np.array(jaccards)
    mean = arr.mean()
    if mean < 1e-12:
        return float(np.std(arr))
    return float(np.std(arr) / mean)


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(filepath, results, metadata=None):
    """Save experiment results to a JSON checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    results : dict
        Must be JSON-serializable (no nx.Graph objects).
    metadata : dict or None
        Extra info (git hash, config snapshot, etc.)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {"results": results}
    if metadata:
        data["metadata"] = metadata
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_checkpoint(filepath):
    """Load a checkpoint file.

    Returns
    -------
    dict with "results" and optionally "metadata" keys.
    """
    with open(filepath) as f:
        return json.load(f)


def _json_default(obj):
    """JSON fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ══════════════════════════════════════════════════════════════════════
# GRAPH SUMMARY STATISTICS (used by Φ-predicate law fitting)
# ══════════════════════════════════════════════════════════════════════

def total_edges(G):
    return G.number_of_edges()


def edges_per_node(G):
    n = G.number_of_nodes()
    return G.number_of_edges() / n if n > 0 else 0


def mean_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return np.mean([d for _, d in G.degree()])


def degree_entropy(G):
    if G.number_of_nodes() < 2:
        return 0.0
    degs = [d for _, d in G.degree()]
    vals, counts = np.unique(degs, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-15)))


def spectral_gap(G):
    if G.number_of_nodes() < 3:
        return 0
    evals = sorted(np.linalg.eigvalsh(
        nx.laplacian_matrix(G).toarray().astype(float)))
    pos = [e for e in evals if e > 1e-8]
    return pos[0] if pos else 0


# Registry of aggregate quantities for law fitting
AGGREGATE_QUANTITIES = {
    "total_edges": total_edges,
    "edges_per_node": edges_per_node,
    "mean_degree": mean_degree,
    "degree_entropy": degree_entropy,
    "spectral_gap": spectral_gap,
}


def law_residual_score(traj, quantity_fn):
    """Fit polynomial models to a trajectory aggregate and return residual.

    Parameters
    ----------
    traj : list of nx.Graph
    quantity_fn : callable
        Graph → float.

    Returns
    -------
    residual : float
        Best relative RMSE across constant/linear/quadratic fits.
    model_name : str
        Name of the best-fit model.
    """
    values = []
    for G in traj:
        try:
            values.append(float(quantity_fn(G)))
        except Exception:
            pass
    if len(values) < 4:
        return float('inf'), "insufficient_data"

    arr = np.array(values)
    t = np.arange(len(arr), dtype=float)
    val_range = arr.max() - arr.min()
    if val_range < 1e-12:
        return 0.0, "constant"

    best_resid = float('inf')
    best_model = "none"
    for degree, name in [(0, "constant"), (1, "linear"), (2, "quadratic")]:
        coeffs = np.polyfit(t, arr, degree)
        fitted = np.polyval(coeffs, t)
        rmse = np.sqrt(np.mean((arr - fitted) ** 2))
        rel_resid = rmse / val_range
        if rel_resid < best_resid:
            best_resid = rel_resid
            best_model = name

    return float(best_resid), best_model
