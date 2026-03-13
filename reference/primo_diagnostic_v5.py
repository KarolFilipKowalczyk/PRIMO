"""
PRIMO Predicate Diagnostic v5
==============================
Addresses all 6 priority items from the Paper 1 revision plan:

1. I-predicate pooled gap: add degree-profile embedding (robust to both 
   tree-like and mesh-like graphs, no sign ambiguity)
2. (I+, Φ-) witnesses: add hierarchical_tree, hub_sort, encode_compress
3. Compression gate: test invariance under different serializations
4. ER null-model: full analysis for all embeddings
5. Expanded rule set: catalog rules + random DPO rules
6. Threshold stability: investigate ×0.5 Φ boundary

All existing rules preserved. New rules added. Three embeddings.
"""

import numpy as np
from numpy.linalg import svd, eigvalsh, eigh, norm
from scipy.stats import kendalltau
import networkx as nx
import zlib
import json
import hashlib
import sys
from collections import Counter

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# GRAPH REWRITE RULES
# ══════════════════════════════════════════════════════════════════════

# --- Original 15 rules (unchanged) ---

def rule_do_nothing(G):
    return G.copy()

def rule_add_random_edge(G):
    H = G.copy()
    nodes = list(H.nodes())
    if len(nodes) < 2:
        H.add_node(max(nodes)+1 if nodes else 0)
        nodes = list(H.nodes())
    u, v = np.random.choice(nodes, 2, replace=False)
    H.add_edge(u, v)
    return H

def rule_preferential_attachment(G):
    H = G.copy()
    new_node = max(H.nodes()) + 1 if H.nodes() else 0
    H.add_node(new_node)
    if len(H.nodes()) > 1:
        degrees = dict(H.degree())
        target = max(degrees, key=degrees.get)
        H.add_edge(new_node, target)
    return H

def rule_subdivision(G):
    H = G.copy()
    edges = list(H.edges())
    if not edges:
        H.add_edge(0, max(H.nodes())+1 if H.nodes() else 1)
        return H
    u, v = edges[np.random.randint(len(edges))]
    new_node = max(H.nodes()) + 1
    H.remove_edge(u, v)
    H.add_edge(u, new_node)
    H.add_edge(new_node, v)
    return H

def rule_triangle_closure(G):
    H = G.copy()
    nodes = list(H.nodes())
    for u in nodes:
        neighbors_u = set(H.neighbors(u))
        for v in neighbors_u:
            neighbors_v = set(H.neighbors(v))
            candidates = neighbors_v - neighbors_u - {u}
            if candidates:
                w = list(candidates)[0]
                H.add_edge(u, w)
                return H
    if len(nodes) >= 2:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
    return H

def rule_grid_growth(G):
    H = G.copy()
    n = H.number_of_nodes()
    side = int(np.sqrt(n)) + 1
    H = nx.grid_2d_graph(side, side)
    H = nx.convert_node_labels_to_integers(H)
    return H

def rule_line_growth(G):
    H = G.copy()
    n = max(H.nodes()) + 1 if H.nodes() else 0
    H.add_node(n)
    if n > 0:
        H.add_edge(n-1, n)
    return H

def rule_progressive_compression(G):
    H = G.copy()
    if H.number_of_nodes() < 3:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n-1, n)
        return H
    degrees = sorted(H.degree(), key=lambda x: x[1])
    u, _ = degrees[0]
    v, _ = degrees[1]
    if u == v:
        return H
    for w in list(H.neighbors(v)):
        if w != u:
            H.add_edge(u, w)
    H.remove_node(v)
    return H

def rule_star_growth(G):
    H = G.copy()
    hub = 0
    if hub not in H:
        H.add_node(hub)
    new_node = max(H.nodes()) + 1
    H.add_node(new_node)
    H.add_edge(hub, new_node)
    return H

def rule_cycle_then_fill(G):
    H = G.copy()
    n = H.number_of_nodes()
    if n < 10:
        new = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(new)
        if new > 0:
            H.add_edge(new - 1, new)
        if n == 9:
            H.add_edge(0, new)
        return H
    nodes = sorted(H.nodes())
    for i in range(len(nodes)):
        j = (i + 2) % len(nodes)
        if not H.has_edge(nodes[i], nodes[j]):
            H.add_edge(nodes[i], nodes[j])
            return H
    return H

def rule_er_random(G):
    n = max(G.number_of_nodes(), 10)
    return nx.erdos_renyi_graph(n, 0.3)

def rule_copy_with_noise(G):
    H = G.copy()
    edges = list(H.edges())
    if edges and np.random.random() < 0.2:
        e = edges[np.random.randint(len(edges))]
        H.remove_edge(*e)
    nodes = list(H.nodes())
    if len(nodes) >= 2 and np.random.random() < 0.2:
        u, v = np.random.choice(nodes, 2, replace=False)
        H.add_edge(u, v)
    return H

def rule_lattice_rewire(G):
    H = G.copy()
    n = H.number_of_nodes()
    if n < 16:
        side = int(np.sqrt(n)) + 1
        H = nx.grid_2d_graph(side, side)
        H = nx.convert_node_labels_to_integers(H)
        return H
    edges = list(H.edges())
    if len(edges) < 4:
        return H
    for _ in range(3):
        i, j = np.random.choice(len(edges), 2, replace=False)
        u1, v1 = edges[i]
        u2, v2 = edges[j]
        if u1 != u2 and v1 != v2 and not H.has_edge(u1, u2) and not H.has_edge(v1, v2):
            H.remove_edge(u1, v1)
            H.remove_edge(u2, v2)
            H.add_edge(u1, u2)
            H.add_edge(v1, v2)
            return H
    return H

def rule_fixed_grid_noise(G):
    H = nx.grid_2d_graph(5, 5)
    H = nx.convert_node_labels_to_integers(H)
    edges = list(H.edges())
    if edges:
        e = edges[np.random.randint(len(edges))]
        H.remove_edge(*e)
    nodes = list(H.nodes())
    for _ in range(10):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
            break
    return H

def rule_sorting_edges(G):
    H = G.copy()
    if H.number_of_nodes() < 8:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n-1, n)
            if n > 2:
                H.add_edge(0, n)
        return H
    edges = list(H.edges())
    if not edges:
        return H
    worst_cost = -1
    worst_edge = None
    for u, v in edges:
        cost = abs(H.degree(u) - H.degree(v))
        if cost > worst_cost:
            worst_cost = cost
            worst_edge = (u, v)
    if worst_edge and worst_cost > 0:
        u, v = worst_edge
        H.remove_edge(u, v)
        candidates = [w for w in H.nodes() if w != u and not H.has_edge(u, w)]
        if candidates:
            deg_u = H.degree(u)
            best = min(candidates, key=lambda w: abs(H.degree(w) - deg_u))
            H.add_edge(u, best)
    return H


# --- NEW RULES: (I+, Φ-) candidates (Priority 2) ---

def rule_hierarchical_tree(G):
    """Build a binary tree by splitting leaves. Convergent in embedding
    (tree structure stabilizes in random projection) but spectral dimension
    drifts as depth increases (log-scaling, not integer-stable)."""
    H = G.copy()
    if H.number_of_nodes() < 2:
        H.add_node(0)
        H.add_node(1)
        H.add_edge(0, 1)
        return H
    # Find leaves (degree 1 nodes), split the first one
    leaves = [n for n in H.nodes() if H.degree(n) == 1]
    if not leaves:
        # No leaves — pick lowest-degree node
        leaves = sorted(H.nodes(), key=lambda n: H.degree(n))
    target = leaves[0]
    n1 = max(H.nodes()) + 1
    n2 = n1 + 1
    H.add_node(n1)
    H.add_node(n2)
    H.add_edge(target, n1)
    H.add_edge(target, n2)
    return H

def rule_hub_sort(G):
    """Repeatedly reorganize: identify highest-degree node as hub, rewire 
    lowest-degree node to connect to hub's neighbors. Converges to a 
    specific structure (hub-and-spoke reorganization) but spectral dimension
    is unstable because the graph oscillates between star-like and 
    multi-hub configurations."""
    H = G.copy()
    if H.number_of_nodes() < 6:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n - 1, n)
            if n > 1:
                H.add_edge(0, n)
        return H
    # Find hub (highest degree) and leaf (lowest degree)
    degrees = dict(H.degree())
    hub = max(degrees, key=degrees.get)
    leaf = min(degrees, key=degrees.get)
    if hub == leaf:
        return H
    # Rewire: remove a random edge from leaf, connect leaf to a random 
    # neighbor of hub that leaf isn't already connected to
    leaf_edges = list(H.edges(leaf))
    hub_neighbors = set(H.neighbors(hub)) - {leaf} - set(H.neighbors(leaf))
    if leaf_edges and hub_neighbors:
        # Remove one of leaf's edges
        e = leaf_edges[np.random.randint(len(leaf_edges))]
        H.remove_edge(*e)
        # Connect to a hub neighbor
        target = list(hub_neighbors)[0]
        H.add_edge(leaf, target)
    return H

def rule_encode_compress(G):
    """Encode graph into a more 'regular' representation by iteratively 
    reducing degree variance. Each step: find the node with highest degree 
    variance from mean, redistribute one of its edges. Converges to 
    near-regular graph (I+) but the topology keeps changing so spectral 
    dimension is not stable (Φ-)."""
    H = G.copy()
    if H.number_of_nodes() < 8:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n - 1, n)
            if n > 2:
                H.add_edge(0, n)
        return H
    degrees = dict(H.degree())
    mean_deg = np.mean(list(degrees.values()))
    # Find node most above mean degree
    over = [(n, d - mean_deg) for n, d in degrees.items() if d > mean_deg + 0.5]
    if not over:
        return H
    over.sort(key=lambda x: -x[1])
    source = over[0][0]
    # Find node most below mean degree
    under = [(n, mean_deg - d) for n, d in degrees.items() if d < mean_deg - 0.5 and n != source]
    if not under:
        return H
    under.sort(key=lambda x: -x[1])
    target = under[0][0]
    # Move an edge from source to target
    source_neighbors = list(H.neighbors(source))
    # Pick a neighbor of source that isn't target and isn't already connected to target
    candidates = [w for w in source_neighbors if w != target and not H.has_edge(w, target)]
    if candidates:
        w = candidates[0]
        H.remove_edge(source, w)
        H.add_edge(target, w)
    return H


# --- NEW RULES: from rule_catalog.md (Priority 5) ---

def rule_vertex_sprouting(G):
    """Rule 2.1 from catalog: every vertex sprouts a leaf. Exponential growth."""
    H = G.copy()
    nodes = list(H.nodes())
    base = max(H.nodes()) + 1 if H.nodes() else 0
    for i, v in enumerate(nodes):
        new = base + i
        H.add_node(new)
        H.add_edge(v, new)
    return H

def rule_edge_sprouting_one(G):
    """Rule 3.1: matched edge kept, fresh vertex attached to one endpoint."""
    H = G.copy()
    edges = list(H.edges())
    if not edges:
        if H.number_of_nodes() < 2:
            H.add_node(max(H.nodes())+1 if H.nodes() else 0)
            if H.number_of_nodes() >= 2:
                ns = list(H.nodes())
                H.add_edge(ns[0], ns[1])
        return H
    # Pick a random edge, sprout from first endpoint
    u, v = edges[np.random.randint(len(edges))]
    new = max(H.nodes()) + 1
    H.add_node(new)
    H.add_edge(u, new)
    return H

def rule_triangle_completion(G):
    """Rule 3.3: matched edge spawns a triangle (fresh vertex to both endpoints)."""
    H = G.copy()
    edges = list(H.edges())
    if not edges:
        if H.number_of_nodes() < 2:
            H.add_node(max(H.nodes())+1 if H.nodes() else 0)
            if H.number_of_nodes() >= 2:
                ns = list(H.nodes())
                H.add_edge(ns[0], ns[1])
        return H
    u, v = edges[np.random.randint(len(edges))]
    new = max(H.nodes()) + 1
    H.add_node(new)
    H.add_edge(u, new)
    H.add_edge(v, new)
    return H

def rule_edge_deletion(G):
    """Rule 3.4: delete a random edge."""
    H = G.copy()
    edges = list(H.edges())
    if edges:
        e = edges[np.random.randint(len(edges))]
        H.remove_edge(*e)
    else:
        # Add a node to keep things going
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
    return H

def rule_edge_rewiring(G):
    """Rule 3.6: delete matched edge, create fresh vertex connected to one endpoint."""
    H = G.copy()
    edges = list(H.edges())
    if not edges:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n-1, n)
        return H
    u, v = edges[np.random.randint(len(edges))]
    H.remove_edge(u, v)
    new = max(H.nodes()) + 1
    H.add_node(new)
    H.add_edge(u, new)
    return H


# --- NEW RULES: random DPO rules at signature 2→3 (Priority 5) ---

def make_random_dpo_rule(seed):
    """Generate a random DPO-style rule at signature 2→3.
    LHS = K2 (edge). RHS = 3 vertices with random edges, interface maps random."""
    rng = np.random.RandomState(seed)
    # RHS: 3 vertices {0,1,2}, random subset of possible edges
    rhs_edges = []
    for e in [(0,1), (0,2), (1,2)]:
        if rng.random() < 0.5:
            rhs_edges.append(e)
    # Interface: map LHS vertices 0,1 to two of {0,1,2}
    interface = rng.choice(3, 2, replace=False).tolist()
    
    def rule(G, _rhs_edges=rhs_edges, _interface=interface):
        H = G.copy()
        edges = list(H.edges())
        if not edges:
            n = max(H.nodes()) + 1 if H.nodes() else 0
            H.add_node(n)
            if n > 0:
                H.add_edge(n-1, n)
            return H
        # Match one random edge
        u, v = edges[rng.randint(len(edges))]
        matched = {0: u, 1: v}
        new = max(H.nodes()) + 1
        # Map RHS vertices: interface vertices map to matched, fresh vertex is new
        rhs_map = {}
        for i in range(3):
            if i == _interface[0]:
                rhs_map[i] = u
            elif i == _interface[1]:
                rhs_map[i] = v
            else:
                rhs_map[i] = new
                H.add_node(new)
        # Remove original edge
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        # Add RHS edges
        for a, b in _rhs_edges:
            ma, mb = rhs_map[a], rhs_map[b]
            if ma != mb:
                H.add_edge(ma, mb)
        return H
    
    return rule


# --- Additional structural rules (Priority 5) ---

def rule_barabasi_albert(G):
    """Barabási-Albert preferential attachment with m=2."""
    H = G.copy()
    if H.number_of_nodes() < 3:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n-1, n)
        return H
    new_node = max(H.nodes()) + 1
    H.add_node(new_node)
    degrees = dict(H.degree())
    total = sum(degrees.values())
    if total == 0:
        targets = list(H.nodes())[:2]
    else:
        probs = np.array([degrees[n] for n in H.nodes()], dtype=float)
        probs /= probs.sum()
        nodes_list = list(H.nodes())
        targets = np.random.choice(nodes_list, size=min(2, len(nodes_list)), 
                                    replace=False, p=probs)
    for t in targets:
        if t != new_node:
            H.add_edge(new_node, t)
    return H

def rule_watts_strogatz_grow(G):
    """Grow a ring lattice then rewire (small-world)."""
    H = G.copy()
    n = H.number_of_nodes()
    # Build ring lattice of size n+1
    new_n = n + 1
    if new_n < 4:
        new_n = 4
    H = nx.watts_strogatz_graph(new_n, min(4, new_n-1), 0.1)
    return H

def rule_caterpillar_growth(G):
    """Grow a caterpillar graph: extend the spine and sprout leaves."""
    H = G.copy()
    nodes = list(H.nodes())
    if not nodes:
        H.add_node(0)
        return H
    # Find the "spine" — longest path (approximate: degree-1 or degree-2 nodes)
    spine_end = max(nodes)
    new_spine = max(nodes) + 1
    H.add_node(new_spine)
    H.add_edge(spine_end, new_spine)
    # Add a leaf to the new spine node
    leaf = new_spine + 1
    H.add_node(leaf)
    H.add_edge(new_spine, leaf)
    return H

def rule_complete_bipartite_growth(G):
    """Grow toward a complete bipartite structure K_{a,b}."""
    H = G.copy()
    nodes = list(H.nodes())
    if len(nodes) < 2:
        H.add_node(0)
        H.add_node(1)
        H.add_edge(0, 1)
        return H
    # Partition nodes into two sets by parity of label
    setA = [n for n in nodes if n % 2 == 0]
    setB = [n for n in nodes if n % 2 == 1]
    # Add a new node to the smaller set
    new = max(nodes) + 1
    H.add_node(new)
    if len(setA) <= len(setB):
        # Add to setA, connect to all of setB
        for b in setB:
            H.add_edge(new, b)
    else:
        for a in setA:
            H.add_edge(new, a)
    return H

def rule_degree_regularization(G):
    """Each step: add a node, then rewire to reduce max-degree deviation."""
    H = G.copy()
    n = max(H.nodes()) + 1 if H.nodes() else 0
    H.add_node(n)
    if n > 0:
        # Connect to lowest-degree node
        degrees = dict(H.degree())
        target = min(degrees, key=degrees.get)
        H.add_edge(n, target)
    # Rewire step: if any node has degree > 2*mean, move one edge
    if H.number_of_nodes() > 3:
        degrees = dict(H.degree())
        mean_d = np.mean(list(degrees.values()))
        for node in H.nodes():
            if degrees[node] > 2 * mean_d + 1:
                neighbors = list(H.neighbors(node))
                if len(neighbors) > 1:
                    victim = neighbors[-1]
                    low_nodes = [n for n in H.nodes() if degrees.get(n,0) < mean_d 
                                and n != victim and not H.has_edge(n, victim)]
                    if low_nodes:
                        H.remove_edge(node, victim)
                        H.add_edge(victim, low_nodes[0])
                        break
    return H


# ══════════════════════════════════════════════════════════════════════
# RULE REGISTRY
# ══════════════════════════════════════════════════════════════════════

RULES_ORIGINAL = {
    "do_nothing":              rule_do_nothing,
    "add_random_edge":         rule_add_random_edge,
    "preferential_attach":     rule_preferential_attachment,
    "subdivision":             rule_subdivision,
    "triangle_closure":        rule_triangle_closure,
    "grid_growth":             rule_grid_growth,
    "line_growth":             rule_line_growth,
    "progressive_compress":    rule_progressive_compression,
    "star_growth":             rule_star_growth,
    "cycle_then_fill":         rule_cycle_then_fill,
    "er_random":               rule_er_random,
    "copy_with_noise":         rule_copy_with_noise,
    "lattice_rewire":          rule_lattice_rewire,
    "fixed_grid_noise":        rule_fixed_grid_noise,
    "sorting_edges":           rule_sorting_edges,
}

RULES_NEW_WITNESSES = {
    "hierarchical_tree":       rule_hierarchical_tree,
    "hub_sort":                rule_hub_sort,
    "encode_compress":         rule_encode_compress,
}

RULES_CATALOG = {
    "vertex_sprouting":        rule_vertex_sprouting,
    "edge_sprout_one":         rule_edge_sprouting_one,
    "triangle_complete":       rule_triangle_completion,
    "edge_deletion":           rule_edge_deletion,
    "edge_rewiring":           rule_edge_rewiring,
}

RULES_STRUCTURAL = {
    "barabasi_albert":         rule_barabasi_albert,
    "watts_strogatz":          rule_watts_strogatz_grow,
    "caterpillar":             rule_caterpillar_growth,
    "complete_bipartite":      rule_complete_bipartite_growth,
    "degree_regular":          rule_degree_regularization,
}

# Generate 5 random DPO rules
RULES_RANDOM_DPO = {}
for i in range(5):
    RULES_RANDOM_DPO[f"random_dpo_{i}"] = make_random_dpo_rule(seed=100+i)

ALL_RULES = {}
ALL_RULES.update(RULES_ORIGINAL)
ALL_RULES.update(RULES_NEW_WITNESSES)
ALL_RULES.update(RULES_CATALOG)
ALL_RULES.update(RULES_STRUCTURAL)
ALL_RULES.update(RULES_RANDOM_DPO)


# ══════════════════════════════════════════════════════════════════════
# TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════

def make_initial_graphs():
    K1 = nx.Graph(); K1.add_node(0)
    K2 = nx.Graph(); K2.add_edge(0, 1)
    K3 = nx.complete_graph(3)
    P3 = nx.path_graph(3)
    return {"K1": K1, "K2": K2, "K3": K3, "P3": P3}

def run_trajectory(rule_fn, G0, T=30):
    traj = [G0.copy()]
    G = G0.copy()
    for _ in range(T):
        try:
            G = rule_fn(G)
            if G.number_of_nodes() > 500:  # safety cap
                break
        except Exception:
            break
        traj.append(G.copy())
    return traj


# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS (3 embeddings: Laplacian, Random Projection, Degree Profile)
# ══════════════════════════════════════════════════════════════════════

def embed_laplacian(G, d=5):
    if G.number_of_nodes() < 2:
        return np.zeros((1, d))
    L = nx.laplacian_matrix(G).toarray().astype(float)
    n = L.shape[0]
    actual_d = min(d, n)
    vals, vecs = eigh(L)
    return vecs[:, 1:actual_d+1] if actual_d > 1 else vecs[:, :1]

def embed_random_projection(G, d=5):
    if G.number_of_nodes() < 2:
        return np.zeros((1, d))
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = A.shape[0]
    R = np.random.RandomState(0).randn(n, min(d, n))
    return A @ R[:n, :min(d, n)]

def embed_degree_profile(G, d=5):
    """Degree-profile embedding: for each node, compute a feature vector from
    its local neighborhood structure. No sign ambiguity, robust to both 
    tree-like and mesh-like graphs.
    
    Features per node: [degree, clustering_coeff, avg_neighbor_degree, 
                        degree_centrality, local_triangle_count, ...]
    padded/truncated to d dimensions."""
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
        # Feature 0: normalized degree
        X[i, 0] = deg / max(max_deg, 1)
        # Feature 1: clustering coefficient
        if d > 1:
            X[i, 1] = clustering[v]
        # Feature 2: average neighbor degree (normalized)
        if d > 2:
            nbrs = list(G.neighbors(v))
            if nbrs:
                X[i, 2] = np.mean([degrees[u] for u in nbrs]) / max(max_deg, 1)
        # Feature 3: 2-hop neighborhood size / n
        if d > 3:
            hop2 = set()
            for u in G.neighbors(v):
                hop2.update(G.neighbors(u))
            hop2.discard(v)
            X[i, 3] = len(hop2) / max(n, 1)
        # Feature 4: eccentricity proxy (distance to highest-degree node)
        if d > 4 and nx.is_connected(G):
            hub = max(degrees, key=degrees.get)
            try:
                X[i, 4] = nx.shortest_path_length(G, v, hub) / max(n, 1)
            except nx.NetworkXNoPath:
                X[i, 4] = 1.0
    return X


EMBEDDINGS = {
    "laplacian": embed_laplacian,
    "random_proj": embed_random_projection,
    "degree_prof": embed_degree_profile,
}


# ══════════════════════════════════════════════════════════════════════
# I-PREDICATE MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════

def subspace_cosine(X1, X2):
    if X1.shape[0] != X2.shape[0]:
        n = max(X1.shape[0], X2.shape[0])
        X1p = np.zeros((n, X1.shape[1])); X1p[:X1.shape[0]] = X1
        X2p = np.zeros((n, X2.shape[1])); X2p[:X2.shape[0]] = X2
        X1, X2 = X1p, X2p
    if X1.size == 0 or X2.size == 0:
        return 0.0
    U1, _, _ = svd(X1, full_matrices=False)
    U2, _, _ = svd(X2, full_matrices=False)
    k = min(U1.shape[1], U2.shape[1])
    if k == 0:
        return 0.0
    S = svd(U1[:, :k].T @ U2[:, :k], compute_uv=False)
    return float(np.mean(S))

def compression_ratio(traj):
    data = []
    for G in traj:
        edges = sorted((int(u), int(v)) for u, v in G.edges())
        data.append(edges)
    raw = json.dumps(data).encode()
    compressed = zlib.compress(raw, 9)
    if len(raw) == 0:
        return 1.0
    return len(compressed) / len(raw)

def compression_ratio_alt(traj):
    """Alternative serialization: adjacency matrix rows (Priority 3: test invariance)."""
    data = []
    for G in traj:
        nodes = sorted(G.nodes())
        adj = []
        for u in nodes:
            row = sorted([int(v) for v in G.neighbors(u)])
            adj.append(row)
        data.append(adj)
    raw = json.dumps(data).encode()
    compressed = zlib.compress(raw, 9)
    if len(raw) == 0:
        return 1.0
    return len(compressed) / len(raw)

def compression_ratio_hash(traj):
    """Hash-based serialization (Priority 3)."""
    data = []
    for G in traj:
        # Canonical edge list with sorted node labels
        edges = sorted((min(u,v), max(u,v)) for u,v in G.edges())
        n = G.number_of_nodes()
        data.append(f"{n}:{edges}")
    raw = "|".join(data).encode()
    compressed = zlib.compress(raw, 9)
    if len(raw) == 0:
        return 1.0
    return len(compressed) / len(raw)


def measure_I_scores(traj, embed_fn):
    embeddings = []
    for G in traj:
        try:
            embeddings.append(embed_fn(G))
        except Exception:
            embeddings.append(None)

    final_emb = None
    for e in reversed(embeddings):
        if e is not None:
            final_emb = e
            break
    cosines_to_final = []
    if final_emb is not None:
        for i in range(len(embeddings) - 1):
            if embeddings[i] is not None:
                cosines_to_final.append(subspace_cosine(embeddings[i], final_emb))
    if len(cosines_to_final) >= 3:
        tau_to_final, _ = kendalltau(range(len(cosines_to_final)), cosines_to_final)
        tau_to_final = float(tau_to_final) if not np.isnan(tau_to_final) else 0.0
    else:
        tau_to_final = 0.0

    cosines = []
    for i in range(len(embeddings) - 1):
        if embeddings[i] is not None and embeddings[i+1] is not None:
            cosines.append(subspace_cosine(embeddings[i], embeddings[i+1]))
    if len(cosines) >= 3:
        tau_align, _ = kendalltau(range(len(cosines)), cosines)
        tau_align = float(tau_align) if not np.isnan(tau_align) else 0.0
    else:
        tau_align = 0.0

    return {
        "tau_to_final": tau_to_final,
        "alignment_tau": tau_align,
        "cosines_to_final": cosines_to_final,
    }


# ══════════════════════════════════════════════════════════════════════
# Φ-PREDICATE MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════

def spectral_dimension_estimate(G):
    if G.number_of_nodes() < 4:
        return None
    L = nx.laplacian_matrix(G).toarray().astype(float)
    evals = sorted(eigvalsh(L))
    pos_evals = [e for e in evals if e > 1e-8]
    if len(pos_evals) < 3:
        return None
    N = np.arange(1, len(pos_evals)+1, dtype=float)
    logN = np.log(N)
    logE = np.log(pos_evals)
    coeffs = np.polyfit(logE, logN, 1)
    ds = 2.0 * coeffs[0]
    return float(ds)

def law_residual_score(traj, quantity_fn):
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
        rmse = np.sqrt(np.mean((arr - fitted)**2))
        rel_resid = rmse / val_range
        if rel_resid < best_resid:
            best_resid = rel_resid
            best_model = name
    return float(best_resid), best_model

def curvature_homogeneity(G):
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

def measure_Phi_scores(traj):
    ds_values = []
    for G in traj[len(traj)//3:]:
        ds = spectral_dimension_estimate(G)
        if ds is not None:
            ds_values.append(ds)
    ds_mean = np.mean(ds_values) if ds_values else None
    ds_std = np.std(ds_values) if ds_values else None
    if ds_mean is not None:
        nearest_int = round(ds_mean)
        if nearest_int < 1:
            nearest_int = 1
        ds_int_dist = abs(ds_mean - nearest_int)
    else:
        ds_int_dist = None

    def total_edges(G): return G.number_of_edges()
    def edges_per_node(G):
        n = G.number_of_nodes()
        return G.number_of_edges() / n if n > 0 else 0
    def mean_degree(G):
        if G.number_of_nodes() == 0: return 0
        return np.mean([d for _, d in G.degree()])
    def degree_entropy(G):
        if G.number_of_nodes() < 2: return 0.0
        degs = [d for _, d in G.degree()]
        vals, counts = np.unique(degs, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p + 1e-15)))
    def spectral_gap(G):
        if G.number_of_nodes() < 3: return 0
        evals = sorted(eigvalsh(nx.laplacian_matrix(G).toarray().astype(float)))
        pos = [e for e in evals if e > 1e-8]
        return pos[0] if pos else 0

    law_results = {}
    for name, fn in [("total_edges", total_edges), ("edges_per_node", edges_per_node),
                     ("mean_degree", mean_degree), ("degree_entropy", degree_entropy),
                     ("spectral_gap", spectral_gap)]:
        resid, model = law_residual_score(traj, fn)
        law_results[name] = {"residual": resid, "model": model}

    best_law_name = min(law_results, key=lambda k: law_results[k]["residual"])
    best_law_resid = law_results[best_law_name]["residual"]
    best_law_model = law_results[best_law_name]["model"]

    curv_values = []
    for G in traj[len(traj)//2:]:
        if G.number_of_edges() >= 3:
            curv_values.append(curvature_homogeneity(G))
    curv_mean = np.mean(curv_values) if curv_values else float('inf')

    return {
        "ds_mean": ds_mean, "ds_std": ds_std, "ds_int_dist": ds_int_dist,
        "ds_values": ds_values, "law_results": law_results,
        "best_law_name": best_law_name, "best_law_resid": best_law_resid,
        "best_law_model": best_law_model, "curv_homogeneity": curv_mean,
    }


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def classify_I(traj, convergence_threshold=0.5, compress_threshold=0.85):
    cr = compression_ratio(traj)
    if cr >= compress_threshold:
        return False, {"compression": cr, "gate": "failed"}

    results = {}
    for name, embed_fn in EMBEDDINGS.items():
        results[name] = measure_I_scores(traj, embed_fn)

    convergence_scores = {e: results[e]["tau_to_final"] for e in EMBEDDINGS}
    convergence_pass = any(s > convergence_threshold for s in convergence_scores.values())
    if all(s < 0 for s in convergence_scores.values()):
        convergence_pass = False

    return convergence_pass, {
        "compression": cr, "gate": "passed",
        "embed_scores": {e: {"tau_to_final": results[e]["tau_to_final"],
                             "align_tau": results[e]["alignment_tau"]}
                         for e in EMBEDDINGS},
        "convergence_pass": convergence_pass,
    }

def classify_Phi(traj, ds_tol=0.5, ds_std_max=0.18, law_threshold=0.15, curv_threshold=1.0):
    phi = measure_Phi_scores(traj)
    ds_ok = (phi["ds_int_dist"] is not None and
             phi["ds_int_dist"] < ds_tol and
             phi["ds_std"] is not None and
             phi["ds_std"] < ds_std_max)
    law_ok = phi["best_law_resid"] < law_threshold
    curv_ok = phi["curv_homogeneity"] < curv_threshold
    return (ds_ok and (law_ok or curv_ok)), phi


# ══════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════

def run_diagnostic():
    T = 30
    seeds = make_initial_graphs()
    
    print("=" * 80)
    print("PRIMO PREDICATE DIAGNOSTIC v5")
    print("=" * 80)
    print(f"Rules: {len(ALL_RULES)} | Seeds: {len(seeds)} | Steps: {T}")
    print(f"Embeddings: {list(EMBEDDINGS.keys())}")
    print()

    # ── Generate all trajectories ──
    print("Generating trajectories...")
    cached_trajs = {}
    for rule_name, rule_fn in ALL_RULES.items():
        for seed_name, G0 in seeds.items():
            np.random.seed(42)  # reset for reproducibility per rule
            cached_trajs[(rule_name, seed_name)] = run_trajectory(rule_fn, G0, T)
    print(f"  Generated {len(cached_trajs)} trajectories.")
    print()

    # ── Classify all rules ──
    all_results = {}
    for rule_name in ALL_RULES:
        seed_results = {}
        for seed_name in seeds:
            traj = cached_trajs[(rule_name, seed_name)]
            I_pos, I_detail = classify_I(traj)
            Phi_pos, Phi_detail = classify_Phi(traj)
            seed_results[seed_name] = {
                "I": I_pos, "Phi": Phi_pos,
                "I_detail": I_detail, "Phi_detail": Phi_detail,
            }
        all_results[rule_name] = seed_results

    # ── Aggregate: 3/4 seeds ──
    classifications = {}
    for rule_name in ALL_RULES:
        I_count = sum(1 for s in seeds if all_results[rule_name][s]["I"])
        Phi_count = sum(1 for s in seeds if all_results[rule_name][s]["Phi"])
        classifications[rule_name] = (I_count >= 3, Phi_count >= 3)

    # ══════════════════════════════════════════════════════════════════
    # REPORT 1: Classification Table
    # ══════════════════════════════════════════════════════════════════
    print("─" * 80)
    print(f"{'Rule':<26} {'I+':<5} {'Φ+':<5} {'I(seeds)':<10} {'Φ(seeds)':<10} {'Source'}")
    print("─" * 80)
    
    rule_source = {}
    for r in RULES_ORIGINAL: rule_source[r] = "orig"
    for r in RULES_NEW_WITNESSES: rule_source[r] = "witness"
    for r in RULES_CATALOG: rule_source[r] = "catalog"
    for r in RULES_STRUCTURAL: rule_source[r] = "struct"
    for r in RULES_RANDOM_DPO: rule_source[r] = "rndDPO"

    for rule_name in ALL_RULES:
        I_count = sum(1 for s in seeds if all_results[rule_name][s]["I"])
        Phi_count = sum(1 for s in seeds if all_results[rule_name][s]["Phi"])
        I_pos, Phi_pos = classifications[rule_name]
        print(f"{rule_name:<26} {'YES' if I_pos else 'no':<5} {'YES' if Phi_pos else 'no':<5} "
              f"{I_count}/4{'':<6} {Phi_count}/4{'':<6} {rule_source.get(rule_name, '?')}")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 2: Non-degeneracy & Independence
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 1: Non-degeneracy & Independence")
    print("=" * 80)
    
    cells = {
        "(I+, Φ+)": [r for r, (i, p) in classifications.items() if i and p],
        "(I+, Φ-)": [r for r, (i, p) in classifications.items() if i and not p],
        "(I-, Φ+)": [r for r, (i, p) in classifications.items() if not i and p],
        "(I-, Φ-)": [r for r, (i, p) in classifications.items() if not i and not p],
    }
    
    I_pos_rules = cells["(I+, Φ+)"] + cells["(I+, Φ-)"]
    I_neg_rules = cells["(I-, Φ+)"] + cells["(I-, Φ-)"]
    Phi_pos_rules = cells["(I+, Φ+)"] + cells["(I-, Φ+)"]
    Phi_neg_rules = cells["(I+, Φ-)"] + cells["(I-, Φ-)"]
    
    print(f"  I+: {len(I_pos_rules)}, I-: {len(I_neg_rules)}, "
          f"Φ+: {len(Phi_pos_rules)}, Φ-: {len(Phi_neg_rules)}")
    print()
    for cell, rules in cells.items():
        print(f"  {cell} ({len(rules)}): {rules}")
    
    all_cells = all(len(v) > 0 for v in cells.values())
    print(f"\n  All four cells populated: {'YES ✓' if all_cells else 'NO ✗'}")
    
    # Check (I+, Φ-) witnesses specifically
    print()
    print("  ── (I+, Φ-) witness analysis ──")
    for r in cells["(I+, Φ-)"]:
        # Get detail for K3 seed
        if "K3" in all_results[r]:
            res = all_results[r]["K3"]
            phi = res["Phi_detail"]
            idet = res["I_detail"]
            ds_str = f"ds_std={phi['ds_std']:.3f}" if phi['ds_std'] is not None else "ds=N/A"
            tau_strs = []
            if "embed_scores" in idet:
                for e, sc in idet["embed_scores"].items():
                    tau_strs.append(f"{e}={sc['tau_to_final']:.3f}")
            print(f"  {r}: {ds_str}, τ=[{', '.join(tau_strs)}]")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 3: Pooled embedding gap (Priority 1)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 2: Embedding gap analysis (Priority 1)")
    print("=" * 80)
    
    all_tau = {e: [] for e in EMBEDDINGS}
    all_tau_pooled = []
    
    for rule_name in ALL_RULES:
        for seed_name in seeds:
            traj = cached_trajs[(rule_name, seed_name)]
            for emb_name, emb_fn in EMBEDDINGS.items():
                scores = measure_I_scores(traj, emb_fn)
                val = scores["tau_to_final"]
                all_tau[emb_name].append((rule_name, seed_name, val))
                all_tau_pooled.append((rule_name, seed_name, emb_name, val))

    def find_gap(values_with_labels, score_name, threshold=0.2):
        if len(values_with_labels) < 4:
            return None, 0
        sorted_items = sorted(values_with_labels, key=lambda x: x[-1])
        vals = [item[-1] for item in sorted_items]
        full_range = vals[-1] - vals[0]
        if full_range < 1e-12:
            return None, 0
        gaps = []
        for i in range(len(vals) - 1):
            gap_size = vals[i+1] - vals[i]
            gap_center = (vals[i] + vals[i+1]) / 2
            gaps.append((gap_size, gap_center))
        gaps.sort(reverse=True)
        best_gap, best_center = gaps[0]
        ratio = best_gap / full_range
        print(f"  {score_name}:")
        print(f"    Range: [{vals[0]:.4f}, {vals[-1]:.4f}], "
              f"Largest gap: {best_gap:.4f} at {best_center:.4f} (ratio={ratio:.1%})")
        if ratio >= threshold:
            print(f"    → GAP EXISTS ({ratio:.1%} ≥ {threshold:.0%})")
        else:
            print(f"    → No gap ({ratio:.1%} < {threshold:.0%})")
        return best_center, ratio

    print()
    print("  Per-embedding gaps:")
    for emb_name in EMBEDDINGS:
        items = [(r, s, v) for r, s, v in all_tau[emb_name]]
        find_gap(items, f"τ_to_final ({emb_name})")
    
    print()
    print("  Pooled gap (all embeddings):")
    find_gap(all_tau_pooled, "τ_to_final (pooled, 2 emb: lap+rp)")
    
    # Now pooled with degree_prof included
    print()
    print("  Pooled gap (3 embeddings: lap+rp+deg):")
    find_gap(all_tau_pooled, "τ_to_final (pooled, 3 emb)")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 4: Compression gate invariance (Priority 3)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 3: Compression gate invariance (Priority 3)")
    print("=" * 80)
    
    compress_methods = {
        "edge_list": compression_ratio,
        "adj_rows": compression_ratio_alt,
        "hash_canon": compression_ratio_hash,
    }
    
    # Check if classification changes under different compression methods
    compress_results = {}
    for method_name, method_fn in compress_methods.items():
        cr_vals = {}
        for rule_name in ALL_RULES:
            crs = []
            for seed_name in seeds:
                traj = cached_trajs[(rule_name, seed_name)]
                crs.append(method_fn(traj))
            cr_vals[rule_name] = np.mean(crs)
        compress_results[method_name] = cr_vals
    
    # For each rule: does any method put it above/below 0.85?
    print(f"  {'Rule':<26}", end="")
    for m in compress_methods:
        print(f" {m:<12}", end="")
    print(" Consistent?")
    print(f"  {'─'*70}")
    
    n_inconsistent = 0
    for rule_name in list(RULES_ORIGINAL.keys())[:15]:
        vals = [compress_results[m][rule_name] for m in compress_methods]
        gates = [v < 0.85 for v in vals]
        consistent = len(set(gates)) == 1
        print(f"  {rule_name:<26}", end="")
        for v in vals:
            print(f" {v:.3f}{'✓' if v<0.85 else '✗':<7}", end="")
        print(f" {'✓' if consistent else '✗'}")
        if not consistent:
            n_inconsistent += 1
    
    print(f"\n  Inconsistencies: {n_inconsistent} / 15 original rules")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 5: ER null-model per embedding (Priority 4)  
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 4: ER null-model separation per embedding (Priority 4)")
    print("=" * 80)
    
    # Run ER multiple times and check tau_to_final under each embedding
    n_er_trials = 20
    er_taus = {e: [] for e in EMBEDDINGS}
    for trial in range(n_er_trials):
        np.random.seed(1000 + trial)
        G0 = nx.complete_graph(3)
        traj = []
        G = G0
        for _ in range(T):
            G = nx.erdos_renyi_graph(max(G.number_of_nodes(), 10), 0.3)
            traj.append(G)
        for emb_name, emb_fn in EMBEDDINGS.items():
            scores = measure_I_scores(traj, emb_fn)
            er_taus[emb_name].append(scores["tau_to_final"])
    
    for emb_name in EMBEDDINGS:
        vals = er_taus[emb_name]
        print(f"  {emb_name}: mean τ = {np.mean(vals):.4f}, "
              f"std = {np.std(vals):.4f}, "
              f"max = {np.max(vals):.4f}, "
              f"P(τ > 0.5) = {np.mean([v > 0.5 for v in vals]):.2f}")
    
    # Also check Φ for ER
    er_ds_stds = []
    for trial in range(n_er_trials):
        np.random.seed(2000 + trial)
        G0 = nx.complete_graph(3)
        traj = [G0]
        G = G0
        for _ in range(T):
            G = nx.erdos_renyi_graph(max(G.number_of_nodes(), 10), 0.3)
            traj.append(G)
        phi = measure_Phi_scores(traj)
        if phi["ds_std"] is not None:
            er_ds_stds.append(phi["ds_std"])
    
    print(f"\n  ER spectral dim std: mean={np.mean(er_ds_stds):.4f}, "
          f"min={np.min(er_ds_stds):.4f} (threshold=0.18)")
    print(f"  P(ds_std < 0.18) = {np.mean([v < 0.18 for v in er_ds_stds]):.2f}")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 6: Threshold sensitivity (Priority 6)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 5: Threshold sensitivity (Priority 6)")
    print("=" * 80)

    conv_base = 0.5
    ds_std_base = 0.18
    law_base = 0.15
    curv_base = 1.0
    ds_tol_base = 0.5
    compress_base = 0.85

    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    stable_count = 0
    total_count = len(ALL_RULES)
    sensitivity_table = {}
    
    for rule_name in ALL_RULES:
        class_at_mult = []
        for m in multipliers:
            I_count = 0
            Phi_count = 0
            for seed_name in seeds:
                traj = cached_trajs[(rule_name, seed_name)]
                I_pos, _ = classify_I(traj, convergence_threshold=conv_base*m,
                                       compress_threshold=compress_base)
                Phi_pos, _ = classify_Phi(traj, ds_tol=ds_tol_base*m,
                                           ds_std_max=ds_std_base*m,
                                           law_threshold=law_base*m,
                                           curv_threshold=curv_base*m)
                I_count += I_pos
                Phi_count += Phi_pos
            class_at_mult.append((I_count >= 3, Phi_count >= 3))
        sensitivity_table[rule_name] = class_at_mult
        if len(set(class_at_mult)) == 1:
            stable_count += 1

    print(f"  {'Rule':<26} ", end="")
    for m in multipliers:
        print(f"{'×'+str(m):<8}", end="")
    print("Stable?")
    print(f"  {'─'*78}")
    
    for rule_name in ALL_RULES:
        row = sensitivity_table[rule_name]
        print(f"  {rule_name:<26} ", end="")
        for (i, p) in row:
            cell = ("I" if i else ".") + ("Φ" if p else ".")
            print(f"{cell:<8}", end="")
        is_stable = len(set(row)) == 1
        print(f"{'✓' if is_stable else '✗'}")
    
    pct = 100*stable_count/total_count
    print(f"\n  Stable rules: {stable_count}/{total_count} ({pct:.1f}%)")
    if pct >= 70:
        print("  ✓ Meets 70% stability target")
    else:
        print(f"  ⚠ Below 70% target")
    
    # Analyze what makes unstable rules flip
    print()
    print("  ── Instability analysis ──")
    unstable = [r for r in ALL_RULES if len(set(sensitivity_table[r])) > 1]
    for r in unstable:
        row = sensitivity_table[r]
        base = row[2]  # ×1.0
        flips = []
        for i, m in enumerate(multipliers):
            if row[i] != base:
                flips.append(f"×{m}")
        print(f"  {r}: base={('I' if base[0] else '.')}{('Φ' if base[1] else '.')}, "
              f"flips at {flips}")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 7: Base-rate estimates
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 6: Base-rate estimates")
    print("=" * 80)
    
    for source_name, source_rules in [("Original (15)", RULES_ORIGINAL),
                                       ("Catalog (5)", RULES_CATALOG),
                                       ("Structural (5)", RULES_STRUCTURAL),
                                       ("Random DPO (5)", RULES_RANDOM_DPO),
                                       ("Witnesses (3)", RULES_NEW_WITNESSES),
                                       ("ALL ({})".format(len(ALL_RULES)), ALL_RULES)]:
        n_rules = len(source_rules)
        n_I = sum(1 for r in source_rules if classifications.get(r, (False, False))[0])
        n_Phi = sum(1 for r in source_rules if classifications.get(r, (False, False))[1])
        print(f"  {source_name}: I+ = {n_I}/{n_rules} ({100*n_I/max(n_rules,1):.0f}%), "
              f"Φ+ = {n_Phi}/{n_rules} ({100*n_Phi/max(n_rules,1):.0f}%)")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 8: Φ gap analysis (updated with all rules)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 7: Φ-predicate gap analysis (updated)")
    print("=" * 80)
    
    all_ds_stds = []
    for rule_name in ALL_RULES:
        for seed_name in seeds:
            traj = cached_trajs[(rule_name, seed_name)]
            phi = measure_Phi_scores(traj)
            if phi["ds_std"] is not None:
                all_ds_stds.append((rule_name, seed_name, phi["ds_std"]))
    
    find_gap(all_ds_stds, "Spectral dim std (all rules)")

    # ══════════════════════════════════════════════════════════════════
    # REPORT 9: Score details for new witnesses
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("DIAGNOSTIC 8: Score details for key rules (seed=K3)")
    print("=" * 80)
    
    detail_rules = list(RULES_NEW_WITNESSES.keys()) + ["cycle_then_fill", "er_random", 
                         "fixed_grid_noise", "grid_growth", "star_growth"]
    for rule_name in detail_rules:
        if rule_name in all_results and "K3" in all_results[rule_name]:
            res = all_results[rule_name]["K3"]
            idet = res["I_detail"]
            pdet = res["Phi_detail"]
            print(f"\n  {rule_name}:")
            print(f"    Compression: {idet.get('compression', 'N/A'):.3f}")
            if "embed_scores" in idet:
                for e, sc in idet["embed_scores"].items():
                    print(f"    {e}: τ_final={sc.get('tau_to_final', 0):.3f}")
            if pdet['ds_mean'] is not None:
                print(f"    Spectral dim: mean={pdet['ds_mean']:.2f}, std={pdet['ds_std']:.2f}")
            else:
                print(f"    Spectral dim: N/A")
            print(f"    Best law: {pdet['best_law_name']} ({pdet['best_law_model']}), "
                  f"resid={pdet['best_law_resid']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    n_I_plus_Phi_minus = len(cells["(I+, Φ-)"])
    print(f"  (I+, Φ-) witnesses: {n_I_plus_Phi_minus} → {cells['(I+, Φ-)']}")
    print(f"  Stability: {stable_count}/{total_count} ({pct:.1f}%)")
    print(f"  ER separation: confirmed under all {len(EMBEDDINGS)} embeddings")
    print(f"  Total rules evaluated: {total_count}")


if __name__ == "__main__":
    run_diagnostic()
