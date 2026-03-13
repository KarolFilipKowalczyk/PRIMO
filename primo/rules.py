"""
PRIMO Rules — graph rewrite rules, catalog I/O, and enumeration.

Contains:
- All 33 rules from primo_diagnostic_v5.py (15 original + 3 witness + 5 catalog
  + 5 structural + 5 random DPO)
- Initial graph constructors (K1, K2, K3, P3)
- Catalog I/O: load/save JSON catalogs from data/catalogs/
- Rule enumeration: generate all DPO rules at a given signature (l→r)
  with optional connected-RHS filter

Usage:
    from primo.rules import ALL_RULES, make_initial_graphs, load_catalog
    rules = ALL_RULES                       # dict: name → callable
    seeds = make_initial_graphs()           # dict: name → nx.Graph
    catalog = load_catalog("2_3")           # list of rule dicts
"""

import json
import itertools
from pathlib import Path

import numpy as np
import networkx as nx

from primo.config import (
    N_MAX, INITIAL_GRAPHS, R_CONNECTED_ONLY, SIGNATURE_R_OFFSET,
)

# ══════════════════════════════════════════════════════════════════════
# INITIAL GRAPHS
# ══════════════════════════════════════════════════════════════════════

def make_initial_graphs(names=None):
    """Return a dict of named initial graphs.

    Parameters
    ----------
    names : list of str or None
        Which graphs to include. Default: config.INITIAL_GRAPHS.
    """
    names = names or INITIAL_GRAPHS
    constructors = {
        'K1': lambda: _single_node(),
        'K2': lambda: nx.complete_graph(2),
        'K3': lambda: nx.complete_graph(3),
        'P3': lambda: nx.path_graph(3),
    }
    result = {}
    for name in names:
        if name not in constructors:
            raise ValueError(f"Unknown initial graph: {name}")
        result[name] = constructors[name]()
    return result


def _single_node():
    G = nx.Graph()
    G.add_node(0)
    return G


# ══════════════════════════════════════════════════════════════════════
# GRAPH REWRITE RULES — Original 15
# ══════════════════════════════════════════════════════════════════════

def rule_do_nothing(G):
    return G.copy()


def rule_add_random_edge(G):
    H = G.copy()
    nodes = list(H.nodes())
    if len(nodes) < 2:
        H.add_node(max(nodes) + 1 if nodes else 0)
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
        H.add_edge(0, max(H.nodes()) + 1 if H.nodes() else 1)
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
    n = G.number_of_nodes()
    side = int(np.sqrt(n)) + 1
    H = nx.grid_2d_graph(side, side)
    return nx.convert_node_labels_to_integers(H)


def rule_line_growth(G):
    H = G.copy()
    n = max(H.nodes()) + 1 if H.nodes() else 0
    H.add_node(n)
    if n > 0:
        H.add_edge(n - 1, n)
    return H


def rule_progressive_compression(G):
    H = G.copy()
    if H.number_of_nodes() < 3:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n - 1, n)
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
        if (u1 != u2 and v1 != v2
                and not H.has_edge(u1, u2) and not H.has_edge(v1, v2)):
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
            H.add_edge(n - 1, n)
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


# ══════════════════════════════════════════════════════════════════════
# NEW RULES: (I+, Φ-) witness candidates
# ══════════════════════════════════════════════════════════════════════

def rule_hierarchical_tree(G):
    """Binary tree by splitting leaves. I+ (embedding converges) but Φ-
    (spectral dimension drifts with depth)."""
    H = G.copy()
    if H.number_of_nodes() < 2:
        H.add_node(0)
        H.add_node(1)
        H.add_edge(0, 1)
        return H
    leaves = [n for n in H.nodes() if H.degree(n) == 1]
    if not leaves:
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
    """Hub-and-spoke reorganization. I+ (converges to hub structure) but
    Φ- (oscillates between star-like and multi-hub)."""
    H = G.copy()
    if H.number_of_nodes() < 6:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n - 1, n)
            if n > 1:
                H.add_edge(0, n)
        return H
    degrees = dict(H.degree())
    hub = max(degrees, key=degrees.get)
    leaf = min(degrees, key=degrees.get)
    if hub == leaf:
        return H
    leaf_edges = list(H.edges(leaf))
    hub_neighbors = set(H.neighbors(hub)) - {leaf} - set(H.neighbors(leaf))
    if leaf_edges and hub_neighbors:
        e = leaf_edges[np.random.randint(len(leaf_edges))]
        H.remove_edge(*e)
        target = list(hub_neighbors)[0]
        H.add_edge(leaf, target)
    return H


def rule_encode_compress(G):
    """Degree-variance reduction. I+ (converges to near-regular graph)
    but Φ- (topology keeps changing)."""
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
    over = [(n, d - mean_deg) for n, d in degrees.items() if d > mean_deg + 0.5]
    if not over:
        return H
    over.sort(key=lambda x: -x[1])
    source = over[0][0]
    under = [(n, mean_deg - d) for n, d in degrees.items()
             if d < mean_deg - 0.5 and n != source]
    if not under:
        return H
    under.sort(key=lambda x: -x[1])
    target = under[0][0]
    source_neighbors = list(H.neighbors(source))
    candidates = [w for w in source_neighbors
                  if w != target and not H.has_edge(w, target)]
    if candidates:
        w = candidates[0]
        H.remove_edge(source, w)
        H.add_edge(target, w)
    return H


# ══════════════════════════════════════════════════════════════════════
# CATALOG RULES (from rule_catalog.md)
# ══════════════════════════════════════════════════════════════════════

def rule_vertex_sprouting(G):
    """Rule 2.1: every vertex sprouts a leaf. Exponential growth."""
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
            H.add_node(max(H.nodes()) + 1 if H.nodes() else 0)
            if H.number_of_nodes() >= 2:
                ns = list(H.nodes())
                H.add_edge(ns[0], ns[1])
        return H
    u, v = edges[np.random.randint(len(edges))]
    new = max(H.nodes()) + 1
    H.add_node(new)
    H.add_edge(u, new)
    return H


def rule_triangle_completion(G):
    """Rule 3.3: matched edge spawns a triangle."""
    H = G.copy()
    edges = list(H.edges())
    if not edges:
        if H.number_of_nodes() < 2:
            H.add_node(max(H.nodes()) + 1 if H.nodes() else 0)
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
            H.add_edge(n - 1, n)
        return H
    u, v = edges[np.random.randint(len(edges))]
    H.remove_edge(u, v)
    new = max(H.nodes()) + 1
    H.add_node(new)
    H.add_edge(u, new)
    return H


# ══════════════════════════════════════════════════════════════════════
# STRUCTURAL RULES
# ══════════════════════════════════════════════════════════════════════

def rule_barabasi_albert(G):
    """Barabási-Albert preferential attachment with m=2."""
    H = G.copy()
    if H.number_of_nodes() < 3:
        n = max(H.nodes()) + 1 if H.nodes() else 0
        H.add_node(n)
        if n > 0:
            H.add_edge(n - 1, n)
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
        targets = np.random.choice(
            nodes_list, size=min(2, len(nodes_list)), replace=False, p=probs)
    for t in targets:
        if t != new_node:
            H.add_edge(new_node, t)
    return H


def rule_watts_strogatz_grow(G):
    """Grow a ring lattice then rewire (small-world)."""
    n = G.number_of_nodes()
    new_n = n + 1
    if new_n < 4:
        new_n = 4
    return nx.watts_strogatz_graph(new_n, min(4, new_n - 1), 0.1)


def rule_caterpillar_growth(G):
    """Grow a caterpillar graph: extend spine and sprout leaves."""
    H = G.copy()
    nodes = list(H.nodes())
    if not nodes:
        H.add_node(0)
        return H
    spine_end = max(nodes)
    new_spine = max(nodes) + 1
    H.add_node(new_spine)
    H.add_edge(spine_end, new_spine)
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
    setA = [n for n in nodes if n % 2 == 0]
    setB = [n for n in nodes if n % 2 == 1]
    new = max(nodes) + 1
    H.add_node(new)
    if len(setA) <= len(setB):
        for b in setB:
            H.add_edge(new, b)
    else:
        for a in setA:
            H.add_edge(new, a)
    return H


def rule_degree_regularization(G):
    """Add a node, then rewire to reduce max-degree deviation."""
    H = G.copy()
    n = max(H.nodes()) + 1 if H.nodes() else 0
    H.add_node(n)
    if n > 0:
        degrees = dict(H.degree())
        target = min(degrees, key=degrees.get)
        H.add_edge(n, target)
    if H.number_of_nodes() > 3:
        degrees = dict(H.degree())
        mean_d = np.mean(list(degrees.values()))
        for node in H.nodes():
            if degrees[node] > 2 * mean_d + 1:
                neighbors = list(H.neighbors(node))
                if len(neighbors) > 1:
                    victim = neighbors[-1]
                    low_nodes = [nn for nn in H.nodes()
                                 if degrees.get(nn, 0) < mean_d
                                 and nn != victim
                                 and not H.has_edge(nn, victim)]
                    if low_nodes:
                        H.remove_edge(node, victim)
                        H.add_edge(victim, low_nodes[0])
                        break
    return H


# ══════════════════════════════════════════════════════════════════════
# RANDOM DPO RULES (signature 2→3)
# ══════════════════════════════════════════════════════════════════════

def make_random_dpo_rule(seed):
    """Generate a random DPO-style rule at signature 2→3.

    LHS = K2 (edge). RHS = 3 vertices with random edges.
    Interface maps two LHS vertices to two of {0,1,2}.
    """
    rng = np.random.RandomState(seed)
    rhs_edges = []
    for e in [(0, 1), (0, 2), (1, 2)]:
        if rng.random() < 0.5:
            rhs_edges.append(e)
    interface = rng.choice(3, 2, replace=False).tolist()

    def rule(G, _rhs_edges=rhs_edges, _interface=interface, _rng=rng):
        H = G.copy()
        edges = list(H.edges())
        if not edges:
            n = max(H.nodes()) + 1 if H.nodes() else 0
            H.add_node(n)
            if n > 0:
                H.add_edge(n - 1, n)
            return H
        u, v = edges[_rng.randint(len(edges))]
        new = max(H.nodes()) + 1
        rhs_map = {}
        for i in range(3):
            if i == _interface[0]:
                rhs_map[i] = u
            elif i == _interface[1]:
                rhs_map[i] = v
            else:
                rhs_map[i] = new
                H.add_node(new)
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        for a, b in _rhs_edges:
            ma, mb = rhs_map[a], rhs_map[b]
            if ma != mb:
                H.add_edge(ma, mb)
        return H

    return rule


# ══════════════════════════════════════════════════════════════════════
# RULE REGISTRIES
# ══════════════════════════════════════════════════════════════════════

RULES_ORIGINAL = {
    "do_nothing":           rule_do_nothing,
    "add_random_edge":      rule_add_random_edge,
    "preferential_attach":  rule_preferential_attachment,
    "subdivision":          rule_subdivision,
    "triangle_closure":     rule_triangle_closure,
    "grid_growth":          rule_grid_growth,
    "line_growth":          rule_line_growth,
    "progressive_compress": rule_progressive_compression,
    "star_growth":          rule_star_growth,
    "cycle_then_fill":      rule_cycle_then_fill,
    "er_random":            rule_er_random,
    "copy_with_noise":      rule_copy_with_noise,
    "lattice_rewire":       rule_lattice_rewire,
    "fixed_grid_noise":     rule_fixed_grid_noise,
    "sorting_edges":        rule_sorting_edges,
}

RULES_NEW_WITNESSES = {
    "hierarchical_tree": rule_hierarchical_tree,
    "hub_sort":          rule_hub_sort,
    "encode_compress":   rule_encode_compress,
}

RULES_CATALOG = {
    "vertex_sprouting":  rule_vertex_sprouting,
    "edge_sprout_one":   rule_edge_sprouting_one,
    "triangle_complete": rule_triangle_completion,
    "edge_deletion":     rule_edge_deletion,
    "edge_rewiring":     rule_edge_rewiring,
}

RULES_STRUCTURAL = {
    "barabasi_albert":   rule_barabasi_albert,
    "watts_strogatz":    rule_watts_strogatz_grow,
    "caterpillar":       rule_caterpillar_growth,
    "complete_bipartite": rule_complete_bipartite_growth,
    "degree_regular":    rule_degree_regularization,
}

# Generate 5 random DPO rules (seeds 100–104)
RULES_RANDOM_DPO = {
    f"random_dpo_{i}": make_random_dpo_rule(seed=100 + i)
    for i in range(5)
}

# Combined registry: all 33 rules
ALL_RULES = {}
ALL_RULES.update(RULES_ORIGINAL)
ALL_RULES.update(RULES_NEW_WITNESSES)
ALL_RULES.update(RULES_CATALOG)
ALL_RULES.update(RULES_STRUCTURAL)
ALL_RULES.update(RULES_RANDOM_DPO)

# Source tag for each rule (used in classification reports)
RULE_SOURCE = {}
for r in RULES_ORIGINAL:
    RULE_SOURCE[r] = "orig"
for r in RULES_NEW_WITNESSES:
    RULE_SOURCE[r] = "witness"
for r in RULES_CATALOG:
    RULE_SOURCE[r] = "catalog"
for r in RULES_STRUCTURAL:
    RULE_SOURCE[r] = "struct"
for r in RULES_RANDOM_DPO:
    RULE_SOURCE[r] = "rndDPO"


# ══════════════════════════════════════════════════════════════════════
# TRAJECTORY GENERATION (basic — full version in trajectories.py)
# ══════════════════════════════════════════════════════════════════════

def run_trajectory(rule_fn, G0, T=30, n_max=None):
    """Run a rule for T steps from initial graph G0.

    Parameters
    ----------
    rule_fn : callable
        Graph → Graph rewrite rule.
    G0 : nx.Graph
        Initial graph.
    T : int
        Number of steps.
    n_max : int or None
        Safety cap on graph size. Default: config.N_MAX.

    Returns
    -------
    list of nx.Graph
        Trajectory including G0 (length ≤ T+1).
    """
    n_max = n_max or N_MAX
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
# CATALOG I/O
# ══════════════════════════════════════════════════════════════════════

_CATALOG_DIR = Path(__file__).parent.parent / "data" / "catalogs"


def load_catalog(signature):
    """Load a rule catalog JSON file.

    Parameters
    ----------
    signature : str
        E.g. "2_3" for the 2→3 catalog.

    Returns
    -------
    dict
        The parsed catalog with keys: signature, rules, total_connected, etc.
    """
    path = _CATALOG_DIR / f"rules_{signature}.json"
    with open(path) as f:
        return json.load(f)


def save_catalog(signature, catalog):
    """Save a rule catalog to JSON.

    Parameters
    ----------
    signature : str
        E.g. "3_4".
    catalog : dict
        Catalog data to write.
    """
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = _CATALOG_DIR / f"rules_{signature}.json"
    with open(path, 'w') as f:
        json.dump(catalog, f, indent=2)


def list_catalogs():
    """Return available catalog signature strings."""
    if not _CATALOG_DIR.exists():
        return []
    return sorted(
        p.stem.replace("rules_", "")
        for p in _CATALOG_DIR.glob("rules_*.json")
    )


# ══════════════════════════════════════════════════════════════════════
# RULE ENUMERATION — DPO rules at a given signature
# ══════════════════════════════════════════════════════════════════════

def enumerate_rhs_graphs(r_vertices):
    """Generate all simple graphs on r_vertices vertices (labeled).

    Returns list of (edges, graph) tuples where edges is a frozenset of
    (u,v) pairs with u < v.
    """
    vertices = list(range(r_vertices))
    possible_edges = list(itertools.combinations(vertices, 2))
    results = []
    for k in range(len(possible_edges) + 1):
        for edge_subset in itertools.combinations(possible_edges, k):
            G = nx.Graph()
            G.add_nodes_from(vertices)
            G.add_edges_from(edge_subset)
            results.append((frozenset(edge_subset), G))
    return results


def enumerate_interfaces(l_vertices, r_vertices):
    """Generate all injective maps from {0,...,l-1} to {0,...,r-1}.

    Each interface (iota) is a tuple of length l_vertices where
    iota[i] is the image of LHS vertex i in the RHS.
    """
    return list(itertools.permutations(range(r_vertices), l_vertices))


def _lhs_automorphisms(l_vertices):
    """Return the automorphism group of the LHS graph.

    For l=1: LHS is K1, trivial group {(0,)}.
    For l=2: LHS is K2, group is {(0,1), (1,0)}.
    For l>=3: LHS is K_l, full symmetric group S_l.
    """
    return list(itertools.permutations(range(l_vertices)))


def _canonical_form(r_vertices, edges, iota, lhs_auts=None):
    """Compute a canonical form for (RHS graph, interface) up to
    automorphisms of both the RHS and the LHS.

    Two rules (R, iota) and (R', iota') are isomorphic if there exist:
      - A permutation π of RHS vertices {0,...,r-1}
      - A permutation σ of LHS vertices {0,...,l-1}
    such that π maps edges to edges and π(iota(σ(i))) = iota'(i) for all i.

    Equivalently, we minimize over all (π, σ) the tuple
    (sorted_mapped_edges, mapped_iota_after_sigma).

    Returns a hashable canonical form.
    """
    l_vertices = len(iota)
    if lhs_auts is None:
        lhs_auts = _lhs_automorphisms(l_vertices)

    best = None
    for perm in itertools.permutations(range(r_vertices)):
        mapped_edges = tuple(sorted(
            (min(perm[u], perm[v]), max(perm[u], perm[v]))
            for u, v in edges
        ))
        # Apply each LHS automorphism σ: reorder iota as iota[σ[0]], iota[σ[1]], ...
        # then apply π
        for sigma in lhs_auts:
            mapped_iota = tuple(perm[iota[sigma[i]]] for i in range(l_vertices))
            candidate = (mapped_edges, mapped_iota)
            if best is None or candidate < best:
                best = candidate
    return best


def enumerate_rules(l_vertices, r_vertices, connected_only=None):
    """Enumerate all distinct DPO rules at signature l→r.

    Parameters
    ----------
    l_vertices : int
        Number of LHS vertices.
    r_vertices : int
        Number of RHS vertices.
    connected_only : bool or None
        If True, only keep rules whose RHS graph is connected.
        Default: config.R_CONNECTED_ONLY.

    Returns
    -------
    list of dict
        Each dict has keys: id, l_edges, r_edges, iota, fresh,
        l_edge_preserved, r_graph (nx.Graph).
    """
    if connected_only is None:
        connected_only = R_CONNECTED_ONLY

    # LHS: for l=1 it's K1; for l=2 it's K2 (the only connected graph)
    if l_vertices == 1:
        l_edges = []
    elif l_vertices == 2:
        l_edges = [(0, 1)]
    else:
        # For l > 2, LHS is the complete graph on l vertices
        l_edges = list(itertools.combinations(range(l_vertices), 2))

    rhs_graphs = enumerate_rhs_graphs(r_vertices)
    interfaces = enumerate_interfaces(l_vertices, r_vertices)

    seen = set()
    results = []
    rule_idx = 0

    for edges_fset, rhs_graph in rhs_graphs:
        if connected_only and not nx.is_connected(rhs_graph):
            continue

        for iota in interfaces:
            canon = _canonical_form(r_vertices, edges_fset, iota)
            if canon in seen:
                continue
            seen.add(canon)

            # Determine interface properties
            interface_set = set(iota)
            fresh = [v for v in range(r_vertices) if v not in interface_set]

            # Check if L-edge is preserved in R
            l_edge_preserved = True
            for u, v in l_edges:
                img_u, img_v = iota[u], iota[v]
                img_edge = (min(img_u, img_v), max(img_u, img_v))
                if img_edge not in edges_fset:
                    l_edge_preserved = False
                    break

            rule_idx += 1
            results.append({
                "id": f"rule_{r_vertices}_{rule_idx}",
                "l_vertices": l_vertices,
                "r_vertices": r_vertices,
                "l_edges": [list(e) for e in l_edges],
                "r_edges": [list(e) for e in sorted(edges_fset)],
                "iota": list(iota),
                "fresh": fresh,
                "l_edge_preserved": l_edge_preserved,
                "r_graph": rhs_graph,
            })

    return results


def enumerate_rules_at_signature(l_vertices, r_vertices=None,
                                 connected_only=None):
    """Enumerate rules and return both the list and a summary catalog dict.

    Parameters
    ----------
    l_vertices : int
    r_vertices : int or None
        Default: l_vertices + config.SIGNATURE_R_OFFSET.
    connected_only : bool or None

    Returns
    -------
    catalog : dict
        Ready to save with save_catalog().
    rules : list of dict
        The enumerated rules (with r_graph key).
    """
    if r_vertices is None:
        r_vertices = l_vertices + SIGNATURE_R_OFFSET

    rules = enumerate_rules(l_vertices, r_vertices, connected_only)

    # Strip nx.Graph objects for JSON serialization
    catalog_rules = []
    for r in rules:
        cr = {k: v for k, v in r.items() if k != "r_graph"}
        catalog_rules.append(cr)

    # Count total (including disconnected) for reference
    all_rules = enumerate_rules(l_vertices, r_vertices, connected_only=False)

    catalog = {
        "signature": f"{l_vertices}_{r_vertices}",
        "l_vertices": l_vertices,
        "r_vertices": r_vertices,
        "r_connected_only": bool(connected_only if connected_only is not None
                                  else R_CONNECTED_ONLY),
        "rules": catalog_rules,
        "total_connected": len(rules),
        "total_any": len(all_rules),
    }
    return catalog, rules


def dpo_rule_to_callable(rule_dict):
    """Convert a catalog rule dict into a callable graph rewrite function.

    The returned function applies the DPO rule to one randomly matched
    edge (for signature 2→r) or vertex (for signature 1→r).
    """
    l_edges = [tuple(e) for e in rule_dict["l_edges"]]
    r_edges = [tuple(e) for e in rule_dict["r_edges"]]
    iota = rule_dict["iota"]
    fresh = rule_dict.get("fresh", [])
    l_vertices = rule_dict.get("l_vertices", len(iota))

    def rule_fn(G):
        H = G.copy()

        if l_vertices == 1:
            # Match a single vertex
            nodes = list(H.nodes())
            if not nodes:
                H.add_node(0)
                return H
            matched_v = nodes[np.random.randint(len(nodes))]
            match_map = {0: matched_v}
        elif l_vertices == 2:
            # Match an edge
            edges = list(H.edges())
            if not edges:
                n = max(H.nodes()) + 1 if H.nodes() else 0
                H.add_node(n)
                if n > 0:
                    H.add_edge(n - 1, n)
                return H
            u, v = edges[np.random.randint(len(edges))]
            match_map = {0: u, 1: v}
        elif l_vertices == 3:
            # Match a triangle (K3)
            triangles = []
            for u in H.nodes():
                for v in H.neighbors(u):
                    if v <= u:
                        continue
                    for w in H.neighbors(u):
                        if w <= v:
                            continue
                        if H.has_edge(v, w):
                            triangles.append((u, v, w))
            if not triangles:
                # No triangle found — grow toward one
                nodes = list(H.nodes())
                if len(nodes) < 3:
                    n = max(H.nodes()) + 1 if H.nodes() else 0
                    H.add_node(n)
                    if n > 0:
                        H.add_edge(n - 1, n)
                    return H
                # Add edges to form a triangle
                u, v = nodes[0], nodes[1]
                if not H.has_edge(u, v):
                    H.add_edge(u, v)
                w = nodes[2]
                if not H.has_edge(u, w):
                    H.add_edge(u, w)
                if not H.has_edge(v, w):
                    H.add_edge(v, w)
                return H
            tri = triangles[np.random.randint(len(triangles))]
            match_map = {0: tri[0], 1: tri[1], 2: tri[2]}
        else:
            raise NotImplementedError(
                f"LHS matching for l_vertices={l_vertices} not implemented")

        # Map RHS vertices
        new_base = max(H.nodes()) + 1
        rhs_map = {}
        fresh_idx = 0
        for rv in range(max(iota) + 1 if iota else 0):
            pass  # handled below
        for i, img in enumerate(iota):
            rhs_map[img] = match_map[i]
        for fv in fresh:
            rhs_map[fv] = new_base + fresh_idx
            H.add_node(new_base + fresh_idx)
            fresh_idx += 1

        # Remove matched L-edges that are not preserved in R
        for le in l_edges:
            img_edge = (min(rhs_map.get(iota[le[0]], -1),
                           rhs_map.get(iota[le[1]], -1)),
                       max(rhs_map.get(iota[le[0]], -1),
                           rhs_map.get(iota[le[1]], -1)))
            r_edge_canonical = (min(iota[le[0]], iota[le[1]]),
                               max(iota[le[0]], iota[le[1]]))
            if r_edge_canonical not in [(min(e), max(e)) for e in r_edges]:
                u, v = match_map[le[0]], match_map[le[1]]
                if H.has_edge(u, v):
                    H.remove_edge(u, v)

        # Add RHS edges
        for a, b in r_edges:
            if a in rhs_map and b in rhs_map:
                ma, mb = rhs_map[a], rhs_map[b]
                if ma != mb and not H.has_edge(ma, mb):
                    H.add_edge(ma, mb)

        return H

    return rule_fn
