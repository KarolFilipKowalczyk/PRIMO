"""
exp02 — Example B analysis: contraction mapping classification.

Paper 2 Section 6.2: A contraction mapping (remove highest-betweenness
edge, add closest-pair edge) that converges but is NOT Bayesian.
The I-predicate should classify it as I-negative.

This experiment:
  1. Implements the Example B contraction rule
  2. Classifies under canonical seeds (K1,K2,K3,P3 bootstrapped to n=20)
  3. Tests robustness on 20 random initial graphs
  4. Analyzes the fixed-point convergence behavior
  5. Measures Grassmannian straightness (discriminator for exp03)
  6. Tests the adaptive variant (grow-then-contract)

Expected results (from paper2_bayesian.md):
  - Canonical: I-negative (2/4 seeds), per-seed τ ≈ 0.50, 0.44, 0.00, 0.50
  - Random seeds: ~7/20 I-positive (boundary behavior)
  - Straightness: ~0.5-0.84 (high = near-geodesic, unlike true inference)
  - Fixed point reached within 3-5 steps

Usage:
    python experiments/exp02_example_b.py
    # or: make exp02
"""

import csv
import json
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, MASTER_SEED, MAJORITY_THRESHOLD,
    EMBEDDING_DIM, DATA_DIR,
)
from primo.rules import make_initial_graphs, run_trajectory
from primo.predicates import (
    classify_I, classify_Phi, measure_I_scores,
)
from primo.trajectories import (
    EMBEDDING_FUNCTIONS, embed_trajectory, subspace_cosine,
)
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# EXAMPLE B: CONTRACTION MAPPING RULE
# ══════════════════════════════════════════════════════════════════════

EXAMPLE_B_N = 20  # target graph size for bootstrapping


def bootstrap_graph(G0, n_target):
    """Bootstrap a small graph to n_target nodes by adding edges
    between random pairs until we have n_target nodes."""
    H = G0.copy()
    while H.number_of_nodes() < n_target:
        new = max(H.nodes()) + 1
        H.add_node(new)
        # Connect to a random existing node
        existing = list(set(H.nodes()) - {new})
        if existing:
            target = existing[np.random.randint(len(existing))]
            H.add_edge(new, target)
    # Add some extra edges to make it denser
    nodes = list(H.nodes())
    for _ in range(n_target // 2):
        u, v = np.random.choice(nodes, 2, replace=False)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
    return H


def example_b_contraction_rule(G):
    """Example B: contraction mapping.

    At each step:
    1. Remove the edge with maximum betweenness centrality
    2. Add an edge between the two closest non-adjacent nodes (by graph distance)
    3. If removal disconnects the graph, reconnect largest components
    """
    H = G.copy()

    if H.number_of_edges() == 0:
        return H

    # Step 1: Remove highest-betweenness edge
    edge_bc = nx.edge_betweenness_centrality(H)
    if not edge_bc:
        return H
    worst_edge = max(edge_bc, key=edge_bc.get)
    H.remove_edge(*worst_edge)

    # Handle disconnection: reconnect largest components
    if not nx.is_connected(H):
        components = sorted(nx.connected_components(H), key=len, reverse=True)
        if len(components) >= 2:
            # Connect highest-degree nodes from two largest components
            c1, c2 = list(components[0]), list(components[1])
            d1 = max(c1, key=lambda n: H.degree(n))
            d2 = max(c2, key=lambda n: H.degree(n))
            H.add_edge(d1, d2)

    # Step 2: Add edge between closest non-adjacent pair
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(H))
    except nx.NetworkXError:
        return H

    best_dist = float('inf')
    best_pair = None
    nodes = list(H.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if not H.has_edge(u, v) and v in lengths.get(u, {}):
                d = lengths[u][v]
                if d < best_dist and d > 1:
                    best_dist = d
                    best_pair = (u, v)
    if best_pair:
        H.add_edge(*best_pair)

    return H


def adaptive_variant_rule(G):
    """Adaptive variant: grow small graph, then contract.

    If graph has < 20 nodes, grow (add node + edge).
    Otherwise, apply contraction rule.
    """
    if G.number_of_nodes() < EXAMPLE_B_N:
        H = G.copy()
        new = max(H.nodes()) + 1
        H.add_node(new)
        existing = list(set(H.nodes()) - {new})
        if existing:
            target = existing[np.random.randint(len(existing))]
            H.add_edge(new, target)
        return H
    return example_b_contraction_rule(G)


# ══════════════════════════════════════════════════════════════════════
# GRASSMANNIAN STRAIGHTNESS
# ══════════════════════════════════════════════════════════════════════

def grassmannian_straightness(traj, embed_fn, d=None):
    """Compute Grassmannian trajectory straightness.

    S = d_G([V_0], [V_T]) / sum(d_G([V_s], [V_{s+1}]))

    Low straightness (~0.3-0.4) = winding path = genuine inference
    High straightness (~0.5-0.8) = near-geodesic = contraction

    Returns
    -------
    float or None (if trajectory too short or all identical)
    """
    d = d or EMBEDDING_DIM
    embeddings = embed_trajectory(traj, embed_fn, d)

    # Filter valid embeddings
    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 3:
        return None

    # Total path length (sum of consecutive distances)
    total_path = 0.0
    for j in range(len(valid) - 1):
        _, e1 = valid[j]
        _, e2 = valid[j + 1]
        cos = subspace_cosine(e1, e2)
        # Convert cosine to angular distance
        cos_clamped = max(-1.0, min(1.0, cos))
        total_path += np.arccos(cos_clamped)

    if total_path < 1e-10:
        return None  # All embeddings identical (fixed point reached instantly)

    # Direct distance (start to end)
    _, e_start = valid[0]
    _, e_end = valid[-1]
    cos_direct = subspace_cosine(e_start, e_end)
    cos_direct = max(-1.0, min(1.0, cos_direct))
    direct_dist = np.arccos(cos_direct)

    return direct_dist / total_path


# ══════════════════════════════════════════════════════════════════════
# FIXED-POINT ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def find_fixed_point_step(traj):
    """Find the step at which the trajectory reaches a fixed point.

    A fixed point is detected when the graph stops changing
    (same edge set for consecutive steps).

    Returns the step number, or None if no fixed point reached.
    """
    for t in range(len(traj) - 1):
        if set(traj[t].edges()) == set(traj[t + 1].edges()):
            return t + 1
    return None


def edit_distance(G1, G2):
    """Simple edge edit distance between two graphs."""
    e1 = set(map(frozenset, G1.edges()))
    e2 = set(map(frozenset, G2.edges()))
    return len(e1.symmetric_difference(e2))


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

# Total ticks: 4 canonical + 20 random + 4 adaptive = 28
N_CANONICAL = 4
N_RANDOM = 20
N_ADAPTIVE = 4
TOTAL_RULES = N_CANONICAL + N_RANDOM + N_ADAPTIVE


def run_exp02(runner):
    """Main experiment: Example B analysis."""
    results_dir = os.path.join(DATA_DIR, "exp02")
    os.makedirs(results_dir, exist_ok=True)

    # ── Part 1: Canonical seeds (K1,K2,K3,P3 bootstrapped to n=20) ──
    with runner.phase("Part 1: Canonical seeds (n=20)"):
        canonical_seeds = make_initial_graphs()
        bootstrapped = {}
        for name, G0 in canonical_seeds.items():
            np.random.seed(MASTER_SEED)
            bootstrapped[name] = bootstrap_graph(G0, EXAMPLE_B_N)

        canonical_results = {}
        I_count = 0
        for seed_name, G0 in bootstrapped.items():
            if runner.should_stop():
                return

            runner.begin_rule(seed_name)
            np.random.seed(MASTER_SEED)
            traj = run_trajectory(example_b_contraction_rule, G0, T_DEFAULT)
            I_pos, I_detail = classify_I(traj)
            Phi_pos, Phi_detail = classify_Phi(traj)

            # Per-embedding tau values
            embed_scores = I_detail.get("embed_scores", {})
            tau_vals = {e: embed_scores.get(e, {}).get("tau_to_final", 0.0)
                        for e in EMBEDDING_FUNCTIONS}

            # Straightness per embedding
            straightness = {}
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                s = grassmannian_straightness(traj, emb_fn)
                straightness[emb_name] = s

            # Fixed-point step
            fp_step = find_fixed_point_step(traj)

            # Edit distances
            edit_dists = [edit_distance(traj[t], traj[t + 1])
                          for t in range(len(traj) - 1)]
            active_steps = sum(1 for d in edit_dists if d > 0)

            canonical_results[seed_name] = {
                "I_positive": I_pos,
                "Phi_positive": Phi_pos,
                "tau_values": tau_vals,
                "straightness": straightness,
                "fixed_point_step": fp_step,
                "active_steps": active_steps,
                "total_steps": len(traj) - 1,
                "edit_distances": edit_dists,
            }

            if I_pos:
                I_count += 1

            cell = ("I+" if I_pos else "I-") + ("Φ+" if Phi_pos else "Φ-")
            runner.tick(seed_name, "canonical", result=cell)
            runner.finish_rule(seed_name, classification=cell)

        # Print canonical results
        print()
        print("=" * 80)
        print("EXAMPLE B — CANONICAL SEEDS (bootstrapped to n=20)")
        print("=" * 80)
        print(f"I-positive: {I_count}/{len(bootstrapped)} "
              f"(majority threshold: {MAJORITY_THRESHOLD})")
        print(f"Classification: {'I-positive' if I_count >= MAJORITY_THRESHOLD else 'I-negative'}")
        print()

        print(f"  {'Seed':<8} {'I?':<6} {'FP step':<10} {'Active':<10} ", end="")
        for emb in EMBEDDING_FUNCTIONS:
            print(f"{'τ_' + emb[:3]:<12}", end="")
        for emb in EMBEDDING_FUNCTIONS:
            print(f"{'S_' + emb[:3]:<12}", end="")
        print()
        print(f"  {'─' * 90}")

        for seed_name, r in canonical_results.items():
            fp = r["fixed_point_step"] if r["fixed_point_step"] else "none"
            print(f"  {seed_name:<8} "
                  f"{'YES' if r['I_positive'] else 'no':<6} "
                  f"{str(fp):<10} "
                  f"{r['active_steps']}/{r['total_steps']:<7} ", end="")
            for emb in EMBEDDING_FUNCTIONS:
                tau = r["tau_values"].get(emb, 0.0)
                print(f"{tau:<12.3f}", end="")
            for emb in EMBEDDING_FUNCTIONS:
                s = r["straightness"].get(emb)
                s_str = f"{s:.3f}" if s is not None else "N/A"
                print(f"{s_str:<12}", end="")
            print()
        print()

    if runner.should_stop():
        return

    # ── Part 2: Random initial graphs (robustness test) ──────────────
    runner.add_extra_ticks(N_RANDOM)
    with runner.phase("Part 2: Random seeds (robustness, 20 graphs)"):
        random_results = {}
        random_I_count = 0

        for trial in range(N_RANDOM):
            if runner.should_stop():
                return

            trial_name = f"random_{trial:02d}"
            runner.begin_rule(trial_name)

            np.random.seed(trial * 100 + 7)
            G0 = nx.erdos_renyi_graph(EXAMPLE_B_N, 0.3)
            while not nx.is_connected(G0):
                G0 = nx.erdos_renyi_graph(EXAMPLE_B_N, 0.3)

            np.random.seed(MASTER_SEED)
            traj = run_trajectory(example_b_contraction_rule, G0, T_DEFAULT)
            I_pos, I_detail = classify_I(traj)

            embed_scores = I_detail.get("embed_scores", {})
            tau_vals = {e: embed_scores.get(e, {}).get("tau_to_final", 0.0)
                        for e in EMBEDDING_FUNCTIONS}

            straightness = {}
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                straightness[emb_name] = grassmannian_straightness(traj, emb_fn)

            fp_step = find_fixed_point_step(traj)

            random_results[trial_name] = {
                "I_positive": I_pos,
                "tau_values": tau_vals,
                "straightness": straightness,
                "fixed_point_step": fp_step,
            }

            if I_pos:
                random_I_count += 1

            result_str = "I+" if I_pos else "I-"
            runner.tick(trial_name, "random", result=result_str)
            runner.finish_rule(trial_name, classification=result_str)

        print("=" * 80)
        print("EXAMPLE B — RANDOM SEEDS (robustness test)")
        print("=" * 80)
        print(f"I-positive: {random_I_count}/{N_RANDOM}")
        print()

        # Separate I+ and I- for comparison
        i_pos_taus = []
        i_neg_taus = []
        i_pos_straightness = []
        i_neg_straightness = []
        for r in random_results.values():
            mean_tau = np.mean([v for v in r["tau_values"].values()])
            s_vals = [v for v in r["straightness"].values() if v is not None]
            mean_s = np.mean(s_vals) if s_vals else None
            if r["I_positive"]:
                i_pos_taus.append(mean_tau)
                if mean_s is not None:
                    i_pos_straightness.append(mean_s)
            else:
                i_neg_taus.append(mean_tau)
                if mean_s is not None:
                    i_neg_straightness.append(mean_s)

        if i_pos_taus:
            print(f"  I+ cases ({len(i_pos_taus)}): "
                  f"mean τ = {np.mean(i_pos_taus):.3f}, "
                  f"mean straightness = {np.mean(i_pos_straightness):.3f}" if i_pos_straightness else "")
        if i_neg_taus:
            print(f"  I- cases ({len(i_neg_taus)}): "
                  f"mean τ = {np.mean(i_neg_taus):.3f}, "
                  f"mean straightness = {np.mean(i_neg_straightness):.3f}" if i_neg_straightness else "")
        print()

    if runner.should_stop():
        return

    # ── Part 3: Adaptive variant (grow-then-contract) ────────────────
    runner.add_extra_ticks(N_ADAPTIVE)
    with runner.phase("Part 3: Adaptive variant (grow-then-contract)"):
        adaptive_results = {}
        adaptive_I_count = 0

        small_seeds = make_initial_graphs()
        for seed_name, G0 in small_seeds.items():
            if runner.should_stop():
                return

            adapt_name = f"adaptive_{seed_name}"
            runner.begin_rule(adapt_name)

            np.random.seed(MASTER_SEED)
            traj = run_trajectory(adaptive_variant_rule, G0, T_DEFAULT)
            I_pos, I_detail = classify_I(traj)

            embed_scores = I_detail.get("embed_scores", {})
            tau_vals = {e: embed_scores.get(e, {}).get("tau_to_final", 0.0)
                        for e in EMBEDDING_FUNCTIONS}

            straightness = {}
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                straightness[emb_name] = grassmannian_straightness(traj, emb_fn)

            adaptive_results[adapt_name] = {
                "I_positive": I_pos,
                "tau_values": tau_vals,
                "straightness": straightness,
            }

            if I_pos:
                adaptive_I_count += 1

            result_str = "I+" if I_pos else "I-"
            runner.tick(adapt_name, "adaptive", result=result_str)
            runner.finish_rule(adapt_name, classification=result_str)

        print("=" * 80)
        print("EXAMPLE B — ADAPTIVE VARIANT (grow-then-contract)")
        print("=" * 80)
        print(f"I-positive: {adaptive_I_count}/{len(small_seeds)}")
        print()
        for name, r in adaptive_results.items():
            tau_str = ", ".join(f"{e[:3]}={v:.3f}"
                                for e, v in r["tau_values"].items())
            s_vals = [v for v in r["straightness"].values() if v is not None]
            s_str = f"{np.mean(s_vals):.3f}" if s_vals else "N/A"
            print(f"  {name}: {'I+' if r['I_positive'] else 'I-'}, "
                  f"τ=[{tau_str}], straightness={s_str}")
        print()

    if runner.should_stop():
        return

    # ── Part 4: Straightness comparison with known I+ rules ─────────
    with runner.phase("Part 4: Straightness comparison"):
        print("=" * 80)
        print("STRAIGHTNESS COMPARISON: Contraction vs known I+ rules")
        print("=" * 80)

        from primo.rules import ALL_RULES
        comparison_rules = ["sorting_edges", "preferential_attach",
                            "line_growth", "star_growth", "grid_growth"]

        print(f"\n  {'System':<26} {'Mean S':<10} {'Mean τ':<10} {'Type'}")
        print(f"  {'─' * 60}")

        # Contraction I+ cases
        canon_s = [np.mean([v for v in r["straightness"].values() if v is not None])
                   for r in canonical_results.values()
                   if r["I_positive"]
                   and any(v is not None for v in r["straightness"].values())]
        canon_tau = [np.mean(list(r["tau_values"].values()))
                     for r in canonical_results.values() if r["I_positive"]]
        if canon_s:
            print(f"  {'Contraction (I+ seeds)':<26} "
                  f"{np.mean(canon_s):<10.3f} "
                  f"{np.mean(canon_tau):<10.3f} contraction")

        # Contraction I- cases
        canon_s_neg = [np.mean([v for v in r["straightness"].values() if v is not None])
                       for r in canonical_results.values()
                       if not r["I_positive"]
                       and any(v is not None for v in r["straightness"].values())]
        canon_tau_neg = [np.mean(list(r["tau_values"].values()))
                         for r in canonical_results.values()
                         if not r["I_positive"]]
        if canon_s_neg:
            print(f"  {'Contraction (I- seeds)':<26} "
                  f"{np.mean(canon_s_neg):<10.3f} "
                  f"{np.mean(canon_tau_neg):<10.3f} contraction")

        # Known I+ rules
        seeds = make_initial_graphs()
        for rule_name in comparison_rules:
            if rule_name not in ALL_RULES:
                continue
            rule_fn = ALL_RULES[rule_name]
            np.random.seed(MASTER_SEED)
            G0 = seeds.get("K3", list(seeds.values())[0])
            traj = run_trajectory(rule_fn, G0, T_DEFAULT)

            s_vals = []
            tau_vals = []
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                s = grassmannian_straightness(traj, emb_fn)
                if s is not None:
                    s_vals.append(s)
                scores = measure_I_scores(traj, emb_fn)
                tau_vals.append(scores["tau_to_final"])

            if s_vals:
                print(f"  {rule_name:<26} "
                      f"{np.mean(s_vals):<10.3f} "
                      f"{np.mean(tau_vals):<10.3f} known I+")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    canonical_I = sum(1 for r in canonical_results.values() if r["I_positive"])
    print(f"  Canonical classification: I-{'positive' if canonical_I >= MAJORITY_THRESHOLD else 'negative'} "
          f"({canonical_I}/{len(canonical_results)})")
    print(f"  Random seed robustness: {random_I_count}/{N_RANDOM} I-positive")
    print(f"  Adaptive variant: {adaptive_I_count}/{len(small_seeds)} I-positive")

    # Fixed-point analysis
    fp_steps = [r["fixed_point_step"] for r in canonical_results.values()
                if r["fixed_point_step"] is not None]
    if fp_steps:
        print(f"  Fixed-point convergence: steps {fp_steps} (mean={np.mean(fp_steps):.1f})")

    active_ratios = [r["active_steps"] / r["total_steps"]
                     for r in canonical_results.values()]
    print(f"  Active dynamics: {np.mean(active_ratios):.1%} of steps have non-zero edits")

    # Assessment
    print()
    if canonical_I < MAJORITY_THRESHOLD:
        print("  ✓ Example B correctly classified as I-negative under canonical spec")
    else:
        print("  ✗ WARNING: Example B classified as I-positive — predicate may be too permissive")

    if adaptive_I_count >= MAJORITY_THRESHOLD:
        print("  ⚠ Adaptive variant classified I-positive — straightness gate needed (exp03)")
    else:
        print("  ✓ Adaptive variant correctly classified as I-negative")

    # ── Save results ─────────────────────────────────────────────────
    _save_results(results_dir, canonical_results, random_results, adaptive_results)

    runner.finish(f"exp02 complete — canonical: {canonical_I}/4, "
                  f"random: {random_I_count}/20, "
                  f"adaptive: {adaptive_I_count}/4")


def _save_results(results_dir, canonical, random_r, adaptive):
    """Save all results to JSON."""
    def _clean(d):
        """Make results JSON-serializable."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = _clean(v)
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, list):
                out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                          for x in v]
            else:
                out[k] = v
        return out

    results = {
        "experiment": "exp02",
        "canonical": _clean(canonical),
        "random": _clean(random_r),
        "adaptive": _clean(adaptive),
    }

    json_path = os.path.join(results_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = StepRunner(
        "exp02_example_b",
        total_rules=N_CANONICAL,
        total_seeds=1,
        phases=[
            "Canonical seeds",
            "Random seeds (robustness)",
            "Adaptive variant",
            "Straightness comparison",
        ],
    )
    runner.run(run_exp02)
