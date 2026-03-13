"""
exp12 — Dehn-twist counterexample: Φ+ but I− without growth.

Paper 4 Theorem 1 says Φ+ → I+ for growth rules. The Remark after the
theorem claims that a Dehn-twist on a toroidal grid is a counterexample
when growth is absent: it should be Φ+ (perfect geometry) but I−
(oscillating embeddings). This experiment verifies that claim.

Construction:
  1. Start with a 10×10 toroidal grid (100 nodes, 4-regular).
  2. At each step, apply a Dehn twist: shift column indices by 1 mod m.
  3. The graph is isomorphic at every step, but node labels change,
     which rotates the Laplacian eigenvectors within degenerate subspaces.

Key subtlety: the degree-profile embedding is invariant under relabeling
(every node has the same local features). If this makes the trajectory
I-positive, then the Dehn twist is NOT a valid counterexample — the
I-predicate correctly sees through the relabeling.

Usage:
    python experiments/exp12_dehn_twist.py
"""

import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, EMBEDDING_DIM, DATA_DIR, MASTER_SEED,
)
from primo.predicates import classify_I, classify_Phi, measure_I_scores
from primo.trajectories import (
    EMBEDDING_FUNCTIONS, subspace_cosine, embed_trajectory,
    compression_ratio,
)
from primo.run_utils import StepRunner


M = 10  # grid dimension (M×M torus = 100 nodes)
T = 30  # trajectory length


# ══════════════════════════════════════════════════════════════════════
# DEHN TWIST CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def make_torus(m):
    """Create an m×m toroidal grid with integer node labels."""
    G = nx.grid_2d_graph(m, m, periodic=True)
    # Relabel (i,j) → i*m + j
    mapping = {(i, j): i * m + j for i in range(m) for j in range(m)}
    return nx.relabel_nodes(G, mapping)


def dehn_twist(G, m):
    """Apply Dehn twist: shift column indices by 1 mod m."""
    mapping = {}
    for i in range(m):
        for j in range(m):
            old_node = i * m + j
            new_node = i * m + ((j + 1) % m)
            mapping[old_node] = new_node
    return nx.relabel_nodes(G, mapping)


def generate_dehn_trajectory(m, T):
    """Generate T-step Dehn-twist trajectory on m×m torus."""
    traj = []
    G = make_torus(m)
    traj.append(G.copy())
    for _ in range(T):
        G = dehn_twist(G, m)
        traj.append(G.copy())
    return traj


# ══════════════════════════════════════════════════════════════════════
# EIGENVALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def count_eigenvalue_crossings(traj, k):
    """Count eigenvalue crossings at position k across trajectory."""
    n_crossings = 0
    prev_eigs = None

    for G in traj:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigs = np.linalg.eigvalsh(L)
        nonzero = np.sort(eigs[eigs > 1e-10])

        if (prev_eigs is not None
                and len(nonzero) >= k + 1
                and len(prev_eigs) >= k + 1):
            curr_gap = nonzero[k] - nonzero[k - 1]
            prev_gap = prev_eigs[k] - prev_eigs[k - 1]
            if curr_gap * prev_gap < 0:
                n_crossings += 1

        prev_eigs = nonzero

    return n_crossings


def compute_cosine_to_final(traj, embed_fn, d):
    """Compute cosine-to-final sequence for one embedding."""
    embeddings = embed_trajectory(traj, embed_fn, d)

    # Find last valid embedding
    final_emb = None
    for emb in reversed(embeddings):
        if emb is not None and emb.shape[0] >= d:
            final_emb = emb
            break

    if final_emb is None:
        return []

    cosines = []
    for emb in embeddings:
        if emb is not None and emb.shape[0] >= d:
            cosines.append(float(subspace_cosine(emb, final_emb)))
        else:
            cosines.append(None)

    return cosines


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp12(runner):
    """Dehn-twist counterexample verification."""
    results_dir = os.path.join(DATA_DIR, "exp12")
    os.makedirs(results_dir, exist_ok=True)

    # ── Generate trajectory ───────────────────────────────────────
    with runner.phase("Generating Dehn-twist trajectory"):
        runner.begin_rule("Dehn twist")
        np.random.seed(MASTER_SEED)
        traj = generate_dehn_trajectory(M, T)
        runner.tick("Dehn twist", f"{M}x{M}", result=f"T={T}, {len(traj)} steps")
        runner.log(f"  Torus: {M}x{M} = {M*M} nodes, T={T}")

    if runner.should_stop():
        return

    # ── Φ-predicate ───────────────────────────────────────────────
    with runner.phase("Classifying Φ-predicate"):
        phi_pos, phi_detail = classify_Phi(traj)
        runner.tick("Dehn twist", "Phi",
                    result=f"{'Φ+' if phi_pos else 'Φ-'}")

    if runner.should_stop():
        return

    # ── I-predicate ───────────────────────────────────────────────
    with runner.phase("Classifying I-predicate"):
        i_pos, i_detail = classify_I(traj)
        runner.tick("Dehn twist", "I",
                    result=f"{'I+' if i_pos else 'I-'}")

    if runner.should_stop():
        return

    # ── Per-embedding analysis ────────────────────────────────────
    with runner.phase("Per-embedding analysis"):
        embed_results = {}
        for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
            scores = measure_I_scores(traj, emb_fn)
            cosines = compute_cosine_to_final(traj, emb_fn, EMBEDDING_DIM)
            valid_cosines = [c for c in cosines if c is not None]

            embed_results[emb_name] = {
                "tau_to_final": scores["tau_to_final"],
                "cosines_to_final": cosines,
                "cosine_min": float(min(valid_cosines)) if valid_cosines else None,
                "cosine_max": float(max(valid_cosines)) if valid_cosines else None,
                "cosine_range": (float(max(valid_cosines)) - float(min(valid_cosines)))
                    if valid_cosines else None,
            }

            runner.tick("Dehn twist", emb_name,
                        result=f"τ={scores['tau_to_final']:.3f}")

    if runner.should_stop():
        return

    # ── Eigenvalue crossings ──────────────────────────────────────
    with runner.phase("Eigenvalue analysis"):
        n_crossings = count_eigenvalue_crossings(traj, EMBEDDING_DIM)

        # Check eigenvalue stability: compute eigenvalues at first and last step
        L0 = nx.laplacian_matrix(traj[0]).toarray().astype(float)
        LT = nx.laplacian_matrix(traj[-1]).toarray().astype(float)
        eigs0 = np.sort(np.linalg.eigvalsh(L0))
        eigsT = np.sort(np.linalg.eigvalsh(LT))
        eig_diff = float(np.linalg.norm(eigs0 - eigsT))

        runner.tick("Dehn twist", "eigenvalues",
                    result=f"crossings={n_crossings}, diff={eig_diff:.6f}")

    if runner.should_stop():
        return

    # ── Compression ratio ─────────────────────────────────────────
    comp_ratio = compression_ratio(traj)

    # ── Summary ───────────────────────────────────────────────────
    with runner.phase("Summary"):
        print()
        print("═" * 70)
        print("DEHN-TWIST COUNTEREXAMPLE")
        print("═" * 70)
        print()
        print(f"Grid size: {M}x{M} ({M*M} nodes)")
        print(f"Trajectory length: T={T}")
        print()

        # Φ-predicate details
        print("Φ-predicate:")
        ds_mean = phi_detail.get("ds_mean")
        ds_std = phi_detail.get("ds_std")
        best_law = phi_detail.get("best_law_resid")
        curv = phi_detail.get("curv_homogeneity")
        print(f"  ds_mean: {ds_mean:.4f}" if ds_mean is not None else "  ds_mean: N/A")
        print(f"  ds_std:  {ds_std:.4f}" if ds_std is not None else "  ds_std:  N/A")
        print(f"  Best law residual: {best_law:.4f}" if best_law is not None else "  Best law residual: N/A")
        print(f"  Curvature CV: {curv:.4f}" if curv is not None else "  Curvature CV: N/A")
        print(f"  Classification: {'Φ-POSITIVE' if phi_pos else 'Φ-negative'}")
        print()

        # I-predicate details
        print("I-predicate:")
        print(f"  Compression ratio: {comp_ratio:.4f}")
        comp_gate = i_detail.get("compression", comp_ratio)
        comp_pass = comp_gate < 0.85  # RHO_STAR
        print(f"  Compression gate: {'PASS' if comp_pass else 'fail'}")

        for emb_name, er in embed_results.items():
            tau = er["tau_to_final"]
            tau_str = f"{tau:.4f}" if not np.isnan(tau) else "NaN"
            gate = ">" if tau > TAU_STAR else "<"
            print(f"  τ_to_final ({emb_name:>11s}): {tau_str} {gate} τ*={TAU_STAR}")

        convergence_pass = i_detail.get("convergence_pass",
                                        any(er["tau_to_final"] > TAU_STAR
                                            for er in embed_results.values()
                                            if not np.isnan(er["tau_to_final"])))
        print(f"  Convergence gate: {'PASS' if convergence_pass else 'fail'}")
        print(f"  Classification: {'I-POSITIVE' if i_pos else 'I-NEGATIVE'}")
        print()

        # Eigenvalue analysis
        print(f"Eigenvalue crossings: {n_crossings} (expected 0)")
        print(f"Eigenvalue spectrum difference (‖λ₀-λ_T‖): {eig_diff:.6f} (expected ~0)")
        print()

        # Cosine-to-final analysis
        print("Cosine-to-final range per embedding:")
        for emb_name, er in embed_results.items():
            if er["cosine_range"] is not None:
                print(f"  {emb_name:>11s}: [{er['cosine_min']:.4f}, {er['cosine_max']:.4f}] "
                      f"range={er['cosine_range']:.4f}")
            else:
                print(f"  {emb_name:>11s}: N/A")
        print()

        # ── Verdict ───────────────────────────────────────────────
        print("VERDICT:")
        is_counterexample = phi_pos and not i_pos

        if is_counterexample:
            print(f"  Φ+, I−: YES")
            print(f"  -> Dehn-twist is a valid (I−, Φ+) counterexample: CONFIRMED")
            verdict = "CONFIRMED"
        else:
            print(f"  Φ+={phi_pos}, I+={i_pos}")
            if i_pos and phi_pos:
                # Check which embedding made it I+
                i_plus_embeds = [n for n, er in embed_results.items()
                                 if not np.isnan(er["tau_to_final"])
                                 and er["tau_to_final"] > TAU_STAR]
                print(f"  -> Dehn-twist is NOT a counterexample.")
                print(f"     I-positive via embedding(s): {', '.join(i_plus_embeds)}")
                if "degree_prof" in i_plus_embeds:
                    print(f"     The degree-profile embedding is invariant under")
                    print(f"     relabeling (vertex-transitive graph), giving τ ≈ 1.0.")
                    print(f"     The I-predicate correctly detects that the graph is")
                    print(f"     structurally identical at every step.")
                verdict = "NOT CONFIRMED (I+ via relabeling-invariant embedding)"
            elif not phi_pos:
                print(f"  -> Not Φ-positive. Cannot test as counterexample.")
                verdict = "NOT CONFIRMED (not Φ+)"
            else:
                print(f"  -> Unexpected: Φ-negative and I-negative.")
                verdict = "NOT CONFIRMED (unexpected)"

        print("═" * 70)

    # ── Save results ──────────────────────────────────────────────
    save_data = {
        "experiment": "exp12_dehn_twist",
        "grid_size": M,
        "n_nodes": M * M,
        "T": T,
        "Phi_positive": phi_pos,
        "Phi_detail": {k: _safe(v) for k, v in phi_detail.items()},
        "I_positive": i_pos,
        "I_detail": {k: _safe(v) for k, v in i_detail.items()},
        "compression_ratio": comp_ratio,
        "eigenvalue_crossings": n_crossings,
        "eigenvalue_spectrum_diff": eig_diff,
        "per_embedding": {
            name: {
                "tau_to_final": _safe(er["tau_to_final"]),
                "cosine_min": er["cosine_min"],
                "cosine_max": er["cosine_max"],
                "cosine_range": er["cosine_range"],
            }
            for name, er in embed_results.items()
        },
        "is_counterexample": is_counterexample,
        "verdict": verdict,
    }

    with open(os.path.join(results_dir, "dehn_twist.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=_json_default)

    print(f"\nResults saved to {results_dir}/")
    runner.finish(f"exp12 — Dehn twist: {verdict}")


def _safe(v):
    """Make a value JSON-safe."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    return v


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


if __name__ == "__main__":
    runner = StepRunner(
        "exp12_dehn_twist",
        total_rules=1,
        total_seeds=6,  # Phi, I, 3 embeddings, eigenvalues
        phases=[
            "Generating Dehn-twist trajectory",
            "Classifying Φ-predicate",
            "Classifying I-predicate",
            "Per-embedding analysis",
            "Eigenvalue analysis",
            "Summary",
        ],
    )
    runner.run(run_exp12)
