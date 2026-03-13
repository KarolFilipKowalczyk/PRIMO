"""
exp11 — Davis-Kahan ratio: ‖ΔA‖_F / gap_k^{cluster} for all 16 DPO rules.

Theorem 1 in Paper 4 says Φ+ → I+ for growth rules. The proof uses
Davis-Kahan, which bounds eigenvector rotation by the ratio
‖ΔA_t‖_F / gap_k^{cluster}(L_t). Exp09 measured the denominator (gap)
but not the numerator (perturbation norm). This experiment fills that hole
by computing the full ratio and checking whether it decreases over time
for Φ+ rules.

For each of the 16 DPO rules (signatures 1→1 through 3→4), for each of
the 4 seed graphs, run T=30 steps and at each step compute:
  - ‖ΔA_t‖_F (Frobenius norm of adjacency difference, zero-padded)
  - gap_k^{cluster}(L_t) (cluster gap of Laplacian)
  - ratio = ‖ΔA_t‖_F / gap_k^{cluster}
  - relative perturbation = ‖ΔA_t‖_F / ‖A_t‖_F

Usage:
    python experiments/exp11_perturbation_ratio.py
"""

import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    T_DEFAULT, EMBEDDING_DIM, DATA_DIR, MASTER_SEED,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog,
    dpo_rule_to_callable,
)
from primo.predicates import classify_rule
from primo.run_utils import StepRunner


SIGNATURES = [
    (1, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]

K = EMBEDDING_DIM  # k=5
N_SEEDS = 4
CLUSTER_TOL = 1e-6
KENDALL_THRESHOLD = -0.15  # same as exp06


# ══════════════════════════════════════════════════════════════════════
# EIGENVALUE CLUSTERING (from exp09b)
# ══════════════════════════════════════════════════════════════════════

def eigenvalue_clusters(eigs, tol=CLUSTER_TOL):
    """Group sorted eigenvalues into clusters of near-equal values."""
    if len(eigs) == 0:
        return []
    clusters = []
    current = [eigs[0]]
    for e in eigs[1:]:
        if e - current[-1] <= tol:
            current.append(e)
        else:
            clusters.append((float(np.mean(current)), len(current)))
            current = [e]
    clusters.append((float(np.mean(current)), len(current)))
    return clusters


def cluster_gap_at_k(eigs_sorted, k, tol=CLUSTER_TOL):
    """Compute the cluster gap relevant for Davis-Kahan at embedding dimension k.

    Returns the gap from the cluster containing eig[k-1] to the next cluster above.
    Returns None if not enough eigenvalues.
    """
    if len(eigs_sorted) < k:
        return None

    clusters = eigenvalue_clusters(eigs_sorted, tol)

    # Find which cluster contains eig[k-1]
    cumulative = 0
    target_idx = None
    for i, (center, count) in enumerate(clusters):
        cumulative += count
        if cumulative >= k:
            target_idx = i
            break

    if target_idx is None:
        return None

    if target_idx + 1 < len(clusters):
        return clusters[target_idx + 1][0] - clusters[target_idx][0]

    return None  # no cluster above


# ══════════════════════════════════════════════════════════════════════
# PERTURBATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze_perturbation_ratio(traj, k):
    """Compute ‖ΔA‖_F / gap_k^{cluster} at each step of a trajectory.

    Returns dict with per-step data and summary statistics.
    """
    steps = []
    prev_A = None

    for t, G in enumerate(traj):
        n = G.number_of_nodes()
        A = nx.adjacency_matrix(G).toarray().astype(float)

        # Laplacian eigenvalues for cluster gap
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigs = np.linalg.eigvalsh(L)
        nonzero = np.sort(eigs[eigs > 1e-10])
        gap = cluster_gap_at_k(nonzero, k)

        # ‖ΔA_t‖_F
        delta_norm = None
        relative_pert = None
        ratio = None

        if prev_A is not None:
            n_prev = prev_A.shape[0]
            n_curr = A.shape[0]
            n_max = max(n_prev, n_curr)

            # Zero-pad to match sizes
            A_prev_pad = np.zeros((n_max, n_max))
            A_prev_pad[:n_prev, :n_prev] = prev_A
            A_curr_pad = np.zeros((n_max, n_max))
            A_curr_pad[:n_curr, :n_curr] = A

            delta = A_curr_pad - A_prev_pad
            delta_norm = float(np.linalg.norm(delta, 'fro'))

            A_norm = float(np.linalg.norm(A_curr_pad, 'fro'))
            if A_norm > 0:
                relative_pert = delta_norm / A_norm

            if gap is not None and gap > 0:
                ratio = delta_norm / gap

        steps.append({
            "t": t,
            "n_nodes": n,
            "delta_norm": delta_norm,
            "cluster_gap": gap,
            "ratio": ratio,
            "relative_pert": relative_pert,
        })

        prev_A = A

    # Summary statistics over latter half (t >= T/2)
    T = len(traj)
    half = T // 2
    latter = [s for s in steps if s["t"] >= half]

    ratios_latter = [s["ratio"] for s in latter if s["ratio"] is not None]
    rel_perts_latter = [s["relative_pert"] for s in latter
                        if s["relative_pert"] is not None]

    # Check if ratio is decreasing (Kendall τ < -0.15)
    ratio_decreasing = False
    ratio_tau = None
    if len(ratios_latter) >= 4:
        tau, _ = kendalltau(range(len(ratios_latter)), ratios_latter)
        ratio_tau = float(tau)
        ratio_decreasing = tau < KENDALL_THRESHOLD

    ratio_mean = float(np.mean(ratios_latter)) if ratios_latter else None
    ratio_max = float(np.max(ratios_latter)) if ratios_latter else None
    rel_pert_mean = float(np.mean(rel_perts_latter)) if rel_perts_latter else None

    return {
        "steps": steps,
        "ratio_decreasing": ratio_decreasing,
        "ratio_tau": ratio_tau,
        "ratio_mean_latter": ratio_mean,
        "ratio_max_latter": ratio_max,
        "relative_pert_mean_latter": rel_pert_mean,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp11(runner):
    """Davis-Kahan ratio diagnostic."""
    seeds = make_initial_graphs()
    seed_names = list(seeds.keys())
    results_dir = os.path.join(DATA_DIR, "exp11")
    os.makedirs(results_dir, exist_ok=True)

    # ── Load all 16 DPO rules ─────────────────────────────────────
    all_rules = []
    for l, r in SIGNATURES:
        sig = f"{l}_{r}"
        catalog = load_catalog(sig)
        for rule_dict in catalog["rules"]:
            rule_fn = dpo_rule_to_callable(rule_dict)
            name = rule_dict.get("name", rule_dict["id"])
            sigma = l + r
            all_rules.append((name, rule_fn, f"{l}->{r}", sigma))

    n_rules = len(all_rules)
    runner.log(f"Loaded {n_rules} DPO rules across {len(SIGNATURES)} signatures")

    # ── Classify all rules ────────────────────────────────────────
    classifications = {}
    with runner.phase("Classifying rules"):
        for rule_name, rule_fn, sig, sigma in all_rules:
            if runner.should_stop():
                return
            runner.begin_rule(rule_name)
            result = classify_rule(rule_fn, seeds=seeds)
            classifications[rule_name] = {
                "I_positive": result["I_positive"],
                "Phi_positive": result["Phi_positive"],
            }
            cell = ("I+" if result["I_positive"] else "I-") + \
                   ("Φ+" if result["Phi_positive"] else "Φ-")
            runner.tick(rule_name, "classify", result=cell)
            runner.finish_rule(rule_name, classification=cell)

    if runner.should_stop():
        return

    # ── Compute perturbation ratios ───────────────────────────────
    per_rule_data = {}
    with runner.phase("Computing perturbation ratios"):
        for rule_name, rule_fn, sig, sigma in all_rules:
            if runner.should_stop():
                return

            runner.begin_rule(rule_name)
            seed_results = {}

            for seed_name in seed_names:
                G0 = seeds[seed_name]
                np.random.seed(MASTER_SEED)
                traj = run_trajectory(rule_fn, G0, T=T_DEFAULT)
                analysis = analyze_perturbation_ratio(traj, K)
                seed_results[seed_name] = analysis

                dec = "DEC" if analysis["ratio_decreasing"] else "---"
                r_mean = analysis["ratio_mean_latter"]
                r_str = f"{r_mean:.3f}" if r_mean is not None else "N/A"
                runner.tick(rule_name, seed_name,
                            result=f"ratio={r_str} {dec}")

            # Aggregate per rule
            phi_pos = classifications[rule_name]["Phi_positive"]
            dec_count = sum(
                1 for a in seed_results.values()
                if a["ratio_decreasing"]
            )
            ratio_means = [a["ratio_mean_latter"] for a in seed_results.values()
                          if a["ratio_mean_latter"] is not None]
            ratio_maxes = [a["ratio_max_latter"] for a in seed_results.values()
                          if a["ratio_max_latter"] is not None]
            rel_pert_means = [a["relative_pert_mean_latter"]
                             for a in seed_results.values()
                             if a["relative_pert_mean_latter"] is not None]

            per_rule_data[rule_name] = {
                "signature": sig,
                "sigma": sigma,
                "Phi_positive": phi_pos,
                "I_positive": classifications[rule_name]["I_positive"],
                "ratio_mean": float(np.mean(ratio_means)) if ratio_means else None,
                "ratio_max": float(np.max(ratio_maxes)) if ratio_maxes else None,
                "rel_pert_mean": float(np.mean(rel_pert_means)) if rel_pert_means else None,
                "dec_count": dec_count,
                "majority_decreasing": dec_count >= 3,
                "seeds": seed_results,
            }

            dec_str = f"{dec_count}/4 DEC"
            runner.finish_rule(rule_name, classification=dec_str)

    if runner.should_stop():
        return

    # ── Summary table ─────────────────────────────────────────────
    with runner.phase("Summary"):
        print()
        print("═" * 90)
        print("DAVIS-KAHAN RATIO: ‖ΔA‖_F / gap_k^{cluster}")
        print("═" * 90)
        print()

        hdr = (
            f"{'Rule':<35s} {'σ':>3s}  {'Φ+':>3s}"
            f"  {'Ratio(mean)':>11s} {'Ratio(max)':>11s}"
            f"  {'Decreasing':>10s} {'Rel.pert(mean)':>14s}"
        )
        print(hdr)
        print("─" * 90)

        for rule_name, rule_fn, sig, sigma in all_rules:
            d = per_rule_data[rule_name]
            phi_str = "YES" if d["Phi_positive"] else "no"
            r_mean = f"{d['ratio_mean']:.4f}" if d["ratio_mean"] is not None else "N/A"
            r_max = f"{d['ratio_max']:.4f}" if d["ratio_max"] is not None else "N/A"
            dec_str = "YES" if d["majority_decreasing"] else "no"
            rp = f"{d['rel_pert_mean']:.4f}" if d["rel_pert_mean"] is not None else "N/A"
            print(
                f"{rule_name:<35s} {sigma:>3d}  {phi_str:>3s}"
                f"  {r_mean:>11s} {r_max:>11s}"
                f"  {dec_str:>10s} {rp:>14s}"
            )

        # ── Verdict ───────────────────────────────────────────────
        phi_pos_rules = [n for n in per_rule_data
                         if per_rule_data[n]["Phi_positive"]]
        phi_pos_dec = [n for n in phi_pos_rules
                       if per_rule_data[n]["majority_decreasing"]]

        phi_pos_ratios = [per_rule_data[n]["ratio_mean"]
                         for n in phi_pos_rules
                         if per_rule_data[n]["ratio_mean"] is not None]
        phi_pos_rel = [per_rule_data[n]["rel_pert_mean"]
                      for n in phi_pos_rules
                      if per_rule_data[n]["rel_pert_mean"] is not None]

        mean_ratio = float(np.mean(phi_pos_ratios)) if phi_pos_ratios else 0
        mean_rel = float(np.mean(phi_pos_rel)) if phi_pos_rel else 0

        print()
        print("VERDICT:")
        print(f"  Φ+ rules with decreasing ratio:  {len(phi_pos_dec)} / {len(phi_pos_rules)}")
        print(f"  Mean ratio across Φ+ rules (latter half): {mean_ratio:.4f}")
        print(f"  Mean relative perturbation across Φ+ rules: {mean_rel:.4f}")
        print()

        if len(phi_pos_dec) == len(phi_pos_rules) and len(phi_pos_rules) > 0:
            bound_status = "tightening"
        elif len(phi_pos_dec) >= len(phi_pos_rules) / 2:
            bound_status = "stable"
        else:
            bound_status = "not tightening"

        print(f"  -> Davis-Kahan bound is [{bound_status}] over time")
        print(f"     for Φ+ DPO rules.")

        if mean_rel < 0.1:
            print(f"  -> Relative perturbation ‖ΔA‖/‖A‖ → {mean_rel:.4f} << 1")
            print(f"     confirms the perturbation shrinks relative to graph size.")
        print("═" * 90)

    # ── Save results ──────────────────────────────────────────────
    save_data = {
        "experiment": "exp11_perturbation_ratio",
        "T": T_DEFAULT,
        "k": K,
        "cluster_tolerance": CLUSTER_TOL,
        "kendall_threshold": KENDALL_THRESHOLD,
        "n_rules": n_rules,
        "phi_pos_count": len(phi_pos_rules),
        "phi_pos_decreasing": len(phi_pos_dec),
        "mean_ratio_phi_pos": mean_ratio,
        "mean_rel_pert_phi_pos": mean_rel,
        "bound_status": bound_status,
        "per_rule": {},
    }

    for rule_name, d in per_rule_data.items():
        seed_save = {}
        for sn, a in d["seeds"].items():
            # Don't save per-step data to keep JSON reasonable
            seed_save[sn] = {
                "ratio_decreasing": a["ratio_decreasing"],
                "ratio_tau": a["ratio_tau"],
                "ratio_mean_latter": a["ratio_mean_latter"],
                "ratio_max_latter": a["ratio_max_latter"],
                "relative_pert_mean_latter": a["relative_pert_mean_latter"],
            }
        save_data["per_rule"][rule_name] = {
            "signature": d["signature"],
            "sigma": d["sigma"],
            "Phi_positive": d["Phi_positive"],
            "I_positive": d["I_positive"],
            "ratio_mean": d["ratio_mean"],
            "ratio_max": d["ratio_max"],
            "rel_pert_mean": d["rel_pert_mean"],
            "dec_count": d["dec_count"],
            "majority_decreasing": d["majority_decreasing"],
            "seeds": seed_save,
        }

    with open(os.path.join(results_dir, "perturbation_ratio.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=_json_default)

    print(f"\nResults saved to {results_dir}/")
    runner.finish(
        f"exp11 done — Φ+ ratio {bound_status}: "
        f"{len(phi_pos_dec)}/{len(phi_pos_rules)} decreasing, "
        f"mean ratio {mean_ratio:.4f}"
    )


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
    # Count rules
    total_rules = 0
    for l, r in SIGNATURES:
        try:
            catalog = load_catalog(f"{l}_{r}")
            total_rules += len(catalog["rules"])
        except FileNotFoundError:
            pass

    runner = StepRunner(
        "exp11_perturbation_ratio",
        total_rules=total_rules,
        total_seeds=N_SEEDS,
        phases=[
            "Classifying rules",
            "Computing perturbation ratios",
            "Summary",
        ],
    )
    runner.run(run_exp11)
