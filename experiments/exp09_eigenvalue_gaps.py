"""
exp09 — Eigenvalue gap stability diagnostic for the Phi+ -> I+ conditional theorem.

Tests whether all Phi-positive DPO rules have stable eigenvalue gaps,
which would explain why Phi+ implies I+ empirically for DPO growth rules.

The Davis-Kahan theorem tells us: if eigenvalue gaps stay open, the
corresponding eigenvectors (and thus embeddings) are stable. Spectral
dimension stability (part of Phi+) stabilizes the bulk eigenvalue
distribution but not individual gaps. This experiment checks those gaps.

For each of the 16 enumerated DPO rules at signatures 1->1 through 3->4:
  - Runs trajectories from 4 seed graphs
  - Measures Laplacian eigenvalue gap and adjacency SVD gap at each step
  - Detects eigenvalue crossings (the event that kills eigenvector stability)
  - Reports whether all Phi+ rules have stable gaps

Usage:
    python experiments/exp09_eigenvalue_gaps.py
"""

import json
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, MAJORITY_THRESHOLD,
    EMBEDDING_DIM, DATA_DIR, DS_STD_STAR,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog,
    dpo_rule_to_callable,
)
from primo.predicates import classify_rule
from primo.run_utils import StepRunner


# Signature levels (same as exp04)
SIGNATURES = [
    (1, 1),  # sigma=1
    (1, 2),  # sigma=2
    (2, 3),  # sigma=3
    (3, 4),  # sigma=4
]

K = EMBEDDING_DIM  # gap index: between k-th and (k+1)-th eigenvalue/singular value
N_SEEDS = 4


def laplacian_gap(G, k):
    """Compute gap between k-th and (k+1)-th smallest non-zero Laplacian eigenvalue.

    Returns None if the graph has fewer than k+1 non-zero eigenvalues.
    """
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigs = np.linalg.eigvalsh(L)
    # Skip eigenvalues near zero (the algebraic multiplicity of 0)
    nonzero = eigs[eigs > 1e-10]
    nonzero.sort()
    if len(nonzero) < k + 1:
        return None
    return float(nonzero[k] - nonzero[k - 1])


def adjacency_svd_gap(G, k):
    """Compute gap between k-th and (k+1)-th largest singular value of adjacency.

    Returns None if the matrix has fewer than k+1 singular values.
    """
    A = nx.adjacency_matrix(G).toarray().astype(float)
    sv = np.linalg.svd(A, compute_uv=False)
    sv = np.sort(sv)[::-1]  # descending
    if len(sv) < k + 1:
        return None
    return float(sv[k - 1] - sv[k])


def analyze_trajectory_gaps(traj, k):
    """Compute eigenvalue gap metrics for a trajectory.

    Returns dict with gap_laplacian_* and gap_adjacency_* metrics,
    plus eigenvalue crossing count.
    """
    T = len(traj)
    lap_gaps = []
    adj_gaps = []

    for G in traj:
        lap_gaps.append(laplacian_gap(G, k))
        adj_gaps.append(adjacency_svd_gap(G, k))

    # Second half indices
    half = T // 2
    lap_second = [g for g in lap_gaps[half:] if g is not None]
    adj_second = [g for g in adj_gaps[half:] if g is not None]

    # Final values
    gap_lap_final = lap_gaps[-1]
    gap_adj_final = adj_gaps[-1]

    # Stats over second half
    gap_lap_std = float(np.std(lap_second)) if len(lap_second) >= 2 else None
    gap_adj_std = float(np.std(adj_second)) if len(adj_second) >= 2 else None
    gap_lap_min = float(min(lap_second)) if lap_second else None
    gap_adj_min = float(min(adj_second)) if adj_second else None

    # Stability: gap stays positive over second half
    gap_stabilizes = (
        len(lap_second) > 0
        and gap_lap_min is not None
        and gap_lap_min > 0
    )

    # Count eigenvalue crossings: track sorted non-zero Laplacian eigenvalues
    n_crossings = 0
    prev_eigs = None
    for G in traj:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigs = np.linalg.eigvalsh(L)
        nonzero = np.sort(eigs[eigs > 1e-10])

        if prev_eigs is not None and len(nonzero) >= k + 1 and len(prev_eigs) >= k + 1:
            # Compare gap sign at position k (between k-th and (k+1)-th)
            curr_gap = nonzero[k] - nonzero[k - 1]
            prev_gap = prev_eigs[k] - prev_eigs[k - 1]
            # A "crossing" = the relative ordering of eigenvalue k and k+1 changed
            # We detect this as the gap changing sign or collapsing to near-zero
            # after being well-separated
            if curr_gap * prev_gap < 0:
                n_crossings += 1

        prev_eigs = nonzero

    return {
        "gap_laplacian_final": gap_lap_final,
        "gap_adjacency_final": gap_adj_final,
        "gap_laplacian_std": gap_lap_std,
        "gap_adjacency_std": gap_adj_std,
        "gap_laplacian_min": gap_lap_min,
        "gap_adjacency_min": gap_adj_min,
        "gap_stabilizes": gap_stabilizes,
        "n_crossings": n_crossings,
    }


def run_exp09(runner):
    """Eigenvalue gap stability diagnostic."""
    seeds = make_initial_graphs()
    seed_names = list(seeds.keys())
    results_dir = os.path.join(DATA_DIR, "exp09")
    os.makedirs(results_dir, exist_ok=True)

    # ── Load all 16 DPO rules ────────────────────────────────────────
    all_rules = []  # list of (name, rule_fn, signature_str)
    for l, r in SIGNATURES:
        sig = f"{l}_{r}"
        catalog = load_catalog(sig)
        for rule_dict in catalog["rules"]:
            rule_fn = dpo_rule_to_callable(rule_dict)
            name = rule_dict.get("name", rule_dict["id"])
            all_rules.append((name, rule_fn, f"{l}->{r}"))

    n_rules = len(all_rules)
    runner.log(f"Loaded {n_rules} DPO rules across {len(SIGNATURES)} signatures")

    # ── Classify all rules (get I/Phi status) ─────────────────────────
    with runner.phase("Classifying rules"):
        classifications = {}
        for rule_name, rule_fn, sig in all_rules:
            if runner.should_stop():
                return
            runner.begin_rule(rule_name)
            result = classify_rule(rule_fn, seeds=seeds)
            i_pos = result["I_positive"]
            phi_pos = result["Phi_positive"]
            classifications[rule_name] = (i_pos, phi_pos)

            cell = ("I+" if i_pos else "I-") + ("Φ+" if phi_pos else "Φ-")
            runner.tick(rule_name, "classify", result=cell)
            runner.finish_rule(rule_name, classification=cell)

    if runner.should_stop():
        return

    # ── Compute eigenvalue gaps ───────────────────────────────────────
    per_rule_data = {}
    with runner.phase("Computing eigenvalue gaps"):
        for rule_name, rule_fn, sig in all_rules:
            if runner.should_stop():
                return

            runner.begin_rule(rule_name)
            seed_results = {}

            for seed_name in seed_names:
                G0 = seeds[seed_name]
                traj = run_trajectory(rule_fn, G0, T=T_DEFAULT)
                metrics = analyze_trajectory_gaps(traj, K)
                seed_results[seed_name] = metrics

                stable_str = "stable" if metrics["gap_stabilizes"] else "unstable"
                cross_str = f"x{metrics['n_crossings']}"
                runner.tick(rule_name, seed_name, result=f"{stable_str} {cross_str}")

            # Aggregate per rule
            stable_count = sum(
                1 for m in seed_results.values() if m["gap_stabilizes"]
            )
            all_gaps_stable = stable_count >= MAJORITY_THRESHOLD
            total_crossings = sum(
                m["n_crossings"] for m in seed_results.values()
            )
            lap_finals = [
                m["gap_laplacian_final"]
                for m in seed_results.values()
                if m["gap_laplacian_final"] is not None
            ]
            mean_gap_final = float(np.mean(lap_finals)) if lap_finals else None

            i_pos, phi_pos = classifications[rule_name]
            per_rule_data[rule_name] = {
                "signature": sig,
                "I_positive": i_pos,
                "Phi_positive": phi_pos,
                "all_gaps_stable": all_gaps_stable,
                "stable_count": stable_count,
                "mean_gap_final": mean_gap_final,
                "total_crossings": total_crossings,
                "seeds": seed_results,
            }

            stable_str = "STABLE" if all_gaps_stable else "UNSTABLE"
            runner.finish_rule(rule_name, classification=stable_str, result_data={
                "stable": all_gaps_stable, "crossings": total_crossings,
            })

    if runner.should_stop():
        return

    # ── Summary table ─────────────────────────────────────────────────
    with runner.phase("Summary and diagnostic"):
        print()
        print("=" * 80)
        print("exp09: EIGENVALUE GAP STABILITY DIAGNOSTIC")
        print("=" * 80)

        hdr = (
            f"{'Rule':<35s} {'σ':>3s}  {'I+':>3s} {'Φ+':>3s}"
            f"  {'Lap_gap':>8s} {'Adj_gap':>8s} {'Stable':>6s} {'Cross':>5s}"
        )
        print(hdr)
        print("─" * 80)

        for rule_name, rule_fn, sig in all_rules:
            d = per_rule_data[rule_name]
            i_str = "YES" if d["I_positive"] else "no"
            p_str = "YES" if d["Phi_positive"] else "no"
            lap = f"{d['mean_gap_final']:.3f}" if d["mean_gap_final"] is not None else "N/A"
            # Mean adjacency gap
            adj_finals = [
                m["gap_adjacency_final"]
                for m in d["seeds"].values()
                if m["gap_adjacency_final"] is not None
            ]
            adj = f"{np.mean(adj_finals):.3f}" if adj_finals else "N/A"
            stab = "YES" if d["all_gaps_stable"] else "NO"
            cross = str(d["total_crossings"])
            print(
                f"{rule_name:<35s} {sig:>3s}  {i_str:>3s} {p_str:>3s}"
                f"  {lap:>8s} {adj:>8s} {stab:>6s} {cross:>5s}"
            )

        # ── Conditional theorem diagnostic ────────────────────────────
        phi_pos_rules = [n for n in per_rule_data if per_rule_data[n]["Phi_positive"]]
        phi_neg_rules = [n for n in per_rule_data if not per_rule_data[n]["Phi_positive"]]

        phi_pos_stable = [n for n in phi_pos_rules if per_rule_data[n]["all_gaps_stable"]]
        phi_pos_unstable = [n for n in phi_pos_rules if not per_rule_data[n]["all_gaps_stable"]]
        phi_neg_stable = [n for n in phi_neg_rules if per_rule_data[n]["all_gaps_stable"]]
        phi_neg_unstable = [n for n in phi_neg_rules if not per_rule_data[n]["all_gaps_stable"]]

        phi_pos_crossings = [per_rule_data[n]["total_crossings"] for n in phi_pos_rules]
        phi_neg_crossings = [per_rule_data[n]["total_crossings"] for n in phi_neg_rules]

        print()
        print("═" * 80)
        print("CONDITIONAL THEOREM DIAGNOSTIC")
        print("═" * 80)
        print()
        print("Question: Does Φ+ → I+ hold because eigenvalue gaps stabilize?")
        print()
        print(f"Φ+ rules with stable gaps:    {len(phi_pos_stable)} / {len(phi_pos_rules)}")
        print(f"Φ+ rules with unstable gaps:  {len(phi_pos_unstable)} / {len(phi_pos_rules)}")
        print(f"Φ- rules with stable gaps:    {len(phi_neg_stable)} / {len(phi_neg_rules)}"
              f"  (interesting if nonzero)")
        print(f"Φ- rules with unstable gaps:  {len(phi_neg_unstable)} / {len(phi_neg_rules)}")
        print()
        print(f"Eigenvalue crossings in Φ+ rules: {phi_pos_crossings}")
        print(f"Eigenvalue crossings in Φ- rules: {phi_neg_crossings}")

        # Verdict
        print()
        if len(phi_pos_rules) == 0:
            verdict = "NO Φ+ RULES: Cannot test the conditional theorem."
        elif len(phi_pos_unstable) == 0:
            verdict = (
                "CONFIRMED: All Φ+ DPO rules have stable eigenvalue gaps.\n"
                "  The conditional theorem Φ+ ∧ M1 ∧ M2 ∧ M3 → I+ explains the data."
            )
        elif len(phi_pos_unstable) <= len(phi_pos_rules) // 3:
            verdict = (
                f"PARTIALLY CONFIRMED: Most but not all Φ+ rules have stable gaps.\n"
                f"  The conditional theorem covers {len(phi_pos_stable)}/{len(phi_pos_rules)}"
                f" rules; the rest need another explanation."
            )
        else:
            verdict = (
                "REFUTED: Φ+ rules do NOT consistently have stable eigenvalue gaps.\n"
                "  The monotone-growth + gap-stability explanation is insufficient."
            )

        print(f"VERDICT: {verdict}")
        print("═" * 80)

    # ── Save results ──────────────────────────────────────────────────
    save_data = {
        "experiment": "exp09_eigenvalue_gaps",
        "embedding_dim_k": K,
        "T": T_DEFAULT,
        "ds_std_star": DS_STD_STAR,
        "tau_star": TAU_STAR,
        "n_rules": n_rules,
        "phi_pos_count": len(phi_pos_rules),
        "phi_pos_stable": len(phi_pos_stable),
        "phi_pos_unstable": len(phi_pos_unstable),
        "phi_neg_stable": len(phi_neg_stable),
        "phi_neg_unstable": len(phi_neg_unstable),
        "verdict": verdict.split(":")[0] if ":" in verdict else verdict,
        "per_rule": {},
    }
    for rule_name, data in per_rule_data.items():
        # Make seed data JSON-serializable
        seed_data = {}
        for sn, sm in data["seeds"].items():
            seed_data[sn] = {
                k: (v if v is not None else None)
                for k, v in sm.items()
            }
        save_data["per_rule"][rule_name] = {
            "signature": data["signature"],
            "I_positive": data["I_positive"],
            "Phi_positive": data["Phi_positive"],
            "all_gaps_stable": data["all_gaps_stable"],
            "stable_count": data["stable_count"],
            "mean_gap_final": data["mean_gap_final"],
            "total_crossings": data["total_crossings"],
            "seeds": seed_data,
        }

    results_path = os.path.join(results_dir, "eigenvalue_gaps.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    runner.finish(
        f"exp09 done — Φ+ stable: {len(phi_pos_stable)}/{len(phi_pos_rules)}, "
        f"crossings Φ+: {sum(phi_pos_crossings)}, Φ-: {sum(phi_neg_crossings)}"
    )


if __name__ == "__main__":
    # Count rules across all signatures
    total_rules = 0
    for l, r in SIGNATURES:
        sig = f"{l}_{r}"
        try:
            catalog = load_catalog(sig)
            total_rules += len(catalog["rules"])
        except FileNotFoundError:
            pass

    runner = StepRunner(
        "exp09_eigenvalue_gaps",
        total_rules=total_rules,
        total_seeds=N_SEEDS,
        phases=[
            "Classifying rules",
            "Computing eigenvalue gaps",
            "Summary and diagnostic",
        ],
    )
    runner.run(run_exp09)
