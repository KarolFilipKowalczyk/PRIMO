"""
exp10 — Perturbation-response test for claim (b).

PRIMO claim (b): physics-like behavior is the equilibrium of inference-like
dynamics. Instead of catching fast natural transients (exp06), we:
  1. Let a Φ+ rule run to geometric equilibrium (T=30 burn-in)
  2. Perturb the graph by randomly adding/removing edges (5%, 15%, 30%)
  3. Continue running the same rule (T=30 recovery) and measure whether
     the return to equilibrium follows an I-positive path

If claim (b) holds, recovery trajectories should show elevated I-scores
(the system "re-infers" its way back to the geometric state), with
dose-dependent response (larger perturbation → stronger I-positive signal).

Controls:
  1. Null recovery: random graphs after perturbation (should NOT be I+)
  2. Unperturbed continuation: no perturbation (should NOT show elevated I)

Usage:
    python experiments/exp10_perturbation_response.py
    # or: make exp10
"""

import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, MASTER_SEED, MAJORITY_THRESHOLD,
    DS_STD_STAR, DS_INT_DIST, LAW_RESIDUAL_STAR, CURVATURE_KAPPA_STAR,
    DATA_DIR, N_MAX, EMBEDDING_DIM,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog,
    dpo_rule_to_callable,
)
from primo.predicates import (
    classify_I, classify_Phi, measure_I_scores, measure_Phi_scores,
)
from primo.trajectories import (
    EMBEDDING_FUNCTIONS, compression_ratio,
)
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

T_BURNIN = 30         # steps to reach equilibrium
T_RECOVERY = 30       # steps after perturbation
WINDOW_SIZE = 8       # sliding window for I-scores
N_SEEDS = 4

PERTURBATION_LEVELS = [
    ("5%", 0.05),
    ("15%", 0.15),
    ("30%", 0.30),
]

SIGNATURES = [
    (1, 1),
    (1, 2),
    (2, 3),
    (3, 4),
]


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def find_phi_positive_rules():
    """Identify all Φ+ DPO rules across signatures."""
    seeds = make_initial_graphs()
    phi_pos = []

    for l, r in SIGNATURES:
        sig_label = f"{l}->{r}"
        catalog = load_catalog(f"{l}_{r}")
        rules = catalog["rules"]

        for rule_dict in rules:
            rule_fn = dpo_rule_to_callable(rule_dict)
            phi_count = 0
            for seed_name, G0 in seeds.items():
                np.random.seed(MASTER_SEED)
                traj = run_trajectory(rule_fn, G0, T_DEFAULT)
                is_phi, _ = classify_Phi(traj)
                if is_phi:
                    phi_count += 1

            if phi_count >= MAJORITY_THRESHOLD:
                phi_pos.append({
                    "rule_dict": rule_dict,
                    "rule_name": rule_dict.get("name", rule_dict["id"]),
                    "signature": sig_label,
                    "phi_count": phi_count,
                })

    return phi_pos


def perturb_graph(G, fraction, seed):
    """Perturb a graph by removing and adding `fraction` of its edges.

    Preserves connectivity: skips removals that would disconnect.
    """
    G_pert = G.copy()
    rng = np.random.RandomState(seed)
    edges = list(G_pert.edges())
    n_perturb = max(1, int(len(edges) * fraction))

    # Remove random edges (skip if would disconnect)
    remove_candidates = rng.permutation(len(edges))
    removed = 0
    for i in remove_candidates:
        if removed >= n_perturb:
            break
        u, v = edges[i]
        G_pert.remove_edge(u, v)
        if not nx.is_connected(G_pert):
            G_pert.add_edge(u, v)  # undo
        else:
            removed += 1

    # Add random edges (non-existing)
    nodes = list(G_pert.nodes())
    added = 0
    attempts = 0
    while added < n_perturb and attempts < n_perturb * 10:
        u, v = rng.choice(nodes, size=2, replace=False)
        if not G_pert.has_edge(u, v):
            G_pert.add_edge(u, v)
            added += 1
        attempts += 1

    return G_pert


def random_recovery_trajectory(G_start, T, seed):
    """Generate a null-model trajectory: ER random graphs matching G_start's
    size and density at each step."""
    n = G_start.number_of_nodes()
    m = G_start.number_of_edges()
    max_edges = n * (n - 1) / 2
    p = m / max_edges if max_edges > 0 else 0.3
    rng = np.random.RandomState(seed)

    traj = [G_start.copy()]
    for _ in range(T):
        G = nx.erdos_renyi_graph(n, p, seed=rng.randint(2**31))
        traj.append(G)
    return traj


def compute_window_I_mean(subtraj):
    """Mean I-score (tau_to_final) across embeddings for a sub-trajectory."""
    vals = []
    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
        result = measure_I_scores(subtraj, emb_fn)
        tau = result["tau_to_final"]
        if not np.isnan(tau):
            vals.append(tau)
    return float(np.mean(vals)) if vals else 0.0


def compute_sliding_I_profile(traj, window=WINDOW_SIZE):
    """Sliding-window I-score profile over a trajectory."""
    if len(traj) < window + 2:
        return []
    profile = []
    for t in range(len(traj) - window):
        subtraj = traj[t:t + window]
        profile.append({"t": t, "I_mean": compute_window_I_mean(subtraj)})
    return profile


def find_phi_recovery_step(rule_fn, traj):
    """Find the first step t where the suffix trajectory [t:] is Φ-positive.

    Returns t_phi_recovery or None if never recovers.
    """
    # Check from each starting point whether the rest is Φ+
    min_len = 10  # need at least 10 steps for meaningful Φ measurement
    for t in range(len(traj)):
        suffix = traj[t:]
        if len(suffix) < min_len:
            break
        is_phi, _ = classify_Phi(suffix)
        if is_phi:
            return t
    return None


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp10(runner):
    """Perturbation-response test for claim (b)."""
    results_dir = os.path.join(DATA_DIR, "exp10")
    os.makedirs(results_dir, exist_ok=True)

    seeds = make_initial_graphs()

    # ── Part 0: Identify Φ+ rules ────────────────────────────────
    with runner.phase("Identifying Φ-positive rules"):
        phi_pos_rules = find_phi_positive_rules()
        runner.log(f"  Found {len(phi_pos_rules)} Φ+ rules")
        for entry in phi_pos_rules:
            runner.log(f"    {entry['signature']}/{entry['rule_name']}")

    if runner.should_stop():
        return

    # Estimate total work: rules × seeds × (3 pert levels + 2 controls)
    n_pairs = len(phi_pos_rules) * N_SEEDS
    total_work = n_pairs * 5  # 3 perturbation + 2 controls
    runner.set_total(len(phi_pos_rules) * 5, N_SEEDS)

    # ── Part 1: Burn-in and perturbation-response ─────────────────
    all_results = []  # flat list of per-(rule, seed, perturbation) results
    control_results = []  # controls

    with runner.phase("Burn-in + perturbation-response"):
        for entry in phi_pos_rules:
            if runner.should_stop():
                return

            rule_name = entry["rule_name"]
            sig_label = entry["signature"]
            rule_fn = dpo_rule_to_callable(entry["rule_dict"])

            for seed_name, G0 in seeds.items():
                if runner.should_stop():
                    return

                # Burn-in: run to equilibrium
                np.random.seed(MASTER_SEED)
                burnin_traj = run_trajectory(rule_fn, G0, T_BURNIN, n_max=N_MAX)

                # Verify Φ+ at equilibrium (last 10 steps)
                if len(burnin_traj) < 11:
                    continue
                eq_suffix = burnin_traj[-10:]
                eq_phi, eq_phi_detail = classify_Phi(eq_suffix)
                if not eq_phi:
                    # Not at Φ equilibrium, skip
                    continue

                G_eq = burnin_traj[-1]

                # Baseline I-scores at equilibrium
                eq_I_pos, eq_I_detail = classify_I(eq_suffix)
                baseline_I_mean = compute_window_I_mean(eq_suffix)

                # ── Perturbation levels ───────────────────────────
                for pert_label, pert_frac in PERTURBATION_LEVELS:
                    if runner.should_stop():
                        return

                    label = f"{sig_label}/{rule_name}/{pert_label}"
                    runner.begin_rule(label)

                    pert_seed = MASTER_SEED + int(pert_frac * 100)
                    G_pert = perturb_graph(G_eq, pert_frac, pert_seed)

                    # Recovery trajectory
                    np.random.seed(MASTER_SEED)
                    recovery_traj = run_trajectory(
                        rule_fn, G_pert, T_RECOVERY, n_max=N_MAX)

                    # Measurements
                    rec_I_pos, rec_I_detail = classify_I(recovery_traj)

                    # Per-embedding tau_to_final
                    rec_tau = {}
                    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                        scores = measure_I_scores(recovery_traj, emb_fn)
                        rec_tau[emb_name] = scores["tau_to_final"]
                    tau_vals = [v for v in rec_tau.values()
                                if not np.isnan(v)]
                    rec_tau_mean = float(np.mean(tau_vals)) if tau_vals else 0.0

                    # Sliding-window I-profile on recovery
                    rec_profile = compute_sliding_I_profile(recovery_traj)
                    if rec_profile:
                        third = len(rec_profile) // 3
                        early_I = float(np.mean(
                            [p["I_mean"] for p in rec_profile[:max(third, 1)]]))
                    else:
                        early_I = 0.0

                    elevated = early_I > baseline_I_mean + 0.05

                    # Φ-recovery step
                    t_phi_rec = find_phi_recovery_step(rule_fn, recovery_traj)

                    # Compression ratio
                    comp_ratio = compression_ratio(recovery_traj)

                    result = {
                        "rule_name": rule_name,
                        "signature": sig_label,
                        "seed": seed_name,
                        "perturbation": pert_label,
                        "pert_fraction": pert_frac,
                        "I_positive_recovery": rec_I_pos,
                        "tau_to_final_mean": rec_tau_mean,
                        "tau_per_embed": rec_tau,
                        "early_I": early_I,
                        "baseline_I": baseline_I_mean,
                        "elevated": elevated,
                        "t_phi_recovery": t_phi_rec,
                        "compression_ratio": comp_ratio,
                    }
                    all_results.append(result)

                    runner.tick(label, seed_name,
                               result=f"I+={rec_I_pos} elev={elevated}")
                    runner.finish_rule(
                        label,
                        classification=f"I+={'Y' if rec_I_pos else 'N'} "
                                       f"elev={'Y' if elevated else 'N'}")

                # ── Controls (medium perturbation only) ───────────
                if runner.should_stop():
                    return

                # Control 1: Null (random) recovery
                ctrl_label = f"{sig_label}/{rule_name}/ctrl_null"
                runner.begin_rule(ctrl_label)

                G_pert_med = perturb_graph(G_eq, 0.15, MASTER_SEED + 15)
                null_traj = random_recovery_trajectory(
                    G_pert_med, T_RECOVERY, MASTER_SEED + 999)
                null_I_pos, _ = classify_I(null_traj)
                null_tau = {}
                for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                    scores = measure_I_scores(null_traj, emb_fn)
                    null_tau[emb_name] = scores["tau_to_final"]
                null_tau_vals = [v for v in null_tau.values()
                                if not np.isnan(v)]
                null_tau_mean = float(np.mean(null_tau_vals)) if null_tau_vals else 0.0

                control_results.append({
                    "rule_name": rule_name,
                    "seed": seed_name,
                    "control": "null_random",
                    "I_positive": null_I_pos,
                    "tau_mean": null_tau_mean,
                })

                runner.tick(ctrl_label, seed_name,
                            result=f"null I+={null_I_pos}")
                runner.finish_rule(ctrl_label,
                                   classification=f"null I+={'Y' if null_I_pos else 'N'}")

                # Control 2: Unperturbed continuation
                ctrl_label2 = f"{sig_label}/{rule_name}/ctrl_unpert"
                runner.begin_rule(ctrl_label2)

                np.random.seed(MASTER_SEED)
                unpert_traj = run_trajectory(
                    rule_fn, G_eq, T_RECOVERY, n_max=N_MAX)
                unpert_profile = compute_sliding_I_profile(unpert_traj)
                if unpert_profile:
                    third = len(unpert_profile) // 3
                    unpert_early_I = float(np.mean(
                        [p["I_mean"] for p in unpert_profile[:max(third, 1)]]))
                else:
                    unpert_early_I = 0.0
                unpert_elevated = unpert_early_I > baseline_I_mean + 0.05

                control_results.append({
                    "rule_name": rule_name,
                    "seed": seed_name,
                    "control": "unperturbed",
                    "I_positive": None,  # not meaningful
                    "early_I": unpert_early_I,
                    "baseline_I": baseline_I_mean,
                    "elevated": unpert_elevated,
                })

                runner.tick(ctrl_label2, seed_name,
                            result=f"unpert elev={unpert_elevated}")
                runner.finish_rule(ctrl_label2,
                                   classification=f"unpert elev={'Y' if unpert_elevated else 'N'}")

    if runner.should_stop():
        return

    # ── Part 2: Analysis ──────────────────────────────────────────
    with runner.phase("Analyzing perturbation-response"):
        print()
        print("═" * 100)
        print("PERTURBATION-RESPONSE TEST: CLAIM (b) EVIDENCE")
        print("═" * 100)
        print()

        header = (f"{'Rule':<30} {'Seed':<5} {'Pert':>5} {'I+ rec?':>8} "
                  f"{'τ_final':>8} {'I_early':>8} {'I_base':>8} "
                  f"{'Elevated?':>10} {'t_Φ_rec':>8}")
        print(header)
        print("─" * len(header))

        for r in all_results:
            i_str = "YES" if r["I_positive_recovery"] else "no"
            elev_str = "YES" if r["elevated"] else "no"
            t_phi = str(r["t_phi_recovery"]) if r["t_phi_recovery"] is not None else "never"
            print(f"{r['rule_name']:<30} {r['seed']:<5} {r['perturbation']:>5} "
                  f"{i_str:>8} {r['tau_to_final_mean']:>8.3f} "
                  f"{r['early_I']:>8.3f} {r['baseline_I']:>8.3f} "
                  f"{elev_str:>10} {t_phi:>8}")

        # ── Aggregate by perturbation level ───────────────────────
        print()
        print("AGGREGATE BY PERTURBATION LEVEL:")
        for pert_label, _ in PERTURBATION_LEVELS:
            subset = [r for r in all_results if r["perturbation"] == pert_label]
            n_total = len(subset)
            n_i_pos = sum(1 for r in subset if r["I_positive_recovery"])
            n_elev = sum(1 for r in subset if r["elevated"])
            mean_tau = float(np.mean([r["tau_to_final_mean"] for r in subset])) if subset else 0
            print(f"  {pert_label:>4}:  I+ recovery in {n_i_pos}/{n_total} pairs, "
                  f"elevated I in {n_elev}/{n_total} pairs, "
                  f"mean τ={mean_tau:.3f}")

        # ── Controls ──────────────────────────────────────────────
        print()
        print("CONTROLS:")
        null_ctrls = [c for c in control_results if c["control"] == "null_random"]
        n_null_i_pos = sum(1 for c in null_ctrls if c["I_positive"])
        print(f"  Null (random) recovery:     I+ in {n_null_i_pos}/{len(null_ctrls)} "
              f"(should be ~0)")

        unpert_ctrls = [c for c in control_results if c["control"] == "unperturbed"]
        n_unpert_elev = sum(1 for c in unpert_ctrls if c["elevated"])
        print(f"  Unperturbed continuation:   Elevated I in {n_unpert_elev}/"
              f"{len(unpert_ctrls)} (should be ~0)")

        # ── Dose-response ─────────────────────────────────────────
        print()
        print("DOSE-RESPONSE:")
        mean_taus = {}
        mean_elevs = {}
        for pert_label, _ in PERTURBATION_LEVELS:
            subset = [r for r in all_results if r["perturbation"] == pert_label]
            mean_taus[pert_label] = float(np.mean(
                [r["tau_to_final_mean"] for r in subset])) if subset else 0
            mean_elevs[pert_label] = (
                sum(1 for r in subset if r["elevated"]) / len(subset)
                if subset else 0)

        tau_values = [mean_taus[pl] for pl, _ in PERTURBATION_LEVELS]
        elev_values = [mean_elevs[pl] for pl, _ in PERTURBATION_LEVELS]

        tau_str = ", ".join(
            f"{pl}={mean_taus[pl]:.3f}" for pl, _ in PERTURBATION_LEVELS)
        print(f"  Mean τ_to_final: {tau_str}")

        elev_str = ", ".join(
            f"{pl}={mean_elevs[pl]:.1%}" for pl, _ in PERTURBATION_LEVELS)
        print(f"  Fraction elevated: {elev_str}")

        # Check for positive dose-response (monotone increase)
        tau_increasing = all(
            tau_values[i] <= tau_values[i + 1]
            for i in range(len(tau_values) - 1))
        elev_increasing = all(
            elev_values[i] <= elev_values[i + 1]
            for i in range(len(elev_values) - 1))
        dose_response = tau_increasing or elev_increasing
        print(f"  Dose-response (larger pert → stronger I+): "
              f"{'YES' if dose_response else 'NO'} "
              f"(τ monotone: {'Y' if tau_increasing else 'N'}, "
              f"elev monotone: {'Y' if elev_increasing else 'N'})")

        # ── Verdict ───────────────────────────────────────────────
        print()
        print("CLAIM (b) VERDICT:")

        # Majority of Φ+ rules show I+ recovery at medium perturbation
        med_subset = [r for r in all_results if r["perturbation"] == "15%"]
        # Group by rule: a rule "shows I+ recovery" if ≥3/4 seeds have I+ recovery
        rule_i_pos_count = {}
        rule_elev_count = {}
        for r in med_subset:
            rn = r["rule_name"]
            rule_i_pos_count.setdefault(rn, 0)
            rule_elev_count.setdefault(rn, 0)
            if r["I_positive_recovery"]:
                rule_i_pos_count[rn] += 1
            if r["elevated"]:
                rule_elev_count[rn] += 1

        n_rules_i_pos = sum(1 for v in rule_i_pos_count.values() if v >= 3)
        n_rules_elev = sum(1 for v in rule_elev_count.values() if v >= 3)
        n_rules = len(rule_i_pos_count)

        controls_clean = (n_null_i_pos <= 2 and n_unpert_elev <= 2)

        if (n_rules_i_pos > n_rules / 2 and dose_response and controls_clean):
            verdict = "SUPPORTED"
            print(f"  SUPPORTED: Majority of Φ+ rules ({n_rules_i_pos}/{n_rules}) "
                  f"return to equilibrium via I-positive paths with "
                  f"dose-dependent response and clean controls.")
        elif n_rules_i_pos > 0 or n_rules_elev > 0:
            verdict = "PARTIAL"
            print(f"  PARTIAL: Some rules show I-positive recovery "
                  f"(I+: {n_rules_i_pos}/{n_rules}, elevated: {n_rules_elev}/{n_rules}) "
                  f"but not majority or missing dose-response/controls.")
        else:
            verdict = "NOT SUPPORTED"
            print(f"  NOT SUPPORTED: Recovery trajectories are not "
                  f"consistently I-positive.")

    # ── Save results ──────────────────────────────────────────────
    save_data = {
        "experiment": "exp10",
        "T_burnin": T_BURNIN,
        "T_recovery": T_RECOVERY,
        "window_size": WINDOW_SIZE,
        "perturbation_levels": [pl for pl, _ in PERTURBATION_LEVELS],
        "verdict": verdict,
        "n_phi_pos_rules": len(phi_pos_rules),
        "dose_response": dose_response,
        "controls_clean": controls_clean,
        "aggregate": {
            pl: {
                "n_I_pos": sum(1 for r in all_results
                               if r["perturbation"] == pl
                               and r["I_positive_recovery"]),
                "n_elevated": sum(1 for r in all_results
                                  if r["perturbation"] == pl
                                  and r["elevated"]),
                "n_total": sum(1 for r in all_results
                               if r["perturbation"] == pl),
                "mean_tau": float(np.mean([
                    r["tau_to_final_mean"] for r in all_results
                    if r["perturbation"] == pl
                ])) if any(r["perturbation"] == pl for r in all_results) else 0,
            }
            for pl, _ in PERTURBATION_LEVELS
        },
        "per_trial": all_results,
        "controls": control_results,
    }

    with open(os.path.join(results_dir, "perturbation_response.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=_json_default)

    print(f"\n  Results saved to {results_dir}/")
    runner.finish(f"exp10 complete — claim(b): {verdict}")


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
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = StepRunner(
        "exp10_perturbation_response",
        total_rules=9 * 5,  # 9 rules × (3 pert + 2 controls)
        total_seeds=N_SEEDS,
        phases=[
            "Identify Φ+ rules",
            "Burn-in + perturbation-response",
            "Analysis",
        ],
    )
    runner.run(run_exp10)
