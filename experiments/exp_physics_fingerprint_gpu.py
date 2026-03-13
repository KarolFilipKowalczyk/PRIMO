"""
exp_physics_fingerprint_gpu — GPU-accelerated physics fingerprint at T=200.

Scale-up of exp_physics_fingerprint.py (T=30 → T=200) with:
  - GPU-accelerated eigenvalue decomposition via primo.backend
  - Tighter stabilization criterion (5% range, t < 0.8*T cutoff)
  - Bootstrap confidence intervals on pairwise precedence
  - Comparison with T=30 baseline results

Supports the paper sketch: papers/intermediate_physics.md

Usage:
    python experiments/exp_physics_fingerprint_gpu.py
    # or: make exp-fingerprint-gpu
"""

import json
import os
import sys
import warnings

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    MASTER_SEED, MAJORITY_THRESHOLD, DATA_DIR, N_MAX,
)
from primo.rules import ALL_RULES, make_initial_graphs, run_trajectory, RULE_SOURCE
from primo.predicates import classify_I, classify_Phi
from primo.trajectories import (
    curvature_homogeneity, normalized_degree_entropy, mean_clustering,
    distance_correlation_ratio, edge_vertex_ratio,
    law_residual_score, AGGREGATE_QUANTITIES,
)
from primo.backend import get_backend
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

T = 200
N_SEEDS = 4
PARAM_NAMES = [
    "d_s", "gap_norm", "curv_cv", "law_resid",
    "deg_entropy", "clustering", "dist_corr", "ev_ratio",
]
N_PARAMS = len(PARAM_NAMES)

# Tighter stabilization: 5% of range, must stabilize before 80% of T
STAB_FRAC = 0.05
STAB_CUTOFF = 0.8  # t_stab must be < STAB_CUTOFF * T

# Bootstrap for precedence CIs
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 99

# Intermediate physics profiles (same as T=30 run)
INTERMEDIATE_PHYSICS = {
    "1D_conserved": {
        "d_s": 1.0, "gap_norm": 0.5, "curv_cv": 0.3,
        "law_resid": 0.05, "deg_entropy": 0.3, "clustering": 0.0,
        "dist_corr": 0.0, "ev_ratio": 1.0,
    },
    "2D_flat": {
        "d_s": 2.0, "gap_norm": 0.3, "curv_cv": 0.3,
        "law_resid": 0.05, "deg_entropy": 0.5, "clustering": 0.0,
        "dist_corr": 0.0, "ev_ratio": 2.0,
    },
    "2D_nonlocal": {
        "d_s": 2.0, "gap_norm": 0.3, "curv_cv": 0.3,
        "law_resid": 0.05, "deg_entropy": 0.5, "clustering": 0.1,
        "dist_corr": 0.5, "ev_ratio": 2.0,
    },
    "geometry_no_laws": {
        "d_s": 2.5, "gap_norm": 0.3, "curv_cv": 0.3,
        "law_resid": 0.5, "deg_entropy": 0.5, "clustering": 0.1,
        "dist_corr": 0.0, "ev_ratio": 2.5,
    },
    "topology_no_metric": {
        "d_s": 2.0, "gap_norm": 0.5, "curv_cv": 2.0,
        "law_resid": 0.3, "deg_entropy": 0.5, "clustering": 0.1,
        "dist_corr": 0.0, "ev_ratio": 2.0,
    },
    "laws_no_geometry": {
        "d_s": None,
        "gap_norm": 0.3, "curv_cv": 2.0,
        "law_resid": 0.05, "deg_entropy": 0.5, "clustering": 0.0,
        "dist_corr": 0.0, "ev_ratio": 1.5,
    },
    "broken_symmetry": {
        "d_s": 3.0, "gap_norm": 0.3, "curv_cv": 0.5,
        "law_resid": 0.05, "deg_entropy": 0.6, "clustering": 0.2,
        "dist_corr": 0.1, "ev_ratio": 3.0,
    },
}

MATCH_TOLERANCE = 0.3

# T=30 baseline for comparison
BASELINE_ORDER = [
    ("gap_norm", 2.9), ("law_resid", 5.0), ("clustering", 5.1),
    ("d_s", 5.3), ("curv_cv", 6.8), ("ev_ratio", 7.5),
    ("dist_corr", 10.3), ("deg_entropy", 14.1),
]
BASELINE_H1 = 0.24
BASELINE_H2 = 0.66
BASELINE_H3 = 0.49


# ══════════════════════════════════════════════════════════════════════
# GPU-ACCELERATED SPECTRAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def spectral_dimension_gpu(G, backend):
    """P1: spectral dimension via GPU eigvalsh."""
    if G.number_of_nodes() < 4:
        return None
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    evals = backend.to_numpy(backend.eigvalsh(backend.to_device(L)))
    evals = sorted(evals)
    pos_evals = [e for e in evals if e > 1e-8]
    if len(pos_evals) < 3:
        return None
    N = np.arange(1, len(pos_evals) + 1, dtype=float)
    logN = np.log(N)
    logE = np.log(pos_evals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(logE, logN, 1)
    return float(2.0 * coeffs[0])


def spectral_gap_gpu(G, backend):
    """P2: spectral gap via GPU eigvalsh."""
    if G.number_of_nodes() < 3:
        return 0.0
    L = nx.laplacian_matrix(G).toarray().astype(np.float64)
    evals = backend.to_numpy(backend.eigvalsh(backend.to_device(L)))
    evals = sorted(evals)
    pos = [e for e in evals if e > 1e-8]
    return float(pos[0]) if pos else 0.0


# ══════════════════════════════════════════════════════════════════════
# FINGERPRINT COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_fingerprint(G, traj_window, backend):
    """Compute 8-parameter physics fingerprint for graph G."""
    # P1: spectral dimension
    ds = spectral_dimension_gpu(G, backend)
    if ds is None:
        ds = 0.0

    # P2: normalized spectral gap
    gap = spectral_gap_gpu(G, backend)
    ds_clamp = max(ds, 0.1)
    gap_norm = gap / ds_clamp

    # P3: curvature homogeneity
    curv_cv = curvature_homogeneity(G)
    if curv_cv == float('inf'):
        curv_cv = 10.0

    # P4: law residual (sliding window)
    best_resid = float('inf')
    if len(traj_window) >= 4:
        for fn in AGGREGATE_QUANTITIES.values():
            resid, _ = law_residual_score(traj_window, fn)
            if resid < best_resid:
                best_resid = resid
    if best_resid == float('inf'):
        best_resid = 1.0

    # P5-P8: CPU graph metrics
    deg_ent = normalized_degree_entropy(G)
    clust = mean_clustering(G)
    dist_corr = distance_correlation_ratio(G)
    ev = edge_vertex_ratio(G)

    return np.array([ds, gap_norm, curv_cv, best_resid,
                     deg_ent, clust, dist_corr, ev])


def compute_all_fingerprints(rules, seeds, runner, backend):
    """Compute fingerprints for all rules x seeds x timesteps."""
    fingerprints = {}
    classifications = {}

    for rule_name, rule_fn in rules.items():
        if runner.should_stop():
            return fingerprints, classifications

        runner.begin_rule(rule_name)
        fingerprints[rule_name] = {}

        i_count = 0
        phi_count = 0

        for seed_name, G0 in seeds.items():
            if runner.should_stop():
                return fingerprints, classifications

            np.random.seed(MASTER_SEED)
            traj = run_trajectory(rule_fn, G0, T, n_max=N_MAX)

            # Classify using standard T=30 prefix for consistency
            traj_short = traj[:min(31, len(traj))]
            i_pos, _ = classify_I(traj_short)
            phi_pos, _ = classify_Phi(traj_short)
            if i_pos:
                i_count += 1
            if phi_pos:
                phi_count += 1

            # Compute fingerprint at each timestep
            n_steps = len(traj)
            fp = np.zeros((n_steps, N_PARAMS))
            for t in range(n_steps):
                window_start = max(0, t - 10)
                traj_window = traj[window_start:t + 1]
                fp[t] = compute_fingerprint(traj[t], traj_window, backend)

            fingerprints[rule_name][seed_name] = fp
            n_nodes = traj[-1].number_of_nodes() if traj else 0
            runner.tick(rule_name, seed_name,
                        result=f"I{'+'if i_pos else '-'}Φ{'+'if phi_pos else '-'} n={n_nodes}")

        i_positive = i_count >= MAJORITY_THRESHOLD
        phi_positive = phi_count >= MAJORITY_THRESHOLD
        classifications[rule_name] = (i_positive, phi_positive)

        cell = f"({'I+'if i_positive else 'I-'}, {'Φ+'if phi_positive else 'Φ-'})"
        runner.finish_rule(rule_name, classification=cell)

    return fingerprints, classifications


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: STABILIZATION ORDER (tighter criterion)
# ══════════════════════════════════════════════════════════════════════

def compute_stabilization_times(fp):
    """Find stabilization time with tighter criterion.

    A parameter is stabilized at time t if:
      1. |P_i(t') - P_i(T)| < 0.05 * range(P_i) for all t' >= t
      2. t < 0.8 * T (must stabilize well before the end)

    Returns n_steps for parameters that never truly stabilize.
    """
    n_steps = fp.shape[0]
    final = fp[-1]
    cutoff = int(STAB_CUTOFF * n_steps)
    stab_times = np.full(N_PARAMS, float(n_steps))

    for i in range(N_PARAMS):
        col = fp[:, i]
        val_range = col.max() - col.min()
        if val_range < 1e-12:
            stab_times[i] = 0
            continue

        threshold = STAB_FRAC * val_range
        for t in range(n_steps):
            if t >= cutoff:
                break  # too late — doesn't count as stabilized
            if np.all(np.abs(col[t:] - final[i]) < threshold):
                stab_times[i] = t
                break

    return stab_times


def bootstrap_precedence(all_stab, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Compute pairwise precedence with bootstrap 95% CIs.

    Returns
    -------
    precedence : (8, 8) array of mean fractions
    ci_low : (8, 8) array of 2.5th percentile
    ci_high : (8, 8) array of 97.5th percentile
    mean_diff : (8, 8) array of mean(t_j - t_i) across samples
    """
    rng = np.random.RandomState(seed)
    n_samples = all_stab.shape[0]

    # Point estimate
    precedence = np.zeros((N_PARAMS, N_PARAMS))
    mean_diff = np.zeros((N_PARAMS, N_PARAMS))
    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                continue
            precedence[i, j] = np.mean(all_stab[:, i] < all_stab[:, j])
            mean_diff[i, j] = np.mean(all_stab[:, j] - all_stab[:, i])

    # Bootstrap CIs
    boot_prec = np.zeros((n_bootstrap, N_PARAMS, N_PARAMS))
    for b in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        sample = all_stab[idx]
        for i in range(N_PARAMS):
            for j in range(N_PARAMS):
                if i == j:
                    continue
                boot_prec[b, i, j] = np.mean(sample[:, i] < sample[:, j])

    ci_low = np.percentile(boot_prec, 2.5, axis=0)
    ci_high = np.percentile(boot_prec, 97.5, axis=0)

    return precedence, ci_low, ci_high, mean_diff


def analyze_stabilization(fingerprints, classifications):
    """Analyze stabilization order across Φ+ rules with bootstrap CIs."""
    phi_plus = [r for r, (i, p) in classifications.items() if p]

    all_stab = []
    stab_data = {}

    for rule_name in phi_plus:
        stab_data[rule_name] = {}
        for seed_name, fp in fingerprints[rule_name].items():
            st = compute_stabilization_times(fp)
            stab_data[rule_name][seed_name] = st.tolist()
            all_stab.append(st)

    if not all_stab:
        z = np.zeros((N_PARAMS, N_PARAMS))
        return stab_data, z, z, z, z, []

    all_stab = np.array(all_stab)
    precedence, ci_low, ci_high, mean_diff = bootstrap_precedence(all_stab)

    mean_times = all_stab.mean(axis=0)
    order = np.argsort(mean_times)
    mean_order = [(PARAM_NAMES[idx], float(mean_times[idx])) for idx in order]

    return stab_data, precedence, ci_low, ci_high, mean_diff, mean_order


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: INTERMEDIATE PHYSICS
# ══════════════════════════════════════════════════════════════════════

def intermediate_distance(fp_vec, profile):
    """Distance between fingerprint and intermediate profile."""
    dists = []
    matched = 0
    for i, pname in enumerate(PARAM_NAMES):
        target = profile.get(pname)
        if target is None:
            continue
        d = abs(fp_vec[i] - target)
        dists.append(d)
        if d < MATCH_TOLERANCE:
            matched += 1
    if not dists:
        return float('inf'), 0
    return float(np.mean(dists)), matched


def identify_intermediates(fingerprints, classifications):
    """For each Φ+ rule at each timestep, find closest intermediate."""
    phi_plus = [r for r, (i, p) in classifications.items() if p]
    visits = {}

    for rule_name in phi_plus:
        visits[rule_name] = {}
        for seed_name, fp in fingerprints[rule_name].items():
            seed_visits = []
            for t in range(fp.shape[0]):
                best_name = None
                best_dist = float('inf')
                best_matched = 0
                for iname, profile in INTERMEDIATE_PHYSICS.items():
                    d, m = intermediate_distance(fp[t], profile)
                    if d < best_dist:
                        best_dist = d
                        best_name = iname
                        best_matched = m
                seed_visits.append({
                    "t": t,
                    "closest": best_name,
                    "distance": float(best_dist),
                    "n_matched": best_matched,
                })
            visits[rule_name][seed_name] = seed_visits

    return visits


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: (I+, Phi-) PLACEMENT
# ══════════════════════════════════════════════════════════════════════

def analyze_i_only(fingerprints, classifications):
    """Terminal fingerprints of (I+, Φ-) rules."""
    i_only = [r for r, (i, p) in classifications.items() if i and not p]
    placements = {}

    for rule_name in i_only:
        terminals = []
        for seed_name, fp in fingerprints[rule_name].items():
            terminals.append(fp[-1])

        mean_terminal = np.mean(terminals, axis=0)
        best_name = None
        best_dist = float('inf')
        for iname, profile in INTERMEDIATE_PHYSICS.items():
            d, _ = intermediate_distance(mean_terminal, profile)
            if d < best_dist:
                best_dist = d
                best_name = iname

        placements[rule_name] = {
            "terminal_vector": mean_terminal.tolist(),
            "closest_intermediate": best_name,
            "distance": float(best_dist),
        }

    return placements


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: COMPLEXITY CORRELATION
# ══════════════════════════════════════════════════════════════════════

def analyze_complexity(fingerprints, classifications):
    """Count stabilized parameters per rule, correlate with source."""
    per_rule = {}
    for rule_name in fingerprints:
        source = RULE_SOURCE.get(rule_name, "?")
        n_stab_list = []
        for seed_name, fp in fingerprints[rule_name].items():
            st = compute_stabilization_times(fp)
            n_stab = int(np.sum(st < fp.shape[0]))
            n_stab_list.append(n_stab)
        per_rule[rule_name] = {
            "mean_n_stabilized": float(np.mean(n_stab_list)),
            "source": source,
            "I_pos": classifications[rule_name][0],
            "Phi_pos": classifications[rule_name][1],
        }
    return per_rule


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_experiment(runner):
    """GPU-scaled physics fingerprint trajectory analysis."""
    results_dir = os.path.join(DATA_DIR, "physics_fingerprint_gpu")
    os.makedirs(results_dir, exist_ok=True)

    # Initialize backend
    backend = get_backend()
    runner.log(f"Backend: {backend.device_name()}")

    rules = ALL_RULES
    seeds = make_initial_graphs()
    n_rules = len(rules)
    runner.set_total(n_rules, N_SEEDS)

    # ── Phase 1: Compute fingerprints ──────────────────────────────
    with runner.phase(f"Computing fingerprints (T={T}, {n_rules} rules)"):
        fingerprints, classifications = compute_all_fingerprints(
            rules, seeds, runner, backend)

    if runner.should_stop():
        return

    # Save raw fingerprints
    rule_names = sorted(rules.keys())
    seed_names = sorted(seeds.keys())
    max_steps = T + 1
    fp_array = np.zeros((n_rules, N_SEEDS, max_steps, N_PARAMS))
    for ri, rn in enumerate(rule_names):
        for si, sn in enumerate(seed_names):
            fp = fingerprints.get(rn, {}).get(sn)
            if fp is not None:
                n_steps = min(fp.shape[0], max_steps)
                fp_array[ri, si, :n_steps] = fp[:n_steps]
    np.save(os.path.join(results_dir, "fingerprints.npy"), fp_array)

    # ── Phase 2: Stabilization analysis ────────────────────────────
    with runner.phase("Analyzing stabilization order"):
        stab_data, precedence, ci_low, ci_high, mean_diff, mean_order = \
            analyze_stabilization(fingerprints, classifications)

    # ── Phase 3: Intermediate physics ──────────────────────────────
    with runner.phase("Identifying intermediate physics"):
        visits = identify_intermediates(fingerprints, classifications)

    # ── Phase 4: (I+, Φ-) placement ───────────────────────────────
    with runner.phase("Analyzing (I+, Φ-) placements"):
        i_only_placements = analyze_i_only(fingerprints, classifications)

    # ── Phase 5: Complexity correlation ────────────────────────────
    with runner.phase("Analyzing complexity correlation"):
        complexity = analyze_complexity(fingerprints, classifications)

    # ── Output ─────────────────────────────────────────────────────
    phi_plus = [r for r, (i, p) in classifications.items() if p]

    cells = {"(I+, Φ+)": [], "(I+, Φ-)": [], "(I-, Φ+)": [], "(I-, Φ-)": []}
    for rn, (ip, pp) in classifications.items():
        key = f"({'I+' if ip else 'I-'}, {'Φ+' if pp else 'Φ-'})"
        cells[key].append(rn)

    print()
    print("═" * 78)
    print(f"PHYSICS FINGERPRINT TRAJECTORIES — GPU SCALE-UP (T={T})")
    print("═" * 78)

    print(f"\nBackend: {backend.device_name()}")
    print(f"Classifications: {len(cells['(I+, Φ+)'])} I+Φ+, "
          f"{len(cells['(I+, Φ-)'])} I+Φ-, "
          f"{len(cells['(I-, Φ+)'])} I-Φ+, "
          f"{len(cells['(I-, Φ-)'])} I-Φ-")

    # ── Analysis 1: Stabilization ──────────────────────────────────
    print(f"\n── ANALYSIS 1: Mean stabilization order (Φ+ rules, n={len(phi_plus)}) ──")
    if mean_order:
        for rank, (pname, mtime) in enumerate(mean_order, 1):
            print(f"  {rank}. {pname:<18s} mean t_stab = {mtime:.1f}")

    # Precedence with CIs
    print("\n  Pairwise precedence [95% CI] (★ = CI excludes 0.5):")
    header = "            " + "".join(f"{p[:7]:>10s}" for p in PARAM_NAMES)
    print(header)
    for i, pname in enumerate(PARAM_NAMES):
        parts = []
        for j in range(N_PARAMS):
            if i == j:
                parts.append(f"{'---':>10s}")
            else:
                sig = "★" if ci_low[i, j] > 0.5 or ci_high[i, j] < 0.5 else " "
                parts.append(f"{precedence[i, j]:.2f}{sig:s}".rjust(10))
        print(f"  {pname[:10]:<10s}" + "".join(parts))

    # Mean time differences
    print("\n  Mean time difference (t_col - t_row, positive = row stabilizes first):")
    header = "            " + "".join(f"{p[:7]:>10s}" for p in PARAM_NAMES)
    print(header)
    for i, pname in enumerate(PARAM_NAMES):
        parts = []
        for j in range(N_PARAMS):
            if i == j:
                parts.append(f"{'---':>10s}")
            else:
                parts.append(f"{mean_diff[i, j]:>+9.1f} ")
        print(f"  {pname[:10]:<10s}" + "".join(parts))

    # Hypothesis tests
    i_ds = PARAM_NAMES.index("d_s")
    i_cv = PARAM_NAMES.index("curv_cv")
    i_lr = PARAM_NAMES.index("law_resid")
    i_cl = PARAM_NAMES.index("clustering")
    i_dc = PARAM_NAMES.index("dist_corr")

    print("\n  Hypothesis tests:")
    for label, i, j, baseline in [
        ("H1 (d_s < curv_cv)", i_ds, i_cv, BASELINE_H1),
        ("H2 (curv_cv < law_resid)", i_cv, i_lr, BASELINE_H2),
        ("H3 (clustering < dist_corr)", i_cl, i_dc, BASELINE_H3),
    ]:
        p = precedence[i, j]
        lo, hi = ci_low[i, j], ci_high[i, j]
        sig = "SIGNIFICANT" if lo > 0.5 or hi < 0.5 else "not significant"
        diff = mean_diff[i, j]
        print(f"    {label}: {p:.2f} [{lo:.2f}, {hi:.2f}] "
              f"Δt={diff:+.1f}  (T=30: {baseline:.2f})  {sig}")

    # ── Analysis 2: Intermediates ──────────────────────────────────
    print(f"\n── ANALYSIS 2: Intermediate physics visits (Φ+ rules) ──")
    if visits:
        visit_counts = {name: 0 for name in INTERMEDIATE_PHYSICS}
        total_trajs = 0
        for rn in visits:
            for sn in visits[rn]:
                total_trajs += 1
                visited_this = set()
                for entry in visits[rn][sn]:
                    if entry["n_matched"] >= 4:
                        visited_this.add(entry["closest"])
                for v in visited_this:
                    visit_counts[v] += 1

        for iname in sorted(visit_counts, key=visit_counts.get, reverse=True):
            print(f"  {iname:<25s} visited by "
                  f"{visit_counts[iname]}/{total_trajs} trajectories")

    # ── Analysis 3: (I+, Φ-) ──────────────────────────────────────
    print(f"\n── ANALYSIS 3: (I+, Φ-) terminal positions ──")
    if i_only_placements:
        for rn, info in sorted(i_only_placements.items()):
            vec_str = ", ".join(f"{v:.2f}" for v in info["terminal_vector"])
            print(f"  {rn:<26s} → {info['closest_intermediate']:<20s} "
                  f"(d={info['distance']:.3f})")
            print(f"    P = [{vec_str}]")
    else:
        print("  No (I+, Φ-) rules found.")

    # ── Analysis 4: Complexity ─────────────────────────────────────
    print(f"\n── ANALYSIS 4: Stabilized parameters by source ──")
    by_source = {}
    for rn, info in complexity.items():
        src = info["source"]
        by_source.setdefault(src, []).append(info["mean_n_stabilized"])
    for src in sorted(by_source):
        vals = by_source[src]
        print(f"  {src:<10s}: mean {np.mean(vals):.1f} / {N_PARAMS} params "
              f"stabilized (n={len(vals)} rules)")

    # ── Comparison with T=30 baseline ──────────────────────────────
    print(f"\n── COMPARISON: T=30 vs T={T} ──")
    if mean_order:
        order_t30 = [p for p, _ in BASELINE_ORDER]
        order_t200 = [p for p, _ in mean_order]
        print(f"  T=30 order:  {' → '.join(order_t30)}")
        print(f"  T={T} order: {' → '.join(order_t200)}")
        # Rank correlation
        from scipy.stats import kendalltau
        ranks_30 = [order_t30.index(p) for p in PARAM_NAMES]
        ranks_200 = [order_t200.index(p) for p in PARAM_NAMES]
        tau, pval = kendalltau(ranks_30, ranks_200)
        print(f"  Kendall τ between orders: {tau:.3f} (p={pval:.3f})")
        if tau > 0.6:
            print(f"  → Ordering is ROBUST across scales")
        elif tau > 0.3:
            print(f"  → Ordering is PARTIALLY robust")
        else:
            print(f"  → Ordering CHANGED at larger scale")

    # ── Save all results ───────────────────────────────────────────
    with runner.phase("Saving results"):
        with open(os.path.join(results_dir, "stabilization_times.json"), "w") as f:
            json.dump(stab_data, f, indent=2, default=_json_default)

        with open(os.path.join(results_dir, "precedence_matrix.json"), "w") as f:
            json.dump({
                "param_names": PARAM_NAMES,
                "matrix": precedence.tolist(),
                "ci_low": ci_low.tolist(),
                "ci_high": ci_high.tolist(),
                "mean_diff": mean_diff.tolist(),
                "mean_order": mean_order,
                "n_bootstrap": N_BOOTSTRAP,
            }, f, indent=2)

        with open(os.path.join(results_dir, "intermediate_visits.json"), "w") as f:
            json.dump(visits, f, indent=2, default=_json_default)

        with open(os.path.join(results_dir, "i_only_placements.json"), "w") as f:
            json.dump(i_only_placements, f, indent=2, default=_json_default)

        summary = {
            "experiment": "exp_physics_fingerprint_gpu",
            "T": T,
            "n_rules": n_rules,
            "n_seeds": N_SEEDS,
            "backend": backend.device_name(),
            "stab_frac": STAB_FRAC,
            "stab_cutoff": STAB_CUTOFF,
            "classifications": {
                rn: {"I": ip, "Phi": pp}
                for rn, (ip, pp) in classifications.items()
            },
            "mean_stabilization_order": mean_order,
            "hypothesis_tests": {
                "H1_ds_before_curv": {
                    "fraction": float(precedence[i_ds, i_cv]),
                    "ci": [float(ci_low[i_ds, i_cv]), float(ci_high[i_ds, i_cv])],
                    "mean_diff": float(mean_diff[i_ds, i_cv]),
                    "baseline_t30": BASELINE_H1,
                },
                "H2_curv_before_law": {
                    "fraction": float(precedence[i_cv, i_lr]),
                    "ci": [float(ci_low[i_cv, i_lr]), float(ci_high[i_cv, i_lr])],
                    "mean_diff": float(mean_diff[i_cv, i_lr]),
                    "baseline_t30": BASELINE_H2,
                },
                "H3_clust_before_distcorr": {
                    "fraction": float(precedence[i_cl, i_dc]),
                    "ci": [float(ci_low[i_cl, i_dc]), float(ci_high[i_cl, i_dc])],
                    "mean_diff": float(mean_diff[i_cl, i_dc]),
                    "baseline_t30": BASELINE_H3,
                },
            },
            "i_only_placements": {
                rn: info["closest_intermediate"]
                for rn, info in i_only_placements.items()
            },
        }
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Physics Fingerprint Trajectories — GPU Scale-Up (T={T})\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Backend: {backend.device_name()}\n")
            f.write(f"Rules: {n_rules}, Seeds: {N_SEEDS}, T: {T}\n")
            f.write(f"Stabilization: {STAB_FRAC*100:.0f}% range, "
                    f"cutoff at {STAB_CUTOFF*100:.0f}% of T\n")
            f.write(f"Cells: {len(cells['(I+, Φ+)'])} I+Φ+, "
                    f"{len(cells['(I+, Φ-)'])} I+Φ-, "
                    f"{len(cells['(I-, Φ+)'])} I-Φ+, "
                    f"{len(cells['(I-, Φ-)'])} I-Φ-\n\n")
            f.write("Mean stabilization order:\n")
            for rank, (pname, mtime) in enumerate(mean_order, 1):
                f.write(f"  {rank}. {pname:<18s} t={mtime:.1f}\n")
            f.write("\nHypothesis tests:\n")
            for label, i, j, bl in [
                ("H1 (d_s < curv_cv)", i_ds, i_cv, BASELINE_H1),
                ("H2 (curv_cv < law_resid)", i_cv, i_lr, BASELINE_H2),
                ("H3 (clustering < dist_corr)", i_cl, i_dc, BASELINE_H3),
            ]:
                p = precedence[i, j]
                lo, hi = ci_low[i, j], ci_high[i, j]
                sig = "SIG" if lo > 0.5 or hi < 0.5 else "n.s."
                f.write(f"  {label}: {p:.2f} [{lo:.2f},{hi:.2f}] "
                        f"(T=30: {bl:.2f}) {sig}\n")

    print(f"\nResults saved to {results_dir}/")
    runner.finish(f"exp_physics_fingerprint_gpu complete — T={T}, "
                  f"{len(phi_plus)} Φ+ rules")


def _json_default(obj):
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
        "exp_physics_fingerprint_gpu",
        total_rules=33,
        total_seeds=N_SEEDS,
        phases=[
            f"Computing fingerprints (T={T})",
            "Analyzing stabilization order",
            "Identifying intermediate physics",
            "Analyzing (I+, Φ-) placements",
            "Analyzing complexity correlation",
            "Saving results",
        ],
        auto_close_delay=3.0,
    )
    runner.run(run_experiment)
