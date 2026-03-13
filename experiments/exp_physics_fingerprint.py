"""
exp_physics_fingerprint — Physics fingerprint trajectories through R^8.

Measures an 8-dimensional "physics parameter vector" P(t) at every timestep
of every rule's trajectory, then analyzes:
  1. Stabilization order of parameters for Phi+ rules
  2. Intermediate physics identification (matching to known profiles)
  3. Terminal positions of (I+, Phi-) rules
  4. Correlation of stabilized-parameter count with signature complexity

Supports the paper sketch: papers/intermediate_physics.md

Usage:
    python experiments/exp_physics_fingerprint.py
    # or: make exp-fingerprint
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    MASTER_SEED, MAJORITY_THRESHOLD, DATA_DIR, N_MAX,
    DS_STD_STAR, DS_INT_DIST, LAW_RESIDUAL_STAR, CURVATURE_KAPPA_STAR,
)
from primo.rules import (
    ALL_RULES, make_initial_graphs, run_trajectory, RULE_SOURCE,
)
from primo.predicates import classify_I, classify_Phi
from primo.trajectories import (
    spectral_dimension_estimate, spectral_gap, curvature_homogeneity,
    normalized_degree_entropy, mean_clustering, distance_correlation_ratio,
    edge_vertex_ratio, law_residual_score, AGGREGATE_QUANTITIES,
)
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

T = 30
N_SEEDS = 4
PARAM_NAMES = [
    "d_s", "gap_norm", "curv_cv", "law_resid",
    "deg_entropy", "clustering", "dist_corr", "ev_ratio",
]
N_PARAMS = len(PARAM_NAMES)

# Stabilization tolerance: parameter is stable if within 10% of its range
# from the final value, for all subsequent timesteps.
STAB_FRAC = 0.1

# Intermediate physics target profiles (from papers/intermediate_physics.md §3)
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
        "d_s": None,  # unstable — skip in distance calc
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

# Distance tolerance for intermediate matching (per normalized parameter)
MATCH_TOLERANCE = 0.3


# ══════════════════════════════════════════════════════════════════════
# FINGERPRINT COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_fingerprint(G, traj_window):
    """Compute the 8-parameter physics fingerprint for graph G.

    Parameters
    ----------
    G : nx.Graph
        Current graph.
    traj_window : list of nx.Graph
        Sliding window of recent graphs (for law residual).

    Returns
    -------
    np.ndarray of shape (8,)
        [d_s, gap_norm, curv_cv, law_resid, deg_entropy, clustering,
         dist_corr, ev_ratio]
    """
    # P1: spectral dimension
    ds = spectral_dimension_estimate(G)
    if ds is None:
        ds = 0.0

    # P2: normalized spectral gap
    gap = spectral_gap(G)
    ds_clamp = max(ds, 0.1)
    gap_norm = gap / ds_clamp

    # P3: curvature homogeneity
    curv_cv = curvature_homogeneity(G)
    if curv_cv == float('inf'):
        curv_cv = 10.0  # cap for numerical sanity

    # P4: law residual (sliding window)
    best_resid = float('inf')
    if len(traj_window) >= 4:
        for fn in AGGREGATE_QUANTITIES.values():
            resid, _ = law_residual_score(traj_window, fn)
            if resid < best_resid:
                best_resid = resid
    if best_resid == float('inf'):
        best_resid = 1.0

    # P5: normalized degree entropy
    deg_ent = normalized_degree_entropy(G)

    # P6: mean clustering
    clust = mean_clustering(G)

    # P7: distance correlation ratio
    dist_corr = distance_correlation_ratio(G)

    # P8: edge-vertex ratio
    ev = edge_vertex_ratio(G)

    return np.array([ds, gap_norm, curv_cv, best_resid,
                     deg_ent, clust, dist_corr, ev])


def compute_all_fingerprints(rules, seeds, runner):
    """Compute fingerprints for all rules × seeds × timesteps.

    Returns
    -------
    fingerprints : dict
        rule_name -> seed_name -> np.ndarray of shape (T+1, 8)
    classifications : dict
        rule_name -> (I_positive, Phi_positive)
    """
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

            # Classify
            i_pos, _ = classify_I(traj)
            phi_pos, _ = classify_Phi(traj)
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
                fp[t] = compute_fingerprint(traj[t], traj_window)

            fingerprints[rule_name][seed_name] = fp
            runner.tick(rule_name, seed_name,
                        result=f"I{'+'if i_pos else '-'}Φ{'+'if phi_pos else '-'}")

        i_positive = i_count >= MAJORITY_THRESHOLD
        phi_positive = phi_count >= MAJORITY_THRESHOLD
        classifications[rule_name] = (i_positive, phi_positive)

        cell = f"({'I+'if i_positive else 'I-'}, {'Φ+'if phi_positive else 'Φ-'})"
        runner.finish_rule(rule_name, classification=cell)

    return fingerprints, classifications


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: STABILIZATION ORDER
# ══════════════════════════════════════════════════════════════════════

def compute_stabilization_times(fp):
    """Find stabilization time for each parameter.

    Parameters
    ----------
    fp : np.ndarray of shape (n_steps, 8)

    Returns
    -------
    stab_times : np.ndarray of shape (8,)
        Stabilization time for each parameter. n_steps if never stabilized.
    """
    n_steps = fp.shape[0]
    final = fp[-1]
    stab_times = np.full(N_PARAMS, n_steps, dtype=float)

    for i in range(N_PARAMS):
        col = fp[:, i]
        val_range = col.max() - col.min()
        if val_range < 1e-12:
            stab_times[i] = 0
            continue

        threshold = STAB_FRAC * val_range
        # Find earliest t such that all subsequent values are within threshold
        for t in range(n_steps):
            if np.all(np.abs(col[t:] - final[i]) < threshold):
                stab_times[i] = t
                break

    return stab_times


def analyze_stabilization(fingerprints, classifications):
    """Analyze stabilization order across all Phi+ rules.

    Returns
    -------
    stab_data : dict
        Per-rule, per-seed stabilization times.
    precedence : np.ndarray of shape (8, 8)
        precedence[i][j] = fraction of Phi+ (rule, seed) pairs where
        parameter i stabilizes before parameter j.
    mean_order : list of (param_name, mean_stab_time)
    """
    phi_plus_rules = [r for r, (i, p) in classifications.items() if p]

    all_stab = []  # list of (8,) arrays
    stab_data = {}

    for rule_name in phi_plus_rules:
        stab_data[rule_name] = {}
        for seed_name, fp in fingerprints[rule_name].items():
            st = compute_stabilization_times(fp)
            stab_data[rule_name][seed_name] = st.tolist()
            all_stab.append(st)

    if not all_stab:
        return stab_data, np.zeros((N_PARAMS, N_PARAMS)), []

    all_stab = np.array(all_stab)  # (n_samples, 8)

    # Pairwise precedence matrix
    n_samples = all_stab.shape[0]
    precedence = np.zeros((N_PARAMS, N_PARAMS))
    for i in range(N_PARAMS):
        for j in range(N_PARAMS):
            if i == j:
                continue
            precedence[i, j] = np.mean(all_stab[:, i] < all_stab[:, j])

    # Mean stabilization order
    mean_times = all_stab.mean(axis=0)
    order = np.argsort(mean_times)
    mean_order = [(PARAM_NAMES[idx], float(mean_times[idx])) for idx in order]

    return stab_data, precedence, mean_order


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: INTERMEDIATE PHYSICS IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════

def intermediate_distance(fp_vec, profile):
    """Compute normalized distance between a fingerprint vector and a profile.

    Returns
    -------
    distance : float
        Mean absolute deviation across defined parameters.
    n_matched : int
        Number of parameters within tolerance.
    """
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
    """For each Phi+ rule at each timestep, find closest intermediate.

    Returns
    -------
    visits : dict
        rule_name -> seed_name -> list of (timestep, best_intermediate, distance, n_matched)
    """
    phi_plus_rules = [r for r, (i, p) in classifications.items() if p]
    visits = {}

    for rule_name in phi_plus_rules:
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
    """Analyze terminal fingerprints of (I+, Phi-) rules.

    Returns
    -------
    placements : dict
        rule_name -> {terminal_vector, closest_intermediate, distance}
    """
    i_only = [r for r, (i, p) in classifications.items() if i and not p]
    placements = {}

    for rule_name in i_only:
        seed_terminals = []
        for seed_name, fp in fingerprints[rule_name].items():
            terminal = fp[-1]
            seed_terminals.append(terminal)

        mean_terminal = np.mean(seed_terminals, axis=0)

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
# ANALYSIS 4: SIGNATURE COMPLEXITY CORRELATION
# ══════════════════════════════════════════════════════════════════════

def analyze_complexity_correlation(fingerprints, classifications):
    """Count stabilized parameters per rule at T, correlate with source.

    Returns
    -------
    per_rule : dict
        rule_name -> {n_stabilized, source}
    """
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
    """Physics fingerprint trajectory analysis."""
    results_dir = os.path.join(DATA_DIR, "physics_fingerprint")
    os.makedirs(results_dir, exist_ok=True)

    rules = ALL_RULES
    seeds = make_initial_graphs()

    n_rules = len(rules)
    runner.set_total(n_rules, N_SEEDS)

    # ── Phase 1: Compute all fingerprints ──────────────────────────
    with runner.phase("Computing physics fingerprints"):
        fingerprints, classifications = compute_all_fingerprints(
            rules, seeds, runner)

    if runner.should_stop():
        return

    # Save raw fingerprints as numpy
    fp_array = np.zeros((n_rules, N_SEEDS, T + 1, N_PARAMS))
    rule_names = sorted(rules.keys())
    seed_names = sorted(seeds.keys())
    for ri, rn in enumerate(rule_names):
        for si, sn in enumerate(seed_names):
            fp = fingerprints.get(rn, {}).get(sn)
            if fp is not None:
                n_steps = min(fp.shape[0], T + 1)
                fp_array[ri, si, :n_steps] = fp[:n_steps]

    np.save(os.path.join(results_dir, "fingerprints.npy"), fp_array)

    # ── Phase 2: Stabilization analysis ────────────────────────────
    with runner.phase("Analyzing stabilization order"):
        stab_data, precedence, mean_order = analyze_stabilization(
            fingerprints, classifications)

    # ── Phase 3: Intermediate physics ──────────────────────────────
    with runner.phase("Identifying intermediate physics"):
        visits = identify_intermediates(fingerprints, classifications)

    # ── Phase 4: (I+, Phi-) placement ─────────────────────────────
    with runner.phase("Analyzing (I+, Φ-) placements"):
        i_only_placements = analyze_i_only(fingerprints, classifications)

    # ── Phase 5: Complexity correlation ────────────────────────────
    with runner.phase("Analyzing complexity correlation"):
        complexity = analyze_complexity_correlation(
            fingerprints, classifications)

    # ── Output ─────────────────────────────────────────────────────
    print()
    print("═" * 78)
    print("PHYSICS FINGERPRINT TRAJECTORIES — RESULTS")
    print("═" * 78)

    # Classifications summary
    cells = {"(I+, Φ+)": [], "(I+, Φ-)": [], "(I-, Φ+)": [], "(I-, Φ-)": []}
    for rn, (ip, pp) in classifications.items():
        key = f"({'I+' if ip else 'I-'}, {'Φ+' if pp else 'Φ-'})"
        cells[key].append(rn)

    print(f"\nClassifications: {len(cells['(I+, Φ+)'])} I+Φ+, "
          f"{len(cells['(I+, Φ-)'])} I+Φ-, "
          f"{len(cells['(I-, Φ+)'])} I-Φ+, "
          f"{len(cells['(I-, Φ-)'])} I-Φ-")

    # Stabilization order
    print("\n── ANALYSIS 1: Mean stabilization order (Φ+ rules) ──")
    if mean_order:
        for rank, (pname, mtime) in enumerate(mean_order, 1):
            print(f"  {rank}. {pname:<18s} mean t_stab = {mtime:.1f}")
    else:
        print("  No Φ+ rules found.")

    # Precedence matrix
    print("\n  Pairwise precedence (fraction where row stabilizes before col):")
    header = "            " + "".join(f"{p[:7]:>8s}" for p in PARAM_NAMES)
    print(header)
    for i, pname in enumerate(PARAM_NAMES):
        row = f"  {pname[:10]:<10s}" + "".join(
            f"{precedence[i, j]:>8.2f}" for j in range(N_PARAMS))
        print(row)

    # Key hypothesis tests
    print("\n  Hypothesis tests (fraction Φ+ rule×seed with ordering):")
    phi_plus = [r for r, (i, p) in classifications.items() if p]
    if phi_plus:
        # H1: d_s before curv_cv
        i_ds = PARAM_NAMES.index("d_s")
        i_cv = PARAM_NAMES.index("curv_cv")
        print(f"    H1 (d_s < curv_cv):      {precedence[i_ds, i_cv]:.2f}")
        # H2: curv_cv before law_resid
        i_lr = PARAM_NAMES.index("law_resid")
        print(f"    H2 (curv_cv < law_resid): {precedence[i_cv, i_lr]:.2f}")
        # H3: clustering before dist_corr
        i_cl = PARAM_NAMES.index("clustering")
        i_dc = PARAM_NAMES.index("dist_corr")
        print(f"    H3 (clustering < dist_corr): {precedence[i_cl, i_dc]:.2f}")

    # Intermediate visits
    print("\n── ANALYSIS 2: Intermediate physics visits (Φ+ rules) ──")
    if visits:
        # Which intermediates are visited at any timestep with n_matched >= 4?
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
            print(f"  {iname:<25s} visited by {visit_counts[iname]}/{total_trajs} trajectories")
    else:
        print("  No Φ+ rules found.")

    # (I+, Phi-) placement
    print("\n── ANALYSIS 3: (I+, Φ-) terminal positions ──")
    if i_only_placements:
        for rn, info in i_only_placements.items():
            vec_str = ", ".join(f"{v:.2f}" for v in info["terminal_vector"])
            print(f"  {rn:<26s} → {info['closest_intermediate']:<20s} "
                  f"(d={info['distance']:.3f})")
            print(f"    P = [{vec_str}]")
    else:
        print("  No (I+, Φ-) rules found.")

    # Complexity correlation
    print("\n── ANALYSIS 4: Stabilized parameters by source ──")
    by_source = {}
    for rn, info in complexity.items():
        src = info["source"]
        by_source.setdefault(src, []).append(info["mean_n_stabilized"])
    for src in sorted(by_source):
        vals = by_source[src]
        print(f"  {src:<10s}: mean {np.mean(vals):.1f} / {N_PARAMS} params stabilized "
              f"(n={len(vals)} rules)")

    # ── Save all results ───────────────────────────────────────────
    with runner.phase("Saving results"):
        # Stabilization times
        with open(os.path.join(results_dir, "stabilization_times.json"), "w") as f:
            json.dump(stab_data, f, indent=2, default=_json_default)

        # Precedence matrix
        with open(os.path.join(results_dir, "precedence_matrix.json"), "w") as f:
            json.dump({
                "param_names": PARAM_NAMES,
                "matrix": precedence.tolist(),
                "mean_order": mean_order,
            }, f, indent=2)

        # Intermediate visits
        with open(os.path.join(results_dir, "intermediate_visits.json"), "w") as f:
            json.dump(visits, f, indent=2, default=_json_default)

        # (I+, Phi-) placements
        with open(os.path.join(results_dir, "i_only_placements.json"), "w") as f:
            json.dump(i_only_placements, f, indent=2, default=_json_default)

        # Summary
        summary = {
            "experiment": "exp_physics_fingerprint",
            "n_rules": n_rules,
            "T": T,
            "n_seeds": N_SEEDS,
            "classifications": {
                rn: {"I": ip, "Phi": pp}
                for rn, (ip, pp) in classifications.items()
            },
            "mean_stabilization_order": mean_order,
            "hypothesis_tests": {},
            "i_only_placements": {
                rn: info["closest_intermediate"]
                for rn, info in i_only_placements.items()
            },
        }
        if phi_plus:
            summary["hypothesis_tests"] = {
                "H1_ds_before_curv": float(precedence[
                    PARAM_NAMES.index("d_s"),
                    PARAM_NAMES.index("curv_cv")]),
                "H2_curv_before_law": float(precedence[
                    PARAM_NAMES.index("curv_cv"),
                    PARAM_NAMES.index("law_resid")]),
                "H3_clust_before_distcorr": float(precedence[
                    PARAM_NAMES.index("clustering"),
                    PARAM_NAMES.index("dist_corr")]),
            }

        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        # Human-readable summary
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Physics Fingerprint Trajectories — Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Rules: {n_rules}, Seeds: {N_SEEDS}, T: {T}\n")
            f.write(f"Cells: {len(cells['(I+, Φ+)'])} I+Φ+, "
                    f"{len(cells['(I+, Φ-)'])} I+Φ-, "
                    f"{len(cells['(I-, Φ+)'])} I-Φ+, "
                    f"{len(cells['(I-, Φ-)'])} I-Φ-\n\n")
            f.write("Mean stabilization order:\n")
            for rank, (pname, mtime) in enumerate(mean_order, 1):
                f.write(f"  {rank}. {pname:<18s} t={mtime:.1f}\n")
            if phi_plus:
                f.write(f"\nH1 (d_s < curv_cv): {precedence[PARAM_NAMES.index('d_s'), PARAM_NAMES.index('curv_cv')]:.2f}\n")
                f.write(f"H2 (curv_cv < law_resid): {precedence[PARAM_NAMES.index('curv_cv'), PARAM_NAMES.index('law_resid')]:.2f}\n")
                f.write(f"H3 (clustering < dist_corr): {precedence[PARAM_NAMES.index('clustering'), PARAM_NAMES.index('dist_corr')]:.2f}\n")

    print(f"\nResults saved to {results_dir}/")
    runner.finish(f"exp_physics_fingerprint complete — "
                  f"{len(phi_plus)} Φ+ rules analyzed")


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
        "exp_physics_fingerprint",
        total_rules=33,
        total_seeds=N_SEEDS,
        phases=[
            "Computing physics fingerprints",
            "Analyzing stabilization order",
            "Identifying intermediate physics",
            "Analyzing (I+, Φ-) placements",
            "Analyzing complexity correlation",
            "Saving results",
        ],
    )
    runner.run(run_experiment)
