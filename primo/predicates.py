"""
PRIMO Predicates — I-predicate and Φ-predicate classification.

Implements the two independent predicates from Paper 1:
- I-predicate (inference-like): embedding convergence + compression gate
- Φ-predicate (physics-like): spectral dimension stability + law fitting + curvature

Each predicate classifies a trajectory as positive or negative.
Classification of a *rule* aggregates over initial graphs (majority vote).

Usage:
    from primo.predicates import classify_rule, classify_I, classify_Phi
    I_pos, I_detail = classify_I(traj)
    Phi_pos, Phi_detail = classify_Phi(traj)
    result = classify_rule(rule_fn, seeds, T=30)
"""

import numpy as np
from scipy.stats import kendalltau

from primo.config import (
    TAU_STAR, RHO_STAR, STRAIGHTNESS_STAR, ANTI_CONVERGENCE_THRESHOLD,
    DS_STD_STAR, DS_INT_DIST, LAW_RESIDUAL_STAR, CURVATURE_KAPPA_STAR,
    MAJORITY_THRESHOLD, T_DEFAULT, MASTER_SEED, EMBEDDING_DIM,
)
from primo.trajectories import (
    embed_trajectory, subspace_cosine, compression_ratio,
    spectral_dimension_estimate, curvature_homogeneity,
    law_residual_score, EMBEDDING_FUNCTIONS, AGGREGATE_QUANTITIES,
)
from primo.rules import make_initial_graphs, run_trajectory


# ══════════════════════════════════════════════════════════════════════
# I-PREDICATE MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════

def measure_I_scores(traj, embed_fn, d=None):
    """Compute I-predicate scores for a trajectory under one embedding.

    Parameters
    ----------
    traj : list of nx.Graph
    embed_fn : callable
        Graph × d → numpy array.
    d : int or None

    Returns
    -------
    dict with keys:
        tau_to_final : float — Kendall τ of cosine-to-final sequence
        alignment_tau : float — Kendall τ of consecutive cosines
        cosines_to_final : list of float
    """
    d = d or EMBEDDING_DIM
    embeddings = embed_trajectory(traj, embed_fn, d)

    # Find the last valid embedding
    final_emb = None
    for e in reversed(embeddings):
        if e is not None:
            final_emb = e
            break

    # Cosine similarity to final embedding
    cosines_to_final = []
    if final_emb is not None:
        for i in range(len(embeddings) - 1):
            if embeddings[i] is not None:
                cosines_to_final.append(
                    subspace_cosine(embeddings[i], final_emb))

    if len(cosines_to_final) >= 3:
        tau_to_final, _ = kendalltau(
            range(len(cosines_to_final)), cosines_to_final)
        tau_to_final = float(tau_to_final) if not np.isnan(tau_to_final) else 0.0
    else:
        tau_to_final = 0.0

    # Consecutive alignment cosines
    cosines = []
    for i in range(len(embeddings) - 1):
        if embeddings[i] is not None and embeddings[i + 1] is not None:
            cosines.append(subspace_cosine(embeddings[i], embeddings[i + 1]))

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

def measure_Phi_scores(traj):
    """Compute Φ-predicate scores for a trajectory.

    Returns
    -------
    dict with keys:
        ds_mean, ds_std, ds_int_dist : spectral dimension stats
        ds_values : list of float
        law_results : dict of {name: {residual, model}}
        best_law_name, best_law_resid, best_law_model : best law fit
        curv_homogeneity : float
    """
    # Spectral dimension from second third of trajectory onward
    ds_values = []
    for G in traj[len(traj) // 3:]:
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

    # Law fitting across aggregate quantities
    law_results = {}
    for name, fn in AGGREGATE_QUANTITIES.items():
        resid, model = law_residual_score(traj, fn)
        law_results[name] = {"residual": resid, "model": model}

    best_law_name = min(law_results, key=lambda k: law_results[k]["residual"])
    best_law_resid = law_results[best_law_name]["residual"]
    best_law_model = law_results[best_law_name]["model"]

    # Curvature homogeneity from second half
    curv_values = []
    for G in traj[len(traj) // 2:]:
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

def classify_I(traj, tau_star=None, rho_star=None):
    """Classify a single trajectory under the I-predicate.

    Parameters
    ----------
    traj : list of nx.Graph
    tau_star : float or None
        Convergence threshold. Default: config.TAU_STAR.
    rho_star : float or None
        Compression gate threshold. Default: config.RHO_STAR.

    Returns
    -------
    is_positive : bool
    detail : dict
    """
    tau_star = tau_star if tau_star is not None else TAU_STAR
    rho_star = rho_star if rho_star is not None else RHO_STAR

    # Compression gate
    cr = compression_ratio(traj)
    if cr >= rho_star:
        return False, {"compression": cr, "gate": "failed"}

    # Compute scores for all embeddings
    results = {}
    for name, embed_fn in EMBEDDING_FUNCTIONS.items():
        results[name] = measure_I_scores(traj, embed_fn)

    convergence_scores = {
        e: results[e]["tau_to_final"] for e in EMBEDDING_FUNCTIONS
    }

    # Pass if any embedding exceeds threshold
    convergence_pass = any(
        s > tau_star for s in convergence_scores.values())

    # Anti-convergence guard: reject if all negative
    if all(s < 0 for s in convergence_scores.values()):
        convergence_pass = False

    return convergence_pass, {
        "compression": cr, "gate": "passed",
        "embed_scores": {
            e: {
                "tau_to_final": results[e]["tau_to_final"],
                "align_tau": results[e]["alignment_tau"],
            }
            for e in EMBEDDING_FUNCTIONS
        },
        "convergence_pass": convergence_pass,
    }


def classify_Phi(traj, ds_tol=None, ds_std_max=None,
                 law_threshold=None, curv_threshold=None):
    """Classify a single trajectory under the Φ-predicate.

    Parameters
    ----------
    traj : list of nx.Graph
    ds_tol : float or None
        Max distance of ds_mean from nearest integer. Default: config.DS_INT_DIST.
    ds_std_max : float or None
        Max spectral dimension std. Default: config.DS_STD_STAR.
    law_threshold : float or None
        Max law residual. Default: config.LAW_RESIDUAL_STAR.
    curv_threshold : float or None
        Max curvature CV. Default: config.CURVATURE_KAPPA_STAR.

    Returns
    -------
    is_positive : bool
    detail : dict
    """
    ds_tol = ds_tol if ds_tol is not None else DS_INT_DIST
    ds_std_max = ds_std_max if ds_std_max is not None else DS_STD_STAR
    law_threshold = law_threshold if law_threshold is not None else LAW_RESIDUAL_STAR
    curv_threshold = curv_threshold if curv_threshold is not None else CURVATURE_KAPPA_STAR

    phi = measure_Phi_scores(traj)

    ds_ok = (phi["ds_int_dist"] is not None
             and phi["ds_int_dist"] < ds_tol
             and phi["ds_std"] is not None
             and phi["ds_std"] < ds_std_max)
    law_ok = phi["best_law_resid"] < law_threshold
    curv_ok = phi["curv_homogeneity"] < curv_threshold

    return (ds_ok and (law_ok or curv_ok)), phi


# ══════════════════════════════════════════════════════════════════════
# RULE-LEVEL CLASSIFICATION (majority over seeds)
# ══════════════════════════════════════════════════════════════════════

def classify_rule(rule_fn, seeds=None, T=None, majority=None, seed=None):
    """Classify a rule by running from all seed graphs and majority voting.

    Parameters
    ----------
    rule_fn : callable
        Graph → Graph rewrite rule.
    seeds : dict or None
        name → nx.Graph. Default: make_initial_graphs().
    T : int or None
        Trajectory length. Default: config.T_DEFAULT.
    majority : int or None
        Seeds needed for positive classification. Default: config.MAJORITY_THRESHOLD.
    seed : int or None
        Random seed. Default: config.MASTER_SEED.

    Returns
    -------
    dict with keys:
        I_positive : bool
        Phi_positive : bool
        I_count : int — number of seeds classified I+
        Phi_count : int — number of seeds classified Φ+
        seed_results : dict — per-seed details
    """
    seeds = seeds or make_initial_graphs()
    T = T if T is not None else T_DEFAULT
    majority = majority if majority is not None else MAJORITY_THRESHOLD
    seed = seed if seed is not None else MASTER_SEED

    seed_results = {}
    for seed_name, G0 in seeds.items():
        np.random.seed(seed)
        traj = run_trajectory(rule_fn, G0, T)
        I_pos, I_detail = classify_I(traj)
        Phi_pos, Phi_detail = classify_Phi(traj)
        seed_results[seed_name] = {
            "I": I_pos, "Phi": Phi_pos,
            "I_detail": I_detail, "Phi_detail": Phi_detail,
        }

    I_count = sum(1 for r in seed_results.values() if r["I"])
    Phi_count = sum(1 for r in seed_results.values() if r["Phi"])

    return {
        "I_positive": I_count >= majority,
        "Phi_positive": Phi_count >= majority,
        "I_count": I_count,
        "Phi_count": Phi_count,
        "seed_results": seed_results,
    }


def classify_all_rules(rules, seeds=None, T=None, majority=None, seed=None):
    """Classify all rules and return structured results.

    Parameters
    ----------
    rules : dict
        name → callable.

    Returns
    -------
    classifications : dict
        rule_name → (I_positive, Phi_positive)
    details : dict
        rule_name → full classify_rule output
    """
    seeds = seeds or make_initial_graphs()
    T = T if T is not None else T_DEFAULT
    majority = majority if majority is not None else MAJORITY_THRESHOLD
    seed = seed if seed is not None else MASTER_SEED

    classifications = {}
    details = {}

    for rule_name, rule_fn in rules.items():
        result = classify_rule(rule_fn, seeds, T, majority, seed)
        classifications[rule_name] = (result["I_positive"], result["Phi_positive"])
        details[rule_name] = result

    return classifications, details


# ══════════════════════════════════════════════════════════════════════
# CLASSIFICATION TABLE & SUMMARY
# ══════════════════════════════════════════════════════════════════════

def classification_cells(classifications):
    """Sort rules into the four (I, Φ) cells.

    Parameters
    ----------
    classifications : dict
        rule_name → (I_positive, Phi_positive)

    Returns
    -------
    dict with keys "(I+, Φ+)", "(I+, Φ-)", "(I-, Φ+)", "(I-, Φ-)"
        Each value is a list of rule names.
    """
    cells = {
        "(I+, Φ+)": [],
        "(I+, Φ-)": [],
        "(I-, Φ+)": [],
        "(I-, Φ-)": [],
    }
    for name, (i_pos, phi_pos) in classifications.items():
        if i_pos and phi_pos:
            cells["(I+, Φ+)"].append(name)
        elif i_pos and not phi_pos:
            cells["(I+, Φ-)"].append(name)
        elif not i_pos and phi_pos:
            cells["(I-, Φ+)"].append(name)
        else:
            cells["(I-, Φ-)"].append(name)
    return cells


def print_classification_table(classifications, details=None, rule_source=None):
    """Print a formatted classification table.

    Parameters
    ----------
    classifications : dict
        rule_name → (I_positive, Phi_positive)
    details : dict or None
        rule_name → classify_rule output (for seed counts).
    rule_source : dict or None
        rule_name → source tag string.
    """
    print("─" * 80)
    print(f"{'Rule':<26} {'I+':<5} {'Φ+':<5} {'I(seeds)':<10} "
          f"{'Φ(seeds)':<10} {'Source'}")
    print("─" * 80)

    for rule_name, (i_pos, phi_pos) in classifications.items():
        i_str = "YES" if i_pos else "no"
        phi_str = "YES" if phi_pos else "no"

        if details and rule_name in details:
            d = details[rule_name]
            i_seeds = f"{d['I_count']}/4"
            phi_seeds = f"{d['Phi_count']}/4"
        else:
            i_seeds = ""
            phi_seeds = ""

        src = rule_source.get(rule_name, "?") if rule_source else ""
        print(f"{rule_name:<26} {i_str:<5} {phi_str:<5} "
              f"{i_seeds:<10} {phi_seeds:<10} {src}")


def print_independence_report(classifications):
    """Print the non-degeneracy and independence diagnostic."""
    cells = classification_cells(classifications)

    i_pos = len(cells["(I+, Φ+)"]) + len(cells["(I+, Φ-)"])
    i_neg = len(cells["(I-, Φ+)"]) + len(cells["(I-, Φ-)"])
    phi_pos = len(cells["(I+, Φ+)"]) + len(cells["(I-, Φ+)"])
    phi_neg = len(cells["(I+, Φ-)"]) + len(cells["(I-, Φ-)"])

    print(f"  I+: {i_pos}, I-: {i_neg}, Φ+: {phi_pos}, Φ-: {phi_neg}")
    print()
    for cell, rules in cells.items():
        print(f"  {cell} ({len(rules)}): {rules}")

    all_populated = all(len(v) > 0 for v in cells.values())
    status = "YES" if all_populated else "NO"
    print(f"\n  All four cells populated: {status}")
    return all_populated


# ══════════════════════════════════════════════════════════════════════
# NULL MODEL: ER random graph baseline
# ══════════════════════════════════════════════════════════════════════

def er_null_model_scores(n=10, p=0.3, trials=20, T=None):
    """Compute I-predicate scores on ER random graph trajectories.

    This establishes the null-model ceiling: no structure should
    yield τ_to_final values above this range.

    Parameters
    ----------
    n : int — ER graph size
    p : float — edge probability
    trials : int — number of independent ER trajectories
    T : int or None

    Returns
    -------
    dict: embedding_name → list of tau_to_final values across trials
    """
    from primo.config import ER_N, ER_P, ER_TRIALS
    n = n or ER_N
    p = p or ER_P
    trials = trials or ER_TRIALS
    T = T if T is not None else T_DEFAULT

    import networkx as nx

    def er_rule(G):
        return nx.erdos_renyi_graph(n, p)

    null_scores = {name: [] for name in EMBEDDING_FUNCTIONS}

    for trial in range(trials):
        np.random.seed(trial * 1000)
        G0 = nx.erdos_renyi_graph(n, p)
        traj = run_trajectory(er_rule, G0, T)
        for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
            scores = measure_I_scores(traj, emb_fn)
            null_scores[emb_name].append(scores["tau_to_final"])

    return null_scores
