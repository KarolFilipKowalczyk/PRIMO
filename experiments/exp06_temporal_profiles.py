"""
exp06 — Temporal I-profiles of Φ-positive rules (claim (b) evidence).

PRIMO claim (b): "physics-like behavior is the equilibrium of inference-like
dynamics." Concretely: Φ-positive rules should show elevated I-scores early
(inference transient) that decay as the Φ-positive state stabilizes.

Tests 9 Φ+ DPO rules (at ds_std*=0.08) with:
  - T=60 trajectories (double default)
  - Sliding-window I-score and Φ-score profiles (W=8, stride 1)
  - Three transient detection methods:
    A) Early-vs-late comparison (thirds, delta > 0.05)
    B) Kendall τ of I-score vs time (τ < -0.15)
    C) Cross-correlation of I-decay and Φ-stabilization

Usage:
    python experiments/exp06_temporal_profiles.py
    # or: make exp06
"""

import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, MASTER_SEED, MAJORITY_THRESHOLD, EMBEDDING_DIM,
    DS_STD_STAR, DS_INT_DIST, LAW_RESIDUAL_STAR, CURVATURE_KAPPA_STAR,
    DATA_DIR, N_MAX,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog, list_catalogs,
    dpo_rule_to_callable,
)
from primo.predicates import (
    classify_Phi, measure_I_scores, measure_Phi_scores,
)
from primo.trajectories import (
    EMBEDDING_FUNCTIONS, spectral_dimension_estimate, law_residual_score,
    AGGREGATE_QUANTITIES,
)
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

T_LONG = 60          # double default trajectory length
WINDOW_SIZE = 8      # sliding window size
N_SEEDS = 4

# Signatures to scan for Φ+ rules
SIGNATURES = [
    (1, 1),   # σ=1
    (1, 2),   # σ=2
    (2, 3),   # σ=3
    (3, 4),   # σ=4
]

# Transient detection thresholds
EARLY_LATE_DELTA = 0.05     # Method A: early - late > this
KENDALL_TAU_THRESH = -0.15  # Method B: τ < this
# Method C: negative Pearson correlation of I_diff and Phi_stab


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def find_phi_positive_rules():
    """Identify all Φ+ DPO rules across signatures by recomputing classify_Phi."""
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
                traj = run_trajectory(rule_fn, G0, 30)
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


def compute_window_I_scores(subtraj):
    """Compute mean I-score (tau_to_final) across embeddings for a sub-trajectory."""
    scores = {}
    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
        result = measure_I_scores(subtraj, emb_fn)
        scores[emb_name] = result["tau_to_final"]
    vals = [v for v in scores.values() if not np.isnan(v)]
    scores["mean"] = float(np.mean(vals)) if vals else 0.0
    return scores


def compute_window_Phi_positive(subtraj):
    """Check if a sub-trajectory would be Φ-positive on its own.

    Returns (is_positive, ds_std, best_law_resid).
    """
    # Spectral dimension across the window
    ds_values = []
    for G in subtraj:
        ds = spectral_dimension_estimate(G)
        if ds is not None:
            ds_values.append(ds)

    if not ds_values:
        return False, None, None

    ds_mean = np.mean(ds_values)
    ds_std = np.std(ds_values)
    nearest_int = max(1, round(ds_mean))
    ds_int_dist = abs(ds_mean - nearest_int)

    ds_ok = (ds_int_dist < DS_INT_DIST and ds_std < DS_STD_STAR)

    # Law fitting on the sub-trajectory
    best_resid = float('inf')
    for name, fn in AGGREGATE_QUANTITIES.items():
        resid, _ = law_residual_score(subtraj, fn)
        if resid < best_resid:
            best_resid = resid

    law_ok = best_resid < LAW_RESIDUAL_STAR

    return (ds_ok and law_ok), ds_std, best_resid


def compute_temporal_profiles(rule_fn, G0):
    """Compute sliding-window I-score and Φ-score profiles at T=60.

    Returns list of dicts, one per window position.
    """
    np.random.seed(MASTER_SEED)
    traj = run_trajectory(rule_fn, G0, T_LONG, n_max=N_MAX)

    if len(traj) < WINDOW_SIZE + 2:
        return []

    profile = []
    for t in range(len(traj) - WINDOW_SIZE):
        subtraj = traj[t:t + WINDOW_SIZE]

        # I-scores
        i_scores = compute_window_I_scores(subtraj)

        # Φ-scores
        phi_pos, ds_std, law_resid = compute_window_Phi_positive(subtraj)

        profile.append({
            "t": t,
            "I_mean": i_scores["mean"],
            "I_per_embed": {k: v for k, v in i_scores.items() if k != "mean"},
            "Phi_positive": phi_pos,
            "ds_std": ds_std,
            "law_resid": law_resid,
        })

    return profile


# ══════════════════════════════════════════════════════════════════════
# TRANSIENT DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_transient(profile):
    """Apply all three transient detection methods.

    Returns dict with method results and overall verdict.
    """
    if len(profile) < 6:
        return {
            "method_a": {"detected": False, "early": None, "mid": None,
                         "late": None, "delta": None},
            "method_b": {"detected": False, "tau": None},
            "method_c": {"detected": False, "correlation": None,
                         "note": "insufficient data"},
            "verdict": "none",
        }

    means = [p["I_mean"] for p in profile]
    n = len(means)
    third = n // 3

    # ── Method A: early-vs-late comparison ────────────────────────
    early = np.mean(means[:third]) if third > 0 else 0
    mid = np.mean(means[third:2 * third]) if third > 0 else 0
    late = np.mean(means[2 * third:]) if third > 0 else 0
    delta = early - late
    method_a_detected = delta > EARLY_LATE_DELTA

    # ── Method B: Kendall τ of I-score vs time ───────────────────
    tau_val, _ = kendalltau(range(n), means)
    tau_val = float(tau_val) if not np.isnan(tau_val) else 0.0
    method_b_detected = tau_val < KENDALL_TAU_THRESH

    # ── Method C: cross-correlation of I-decay and Φ-stabilization
    phi_states = [1 if p["Phi_positive"] else 0 for p in profile]
    method_c_detected = False
    method_c_corr = None
    method_c_note = None

    # Check if Phi varies
    if len(set(phi_states)) <= 1:
        method_c_note = "N/A — no Φ transition detected"
    else:
        # I_diff[t] = I_mean[t+1] - I_mean[t]
        i_diff = [means[t + 1] - means[t] for t in range(n - 1)]
        # Phi_stab aligned with I_diff
        phi_stab = phi_states[1:]  # align with diff indices

        if len(i_diff) >= 3 and np.std(i_diff) > 1e-10 and np.std(phi_stab) > 1e-10:
            corr, _ = pearsonr(i_diff, phi_stab)
            method_c_corr = float(corr) if not np.isnan(corr) else None
            if method_c_corr is not None and method_c_corr < 0:
                method_c_detected = True
        else:
            method_c_note = "N/A — insufficient variance"

    # Overall verdict
    n_detected = sum([method_a_detected, method_b_detected, method_c_detected])
    if n_detected >= 2:
        verdict = "TRANSIENT"
    elif n_detected == 1:
        verdict = "weak"
    else:
        verdict = "none"

    return {
        "method_a": {
            "detected": method_a_detected,
            "early": float(early),
            "mid": float(mid),
            "late": float(late),
            "delta": float(delta),
        },
        "method_b": {
            "detected": method_b_detected,
            "tau": tau_val,
        },
        "method_c": {
            "detected": method_c_detected,
            "correlation": method_c_corr,
            "note": method_c_note,
        },
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp06(runner):
    """Temporal I-profiles for Φ-positive DPO rules."""
    results_dir = os.path.join(DATA_DIR, "exp06")
    os.makedirs(results_dir, exist_ok=True)

    seeds = make_initial_graphs()

    # ── Part 0: Identify Φ+ rules ────────────────────────────────
    with runner.phase("Identifying Φ-positive rules"):
        phi_pos_rules = find_phi_positive_rules()
        runner.log(f"  Found {len(phi_pos_rules)} Φ+ rules")
        for entry in phi_pos_rules:
            runner.log(f"    {entry['signature']}/{entry['rule_name']} "
                       f"(Φ: {entry['phi_count']}/4)")

    if runner.should_stop():
        return

    runner.set_total(len(phi_pos_rules), N_SEEDS)

    # ── Part 1: Compute temporal profiles ─────────────────────────
    all_profiles = {}  # rule_name -> {seed_name -> profile}
    all_detections = {}  # rule_name -> {seed_name -> detection}

    with runner.phase("Computing temporal I/Φ profiles (T=60)"):
        for entry in phi_pos_rules:
            if runner.should_stop():
                return

            rule_name = entry["rule_name"]
            sig_label = entry["signature"]
            label = f"{sig_label}/{rule_name}"
            runner.begin_rule(label)

            rule_fn = dpo_rule_to_callable(entry["rule_dict"])

            profiles = {}
            detections = {}
            for seed_name, G0 in seeds.items():
                if runner.should_stop():
                    return

                profile = compute_temporal_profiles(rule_fn, G0)
                profiles[seed_name] = profile

                detection = detect_transient(profile)
                detections[seed_name] = detection

                runner.tick(label, seed_name,
                            result=detection["verdict"])

            all_profiles[rule_name] = profiles
            all_detections[rule_name] = detections

            # Quick summary for this rule
            n_transient = sum(
                1 for d in detections.values()
                if d["verdict"] == "TRANSIENT"
            )
            runner.finish_rule(
                label,
                classification=f"{n_transient}/4 transient"
            )

    if runner.should_stop():
        return

    # ── Part 2: Analysis and output ───────────────────────────────
    with runner.phase("Analyzing claim (b)"):
        print()
        print("═" * 78)
        print("TEMPORAL I-PROFILES: CLAIM (b) EVIDENCE")
        print("═" * 78)
        print()

        header = (f"{'Rule':<30} {'Seed':<5} {'Early':>6} {'Mid':>6} "
                  f"{'Late':>6} {'Delta':>7} {'τ(I,t)':>7} "
                  f"{'I-Φ corr':>9} {'Verdict'}")
        print(header)
        print("─" * len(header))

        # Per-rule, per-seed results
        rules_with_any_transient = 0
        rules_with_majority_transient = 0
        all_tau_values = []
        all_corr_values = []

        for entry in phi_pos_rules:
            rule_name = entry["rule_name"]
            detections = all_detections[rule_name]

            n_transient_seeds = 0
            for seed_name in seeds:
                d = detections[seed_name]
                ma = d["method_a"]
                mb = d["method_b"]
                mc = d["method_c"]

                early_str = f"{ma['early']:.3f}" if ma['early'] is not None else "N/A"
                mid_str = f"{ma['mid']:.3f}" if ma['mid'] is not None else "N/A"
                late_str = f"{ma['late']:.3f}" if ma['late'] is not None else "N/A"
                delta_str = f"{ma['delta']:+.3f}" if ma['delta'] is not None else "N/A"
                tau_str = f"{mb['tau']:.3f}" if mb['tau'] is not None else "N/A"

                if mc['correlation'] is not None:
                    corr_str = f"{mc['correlation']:.3f}"
                    all_corr_values.append(mc['correlation'])
                elif mc['note']:
                    corr_str = "N/A"
                else:
                    corr_str = "N/A"

                if mb['tau'] is not None:
                    all_tau_values.append(mb['tau'])

                verdict = d["verdict"]
                if verdict == "TRANSIENT":
                    n_transient_seeds += 1

                print(f"{rule_name:<30} {seed_name:<5} {early_str:>6} "
                      f"{mid_str:>6} {late_str:>6} {delta_str:>7} "
                      f"{tau_str:>7} {corr_str:>9} {verdict}")

            if n_transient_seeds >= 1:
                rules_with_any_transient += 1
            if n_transient_seeds >= 3:
                rules_with_majority_transient += 1

        n_phi_pos = len(phi_pos_rules)
        mean_tau = float(np.mean(all_tau_values)) if all_tau_values else 0.0
        mean_corr = float(np.mean(all_corr_values)) if all_corr_values else None

        print()
        print("AGGREGATE:")
        print(f"  Rules with transient in ≥ 1 seed:   "
              f"{rules_with_any_transient} / {n_phi_pos}")
        print(f"  Rules with transient in ≥ 3 seeds:  "
              f"{rules_with_majority_transient} / {n_phi_pos}  (majority)")
        print(f"  Mean I-decay τ across all Φ+ rules: {mean_tau:.4f}")
        if mean_corr is not None:
            print(f"  Mean I-Φ correlation (where available): {mean_corr:.4f}")
        else:
            print(f"  Mean I-Φ correlation: N/A (no Φ transitions detected)")

        # Claim (b) assessment
        print()
        print("CLAIM (b) ASSESSMENT:")
        if (rules_with_majority_transient > n_phi_pos / 2
                and mean_corr is not None and mean_corr < 0):
            assessment = "SUPPORTED"
            print(f"  SUPPORTED: majority of Φ+ rules show I-positive transient "
                  f"decay anti-correlated with Φ stabilization")
        elif rules_with_any_transient > n_phi_pos / 2:
            assessment = "PARTIAL"
            print(f"  PARTIAL: some Φ+ rules show transient, but not "
                  f"majority with strong anti-correlation")
        elif rules_with_any_transient > 0:
            if mean_corr is not None and mean_corr < 0:
                assessment = "WEAK"
                print(f"  WEAK: transients detected and anti-correlated with Φ "
                      f"stabilization, but not in majority of rules")
            else:
                assessment = "WEAK"
                print(f"  WEAK: transients detected but not anti-correlated "
                      f"with Φ stabilization")
        else:
            assessment = "NOT SUPPORTED"
            print(f"  NOT SUPPORTED: no meaningful transients detected")

    # ── Save results ──────────────────────────────────────────────
    # Full per-rule, per-seed, per-window data
    full_data = {}
    for entry in phi_pos_rules:
        rule_name = entry["rule_name"]
        rule_data = {
            "signature": entry["signature"],
            "phi_count": entry["phi_count"],
            "seeds": {},
        }
        for seed_name in seeds:
            profile = all_profiles[rule_name][seed_name]
            detection = all_detections[rule_name][seed_name]
            rule_data["seeds"][seed_name] = {
                "profile": [
                    {
                        "t": p["t"],
                        "I_mean": p["I_mean"],
                        "I_per_embed": p["I_per_embed"],
                        "Phi_positive": p["Phi_positive"],
                        "ds_std": float(p["ds_std"]) if p["ds_std"] is not None else None,
                        "law_resid": float(p["law_resid"]) if p["law_resid"] is not None else None,
                    }
                    for p in profile
                ],
                "detection": detection,
            }
        full_data[rule_name] = rule_data

    with open(os.path.join(results_dir, "temporal_profiles.json"), "w") as f:
        json.dump(full_data, f, indent=2, default=_json_default)

    # Summary file
    summary = {
        "experiment": "exp06",
        "T": T_LONG,
        "window_size": WINDOW_SIZE,
        "n_phi_pos_rules": n_phi_pos,
        "rules_with_transient_any": rules_with_any_transient,
        "rules_with_transient_majority": rules_with_majority_transient,
        "mean_I_decay_tau": mean_tau,
        "mean_I_Phi_correlation": mean_corr,
        "assessment": assessment,
        "per_rule_summary": {
            entry["rule_name"]: {
                "signature": entry["signature"],
                "transient_seeds": sum(
                    1 for d in all_detections[entry["rule_name"]].values()
                    if d["verdict"] == "TRANSIENT"
                ),
                "mean_tau": float(np.mean([
                    all_detections[entry["rule_name"]][s]["method_b"]["tau"]
                    for s in seeds
                    if all_detections[entry["rule_name"]][s]["method_b"]["tau"] is not None
                ])) if any(
                    all_detections[entry["rule_name"]][s]["method_b"]["tau"] is not None
                    for s in seeds
                ) else None,
            }
            for entry in phi_pos_rules
        },
    }

    with open(os.path.join(results_dir, "claim_b_assessment.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"\n  Results saved to {results_dir}/")

    runner.finish(f"exp06 complete — claim(b): {assessment}")


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
    # Estimate rule count (will be updated once Φ+ rules identified)
    runner = StepRunner(
        "exp06_temporal_profiles",
        total_rules=9,  # expected ~9 Φ+ rules
        total_seeds=N_SEEDS,
        phases=[
            "Identify Φ+ rules",
            "Temporal I/Φ profiles",
            "Analysis",
        ],
    )
    runner.run(run_exp06)
