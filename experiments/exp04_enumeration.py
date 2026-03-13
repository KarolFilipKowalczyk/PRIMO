"""
exp04 — First PRIMO enumeration (single rules <= 2->3).

Tests the PRIMO conjecture claims at small signature levels:
  sigma=1: signature 1->1 (1 rule: identity)
  sigma=2: signature 1->2 (1 rule: vertex sprouting)
  sigma=3: signature 2->3 (3 connected rules)

Claims tested:
  (a) Ordering: N_I^min < N_Phi^min
      (inference appears at shorter programs than physics)
  (b) Equilibrium: Phi-positive rules show I-positive transient
      that decays as the Phi-positive state stabilizes

Secondary hypotheses:
  S1: I-only programs exist at shortest lengths
  S2: I(N) > Phi(N) at each signature level

Also applies the straightness gate (calibrated in exp03) as a
post-filter on I-positive classifications.

Usage:
    python experiments/exp04_enumeration.py
    # or: make exp04
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, MASTER_SEED, MAJORITY_THRESHOLD,
    EMBEDDING_DIM, STRAIGHTNESS_STAR, DATA_DIR,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog, list_catalogs,
    dpo_rule_to_callable,
)
from primo.predicates import (
    classify_rule, classify_I, classify_Phi, measure_I_scores,
    classification_cells,
)
from primo.trajectories import EMBEDDING_FUNCTIONS, embed_trajectory, subspace_cosine
from primo.run_utils import StepRunner
from experiments.exp02_example_b import grassmannian_straightness


# Signature levels to enumerate, ordered by complexity
SIGNATURES = [
    (1, 1),  # sigma=1
    (1, 2),  # sigma=2
    (2, 3),  # sigma=3
]

WINDOW_SIZE = 8  # sliding window for temporal I-profiles
N_SEEDS = 4


def compute_straightness(traj):
    """Mean Grassmannian straightness across all embeddings."""
    values = []
    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
        s = grassmannian_straightness(traj, emb_fn)
        if s is not None:
            values.append(s)
    return np.mean(values) if values else None


def compute_temporal_I_profile(rule_fn, G0, T=None, window=None):
    """Compute I-score (tau_to_final) over sliding windows.

    Returns list of dicts, one per window position, each with
    per-embedding tau_to_final values.
    """
    T = T or T_DEFAULT
    window = window or WINDOW_SIZE

    np.random.seed(MASTER_SEED)
    traj = run_trajectory(rule_fn, G0, T)

    if len(traj) < window + 2:
        return []

    profile = []
    for t in range(len(traj) - window):
        subtraj = traj[t:t + window]
        scores = {}
        for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
            result = measure_I_scores(subtraj, emb_fn)
            scores[emb_name] = result["tau_to_final"]
        # Mean across embeddings
        vals = [v for v in scores.values() if not np.isnan(v)]
        scores["mean"] = float(np.mean(vals)) if vals else 0.0
        profile.append({"t": t, **scores})

    return profile


def run_exp04(runner):
    """First PRIMO enumeration."""
    results_dir = os.path.join(DATA_DIR, "exp04")
    os.makedirs(results_dir, exist_ok=True)

    seeds = make_initial_graphs()
    all_results = {}  # signature -> list of rule results

    # ── Part 1: Enumerate and classify at each signature level ─────
    total_rules = sum(
        len(load_catalog(f"{l}_{r}")["rules"]) for l, r in SIGNATURES
    )
    runner.add_extra_ticks(total_rules * 2)  # extra for straightness + profiles

    with runner.phase("Enumerating and classifying rules"):
        for l, r in SIGNATURES:
            if runner.should_stop():
                return

            sig_label = f"{l}->{r}"
            catalog = load_catalog(f"{l}_{r}")
            rules = catalog["rules"]
            sig_results = []

            for rule_dict in rules:
                if runner.should_stop():
                    return

                rule_id = rule_dict["id"]
                rule_name = rule_dict.get("name", rule_id)
                runner.begin_rule(f"{sig_label}/{rule_name}")

                rule_fn = dpo_rule_to_callable(rule_dict)

                # Classify with I and Phi predicates
                result = classify_rule(rule_fn, seeds)
                I_pos = result["I_positive"]
                Phi_pos = result["Phi_positive"]

                # Measure straightness (for gate)
                straightness_per_seed = {}
                for seed_name, G0 in seeds.items():
                    np.random.seed(MASTER_SEED)
                    traj = run_trajectory(rule_fn, G0, T_DEFAULT)
                    s = compute_straightness(traj)
                    straightness_per_seed[seed_name] = s
                    runner.tick(f"{sig_label}/{rule_name}", seed_name,
                               result="classified")

                s_vals = [v for v in straightness_per_seed.values()
                          if v is not None]
                mean_straightness = float(np.mean(s_vals)) if s_vals else None

                # Apply straightness gate
                I_pos_gated = I_pos
                if I_pos and mean_straightness is not None:
                    if mean_straightness > STRAIGHTNESS_STAR:
                        I_pos_gated = False

                entry = {
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "signature": sig_label,
                    "I_positive": I_pos,
                    "I_positive_gated": I_pos_gated,
                    "Phi_positive": Phi_pos,
                    "I_count": result["I_count"],
                    "Phi_count": result["Phi_count"],
                    "mean_straightness": mean_straightness,
                    "straightness_per_seed": {
                        k: float(v) if v is not None else None
                        for k, v in straightness_per_seed.items()
                    },
                }
                sig_results.append(entry)

                cell = ("I+" if I_pos_gated else "I-") + \
                       ("Phi+" if Phi_pos else "Phi-")
                runner.finish_rule(f"{sig_label}/{rule_name}",
                                   classification=cell)

            all_results[sig_label] = sig_results

    if runner.should_stop():
        return

    # ── Part 2: Temporal I-profiles for Phi-positive rules ─────────
    phi_pos_rules = []
    for sig_label, sig_results in all_results.items():
        for entry in sig_results:
            if entry["Phi_positive"]:
                phi_pos_rules.append(entry)

    with runner.phase("Computing temporal I-profiles"):
        for entry in phi_pos_rules:
            if runner.should_stop():
                return

            rule_id = entry["rule_id"]
            sig_label = entry["signature"]
            rule_name = entry["rule_name"]
            label = f"profile/{sig_label}/{rule_name}"
            runner.begin_rule(label)

            # Re-load rule
            l, r = sig_label.split("->")
            catalog = load_catalog(f"{l}_{r}")
            rule_dict = next(rd for rd in catalog["rules"]
                            if rd["id"] == rule_id)
            rule_fn = dpo_rule_to_callable(rule_dict)

            # Compute profiles for each seed
            profiles = {}
            for seed_name, G0 in seeds.items():
                profile = compute_temporal_I_profile(rule_fn, G0)
                profiles[seed_name] = profile
                runner.tick(label, seed_name, result="profiled")

            entry["temporal_profiles"] = profiles

            # Check for I-positive transient: is early window I-score
            # higher than late window I-score?
            transient_detected = False
            for seed_name, profile in profiles.items():
                if len(profile) >= 4:
                    early = np.mean([p["mean"] for p in profile[:3]])
                    late = np.mean([p["mean"] for p in profile[-3:]])
                    if early > late + 0.1:  # meaningful decay
                        transient_detected = True
                        break

            entry["transient_detected"] = transient_detected
            runner.finish_rule(label,
                               classification="transient" if transient_detected
                               else "no transient")

    if runner.should_stop():
        return

    # ── Analysis ───────────────────────────────────────────────────
    with runner.phase("Analyzing PRIMO claims"):
        print()
        print("=" * 80)
        print("PRIMO ENUMERATION RESULTS")
        print("=" * 80)

        # Per-signature summary
        for sig_label, sig_results in all_results.items():
            n_total = len(sig_results)
            n_I = sum(1 for e in sig_results if e["I_positive_gated"])
            n_Phi = sum(1 for e in sig_results if e["Phi_positive"])

            print(f"\n  Signature {sig_label}: {n_total} rules")
            print(f"    I+  (gated): {n_I}/{n_total}")
            print(f"    Phi+:        {n_Phi}/{n_total}")

            for entry in sig_results:
                I_str = "I+" if entry["I_positive_gated"] else "I-"
                P_str = "Phi+" if entry["Phi_positive"] else "Phi-"
                s_str = (f"S={entry['mean_straightness']:.3f}"
                         if entry["mean_straightness"] is not None
                         else "S=N/A")
                gate_note = ""
                if entry["I_positive"] and not entry["I_positive_gated"]:
                    gate_note = " [gated out by straightness]"
                print(f"      {entry['rule_name']:<35} {I_str} {P_str}  "
                      f"{s_str}{gate_note}")

        # ── Claim (a): N_I^min < N_Phi^min ─────────────────────────
        print()
        print("=" * 80)
        print("CLAIM (a): ORDERING N_I^min < N_Phi^min")
        print("=" * 80)

        sigma_levels = {
            "1->1": 1,
            "1->2": 2,
            "2->3": 3,
        }

        N_I_min = None
        N_Phi_min = None

        for sig_label, sig_results in all_results.items():
            sigma = sigma_levels[sig_label]
            has_I = any(e["I_positive_gated"] for e in sig_results)
            has_Phi = any(e["Phi_positive"] for e in sig_results)

            if has_I and N_I_min is None:
                N_I_min = sigma
            if has_Phi and N_Phi_min is None:
                N_Phi_min = sigma

        print(f"\n  N_I^min  = sigma {N_I_min if N_I_min else 'not found'}")
        print(f"  N_Phi^min = sigma {N_Phi_min if N_Phi_min else 'not found'}")

        if N_I_min is not None and N_Phi_min is not None:
            if N_I_min < N_Phi_min:
                claim_a = "SUPPORTED"
                print(f"\n  Result: N_I^min ({N_I_min}) < N_Phi^min "
                      f"({N_Phi_min}) -- claim (a) SUPPORTED")
            elif N_I_min == N_Phi_min:
                claim_a = "INCONCLUSIVE"
                print(f"\n  Result: N_I^min = N_Phi^min = {N_I_min} "
                      f"-- claim (a) INCONCLUSIVE (tie)")
            else:
                claim_a = "REFUTED"
                print(f"\n  Result: N_I^min ({N_I_min}) > N_Phi^min "
                      f"({N_Phi_min}) -- claim (a) REFUTED")
        else:
            claim_a = "INSUFFICIENT DATA"
            print("\n  Result: insufficient data (no I+ or Phi+ rules found)")

        # ── Claim (b): Transient I-positive in Phi-positive rules ──
        print()
        print("=" * 80)
        print("CLAIM (b): I-POSITIVE TRANSIENT IN Phi-POSITIVE RULES")
        print("=" * 80)

        if phi_pos_rules:
            n_transient = sum(1 for e in phi_pos_rules
                              if e.get("transient_detected"))
            print(f"\n  Phi-positive rules tested: {len(phi_pos_rules)}")
            print(f"  Transient I-positive detected: {n_transient}/"
                  f"{len(phi_pos_rules)}")

            for entry in phi_pos_rules:
                t_str = ("YES" if entry.get("transient_detected")
                         else "no")
                print(f"    {entry['rule_name']}: transient = {t_str}")

                # Print profile summary for each seed
                for seed_name, profile in entry.get(
                        "temporal_profiles", {}).items():
                    if profile:
                        means = [p["mean"] for p in profile]
                        early = np.mean(means[:3]) if len(means) >= 3 else 0
                        late = np.mean(means[-3:]) if len(means) >= 3 else 0
                        print(f"      {seed_name}: early={early:.3f} "
                              f"late={late:.3f} "
                              f"delta={early - late:+.3f}")

            if n_transient == len(phi_pos_rules) and len(phi_pos_rules) > 0:
                claim_b = "SUPPORTED"
            elif n_transient > 0:
                claim_b = "PARTIAL"
            else:
                claim_b = "NOT SUPPORTED"

            print(f"\n  Claim (b): {claim_b}")
        else:
            claim_b = "NO PHI+ RULES"
            print("\n  No Phi-positive rules found at these signature levels.")

        # ── Secondary hypotheses ───────────────────────────────────
        print()
        print("=" * 80)
        print("SECONDARY HYPOTHESES")
        print("=" * 80)

        # S1: I-only programs at shortest lengths
        print("\n  S1 (Complexity ladder): I-only programs at shortest lengths")
        for sig_label, sig_results in all_results.items():
            I_only = [e for e in sig_results
                      if e["I_positive_gated"] and not e["Phi_positive"]]
            if I_only:
                names = [e["rule_name"] for e in I_only]
                print(f"    {sig_label}: {len(I_only)} I-only rule(s): "
                      f"{', '.join(names)}")
            else:
                print(f"    {sig_label}: no I-only rules")

        # S2: I(N) > Phi(N)
        print("\n  S2 (Frequency dominance): I(N) > Phi(N)")
        for sig_label, sig_results in all_results.items():
            n_I = sum(1 for e in sig_results if e["I_positive_gated"])
            n_Phi = sum(1 for e in sig_results if e["Phi_positive"])
            n = len(sig_results)
            result = "I > Phi" if n_I > n_Phi else (
                "I = Phi" if n_I == n_Phi else "I < Phi")
            print(f"    {sig_label}: I+={n_I}/{n}, Phi+={n_Phi}/{n} "
                  f"-> {result}")

        # ── 2x2 classification table ───────────────────────────────
        print()
        print("=" * 80)
        print("2x2 CLASSIFICATION TABLE (all enumerated rules)")
        print("=" * 80)

        # Build classifications dict using gated I-positive
        all_classifications = {}
        for sig_label, sig_results in all_results.items():
            for entry in sig_results:
                key = f"{sig_label}/{entry['rule_name']}"
                all_classifications[key] = (
                    entry["I_positive_gated"], entry["Phi_positive"])

        cells = classification_cells(all_classifications)
        for cell_name in sorted(cells.keys()):
            members = cells.get(cell_name, [])
            print(f"\n  {cell_name}: {len(members)} rules")
            for m in members:
                print(f"    - {m}")

    # ── Save results ───────────────────────────────────────────────
    # Strip non-serializable temporal profile data for JSON
    save_results = {}
    for sig_label, sig_results in all_results.items():
        save_list = []
        for entry in sig_results:
            save_entry = {k: v for k, v in entry.items()
                          if k != "temporal_profiles"}
            # Summarize temporal profiles
            if "temporal_profiles" in entry:
                profile_summary = {}
                for seed_name, profile in entry["temporal_profiles"].items():
                    if profile:
                        means = [p["mean"] for p in profile]
                        profile_summary[seed_name] = {
                            "early_mean": float(np.mean(means[:3]))
                            if len(means) >= 3 else None,
                            "late_mean": float(np.mean(means[-3:]))
                            if len(means) >= 3 else None,
                            "all_means": [float(m) for m in means],
                        }
                save_entry["temporal_profile_summary"] = profile_summary
            save_list.append(save_entry)
        save_results[sig_label] = save_list

    summary = {
        "experiment": "exp04",
        "signatures_tested": [f"{l}->{r}" for l, r in SIGNATURES],
        "total_rules": sum(len(v) for v in all_results.values()),
        "N_I_min": N_I_min,
        "N_Phi_min": N_Phi_min,
        "claim_a": claim_a,
        "claim_b": claim_b,
        "straightness_gate": STRAIGHTNESS_STAR,
        "per_signature": save_results,
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")

    runner.finish(f"exp04 complete -- N_I^min={N_I_min}, "
                  f"N_Phi^min={N_Phi_min}, claim(a)={claim_a}")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    total_rules = sum(
        len(load_catalog(f"{l}_{r}")["rules"]) for l, r in SIGNATURES
    )
    runner = StepRunner(
        "exp04_enumeration",
        total_rules=total_rules,
        total_seeds=N_SEEDS,
        phases=[
            "Enumerate and classify",
            "Temporal I-profiles",
            "Analysis",
        ],
    )
    runner.run(run_exp04)
