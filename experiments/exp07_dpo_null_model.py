"""
exp07 — DPO null-model recalibration sweep.

Problem: all 16 enumerated DPO rules at sigma >= 2 are I+ Phi+ — the
predicates don't separate. The I-predicate threshold tau* = 0.5 was
calibrated against ER random graphs, but DPO growth rules are all
constructive, so they all pass easily. We need a DPO-specific null
model to recalibrate.

Steps:
  1. Load all 16 enumerated rules, compute full I-score distributions.
  2. Compute DPO null-model ceiling (max/mean/p95 of tau_to_final).
  3. Sweep tau_star from 0.50 to 0.99 — find where I-separation appears.
  4. Sweep ds_std_star from 0.01 to 0.18 — find where Phi-separation appears.
  5. Report (tau_star, ds_std_star) combos that produce predicate separation.
  6. Cross-check: re-classify 33 original rules at promising thresholds.

All sweeps use pre-computed scores (no re-running classify_I/classify_Phi).

Usage:
    python experiments/exp07_dpo_null_model.py
    # or: make exp07
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, DS_STD_STAR, RHO_STAR, STRAIGHTNESS_STAR,
    DS_INT_DIST, LAW_RESIDUAL_STAR, CURVATURE_KAPPA_STAR,
    T_DEFAULT, MASTER_SEED, MAJORITY_THRESHOLD, DATA_DIR,
)
from primo.rules import (
    make_initial_graphs, run_trajectory, load_catalog, ALL_RULES,
    dpo_rule_to_callable,
)
from primo.predicates import (
    measure_I_scores, measure_Phi_scores, classification_cells,
)
from primo.trajectories import EMBEDDING_FUNCTIONS, compression_ratio
from primo.run_utils import StepRunner
from experiments.exp02_example_b import grassmannian_straightness


# ── Configuration ────────────────────────────────────────────────────

SIGNATURES = [
    (1, 1),  # sigma=1
    (1, 2),  # sigma=2
    (2, 3),  # sigma=3
    (3, 4),  # sigma=4
]

TAU_SWEEP = np.arange(0.50, 1.00, 0.05)
DS_STD_SWEEP = np.arange(0.01, 0.19, 0.01)

N_SEEDS = 4


# ══════════════════════════════════════════════════════════════════════
# HELPERS — fast classification from pre-computed scores
# ══════════════════════════════════════════════════════════════════════

def compute_straightness(traj):
    """Mean Grassmannian straightness across all embeddings."""
    values = []
    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
        s = grassmannian_straightness(traj, emb_fn)
        if s is not None:
            values.append(s)
    return np.mean(values) if values else None


def score_seed(traj):
    """Pre-compute all raw scores for a single trajectory.

    Returns a dict of scalar scores that can be thresholded later
    without re-running any embedding or scoring code.
    """
    # I-predicate scores per embedding
    i_scores = {}
    for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
        i_scores[emb_name] = measure_I_scores(traj, emb_fn)

    # Compression ratio
    cr = compression_ratio(traj)

    # Phi-predicate scores
    phi_scores = measure_Phi_scores(traj)

    # Straightness
    straightness = compute_straightness(traj)

    # Extract the key numbers for fast threshold sweeps
    tau_values = {emb: i_scores[emb]["tau_to_final"] for emb in EMBEDDING_FUNCTIONS}
    max_tau = max(tau_values.values())

    return {
        "i_scores": i_scores,
        "phi_scores": phi_scores,
        "cr": cr,
        "straightness": straightness,
        "tau_values": tau_values,
        "max_tau": max_tau,
        "ds_std": phi_scores["ds_std"],
        "ds_int_dist": phi_scores["ds_int_dist"],
        "best_law_resid": phi_scores["best_law_resid"],
        "curv_homogeneity": phi_scores["curv_homogeneity"],
    }


def fast_classify_I(seed_score, tau_star):
    """Classify I-predicate from pre-computed scores."""
    # Compression gate
    if seed_score["cr"] >= RHO_STAR:
        return False

    # Any embedding exceeds threshold
    convergence_pass = any(
        t > tau_star for t in seed_score["tau_values"].values())

    # Anti-convergence guard
    if all(t < 0 for t in seed_score["tau_values"].values()):
        convergence_pass = False

    # Straightness gate
    if convergence_pass and seed_score["straightness"] is not None:
        if seed_score["straightness"] > STRAIGHTNESS_STAR:
            convergence_pass = False

    return convergence_pass


def fast_classify_Phi(seed_score, ds_std_star):
    """Classify Phi-predicate from pre-computed scores."""
    ds_ok = (seed_score["ds_int_dist"] is not None
             and seed_score["ds_int_dist"] < DS_INT_DIST
             and seed_score["ds_std"] is not None
             and seed_score["ds_std"] < ds_std_star)
    law_ok = seed_score["best_law_resid"] < LAW_RESIDUAL_STAR
    curv_ok = seed_score["curv_homogeneity"] < CURVATURE_KAPPA_STAR
    return ds_ok and (law_ok or curv_ok)


def classify_rule_fast(seed_scores, tau_star, ds_std_star):
    """Classify a rule from pre-computed per-seed scores.

    Returns (I_positive, Phi_positive).
    """
    I_count = sum(1 for ss in seed_scores.values()
                  if fast_classify_I(ss, tau_star))
    Phi_count = sum(1 for ss in seed_scores.values()
                    if fast_classify_Phi(ss, ds_std_star))
    return I_count >= MAJORITY_THRESHOLD, Phi_count >= MAJORITY_THRESHOLD


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp07(runner):
    """DPO null-model recalibration."""
    results_dir = os.path.join(DATA_DIR, "exp07")
    os.makedirs(results_dir, exist_ok=True)

    seeds = make_initial_graphs()

    # Extra ticks for cross-check phase (33 rules × 4 seeds × ~4 thresholds)
    runner.add_extra_ticks(33 * 4)

    # ── Part 1: Load all 16 DPO rules, compute full score distributions ──
    all_rules_data = []  # list of {rule_id, rule_name, signature, seed_scores}

    with runner.phase("Computing I/Phi score distributions for all DPO rules"):
        for l, r in SIGNATURES:
            if runner.should_stop():
                return

            sig_label = f"{l}->{r}"
            catalog = load_catalog(f"{l}_{r}")
            rules = catalog["rules"]

            for rule_dict in rules:
                if runner.should_stop():
                    return

                rule_id = rule_dict["id"]
                rule_name = rule_dict.get("name", rule_id)
                runner.begin_rule(f"{sig_label}/{rule_name}")

                rule_fn = dpo_rule_to_callable(rule_dict)

                seed_scores = {}
                for seed_name, G0 in seeds.items():
                    np.random.seed(MASTER_SEED)
                    traj = run_trajectory(rule_fn, G0, T_DEFAULT)
                    seed_scores[seed_name] = score_seed(traj)
                    runner.tick(f"{sig_label}/{rule_name}", seed_name,
                               result="scored")

                all_rules_data.append({
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "signature": sig_label,
                    "seed_scores": seed_scores,
                })
                runner.finish_rule(f"{sig_label}/{rule_name}",
                                   classification="scored")

    if runner.should_stop():
        return

    # ── Part 2: DPO null-model ceiling ───────────────────────────────
    with runner.phase("Computing DPO null-model ceiling"):
        print()
        print("=" * 80)
        print("DPO NULL-MODEL CEILING")
        print("=" * 80)

        # Collect all tau_to_final values per embedding
        dpo_taus = {emb: [] for emb in EMBEDDING_FUNCTIONS}
        dpo_ds_stds = []

        for rd in all_rules_data:
            for seed_name, ss in rd["seed_scores"].items():
                for emb_name in EMBEDDING_FUNCTIONS:
                    dpo_taus[emb_name].append(ss["tau_values"][emb_name])
                if ss["ds_std"] is not None:
                    dpo_ds_stds.append(ss["ds_std"])

        print("\n  I-predicate (tau_to_final) distribution across all DPO rules:")
        for emb_name, taus in dpo_taus.items():
            arr = np.array(taus)
            print(f"    {emb_name}: mean={arr.mean():.4f}, "
                  f"std={arr.std():.4f}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}, "
                  f"p5={np.percentile(arr, 5):.4f}, "
                  f"p50={np.percentile(arr, 50):.4f}, "
                  f"p95={np.percentile(arr, 95):.4f}")

        pooled = []
        for taus in dpo_taus.values():
            pooled.extend(taus)
        pooled = np.array(pooled)
        print(f"\n    POOLED: mean={pooled.mean():.4f}, "
              f"std={pooled.std():.4f}, "
              f"min={pooled.min():.4f}, max={pooled.max():.4f}, "
              f"p5={np.percentile(pooled, 5):.4f}, "
              f"p95={np.percentile(pooled, 95):.4f}")
        print(f"    Current tau*={TAU_STAR}: "
              f"{np.mean(pooled > TAU_STAR)*100:.1f}% of DPO scores exceed it")

        print("\n  Phi-predicate (ds_std) distribution across all DPO rules:")
        ds_arr = np.array(dpo_ds_stds)
        print(f"    mean={ds_arr.mean():.4f}, std={ds_arr.std():.4f}, "
              f"min={ds_arr.min():.4f}, max={ds_arr.max():.4f}, "
              f"p5={np.percentile(ds_arr, 5):.4f}, "
              f"p95={np.percentile(ds_arr, 95):.4f}")
        print(f"    Current ds_std*={DS_STD_STAR}: "
              f"{np.mean(ds_arr < DS_STD_STAR)*100:.1f}% of DPO scores pass")

    if runner.should_stop():
        return

    # ── Part 3: tau_star sweep (fast — uses pre-computed scores) ─────
    with runner.phase("Sweeping tau_star threshold"):
        print()
        print("=" * 80)
        print("TAU_STAR SWEEP (I-predicate recalibration)")
        print("=" * 80)
        print(f"\n  {'tau*':<8} {'I+':<5} {'I-':<5} "
              f"{'Phi+':<5} {'Phi-':<5} "
              f"{'I+Phi+':<8} {'I+Phi-':<8} "
              f"{'I-Phi+':<8} {'I-Phi-':<8} Separates?")
        print(f"  {'─' * 80}")

        tau_sweep_results = []
        first_flip_tau = None

        for tau_val in TAU_SWEEP:
            classifications = {}
            for rd in all_rules_data:
                key = f"{rd['signature']}/{rd['rule_name']}"
                I_pos, Phi_pos = classify_rule_fast(
                    rd["seed_scores"], tau_val, DS_STD_STAR)
                classifications[key] = (I_pos, Phi_pos)

            cells = classification_cells(classifications)
            n_total = len(classifications)
            n_I = len(cells["(I+, Φ+)"]) + len(cells["(I+, Φ-)"])
            n_I_neg = n_total - n_I
            n_Phi = len(cells["(I+, Φ+)"]) + len(cells["(I-, Φ+)"])
            n_Phi_neg = n_total - n_Phi

            separates = (len(cells["(I+, Φ+)"]) > 0 and
                         len(cells["(I-, Φ-)"])  > 0 and
                         (len(cells["(I+, Φ-)"]) > 0 or
                          len(cells["(I-, Φ+)"]) > 0))

            sep_str = "YES" if separates else "no"
            print(f"  {tau_val:<8.2f} {n_I:<5} {n_I_neg:<5} "
                  f"{n_Phi:<5} {n_Phi_neg:<5} "
                  f"{len(cells['(I+, Φ+)']):<8} "
                  f"{len(cells['(I+, Φ-)']):<8} "
                  f"{len(cells['(I-, Φ+)']):<8} "
                  f"{len(cells['(I-, Φ-)']):<8} {sep_str}")

            tau_sweep_results.append({
                "tau_star": float(tau_val),
                "cells": {k: len(v) for k, v in cells.items()},
                "separates": separates,
                "I_plus": n_I,
                "I_minus": n_I_neg,
                "cell_members": {k: list(v) for k, v in cells.items()},
            })

            if first_flip_tau is None and n_I_neg > 1:
                first_flip_tau = tau_val

        if first_flip_tau is not None:
            print(f"\n  First tau* where >1 rule becomes I-: "
                  f"{first_flip_tau:.2f}")
        else:
            print("\n  No tau* in range produced additional I- rules")

    if runner.should_stop():
        return

    # ── Part 4: ds_std_star sweep (fast) ─────────────────────────────
    with runner.phase("Sweeping ds_std_star threshold"):
        print()
        print("=" * 80)
        print("DS_STD_STAR SWEEP (Phi-predicate recalibration)")
        print("=" * 80)
        print(f"\n  {'ds_std*':<9} {'Phi+':<5} {'Phi-':<5} "
              f"{'I+Phi+':<8} {'I+Phi-':<8} "
              f"{'I-Phi+':<8} {'I-Phi-':<8} Separates?")
        print(f"  {'─' * 70}")

        ds_sweep_results = []
        first_flip_ds = None

        for ds_val in DS_STD_SWEEP:
            classifications = {}
            for rd in all_rules_data:
                key = f"{rd['signature']}/{rd['rule_name']}"
                I_pos, Phi_pos = classify_rule_fast(
                    rd["seed_scores"], TAU_STAR, ds_val)
                classifications[key] = (I_pos, Phi_pos)

            cells = classification_cells(classifications)
            n_Phi = len(cells["(I+, Φ+)"]) + len(cells["(I-, Φ+)"])
            n_Phi_neg = len(classifications) - n_Phi

            separates = (len(cells["(I+, Φ+)"]) > 0 and
                         len(cells["(I-, Φ-)"])  > 0 and
                         (len(cells["(I+, Φ-)"]) > 0 or
                          len(cells["(I-, Φ+)"]) > 0))

            sep_str = "YES" if separates else "no"
            print(f"  {ds_val:<9.3f} {n_Phi:<5} {n_Phi_neg:<5} "
                  f"{len(cells['(I+, Φ+)']):<8} "
                  f"{len(cells['(I+, Φ-)']):<8} "
                  f"{len(cells['(I-, Φ+)']):<8} "
                  f"{len(cells['(I-, Φ-)']):<8} {sep_str}")

            ds_sweep_results.append({
                "ds_std_star": float(ds_val),
                "cells": {k: len(v) for k, v in cells.items()},
                "separates": separates,
                "Phi_plus": n_Phi,
                "Phi_minus": n_Phi_neg,
                "cell_members": {k: list(v) for k, v in cells.items()},
            })

            if first_flip_ds is None and n_Phi_neg > 1:
                first_flip_ds = ds_val

        if first_flip_ds is not None:
            print(f"\n  First ds_std* where >1 rule becomes Phi-: "
                  f"{first_flip_ds:.3f}")
        else:
            print("\n  No ds_std* in range produced additional Phi- rules")

    if runner.should_stop():
        return

    # ── Part 5: Joint sweep (fast — instant arithmetic) ──────────────
    with runner.phase("Joint (tau*, ds_std*) sweep for separation"):
        print()
        print("=" * 80)
        print("JOINT SWEEP: (tau*, ds_std*) COMBINATIONS PRODUCING SEPARATION")
        print("=" * 80)

        separating_combos = []

        for tau_val in TAU_SWEEP:
            for ds_val in DS_STD_SWEEP:
                classifications = {}
                for rd in all_rules_data:
                    key = f"{rd['signature']}/{rd['rule_name']}"
                    I_pos, Phi_pos = classify_rule_fast(
                        rd["seed_scores"], tau_val, ds_val)
                    classifications[key] = (I_pos, Phi_pos)

                cells = classification_cells(classifications)
                separates = (len(cells["(I+, Φ+)"]) > 0 and
                             len(cells["(I-, Φ-)"])  > 0 and
                             (len(cells["(I+, Φ-)"]) > 0 or
                              len(cells["(I-, Φ+)"]) > 0))

                if separates:
                    separating_combos.append({
                        "tau_star": float(tau_val),
                        "ds_std_star": float(ds_val),
                        "cells": {k: len(v) for k, v in cells.items()},
                        "cell_members": {k: list(v) for k, v in cells.items()},
                    })

        if separating_combos:
            print(f"\n  Found {len(separating_combos)} separating combinations:")
            print(f"\n  {'tau*':<8} {'ds_std*':<9} "
                  f"{'I+Phi+':<8} {'I+Phi-':<8} "
                  f"{'I-Phi+':<8} {'I-Phi-':<8}")
            print(f"  {'─' * 55}")

            for combo in separating_combos:
                c = combo["cells"]
                print(f"  {combo['tau_star']:<8.2f} "
                      f"{combo['ds_std_star']:<9.3f} "
                      f"{c['(I+, Φ+)']:<8} "
                      f"{c['(I+, Φ-)']:<8} "
                      f"{c['(I-, Φ+)']:<8} "
                      f"{c['(I-, Φ-)']:<8}")

            # Show members at the first separating combo
            best = separating_combos[0]
            print(f"\n  Example: tau*={best['tau_star']:.2f}, "
                  f"ds_std*={best['ds_std_star']:.3f}")
            for cell_name, members in best["cell_members"].items():
                if members:
                    print(f"    {cell_name}: {members}")
        else:
            print("\n  NO separating combination found in the swept range.")
            print("  The DPO null model is uniformly I+ Phi+ "
                  "at all tested thresholds.")

    if runner.should_stop():
        return

    # ── Part 6: Cross-check on 33 original rules ────────────────────
    # Pre-compute scores for all 33 rules (once), then sweep thresholds fast.
    with runner.phase("Cross-checking 33 original rules at promising thresholds"):
        print()
        print("=" * 80)
        print("CROSS-CHECK: 33 ORIGINAL RULES AT RECALIBRATED THRESHOLDS")
        print("=" * 80)

        # Score all 33 rules once
        runner.log("Scoring all 33 rules (one-time cost)...")
        orig_rule_scores = {}  # rule_name -> {seed_name -> seed_score}
        for rule_name, rule_fn in ALL_RULES.items():
            if runner.should_stop():
                return
            seed_scores = {}
            for seed_name, G0 in seeds.items():
                np.random.seed(MASTER_SEED)
                traj = run_trajectory(rule_fn, G0, T_DEFAULT)
                seed_scores[seed_name] = score_seed(traj)
                runner.tick(rule_name, seed_name, result="cross-check")
            orig_rule_scores[rule_name] = seed_scores

        # Pick thresholds to test
        test_thresholds = []
        if separating_combos:
            best = separating_combos[0]
            test_thresholds.append(
                (best["tau_star"], best["ds_std_star"], "first separating"))
        if first_flip_tau is not None:
            test_thresholds.append(
                (float(first_flip_tau), DS_STD_STAR, "first I- flip"))
        test_thresholds.append((0.80, DS_STD_STAR, "tau=0.80"))
        test_thresholds.append((0.90, DS_STD_STAR, "tau=0.90"))

        for tau_test, ds_test, label in test_thresholds:
            print(f"\n  -- tau*={tau_test:.2f}, ds_std*={ds_test:.3f} "
                  f"({label}) --")

            classifications = {}
            for rule_name, seed_scores in orig_rule_scores.items():
                I_pos, Phi_pos = classify_rule_fast(
                    seed_scores, tau_test, ds_test)
                classifications[rule_name] = (I_pos, Phi_pos)

            cells = classification_cells(classifications)
            n_I = len(cells["(I+, Φ+)"]) + len(cells["(I+, Φ-)"])
            n_Phi = len(cells["(I+, Φ+)"]) + len(cells["(I-, Φ+)"])

            print(f"    I+: {n_I}/33, Phi+: {n_Phi}/33")
            for cell_name in sorted(cells.keys()):
                members = cells[cell_name]
                print(f"    {cell_name} ({len(members)}): "
                      f"{', '.join(sorted(members)[:5])}"
                      f"{'...' if len(members) > 5 else ''}")

            all_populated = all(len(v) > 0 for v in cells.values())
            print(f"    All four cells populated: "
                  f"{'YES' if all_populated else 'NO'}")

    if runner.should_stop():
        return

    # ── Per-rule score detail table ──────────────────────────────────
    with runner.phase("Printing per-rule score details"):
        print()
        print("=" * 80)
        print("PER-RULE DPO SCORE DETAILS")
        print("=" * 80)
        print(f"\n  {'Rule':<35} {'Sig':<6} "
              f"{'tau_max':<9} {'tau_mean':<9} "
              f"{'ds_std':<8} {'S':<7} {'Cell(0.5)':<10}")
        print(f"  {'─' * 90}")

        for rd in all_rules_data:
            all_taus = []
            all_ds_stds = []
            all_S = []
            for seed_name, ss in rd["seed_scores"].items():
                for emb_name in EMBEDDING_FUNCTIONS:
                    all_taus.append(ss["tau_values"][emb_name])
                if ss["ds_std"] is not None:
                    all_ds_stds.append(ss["ds_std"])
                if ss["straightness"] is not None:
                    all_S.append(ss["straightness"])

            tau_max = max(all_taus)
            tau_mean = np.mean(all_taus)
            ds_std_mean = np.mean(all_ds_stds) if all_ds_stds else float('nan')
            s_mean = np.mean(all_S) if all_S else float('nan')

            cell = ("I+" if tau_max > TAU_STAR else "I-") + \
                   ("Phi+" if ds_std_mean < DS_STD_STAR else "Phi-")

            print(f"  {rd['rule_name']:<35} {rd['signature']:<6} "
                  f"{tau_max:<9.4f} {tau_mean:<9.4f} "
                  f"{ds_std_mean:<8.4f} {s_mean:<7.3f} {cell:<10}")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n  DPO null-model ceiling (pooled tau_to_final):")
    print(f"    mean={pooled.mean():.4f}, p95={np.percentile(pooled, 95):.4f}, "
          f"max={pooled.max():.4f}")
    print(f"    Current tau*={TAU_STAR} -- "
          f"{np.mean(pooled > TAU_STAR)*100:.1f}% exceed it")

    if separating_combos:
        best = separating_combos[0]
        print(f"\n  Minimum separating thresholds: "
              f"tau*={best['tau_star']:.2f}, "
              f"ds_std*={best['ds_std_star']:.3f}")
        print(f"    Cell counts: {best['cells']}")
    else:
        print(f"\n  No (tau*, ds_std*) combination separates the 16 DPO rules.")
        print(f"  All growth rules are uniformly I+ Phi+ -- the predicates")
        print(f"  lack discriminating power within the DPO class at sigma<=4.")

    if first_flip_tau is not None:
        print(f"\n  First tau* producing >1 I- rule: {first_flip_tau:.2f}")
    if first_flip_ds is not None:
        print(f"\n  First ds_std* producing >1 Phi- rule: {first_flip_ds:.3f}")

    # ── Save results ─────────────────────────────────────────────────
    save_data = {
        "experiment": "exp07",
        "dpo_null_model": {
            "pooled_tau": {
                "mean": float(pooled.mean()),
                "std": float(pooled.std()),
                "min": float(pooled.min()),
                "max": float(pooled.max()),
                "p5": float(np.percentile(pooled, 5)),
                "p95": float(np.percentile(pooled, 95)),
            },
            "ds_std": {
                "mean": float(ds_arr.mean()),
                "std": float(ds_arr.std()),
                "min": float(ds_arr.min()),
                "max": float(ds_arr.max()),
                "p5": float(np.percentile(ds_arr, 5)),
                "p95": float(np.percentile(ds_arr, 95)),
            },
            "per_embedding_tau": {
                emb: {
                    "mean": float(np.mean(dpo_taus[emb])),
                    "std": float(np.std(dpo_taus[emb])),
                    "min": float(np.min(dpo_taus[emb])),
                    "max": float(np.max(dpo_taus[emb])),
                }
                for emb in EMBEDDING_FUNCTIONS
            },
        },
        "tau_sweep": [
            {k: v for k, v in r.items() if k != "cell_members"}
            for r in tau_sweep_results
        ],
        "ds_std_sweep": [
            {k: v for k, v in r.items() if k != "cell_members"}
            for r in ds_sweep_results
        ],
        "separating_combos_count": len(separating_combos),
        "separating_combos": [
            {k: v for k, v in c.items() if k != "cell_members"}
            for c in separating_combos[:20]
        ],
        "first_flip_tau": float(first_flip_tau) if first_flip_tau is not None else None,
        "first_flip_ds": float(first_flip_ds) if first_flip_ds is not None else None,
        "per_rule_scores": [
            {
                "rule_id": rd["rule_id"],
                "rule_name": rd["rule_name"],
                "signature": rd["signature"],
                "tau_values": {
                    seed_name: {
                        emb_name: float(ss["tau_values"][emb_name])
                        for emb_name in EMBEDDING_FUNCTIONS
                    }
                    for seed_name, ss in rd["seed_scores"].items()
                },
                "ds_std_values": {
                    seed_name: float(ss["ds_std"])
                    if ss["ds_std"] is not None else None
                    for seed_name, ss in rd["seed_scores"].items()
                },
                "straightness": {
                    seed_name: float(ss["straightness"])
                    if ss["straightness"] is not None else None
                    for seed_name, ss in rd["seed_scores"].items()
                },
            }
            for rd in all_rules_data
        ],
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")

    summary_msg = (f"exp07 complete -- "
                   f"{len(separating_combos)} separating combos, "
                   f"first I-flip at tau*="
                   f"{first_flip_tau:.2f}" if first_flip_tau is not None
                   else f"exp07 complete -- no I-flip found")
    runner.finish(summary_msg)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    total_rules = sum(
        len(load_catalog(f"{l}_{r}")["rules"]) for l, r in SIGNATURES
    )
    runner = StepRunner(
        "exp07_dpo_null_model",
        total_rules=total_rules,
        total_seeds=N_SEEDS,
        phases=[
            "Score distributions",
            "DPO null-model ceiling",
            "tau_star sweep",
            "ds_std_star sweep",
            "Joint sweep",
            "Cross-check 33 rules",
            "Per-rule details",
        ],
        auto_close_delay=5.0,
    )
    runner.run(run_exp07)
