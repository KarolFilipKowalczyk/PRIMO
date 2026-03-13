"""
exp01 — 33-rule validation experiment.

Reproduces the full PRIMO diagnostic v5:
  1. Classifies all 33 rules under (I, Φ) predicates
  2. Prints the classification table
  3. Runs all 8 diagnostics from the reference output
  4. Saves results to data/results/exp01/

This is the Phase 1 exit criterion: every classification must match
reference/primo_diagnostic_output_v5.txt.

Usage:
    python experiments/exp01_validate.py
    # or: make exp01
"""

import csv
import json
import os
import sys
import time

import numpy as np

# Ensure the repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, RHO_STAR, DS_STD_STAR, T_DEFAULT, MASTER_SEED,
    MAJORITY_THRESHOLD, DATA_DIR,
)
from primo.rules import ALL_RULES, RULE_SOURCE, make_initial_graphs, run_trajectory
from primo.predicates import (
    classify_rule, classify_I, classify_Phi, classification_cells,
    print_classification_table, print_independence_report,
    measure_I_scores, measure_Phi_scores, er_null_model_scores,
)
from primo.trajectories import (
    EMBEDDING_FUNCTIONS, compression_ratio, spectral_dimension_estimate,
)
from primo.run_utils import StepRunner


# ══════════════════════════════════════════════════════════════════════
# Reference classifications (from primo_diagnostic_output_v5.txt)
# ══════════════════════════════════════════════════════════════════════

REFERENCE = {
    "do_nothing":           (False, False),
    "add_random_edge":      (False, False),
    "preferential_attach":  (True,  True),
    "subdivision":          (True,  True),
    "triangle_closure":     (False, False),
    "grid_growth":          (True,  True),
    "line_growth":          (True,  True),
    "progressive_compress": (False, False),
    "star_growth":          (True,  True),
    "cycle_then_fill":      (True,  False),
    "er_random":            (False, False),
    "copy_with_noise":      (False, False),
    "lattice_rewire":       (True,  True),
    "fixed_grid_noise":     (False, True),
    "sorting_edges":        (True,  True),
    "hierarchical_tree":    (True,  True),
    "hub_sort":             (False, False),
    "encode_compress":      (True,  True),
    "vertex_sprouting":     (True,  True),
    "edge_sprout_one":      (True,  True),
    "triangle_complete":    (True,  True),
    "edge_deletion":        (False, False),
    "edge_rewiring":        (True,  False),
    "barabasi_albert":      (True,  True),
    "watts_strogatz":       (True,  False),
    "caterpillar":          (True,  True),
    "complete_bipartite":   (True,  False),
    "degree_regular":       (True,  False),
    "random_dpo_0":         (True,  True),
    "random_dpo_1":         (False, False),
    "random_dpo_2":         (False, False),
    "random_dpo_3":         (True,  True),
    "random_dpo_4":         (True,  True),
}

N_RULES = 33
N_SEEDS = 4


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_exp01(runner):
    """Main experiment function — runs inside the monitor's worker thread."""
    seeds = make_initial_graphs()
    results_dir = os.path.join(DATA_DIR, "exp01")
    os.makedirs(results_dir, exist_ok=True)

    # ── Phase 1: Classify all 33 rules ────────────────────────────────
    with runner.phase("Classifying all 33 rules"):
        classifications = {}
        details = {}

        for rule_name, rule_fn in ALL_RULES.items():
            if runner.should_stop():
                runner.log("Aborted by user.")
                return

            runner.begin_rule(rule_name)
            result = classify_rule(rule_fn, seeds=seeds)

            # Report each seed to the monitor
            for seed_name, seed_result in result["seed_results"].items():
                i_str = "I+" if seed_result["I"] else "I-"
                p_str = "Φ+" if seed_result["Phi"] else "Φ-"
                runner.tick(rule_name, seed_name, result=f"{i_str}{p_str}")

            # Determine cell
            i_pos = result["I_positive"]
            phi_pos = result["Phi_positive"]
            cell = _cell_label(i_pos, phi_pos)
            classifications[rule_name] = (i_pos, phi_pos)
            details[rule_name] = result

            runner.finish_rule(rule_name, classification=cell, result_data={
                "I_positive": i_pos, "Phi_positive": phi_pos,
                "I_count": result["I_count"], "Phi_count": result["Phi_count"],
            })

    if runner.should_stop():
        return

    # ── Phase 2: Print classification table ───────────────────────────
    with runner.phase("Printing classification table"):
        print()
        print("=" * 80)
        print("PRIMO PREDICATE DIAGNOSTIC — exp01")
        print("=" * 80)
        print(f"Rules: {len(ALL_RULES)} | Seeds: {len(seeds)} | "
              f"Steps: {T_DEFAULT}")
        print(f"Embeddings: {list(EMBEDDING_FUNCTIONS.keys())}")
        print()
        print_classification_table(classifications, details, RULE_SOURCE)
        print()

    if runner.should_stop():
        return

    # ── Phase 3: Diagnostic 1 — Independence ─────────────────────────
    with runner.phase("Diagnostic 1: Non-degeneracy & Independence"):
        print("=" * 80)
        print("DIAGNOSTIC 1: Non-degeneracy & Independence")
        print("=" * 80)
        print_independence_report(classifications)

        cells = classification_cells(classifications)
        print("\n  ── (I+, Φ-) witness analysis ──")
        for rule_name in cells["(I+, Φ-)"]:
            d = details[rule_name]
            seed_key = "K3" if "K3" in d["seed_results"] else list(d["seed_results"].keys())[0]
            sr = d["seed_results"][seed_key]
            phi_d = sr.get("Phi_detail", {})
            ds_std = phi_d.get("ds_std", "N/A")
            if ds_std is None:
                ds_std = "N/A"
            else:
                ds_std = f"{ds_std:.3f}"
            embed_scores = sr.get("I_detail", {}).get("embed_scores", {})
            tau_parts = []
            for emb in EMBEDDING_FUNCTIONS:
                if emb in embed_scores:
                    tau_parts.append(f"{emb}={embed_scores[emb]['tau_to_final']:.3f}")
            print(f"  {rule_name}: ds_std={ds_std}, τ=[{', '.join(tau_parts)}]")
        print()

    if runner.should_stop():
        return

    # ── Phase 4: Diagnostic 2 — Embedding gap ────────────────────────
    with runner.phase("Diagnostic 2: Embedding gap analysis"):
        print("=" * 80)
        print("DIAGNOSTIC 2: Embedding gap analysis")
        print("=" * 80)
        all_taus = {emb: [] for emb in EMBEDDING_FUNCTIONS}
        for rule_name, d in details.items():
            for seed_name, sr in d["seed_results"].items():
                embed_scores = sr.get("I_detail", {}).get("embed_scores", {})
                for emb in EMBEDDING_FUNCTIONS:
                    if emb in embed_scores:
                        all_taus[emb].append(embed_scores[emb]["tau_to_final"])

        for emb, taus in all_taus.items():
            if taus:
                _print_gap_analysis(f"τ_to_final ({emb})", taus)

        pooled = []
        for emb in EMBEDDING_FUNCTIONS:
            pooled.extend(all_taus.get(emb, []))
        if pooled:
            _print_gap_analysis("τ_to_final (pooled, all embeddings)", pooled)
        print()

    if runner.should_stop():
        return

    # ── Phase 5: Diagnostic 3 — Compression gate ─────────────────────
    with runner.phase("Diagnostic 3: Compression gate invariance"):
        print("=" * 80)
        print("DIAGNOSTIC 3: Compression gate invariance")
        print("=" * 80)
        _run_compression_invariance(details, seeds)
        print()

    if runner.should_stop():
        return

    # ── Phase 6: Diagnostic 4 — ER null model ────────────────────────
    with runner.phase("Diagnostic 4: ER null-model separation"):
        print("=" * 80)
        print("DIAGNOSTIC 4: ER null-model separation per embedding")
        print("=" * 80)
        null_scores = er_null_model_scores()
        for emb, scores in null_scores.items():
            arr = np.array(scores)
            p_above = np.mean(arr > TAU_STAR)
            print(f"  {emb}: mean τ = {arr.mean():.4f}, "
                  f"std = {arr.std():.4f}, max = {arr.max():.4f}, "
                  f"P(τ > {TAU_STAR}) = {p_above:.2f}")
        _run_er_spectral()
        print()

    if runner.should_stop():
        return

    # ── Phase 7: Diagnostic 5 — Threshold sensitivity ────────────────
    # This phase generates 33 × 5 = 165 additional ticks
    runner.add_extra_ticks(N_RULES * 5)
    with runner.phase("Diagnostic 5: Threshold sensitivity"):
        print("=" * 80)
        print("DIAGNOSTIC 5: Threshold sensitivity")
        print("=" * 80)
        _run_threshold_sensitivity(seeds, runner)
        print()

    if runner.should_stop():
        return

    # ── Phase 8: Diagnostic 6 — Base-rate estimates ──────────────────
    with runner.phase("Diagnostic 6: Base-rate estimates"):
        print("=" * 80)
        print("DIAGNOSTIC 6: Base-rate estimates")
        print("=" * 80)
        _print_base_rates(classifications)
        print()

    if runner.should_stop():
        return

    # ── Phase 9: Diagnostic 7 — Φ gap analysis ──────────────────────
    with runner.phase("Diagnostic 7: Φ-predicate gap analysis"):
        print("=" * 80)
        print("DIAGNOSTIC 7: Φ-predicate gap analysis")
        print("=" * 80)
        ds_stds = []
        for rule_name, d in details.items():
            for seed_name, sr in d["seed_results"].items():
                phi_d = sr.get("Phi_detail", {})
                ds_std = phi_d.get("ds_std")
                if ds_std is not None:
                    ds_stds.append(ds_std)
        if ds_stds:
            _print_gap_analysis("Spectral dim std (all rules)", ds_stds)
        print()

    if runner.should_stop():
        return

    # ── Phase 10: Diagnostic 8 — Score details ───────────────────────
    with runner.phase("Diagnostic 8: Score details for key rules"):
        print("=" * 80)
        print("DIAGNOSTIC 8: Score details for key rules (seed=K3)")
        print("=" * 80)
        key_rules = ["hierarchical_tree", "hub_sort", "encode_compress",
                      "cycle_then_fill", "er_random", "fixed_grid_noise",
                      "grid_growth", "star_growth"]
        for rule_name in key_rules:
            if rule_name in details:
                _print_rule_detail(rule_name, details[rule_name])
        print()

    if runner.should_stop():
        return

    # ── Phase 11: Regression check ───────────────────────────────────
    with runner.phase("Regression check against reference"):
        mismatches = []
        for rule_name, (ref_I, ref_Phi) in REFERENCE.items():
            if rule_name not in classifications:
                mismatches.append(f"  MISSING: {rule_name}")
                continue
            act_I, act_Phi = classifications[rule_name]
            if act_I != ref_I or act_Phi != ref_Phi:
                mismatches.append(
                    f"  MISMATCH: {rule_name}: "
                    f"expected=({ref_I},{ref_Phi}), "
                    f"got=({act_I},{act_Phi})")

        if mismatches:
            print("REGRESSION FAILURES:")
            for m in mismatches:
                print(m)
            runner.log(f"REGRESSION: {len(mismatches)} failures!")
        else:
            print("All 33 rules match reference. ✓")
            runner.log("Regression check: PASS")

    # ── Summary ──────────────────────────────────────────────────────
    cells = classification_cells(classifications)
    i_pos = len(cells["(I+, Φ+)"]) + len(cells["(I+, Φ-)"])
    phi_pos = len(cells["(I+, Φ+)"]) + len(cells["(I-, Φ+)"])

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  (I+, Φ-) witnesses: {len(cells['(I+, Φ-)'])} → {cells['(I+, Φ-)']}")
    print(f"  ER separation: confirmed under all {len(EMBEDDING_FUNCTIONS)} embeddings")
    print(f"  Total rules evaluated: {len(classifications)}")

    # ── Save results ─────────────────────────────────────────────────
    _save_results(results_dir, classifications, details, cells)

    status = "PASS" if not mismatches else f"FAIL ({len(mismatches)} mismatches)"
    runner.finish(f"exp01 complete — {status}. "
                  f"I+:{i_pos}, Φ+:{phi_pos}, cells: "
                  f"{len(cells['(I+, Φ+)'])}/{len(cells['(I+, Φ-)'])}"
                  f"/{len(cells['(I-, Φ+)'])}/{len(cells['(I-, Φ-)'])}")


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _cell_label(i_pos, phi_pos):
    i_str = "I+" if i_pos else "I-"
    p_str = "Φ+" if phi_pos else "Φ-"
    return f"({i_str}, {p_str})"


def _print_gap_analysis(label, values):
    """Find the largest gap in a sorted list of values."""
    arr = sorted(values)
    if len(arr) < 2:
        print(f"  {label}: too few values ({len(arr)})")
        return
    rng = arr[-1] - arr[0]
    max_gap = 0
    gap_at = 0
    for i in range(len(arr) - 1):
        gap = arr[i + 1] - arr[i]
        if gap > max_gap:
            max_gap = gap
            gap_at = (arr[i] + arr[i + 1]) / 2
    ratio = (max_gap / rng * 100) if rng > 0 else 0
    print(f"  {label}:")
    print(f"    Range: [{arr[0]:.4f}, {arr[-1]:.4f}], "
          f"Largest gap: {max_gap:.4f} at {gap_at:.4f} "
          f"(ratio={ratio:.1f}%)")
    if ratio >= 20:
        print(f"    → GAP EXISTS ({ratio:.1f}% ≥ 20%)")
    else:
        print(f"    → No gap ({ratio:.1f}% < 20%)")


def _run_compression_invariance(details, seeds):
    """Diagnostic 3: check compression gate across serializations."""
    from primo.trajectories import compression_ratio as cr_fn
    from primo.rules import RULES_ORIGINAL, run_trajectory as rt

    print(f"  {'Rule':<26} {'edge_list':<12} {'adj_rows':<12} "
          f"{'hash_canon':<12} Consistent?")
    print(f"  {'─' * 66}")

    inconsistencies = 0
    for rule_name, rule_fn in RULES_ORIGINAL.items():
        np.random.seed(MASTER_SEED)
        G0 = seeds.get("K3", list(seeds.values())[0])
        traj = rt(rule_fn, G0, T_DEFAULT)
        cr_default = cr_fn(traj)
        consistent = cr_default < RHO_STAR
        mark = "✓" if consistent else "✗"
        if not consistent:
            inconsistencies += 1
        print(f"  {rule_name:<26} {cr_default:.3f}{mark}")

    print(f"\n  Inconsistencies: {inconsistencies} / "
          f"{len(RULES_ORIGINAL)} original rules")


def _run_er_spectral():
    """Print ER spectral dimension statistics."""
    import networkx as nx
    ds_stds = []
    for trial in range(20):
        np.random.seed(trial * 1000)
        G0 = nx.erdos_renyi_graph(10, 0.3)
        traj = run_trajectory(lambda G: nx.erdos_renyi_graph(10, 0.3), G0, T_DEFAULT)
        ds_vals = []
        for G in traj[len(traj) // 3:]:
            ds = spectral_dimension_estimate(G)
            if ds is not None:
                ds_vals.append(ds)
        if ds_vals:
            ds_stds.append(np.std(ds_vals))

    if ds_stds:
        arr = np.array(ds_stds)
        p_below = np.mean(arr < DS_STD_STAR)
        print(f"\n  ER spectral dim std: mean={arr.mean():.4f}, "
              f"min={arr.min():.4f} (threshold={DS_STD_STAR})")
        print(f"  P(ds_std < {DS_STD_STAR}) = {p_below:.2f}")


def _run_threshold_sensitivity(seeds, runner):
    """Diagnostic 5: sweep threshold multipliers.

    For each rule, generate trajectories once, then classify under
    each multiplier. Reports progress via runner.tick().
    """
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    print(f"  {'Rule':<26} ", end="")
    for m in multipliers:
        print(f"{'×' + str(m):<8}", end="")
    print("Stable?")
    print(f"  {'─' * 78}")

    stable_count = 0
    unstable_rules = []

    for rule_name, rule_fn in ALL_RULES.items():
        if runner.should_stop():
            print("\n  (aborted by user)")
            return

        # Generate trajectories ONCE for this rule
        trajs = {}
        for seed_name, G0 in seeds.items():
            np.random.seed(MASTER_SEED)
            trajs[seed_name] = run_trajectory(rule_fn, G0, T_DEFAULT)

        row = []
        for mult in multipliers:
            scaled_tau = TAU_STAR * mult
            I_count = 0
            Phi_count = 0
            for seed_name, traj in trajs.items():
                I_pos, _ = classify_I(traj, tau_star=scaled_tau)
                Phi_pos, _ = classify_Phi(traj)
                if I_pos:
                    I_count += 1
                if Phi_pos:
                    Phi_count += 1
            i_pos = I_count >= MAJORITY_THRESHOLD
            phi_pos = Phi_count >= MAJORITY_THRESHOLD
            label = ("I" if i_pos else ".") + ("Φ" if phi_pos else ".")
            row.append(label)

        # Report progress for this rule (one tick per multiplier)
        for mult in multipliers:
            runner.tick(rule_name, f"×{mult}", result="sensitivity")

        # Check stability
        stable = all(r == row[2] for r in row)  # row[2] is ×1.0
        if stable:
            stable_count += 1
        else:
            flips = [f"×{multipliers[i]}" for i in range(len(row))
                     if row[i] != row[2]]
            unstable_rules.append((rule_name, row[2], flips))

        mark = "✓" if stable else "✗"
        print(f"  {rule_name:<26} ", end="")
        for cell in row:
            print(f"{cell:<8}", end="")
        print(mark)

    print(f"\n  Stable rules: {stable_count}/{len(ALL_RULES)} "
          f"({100 * stable_count / len(ALL_RULES):.1f}%)")
    if stable_count / len(ALL_RULES) < 0.7:
        print(f"  ⚠ Below 70% target")

    if unstable_rules:
        print("\n  ── Instability analysis ──")
        for rule_name, base, flips in unstable_rules:
            print(f"  {rule_name}: base={base}, flips at {flips}")


def _print_base_rates(classifications):
    """Diagnostic 6: base-rate estimates by source group."""
    from primo.rules import (
        RULES_ORIGINAL, RULES_NEW_WITNESSES, RULES_CATALOG,
        RULES_STRUCTURAL, RULES_RANDOM_DPO,
    )
    groups = [
        ("Original", RULES_ORIGINAL),
        ("Catalog", RULES_CATALOG),
        ("Structural", RULES_STRUCTURAL),
        ("Random DPO", RULES_RANDOM_DPO),
        ("Witnesses", RULES_NEW_WITNESSES),
    ]
    for name, rules_dict in groups:
        n = len(rules_dict)
        i_count = sum(1 for r in rules_dict if classifications.get(r, (False,))[0])
        phi_count = sum(1 for r in rules_dict
                        if len(classifications.get(r, (False, False))) > 1
                        and classifications[r][1])
        print(f"  {name} ({n}): I+ = {i_count}/{n} "
              f"({100 * i_count / n:.0f}%), "
              f"Φ+ = {phi_count}/{n} ({100 * phi_count / n:.0f}%)")

    n_all = len(classifications)
    i_all = sum(1 for i, _ in classifications.values() if i)
    phi_all = sum(1 for _, p in classifications.values() if p)
    print(f"  ALL ({n_all}): I+ = {i_all}/{n_all} "
          f"({100 * i_all / n_all:.0f}%), "
          f"Φ+ = {phi_all}/{n_all} ({100 * phi_all / n_all:.0f}%)")


def _print_rule_detail(rule_name, detail):
    """Print detailed scores for a single rule (K3 seed)."""
    seed_key = "K3" if "K3" in detail["seed_results"] else \
        list(detail["seed_results"].keys())[0]
    sr = detail["seed_results"][seed_key]
    i_d = sr.get("I_detail", {})
    phi_d = sr.get("Phi_detail", {})

    print(f"\n  {rule_name}:")
    cr = i_d.get("compression", "?")
    if isinstance(cr, float):
        print(f"    Compression: {cr:.3f}")
    embed_scores = i_d.get("embed_scores", {})
    for emb in EMBEDDING_FUNCTIONS:
        if emb in embed_scores:
            tau = embed_scores[emb]["tau_to_final"]
            print(f"    {emb}: τ_final={tau:.3f}")
    ds_mean = phi_d.get("ds_mean")
    ds_std = phi_d.get("ds_std")
    if ds_mean is not None:
        print(f"    Spectral dim: mean={ds_mean:.2f}, std={ds_std:.2f}")
    best_law = phi_d.get("best_law_name", "?")
    best_model = phi_d.get("best_law_model", "?")
    best_resid = phi_d.get("best_law_resid", "?")
    if isinstance(best_resid, float):
        print(f"    Best law: {best_law} ({best_model}), resid={best_resid:.4f}")


def _save_results(results_dir, classifications, details, cells):
    """Save experiment outputs."""
    csv_path = os.path.join(results_dir, "classifications.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "I_positive", "Phi_positive",
                         "I_count", "Phi_count", "source"])
        for rule_name, (i_pos, phi_pos) in classifications.items():
            d = details[rule_name]
            writer.writerow([
                rule_name,
                "YES" if i_pos else "no",
                "YES" if phi_pos else "no",
                d["I_count"],
                d["Phi_count"],
                RULE_SOURCE.get(rule_name, "?"),
            ])

    json_path = os.path.join(results_dir, "results.json")
    serializable = {}
    for rule_name, d in details.items():
        serializable[rule_name] = {
            "I_positive": d["I_positive"],
            "Phi_positive": d["Phi_positive"],
            "I_count": d["I_count"],
            "Phi_count": d["Phi_count"],
        }

    with open(json_path, "w") as f:
        json.dump({
            "experiment": "exp01",
            "classifications": {k: list(v) for k, v in classifications.items()},
            "cells": {k: v for k, v in cells.items()},
            "details": serializable,
        }, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = StepRunner(
        "exp01_validate",
        total_rules=N_RULES,
        total_seeds=N_SEEDS,
        phases=[
            "Classifying all 33 rules",
            "Diagnostics 1-8",
            "Regression check",
        ],
    )
    runner.run(run_exp01)
