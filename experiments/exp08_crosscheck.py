"""
exp08 — Cross-check 33 original rules at tightened ds_std* = 0.08.

Verifies that all four (I, Phi) cells remain populated after the
ds_std threshold change from 0.18 to 0.08. This is a hard constraint:
if any cell empties, the threshold is too aggressive.

Also shows which rules changed classification and the new cell counts.

Usage:
    python experiments/exp08_crosscheck.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import TAU_STAR, DS_STD_STAR, T_DEFAULT, MASTER_SEED, DATA_DIR
from primo.rules import ALL_RULES, RULE_SOURCE, make_initial_graphs
from primo.predicates import (
    classify_rule, classification_cells, print_classification_table,
    print_independence_report,
)
from primo.run_utils import StepRunner

# Old reference classifications (at ds_std*=0.18, tau*=0.50)
OLD_REFERENCE = {
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


def run_exp08(runner):
    """Cross-check 33 rules at new ds_std*."""
    seeds = make_initial_graphs()
    results_dir = os.path.join(DATA_DIR, "exp08")
    os.makedirs(results_dir, exist_ok=True)

    # ── Classify all 33 rules ──────────────────────────────────────
    with runner.phase("Classifying all 33 rules"):
        classifications = {}
        details = {}

        for rule_name, rule_fn in ALL_RULES.items():
            if runner.should_stop():
                return

            runner.begin_rule(rule_name)
            result = classify_rule(rule_fn, seeds=seeds)

            for seed_name, seed_result in result["seed_results"].items():
                i_str = "I+" if seed_result["I"] else "I-"
                p_str = "Phi+" if seed_result["Phi"] else "Phi-"
                runner.tick(rule_name, seed_name, result=f"{i_str}{p_str}")

            i_pos = result["I_positive"]
            phi_pos = result["Phi_positive"]
            classifications[rule_name] = (i_pos, phi_pos)
            details[rule_name] = result

            cell = ("I+" if i_pos else "I-") + ("Phi+" if phi_pos else "Phi-")
            runner.finish_rule(rule_name, classification=cell, result_data={
                "I_positive": i_pos, "Phi_positive": phi_pos,
                "I_count": result["I_count"], "Phi_count": result["Phi_count"],
            })

    if runner.should_stop():
        return

    # ── Analysis ───────────────────────────────────────────────────
    with runner.phase("Analyzing changes"):
        print()
        print("=" * 80)
        print(f"CROSS-CHECK: 33 RULES AT ds_std*={DS_STD_STAR}, tau*={TAU_STAR}")
        print("=" * 80)

        # Classification table
        print()
        print_classification_table(classifications, details, RULE_SOURCE)

        # Independence report
        print()
        print("=" * 80)
        print("INDEPENDENCE CHECK")
        print("=" * 80)
        all_populated = print_independence_report(classifications)

        # Changes from old reference
        print()
        print("=" * 80)
        print("CHANGES FROM OLD THRESHOLD (ds_std*=0.18)")
        print("=" * 80)

        changes = []
        for rule_name in classifications:
            old = OLD_REFERENCE.get(rule_name)
            new = classifications[rule_name]
            if old != new:
                old_cell = ("I+" if old[0] else "I-") + ("Phi+" if old[1] else "Phi-")
                new_cell = ("I+" if new[0] else "I-") + ("Phi+" if new[1] else "Phi-")
                changes.append((rule_name, old_cell, new_cell))
                print(f"  {rule_name}: {old_cell} -> {new_cell}")

        if not changes:
            print("  No changes.")
        else:
            print(f"\n  Total changes: {len(changes)}")

        # Per-seed ds_std details for changed rules
        if changes:
            print()
            print("=" * 80)
            print("ds_std DETAILS FOR CHANGED RULES")
            print("=" * 80)
            for rule_name, old_cell, new_cell in changes:
                d = details[rule_name]
                print(f"\n  {rule_name}: {old_cell} -> {new_cell}")
                for seed_name, sr in d["seed_results"].items():
                    phi_d = sr.get("Phi_detail", {})
                    ds_std = phi_d.get("ds_std")
                    ds_str = f"{ds_std:.4f}" if ds_std is not None else "N/A"
                    phi_str = "Phi+" if sr["Phi"] else "Phi-"
                    print(f"    {seed_name}: ds_std={ds_str} -> {phi_str}")

        # HARD CONSTRAINT CHECK
        print()
        print("=" * 80)
        if all_populated:
            print("PASS: All four (I, Phi) cells populated.")
        else:
            print("FAIL: Not all cells populated! Threshold too aggressive.")
        print("=" * 80)

        # Cell counts
        cells = classification_cells(classifications)
        cell_counts = {k: len(v) for k, v in cells.items()}
        old_cells = classification_cells(OLD_REFERENCE)
        old_counts = {k: len(v) for k, v in old_cells.items()}

        print(f"\n  Old: {old_counts}")
        print(f"  New: {cell_counts}")

    # ── Save results ───────────────────────────────────────────────
    save_data = {
        "experiment": "exp08_crosscheck",
        "ds_std_star": DS_STD_STAR,
        "tau_star": TAU_STAR,
        "all_populated": all_populated,
        "cell_counts": cell_counts,
        "old_cell_counts": old_counts,
        "changes": [
            {"rule": r, "old": o, "new": n} for r, o, n in changes
        ],
        "classifications": {
            name: {"I_positive": ip, "Phi_positive": pp,
                   "I_count": details[name]["I_count"],
                   "Phi_count": details[name]["Phi_count"]}
            for name, (ip, pp) in classifications.items()
        },
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")

    status = "PASS" if all_populated else "FAIL"
    runner.finish(f"exp08 {status} -- cells: {cell_counts}")


if __name__ == "__main__":
    runner = StepRunner(
        "exp08_crosscheck",
        total_rules=N_RULES,
        total_seeds=N_SEEDS,
        phases=[
            "Classify all 33 rules",
            "Analysis",
        ],
    )
    runner.run(run_exp08)
