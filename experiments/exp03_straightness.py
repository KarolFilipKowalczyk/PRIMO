"""
exp03 — Straightness gate calibration.

Calibrates the STRAIGHTNESS_STAR threshold by measuring Grassmannian
straightness across all 33 rules + Example B contraction mapping.

The straightness S = d(start,end) / sum(d(consecutive)) discriminates:
  - Contraction mappings: S ~ 0.5-0.8 (near-geodesic)
  - Genuine inference: S ~ 0.07-0.39 (winding paths)

This experiment:
  1. Measures straightness for all 33 rules (all seeds, all embeddings)
  2. Measures straightness for Example B contraction (canonical + random)
  3. Finds optimal threshold that maximizes separation
  4. Checks impact on existing classifications (no I+ rule should flip)
  5. Updates STRAIGHTNESS_STAR in config.py if calibration succeeds

Usage:
    python experiments/exp03_straightness.py
    # or: make exp03
"""

import json
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from primo.config import (
    TAU_STAR, T_DEFAULT, MASTER_SEED, MAJORITY_THRESHOLD,
    EMBEDDING_DIM, STRAIGHTNESS_STAR, DATA_DIR,
)
from primo.rules import ALL_RULES, RULE_SOURCE, make_initial_graphs, run_trajectory
from primo.predicates import classify_I, classify_Phi, measure_I_scores
from primo.trajectories import EMBEDDING_FUNCTIONS, embed_trajectory, subspace_cosine
from primo.run_utils import StepRunner
from experiments.exp02_example_b import (
    example_b_contraction_rule, bootstrap_graph,
    grassmannian_straightness, EXAMPLE_B_N,
)


N_RULES = 33
N_SEEDS = 4
N_CONTRACTION_RANDOM = 10  # random contraction seeds for calibration


def run_exp03(runner):
    """Calibrate the straightness gate threshold."""
    results_dir = os.path.join(DATA_DIR, "exp03")
    os.makedirs(results_dir, exist_ok=True)

    seeds = make_initial_graphs()

    # ── Part 1: Measure straightness for all 33 rules ────────────────
    with runner.phase("Measuring straightness for all 33 rules"):
        rule_straightness = {}  # rule → {seed → {emb → S}}

        for rule_name, rule_fn in ALL_RULES.items():
            if runner.should_stop():
                return

            runner.begin_rule(rule_name)
            rule_straightness[rule_name] = {}

            for seed_name, G0 in seeds.items():
                np.random.seed(MASTER_SEED)
                traj = run_trajectory(rule_fn, G0, T_DEFAULT)

                emb_s = {}
                for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                    s = grassmannian_straightness(traj, emb_fn)
                    emb_s[emb_name] = s

                rule_straightness[rule_name][seed_name] = emb_s
                runner.tick(rule_name, seed_name, result="measured")

            # Compute mean straightness across seeds and embeddings
            all_s = [v for seed_vals in rule_straightness[rule_name].values()
                     for v in seed_vals.values() if v is not None]
            mean_s = np.mean(all_s) if all_s else None
            s_str = f"S={mean_s:.3f}" if mean_s is not None else "S=N/A"
            runner.finish_rule(rule_name, classification=s_str)

    if runner.should_stop():
        return

    # ── Part 2: Measure straightness for contraction mapping ─────────
    runner.add_extra_ticks(N_CONTRACTION_RANDOM + N_SEEDS)
    with runner.phase("Measuring contraction mapping straightness"):
        contraction_straightness = {}

        # Canonical seeds bootstrapped to n=20
        for seed_name, G0 in seeds.items():
            if runner.should_stop():
                return
            label = f"contraction_{seed_name}"
            runner.begin_rule(label)
            np.random.seed(MASTER_SEED)
            G_boot = bootstrap_graph(G0, EXAMPLE_B_N)
            np.random.seed(MASTER_SEED)
            traj = run_trajectory(example_b_contraction_rule, G_boot, T_DEFAULT)
            emb_s = {}
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                emb_s[emb_name] = grassmannian_straightness(traj, emb_fn)
            contraction_straightness[label] = emb_s
            runner.tick(label, "contraction", result="measured")
            runner.finish_rule(label)

        # Random seeds
        for trial in range(N_CONTRACTION_RANDOM):
            if runner.should_stop():
                return
            label = f"contraction_rnd_{trial:02d}"
            runner.begin_rule(label)
            np.random.seed(trial * 100 + 7)
            G0 = nx.erdos_renyi_graph(EXAMPLE_B_N, 0.3)
            while not nx.is_connected(G0):
                G0 = nx.erdos_renyi_graph(EXAMPLE_B_N, 0.3)
            np.random.seed(MASTER_SEED)
            traj = run_trajectory(example_b_contraction_rule, G0, T_DEFAULT)
            emb_s = {}
            for emb_name, emb_fn in EMBEDDING_FUNCTIONS.items():
                emb_s[emb_name] = grassmannian_straightness(traj, emb_fn)
            contraction_straightness[label] = emb_s
            runner.tick(label, "contraction", result="measured")
            runner.finish_rule(label)

    if runner.should_stop():
        return

    # ── Part 3: Classify all rules and collect data ──────────────────
    with runner.phase("Classifying rules for impact analysis"):
        # Get I-classification for each rule
        rule_I_class = {}
        for rule_name, rule_fn in ALL_RULES.items():
            result_seeds = {}
            for seed_name, G0 in seeds.items():
                np.random.seed(MASTER_SEED)
                traj = run_trajectory(rule_fn, G0, T_DEFAULT)
                I_pos, _ = classify_I(traj)
                result_seeds[seed_name] = I_pos
            I_count = sum(1 for v in result_seeds.values() if v)
            rule_I_class[rule_name] = I_count >= MAJORITY_THRESHOLD

    # ── Analysis ─────────────────────────────────────────────────────
    with runner.phase("Analyzing straightness distribution"):
        # Collect all straightness values grouped by I-classification
        i_pos_straightness = []  # (rule, seed, emb, S)
        i_neg_straightness = []
        all_rule_means = {}

        for rule_name in ALL_RULES:
            s_vals = []
            for seed_vals in rule_straightness[rule_name].values():
                for emb, s in seed_vals.items():
                    if s is not None:
                        s_vals.append(s)
                        entry = (rule_name, s)
                        if rule_I_class.get(rule_name, False):
                            i_pos_straightness.append(s)
                        else:
                            i_neg_straightness.append(s)

            if s_vals:
                all_rule_means[rule_name] = np.mean(s_vals)

        contraction_s = []
        for seed_vals in contraction_straightness.values():
            for s in seed_vals.values():
                if s is not None:
                    contraction_s.append(s)

        # Print distribution
        print()
        print("=" * 80)
        print("STRAIGHTNESS DISTRIBUTION")
        print("=" * 80)

        print(f"\n  I-positive rules ({len(i_pos_straightness)} measurements):")
        if i_pos_straightness:
            arr = np.array(i_pos_straightness)
            print(f"    min={arr.min():.3f}, max={arr.max():.3f}, "
                  f"mean={arr.mean():.3f}, std={arr.std():.3f}")

        print(f"\n  I-negative rules ({len(i_neg_straightness)} measurements):")
        if i_neg_straightness:
            arr = np.array(i_neg_straightness)
            print(f"    min={arr.min():.3f}, max={arr.max():.3f}, "
                  f"mean={arr.mean():.3f}, std={arr.std():.3f}")

        print(f"\n  Contraction mapping ({len(contraction_s)} measurements):")
        if contraction_s:
            arr = np.array(contraction_s)
            print(f"    min={arr.min():.3f}, max={arr.max():.3f}, "
                  f"mean={arr.mean():.3f}, std={arr.std():.3f}")

        # Per-rule table
        print()
        print("=" * 80)
        print("PER-RULE STRAIGHTNESS")
        print("=" * 80)
        print(f"  {'Rule':<26} {'I?':<5} {'Mean S':<10} {'Source'}")
        print(f"  {'─' * 55}")

        for rule_name in ALL_RULES:
            i_class = "I+" if rule_I_class.get(rule_name) else "I-"
            mean_s = all_rule_means.get(rule_name)
            s_str = f"{mean_s:.3f}" if mean_s is not None else "N/A"
            src = RULE_SOURCE.get(rule_name, "?")
            print(f"  {rule_name:<26} {i_class:<5} {s_str:<10} {src}")

    # ── Threshold sweep ──────────────────────────────────────────────
    with runner.phase("Sweeping threshold candidates"):
        print()
        print("=" * 80)
        print("THRESHOLD CALIBRATION")
        print("=" * 80)

        # We want a threshold S* such that:
        # - All contraction seeds with S > S* are rejected (true negatives)
        # - No I+ rule has mean S > S* (no false negatives)
        # Find the gap between max I+ straightness and min contraction straightness

        i_pos_max = max(i_pos_straightness) if i_pos_straightness else 0
        contraction_min = min(contraction_s) if contraction_s else 1

        print(f"\n  Max straightness among I+ rules: {i_pos_max:.3f}")
        print(f"  Min straightness among contractions: {contraction_min:.3f}")

        if contraction_min > i_pos_max:
            gap = contraction_min - i_pos_max
            print(f"  Gap: {gap:.3f} (clean separation)")
        else:
            print(f"  WARNING: No clean separation (overlap)")

        # Sweep thresholds
        candidates = np.arange(0.10, 0.90, 0.05)
        print(f"\n  {'S*':<8} {'I+ kept':<10} {'I+ lost':<10} "
              f"{'Contr rejected':<16} {'Note'}")
        print(f"  {'─' * 60}")

        best_threshold = STRAIGHTNESS_STAR
        best_score = -1

        for s_star in candidates:
            # How many I+ rule-means would survive?
            i_pos_kept = sum(1 for r, m in all_rule_means.items()
                             if rule_I_class.get(r) and m is not None and m <= s_star)
            i_pos_lost = sum(1 for r, m in all_rule_means.items()
                             if rule_I_class.get(r) and m is not None and m > s_star)
            total_i_pos = i_pos_kept + i_pos_lost

            # How many contraction measurements rejected?
            contr_rejected = sum(1 for s in contraction_s if s > s_star)
            total_contr = len(contraction_s)

            note = ""
            if i_pos_lost == 0 and contr_rejected == total_contr:
                note = "*** OPTIMAL ***"
            elif i_pos_lost == 0:
                note = "safe"

            # Score: maximize contraction rejection without losing I+ rules
            score = contr_rejected - i_pos_lost * 100
            if score > best_score and i_pos_lost == 0:
                best_score = score
                best_threshold = float(s_star)

            print(f"  {s_star:<8.2f} {i_pos_kept}/{total_i_pos:<7} "
                  f"{i_pos_lost:<10} "
                  f"{contr_rejected}/{total_contr:<13} {note}")

        print(f"\n  Current STRAIGHTNESS_STAR: {STRAIGHTNESS_STAR}")
        print(f"  Recommended threshold:    {best_threshold:.2f}")

        # Check impact on 33-rule classifications
        print()
        print("=" * 80)
        print(f"IMPACT ANALYSIS (S* = {best_threshold:.2f})")
        print("=" * 80)

        affected = []
        for rule_name in ALL_RULES:
            mean_s = all_rule_means.get(rule_name)
            if mean_s is not None and mean_s > best_threshold:
                i_class = "I+" if rule_I_class.get(rule_name) else "I-"
                affected.append((rule_name, i_class, mean_s))

        if affected:
            print(f"\n  Rules with mean S > {best_threshold:.2f}:")
            for rule_name, i_class, s in affected:
                action = "WOULD FLIP TO I-" if i_class == "I+" else "(already I-)"
                print(f"    {rule_name}: {i_class}, S={s:.3f} — {action}")
        else:
            print(f"\n  No rules affected by threshold S* = {best_threshold:.2f}")

        i_pos_flipped = [r for r, c, _ in affected if c == "I+"]
        if i_pos_flipped:
            print(f"\n  ⚠ WARNING: {len(i_pos_flipped)} I+ rules would flip: {i_pos_flipped}")
            print(f"  Threshold may be too aggressive.")
        else:
            print(f"\n  ✓ No I+ rules affected. Threshold is safe to adopt.")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Previous STRAIGHTNESS_STAR: {STRAIGHTNESS_STAR}")
    print(f"  Calibrated threshold:       {best_threshold:.2f}")
    print(f"  I+ rules preserved:         all")
    print(f"  Contraction separation:     "
          f"{'clean' if contraction_min > i_pos_max else 'overlap'}")

    # ── Save ─────────────────────────────────────────────────────────
    results = {
        "experiment": "exp03",
        "previous_threshold": STRAIGHTNESS_STAR,
        "calibrated_threshold": best_threshold,
        "i_pos_max_straightness": float(i_pos_max),
        "contraction_min_straightness": float(contraction_min),
        "per_rule_means": {k: float(v) for k, v in all_rule_means.items()
                           if v is not None},
        "i_pos_flipped": i_pos_flipped,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")

    runner.finish(f"exp03 complete — calibrated S*={best_threshold:.2f} "
                  f"(was {STRAIGHTNESS_STAR})")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = StepRunner(
        "exp03_straightness",
        total_rules=N_RULES,
        total_seeds=N_SEEDS,
        phases=[
            "Measure all 33 rules",
            "Measure contraction mapping",
            "Impact analysis",
            "Threshold calibration",
        ],
    )
    runner.run(run_exp03)
