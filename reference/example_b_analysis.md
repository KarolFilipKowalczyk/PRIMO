# Example B Contraction Mapping: Test Results and Analysis

## Executive Summary

**The Example B test produces a QUALIFIED PASS for the I-predicate, but reveals a serious structural weakness that must be addressed before the forward theorem proof has full value.**

Under the exact specification (4 canonical 20-node seeds, T=30, 3 embeddings), Example B classifies as **I-negative** (2/4 seeds, below the 3/4 majority threshold). However:

- **5 out of 10 random 20-node seeds produce I-positive classifications**, with τ values up to 0.71
- **The adaptive variant (grow-then-contract) is unanimously I-positive** (4/4 seeds, all τ > 0.87)
- **The mechanism of failure is now understood**: the contraction mapping reaches a fixed point within ~5 steps, after which all cosines are identically 1.0. The τ_to_final score depends entirely on the behavior during the first 4-5 transient steps.

## Detailed Findings

### 1. The Fixed-Point Problem

The contraction mapping converges to a fixed graph within 4-5 steps for most initial conditions. After convergence:
- All cosines-to-final = 1.0 (the graph IS the final graph)
- The τ_to_final is determined solely by the pre-convergence trajectory
- For a 30-step trajectory where the graph is frozen for steps 5-30, the Kendall τ computation is dominated by the long flat tail of 1.0 values

This means the I-predicate is effectively testing whether the first 4-5 steps show progressive alignment — a much weaker condition than intended.

### 2. Why Some Seeds Are I-Positive

The I-positive cases have initial graphs whose betweenness-centrality structure produces a monotone convergence in the embedding during the first few steps. The I-negative cases have initial graphs that either:
- Start very close to the fixed point (no transient to measure)
- Have a non-monotone transient (embedding oscillates before settling)

The discrimination is an artifact of the initial condition, not a property of the rule.

### 3. The Straightness Signal

The deep diagnostic reveals a potential discriminator:

| System Type | Mean Straightness | Mean τ |
|---|---|---|
| Contraction (I+ cases) | 0.49 | 0.58 |
| Contraction (I- cases) | 0.84 | 0.32 |
| sorting_edges (known I+) | 0.35 | 0.55 |

**Key insight**: Systems that reach a fixed point quickly have high straightness (the Grassmannian trajectory is nearly geodesic — straight line to the fixed point). Genuinely inference-like systems have LOW straightness (the trajectory winds on the Grassmannian as it accumulates evidence). But this relationship is inverted in the τ scores: the fast-converging cases happen to produce higher τ because their monotone transient is clean.

### 4. The Adaptive Variant Failure

The adaptive variant (grow from small seed, then contract) is **unanimously I-positive** with very high τ (0.87-0.92). This is because the growth phase naturally produces progressive alignment (the graph is getting bigger and approaching its final state), and the contraction phase then converges quickly. The I-predicate cannot distinguish "growing toward a target" from "inferring toward a posterior."

This is a more fundamental problem than the fixed-point issue.

## Assessment

### What works:
- The 3-of-4 majority rule saves the exact specification from misclassification
- The compression gate passes easily (all trajectories are highly compressible)
- The anti-convergence guard is not triggered
- hub_sort (the v5 proxy) correctly classifies as I-negative across all seeds

### What doesn't work:
- **The predicate is not robust across initial conditions**: 5/10 random seeds are I-positive
- **The adaptive variant fails completely**: growth-then-contract mimics inference perfectly
- **The discrimination margin is razor-thin**: τ values of 0.49-0.50 for the canonical seeds, just barely below τ*=0.5
- **Fast convergence to a fixed point inflates τ**: the long tail of identical cosines (= 1.0) artificially boosts the Kendall correlation

## Recommendations for Paper 2

### Option A: Proceed with the forward theorem, but strengthen the predicate

The exact specification passes. Paper 2 can proceed with the proof of Steps 4-5, but must add one of the following discriminators to Definition 4:

1. **Convergence rate gate**: Reject trajectories where the graph reaches a fixed point before T/3 steps. Rationale: Bayesian posterior concentration is gradual (O(1/t)), not instantaneous. A system that converges in O(1) steps cannot be implementing posterior updating with meaningful information gain.

2. **Drift curvature criterion** (already suggested in paper2_sketch.md): Measure deviation from Grassmannian geodesics. Contraction mappings follow near-geodesic paths (straightness ≈ 0.5-1.0); Bayesian systems should show evidence-dependent winding (straightness ≈ 0.3-0.4). This is the "drift curvature" idea from Section 5.2.

3. **Active dynamics gate**: Reject trajectories where edit distance d(G_t, G_{t+1}) = 0 for more than T/3 consecutive steps. This directly catches the fixed-point problem.

### Option B: Redesign before proving

If the adaptive variant failure is considered disqualifying, the I-predicate needs a more fundamental change. The core issue is that progressive alignment (τ_to_final > 0.5) is satisfied by ANY system that converges monotonically to a fixed point, which is a much larger class than Bayesian systems. The forward theorem's value depends on the predicate being specific enough that its implication is non-trivial.

### Recommended path: Option A with the active dynamics gate

The active dynamics gate is the simplest, most principled, and most effective fix:

```
Definition 4' (amended): Add to condition (i):
  (iv) Active dynamics: The edit distance d(G_t, G_{t+1}) > 0 for at least 2T/3 steps.
```

This eliminates the fixed-point problem entirely. The contraction mapping fails because it freezes within 5 steps. Bayesian systems (HMM example) continue to evolve at every step as the posterior updates. The gate is also motivated by the forward theorem: Theorem 1(i) bounds the *rate* of alignment by the *per-step KL divergence*, which requires non-trivial updates at every step.

Under this amendment:
- Example B: REJECTED by active dynamics gate (frozen after step 5)
- HMM (Example A): PASSES (updates at every step)
- hub_sort: Already I-negative, remains I-negative
- All v5 I-positive rules: Not affected (none have early fixed points)
