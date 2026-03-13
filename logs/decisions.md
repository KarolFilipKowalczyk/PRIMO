# Decision Log

Append-only. Every threshold, design choice, and retraction is documented here with justification.

---

## 2026-03-13: Threshold τ* = 0.5 (I-predicate convergence)

**Decision:** I-predicate convergence threshold set to 0.5.
**Justification:** Above ER null-model ceiling (max τ = 0.31 for laplacian, 0.31 for random_proj, 0.13 for degree_prof across 20 trials). P(τ > 0.5) = 0.00 under all three embeddings. Midpoint between null model (τ ≈ 0) and Bayesian optimum (τ = 1, proved in Paper 2 Theorem 1). No natural gap exists in the score distribution (7.5% pooled).
**Alternatives considered:** 0.4 (too close to ER ceiling), 0.6 (excludes borderline convergent rules like sorting_edges at τ = 0.56).
**Status:** Active.

## 2026-03-13: Threshold ρ* = 0.85 (compression gate)

**Decision:** Compression gate threshold set to 0.85.
**Justification:** All 33 rules have compression ratios well below threshold (max observed: 0.26). Tested under three serialization methods (canonical edge list, adjacency-row, hash-canonical) with zero inconsistencies across 15 original rules. The gate is a permissive filter — it excludes only random/incompressible trajectories.
**Status:** Active.

## 2026-03-13: Threshold ds_std = 0.18 (spectral dimension stability)

**Decision:** Φ-predicate spectral dimension stability threshold set to 0.18.
**Justification:** A 39% gap exists in the ds_std distribution across 33 rules × 4 seeds at 0.18. ER null model has min ds_std = 0.21, mean = 0.32 — all above threshold. P(ds_std < 0.18 | ER) = 0.00.
**Status:** Active.

## 2026-03-13: Provisional straightness gate = 0.45

**Decision:** Grassmannian straightness gate provisionally set to 0.45.
**Justification:** Example B analysis shows contraction mappings have straightness 0.49–0.84, while genuinely I-positive rules (sorting_edges) have straightness ~0.35. The threshold 0.45 separates these populations. However, this has NOT been validated on the full 33-rule set.
**Status:** Provisional — pending exp03 calibration. Do not use for PRIMO enumeration until calibrated.

## 2026-03-13: Retraction of "partially blind one-counter" characterization

**Decision:** The claim that signature 1→2 is equivalent to partially blind counter machines is retracted.
**Reason:** Category error — a counter machine has a finite control that reads input; vertex sprouting has no control and processes no input. See Paper 3 Section 3.2.
**Status:** Permanent.

## 2026-03-13: Example B classified as I-negative (2/4 seeds)

**Decision:** Example B (betweenness-centrality contraction mapping) is classified I-negative under the canonical protocol (4 seeds, T=30, 3 embeddings, majority 3/4).
**Key finding:** The contraction mapping reaches a fixed point within ~5 steps. τ_to_final depends solely on the first 4–5 transient steps. 5/10 random seeds produce I-positive; the canonical 4-seed protocol correctly classifies as I-negative. The adaptive variant (grow-then-contract) is unanimously I-positive — a fundamental limitation of the current predicate.
**Recommended fix:** Active dynamics gate — require edit distance d(G_t, G_{t+1}) > 0 for at least 2T/3 steps.
**Status:** Documented. Fix deferred to exp03 (straightness gate may address the same issue).

## 2026-03-13: PRIMO conjecture scoped to signature level σ (not bit-level K(p))

**Decision:** State the PRIMO conjecture in terms of signature level σ as the primary ordering, not raw Kolmogorov complexity K(p).
**Justification:** Within a single signature level, different rules have different description lengths. The conjecture asks about the first *level* where I-positivity/Φ-positivity appears. Using σ sidesteps the universal-language-dependence problem at short program lengths — σ is formalism-specific but well-defined within DPO. Bit-level ordering within a level is noted as a refinement for future work.
**Status:** Active.

## 2026-03-13: Threshold stability at 51.5% (below 70% target)

**Decision:** Acknowledged that only 17/33 rules (51.5%) have stable classifications under ×0.5 to ×1.5 threshold variation. This is below the 70% target from the PRIMO plan.
**Analysis:** Most instability is at the Φ boundary (rules that are borderline Φ-positive flip when ds_std threshold is scaled). The I-predicate is more stable. The instability is concentrated in rules whose scores sit near threshold boundaries, not in rules whose classifications are clear.
**Status:** Acknowledged. Does not block Paper 1 (which documents this honestly). May improve after straightness gate calibration.

## 2026-03-13: exp02 — Example B contraction mapping analysis

**Decision:** Example B canonical classification confirmed as I-negative (0/4 seeds, stronger than paper's 2/4).
**Key findings:**
- Fixed point reached at step 4 for all canonical seeds (10% active dynamics)
- Random seed robustness: 18/20 I-positive (worse than paper's ~7/20 — predicate is permissive)
- Adaptive variant: 4/4 I-positive (confirms need for straightness gate)
- Straightness discriminator: contraction I- seeds ~0.61, known I+ rules 0.07–0.39
**Status:** Complete. Feeds into exp03.

## 2026-03-13: exp03 — Straightness gate calibrated at 0.35

**Decision:** STRAIGHTNESS_STAR updated from 0.45 (provisional) to 0.35 (calibrated).
**Justification:** Swept thresholds 0.10–0.85 across all 33 rules (4 seeds × 3 embeddings = 264 I+ measurements) plus 14 contraction mapping trajectories (42 measurements). At S*=0.35: zero I+ rules lost, 20/42 contraction measurements rejected. No clean separation exists (overlap zone 0.13–0.50), but 0.35 is the most aggressive safe threshold.
**Impact:** Only `edge_deletion` (I-, S=0.511) has mean straightness above 0.35 — already I-negative.
**Status:** Active. Replaces provisional 0.45.

## 2026-03-13: exp04 — First PRIMO enumeration (signatures ≤ 2→3)

**Decision:** Enumeration at signatures 1→1, 1→2, 2→3 complete. 5 total rules classified with straightness gate.
**Results:**
- σ=1 (1→1): 1 rule — Identity (I- Φ-)
- σ=2 (1→2): 1 rule — Vertex Sprouting (I+ Φ+, S=0.124)
- σ=3 (2→3): 3 rules — all I+ Φ+ (S=0.082–0.122)
- N_I^min = σ=2, N_Φ^min = σ=2 — **tie** (claim (a) inconclusive)
- Claim (b) not supported: all Φ+ rules show increasing I-scores over time (convergence strengthening), no transient decay
- S1 (I-only at shortest lengths): no I-only rules found
- S2 (frequency dominance): I+ = Φ+ at every signature (tied)
**Interpretation:** At this scale (5 rules), all non-trivial growth rules are both I+ and Φ+. The conjecture cannot be tested meaningfully until signature 3→4 provides enough rules for the predicates to separate.
**Status:** Complete. Motivates extending enumeration to 3→4.

## 2026-03-13: Base rates — I+ = 67%, Φ+ = 55% across 33 rules

**Decision:** Documented. Base rates by source: Original (15): I+=53% Φ+=53%. Catalog (5): I+=80% Φ+=60%. Structural (5): I+=100% Φ+=40%. Random DPO (5): I+=60% Φ+=60%. Witnesses (3): I+=67% Φ+=67%.
**Note:** High I+ base rate (67%) raises concern that the I-predicate may be too permissive. The straightness gate should reduce this.
**Status:** Documented.
