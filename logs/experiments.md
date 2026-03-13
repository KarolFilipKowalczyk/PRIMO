# Experiment Log

Append-only. Every experiment run is logged with git hash, config, duration, and result.

---

## exp04 — PRIMO enumeration extended to 3→4

**Date:** 2026-03-13
**Git hash:** 80d38a7 (base) + uncommitted changes
**Config:** T=30, N_MAX=500, 4 seeds (K1,K2,K3,P3), STRAIGHTNESS_STAR=0.35, DEVICE=cuda
**Signatures:** 1→1 (1 rule), 1→2 (1 rule), 2→3 (3 rules), 3→4 (11 rules) — 16 total
**Duration:** ~3 min

**Results:**
- All 11 σ=4 rules: I+ Φ+ (S=0.050–0.123)
- Claim (a): INCONCLUSIVE (N_I^min = N_Φ^min = σ=2)
- Claim (b): PARTIAL (4/15 Φ+ rules show I-positive transient decay, all at σ=4)
- S1: NOT SUPPORTED (no I-only rules)
- S2: INCONCLUSIVE (I+ = Φ+ at every level)
- 2×2 table: 15 I+Φ+, 0 I+Φ-, 0 I-Φ+, 1 I-Φ- (Identity)

**Key observation:** Triangle matching (l=3) implemented for DPO rule application. Seeds without triangles bootstrap by adding edges to form K3 first.

## exp07 — DPO null-model recalibration sweep

**Date:** 2026-03-13
**Git hash:** TBD (this commit)
**Config:** T=30, N_MAX=500, 4 seeds (K1,K2,K3,P3), tau_star sweep 0.50–0.95, ds_std sweep 0.01–0.18
**Duration:** ~4 min

**Results:**
- DPO null-model ceiling: pooled tau mean=0.70, p95=0.99 — 78% exceed tau*=0.5
- 100% of DPO ds_std values pass current threshold 0.18
- First tau* with >1 I- rule: 0.60
- 174 separating (tau*, ds_std*) combinations found
- All DPO separation is I+Phi- type — no I-Phi+ DPO rules exist
- Cross-check: 33 rules retain all 4 cells at tau*=0.60 and tau*=0.80
- Per-rule scores: tau_max ranges from 0.63 (Path-4 partial preserve) to 1.00 (Diamond fresh vertex)
- Identity is the only I- DPO rule at all thresholds up to tau*=0.55

## exp09 — Eigenvalue gap stability diagnostic

**Date:** 2026-03-13
**Git hash:** d29c327 + uncommitted changes
**Config:** T=30, k=5 (EMBEDDING_DIM), 4 seeds (K1,K2,K3,P3), ds_std*=0.08
**Rules:** All 16 DPO rules at signatures 1→1 through 3→4
**Duration:** ~2 min

**Results:**
- Zero eigenvalue crossings across all 16 rules (all seeds)
- 6/9 Phi+ rules have stable raw gaps; 3/9 "unstable" (Edge Sprouting one-sided, Tri+pendant preserved/shifted)
- 2/7 Phi- rules have stable gaps (Star-3 fresh hub, K4 completion)
- Verdict: PARTIALLY CONFIRMED (raw gap measure)
- Borderline 3 rules have raw gap=0 due to high eigenvalue multiplicity (degeneracy), not crossings

**Key observation:** Raw eigenvalue gap is too strict a measure — eigenvalue degeneracy (multiplicity > 1) gives gap=0 but the eigenSPACE is perfectly stable. Davis-Kahan theorem applies to eigenspaces. Need cluster gap measure (exp09b).

## exp09b — Borderline Phi+ rule confirmation (T=60)

**Date:** 2026-03-13
**Git hash:** d29c327 + uncommitted changes
**Config:** T=60, k=5, 4 seeds, cluster_tol=1e-6
**Rules:** 3 borderline Phi+ rules from exp09
**Duration:** <1 min

**Results:**
- All 3 rules reach sufficient graph size (t_sufficient = 3–7 steps)
- All 12/12 (rule, seed) pairs have stable CLUSTER gaps (YES)
- All 12/12 have zero eigenvalue crossings after sufficient size
- Cluster gap minimums: 0.56–6.0 (well above zero)
- Eigenvalue multiplicity at k: 57–60 (high degeneracy explains raw gap=0)
- Verdict: M3' (eigenspace gap stability) HOLDS for all Phi+ DPO rules

**Key finding:** The 3 "unstable" rules from exp09 produce tree-like graphs where Laplacian eigenvalue 1 has multiplicity ~60. The raw gap (eig[k]-eig[k-1]) is zero because both sit inside this degenerate cluster. But the cluster gap (distance to the next distinct eigenvalue) is large and stable. Davis-Kahan guarantees the embedding subspace is stable under these conditions. Conditional theorem Phi+ ∧ M1 ∧ M2 ∧ M3' → I+ is fully supported for all 16 DPO rules.

## exp06 — Temporal I-profiles for Φ-positive rules (claim (b))

**Date:** 2026-03-13
**Git hash:** TBD (this commit)
**Config:** T=60, W=8 (window), stride=1, ds_std*=0.08, 4 seeds (K1,K2,K3,P3), DEVICE=cuda
**Rules:** 9 Φ+ DPO rules (σ=2: 1, σ=3: 3, σ=4: 5)
**Duration:** ~5 min

**Results:**
- 4/9 Φ+ rules show transient in ≥1 seed
- 2/9 show transient in ≥3 seeds (majority): Path-4 fresh middle (4/4), Tri+pendant shifted (4/4)
- Mean I-decay Kendall τ: -0.017 (near zero overall, but strong in transient rules)
- Mean I-Φ Pearson correlation (where Φ transition exists): -0.076 (weakly anti-correlated)
- Claim (b) assessment: **WEAK**

**Strong transient rules:**
- Path-4 fresh middle: delta +0.19 to +0.27, τ -0.30 to -0.43, I-Φ corr -0.05 to -0.16 (all 4 seeds)
- Tri+pendant shifted: delta +0.28 to +0.30, τ -0.33 to -0.43 (all 4 seeds)

**No transient:**
- Edge Replacement (subdivision): I-scores increasing (τ +0.41 to +0.49)
- Tri+pendant preserved: flat I-scores, no decay
- Diamond fresh vertex: I-scores slightly increasing

**Key observation:** Transient detection is method-dependent. Method A (early-vs-late) and B (Kendall τ) agree well. Method C (I-Φ cross-correlation) is limited by the fact that most Φ+ rules are Φ+ throughout the entire trajectory (no transition from Φ- to Φ+), making cross-correlation undefined. The claim (b) signal is concentrated in 2 rules with clear monotonic I-decay.

## exp10 — Perturbation-response test for claim (b)

**Date:** 2026-03-13
**Git hash:** TBD (this commit)
**Config:** T_burnin=30, T_recovery=30, W=8 (window), perturbation levels 5%/15%/30%, 4 seeds, ds_std*=0.08
**Rules:** 9 Φ+ DPO rules (same as exp06)
**Duration:** ~8 min

**Results:**
- **100% I+ recovery**: 108/108 (rule, seed, perturbation) pairs recover via I-positive trajectories
- Dose-response in τ_to_final: 5%=0.731, 15%=0.797, 30%=0.835 (monotone increasing)
- Elevated-I (early recovery > baseline + 0.05): 5%=42%, 15%=47%, 30%=44%
- 3/9 rules show elevated I at majority of seeds (≥3/4): Edge Sprouting one-sided, Tri+pendant shifted, Path-4 fresh middle (partial)
- Null control (random recovery): 0/36 I+ — clean
- Unperturbed continuation control: 8/36 elevated — slightly noisy
- Claim (b) assessment: **PARTIAL**

**Per-rule highlights:**
- Tri+pendant shifted: strongest signal — I_early well above baseline at all levels, all seeds. Low baseline (τ ≈ 0.04–0.41) makes elevation easy to detect.
- Edge Sprouting (one-sided): clear dose-response — τ increases from 0.39→0.45→0.81 at K1, elevated at larger perturbations
- Vertex Sprouting, Edge Replacement, Triangle completion: very high baseline I-scores (>0.9), so recovery stays high but isn't "elevated" above baseline
- Φ-recovery nearly instant: t_phi_recovery = 0 for almost all pairs (one "never" at Edge Sprouting 15%)

**Key observations:**
1. The 100% I+ recovery rate with 0% null-recovery is the strongest evidence for claim (b) — perturbed Φ+ systems consistently return to equilibrium via I-positive paths, and this is not a property of arbitrary trajectories.
2. The "elevated" criterion is hard to trigger when baseline I-scores are already high (>0.9 for 5/9 rules). The test design favors rules with moderate baseline I (Edge Sprouting, Tri+pendant shifted, Path-4 fresh middle).
3. The positive dose-response in mean τ (monotone increasing with perturbation size) supports the claim that larger perturbations require more "re-inference."
4. Φ-recovery is nearly instantaneous (t=0), suggesting the DPO rule structure itself forces Φ-positive geometry quickly — the I-positive recovery is about the embedding convergence path, not the return of Φ-positivity.

## exp11 — Davis-Kahan ratio ‖ΔA‖_F / gap_k^{cluster}

**Date:** 2026-03-13
**Git hash:** TBD (this commit)
**Config:** T=30, k=5, 4 seeds, cluster_tol=1e-6, Kendall threshold -0.15
**Rules:** All 16 DPO rules at signatures 1→1 through 3→4
**Duration:** <3 min

**Results:**
- Only 3/9 Φ+ rules have majority-decreasing ratio (Edge Sprouting one-sided, Tri+pendant shifted, Edge Sprouting triangle completion)
- Mean ratio across Φ+ rules (latter half): 10.9996 — large, not close to zero
- Mean relative perturbation ‖ΔA‖/‖A‖ across Φ+ rules: 0.3060 — not vanishing
- Identity has N/A ratio (graph doesn't grow → no cluster gap computable)
- Verdict: Davis-Kahan bound is **not tightening** over time for Φ+ DPO rules

**Key observations:**
1. The raw perturbation norms ‖ΔA‖_F are large relative to the cluster gaps, giving ratios >> 1 for most rules. Davis-Kahan gives a bound, not an equality — the eigenvectors can still be stable even when the bound is loose.
2. The relative perturbation ‖ΔA‖/‖A‖ ≈ 0.31 is not decreasing toward zero, meaning each step adds a non-negligible fraction of new edges. Growth adds structure but also adds perturbation.
3. The theorem verification should rely primarily on the gap stability result (exp09/09b) which directly shows eigenspace stability, rather than on the Davis-Kahan bound tightening. The bound explains *why* stability is possible (gaps open), not *how tight* the stability is.
4. Rules with small σ (Vertex Sprouting, σ=3) have the largest ratios (~19) because they add relatively many edges per step. Higher-σ rules add proportionally fewer edges, giving smaller ratios.

## exp12 — Dehn-twist counterexample

**Date:** 2026-03-13
**Git hash:** TBD (this commit)
**Config:** Grid 10×10 (100 nodes), T=30, 4-regular torus
**Duration:** <1 min

**Results:**
- Φ-positive: ds_mean=2.58, ds_std=0.000, law_resid=0.000, curv_CV=0.000
- I-negative: compression=0.028 (PASS), all three τ_to_final ≤ 0 (FAIL convergence gate)
- τ_to_final: laplacian=0.000, random_proj=0.000, degree_prof=-0.093
- Eigenvalue crossings: 0 (spectrum is identical at every step)
- Eigenvalue spectrum difference: 0.000000 (isometry)
- Cosine-to-final: laplacian=[1.0,1.0], random_proj=[1.0,1.0], degree_prof=[0.81,1.0]
- Verdict: **CONFIRMED** as (I−, Φ+) counterexample

**Key observations:**
1. The Dehn twist preserves the spectrum (eigenvalues identical at every step) but rotates eigenvectors within degenerate eigenspaces. This produces τ_to_final = 0 for laplacian and random_proj embeddings (subspace cosine is constant = 1.0, so Kendall τ of a constant sequence is 0).
2. The degree-profile embedding gives τ = -0.093 (slightly negative). Despite the torus being vertex-transitive, the degree-profile embedding is NOT perfectly constant across steps — the embedding depends on node labels (neighbor indices), not just structural features. This gives slight oscillation.
3. The compression ratio is extremely low (0.028) because the trajectory is highly regular — the compression gate passes easily. The convergence gate is what fails.
4. This confirms the Remark in Paper 4: growth (M1) is essential for Φ+ → I+. Without growth, a symmetry-based rewrite can produce Φ+ geometry with no convergence. The Dehn twist is an isometry that rotates the embedding without changing the geometry.
