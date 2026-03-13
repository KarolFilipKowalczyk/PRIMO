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
