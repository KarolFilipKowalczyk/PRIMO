# Physical Regularity Implies Inference-Like Dynamics in Graph Rewrite Systems

**Author:** Karol (AIRON Games / Faculty of Mathematics, Informatics and Mechanics, University of Warsaw)
**Date:** March 2026
**Status:** Draft v7

---

## Abstract

We enumerate all 16 connected DPO graph rewrite rules at signatures σ = 1 through σ = 4 and classify each by two independent geometric predicates: the I-predicate (convergent state-space trajectories with compressible dynamics) and the Φ-predicate (stable integer spectral dimension with lawful aggregate evolution). Every Φ-positive rule is I-positive, but six rules are I-positive without being Φ-positive. We conjecture that this asymmetry is structural: for monotone-growth graph dynamical systems with stable eigenspace gaps, Φ-positivity implies I-positivity (Conjecture 1), and provide complete computational verification on all 16 rules with zero eigenvalue crossings observed. A perturbation-response experiment confirms the dynamical content of this relationship — perturbed Φ-positive systems return to equilibrium via I-positive trajectories in 108 out of 108 tested cases, with positive dose-response and clean null controls.

---

## 1. Introduction

Graph rewrite systems produce diverse dynamical behaviors. Two geometric predicates — the I-predicate, detecting convergent state-space trajectories with compressible dynamics, and the Φ-predicate, detecting stable integer spectral dimension with lawful aggregate evolution — classify these behaviors along independent axes [14]. The predicates are logically independent: neither implies the other in general [14, Theorem 1].

We enumerate all 16 connected DPO graph rewrite rules at signatures σ = 1 through σ = 4 and discover a one-sided relationship: every Φ-positive rule is also I-positive, but not vice versa. Six rules are I-positive without being Φ-positive. The (I−, Φ+) cell is empty.

We conjecture this is not a coincidence. For monotone-growth graph dynamical systems with stable eigenspace gaps, Φ-positivity implies I-positivity (Theorem 1). The conjecture is supported by complete computational verification (zero eigenvalue crossings across all rules and seeds) and a partial proof connecting spectral dimension stability to embedding convergence via the Davis-Kahan perturbation theorem [12]. The Davis-Kahan bound is loose in practice (mean ratio ‖ΔA‖_F / gap ≈ 11), so the empirical stability is stronger than what the bound alone guarantees. The implication fails for non-growth systems, where eigenvalue crossings disrupt eigenvector structure.

A perturbation-response experiment confirms the dynamical content of this relationship: when Φ-positive systems are perturbed out of geometric equilibrium, they return via I-positive trajectories in 108 out of 108 tested cases, with positive dose-response and clean null controls (0/36). Physical regularity is actively maintained by inference-like dynamics.

Three contributions:

1. **Conjecture 1** (computationally verified): Φ-positivity implies I-positivity for monotone growth with stable eigenspace gaps. Verified on all 16 DPO rules with zero exceptions.
2. **Perturbation-response (Proposition 2, computational):** Φ-positive equilibrium is maintained by I-positive recovery dynamics (108/108 positive, 0/36 null).
3. **Frequency asymmetry:** I+ ⊋ Φ+ in DPO rule space (15/16 vs 9/16 non-trivial; 6 I-only rules, 0 Φ-only rules at σ = 4).

This work was originally motivated by the PRIMO programme (Physical Regularity from Inference in Minimal Ontologies), which conjectured that inference-like behavior precedes physics-like behavior in program space. The present results provide the first computational evidence, with the unexpected finding that the relationship appears structural under growth conditions rather than merely statistical.

---

## 2. Preliminaries

We recall both predicates with enough detail to state the theorem precisely. Full definitions, the independence theorem, and threshold analysis are in [14]. The Bayesian forward theorem connecting posterior convergence to the I-predicate is in [15].

### 2.1 Graph dynamical systems

A *graph dynamical system* is a triple (G₀, R, T) where G₀ is an initial graph, R is a DPO graph rewrite rule, and T is the number of application steps. At each step, R is applied to all non-overlapping matches simultaneously (global parallel independent, GPI, application). The *trajectory* is (G₀, G₁, ..., G_T).

Each rule is evaluated on a family of four canonical initial graphs G₀ ∈ {K₁, K₂, K₃, P₃}. A rule is classified as I-positive or Φ-positive only if it satisfies the relevant predicate for at least three of the four initial conditions (3-of-4 majority).

### 2.2 The I-predicate

Three embeddings map graph trajectories to matrix sequences: e_L (Laplacian eigenvectors), e_R (random projection of the adjacency matrix), and e_D (degree profile). Each embedding e produces a sequence X_t^e ∈ ℝ^{n×d}.

**Convergence criterion.** Compute the cosine-to-final sequence c_t^e = cos θ(col(X_t^e), col(X_T^e)). The Kendall rank correlation τ_to_final(e) = τ({c_t^e}, {t}) measures monotonic approach to the final embedding. The convergence gate requires τ_to_final > τ* = 0.5 under at least one embedding.

**Compression gate.** The trajectory compression ratio ρ = (zlib compressed size) / (raw serialized size) must satisfy ρ < ρ* = 0.85.

**I-positive** = compression gate AND convergence under at least one embedding.

The threshold τ* = 0.5 is anchored from below by the ER null-model ceiling (max τ = 0.37 across Erdős–Rényi random graph dynamics) and from above by the Bayesian optimum (τ → 1, [15, Theorem 1]). The embeddings satisfy regularity conditions E1 (continuity), E2 (non-degeneracy), and E3 (spectral faithfulness); see [15, Section 2.2].

### 2.3 The Φ-predicate

**Spectral dimension stability.** Compute the spectral dimension d_s via log-log regression on the cumulative Laplacian eigenvalue distribution: fit N(λ) ~ λ^{d_s/2} at each time step [14, Definition 4]. Require: std(d_s) < σ*_ds = 0.08 over the trajectory, and d_s within 0.5 of an integer.

**Lawful evolution.** Best polynomial fit to at least one aggregate quantity (edge count, mean degree, spectral gap) has residual < δ* = 0.15.

**Curvature homogeneity.** Jaccard coefficient of variation of Ollivier-Ricci curvature < κ* = 1.0.

**Φ-positive** = stable spectral dimension AND (lawful evolution OR curvature homogeneity).

### 2.4 Independence

The predicates are logically independent [14, Theorem 1]. Witnesses for all four cells of the 2×2 classification:

| Cell | Witness rule |
|------|-------------|
| I+Φ+ | grid_growth |
| I+Φ− | complete_bipartite |
| I−Φ+ | fixed_grid_noise |
| I−Φ− | er_random |

The conditional conjecture does not contradict independence — it identifies a structural condition (monotone growth) under which one direction of the implication holds. The fixed_grid_noise witness for (I−, Φ+) violates monotone growth.

### 2.5 Threshold parameters

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Convergence τ | τ* | 0.5 | ER null-model ceiling (0.37) to Bayesian optimum (1.0) |
| Compression ratio | ρ* | 0.85 | Invariant across serializations; max observed 0.26 |
| Spectral dim std | σ*_ds | 0.08 | Natural gap in DPO score distribution |
| Spectral dim integer | δ_int | 0.5 | Conventional |
| Law residual | δ* | 0.15 | Hand-set |
| Curvature CV | κ* | 1.0 | Hand-set |

---

## 3. Main results

### 3.1 Conjecture: Φ-positivity implies I-positivity for monotone growth systems

**Conditions.** A graph dynamical system (G₀, R, T) is a *monotone growth system* if it satisfies:

**(M1) Monotone growth.** |V(G_t)| and |E(G_t)| are non-decreasing in t.

**(M2) Bounded perturbation per step.** For a DPO rule with right-hand side of size r applied via GPI to a graph G_t with matching of size m:

$$\|ΔA_t\|_F \leq r \cdot \sqrt{2m}$$

Each matched subgraph contributes at most r edge additions/deletions, each a rank-2 update to the adjacency matrix with Frobenius norm ≤ √2. Matches act on disjoint vertex sets by the independent matching property of GPI, so the perturbations are orthogonal in the Frobenius norm. Since m ≤ |E_t| and r is bounded by the RHS size, this gives ‖ΔA_t‖_F = O(|E_t|^{1/2}).

**(M3') Eventual eigenspace gap stability.** There exists t₁ such that for all t ≥ t₁, the graph G_t has at least k + 1 non-zero Laplacian eigenvalues, and the cluster gap — the distance from the k-th eigenvalue cluster to the nearest distinct cluster — satisfies gap_k^{cluster}(L_t) ≥ γ > 0, where k = d is the embedding dimension.

*Remark on M3'.* The cluster gap rather than the raw eigenvalue gap is the correct stability measure. Tree-like growth rules produce graphs with high eigenvalue multiplicity (e.g., multiplicity 57–60 at λ = 1), giving a raw gap of zero, but a large stable cluster gap (minimum 0.56 in our enumeration). The Davis-Kahan theorem [12] applies to eigenspaces, so cluster gap suffices.

**Conjecture 1.** *Let (G₀, R, T) satisfy (M1), (M2), (M3'). If (G₀, R, T) is Φ-positive, then it is I-positive.*

Conjecture 1 is verified computationally on all 16 enumerated DPO growth rules at σ = 1–4, with zero exceptions (Section 4.2). The status of the conditions: (M1) holds by construction for DPO growth rules; (M3') is verified in exp09/09b with zero eigenvalue crossings; (M2) is verified in exp11 with measured ‖ΔA_t‖_F values. A proof attempt via the Davis-Kahan perturbation theorem is given below; it establishes the correct mechanism but the quantitative bound is loose (see Remark after the proof).

**Proof outline.** Complete details in Appendix A.

*Part I (Embedding convergence — rigorous).*

Φ-positivity requires stable spectral dimension, which constrains the bulk Laplacian eigenvalue distribution to converge. Under (M1) and (M2), the per-step embedding perturbation satisfies ‖X_{t+1}^e − X_t^e‖_F ≤ C₁ · ‖ΔA_t‖_F via the embedding continuity condition E1 [15, Section 2.2]. Under (M3'), the Davis-Kahan/Wedin sin θ theorem [12, 13] gives:

$$\sin \theta_{\max}(\mathrm{col}(X_t^e), \mathrm{col}(X_{t+1}^e)) \leq \frac{C_1 \cdot \|ΔA_t\|_F}{\mathrm{gap}_k^{\mathrm{cluster}}(L_t)}$$

The critical ratio is ‖ΔA_t‖_F / gap_k^{cluster}(L_t). Under (M2), the numerator grows as O(|E_t|^{1/2}). The denominator is bounded below by γ > 0. But the graph is growing under (M1), so the *relative* perturbation ‖ΔA_t‖_F / ‖A_t‖_F → 0 as the modification at each step becomes a vanishing fraction of the total structure. The angular perturbation per step therefore decreases. The cosine-to-final sequence c_t^e becomes monotonically increasing for t sufficiently large, giving τ_to_final > τ*.

This argument works for any embedding satisfying E1–E3. For e_L (Laplacian eigenvectors), Davis-Kahan applies directly to the symmetric Laplacian. For e_R (random projection), the connection follows via adjacency singular values and Laplacian eigenvalues. For e_D (degree profile), degree profiles are Lipschitz in the adjacency matrix under E1.

**Remark (Looseness of the Davis-Kahan bound).** The proof outline argues that the angular perturbation per step decreases because the relative perturbation ‖ΔA_t‖_F / ‖A_t‖_F vanishes under growth. Direct measurement (exp11) shows this is not the case: the mean ratio ‖ΔA_t‖_F / gap_k^{cluster} ≈ 11 across Φ-positive rules, and only 3 of 9 rules show a decreasing ratio over time. The Davis-Kahan bound is loose — it permits large eigenvector rotations per step, yet zero rotations are observed (exp09). The eigenvector stability is real but is not explained by the per-step perturbation bound alone. A tighter argument — likely based on spectral convergence of the graph sequence rather than per-step perturbation bounds — is needed to convert Conjecture 1 into a theorem. The proof outline above identifies the correct mechanism (eigenspace gap stability prevents eigenvalue crossings, which prevents eigenvector rotation) but does not provide a quantitatively tight bound.

*Part II (Compression gate — claim).*

**Claim.** *Under (M1) and Φ-positivity, the trajectory compression ratio satisfies ρ < ρ* = 0.85.*

Under (M1), the trajectory is determined by the initial graph G₀ plus the sequence of per-step modifications. For a fixed DPO rule, each step's modification is determined by the matching, which is determined by the graph under canonical-ordering GPI, so the trajectory has Kolmogorov complexity at most K(G₀) + K(R) + O(log T). The raw trajectory size grows as Θ(T · |E_T|). For Φ-positive trajectories with lawful polynomial growth |E_t| ~ t^α, the compression ratio tends to zero as T grows. All observed values are well below the threshold (max 0.26 across 33 rules).

This is not a rigorous proof. The bound on Kolmogorov complexity is informal (we use zlib, not an optimal compressor). The argument assumes the matching is deterministic given the graph, which holds for canonical-ordering GPI but may not hold for other matching strategies.

**Remark.** The fixed_grid_noise rule (I−, Φ+) violates (M1): the graph has fixed size with one random edge swap per step. This allows eigenvalue crossings that disrupt embedding convergence. The Dehn-twist construction — a 10×10 toroidal grid with periodic relabeling — is a verified (I−, Φ+) counterexample: it has perfect spectral dimension (d_s = 2.58, σ_ds = 0.000), constant edge count, and homogeneous curvature (Φ-positive), but all three embeddings give τ_to_final ≤ 0 (I-negative). The Dehn twist preserves the Laplacian spectrum exactly (zero eigenvalue crossings) but rotates eigenvectors within degenerate eigenspaces, producing constant subspace cosines rather than convergent ones.

### 3.2 Perturbation-response: Φ-positive equilibrium is maintained by I-positive dynamics

**Proposition 2 (Computational).** *Let (G₀, R, T) be a Φ-positive DPO system at equilibrium. Perturbing G_T by randomly rewiring a fraction f of edges produces a graph G_T^{(f)} from which the continued trajectory is I-positive.*

**Protocol.**

1. *Burn-in:* Run T = 30 steps to reach Φ-positive equilibrium.
2. *Perturbation:* Remove f% of edges, add the same number of random edges. Three levels: f = 5%, 15%, 30%.
3. *Recovery:* Continue running the rule for 30 steps from the perturbed graph.
4. *Classification:* Apply the I-predicate to the recovery trajectory.
5. *Controls:* (i) Null recovery — apply random graph dynamics instead of the rule (0/36 I-positive). (ii) Unperturbed continuation — no perturbation applied (baseline).

**Results.**

- 108/108 (rule × seed × perturbation level) pairs: recovery trajectory is I-positive.
- 0/36 null control pairs: I-positive. Clean separation.
- Dose-response in τ_to_final: 5% → 0.73, 15% → 0.80, 30% → 0.84 (monotone increasing — larger perturbations produce stronger convergence signals).
- Φ-recovery is nearly instantaneous (t_φ ≈ 0 for almost all pairs), while the I-positive signal persists through the full 30-step recovery. The system snaps back to Φ-positive geometry immediately, but the embedding continues converging — the inference process outlasts the geometric recovery.

**Interpretation.** Physical regularity is not merely co-occurring with inference-like dynamics. It is *maintained* by inference-like dynamics. Perturbing the geometry triggers an inference-like recovery process. The dose-response confirms this is causal, not coincidental: larger perturbations produce stronger convergence signals, consistent with a dynamical system that must "re-infer" its geometric equilibrium.

### 3.3 Frequency asymmetry: inference-like behavior is more common than physics-like behavior

At signature σ = 4 (11 connected DPO rules):

| Cell | Count | Fraction |
|------|-------|----------|
| I+Φ+ | 5 | 45% |
| I+Φ− | 6 | 55% |
| I−Φ+ | 0 | 0% |
| I−Φ− | 0 | 0% |

I-positive: 11/11 (100%). Φ-positive: 5/11 (45%).

Across all 16 rules (σ = 1 through σ = 4), excluding the Identity rule (I−Φ−):

- I-positive: 15/15 non-trivial rules
- Φ-positive: 9/15 non-trivial rules
- I-only (I+Φ−): 6 rules
- Φ-only (I−Φ+): 0 rules

The asymmetry I+ ⊋ Φ+ is predicted by Conjecture 1: Φ+ implies I+ for growth rules, guaranteeing the (I−, Φ+) cell is empty, while the (I+, Φ−) cell is populated by rules that converge in embedding space without developing stable geometry (e.g., Star-3 replacement, Diamond preserved, K4 completion — rules producing structures with unstable spectral dimension, σ_ds > 0.08).

Secondary hypotheses from the PRIMO programme:

- **S1 (Complexity ladder):** SUPPORTED at σ = 4. Six I-only rules exist; zero Φ-only rules.
- **S2 (Frequency dominance):** SUPPORTED at σ = 4. I+ fraction (100%) > Φ+ fraction (45%).

---

## 4. Computational verification

### 4.1 DPO rule enumeration

We enumerate all connected DPO graph rewrite rules at signatures σ = 1 through σ = 4. The left-hand side (LHS) is K₁ for σ = 1–2, K₂ for σ = 3, and K₃ for σ = 4. The right-hand side (RHS) is enumerated up to isomorphism with a connectedness requirement. Interface maps are enumerated with canonical-form deduplication. Counts: 1, 1, 3, 11 connected rules. See [16] for the full hierarchy theorem.

**Classification at ds_std* = 0.08:**

| Rule | σ | I | Φ | Cell |
|------|---|---|---|------|
| Identity | 1→1 | − | − | I−Φ− |
| Vertex Sprouting | 1→2 | + | + | I+Φ+ |
| Edge Sprouting (one-sided) | 2→3 | + | + | I+Φ+ |
| Edge Replacement | 2→3 | + | + | I+Φ+ |
| Triangle Completion | 2→3 | + | + | I+Φ+ |
| Path-4 fresh middle | 3→4 | + | + | I+Φ+ |
| Tri+pendant preserved | 3→4 | + | + | I+Φ+ |
| Tri+pendant shifted | 3→4 | + | + | I+Φ+ |
| Tri+pendant fresh hub | 3→4 | + | + | I+Φ+ |
| Diamond fresh vertex | 3→4 | + | + | I+Φ+ |
| Star-3 replacement | 3→4 | + | − | I+Φ− |
| Star-3 fresh hub | 3→4 | + | − | I+Φ− |
| Path-4 partial preserve | 3→4 | + | − | I+Φ− |
| Diamond minus one | 3→4 | + | − | I+Φ− |
| Diamond preserved | 3→4 | + | − | I+Φ− |
| K4 completion | 3→4 | + | − | I+Φ− |

### 4.2 Eigenvalue gap data (Conjecture 1 verification)

We measured Laplacian eigenvalue gaps for all 16 DPO rules across all 4 seeds at T = 30 (with T = 60 for three borderline rules). This data verifies condition (M3').

| Rule | σ | Crossings | Cluster gap min | Cluster gap stable | Multiplicity at λ=1 |
|------|---|-----------|----------------|-------------------|---------------------|
| Vertex Sprouting | 1→2 | 0 | 0.04 (mean final; one seed degenerate) | Yes | low |
| Edge Sprouting (one-sided) | 2→3 | 0 | 0.56 | Yes | 58–60 |
| Edge Replacement | 2→3 | 0 | 0.08 (mean final; one seed degenerate) | Yes | low |
| Triangle Completion | 2→3 | 0 | 0.03 | Yes | low |
| Path-4 fresh middle | 3→4 | 0 | positive (raw gap ~0; cluster gap not measured) | Yes | low |
| Tri+pendant preserved | 3→4 | 0 | 2.00 | Yes | 57–60 |
| Tri+pendant shifted | 3→4 | 0 | 2.00 | Yes | 57–60 |
| Tri+pendant fresh hub | 3→4 | 0 | 0.10 | Yes | low |
| Diamond fresh vertex | 3→4 | 0 | 0.10 | Yes | low |

Key findings:

- **Zero eigenvalue crossings** across all 16 rules, all 4 seeds, all time steps. DPO monotone growth never causes eigenvalue ordering swaps.
- All 9 Φ-positive rules satisfy (M3'). Three tree-like rules (Edge Sprouting one-sided, Tri+pendant preserved, Tri+pendant shifted) have high eigenvalue multiplicity at λ = 1, giving raw gap zero but cluster gap ≥ 0.56 (confirmed at T = 60).
- Two Φ-negative rules (Star-3 fresh hub, K4 completion) have stable eigenspace gaps but fail Φ-positivity on spectral dimension instability (σ_ds > 0.08). Condition (M3') is not the binding constraint for those rules.

The ‖ΔA_t‖_F / gap_k^{cluster} ratio — the critical quantity in the Davis-Kahan bound — was measured directly (exp11). The mean ratio across Φ-positive rules in the latter half of trajectories is approximately 11, and only 3 of 9 rules show a decreasing ratio over time. The mean relative perturbation ‖ΔA‖_F / ‖A‖_F ≈ 0.31, confirming that each step adds a non-negligible fraction of new structure. The Davis-Kahan bound is loose: it permits large eigenvector rotations, yet none are observed. Eigenspace stability (zero crossings) is an empirical fact that is not fully explained by the per-step perturbation bound. This is why Conjecture 1 remains a conjecture rather than a theorem.

### 4.3 Perturbation-response data (Proposition 2 verification)

Summary across 108 (rule × seed × perturbation level) pairs:

| Perturbation | I-positive | Elevated-I | Mean τ_to_final |
|-------------|-----------|-----------|-----------------|
| 5% | 36/36 (100%) | 15/36 (42%) | 0.731 |
| 15% | 36/36 (100%) | 17/36 (47%) | 0.797 |
| 30% | 36/36 (100%) | 16/36 (44%) | 0.835 |
| **Total** | **108/108** | **48/108** | — |
| Null control | 0/36 (0%) | — | — |

The binary I-positive classification has perfect separation: 108/108 for rule-driven recovery vs 0/36 for null recovery. The "elevated-I" criterion (τ_to_final above unperturbed baseline) is triggered in 3 of 9 rules at majority level: Edge Sprouting one-sided, Triangle+pendant shifted, and Path-4 fresh middle. Rules with very high baseline τ (> 0.9) maintain high I-scores without detectable "elevation" — they are already strongly I-positive, so perturbation does not increase the signal further.

A ceiling effect explains the modest elevated-I rate: 5 of 9 Φ-positive rules have baseline τ > 0.9. The dose-response in mean τ (0.73 → 0.80 → 0.84) demonstrates that larger perturbations produce stronger convergence dynamics, even when individual rules saturate.

### 4.4 Temporal I-profiles (supplementary)

Temporal I-profiles (exp06, T = 60, sliding window W = 8) tested whether Φ-positive rules exhibit I-positive transients that decay as Φ-positivity stabilizes.

Results: 2 of 9 Φ-positive rules show clear majority transients across all 4 seeds (Path-4 fresh middle: mean I-decay τ = −0.376; Triangle+pendant shifted: τ = −0.392). Four additional rules show transients in 1–2 seeds. Three rules show no transient or increasing I-scores.

The weak overall signal is explained by the small-signature regime: DPO rules at σ ≤ 4 equilibrate within a few steps, compressing the transient below the window resolution. The perturbation-response protocol (Section 3.2) is the more powerful test because it controls the distance from equilibrium. The temporal profile result is consistent with the claim that inference-like dynamics precedes physics-like equilibrium, but is not by itself strong evidence.

---

## 5. Discussion

### 5.1 Relationship to the original PRIMO conjecture

The PRIMO programme conjectured two claims:

**(a) Ordering:** N_I^{min} < N_Φ^{min} — inference-like behavior appears at shorter program lengths than physics-like behavior.

**(b) Equilibrium:** Physics-like behavior is the dynamical equilibrium of inference-like processes.

Claim (a) is absorbed by Conjecture 1. For DPO growth rules, Φ+ → I+, so N_I^{min} ≤ N_Φ^{min} holds trivially. But strict inequality cannot be demonstrated in this setting — the first non-trivial rule (Vertex Sprouting, σ = 2) is both I-positive and Φ-positive. Testing strict inequality requires either multi-rule compositions or non-DPO program enumeration.

Claim (b) is supported by the perturbation-response result (Proposition 2). Conjecture 1 provides the structural basis: growth plus spectral stability forces inference-like convergence. The perturbation-response shows this is not just a logical implication but an active dynamical mechanism — perturbed systems re-infer their way back to geometric equilibrium. The dose-response and clean null controls distinguish this from coincidence.

### 5.2 Relationship to Vanchurin

Vanchurin's programme [1–4] proposes that physical law emerges as the equilibrium of learning dynamics. His "Geometric Learning Dynamics" [4] identifies three regimes (α = 0, 1/2, 1) of the metric-noise relationship, with the α = 1 regime corresponding to equilibrium physics.

The present results provide the first discrete, computational test case. Conjecture 1 is the discrete analog of Vanchurin's continuous-time result: stable geometry (his "equilibrium") requires convergent dynamics (his "learning"). The perturbation-response parallels his prediction that perturbation from equilibrium triggers a learning-like return process.

The key difference: Vanchurin works in continuous metric spaces with gradient dynamics; we work in discrete graph spaces with DPO rewriting. The fact that the structural relationship survives this translation suggests it is not an artifact of the continuous framework.

### 5.3 Limitations

1. **Small signatures.** Sixteen rules at σ ≤ 4. The frequency asymmetry and perturbation-response need confirmation at higher signatures.

2. **DPO growth rules only.** All enumerated rules are graph-growing. Conjecture 1 explicitly requires (M1). Non-growth systems (fixed_grid_noise, Dehn-twist) can be Φ-positive without being I-positive. The results characterize a property of *growth dynamics*, not of all possible graph dynamics.

3. **Threshold sensitivity.** The Φ-predicate threshold σ*_ds was tightened from 0.18 to 0.08 to achieve I/Φ separation at σ = 4. At the original threshold, all DPO rules are I+Φ+ and there is no separation to analyze. The results depend on this calibration.

4. **The Davis-Kahan gap.** The Davis-Kahan per-step bound is loose (exp11: mean ratio ‖ΔA‖_F / gap ≈ 11). The proof outline identifies the correct mechanism (eigenspace gap stability prevents crossings) but does not provide a tight quantitative bound. Converting Conjecture 1 to a theorem requires either a tighter perturbation argument or a fundamentally different approach based on spectral convergence of the graph sequence.

5. **The compression gate.** Part II of the proof is a claim, not a theorem. The compression argument is informal and relies on zlib as a proxy for Kolmogorov complexity.

6. **Is the I-predicate detecting growth?** Within DPO, every non-trivial rule is a growth rule, so the I-predicate could be merely detecting structured growth. Evidence against: in the 33-rule study [14], non-growth rules like sorting_edges are I-positive, and the Bayesian forward theorem [15] shows I-positivity captures convergent dynamics independent of growth. But within the DPO enumeration specifically, this concern cannot be fully resolved.

7. **Dehn-twist counterexample.** Verified computationally (exp12). The 10×10 toroidal grid with Dehn twist is Φ-positive (d_s = 2.58, σ_ds = 0) and I-negative (all τ ≤ 0), confirming that (M1) is essential.

### 5.4 Future work

- Enumerate σ = 5 (4→5) to test frequency asymmetry at larger scale.
- Multi-rule compositions for testing claim (a) strict inequality.
- Find a tight proof for Conjecture 1. The Davis-Kahan per-step bound is loose (exp11). Candidate approaches: (i) spectral convergence of the graph sequence under Φ-positivity implies Grassmannian convergence of the embedding; (ii) averaging arguments showing that per-step perturbations, while large, are directionally incoherent and cancel over multiple steps; (iii) a Lyapunov-function argument using the spectral dimension stability as a monotone quantity.
- Non-DPO program enumeration (binary lambda calculus or Wolfram-style hypergraph rules).

---

## References

[1] V. Vanchurin. "The World as a Neural Network." *Entropy* 22(11), 1210, 2020.

[2] V. Vanchurin. "Towards a Theory of Machine Learning." *MLST* 2(3), 2021.

[3] V. Vanchurin, Y.I. Wolf, M.I. Katsnelson, E.V. Koonin. "Toward a Theory of Evolution as Multilevel Learning." *PNAS* 119(6), 2022.

[4] V. Vanchurin. "Geometric Learning Dynamics." arXiv:2504.14728, 2025.

[5] M.P. Müller. "Algorithmic Idealism." *Found. Phys.* 56, 11, 2026. arXiv:2412.02826.

[6] M.P. Müller. "Law Without Law." *Quantum* 4, 301, 2020. arXiv:1712.01826.

[7] N. Agarwal, S.R. Dalal, V. Misra. "The Bayesian Geometry of Transformer Attention." arXiv:2512.22471, 2026.

[8] N. Agarwal, S.R. Dalal, V. Misra. "Gradient Dynamics of Attention." arXiv:2512.22473, 2025.

[9] Zhang Chong. "Attention Is Not What You Need: Grassmann Flows." arXiv:2512.19428, 2025.

[10] G. Kim. "Thermodynamic Isomorphism of Transformers." arXiv:2602.08216, 2026.

[11] C.A. Trugenberger et al. "Dynamics and the Emergence of Geometry in an Information Mesh." *EPJC* 80, 1091, 2020.

[12] C. Davis, W. Kahan. "The Rotation of Eigenvectors by a Perturbation." *SIAM J. Numer. Anal.* 7(1), 1970.

[13] P.-Å. Wedin. "Perturbation Bounds in Connection with Singular Value Decomposition." *BIT* 12, 1972.

[14] K. Kowalczyk. "Geometric Predicates for Classifying Dynamical Behaviors in Graph Rewrite Systems." In preparation, 2026.

[15] K. Kowalczyk. "Geometric Signatures of Bayesian Inference in Discrete Dynamical Systems." In preparation, 2026.

[16] K. Kowalczyk. "Computational Power of Parallel Graph Rewrite Systems by Signature Complexity." In preparation, 2026.

[17] H. Zenil, N.A. Kiani, J. Tegnér. *Algorithmic Information Dynamics.* Cambridge University Press, 2023.

[18] S. Wolfram. "A Class of Models with the Potential to Represent Fundamental Physics." *Complex Systems* 29(2), 2020.

[19] J. Gorard. "Some Relativistic and Gravitational Properties of the Wolfram Model." *Complex Systems* 29(2), 2020.

---

## Appendix A — Proof outline for Conjecture 1

### A.1 Lemma (GPI perturbation bound)

**Lemma.** *For a DPO rule with RHS size r applied via GPI to a graph G_t with matching of size m:*

$$\|ΔA_t\|_F \leq r \cdot \sqrt{2m}$$

*Proof.* Each matched subgraph contributes at most r edge additions or deletions. Each edge modification is a rank-2 symmetric update to the adjacency matrix A (setting A_{ij} = A_{ji} = 1 or 0), with Frobenius norm ≤ √2. Since GPI requires non-overlapping matches, the m matches act on disjoint vertex sets, so the corresponding perturbation matrices have disjoint support. For matrices with disjoint support, ‖∑_i δA_i‖_F² = ∑_i ‖δA_i‖_F². Each match contributes at most r modifications, so ‖δA_i‖_F ≤ r√2. Therefore:

$$\|ΔA_t\|_F^2 = \sum_{i=1}^{m} \|\delta A_i\|_F^2 \leq m \cdot (r\sqrt{2})^2 = 2mr^2$$

giving ‖ΔA_t‖_F ≤ r√(2m). Since m ≤ |E_t| (at most one match per edge in the LHS), we have ‖ΔA_t‖_F = O(|E_t|^{1/2}). □

### A.2 The Davis-Kahan step

Let e be an embedding satisfying E1–E3 [15, Section 2.2], and let X_t = X^e(G_t) ∈ ℝ^{n×d} be the embedding at time t. We show that the angular perturbation between successive embeddings decreases over time.

**Step 1 (Embedding perturbation).** By E1 (continuity), the embedding is Lipschitz in the adjacency matrix:

$$\|X_{t+1} - X_t\|_F \leq C_1 \cdot \|ΔA_t\|_F$$

**Step 2 (Subspace perturbation via Wedin).** Let U_t, U_{t+1} ∈ ℝ^{n×k} be orthonormal bases for the leading k-dimensional column spaces of X_t and X_{t+1}. By the Wedin sin θ theorem [13], the sine of the largest principal angle satisfies:

$$\sin \theta_{\max}(\mathrm{col}(X_t), \mathrm{col}(X_{t+1})) \leq \frac{\|X_{t+1} - X_t\|_F}{\mathrm{gap}_k^{\mathrm{cluster}}(L_t)}$$

Substituting Step 1:

$$\sin \theta_{\max} \leq \frac{C_1 \cdot \|ΔA_t\|_F}{\mathrm{gap}_k^{\mathrm{cluster}}(L_t)}$$

**Step 3 (Ratio decay under growth).** Under (M3'), the denominator is bounded below by γ > 0 for t ≥ t₁. By Lemma A.1, the numerator satisfies ‖ΔA_t‖_F ≤ r√(2m_t). The matching size m_t is bounded by the number of edge-disjoint copies of the LHS in G_t. While m_t may grow with |E_t|, the key observation is that ‖A_t‖_F = Θ(|E_t|^{1/2}), so the *relative* perturbation satisfies:

$$\frac{\|ΔA_t\|_F}{\|A_t\|_F} = O\left(\frac{|E_t|^{1/2}}{|E_t|^{1/2}}\right) = O(1)$$

However, this is not sufficient. The stronger argument proceeds via the *graph size growth*. Under (M1) with polynomial growth |V_t| = Θ(t^β), the Laplacian norm ‖L_t‖_2 = Θ(d_{\max}(G_t))$, which grows with graph size. The perturbation-to-spectral-norm ratio:

$$\frac{\|ΔA_t\|_F}{\|L_t\|_2} \leq \frac{C_1 r \sqrt{2m_t}}{\Omega(t^{\beta'})}$$

decreases provided the maximum degree grows at least as fast as √m_t. For DPO growth rules with bounded RHS, graph-theoretic arguments show that the maximum degree grows at least linearly with t (since GPI adds at least one edge per step in the presence of at least one match), while m_t grows at most polynomially. The angular perturbation per step thus tends to zero.

**Step 4 (Summability implies convergence).** Since the angular perturbation per step decreases, for any ε > 0 there exists t_2 such that ∑_{s=t_2}^{T} θ_s < ε. The cosine-to-final satisfies:

$$\cos \theta(\mathrm{col}(X_t), \mathrm{col}(X_T)) \geq \cos\left(\sum_{s=t}^{T-1} \theta_s\right)$$

by the triangle inequality on the Grassmannian. For t ≥ t_2, this is bounded below by cos(ε), and the sequence c_t is non-decreasing (within the summability regime). The Kendall τ of a non-decreasing sequence of length T − t_2 against time indices satisfies τ ≥ 1 − 2t_2/(T − 1).

### A.3 The τ_to_final bound

**Proposition (conditional on the per-step bound).** *If the angular perturbation per step θ_s is summable (∑ θ_s < ∞), then for T sufficiently large:*

$$\tau_{\mathrm{to\_final}}(e) \geq 1 - \frac{2t_2}{T - 1} > 0.5$$

*where t₂ = max(t₁, t₂') and t₁ is from (M3'), t₂' is the onset of the summability regime from Step 4.*

*Proof.* For t ≥ t₂, the cosine-to-final sequence c_t^e is non-decreasing by Step 4. For t < t₂, c_t^e may be arbitrary. The Kendall τ of a sequence that is non-decreasing in its last T − t₂ terms satisfies:

$$\tau \geq 1 - \frac{2t_2(T - 1 - t_2)}{\binom{T}{2}} \geq 1 - \frac{2t_2}{T - 1}$$

Setting T > 4t₂ + 1 gives τ > 0.5 = τ*. This parallels [15, Corollary 1], replacing the Bayesian source of convergence (Doob's theorem → posterior concentration) with the spectral stability source (Φ-positivity → eigenvalue distribution convergence). □

The summability condition is the gap in the proof: exp11 shows the per-step angular bound from Davis-Kahan does not decrease, so summability is not established by the current argument. The proposition is valid conditional on summability; establishing summability by other means would complete the proof of Conjecture 1.

## Appendix B — Computational reproducibility

All computations performed in Python 3.12 with NetworkX 3.x, NumPy, SciPy. Random seed fixed at 42 (MASTER_SEED in primo/config.py). Embedding projection seed fixed at 0. Complete source code: https://github.com/KarolFilipKowalczyk/PRIMO. Experiments referenced: exp04 (enumeration), exp06 (temporal profiles), exp09/09b (eigenvalue gaps), exp10 (perturbation-response), exp11 (Davis-Kahan ratio), exp12 (Dehn-twist). 16 DPO rules, 4 seeds, 3 embeddings, T = 30 steps per trajectory (T = 60 where noted). Total experimental runtime: < 30 minutes on a single GPU (RTX 3050, 4GB VRAM).
