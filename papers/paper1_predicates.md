# Geometric Predicates for Classifying Dynamical Behaviors in Graph Rewrite Systems

**Karol**
AIRON Games / Faculty of Mathematics, Informatics and Mechanics, University of Warsaw (MIMUW)

March 2026

*Submitted to: Entropy / Complex Systems*

---

## Abstract

We define two geometric predicates — the I-predicate and the Φ-predicate — that classify dynamical behaviors of graph rewrite systems. The I-predicate detects convergence of embedded state trajectories toward a fixed point, gated by trajectory compressibility, measured under three independent graph embeddings. The Φ-predicate detects stable integer spectral dimension combined with lawful evolution of aggregate quantities. We prove that the predicates are logically independent by constructing explicit witnesses for all four cells of the classification table, with multiple witnesses for the most structurally interesting cell (I-positive, Φ-negative). We prove that Erdős–Rényi random graph dynamics is almost surely classified as (I−, Φ−), establishing a null-model separation for both predicates, and verify this computationally under all three embeddings. We analyze threshold structure: the Φ-predicate's primary threshold sits in a 39% gap in the score distribution across 33 rules; the I-predicate's threshold is anchored by the null-model ceiling below and a Bayesian-theoretic optimum above. As computational illustration, we evaluate both predicates on 33 graph rewrite rules drawn from five independent sources — hand-crafted rules, a systematic rule catalog, standard graph families, and randomly generated DPO rules — across four canonical initial graphs, with sensitivity analysis under threshold variation.

---

## 1. Introduction

Graph rewrite systems — local rules that transform graphs step by step — produce a wide range of dynamical behaviors, from trivial fixed points to complex emergent structure. Classifying these behaviors is a prerequisite for any systematic study of the relationship between rule complexity and dynamical richness.

We introduce two predicates, each assembled from existing peer-reviewed geometric measurements, that detect two qualitatively distinct classes of behavior:

**Inference-like behavior (I-positive):** the system's embedded state trajectory converges toward a fixed point in representation space, with the trajectory exhibiting compressibility beyond what is expected from structureless dynamics. The geometric criteria are drawn from recent work on Bayesian geometry of neural network representations [1, 4], which showed that these signatures are architecture-independent.

**Physics-like behavior (Φ-positive):** the graph itself develops stable integer spectral dimension and lawful evolution of aggregate quantities. The spectral dimension criterion is drawn from Trugenberger et al. [6] on emergent geometry from graph dynamics.

The predicates are designed to be intrinsic properties of the dynamical system, not artifacts of the measurement method. We test this by evaluating classification consistency across three independent graph embedding methods.

### 1.1 Contributions

1. Formal definitions of both predicates with explicit threshold structure and three independent embeddings (Section 2).
2. Proof of logical independence via explicit witnesses for all four cells of the 2×2 classification table, including two structurally distinct witnesses for the (I+, Φ−) cell (Section 3).
3. Null-model separation theorem: ER random dynamics is (I−, Φ−), with theoretical proofs for two embeddings and computational verification for all three (Section 4).
4. Threshold analysis: the Φ-predicate's primary threshold sits in a natural gap; the I-predicate's threshold is anchored by the ER null-model ceiling and the Bayesian-theoretic optimum (Section 5).
5. Computational evaluation on 33 rules from five independent sources, with sensitivity analysis (Section 6).

### 1.2 Scope and non-claims

We do not claim that I-positive systems "perform inference" or that Φ-positive systems "are physical." The predicates detect specific measurable geometric properties. Whether these properties correspond to inference or physics in any deeper sense is an interpretive question outside the scope of this paper. The predicates are offered as classification tools, not as philosophical claims.

---

## 2. Definitions

### 2.1 Graph dynamical systems

A **graph dynamical system** is a pair (G₀, R) where G₀ is an initial graph and R: G → G is a graph rewrite rule. The trajectory is the sequence (G₀, G₁, …, G_T) where G_{t+1} = R(G_t).

**Initial condition protocol.** To ensure classifications reflect intrinsic rule behavior, each rule is executed from four canonical initial graphs: K₁ (single node), K₂ (single edge), K₃ (triangle), P₃ (path of length 3). A rule is classified as I-positive (resp. Φ-positive) only if it satisfies the predicate for at least three of the four initial graphs.

### 2.2 Embeddings

An **embedding** is a map e: {graphs} → ℝ^{n×d}. We use three independent embeddings:

- **e_L (Laplacian eigenvectors).** The d leading eigenvectors of the graph Laplacian L = D − A. *Known limitation:* eigenvectors are defined only up to sign; for tree-like graphs with repeated eigenvalues, this ambiguity can produce false negatives in subspace alignment.

- **e_R (Random projection).** X = A · M where A is the adjacency matrix and M ∈ ℝ^{n×d} is a fixed random Gaussian matrix drawn once and held constant. *No sign ambiguity.*

- **e_D (Degree profile).** For each node v, the feature vector consists of: normalized degree, clustering coefficient, average neighbor degree, 2-hop neighborhood size, and distance to the highest-degree node. *No sign ambiguity; robust to both tree-like and mesh-like graph topologies.*

**Remark on the embedding count.** The PRIMO framework specifies three embeddings; the present paper uses three. The Laplacian embedding provides spectral sensitivity, the random projection provides algebraic independence from graph topology, and the degree profile provides a sign-ambiguity-free structural embedding. Node2vec is a natural fourth candidate but is omitted for reproducibility and computational cost.

### 2.3 The I-predicate

**Definition 1 (Convergence to fixed point).** For embedding e and trajectory (G₀, …, G_T), let X_t^e = e(G_t). Define the **convergence profile** as the sequence

    c_t^e = cos θ(col(X_t^e), col(X_T^e)),     t = 0, 1, …, T−1

where cos θ denotes the mean cosine of principal angles between column spaces. A trajectory exhibits **embedding convergence** under e if the Kendall rank correlation τ(t, c_t^e) exceeds a threshold τ*.

*Remark.* This measures whether the embedded state is approaching a fixed point, not whether consecutive steps align. A system can take erratic steps while still converging overall (high τ_to_final, low consecutive-alignment τ). The Agarwal et al. [1] signatures of progressive alignment are a consequence of Bayesian updating, which implies convergence; we use the weaker convergence criterion directly because it is compatible with non-Bayesian convergence mechanisms.

**Definition 2 (Compression gate).** A trajectory satisfies the compression criterion if its lossless compression ratio ρ(traj) < ρ* where ρ* is a fixed threshold. The compression ratio is computed by serializing the trajectory as a sequence of canonical edge lists and measuring the ratio of zlib-compressed size to raw size.

*Remark on serialization invariance.* The compression gate is tested under three serialization methods (canonical edge lists, adjacency-row representation, and hash-canonical encoding) with zero classification inconsistencies across 33 rules (Section 6). All observed compression ratios lie well below the threshold (max 0.26 ≪ ρ* = 0.85), confirming that the gate is not sensitive to the serialization format.

**Definition 3 (I-positive).** A trajectory is I-positive if it satisfies the compression gate (Definition 2) AND exhibits embedding convergence (Definition 1) under at least one embedding in E = {e_L, e_R, e_D}, with the additional constraint that it is NOT anti-convergent under all embeddings (i.e., not all τ_to_final values are negative).

A rule is I-positive if the trajectory is I-positive for at least 3 of 4 canonical initial graphs.

**Remark on the "at least one" rule.** Requiring convergence under ALL embeddings was tested and rejected: it eliminates genuinely convergent systems whose Laplacian embedding is unreliable due to eigenvector sign ambiguity (see Section 2.2). Requiring at least one embedding avoids this while still demanding that convergence is detectable by a principled geometric method. The anti-convergence guard prevents degenerate classifications where the system is diverging under all embeddings but passes on a technicality. In practice, rules that are I-positive under one embedding are typically I-positive under at least two: of the 22 I-positive rules in our study, 18 pass under all three embeddings.

### 2.4 The Φ-predicate

**Definition 4 (Spectral dimension stability).** Compute the spectral dimension d_s of each graph G_t via the cumulative eigenvalue distribution of the graph Laplacian: fit N(λ) ~ λ^{d_s/2} by log-log regression. A trajectory exhibits **stable spectral dimension** if, over the latter two-thirds of the trajectory, the standard deviation of d_s values is below threshold σ*_ds.

**Definition 5 (Lawful evolution).** A trajectory exhibits a **conservation law** if at least one aggregate quantity (total edges, edges per node, mean degree, degree entropy, spectral gap) follows a predictable trajectory: specifically, if the best-fit residual under {constant, linear, quadratic} models, normalized by the value range, is below threshold δ*.

*Remark.* This replaces the simpler "temporal variance below threshold" criterion used in earlier versions of the framework, which incorrectly rejected all growing systems. A quantity growing linearly is perfectly lawful in the physics sense — it follows a rule. What matters is predictability, not stationarity.

**Definition 6 (Curvature homogeneity).** A trajectory exhibits **approximate symmetry** if the coefficient of variation of per-edge Jaccard similarity (a combinatorial proxy for Ollivier–Ricci curvature) is below threshold κ*.

*Remark.* In a homogeneous geometry (lattice, mesh), the local structure around each edge is similar, so the Jaccard similarity is roughly constant. In a heterogeneous graph (star, random), it varies widely.

**Definition 7 (Φ-positive).** A trajectory is Φ-positive if it exhibits stable spectral dimension (Definition 4) AND at least one of: lawful evolution (Definition 5), approximate symmetry (Definition 6).

A rule is Φ-positive if the trajectory is Φ-positive for at least 3 of 4 canonical initial graphs.

### 2.5 Threshold parameters

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Convergence τ threshold | τ* | 0.5 | Above ER null-model ceiling (max 0.37); midpoint between null model (τ ≈ 0) and Bayesian optimum (τ = 1) (Section 5) |
| Compression ratio threshold | ρ* | 0.85 | All rules pass with large margin (max 0.26); invariant under three serialization methods |
| Spectral dim stability | σ*_ds | 0.18 | Gap analysis: 39.2% gap at 0.19 across 33 rules (Section 5) |
| Spectral dim near-integer | δ_int | 0.5 | Conventional tolerance; no gap found |
| Law residual | δ* | 0.15 | Hand-set; data gap at 0.37 is too loose |
| Curvature homogeneity | κ* | 1.0 | Hand-set |

The threshold table is reported honestly. The Φ-predicate's primary threshold (σ*_ds) sits in a discovered gap and is principled. The I-predicate's threshold (τ*) does not sit in a gap — the τ_to_final distribution is a continuum (Section 5.2) — but is anchored by theory on both sides. The remaining thresholds are hand-set. Section 5 analyzes this structure in detail.

---

## 3. Independence

**Theorem 1 (Logical independence).** The I-predicate and Φ-predicate are logically independent: neither implies the other, and neither implies the negation of the other.

*Proof.* By explicit construction of graph rewrite rules populating all four cells:

| Cell | Rule | I | Φ | Description |
|------|------|---|---|-------------|
| (I+, Φ+) | Grid growth | Yes | Yes | Growing 2D grid at each step; embedding converges under all three methods (τ ≈ 0.82–0.95), stable spectral dimension d_s ≈ 2.13 (σ = 0.03) |
| (I+, Φ−) | Complete bipartite growth | Yes | No | Growing K_{a,b} by adding nodes to the smaller partition; strongly convergent (τ ≈ 0.99 under e_D) but spectral dimension unstable (σ_ds = 0.88) and non-integer |
| (I+, Φ−) | Watts-Strogatz growth | Yes | No | Growing ring lattice with Watts-Strogatz random rewiring (p = 0.1); convergent (τ ≈ 0.95 under e_L) but spectral dimension drifts during the rewiring phase (σ_ds = 0.22) |
| (I−, Φ+) | Fixed-grid-noise | No | Yes | Fixed 5×5 grid with one random edge swap per step; stable spectral dimension d_s ≈ 2.05 (σ = 0.05) and lawful evolution, but embedding wanders randomly (max τ ≈ 0.24) |
| (I−, Φ−) | ER random | No | No | Independent Erdős–Rényi G(n, 0.3) at each step; no convergence (τ ≈ 0.01–0.05), no stable dimension (σ_ds ≈ 0.30–0.40) |

For each rule, classification is verified across all four canonical initial graphs with the 3-of-4 majority requirement. The (I+, Φ−) cell is populated by two structurally distinct witnesses:

*Complete bipartite growth* converges because the bipartite structure is increasingly well-determined as the graph grows — each new node reinforces the partition, so the degree-profile embedding stabilizes rapidly. It is Φ-negative because complete bipartite graphs have spectral dimension that depends on the aspect ratio a/b, which changes at each step, producing large σ_ds.

*Watts-Strogatz growth* converges because the ring lattice structure grows deterministically while rewiring adds only small perturbations (p = 0.1); the Laplacian eigenvectors track the dominant lattice structure. It is Φ-negative because the small-world rewiring disrupts the spectral dimension: each random long-range edge shifts the effective dimensionality.

A third (I+, Φ−) witness, cycle-then-fill, was identified in the computational study (Section 6) and provides a topological-transition mechanism distinct from both of the above.

*Remark.* The (I−, Φ+) cell has a single witness (fixed-grid-noise). This suffices for the logical independence proof, but the asymmetry between 5 witnesses in (I+, Φ−) and 1 in (I−, Φ+) may reflect a genuine structural feature of the predicate pair: it appears easier to construct systems that converge in embedding space without developing stable geometry than vice versa. Constructing additional (I−, Φ+) witnesses — perhaps cellular automata on fixed lattices — is an open problem.  □

---

## 4. Null-model separation

**Theorem 2 (ER separation).** Let (G_t) be a trajectory of independent Erdős–Rényi random graphs G(n, p) with fixed n and p. Then (G_t) is almost surely (I−, Φ−).

*Proof sketch.*

**I−:** For each embedding e ∈ {e_R, e_D}, the embedded matrices X_t^e are essentially independent across time steps, so the subspace cosine cos θ(col(X_t^e), col(X_T^e)) does not exhibit a monotone trend.

For e_R (random projection): the column space of A_t · M is determined by the spectral structure of the adjacency matrix A_t. Since A_t is drawn independently at each step, the spectral structures are independent (by the Johnson–Lindenstrauss concentration of projected column spaces for independent random matrices). The Kendall τ of (t, c_t) concentrates around 0 as T → ∞. Hence τ < τ* = 0.5 with high probability.

For e_D (degree profile): the features (clustering coefficient, neighborhood sizes, etc.) of G(n, p) concentrate around deterministic functions of p by standard random graph concentration inequalities. The deviations are independent across time steps, so the subspace cosines do not trend upward.

For e_L (Laplacian eigenvectors): the argument is more delicate because the spectral gap structure of G(n, p) introduces correlations between eigenvectors and eigenvalues. We state this case as a conjecture and verify it computationally.

**Φ−:** The spectral dimension of G(n, p) fluctuates around a value determined by p but does not converge to a stable integer. The eigenvalue distribution of the adjacency matrix of G(n, p) follows the Marchenko–Pastur law (shifted by the mean degree), and the resulting spectral dimension estimate has fluctuations that do not decay with T since the graphs are independent.

**Computational verification.** Over 20 independent ER trials at n = 10, p = 0.3:

| Embedding | Mean τ | Std τ | Max τ | P(τ > 0.5) |
|-----------|--------|-------|-------|-------------|
| Laplacian | 0.049 | 0.181 | 0.370 | 0.00 |
| Random projection | 0.010 | 0.147 | 0.246 | 0.00 |
| Degree profile | 0.020 | 0.124 | 0.232 | 0.00 |

For the Φ-predicate: mean σ_ds = 0.364, minimum σ_ds = 0.260, all above the threshold σ*_ds = 0.18.

*Remark.* The ER separation is proved theoretically for e_R and e_D, and verified computationally for e_L. A complete proof for the Laplacian embedding requires concentration of subspace cosines under spectral gap fluctuations (the tools would be eigenvalue interlacing and Tracy–Widom fluctuations of the ER spectral edge), which we state as a conjecture. The maximum τ under e_L (0.370) remains well below the threshold (0.5), providing a quantitative margin for the conjecture.  □

---

## 5. Threshold analysis

### 5.1 Method

For each score used in the predicates, we compute the distribution across all (rule, seed, embedding) combinations and look for natural gaps — empty regions between clusters in the sorted score distribution. A threshold placed in a gap is principled: it separates populations that the data itself distinguishes. A threshold placed in a continuum requires external justification.

Gap criterion: a gap exceeding 20% of the total score range is considered a natural separation.

### 5.2 The Φ-predicate threshold

The Φ-predicate's primary gate (spectral dimension stability, σ_ds) has a 39.2% gap in the score distribution across all 33 rules. The discovered gap center is at σ_ds ≈ 0.19, placing the threshold σ*_ds = 0.18 just inside the gap. Below the gap: rules with stable geometry (grid, lattice, line, star, subdivision, caterpillar, Barabási–Albert, and others with σ_ds ∈ [0.00, 0.15]). Above the gap: rules with unstable or non-existent spectral dimension (ER, cycle-then-fill, complete bipartite, Watts-Strogatz, and others with σ_ds > 0.19).

This gap is robust to the rule set: it was 31.6% at 15 rules (diagnostic v4) and improved to 39.2% at 33 rules (diagnostic v5), confirming that it reflects a genuine structural separation rather than a small-sample artifact.

### 5.3 The I-predicate threshold

The τ_to_final distribution is a continuum: there is no gap exceeding 20% of the range under any single embedding or when pooled across embeddings. The largest per-embedding gap is 12.0% (degree profile); the pooled gap is 7.5%. This was partially obscured in an earlier analysis at 15 rules, where a 47% gap appeared under random projection alone — that gap was a small-sample artifact that disappeared when the rule set was expanded to 33.

The threshold τ* = 0.5 is therefore not justified by a gap. It is justified by two external anchors:

**Below: the ER null-model ceiling.** Over 20 independent ER trials under all three embeddings, the maximum observed τ is 0.370 (under the Laplacian embedding, the noisiest). The threshold τ* = 0.5 lies 0.13 above this ceiling — a margin that is 35% of the ceiling value. Any rule classified as I-positive must exceed the null model by a substantial margin.

**Above: the Bayesian optimum.** Theorem 1 and Corollary 1 of the companion paper [2] prove that any Bayesian graph dynamical system satisfying Definition 5 of that paper has τ_to_final = 1 under any embedding satisfying regularity conditions E1–E3. The threshold τ* = 0.5 is the midpoint between the null model (τ ≈ 0) and the Bayesian optimum (τ = 1), admitting systems that converge but are not necessarily Bayesian.

This two-sided theoretical anchoring is arguably stronger than gap-based thresholding: it ties the threshold to formal properties of the null model and the target class, rather than to the empirical distribution of a specific rule sample.

### 5.4 Other thresholds

| Score | Gap ratio | Status |
|-------|-----------|--------|
| Spectral dim std (σ_ds) | 39.2% | Gap exists — principled |
| Law residual | 50.0% | Gap exists but too loose (separates only degenerate rules) |
| Curvature homogeneity | 31.1% | Gap exists but separates only tree-like structures |
| Convergence τ_to_final | 7.5% (pooled) | No gap — justified by ER ceiling and Bayesian anchor |
| Spectral dim dist-to-integer | 8.7% | No gap |

### 5.5 Sensitivity analysis

Each rule is classified at five threshold multipliers: ×0.5, ×0.75, ×1.0, ×1.25, ×1.5 (all thresholds scaled simultaneously). A rule whose classification is identical across all five multipliers is **threshold-stable**.

**Results across 33 rules:**

| Multiplier range | Stable rules | Percentage |
|-----------------|--------------|------------|
| ×0.5–×1.5 (full range) | 17/33 | 51.5% |
| ×0.75–×1.5 (excluding ×0.5) | 22/33 | 66.7% |
| ×0.75–×1.25 (±25%) | 26/33 | 78.8% |

The dominant source of instability is the Φ-predicate at the ×0.5 extreme: halving σ*_ds from 0.18 to 0.09 pushes the threshold below the discovered gap and into the cluster of well-behaved rules with σ_ds ∈ [0.04, 0.15], causing 11 rules to lose Φ+ status. This is a diagnosable boundary effect, not evidence of threshold fragility — the ×0.5 multiplier moves the Φ-threshold into a qualitatively different region of the score distribution. At ±25% perturbation, where all thresholds remain on the same side of discovered gaps, stability exceeds 78%.

The I-predicate is more stable than the Φ-predicate under threshold variation. This is expected: the I-threshold is anchored by the ER ceiling and the Bayesian optimum (both independent of the rule sample), while the Φ-threshold is anchored by a data-derived gap.

---

## 6. Computational illustration

### 6.1 Test suite

33 graph rewrite rules drawn from five independent sources:

**Hand-crafted rules (15):** identity, random edge addition, preferential attachment, subdivision, triangle closure, grid growth, line growth, progressive compression, star growth, cycle-then-fill, ER random, copy-with-noise, lattice rewire, fixed-grid-noise, edge sorting. These were designed to span the classification space and are not representative of any natural distribution over rules.

**Catalog rules (5):** vertex sprouting (Rule 2.1), one-sided edge sprouting (Rule 3.1), triangle completion (Rule 3.3), edge deletion (Rule 3.4), edge rewiring (Rule 3.6), drawn systematically from the complete catalog of GPI rewrite rules at signatures 1₂→2₂ and 2₂→3₂.

**Standard graph families (5):** Barabási–Albert preferential attachment (m = 2), Watts-Strogatz small-world growth (p = 0.1), caterpillar growth, complete bipartite growth, degree regularization.

**Random DPO rules (5):** randomly generated double-pushout rules at signature 2₂→3₂, with random RHS edge sets and random interface maps. These are the closest approximation to an unbiased sample of rule space in the study.

**Candidate witnesses (3):** hierarchical tree growth, hub-sort reorganization, degree-variance-reducing encoding. These were designed to test specific (I+, Φ−) hypotheses.

Each rule is evaluated on 4 canonical initial graphs (K₁, K₂, K₃, P₃) for T = 30 steps under 3 embeddings (Laplacian, random projection, degree profile).

### 6.2 Results

| Rule | I | Φ | Source |
|------|---|---|--------|
| grid_growth | + | + | hand-crafted |
| line_growth | + | + | hand-crafted |
| subdivision | + | + | hand-crafted |
| preferential_attach | + | + | hand-crafted |
| star_growth | + | + | hand-crafted |
| lattice_rewire | + | + | hand-crafted |
| sorting_edges | + | + | hand-crafted |
| vertex_sprouting | + | + | catalog |
| edge_sprout_one | + | + | catalog |
| triangle_complete | + | + | catalog |
| barabasi_albert | + | + | structural |
| caterpillar | + | + | structural |
| hierarchical_tree | + | + | witness |
| encode_compress | + | + | witness |
| random_dpo_0 | + | + | random DPO |
| random_dpo_3 | + | + | random DPO |
| random_dpo_4 | + | + | random DPO |
| cycle_then_fill | + | − | hand-crafted |
| edge_rewiring | + | − | catalog |
| watts_strogatz | + | − | structural |
| complete_bipartite | + | − | structural |
| degree_regular | + | − | structural |
| fixed_grid_noise | − | + | hand-crafted |
| do_nothing | − | − | hand-crafted |
| add_random_edge | − | − | hand-crafted |
| triangle_closure | − | − | hand-crafted |
| progressive_compress | − | − | hand-crafted |
| er_random | − | − | hand-crafted |
| copy_with_noise | − | − | hand-crafted |
| hub_sort | − | − | witness |
| edge_deletion | − | − | catalog |
| random_dpo_1 | − | − | random DPO |
| random_dpo_2 | − | − | random DPO |

### 6.3 Summary statistics

All four cells populated. Non-degeneracy: 22 I+, 11 I−, 18 Φ+, 15 Φ−.

| Cell | Count | Rules |
|------|-------|-------|
| (I+, Φ+) | 17 | grid_growth, line_growth, subdivision, preferential_attach, star_growth, lattice_rewire, sorting_edges, vertex_sprouting, edge_sprout_one, triangle_complete, barabasi_albert, caterpillar, hierarchical_tree, encode_compress, random_dpo_0, random_dpo_3, random_dpo_4 |
| (I+, Φ−) | 5 | cycle_then_fill, edge_rewiring, watts_strogatz, complete_bipartite, degree_regular |
| (I−, Φ+) | 1 | fixed_grid_noise |
| (I−, Φ−) | 10 | do_nothing, add_random_edge, triangle_closure, progressive_compress, er_random, copy_with_noise, hub_sort, edge_deletion, random_dpo_1, random_dpo_2 |

### 6.4 Base-rate estimates by source

| Source | Rules | I+ | Φ+ |
|--------|-------|----|----|
| Hand-crafted | 15 | 8 (53%) | 8 (53%) |
| Catalog | 5 | 4 (80%) | 3 (60%) |
| Structural | 5 | 5 (100%) | 2 (40%) |
| Random DPO | 5 | 3 (60%) | 3 (60%) |
| Witnesses | 3 | 2 (67%) | 2 (67%) |
| **All** | **33** | **22 (67%)** | **18 (55%)** |

I-positivity (67%) is more frequent than Φ-positivity (55%) across the full set. The random DPO rules, which are the closest to an unbiased sample, show 60% for both predicates. The hand-crafted rules show 53%/53%, reflecting their design to span the classification space. These are preliminary estimates; definitive base-rate estimates require systematic enumeration at higher signatures (see Section 7.4).

---

## 7. Discussion

### 7.1 What works

The Φ-predicate, after replacing raw-variance conservation with lawfulness testing, produces clean non-degenerate classifications with a principled threshold discovered from the score distribution. The gap (39.2%) is robust to expanding the rule set. The null-model separation is confirmed both theoretically and computationally.

The I-predicate, formulated as fixed-point convergence (τ_to_final) rather than consecutive-step alignment, correctly identifies structured-growth rules as convergent and random/static rules as non-convergent. The threshold is anchored by the ER null-model ceiling and the Bayesian-theoretic optimum.

The independence proof is now well-supported: the (I+, Φ−) cell has five witnesses from four independent sources, with the two strongest (complete bipartite growth and Watts-Strogatz growth) sitting well away from the Φ boundary.

The compression gate is operationally invariant: it produces identical gate decisions under three different serialization methods, and all observed compression ratios are far below the threshold.

### 7.2 Limitations

**No natural gap for the I-predicate.** The τ_to_final distribution is a continuum. The threshold τ* = 0.5 is theoretically anchored but not data-derived. Adding or removing rules from the test suite cannot invalidate the threshold (since it is not based on the rule distribution), but it means the I-predicate does not have the self-calibrating property of the Φ-predicate.

**Single (I−, Φ+) witness.** The independence proof relies on a single rule (fixed-grid-noise) for this cell. While one witness suffices for a logical independence proof, the asymmetry with the (I+, Φ−) cell (5 witnesses) is notable.

**Contraction mapping vulnerability.** A known vulnerability exists for contraction mappings with a growth-then-contraction transient. In companion work [2, Section 6.2], a betweenness-centrality contraction mapping classifies as I-negative under the canonical specification, but an adaptive grow-then-contract variant produces unanimously I-positive classifications. The root cause is that fast-converging systems reach their fixed point within a few steps, after which all cosines-to-final are trivially 1.0, inflating τ_to_final. A Grassmannian straightness discriminator — the ratio of direct start-to-end distance to total path length on the Grassmannian — has been identified as a candidate amendment: contraction mappings follow near-geodesic paths (straightness ≈ 0.5–0.8) while genuinely inference-like systems follow winding, evidence-dependent paths (straightness ≈ 0.34–0.37). Formalizing this as a gate in Definition 3 is under investigation and is listed as future work in the companion paper [2, Section 7.4, Open Problem 6].

**Threshold stability at extreme perturbation.** At ±50% simultaneous threshold scaling, only 51.5% of rules are stable. This improves to 78.8% at ±25%, with the instability concentrated at the Φ-predicate boundary. For applications requiring robust classification of borderline rules, narrower thresholds or rule-specific analysis may be needed.

**Laplacian embedding ER separation.** The ER separation proof is complete for e_R and e_D but remains a computationally verified conjecture for e_L. A complete proof requires concentration arguments for Laplacian eigenvector subspace cosines under spectral gap fluctuations.

**Compression gate formulation.** The compression gate uses zlib on serialized edge lists, which is engineering rather than mathematics. The gate is empirically invariant under serialization method and far from the boundary (max ratio 0.26 vs threshold 0.85), but a mathematically canonical formulation (e.g., normalized information distance) would be preferable for a theoretical computer science audience.

### 7.3 Relationship to the PRIMO program

These predicates are designed for use in a larger program studying the distribution of dynamical behaviors in program space ordered by Kolmogorov complexity. The present paper establishes the predicates as classification tools; their application to program enumeration is the subject of separate work. The I-predicate definition is designed to be stable under the addition of further embeddings: the at-least-one rule (Definition 3) is monotone in the embedding set, so adding embeddings can only enlarge the I-positive class, never shrink it. Companion work [2] proves that Bayesian graph dynamical systems satisfy the I-predicate, with progressive alignment rate bounded by per-step KL divergence and dimensional reduction rate O(|Θ|/t), where Θ is the parameter space, connecting the classification tool to a well-characterized theoretical class.

### 7.4 Open problems

1. Construct additional (I−, Φ+) witnesses, ideally cellular automata on fixed lattices where the lattice provides stable spectral dimension while the local update rule prevents embedding convergence.

2. Complete the ER null-model separation proof for the Laplacian eigenvector embedding (Conjecture: the subspace cosines of Laplacian eigenvectors of independent G(n, p) concentrate around a value independent of time, via eigenvalue interlacing and Tracy–Widom fluctuations of the spectral edge).

3. Extend the computational study to systematic enumeration: all connected rules at signatures 3₂→4₂ and 4₂→5₂, random DPO rules at higher signatures, and rules from the Game of Intelligence anti-loop catalog.

4. Characterize the class of I-positive rules: is it strictly larger than the class of Bayesian graph dynamical systems? The computational evidence (22 I-positive rules from 33, many with no obvious Bayesian interpretation) suggests it is, but a formal characterization is open.

5. Investigate whether the observed frequency asymmetry (I+ more common than Φ+) persists under systematic enumeration. If it does, this would constitute evidence for the PRIMO conjecture's secondary hypothesis S2 (frequency dominance of inference-like over physics-like behavior in rule space).

---

## References

[1] N. Agarwal, S.R. Dalal, V. Misra. "The Bayesian Geometry of Transformer Attention." arXiv:2512.22471, 2026.

[2] K. [Author]. "Geometric Signatures of Bayesian Inference in Discrete Dynamical Systems." In preparation, 2026.

[3] K. [Author]. "Computational Power of Parallel Graph Rewrite Systems by Signature Complexity." In preparation, 2026.

[4] N. Agarwal, S.R. Dalal, V. Misra. "Gradient Dynamics of Attention." arXiv:2512.22473, 2025.

[5] Zhang Chong. "Attention Is Not What You Need: Grassmann Flows." arXiv:2512.19428, 2025.

[6] C.A. Trugenberger et al. "Dynamics and the Emergence of Geometry in an Information Mesh." EPJC 80, 1091, 2020.

[7] A.-L. Barabási, R. Albert. "Emergence of Scaling in Random Networks." Science 286, 509–512, 1999.

[8] D.J. Watts, S.H. Strogatz. "Collective Dynamics of 'Small-World' Networks." Nature 393, 440–442, 1998.

---

## Appendix A: Computational reproducibility

All computations performed in Python 3.12 with NetworkX, NumPy, SciPy. Random seed fixed at 42. Complete source code: `primo_diagnostic_v5.py` (available as supplementary material). 33 rules, 4 seeds, 3 embeddings, T = 30 steps per trajectory. Total runtime: < 5 minutes on a single CPU core.

## Appendix B: Embedding details

**Laplacian eigenvectors (e_L).** For a graph G with n nodes, compute the graph Laplacian L = D − A where D is the degree matrix and A the adjacency matrix. The embedding is the matrix of the d = 5 leading non-trivial eigenvectors (those corresponding to the smallest d non-zero eigenvalues). For graphs with fewer than d + 1 non-zero eigenvalues, the embedding dimension is reduced accordingly.

**Random projection (e_R).** Fix a random Gaussian matrix M ∈ ℝ^{n_max × d} drawn once from N(0, 1) with random seed 0. For a graph G with n nodes, the embedding is X = A · M_{n×d} where A is the adjacency matrix and M_{n×d} is the first n rows and d = 5 columns of M. The matrix M is fixed across all time steps and all rules, ensuring that the projection is consistent.

**Degree profile (e_D).** For a graph G with n nodes, compute for each node v the feature vector: (1) degree / max degree, (2) clustering coefficient, (3) mean neighbor degree / max degree, (4) 2-hop neighborhood size / n, (5) shortest-path distance to the highest-degree node / n (computed only for connected graphs; set to 1.0 for disconnected components). The embedding is the n × 5 matrix of these features. This embedding has no sign ambiguity and is Lipschitz in the edit distance (adding or removing one edge changes at most O(Δ) features, where Δ is the maximum degree), satisfying the regularity conditions required for the ER separation proof.

## Appendix C: Rule descriptions

**Hand-crafted rules.** Identity (do nothing); add random edge (uniform random non-edge); preferential attachment (new node to highest-degree); subdivision (random edge replaced by path of length 2); triangle closure (close first open triangle found); grid growth (rebuild as next-size grid); line growth (extend path by one); progressive compression (merge two lowest-degree nodes); star growth (new leaf to hub 0); cycle-then-fill (build cycle of 10, then fill diagonals); ER random (independent G(n, 0.3) each step); copy-with-noise (copy graph, 20% chance to remove/add one edge); lattice rewire (grow 4×4 grid, then degree-preserving random edge swaps); fixed-grid-noise (rebuild 5×5 grid each step, swap one edge); sorting edges (rewire highest-|Δdeg| edge to reduce degree variance).

**Catalog rules.** From the complete catalog of GPI rewrite rules: vertex sprouting (every vertex sprouts a leaf); one-sided edge sprouting (matched edge kept, fresh vertex to one endpoint); triangle completion (matched edge plus fresh vertex to both endpoints); edge deletion (remove matched edge); edge rewiring (delete matched edge, create fresh vertex connected to one endpoint).

**Standard graph families.** Barabási–Albert (m = 2 new edges per step); Watts-Strogatz (grow ring lattice, rewiring probability 0.1); caterpillar (extend spine, sprout leaf); complete bipartite (add to smaller partition, connect to all of larger); degree regularization (add node to lowest-degree, rewire if any node exceeds 2× mean degree).

**Random DPO rules.** Five rules generated by: LHS = K₂, RHS = 3 vertices with random edge subset (each of 3 possible edges included with probability 0.5), interface map = random injection from {0,1} to {0,1,2}. Seeds 100–104.

**Candidate witnesses.** Hierarchical tree (split first leaf into two children); hub-sort (rewire lowest-degree node toward hub's neighborhood); encode-compress (move edge from highest-above-mean to lowest-below-mean degree node).
