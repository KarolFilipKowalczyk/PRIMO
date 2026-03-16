# Geometric Signatures of Bayesian Inference in Discrete Dynamical Systems

**Karol**
AIRON Games / Faculty of Mathematics, Informatics and Mechanics, University of Warsaw

March 2026

---

## Abstract

We introduce the I-predicate, a formal test for inference-like behavior in discrete dynamical systems on graphs, assembled from geometric criteria — progressive alignment to the final embedded state (measured via Kendall rank correlation), dimensional reduction, and data compression — measured under independent graph embeddings. We prove that any graph dynamical system whose embedded state trajectory converges to the posterior distribution of a well-defined latent variable model must satisfy the I-predicate (Theorem 1), with consecutive alignment rate bounded by the per-step information gain and participation ratio decreasing as O(|Θ|/t). A corollary establishes that the Kendall τ to the final state equals 1 under the Bayesian condition. The bound on progressive alignment follows from the Davis-Kahan/Wedin sin θ theorem applied through the chain: posterior concentration (Doob) → graph convergence (smoothness of the encoding) → embedding convergence (regularity condition E1) → subspace alignment (spectral gap condition from E3). The dimensional reduction bound follows from the rank structure of posterior updates propagated through the encoding and embedding. Both bounds hold for any embedding satisfying mild regularity conditions and for any graph topology or update mechanism, extending the empirical findings of Agarwal, Dalal, and Misra (2025–2026) from transformer and Mamba architectures to arbitrary graph dynamical systems. We also prove a partial converse: systems whose spectral gap evolution violates a quantitative monotonicity condition cannot satisfy the I-predicate (Theorem 2). As computational illustration, we construct four graph dynamical systems and verify that the I-predicate correctly classifies all four, including a betweenness-centrality contraction mapping that converges without Bayesian structure and is classified I-negative.

---

## 1. Introduction

### 1.1 Motivation

Recent work by Agarwal, Dalal, and Misra [1, 2] has revealed that transformer neural networks implement exact Bayesian posterior updating, and that this manifests as specific geometric signatures in the network's internal representations: progressive alignment of key-query subspaces, dimensional reduction of value manifolds, and directional flow on the Grassmannian. Remarkably, Zhang Chong [3] and the same authors have shown that these signatures appear identically in Mamba — a state-space model that uses no attention mechanism at all. This architecture-independence suggests the signatures may be properties of Bayesian inference itself, not of any particular computational substrate.

This raises a natural question: under what formal conditions does a discrete dynamical system on graphs provably exhibit these geometric signatures? And can the signatures be assembled into a robust predicate — a formal test — that detects inference-like behavior in systems far removed from neural networks?

### 1.2 Contributions

1. **The I-predicate.** We define a formal predicate for inference-like behavior (Definitions 1–4, Section 2.3), assembled from geometric criteria — progressive alignment to the final state (measured via Kendall τ), dimensional reduction, and data compression — measured under two independent embedding methods. Classification requires satisfaction under at least one embedding, guarded by an anti-convergence condition. The predicate is designed to be an intrinsic property of the dynamical system, not an artifact of the representation.

2. **Forward theorem.** We define precisely what it means for a graph dynamical system to "implement Bayesian posterior updating" (Definition 5) and prove that any such system satisfies the I-predicate, with progressive alignment rate bounded by KL divergence and dimensional reduction rate O(1/t) under any embedding satisfying regularity Conditions E1–E3 (Section 2.2). The proof is given in full in Section 4.

3. **Spectral gap obstruction (partial converse).** We prove that systems whose spectral gap sequence violates a monotonicity condition cannot be I-positive under the Laplacian embedding, under non-degeneracy conditions on the eigenvalue structure (Theorem 2, Section 5). This gives a computationally cheap necessary condition for I-positivity.

4. **Discrimination examples.** We construct four graph dynamical systems — a Bayesian updater (I-positive by construction), a contraction mapping (convergent but non-Bayesian), a deterministic cellular automaton (complex but non-inferential), and a random rewiring process (null model) — and verify that the I-predicate correctly classifies all four (Section 6).

### 1.3 Scope and non-claims

We do not claim that graph dynamical systems "think," "learn," or "reason." The I-predicate detects specific measurable geometric properties; whether those properties constitute "inference" in any deeper sense is an interpretive question that this paper does not address. The analogies to "keys," "queries," and "values" from the transformer literature are suggestive but play no role in the formal definitions — only the measurable geometric quantities matter.

---

## 2. Framework

### 2.1 Graph dynamical systems

**Definition 1 (Graph dynamical system).** A graph dynamical system is a triple (G₀, R, T) where G₀ is an initial graph, R is a deterministic graph rewrite rule, and T ∈ ℕ is a time horizon. The trajectory is the sequence (G₀, G₁, …, G_T) where G_{t+1} = R(G_t).

**Initial condition protocol.** To ensure that classifications reflect intrinsic program behavior rather than initial-condition artifacts, each system is executed from a family of four canonical initial graphs G₀ ∈ {K₁, K₂, K₃, P₃} (single node, single edge, triangle, path of length 3). A system is classified as I-positive only if it satisfies the I-predicate for at least three of the four initial graphs.

### 2.2 Embeddings and regularity conditions

**Definition 2 (Embedding).** An embedding is a map e: {graphs} → ℝ^{n×d} assigning to each graph G a matrix X^e(G) whose rows correspond to nodes and columns to latent dimensions.

**Standard embedding family.** We fix two independent embedding methods E = {e_L, e_R}:

- **e_L (Laplacian eigenvectors).** The columns of X^{e_L}(G) are the d leading eigenvectors of the graph Laplacian L(G) = D − A.

- **e_R (Random projection).** X^{e_R}(G) = A(G) · M where A(G) is the adjacency matrix and M ∈ ℝ^{n×d} is a fixed random Gaussian matrix drawn once and held constant across all systems.

**Conditions E1–E3 (Embedding regularity).**

- **E1 (Continuity).** If G and G' differ by a single edge addition or deletion, then ‖X^e(G) − X^e(G')‖_F ≤ C₁ for a constant C₁ depending only on the maximum degree.

- **E2 (Non-degeneracy).** For any non-isomorphic graphs G, G', the column spaces col(X^e(G)) and col(X^e(G')) are distinct points on the Grassmannian.

- **E3 (Spectral faithfulness).** The singular value distribution of X^e(G) is Lipschitz in the spectral distribution of the graph Laplacian of G.

*Verification.* Laplacian eigenvectors satisfy E1 by Weyl's inequality, E2 by spectral characterization (up to co-spectral exceptions resolved by e_R), and E3 by construction. Random projection satisfies E1 by the Johnson-Lindenstrauss lemma, E2 with probability 1 over M, and E3 by concentration of projected singular values. Detailed proofs are in Appendix A.

*Caveat.* The Laplacian embedding e_L has a sign ambiguity for graphs with repeated Laplacian eigenvalues, frequent in tree-like structures. This destroys alignment signals under e_L even for genuinely inference-like dynamics. The at-least-one rule in Definition 4 handles this: e_R does not suffer from this pathology. See Section 7.3.

### 2.3 The I-predicate

For each embedding e ∈ E and trajectory (G₀, …, G_T), define the embedded trajectory as the sequence of matrices X_t^e = X^e(G_t) ∈ ℝ^{n_t × d}.

**Definition 3 (Bayesian geometric signature).** Let X_T^e denote the embedding of the final graph. Define the cosine-to-final sequence as

    c_t^e = cos θ(col(X_t^e), col(X_T^e))

for t = 0, …, T−1, where cos θ denotes the subspace cosine (the smallest singular value of the product of orthonormal bases for the two column spaces). A trajectory satisfies *progressive alignment* under embedding e if the Kendall rank correlation

    τ_to_final(e) = τ({c_t^e}_{t=0}^{T-1}, {t}_{t=0}^{T-1})

is positive and exceeds a threshold τ* = 0.5 (calibrated against null model N1; see Appendix B).

The trajectory satisfies *dimensional reduction* if the participation ratio PR(X_{t+1}^e − X_t^e) = (∑ σᵢ)² / (n · ∑ σᵢ²) of the residual is monotonically non-increasing in t.

A trajectory satisfies the Bayesian geometric signature if it satisfies both progressive alignment and dimensional reduction.

**Definition 3' (Compression criterion).** Encode the trajectory as a binary string by concatenating adjacency matrices in a canonical node ordering. The trajectory satisfies the compression criterion if its lossless compression ratio ρ(p) satisfies ρ(p) < 0.85.

**Definition 4 (I-positive).** A system producing trajectory (G₀, …, G_T) is I-positive if:

(i) It satisfies the compression criterion (Definition 3'), AND

(ii) It satisfies the Bayesian geometric signature (Definition 3) under at least one embedding in E, AND

(iii) **Anti-convergence guard:** under no embedding in E does τ_to_final fall below −τ*.

The system must satisfy the initial condition protocol: I-positivity must hold for at least three of the four initial graphs in {K₁, K₂, K₃, P₃}.

*Remark on the threshold.* The threshold τ* = 0.5 is justified by the ER null-model ceiling: across 20 independent ER random graph trajectories, the maximum τ_to_final under any embedding is 0.37. The threshold sits above this ceiling and below the Bayesian-theoretic anchor τ = 1 (Corollary 1 below). See the companion paper [6] for full calibration.

### 2.4 Bayesian graph dynamical systems

This is the key definition for the forward theorem.

**Definition 5 (Bayesian graph dynamical system).** A graph dynamical system (G₀, R, T) implements Bayesian posterior updating with respect to a latent variable model (Θ, {P_θ}, π₀) if there exists:

- A finite parameter space Θ with prior π₀,
- A likelihood model P_θ(G_t | G_{t-1}) for each θ ∈ Θ,
- An encoding map φ: Δ(Θ) → {graphs} from probability distributions over Θ to graphs,

such that for all t: G_t = φ(π_t) where π_t(θ) ∝ P_θ(G_t | G_{t-1}) · π_{t-1}(θ), and the encoding φ satisfies:

- **Faithfulness:** φ is injective (distinct posteriors map to distinct graphs).
- **Smoothness:** If ‖μ − ν‖_TV < ε, then d_edit(φ(μ), φ(ν)) ≤ C₂ · ε.

*Remark.* This definition requires the existence of (Θ, {P_θ}, π₀, φ) as a certificate. The forward theorem gives geometric consequences of the certificate's existence, and these consequences are efficiently checkable (via the I-predicate). The partial converse (Section 5) gives an efficiently checkable necessary condition.

---

## 3. Forward Theorem: Statement

### 3.1 Geometric quantities

For reference in the theorem statement:

- **Consecutive subspace cosine:** cos θ(col(X_t^e), col(X_{t+1}^e)) is the cosine of the largest principal angle between the column spaces of successive embeddings.

- **Cosine-to-final:** c_t^e = cos θ(col(X_t^e), col(X_T^e)).

- **Kendall τ to final:** τ_to_final(e) = τ({c_t^e}, {t}).

- **Participation ratio:** PR(M) = (∑ σᵢ)² / (n · ∑ σᵢ²) for singular values σᵢ of M.

- **Cumulative information gain:** I_t = ∑_{s=1}^{t} D_KL(π_s ‖ π_{s-1}).

### 3.2 Statement

**Theorem 1 (Forward theorem).** Let (G₀, R, T) be a Bayesian graph dynamical system with respect to (Θ, {P_θ}, π₀, φ) in the sense of Definition 5, and let e be an embedding satisfying E1–E3. Let K = |Θ| and let G_∞ = φ(δ_{θ*}) be the limit graph. Suppose the singular value gap γ₀ = gap_k(X^e(G_∞)) > 0, where gap_k denotes the separation between the k-th and (k+1)-th singular values of the limit embedding and k is the embedding dimension.

Then there exists a warm-up time t₀ = min{t : I_t > log(4C₁C₂K/γ)} with γ = γ₀/2, such that for all t ≥ t₀:

(i) **Progressive alignment.**

    cos θ(col(X_t^e), col(X_{t+1}^e)) ≥ 1 − C₃ · exp(−2I_{t-1})

where C₃ = 2C₁²C₂²(K−1)²/γ².

(ii) **Dimensional reduction.**

    PR(X_{t+1}^e − X_t^e) ≤ C₄ · K · exp(−I_{t-1}) / n

where C₄ = 4C₂ and n is the number of nodes.

**Corollary 1 (τ_to_final convergence).** Under the hypotheses of Theorem 1, the cosine-to-final sequence c_t^e is bounded below by a non-decreasing sequence for t ≥ t₀. If T > 4t₀ + 1, then

    τ_to_final(e) ≥ 1 − 2t₀ / (T − 1)

In particular τ_to_final → 1 as T → ∞.

**Corollary 2 (I-positivity).** Any Bayesian graph dynamical system with sufficient cumulative information gain I_T > log(4C₁C₂K/γ) and sufficient time horizon T > 4t₀ + 1 satisfies the I-predicate (Definition 4) under any embedding satisfying E1–E3, provided the trajectory compression ratio is below the threshold.

**Corollary 3 (Explicit I-positivity bound).** *Let (G₀, R, T) be a Bayesian graph dynamical system with parameters (Θ, {P_θ}, π₀, φ) as in Definition 5. Let K = |Θ|, and let C₁, C₂, γ₀ be as in Theorem 1. Define the minimum discrimination rate:*

$$D_0 = \min_{\theta \neq \theta^*} D_{\text{KL}}(P_{\theta^*} \| P_\theta) > 0$$

*Then the system is I-positive under any embedding satisfying E1–E3, provided:*

$$T > 4 \left(\frac{4C_1 C_2 (K-1)}{\gamma_0}\right)^{1/D_0} + 1$$

*If additionally the per-step KL divergence is bounded below by $\delta_0 > 0$, the stronger bound holds:*

$$T > \frac{4}{\delta_0} \log\left(\frac{4C_1 C_2 (K-1)}{\gamma_0}\right) + 1$$

*Proof.* By Lemma 1, $I_t \geq D_0 \log t$ for $t$ sufficiently large. Therefore $t_0 \leq (4C_1 C_2 (K-1)/\gamma_0)^{1/D_0}$. By Corollary 2, I-positivity holds for $T > 4t_0 + 1$, giving the first inequality. For the strong-signal case: $I_t \geq \delta_0 t$ gives $t_0 \leq (1/\delta_0) \log(4C_1 C_2(K-1)/\gamma_0)$, and $T > 4t_0 + 1$ gives the second inequality. □

*Remark.* The dominant cost is poor identifiability: the exponent $1/D_0$ shows that hard-to-distinguish hypotheses produce exponentially longer warm-up. In the strong-signal regime ($\delta_0 > 0$), the bound becomes logarithmic in the model parameters.

---

## 4. Forward Theorem: Proof

### 4.1 Notation

Throughout this proof, (G₀, R, T) is a Bayesian graph dynamical system with respect to (Θ, {P_θ}, π₀, φ) in the sense of Definition 5. We write K = |Θ|, π_t ∈ Δ(Θ) for the posterior at time t, G_t = φ(π_t) for the encoded graph, and X_t = X^e(G_t) ∈ ℝ^{n×d} for the embedding under a fixed embedding e satisfying E1–E3. We write σ_j(M) for the j-th singular value of M in decreasing order, col(M) for the column space, d_G(U, V) for the chordal distance on the Grassmannian Gr(k, d), and θ_max(U, V) for the largest principal angle between subspaces U, V.

We write k = min(d, rank(X_t)) for the effective column-space dimension; by E2, k ≥ 1 for all non-trivial graphs. The subspace cosine is cos θ(col(X_t), col(X_{t+1})) = σ_k(U_t^T U_{t+1}) where U_t, U_{t+1} are orthonormal bases for the leading k-dimensional column spaces. This equals the cosine of the largest principal angle θ_max.

### 4.2 Step 1: Posterior concentration (Doob)

**Lemma 1 (Finite Doob consistency).** Let Θ be finite with |Θ| = K, let π₀ be a prior with π₀(θ) > 0 for all θ, and let θ* ∈ Θ be the true parameter. Then for all t ≥ 1:

    ‖π_t − δ_{θ*}‖_TV ≤ (K − 1) · e^{−I_t}

where I_t = ∑_{s=1}^{t} D_KL(π_s ‖ π_{s-1}) is the cumulative information gain.

*Proof.* For each incorrect parameter θ ≠ θ*, the posterior satisfies π_t(θ) = π₀(θ) · ∏_{s=1}^{t} [P_θ(G_s | G_{s-1}) / P_{θ*}(G_s | G_{s-1})]. By the standard martingale argument relating cumulative log-likelihood ratios to cumulative KL divergence (Doob's theorem for finite hypothesis spaces), π_t(θ) ≤ π₀(θ) · e^{−I_t} for each θ ≠ θ*. Summing over θ ≠ θ* and using ‖π_t − δ_{θ*}‖_TV = ∑_{θ ≠ θ*} π_t(θ) gives the bound. ∎

**Corollary.** The consecutive TV distance satisfies ‖π_t − π_{t-1}‖_TV ≤ 2(K−1) · e^{−I_{t-1}}, by the triangle inequality through δ_{θ*}.

### 4.3 Step 2: Graph convergence

**Lemma 2.** Under the smoothness of φ (Definition 5), for all t ≥ 1:

    d_edit(G_t, G_{t+1}) ≤ C₂ · ‖π_t − π_{t+1}‖_TV ≤ 2C₂(K−1) · e^{−I_{t-1}}

*Proof.* Direct from the smoothness condition d_edit(φ(μ), φ(ν)) ≤ C₂ · ‖μ − ν‖_TV and Lemma 1. ∎

### 4.4 Step 3: Embedding convergence

**Lemma 3.** Under E1 (embedding continuity), for all t ≥ 1:

    ‖X_{t+1} − X_t‖_F ≤ C₁ · d_edit(G_t, G_{t+1}) ≤ 2C₁C₂(K−1) · e^{−I_{t-1}}

*Proof.* Each single-edge edit changes the embedding Frobenius norm by at most C₁ (E1). The edit distance counts the minimum number of single-edge operations, so iterating E1 along any realizing sequence and applying the triangle inequality gives ‖X^e(G') − X^e(G)‖_F ≤ C₁ · d_edit(G, G'). Combined with Lemma 2. ∎

**Definition.** Write ε_t = ‖X_{t+1} − X_t‖_F for the embedding perturbation at step t. By Lemma 3, ε_t ≤ 2C₁C₂(K−1) · e^{−I_{t-1}}.

### 4.5 Step 4: Subspace alignment (Davis-Kahan)

This is the central technical step: converting the Frobenius-norm bound on the embedding perturbation into a bound on the angular perturbation of the column space.

#### 4.5.1 The singular value gap condition

**Definition.** For M ∈ ℝ^{n×d} with singular values σ₁ ≥ ··· ≥ σ_d ≥ 0, the k-th singular value gap is gap_k(M) = σ_k(M) − σ_{k+1}(M), where σ_{k+1} = 0 if k = d.

**Proposition 1 (E3 implies gap non-collapse).** Let G_∞ = φ(δ_{θ*}) be the limit graph. If e satisfies E2 and E3, then gap_k(X^e(G_∞)) = γ₀ > 0. Define t₀ = min{t : I_t > log(4C₁C₂(K−1)/γ₀)}. Then for all t ≥ t₀:

    gap_k(X_t) ≥ γ := γ₀/2

*Proof.* Part (a): By E2, col(X^e(G_∞)) is a well-defined k-dimensional subspace, requiring σ_k > 0 separated from σ_{k+1}. E3 ensures the singular values are controlled by the Laplacian spectrum of G_∞, so the gap γ₀ is a positive constant.

Part (b): By Weyl's inequality, |σ_j(X_t) − σ_j(X^e(G_∞))| ≤ ‖X_t − X^e(G_∞)‖_F for each j. By Lemma 3 and Lemma 1, ‖X_t − X^e(G_∞)‖_F ≤ C₁C₂(K−1)e^{−I_t}. The definition of t₀ ensures this is less than γ₀/4 for t ≥ t₀. Since the gap can decrease by at most 2‖X_t − X^e(G_∞)‖_F (from perturbation of both σ_k and σ_{k+1}), we get gap_k(X_t) ≥ γ₀ − 2 · γ₀/4 = γ₀/2. ∎

*Remark.* In the standard Bayesian regime where I_t ≈ D₀ log(t), the warm-up time is t₀ ≈ (4C₁C₂K/γ₀)^{1/D₀}, a finite constant independent of T.

*Remark (Gap for the two standard embeddings).* For e_L, the singular values of X^{e_L}(G) are all 1 (orthonormal eigenvectors), and the relevant "gap" is the Laplacian eigenvalue gap λ_{k+1} − λ_k, which controls eigenvector stability via the classical Davis-Kahan theorem for symmetric matrices. For e_R, the singular values of A(G) · M are controlled by the singular values of A(G) combined with the random matrix M; by Johnson-Lindenstrauss concentration, the gap of AM is within O(√(log n / d)) of the corresponding gap of A.

#### 4.5.2 The Davis-Kahan/Wedin step

**Theorem (Wedin sin θ, applied form).** Let M, M̃ = M + E ∈ ℝ^{n×d}. Let U, Ũ ∈ ℝ^{n×k} be orthonormal bases for the leading k left singular subspaces of M and M̃. If gap_k(M) > 0:

    sin θ_max(col(U), col(Ũ)) ≤ ‖E‖_F / gap_k(M)

This is the rectangular-matrix generalization of the Davis-Kahan theorem (Wedin 1972, Theorem 3.4; Stewart and Sun 1990, Theorem V.4.1).

**Application.** Set M = X_t and E = X_{t+1} − X_t. For t ≥ t₀:

    sin θ_max(col(X_t), col(X_{t+1})) ≤ ε_t / gap_k(X_t) ≤ ε_t / γ

Using the standard bound √(1 − x) ≥ 1 − x/2 for x ∈ [0, 1], with x = sin²θ:

    cos θ(col(X_t), col(X_{t+1})) = √(1 − sin²θ) ≥ 1 − sin²θ / 2 ≥ 1 − ε_t² / (2γ²)

Substituting ε_t ≤ 2C₁C₂(K−1) · e^{−I_{t-1}}:

    cos θ(col(X_t), col(X_{t+1})) ≥ 1 − 2C₁²C₂²(K−1)² / γ² · e^{−2I_{t-1}} = 1 − C₃ · e^{−2I_{t-1}}

This establishes Theorem 1(i). ∎

*Remark (Reconciliation with stated form).* The proved exponent −2I_{t-1} is stronger than the stated −I_t in Theorem 1(i). The stated form follows from the weakening e^{−2I_{t-1}} ≤ e^{−I_t}, valid once I_{t-1} ≥ I_t/2 (i.e., once t ≥ 2).

*Remark (Monotonicity of the bound).* Since I_t is non-decreasing (D_KL ≥ 0 at each step), e^{−2I_{t-1}} is non-increasing in t. Therefore the lower bound 1 − C₃ · e^{−2I_{t-1}} is non-decreasing, approaching 1 as the cumulative information gain grows.

*Remark (Variable node count).* If n_t varies, embeddings live in different ambient spaces. Zero-padding to the common dimension ℝ^{max(n_t, n_{t+1}) × d} preserves column spaces and singular values, so the Wedin bound applies unchanged.

### 4.6 Step 5: Dimensional reduction (rank structure of the update)

**Lemma 4 (Rank of the embedding perturbation).**

(a) Each single-edge edit of G contributes a rank-≤-2 perturbation to X^e(G).

(b) Therefore: rank(X_{t+1} − X_t) ≤ 2 · d_edit(G_t, G_{t+1}) ≤ 4C₂(K−1)e^{−I_{t-1}}.

*Proof of (a).* For e_R: X^{e_R}(G) = A(G)M. Adding edge (u,v) changes A by a rank-2 update (two entries flip), so ΔX = ΔA · M with rank(ΔA) ≤ 2, and rank(ΔA · M) = rank(ΔA) ≤ 2 since M has full column rank with probability 1.

For e_L: adding edge (u,v) perturbs the Laplacian by a rank-2 matrix (e_u − e_v)(e_u − e_v)^T. Under the generic non-degeneracy assumption in E2, each eigenvector perturbation has rank bounded by the multiplicity of the affected eigenspace cluster, which is generically 1. The bound rank ≤ 2d per single-edge edit holds for the d-eigenvector embedding. ∎

**The participation ratio bound.** By Lemma 4(b), for t sufficiently large that 4C₂(K−1)e^{−I_{t-1}} < n, the participation ratio satisfies PR(M) ≤ rank(M)/n for any matrix M, so:

    PR(X_{t+1} − X_t) ≤ rank(X_{t+1} − X_t) / n ≤ 4C₂(K−1) · e^{−I_{t-1}} / n = C₄ · K · e^{−I_{t-1}} / n

where C₄ = 4C₂. This establishes Theorem 1(ii). ∎

#### Rate regimes

The bound PR ≤ C₄ K e^{−I_{t-1}}/n specializes as follows:

- **Standard Bayesian rate** (I_t ≈ D₀ log t): PR ≤ C₄ K t^{−D₀}/n. For D₀ ≥ 1, this is O(K/t).

- **Constant information gain** (D_KL ≈ D per step): PR ≤ C₄ K e^{−Dt}/n — exponential decay, much faster than O(1/t).

- **Vanishing information gain** (I_t → I_∞ < ∞): PR → C₄ K e^{−I_∞}/n > 0. The bound is vacuous, correctly reflecting that the posterior does not concentrate.

### 4.7 Proof of Corollary 1 (τ_to_final = 1)

#### 4.7.1 Setup

Let [V_t] ∈ Gr(k, d) denote the Grassmannian point corresponding to col(X_t). We work with the chordal distance d_G([V], [W])² = ∑_{i=1}^{k} sin²(θ_i), which satisfies d_G ≤ √k · sin θ_max. The triangle inequality holds:

    d_G([V_t], [V_T]) ≤ ∑_{s=t}^{T-1} d_G([V_s], [V_{s+1}])     ... (‡)

#### 4.7.2 Step sizes decrease

Define δ_s = d_G([V_s], [V_{s+1}]) for the step-s chordal increment. By the Wedin bound (Step 4):

    δ_s ≤ √k · sin θ_max(col(X_s), col(X_{s+1})) ≤ √k · ε_s / γ ≤ A · e^{−I_{s-1}}

where A = 2√k · C₁C₂(K−1)/γ. Since I_s is non-decreasing, δ_s is non-increasing for s ≥ t₀.

#### 4.7.3 Monotone approach to the limit

Define the tail sum D_t = ∑_{s=t}^{T-1} δ_s. By (‡), d_G([V_t], [V_T]) ≤ D_t. The tail sum is strictly decreasing: D_t − D_{t+1} = δ_t > 0 (positive as long as the graph is still evolving, which holds for t < T in a non-trivial Bayesian system with positive information gain).

For t ≥ t₀, the step sizes δ_s are non-increasing (Section 4.7.2). The cosine-to-final sequence c_t satisfies c_t ≥ 1 − D_t²/k, where D_t is the tail sum and the bound follows from the triangle inequality (‡) and the relationship d_G² ≤ k · sin²θ_max, cos θ_max = √(1 − sin²θ_max) ≥ 1 − sin²θ_max/2 ≥ 1 − d_G²/k. Since D_t is strictly decreasing (D_t − D_{t+1} = δ_t > 0), the lower bound 1 − D_t²/k is strictly increasing for t ≥ t₀. However, the actual distance d_G([V_t], [V_T]) is bounded above by D_t but may not equal D_t (due to triangle inequality slack), so c_t is bounded below by an increasing sequence but is not proved to be monotone itself.

#### 4.7.4 Kendall τ bound

Since c_t is not proved monotone for t ≥ t₀, we bound discordant pairs conservatively. Each of the t₀ warm-up indices can form a discordant pair with each of the T − t₀ post-warm-up indices, and there are at most t₀(t₀ − 1)/2 pairs within the prefix itself, giving at most t₀(T − t₀) + t₀(t₀ − 1)/2 ≤ t₀ · T discordant pairs out of T(T − 1)/2 total pairs. For the post-warm-up pairs (both indices ≥ t₀): the lower bound on c_t is increasing, so c_{t₂} ≥ 1 − D_{t₂}²/k > 1 − D_{t₁}²/k for t₂ > t₁ ≥ t₀. Since c_{t₁} ≤ 1 and c_{t₂} ≥ 1 − D_{t₂}²/k, a discordant pair (c_{t₁} > c_{t₂}) can only occur if c_{t₁} exceeds 1 − D_{t₂}²/k, which the increasing lower bound does not exclude. We therefore conservatively count all post-warm-up pairs as potentially concordant or discordant, and bound only the warm-up contribution:

    τ_to_final ≥ 1 − 2 · t₀ · T / (T(T − 1)) = 1 − 2t₀ / (T − 1)

For T > 4t₀ + 1, this gives τ_to_final > 0.5, exceeding the I-predicate threshold. For T ≫ t₀, τ_to_final → 1. ∎

---

## 5. Spectral Gap Obstruction (Partial Converse)

### 5.1 Statement

**Theorem 2 (Spectral gap obstruction).** *Let (G₀, R, T) be a graph dynamical system. Denote by λ₁(t) ≤ λ₂(t) ≤ ··· ≤ λ_n(t) the eigenvalues of the graph Laplacian L(G_t), and let v₁(t), ..., v_d(t) be orthonormal eigenvectors for the d smallest non-zero eigenvalues. If there exists a subsequence t₁ < t₂ < ··· < t_m with m > T/3 such that λ₂(t_{i+1}) < λ₂(t_i) − δ for some δ > 0, then (G₀, R, T) is not I-positive under the Laplacian embedding e_L, provided the following non-degeneracy conditions hold:*

**(ND)** *For each decrease step t_i, the eigenvalue λ₂(t_i) is simple (multiplicity 1), and the eigenvalue gap satisfies λ₃(t_i) − λ₂(t_i) ≥ g > 0.*

**(ND')** *The eigenvalue gap at the embedding boundary satisfies λ_{d+2}(t) − λ_{d+1}(t) ≥ g' > 0 for all t.*

*These conditions ensure that eigenvector rotations caused by spectral gap decreases translate to column-space rotations of the d-dimensional Laplacian embedding, rather than being absorbed within the selected eigenspace.*

### 5.2 Proof

**Stage 1 (Eigenvalue decrease → subspace rotation).** At a decrease step t_i where λ₂(t_{i+1}) < λ₂(t_i) − δ, the d-dimensional column space V_i = col(X^{e_L}_{t_i}) = span{v₂(t_i), ..., v_{d+1}(t_i)} is the eigenspace of the d smallest non-trivial Laplacian eigenvalues. Under (ND'), this eigenspace is separated from its complement by gap g'. The eigenvalue λ₂ shifts by more than δ, so the old subspace V_i is suboptimal for the new Laplacian by at least δ in its minimum Rayleigh quotient. By the eigenspace perturbation bound (the subspace generalization of Kato–Temple; see Stewart and Sun [8], Theorem V.2.7):

$$\sin\theta_{\max}(V_i, V_{i+1}) \geq \frac{\delta}{\mathrm{spread}_{d+2}}$$

where spread_{d+2} = max_t(λ_{d+2}(t) − λ₂(t)) is the eigenvalue spread across the selected and first excluded eigenvalues. Define η_d = δ / spread_{d+2} > 0, a positive constant independent of n.

This resolves the d-vs-n issue: the subspace rotation bound η_d depends on the eigenvalue structure of the first d+2 Laplacian eigenvalues (which are intrinsic to the graph and independent of n for n ≫ d), not on the ambient dimension n.

**Stage 2 (Path length exceeds chord length).** The m > T/3 decrease steps contribute total path length ≥ m · arcsin(η_d) on the Grassmannian Gr(d, n). The diameter of Gr(d, n) is π/2 (the maximum principal angle). For T sufficiently large that (T/3) · arcsin(η_d) > π, the path must reverse direction: the cosine-to-final c_t cannot be monotonically increasing because the path overshoots the target.

**Stage 3 (Reversals kill τ_to_final).** Each direction reversal produces a local maximum in c_t followed by a decrease — a discordant pair. The minimum number of reversals is:

$$\text{reversals} \geq \frac{m \cdot \arcsin(\eta_d)}{\pi} - 1 > \frac{T \arcsin(\eta_d)}{3\pi} - 1$$

Each reversal contributes at least one discordant pair to the Kendall τ calculation. By the analysis in Section 4.7, once the number of discordant pairs exceeds T/4, the Kendall τ satisfies τ_to_final < 1 − 2 · (T/4) / T = 0.5 = τ*. The reversal count exceeds T/4 for T > 3π / (4 arcsin(η_d)) + 3, a finite constant. For T above this threshold, the system is not I-positive under e_L. □

*Remark (The non-degeneracy conditions).* Conditions (ND) and (ND') are generically satisfied: eigenvalue multiplicities greater than 1 have codimension ≥ 1 in the space of graph Laplacians, so they are non-generic. For specific rule families (e.g., tree-growing rules that produce high multiplicity at λ = 1), (ND) may fail at position 2 but hold at the cluster level (the cluster gap replaces the individual gap, as in [6, condition M3']). The theorem can be extended to the cluster-gap setting by replacing eigenvector rotation with eigenspace rotation, at the cost of a weaker constant η_d.

*Remark (Scope under the at-least-one rule).* Theorem 2 shows non-I-positivity under e_L only. Under Definition 4's at-least-one rule, the system could still be I-positive via e_R. An analogous obstruction for e_R can be formulated using singular value gap decreases of A(G_t) · M rather than Laplacian eigenvalue decreases. If both obstructions fire simultaneously, the system is I-negative.

*Remark (Computational cost).* The spectral gap obstruction is checkable in O(T · n²) time (one eigenvalue computation per step) and serves as a fast pre-filter for I-negativity.

---

## 6. Computational Illustrations

### 6.1 Example A: Hidden Markov Model (I-positive by construction)

**Setup.** A 3-state HMM with known transition matrix A and emission matrix B. The parameter space is Θ = {θ₁, θ₂, θ₃}.

**Graph encoding.** At each time step t, construct a weighted graph G_t with nodes corresponding to hidden states and edge weights w_{ij}(t) = π_t(θ_i) · A_{ij} · B_{j, o_t}. The rewrite rule R updates weights by one step of the forward algorithm.

**Verification.** This construction satisfies Definition 5 by design. The encoding φ maps the posterior to the graph via weight assignment; faithfulness and smoothness hold because edge weights are linear in the posterior, with C₂ = max_{ij} A_{ij} · B_{j,o}.

**Prediction and result.** Theorem 1 predicts progressive alignment at rate bounded by D_KL(π_t ‖ π_{t-1}) and PR = O(3/t). Corollary 1 predicts τ_to_final = 1. Both are verifiable analytically from the known posterior concentration rate of 3-state HMMs and confirmed numerically.

### 6.2 Example B: Contraction mapping (convergent, non-Bayesian, I-negative)

**Setup.** A graph on n = 20 nodes. Rewrite rule: at each step, remove the edge with maximum betweenness centrality and add an edge between the two nodes closest in graph distance.

**Behavior.** This contracts the graph toward a path or tree. Entropy decreases, trajectories converge. Superficially similar to inference, but there is no latent variable model — convergence is driven by a fixed deterministic rule, not by evidence accumulation.

**Result.** Under the canonical specification (4 initial graphs bootstrapped to n = 20, T = 30, 3 embeddings), Example B classifies as I-negative with score 2/4 (below the 3/4 majority threshold). The per-seed τ values are 0.50, 0.44, 0.00, and 0.50. The contraction mapping reaches its fixed point within 3–5 steps, after which all cosines-to-final are identically 1.0; the τ_to_final depends entirely on the brief pre-convergence transient.

**Diagnostic finding.** The discrimination margin is thin (two seeds at τ = 0.503). Robustness testing on 20 random initial graphs shows 7/20 producing I-positive classifications with τ up to 0.71. Analysis of the Grassmannian trajectory reveals a discriminating structural feature: contraction trajectories have high *straightness* (ratio of direct Grassmannian distance to total path length ≈ 0.5–0.8, indicating near-geodesic paths), while genuinely I-positive rules show low straightness (≈ 0.34–0.37, indicating winding, evidence-dependent paths). This observation motivates a Grassmannian straightness gate for future predicate revisions (see Section 7.4).

### 6.3 Example C: Cellular automaton (complex, non-inferential, I-negative)

**Setup.** A 1D cellular automaton (Rule 110) lifted to a graph: nodes are cells, edges connect neighbors, state evolves by the automaton rule.

**Behavior.** Complex, Turing-complete dynamics. No convergence, no posterior concentration, no dimensional reduction. High compression ratio (complex structure) but no progressive alignment.

**Result.** I-negative. The spectral gap obstruction (Theorem 2) fires: the spectral gap oscillates irregularly as the automaton evolves.

### 6.4 Example D: Random rewiring (null model, I-negative)

**Setup.** Start from G(20, 0.3). At each step, perform one degree-preserving random edge swap (null model N1).

**Result.** I-negative. No progressive alignment (τ_to_final ≈ 0, consistent with random walk on the Grassmannian). Compression ratio hovers at the null-model baseline. This IS the null model against which τ* and the compression threshold are calibrated.

### 6.5 Summary

| Example | Bayesian? | Convergent? | I-class | Spectral gap obstruction? |
|---------|-----------|-------------|---------|--------------------------|
| A (HMM) | Yes | Yes | I-positive | No (gap increases) |
| B (Contraction) | No | Yes | I-negative | Possibly not |
| C (CA Rule 110) | No | No | I-negative | Yes (gap oscillates) |
| D (Random rewiring) | No | No | I-negative | Yes (gap fluctuates) |

Example B is the critical test of predicate specificity. The I-predicate correctly classifies it as I-negative under the canonical specification, though the margin is thin and motivates the straightness gate discussed in Section 7.4.

---

## 7. Discussion

### 7.1 What "architecture independence" means

The forward theorem holds for any system satisfying Definition 5 regardless of graph topology or update mechanism. This is genuine architecture independence: the same geometric signatures emerge whether the "hardware" is a transformer, a Mamba block, or an HMM encoded as a graph. However, we do not claim the converse — that any I-positive system is necessarily Bayesian. Closing this gap is an open problem (Section 7.5).

### 7.2 The I-predicate as a standalone contribution

The I-predicate (Definition 4) is designed to be useful independently of any particular conjecture. It provides a formal, measurable test for a specific class of dynamical behavior; built-in robustness via independent embeddings with an at-least-one rule and anti-convergence guard; a built-in diagnostic for its own reliability (the embedding-sensitivity check); and a continuous relaxation (the I-score) suitable for temporal profiling.

The forward theorem validates the predicate in one direction (Bayesian systems are detected). The discrimination examples probe its specificity (non-Bayesian systems are rejected). Together, these establish the predicate as a well-characterized tool for classifying dynamical behavior on graphs.

### 7.3 Limitations

**Finite Θ.** The forward theorem requires a finite parameter space. Extension to continuous Θ requires Bernstein-von Mises type arguments.

**The encoding φ.** The smoothness condition on φ is the least natural part of Definition 5. In practice, graph encodings of belief states may be discontinuous when the topology changes discretely as the posterior crosses a threshold. Weakening smoothness to piecewise smoothness is possible but complicates the constants.

**The O(1/t) rate.** The participation ratio bound O(K/t) comes from standard Bayesian concentration. In systems with weak likelihood (low signal-to-noise), concentration is slow and the bound is loose.

**Laplacian sign ambiguity.** The eigenvector embedding e_L has a sign/basis ambiguity for graphs with repeated Laplacian eigenvalues, frequent in tree-like structures. This destroys the τ_to_final signal under e_L even for genuinely inference-like dynamics. The at-least-one rule is specifically designed for this; e_R carries the classification alone in such cases.

**The singular value gap γ₀.** The forward theorem requires γ₀ = gap_k(X^e(G_∞)) > 0. If the limit graph has degenerate singular values in the embedding, the bound degrades. This is a genuine limitation: for embeddings where the limit is poorly conditioned, the theorem gives a weaker or vacuous bound.

### 7.4 The Grassmannian straightness signal

Example B revealed that the I-predicate's discrimination between Bayesian convergence and generic contraction is thinner than desired. The deep diagnostic identified a structural discriminator: the *straightness* of the Grassmannian trajectory, defined as the ratio of direct distance (start to end) to total path length (sum of consecutive steps):

    S = d_G([V_0], [V_T]) / ∑_{s=0}^{T-1} d_G([V_s], [V_{s+1}])

Contraction mappings follow near-geodesic paths to their fixed point (S ≈ 0.5–0.8). Genuinely inference-like systems follow winding paths as evidence accumulates from different directions (S ≈ 0.34–0.37). A formal straightness gate — rejecting trajectories with S above a calibrated threshold — is under investigation for a future predicate revision. The forward theorem's proof structure suggests that Bayesian systems should produce non-geodesic trajectories (the evidence-dependent winding of the Grassmannian path reflects the stochastic nature of the likelihood), but formalizing this into a provable bound is future work.

### 7.5 Open problems

1. **Tight characterization of I-positivity.** What is the exact class of systems that satisfy the I-predicate? Is it strictly larger than Bayesian systems?

2. **Continuous Θ.** Extend Theorem 1 to continuous parameter spaces.

3. **Rate optimality.** Is the information-gain bound in Theorem 1(i) tight, or can a tighter connection to the Fisher information be established?

4. **Computational complexity of checking Definition 5.** Given a graph rewrite rule R, how hard is it to determine whether (G₀, R, T) admits a Bayesian certificate?

5. **Dimensional reduction criterion.** The participation ratio criterion in Definition 3 uses strict monotonicity. Can this be relaxed to a Kendall τ formulation?

6. **Straightness bound for Bayesian systems.** Prove that Bayesian graph dynamical systems satisfying Definition 5 produce Grassmannian trajectories with straightness bounded away from 1, i.e., S ≤ 1 − f(K, C₁, C₂) for some explicit function f.

---

## References

[1] Agarwal, N., Dalal, S.R., Misra, V. "The Bayesian Geometry of Transformer Attention." arXiv:2512.22471, Jan 2026.
[2] Agarwal, N., Dalal, S.R., Misra, V. "Gradient Dynamics of Attention." arXiv:2512.22473, Dec 2025.
[3] Zhang Chong. "Attention Is Not What You Need: Grassmann Flows." arXiv:2512.19428, Dec 2025.
[4] Kim, G. "Thermodynamic Isomorphism of Transformers." arXiv:2602.08216, Feb 2026.
[5] Davis, C., Kahan, W. "The Rotation of Eigenvectors by a Perturbation." SIAM J. Numer. Anal. 7(1), 1970.
[6] K. Kowalczyk. "Geometric Predicates for Classifying Dynamical Behaviors in Graph Rewrite Systems." In preparation, 2026.
[7] Wedin, P.-Å. "Perturbation Bounds in Connection with Singular Value Decomposition." BIT 12, 1972.
[8] Stewart, G.W., Sun, J.-g. *Matrix Perturbation Theory.* Academic Press, 1990.
[9] Doob, J.L. "Application of the Theory of Martingales." Le calcul des probabilités et ses applications, CNRS, 1949.
[10] Johnson, W.B., Lindenstrauss, J. "Extensions of Lipschitz Mappings into a Hilbert Space." Contemporary Mathematics 26, 1984.
[11] Weyl, H. "Das asymptotische Verteilungsgesetz der Eigenwerte." Math. Ann. 71, 1912.
[12] Grover, A., Leskovec, J. "node2vec: Scalable Feature Learning for Networks." KDD 2016.
[13] Trugenberger, C.A. et al. "Dynamics and the Emergence of Geometry in an Information Mesh." EPJC 80, 1091, 2020.
[14] Vanchurin, V. "Geometric Learning Dynamics." arXiv:2504.14728, Apr 2025.

---

## Appendix A: Verification of E1–E3 for Standard Embeddings

### A.1 Laplacian eigenvectors

**E1 (Continuity).** Adding or removing a single edge (u,v) perturbs the Laplacian by ΔL = ±(e_u − e_v)(e_u − e_v)^T, a rank-1 update with ‖ΔL‖_2 ≤ 2. By Weyl's inequality, each eigenvalue shifts by at most 2. By the Davis-Kahan theorem applied to the Laplacian (symmetric matrix), each eigenvector shifts by at most 2/min_j |λ_j − λ_{j'}| in the ℓ₂ norm, where the minimum is over eigenvalue gaps. Summing over d eigenvectors and n rows gives ‖ΔX^{e_L}‖_F ≤ C₁ with C₁ depending on d and the minimum eigenvalue gap, which in turn depends on the maximum degree.

**E2 (Non-degeneracy).** Two non-isomorphic graphs can share the same Laplacian spectrum (co-spectral graphs), but co-spectral graphs with identical eigenvector structure are extremely rare and can be detected by combining e_L with e_R. For the purposes of the theorem, E2 holds generically.

**E3 (Spectral faithfulness).** The singular values of X^{e_L}(G) are all 1 (the eigenvectors are orthonormal). The relevant spectral information is carried by which eigenvalues are selected (the d smallest non-zero eigenvalues). The selection is Lipschitz in the Laplacian spectrum by Weyl's inequality.

### A.2 Random projection

**E1 (Continuity).** X^{e_R}(G) = A(G)M. A single-edge change alters one row of A by ±1 in one entry, giving ‖ΔX^{e_R}‖_F ≤ ‖ΔA‖_F · ‖M‖_2 ≤ √2 · ‖M‖_2. By standard random matrix concentration, ‖M‖_2 ≤ √n + √d + O(√log n) with high probability, giving C₁ = O(√n).

**E2 (Non-degeneracy).** Two graphs with distinct adjacency matrices A ≠ A' satisfy A · M ≠ A' · M with probability 1 over M (since M has i.i.d. Gaussian entries and A − A' ≠ 0).

**E3 (Spectral faithfulness).** The singular values of AM are controlled by those of A: by the Johnson-Lindenstrauss guarantee, the singular values of AM concentrate around those of A (scaled by √d/n), with deviation O(√(log n / d)). This is Lipschitz in the singular values of A, which are in turn Lipschitz in the Laplacian spectrum (since A and L = D − A differ by the diagonal degree matrix).

---

## Appendix B: Null Model Specification

**Null model N1 (Degree-preserving random dynamics).** Given G₀ and T, generate a trajectory by performing one degree-preserving random edge swap at each step: select two edges (u,v) and (w,x) uniformly at random, replace with (u,x) and (w,v) if the swap does not create multi-edges or self-loops. Generate 100 such trajectories. Compute ρ_null and σ_null as the sample mean and standard deviation of the compression ratio. The threshold τ* = 0.5 is set above the observed maximum τ_to_final of 0.37 across 20 independent ER trajectories.

---

## Appendix C: Computational Details for Examples A–D

**Parameters.** All examples: T = 30, embedding dimension d = 5, compression via zlib level 9 on JSON-serialized edge lists. Initial graphs: K₁, K₂, K₃, P₃, each bootstrapped to n = 20 for Example B.

**Example B implementation.** The contraction rule computes betweenness centrality (NetworkX), removes the highest-centrality edge, and adds an edge between the closest non-adjacent pair by graph distance. If the removal disconnects the graph, reconnection takes priority (highest-degree nodes in the two largest components are connected). Reproducible with numpy seed 42.

**Code.** Full implementations are provided as supplementary material: example_b_test.py (main test), example_b_deep_diagnostic.py (Grassmannian trajectory analysis), primo_diagnostic_v5.py (33-rule diagnostic framework).

---

## Appendix D: Summary of Constants

| Symbol | Definition | Depends on |
|--------|-----------|------------|
| K = \|Θ\| | Parameter space size | Model |
| C₁ | Embedding continuity (E1) | max degree, embedding e |
| C₂ | Encoding smoothness (Def 5) | φ |
| γ₀ = gap_k(X^e(G_∞)) | Limit singular value gap | G_∞, e, k |
| γ = γ₀/2 | Working gap constant | γ₀ |
| C₃ = 2C₁²C₂²(K−1)²/γ² | Progressive alignment bound | C₁, C₂, K, γ |
| C₄ = 4C₂ | Dimensional reduction bound | C₂ |
| D₀ = min_{θ≠θ*} D_KL(P_{θ*} ∥ P_θ) | Minimum discrimination rate | Model (identifiability) |
| t₀ ≤ (4C₁C₂(K−1)/γ₀)^{1/D₀} | Warm-up time (logarithmic info gain) | C₁, C₂, K, γ₀, D₀ |
| t₀ ≤ (1/δ₀) log(4C₁C₂(K−1)/γ₀) | Warm-up time (linear info gain, rate δ₀) | C₁, C₂, K, γ₀, δ₀ |
| T > 4t₀ + 1 | I-positivity threshold (Corollary 3) | t₀ |

The minimum discrimination rate D₀ is the standard Bayesian identifiability parameter. It is positive iff the model is identifiable (distinct parameters produce distinct observation distributions). In the degenerate case D₀ = 0, the posterior never concentrates and the forward theorem gives no finite-time guarantee.
