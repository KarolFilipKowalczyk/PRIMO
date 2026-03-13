# PRIMO: Primacy of Inference over Physics in the Space of Minimal Programs

**Author:** Karol (AIRON Games / MIMUW)
**Date:** March 2026
**Status:** Draft v6

---

## 1. Overview

We conjecture that in the space of all programs ordered by Kolmogorov complexity, inference-like behavior appears at strictly shorter program lengths than physics-like behavior, and that physics-like behavior is generically the dynamical equilibrium of an inference-like process. The conjecture is formalized using predicates assembled from existing peer-reviewed frameworks and is testable via program enumeration.

---

## 2. Definitions

Throughout, fix a universal prefix-free Turing machine $U$. Let $\mathcal{P}_N = \{p : |p| \leq N\}$ denote the set of all programs of length at most $N$. Each $p \in \mathcal{P}_N$ is executed on a family of canonical initial graphs $\mathcal{G}_0 = \{K_1, K_2, K_3, P_3\}$ (single node, single edge, triangle, path of length 3) for $T$ steps, producing a trajectory $(G_0, G_1, \ldots, G_T)$ for each $G_0 \in \mathcal{G}_0$. A program is classified as $I$-positive or $\Phi$-positive only if it satisfies the relevant predicate for *at least three of the four* initial graphs in $\mathcal{G}_0$, ensuring that classifications reflect intrinsic program behavior rather than initial-condition artifacts.

### 2.1 The $I$-predicate (inference-like behavior)

We require three independent embedding methods $\mathcal{E} = \{e_L, e_N, e_R\}$: graph Laplacian eigenvectors, node2vec, and random projection of the adjacency matrix. For each embedding $e \in \mathcal{E}$, the trajectory is mapped to a sequence of matrices $X_t^e \in \mathbb{R}^{n \times d}$.

**Definition 1** (Bayesian geometric signature). A trajectory satisfies *progressive alignment* under embedding $e$ if $\cos\theta(\mathrm{col}(X_t^e), \mathrm{col}(X_{t+1}^e))$ is monotonically increasing in $t$, and satisfies *dimensional reduction* if the participation ratio of singular values of $X_{t+1}^e - X_t^e$ is monotonically decreasing in $t$. (After Agarwal, Dalal, Misra [7, 8].)

**Definition 2** (Grassmann flow). A trajectory exhibits *directional flow* under embedding $e$ if the sequence of points $[\mathrm{col}(X_t^e)]$ on the Grassmannian $\mathrm{Gr}(k, d)$ has non-zero mean drift exceeding a null-model baseline. It exhibits *semantic clustering* if trajectories from nearby initial conditions converge on $\mathrm{Gr}(k, d)$. (After Zhang Chong [9].)

**Definition 3** (Compression gate). A trajectory satisfies the *compression criterion* if its lossless compression ratio $\rho(p)$ satisfies $\rho(p) < \rho_{\mathrm{null}} - 2\sigma_{\mathrm{null}}$, where $\rho_{\mathrm{null}}$ and $\sigma_{\mathrm{null}}$ are the mean and standard deviation over a degree-preserving random graph dynamics null model.

**Definition 4** ($I$-positive). A program $p$ is *$I$-positive* if it satisfies the compression criterion (Definition 3) and at least one of Definitions 1, 2, under *all three* embeddings in $\mathcal{E}$.

A program is *embedding-sensitive* if its classification changes across embeddings. If the majority of candidate $I$-positive programs are embedding-sensitive, the predicate requires redesign.

**Remark.** The intersection rule across embeddings ensures that $I$-positivity is an intrinsic dynamical property, not an artifact of the representation. The analogies to "keys," "queries," and "values" in Definitions 1–2 are suggestive but play no role in the formal definitions; only the measurable geometric quantities matter (progressive alignment, dimensional reduction, directional flow, compression). The architecture-neutrality of these signatures is established empirically in [7, 8]: Mamba, a state-space model without attention, exhibits identical geometric structure.

### 2.2 The $\Phi$-predicate (physics-like behavior)

**Definition 5** (Spectral dimension). Compute the spectral dimension via diffusion return probability: $d_s(\sigma) = -2\, d\log P(\sigma) / d\log\sigma$. A trajectory satisfies *stable spectral dimension* if $d_s$ converges to a value in $\{1, 2, 3, 4, \ldots\}$ within tolerance $\epsilon$ across a range of scales, persisting over the evolution. (After Trugenberger [11].)

**Definition 6** (Approximate symmetry). A trajectory exhibits *approximate isometry invariance* if the variance of Ollivier-Ricci curvature under random graph relabelings consistent with the emergent geometry is below threshold $\delta_1$.

**Definition 7** (Conservation). A trajectory exhibits a *conservation law* if at least one nontrivial aggregate quantity (total edges, degree moments, spectral gap) has temporal variance below threshold $\delta_2$ over the observation window.

**Definition 8** ($\Phi$-positive). A program $p$ is *$\Phi$-positive* if it satisfies stable spectral dimension (Definition 5) and at least one of Definitions 6, 7.

**Null models.** Erdős–Rényi random graphs do not produce stable integer spectral dimension, providing a clean baseline for Definition 5. Degree-preserving random rewiring serves as baseline for Definitions 6–7. As with the $I$-predicate, these definitions rest on measurable graph-theoretic quantities alone; whether stable integer spectral dimension constitutes "emergent spacetime" is an interpretive claim of the source literature that plays no role in the formal framework.

### 2.3 Independence of predicates

The $I$-predicate measures geometric properties of state-space trajectories (alignment, dimensional reduction, Grassmann flow). The $\Phi$-predicate measures geometric properties of the graph itself (spectral dimension, curvature symmetry, conservation). These are independent measurement domains: a program may be $I$-positive, $\Phi$-positive, both, or neither. The conjecture that these classes are structurally related is therefore nontrivial.

---

## 3. The PRIMO Conjecture

Define:
$$N_I^{\min} = \min\{|p| : p \text{ is } I\text{-positive}\}, \qquad N_\Phi^{\min} = \min\{|p| : p \text{ is } \Phi\text{-positive}\}.$$

For a $\Phi$-positive program $p$, define its *temporal $I$-profile* as the function $t \mapsto I\text{-score}(p, [t, t+W])$ computed over a sliding window of width $W$.

**Conjecture (PRIMO, core).**

**(a) Ordering.** $N_I^{\min} < N_\Phi^{\min}$.

**(b) Equilibrium.** For $\Phi$-positive programs $p$ with $|p| \leq N_\Phi^{\min} + \Delta$ (where $\Delta$ is initially set to 5 bits and varied in sensitivity analysis): the temporal $I$-profile exhibits an $I$-positive transient phase that precedes and decays as the $\Phi$-positive state stabilizes. Moreover, perturbing such a program away from its $\Phi$-positive fixed point produces a return trajectory with elevated $I$-scores.

Claim (b) is the structural heart of the conjecture: physics-like behavior is the equilibrium of inference-like dynamics, not a co-occurring phenomenon.

**Secondary hypotheses** (tracked, not part of the core conjecture):

*S1 (Complexity ladder).* Define $I_0 = \{p : p \text{ is } I\text{-positive}, p \text{ is never } \Phi\text{-positive}\}$ and $I_+ = \{p : p \text{ is } I\text{-positive and enters a } \Phi\text{-positive region}\}$. Then $\min\{|p| : p \in I_0\} < N_\Phi^{\min} < \min\{|p| : p \in I_+\}$.

*S2 (Frequency dominance).* Let $I(N)$ and $\Phi(N)$ denote the fractions of $I$-positive and $\Phi$-positive programs in $\mathcal{P}_N$. Then $I(N) > \Phi(N)$ for all sufficiently large $N$.

---

## 4. Prior Work

The conjecture synthesizes four independent lines of research. We state what each contributes and what it does not.

**Vanchurin** [1–4] shows that learning dynamics reproduces quantum mechanics near equilibrium (Madelung equations) and classical mechanics far from equilibrium (Hamilton-Jacobi equations), and with Koonin and Katsnelson extends this to biological evolution as multilevel learning. His "Geometric Learning Dynamics" [4] identifies three regimes ($\alpha = 0, \tfrac{1}{2}, 1$) of the metric-noise relationship. This directly motivates claim (b): physics as equilibrium of learning. It does not address program-space ordering (claim (a)), nor does it provide discrete-graph predicates.

**Müller** [5, 6] establishes that Solomonoff induction over observer self-states yields external regularities as asymptotic statistical phenomena. His algorithmic idealism framework provides the formal setting for restating PRIMO as a claim about the Solomonoff prior: $M(I) = \sum_{p \in I} 2^{-|p|} > M(\Phi) = \sum_{p \in \Phi} 2^{-|p|}$. It does not address the temporal relationship between $I$ and $\Phi$ within individual programs.

**Agarwal, Dalal, Misra** [7, 8] demonstrate that transformers implement exact Bayesian inference, with specific geometric signatures (orthogonal key bases, progressive query-key alignment, low-dimensional value manifolds). Crucially, these signatures are architecture-neutral — they appear identically in Mamba. This supplies the measurable criteria for the $I$-predicate. The papers do not address arbitrary dynamical systems on graphs; we extract only the mathematical properties (Definitions 1–2), not the interpretive claims.

**Trugenberger** [11] shows that spacetime emerges as the ground state of graph Hamiltonians via Ollivier-Ricci curvature, with spectral dimension convergence as the diagnostic. This supplies the $\Phi$-predicate. The work does not address inference-like behavior.

**Zenil** [*] has developed Algorithmic Information Dynamics — a framework for enumerating small Turing machines and classifying outputs by algorithmic complexity. His Coding Theorem Method and Block Decomposition Method are operationally adjacent to our Approach B, but classify by complexity rather than by dynamical behavior type. PRIMO's enumeration engine should build on or interface with this infrastructure.

**Additional sources:** Grassmann flows [9] supply the geometric characterization of Definition 2. Kim's thermodynamic isomorphism [10] establishes the instability of the zero-inference fixed point — the system is repelled from "no inference" — which provides heuristic motivation for claim (a) but not a proof. Van Raamsdonk [12] and the relational-informational framework [13] supply background on emergent geometry from entanglement and information constraints. Wolfram and Gorard [15, 16] demonstrate that simple graph-rewriting rules can produce physics; they do not address relative frequency of behavioral classes. Ramsauer et al. [14] connect Hopfield networks to attention.

---

## 5. Theoretical Approach

### 5.1 The Vanchurin fixed-point argument

Vanchurin's derivation operates in a continuous setting. The key technical challenge is formalizing "equilibrium" for discrete graph dynamics.

**Proposed definition.** A graph dynamical system $(G_t)$ is at *discrete equilibrium* at time $t_0$ if for all $t > t_0$: (i) the $\Phi$-score remains above threshold, and (ii) the $I$-score over sliding windows $[t, t+W]$ drops below the null-model baseline.

If this definition captures Vanchurin's continuous equilibrium — i.e., if the vanishing of the loss gradient corresponds to the $I$-score dropping to baseline — then claim (b) follows from his results, provided our $I$-predicate detects the relevant features of learning dynamics.

The perturbation-response test strengthens this: if perturbing a $\Phi$-positive system away from equilibrium produces a return trajectory with elevated $I$-scores, the physics-like state is actively maintained by inference-like dynamics.

**Risk.** The notion of "gradient" has no obvious discrete analog. The discrete equilibrium definition may fail to capture the same phenomenon. This is the principal theoretical gap. However, the definition is *falsifiable independently of PRIMO*: one can test whether it agrees with Vanchurin's continuous equilibrium in cases where both frameworks apply — e.g., small neural networks whose continuous learning dynamics is known and which can simultaneously be represented as graph dynamical systems. This validation step should precede any use of the definition in testing the conjecture.

### 5.2 Complexity bounds via Müller

In Müller's framework, PRIMO becomes: $M(I) > M(\Phi)$, where $M(\cdot)$ is algorithmic probability. If $I$-positive programs are shorter (claim (a)) and more numerous (hypothesis S2), this follows. The framework handles the uncomputability of $K$ through asymptotic bounds on the Solomonoff prior.

**Risk.** Müller's asymptotic machinery may not apply at the finite $N$ values ($N = 20$–$40$) where computation is feasible.

### 5.3 What the thermodynamic analogy does and does not do

Kim's thermodynamic isomorphism [10] suggests that $\Phi$-positive behavior (long-range geometric order) generically requires more structure than $I$-positive behavior (local alignment), just as long-range order requires more entropy reduction than short-range order. This is a heuristic for claim (a), not a proof. Program space is not a thermodynamic system; it has no temperature, no spatial locality, no partition function.

---

## 6. Computational Approach

### 6.1 Enumeration

Choose a minimal universal language for graph rewriting. Candidates: binary lambda calculus (clean complexity theory), Wolfram-style hypergraph rules (comparability with existing results), or a custom graph-rewrite DSL building on the Game of Intelligence framework. Enumerate all programs of length $|p| \leq N$ for target $N = 20$–$40$ bits. Execute each on the family $\mathcal{G}_0$ for $T$ steps.

### 6.2 Measurement

For each program, compute $I$-scores under the three-embedding protocol (Definition 4) and $\Phi$-scores via spectral dimension, curvature, and conservation (Definition 8), both against documented null models.

### 6.3 Tests

**Ordering test.** Measure $N_I^{\min}$ and $N_\Phi^{\min}$. Claim (a) predicts $N_I^{\min} < N_\Phi^{\min}$.

**Temporal profile test.** For each $\Phi$-positive program, compute the temporal $I$-profile. Claim (b) predicts an $I$-positive transient preceding $\Phi$-positive stabilization.

**Perturbation-response test.** For the shortest $\Phi$-positive programs, perturb away from the fixed point (random edge additions/deletions) and measure whether the return trajectory is $I$-positive.

**Null controls.** (i) Shuffled temporal profiles: randomly permute time steps before computing scores, to test whether $I$-before-$\Phi$ ordering is meaningful. (ii) Reversed trajectories: run programs in reverse time to check directionality. (iii) Threshold sensitivity: repeat all analysis across a range of threshold values.

### 6.4 Case studies

Manually analyze the shortest $\Phi$-positive programs: temporal profile, nature of the $I$-positive transient, perturbation response, and whether any known Wolfram rules appear. Any $\Phi$-positive program with no detectable $I$-positive transient is a counterexample to claim (b) and requires careful analysis.

### 6.5 Connection to existing infrastructure

The Game of Intelligence framework provides anti-loop graph growth rules (a constrained search through graph-rewriting programs), Kolmogorov complexity constraints, and degree distribution analysis. The first experiment: measure $I$-scores on the anti-loop rules that produce the most interesting emergent structure.

### 6.6 Risks

Computational cost ($N = 30$ yields $\sim 10^9$ programs, tripled by the three-embedding protocol). The intersection rule may be too conservative. The perturbation model (random edge changes) is not canonical. The interesting region may require $N$ beyond reach. Null models may exhibit nontrivial structure.

---

## 7. Falsification

**Against (a).** Finding $\Phi$-positive programs at lengths $\leq N_I^{\min}$ refutes the ordering claim.

**Against (b).** If $> 20\%$ of $\Phi$-positive programs show no $I$-positive transient, or if perturbed $\Phi$-positive systems return to equilibrium via non-$I$-positive trajectories, the equilibrium claim fails.

**Against the predicates.** If most $I$-classifications are embedding-sensitive, the $I$-predicate does not capture an intrinsic property and the framework needs redesign before any claims can be tested.

**Global.** If the $(I, \Phi)$ phase diagram shows no structure — $I$-positivity and $\Phi$-positivity are statistically independent of each other and of program length — PRIMO is wrong.

**Against secondary hypotheses.** $I_+$ programs at lengths $\leq N_\Phi^{\min}$ refute S1. $\Phi(N) \geq I(N)$ for a range of $N$ refutes S2. Failure of S1 or S2 does not refute the core conjecture.

---

## 8. Implications

If claims (a) and (b) hold:

**Algorithmic information theory.** The distribution of dynamical behaviors in program space has nontrivial structure: inference-like behaviors appear at shorter description lengths, and the two classes are related by a fixed-point/transient relationship. This is a structural result about the Solomonoff prior.

**Foundations of physics.** Physical law as a fixed-point phenomenon of inference-like dynamics (building on Vanchurin [1–4]) gains complexity-theoretic support.

**Deep learning theory.** If claim (a) holds, then inference-like operations are algorithmically simpler than physics-like operations, providing a complexity-theoretic explanation for the empirical effectiveness of attention-based and related architectures.

PRIMO makes no claims about consciousness, intelligence, or the hard problem.

---

## 9. Plan

| Month | Deliverable |
|-------|------------|
| 1 | $I$-predicate implementation with embedding robustness protocol on existing Game of Intelligence data. Null model validation. Embedding-sensitivity diagnostic. **Go/no-go criterion:** if fewer than 70% of candidate $I$-positive programs are classified consistently across all three embeddings, halt and redesign the predicate before proceeding. |
| 2–3 | $\Phi$-predicate implementation. First $(I, \Phi)$ scatter plots and temporal profiles on existing data. |
| 3–6 | Enumeration engine. Ordering test, temporal profile test, perturbation-response test at small $N$. |
| 1–6 (parallel) | Discrete equilibrium formalization (§5.1). |
| 6+ (conditional) | If core claims confirm: secondary hypotheses S1, S2 at larger $N$. If not: characterize actual structure. |

---

## 10. References

### Core
1. Vanchurin, V. "The World as a Neural Network." *Entropy* 22(11), 1210, 2020.
2. Vanchurin, V. "Towards a Theory of Machine Learning." *MLST* 2(3), 2021.
3. Vanchurin, V., Wolf, Y.I., Katsnelson, M.I., Koonin, E.V. "Toward a Theory of Evolution as Multilevel Learning." *PNAS* 119(6), 2022.
4. Vanchurin, V. "Geometric Learning Dynamics." arXiv:2504.14728, Apr 2025.
5. Müller, M.P. "Algorithmic Idealism." *Found. Phys.* 56, 11, 2026. arXiv:2412.02826.
6. Müller, M.P. "Law Without Law." *Quantum* 4, 301, 2020. arXiv:1712.01826.

### Inference signatures
7. Agarwal, N., Dalal, S.R., Misra, V. "The Bayesian Geometry of Transformer Attention." arXiv:2512.22471, Jan 2026.
8. Agarwal, N., Dalal, S.R., Misra, V. "Gradient Dynamics of Attention." arXiv:2512.22473, Dec 2025.
9. Zhang Chong. "Attention Is Not What You Need: Grassmann Flows." arXiv:2512.19428, Dec 2025.
10. Kim, G. "Thermodynamic Isomorphism of Transformers." arXiv:2602.08216, Feb 2026.

### Physics-like behavior
11. Trugenberger, C.A. et al. "Dynamics and the Emergence of Geometry in an Information Mesh." *EPJC* 80, 1091, 2020.
12. Van Raamsdonk, M. "Building up Spacetime with Quantum Entanglement." *Gen. Rel. Grav.* 42, 2323–2329, 2010.
13. "An Axiomatic Relational-Informational Framework." *Axioms* 15(2), 154, Feb 2026.

### Infrastructure and methods
14. Ramsauer, H. et al. "Hopfield Networks is All You Need." arXiv:2008.02217, 2020.
15. Wolfram, S. "A Class of Models with the Potential to Represent Fundamental Physics." *Complex Systems* 29(2), 2020.
16. Gorard, J. "Some Relativistic and Gravitational Properties of the Wolfram Model." *Complex Systems* 29(2), 2020.
17. Zenil, H., Kiani, N.A., Tegnér, J. *Algorithmic Information Dynamics.* Cambridge University Press, 2023.
