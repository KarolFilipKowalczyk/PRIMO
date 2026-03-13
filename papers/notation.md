# PRIMO Notation Glossary

Shared notation across all four papers. During development this file is the reference.
At Phase 4 (LaTeX conversion) this becomes `primo-macros.sty`.

---

## Predicates

| Symbol | Meaning | Defined in |
|--------|---------|------------|
| I-positive | Satisfies I-predicate: compression gate + embedding convergence (τ > τ*) under ≥1 embedding, for ≥3/4 seeds | Paper 1 Def 3 |
| I-negative | Does not satisfy I-predicate | Paper 1 |
| Φ-positive | Satisfies Φ-predicate: stable integer spectral dimension + (lawful evolution or curvature homogeneity), for ≥3/4 seeds | Paper 1 Def 8 (in primo.md) |
| Φ-negative | Does not satisfy Φ-predicate | Paper 1 |
| I-score | Continuous version: τ_to_final under a given embedding. Used for temporal profiling in PRIMO claim (b) | Paper 1, PRIMO §3 |

## Thresholds

| Symbol | Value | Meaning |
|--------|-------|---------|
| τ* | 0.5 | Convergence threshold (Kendall τ of cosines-to-final) |
| ρ* | 0.85 | Compression gate threshold |
| σ* | 0.45 | Straightness gate threshold (provisional) |
| ds_std* | 0.18 | Spectral dimension stability (std across trajectory) |
| ds_int | 0.5 | Max distance of ds_mean from nearest integer |
| law* | 0.15 | Lawful evolution residual threshold |
| κ* | 1.0 | Curvature homogeneity (CV of Jaccard curvature) |

## Classes (Paper 3 — hierarchy)

| Symbol | Meaning |
|--------|---------|
| C_l | Function class at level l: partial graph functions computed by rules with LHS size l |
| D_l | Dynamics class at level l: trajectory properties at LHS size l |
| Ĉ_l | Extended function class (rule-set extension with priority) |
| D̂_l | Extended dynamics class |
| σ(f) | Signature complexity: LHS size l of the rule computing f |
| k* | Universality threshold: min l such that C_l contains all TM-computable functions. k* ≤ 7 for small TMs, k* ≤ 8 for arbitrary TMs |

## Operations

| Symbol | Meaning |
|--------|---------|
| GPI | Greedy parallel-independent application: find maximal matching of LHS, apply all matches simultaneously |
| DPO | Double pushout: the categorical framework for graph rewriting |
| seq | Sequential application: apply rule to the first (by canonical ordering) match only |

## Embeddings

| Symbol | Meaning |
|--------|---------|
| e_L | Laplacian eigenvector embedding: d leading eigenvectors of graph Laplacian L = D − A |
| e_R | Random projection embedding: X = A · M where M is a fixed random Gaussian matrix |
| e_D | Degree-profile embedding: per-node features (normalized degree, clustering coefficient, avg neighbor degree, 2-hop size, hub distance) |

## Graph notation

| Symbol | Meaning |
|--------|---------|
| K_n | Complete graph on n vertices |
| P_n | Path graph on n vertices |
| G_0 | Initial graph in a trajectory |
| (G_0, G_1, …, G_T) | Trajectory of length T |

## PRIMO-specific

| Symbol | Meaning |
|--------|---------|
| N_I^min | Minimum signature level at which an I-positive rule exists |
| N_Φ^min | Minimum signature level at which a Φ-positive rule exists |
| I(N) | Fraction of I-positive programs among all programs of length ≤ N |
| Φ(N) | Fraction of Φ-positive programs among all programs of length ≤ N |

## Grassmannian

| Symbol | Meaning |
|--------|---------|
| Gr(k, d) | Grassmannian: space of k-dimensional subspaces of R^d |
| d_Gr | Geodesic distance on the Grassmannian |
| cos θ(U, V) | Mean cosine of principal angles between subspaces U, V |

## Information theory

| Symbol | Meaning |
|--------|---------|
| D_KL | Kullback-Leibler divergence |
| ρ(traj) | Compression ratio: len(zlib(serialize(traj))) / len(serialize(traj)) |
| d_s | Spectral dimension (from diffusion return probability / eigenvalue counting) |
