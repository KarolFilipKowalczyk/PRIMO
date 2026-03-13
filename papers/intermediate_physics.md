# Mandatory Intermediate Physics in the Space of Minimal Programs

**Author:** Karol Jalochowski
**Affiliation:** Independent researcher
**Date:** March 2026
**Status:** Sketch (v0) — [TODO] markers indicate missing proofs, data, or arguments

---

## Abstract

The conditional theorem Φ+ ⊂ I+ (Papers 1–2) establishes that every physics-like graph rewrite trajectory is also inference-like, but the gap between I+ and Φ+ remains uncharacterized. We introduce a continuous *physics parameter vector* P ∈ ℝ⁸ that replaces the binary Φ-predicate with a multidimensional fingerprint measuring spectral dimension, propagation, curvature homogeneity, lawfulness, structural complexity, clustering, nonlocal correlations, and dimensional consistency. We identify eight recognizable intermediate physics — specific parameter profiles corresponding to known physical theories (1+1D gravity, topological order, geometry without conservation laws, etc.) — and hypothesize a partial order among them forced by algebraic dependencies between parameters. The central prediction is that on the path from I+ to Φ+, strictly weaker physical theories must stabilize first, in a specific order: dimension before metric, metric before conservation laws, local structure before nonlocal correlations. We describe an experimental protocol using the existing 33-rule catalog to test whether Φ+ rules universally pass through these intermediate stages, whether I+Φ− rules are "stuck" at identifiable stages, and whether program length (signature complexity) correlates with the furthest stage reached. [TODO: experiment]

---

## 1. Introduction

### 1.1 Context

Papers 1–3 are companion papers to this work. Paper 1 [1] defines the I-predicate (inference-like behavior) and Φ-predicate (physics-like behavior) on graph rewrite trajectories and proves their logical independence. Paper 2 [2] proves that Bayesian graph dynamical systems satisfy the I-predicate (forward theorem). Paper 3 [3] establishes a computational hierarchy of DPO rewrite rules by signature complexity. The conditional theorem, proved across Papers 1–2 using Davis–Kahan perturbation theory, establishes Φ+ ⊂ I+: every physics-like trajectory is also inference-like. The PRIMO conjecture (Paper 4 [4]) claims that inference appears at shorter program lengths than physics.

This paper asks: *is the gap between I+ and Φ+ structured?* Does it contain mandatory intermediate stages — recognizable fragments of physical theory that must appear before full physics can emerge?

### 1.2 Motivation

The binary Φ-predicate treats physics as all-or-nothing: a trajectory either satisfies the spectral dimension, curvature, and lawfulness criteria simultaneously, or it does not. But real physics has layers. One-dimensional physics is simpler than two-dimensional physics. Geometry without conservation laws is simpler than geometry with them. If these layers correspond to nested subsets of program space — each requiring longer programs to reach — then the emergence of physics has mandatory intermediate stages, and the PRIMO conjecture can be sharpened from a single ordering (I before Φ) to a graded sequence.

### 1.3 Contributions

1. A continuous physics parameter vector P ∈ ℝ⁸ that refines the binary Φ-predicate (Section 2).
2. Eight recognizable intermediate physics, each identified by a specific parameter profile corresponding to a known physical theory (Section 3).
3. Four hypotheses about the ordering of these intermediates, grounded in algebraic dependencies between parameters (Section 4).
4. An experimental protocol to test these hypotheses on the existing 33-rule catalog (Section 5).

### 1.4 Scope

This paper does not claim that the universe passed through these intermediate stages historically. The ordering is in *program space* (description length), not in cosmological time. It does not claim that the intermediate physics "exist" as possible universes — they are dynamical behaviors of graph rewrite rules. Whether they correspond to consistent physical theories is an interpretive question outside scope.

---

## 2. The Physics Parameter Vector

We replace the binary Φ-predicate with a continuous parameter vector P(trajectory) ∈ ℝ⁸. Each component measures a property that has a well-defined meaning both on finite graphs and in known physics.

### 2.1 P₁: Spectral dimension (d_s)

**On graph.** Scaling exponent of the cumulative eigenvalue distribution N(λ) ∼ λ^{d_s/2} of the graph Laplacian.

**Physics target.** d_s = 4 (3+1 spacetime). Possibly d_s = 2 at Planck scale, consistent with dimensional reduction observed in multiple quantum gravity approaches [5, 6].

**What it measures.** Effective number of directions in space — the intrinsic dimensionality experienced by a diffusion process on the graph.

### 2.2 P₂: Normalized spectral gap (λ₁/d_s)

**On graph.** Smallest nonzero eigenvalue of the graph Laplacian, divided by d_s.

**Physics target.** Small but nonzero. Zero gap implies a massless field (photon-like propagation). Large gap implies no long-range propagation — the system is gapped, with only short-range correlations.

**What it measures.** Whether signals can propagate over large distances. The spectral gap controls the relaxation time of diffusion processes and is the graph-theoretic analog of the mass gap in quantum field theory.

### 2.3 P₃: Curvature homogeneity (CV_J)

**On graph.** Coefficient of variation of per-edge Jaccard similarity, where Jaccard similarity of edge (u,v) is |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.

**Physics target.** Low (< 0.5) at large scales — the cosmological principle states that the universe is homogeneous and isotropic on sufficiently large scales. High at small scales (atoms vs vacuum).

**What it measures.** Whether space looks the same everywhere. Low CV_J means curvature is approximately uniform; high CV_J means some regions are highly curved while others are flat.

### 2.4 P₄: Lawfulness (R_law)

**On graph.** Lowest normalized residual of {constant, linear, quadratic} polynomial fit to aggregate evolution observables: total edges, mean degree, degree entropy, spectral gap, edges per node.

**Physics target.** R_law → 0. Conservation laws in physics are exact. Approximate lawfulness (low residual) is the graph analog of deterministic evolution governed by differential equations.

**What it measures.** Whether evolution is predictable — whether "laws of nature" exist for this trajectory.

### 2.5 P₅: Normalized degree entropy (H_deg)

**On graph.** Shannon entropy of the degree distribution, normalized by log(n): H_deg = −Σ_k p_k log p_k / log(n), where p_k is the fraction of vertices with degree k.

**Physics target.** Moderate (0.4–0.7). H_deg = 0 implies a perfectly regular lattice (all vertices identical). H_deg = 1 implies a maximally disordered degree distribution. Real physics lies between — structured but not regular.

**What it measures.** Complexity and richness of local structure. A proxy for the variety of "particle types" or "local geometries" present.

### 2.6 P₆: Mean clustering coefficient (C)

**On graph.** Average over vertices of the triangle proportion: C(v) = 2T(v) / (deg(v)(deg(v)−1)), where T(v) is the number of triangles containing v.

**Physics target.** Nonzero. Regular lattices have low clustering. Erdős–Rényi random graphs have low clustering. Small-world networks (which share structural features with physical space) have high clustering.

**What it measures.** Local geometric coherence — whether neighbors of a vertex's neighbors are also neighbors. The graph analog of local flatness or triangulability.

### 2.7 P₇: Distance correlation ratio (ξ)

**On graph.** Ratio corr(r=3)/corr(r=1), where corr(r) is the Pearson correlation between degrees of vertex pairs at graph distance r, averaged over all such pairs.

**Physics target (classical).** Decays exponentially with r, so ξ is small. Classical correlations are local.

**Physics target (quantum).** Decays slower than exponential (EPR-type correlations), so ξ is larger. Nonlocal correlations are the signature of quantum mechanics.

**What it measures.** Whether distant parts of the system know about each other — a proxy for nonlocal correlations.

### 2.8 P₈: Edge-to-vertex ratio (e/v)

**On graph.** |E|/|V|.

**Physics target.** e/v ≈ d_s for lattice-like geometry. A d-dimensional cubic lattice has e/v = d. Consistency between connectivity density and spectral dimension is a signature of genuine geometry rather than a high-dimensional graph pretending to be low-dimensional.

**What it measures.** Dimensional consistency — whether the graph's connectivity matches its spectral dimension.

### 2.9 Target vector

The target vector for "our physics" (3+1D spacetime with quantum correlations):

| Parameter | Symbol | Target value | Interpretation |
|-----------|--------|-------------|----------------|
| Spectral dimension | d_s | 4 (or 2 at Planck scale) | 3+1 spacetime |
| Normalized spectral gap | λ₁/d_s | small, > 0 | Long-range propagation |
| Curvature homogeneity | CV_J | < 0.5 | Homogeneous space |
| Lawfulness | R_law | → 0 | Exact conservation laws |
| Degree entropy | H_deg | 0.4–0.7 | Structured but not regular |
| Clustering | C | > 0 | Local geometric coherence |
| Distance correlation ratio | ξ | > 0 | Nonlocal correlations (quantum) |
| Edge/vertex ratio | e/v | ≈ d_s | Dimensional consistency |

*Remark.* The target vector is not a single point but a region in ℝ⁸. The tolerances around each target value are themselves parameters of the analysis. We use the same threshold philosophy as Paper 1: set thresholds by null-model separation rather than by fiat. [TODO: calibrate tolerances using ER null models for each parameter]

---

## 3. Recognizable Intermediate Physics

We identify specific regions in ℝ⁸ that correspond to known, well-studied physical theories. Each is characterized by a parameter profile — which parameters are near their physical targets and which are not. These are *predictions*: we hypothesize that Φ+ rules pass through these regions on their way to the full target vector.

### 3.1 One-dimensional physics with conservation laws

**Profile.** d_s ≈ 1, λ₁/d_s > 0, CV_J low, R_law → 0, ξ ≈ 0, e/v ≈ 1.

**Theory.** 1+1D gravity (Jackiw–Teitelboim [7]), Liouville field theory.

**World.** A line. Particles move forward and backward. Conservation laws exist. No way to go around each other — all collisions are head-on. Gravity is a uniform force, not inverse-square. Exactly solvable. This is the simplest physics that has both geometry and dynamics.

### 3.2 Two-dimensional flat geometry

**Profile.** d_s ≈ 2, λ₁/d_s > 0, CV_J low, R_law → 0, ξ ≈ 0, e/v ≈ 2.

**Theory.** 2+1D gravity (Witten [8], Carlip [9]), Flatland physics.

**World.** A surface. No gravitational waves — the Weyl tensor vanishes identically in 2+1 dimensions. Particles can be anyons (neither bosons nor fermions — fractional statistics are possible only in 2D). Gravity is topological: a point mass creates a conical deficit, not a force field. Exactly solvable.

### 3.3 Two-dimensional physics with nonlocal correlations

**Profile.** d_s ≈ 2, λ₁/d_s > 0, CV_J low, R_law → 0, ξ > 0, e/v ≈ 2.

**Theory.** Fractional quantum Hall effect, topological order (Wen [10]).

**World.** A 2D surface where distant particles are correlated in ways that no local mechanism can explain. Information is stored globally, not locally — the ground state degeneracy depends on the topology of the surface, not on any local observable. The physical basis for topological quantum computing.

### 3.4 Geometry without conservation laws

**Profile.** d_s ≈ 2–3, CV_J low, R_law high, e/v ≈ d_s.

**Theory.** Far-from-equilibrium physics: turbulence, quark-gluon plasma, early-universe reheating.

**World.** Space exists, has dimension and metric, but evolution is unpredictable. Like the universe moments after the Big Bang before it cooled enough for conservation laws to emerge macroscopically. Or like fully developed turbulence: spatial structure exists but no simple dynamical laws govern the evolution. Energy cascades from scale to scale without a conserved quantity at any one scale.

### 3.5 Topology without metric

**Profile.** d_s stable and integer, λ₁ > 0, CV_J high, R_law variable.

**Theory.** Pre-geometric phase in loop quantum gravity [11], spin foam models, Regge calculus before the continuum limit.

**World.** You know you are on something with a definite number of dimensions, but there is no meaningful notion of distance. Curvature varies wildly from place to place. Like a foam — it has combinatorial structure (triangulation, cell complex) but no smooth geometry. Spectral dimension is well-defined because it depends on the Laplacian spectrum, which exists for any graph, but the Jaccard curvature varies enormously because adjacent regions have completely different local geometries.

### 3.6 Conservation laws without geometry

**Profile.** d_s unstable or undefined, R_law → 0, CV_J high.

**Theory.** Integrable systems without spatial interpretation — number-theoretic dynamical systems with conserved quantities, cellular automata with exact invariants but no emergent spatiality.

**World.** Evolution is perfectly predictable, but there is no space to speak of. Like a clock without a room — it keeps time but is nowhere. The trajectory has exact polynomial fits (low R_law) but spectral dimension fluctuates or does not converge. Analogous to the difference between a Hamiltonian system (which has conservation laws by construction) and a spatial system (which also has geometry).

### 3.7 Homogeneous space with broken symmetry

**Profile.** d_s ≈ 2–4, CV_J low globally but bimodal locally, R_law → 0.

**Theory.** Phase transitions, spontaneous symmetry breaking, Higgs mechanism, Landau theory [12].

**World.** Space is smooth and homogeneous at large scales, but at smaller scales has distinct "phases" — like ice and water coexisting, or a ferromagnet that has chosen a magnetization direction. The Jaccard curvature distribution is bimodal: most edges are in one of two clusters, reflecting two locally distinct geometries. The universe after electroweak symmetry breaking has exactly this character: homogeneous on cosmological scales but with distinct broken-symmetry phases at particle-physics scales.

### 3.8 Full physics (target)

**Profile.** d_s ≈ 4, λ₁/d_s small > 0, CV_J low, R_law → 0, H_deg moderate, C > 0, ξ > 0, e/v ≈ 4.

**Theory.** 3+1D physics with quantum correlations — the Standard Model coupled to general relativity.

**World.** Our world.

*Remark.* Not every combination of parameter values corresponds to a recognized physical theory. The intermediate physics listed here are selected because they are (a) well-studied, (b) definable in terms of the parameter vector, and (c) plausible as intermediate stages based on their parameter profiles being strict subsets of the full target. Other intermediate stages may exist — see Section 6.4.

---

## 4. Hypothesized Ordering

The central hypothesis of this paper is that the intermediate physics of Section 3 form a partial order, and that this order is not arbitrary but is forced by algebraic dependencies between the parameters.

### 4.1 Hypothesis 1: Dimension before metric

[HYPOTHESIS]

**Statement.** Stable spectral dimension (d_s converges) must precede curvature homogeneity (CV_J drops below threshold).

**Justification sketch.** Cheeger's inequality [13] relates the spectral gap λ₁ to the isoperimetric constant h: λ₁/2 ≤ h ≤ √(2λ₁). The isoperimetric constant depends on the notion of "volume," which requires stable spectral dimension to be meaningful. More directly: curvature homogeneity asks "is curvature approximately uniform?" — but curvature is defined with respect to a dimension. A graph with fluctuating d_s has no stable notion of what "uniform curvature" means. Homogeneity of curvature presupposes that there is a stable notion of curvature, which presupposes stable dimension.

[TODO: proof — formalize the dependence of Jaccard curvature statistics on spectral dimension stability]

### 4.2 Hypothesis 2: Metric before conservation laws

[HYPOTHESIS]

**Statement.** Curvature homogeneity must precede or co-emerge with lawful evolution (low R_law).

**Justification sketch.** Noether's theorem ties conservation laws to continuous symmetries of the action. In a graph rewrite context, the analog is that homogeneous curvature (approximate isometry invariance) provides the symmetries that enable conserved quantities. A system with wildly varying curvature has no approximate symmetries, hence no mechanism for conservation laws to emerge. Concretely: if CV_J is high, aggregate observables depend on *which* region of the graph is sampled, making polynomial fits to global observables poor (high R_law).

[TODO: proof — prove that high CV_J implies high R_law under reasonable assumptions on the observable averaging procedure]

*Remark.* This is the most speculative hypothesis. Conservation laws without geometry (Section 3.6) shows that the implication is not logically necessary — one can have low R_law with unstable d_s. The hypothesis claims that this combination is not reachable *on the path from I+ to Φ+*, not that it is logically impossible.

### 4.3 Hypothesis 3: Local structure before nonlocal correlations

[HYPOTHESIS]

**Statement.** The clustering coefficient C stabilizes at a nonzero value before the distance correlation ratio ξ becomes nonzero.

**Justification sketch.** Nonlocal correlations require a notion of "distance" to be meaningful — corr(r=3)/corr(r=1) requires that graph distance is a meaningful proxy for physical distance. Distance is meaningful only if local neighborhoods have coherent structure (nonzero clustering). On a graph with C ≈ 0, graph distance is unreliable as a geometric proxy because the graph lacks local triangulability. Thus ξ > 0 is uninterpretable without C > 0.

[TODO: proof — show that ξ > 0 with C ≈ 0 does not correspond to genuine nonlocality but to degree-degree correlations in tree-like graphs]

### 4.4 Hypothesis 4: All intermediates before full physics

[HYPOTHESIS]

**Statement.** The full target vector (Section 2.9) is reached only after all intermediate stages have been passed through. No trajectory jumps directly from I+ to Φ_full without visiting intermediate parameter profiles.

**Justification sketch.** This follows from Hypotheses 1–3 if the parameter dependencies are strict. If dimension must precede metric, and metric must precede conservation laws, then any trajectory reaching the full target must pass through states where dimension is stable but metric is not, and states where metric is stable but conservation laws are not. These states correspond to the intermediate physics of Section 3.

[TODO: proof — formalize as a topological argument about connected paths in the admissible region of ℝ⁸]

### 4.5 Conjectured partial order

Combining the hypotheses, we conjecture the following Hasse diagram:

```
I+ (inference)
 │
 ↓
Φ_topo (stable dimension, spectral gap)          [Section 3.5]
 │
 ↓
Φ_geom (+ curvature homogeneity)                 [Sections 3.1, 3.2]
 │         ╲
 ↓          ↓
Φ_laws      Φ_broken                              [Sections 3.4→inverse, 3.7]
(+ conservation laws)  (+ symmetry breaking)
 │         ╱
 ↓
Φ_nonlocal (+ distance correlations)              [Section 3.3]
 │
 ↓
Φ_full (our physics)                              [Section 3.8]
```

*Remark.* Sections 3.4 (geometry without conservation laws) and 3.6 (conservation laws without geometry) sit on *opposite sides* of the partial order: 3.4 is on the path from Φ_geom to Φ_laws, while 3.6 is off the main path entirely — reachable from I+ but not leading to Φ_full. If this is correct, the (I+, Φ−) rules in Paper 1's catalog should include examples of Section 3.6 (rules that achieve lawfulness without achieving geometry).

---

## 5. Experimental Design

### 5.1 What we measure

For each rule r in the 33-rule catalog, for each of the 4 canonical initial graphs G₀, at each timestep t = 0, 1, …, T:

- Apply the rule to obtain G_t.
- Compute all 8 parameters: P(t) = (d_s(t), λ₁(t)/d_s(t), CV_J(t), R_law(t), H_deg(t), C(t), ξ(t), e/v(t)).

This gives a trajectory P(t) through ℝ⁸ — the *physics development path* of that rule under that initial condition.

*Remark.* R_law(t) is computed using a sliding window of the most recent w steps (w = 10 by default), since polynomial fits to cumulative observables require a time series. At early times (t < w), R_law is undefined.

### 5.2 Stabilization times

**Definition (Stabilization).** Parameter P_i is *stabilized* at time t_i if |P_i(t) − P_i(T)| < ε_i for all t ≥ t_i, where ε_i is a tolerance calibrated per parameter. [TODO: calibrate ε_i using ER null-model variance for each parameter]

For each Φ+ rule, record the vector of stabilization times (t₁, t₂, …, t₈). The ordering of these times is the *stabilization order* of that rule.

### 5.3 Tests

**Test 1 (Universal ordering).** For each Φ+ rule, compute the stabilization order. Is it universal (identical across all Φ+ rules), partially universal (consistent partial order but varying total orders), or rule-dependent (no consistent ordering)?

Specifically, test:
- t(d_s) < t(CV_J) for all Φ+ rules? (Hypothesis 1)
- t(CV_J) < t(R_law) for all Φ+ rules? (Hypothesis 2)
- t(C) < t(ξ) for all Φ+ rules? (Hypothesis 3)

**Test 2 (Mandatory intermediates).** For each Φ+ rule, at the time t when d_s has stabilized but CV_J has not yet dropped: does the parameter vector match Section 3.5 (topology without metric)? At the time when d_s and CV_J are stable but R_law is still high: does it match Section 3.4 (geometry without conservation laws)?

Operationally: define a "match score" between the observed P(t) and each intermediate profile as the number of parameters within tolerance of the profile's target values, weighted by the profile's specificity (number of constrained parameters).

**Test 3 (I+Φ− rules).** The 5 rules classified as (I+, Φ−) in Paper 1 — where do they sit in ℝ⁸ at their final timestep T? Do their terminal parameter vectors correspond to any of the intermediate physics in Section 3? Are they "stuck" at an identifiable intermediate stage?

**Test 4 (Correlation with program length).** Using signature complexity from Paper 3: for rules at each hierarchy level, what is the furthest stage reached in the partial order of Section 4.5? Do shorter programs (lower signature) get stuck at earlier stages? Do longer programs reach further?

[TODO: experiment — implement and run all four tests on the 33-rule catalog]

### 5.4 Data

The experiment uses the existing 33-rule catalog from Paper 1 (reference oracle: `primo_diagnostic_v5.py`) with T = 30 steps and 4 canonical initial graphs. This is the same dataset used in Papers 1–4, ensuring consistency.

Most of the 8 parameters are already computed in the existing diagnostic infrastructure:
- d_s, λ₁, CV_J, R_law: computed in `primo/predicates.py`.
- H_deg, C, ξ, e/v: not yet computed. [TODO: implement these four parameters in `primo/predicates.py`]

If results are promising, extend to T = 100 and to newly enumerated rules at higher signatures (3→4, 4→5 from Phase 3).

### 5.5 Implementation notes

Output format: for each (rule, seed, timestep) triple, an 8-vector P(t). Stored as a tensor of shape (n_rules × n_seeds × T × 8).

Analysis pipeline:
1. Compute stabilization times per parameter per rule.
2. Compute pairwise ordering statistics across Φ+ rules.
3. Compute match scores against intermediate profiles.
4. Compute terminal positions of I+Φ− rules.
5. Correlate furthest-stage-reached with signature complexity.

[TODO: write experiment script `experiments/exp_intermediate_physics.py`]

---

## 6. Possible Outcomes

### 6.1 Strong positive

All Φ+ rules pass through the same sequence of intermediate physics, in the same order, and the order matches the Hasse diagram from Section 4.5. The (I+, Φ−) rules correspond to identified intermediate stages. Shorter programs get stuck at earlier stages.

**Interpretation.** The path to physics is unique (up to partial ordering), and the intermediate physical theories are mandatory stations — not optional detours. The emergence of physics from inference is a structured, staged process. The PRIMO conjecture sharpens from "I before Φ" to "I before Φ_topo before Φ_geom before Φ_laws before Φ_full."

### 6.2 Weak positive

The ordering is partially universal — all Φ+ rules stabilize dimension before curvature (Hypothesis 1 confirmed), but the ordering of conservation laws vs symmetry breaking (Hypotheses 2–3) varies between rules. Some (I+, Φ−) rules correspond to identified intermediates, others do not.

**Interpretation.** Some intermediate physics are mandatory (the early ones — topology and geometry), while others are contingent (conservation laws, nonlocal correlations). This still identifies which aspects of physics are algebraically forced and which are "choices" that depend on the specific rewrite rule.

### 6.3 Negative

No universal ordering. Different Φ+ rules take completely different paths through ℝ⁸. The intermediate physics from Section 3 are not visited as recognizable stages.

**Interpretation.** The gap between I+ and Φ+ is unstructured. Physics assembles from its components in no particular order. The binary Φ-predicate cannot be meaningfully decomposed into intermediate stages. The continuous parameter vector is useful for description but not for prediction.

### 6.4 Surprise

The trajectories cluster in ℝ⁸ around parameter profiles *not* listed in Section 3 — combinations of parameter values that do not correspond to any recognized physical theory.

**Interpretation.** This would be the most interesting outcome: discovery of "alien physics" that is mathematically consistent (well-defined dynamical behavior on graphs) but not studied by physicists because it does not describe our universe. These would be genuinely novel physical theories, identifiable by their parameter profiles, that emerge naturally in program space. Whether they correspond to internally consistent physical theories with their own "inhabitants" is a question for mathematical physics.

---

## 7. What This Paper Does Not Claim

**Not a cosmological history.** The ordering in Section 4 is in *program space* (description length / signature complexity), not in *cosmological time*. We do not claim the universe passed through 1+1D gravity on its way to 3+1D physics. The claim is about the structure of the space of minimal programs, not about the history of our universe.

**Not an existence claim.** The intermediate physics of Section 3 are dynamical behaviors of graph rewrite rules, not physical theories. Whether they correspond to internally consistent quantum field theories or general-relativistic spacetimes is an interpretive question that this paper does not address.

**Not a resolution of the Vanchurin tension.** Vanchurin [14, 15] argues that quantum mechanics emerges near equilibrium, before classical mechanics (which requires far-from-equilibrium conditions). Our ordering is by program complexity, not by distance from equilibrium. A rule with low signature complexity that exhibits quantum-like correlations (ξ > 0) at equilibrium is not contradicted by our ordering — we predict only that achieving ξ > 0 requires a program complex enough to first achieve stable d_s, low CV_J, and nonzero C. The two orderings (Vanchurin's thermodynamic ordering and our complexity ordering) may be compatible, complementary, or in tension. Resolving this requires experiments on both axes simultaneously. [TODO: design experiment comparing thermodynamic distance from equilibrium with signature complexity]

**Not complete.** The eight parameters of Section 2 do not exhaust the properties of physical theories. Gauge symmetry, chirality, supersymmetry, and many other structures are not captured. The parameter vector is a first-order approximation to the "physics fingerprint" — sufficient for the coarse-grained ordering question, but not for fine-grained identification of specific physical theories.

---

## 8. Discussion

### 8.1 Relation to "More is Different"

Anderson's "More is Different" [16] argues that each level of physical organization involves genuinely new principles not derivable from the level below. Our Hasse diagram (Section 4.5) can be read as a formalization of this idea in the graph rewrite setting: each stage in the partial order introduces a new parameter crossing its physical threshold, corresponding to a new physical principle (dimensionality, homogeneity, lawfulness, nonlocality) that is not reducible to the preceding ones.

The key difference is that Anderson's hierarchy is empirical (observed in nature), while ours is computational (predicted from program space structure). If the experimental tests of Section 5 confirm the ordering, it would suggest that "more is different" is not merely an empirical observation but a structural necessity — forced by the algebraic dependencies between the mathematical quantities that define physical theories.

### 8.2 Relation to causal set theory

Sorkin's causal set program [17] posits that spacetime is fundamentally a partial order (causal set) from which geometry emerges. Our approach is complementary: we posit that the *space of programs that generate spacetime-like structures* has a partial order (the Hasse diagram), from which the sequence of emergence is read off. The graph rewrite rules are not causal sets, but they generate graphs that may or may not have causal-set-like properties (stable dimension, good propagation, etc.).

### 8.3 Falsifiability

Each hypothesis in Section 4 makes a specific, testable prediction about the ordering of stabilization times. If t(CV_J) < t(d_s) for even a single Φ+ rule, Hypothesis 1 is falsified. The experiment of Section 5 is designed so that every hypothesis can be independently confirmed or refuted.

### 8.4 Open problems

1. **Sharpening the intermediate profiles.** The profiles in Section 3 are qualitative (d_s ≈ 2, CV_J low). Making them quantitative requires calibrating thresholds for each parameter, analogous to the threshold calibration in Paper 1.

2. **Completeness of the parameter vector.** Are 8 parameters sufficient to distinguish all physically meaningful intermediate stages? Are there intermediate physics that are indistinguishable in ℝ⁸ but physically distinct?

3. **Algebraic proof of ordering.** Hypotheses 1–3 are justified by sketches. Full proofs require formalizing the dependence of each parameter on the preceding ones in the graph Laplacian framework.

4. **Extension to higher signatures.** The 33-rule catalog has limited signature range. Testing at 3→4 and 4→5 signatures (Phase 3 of the PRIMO program) would provide more data points for the correlation between program length and furthest stage reached.

5. **Reconciliation with Vanchurin.** Is the complexity ordering (our Hasse diagram) consistent with the thermodynamic ordering (Vanchurin's near-equilibrium → far-from-equilibrium)?

---

## References

[1] K. Jalochowski, "I-Predicate and Φ-Predicate on Graph Rewrite Trajectories: Definitions, Independence, and Computational Validation." Paper 1 (companion).

[2] K. Jalochowski, "Bayesian Graph Dynamical Systems Satisfy the I-Predicate." Paper 2 (companion).

[3] K. Jalochowski, "A Strict Computational Hierarchy for Graph Rewrite Systems by Signature Complexity." Paper 3 (companion).

[4] K. Jalochowski, "The PRIMO Conjecture: Inference Before Physics in the Space of Minimal Programs." Paper 4 (companion).

[5] O. Lauscher and M. Reuter, "Fractal spacetime structure in asymptotically safe gravity," JHEP 0510 (2005) 050.

[6] P. Hořava, "Spectral dimension of the universe in quantum gravity at a Lifshitz point," Phys. Rev. Lett. 102 (2009) 161301.

[7] R. Jackiw, "Lower dimensional gravity," Nuclear Physics B 252 (1985) 343–356.

[8] E. Witten, "2+1 dimensional gravity as an exactly soluble system," Nuclear Physics B 311 (1988) 46–78.

[9] S. Carlip, "Quantum Gravity in 2+1 Dimensions," Cambridge University Press (1998).

[10] X.-G. Wen, "Topological orders in rigid states," Int. J. Mod. Phys. B 4 (1990) 239–271.

[11] C. A. Trugenberger, "Combinatorial quantum gravity: geometry from random bits," JHEP 2017 (2017) 45.

[12] L. D. Landau, "On the theory of phase transitions," Zh. Eksp. Teor. Fiz. 7 (1937) 19–32.

[13] J. Cheeger, "A lower bound for the smallest eigenvalue of the Laplacian," Problems in Analysis (1970) 195–199.

[14] V. Vanchurin, "The world as a neural network," Entropy 22 (2020) 1210.

[15] V. Vanchurin, "Towards a theory of quantum gravity from neural networks," Entropy 24 (2022) 7.

[16] P. W. Anderson, "More is different," Science 177 (1972) 393–396.

[17] R. D. Sorkin, "Causal sets: discrete gravity," in Lectures on Quantum Gravity, Springer (2005) 305–327.

---

## Appendix A: Parameter Computation Details

[TODO: specify exact algorithms for each of the 8 parameters, including edge cases (disconnected graphs, graphs too small for spectral dimension fitting, etc.)]

## Appendix B: Intermediate Physics Identification Algorithm

[TODO: specify the match score function for identifying which intermediate physics a parameter vector P(t) corresponds to, including handling of partial matches and ambiguous cases]

## Appendix C: Stabilization Detection

[TODO: specify the algorithm for detecting stabilization times, including robustness to fluctuations and the choice of window size for R_law computation]
