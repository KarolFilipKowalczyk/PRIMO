# Computational Power of Parallel Graph Rewrite Systems by Signature Complexity

**Author:** Karol (AIRON Games / MIMUW)  
**Date:** March 2026  
**Target venue:** Theoretical Computer Science / Fundamenta Informaticae

---

## Abstract

We study a parameterized family of graph rewrite systems operating on finite simple ordered graphs via DPO rules in the category $\mathbf{Graph}_{\mathrm{inj}}$. The parameter — the *LHS size* $l$ — determines computational power. We formalize two notions of hierarchy: a *dynamics class* $\mathcal{D}_l$ (trajectory properties: growth, reachable graphs, structural invariants) and a *function class* $\mathcal{C}_l$ (partial graph functions defined by halting). We establish a strict dynamics-class hierarchy: level-1 trajectories have constant graph size (finite-state); level-2 trajectories exhibit unbounded uniform growth but cannot produce degree-2 fresh vertices; level-3 trajectories can condition on edge existence and perform topological operations (subdivision, triangle completion) impossible at level 2. Two strict separations are proved: constant size vs. unbounded growth (level 1 vs. 2), and the No-Betweenness Invariant showing that vertex sprouting cannot produce subdivided edges (level 2 vs. 3). We prove a universality threshold: $k^* \leq 7$ for TMs with $q|\Gamma| \leq 8$ (using a $K_3$ marker); $k^* \leq 8$ for arbitrary TMs (using a $K_4$ marker and the Neary–Woods $(15,2)$ UTM [5]). The threshold is identical for sequential and greedy parallel-independent (GPI) application. A complexity bridge theorem shows that every function at level $l$ is computable at level $l+1$ with explicit description overhead $C_l = O(l)$, for function equivalence (not trajectory identity). We identify a *conditional branching threshold*: at $l \leq 2$, all rules share a unique LHS ($K_1$ or $K_2$), making rule-set priority vacuous and conditional computation impossible; at $l \geq 3$, multiple connected LHS patterns exist, enabling conditional branching for the first time. We discuss connections to pebble games and bounded-variable logics, identifying where the static correspondence holds and where it breaks down.

---

## 1. Introduction

### 1.1 The question

How does the computational power of a graph rewriting system depend on the size of its rewrite rules? We formalize this question for DPO rules on finite simple ordered graphs, with two application models (sequential and greedy parallel-independent), and give precise answers.

### 1.2 Conceptual framing: dynamical systems vs. computing devices

Before stating results, we resolve a foundational question. A single rule $\rho$ applied iteratively from a fixed initial graph $G_0$ produces a single deterministic trajectory. This is a dynamical system. To obtain a hierarchy of computational classes, we formalize two notions.

**Function class.** A system $\mathcal{S} = (\rho, \alpha)$ computes a partial function $f_{\mathcal{S}}: \mathcal{G} \rightharpoonup \mathcal{G}$:

$$f_{\mathcal{S}}(G_0) = \begin{cases} G_T & \text{if the trajectory reaches a fixed point } G_T = G_{T+1} \\ \bot & \text{otherwise} \end{cases}$$

The *function class at level $l$* is $\mathcal{C}_l^{\alpha} = \{f_{\mathcal{S}} : \mathcal{S} \text{ has LHS size } l\}$.

**Dynamics class.** The system $\mathcal{S}$ generates a trajectory map $\tau_{\mathcal{S}}: \mathcal{G} \to \mathcal{G}^{\omega}$ sending each initial graph to its (possibly infinite) trajectory. The *dynamics class at level $l$* is $\mathcal{D}_l^{\alpha} = \{\tau_{\mathcal{S}} : \mathcal{S} \text{ has LHS size } l\}$. Trajectory properties — growth rates, periodicity, reachable graph families — are $\mathcal{D}_l$ properties.

**Why both are needed.** At level 1, $\mathcal{C}_1 = \{\text{id}\}$ (every graph is an immediate fixed point). At level 2, $\mathcal{C}_2 = \{\bot\}$ (vertex sprouting never halts). The hierarchy between $\{\text{id}\}$ and $\{\bot\}$ is not informative. But $\mathcal{D}_1$ trajectories have constant graph size, while $\mathcal{D}_2$ trajectories exhibit unbounded growth — a genuine separation. Conversely, at the universality threshold, $\mathcal{C}_{k^*}$ contains all Turing-computable graph functions — a function-class result.

The hierarchy theorem therefore has two components: a $\mathcal{D}$-class strict hierarchy at all levels (by invariant arguments on trajectories), and a $\mathcal{C}$-class hierarchy at and above the universality threshold.

**Rule-set extension.** We allow a *finite ordered tuple* of rules $(\rho_1, \ldots, \rho_k)$ at a given signature level, with *per-step priority*: at each step, apply the first rule $\rho_i$ (in the fixed ordering) that has at least one injective match in the current graph. That rule is applied under the chosen model (sequential or GPI). Lower-priority rules are not considered at that step. If no rule has a match, the system halts. Write $\hat{\mathcal{C}}_l^{\alpha}$ and $\hat{\mathcal{D}}_l^{\alpha}$ for the extended classes.

**Critical observation.** At signature $l_2 \to (l+1)_2$, if there is only one connected graph on $l$ vertices (which is the case for $l = 1$: $K_1$, and $l = 2$: $K_2$), then all rules at this level share the same LHS. Per-step priority between rules with identical LHS is vacuous: the highest-priority rule always fires whenever the graph contains a copy of $L$. Therefore $\hat{\mathcal{C}}_l = \mathcal{C}_l$ and $\hat{\mathcal{D}}_l = \mathcal{D}_l$ at levels 1 and 2. At level 3 ($l = 3$), there are two connected graphs on 3 vertices ($K_3$ and $P_3$), so rules can have different LHS patterns, and the rule-set extension genuinely adds power. See Section 3.3 and Appendix C.

### 1.3 Results

**Theorem 1 (Hierarchy).** The dynamics class at each LHS size $l$ (with $r = l + 1$) forms a strict hierarchy. Adjacent levels are strictly separated.

**Theorem 2 (Strict separations).** Unbounded growth separates $\mathcal{D}_1$ from $\mathcal{D}_2$. Edge subdivision separates $\mathcal{D}_2$ from $\mathcal{D}_3$ (proved via a graph-theoretic invariant).

**Theorem 3 (Universality).** $k^* \leq 7$ for TMs with $q|\Gamma| \leq 8$; $k^* \leq 8$ for arbitrary TMs. The threshold is model-independent (sequential = GPI) because the construction uses unique-match rules.

**Theorem 4 (Complexity bridge).** Every function at level $l$ is computable at level $l+1$ with description overhead $C_l = O(l)$ bits. This is function equivalence, not trajectory identity.

### 1.4 Related work

**Graph transformation.** Our model is a restriction of the DPO approach (Ehrig, Ehrig, Prange, Taentzer 2006) to simple undirected graphs with fixed signature. We use the standard formalism and notation from *Fundamentals of Algebraic Graph Transformation* [1] throughout.

**L-systems.** Our level-1 (context-free vertex replacement) is the graph-theoretic analogue of D0L-systems. Level 2 (edge-conditioned) introduces context-sensitivity analogous to IL-systems. The rule-set extension corresponds to DT0L.

**MSO-definable transductions (Courcelle).** Matching a fixed $k$-vertex pattern is first-order definable. Selecting a maximal independent match set is MSO-definable. The iterated application of an MSO-definable transduction gives a natural upper bound on our formalism's power. For bounded-treewidth graphs, Courcelle's theorem implies decidability of MSO-definable properties, which constrains what our rules can decide on such graphs.

**Pebble games and finite model theory.** The LHS vertex count plays a role analogous to the pebble count in Immerman's $k$-pebble game. We make the correspondence precise in Section 6.4 (static case) and identify where it breaks down (dynamic case).

**Small UTMs.** The universality construction uses the Neary–Woods $(15,2)$ UTM [5], a standard-model universal Turing machine with 15 states and 2 symbols.

### 1.5 Organization

Section 2: Formal model. Section 3: Lower hierarchy levels. Section 4: Separations. Section 5: Universality. Section 6: Complexity bridge and logic. Section 7: Illustration. Section 8: Open problems.

---

## 2. Definitions

### 2.1 Ordered graphs

A *graph* is a pair $G = (V, E)$ where $V \subset \mathbb{N}$ is a finite set of vertices carrying the canonical total order inherited from $\mathbb{N}$, and $E \subseteq \binom{V}{2}$ is a set of undirected edges (no loops, no parallel edges). Write $\mathcal{G}$ for the class of all such graphs.

**On labels and ordering.** Our graphs carry an order on vertices. This is necessary for the deterministic tie-breaking in both application models (sequential and GPI). The order is not a "label" in the graph-grammar sense (it carries no semantic content beyond determining the evaluation schedule), but it does affect trajectories: different orderings on the same abstract graph may produce different GPI trajectories. We note this explicitly:

**Convention.** All results about the *computational class* at a given level (i.e., the set of computable graph functions) are stated for the canonical ordering. We do not claim that the results are independent of ordering choice; this independence is an open question for GPI (and is trivially true for sequential application when all matches of $L$ in $G$ are automorphic, which holds at $1_2 \to 2_2$).

### 2.2 Signatures and the restriction $r = l+1$

A *signature* is a pair $(l, r)$ with $1 \leq l \leq r$. The hierarchy is indexed by $l$ alone, with $r = l + 1$ fixed. This restriction is justified:

**Proposition 2 (Justification of $r = l+1$).**

(i) *$r = l$ (no fresh vertices).* A rule with $r = l$ adds no new vertices. Since it also cannot delete vertices (see Section 2.4), $|V(G_t)| = |V(G_0)|$ for all $t$. The state space is $2^{\binom{n}{2}}$ (the set of all graphs on $n$ vertices), which is finite. The system is a finite automaton regardless of $l$. Thus all levels with $r = l$ collapse to finite-state, making the parameter $l$ irrelevant.

(ii) *$r = l + j$ for $j \geq 2$ (multiple fresh vertices per match).* The computational power is dominated by the LHS size $l$ (which controls the bandwidth of information the rule *reads*), not the number of fresh vertices $j = r - l$ (which controls the bandwidth of information the rule *writes*). Formally: any rule at signature $l_2 \to (l+j)_2$ can be decomposed into $j$ sequential applications of rules at signature $l_2 \to (l+1)_2$ (each adding one fresh vertex), provided the intermediate rules are chosen appropriately. This decomposition is not exact (the intermediate graphs differ), but it shows that the computational class at $l_2 \to (l+j)_2$ is contained in the class achievable by $j$-fold iteration at $l_2 \to (l+1)_2$. We state a partial result:

**Claim.** For the *upper bounds* on computational power: the class at $l_2 \to (l+j)_2$ is contained in the class at $l_2 \to (l+1)_2$ under multi-step iteration. For the *lower bounds*: the universality construction at level $k^*$ uses $r = k^* + 1$, and padding to higher $r$ does not increase the computational class (already Turing-complete). We therefore lose nothing by restricting to $r = l + 1$.

**What is genuinely lost.** At a fixed level $l$, rules with $r = l + 2$ can create *two* fresh vertices per match, which may allow more complex *single-step* operations (e.g., inserting a path of length 2 in one step rather than two). This affects the efficiency (number of steps) but not the computational class (set of computable functions). A formal proof of this for all $l$ is deferred to future work; for the specific levels we analyze ($l = 1, 2$), the claim is verified directly.

### 2.3 Rewrite rules (DPO formalism)

We use the standard Double Pushout (DPO) approach in the category $\mathbf{Graph}_{\mathrm{inj}}$ of graphs with injective morphisms [1].

A *rewrite rule* is a span $\rho: L \xleftarrow{l_K} K \xrightarrow{r_K} R$ where $L$, $K$, $R$ are graphs and $l_K$, $r_K$ are injective graph morphisms. The graph $K$ is the *interface*: it specifies which parts of $L$ are preserved in $R$.

**Our restriction.** In this paper, $l_K$ is always a bijection on vertices: $V(K) = V(L)$ (equivalently, $l_K$ is the inclusion — all LHS vertices are in the interface). This means:

- **No vertex deletion.** Every vertex of $L$ is preserved in $R$. Vertices may only be added ($|V(R)| \geq |V(L)|$).
- **Edge deletion is possible.** An edge $(u,v) \in E(L)$ is deleted if $(r_K(l_K^{-1}(u)), r_K(l_K^{-1}(v))) \notin E(R)$.
- **Edge creation is possible.** An edge in $R$ between preserved vertices (not corresponding to an $L$-edge) or involving a fresh vertex is created.

We may therefore represent a rule equivalently as a triple $\rho = (L, R, \iota)$ where $\iota = r_K \circ l_K^{-1}: V(L) \hookrightarrow V(R)$ is the injective interface map. This is the notation used in the rest of the paper.

### 2.4 Rule application

Given rule $\rho = (L \leftarrow K \rightarrow R)$, host graph $G$, and injective match $m: L \hookrightarrow G$, the DPO application constructs:

$$L \xleftarrow{l_K} K \xrightarrow{r_K} R$$
$$\downarrow^m \qquad \downarrow^{m_K} \qquad \downarrow^{m_R}$$
$$G \xleftarrow{} D \xrightarrow{} H$$

where $D$ is the pushout complement and $H$ is the pushout (result graph).

**Proposition 3 (Application conditions are vacuous).** *Under our restriction ($l_K$ bijective on vertices, $m$ injective):*

(i) *The dangling condition is vacuously satisfied.* The dangling condition requires that no edge in $G \setminus m(E(L))$ is incident to a vertex in $m(V(L)) \setminus m(V(K))$. Since $V(K) = V(L)$, we have $m(V(L)) \setminus m(V(K)) = \emptyset$. The condition holds vacuously.

(ii) *The identification condition is vacuously satisfied.* The identification condition requires that $m$ does not merge vertices that the rule treats differently. Since $m$ is injective, no merging occurs.

(iii) *Consequently, the pushout complement $D$ always exists.* DPO application is always defined for any injective match of our rules. No case analysis is needed.

*Proof.* Direct from the definitions. See [1], Theorem 3.11. $\square$

**Remark (no vertex deletion).** This means the graph can only grow or stay the same in vertex count. For the universality construction, this is not a limitation: TM simulation requires moving a head marker by *rewiring edges*, not by deleting and recreating vertices. The head marker "moves" by having its connecting edge deleted (from the old cell) and a new connecting edge created (to the new cell). The vertex count grows monotonically (one fresh vertex per step at the head position, used as workspace). We verify this explicitly in Section 5.

### 2.5 Application models

**Definition (Sequential step).** Given $G$, $\rho$, select the lexicographically first injective match $m: L \hookrightarrow G$ (ordering matches by the tuple $(m(v_1), \ldots, m(v_l))$ in the canonical vertex order of $L$). Apply $\rho$ at $m$. If no match exists, $G' = G$.

**Definition (GPI step).** Enumerate all injective matches of $L$ in $G$ in lexicographic order. Greedily select: include $m_i$ if $m_i(V(L)) \cap \bigcup_{j < i, m_j \text{ selected}} m_j(V(L)) = \emptyset$. Apply $\rho$ simultaneously at all selected matches. By Proposition 3, all applications are well-defined. By vertex-disjointness, the applications are parallel-independent (in the sense of DPO theory: the pushout complements and pushouts commute).

**Notation.** $G \xrightarrow{\rho}_{\text{seq}} G'$ and $G \xrightarrow{\rho}_{\text{GPI}} G'$.

### 2.6 Graph functions, dynamics, and computational classes

**Definition (Graph function).** A system $\mathcal{S} = (\rho, \alpha)$ computes the partial graph function $f_{\mathcal{S}}: \mathcal{G} \rightharpoonup \mathcal{G}$ defined by: $f_{\mathcal{S}}(G_0) = G_T$ if the trajectory reaches a fixed point $G_T = G_{T+1}$; $f_{\mathcal{S}}(G_0) = \bot$ otherwise.

**Definition (Trajectory map).** $\tau_{\mathcal{S}}: \mathcal{G} \to \mathcal{G}^{\omega}$ sends $G_0$ to the sequence $(G_0, G_1, G_2, \ldots)$.

**Definition (Classes).** $\mathcal{C}_l^{\alpha} = \{f_{\mathcal{S}} : \mathcal{S} \text{ at level } l\}$ (function class). $\mathcal{D}_l^{\alpha} = \{\tau_{\mathcal{S}} : \mathcal{S} \text{ at level } l\}$ (dynamics class). Extended classes $\hat{\mathcal{C}}_l^{\alpha}$, $\hat{\mathcal{D}}_l^{\alpha}$ use rule sets with per-step priority.

### 2.7 Sequential–GPI relationship

**Proposition 4.**

(i) *Unique-match agreement.* If $|\mathcal{M}_t| = 1$ at every step, sequential and GPI trajectories coincide.

(ii) *GPI advantage.* For some $(\rho, G_0)$, GPI reaches graphs unreachable by sequential. Example: vertex sprouting on $K_2$ (Section 3.2).

(iii) *Identical $k^*$.* The universality construction uses unique-match rules. Hence $k^*$ is model-independent.

*Proofs.* (i) Trivial: the greedy selection of a single match equals the first match. (ii) Vertex sprouting on $K_2$: sequential sprouts at vertex 0 only ($P_3$); GPI sprouts at both ($K_2$ plus two pendants). These are non-isomorphic. (iii) Follows from (i) and the construction in Section 5. $\square$

### 2.8 Rule catalog

| Signature | Connected $R$ | Rules |
|-----------|---------------|-------|
| $1_2 \to 1_2$ | 1 | Identity (trivial) |
| $1_2 \to 2_2$ | 1 | Vertex sprouting |
| $2_2 \to 3_2$ | 3 | One-sided sprouting (3.1); Subdivision (3.2); Triangle completion (3.3) |

If $R$-connectivity is dropped: 1, 2, and 6 rules respectively. See Appendix B for the exhaustive catalog.

---

## 3. Lower levels of the hierarchy

### 3.1 Level 1: finite-state

**Theorem 5.** *$\mathcal{C}_1^{\alpha}$ consists entirely of finite-state graph functions (for either $\alpha$). That is, the trajectory is eventually periodic, with period and transient bounded by $2^{\binom{n}{2}}$ where $n = |V(G_0)|$.*

**Proof.** At signature $1_2 \to 1_2$, $l = r = 1$, so no fresh vertices are created and no vertices are deleted. Hence $|V(G_t)| = |V(G_0)| = n$ for all $t$. The system evolves on the finite state space of all simple graphs on vertex set $V(G_0)$ (which has $2^{\binom{n}{2}}$ elements). Since the dynamics is a deterministic function on a finite set, the trajectory is eventually periodic. The graph function $f_\mathcal{S}$ is defined iff the trajectory reaches a fixed point; in all cases the trajectory eventually cycles with period dividing $2^{\binom{n}{2}}!$.

In fact, at this signature the only connected $L$ and $R$ are $K_1$ (the isolated vertex), and the only interface map is the identity. The rule is trivially the identity: $G_t = G_0$ for all $t$. Every graph is a fixed point. The computational class $\mathcal{C}_1^{\alpha}$ is therefore the singleton $\{\text{id}\}$. $\square$

**Remark.** The class $\hat{\mathcal{C}}_1^{\alpha}$ (rule sets) is also finite-state, since rule sets at $1_2 \to 1_2$ are all identity rules (with $R$ connected). The level is computationally trivial regardless of extension.

### 3.2 Level 2: the single context-free rule

**Theorem 6.** *The class $\mathcal{C}_2^{\text{GPI}}$ (single connected rule at $1_2 \to 2_2$) is the singleton $\{$vertex-sprouting function$\}$. Under GPI, the function maps $G_0$ to the unique trajectory $(G_0, G_1, G_2, \ldots)$ where $G_{t+1}$ is obtained by attaching a fresh pendant to every vertex of $G_t$. This trajectory never reaches a fixed point: the graph function $f_\mathcal{S}(G_0) = \bot$ for all $G_0$. Under sequential application, $f_\mathcal{S}(G_0) = \bot$ likewise (the graph grows by one vertex per step, never stabilizing).*

*In particular, $\mathcal{C}_2^{\alpha}$ is degenerate: it contains exactly one (everywhere-undefined) function. This degeneracy is a genuine feature of the formalism, not a bug — at this signature level, there is exactly one connected rule and it cannot condition on local topology, so it cannot implement halting.*

**Proof.** There is exactly one connected rule at $1_2 \to 2_2$: $L = K_1$, $R = K_2$, $\iota(0) = 0$. This matches every vertex (every graph contains $K_1$). Under GPI, all vertices are matched simultaneously (single-vertex matches are always vertex-disjoint). Each matched vertex sprouts a pendant. The graph grows at every step: $|V(G_{t+1})| > |V(G_t)|$. No fixed point is ever reached.

Under sequential application: one vertex sprouts per step. Again, $|V(G_{t+1})| = |V(G_t)| + 1$. No fixed point.

Since there is only one rule, $\mathcal{C}_2^{\alpha} = \{f_{\text{sprout}}\}$ where $f_{\text{sprout}}$ is everywhere undefined (divergent). $\square$

**Why "partially blind one-counter" is the wrong characterization for $\mathcal{C}_2$.** In the v1 draft, we claimed this level was "equivalent to partially blind one-counter machines." This is a category error. A partially blind counter machine is a device with a finite control that reads input symbols and manipulates a counter. Our single vertex-sprouting rule has no finite control, reads no input symbols, and performs no conditional operations. The "counter" (vertex count) increments unconditionally at every step. There is no sense in which this system processes input or branches on state. The characterization "partially blind one-counter" would require the rule-set extension $\hat{\mathcal{C}}_2$ and an input-encoding convention; even then, it would require rules at this level that can implement conditional behavior. Since $L = K_1$ provides zero topological context, no conditional behavior is possible at this signature: the rule fires identically at every vertex regardless of its neighborhood.

**What the level CAN do (as a dynamical system, not a computing device).** Vertex sprouting produces a specific family of growing graphs — the *sprouting trees* — that have a clean recursive structure determined entirely by $G_0$ and $t$. This is the object of interest for PRIMO: classifying the dynamical behavior of this operation across initial graphs.

**The generation invariant.** Define the *generation* of a vertex $v$: gen$(v) = 0$ if $v \in V(G_0)$; gen$(v) = t$ if $v$ was created at step $t$. Under GPI sprouting:

**Proposition 5 (Structural invariant).** *The graph $G_t$ is determined up to isomorphism by $G_0$ and $t$. Every generation-$t$ vertex is a pendant attached to exactly one vertex of generation $< t$. The generation structure forms a forest: each vertex of generation $\geq 1$ has exactly one parent of lower generation.*

*Proof.* Induction on $t$. At step $t$, every vertex in $G_{t-1}$ sprouts one pendant. The fresh vertices form generation $t$; each connects to exactly one parent (its match vertex). The operation is uniform, so the result depends only on the structure of $G_{t-1}$, which by induction depends only on $G_0$ and $t-1$. $\square$

### 3.3 Level 3: edge-conditioned computation

At $2_2 \to 3_2$, $L = K_2$. Rules match edges. This is the first level with topological context: the rule fires only where edges exist, unlike level 2 which fires at every vertex.

With a single rule, $\mathcal{D}_3^{\alpha}$ contains three trajectory maps (one per connected rule): one-sided sprouting, subdivision, and triangle completion of matched edges. Each is a single deterministic operation producing a characteristic growth pattern.

**The rule-set extension is vacuous at this level.** All three connected rules share $L = K_2$. Under per-step priority, the highest-priority rule fires whenever the graph has at least one edge. Lower-priority rules never fire. Therefore $\hat{\mathcal{D}}_3 = \mathcal{D}_3$ and $\hat{\mathcal{C}}_3 = \mathcal{C}_3$: the rule-set extension adds no power at signature $2_2 \to 3_2$. This is because there is only one connected graph on 2 vertices ($K_2$), so all rules match the same pattern. (See Appendix C for the detailed argument.)

**Consequence for conditional computation.** A DPDA requires conditional branching (push vs. pop based on input symbol), but every rule at $2_2 \to 3_2$ performs the same unconditional operation at every matched edge. No combination of per-step priority and GPI/sequential application can produce conditional behavior. The precise obstruction is analyzed in Appendix C.

**What IS proved at level 3.** The dynamics-class separation $\mathcal{D}_2 \neq \mathcal{D}_3$ (Theorem 9, Section 4.2): the No-Betweenness Invariant shows that level-2 trajectories cannot produce degree-2 fresh vertices, while level-3 Rule 3.2 can. This is a genuine structural separation. Level 3 is strictly more powerful than level 2 — it can perform *different kinds* of graph operations (edge deletion, triangle formation, subdivision) — but it cannot perform *conditional* operations.

**Open Problem 1.** *Characterize the computational class at signature $2_2 \to 3_2$. Each of the three connected rules produces a specific unconditional graph transformation. No rule set at this level can implement conditional branching. We conjecture the class is a graph-theoretic analogue of D0L-systems on edges: context-free parallel edge replacement. This is strictly between level 2 (context-free vertex replacement) and any level with conditional operations (which requires $L$ size $\geq 3$, giving different LHS patterns for different rules).*

---

## 4. Strict separations

### 4.1 Level 1 vs. Level 2: unbounded growth

**Theorem 8 ($\mathcal{D}$-class separation).** *$\mathcal{D}_1 \neq \mathcal{D}_2$. Every trajectory in $\mathcal{D}_1$ has constant vertex count. Every trajectory in $\mathcal{D}_2$ from a non-empty initial graph has unbounded vertex count.*

**Proof.** At level 1, $|V(G_t)| = |V(G_0)|$ for all $t$ (Theorem 5). At level 2, the trajectory from any non-empty $G_0$ satisfies $|V(G_t)| \to \infty$ (vertex sprouting adds vertices at every step). The property "the trajectory exhibits unbounded vertex growth" is present in $\mathcal{D}_2$ and absent from $\mathcal{D}_1$. $\square$

### 4.2 Level 2 vs. Level 3: edge subdivision

This is the substantive separation.

**Definition.** The *one-step GPI subdivision function* $\text{Sub}_1: \mathcal{G} \to \mathcal{G}$ maps $G$ to the graph obtained by applying Rule 3.2 under GPI: every edge in the greedy maximal matching is subdivided (replaced by a 2-path through a fresh vertex).

**Theorem 9 ($\mathcal{D}$-class separation).** *$\mathcal{D}_2 \neq \mathcal{D}_3$. $\text{Sub}_1$ is realizable in $\mathcal{D}_3$ (by Rule 3.2). No trajectory in $\mathcal{D}_2$ from any initial graph $G_0$ with $|E(G_0)| \geq 1$ ever produces $\text{Sub}_1(G_0)$.*

**Proof of realizability.** Rule 3.2 at $2_2 \to 3_2$ matches an edge and subdivides it. Under GPI, a maximal matching of edges is selected; all are subdivided simultaneously. The result is $\text{Sub}_1(G_0)$.

**Proof of impossibility at level 2.** We prove a structural invariant:

**Lemma 10 (No-Betweenness Invariant).** *Let $(G_0, G_1, G_2, \ldots)$ be the trajectory of Rule 2.1 (vertex sprouting) under either sequential or GPI application, from any initial graph $G_0$. For every $t \geq 1$ and every vertex $w$ created at step $t$:*

$$|N_{G_t}(w) \cap V(G_{t-1})| = 1$$

*That is, every fresh vertex is adjacent to exactly one previously existing vertex (its parent). No fresh vertex is ever adjacent to two or more vertices of $V(G_{t-1})$.*

**Proof.** By inspection of Rule 2.1: $L = K_1$ (no edges), $R = K_2$ (one edge between vertex 0 and fresh vertex 1), $\iota(0) = 0$. Application at match $m(0) = v$: the fresh vertex $w$ is connected to $v$ by the single edge $\{v, w\}$ created in $R$. No other edges are created (the rule creates exactly one edge per match). No existing edges are modified (the LHS has no edges, so there are no edges to delete or preserve — only the external edges of $v$ are retained via the interface).

Under GPI: multiple vertices matched simultaneously. Vertex $v_i$ sprouts pendant $w_i$. The pendant $w_i$ is adjacent to $v_i$ only. Crucially, $w_i$ is NOT adjacent to $w_j$ for $i \neq j$ (the rule creates no edges between fresh vertices of distinct matches), and NOT adjacent to $v_j$ for $j \neq i$ (the rule creates an edge only between the matched vertex and its pendant).

Therefore: $N_{G_t}(w_i) \cap V(G_{t-1}) = \{v_i\}$, giving $|N_{G_t}(w_i) \cap V(G_{t-1})| = 1$. $\square$

**Completion of Theorem 9.** Edge subdivision of $G_0$ at any edge $\{u, v\} \in E(G_0)$ creates a fresh vertex $w$ with $N(w) \supseteq \{u, v\}$, so $|N(w) \cap V(G_0)| \geq 2$. By Lemma 10, this cannot occur in any trajectory at level 2. Therefore $\text{Sub}_1(G_0) \neq G_t$ for any $t$. $\square$

**Remark (generalization).** Lemma 10 says: at level 2, every fresh vertex has *creation degree* 1 (pendant upon creation). At level 3, Rule 3.2 creates vertices with creation degree 2 (connected to both endpoints of the subdivided edge). Rule 3.3 also creates degree-2 vertices (connected to both endpoints of the matched edge). This structural observation — *creation degree increases with signature level* — suggests a general separation technique:

**Conjecture (Creation-Degree Separation).** At level $l$ (signature $l_2 \to (l+1)_2$), the maximum creation degree of a fresh vertex is $\min(l, r-1)$. At level $l+1$, fresh vertices with creation degree $l+1$ can be produced. The function "create a vertex of degree $l+1$ among previously existing vertices" separates level $l$ from level $l+1$.

### 4.3 Higher separations

For $l \geq k^*$: all levels are Turing-complete, so no separation exists.

For $3 \leq l < k^*$: each level is sub-universal but potentially capable of conditional branching ($l \geq 3$ provides multiple LHS patterns). The separation $l$ vs. $l+1$ follows if we can exhibit a function computable at $l+1$ but not $l$. The Creation-Degree Conjecture would give this for all levels, but a proof requires formalizing the creation-degree bound as a function of $l$, which we leave open.

Between level $k^* - 1$ (sub-universal) and level $k^*$ (Turing-complete): the separation is immediate — any Turing-computable function not in the sub-universal class witnesses it.

---

## 5. Universality

### 5.1 Overview

We construct a rule set at signature $7_2 \to 8_2$ that simulates an arbitrary Turing machine. The construction uses per-step priority between rules with *different LHS edge patterns* (but the same vertex count $l = 7$). Different edge patterns in the LHS encode different match conditions, giving the conditional branching needed for TM simulation. This is the mechanism that is unavailable at level 2 (where $L = K_2$ is the unique connected 2-vertex graph).

**Why $l = 7$ and not less.** The LHS must simultaneously contain:
- The current tape cell $c_i$ (1 vertex)
- The pendant $p_i$ of $c_i$, for symbol reading (1 vertex)
- The head marker, a $K_3$ subgraph encoding the TM state (3 vertices)
- Both neighboring cells $c_{i-1}$ and $c_{i+1}$, for bidirectional movement (2 vertices)

Total: $1 + 1 + 3 + 2 = 7$ vertices. Fewer vertices would sacrifice either symbol reading (no pendant), state encoding (marker too small), or bidirectional movement (only one neighbor).

### 5.2 Tape encoding

**Cells.** Each tape cell is a single vertex $c_i$. The tape is a path: $c_0 - c_1 - c_2 - \cdots - c_n$. A left sentinel $s$ is connected to $c_0$ (acting as $c_{-1}$). A right sentinel $s_R$ is connected to $c_n$ (acting as $c_{n+1}$).

**Pendants.** Every cell $c_i$ has a pendant vertex $p_i$ connected by edge $\{c_i, p_i\}$. The pendant is always present regardless of the tape symbol.

**Symbol encoding.** The symbol at cell $c_i$ is encoded by a "leaning" edge:
- Symbol 1: edge $\{c_{i-1}, p_i\}$ exists (the pendant "leans left" toward the predecessor cell).
- Symbol 0: edge $\{c_{i-1}, p_i\}$ does not exist (the pendant connects only to $c_i$).

This encoding is a static property of the graph, set at initialization. No marker involvement is needed to read a symbol — the LHS simply checks whether the leaning edge exists.

**Head marker.** A triangle $K_3$ on vertices $\{m_1, m_2, m_3\}$, attached to the current cell $c_i$ by edge $\{c_i, m_1\}$. The marker is the unique triangle in the host graph (the tape path and pendants contain no triangles), ensuring exactly one match per step.

**State encoding.** TM state is encoded by the marker's internal edge pattern:
- State $q_0$ (active): all three edges $\{m_1,m_2\}, \{m_2,m_3\}, \{m_1,m_3\}$ present ($K_3$).
- State $q_1$ (halt): edge $\{m_2,m_3\}$ removed (marker is a path, not a triangle).

For TMs with more states, a larger marker ($K_4$ with $2^{\binom{4}{2}} = 64$ edge patterns) suffices. For the binary incrementer (2 states, 2 symbols), $K_3$ with 2 distinguishable patterns is enough.

### 5.3 The explicit DPO rules (binary incrementer)

The binary incrementer has transitions: $(q_0, 1) \to (q_0, 0, R)$ and $(q_0, 0) \to (q_1, 1, L)$.

**LHS canonical labeling (7 vertices):**

| LHS vertex | Role | Notation |
|---|---|---|
| 0 | Left neighbor $c_{i-1}$ | $c_L$ |
| 1 | Current cell $c_i$ | $c$ |
| 2 | Pendant $p_i$ | $p$ |
| 3 | Marker vertex $m_1$ | $m_1$ |
| 4 | Marker vertex $m_2$ | $m_2$ |
| 5 | Marker vertex $m_3$ | $m_3$ |
| 6 | Right neighbor $c_{i+1}$ | $c_R$ |

**Base edges** (present in both rules' LHS):

$$E_{\text{base}} = \{\{0,1\},\; \{1,2\},\; \{1,3\},\; \{1,6\},\; \{3,4\},\; \{4,5\},\; \{3,5\}\}$$

These encode: tape left linkage, cell-pendant, cell-marker attachment, tape right linkage, and the marker $K_3$.

**Rule $\rho_A$: transition $(q_0, 1) \to (q_0, 0, R)$.** Write 0, move right, stay $q_0$.

$L_A$: vertices $\{0,\ldots,6\}$, edges $E_{\text{base}} \cup \{\{0,2\}\}$. The edge $\{0,2\} = \{c_L, p\}$ is the symbol-1 indicator. Total: 8 edges, connected.

$R_A$: vertices $\{0,\ldots,7\}$ (vertex 7 = fresh), edges:
$$\{\{0,1\},\; \{1,2\},\; \{1,6\},\; \{3,4\},\; \{4,5\},\; \{3,5\},\; \{6,3\},\; \{1,7\}\}$$

Interface map $\iota_A$: identity on $\{0,\ldots,6\}$.

| LHS edge | Status in $R_A$ | Meaning |
|---|---|---|
| $\{0,1\}$ | preserved | tape left |
| $\{1,2\}$ | preserved | pendant stays |
| $\{1,3\}$ | **deleted** | detach marker from $c$ |
| $\{1,6\}$ | preserved | tape right |
| $\{3,4\}$ | preserved | marker $K_3$ |
| $\{4,5\}$ | preserved | marker $K_3$ |
| $\{3,5\}$ | preserved | marker $K_3$ |
| $\{0,2\}$ | **deleted** | write 0 (remove symbol indicator) |
| — | $\{6,3\}$ created | attach marker to $c_R$ |
| — | $\{1,7\}$ created | fresh pendant on $c$ (workspace) |

$R_A$ is connected: $0\!-\!1\!-\!2$, $1\!-\!6\!-\!3\!-\!4\!-\!5$, $1\!-\!7$. ✓

**Rule $\rho_B$: transition $(q_0, 0) \to (q_1, 1, L)$.** Write 1, move left, enter halt state.

$L_B$: vertices $\{0,\ldots,6\}$, edges $E_{\text{base}}$ (no $\{0,2\}$ edge — symbol 0). Total: 7 edges, connected.

$R_B$: vertices $\{0,\ldots,7\}$ (vertex 7 = fresh), edges:
$$\{\{0,1\},\; \{1,2\},\; \{1,6\},\; \{3,4\},\; \{3,5\},\; \{0,2\},\; \{0,3\},\; \{0,7\}\}$$

Interface map $\iota_B$: identity on $\{0,\ldots,6\}$.

| LHS edge | Status in $R_B$ | Meaning |
|---|---|---|
| $\{0,1\}$ | preserved | tape left |
| $\{1,2\}$ | preserved | pendant stays |
| $\{1,3\}$ | **deleted** | detach marker from $c$ |
| $\{1,6\}$ | preserved | tape right |
| $\{3,4\}$ | preserved | marker edge |
| $\{4,5\}$ | **deleted** | state $q_0 \to q_1$ (break $K_3$) |
| $\{3,5\}$ | preserved | marker edge |
| — | $\{0,2\}$ created | write 1 (add symbol indicator) |
| — | $\{0,3\}$ created | attach marker to $c_L$ (move left) |
| — | $\{0,7\}$ created | fresh pendant on $c_L$ (workspace) |

$R_B$ is connected: $0\!-\!1\!-\!2$, $0\!-\!3\!-\!4$, $3\!-\!5$, $0\!-\!7$, $0\!-\!2$, $1\!-\!6$. ✓

**Priority.** $\rho_A > \rho_B$ (try $\rho_A$ first at each step). Since $L_A$ requires edge $\{0,2\}$ while $L_B$ does not:
- Symbol 1 (edge $\{c_L, p\}$ present): $\rho_A$ matches. $\rho_A$ fires.
- Symbol 0 (edge $\{c_L, p\}$ absent): $\rho_A$ fails. $\rho_B$ matches. $\rho_B$ fires.
- State $q_1$ (marker not $K_3$): neither matches ($K_3$ edges required by both). Halt.

### 5.4 Tape extension

Each TM step is simulated in one rewriting step for interior cells and at most two rewriting steps when the head reaches a tape boundary (one step for the transition, one step to extend the tape by creating a new cell vertex and pendant from the fresh vertex). This constant-factor slowdown does not affect the computability result. For the binary incrementer on a finite pre-allocated tape, boundary extension is not needed: the computation uses at most $n + 1$ cells for an $n$-cell input. For the general universality result (Theorem 11), a two-step boundary protocol ensures that the tape can grow without bound. The walkthrough in Section 5.5 demonstrates the interior case; the boundary case is not exercised in this example because the tape is pre-allocated with sentinels.

### 5.5 Verified walkthrough: binary incrementer on $[1,1,0]$

**Initial graph.** Tape $[1,1,0]$ = binary 011 = 3. Head at $c_0$, state $q_0$.

Vertices (with their roles):

| Vertex | Role |
|---|---|
| 0 | Sentinel $s$ (= $c_{-1}$) |
| 1 | Cell $c_0$ |
| 2 | Pendant $p_0$ |
| 3 | Cell $c_1$ |
| 4 | Pendant $p_1$ |
| 5 | Cell $c_2$ |
| 6 | Pendant $p_2$ |
| 7 | Marker $m_1$ |
| 8 | Marker $m_2$ |
| 9 | Marker $m_3$ |
| 10 | Right sentinel $s_R$ (= $c_3$) |

Initial edges:
- Tape: $\{0,1\}, \{1,3\}, \{3,5\}, \{5,10\}$
- Pendants: $\{1,2\}, \{3,4\}, \{5,6\}$
- Symbol 1 at $c_0$: $\{0,2\}$ ($s$-to-$p_0$ leaning edge)
- Symbol 1 at $c_1$: $\{1,4\}$ ($c_0$-to-$p_1$ leaning edge)
- Symbol 0 at $c_2$: no leaning edge
- Marker $K_3$: $\{7,8\}, \{8,9\}, \{7,9\}$
- Head at $c_0$: $\{1,7\}$

**Step 1.** Head at $c_0$. Check $\rho_A$ match (requires $\{c_L, p\} = \{0, 2\}$):

LHS mapping: $0 \mapsto s = 0$, $1 \mapsto c_0 = 1$, $2 \mapsto p_0 = 2$, $3 \mapsto m_1 = 7$, $4 \mapsto m_2 = 8$, $5 \mapsto m_3 = 9$, $6 \mapsto c_1 = 3$.

Check all required edges of $L_A$: $\{0,1\}$✓, $\{1,2\}$✓, $\{1,7\}$(= LHS $\{1,3\}$)✓, $\{1,3\}$(= LHS $\{1,6\}$)✓, $\{7,8\}$(= LHS $\{3,4\}$)✓, $\{8,9\}$(= LHS $\{4,5\}$)✓, $\{7,9\}$(= LHS $\{3,5\}$)✓, $\{0,2\}$✓. All present. **$\rho_A$ fires.**

Apply $\rho_A$: delete $\{1,7\}$ (detach marker), delete $\{0,2\}$ (write 0). Create $\{3,7\}$ (attach marker to $c_1$), create $\{1,11\}$ (fresh pendant, vertex 11).

**After step 1:** Tape $[0,1,0]$. Head at $c_1$. State $q_0$.

**Step 2.** Head at $c_1$. LHS mapping: $0 \mapsto c_0 = 1$, $1 \mapsto c_1 = 3$, $2 \mapsto p_1 = 4$, $3 \mapsto m_1 = 7$, $4 \mapsto m_2 = 8$, $5 \mapsto m_3 = 9$, $6 \mapsto c_2 = 5$.

Check $\rho_A$: need $\{0,2\} = \{1, 4\}$. Symbol 1 at $c_1$ means $\{c_0, p_1\} = \{1, 4\}$ exists. ✓ **$\rho_A$ fires.**

Apply $\rho_A$: delete $\{3,7\}$ (detach), delete $\{1,4\}$ (write 0). Create $\{5,7\}$ (attach to $c_2$), $\{3,12\}$ (fresh pendant, vertex 12).

**After step 2:** Tape $[0,0,0]$. Head at $c_2$. State $q_0$.

**Step 3.** Head at $c_2$. LHS mapping: $0 \mapsto c_1 = 3$, $1 \mapsto c_2 = 5$, $2 \mapsto p_2 = 6$, $3 \mapsto m_1 = 7$, $4 \mapsto m_2 = 8$, $5 \mapsto m_3 = 9$, $6 \mapsto s_R = 10$.

Check $\rho_A$: need $\{0,2\} = \{3, 6\}$. Symbol 0 at $c_2$: edge $\{c_1, p_2\} = \{3, 6\}$ does NOT exist. **$\rho_A$ fails.** Try $\rho_B$: need all base edges but NOT $\{0,2\}$. Check: $\{3,5\}$✓, $\{5,6\}$✓, $\{5,7\}$✓, $\{5,10\}$✓, $\{7,8\}$✓, $\{8,9\}$✓, $\{7,9\}$✓. All present. **$\rho_B$ fires.**

Apply $\rho_B$: delete $\{5,7\}$ (detach), delete $\{8,9\}$ (state $\to q_1$). Create $\{3,6\}$ (write 1), $\{3,7\}$ (attach marker to $c_1$), $\{3,13\}$ (fresh pendant, vertex 13).

**After step 3:** Tape $[0,0,1]$. Head at $c_1$. State $q_1$ (marker edges: $\{7,8\}, \{7,9\}$, missing $\{8,9\}$).

**Step 4.** Head at $c_1$. Check $\rho_A$: requires $K_3$ on marker, i.e., edges $\{7,8\}, \{8,9\}, \{7,9\}$. Edge $\{8,9\}$ absent. **$\rho_A$ fails.** Check $\rho_B$: also requires $K_3$. **$\rho_B$ fails.** No match. **HALT.**

**Result:** Tape $[0,0,1]$ = binary $100$ = $4$. Computation $3 + 1 = 4$. ✓

### 5.6 General universality

**Theorem 11 (Universality).** *$k^* \leq 7$ for TMs with $q|\Gamma| \leq 8$; $k^* \leq 8$ for arbitrary TMs. Specifically: a rule set of $2q|\Gamma|$ rules at signature $7_2 \to 8_2$ (using a $K_3$ marker) simulates any TM with $q$ states and alphabet $\Gamma$ satisfying $q|\Gamma| \leq 8$, e.g., 4 states $\times$ 2 symbols. For larger TMs, use a $K_4$ marker ($2^6 = 64$ edge patterns), increasing the LHS to 8 vertices and the signature to $8_2 \to 9_2$. The construction uses:*
- *Cell encoding: 1 vertex per cell, 1 pendant per cell, symbol via leaning edge.*
- *Head marker: $K_3$ (or $K_4$ for $> 8$ state-symbol configurations).*
- *Both tape neighbors in LHS for bidirectional movement.*
- *Per-step priority between rules with different LHS edge patterns for conditional branching.*
- *Unique-match property: $K_3$ (resp. $K_4$) appears only in the marker.*

*Any Turing machine can be simulated by a binary-alphabet TM with at most a polynomial overhead in time and a constant-factor overhead in tape (see Arora and Barak [18], Theorem 1.13). The Neary–Woods $(15,2)$ UTM [5] — a standard-model universal Turing machine with 15 states and 2 symbols — gives $q|\Gamma| = 30$, which exceeds 8 and requires a $K_4$ marker (signature $8_2 \to 9_2$, giving $k^* \leq 8$). For TMs with $q|\Gamma| \leq 8$ (which suffice for any specific computation, though not for a single UTM), $K_3$ gives $k^* \leq 7$.*

*The threshold $k^*$ is identical for sequential and GPI application (Proposition 4(i)): the unique-match property ensures a single match per step.*

*Proof.* The explicit construction for the binary incrementer (Section 5.3) demonstrates the mechanism. The walkthrough (Section 5.5) verifies correctness on a concrete input. The generalization to arbitrary TMs replaces the two rules $\rho_A, \rho_B$ with a family of rules: one rule per (state, symbol) pair, each with the appropriate LHS edge pattern (encoding the symbol being read) and RHS edge pattern (encoding the symbol written, the direction moved, and the new state). Per-step priority orders these rules, with more specific LHS patterns (more edges) having higher priority. The construction is standard given the encoding; the key contribution is the encoding itself (leaning-edge symbol, $K_3$ marker, 7-vertex LHS window). $\square$

---

## 6. Complexity bridge and logic connections

### 6.1 Signature complexity

**Definition.** $\sigma(f) = \min\{l : f \in \mathcal{C}_l^{\alpha}\}$ (or $\hat{\mathcal{C}}_l^{\alpha}$ under rule-set extension). This is well-defined for any computable graph function and equals $k^*$ for Turing-complete functions.

### 6.2 Complexity bridge

**Theorem 12 (Complexity bridge).** *For all $l \geq 1$, every rule $\rho$ at signature $l_2 \to (l+1)_2$ can be simulated by a rule $\rho'$ at signature $(l+1)_2 \to (l+2)_2$ such that:*

(a) *Function equivalence: $f_{(\rho', \alpha)}(G_0) = f_{(\rho, \alpha)}(G_0)$ for all $G_0$ in a cofinite subset of $\mathcal{G}$ (specifically, all graphs where every vertex in any match of $L$ has degree $\geq 1$).*

(b) *Description overhead: the rule $\rho'$ can be specified from $\rho$ using $C_l = O(l)$ additional bits.*

**Proof.** *Construction.* Given $\rho = (L, R, \iota)$, choose $v \in V(L)$. Define $L' = L + \{v'\}$ with edge $\{v, v'\}$ (so $|V(L')| = l + 1$, and $L'$ is connected). Define $R' = R + \{v', w\}$ with edge $\{\iota(v), w\}$ (so $|V(R')| = l + 2$, and $R'$ is connected if $R$ is). The interface map: $\iota'(u) = \iota(u)$ for $u \in V(L)$; $\iota'(v') = v'$. Fresh vertex: $w$.

*Function equivalence.* A match of $L'$ in $G$ is an injective morphism mapping $L$ to some subgraph of $G$ AND mapping $v'$ to a neighbor of $m(v)$ outside the match of $L$. This exists whenever $m(v)$ has a neighbor not in $m(V(L))$, which holds for all vertices of degree $\geq l$ in any graph with $|V(G)| > l$. For such graphs, every match of $L$ extends to a match of $L'$, so the set of "active" positions is the same (possibly with different multiplicities, which affects GPI but not the computed function).

The local rewrite at each match position is: apply $\rho$ (same effect on the $L$ portion) plus create a pendant $w$ attached to $\iota(v)$. The pendant $w$ is workspace that does not interact with other parts of the graph. The computed function is the same (the additional pendant does not affect halting or the graph structure at fixed point, modulo the pendants — which can be stripped by convention).

*Overhead.* Specifying $\rho'$ from $\rho$ requires: choice of $v$ ($\lceil \log_2 l \rceil$ bits), adjacency of $v'$ in $L'$ ($l$ bits), adjacency of $v', w$ in $R'$ ($l + 1$ bits). Total: $C_l = O(l)$. $\square$

### 6.3 Relationship to Kolmogorov complexity

The interesting content of the complexity bridge is internal to the formalism: the signature ordering is monotone with description complexity. This is a statement about the specific GPI formalism as a programming language, not a Kolmogorov-complexity result (the invariance theorem gives the reverse inequality trivially for any formalism).

**Corollary (Signature monotonicity).** *The sequence of minimum description lengths $|S_l(f)|$ for a fixed function $f$ is non-increasing in $l$: $|S_{l+1}(f)| \leq |S_l(f)| + C_l$. Combined with the strict separations (some functions require level $l+1$ and are not computable at level $l$), this gives a quantitative hierarchy: higher levels are strictly more expressive and every function at a lower level has a slightly longer description at the higher level.*

### 6.4 Connections to bounded-variable logics

**Static correspondence.** The LHS size $l$ corresponds to the pebble count $k$ in Immerman's $k$-pebble game. A match of $L$ in $G$ is an injection from $l$ vertices to $G$ — a "pebble placement." Two graphs are $\mathcal{L}^l_{\infty\omega}$-equivalent (Duplicator wins the $l$-pebble game) iff they agree on all subgraph counts for patterns of size $\leq l$. Since the match count $|\{m : L \hookrightarrow G\}|$ for $|V(L)| = l$ is determined by $\mathcal{L}^l_{\infty\omega}$-equivalence class (Lovász 1967; Dell–Grohe–Rattan 2018):

**Proposition 13.** *If $G \equiv_{\mathcal{L}^l_{\infty\omega}} H$, then for any rule $\rho$ at level $l$, the number of matches of $L$ in $G$ equals the number of matches of $L$ in $H$.*

This means the $l$-pebble equivalence is a sufficient condition for two graphs to "look the same" to any level-$l$ rule at a single step.

**Dynamic breakdown.** Over multiple steps, the correspondence fails: a level-$l$ rule applied for $t$ steps effectively "scans" $O(t \cdot l)$ vertices, accumulating information in the graph's evolving structure. The system's discriminating power after $t$ steps exceeds $\mathcal{L}^l_{\infty\omega}$. A precise characterization would require a *temporal bounded-variable logic* — something like $l$-variable logic with a constructive least-fixed-point operator, restricted to operations realizable by DPO rules. The known logics FP$^k$ (fixed-point logic with $k$ variables) are close but do not exactly match, because our rules are *constructive* (they add vertices and change edges) rather than *observational* (they test properties).

**Courcelle's MSO transductions.** Checking whether a fixed pattern $L$ has an injective match in $G$ is first-order definable (a sentence with $l$ existentially quantified vertex variables). Selecting a maximal independent match set is MSO-definable (quantify over sets of matches). The iterated GPI application of a fixed rule is therefore an iterated MSO-definable transduction. Courcelle's theory of MSO-definable graph transductions provides general tools (closure under composition, decidability on bounded-treewidth graphs) that may yield upper bounds on our formalism's power. Specifically: if a class of graphs has bounded treewidth, then all MSO-definable properties of its elements are decidable. Our lower-level rules produce graphs of bounded treewidth (level-2 sprouting produces trees), so Courcelle's theorem applies and gives decidability of trajectory properties — consistent with the "finite-state-like" behavior at low levels.

**Open Problem 2 (Descriptive complexity connection).** *Find a logic $\mathcal{L}_l$ such that the graph functions computable by level-$l$ GPI systems (running for arbitrarily many steps) are exactly the $\mathcal{L}_l$-definable transductions. Candidate: $l$-variable constructive fixpoint logic over DPO spans.*

---

## 7. Computational illustration

See Section 5.5 for the binary incrementer walkthrough. The companion study (Paper 1, in preparation) evaluates Rule 3.2 (subdivision) on initial graphs $K_1, K_2, K_3, P_3$ and classifies it as $I$-positive and $\Phi$-positive. The theoretical hierarchy places this rule at the first level with topological context ($l = 2$), consistent with the empirical observation that complex dynamical behavior requires context-sensitivity (the L-system analogy: context-free D0L produces regular behavior; context-dependent IL produces complex behavior).

**Rule count reconciliation.** This paper catalogs 5 connected rules across 3 signature levels. The companion study's $\sim$8,000 count arises from: (i) signatures $l_2 \to r_2$ with $r \neq l+1$ (including $2_2 \to 2_2$, $1_2 \to 3_2$, $3_2 \to 4_2$, etc.); (ii) disconnected $R$; (iii) rule instances on specific initial graphs from $\mathcal{G}_0 = \{K_1, K_2, K_3, P_3\}$ (multiplying by 4); (iv) higher signatures. The theoretical hierarchy depends only on LHS size and the $r = l+1$ convention.

---

## 8. Open problems

**1. Conditional branching threshold.** At what LHS size does conditional branching first become possible? At $l = 2$, all rules share $L = K_2$ and no conditional behavior is achievable (Appendix C). At $l = 7$, the universality construction implements conditional branching via different LHS edge patterns. What happens at $l = 3, 4, 5, 6$? At $l = 3$, there are two connected LHS patterns ($K_3$ and $P_3$), so rules can have genuinely different match conditions, and per-step priority is non-vacuous. Can conditional branching at $l = 3$ simulate counter machines? DPDAs? The answer determines the "complexity gap" between the unconditional levels ($l \leq 2$) and the universal level ($l \leq 8$).

**2. Tighten $k^*$.** Current: $k^* \leq 7$ (with $K_3$ marker, for TMs with $q|\Gamma| \leq 8$) or $k^* \leq 8$ (with $K_4$ marker, for arbitrary TMs via the Neary–Woods UTM). Can the LHS be reduced to 6 vertices? This requires either sharing a vertex between the marker and the cell gadget, or using a 2-vertex marker (which encodes only $2^1 = 2$ states). A lower bound on $k^*$ would require proving that no rule set at signature $6_2 \to 7_2$ can simulate an arbitrary TM.

**3. Sequential vs. GPI.** At sub-universal levels, is GPI strictly more powerful? Characterize when trajectories coincide (confluence conditions).

**4. Justify $r = l+1$ formally.** Prove that for all $l$ and $j \geq 2$, $\mathcal{C}^{\alpha}(l, l+j) = \mathcal{C}^{\alpha}(l, l+1)$ (multi-step iteration at $r = l+1$ covers the same class as single-step at $r = l+j$).

**5. Descriptive complexity.** Find logic $\mathcal{L}_l$ matching level-$l$ GPI power (Open Problem 2).

**6. Creation-degree separation.** Formalize the conjecture that level $l$ cannot create vertices of degree $> l-1$ among existing vertices, and prove it for all $l$.

**7. Ordering independence.** Prove or disprove: the computational class $\mathcal{C}_l^{\text{GPI}}$ is independent of the vertex ordering on the host graph.

**8. Hyperedge generalization.** Extend to arity-$a$ hyperedges. How does $(l, r, a)$ affect the hierarchy?

---

## Appendix A: Notation

| Symbol | Meaning |
|--------|---------|
| $l_2 \to r_2$ | Signature |
| $\rho = (L, R, \iota)$ | Rule with interface map |
| $\mathcal{C}_l^{\alpha}$ | Function class (halting-based) |
| $\mathcal{D}_l^{\alpha}$ | Dynamics class (trajectory-based) |
| $\hat{\mathcal{C}}_l^{\alpha}, \hat{\mathcal{D}}_l^{\alpha}$ | Extended classes (rule sets, per-step priority) |
| $f_{\mathcal{S}}$ | Graph function of system $\mathcal{S}$ |
| $\tau_{\mathcal{S}}$ | Trajectory map of system $\mathcal{S}$ |
| $k^*$ | Universality threshold ($\leq 7$ or $\leq 8$) |

## Appendix B: Complete rule catalog

See rule_catalog.md for the exhaustive enumeration and verification of all rules at signatures $1_2 \to 1_2$, $1_2 \to 2_2$, and $2_2 \to 3_2$.

## Appendix C: Why the DPDA simulation fails at $2_2 \to 3_2$

This appendix provides the complete analysis of why conditional branching — and hence DPDA simulation — is not achievable at signature $2_2 \to 3_2$ with plain DPO and per-step priority.

### C.1 The obstruction: all rules share $L = K_2$

At signature $2_2 \to 3_2$, the LHS must be a connected graph on 2 vertices. The only such graph is $K_2$ (a single edge). All three connected rules (3.1, 3.2, 3.3) and all three disconnected-$R$ rules (3.4, 3.5, 3.6) have $L = K_2$.

Under per-step priority with rule tuple $(\rho_1, \ldots, \rho_k)$: at each step, try $\rho_1$ first. Rule $\rho_1$ has $L_1 = K_2$. It has an injective match in the host graph iff the graph contains at least one edge. If the graph has an edge, $\rho_1$ fires. Rules $\rho_2, \ldots, \rho_k$ are never reached.

If the graph has no edges (edgeless), no rule fires and the system halts.

**Therefore:** the system reduces to a single rule $\rho_1$ (the highest-priority rule). The rule-set extension $\hat{\mathcal{D}}_3 = \mathcal{D}_3$ at this level. The three connected rules give three distinct single-rule dynamics, but no multi-rule dynamics.

### C.2 Why this prevents conditional branching

A DPDA computes by selecting different transitions based on the current state and input symbol. This requires the rewriting system to *behave differently* at different steps, depending on the graph's structure.

With a single rule acting unconditionally (same operation at every matched edge), the system cannot branch. Rule 3.2 always subdivides. Rule 3.1 always sprouts. Rule 3.3 always triangulates. There is no "if the cell contains symbol 1, then push; if symbol 0, then pop."

### C.3 Could alternative rule-set mechanisms help?

**Per-match priority** (choose between rules at each match position based on local structure): this requires the rules to have *different* LHS patterns, so that one matches at position $m$ and another does not. Since all rules share $L = K_2$, any match of one rule is a match of all rules. Per-match priority is also vacuous.

**Negative application conditions (NACs):** a NAC forbids a rule from firing when a larger pattern is present. With NACs, Rule $\rho_1$ could fire only when the matched edge's endpoint has no pendant (detectable by a NAC requiring $K_{1,1}$ absence). NACs are not part of our plain DPO model.

**Vertex labels or colors:** with colored vertices, different rules could match different colors. This is equivalent to having different LHS patterns. Vertex coloring is not part of our model.

**Larger LHS ($l \geq 3$):** at $l = 3$, there are two connected LHS patterns ($K_3$ and $P_3$). Rules with $L = K_3$ match only at triangles; rules with $L = P_3$ match at 2-paths. These are genuinely different match conditions, so per-step priority is non-vacuous. Conditional branching becomes possible.

### C.4 What IS achievable at $2_2 \to 3_2$

Each connected rule at this level computes a single unconditional graph transformation:
- Rule 3.1: sprout a pendant at one endpoint of each matched edge.
- Rule 3.2: subdivide each matched edge.
- Rule 3.3: complete a triangle at each matched edge.

Under GPI, the maximal matching determines which edges are processed. The *selection* of which edges are matched is data-dependent (it depends on the graph structure), but the *operation* at each matched edge is fixed. This is analogous to a D0L-system on edges: context-free parallel edge replacement.

The dynamics class $\mathcal{D}_3$ is strictly richer than $\mathcal{D}_2$: it contains three qualitatively different transformation types (vs. one at level 2), it can condition on edge existence (rules fire only where edges are), and it can delete edges (Rule 3.2). The No-Betweenness separation (Theorem 9) is the formal witness.

---

## References

1. Ehrig, H., Ehrig, K., Prange, U., Taentzer, G. *Fundamentals of Algebraic Graph Transformation.* EATCS Monographs in TCS, Springer, 2006.
2. Ehrig, H., Kreowski, H.-J. Parallelism of manipulations in multidimensional information structures. *MFCS*, LNCS 45, 1976.
3. Rozenberg, G., Salomaa, A. (eds.) *The Mathematical Theory of L Systems.* Academic Press, 1980.
4. Rogozhin, Y. Small universal Turing machines. *TCS* 168(2), 215–240, 1996.
5. Neary, T., Woods, D. Four small universal Turing machines. *Fundamenta Informaticae* 91(1), 123–144, 2009.
6. Greibach, S.A. Remarks on blind and partially blind one-way multicounter machines. *TCS* 7(3), 311–324, 1978.
7. Immerman, N. Upper and lower bounds for first order expressibility. *JCSS* 25, 76–98, 1982.
8. Abramsky, S., Dawar, A., Wang, P. The pebbling comonad in finite model theory. *LICS*, 2017.
9. Courcelle, B. Monadic second-order definable graph transductions: a survey. *TCS* 126, 53–75, 1994.
10. Courcelle, B. The monadic second-order logic of graphs I. *Inf. Comput.* 85, 12–75, 1990.
11. Courcelle, B., Engelfriet, J. *Graph Structure and Monadic Second-Order Logic.* Cambridge University Press, 2012.
12. Lovász, L. Operations with structures. *Acta Math. Acad. Sci. Hung.* 18, 321–328, 1967.
13. Dell, H., Grohe, M., Rattan, G. Lovász meets Weisfeiler and Leman. *ICALP*, 2018.
14. Otto, M. *Bounded Variable Logics and Counting.* Lecture Notes in Logic 9, Springer, 1997.
15. Montacute, Y., Shah, N. The pebble-relation comonad in finite model theory. *LMCS* 20(2), 2024.
16. Wolfram, S. A class of models with the potential to represent fundamental physics. *Complex Systems* 29(2), 2020.
17. Zenil, H., Kiani, N.A., Tegnér, J. *Algorithmic Information Dynamics.* Cambridge University Press, 2023.
18. Arora, S., Barak, B. *Computational Complexity: A Modern Approach.* Cambridge University Press, 2009.
19. Author. Geometric predicates for classifying dynamical behaviors in graph rewrite systems (Paper 1). In preparation, 2026.
20. Author. Geometric signatures of Bayesian inference in discrete dynamical systems (Paper 2). In preparation, 2026.
