# PRIMO

**Primacy of Inference over Physics in the Space of Minimal Programs**

We conjecture that in the space of all programs ordered by complexity, inference-like behavior appears at strictly shorter program lengths than physics-like behavior, and that physics-like behavior is generically the dynamical equilibrium of an inference-like process.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test          # unit + regression (~2 min)
make exp01         # reproduce 33-rule validation (~5 min, requires a display)
```

Requires Python 3.10+, numpy, scipy, networkx, matplotlib. GPU acceleration (optional): PyTorch with CUDA. All experiments open a Tkinter monitor window and cannot run headless.

## Current state

**Last updated:** 2026-03-13 (exp06 temporal I-profiles)

**Phase 1: COMPLETE.** All 6 library files, all tests passing, exp01 reproduces the full 33-rule diagnostic.

**Phase 2: IN PROGRESS.** exp01–04, exp06–09 complete. I/Φ separation achieved at σ=4. exp05 skipped (see below).

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3, 3→4 (1, 1, 3, 11 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- `primo/`: all 6 library files (backend, rules, trajectories, predicates, monitor, run_utils)
- `exp01`: 33-rule validation — all rules match reference at original thresholds
- `exp02`: Example B analysis — canonical I-negative (0/4), robustness 18/20 I-positive, adaptive 4/4 I-positive
- `exp03`: Straightness gate calibrated at 0.35 (was provisional 0.45) — zero I+ rules affected
- `exp04`: PRIMO enumeration (16 rules, signatures ≤ 3→4) — rerun at ds_std*=0.08
- `exp07`: DPO null-model recalibration — 174 separating threshold combos found
- `exp08`: Cross-check 33 rules at ds_std*=0.08 — all 4 cells populated (16, 6, 1, 10)
- `exp09`: Eigenvalue gap stability diagnostic — zero crossings across all 16 DPO rules. 3 borderline Φ+ rules (small graphs) confirmed stable at T=60 (exp09b). Conditional theorem M3' verified: all Φ+ DPO rules have stable eigenvalue gaps once graphs reach sufficient size.
- `exp06`: Temporal I-profiles for 9 Φ+ rules at T=60 — claim (b) assessment: WEAK. 4/9 rules show transient in ≥1 seed; 2/9 (Path-4 fresh middle, Tri+pendant shifted) show strong transients in all 4 seeds. Mean I-Φ correlation -0.076 (weakly anti-correlated).

**exp08 key findings (ds_std* tightened to 0.08):**
- Only 1 rule changed among 33: lattice_rewire moved I+Φ+ → I+Φ- (ds_std=0.149)
- All 4 cells populated: (16, 6, 1, 10) — hard constraint satisfied
- Critical I-Φ+ cell (fixed_grid_noise) preserved

**exp04 rerun findings (at ds_std*=0.08):**
- σ=1: Identity (I- Φ-) — unchanged
- σ=2: Vertex Sprouting (I+ Φ+) — unchanged
- σ=3: 3 rules, all I+ Φ+ — unchanged
- σ=4: **5 I+Φ+, 6 I+Φ-** — separation achieved!
  - I+Φ-: Star-3 replacement, Star-3 fresh hub, Path-4 partial preserve, Diamond minus one, Diamond preserved, K4 completion
  - I+Φ+: Path-4 fresh middle, Tri+pendant (preserved/shifted/fresh hub), Diamond fresh vertex
- N_I^min = N_Φ^min = σ=2 — claim (a) still inconclusive (tie)
- Claim (b) PARTIAL: 3/9 Φ+ rules show I-positive transient decay
  - Path-4 fresh middle (K3: delta +0.28), Tri+pendant preserved (P3: +0.16), Tri+pendant shifted (K3: +0.50)
- **S1 SUPPORTED at σ=4**: 6 I-only rules, 0 Φ-only rules
- **S2 SUPPORTED at σ=4**: I+=11/11 > Φ+=5/11

**What's next:**
- exp05: SKIPPED. The conditional theorem (Proposition 2, Paper 1 §7.5) proves Φ+ → I+ for all DPO growth rules, so N_I^min ≤ N_Φ^min holds trivially in the DPO setting. Higher-σ enumeration cannot break the tie. Claim (a) requires multi-rule compositions or non-DPO program enumeration to test properly.
- Paper 4 drafting: exp04 (enumeration) + exp06 (temporal profiles) + exp09 (eigenvalue gaps) provide the experimental core. Begin assembling results into primo_conjecture.md.
- Paper 1 update: add Proposition (Φ+ → I+ for monotone growth with stable eigenvalue gaps) to Section 7, with exp09/exp09b data as evidence

**Known issues:**
- `watts_strogatz` is a boundary rule: Φ classification varies across runs (2–4 seeds)
- Straightness gate overlap: no clean separation between I+ rules and contractions (overlap 0.13–0.50)
- Adaptive variant not caught by straightness gate (S ~0.13–0.17); identified mitigation: active dynamics gate requiring edit distance > 0 for at least 2T/3 steps (see `logs/decisions.md`)
- Example B seed sensitivity: 18/20 random seeds classify I-positive despite canonical protocol giving I-negative; predicate is sensitive to initial conditions (see `reference/example_b_analysis.md`)
- At ds_std*=0.08, all σ≤3 DPO rules are I+ Φ+. Separation only appears at σ=4. Claim (a) requires σ=5 enumeration or is structurally impossible for single-rule DPO.
- Φ+ → I+ holds for all 16 DPO rules. Explained by: zero eigenvalue crossings under monotone DPO growth + Davis-Kahan theorem. Does NOT hold in general (fixed_grid_noise counterexample). Conditional theorem drafted, pending Paper 1 integration.
- Claim (a) structurally untestable in single-rule DPO: conditional theorem guarantees N_I^min ≤ N_Φ^min. Testing strict inequality requires richer program spaces.
- License TBD — must be resolved before Paper 4 data availability statement

**Hardware:**
- Development: RTX 3050 (4GB VRAM), `DEVICE="cuda"`, `GPU_BATCH_SIZE=8`
- Target: university cluster (A100 80GB), `GPU_BATCH_SIZE=256`

## Project structure

```
primo/                     # Python library
experiments/               # One script per experiment
tests/                     # Unit + regression tests
papers/                    # Markdown drafts (LaTeX conversion at Phase 4)
data/catalogs/             # Rule catalogs (small JSON, committed)
data/results/              # Experiment outputs (.gitignored)
logs/                      # Decision log, experiment log
reference/                 # Historical oracle files (do not modify)
```

See `CLAUDE.md` for the full five-phase plan from bootstrap to submission.

## Papers

| # | Title | Target venue | Status |
|---|-------|-------------|--------|
| 1 | Geometric Predicates for Classifying Dynamical Behaviors in Graph Rewrite Systems | Entropy | Draft complete |
| 2 | Geometric Signatures of Bayesian Inference in Discrete Dynamical Systems | JMLR | Draft complete |
| 3 | Computational Power of Parallel Graph Rewrite Systems by Signature Complexity | Theoretical CS | Draft complete |
| 4 | PRIMO: Primacy of Inference over Physics in the Space of Minimal Programs | TBD | Draft v6, awaiting experimental results |

## License

TBD
