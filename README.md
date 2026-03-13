# PRIMO

**Primacy of Inference over Physics in the Space of Minimal Programs**

We conjecture that in the space of all programs ordered by complexity, inference-like behavior appears at strictly shorter program lengths than physics-like behavior, and that physics-like behavior is generically the dynamical equilibrium of an inference-like process.

## Quick start

```bash
pip install -r requirements.txt
make test          # unit + regression (~2 min)
make exp01         # reproduce 33-rule validation (~5 min, opens Tkinter monitor)
```

Requires Python 3.10+, numpy, scipy, networkx, matplotlib. GPU acceleration (optional): PyTorch with CUDA.

## Current state

**Last updated:** 2026-03-13

**Phase 1: COMPLETE.** All 6 library files, 301 tests passing, exp01 reproduces the full 33-rule diagnostic.

**Phase 2: IN PROGRESS.** exp01–03 complete, exp04–06 next.

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3 (1, 1, 3 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- `primo/`: all 6 library files built (backend, rules, trajectories, predicates, monitor, run_utils)
- Tests: 301 passing
- `exp01`: 33-rule validation — all rules match reference, stability 75.8%
- `exp02`: Example B analysis — canonical I-negative (0/4), robustness 18/20 I-positive, adaptive 4/4 I-positive
- `exp03`: Straightness gate calibrated at 0.35 (was provisional 0.45) — zero I+ rules affected

**exp02 key findings:**
- Contraction mapping reaches fixed point at step 4 (10% active dynamics)
- 18/20 random seeds are I-positive — predicate is permissive without straightness gate
- Adaptive variant (grow-then-contract) fools predicate entirely (4/4 I-positive)
- Straightness discriminator: contraction ~0.61 vs known I+ rules ~0.07–0.39

**exp03 key findings:**
- Swept S* from 0.10–0.85 across 264 I+ and 42 contraction measurements
- No clean separation (overlap 0.13–0.50), but S*=0.35 is optimal safe threshold
- Zero I+ rules lost, 20/42 contraction measurements rejected

**What's next:**
- exp04: First PRIMO enumeration (single rules ≤ 2→3) — Paper 4
- exp05: Ordering test N_I^min vs N_Φ^min — Paper 4
- exp06: Temporal I-profiles of Φ-positive programs — Paper 4

**Known issues:**
- `watts_strogatz` is a boundary rule: Φ classification varies across runs (2–4 seeds)
- Straightness gate overlap: no clean separation between I+ rules and contractions
- Adaptive variant not caught by straightness gate (S ~0.13–0.17)

**Hardware:**
- Development: RTX 3050 (4GB VRAM), `DEVICE="cuda"`, `GPU_BATCH_SIZE=8`
- Target: university cluster (A100 80GB), `GPU_BATCH_SIZE=256`

## Project structure

```
primo/                     # Library (6 of 6 files built)
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
