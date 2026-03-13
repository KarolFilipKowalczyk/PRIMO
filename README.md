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

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3 (1, 1, 3 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- Repository structure and plan: `CLAUDE.md`
- `primo/backend.py`: numpy↔torch GPU abstraction — CUDA verified on RTX 3050
- `primo/monitor.py`: Tkinter experiment dashboard with progress tracking and auto-close
- `primo/run_utils.py`: StepRunner for experiment orchestration
- `primo/rules.py`: all 33 rules ported, rule enumeration with isomorphism, catalog I/O
- `primo/trajectories.py`: trajectory generation, 3 embeddings, tensor conversion, compression ratios, spectral dimension
- `primo/predicates.py`: I-predicate and Φ-predicate classification, majority voting, ER null model
- `experiments/exp01_validate.py`: full 8-diagnostic reproduction — all 33 rules match reference
- Tests: 301 passing (test_backend, test_rules, test_predicates, test_regression, test_monitor)

**exp01 results:**
- All 33 rules match reference classifications
- Threshold stability: 75.8% (above 70% target)
- ER null-model separation confirmed under all 3 embeddings
- Φ-predicate gap: 43.8% (exists)
- All four (I, Φ) cells populated: (I+,Φ+)=17, (I+,Φ-)=5, (I-,Φ+)=1, (I-,Φ-)=10

**What's in progress:**
- Phase 2: foundation experiments for Papers 1–3

**What's next:**
- exp02: Example B analysis (I-negative, 2/4 seeds) — Paper 2
- exp03: Straightness gate calibration — Paper 1
- exp04–06: PRIMO enumeration and ordering tests — Paper 4

**Known issues:**
- `watts_strogatz` is a boundary rule: Φ classification varies across runs (2–4 seeds)
- Straightness gate threshold (0.45) is provisional, pending exp03 calibration
- Example B seed sensitivity — see `reference/example_b_analysis.md`

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
