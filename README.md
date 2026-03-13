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

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3 (1, 1, 3 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- Repository structure and plan: `CLAUDE.md`
- `primo/backend.py`: numpy↔torch GPU abstraction — CUDA verified on RTX 3050, 59 tests passing
- `primo/monitor.py`: Tkinter experiment dashboard with progress tracking, auto-close, checkpointing
- `primo/run_utils.py`: StepRunner for experiment orchestration

**What's in progress:**
- Phase 1 bootstrap: building the `primo/` engine from the reference diagnostic

**What's next:**
- Build `primo/rules.py` (port 33 rules + enumeration from reference)
- Build `primo/trajectories.py`, `primo/predicates.py`
- Write tests (`test_rules.py`, `test_predicates.py`, `test_regression.py`)
- Write `experiments/exp01_validate.py`

**Known issues:**
- Threshold stability at 51.5% (below 70% target) — see `reference/primo_diagnostic_output_v5.txt` diagnostic 5
- Example B seed sensitivity — see `reference/example_b_analysis.md`
- Straightness gate threshold (0.45) is provisional, pending exp03 calibration

**Hardware:**
- Development: RTX 3050 (4GB VRAM), `DEVICE="cuda"`, `GPU_BATCH_SIZE=8`
- Target: university cluster (A100 80GB), `GPU_BATCH_SIZE=256`

## Project structure

```
primo/                     # Library (3 of 6 files built: backend, monitor, run_utils)
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
