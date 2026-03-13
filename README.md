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

**Last updated:** 2026-03-13 (exp04 session)

**Phase 1: COMPLETE.** All 6 library files, all tests passing, exp01 reproduces the full 33-rule diagnostic.

**Phase 2: IN PROGRESS.** exp01–04 complete, exp05–06 next (blocked on 3→4 enumeration).

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3 (1, 1, 3 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- `primo/`: all 6 library files (backend, rules, trajectories, predicates, monitor, run_utils)
- `exp01`: 33-rule validation — all rules match reference, stability 75.8%, all four (I,Φ) cells populated (17, 5, 1, 10)
- `exp02`: Example B analysis — canonical I-negative (0/4), robustness 18/20 I-positive, adaptive 4/4 I-positive
- `exp03`: Straightness gate calibrated at 0.35 (was provisional 0.45) — zero I+ rules affected
- `exp04`: First PRIMO enumeration (5 rules, signatures ≤ 2→3) — claim (a) inconclusive (tie at σ=2), claim (b) not supported

**exp04 key findings:**
- Enumerated all connected DPO rules at signatures 1→1 (1), 1→2 (1), 2→3 (3)
- σ=1: Identity is I- Φ- (trivial, no dynamics)
- σ=2: Vertex Sprouting is I+ Φ+ (S=0.124)
- σ=3: All 3 rules are I+ Φ+ (S=0.082–0.122)
- N_I^min = N_Φ^min = σ=2 — claim (a) inconclusive (tie)
- No I-positive transient decay in Φ+ rules — claim (b) not supported
- No I-only rules, no frequency separation — S1 and S2 inconclusive
- All non-trivial growth rules are both I+ and Φ+ at this scale

**What's next:**
- Enumerate connected rules at signature 3→4 (count TBD) to get enough rules for predicate separation
- exp05: Ordering test N_I^min vs N_Φ^min — requires 3→4 data
- exp06: Temporal I-profiles of Φ-positive programs — requires 3→4 data

**Known issues:**
- `watts_strogatz` is a boundary rule: Φ classification varies across runs (2–4 seeds)
- Straightness gate overlap: no clean separation between I+ rules and contractions (overlap 0.13–0.50)
- Adaptive variant not caught by straightness gate (S ~0.13–0.17); identified mitigation: active dynamics gate requiring edit distance > 0 for at least 2T/3 steps (see `logs/decisions.md`)
- Example B seed sensitivity: 18/20 random seeds classify I-positive despite canonical protocol giving I-negative; predicate is sensitive to initial conditions (see `reference/example_b_analysis.md`)
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
