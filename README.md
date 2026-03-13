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

**Last updated:** 2026-03-13 (exp04 extended to 3→4)

**Phase 1: COMPLETE.** All 6 library files, all tests passing, exp01 reproduces the full 33-rule diagnostic.

**Phase 2: IN PROGRESS.** exp01–04 complete (including 3→4 enumeration), exp05–06 next.

**What's done:**
- Papers 1–3: complete markdown drafts in `papers/`
- Rule catalog: verified at signatures 1→1, 1→2, 2→3, 3→4 (1, 1, 3, 11 connected rules)
- 33-rule diagnostic: classifications established in `reference/primo_diagnostic_output_v5.txt`
- PRIMO conjecture: draft v6 in `papers/primo_conjecture.md`
- `primo/`: all 6 library files (backend, rules, trajectories, predicates, monitor, run_utils)
- `exp01`: 33-rule validation — all rules match reference, stability 75.8%, all four (I,Φ) cells populated (17, 5, 1, 10)
- `exp02`: Example B analysis — canonical I-negative (0/4), robustness 18/20 I-positive, adaptive 4/4 I-positive
- `exp03`: Straightness gate calibrated at 0.35 (was provisional 0.45) — zero I+ rules affected
- `exp04`: PRIMO enumeration (16 rules, signatures ≤ 3→4) — claim (a) inconclusive (tie at σ=2), claim (b) partial (4/15)

**exp04 key findings (extended to 3→4):**
- Enumerated all connected DPO rules at signatures 1→1 (1), 1→2 (1), 2→3 (3), 3→4 (11) — 16 total
- σ=1: Identity is I- Φ- (trivial, no dynamics)
- σ=2: Vertex Sprouting is I+ Φ+ (S=0.124)
- σ=3: All 3 rules are I+ Φ+ (S=0.082–0.122)
- σ=4: All 11 rules are I+ Φ+ (S=0.050–0.123)
- N_I^min = N_Φ^min = σ=2 — claim (a) still inconclusive (tie)
- Claim (b) PARTIAL: 4/15 Φ+ rules show I-positive transient decay (all at σ=4)
  - Path-4 partial preserve: early-late delta +0.15 to +0.27 across seeds
  - Path-4 fresh middle: mixed, delta +0.28 on K3
  - Triangle + pendant (preserved): delta +0.10 to +0.16 on K2, P3
  - Triangle + pendant (shifted): delta +0.16 to +0.50 on K2, K3, P3
- No I-only rules at any level — S1 not supported
- I+ = Φ+ at every signature level — S2 inconclusive (tied)
- All non-trivial growth rules are I+ Φ+ through σ=4 — predicates do not separate at this scale

**What's next:**
- exp05: Ordering test N_I^min vs N_Φ^min — may need σ=5 (4→5) for separation
- exp06: Temporal I-profiles — 4 rules with transient decay provide first evidence for claim (b)
- Consider whether the I-predicate is too permissive for enumerated DPO rules (all growth rules pass)

**Known issues:**
- `watts_strogatz` is a boundary rule: Φ classification varies across runs (2–4 seeds)
- Straightness gate overlap: no clean separation between I+ rules and contractions (overlap 0.13–0.50)
- Adaptive variant not caught by straightness gate (S ~0.13–0.17); identified mitigation: active dynamics gate requiring edit distance > 0 for at least 2T/3 steps (see `logs/decisions.md`)
- Example B seed sensitivity: 18/20 random seeds classify I-positive despite canonical protocol giving I-negative; predicate is sensitive to initial conditions (see `reference/example_b_analysis.md`)
- All enumerated growth rules (σ≥2) are I+ Φ+ — the I/Φ predicates may lack discriminating power for small DPO rules. The 33-rule set includes non-DPO rules that do separate.
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
