# CLAUDE.md — PRIMO Repository Master Plan

This file is the primary instruction document for Claude Code agents working on the PRIMO project. Read this first, every session.

---

## What this project is

PRIMO (Primacy of Inference over Physics in the Space of Minimal Programs) is a research project producing four papers:

1. **Paper 1 — Predicates.** Defines the I-predicate and Φ-predicate, proves independence, validates on 33 rules.
2. **Paper 2 — Bayesian.** Proves that Bayesian posterior updating implies I-positivity (forward theorem), with partial converse.
3. **Paper 3 — Hierarchy.** Establishes a strict computational hierarchy for graph rewrite systems by signature complexity.
4. **Paper 4 — PRIMO Conjecture.** The crown jewel. Uses Papers 1–3 as infrastructure to test whether inference-like behavior appears at shorter program lengths than physics-like behavior.

Papers 1–3 are complete as markdown drafts. Paper 4 depends on experimental results from the PRIMO enumeration.

---

## How this repo is developed

**You are a Claude Code agent. Karol supervises you.** There is no second human developer. The repo is designed so that:

- You orient yourself by reading this file and `README.md`
- You understand parameters by reading `primo/config.py`
- You understand any subsystem by reading one library file (~500 lines)
- Experiments run with a mandatory Tkinter monitor so Karol can watch and stop them
- Between sessions, state is captured in `status.json` files and `README.md`

**Your first action every session:** Read `README.md` for current project state.

---

## Repository structure

```
primo/
├── CLAUDE.md                    # THIS FILE — master plan, read first
├── README.md                    # Current state, updated after every session
├── Makefile
├── requirements.txt
├── primo/                       # Library (6 files)
│   ├── config.py                # All parameters
│   ├── rules.py                 # Rules, catalog, enumeration
│   ├── trajectories.py          # Generation, tensor conversion, checkpointing
│   ├── predicates.py            # Embeddings, I/Φ predicates, null models
│   ├── backend.py               # numpy↔torch device abstraction
│   └── monitor.py               # Tkinter dashboard (mandatory)
├── experiments/                 # One script per experiment
├── tests/                       # Unit + regression
├── papers/                      # Markdown → LaTeX at Phase 4
├── data/                        # Catalogs (committed) + results (gitignored)
├── logs/                        # Decision log, experiment log
└── reference/                   # Historical oracle files
```

---

## The plan: five phases

### ══════════════════════════════════════════════════════
### PHASE 1: Bootstrap (upload initial files, build the engine)
### ══════════════════════════════════════════════════════

**Goal:** Get the repo to the point where `make test` passes and `make exp01` reproduces the 33-rule diagnostic.

**Upload to GitHub (initial commit):**

```
README.md                        # Initial project description + setup instructions
CLAUDE.md                        # This file
Makefile                         # Targets: test, exp01..exp06, reproduce-paper-1..3
requirements.txt                 # numpy, scipy, networkx, matplotlib, tqdm, torch (optional)
pyproject.toml                   # Package metadata
.gitignore                       # data/results/, *.db, __pycache__, .env

primo/__init__.py
primo/config.py                  # All thresholds, device settings, seeds

papers/paper1_predicates.md      # From project: paper1_final.md
papers/paper2_bayesian.md        # From project: paper2_final.md
papers/paper3_hierarchy.md       # From project: paper3_final.md
papers/primo_conjecture.md       # From project: primo.md
papers/notation.md               # Shared notation glossary (extracted from papers)

data/catalogs/rules_1_1.json     # From rule_catalog.md: 1 rule
data/catalogs/rules_1_2.json     # 1 rule
data/catalogs/rules_2_3.json     # 3 rules (connected)

logs/decisions.md                # Seed with existing decisions (thresholds, retractions)

reference/primo_diagnostic_v5.py           # Test oracle
reference/primo_diagnostic_output_v5.txt   # Regression target
```

**Then build, in this order:**

| Step | Task | Depends on | Deliverable |
|------|------|------------|-------------|
| 1.1 | Write `primo/backend.py` | nothing | numpy↔torch abstraction, ~150 lines |
| 1.2 | Write `primo/rules.py` | `config.py` | Port 33 rules from `primo_diagnostic_v5.py`, add enumeration from `enumerate_rules.py`, add catalog I/O. ~500 lines |
| 1.3 | Write `primo/trajectories.py` | `rules.py`, `backend.py` | Trajectory generation, tensor conversion, checkpointing. ~300 lines |
| 1.4 | Write `primo/predicates.py` | `trajectories.py`, `backend.py` | Port embeddings + I/Φ predicates from `primo_diagnostic_v5.py`, add batched GPU paths. ~500 lines |
| 1.5 | Write `primo/monitor.py` | nothing | Tkinter dashboard + JSON progress. ~300 lines |
| 1.6 | Write `experiments/run_utils.py` | `monitor.py` | StepRunner, monitor integration. ~150 lines |
| 1.7 | Write `tests/test_rules.py` | `rules.py` | Enumeration counts: 1, 1, 3 (connected). Isomorphism checks. |
| 1.8 | Write `tests/test_predicates.py` | `predicates.py` | Hand-crafted trajectories with known I/Φ classification. |
| 1.9 | Write `tests/test_backend.py` | `backend.py` | numpy vs torch produce identical results (within float tolerance). |
| 1.10 | Write `tests/test_regression.py` | all `primo/` | All 33 rules match `reference/primo_diagnostic_output_v5.txt`. |
| 1.11 | Write `experiments/exp01_validate.py` | all above | Reproduces the full 33-rule diagnostic with Tkinter monitor. |

**Phase 1 exit criterion:** `make test` passes. `make exp01` opens a Tkinter window, runs the 33-rule validation, and all classifications match the v5 reference output.

---

### ══════════════════════════════════════════════════════
### PHASE 2: Foundation experiments (Papers 1–3 validation)
### ══════════════════════════════════════════════════════

**Goal:** Run the experiments that validate the claims of Papers 1–3. Generate all figures and tables needed for submission.

**Experiment sequence (order matters):**

| Exp | What it tests | Paper | Depends on | Key output |
|-----|--------------|-------|------------|------------|
| exp01 | 33-rule classification match | Paper 1 | Phase 1 | classifications.csv, 2×2 table |
| exp02 | Example B analysis (I-negative, 2/4 seeds) | Paper 2 | exp01 | Example B classification, fixed-point analysis |
| exp03 | Straightness gate calibration | Paper 1 | exp01 | Calibrated `STRAIGHTNESS_STAR` value in config.py |
| exp04 | First PRIMO enumeration (single rules ≤ 2→3) | Paper 4 | exp03 | N_I^min, N_Φ^min at signature level |
| exp05 | Ordering test: N_I^min vs N_Φ^min | Paper 4 | exp04 | Statistical test result |
| exp06 | Temporal I-profiles of Φ-positive programs | Paper 4 | exp04 | Transient analysis for claim (b) |

**Critical dependency:** exp03 (straightness) must complete before exp04 (enumeration). The straightness gate changes I-predicate classifications, and the enumeration must use the final predicate.

**After each experiment:**
1. Update `logs/experiments.md` with git hash, config, duration, result
2. Update `README.md` current state section
3. If a threshold changes, update `primo/config.py` AND `logs/decisions.md`

**Figures to generate (stored in `data/results/figures/`):**

For Paper 1:
- 2×2 classification table (all 33 rules)
- Threshold sensitivity heatmap (τ* × ρ* sweep)
- ER null-model separation per embedding
- Φ-predicate gap analysis (spectral dim std distribution)
- Compression gate invariance across serializations

For Paper 2:
- Example B convergence profile (τ vs t for all seeds)
- Forward theorem illustration: Bayesian system trajectory on Grassmannian
- Spectral gap obstruction example

For Paper 3:
- Signature hierarchy diagram (levels 1–8)
- Binary incrementer walkthrough (step-by-step)
- TM encoding diagram

For Paper 4 (PRIMO):
- (I, Φ) phase diagram by signature level
- N_I^min vs N_Φ^min bar chart
- Temporal I-profiles of Φ-positive programs
- Perturbation-response curves

**Phase 2 exit criterion:** All experiments pass. All figures generated. Paper drafts updated with final numbers (still in markdown).

---

### ══════════════════════════════════════════════════════
### PHASE 3: Scale-up (cluster experiments for PRIMO)
### ══════════════════════════════════════════════════════

**Goal:** Extend the PRIMO enumeration to higher signatures (3→4, potentially 4→5) on a university cluster. This is where the GPU architecture pays off.

**What changes from Phase 2:**
- `config.py`: `DEVICE = "cuda"`, `GPU_BATCH_SIZE = 256`, `N_MAX = 2000`
- Enumerate connected rules at signature 3→4 (count TBD — this is the first thing to compute)
- Possibly enumerate at 4→5 if 3→4 is tractable
- Run exp04/05/06 at the larger scale

**Before going to cluster:**
- Verify `test_backend.py` passes on cluster hardware
- Benchmark: measure wall-clock time per rule on cluster GPU vs 3050
- Estimate total compute budget for the enumeration scope

**New experiments at this phase:**
- exp07: Rule-set enumeration (ordered tuples of rules at a single signature level)
- exp08: Secondary hypothesis S2 — frequency dominance I(N) > Φ(N)
- exp09: Perturbation-response test for claim (b)

**Phase 3 exit criterion:** PRIMO claims (a) and (b) are either confirmed or refuted at a statistically meaningful scale. All data for Paper 4 is in hand.

---

### ══════════════════════════════════════════════════════
### PHASE 4: LaTeX conversion and unified presentation
### ══════════════════════════════════════════════════════

**Goal:** Convert all four papers from markdown to LaTeX with unified style, bibliography, and illustration conventions. This happens ONCE, after all experimental results are final.

**Why not earlier:** LaTeX during development creates maintenance burden. Agents edit markdown easily. LaTeX cross-file dependencies (.sty, .bib) are fragile. Converting once at the end with all numbers final is cleaner than maintaining parallel formats.

**The conversion task:**

| Step | Task | Deliverable |
|------|------|-------------|
| 4.1 | Create `papers/shared/primo.bib` | Unified bibliography from all four papers' reference sections |
| 4.2 | Create `papers/shared/primo-macros.sty` | From `papers/notation.md`: `\Ipos`, `\Phineg`, `\GPI`, `\DPO`, etc. |
| 4.3 | Create `papers/shared/primo-theorems.sty` | Shared theorem environments |
| 4.4 | Create `papers/shared/primo-style.sty` | Typography, colors, figure style (venue-agnostic base) |
| 4.5 | Convert `paper1_predicates.md` → `papers/predicates/predicates.tex` | Full LaTeX with all figures embedded |
| 4.6 | Create `papers/predicates/predicates-entropy.tex` | Entropy venue wrapper (imports predicates.tex sections) |
| 4.7 | Convert `paper2_bayesian.md` → `papers/bayesian/bayesian.tex` | Full LaTeX |
| 4.8 | Convert `paper3_hierarchy.md` → `papers/hierarchy/hierarchy.tex` | Full LaTeX |
| 4.9 | Convert `primo_conjecture.md` → `papers/primo/primo.tex` | Full LaTeX, references Papers 1–3 |
| 4.10 | Per-paper Makefiles | `make pdf` in each paper directory |
| 4.11 | Verify: `make reproduce-paper-1` regenerates all figures and compiles PDF | End-to-end test |

**Unified style conventions (defined in Phase 4, applied to all papers):**

Illustrations:
- All plots use the same matplotlib style file (`primo/viz_style.mplstyle`)
- Color palette: a fixed 6-color qualitative palette, same across all papers
- Font: matching the LaTeX document font (Computer Modern or the venue's font)
- Figure sizes: consistent widths (single-column, double-column) across papers
- Phase diagrams: I-axis horizontal, Φ-axis vertical, consistent marker shapes per classification cell

Data presentation:
- Classification tables: same layout in all papers (rule name, I seeds, Φ seeds, cell)
- Score distributions: violin plots with null-model baseline marked
- Threshold sensitivity: heatmaps with the same color scale
- Temporal profiles: time on x-axis, I-score on primary y-axis, Φ-score on secondary

Typography:
- Notation from `primo-macros.sty` used consistently (never raw `I\text{-positive}`, always `\Ipos`)
- Theorem numbering: per-section within each paper
- Cross-references between papers: by title and result number, not by internal label

**After conversion, the papers/ directory looks like:**

```
papers/
├── shared/
│   ├── primo.bib
│   ├── primo-macros.sty
│   ├── primo-theorems.sty
│   └── primo-style.sty
├── predicates/
│   ├── predicates.tex
│   ├── predicates-entropy.tex      # venue wrapper
│   ├── figures/
│   ├── supplementary/
│   └── Makefile
├── bayesian/
│   ├── bayesian.tex
│   ├── figures/
│   ├── supplementary/
│   └── Makefile
├── hierarchy/
│   ├── hierarchy.tex
│   ├── figures/
│   ├── supplementary/
│   └── Makefile
└── primo/
    ├── primo.tex
    ├── figures/
    ├── supplementary/
    └── Makefile
```

**Phase 4 exit criterion:** All four papers compile to PDF. `make reproduce-paper-X` regenerates figures from data and compiles for each paper. The PDFs look professional and visually unified.

---

### ══════════════════════════════════════════════════════
### PHASE 5: Submission
### ══════════════════════════════════════════════════════

**Goal:** Submit papers to target venues with all supplementary materials.

**Submission order:**

1. **Paper 1 (Predicates) → Entropy / Complex Systems.** Submitted first because it establishes the measurement tools that Papers 2–4 depend on. Self-contained: defines predicates, proves independence, validates computationally.

2. **Paper 3 (Hierarchy) → Theoretical Computer Science / Fundamenta Informaticae.** Submitted second (or simultaneously with Paper 1). Self-contained: pure graph rewriting theory, no dependency on the predicates.

3. **Paper 2 (Bayesian) → Journal of Machine Learning Research / Mathematical Structures in Computer Science.** Depends on Paper 1's predicate definitions. Can reference Paper 1 as "companion paper" or include the definitions self-containedly.

4. **Paper 4 (PRIMO Conjecture) → Last.** References all three preceding papers. Submitted after at least Papers 1 and 3 are accepted or posted as preprints.

**For each submission:**
- Venue-specific `.tex` wrapper (font, margins, bibliography style)
- Supplementary materials: code archive (the `primo/` package + relevant experiment script)
- Data availability statement pointing to the GitHub repository
- Reproducibility checklist if required by venue

**Phase 5 exit criterion:** All four papers submitted. GitHub repo is public with a release tag matching each submission.

---

## Dependency graph

```
Phase 1 (Bootstrap)
  │
  ├── 1.1 backend.py
  ├── 1.2 rules.py ─────────────────────────────────┐
  ├── 1.3 trajectories.py                            │
  ├── 1.4 predicates.py                              │
  ├── 1.5 monitor.py                                 │
  ├── 1.6 run_utils.py                               │
  ├── 1.7-1.10 tests                                 │
  └── 1.11 exp01 (33-rule validation)                │
        │                                            │
Phase 2 (Foundation experiments)                     │
  │                                                  │
  ├── exp01 ✓ (from Phase 1)                         │
  ├── exp02 (Example B) ── figures for Paper 2       │
  ├── exp03 (straightness) ── updates config.py      │
  │     │                                            │
  │     ▼                                            │
  ├── exp04 (PRIMO enumeration, small) ──────────────┤
  │     │                                            │
  │     ├── exp05 (ordering test) ── figures for P4  │
  │     └── exp06 (temporal profiles) ── figures P4  │
  │                                                  │
  │   ► All Paper 1-3 figures generated              │
  │   ► Paper markdowns updated with final numbers   │
  │                                                  │
Phase 3 (Scale-up) ─── only if 3050 results warrant  │
  │                                                  │
  ├── exp07 (rule-set enumeration)                   │
  ├── exp08 (frequency dominance)                    │
  └── exp09 (perturbation-response)                  │
        │                                            │
Phase 4 (LaTeX conversion)                           │
  │                                                  │
  ├── 4.1-4.4 shared infrastructure ◄───────────────┘
  ├── 4.5-4.6 Paper 1 LaTeX + venue wrapper
  ├── 4.7 Paper 2 LaTeX
  ├── 4.8 Paper 3 LaTeX
  ├── 4.9 Paper 4 LaTeX
  └── 4.10-4.11 Makefiles + reproduce targets
        │
Phase 5 (Submission)
  │
  ├── Paper 1 → Entropy
  ├── Paper 3 → TCS (parallel with Paper 1)
  ├── Paper 2 → JMLR (after Paper 1)
  └── Paper 4 → (after Papers 1+3 accepted/posted)
```

---

## Agent work instructions

### Starting a session

1. `cat README.md` — understand current state
2. Check which phase we're in
3. Check if any experiment is paused (look for `checkpoint.json` files in `data/results/`)
4. Ask Karol what to work on, or continue the next task in sequence

### During a session

- **Never run an experiment without the Tkinter monitor.** Import and use `ExperimentMonitor` from `primo/monitor.py`. Every experiment script must launch the monitor window.
- If writing new code, run `make test` before committing
- If a threshold changes, update BOTH `primo/config.py` AND `logs/decisions.md`
- If an experiment completes, update `logs/experiments.md`

### Ending a session

1. Update `README.md` with what was done and what's next
2. If an experiment is mid-run, note the checkpoint path
3. Commit with a descriptive message: `"exp03: straightness gate calibrated at 0.45"` not `"update files"`

### File size discipline

- Library files (`primo/*.py`): target 300–500 lines each. Split at ~800 lines.
- Experiment scripts: target 100–200 lines each. All config comes from `primo/config.py`.
- Tests: as long as needed, but one file per subsystem.

### What NOT to do

- Don't create files outside the structure defined above without discussing with Karol
- Don't modify `reference/` files — they are the test oracle
- Don't run experiments headless — the Tkinter monitor is mandatory
- Don't change thresholds in `config.py` without logging the decision
- Don't start Phase 4 (LaTeX) until all experimental results are final
- Don't install packages not in `requirements.txt` without adding them

---

## Key thresholds (current values — see `primo/config.py` for authoritative source)

| Parameter | Value | Set by | Status |
|-----------|-------|--------|--------|
| τ* (convergence) | 0.5 | ER null-model ceiling + Bayesian optimum | Active |
| ρ* (compression gate) | 0.85 | Invariant across serializations | Active |
| Straightness gate | 0.45 | Pending exp03 calibration | Provisional |
| ds_std (spectral dim) | 0.18 | 39% gap in score distribution | Active |
| Law residual | 0.15 | Polynomial fit residual threshold | Active |
| Curvature CV | 1.0 | Coefficient of variation threshold | Active |
| N_MAX (graph size cap) | 500 | Memory budget on 3050 | Active |
| GPU_BATCH_SIZE | 8 | 3050 4GB VRAM constraint | Active (3050) |

---

## Venue targets

| Paper | Primary venue | Backup venue | Style |
|-------|-------------|--------------|-------|
| Paper 1 (Predicates) | Entropy (MDPI) | Complex Systems | MDPI template |
| Paper 2 (Bayesian) | JMLR | Math Structures in CS | JMLR template |
| Paper 3 (Hierarchy) | Theoretical CS (Elsevier) | Fundamenta Informaticae | Elsevier template |
| Paper 4 (PRIMO) | TBD — depends on results | — | venue-agnostic first |
