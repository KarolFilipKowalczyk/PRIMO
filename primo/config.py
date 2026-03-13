"""
PRIMO Configuration — single source of truth for all parameters.

Every threshold, seed, and tunable constant lives here.
Experiments that need different values modify a local copy of the config
dict at runtime and log the diff to logs/decisions.md.

Changes to this file MUST be accompanied by an entry in logs/decisions.md.
"""

# === Device ===
DEVICE = "cuda"              # "cuda" | "cpu"
GPU_BATCH_SIZE = 8           # trajectories per GPU batch (8 for 3050, 256 for A100)
CPU_WORKERS = 4              # multiprocessing workers for trajectory generation

# === Random state ===
MASTER_SEED = 42
EMBEDDING_PROJECTION_SEED = 0  # seed for the random projection matrix M

# === Simulation ===
T_DEFAULT = 30               # default trajectory length (steps)
N_MAX = 500                  # max graph size (safety cap + tensor padding bound)
INITIAL_GRAPHS = ['K1', 'K2', 'K3', 'P3']
MAJORITY_THRESHOLD = 3       # out of 4 initial graphs required for classification

# === Embeddings ===
EMBEDDING_DIM = 5            # d for all embeddings
EMBEDDINGS = ['laplacian', 'random_proj', 'degree_prof']

# === I-predicate thresholds ===
# τ*: convergence threshold.
#   Justification: above ER null-model ceiling (max 0.37 across 20 trials,
#   3 embeddings). Midpoint between null model (τ ≈ 0) and Bayesian optimum
#   (τ = 1, proved in Paper 2 Theorem 1).
#   See logs/decisions.md entry 2026-03-13.
TAU_STAR = 0.5

# ρ*: compression gate threshold.
#   Justification: invariant across 3 serialization methods with zero
#   inconsistencies on 33 rules. All observed ratios well below (max 0.26).
#   See diagnostic output v5, diagnostic 3.
RHO_STAR = 0.85

# Straightness gate threshold.
#   Status: PROVISIONAL — pending calibration in exp03.
#   See example_b_analysis.md: contraction mappings have straightness ~0.5-0.84,
#   genuine I-positive rules have straightness ~0.35.
STRAIGHTNESS_STAR = 0.45

# Anti-convergence guard: if ALL τ_to_final values are below this, reject.
ANTI_CONVERGENCE_THRESHOLD = -0.5

# === Φ-predicate thresholds ===
# Spectral dimension stability.
#   Justification: 39% gap exists in the ds_std distribution at 0.18.
#   ER null model: min ds_std = 0.21, mean = 0.32 (all above threshold).
#   See diagnostic output v5, diagnostics 4 and 7.
DS_STD_STAR = 0.18
DS_INT_DIST = 0.5            # max distance from nearest integer for ds_mean

# Law residual: polynomial fit residual for "lawful evolution" of aggregates.
LAW_RESIDUAL_STAR = 0.15

# Curvature homogeneity: coefficient of variation of Jaccard curvature.
CURVATURE_KAPPA_STAR = 1.0

# === Null models ===
ER_N = 10                    # ER graph size
ER_P = 0.3                   # ER edge probability
ER_TRIALS = 20               # number of ER trials for null-model calibration
NULL_MODEL_REPLICATES = 100   # degree-preserving random dynamics replicates

# === Enumeration ===
R_CONNECTED_ONLY = True       # require connected RHS in rule enumeration
SIGNATURE_R_OFFSET = 1        # r = l + offset (fixed at 1 for growth rules)

# === Data ===
CHECKPOINT_INTERVAL = 100     # save state every N rules during enumeration
DATA_DIR = "data/results"     # output directory for experiment results
