"""
PRIMO Run Utilities — StepRunner + monitor integration.

Provides a StepRunner that manages the experiment lifecycle:
  1. Set up the monitor
  2. Run named phases with progress tracking
  3. Handle checkpointing and resumption
  4. Enforce the "never headless" rule

Usage:
    from primo.run_utils import StepRunner

    def my_experiment(runner):
        with runner.phase("Generating trajectories"):
            for rule in rules:
                runner.begin_rule(rule)
                for seed in seeds:
                    traj = generate(rule, seed)
                    runner.tick(rule, seed)
                runner.finish_rule(rule, classification="(I+, Φ-)")

    runner = StepRunner("exp01", total_rules=33, total_seeds=4)
    runner.run(my_experiment)
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from primo.config import DATA_DIR, CHECKPOINT_INTERVAL
from primo.monitor import ExperimentMonitor


class StepRunner:
    """Orchestrates an experiment with monitor integration and checkpointing."""

    def __init__(self, experiment_name, total_rules=0, total_seeds=4,
                 phases=None, checkpoint_dir=None, auto_close=True,
                 auto_close_delay=2.0):
        self.experiment_name = experiment_name
        self.total_rules = total_rules
        self.total_seeds = total_seeds
        self.phases = phases or []
        self.auto_close = auto_close
        self.auto_close_delay = auto_close_delay

        self._checkpoint_dir = checkpoint_dir or os.path.join(
            DATA_DIR, experiment_name
        )
        self._checkpoint_path = os.path.join(
            self._checkpoint_dir, "checkpoint.json"
        )
        self._monitor = None
        self._results = {}
        self._tick_count = 0

    # ── Public API ───────────────────────────────────────────────────

    def run(self, experiment_fn):
        """Launch the experiment with a Tkinter monitor.

        experiment_fn receives the StepRunner as its argument.
        The monitor window opens immediately; experiment_fn runs
        in a worker thread.
        """
        mon = ExperimentMonitor(
            self.experiment_name,
            total_rules=self.total_rules,
            total_seeds=self.total_seeds,
            phases=self.phases,
            auto_close=self.auto_close,
            auto_close_delay=self.auto_close_delay,
        )
        self._monitor = mon

        def _wrapped(m):
            experiment_fn(self)

        mon.run(_wrapped)

    @contextmanager
    def phase(self, phase_name):
        """Context manager for a named experiment phase."""
        self._monitor.set_phase(phase_name)
        yield
        self._monitor.log(f"  ✓ {phase_name}")

    def begin_rule(self, rule_name):
        """Signal that processing has started for a rule."""
        self._monitor.begin_rule(rule_name)

    def tick(self, rule_name, seed_name, result=""):
        """Signal completion of one (rule, seed) unit of work."""
        self._monitor.tick(rule_name, seed_name, result)
        self._tick_count += 1

        # Periodic checkpoint
        if self._tick_count % (CHECKPOINT_INTERVAL * self.total_seeds) == 0:
            self.save_checkpoint()

    def finish_rule(self, rule_name, classification="", result_data=None):
        """Signal that a rule is fully classified.

        result_data: optional dict of per-rule results to include in checkpoint.
        """
        self._monitor.finish_rule(rule_name, classification)
        if result_data is not None:
            self._results[rule_name] = result_data

    def log(self, text):
        """Add a line to the monitor log."""
        self._monitor.log(text)

    def finish(self, summary=""):
        """Signal experiment completion."""
        self.save_checkpoint()
        self._monitor.finish(summary)

    def should_stop(self):
        """Check if the user pressed Stop."""
        return self._monitor.should_stop()

    def set_total(self, total_rules, total_seeds=None):
        """Update totals mid-experiment."""
        self.total_rules = total_rules
        if total_seeds is not None:
            self.total_seeds = total_seeds
        self._monitor.set_total(total_rules, total_seeds)

    # ── Checkpointing ────────────────────────────────────────────────

    def save_checkpoint(self):
        """Save current results to a checkpoint file."""
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        state = {
            "experiment": self.experiment_name,
            "timestamp": time.time(),
            "rules_done": list(self._results.keys()),
            "results": self._results,
        }
        tmp_path = self._checkpoint_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self._checkpoint_path)

    def load_checkpoint(self):
        """Load results from a previous checkpoint. Returns dict or None."""
        if not os.path.exists(self._checkpoint_path):
            return None
        with open(self._checkpoint_path) as f:
            state = json.load(f)
        self._results = state.get("results", {})
        self._monitor.log(
            f"Resumed from checkpoint: {len(self._results)} rules done"
        )
        return state

    def is_rule_done(self, rule_name):
        """Check if a rule was already completed in a prior checkpoint."""
        return rule_name in self._results
