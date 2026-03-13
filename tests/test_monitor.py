"""
Tests for primo.monitor and primo.run_utils.

Includes:
1. Unit tests for message handling (no GUI)
2. A visual smoke test that opens the Tkinter window briefly
   Run with: pytest tests/test_monitor.py -v -k smoke --no-header
   Or directly: python tests/test_monitor.py
"""

import json
import os
import time
import threading
import pytest

from primo.monitor import (
    ExperimentMonitor, _MsgBeginRule, _MsgTick,
    _MsgFinishRule, _MsgFinish, _MsgPhase, _MsgLog,
)
from primo.run_utils import StepRunner


# ── Unit tests (no GUI) ──────────────────────────────────────────────

class TestMonitorState:
    """Test the monitor's state tracking without opening a window."""

    def test_message_queue(self):
        mon = ExperimentMonitor("test", total_rules=3, total_seeds=2)
        # Messages should queue without a GUI
        mon.begin_rule("rule_a")
        mon.tick("rule_a", "K1", result="I+")
        mon.tick("rule_a", "K2", result="I-")
        mon.finish_rule("rule_a", classification="(I+, Φ-)")
        assert mon._queue.qsize() == 4

    def test_stop_event(self):
        mon = ExperimentMonitor("test", total_rules=1, total_seeds=1)
        assert not mon.should_stop()
        mon._stop_event.set()
        assert mon.should_stop()

    def test_set_total(self):
        mon = ExperimentMonitor("test", total_rules=10, total_seeds=4)
        assert mon._total_ticks == 40
        mon.total_rules = 20
        mon.total_seeds = 4
        mon._total_ticks = 20 * 4
        assert mon._total_ticks == 80

    def test_fmt_time(self):
        assert ExperimentMonitor._fmt_time(5) == "5s"
        assert ExperimentMonitor._fmt_time(65) == "1m 5s"
        assert ExperimentMonitor._fmt_time(3665) == "1h 1m 5s"


class TestStepRunnerCheckpoint:
    """Test checkpoint save/load without GUI."""

    def test_save_and_load(self, tmp_path):
        runner = StepRunner("test_exp", total_rules=2, total_seeds=2,
                            checkpoint_dir=str(tmp_path))
        # Fake monitor to avoid Tkinter
        runner._monitor = ExperimentMonitor("test", total_rules=2, total_seeds=2)

        runner._results = {
            "rule_a": {"I": True, "Phi": False},
            "rule_b": {"I": False, "Phi": True},
        }
        runner.save_checkpoint()

        # Check file exists
        cp_path = tmp_path / "checkpoint.json"
        assert cp_path.exists()

        # Load and verify
        with open(cp_path) as f:
            data = json.load(f)
        assert "rule_a" in data["results"]
        assert "rule_b" in data["results"]
        assert data["experiment"] == "test_exp"

    def test_is_rule_done(self, tmp_path):
        runner = StepRunner("test", checkpoint_dir=str(tmp_path))
        runner._monitor = ExperimentMonitor("test")
        runner._results = {"rule_a": {}}
        assert runner.is_rule_done("rule_a")
        assert not runner.is_rule_done("rule_b")


# ── Visual smoke test ────────────────────────────────────────────────

FAKE_RULES = [
    "preferential_attach",
    "subdivision",
    "triangle_closure",
    "grid_growth",
    "line_growth",
    "star_growth",
    "cycle_then_fill",
    "er_random",
]
FAKE_SEEDS = ["K1", "K2", "K3", "P3"]
FAKE_CLASSES = [
    "(I+, Φ+)", "(I+, Φ-)", "(I-, Φ+)", "(I-, Φ-)",
    "(I+, Φ+)", "(I+, Φ-)", "(I-, Φ+)", "(I-, Φ-)",
]


def _simulated_experiment(mon):
    """Simulate an experiment with fake rules — for visual testing."""
    import random
    random.seed(42)

    mon.set_phase("Phase 1: Generating trajectories")
    time.sleep(0.3)

    mon.set_phase("Phase 2: Classifying rules")
    for i, rule in enumerate(FAKE_RULES):
        if mon.should_stop():
            mon.log("Stopped by user.")
            return

        mon.begin_rule(rule)
        for seed in FAKE_SEEDS:
            time.sleep(0.08)  # simulate work
            result = random.choice(["I+", "I-", ""])
            mon.tick(rule, seed, result=result)

        mon.finish_rule(rule, classification=FAKE_CLASSES[i])

    mon.set_phase("Phase 3: Diagnostics")
    time.sleep(0.3)
    mon.log("All diagnostics passed.")

    mon.finish(f"All {len(FAKE_RULES)} rules classified successfully.")


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Tkinter not available in CI",
)
class TestMonitorSmoke:
    """Visual smoke test — opens a real Tkinter window.

    Run with: pytest tests/test_monitor.py -v -k smoke
    Or directly: python tests/test_monitor.py
    """

    def test_smoke_monitor_direct(self):
        """Open the monitor with a simulated experiment.

        Auto-closes after completion.
        """
        mon = ExperimentMonitor(
            "smoke_test",
            total_rules=len(FAKE_RULES),
            total_seeds=len(FAKE_SEEDS),
            phases=["Generating trajectories", "Classifying rules", "Diagnostics"],
            auto_close_delay=1.0,
        )
        mon.run(_simulated_experiment)

    def test_smoke_step_runner(self):
        """Test StepRunner with the monitor (full integration)."""

        def experiment(runner):
            with runner.phase("Classifying rules"):
                for i, rule in enumerate(FAKE_RULES[:4]):
                    if runner.should_stop():
                        return
                    runner.begin_rule(rule)
                    for seed in FAKE_SEEDS:
                        time.sleep(0.05)
                        runner.tick(rule, seed, result="ok")
                    runner.finish_rule(rule, classification=FAKE_CLASSES[i],
                                       result_data={"I": True})
            runner.finish("Smoke test passed.")

        runner = StepRunner("smoke_runner", total_rules=4, total_seeds=4)
        runner.run(experiment)


# ── Direct execution: visual demo ────────────────────────────────────

if __name__ == "__main__":
    print("Opening PRIMO monitor smoke test...")
    print("The window will auto-close after the simulation completes.")
    print()

    mon = ExperimentMonitor(
        "smoke_test_demo",
        total_rules=len(FAKE_RULES),
        total_seeds=len(FAKE_SEEDS),
        auto_close_delay=2.0,
    )

    mon.run(_simulated_experiment)
    print("Done.")
