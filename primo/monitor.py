"""
PRIMO Experiment Monitor — Tkinter dashboard (mandatory for all experiments).

Provides a live GUI window showing experiment progress, per-rule status,
throughput stats, and a stop button. Also writes JSON progress files for
programmatic monitoring.

Usage:
    from primo.monitor import ExperimentMonitor

    with ExperimentMonitor("exp01", total_rules=33, total_seeds=4) as mon:
        for rule in rules:
            mon.begin_rule(rule)
            for seed in seeds:
                # ... do work ...
                mon.tick(rule, seed, result="I+Φ-")
            mon.finish_rule(rule, classification="(I+, Φ-)")
        mon.finish("All 33 rules classified.")

The monitor runs Tkinter on the main thread. Experiment logic runs in a
worker thread, communicating via a thread-safe queue.
"""

import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from primo.config import DATA_DIR


# ── Message types for thread-safe communication ──────────────────────

class _Msg:
    """Base message from worker → GUI."""
    pass

class _MsgInit(_Msg):
    def __init__(self, experiment, total_rules, total_seeds, phases=None):
        self.experiment = experiment
        self.total_rules = total_rules
        self.total_seeds = total_seeds
        self.phases = phases or []

class _MsgPhase(_Msg):
    def __init__(self, phase_name):
        self.phase_name = phase_name

class _MsgBeginRule(_Msg):
    def __init__(self, rule_name):
        self.rule_name = rule_name

class _MsgTick(_Msg):
    def __init__(self, rule_name, seed_name, result=""):
        self.rule_name = rule_name
        self.seed_name = seed_name
        self.result = result

class _MsgFinishRule(_Msg):
    def __init__(self, rule_name, classification=""):
        self.rule_name = rule_name
        self.classification = classification

class _MsgLog(_Msg):
    def __init__(self, text):
        self.text = text

class _MsgFinish(_Msg):
    def __init__(self, summary=""):
        self.summary = summary

class _MsgError(_Msg):
    def __init__(self, error_text):
        self.error_text = error_text


# ── Monitor GUI ──────────────────────────────────────────────────────

class ExperimentMonitor:
    """Tkinter experiment dashboard.

    Use as a context manager. The experiment function runs in a worker
    thread; Tkinter runs on the main thread.
    """

    # How often (ms) the GUI polls the message queue
    POLL_INTERVAL = 50

    def __init__(self, experiment_name, total_rules=0, total_seeds=4,
                 phases=None, json_path=None, auto_close=True,
                 auto_close_delay=2.0):
        self.experiment_name = experiment_name
        self.total_rules = total_rules
        self.total_seeds = total_seeds
        self.phases = phases or []
        self.auto_close = auto_close
        self.auto_close_delay = auto_close_delay
        self.json_path = json_path or os.path.join(
            DATA_DIR, f"{experiment_name}_progress.json"
        )

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._worker_fn = None
        self._worker_exception = None

        # State tracked for the GUI
        self._start_time = None
        self._rules_done = 0
        self._ticks_done = 0
        self._total_ticks = total_rules * total_seeds
        self._current_phase = ""
        self._rule_status = {}   # rule_name → {"ticks": 0, "class": "", "state": ""}
        self._log_lines = []
        self._finished = False

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._queue.put(_MsgError(f"{exc_type.__name__}: {exc_val}"))
        return False  # don't suppress exceptions

    # ── Public API (called from worker thread) ───────────────────────

    def begin_rule(self, rule_name):
        """Signal that processing has started for a rule."""
        self._queue.put(_MsgBeginRule(rule_name))

    def tick(self, rule_name, seed_name, result=""):
        """Signal completion of one (rule, seed) unit of work."""
        self._queue.put(_MsgTick(rule_name, seed_name, result))

    def finish_rule(self, rule_name, classification=""):
        """Signal that a rule is fully classified."""
        self._queue.put(_MsgFinishRule(rule_name, classification))

    def set_phase(self, phase_name):
        """Signal a phase transition (e.g., 'Generating trajectories')."""
        self._queue.put(_MsgPhase(phase_name))

    def log(self, text):
        """Add a line to the log panel."""
        self._queue.put(_MsgLog(text))

    def finish(self, summary=""):
        """Signal that the experiment is complete."""
        self._queue.put(_MsgFinish(summary))

    def should_stop(self):
        """Check if the user pressed Stop. Call this in your loop."""
        return self._stop_event.is_set()

    def set_total(self, total_rules, total_seeds=None):
        """Update totals mid-experiment (e.g., after enumeration)."""
        self.total_rules = total_rules
        if total_seeds is not None:
            self.total_seeds = total_seeds
        self._total_ticks = total_rules * self.total_seeds
        self._queue.put(_MsgInit(
            self.experiment_name, total_rules,
            total_seeds or self.total_seeds, self.phases,
        ))

    # ── Run: main thread gets Tkinter, worker gets experiment ────────

    def run(self, experiment_fn):
        """Launch the monitor GUI and run experiment_fn in a background thread.

        experiment_fn receives the monitor as its only argument:
            def my_experiment(mon):
                mon.set_phase("Generating trajectories")
                ...

        This method blocks until both the GUI is closed and the worker
        thread finishes.
        """
        self._worker_fn = experiment_fn
        self._start_time = time.time()
        self._build_gui()

        # Start the worker thread
        self._worker_thread = threading.Thread(
            target=self._run_worker, daemon=True
        )
        self._worker_thread.start()

        # Run Tkinter mainloop (blocks until window closed)
        self._root.mainloop()

        # If user closed window early, signal stop
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

        # Re-raise worker exceptions
        if self._worker_exception is not None:
            raise self._worker_exception

    def _run_worker(self):
        """Worker thread entry point."""
        try:
            self._worker_fn(self)
        except Exception as e:
            self._worker_exception = e
            self._queue.put(_MsgError(f"{type(e).__name__}: {e}"))

    # ── GUI construction ─────────────────────────────────────────────

    def _build_gui(self):
        self._root = tk.Tk()
        self._root.title(f"PRIMO — {self.experiment_name}")
        self._root.geometry("820x620")
        self._root.configure(bg="#1e1e2e")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Ensure window appears in front on Windows
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.after(100, lambda: self._root.attributes("-topmost", False))

        style = ttk.Style()
        style.theme_use("clam")

        # Colors — Claude dark theme
        bg = "#2b2a27"           # warm dark background
        fg = "#e8e4d9"           # warm off-white text
        accent = "#d97706"       # Claude amber/orange
        green = "#8fbc6a"        # muted sage green
        red = "#c75a5a"          # muted warm red
        yellow = "#e0a84c"       # warm gold
        surface = "#393835"      # slightly lighter surface
        heading_bg = "#46443f"   # table heading background

        style.configure("Title.TLabel", background=bg, foreground=accent,
                        font=("Consolas", 16, "bold"))
        style.configure("Info.TLabel", background=bg, foreground=fg,
                        font=("Consolas", 10))
        style.configure("Phase.TLabel", background=bg, foreground=yellow,
                        font=("Consolas", 11, "bold"))
        style.configure("Stats.TLabel", background=bg, foreground=fg,
                        font=("Consolas", 10))
        style.configure("Green.TLabel", background=bg, foreground=green,
                        font=("Consolas", 10, "bold"))
        style.configure("Red.TLabel", background=bg, foreground=red,
                        font=("Consolas", 10, "bold"))
        style.configure("Horizontal.TProgressbar",
                        background=accent, troughcolor=surface, thickness=20)
        style.configure("Stop.TButton", background=heading_bg, foreground=fg,
                        font=("Consolas", 10, "bold"))

        # ── Header ───────────────────────────────────────────────────
        header = tk.Frame(self._root, bg=bg)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))

        self._title_label = ttk.Label(
            header, text=f"PRIMO  ·  {self.experiment_name}",
            style="Title.TLabel",
        )
        self._title_label.pack(side=tk.LEFT)

        self._stop_btn = ttk.Button(
            header, text="  ABORT  ", style="Stop.TButton",
            command=self._on_stop,
        )
        self._stop_btn.pack(side=tk.RIGHT)

        # ── Phase + progress ─────────────────────────────────────────
        info_frame = tk.Frame(self._root, bg=bg)
        info_frame.pack(fill=tk.X, padx=12, pady=(4, 2))

        self._phase_label = ttk.Label(
            info_frame, text="Initializing...", style="Phase.TLabel",
        )
        self._phase_label.pack(anchor=tk.W)

        prog_frame = tk.Frame(self._root, bg=bg)
        prog_frame.pack(fill=tk.X, padx=12, pady=(2, 4))

        self._progress_bar = ttk.Progressbar(
            prog_frame, orient=tk.HORIZONTAL, length=780, mode="determinate",
            style="Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill=tk.X)

        self._progress_label = ttk.Label(
            prog_frame, text="0 / 0", style="Stats.TLabel",
        )
        self._progress_label.pack(anchor=tk.E, pady=(2, 0))

        # ── Stats row ────────────────────────────────────────────────
        stats_frame = tk.Frame(self._root, bg=bg)
        stats_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        self._elapsed_label = ttk.Label(
            stats_frame, text="Elapsed: 0s", style="Stats.TLabel",
        )
        self._elapsed_label.pack(side=tk.LEFT, padx=(0, 20))

        self._rate_label = ttk.Label(
            stats_frame, text="Rate: —", style="Stats.TLabel",
        )
        self._rate_label.pack(side=tk.LEFT, padx=(0, 20))

        self._eta_label = ttk.Label(
            stats_frame, text="ETA: —", style="Stats.TLabel",
        )
        self._eta_label.pack(side=tk.LEFT)

        self._rules_done_label = ttk.Label(
            stats_frame, text="Rules: 0/0", style="Stats.TLabel",
        )
        self._rules_done_label.pack(side=tk.RIGHT)

        # ── Rule table (scrollable) ──────────────────────────────────
        table_frame = tk.Frame(self._root, bg=bg)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 4))

        columns = ("rule", "progress", "classification", "state")
        self._tree = ttk.Treeview(
            table_frame, columns=columns, show="headings",
            height=12, selectmode="none",
        )
        self._tree.heading("rule", text="Rule")
        self._tree.heading("progress", text="Seeds")
        self._tree.heading("classification", text="Class")
        self._tree.heading("state", text="State")

        self._tree.column("rule", width=220, minwidth=150)
        self._tree.column("progress", width=100, minwidth=60, anchor=tk.CENTER)
        self._tree.column("classification", width=100, minwidth=60, anchor=tk.CENTER)
        self._tree.column("state", width=340, minwidth=100)

        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self._tree.yview
        )
        self._tree.configure(yscrollcommand=scrollbar.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Style the treeview
        style.configure("Treeview",
                        background=surface, foreground=fg, fieldbackground=surface,
                        font=("Consolas", 9), rowheight=22)
        style.configure("Treeview.Heading",
                        background=heading_bg, foreground=fg,
                        font=("Consolas", 9, "bold"))
        style.map("Treeview",
                  background=[("selected", accent)],
                  foreground=[("selected", bg)])

        # ── Log panel ────────────────────────────────────────────────
        log_frame = tk.Frame(self._root, bg=bg)
        log_frame.pack(fill=tk.X, padx=12, pady=(4, 10))

        self._log_text = tk.Text(
            log_frame, height=6, bg=surface, fg=fg,
            font=("Consolas", 9), relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self._log_text.pack(fill=tk.X)

        # Map of tree item IDs by rule name
        self._tree_items = {}

        # Start polling
        self._poll()

    # ── GUI event handlers ───────────────────────────────────────────

    def _on_stop(self):
        self._stop_event.set()
        self._stop_btn.configure(text="ABORTING...")
        self._append_log(">> Stop requested by user")

    def _on_close(self):
        self._stop_event.set()
        self._root.destroy()

    # ── Message polling ──────────────────────────────────────────────

    def _poll(self):
        """Process all pending messages, then reschedule."""
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        # Update elapsed time
        if self._start_time and not self._finished:
            elapsed = time.time() - self._start_time
            self._elapsed_label.configure(text=f"Elapsed: {self._fmt_time(elapsed)}")

            # Update rate and ETA
            if self._ticks_done > 0 and self._total_ticks > 0:
                rate = self._ticks_done / elapsed
                self._rate_label.configure(text=f"Rate: {rate:.1f} ticks/s")
                remaining = self._total_ticks - self._ticks_done
                if rate > 0:
                    eta = remaining / rate
                    self._eta_label.configure(text=f"ETA: {self._fmt_time(eta)}")

        if not self._finished:
            self._root.after(self.POLL_INTERVAL, self._poll)

    def _handle_msg(self, msg):
        if isinstance(msg, _MsgInit):
            self._update_totals()

        elif isinstance(msg, _MsgPhase):
            self._current_phase = msg.phase_name
            self._phase_label.configure(text=msg.phase_name)
            self._append_log(f"Phase: {msg.phase_name}")

        elif isinstance(msg, _MsgBeginRule):
            self._rule_status[msg.rule_name] = {
                "ticks": 0, "class": "", "state": "running",
            }
            iid = self._tree.insert("", tk.END, values=(
                msg.rule_name, f"0/{self.total_seeds}", "", "running..."
            ))
            self._tree_items[msg.rule_name] = iid
            # Auto-scroll to latest
            self._tree.see(iid)

        elif isinstance(msg, _MsgTick):
            status = self._rule_status.get(msg.rule_name, {})
            status["ticks"] = status.get("ticks", 0) + 1
            status["state"] = msg.result or "running"
            self._rule_status[msg.rule_name] = status
            self._ticks_done += 1

            iid = self._tree_items.get(msg.rule_name)
            if iid:
                self._tree.item(iid, values=(
                    msg.rule_name,
                    f"{status['ticks']}/{self.total_seeds}",
                    status.get("class", ""),
                    status["state"],
                ))

            self._update_progress()

        elif isinstance(msg, _MsgFinishRule):
            status = self._rule_status.get(msg.rule_name, {})
            status["class"] = msg.classification
            status["state"] = "done"
            self._rule_status[msg.rule_name] = status
            self._rules_done += 1

            iid = self._tree_items.get(msg.rule_name)
            if iid:
                self._tree.item(iid, values=(
                    msg.rule_name,
                    f"{status.get('ticks', '?')}/{self.total_seeds}",
                    msg.classification,
                    "done ✓",
                ))

            self._rules_done_label.configure(
                text=f"Rules: {self._rules_done}/{self.total_rules}"
            )
            self._write_json()

        elif isinstance(msg, _MsgLog):
            self._append_log(msg.text)

        elif isinstance(msg, _MsgFinish):
            self._finished = True
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._phase_label.configure(text="COMPLETE")
            self._elapsed_label.configure(
                text=f"Total: {self._fmt_time(elapsed)}"
            )
            self._eta_label.configure(text="ETA: done")
            self._stop_btn.configure(text="  CLOSE  ")
            summary = msg.summary or "Experiment finished."
            self._append_log(f">> {summary}")
            self._write_json()

            # Auto-close on success (not on stop/error)
            if self.auto_close and not self._stop_event.is_set():
                delay_ms = int(self.auto_close_delay * 1000)
                self._root.after(delay_ms, self._root.destroy)

        elif isinstance(msg, _MsgError):
            self._append_log(f"ERROR: {msg.error_text}")
            self._phase_label.configure(text="ERROR")

    # ── GUI helpers ──────────────────────────────────────────────────

    def _update_totals(self):
        self._total_ticks = self.total_rules * self.total_seeds
        self._update_progress()

    def _update_progress(self):
        if self._total_ticks > 0:
            pct = (self._ticks_done / self._total_ticks) * 100
            self._progress_bar["value"] = pct
            self._progress_label.configure(
                text=f"{self._ticks_done} / {self._total_ticks}  ({pct:.0f}%)"
            )

    def _append_log(self, text):
        self._log_lines.append(text)
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _write_json(self):
        """Write current progress to a JSON file."""
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        state = {
            "experiment": self.experiment_name,
            "phase": self._current_phase,
            "rules_done": self._rules_done,
            "rules_total": self.total_rules,
            "ticks_done": self._ticks_done,
            "ticks_total": self._total_ticks,
            "elapsed_s": time.time() - self._start_time if self._start_time else 0,
            "finished": self._finished,
            "stopped": self._stop_event.is_set(),
            "rules": {
                name: {
                    "ticks": s["ticks"],
                    "classification": s["class"],
                    "state": s["state"],
                }
                for name, s in self._rule_status.items()
            },
        }
        try:
            with open(self.json_path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            pass  # non-critical

    @staticmethod
    def _fmt_time(seconds):
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m {s}s"
