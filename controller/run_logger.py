"""Structured per-run logging (spec section "Logging and evaluation"). Every
run gets its own timestamped directory under runs/ with:
  launch_command.txt   exact fgfs argv
  init_conditions.json initial conditions + weather + scenario
  cycles.jsonl          one JSON line per 10 Hz control cycle (raw telemetry,
                         physical + normalized features, FSM state, raw model
                         predictions, commands sent, saturation/safety events,
                         latency)
  transitions.json      Flight Manager state transitions
  summary.json          post-flight summary (spec: completed/aborted/crashed,
                         tracking errors, touchdown metrics, bounce events,
                         saturation, rollout result, abort reason)
  fgfs_stdout.log        FlightGear's own stdout/stderr
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from controller.config import RUNS_DIR


def _default(o):
    if is_dataclass(o):
        return asdict(o)
    return str(o)


class RunLogger:
    def __init__(self, mode: str, scenario: str):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = RUNS_DIR / f"{ts}_{mode}_{scenario}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._cycles_f = open(self.run_dir / "cycles.jsonl", "w")
        self.n_cycles = 0

    def log_launch_command(self, argv: list[str]):
        (self.run_dir / "launch_command.txt").write_text(" ".join(argv) + "\n")

    def log_init_conditions(self, data: dict):
        (self.run_dir / "init_conditions.json").write_text(json.dumps(data, indent=2, default=_default))

    def log_cycle(self, record: dict):
        self._cycles_f.write(json.dumps(record, default=_default) + "\n")
        self.n_cycles += 1
        if self.n_cycles % 20 == 0:
            self._cycles_f.flush()

    def log_transitions(self, transitions: list):
        (self.run_dir / "transitions.json").write_text(json.dumps(transitions, indent=2, default=_default))

    def log_summary(self, summary: dict):
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_default))

    def fgfs_log_path(self) -> Path:
        return self.run_dir / "fgfs_stdout.log"

    def close(self):
        self._cycles_f.close()
