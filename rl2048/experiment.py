from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


METRIC_FIELDS = [
    "episode",
    "episode_reward",
    "game_score",
    "max_tile",
    "episode_length",
    "epsilon",
    "loss",
]


class ExperimentLogger:
    def __init__(self, runs_dir: str | Path, config: dict[str, Any]) -> None:
        self.config = config
        self.run_id = self._build_run_id(config.get("run_name", "run"))
        self.run_dir = Path(runs_dir) / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.yaml"
        self.start_time = datetime.now().isoformat(timespec="seconds")
        self.latest_checkpoint_path: Path | None = None

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def log_metric(self, metric: dict[str, Any]) -> None:
        write_header = not self.metrics_path.exists()
        row = {field: metric.get(field, "") for field in METRIC_FIELDS}
        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def checkpoint_path(self, name: str) -> Path:
        path = self.checkpoints_dir / name
        self.latest_checkpoint_path = path
        return path

    def write_summary(
        self,
        total_episodes: int,
        best_score: int,
        best_max_tile: int,
    ) -> None:
        summary = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "total_episodes": total_episodes,
            "best_score": best_score,
            "best_max_tile": best_max_tile,
            "latest_checkpoint_path": (
                str(self.latest_checkpoint_path)
                if self.latest_checkpoint_path is not None
                else None
            ),
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _write_config(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def _build_run_id(self, run_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in run_name
        ).strip("_")
        return f"{timestamp}_{safe_name or 'run'}"
