import csv
import json

import yaml

from rl2048.experiment import ExperimentLogger


def test_experiment_logger_writes_metrics_and_summary(tmp_path):
    config = {"run_name": "test_run", "training": {"episodes": 1}}
    logger = ExperimentLogger(tmp_path, config)

    logger.log_metric(
        {
            "episode": 1,
            "episode_reward": 10.0,
            "game_score": 8,
            "max_tile": 4,
            "episode_length": 3,
            "epsilon": 0.9,
            "loss": 1.2,
        }
    )
    logger.write_summary(total_episodes=1, best_score=8, best_max_tile=4)

    assert logger.run_dir.exists()
    assert (logger.run_dir / "config.yaml").exists()
    assert (logger.run_dir / "metrics.csv").exists()
    assert (logger.run_dir / "summary.json").exists()
    assert (logger.run_dir / "checkpoints").exists()

    with (logger.run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        assert yaml.safe_load(f) == config

    with (logger.run_dir / "metrics.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["episode"] == "1"
    assert rows[0]["max_tile"] == "4"

    with (logger.run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_episodes"] == 1
    assert summary["best_score"] == 8
    assert summary["best_max_tile"] == 4
    assert summary["latest_checkpoint_path"] is None
