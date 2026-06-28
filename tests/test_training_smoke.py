from rl2048.trainer import train


def test_short_training_creates_metrics_and_checkpoint(tmp_path):
    config = {
        "seed": 1,
        "run_name": "smoke",
        "env": {
            "invalid_move_penalty": 2.0,
            "empty_cell_weight": 0.1,
            "max_tile_weight": 0.01,
        },
        "training": {
            "episodes": 2,
            "max_steps_per_episode": 20,
            "batch_size": 4,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "replay_capacity": 100,
            "learning_starts": 4,
            "train_every": 1,
            "target_update_interval": 5,
            "checkpoint_interval": 1,
            "metrics_window": 2,
        },
        "agent": {
            "hidden_sizes": [32],
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 10,
        },
        "paths": {"runs_dir": str(tmp_path)},
    }

    run_dir = train(config)

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
