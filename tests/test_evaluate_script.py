import subprocess
import sys


def test_evaluate_reports_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "empty_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        """
seed: 1
env:
  invalid_move_penalty: 2.0
  empty_cell_weight: 0.1
  max_tile_weight: 0.01
training:
  learning_rate: 0.001
  gamma: 0.99
  max_steps_per_episode: 20
agent:
  hidden_sizes: [32]
""".strip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run", str(run_dir)],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "latest.pt" in result.stderr
