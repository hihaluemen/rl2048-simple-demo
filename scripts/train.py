from __future__ import annotations

import argparse

from rl2048.trainer import train
from rl2048.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on 2048.")
    parser.add_argument(
        "--config",
        default="configs/dqn_2048.yaml",
        help="Path to YAML training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_dir = train(config)
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
