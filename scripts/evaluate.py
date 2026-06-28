from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from rl2048.dqn import DQNAgent
from rl2048.env import Game2048Env
from rl2048.utils import get_device, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--run", required=True, help="Path to a run directory.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print the board after each step.",
    )
    return parser.parse_args()


def load_agent(config: dict, checkpoint_path: Path, device: torch.device) -> DQNAgent:
    training_config = config["training"]
    agent_config = config["agent"]
    agent = DQNAgent(
        observation_size=16,
        action_size=4,
        hidden_sizes=list(agent_config["hidden_sizes"]),
        learning_rate=float(training_config["learning_rate"]),
        gamma=float(training_config["gamma"]),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        device=device,
    )
    agent.load(checkpoint_path)
    return agent


def evaluate(run_dir: Path, episodes: int, render: bool) -> tuple[float, int]:
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_yaml(config_path)
    device = get_device()
    agent = load_agent(config, checkpoint_path, device)
    env = Game2048Env(seed=config.get("seed"), **config.get("env", {}))

    scores: list[int] = []
    best_max_tile = 0
    max_steps = int(config.get("training", {}).get("max_steps_per_episode", 2000))
    for episode in range(1, episodes + 1):
        observation, _ = env.reset(seed=config.get("seed", 42) + episode)
        if render:
            print(env.render())
            print()
        for _ in range(max_steps):
            action = agent.select_action(observation, step=10**9)
            observation, _, terminated, truncated, _ = env.step(action)
            if render:
                print(env.render())
                print()
            if terminated or truncated:
                break
        scores.append(env.score)
        best_max_tile = max(best_max_tile, int(env.board.max()))

    average_score = sum(scores) / max(1, len(scores))
    return average_score, best_max_tile


def main() -> int:
    args = parse_args()
    try:
        average_score, best_max_tile = evaluate(
            run_dir=Path(args.run),
            episodes=args.episodes,
            render=args.render,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"average_score: {average_score:.2f}")
    print(f"best_max_tile: {best_max_tile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
