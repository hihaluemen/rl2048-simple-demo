from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rl2048.dqn import DQNAgent
from rl2048.env import Game2048Env
from rl2048.experiment import ExperimentLogger
from rl2048.replay_buffer import ReplayBuffer
from rl2048.utils import get_device, set_seed


def train(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    env_config = config.get("env", {})
    training_config = config["training"]
    agent_config = config["agent"]
    runs_dir = config.get("paths", {}).get("runs_dir", "runs")

    env = Game2048Env(seed=seed, **env_config)
    device = get_device()
    agent = DQNAgent(
        observation_size=16,
        action_size=4,
        hidden_sizes=list(agent_config["hidden_sizes"]),
        learning_rate=float(training_config["learning_rate"]),
        gamma=float(training_config["gamma"]),
        epsilon_start=float(agent_config["epsilon_start"]),
        epsilon_end=float(agent_config["epsilon_end"]),
        epsilon_decay_steps=int(agent_config["epsilon_decay_steps"]),
        device=device,
    )
    buffer = ReplayBuffer(
        capacity=int(training_config["replay_capacity"]),
        observation_shape=(16,),
    )
    logger = ExperimentLogger(runs_dir, config)

    total_steps = 0
    best_score = 0
    best_max_tile = 0
    last_loss: float | None = None
    episodes = int(training_config["episodes"])
    max_steps_per_episode = int(training_config["max_steps_per_episode"])
    batch_size = int(training_config["batch_size"])
    learning_starts = int(training_config["learning_starts"])
    train_every = int(training_config["train_every"])
    target_update_interval = int(training_config["target_update_interval"])
    checkpoint_interval = int(training_config["checkpoint_interval"])

    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        episode_reward = 0.0
        episode_score_start = env.score
        episode_length = 0

        for _ in range(max_steps_per_episode):
            action = agent.select_action(observation, step=total_steps)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(observation, action, reward, next_observation, done)

            observation = next_observation
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if (
                len(buffer) >= max(batch_size, learning_starts)
                and total_steps % train_every == 0
            ):
                last_loss = agent.update(buffer.sample(batch_size))

            if total_steps % target_update_interval == 0:
                agent.sync_target()

            if done:
                break

        game_score = env.score - episode_score_start
        max_tile = int(env.board.max())
        best_score = max(best_score, env.score)
        best_max_tile = max(best_max_tile, max_tile)
        logger.log_metric(
            {
                "episode": episode,
                "episode_reward": round(episode_reward, 6),
                "game_score": game_score,
                "max_tile": max_tile,
                "episode_length": episode_length,
                "epsilon": round(agent.epsilon(total_steps), 6),
                "loss": "" if last_loss is None else round(last_loss, 6),
            }
        )

        if episode % checkpoint_interval == 0 or episode == episodes:
            checkpoint_path = logger.checkpoint_path(f"checkpoint_{episode:06d}.pt")
            agent.save(checkpoint_path)
            latest_path = logger.checkpoint_path("latest.pt")
            shutil.copyfile(checkpoint_path, latest_path)

    logger.write_summary(
        total_episodes=episodes,
        best_score=best_score,
        best_max_tile=best_max_tile,
    )
    return logger.run_dir
