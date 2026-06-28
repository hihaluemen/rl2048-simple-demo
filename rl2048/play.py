from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from rl2048.env import ACTION_NAMES, Game2048Env


class Policy(Protocol):
    def select_action(self, observation: np.ndarray, step: int) -> int:
        ...


@dataclass(frozen=True)
class PlayFrame:
    board: np.ndarray
    action: int | None
    action_name: str | None
    reward: float
    score: int
    max_tile: int
    valid_move: bool


@dataclass(frozen=True)
class PlayResult:
    frames: list[PlayFrame]
    score: int
    max_tile: int
    steps: int
    terminated: bool


def play_episode(
    env: Game2048Env,
    policy: Policy,
    seed: int | None,
    max_steps: int,
) -> PlayResult:
    observation, info = env.reset(seed=seed)
    frames = [
        PlayFrame(
            board=env.board.copy(),
            action=None,
            action_name=None,
            reward=0.0,
            score=int(info["score"]),
            max_tile=int(info["max_tile"]),
            valid_move=True,
        )
    ]
    terminated = False

    for step in range(max_steps):
        action = select_play_action(env, policy, observation, step)
        observation, reward, terminated, truncated, info = env.step(action)
        frames.append(
            PlayFrame(
                board=env.board.copy(),
                action=action,
                action_name=ACTION_NAMES[action],
                reward=float(reward),
                score=int(info["score"]),
                max_tile=int(info["max_tile"]),
                valid_move=bool(info["valid_move"]),
            )
        )
        if terminated or truncated:
            break

    return PlayResult(
        frames=frames,
        score=int(env.score),
        max_tile=int(env.board.max()),
        steps=len(frames) - 1,
        terminated=terminated,
    )


def select_play_action(
    env: Game2048Env,
    policy: Policy,
    observation: np.ndarray,
    step: int,
) -> int:
    legal_actions = env.legal_actions()
    if not legal_actions:
        return 0

    if hasattr(policy, "q_network") and hasattr(policy, "device"):
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=policy.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = policy.q_network(obs_tensor).squeeze(0).detach().cpu().numpy()
        return max(legal_actions, key=lambda action: float(q_values[action]))

    action = int(policy.select_action(observation, step=10**9 + step))
    if action in legal_actions:
        return action
    return legal_actions[0]
