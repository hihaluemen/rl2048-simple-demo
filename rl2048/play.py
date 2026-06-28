from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

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
        action = int(policy.select_action(observation, step=10**9 + step))
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
