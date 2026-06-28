from __future__ import annotations

import numpy as np

from rl2048.env import Game2048Env
from rl2048.play import play_episode


class AlwaysLeftAgent:
    def select_action(self, observation: np.ndarray, step: int) -> int:
        return 3


def test_play_episode_records_boards_actions_and_scores():
    env = Game2048Env(seed=1)

    result = play_episode(env, AlwaysLeftAgent(), seed=1, max_steps=5)

    assert len(result.frames) >= 1
    assert result.frames[0].board.shape == (4, 4)
    assert result.frames[0].action is None
    assert all(frame.board.shape == (4, 4) for frame in result.frames)
    assert result.steps == len(result.frames) - 1
    assert result.score >= 0
    assert result.max_tile >= 2
