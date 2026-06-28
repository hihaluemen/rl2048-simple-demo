from __future__ import annotations

import numpy as np

from rl2048.env import Game2048Env
from rl2048.play import play_episode
from scripts.app import board_html


class AlwaysLeftAgent:
    def select_action(self, observation: np.ndarray, step: int) -> int:
        return 3


class AlwaysInvalidUpAgent:
    def select_action(self, observation: np.ndarray, step: int) -> int:
        return 0


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


def test_board_html_is_not_indented_markdown_code():
    html = board_html(np.array([[0, 2, 4, 8], [16, 32, 64, 128], [0, 0, 2, 4], [8, 16, 32, 64]]))

    assert html.startswith("<style>")
    assert "\n                <div" not in html
    assert '<div class="board">' in html
    assert "2048" not in html


def test_play_episode_falls_back_to_legal_action():
    env = Game2048Env(seed=1)

    result = play_episode(env, AlwaysInvalidUpAgent(), seed=1, max_steps=8)

    played_frames = result.frames[1:]
    assert played_frames
    assert all(frame.valid_move for frame in played_frames)
