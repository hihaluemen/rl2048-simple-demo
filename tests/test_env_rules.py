import numpy as np

from rl2048.env import Game2048Env


def test_merge_left_keeps_single_merge_per_tile():
    env = Game2048Env(seed=1)
    row, gained = env._merge_row_left(np.array([2, 2, 2, 0]))
    assert row.tolist() == [4, 2, 0, 0]
    assert gained == 4


def test_merge_left_two_pairs():
    env = Game2048Env(seed=1)
    row, gained = env._merge_row_left(np.array([2, 2, 4, 4]))
    assert row.tolist() == [4, 8, 0, 0]
    assert gained == 12


def test_invalid_move_does_not_spawn_tile():
    env = Game2048Env(seed=1)
    env.board = np.array(
        [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
        ]
    )
    before = env.board.copy()

    _, reward, terminated, truncated, info = env.step(0)

    assert np.array_equal(env.board, before)
    assert info["valid_move"] is False
    assert reward < 0
    assert terminated is False
    assert truncated is False


def test_game_over_detection():
    env = Game2048Env(seed=1)
    env.board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
    )
    assert env.is_game_over()


def test_legal_actions_excludes_noop_moves():
    env = Game2048Env(seed=1)
    env.board = np.array(
        [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
        ]
    )

    assert 0 not in env.legal_actions()
    assert set(env.legal_actions()) == {1}
