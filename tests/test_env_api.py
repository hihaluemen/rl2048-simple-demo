import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils.env_checker import check_env

from rl2048.env import Game2048Env


def test_reset_returns_observation_and_info():
    env = Game2048Env(seed=123)
    obs, info = env.reset(seed=123)

    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert info["score"] == 0
    assert np.count_nonzero(env.board) == 2


def test_seeded_reset_is_reproducible():
    env1 = Game2048Env()
    env2 = Game2048Env()

    env1.reset(seed=42)
    env2.reset(seed=42)

    assert np.array_equal(env1.board, env2.board)


def test_step_returns_gymnasium_style_tuple():
    env = Game2048Env(seed=123)
    env.reset(seed=123)

    result = env.step(3)

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (16,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_tile" in info


def test_render_returns_ansi_board():
    env = Game2048Env(seed=123)
    env.reset(seed=123)

    rendered = env.render()

    assert isinstance(rendered, str)
    assert len(rendered.splitlines()) == 4


def test_env_is_strictly_gymnasium_compatible():
    env = Game2048Env(seed=123)

    assert isinstance(env, Env)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 4
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (16,)
    assert env.observation_space.dtype == np.float32
    check_env(env, skip_render_check=True)
