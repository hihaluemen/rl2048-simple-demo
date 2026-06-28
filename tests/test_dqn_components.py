import numpy as np
import torch

from rl2048.dqn import DQNAgent, QNetwork
from rl2048.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=10, observation_shape=(16,))
    for i in range(6):
        obs = np.full((16,), i, dtype=np.float32)
        buffer.push(obs, 1, 1.0, obs + 1, False)

    batch = buffer.sample(4)

    assert batch.observations.shape == (4, 16)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_observations.shape == (4, 16)
    assert batch.dones.shape == (4,)


def test_q_network_output_shape():
    net = QNetwork(input_size=16, hidden_sizes=[32], output_size=4)

    out = net(torch.zeros((2, 16), dtype=torch.float32))

    assert out.shape == (2, 4)


def test_agent_select_action_is_valid():
    agent = DQNAgent(
        observation_size=16,
        action_size=4,
        hidden_sizes=[32],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        device=torch.device("cpu"),
    )

    action = agent.select_action(np.zeros((16,), dtype=np.float32), step=0)

    assert action in {0, 1, 2, 3}


def test_agent_update_returns_loss():
    buffer = ReplayBuffer(capacity=10, observation_shape=(16,))
    for i in range(6):
        obs = np.full((16,), i, dtype=np.float32)
        buffer.push(obs, i % 4, 1.0, obs + 1, False)
    batch = buffer.sample(4)
    agent = DQNAgent(
        observation_size=16,
        action_size=4,
        hidden_sizes=[32],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        device=torch.device("cpu"),
    )

    loss = agent.update(batch)

    assert isinstance(loss, float)
    assert loss >= 0.0
