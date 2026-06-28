from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl2048.replay_buffer import TransitionBatch


class QNetwork(nn.Module):
    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: list[int],
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        device: torch.device,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.device = device

        self.q_network = QNetwork(observation_size, hidden_sizes, action_size).to(device)
        self.target_network = QNetwork(observation_size, hidden_sizes, action_size).to(device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def epsilon(self, step: int) -> float:
        progress = min(max(step, 0) / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, observation: np.ndarray, step: int) -> int:
        if random.random() < self.epsilon(step):
            return random.randrange(self.action_size)
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: TransitionBatch) -> float:
        observations = torch.as_tensor(
            batch.observations, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(
            batch.next_observations, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        current_q = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_observations).max(dim=1).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
