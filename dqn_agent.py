import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


class DQN(nn.Module):
    """
    Simple fully-connected Deep Q-Network.

    Input: flattened MiniGrid observation vector.
    Output: Q-values for all actions.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    DQN agent with:
    - epsilon-greedy exploration
    - replay buffer
    - target network
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.learn_step = 0

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy strategy during training.
        During testing, always choose the action with the highest Q-value.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Tuple[float, bool]:
        """
        Train the policy network for one step.

        Returns:
            loss_value: training loss. If not enough samples, returns 0.0.
            updated: whether a training update was performed.
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0, False

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size,
            self.device,
        )

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item()), True

    def update_target_network(self) -> None:
        """Copy policy network parameters to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Gradually reduce exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        # PyTorch 2.6+ 默认 weights_only=True，但这里保存的是包含 optimizer、epsilon 等信息的完整 checkpoint。
        # 该模型文件由本项目训练生成，因此可以显式设置 weights_only=False。
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.policy_net.eval()
        self.target_net.eval()