import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


class DQN(nn.Module):
    """
    Dueling DQN 网络。

    输入：展平后的 MiniGrid 状态向量。
    输出：每个动作对应的 Q 值。

    Dueling 结构将网络拆成两条分支：
    - value_stream：估计当前状态本身的价值 V(s)
    - advantage_stream：估计每个动作相对于平均水平的优势 A(s, a)

    最后合成：
        Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DQNAgent:
    """
    Dueling Double DQN 智能体。

    主要功能：
    - epsilon-greedy 探索策略
    - 经验回放池 Replay Buffer
    - 目标网络 Target Network
    - Double DQN 更新目标
    - Dueling DQN 网络结构

    Double DQN：
    使用 policy_net 选择下一步动作，使用 target_net 评估该动作价值，减少 Q 值高估。

    Dueling DQN：
    将 Q 值拆成状态价值 V(s) 和动作优势 A(s, a)，让模型更容易判断“当前位置本身好不好”。

    这两者结合后，可以提升复杂迷宫环境下训练的稳定性。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 5e-4,
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
            # Double DQN 核心思想：
            # 1. 使用 policy_net 选择下一状态下 Q 值最大的动作
            # 2. 使用 target_net 评估该动作对应的 Q 值
            #
            # 普通 DQN：
            #     next_q = max(target_net(next_state))
            #
            # Double DQN：
            #     next_action = argmax(policy_net(next_state))
            #     next_q = target_net(next_state, next_action)
            #
            # 这样能够降低 max 操作导致的 Q 值高估，使训练过程更加稳定。
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
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