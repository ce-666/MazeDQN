import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    DQN experience replay buffer.

    It stores transitions in the form:
        (state, action, reward, next_state, done)

    During training, the agent samples a random batch from this buffer.
    This reduces correlation between consecutive experiences and makes DQN
    training more stable.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        """
        Randomly sample a batch and convert it to PyTorch tensors.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if batch_size > len(self.buffer):
            raise ValueError("batch_size cannot be larger than the current buffer size")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Return True when enough samples are available for one training step."""
        return len(self.buffer) >= batch_size