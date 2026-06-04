from pathlib import Path

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
import numpy as np
import torch

from dqn_agent import DQNAgent
from plot_utils import plot_rewards, plot_success_rate


# 训练使用的 MiniGrid 环境名称
ENV_NAME = "MiniGrid-SimpleCrossingS9N1-v0"

# 总训练轮数：每一轮 episode 表示智能体从起点开始尝试一次寻路
MAX_EPISODES = 1500

# 每一轮最多允许智能体执行的动作次数，防止智能体无限乱走
MAX_STEPS_PER_EPISODE = 300

# 计算成功率时使用的滑动窗口大小，例如最近 100 轮的平均成功率
SUCCESS_WINDOW = 100

# 提前停止条件：至少训练这么多轮后，才允许提前停止
MIN_EPISODES_BEFORE_EARLY_STOP = 1000

# 提前停止条件：最近 SUCCESS_WINDOW 轮的成功率达到该阈值
EARLY_STOP_SUCCESS_RATE = 0.9999

# 提前停止条件：最近 SUCCESS_WINDOW 轮的平均奖励达到该阈值
EARLY_STOP_AVG_REWARD = 1.0

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pth"

# MiniGrid 默认有 7 个动作，但 Empty-5x5 寻路只需要 3 个动作：
# 0=左转，1=右转，2=前进。
# 如果让智能体学习全部 7 个动作，它可能学到无意义的 done/pickup/drop 等动作。
USEFUL_ACTIONS = [0, 1, 2]
ACTION_NAMES = {
    0: "左转",
    1: "右转",
    2: "前进",
}


def make_env(render_mode=None):
    """创建 MiniGrid 环境，并使用全局观测包装器，让智能体能看到完整迷宫。"""
    env = gym.make(ENV_NAME, render_mode=render_mode)
    env = FullyObsWrapper(env)
    return env


def get_device() -> torch.device:
    """自动选择计算设备。M1 Mac 优先使用 mps，否则使用 cpu。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 将 MiniGrid 返回的观测信息转换成一维特征向量，作为神经网络输入
def preprocess_observation(obs: dict) -> np.ndarray:
    """
    将 MiniGrid 的观测字典转换成一维数值向量。

    MiniGrid 的 image 通常是 (7, 7, 3) 的三维数组。
    这里先将其展平成一维向量，再做简单归一化，方便 DQN 网络训练。
    另外加入 direction 信息，让智能体知道自己当前朝向。
    """
    image = obs["image"].astype(np.float32).flatten()
    image = image / 10.0

    direction = np.array([obs["direction"] / 3.0], dtype=np.float32)

    return np.concatenate([image, direction], axis=0)


def train() -> None:
    device = get_device()
    print(f"Using device: {device}")
    print(f"Environment: {ENV_NAME}")

    # 创建 MiniGrid 迷宫环境。这里使用全局观测，降低训练难度，提高稳定性。
    env = make_env(render_mode=None)
    obs, info = env.reset(seed=42)

    sample_state = preprocess_observation(obs)
    state_dim = sample_state.shape[0]
    action_dim = len(USEFUL_ACTIONS)

    print(f"state_dim: {state_dim}")
    print(f"action_dim: {action_dim}")

    # 初始化 DQN 智能体
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=50_000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        target_update_freq=100,
    )

    episode_rewards = []
    success_history = []
    success_rates = []
    loss_history = []

    best_success_rate = 0.0

    # 主训练循环：每一轮让智能体从起点开始探索一次
    for episode in range(1, MAX_EPISODES + 1):
        obs, info = env.reset()
        state = preprocess_observation(obs)

        total_reward = 0.0
        success = False
        losses = []

        for step in range(MAX_STEPS_PER_EPISODE):
            agent_action = agent.select_action(state, training=True)
            env_action = USEFUL_ACTIONS[agent_action]

            # 记录执行动作前的位置，用于判断是否撞墙。
            old_pos = tuple(env.unwrapped.agent_pos)
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            new_pos = tuple(env.unwrapped.agent_pos)
            done = terminated or truncated

            next_state = preprocess_observation(next_obs)

            # 奖励塑形：MiniGrid 原始奖励比较稀疏
            # 这里给每一步增加小惩罚，鼓励智能体用更短路径到达目标。
            # 如果智能体选择“前进”但位置没有变化，说明前方是墙或障碍物，需要额外惩罚，避免学成无脑撞墙策略。
            shaped_reward = float(reward)
            shaped_reward -= 0.01

            if env_action == 2 and old_pos == new_pos and not terminated:
                shaped_reward -= 0.20

            if terminated and reward > 0:
                shaped_reward += 1.0
                success = True

            if truncated:
                shaped_reward -= 0.1

            # 将本次交互经验存入经验回放池
            agent.store_transition(state, agent_action, shaped_reward, next_state, done)
            # 从经验回放池随机采样一批数据，并更新 DQN 网络参数
            loss, updated = agent.update()
            if updated:
                losses.append(loss)

            state = next_state
            total_reward += shaped_reward

            if done:
                break

        # 每轮结束后降低探索率，让智能体逐渐从随机探索转向利用已学到的策略
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        success_history.append(1 if success else 0)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        loss_history.append(avg_loss)

        recent_success_rate = float(np.mean(success_history[-SUCCESS_WINDOW:]))
        success_rates.append(recent_success_rate)

        if recent_success_rate > best_success_rate and episode >= SUCCESS_WINDOW:
            best_success_rate = recent_success_rate
            agent.save(str(BEST_MODEL_PATH))

        if episode % 50 == 0:
            recent_reward = float(np.mean(episode_rewards[-SUCCESS_WINDOW:]))
            print(
                f"Episode {episode:4d} | "
                f"AvgReward({SUCCESS_WINDOW})={recent_reward:7.3f} | "
                f"SuccessRate({SUCCESS_WINDOW})={recent_success_rate:5.2f} | "
                f"Epsilon={agent.epsilon:5.3f} | "
                f"Loss={avg_loss:8.5f}"
            )

            # 如果模型已经稳定收敛，就提前停止训练，避免无意义地继续跑满 MAX_EPISODES。
            if (
                episode >= MIN_EPISODES_BEFORE_EARLY_STOP
                and recent_success_rate >= EARLY_STOP_SUCCESS_RATE
                and recent_reward >= EARLY_STOP_AVG_REWARD
            ):
                print(
                    "模型已稳定收敛，提前停止训练："
                    f"Episode={episode}, "
                    f"SuccessRate={recent_success_rate:.2f}, "
                    f"AvgReward={recent_reward:.3f}"
                )
                break

    agent.save(str(FINAL_MODEL_PATH))
    env.close()

    # 生成训练结果图：奖励曲线和成功率曲线
    plot_rewards(episode_rewards, RESULTS_DIR / "reward_curve.png")
    plot_success_rate(success_rates, RESULTS_DIR / "success_curve.png")

    print("Training finished.")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Final model saved to: {FINAL_MODEL_PATH}")
    print(f"Reward curve saved to: {RESULTS_DIR / 'reward_curve.png'}")
    print(f"Success curve saved to: {RESULTS_DIR / 'success_curve.png'}")


if __name__ == "__main__":
    train()