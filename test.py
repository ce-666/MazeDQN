from pathlib import Path
import time

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
import numpy as np
import torch

from dqn_agent import DQNAgent
from train import ENV_NAME, preprocess_observation, get_device, USEFUL_ACTIONS, ACTION_NAMES, make_env


MODEL_PATH = Path("models/best_model.pth")
MAX_TEST_STEPS = 100

# 防止测试时贪心策略一直撞墙：连续撞墙达到该次数后，临时改为转向动作
STUCK_LIMIT = 3


def test() -> None:
    """加载训练好的 DQN 模型，并在 MiniGrid 环境中进行可视化测试。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"没有找到模型文件：{MODEL_PATH}\n"
            "请先运行 train.py，生成 models/best_model.pth。"
        )

    device = get_device()
    print(f"使用设备: {device}")
    print(f"测试环境: {ENV_NAME}")
    print(f"加载模型: {MODEL_PATH}")

    # render_mode='human' 会打开可视化窗口，方便展示智能体寻路过程
    # 测试时必须使用和训练时相同的环境包装器。
    env = make_env(render_mode="human")
    obs, info = env.reset()

    state = preprocess_observation(obs)
    state_dim = state.shape[0]
    action_dim = len(USEFUL_ACTIONS)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    agent.load(str(MODEL_PATH))

    total_reward = 0.0
    success = False

    # 用于检测智能体是否连续撞墙或原地旋转卡住
    stuck_count = 0
    no_move_count = 0

    print("开始测试智能体...")

    for step in range(1, MAX_TEST_STEPS + 1):
        # 测试阶段默认使用贪心策略，直接选择 Q 值最大的动作。
        # 如果连续多步选择“前进”但位置不变，说明智能体可能卡在墙或障碍物前，
        # 此时临时改为在“左转/右转”中选择 Q 值较大的动作，避免一直撞墙。
        agent_action = agent.select_action(state, training=False)
        env_action = USEFUL_ACTIONS[agent_action]

        old_pos = tuple(env.unwrapped.agent_pos)
        obs, reward, terminated, truncated, info = env.step(env_action)
        new_pos = tuple(env.unwrapped.agent_pos)

        if new_pos == old_pos and not terminated:
            no_move_count += 1
        else:
            no_move_count = 0

        if env_action == 2 and old_pos == new_pos and not terminated:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= STUCK_LIMIT:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor).squeeze(0)
                turn_candidates = [0, 1]
                agent_action = max(turn_candidates, key=lambda a: float(q_values[a].item()))
                env_action = USEFUL_ACTIONS[agent_action]

            obs, reward, terminated, truncated, info = env.step(env_action)
            new_pos = tuple(env.unwrapped.agent_pos)
            stuck_count = 0
            no_move_count = 0

        elif no_move_count >= STUCK_LIMIT:
            # 如果连续多步都没有改变位置，通常说明智能体在原地左右转。
            # 这时临时尝试“前进”，打破转向循环。
            agent_action = 2
            env_action = USEFUL_ACTIONS[agent_action]
            obs, reward, terminated, truncated, info = env.step(env_action)
            new_pos = tuple(env.unwrapped.agent_pos)
            no_move_count = 0

        state = preprocess_observation(obs)

        total_reward += float(reward)

        print(
            f"Step {step:03d} | "
            f"AgentAction={agent_action} | "
            f"EnvAction={env_action}({ACTION_NAMES.get(env_action, '未知')}) | "
            f"Reward={reward:.3f} | "
            f"Terminated={terminated} | "
            f"Truncated={truncated}"
        )

        time.sleep(0.25)

        if terminated:
            success = reward > 0
            break

        if truncated:
            break

    env.close()

    print("测试结束。")
    print(f"总奖励: {total_reward:.3f}")
    print(f"是否成功到达终点: {'是' if success else '否'}")


if __name__ == "__main__":
    test()