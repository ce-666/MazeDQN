import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
import torch

from dqn_agent import DQNAgent
from test import STUCK_LIMIT
from train import (
    ACTION_NAMES,
    ENV_NAME,
    MAX_STEPS_PER_EPISODE,
    RESULTS_DIR,
    USEFUL_ACTIONS,
    get_device,
    make_env,
    preprocess_observation,
)


DEFAULT_MODEL_PATH = Path("models/best_model.pth")
DEFAULT_EPISODES = 100
DEFAULT_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained MiniGrid Dueling DQN agent without human rendering.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained checkpoint. Default: models/best_model.pth",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of evaluation episodes. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed. Episode i uses seed + i. Default: 2026",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help=f"Maximum environment steps per episode. Default: {MAX_STEPS_PER_EPISODE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for evaluation_summary.txt/json. Default: results/",
    )
    parser.add_argument(
        "--pure-only",
        action="store_true",
        help="Only run the pure greedy policy test, without the assisted-rule comparison.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_agent(model_path: Path, device: torch.device) -> DQNAgent:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run train.py first, or pass --model to an existing checkpoint."
        )

    env = make_env(render_mode=None)
    obs, _ = env.reset(seed=DEFAULT_SEED)
    sample_state = preprocess_observation(obs)
    env.close()

    agent = DQNAgent(
        state_dim=sample_state.shape[0],
        action_dim=len(USEFUL_ACTIONS),
        device=device,
    )
    agent.load(str(model_path))
    agent.epsilon = 0.0
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def greedy_action(agent: DQNAgent, state: np.ndarray) -> int:
    return agent.select_action(state, training=False)


def best_turn_action(agent: DQNAgent, state: np.ndarray) -> int:
    state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).squeeze(0)
    return max([0, 1], key=lambda action: float(q_values[action].item()))


def run_one_episode(
    agent: DQNAgent,
    seed: int,
    max_steps: int,
    assisted: bool,
) -> Dict[str, float]:
    env = make_env(render_mode=None)
    obs, _ = env.reset(seed=seed)
    state = preprocess_observation(obs)

    total_reward = 0.0
    steps = 0
    success = False
    terminated = False
    truncated = False
    assisted_actions = 0
    stuck_count = 0
    no_move_count = 0

    while steps < max_steps:
        agent_action = greedy_action(agent, state)
        env_action = USEFUL_ACTIONS[agent_action]
        old_pos = tuple(env.unwrapped.agent_pos)

        obs, reward, terminated, truncated, _ = env.step(env_action)
        steps += 1
        total_reward += float(reward)
        new_pos = tuple(env.unwrapped.agent_pos)

        if terminated and reward > 0:
            success = True

        if assisted and not (terminated or truncated):
            if new_pos == old_pos:
                no_move_count += 1
            else:
                no_move_count = 0

            if env_action == 2 and old_pos == new_pos:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= STUCK_LIMIT and steps < max_steps:
                agent_action = best_turn_action(agent, state)
                env_action = USEFUL_ACTIONS[agent_action]
                obs, reward, terminated, truncated, _ = env.step(env_action)
                steps += 1
                assisted_actions += 1
                total_reward += float(reward)
                stuck_count = 0
                no_move_count = 0
                if terminated and reward > 0:
                    success = True

            elif no_move_count >= STUCK_LIMIT and steps < max_steps:
                env_action = 2
                obs, reward, terminated, truncated, _ = env.step(env_action)
                steps += 1
                assisted_actions += 1
                total_reward += float(reward)
                no_move_count = 0
                if terminated and reward > 0:
                    success = True

        state = preprocess_observation(obs)

        if terminated or truncated:
            break

    if not terminated and not truncated and steps >= max_steps:
        truncated = True

    env.close()

    return {
        "seed": seed,
        "success": int(success),
        "total_reward": total_reward,
        "steps": steps,
        "episode_length": steps,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "assisted_actions": assisted_actions,
    }


def summarize(mode: str, episodes: List[Dict[str, float]], env_name: str) -> Dict[str, float]:
    episode_count = len(episodes)
    success_count = sum(item["success"] for item in episodes)
    terminated_count = sum(item["terminated"] for item in episodes)
    truncated_count = sum(item["truncated"] for item in episodes)

    return {
        "mode": mode,
        "environment": env_name,
        "episodes": episode_count,
        "success_count": int(success_count),
        "success_rate": success_count / episode_count if episode_count else 0.0,
        "average_cumulative_reward": mean(item["total_reward"] for item in episodes),
        "average_steps": mean(item["steps"] for item in episodes),
        "average_episode_length": mean(item["episode_length"] for item in episodes),
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "assisted_actions": int(sum(item["assisted_actions"] for item in episodes)),
    }


def write_outputs(
    output_dir: Path,
    model_path: Path,
    base_seed: int,
    max_steps: int,
    summaries: List[Dict[str, float]],
    details: Dict[str, List[Dict[str, float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_path": str(model_path),
        "environment": ENV_NAME,
        "base_seed": base_seed,
        "seed_policy": "episode_seed = base_seed + episode_index",
        "max_steps_per_episode": max_steps,
        "action_space": {
            str(index): ACTION_NAMES.get(action, str(action))
            for index, action in enumerate(USEFUL_ACTIONS)
        },
        "summaries": summaries,
        "episodes": details,
    }

    json_path = output_dir / "evaluation_summary.json"
    txt_path = output_dir / "evaluation_summary.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Independent Evaluation Summary",
        f"Model: {model_path}",
        f"Environment: {ENV_NAME}",
        f"Base seed: {base_seed}",
        f"Seed policy: episode_seed = base_seed + episode_index",
        f"Max steps per episode: {max_steps}",
        "",
        (
            "Mode\tEpisodes\tSuccess\tSuccessRate\tAvgReward\tAvgSteps\t"
            "AvgEpisodeLength\tTerminated\tTruncated\tAssistedActions"
        ),
    ]
    for item in summaries:
        lines.append(
            f"{item['mode']}\t{item['episodes']}\t{item['success_count']}\t"
            f"{item['success_rate']:.4f}\t{item['average_cumulative_reward']:.4f}\t"
            f"{item['average_steps']:.2f}\t{item['average_episode_length']:.2f}\t"
            f"{item['terminated_count']}\t{item['truncated_count']}\t"
            f"{item['assisted_actions']}"
        )

    lines.extend(
        [
            "",
            "说明：pure_greedy 为 epsilon=0 的纯模型贪心策略，不使用 test.py 中的防卡住辅助规则；",
            "assisted_rules 在纯贪心基础上加入连续撞墙/原地卡住时的临时转向或前进规则，仅作对照。",
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Environment: {ENV_NAME}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")

    agent = build_agent(args.model, device)

    modes = [("pure_greedy", False)]
    if not args.pure_only:
        modes.append(("assisted_rules", True))

    summaries = []
    details = {}

    for mode, assisted in modes:
        print(f"Running {mode} evaluation...")
        episode_results = [
            run_one_episode(
                agent=agent,
                seed=args.seed + episode_index,
                max_steps=args.max_steps,
                assisted=assisted,
            )
            for episode_index in range(args.episodes)
        ]
        summary = summarize(mode, episode_results, ENV_NAME)
        summaries.append(summary)
        details[mode] = episode_results
        print(
            f"{mode}: success={summary['success_count']}/{summary['episodes']} "
            f"({summary['success_rate']:.2%}), "
            f"avg_reward={summary['average_cumulative_reward']:.4f}, "
            f"avg_steps={summary['average_steps']:.2f}"
        )

    write_outputs(
        output_dir=args.output_dir,
        model_path=args.model,
        base_seed=args.seed,
        max_steps=args.max_steps,
        summaries=summaries,
        details=details,
    )
    print(f"Saved: {args.output_dir / 'evaluation_summary.txt'}")
    print(f"Saved: {args.output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
