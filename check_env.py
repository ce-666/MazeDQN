import gymnasium as gym
import minigrid


def main():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    obs, info = env.reset(seed=42)

    print("环境创建成功")
    print("观测类型:", type(obs))
    print("观测键:", obs.keys())
    print("动作空间:", env.action_space)

    print("\nimage shape:")
    print(obs["image"].shape)

    print("\ndirection:")
    print(obs["direction"])

    print("\nmission:")
    print(obs["mission"])

    print("\nimage:")
    print(obs["image"])

    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={step}, action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()