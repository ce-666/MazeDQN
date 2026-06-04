from pathlib import Path

import matplotlib.pyplot as plt


# 解决中文标题和中文坐标轴显示问题。
# macOS 上优先使用 PingFang SC；如果不可用，再尝试 Arial Unicode MS 或 SimHei。
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def plot_rewards(rewards, save_path=None):
    """绘制每轮训练的累计奖励变化曲线。"""
    if save_path is None:
        save_path = RESULTS_DIR / "reward_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.title("训练奖励变化曲线")
    plt.xlabel("训练轮次")
    plt.ylabel("累计奖励")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_success_rate(success_rates, save_path=None):
    """绘制训练过程中的成功率变化曲线。"""
    if save_path is None:
        save_path = RESULTS_DIR / "success_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(success_rates)
    plt.title("成功率变化曲线")
    plt.xlabel("训练轮次")
    plt.ylabel("成功率")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    demo_rewards = [1, 2, 3, 5, 8, 12, 15]
    demo_success = [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95]

    plot_rewards(demo_rewards)
    plot_success_rate(demo_success)

    print("测试图片生成成功。")