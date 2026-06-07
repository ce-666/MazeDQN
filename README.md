# MazeDQN

基于 DQN、Double DQN 和 Dueling DQN 的 MiniGrid 迷宫寻路强化学习项目（个人课程作业）。

## 项目简介

本项目在 MiniGrid-SimpleCrossing 系列环境（S9N1、S9N2、S9N3）中，从基础 DQN 出发逐步引入 Double DQN 目标更新、Dueling DQN 网络结构、经验回放、目标网络和基于 BFS 的奖励塑形，最终在 S9N3 环境中达到约 99% 的成功率。项目使用 Git 进行版本管理，N1、N2、N3 三个阶段分别打 tag 保存。

## 环境依赖

- Python 3.11+
- PyTorch 2.6+
- Gymnasium
- MiniGrid
- Matplotlib

```bash
pip install torch gymnasium minigrid matplotlib
```

## 项目结构

```
MazeDQN/
├── train.py          # 训练主程序（环境创建、奖励塑形、训练循环）
├── dqn_agent.py      # Dueling Double DQN 智能体（网络定义 + 训练逻辑）
├── replay_buffer.py  # 经验回放池
├── plot_utils.py     # 奖励曲线 / 成功率曲线绘制
├── test.py           # 模型测试与可视化
├── check_env.py      # 环境连通性检查
├── checkpoints/      # 各阶段训练结果
│   ├── N1/           # 基础 DQN，S9N1 环境
│   ├── N2/           # Double DQN + 初步奖励塑形，S9N2 环境
│   └── N3/           # Dueling Double DQN + BFS 奖励塑形，S9N3 环境
├── models/           # 通用模型文件
├── results/          # 训练曲线图片
└── README.md
```

## 核心技术

### DQN
使用神经网络近似动作价值函数 Q(s,a)，以 MiniGrid 展平观测为输入，输出各动作的 Q 值。配合经验回放池打破样本相关性，使用目标网络稳定 TD 目标。

### Double DQN
将动作选择和动作评估分离：policy_net 选择下一状态的最优动作，target_net 评估该动作价值，降低 max 操作带来的 Q 值高估。

### Dueling DQN
将网络拆分为价值分支（value_stream）和优势分支（advantage_stream），分别估计 V(s) 和 A(s,a)，最终 Q = V(s) + A(s,a) - mean(A(s,a))。

### MiniGrid
轻量级网格世界强化学习环境。本项目使用 SimpleCrossing 系列（9×9 网格），通过 FullyObsWrapper 提供全局观测。

## 运行方法

```bash
# 训练（默认 S9N3 环境）
python train.py

# 测试
python test.py

# 环境检查
python check_env.py
```

训练超参数可在 `train.py` 中修改。

## 实验结果

| 阶段 | 环境 | 算法 | 最终成功率 |
|------|------|------|-----------|
| N1 | S9N1 | 基础 DQN | ≈ 0.96 |
| N2 | S9N2 | Double DQN + 奖励塑形 | ≈ 0.90 |
| N3 | S9N3 | Dueling Double DQN + BFS 奖励塑形 | ≈ 0.99 |

训练曲线和模型文件保存在 `checkpoints/` 目录下。

## 版本标签

```bash
git checkout N1  # 基础 DQN
git checkout N2  # Double DQN
git checkout N3  # Dueling Double DQN（最终版）
```
