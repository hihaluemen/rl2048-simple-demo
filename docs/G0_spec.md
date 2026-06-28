# G0 需求结构化

## 原始需求

用 2048 游戏作为训练场景，搭建一个 RL 训练实验平台 MVP。平台需要包含 2048 环境、至少一种 RL 算法、实验记录、可视化界面、一键启动说明和测试。

## 需求理解

这个项目不是单纯的 2048 AI，而是一个小型训练实验平台。MVP 需要同时证明三件事：

- 环境正确：2048 规则从零实现，并通过测试锁定。
- 训练链路可跑：DQN 能和环境交互，生成 metrics 和 checkpoint。
- 实验可观察：用户能查看 reward、score、max tile 曲线，并加载 checkpoint 观看模型自动玩 2048。

## 首版范围

- 语言：Python。
- 环境：从零实现 2048，继承 `gymnasium.Env`，提供标准 `reset`、`step`、`render`、`action_space`、`observation_space`。
- 算法：DQN baseline。
- 配置：YAML。
- 实验记录：`runs/<run_id>/config.yaml`、`metrics.csv`、`summary.json`、`checkpoints/latest.pt`。
- 可视化：Streamlit，支持训练曲线、实验对比和模型自动回放。
- 测试：pytest。
- 启动：直接 Python 命令，不依赖 Makefile 或 Docker。

## 非目标

- 不实现 PPO。
- 不做分布式训练。
- 不做多智能体协同控制。
- 不接入外部仿真器。
- 不承诺本地短时间从零训练一定达到 2048。

## 验收口径

- `pytest` 全部通过。
- 能运行短训练并生成实验目录。
- 能评估 `runs/sample_dqn_2048` 的 checkpoint。
- Streamlit 能展示 sample run 曲线，并加载 checkpoint 自动玩一局 2048。
- 能评估 `runs/stronger_dqn_2048` 的 checkpoint，并查看 5000 episodes 训练曲线。
- 长训练配置 `configs/dqn_2048_stronger.yaml` 可用于生成更强的本地模型；当前 curated stronger run 达到过 `best_score=6024`、`best_max_tile=512`。
- README 能让面试官按命令复现。

## 边界说明

- 仓库提交的 `runs/sample_dqn_2048` 用于稳定演示平台链路，不代表最强策略。
- 仓库提交的 `runs/stronger_dqn_2048` 用于展示较长训练效果，只保留 `latest.pt`，不提交全部中间 checkpoint。
