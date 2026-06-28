# RL 2048 训练实验平台

这是一个纯 Python 的强化学习训练实验平台 MVP，以 2048 作为训练环境，实现了：

- 从零实现的 2048 环境，继承 `gymnasium.Env`，提供标准 `reset`、`step`、`render` 接口和 action/observation space。
- DQN baseline。
- YAML 配置管理。
- 实验指标、配置、summary 和 checkpoint 自动记录。
- Streamlit 可视化界面。
- pytest 测试。
- G0 到 G4 研发过程文档。

## 快速开始

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

运行测试：

```bash
.venv/bin/python -m pytest -v
```

查看示例 checkpoint 评估：

```bash
.venv/bin/python scripts/evaluate.py --run runs/sample_dqn_2048
```

启动可视化界面：

```bash
.venv/bin/streamlit run scripts/app.py
```

也可以使用一键 demo 脚本启动：

```bash
bash scripts/demo.sh
```

## 从零训练

运行默认训练：

```bash
.venv/bin/python scripts/train.py --config configs/dqn_2048.yaml
```

训练结束后会在 `runs/` 下生成新的实验目录：

```text
runs/<run_id>/
├── config.yaml
├── metrics.csv
├── summary.json
└── checkpoints/
    ├── checkpoint_*.pt
    └── latest.pt
```

默认训练配置在 [configs/dqn_2048.yaml](configs/dqn_2048.yaml)。

如果想训练一个更像样的 DQN baseline，可以使用更长训练配置：

```bash
.venv/bin/python scripts/train.py --config configs/dqn_2048_stronger.yaml
```

这个配置会训练 5000 episodes，使用 `[256, 256]` MLP、更大的 replay buffer 和更长 epsilon 衰减。训练耗时取决于机器性能，可能需要几十分钟到数小时。训练完成后会输出新的 run 目录，例如：

```text
Run saved to: runs/20260629_xxxxxx_dqn_2048_stronger
```

评估新模型：

```bash
.venv/bin/python scripts/evaluate.py --run runs/你的run目录 --episodes 10
```

仓库内置了一个 curated stronger run，便于 clone 后直接查看更像样的训练曲线和 checkpoint：

```text
runs/stronger_dqn_2048/
├── config.yaml
├── metrics.csv
├── summary.json
└── checkpoints/latest.pt
```

它来自 5000 episodes 的 stronger 配置训练，summary 记录 `best_score=6024`、`best_max_tile=512`。如果新模型效果更好，可以替换这个展示 run：

```bash
cp runs/你的run目录/config.yaml runs/stronger_dqn_2048/config.yaml
cp runs/你的run目录/metrics.csv runs/stronger_dqn_2048/metrics.csv
cp runs/你的run目录/summary.json runs/stronger_dqn_2048/summary.json
cp runs/你的run目录/checkpoints/latest.pt runs/stronger_dqn_2048/checkpoints/latest.pt
```

## 快速 sample checkpoint

仓库内置了一个小型 sample run：

```text
runs/sample_dqn_2048/
├── config.yaml
├── metrics.csv
├── summary.json
└── checkpoints/latest.pt
```

它的目标是保证演示链路稳定：UI 能读取曲线，评估脚本能加载 checkpoint。这个 checkpoint 由短训练生成，不代表强策略。

在 UI 底部的“模型自动玩 2048”区域，可以加载 checkpoint，让模型自动运行一局 2048，并用棋盘格实时/逐步展示每一步动作、分数和最大 tile。

如需重新生成一个轻量 sample checkpoint：

```bash
.venv/bin/python scripts/train.py --config configs/sample_dqn_2048.yaml
```

## 项目结构

```text
.
├── configs/
│   ├── dqn_2048.yaml
│   ├── dqn_2048_stronger.yaml
│   └── sample_dqn_2048.yaml
├── rl2048/
│   ├── env.py
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── trainer.py
│   ├── experiment.py
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── app.py
│   └── demo.sh
├── tests/
├── docs/
├── runs/sample_dqn_2048/
├── runs/stronger_dqn_2048/
└── requirements.txt
```

## 关键设计

2048 环境内部使用真实 tile 值，例如 `0`、`2`、`4`、`8`。模型输入使用 `log2(tile)` 编码后的长度 16 向量。

动作空间：

- `0`: up
- `1`: right
- `2`: down
- `3`: left

训练同时记录：

- `episode_reward`：训练 reward，包含 reward shaping。
- `game_score`：2048 原始分数。
- `max_tile`：单局最大 tile。

这样可以区分“训练信号”和“真实游戏表现”。

## 过程文档

- [G0 需求结构化](docs/G0_spec.md)
- [G1 测试计划](docs/G1_test_plan.md)
- [G2 任务规划](docs/G2_task_plan.md)
- [G3 开发与 Review 报告](docs/G3_review_report.md)
- [G4 验收记录](docs/G4_acceptance.md)

更详细的设计和实现计划：

- [设计文档](docs/superpowers/specs/2026-06-29-rl2048-platform-design.md)
- [实现计划](docs/superpowers/plans/2026-06-29-rl2048-platform-implementation.md)

## 已知限制

- 首版只实现 DQN，没有 PPO。
- 没有分布式训练、多智能体协同或外部仿真器集成。
- sample checkpoint 不保证达到 2048，只用于稳定展示平台链路；`runs/stronger_dqn_2048` 用于展示更长训练结果。
- UI 是 MVP，没有后台训练任务管理。
- 首版不依赖 Docker 或 Makefile。
