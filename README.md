# RL 2048 训练实验平台

这是一个纯 Python 的强化学习训练实验平台 MVP，以 2048 作为训练环境，实现了：

- 从零实现的 2048 环境，提供 Gymnasium 风格 `reset`、`step`、`render` 接口。
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

如需重新生成一个轻量 sample checkpoint：

```bash
.venv/bin/python scripts/train.py --config configs/sample_dqn_2048.yaml
```

## 项目结构

```text
.
├── configs/
│   ├── dqn_2048.yaml
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
│   └── app.py
├── tests/
├── docs/
├── runs/sample_dqn_2048/
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
- sample checkpoint 不保证达到 2048，只用于稳定展示平台链路。
- UI 是 MVP，没有后台训练任务管理。
- 首版不依赖 Docker 或 Makefile。
