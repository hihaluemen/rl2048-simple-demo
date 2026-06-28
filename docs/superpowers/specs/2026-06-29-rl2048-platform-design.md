# RL 2048 训练实验平台设计

## 背景

面试题要求构建一个强化学习训练实验平台 MVP，以 2048 作为训练环境。这个项目的重点不是只写一个算法 demo，而是展示如何把模糊需求转成可运行、可测试、可观测、可复现的工程交付。

当前工作区只有原始题目文件，没有已有项目结构需要兼容。

## 目标

- 构建一个纯 Python MVP，方便面试时快速理解和运行。
- 从零实现 2048 环境，并提供 Gymnasium 风格接口：`reset`、`step`、`render`。
- 实现一种强化学习算法：DQN。
- 通过 YAML 配置管理 DQN 和训练超参数。
- 每次训练自动记录实验配置、训练指标、模型检查点和实验摘要。
- 提供 Streamlit 可视化界面，用于查看实验曲线、对比不同实验、观看 checkpoint 自动玩 2048。
- 提供测试，证明 2048 规则、环境接口和训练链路满足 MVP 交付要求。
- 按题目要求补齐 G0 到 G4 的研发过程文档。

## 非目标

- 首版不实现 PPO。
- 首版不做多算法插件框架。
- 首版不做分布式训练、多智能体协同或外部仿真器集成。
- 首版不要求 Docker 或 Makefile。
- 不承诺本地从零训练在 5 分钟内一定达到 2048。

## 目标使用体验

README 应支持以下直接命令：

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/dqn_2048.yaml
python scripts/evaluate.py --run runs/sample_dqn_2048
streamlit run scripts/app.py
pytest
```

快速演示路径应默认使用仓库内置的 sample run。面试官打开 Streamlit 后，可以直接看到训练指标和 checkpoint 效果，不需要等待现场训练收敛。

从零训练路径应运行真实 DQN 训练循环，并在 `runs/` 下生成新的实验目录。

## 项目结构

```text
.
├── configs/
│   └── dqn_2048.yaml
├── rl2048/
│   ├── __init__.py
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
│   ├── test_env_rules.py
│   ├── test_env_api.py
│   └── test_training_smoke.py
├── docs/
│   ├── G0_spec.md
│   ├── G1_test_plan.md
│   ├── G2_task_plan.md
│   ├── G3_review_report.md
│   └── G4_acceptance.md
├── runs/
│   └── sample_dqn_2048/
├── requirements.txt
└── README.md
```

## 2048 环境设计

### 接口

`rl2048.env.Game2048Env` 提供：

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render(mode="ansi")`

接口风格遵循 Gymnasium，但实现保持轻量，不引入不必要的平台抽象。

### 状态表示

棋盘内部使用 `4x4` 整数矩阵，保存真实 tile 值，例如 `0`、`2`、`4`、`8`。

模型输入使用长度为 16 的扁平向量，采用 `log2(tile)` 编码：

- 空格：`0`
- tile `2`：`1`
- tile `4`：`2`
- tile `8`：`3`

这个表示方式紧凑、确定性强，适合 MLP DQN baseline。

### 动作空间

动作空间为 4 个离散动作：

- `0`：上
- `1`：右
- `2`：下
- `3`：左

### 规则要求

环境必须正确实现：

- 初始棋盘随机生成两个 tile。
- 四个方向的移动逻辑。
- 每个 tile 在一次移动中最多合并一次。
- 合并后按新 tile 值增加游戏分数。
- 非法移动不改变棋盘，也不生成新 tile。
- 没有合法移动时进入终止状态。
- 提供 seed 时，`reset` 行为可复现。

### 奖励设计

训练时同时记录训练奖励和真实游戏指标。

`game_score` 表示 2048 原始合并分数，用于真实效果评估。

`train_reward` 可以使用 reward shaping，帮助 DQN 更快学习：

```text
train_reward = score_delta
             + empty_cell_bonus
             + max_tile_bonus
             - invalid_move_penalty
```

具体权重放在 `configs/dqn_2048.yaml` 中，便于调参和复现实验。

## DQN 设计

首版只实现一个 DQN baseline。

核心组件：

- MLP Q-network：输入维度 `16`，隐藏层 `256`、`256`，输出维度 `4`。
- target network。
- replay buffer。
- epsilon-greedy 探索。
- Bellman loss，使用 MSE 或 Huber loss。
- 定期同步 target network。
- 定期保存 checkpoint。

实现优先保证清晰、可读、可 review，不做过度抽象。

## 实验记录

每次训练生成一个实验目录：

```text
runs/<run_id>/
├── config.yaml
├── metrics.csv
├── summary.json
└── checkpoints/
    ├── checkpoint_000200.pt
    └── latest.pt
```

`metrics.csv` 至少包含：

- episode
- episode_reward
- game_score
- max_tile
- episode_length
- epsilon
- loss，如果当轮存在训练 loss

`summary.json` 至少包含：

- run id
- start time
- end time
- total episodes
- best score
- best max tile
- latest checkpoint path

## Streamlit UI

`scripts/app.py` 提供：

- 从 `runs/` 下选择实验。
- 展示 reward、game score、max tile 曲线。
- 展示移动平均曲线，降低训练曲线噪声。
- 多实验对比 reward 和 max tile。
- 加载 checkpoint，运行一局评估并展示棋盘过程。

如果存在 `runs/sample_dqn_2048`，UI 默认选择它。

## 脚本设计

### `scripts/train.py`

职责：

- 解析 `--config`。
- 创建实验目录。
- 将配置复制到实验目录。
- 启动 DQN 训练。
- 写入 metrics 和 checkpoints。
- 写入 summary。

### `scripts/evaluate.py`

职责：

- 解析 `--run`。
- 加载配置和最新 checkpoint。
- 运行一局或多局评估。
- 输出 score、max tile 和可选棋盘渲染。

### `scripts/app.py`

职责：

- 加载实验元数据和训练指标。
- 渲染曲线。
- 对比多个实验。
- 加载 checkpoint 做简单可视化回放。

## 测试策略

测试重点放在确定性规则和训练链路正确性上。

必须覆盖：

- 左移合并，例如 `[2, 2, 2, 0] -> [4, 2, 0, 0]`。
- 一个 tile 在一次移动中不能连续合并两次。
- 非法动作不改变棋盘，也不生成新 tile。
- game over 判断正确。
- seeded reset 可复现。
- `reset` 和 `step` 返回 Gymnasium 风格元组。
- 短训练 smoke test 能生成 `metrics.csv` 和 checkpoint。

测试不需要证明 DQN 一定能收敛到 2048。

## 文档设计

项目需要包含题目要求的过程文档：

- `docs/G0_spec.md`：结构化产品和技术规格。
- `docs/G1_test_plan.md`：验收标准和测试用例。
- `docs/G2_task_plan.md`：实现任务、顺序和风险。
- `docs/G3_review_report.md`：AI 生成代码后的人工 review 记录。
- `docs/G4_acceptance.md`：最终验收命令和结果。

README 需要说明：

- 项目做什么。
- 快速演示命令。
- 从零训练命令。
- 如何打开 UI。
- 每次实验会产生哪些 artifacts。
- 已知限制。

## 风险和缓解

- 风险：现场从零训练不一定能快速看到明显收敛。缓解：仓库内置 sample run 和 checkpoint，保证快速演示稳定。
- 风险：reward shaping 可能让训练奖励看起来比真实表现更好。缓解：单独记录 `game_score` 和 `max_tile`。
- 风险：DQN 训练曲线噪声较大。缓解：UI 展示移动平均，并支持多实验对比。
- 风险：为了“平台感”做过多抽象导致交付变慢。缓解：首版保持单 DQN 路径，文档中说明后续扩展点。

## 验收标准

MVP 满足以下条件即视为完成：

- `pytest` 通过。
- `python scripts/train.py --config configs/dqn_2048.yaml` 能创建有效实验目录。
- `python scripts/evaluate.py --run runs/sample_dqn_2048` 能加载 checkpoint，或在 checkpoint 不存在时给出清晰提示。
- `streamlit run scripts/app.py` 能打开 UI，并至少查看一个实验。
- README 启动步骤不依赖 Makefile 或 Docker。
- G0 到 G4 文档存在，并且与实际实现范围一致。
