# RL 2048 训练实验平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个纯 Python 的 2048 DQN 训练实验平台 MVP，包含环境、训练、实验记录、可视化、测试和 G0-G4 文档。

**Architecture:** 核心库放在 `rl2048/`，脚本入口放在 `scripts/`，测试放在 `tests/`。实现先锁定 2048 规则和 Gymnasium API，再接入实验记录、DQN 训练、评估脚本和 Streamlit UI，最后补齐 README 与过程文档。

**Tech Stack:** Python 3.11+, NumPy, PyTorch, PyYAML, pandas, Streamlit, pytest.

---

## 执行结果补充

该计划已按 MVP 范围落地，最终实现相对原计划增加了以下内容：

- 新增 `rl2048/play.py`，负责 checkpoint 加载、模型自动游玩和棋盘帧记录。
- 新增 `Game2048Env.legal_actions()`，用于环境规则验证、DQN 合法动作选择和 UI 回放阶段动作过滤。
- 补齐严格 Gymnasium 兼容：继承 `gymnasium.Env`，声明 action/observation space，并增加 `check_env` 测试。
- 新增 `scripts/demo.sh`，作为一键启动入口。
- 新增 `configs/sample_dqn_2048.yaml` 与 `configs/dqn_2048_stronger.yaml`，分别支持快速样例生成和更长训练。
- 新增模型自动游玩相关测试，当前测试集覆盖环境、DQN 组件、训练 smoke、评估脚本和回放逻辑。
- curated 展示 run `runs/stronger_dqn_2048` 训练 5000 episodes，summary 记录 `best_score=6024`、`best_max_tile=512`。

原始 checklist 保留为实施过程记录；最终验收以 `docs/G4_acceptance.md` 和 `docs/G3_review_report.md` 为准。

---

## 参考文档

- 设计文档：`docs/superpowers/specs/2026-06-29-rl2048-platform-design.md`
- 原始题目：`tmp/动态2题目：RL训练实验平台.md`

## 文件职责

- `requirements.txt`：项目依赖。
- `.gitignore`：忽略缓存、虚拟环境、临时训练输出；保留 `runs/sample_dqn_2048/`。
- `configs/dqn_2048.yaml`：默认 DQN 训练配置。
- `rl2048/env.py`：2048 规则、Gymnasium 环境、渲染。
- `rl2048/replay_buffer.py`：DQN 经验回放。
- `rl2048/dqn.py`：Q 网络和 DQN agent。
- `rl2048/experiment.py`：run 目录、metrics、summary、checkpoint 路径管理。
- `rl2048/trainer.py`：训练循环。
- `rl2048/utils.py`：配置加载、随机种子、设备选择等通用函数。
- `scripts/train.py`：训练入口。
- `scripts/evaluate.py`：checkpoint 评估入口。
- `scripts/app.py`：Streamlit 可视化入口。
- `tests/test_env_rules.py`：2048 规则测试。
- `tests/test_env_api.py`：环境 API 测试。
- `tests/test_training_smoke.py`：训练链路 smoke test。
- `docs/G0_spec.md` 到 `docs/G4_acceptance.md`：题目要求的门控文档。
- `README.md`：安装、训练、评估、UI、测试和限制说明。

---

### Task 1: 项目骨架和依赖

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/dqn_2048.yaml`
- Create: `rl2048/__init__.py`
- Create: `rl2048/utils.py`

- [ ] **Step 1: 创建基础依赖文件**

`requirements.txt` 包含：

```text
numpy>=1.26
pandas>=2.2
pyyaml>=6.0
torch>=2.2
streamlit>=1.35
pytest>=8.0
```

`.gitignore` 至少包含：

```text
__pycache__/
.pytest_cache/
.venv/
*.pyc
runs/*
!runs/sample_dqn_2048/
!runs/sample_dqn_2048/**
```

- [ ] **Step 2: 创建默认配置**

`configs/dqn_2048.yaml` 内容：

```yaml
seed: 42
run_name: dqn_2048

env:
  invalid_move_penalty: 2.0
  empty_cell_weight: 0.1
  max_tile_weight: 0.01

training:
  episodes: 500
  max_steps_per_episode: 2000
  batch_size: 128
  gamma: 0.99
  learning_rate: 0.0005
  replay_capacity: 50000
  learning_starts: 1000
  train_every: 4
  target_update_interval: 1000
  checkpoint_interval: 100
  metrics_window: 50

agent:
  hidden_sizes: [256, 256]
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 50000

paths:
  runs_dir: runs
```

- [ ] **Step 3: 创建 utils**

`rl2048/utils.py` 提供：

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 4: 检查导入**

Run: `python -c "from rl2048.utils import load_yaml, set_seed, get_device; print(get_device())"`

Expected: 输出一个 torch device，例如 `cpu`。

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt configs/dqn_2048.yaml rl2048/__init__.py rl2048/utils.py
git commit -m "添加项目骨架和默认配置"
```

---

### Task 2: 2048 规则测试

**Files:**
- Create: `tests/test_env_rules.py`
- Create: `tests/test_env_api.py`

- [ ] **Step 1: 写规则失败测试**

`tests/test_env_rules.py` 覆盖：

```python
import numpy as np

from rl2048.env import Game2048Env


def test_merge_left_keeps_single_merge_per_tile():
    env = Game2048Env(seed=1)
    row, gained = env._merge_row_left(np.array([2, 2, 2, 0]))
    assert row.tolist() == [4, 2, 0, 0]
    assert gained == 4


def test_merge_left_two_pairs():
    env = Game2048Env(seed=1)
    row, gained = env._merge_row_left(np.array([2, 2, 4, 4]))
    assert row.tolist() == [4, 8, 0, 0]
    assert gained == 12


def test_invalid_move_does_not_spawn_tile():
    env = Game2048Env(seed=1)
    env.board = np.array([
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        [8, 0, 0, 0],
        [16, 0, 0, 0],
    ])
    before = env.board.copy()
    _, reward, terminated, truncated, info = env.step(0)
    assert np.array_equal(env.board, before)
    assert info["valid_move"] is False
    assert reward < 0
    assert terminated is False
    assert truncated is False


def test_game_over_detection():
    env = Game2048Env(seed=1)
    env.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])
    assert env.is_game_over()
```

- [ ] **Step 2: 写 API 失败测试**

`tests/test_env_api.py` 覆盖：

```python
import numpy as np

from rl2048.env import Game2048Env


def test_reset_returns_observation_and_info():
    env = Game2048Env(seed=123)
    obs, info = env.reset(seed=123)
    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert info["score"] == 0
    assert np.count_nonzero(env.board) == 2


def test_seeded_reset_is_reproducible():
    env1 = Game2048Env()
    env2 = Game2048Env()
    env1.reset(seed=42)
    env2.reset(seed=42)
    assert np.array_equal(env1.board, env2.board)


def test_step_returns_gymnasium_style_tuple():
    env = Game2048Env(seed=123)
    env.reset(seed=123)
    result = env.step(3)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (16,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "max_tile" in info
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_env_rules.py tests/test_env_api.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'rl2048.env'`。

- [ ] **Step 4: Commit**

```bash
git add tests/test_env_rules.py tests/test_env_api.py
git commit -m "添加2048环境规则测试"
```

---

### Task 3: 2048 环境实现

**Files:**
- Create: `rl2048/env.py`
- Modify: `tests/test_env_rules.py`
- Modify: `tests/test_env_api.py`

- [ ] **Step 1: 实现 Game2048Env**

`rl2048/env.py` 需要包含：

- `ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}`
- `Game2048Env.__init__`
- `reset`
- `step`
- `render`
- `_merge_row_left`
- `_move`
- `_spawn_tile`
- `_get_observation`
- `_calculate_reward`
- `has_legal_moves`
- `is_game_over`

关键行为：

```python
def _merge_row_left(self, row: np.ndarray) -> tuple[np.ndarray, int]:
    values = [int(v) for v in row if v != 0]
    merged = []
    gained = 0
    i = 0
    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            new_value = values[i] * 2
            merged.append(new_value)
            gained += new_value
            i += 2
        else:
            merged.append(values[i])
            i += 1
    merged.extend([0] * (4 - len(merged)))
    return np.array(merged, dtype=np.int64), gained
```

- [ ] **Step 2: 使用 numpy RNG 支持 seed**

`reset(seed=...)` 使用 `np.random.default_rng(seed)` 重建 RNG。`_spawn_tile` 从空格中随机选位置，tile 值按 90% 生成 2、10% 生成 4。

- [ ] **Step 3: 实现 step 返回 info**

`info` 至少包含：

```python
{
    "score": self.score,
    "score_delta": score_delta,
    "max_tile": int(self.board.max()),
    "valid_move": valid_move,
}
```

- [ ] **Step 4: 运行环境测试**

Run: `pytest tests/test_env_rules.py tests/test_env_api.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add rl2048/env.py tests/test_env_rules.py tests/test_env_api.py
git commit -m "实现2048环境"
```

---

### Task 4: 实验记录模块

**Files:**
- Create: `rl2048/experiment.py`
- Create: `tests/test_experiment.py`

- [ ] **Step 1: 写失败测试**

`tests/test_experiment.py`：

```python
from pathlib import Path

from rl2048.experiment import ExperimentLogger


def test_experiment_logger_writes_metrics_and_summary(tmp_path):
    config = {"run_name": "test_run", "training": {"episodes": 1}}
    logger = ExperimentLogger(tmp_path, config)
    logger.log_metric({
        "episode": 1,
        "episode_reward": 10.0,
        "game_score": 8,
        "max_tile": 4,
        "episode_length": 3,
        "epsilon": 0.9,
        "loss": 1.2,
    })
    logger.write_summary(total_episodes=1, best_score=8, best_max_tile=4)

    assert logger.run_dir.exists()
    assert (logger.run_dir / "config.yaml").exists()
    assert (logger.run_dir / "metrics.csv").exists()
    assert (logger.run_dir / "summary.json").exists()
    assert (logger.run_dir / "checkpoints").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_experiment.py -v`

Expected: FAIL with missing `rl2048.experiment`。

- [ ] **Step 3: 实现 ExperimentLogger**

实现要求：

- 初始化时创建 `runs_dir/<timestamp>_<run_name>/`
- 写入 `config.yaml`
- 追加写入 `metrics.csv`，首次写 header
- 创建 `checkpoints/`
- 提供 `checkpoint_path(name: str) -> Path`
- 提供 `write_summary(...)`

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_experiment.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add rl2048/experiment.py tests/test_experiment.py
git commit -m "添加实验记录模块"
```

---

### Task 5: Replay Buffer 和 DQN Agent

**Files:**
- Create: `rl2048/replay_buffer.py`
- Create: `rl2048/dqn.py`
- Create: `tests/test_dqn_components.py`

- [ ] **Step 1: 写失败测试**

`tests/test_dqn_components.py` 覆盖：

```python
import numpy as np
import torch

from rl2048.dqn import DQNAgent, QNetwork
from rl2048.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=10, observation_shape=(16,))
    for i in range(6):
        obs = np.full((16,), i, dtype=np.float32)
        buffer.push(obs, 1, 1.0, obs + 1, False)
    batch = buffer.sample(4)
    assert batch.observations.shape == (4, 16)
    assert batch.actions.shape == (4,)


def test_q_network_output_shape():
    net = QNetwork(input_size=16, hidden_sizes=[32], output_size=4)
    out = net(torch.zeros((2, 16), dtype=torch.float32))
    assert out.shape == (2, 4)


def test_agent_select_action_is_valid():
    agent = DQNAgent(
        observation_size=16,
        action_size=4,
        hidden_sizes=[32],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        device=torch.device("cpu"),
    )
    action = agent.select_action(np.zeros((16,), dtype=np.float32), step=0)
    assert action in {0, 1, 2, 3}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_dqn_components.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: 实现 ReplayBuffer**

实现 `TransitionBatch` dataclass 和 `ReplayBuffer`：

- `push(observation, action, reward, next_observation, done)`
- `sample(batch_size)`
- `__len__`

- [ ] **Step 4: 实现 QNetwork 和 DQNAgent**

`DQNAgent` 提供：

- `select_action(observation, step)`
- `epsilon(step)`
- `update(batch)`
- `sync_target()`
- `save(path)`
- `load(path)`

`update` 使用 target network 计算 TD target，返回 float loss。

- [ ] **Step 5: 运行组件测试**

Run: `pytest tests/test_dqn_components.py -v`

Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add rl2048/replay_buffer.py rl2048/dqn.py tests/test_dqn_components.py
git commit -m "实现DQN基础组件"
```

---

### Task 6: 训练循环和训练脚本

**Files:**
- Create: `rl2048/trainer.py`
- Create: `scripts/train.py`
- Create: `tests/test_training_smoke.py`

- [ ] **Step 1: 写 smoke test**

`tests/test_training_smoke.py`：

```python
from pathlib import Path

from rl2048.trainer import train


def test_short_training_creates_metrics_and_checkpoint(tmp_path):
    config = {
        "seed": 1,
        "run_name": "smoke",
        "env": {
            "invalid_move_penalty": 2.0,
            "empty_cell_weight": 0.1,
            "max_tile_weight": 0.01,
        },
        "training": {
            "episodes": 2,
            "max_steps_per_episode": 20,
            "batch_size": 4,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "replay_capacity": 100,
            "learning_starts": 4,
            "train_every": 1,
            "target_update_interval": 5,
            "checkpoint_interval": 1,
            "metrics_window": 2,
        },
        "agent": {
            "hidden_sizes": [32],
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 10,
        },
        "paths": {"runs_dir": str(tmp_path)},
    }
    run_dir = train(config)
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_training_smoke.py -v`

Expected: FAIL with missing `rl2048.trainer`。

- [ ] **Step 3: 实现 train(config)**

训练循环要求：

- 初始化 seed、env、agent、buffer、ExperimentLogger。
- 每个 episode 调用 env.reset。
- epsilon-greedy 选动作。
- 将 transition 放入 replay buffer。
- 达到 `learning_starts` 后按 `train_every` 更新 agent。
- 按 `target_update_interval` 同步 target network。
- 按 `checkpoint_interval` 保存 checkpoint。
- 每 episode 写 metrics。
- 结束写 summary。
- 返回 `run_dir`。

- [ ] **Step 4: 实现 scripts/train.py**

入口要求：

```bash
python scripts/train.py --config configs/dqn_2048.yaml
```

脚本只负责解析参数、加载 YAML、调用 `train(config)`、打印 run 目录。

- [ ] **Step 5: 运行训练 smoke test**

Run: `pytest tests/test_training_smoke.py -v`

Expected: PASS。

- [ ] **Step 6: 运行全量测试**

Run: `pytest -v`

Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add rl2048/trainer.py scripts/train.py tests/test_training_smoke.py
git commit -m "实现训练循环和训练脚本"
```

---

### Task 7: 评估脚本

**Files:**
- Create: `scripts/evaluate.py`
- Create: `tests/test_evaluate_script.py`

- [ ] **Step 1: 写轻量测试**

`tests/test_evaluate_script.py` 测试缺少 checkpoint 时能清晰失败：

```python
import subprocess
import sys
from pathlib import Path


def test_evaluate_reports_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "empty_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run", str(run_dir)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "latest.pt" in result.stderr
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_evaluate_script.py -v`

Expected: FAIL because script does not exist.

- [ ] **Step 3: 实现 evaluate.py**

脚本要求：

- 解析 `--run`、`--episodes`、`--render`。
- 查找 `<run>/config.yaml` 和 `<run>/checkpoints/latest.pt`。
- checkpoint 不存在时向 stderr 输出清晰错误并返回非 0。
- checkpoint 存在时加载 DQNAgent，运行评估，打印平均 score 和 max tile。

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_evaluate_script.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate_script.py
git commit -m "添加模型评估脚本"
```

---

### Task 8: Streamlit UI 和示例实验

**Files:**
- Create: `scripts/app.py`
- Create: `runs/sample_dqn_2048/config.yaml`
- Create: `runs/sample_dqn_2048/metrics.csv`
- Create: `runs/sample_dqn_2048/summary.json`
- 最终创建: `runs/sample_dqn_2048/checkpoints/latest.pt`

- [ ] **Step 1: 准备 sample run 曲线数据**

创建轻量 sample run，至少能让 UI 展示曲线。`metrics.csv` 可先使用手写的小样本数据。

`runs/sample_dqn_2048/metrics.csv` 至少包含：

```csv
episode,episode_reward,game_score,max_tile,episode_length,epsilon,loss
1,12,8,4,20,1.0,
2,32,24,8,34,0.9,1.2
3,80,72,16,50,0.8,0.9
4,160,144,32,80,0.7,0.7
5,320,288,64,120,0.6,0.5
```

- [ ] **Step 2: 实现 app.py**

UI 要求：

- 读取 `runs/` 下所有含 `metrics.csv` 的目录。
- 默认选择 `runs/sample_dqn_2048`。
- 用 pandas 读取 metrics。
- 展示 reward、game_score、max_tile 折线图。
- 支持多选 run 对比 max_tile。
- 若 checkpoint 不存在，显示提示而不是报错。

- [ ] **Step 3: 生成 sample checkpoint**

在训练链路可用后，运行一次较短训练，生成真实 checkpoint，再整理为 `runs/sample_dqn_2048/checkpoints/latest.pt`。这个 checkpoint 不承诺达到 2048，但必须能被 `scripts/evaluate.py` 加载并完成一局评估。

推荐流程：

```bash
python scripts/train.py --config configs/dqn_2048.yaml
```

然后将生成 run 中的 `checkpoints/latest.pt` 复制到 `runs/sample_dqn_2048/checkpoints/latest.pt`，并根据实际结果更新 `summary.json`。如果训练耗时过长，可以降低 sample 配置 episodes，但保留默认配置用于正常训练。

- [ ] **Step 4: 运行语法检查**

Run: `python -m py_compile scripts/app.py`

Expected: PASS。

- [ ] **Step 5: 可选本地启动检查**

Run: `streamlit run scripts/app.py`

Expected: 浏览器或终端显示 Streamlit 服务启动。检查完成后停止服务。

- [ ] **Step 6: Commit**

```bash
git add scripts/app.py runs/sample_dqn_2048/config.yaml runs/sample_dqn_2048/metrics.csv runs/sample_dqn_2048/summary.json runs/sample_dqn_2048/checkpoints/latest.pt
git commit -m "添加可视化界面和示例实验"
```

---

### Task 9: G0-G4 文档和 README

**Files:**
- Create: `docs/G0_spec.md`
- Create: `docs/G1_test_plan.md`
- Create: `docs/G2_task_plan.md`
- Create: `docs/G3_review_report.md`
- Create: `docs/G4_acceptance.md`
- Create: `README.md`

- [ ] **Step 1: 编写 G0 结构化需求**

`docs/G0_spec.md` 说明：

- 原始需求。
- 明确范围：DQN、2048、纯 Python、Streamlit、pytest。
- 非目标：PPO、Docker、分布式、多智能体。
- 验收口径：测试通过、训练可跑、UI 可看 sample run。

- [ ] **Step 2: 编写 G1 测试计划**

`docs/G1_test_plan.md` 列出测试用例和验收标准，和 `tests/` 文件逐项对应。

- [ ] **Step 3: 编写 G2 任务计划**

`docs/G2_task_plan.md` 总结实现拆解、顺序和风险。

- [ ] **Step 4: 编写 G3 Review 报告**

`docs/G3_review_report.md` 记录人工 review 关注点：

- 2048 合并规则是否正确。
- 非法动作是否 spawn tile。
- 训练指标是否区分 shaped reward 和真实 score。
- checkpoint 缺失时脚本是否清晰失败。
- UI 是否能在无 checkpoint 时展示 sample metrics。

- [ ] **Step 5: 编写 G4 验收文档**

`docs/G4_acceptance.md` 记录最终要运行的命令：

```bash
pytest
python scripts/train.py --config configs/dqn_2048.yaml
python scripts/evaluate.py --run runs/sample_dqn_2048
streamlit run scripts/app.py
```

- [ ] **Step 6: 编写 README**

README 包含：

- 项目简介。
- 环境安装。
- 快速查看 UI。
- 从零训练。
- 评估 checkpoint。
- 运行测试。
- 目录结构。
- 已知限制。

- [ ] **Step 7: Commit**

```bash
git add README.md docs/G0_spec.md docs/G1_test_plan.md docs/G2_task_plan.md docs/G3_review_report.md docs/G4_acceptance.md
git commit -m "补充项目文档和验收说明"
```

---

### Task 10: 最终验收和远程推送

**Files:**
- Modify: `docs/G4_acceptance.md`
- Possibly modify: `README.md`

- [ ] **Step 1: 安装依赖**

Run:

```bash
pip install -r requirements.txt
```

Expected: 依赖安装成功。

- [ ] **Step 2: 运行测试**

Run:

```bash
pytest -v
```

Expected: PASS。

- [ ] **Step 3: 运行短训练**

Run:

```bash
python scripts/train.py --config configs/dqn_2048.yaml
```

Expected: 输出新 run 目录，且包含 `metrics.csv`、`summary.json`、`checkpoints/latest.pt`。

- [ ] **Step 4: 运行评估**

Run:

```bash
python scripts/evaluate.py --run runs/sample_dqn_2048
```

Expected: 加载 `runs/sample_dqn_2048/checkpoints/latest.pt`，输出 score 和 max tile。若 checkpoint 被手动删除，应清晰提示 `latest.pt` 缺失。

- [ ] **Step 5: 检查 UI 启动**

Run:

```bash
streamlit run scripts/app.py
```

Expected: UI 能加载 sample metrics。检查完成后停止服务。

- [ ] **Step 6: 更新 G4 验收结果**

将实际命令和结果写入 `docs/G4_acceptance.md`。

- [ ] **Step 7: 最终 commit**

```bash
git add docs/G4_acceptance.md README.md
git commit -m "记录最终验收结果"
```

- [ ] **Step 8: 推送远程**

```bash
git push -u origin main
```

Expected: 推送到 `git@github.com:hihaluemen/rl2048-simple-demo.git`。
