# G1 测试计划

## 测试目标

测试重点放在确定性规则和工程链路，不以“模型一定收敛到 2048”作为自动化测试目标。

## 2048 规则测试

对应文件：`tests/test_env_rules.py`

- `[2, 2, 2, 0]` 左移后应为 `[4, 2, 0, 0]`。
- `[2, 2, 4, 4]` 左移后应为 `[4, 8, 0, 0]`。
- 非法移动不改变棋盘，也不生成新 tile。
- 无空位且无相邻可合并 tile 时，game over 判断为真。
- `legal_actions` 只返回会改变棋盘的动作。

## 环境 API 测试

对应文件：`tests/test_env_api.py`

- `reset` 返回 `(observation, info)`。
- observation 是 shape `(16,)` 的 `float32` 向量。
- 初始棋盘有两个 tile。
- 相同 seed 的 reset 可复现。
- `step` 返回 Gymnasium 风格五元组。

## 工具和实验记录测试

对应文件：

- `tests/test_utils.py`
- `tests/test_experiment.py`

覆盖：

- YAML 配置读取。
- seed 设置可复现。
- 实验目录创建。
- `config.yaml`、`metrics.csv`、`summary.json` 写入。
- `checkpoints/` 目录创建。

## DQN 组件测试

对应文件：`tests/test_dqn_components.py`

覆盖：

- replay buffer sample shape。
- QNetwork 输出 shape。
- DQNAgent 选择合法动作。
- 单次 update 返回非负 loss。

## 训练链路测试

对应文件：`tests/test_training_smoke.py`

覆盖：

- 运行 2 episode 短训练。
- 生成 `metrics.csv`。
- 生成 `summary.json`。
- 生成 `checkpoints/latest.pt`。

## 评估脚本测试

对应文件：`tests/test_evaluate_script.py`

覆盖：

- checkpoint 缺失时，脚本返回非 0。
- stderr 明确包含 `latest.pt`，便于定位问题。

## 模型自动游玩测试

对应文件：`tests/test_play.py`

覆盖：

- 自动游玩会记录初始棋盘和后续每一步棋盘。
- 每一帧包含动作、分数、最大 tile 和动作是否合法。
- 回放阶段会屏蔽无效动作，避免模型反复执行 no-op。
- 回放结果会记录结束原因，例如 game over 或达到最大步数。
- 结果中的 `steps` 与帧数量一致。

## 验收命令

```bash
.venv/bin/python -m pytest -v
```
