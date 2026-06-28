# G3 开发与 Review 报告

## Review 范围

本项目代码由 AI 辅助生成，但关键行为经过人工 review 和测试验证。Review 重点放在容易影响正确性和演示稳定性的部分。

## 2048 环境

检查项：

- 单次移动中每个 tile 只能合并一次。
- 非法移动不能生成新 tile。
- `score_delta` 使用合并后的 tile 值。
- `reset(seed=...)` 可复现。
- `step` 返回 Gymnasium 风格五元组。

结论：已用 `tests/test_env_rules.py` 和 `tests/test_env_api.py` 覆盖。

## DQN 训练链路

检查项：

- replay buffer sample shape 正确。
- QNetwork 输出动作维度为 4。
- DQN update 使用 target network 计算 TD target。
- 训练循环按 episode 写 metrics。
- checkpoint 同时保存周期文件和 `latest.pt`。

结论：组件测试和 smoke test 通过。

## 实验记录

检查项：

- run 目录包含 config、metrics、summary、checkpoints。
- metrics 使用 CSV，便于 Streamlit 和 pandas 读取。
- summary 中记录 best score、best max tile 和 latest checkpoint path。

结论：`tests/test_experiment.py` 已覆盖核心写入行为。

## 脚本和 UI

检查项：

- `scripts/train.py` 和 `scripts/evaluate.py` 可直接用 Python 运行。
- checkpoint 缺失时，评估脚本给出清晰错误。
- 模型自动游玩逻辑集中在 `rl2048/play.py`，CLI 和 UI 复用同一条 rollout 链路。
- 回放阶段屏蔽无效动作，避免弱模型反复选择 no-op 导致演示卡住。
- Streamlit 在无 checkpoint 时提示，在有 checkpoint 时加载模型并用棋盘格展示自动游玩过程。

结论：评估脚本测试和自动游玩测试通过，UI 通过语法检查和 sample checkpoint 手动评估验证。

## 已知限制

- sample checkpoint 只用于证明加载和回放链路，不代表强策略。
- 默认 DQN 没有 Double DQN、Dueling DQN 或 prioritized replay。
- UI 是 MVP，不包含复杂实验筛选、权限或后台训练任务管理。
