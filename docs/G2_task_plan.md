# G2 任务规划

## 实现顺序

1. 项目骨架：依赖、配置、工具函数、`.gitignore`。
2. 2048 环境测试：先锁定合并、非法移动、game over 和 API 行为。
3. 2048 环境实现：完成棋盘移动、spawn、reward、render。
4. 实验记录模块：生成 run 目录和 artifacts。
5. DQN 组件：replay buffer、QNetwork、DQNAgent。
6. 训练循环：串联 env、agent、buffer、logger。
7. 评估脚本：加载 run 和 checkpoint，输出 score/max tile。
8. 模型回放模块：加载 checkpoint，用合法动作过滤避免无效动作循环，并输出每一步棋盘帧。
9. Streamlit UI：查看曲线、对比实验、加载 checkpoint 自动玩 2048。
10. 示例实验：提交 sample metrics 和 sample checkpoint。
11. 长训练配置：提供 `configs/dqn_2048_stronger.yaml`，用于面试前离线训练更强 baseline。
12. 文档和验收：补齐 README、G0-G4、Review 报告和最终验证结果。

## 技术方案

- 环境状态：内部保留真实 tile 值，模型输入使用 `log2(tile)` 编码。
- 动作空间：`0=up`、`1=right`、`2=down`、`3=left`。
- 算法：DQN baseline，MLP 网络。
- 实验记录：CSV 记录 episode 指标，JSON 记录 run 摘要。
- 可视化：Streamlit 读取 `runs/` 下的实验目录。
- 回放策略：评估/展示阶段只在 `legal_actions()` 中选择 Q 值最高动作，避免模型卡在 no-op。

## 风险和处理

- DQN 训练不稳定：测试只验证训练链路，展示使用 sample checkpoint；更好效果通过 stronger 配置离线训练获得。
- reward shaping 可能误导：同时记录 `episode_reward`、`game_score`、`max_tile`。
- 现场安装依赖耗时：README 提供清晰命令，依赖集中在 `requirements.txt`。
- 模型文件可能过大：sample checkpoint 使用小网络和短训练生成。
- 回放卡住不结束：UI 限制最大步数，并在回放结果中显示 `game_over`、`truncated` 或 `max_steps` 结束原因。

## 当前取舍

首版不抽象多算法框架。这样代码更短，面试时更容易解释，也更符合 MVP 交付目标。

## 当前实现补充

- 已增加 `rl2048/play.py`，负责 checkpoint 加载和模型自动游玩。
- 已增加 `Game2048Env.legal_actions()`，用于规则测试、agent 动作选择和回放阶段动作过滤。
- 已增加 `tests/test_play.py`，覆盖自动游玩帧、合法动作过滤和结束原因。
- 当前本地 stronger run：`runs/20260629_021317_dqn_2048_stronger`，训练 5000 episodes，summary 记录 `best_score=6024`、`best_max_tile=512`。
