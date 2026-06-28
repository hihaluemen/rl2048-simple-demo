# G2 任务规划

## 实现顺序

1. 项目骨架：依赖、配置、工具函数、`.gitignore`。
2. 2048 环境测试：先锁定合并、非法移动、game over 和 API 行为。
3. 2048 环境实现：完成棋盘移动、spawn、reward、render。
4. 实验记录模块：生成 run 目录和 artifacts。
5. DQN 组件：replay buffer、QNetwork、DQNAgent。
6. 训练循环：串联 env、agent、buffer、logger。
7. 评估脚本：加载 run 和 checkpoint，输出 score/max tile。
8. Streamlit UI：查看曲线、对比实验、加载 checkpoint 回放。
9. 示例实验：提交 sample metrics 和 sample checkpoint。
10. 文档和验收：补齐 README、G0-G4、最终验证结果。

## 技术方案

- 环境状态：内部保留真实 tile 值，模型输入使用 `log2(tile)` 编码。
- 动作空间：`0=up`、`1=right`、`2=down`、`3=left`。
- 算法：DQN baseline，MLP 网络。
- 实验记录：CSV 记录 episode 指标，JSON 记录 run 摘要。
- 可视化：Streamlit 读取 `runs/` 下的实验目录。

## 风险和处理

- DQN 训练不稳定：测试只验证训练链路，展示使用 sample checkpoint。
- reward shaping 可能误导：同时记录 `episode_reward`、`game_score`、`max_tile`。
- 现场安装依赖耗时：README 提供清晰命令，依赖集中在 `requirements.txt`。
- 模型文件可能过大：sample checkpoint 使用小网络和短训练生成。

## 当前取舍

首版不抽象多算法框架。这样代码更短，面试时更容易解释，也更符合 MVP 交付目标。
