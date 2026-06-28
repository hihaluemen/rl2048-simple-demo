# Third Party Notices

## Source Code Statement

本仓库的业务源码为项目内原创实现，没有直接复制第三方项目源码。

2048 游戏环境、DQN 训练逻辑、实验记录、Streamlit 页面和测试代码均在本项目中从零编写。Gymnasium 相关实现是基于其公开 API 约定进行接口适配，没有拷贝 Gymnasium 内部环境源码。

训练产物 `runs/sample_dqn_2048` 和 `runs/stronger_dqn_2048` 是使用本仓库训练脚本生成的模型和指标文件，不包含外部预训练模型。

## Direct Runtime And Development Dependencies

| Package | Purpose | License |
|---|---|---|
| NumPy | 2048 棋盘矩阵、随机数和数值计算 | BSD-3-Clause |
| pandas | 读取和展示训练指标 CSV | BSD-3-Clause |
| PyYAML | 读取 YAML 配置文件 | MIT |
| PyTorch | DQN 神经网络、优化器和 checkpoint 保存/加载 | BSD-3-Clause |
| Gymnasium | RL 环境标准接口、spaces 和 env checker | MIT |
| Streamlit | 可视化界面 | Apache-2.0 |
| pytest | 自动化测试 | MIT |

依赖版本以 `requirements.txt` 和本地安装环境为准。

## Notes

- 本文件用于面试项目交付说明，不替代各依赖项目自身的许可证文本。
- 如果后续复制第三方代码片段、引入外部模型或使用外部数据集，需要在本文件中补充来源、许可证和修改说明。
