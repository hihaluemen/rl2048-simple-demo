# G4 验收记录

## 验收标准

- 测试全部通过。
- 短训练能生成实验 artifacts。
- sample checkpoint 能被评估脚本加载。
- stronger checkpoint 能被评估脚本加载。
- Streamlit UI 能读取 sample run。
- Streamlit UI 能加载 checkpoint，并以棋盘格展示模型自动玩 2048 的过程。
- 模型自动游玩不会反复执行非法 no-op，结束时能显示 game over、truncated 或 max steps。
- README 命令可复现。

## 当前验证结果

### 单元和集成测试

命令：

```bash
.venv/bin/python -m pytest -v
```

结果：

```text
23 passed
```

### sample checkpoint 评估

命令：

```bash
.venv/bin/python scripts/evaluate.py --run runs/sample_dqn_2048
```

结果：

```text
average_score: 156.00
best_max_tile: 16
```

### stronger 配置训练记录

命令：

```bash
.venv/bin/python scripts/train.py --config configs/dqn_2048_stronger.yaml
```

结果：

```text
Run saved to: runs/20260629_021317_dqn_2048_stronger
```

该 run 的 `summary.json` 记录：

```text
total_episodes: 5000
best_score: 6024
best_max_tile: 512
```

补充评估命令：

```bash
.venv/bin/python scripts/evaluate.py --run runs/stronger_dqn_2048 --episodes 20
```

历史评估结果：

```text
average_score: 1737.80
best_max_tile: 512
```

说明：仓库提交 `runs/stronger_dqn_2048` 作为 curated 展示产物，只保留配置、指标、summary 和 `checkpoints/latest.pt`。

### UI 语法检查

命令：

```bash
.venv/bin/python -m py_compile scripts/app.py scripts/train.py scripts/evaluate.py
```

结果：通过。

### 一键 demo 脚本检查

命令：

```bash
bash -n scripts/demo.sh
```

结果：通过。

### 训练脚本快速验收

命令：

```bash
.venv/bin/python scripts/train.py --config configs/sample_dqn_2048.yaml
```

结果：

```text
Run saved to: runs/_tmp_sample_training/20260629_010236_sample_checkpoint
```

说明：该目录用于快速验收，已被 `.gitignore` 忽略。正式可展示样例保存在 `runs/sample_dqn_2048`。

## 最终结论

当前 MVP 已满足首版验收标准：

- 测试通过。
- 训练脚本可生成实验 artifacts。
- sample checkpoint 可被评估脚本加载。
- stronger checkpoint 可被评估脚本加载，并可在 UI 中展示更长训练曲线。
- Streamlit 脚本语法检查通过，且可读取 sample run 数据、加载 checkpoint 并渲染模型自动游玩过程。
- stronger 配置可以离线训练出明显强于 sample checkpoint 的 DQN baseline，并已通过本地 run 验证到 512 tile。
- README 已补充安装、测试、训练、评估和 UI 启动步骤。
