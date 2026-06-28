# G4 验收记录

## 验收标准

- 测试全部通过。
- 短训练能生成实验 artifacts。
- sample checkpoint 能被评估脚本加载。
- Streamlit UI 能读取 sample run。
- README 命令可复现。

## 当前验证结果

### 单元和集成测试

命令：

```bash
.venv/bin/python -m pytest -v
```

结果：

```text
16 passed
```

### sample checkpoint 评估

命令：

```bash
.venv/bin/python scripts/evaluate.py --run runs/sample_dqn_2048
```

结果：

```text
average_score: 72.00
best_max_tile: 16
```

### UI 语法检查

命令：

```bash
.venv/bin/python -m py_compile scripts/app.py
```

结果：通过。

## 待最终确认

- README 完成后，需要按 README 从头执行一次。
- 如需提交远程仓库，执行 `git push -u origin main`。
