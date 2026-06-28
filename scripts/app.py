from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from scripts.evaluate import load_agent
from rl2048.env import Game2048Env
from rl2048.utils import get_device, load_yaml


RUNS_DIR = Path("runs")
SAMPLE_RUN = RUNS_DIR / "sample_dqn_2048"


def find_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [path for path in RUNS_DIR.iterdir() if (path / "metrics.csv").exists()],
        key=lambda path: path.name,
    )


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    for column in ["episode_reward", "game_score", "max_tile"]:
        metrics[f"{column}_avg"] = metrics[column].rolling(window=10, min_periods=1).mean()
    return metrics


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_board(board) -> None:
    styled = []
    for row in board:
        styled.append(" ".join(f"{int(value):5d}" if value else "    ." for value in row))
    st.code("\n".join(styled), language="text")


def render_checkpoint_preview(run_dir: Path) -> None:
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    config_path = run_dir / "config.yaml"
    if not checkpoint_path.exists():
        st.info(f"未找到 checkpoint：{checkpoint_path}")
        return
    if not config_path.exists():
        st.info(f"未找到配置文件：{config_path}")
        return

    try:
        config = load_yaml(config_path)
        agent = load_agent(config, checkpoint_path, get_device())
    except Exception as exc:
        st.error(f"加载 checkpoint 失败：{exc}")
        return

    max_steps = st.slider("回放步数", min_value=1, max_value=200, value=50, step=1)
    env = Game2048Env(seed=config.get("seed"), **config.get("env", {}))
    observation, _ = env.reset(seed=config.get("seed", 42))
    frames = [env.board.copy()]
    for _ in range(max_steps):
        action = agent.select_action(observation, step=10**9)
        observation, _, terminated, truncated, _ = env.step(action)
        frames.append(env.board.copy())
        if terminated or truncated:
            break

    st.caption(f"score={env.score}, max_tile={int(env.board.max())}, steps={len(frames) - 1}")
    frame_index = st.slider("棋盘帧", min_value=0, max_value=len(frames) - 1, value=len(frames) - 1)
    render_board(frames[frame_index])


def main() -> None:
    st.set_page_config(page_title="RL 2048 实验平台", layout="wide")
    st.title("RL 2048 实验平台")

    runs = find_runs()
    if not runs:
        st.warning("未找到实验数据。请先运行训练脚本，或确认 runs/sample_dqn_2048 存在。")
        return

    default_index = 0
    if SAMPLE_RUN in runs:
        default_index = runs.index(SAMPLE_RUN)

    selected_run = st.sidebar.selectbox(
        "实验",
        runs,
        index=default_index,
        format_func=lambda path: path.name,
    )
    compare_runs = st.sidebar.multiselect(
        "对比实验",
        runs,
        default=[selected_run],
        format_func=lambda path: path.name,
    )

    metrics = load_metrics(selected_run)
    summary = load_summary(selected_run)

    st.subheader(selected_run.name)
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Score", summary.get("best_score", "-"))
    col2.metric("Best Max Tile", summary.get("best_max_tile", "-"))
    col3.metric("Episodes", summary.get("total_episodes", len(metrics)))

    st.subheader("训练曲线")
    reward_tab, score_tab, tile_tab = st.tabs(["Reward", "Game Score", "Max Tile"])
    with reward_tab:
        st.line_chart(metrics.set_index("episode")[["episode_reward", "episode_reward_avg"]])
    with score_tab:
        st.line_chart(metrics.set_index("episode")[["game_score", "game_score_avg"]])
    with tile_tab:
        st.line_chart(metrics.set_index("episode")[["max_tile", "max_tile_avg"]])

    st.subheader("实验对比")
    comparison_frames = []
    for run_dir in compare_runs:
        run_metrics = load_metrics(run_dir)
        frame = run_metrics[["episode", "max_tile"]].copy()
        frame["run"] = run_dir.name
        comparison_frames.append(frame)
    if comparison_frames:
        comparison = pd.concat(comparison_frames, ignore_index=True)
        pivot = comparison.pivot(index="episode", columns="run", values="max_tile")
        st.line_chart(pivot)

    st.subheader("Checkpoint")
    render_checkpoint_preview(selected_run)


if __name__ == "__main__":
    main()
