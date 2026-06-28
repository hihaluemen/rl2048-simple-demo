from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from scripts.evaluate import load_agent
from rl2048.env import Game2048Env
from rl2048.play import PlayResult, play_episode
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


def tile_color(value: int) -> str:
    colors = {
        0: "#cdc1b4",
        2: "#eee4da",
        4: "#ede0c8",
        8: "#f2b179",
        16: "#f59563",
        32: "#f67c5f",
        64: "#f65e3b",
        128: "#edcf72",
        256: "#edcc61",
        512: "#edc850",
        1024: "#edc53f",
        2048: "#edc22e",
    }
    return colors.get(value, "#3c3a32")


def tile_font_color(value: int) -> str:
    return "#776e65" if value in {0, 2, 4} else "#f9f6f2"


def board_html(board) -> str:
    cells = []
    for row in board:
        for raw_value in row:
            value = int(raw_value)
            label = "" if value == 0 else str(value)
            cells.append(
                f'<div class="tile" style="background:{tile_color(value)};'
                f'color:{tile_font_color(value)};">{label}</div>'
            )
    return (
        "<style>"
        "body{margin:0;background:transparent;}"
        ".board{display:grid;grid-template-columns:repeat(4,76px);"
        "gap:10px;padding:12px;background:#bbada0;width:max-content;border-radius:8px;}"
        ".tile{width:76px;height:76px;border-radius:6px;display:flex;"
        "align-items:center;justify-content:center;font-family:-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;font-size:26px;font-weight:700;line-height:1;}"
        "</style>"
        f'<div class="board">{"".join(cells)}</div>'
    )


def render_board(board) -> None:
    components.html(board_html(board), height=400, scrolling=False)


@st.cache_data(show_spinner=False)
def build_play_result(run_dir_text: str, max_steps: int, seed: int) -> PlayResult:
    run_dir = Path(run_dir_text)
    config = load_yaml(run_dir / "config.yaml")
    agent = load_agent(config, run_dir / "checkpoints" / "latest.pt", get_device())
    env = Game2048Env(seed=seed, **config.get("env", {}))
    return play_episode(env, agent, seed=seed, max_steps=max_steps)


def render_checkpoint_preview(run_dir: Path) -> None:
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    config_path = run_dir / "config.yaml"
    if not checkpoint_path.exists():
        st.info(f"未找到 checkpoint：{checkpoint_path}")
        return
    if not config_path.exists():
        st.info(f"未找到配置文件：{config_path}")
        return

    config = load_yaml(config_path)
    default_seed = int(config.get("seed", 42))
    left, right = st.columns([1, 2])
    with left:
        seed = st.number_input("回放 seed", min_value=0, value=default_seed, step=1)
        max_steps = st.slider("最大步数", min_value=1, max_value=500, value=120, step=1)
        autoplay = st.toggle("自动播放", value=False)
        speed = st.slider("播放速度（秒/步）", min_value=0.1, max_value=2.0, value=0.4, step=0.1)

    try:
        result = build_play_result(str(run_dir), int(max_steps), int(seed))
    except Exception as exc:
        st.error(f"加载或回放 checkpoint 失败：{exc}")
        return

    if autoplay:
        frame_index = st.empty()
        board_slot = st.empty()
        for index, frame in enumerate(result.frames):
            frame_index.caption(
                f"step={index}/{result.steps}, action={frame.action_name or '-'}, "
                f"score={frame.score}, max_tile={frame.max_tile}"
            )
            with board_slot.container():
                render_board(frame.board)
            if index < len(result.frames) - 1:
                import time

                time.sleep(speed)
        return

    with left:
        selected_frame = st.slider(
            "棋盘帧",
            min_value=0,
            max_value=len(result.frames) - 1,
            value=len(result.frames) - 1,
        )
    frame = result.frames[selected_frame]
    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", frame.score)
        c2.metric("Max Tile", frame.max_tile)
        c3.metric("Step", selected_frame)
        c4.metric("Action", frame.action_name or "-")
        render_board(frame.board)
        st.caption(
            f"本局总步数={result.steps}，最终 score={result.score}，"
            f"最终 max_tile={result.max_tile}，terminated={result.terminated}"
        )


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

    st.subheader("模型自动玩 2048")
    render_checkpoint_preview(selected_run)


if __name__ == "__main__":
    main()
