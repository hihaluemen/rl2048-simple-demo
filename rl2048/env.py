from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


@dataclass(frozen=True)
class RewardConfig:
    invalid_move_penalty: float = 2.0
    empty_cell_weight: float = 0.1
    max_tile_weight: float = 0.01


class Game2048Env:
    def __init__(
        self,
        seed: int | None = None,
        invalid_move_penalty: float = 2.0,
        empty_cell_weight: float = 0.1,
        max_tile_weight: float = 0.01,
    ) -> None:
        self.reward_config = RewardConfig(
            invalid_move_penalty=invalid_move_penalty,
            empty_cell_weight=empty_cell_weight,
            max_tile_weight=max_tile_weight,
        )
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self._get_observation(), self._info(score_delta=0, valid_move=True)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action {action}; expected one of {sorted(ACTION_NAMES)}")

        moved_board, score_delta = self._move(action)
        valid_move = not np.array_equal(moved_board, self.board)
        if valid_move:
            self.board = moved_board
            self.score += score_delta
            self._spawn_tile()

        terminated = self.is_game_over()
        truncated = False
        reward = self._calculate_reward(score_delta=score_delta, valid_move=valid_move)
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._info(score_delta=score_delta, valid_move=valid_move),
        )

    def render(self, mode: str = "ansi") -> str:
        if mode != "ansi":
            raise ValueError("Only ansi render mode is supported")
        rows = []
        for row in self.board:
            rows.append(" ".join(f"{value:4d}" if value else "   ." for value in row))
        return "\n".join(rows)

    def _merge_row_left(self, row: np.ndarray) -> tuple[np.ndarray, int]:
        values = [int(value) for value in row if value != 0]
        merged: list[int] = []
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

    def _move(self, action: int) -> tuple[np.ndarray, int]:
        if action == 3:
            return self._move_left(self.board)
        if action == 1:
            moved, gained = self._move_left(np.fliplr(self.board))
            return np.fliplr(moved), gained
        if action == 0:
            moved, gained = self._move_left(self.board.T)
            return moved.T, gained
        moved, gained = self._move_left(np.flipud(self.board).T)
        return np.flipud(moved.T), gained

    def _move_left(self, board: np.ndarray) -> tuple[np.ndarray, int]:
        rows = []
        total_gained = 0
        for row in board:
            merged, gained = self._merge_row_left(row)
            rows.append(merged)
            total_gained += gained
        return np.vstack(rows).astype(np.int64), total_gained

    def _spawn_tile(self) -> bool:
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            return False
        row, col = empty_cells[self.rng.integers(len(empty_cells))]
        self.board[row, col] = 4 if self.rng.random() < 0.1 else 2
        return True

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros_like(self.board, dtype=np.float32)
        non_empty = self.board > 0
        obs[non_empty] = np.log2(self.board[non_empty]).astype(np.float32)
        return obs.flatten()

    def _calculate_reward(self, score_delta: int, valid_move: bool) -> float:
        if not valid_move:
            return -self.reward_config.invalid_move_penalty
        empty_cells = int(np.count_nonzero(self.board == 0))
        max_tile = int(self.board.max())
        return float(
            score_delta
            + self.reward_config.empty_cell_weight * empty_cells
            + self.reward_config.max_tile_weight * max_tile
        )

    def has_legal_moves(self) -> bool:
        if np.any(self.board == 0):
            return True
        for action in ACTION_NAMES:
            moved_board, _ = self._move(action)
            if not np.array_equal(moved_board, self.board):
                return True
        return False

    def is_game_over(self) -> bool:
        return not self.has_legal_moves()

    def _info(self, score_delta: int, valid_move: bool) -> dict:
        return {
            "score": self.score,
            "score_delta": score_delta,
            "max_tile": int(self.board.max()),
            "valid_move": valid_move,
        }
