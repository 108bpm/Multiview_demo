"""Deterministic stand-in policy: no torch, instant. Used by
configs/example.json, configs/home.json, and as the reference for writing
new adapters.

The full adapter contract (see also base.py):
  info() -> {"name": str, "image_keys": list[str], "state_dim": int,
             "action_dim": int, "chunk_size": int, "fps": float,
             "checkpoint": str | None}
  predict_chunk(images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray
      images: {image_key: HWC uint8 RGB array} — exactly the keys in
              info()["image_keys"]; state: float32 (state_dim,); task: str;
              consumed/delay_ticks: RTC hints from the client (see base.py).
      Returns float32 (chunk_size, action_dim): absolute motor targets in the
      same units/order the robot's action_features use.
  reset() -> None — clear per-episode state (action queues, KV caches, ...).

Constructor kwargs arrive as STRINGS (forwarded from --key=value server
flags), so coerce types yourself, as done below.

Default action row t is the constant vector [t, t, ..., t], so tests can tell
which step of a chunk got executed. For robot smoke tests, pass --mode=home to
return a short linear trajectory from the current state toward a target vector
(all zeros by default).
"""
from __future__ import annotations

import numpy as np

from .base import PolicyAdapter


class DummyAdapter(PolicyAdapter):
    def __init__(
        self,
        state_dim=14,
        action_dim=14,
        chunk_size=10,
        fps=30.0,
        image_keys="camera1,camera2",
        mode="steps",
        target="",
        state_scale="",
        action_min="",
        action_max="",
        checkpoint=None,
        fail=False,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.fps = float(fps)
        self.image_keys = [key for key in str(image_keys).split(",") if key]
        self.mode = str(mode)
        self.target = self._parse_vector(target, default=0.0, name="target")
        self.state_scale = self._parse_vector(state_scale, default=1.0, name="state_scale")
        self.action_min = self._parse_vector(action_min, default=-np.inf, name="action_min")
        self.action_max = self._parse_vector(action_max, default=np.inf, name="action_max")
        self.checkpoint = checkpoint
        self.fail = bool(int(fail)) if isinstance(fail, str) else bool(fail)
        self.reset_count = 0
        self.last_meta = None

    def _parse_vector(self, value, default: float, name: str) -> np.ndarray:
        if value is None or str(value) == "":
            return np.full(self.action_dim, default, dtype=np.float32)
        values = [float(x) for x in str(value).split(",")]
        if len(values) != self.action_dim:
            raise ValueError(f"{name} has {len(values)} values, expected action_dim={self.action_dim}")
        return np.asarray(values, dtype=np.float32)

    def info(self) -> dict:
        return {
            "name": "dummy",
            "image_keys": self.image_keys,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "fps": self.fps,
            "checkpoint": self.checkpoint,
        }

    def predict_chunk(self, images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray:
        self.last_meta = {"consumed": consumed, "delay_ticks": delay_ticks}
        if self.fail:
            raise RuntimeError("dummy failure requested")
        if self.mode == "home":
            start = np.asarray(state, dtype=np.float32).reshape(-1) * self.state_scale
            if start.shape[0] != self.action_dim:
                raise ValueError(f"state has dim {start.shape[0]}, expected action_dim={self.action_dim}")
            alpha = np.linspace(1.0 / self.chunk_size, 1.0, self.chunk_size, dtype=np.float32)[:, None]
            chunk = start[None, :] + alpha * (self.target[None, :] - start[None, :])
            return np.clip(chunk, self.action_min, self.action_max)
        if self.mode == "constant":
            chunk = np.repeat(self.target[None, :], self.chunk_size, axis=0)
            return np.clip(chunk, self.action_min, self.action_max)
        if self.mode != "steps":
            raise ValueError("mode must be one of: steps, home, constant")
        steps = np.arange(self.chunk_size, dtype=np.float32)[:, None]
        return np.repeat(steps, self.action_dim, axis=1)

    def reset(self) -> None:
        self.reset_count += 1
