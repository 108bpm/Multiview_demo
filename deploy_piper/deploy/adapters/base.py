"""The one interface a policy must implement to be deployable on the Piper."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicyAdapter(ABC):
    """Serve one policy. Implement these methods in whatever conda env the
    policy needs; the server, transport, and robot client never change."""

    @abstractmethod
    def info(self) -> dict:
        """Static metadata:
        {"name": str, "image_keys": list[str], "state_dim": int,
         "action_dim": int, "chunk_size": int, "fps": float,
         "checkpoint": str | None}   # exact value the adapter was given, or None"""

    @abstractmethod
    def predict_chunk(
        self, images: dict[str, np.ndarray], state: np.ndarray, task: str,
        consumed: int = -1, delay_ticks: int = 0,
    ) -> np.ndarray:
        """images: HWC uint8 RGB keyed by the policy's image keys;
        state: (state_dim,). consumed: rows of the previously returned chunk
        already executed client-side (-1 = no chunk yet); delay_ticks: client's
        predicted inference delay in control ticks. RTC-capable adapters use
        these to blend consecutive chunks; others may ignore them.
        Returns finite (chunk_size, action_dim) float32 absolute motor targets,
        in the same order and units as the robot's action_features."""

    def reset(self) -> None:
        """Clear per-episode state (action queues etc.). Optional."""
