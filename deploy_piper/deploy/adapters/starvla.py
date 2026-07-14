"""StarVLA adapter for deploy_piper.

The deploy_piper side owns the hardware contract:
- images are keyed as top/l_wrist/r_wrist and are HWC uint8 RGB;
- state/action are 14-D vectors in robot.action_features order;
- returned actions are absolute motor targets.
"""
from __future__ import annotations

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework

from .base import PolicyAdapter


DEFAULT_IMAGE_KEYS = ("top", "l_wrist", "r_wrist")


class StarVLAAdapter(PolicyAdapter):
    def __init__(
        self,
        checkpoint: str = "",
        device: str = "",
        fps: str = "30",
        image_keys: str = ",".join(DEFAULT_IMAGE_KEYS),
        unnorm_key: str = "",
        use_bf16: str = "0",
    ):
        if not checkpoint:
            raise ValueError("--checkpoint=<StarVLA .pt checkpoint> is required")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint = checkpoint
        self.device = device
        self.fps = float(fps)
        self.image_keys = [key for key in str(image_keys).split(",") if key]
        self.unnorm_key = unnorm_key or None

        model = baseframework.from_pretrained(checkpoint)
        if bool(int(use_bf16)):
            model = model.to(torch.bfloat16)
        self.model = model.to(device).eval()

        stats_key = self._resolve_stats_key(self.unnorm_key)
        self.stats_key = stats_key
        self.action_stats = self.model.norm_stats[stats_key]["action"]
        self.state_stats = self.model.norm_stats[stats_key].get("state")

        self.action_dim = len(self._stat_vector(self.action_stats, "min", "q01"))
        self.state_dim = (
            len(self._stat_vector(self.state_stats, "min", "q01"))
            if self.state_stats is not None
            else self.action_dim
        )
        self.chunk_size = int(self.model.config.framework.action_model.future_action_window_size) + 1

    def _resolve_stats_key(self, unnorm_key: str | None) -> str:
        stats = self.model.norm_stats
        if unnorm_key is None:
            if len(stats) != 1:
                raise ValueError(f"--unnorm_key is required; available keys: {list(stats)}")
            return next(iter(stats))
        if unnorm_key not in stats:
            raise ValueError(f"unknown --unnorm_key={unnorm_key}; available keys: {list(stats)}")
        return unnorm_key

    @staticmethod
    def _stat_vector(stats: dict, primary: str, fallback: str) -> np.ndarray:
        if primary in stats:
            return np.asarray(stats[primary], dtype=np.float32)
        return np.asarray(stats[fallback], dtype=np.float32)

    @classmethod
    def _normalize(cls, values: np.ndarray, stats: dict | None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if stats is None:
            return values
        low = cls._stat_vector(stats, "min", "q01")
        high = cls._stat_vector(stats, "max", "q99")
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        denom = high - low
        normalized = np.zeros_like(values, dtype=np.float32)
        valid = mask & (denom != 0)
        normalized[..., valid] = 2.0 * (values[..., valid] - low[valid]) / denom[valid] - 1.0
        normalized[..., ~valid] = values[..., ~valid]
        return np.clip(normalized, -1.0, 1.0)

    @classmethod
    def _denormalize(cls, values: np.ndarray, stats: dict) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        low = cls._stat_vector(stats, "min", "q01")
        high = cls._stat_vector(stats, "max", "q99")
        mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
        clipped = np.clip(values, -1.0, 1.0)
        denormalized = np.where(mask, 0.5 * (clipped + 1.0) * (high - low) + low, clipped)
        return denormalized.astype(np.float32)

    def info(self) -> dict:
        return {
            "name": f"starvla:{self.checkpoint}",
            "image_keys": self.image_keys,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "fps": self.fps,
            "checkpoint": self.checkpoint,
            "unnorm_key": self.stats_key,
        }

    @torch.inference_mode()
    def predict_chunk(self, images, state, task, consumed=-1, delay_ticks=0) -> np.ndarray:
        missing = [key for key in self.image_keys if key not in images]
        if missing:
            raise ValueError(f"missing images for keys {missing}; got {sorted(images)}")

        ordered_images = [np.asarray(images[key], dtype=np.uint8) for key in self.image_keys]
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state has shape {state.shape}; expected ({self.state_dim},)")
        normalized_state = self._normalize(state, self.state_stats)[None, :]

        output = self.model.predict_action(
            examples=[{"image": ordered_images, "lang": task, "state": normalized_state}]
        )
        normalized_actions = np.asarray(output["normalized_actions"], dtype=np.float32)[0]
        if normalized_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"model returned action dim {normalized_actions.shape[-1]}; expected {self.action_dim}"
            )
        return self._denormalize(normalized_actions, self.action_stats)

    def reset(self) -> None:
        pass
