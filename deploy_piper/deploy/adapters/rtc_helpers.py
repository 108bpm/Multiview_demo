"""Pure helpers for RTC (real-time chunking) serving. Shapes and int values
passed into the torch.compile'd sample_actions must be STABLE across calls
(max-autotune uses cudagraphs; every new shape or int value re-captures):
leftover is always (T, A) zero-padded, inference_delay is quantized.
"""
from __future__ import annotations

import torch


def clamp_delay(delay_ticks: int, execution_horizon: int, quantum: int = 5) -> int:
    d = max(1, min(int(delay_ticks), execution_horizon))
    return min(execution_horizon, ((d + quantum - 1) // quantum) * quantum)


def slice_leftover(
    raw_chunk: torch.Tensor | None, consumed: int, execution_horizon: int
) -> torch.Tensor | None:
    if raw_chunk is None or consumed < 0:
        return None
    total = raw_chunk.shape[0]
    remaining = total - min(int(consumed), total)
    if remaining < execution_horizon:
        return None  # zero-padding inside the guidance window would pull toward 0
    out = torch.zeros_like(raw_chunk)
    out[:remaining] = raw_chunk[total - remaining:]
    return out


def build_rtc_kwargs(
    raw_chunk: torch.Tensor | None, consumed: int, delay_ticks: int, execution_horizon: int
) -> dict:
    leftover = slice_leftover(raw_chunk, consumed, execution_horizon)
    if leftover is None:
        return {}
    return {
        "prev_chunk_left_over": leftover.unsqueeze(0),
        "inference_delay": clamp_delay(delay_ticks, execution_horizon),
    }
