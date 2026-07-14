"""Pure queue logic for the async-overlap control loop. No I/O, no threads —
the client calls everything from its single control-loop thread.

Per control tick:
    action = executor.next_action()          # None -> queue dry, hold position
    if executor.should_request():
        # capture obs NOW, then:
        executor.mark_requested()
        # ... send obs to the server on a background thread ...
    # when the reply lands (any later tick):
    executor.on_chunk(chunk)                  # or executor.on_request_failed()
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ChunkExecutor:
    def __init__(self, chunk_threshold: float = 0.5):
        if not 0.0 < chunk_threshold <= 1.0:
            raise ValueError(f"chunk_threshold must be in (0, 1], got {chunk_threshold}")
        self.chunk_threshold = chunk_threshold
        self._queue: deque = deque()
        self._chunk_len = 0  # length of the chunk the current queue came from
        self._tick = 0
        self._request_tick: int | None = None
        self._executed_since_request = 0
        self._last_skip = 0  # rows dropped at the last install == measured in-flight delay

    @property
    def in_flight(self) -> bool:
        return self._request_tick is not None

    def next_action(self):
        """Advance one control tick; return the next action row or None if dry."""
        self._tick += 1
        if self._queue:
            row = self._queue.popleft()
            if self.in_flight:
                self._executed_since_request += 1
            return row
        return None

    def should_request(self) -> bool:
        if self.in_flight:
            return False
        if self._chunk_len == 0 or not self._queue:
            return True
        consumed = 1.0 - len(self._queue) / self._chunk_len
        return consumed >= self.chunk_threshold

    def mark_requested(self) -> None:
        """Call at the tick the observation was captured."""
        self._request_tick = self._tick
        self._executed_since_request = 0

    def on_chunk(self, chunk: np.ndarray) -> int:
        """Install a fresh chunk, skipping the rows already executed during flight.

        Row 0 of the chunk is the action for the observation's tick. While the
        request was in flight, some rows of the *old* queue may have been
        executed (the arm moved); those many rows of the new chunk are stale
        and are skipped. If the queue ran dry during flight (the arm held
        position, unchanged from the observed state), no ticks were executed
        and no rows are skipped — held ticks don't consume rows.

        Returns the number of usable rows installed, so the caller can detect
        a fully-stale chunk (0 usable rows) and log/re-request.
        """
        skip = self._executed_since_request if self.in_flight else 0
        self._last_skip = skip
        self._request_tick = None
        rows = list(np.asarray(chunk))
        usable = rows[skip:]
        self._queue = deque(usable)
        self._chunk_len = len(usable)
        return len(usable)

    def on_request_failed(self) -> None:
        self._request_tick = None

    @property
    def consumed_rows(self) -> int:
        """Rows of the most recent chunk that are in the past, in FULL-chunk
        coordinates (install-skip included). Server-side leftover start index."""
        return self._last_skip + (self._chunk_len - len(self._queue))

    @property
    def last_skip(self) -> int:
        """Rows skipped at the last install == measured in-flight delay (ticks)."""
        return self._last_skip
