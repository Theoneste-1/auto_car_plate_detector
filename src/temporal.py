from __future__ import annotations

import time
from collections import Counter, deque
from typing import Deque

BUFFER_SIZE = 5
MIN_CONFIRMATIONS = 3
COOLDOWN_SECONDS = 10.0

class TemporalPlateTracker:
    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        min_confirmations: int = MIN_CONFIRMATIONS,
        cooldown_seconds: float = COOLDOWN_SECONDS,
    ) -> None:
        self.buffer: Deque[str] = deque(maxlen=buffer_size)
        self.min_confirmations = min_confirmations
        self.cooldown_seconds = cooldown_seconds
        self.last_saved_plate: str | None = None
        self.last_saved_time = 0.0

    def observe(self, plate: str, now: float | None = None) -> tuple[str | None, bool]:
        if not plate:
            return None, False

        current_time = now if now is not None else time.time()
        self.buffer.append(plate)
        most_common_plate, votes = Counter(self.buffer).most_common(1)[0]
        if votes < self.min_confirmations:
            return None, False

        should_log = (
            most_common_plate != self.last_saved_plate
            or (current_time - self.last_saved_time) >= self.cooldown_seconds
        )
        if should_log:
            self.last_saved_plate = most_common_plate
            self.last_saved_time = current_time
        return most_common_plate, should_log
