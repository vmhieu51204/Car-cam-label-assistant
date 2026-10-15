"""Adaptive temporal batching: pick the window (frames per SAM3 session) that fits VRAM.

Every window is one SAM3 session. Memory grows with window length (the tracker keeps
per-frame outputs for the whole session), so:

* on CUDA OOM the same start frame is retried with a halved window;
* after ``grow_after`` consecutive windows whose peak stayed under ``grow_below`` of
  the VRAM budget, the window grows by ``grow_factor`` (capped at ``max_size``).

Consecutive windows overlap by ``overlap`` frames so tracks can be stitched.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class WindowTooSmallError(RuntimeError):
    """OOM even at ``min_size`` — lower --max-objects, enable offload, or lower resolution."""


@dataclass
class AdaptiveWindowScheduler:
    total: int
    size: int
    min_size: int = 8
    max_size: int = 160
    overlap: int = 4
    grow_below: float = 0.60
    grow_factor: float = 1.5
    grow_after: int = 2
    start: int = 0
    events: list[dict] = field(default_factory=list)
    _calm_windows: int = 0

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("no frames to schedule")
        self.min_size = max(2, self.min_size)
        self.max_size = max(self.min_size, self.max_size)
        self.size = min(max(self.size, self.min_size), self.max_size)
        # overlap must leave room for progress even at the smallest window
        self.overlap = max(0, min(self.overlap, self.min_size - 1))

    @property
    def done(self) -> bool:
        return self.start >= self.total

    def current(self) -> tuple[int, int]:
        """Half-open global frame range [start, end) for the next window."""
        return self.start, min(self.start + self.size, self.total)

    def on_success(self, end: int, peak_fraction: float | None) -> None:
        if end >= self.total:
            self.start = self.total
        else:
            self.start = max(end - self.overlap, self.start + 1)
        if peak_fraction is not None and peak_fraction < self.grow_below:
            self._calm_windows += 1
        else:
            self._calm_windows = 0
        if self._calm_windows >= self.grow_after and self.size < self.max_size:
            new_size = min(self.max_size, int(self.size * self.grow_factor))
            self.events.append({"type": "grow", "at_frame": self.start, "from": self.size, "to": new_size,
                                "peak_fraction": peak_fraction})
            self.size = new_size
            self._calm_windows = 0

    def on_oom(self) -> None:
        if self.size <= self.min_size:
            self.events.append({"type": "oom_at_min", "at_frame": self.start, "size": self.size})
            raise WindowTooSmallError(f"CUDA OOM at the minimum window size ({self.min_size} frames)")
        new_size = max(self.min_size, self.size // 2)
        self.events.append({"type": "oom_backoff", "at_frame": self.start, "from": self.size, "to": new_size})
        self.size = new_size
        self._calm_windows = 0
