"""One-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

Strong smoothing while the gaze is still (fixations), little lag during fast
movement (saccades) — unlike the fixed 3-sample moving average it replaces.
"""

from __future__ import annotations

import math

from gaze_engine import config


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter2D:
    def __init__(
        self,
        min_cutoff_hz: float = config.FILTER_MIN_CUTOFF_HZ,
        beta: float = config.FILTER_BETA,
        d_cutoff_hz: float = config.FILTER_D_CUTOFF_HZ,
        reset_gap_s: float = 0.35,
    ) -> None:
        self.min_cutoff = min_cutoff_hz
        self.beta = beta
        self.d_cutoff = d_cutoff_hz
        self.reset_gap = reset_gap_s
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._x: list[float] | None = None
        self._dx = [0.0, 0.0]

    def __call__(self, t_s: float, point: tuple[float, float]) -> tuple[float, float]:
        if self._t is None or self._x is None or t_s <= self._t or t_s - self._t > self.reset_gap:
            self._t, self._x, self._dx = t_s, [float(point[0]), float(point[1])], [0.0, 0.0]
            return point[0], point[1]
        dt = t_s - self._t
        out = []
        for i in range(2):
            dx = (point[i] - self._x[i]) / dt
            a_d = _alpha(self.d_cutoff, dt)
            self._dx[i] = a_d * dx + (1 - a_d) * self._dx[i]
            cutoff = self.min_cutoff + self.beta * abs(self._dx[i])
            a = _alpha(cutoff, dt)
            out.append(a * point[i] + (1 - a) * self._x[i])
        self._t, self._x = t_s, out
        return out[0], out[1]
