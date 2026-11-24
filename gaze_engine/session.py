"""Per-connection gaze session: calibration, validation, drift checks, live output.

Protocol (JSON control messages; frames arrive separately with their capture time):

  hello            {screen: {width, height}}                 -> ready
  target_start     {phase, id, x, y, t}                       (no reply)
  target_end       {id}                                       -> target_result
  fit              {}                                         -> fit_result
  validation_summary {}                                       -> validation_result
  reset            {}                                         -> ready

``phase`` is ``calibration``, ``validation`` or ``drift``. Every frame returns a
``gaze`` message. Nothing here touches the model, so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gaze_engine import config
from gaze_engine.calibration import (
    GazeMapping,
    TargetCapture,
    fit_mapping,
    quality_band,
    summarise_errors,
    summarise_target,
)
from gaze_engine.filters import OneEuroFilter2D
from gaze_engine.geometry import approx_visual_angle_deg
from gaze_engine.observation import FrameObservation

PHASES = ("calibration", "validation", "drift")


@dataclass
class _ActiveTarget:
    phase: str
    target_id: str
    target_px: tuple[float, float]
    shown_at_ms: float
    features: list[tuple[float, tuple[float, float], float | None]] = field(default_factory=list)
    predictions: list[tuple[float, tuple[float, float], float | None]] = field(default_factory=list)
    # Re-measuring a target replaces the earlier measurement at the same location.
    replaces: str | None = None


class GazeSession:
    def __init__(self) -> None:
        self.screen: tuple[int, int] | None = None
        self.reset()

    def reset(self) -> None:
        self.captures: dict[str, TargetCapture] = {}
        self.mapping: GazeMapping | None = None
        self.fit_report: dict[str, Any] | None = None
        self.validation_errors: dict[str, float] = {}
        self.validation: dict[str, Any] | None = None
        self.active: _ActiveTarget | None = None
        self.filter = OneEuroFilter2D()
        self.openness_samples: list[float] = []
        self.reference_distance_m: float | None = None
        self.drift_corrections = 0

    # -- control messages -----------------------------------------------
    def handle_control(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        kind = message.get("type")
        if kind == "hello":
            screen = message.get("screen") or {}
            width, height = int(screen.get("width", 0)), int(screen.get("height", 0))
            if width <= 0 or height <= 0:
                return [_error("hello needs screen.width and screen.height")]
            self.screen = (width, height)
            self.reset()
            return [{"type": "ready", "screen": {"width": width, "height": height}}]
        if kind == "reset":
            self.reset()
            return [{"type": "ready"}]
        if kind == "target_start":
            phase = message.get("phase")
            if phase not in PHASES:
                return [_error(f"unknown phase {phase!r}")]
            if phase != "calibration" and self.mapping is None:
                return [_error("fit a calibration before validation or drift targets")]
            try:
                target = (float(message["x"]), float(message["y"]))
                shown_at = float(message["t"])
            except (KeyError, TypeError, ValueError):
                return [_error("target_start needs numeric x, y and t")]
            replaces = message.get("replaces")
            self.active = _ActiveTarget(
                phase, str(message.get("id", "")), target, shown_at, replaces=str(replaces) if replaces else None
            )
            if phase == "calibration":
                self.filter.reset()
            return []
        if kind == "target_end":
            return [self._finish_target(str(message.get("id", "")))]
        if kind == "fit":
            return [self._fit()]
        if kind == "validation_summary":
            return [self._validation_summary()]
        return [_error(f"unknown message type {kind!r}")]

    def _finish_target(self, target_id: str) -> dict[str, Any]:
        active, self.active = self.active, None
        if active is None or active.target_id != target_id:
            return {"type": "target_result", "id": target_id, "accepted": False, "reason": "no_active_target"}
        base = {"type": "target_result", "id": target_id, "phase": active.phase}
        if active.phase == "calibration":
            summary, reason = summarise_target(active.features, active.shown_at_ms)
            if summary is None:
                return {**base, "accepted": False, "reason": reason, "n": len(active.features)}
            feature, n, dispersion, distance = summary
            if active.replaces:
                self.captures.pop(active.replaces, None)
            self.captures[target_id] = TargetCapture(target_id, active.target_px, feature, n, dispersion, distance)
            return {**base, "accepted": True, "reason": "ok", "n": n, "dispersion_mm": round(dispersion * 1000, 1)}

        summary, reason = summarise_target(
            active.predictions,
            active.shown_at_ms,
            # Predictions are already in pixels; reuse the same settle/trim
            # rules but with a pixel-sized dispersion limit (~3 deg at 60 cm).
            max_dispersion_m=float("inf"),
        )
        if summary is None:
            return {**base, "accepted": False, "reason": reason, "n": len(active.predictions)}
        predicted, n, _dispersion, distance = summary
        offset = np.asarray(active.target_px) - np.asarray(predicted)
        error = float(np.linalg.norm(offset))
        result = {
            **base,
            "accepted": True,
            "reason": "ok",
            "n": n,
            "error_px": round(error, 1),
            "predicted": [round(predicted[0], 1), round(predicted[1], 1)],
            "approx_error_deg": _round(approx_visual_angle_deg(error, distance, config.ASSUMED_CSS_PX_PER_CM)),
        }
        if active.phase == "validation":
            if active.replaces:
                result["replaced_error_px"] = _round(self.validation_errors.pop(active.replaces, None), 1)
            self.validation_errors[target_id] = error
            return result
        # Drift check: small offsets are corrected in place, large ones need a new calibration.
        if error <= config.DRIFT_CORRECT_LIMIT_PX and self.mapping is not None:
            self.mapping.offset = self.mapping.offset + config.DRIFT_BLEND * offset
            self.drift_corrections += 1
            self.filter.reset()
            result["action"] = "corrected"
        else:
            result["action"] = "recalibrate"
        return result

    def _fit(self) -> dict[str, Any]:
        try:
            self.mapping, self.fit_report = fit_mapping(list(self.captures.values()))
        except ValueError as exc:
            self.mapping = None
            return {"type": "fit_result", "ok": False, "reason": str(exc), "points": len(self.captures)}
        distances = [c.face_distance_m for c in self.captures.values() if c.face_distance_m]
        self.reference_distance_m = float(np.median(distances)) if distances else None
        self.validation_errors = {}
        self.validation = None
        self.filter.reset()
        return {"type": "fit_result", "ok": True, **self.fit_report}

    def _validation_summary(self) -> dict[str, Any]:
        stats = summarise_errors(self.validation_errors.values())
        quality = quality_band(stats["median_px"], stats["p90_px"])
        self.validation = {**stats, "quality": quality}
        return {
            "type": "validation_result",
            **self.validation,
            "approx_median_deg": _round(
                approx_visual_angle_deg(stats["median_px"] or 0.0, self.reference_distance_m, config.ASSUMED_CSS_PX_PER_CM)
            ) if stats["median_px"] is not None else None,
            "calibration": self.fit_report,
            "errors": {k: round(v, 1) for k, v in self.validation_errors.items()},
        }

    # -- frames -------------------------------------------------------------
    def _openness_baseline(self) -> float | None:
        if len(self.openness_samples) < 15:
            return None
        return float(np.percentile(self.openness_samples, 75))

    def _is_blink(self, openness: float | None) -> bool:
        if openness is None:
            return False
        if openness < config.ABSOLUTE_MIN_OPENNESS:
            return True
        baseline = self._openness_baseline()
        return baseline is not None and openness < config.BLINK_RATIO * baseline

    def handle_frame(self, observation: FrameObservation, t_ms: float) -> dict[str, Any]:
        reply: dict[str, Any] = {"type": "gaze", "t": t_ms, "valid": False, "reason": observation.reason}
        if observation.face_box is not None:
            reply["face_box"] = [round(v, 4) for v in observation.face_box]
        if observation.head_yaw_deg is not None:
            reply["yaw_deg"] = round(observation.head_yaw_deg, 1)
            reply["pitch_deg"] = round(observation.head_pitch_deg, 1)
        if observation.face_distance_m is not None:
            reply["distance_cm"] = round(observation.face_distance_m * 100, 1)
            if self.reference_distance_m:
                change = abs(observation.face_distance_m - self.reference_distance_m) / self.reference_distance_m
                reply["head_moved"] = change > config.DISTANCE_WARNING_RATIO
        if not observation.usable:
            self.filter.reset()
            return reply
        if self._is_blink(observation.eye_openness):
            reply["reason"] = "blink"
            self.filter.reset()
            return reply
        if observation.eye_openness is not None and len(self.openness_samples) < 600:
            self.openness_samples.append(observation.eye_openness)

        point = observation.point_m
        if self.active is not None and self.active.phase == "calibration":
            self.active.features.append((t_ms, point, observation.face_distance_m))

        reply["valid"] = True
        reply["reason"] = "ok"
        if self.mapping is None:
            return reply
        raw = self.mapping.predict(point)
        raw_xy = (float(raw[0]), float(raw[1]))
        if self.active is not None and self.active.phase != "calibration":
            self.active.predictions.append((t_ms, raw_xy, observation.face_distance_m))
        fx, fy = self.filter(t_ms / 1000.0, raw_xy)
        reply.update(raw_x=round(raw_xy[0], 1), raw_y=round(raw_xy[1], 1), x=round(fx, 1), y=round(fy, 1))
        return reply


def _error(detail: str) -> dict[str, Any]:
    return {"type": "error", "detail": detail}


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)
