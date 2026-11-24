import math

import numpy as np
import pytest

from gaze_engine.backend import observation_from_geometry
from gaze_engine.calibration import GazeMapping, TargetCapture, fit_mapping, quality_band, summarise_target
from gaze_engine.filters import OneEuroFilter2D
from gaze_engine.geometry import camera_matrix, eye_openness, intersect_camera_plane
from gaze_engine.observation import FrameObservation
from gaze_engine.session import GazeSession


def test_camera_matrix_uses_field_of_view():
    k = camera_matrix(1280, 720, 90.0)
    assert k[0, 0] == pytest.approx(640.0)
    assert k[0, 2] == 640 and k[1, 2] == 360
    with pytest.raises(ValueError):
        camera_matrix(0, 720, 70)


def test_plane_intersection_is_independent_of_head_position():
    """The property the old iris-offset model lacked: same screen point, different heads."""
    screen_point = np.array([0.05, 0.12, 0.0])
    for head in ([0.0, 0.0, 0.6], [0.08, -0.03, 0.5], [-0.1, 0.05, 0.75]):
        origin = np.array(head)
        hit = intersect_camera_plane(origin, screen_point - origin)
        assert hit == pytest.approx((0.05, 0.12), abs=1e-9)


def test_plane_intersection_rejects_rays_that_miss():
    assert intersect_camera_plane([0, 0, 0.6], [0, 0, 1]) is None  # looking away from screen
    assert intersect_camera_plane([0, 0, 0.6], [1, 0, -0.05]) is None  # grazing
    assert intersect_camera_plane([0, 0, 0.6], [0, 0, 0]) is None


def test_eye_openness_from_landmarks():
    pts = np.zeros((468, 2))
    for (c1, c2), (v1, v2) in (((33, 133), (159, 145)), ((362, 263), (386, 374))):
        pts[c1], pts[c2] = (0, 0), (10, 0)
        pts[v1], pts[v2] = (5, -1.5), (5, 1.5)
    assert eye_openness(pts) == pytest.approx(0.3)
    assert eye_openness(np.zeros((10, 2))) is None


def test_observation_from_geometry_filters_bad_distance_and_direction():
    ok = observation_from_geometry(np.array([0, 0, 0.6]), np.array([0, 0.1, -1.0]), np.zeros(3), np.zeros(2), 0.3)
    assert ok.usable and ok.point_m[1] == pytest.approx(0.06)
    far = observation_from_geometry(np.array([0, 0, 3.0]), np.array([0, 0, -1.0]), np.zeros(3), np.zeros(2), 0.3)
    assert far.reason == "bad_distance" and not far.usable
    away = observation_from_geometry(np.array([0, 0, 0.6]), np.array([0, 0, 1.0]), np.zeros(3), np.zeros(2), 0.3)
    assert away.reason == "looking_away"


def test_target_summary_drops_the_saccade_at_the_start():
    # First 600 ms the eye is still travelling from the previous target.
    samples = [(t, (0.2, 0.2), 0.6) for t in range(0, 600, 50)]
    samples += [(t, (0.01, 0.02), 0.6) for t in range(600, 1900, 50)]
    summary, reason = summarise_target(samples, shown_at_ms=0)
    assert reason == "ok"
    feature, n, dispersion, distance = summary
    assert feature == pytest.approx((0.01, 0.02))
    assert n == 26 and dispersion == pytest.approx(0) and distance == 0.6


def test_target_summary_rejects_wandering_gaze():
    rng = np.random.default_rng(1)
    samples = [(600 + 50 * i, tuple(rng.normal(0, 0.08, 2)), None) for i in range(25)]
    summary, reason = summarise_target(samples, shown_at_ms=0)
    assert summary is None and reason == "unstable_fixation"
    assert summarise_target([(700, (0, 0), None)] * 2, 0)[1] == "too_few_samples"


def _synthetic_grid(points=13):
    ratios = [(.1, .1), (.5, .1), (.9, .1), (.1, .5), (.5, .5), (.9, .5), (.1, .9), (.5, .9), (.9, .9),
              (.3, .3), (.7, .3), (.3, .7), (.7, .7)][:points]
    return [(1440 * x, 900 * y) for x, y in ratios]


def _screen_to_plane(px):
    # A plausible true geometry: 1 px ~ 0.25 mm, screen below the camera, slight kappa offset.
    return (px[0] - 720) * 0.00025 + 0.004, px[1] * 0.00025 + 0.015


def test_fit_mapping_recovers_geometry_and_drops_a_bad_target():
    rng = np.random.default_rng(0)
    targets = _synthetic_grid()
    captures = []
    for i, target in enumerate(targets):
        feature = np.asarray(_screen_to_plane(target)) + rng.normal(0, 0.0015, 2)
        if i == 4:
            feature = feature + 0.08  # participant looked away on this target
        captures.append(TargetCapture(str(i), target, tuple(feature), 20, 0.003))
    mapping, report = fit_mapping(captures)
    assert report["dropped_targets"] == ["4"]
    assert report["cv_median_px"] < 25
    pred = mapping.predict(_screen_to_plane((1000, 400)))
    assert np.linalg.norm(pred - np.array([1000, 400])) < 25


def test_fit_mapping_needs_enough_targets():
    with pytest.raises(ValueError):
        fit_mapping([TargetCapture("a", (0, 0), (0, 0), 10, 0.0)] * 3)


def test_quality_bands():
    assert quality_band(80, 150) == "good"
    assert quality_band(150, 300) == "fair"
    assert quality_band(250, 400) == "poor"
    assert quality_band(None, None) == "poor"


def test_one_euro_filter_smooths_fixation_and_follows_saccade():
    rng = np.random.default_rng(3)
    f = OneEuroFilter2D()
    raw, out = [], []
    for i in range(60):
        point = (500 + rng.normal(0, 30), 300 + rng.normal(0, 30))
        raw.append(point)
        out.append(f(i / 20, point))
    assert np.std(np.asarray(out[10:])[:, 0]) < 0.6 * np.std(np.asarray(raw[10:])[:, 0])
    for i in range(60, 75):
        last = f(i / 20, (1200, 300))
    assert abs(last[0] - 1200) < 20  # caught up with the saccade


def _frame(point, distance=0.6, openness=0.3):
    return FrameObservation(face_detected=True, point_m=point, face_distance_m=distance, eye_openness=openness)


def _show(session, phase, target_id, target, start_ms, point_fn):
    session.handle_control({"type": "target_start", "phase": phase, "id": target_id, "x": target[0], "y": target[1], "t": start_ms})
    for t in range(int(start_ms), int(start_ms) + 2000, 60):
        session.handle_frame(_frame(point_fn(t)), t)
    return session.handle_control({"type": "target_end", "id": target_id})[0]


def test_session_end_to_end_calibration_validation_and_drift():
    rng = np.random.default_rng(7)
    s = GazeSession()
    assert s.handle_control({"type": "hello", "screen": {"width": 1440, "height": 900}})[0]["type"] == "ready"
    clock = 0
    for i, target in enumerate(_synthetic_grid()):
        truth = np.asarray(_screen_to_plane(target))
        result = _show(s, "calibration", f"c{i}", target, clock, lambda t: tuple(truth + rng.normal(0, 0.001, 2)))
        assert result["accepted"], result
        clock += 2500
    fit = s.handle_control({"type": "fit"})[0]
    assert fit["ok"] and fit["cv_median_px"] < 30

    for i, target in enumerate([(288, 198), (720, 198), (1152, 702), (288, 702)]):
        truth = np.asarray(_screen_to_plane(target))
        result = _show(s, "validation", f"v{i}", target, clock, lambda t: tuple(truth + rng.normal(0, 0.001, 2)))
        assert result["accepted"] and result["error_px"] < 30
        clock += 2500
    summary = s.handle_control({"type": "validation_summary"})[0]
    assert summary["quality"] == "good" and summary["n"] == 4

    # The participant's mapping drifts by ~100 px; a drift check corrects most of it.
    shift = np.array([0.025, 0.0])
    target = (720, 450)
    truth = np.asarray(_screen_to_plane(target)) + shift
    drift = _show(s, "drift", "d0", target, clock, lambda t: tuple(truth))
    assert drift["action"] == "corrected" and drift["error_px"] == pytest.approx(100, abs=5)
    after = s.mapping.predict(truth)
    assert np.linalg.norm(after - np.asarray(target)) < 40

    gaze = s.handle_frame(_frame(tuple(truth)), clock + 5000)
    assert gaze["valid"] and "x" in gaze and gaze["head_moved"] is False
    moved = s.handle_frame(_frame(tuple(truth), distance=0.85), clock + 5100)
    assert moved["head_moved"] is True


def test_session_rejects_blinks_relative_to_baseline_and_orders_phases():
    s = GazeSession()
    s.handle_control({"type": "hello", "screen": {"width": 100, "height": 100}})
    assert s.handle_control({"type": "target_start", "phase": "validation", "id": "x", "x": 1, "y": 1, "t": 0})[0]["type"] == "error"
    for i in range(20):
        assert s.handle_frame(_frame((0.0, 0.0), openness=0.32), i * 50)["valid"]
    blink = s.handle_frame(_frame((0.0, 0.0), openness=0.12), 2000)
    assert blink["reason"] == "blink" and not blink["valid"]
    no_face = s.handle_frame(FrameObservation(face_detected=False, reason="no_face"), 2100)
    assert no_face["reason"] == "no_face"


def test_positioning_fields_flow_through_to_the_gaze_reply():
    obs = observation_from_geometry(
        np.array([0, 0, 0.55]), np.array([0, 0.1, -1.0]), np.zeros(3), np.zeros(2), 0.3,
        face_box=(0.4, 0.3, 0.6, 0.6), head_angles=(-3.0, 8.0),
    )
    reply = GazeSession().handle_frame(obs, 0)
    assert reply["face_box"] == [0.4, 0.3, 0.6, 0.6]
    assert reply["yaw_deg"] == 8.0 and reply["pitch_deg"] == -3.0 and reply["distance_cm"] == 55.0
    # Pose is still reported when the frame is unusable (e.g. sitting too close), so the guide can say so.
    near = observation_from_geometry(np.array([0, 0, 0.2]), np.array([0, 0, -1.0]), np.zeros(3), np.zeros(2), 0.3,
                                     face_box=(0.2, 0.1, 0.8, 0.9), head_angles=(0, 0))
    assert near.reason == "bad_distance" and near.face_box is not None


def test_remeasured_targets_replace_their_earlier_measurement():
    rng = np.random.default_rng(11)
    s = GazeSession()
    s.handle_control({"type": "hello", "screen": {"width": 1440, "height": 900}})
    clock = 0
    grid = _synthetic_grid()
    for i, target in enumerate(grid):
        truth = np.asarray(_screen_to_plane(target))
        if i == 2:
            truth = truth + np.array([0.03, -0.02])  # looked slightly off this dot
        _show(s, "calibration", f"c{i}", target, clock, lambda t: tuple(truth + rng.normal(0, 0.0008, 2)))
        clock += 2500
    first = s.handle_control({"type": "fit"})[0]
    assert set(first["residuals"]) | set(first["dropped_targets"]) == {f"c{i}" for i in range(13)}
    worst = max(first["residuals"], key=first["residuals"].get) if not first["dropped_targets"] else first["dropped_targets"][0]
    assert worst == "c2"
    truth = np.asarray(_screen_to_plane(grid[2]))
    s.handle_control({"type": "target_start", "phase": "calibration", "id": "c2-redo", "replaces": "c2",
                      "x": grid[2][0], "y": grid[2][1], "t": clock})
    for t in range(clock, clock + 2000, 60):
        s.handle_frame(_frame(tuple(truth)), t)
    s.handle_control({"type": "target_end", "id": "c2-redo"})
    assert "c2" not in s.captures and "c2-redo" in s.captures
    second = s.handle_control({"type": "fit"})[0]
    assert second["cv_median_px"] <= first["cv_median_px"] + 1 and not second["dropped_targets"]

    # A validation dot the participant missed is re-measured, not averaged in.
    clock += 5000
    target = (720, 450)
    away = np.asarray(_screen_to_plane((100, 800)))
    bad = _show(s, "validation", "v0", target, clock, lambda t: tuple(away))
    assert bad["error_px"] > 500
    s.handle_control({"type": "target_start", "phase": "validation", "id": "v0-r", "replaces": "v0", "x": 720, "y": 450, "t": clock + 3000})
    good_truth = np.asarray(_screen_to_plane(target))
    for t in range(clock + 3000, clock + 5000, 60):
        s.handle_frame(_frame(tuple(good_truth)), t)
    retry = s.handle_control({"type": "target_end", "id": "v0-r"})[0]
    assert retry["replaced_error_px"] > 500 and retry["error_px"] < 30
    assert list(s.validation_errors) == ["v0-r"]
