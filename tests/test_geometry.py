"""Geometry tests.

These are the parts that cannot be checked by looking at a picture: a sign error
in the pitch convention or a transposed rotation produces distances that look
plausible and are wrong everywhere.  Each test builds a synthetic scene with
known ground truth and checks the code recovers it.

    python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distance.calibrate import CalibrationSample, fit_extrinsics  # noqa: E402
from src.distance.camera import CameraModel  # noqa: E402
from src.distance.estimator import DistanceEstimator, ObjectPrior  # noqa: E402
from src.geometry.bev import BEVTransform  # noqa: E402
from src.geometry.epipolar import (depth_to_disparity, depth_uncertainty,  # noqa: E402
                                   disparity_to_depth, max_useful_range)


def make_camera(pitch_deg=0.0, height=1.3):
    return CameraModel.from_hfov(1920, 1080, 60.0, camera_height_m=height,
                                 pitch_deg=pitch_deg)


def project(cam, x, y, z):
    """Project a level-frame point to pixels using the forward model."""
    t = cam.pitch_rad
    c, s = math.cos(t), math.sin(t)
    p_c = np.array([x, y * c - z * s, y * s + z * c])
    return cam.fx * p_c[0] / p_c[2] + cam.cx, cam.fy * p_c[1] / p_c[2] + cam.cy


# --------------------------------------------------------------------------- #
# camera model
# --------------------------------------------------------------------------- #
def test_fx_matches_fov():
    cam = make_camera()
    assert cam.fx == pytest.approx(960.0 / math.tan(math.radians(30.0)), rel=1e-9)
    assert cam.hfov_deg == pytest.approx(60.0, abs=1e-9)


@pytest.mark.parametrize("pitch", [-6.0, -2.0, 0.0, 3.0, 8.0])
def test_ground_range_roundtrip(pitch):
    """Project a known ground point, then recover its range from its image row."""
    cam = make_camera(pitch_deg=pitch)
    for z_true in (3.0, 7.5, 15.0, 30.0, 60.0):
        _, v = project(cam, 0.0, cam.camera_height_m, z_true)
        z_est = cam.ground_plane_range(v, cam.camera_height_m)
        assert z_est == pytest.approx(z_true, rel=1e-6), f"pitch={pitch} z={z_true}"


@pytest.mark.parametrize("pitch", [-5.0, 0.0, 4.0])
def test_forward_to_row_is_inverse(pitch):
    cam = make_camera(pitch_deg=pitch)
    for z in (2.0, 10.0, 40.0):
        v = cam.forward_to_row(z, cam.camera_height_m)
        assert cam.ground_plane_range(v, cam.camera_height_m) == pytest.approx(z, rel=1e-6)


@pytest.mark.parametrize("pitch", [-7.0, 0.0, 5.0])
def test_horizon_is_the_limit_of_the_ground_plane(pitch):
    cam = make_camera(pitch_deg=pitch)
    v_far = cam.forward_to_row(1e7, cam.camera_height_m)
    assert v_far == pytest.approx(cam.horizon_row, abs=1e-3)
    # just above the horizon the plane is not intersected in front of the camera
    assert not math.isfinite(cam.ground_plane_range(cam.horizon_row - 1e-3,
                                                    cam.camera_height_m))


def test_pitch_down_raises_the_horizon():
    """Positive pitch is nose-down, so the horizon moves up the image (smaller v)."""
    assert make_camera(pitch_deg=5.0).horizon_row < make_camera(pitch_deg=0.0).horizon_row


def test_backprojection_inverts_projection():
    cam = make_camera(pitch_deg=3.0)
    x, y, z = 2.0, 1.3, 12.0
    u, v = project(cam, x, y, z)
    t = cam.pitch_rad
    depth_zc = y * math.sin(t) + z * math.cos(t)
    p = cam.backproject(u, v, depth_zc)
    assert p == pytest.approx([x, y, z], rel=1e-6)


def test_rescaled_camera_gives_the_same_range():
    cam = make_camera(pitch_deg=2.0)
    _, v = project(cam, 0.0, cam.camera_height_m, 20.0)
    half = cam.rescaled(960, 540)
    assert half.ground_plane_range(v / 2, half.camera_height_m) == pytest.approx(20.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# distance estimator
# --------------------------------------------------------------------------- #
def synthetic_box(cam, z_true, height_m, lateral=0.0):
    """Pixel box of an upright object standing on the road at range ``z_true``."""
    u_b, v_b = project(cam, lateral, cam.camera_height_m, z_true)
    u_t, v_t = project(cam, lateral, cam.camera_height_m - height_m, z_true)
    half_w = 0.5 * abs(v_b - v_t)
    return [u_b - half_w, v_t, u_b + half_w, v_b], (u_b, v_b)


def make_estimator(cam, **prior_kw):
    priors = {"cone": ObjectPrior("cone", height_m=0.75, height_sigma_m=0.14,
                                  ground_contact=True, **prior_kw)}
    return DistanceEstimator(cam, priors)


@pytest.mark.parametrize("pitch", [0.0, 3.0, -3.0])
@pytest.mark.parametrize("z_true", [5.0, 10.0, 25.0])
def test_both_cues_recover_a_perfect_synthetic_cone(pitch, z_true):
    """With an exact box and an exact prior, both cues must land on the truth."""
    cam = make_camera(pitch_deg=pitch)
    box, _ = synthetic_box(cam, z_true, 0.75)
    est = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert est.valid
    # euclidean range to the base point, not the horizontal range
    expected = math.hypot(math.hypot(0.0, cam.camera_height_m), z_true)
    assert est.plane_m == pytest.approx(expected, rel=2e-3)
    assert est.size_m == pytest.approx(expected, rel=2e-2)
    assert est.distance_m == pytest.approx(expected, rel=2e-2)
    assert est.method == "fused"


def test_size_cue_scales_with_the_prior():
    """A cone assumed twice as tall must be reported twice as far."""
    cam = make_camera()
    box, _ = synthetic_box(cam, 10.0, 0.75)
    a = DistanceEstimator(cam, {"cone": ObjectPrior("cone", height_m=0.75)},
                          fusion="size_only").estimate(box, "cone", (1080, 1920))
    b = DistanceEstimator(cam, {"cone": ObjectPrior("cone", height_m=1.50)},
                          fusion="size_only").estimate(box, "cone", (1080, 1920))
    assert b.distance_m / a.distance_m == pytest.approx(2.0, rel=1e-6)


def test_plane_cue_uncertainty_grows_faster_than_linearly():
    """sigma_Z/Z for the ground cue should grow with range - the quadratic blow-up."""
    cam = make_camera()
    est = DistanceEstimator(cam, {"cone": ObjectPrior("cone", height_m=0.75)},
                            fusion="ground_only")
    rel = []
    for z in (5.0, 15.0, 40.0):
        box, _ = synthetic_box(cam, z, 0.75)
        e = est.estimate(box, "cone", (1080, 1920))
        rel.append(e.sigma_m / e.distance_m)
    assert rel[0] < rel[1] < rel[2]


def test_size_cue_relative_error_has_a_floor_from_the_prior():
    """Far away, the size cue's relative error tends to sigma_H/H, not to zero."""
    cam = make_camera()
    est = DistanceEstimator(cam, {"cone": ObjectPrior("cone", height_m=0.75,
                                                      height_sigma_m=0.15)},
                            fusion="size_only")
    box, _ = synthetic_box(cam, 30.0, 0.75)
    e = est.estimate(box, "cone", (1080, 1920))
    assert e.sigma_m / e.distance_m > 0.15


def test_truncated_box_falls_back_to_the_plane_cue():
    cam = make_camera()
    box, _ = synthetic_box(cam, 6.0, 0.75)
    box[1] = 1.0                       # object runs off the top of the frame
    e = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert e.valid and e.method == "plane"
    assert "size:truncated" in e.flags


def test_box_running_off_the_bottom_edge_yields_no_range():
    """If the base is cut off, neither cue is trustworthy and nothing is reported.

    The plane cue cannot see where the object meets the road, and the box height
    is then only a lower bound, so the size cue is biased long. Returning
    ``valid=False`` is the correct answer, not a guess.
    """
    cam = make_camera()
    box, _ = synthetic_box(cam, 3.0, 0.75)
    box[3] = 1079.0                    # base is cut off by the frame
    e = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert not e.valid
    assert "plane:base_cut_off" in e.flags and "size:truncated" in e.flags


def test_near_field_blind_spot_is_real():
    """A forward-facing camera cannot see the road right in front of it.

    With h=1.3 m, fy=1662.8 and a 1080-row sensor the ground first becomes
    visible at h*fy/(H-cy) = 4.0 m. Anything closer is below the frame, which is
    a mounting constraint on the robot, not a bug in the estimator.
    """
    cam = make_camera()
    nearest = cam.camera_height_m * cam.fy / (cam.image_height - cam.cy)
    assert nearest == pytest.approx(4.00, abs=0.02)
    assert cam.forward_to_row(nearest, cam.camera_height_m) == pytest.approx(
        cam.image_height, abs=1e-6)


def test_object_above_the_horizon_is_rejected_by_the_plane_cue():
    cam = make_camera()
    box = [900.0, 100.0, 960.0, 200.0]         # entirely above the horizon
    e = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert "plane:above_horizon" in e.flags


def test_elevated_prior_uses_the_offset_plane():
    """A stop sign 2 m up must be ranged from a plane above the camera."""
    cam = make_camera()
    priors = {"stop_sign": ObjectPrior("stop_sign", height_m=0.79, height_sigma_m=0.09,
                                       ground_contact=False, mount_height_m=2.0,
                                       mount_height_sigma_m=0.45)}
    est = DistanceEstimator(cam, priors, fusion="ground_only")
    z_true, x_true = 12.0, 1.5
    y_true = cam.camera_height_m - 2.0          # sign foot, 2 m above the road
    u_b, v_b = project(cam, x_true, y_true, z_true)
    e = est.estimate([u_b - 30.0, v_b - 60.0, u_b + 30.0, v_b], "stop_sign", (1080, 1920))
    assert e.valid
    expected = math.sqrt(x_true ** 2 + y_true ** 2 + z_true ** 2)
    assert e.distance_m == pytest.approx(expected, rel=5e-3)
    assert e.forward_m == pytest.approx(z_true, rel=5e-3)


def test_lateral_offset_sign_and_magnitude():
    cam = make_camera()
    box, _ = synthetic_box(cam, 10.0, 0.75, lateral=2.5)
    e = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert e.lateral_m == pytest.approx(2.5, rel=0.03)
    assert e.forward_m == pytest.approx(10.0, rel=0.03)


def test_cue_disagreement_is_flagged():
    """A box half its true height should make the two cues disagree loudly."""
    cam = make_camera()
    box, _ = synthetic_box(cam, 10.0, 0.75)
    box[1] = 0.5 * (box[1] + box[3])        # halve the box height
    e = make_estimator(cam).estimate(box, "cone", (1080, 1920))
    assert any(f.startswith("cue_disagreement") for f in e.flags)


def test_unknown_class_is_invalid_not_a_crash():
    e = make_estimator(make_camera()).estimate([100, 100, 150, 200], "banana", (1080, 1920))
    assert not e.valid and e.flags == ["no_prior:banana"]


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_h,true_pitch", [(1.55, 2.5), (1.10, -1.5), (1.30, 0.0)])
def test_calibration_recovers_known_extrinsics(true_h, true_pitch):
    """Synthesise cones seen by a camera with known pose; the fit must find it."""
    truth = make_camera(pitch_deg=true_pitch, height=true_h)
    rng = np.random.default_rng(0)
    samples = []
    for z in rng.uniform(4.0, 35.0, 400):
        h_m = float(rng.normal(0.75, 0.05))       # real spread of cone heights
        box, _ = synthetic_box(truth, float(z), h_m)
        samples.append(CalibrationSample(v_bottom=box[3], h_px=box[3] - box[1],
                                         height_m=0.75))   # the *prior*, not the truth

    guess = make_camera(pitch_deg=0.0, height=1.3)
    report = fit_extrinsics(samples, guess, val_samples=samples)
    assert report["fitted"]["camera_height_m"] == pytest.approx(true_h, rel=0.05)
    assert report["fitted"]["pitch_deg"] == pytest.approx(true_pitch, abs=0.5)
    assert report["train_after"]["median_abs_log_ratio"] < 0.05


def test_calibration_needs_enough_samples():
    with pytest.raises(ValueError):
        fit_extrinsics([CalibrationSample(800.0, 100.0, 0.75)] * 10, make_camera())


# --------------------------------------------------------------------------- #
# epipolar / stereo
# --------------------------------------------------------------------------- #
def test_disparity_depth_roundtrip():
    f, B = 700.0, 0.12
    for z in (1.0, 5.0, 20.0, 50.0):
        d = depth_to_disparity(z, f, B)
        assert float(disparity_to_depth(d, f, B)) == pytest.approx(z, rel=1e-9)


def test_stereo_error_is_quadratic_in_range():
    f, B = 700.0, 0.12
    s10 = float(depth_uncertainty(10.0, f, B))
    s20 = float(depth_uncertainty(20.0, f, B))
    assert s20 / s10 == pytest.approx(4.0, rel=1e-9)


def test_zero_disparity_is_infinite_depth():
    assert math.isinf(float(disparity_to_depth(0.0, 700.0, 0.12)))


def test_max_useful_range_scales_with_baseline():
    a = max_useful_range(700.0, 0.12, 0.25, 0.10)
    b = max_useful_range(700.0, 0.24, 0.25, 0.10)
    assert b == pytest.approx(2.0 * a, rel=1e-9)


# --------------------------------------------------------------------------- #
# bird's-eye view
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pitch", [0.0, 3.0, -2.0])
def test_ipm_homography_matches_the_projection_model(pitch):
    """H_image_from_ground must agree with projecting the point directly."""
    cam = make_camera(pitch_deg=pitch)
    bev = BEVTransform(cam)
    H = bev.H_image_from_ground()
    for x, z in [(0.0, 10.0), (2.0, 5.0), (-3.0, 25.0)]:
        p = H @ np.array([x, z, 1.0])
        u, v = p[0] / p[2], p[1] / p[2]
        eu, ev = project(cam, x, cam.camera_height_m, z)
        assert (u, v) == pytest.approx((eu, ev), rel=1e-9)


def test_bev_canvas_places_metres_where_expected():
    cam = make_camera()
    bev = BEVTransform(cam, x_range=(-10, 10), z_range=(0, 40), px_per_m=10)
    assert bev.ground_to_bev(0.0, 40.0) == (100, 0)        # far edge, top centre
    assert bev.ground_to_bev(0.0, 0.0) == (100, 400)       # at the robot, bottom
    assert bev.ground_to_bev(10.0, 40.0) == (200, 0)       # right edge


def test_bev_image_to_ground_inverts():
    cam = make_camera(pitch_deg=2.0)
    bev = BEVTransform(cam)
    H_inv = np.linalg.inv(bev.H_image_from_ground())
    u, v = project(cam, 1.0, cam.camera_height_m, 18.0)
    p = H_inv @ np.array([u, v, 1.0])
    assert (p[0] / p[2], p[1] / p[2]) == pytest.approx((1.0, 18.0), rel=1e-6)
