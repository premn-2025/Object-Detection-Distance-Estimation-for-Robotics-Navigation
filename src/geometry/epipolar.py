"""Epipolar geometry, the disparity-depth relation, and motion stereo.

**The relation, derived.**  Two identical pinhole cameras, optical axes parallel,
separated by a baseline ``B`` along the ``X`` axis, focal length ``f`` px.  A
world point at depth ``Z`` and lateral offset ``X`` projects to

    u_L = f * X / Z + cx          u_R = f * (X - B) / Z + cx

so the disparity is

    d = u_L - u_R = f * B / Z      =>      Z = f * B / d

Depth is inversely proportional to disparity, which is the single most important
fact about stereo: differentiating gives

    dZ/dd = -f * B / d^2 = -Z^2 / (f * B)

so a fixed disparity error of ``sigma_d`` pixels becomes a range error that grows
with the **square** of the range,

    sigma_Z = Z^2 * sigma_d / (f * B)

A 12 cm baseline with f = 700 px and quarter-pixel matching gives ~4 cm error at
5 m and ~1.7 m at 30 m.  This is the same quadratic blow-up the monocular
ground-plane cue suffers in :mod:`src.distance.estimator`, and for the same
structural reason: both recover depth from a small image-space difference.

**Why this matters here.**  The datasets used in this project are monocular, so
there is no second camera.  But a moving camera is a stereo rig spread over time:
consecutive frames have a baseline equal to how far the robot travelled.  That
gives a genuinely independent check on the size-prior distances - see
:func:`motion_stereo_depth`.  The catch is that structure-from-motion recovers
translation only up to scale, so the baseline has to come from somewhere else
(wheel odometry, IMU, or - as in :func:`scale_from_known_height` - the very size
prior we were trying to check).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# the textbook relation
# --------------------------------------------------------------------------- #
def disparity_to_depth(disparity_px, focal_px: float, baseline_m: float):
    """``Z = f * B / d``. Non-positive disparity -> inf."""
    d = np.asarray(disparity_px, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = focal_px * baseline_m / d
    return np.where(d > 0, z, np.inf)


def depth_to_disparity(depth_m, focal_px: float, baseline_m: float):
    z = np.asarray(depth_m, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, focal_px * baseline_m / z, np.inf)


def depth_uncertainty(depth_m, focal_px: float, baseline_m: float,
                      disparity_sigma_px: float = 0.25):
    """``sigma_Z = Z^2 * sigma_d / (f * B)`` - the quadratic growth of stereo error."""
    z = np.asarray(depth_m, np.float64)
    return z ** 2 * disparity_sigma_px / (focal_px * baseline_m)


def max_useful_range(focal_px: float, baseline_m: float,
                     disparity_sigma_px: float = 0.25,
                     relative_error: float = 0.10) -> float:
    """Range at which stereo error exceeds ``relative_error`` of the range."""
    return relative_error * focal_px * baseline_m / disparity_sigma_px


# --------------------------------------------------------------------------- #
# fundamental / essential matrices
# --------------------------------------------------------------------------- #
@dataclass
class RelativePose:
    R: np.ndarray            # 3x3 rotation, frame 1 -> frame 2
    t: np.ndarray            # 3x1 unit translation (scale is unobservable)
    E: np.ndarray            # essential matrix
    F: np.ndarray            # fundamental matrix
    pts1: np.ndarray         # inlier correspondences
    pts2: np.ndarray
    n_inliers: int


def match_features(img1: np.ndarray, img2: np.ndarray, max_features: int = 4000,
                   ratio: float = 0.75) -> tuple:
    """ORB + Lowe ratio test. Returns matched point arrays ``(N,2)``."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    orb = cv2.ORB_create(max_features)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def estimate_relative_pose(img1: np.ndarray, img2: np.ndarray, K: np.ndarray,
                           ratio: float = 0.75) -> RelativePose:
    """Essential-matrix pose between two views of a rigid scene."""
    p1, p2 = match_features(img1, img2, ratio=ratio)
    if len(p1) < 8:
        raise RuntimeError(f"only {len(p1)} matches; need >= 8")
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("essential matrix estimation failed")
    mask = mask.ravel().astype(bool)
    p1i, p2i = p1[mask], p2[mask]
    _, R, t, _ = cv2.recoverPose(E, p1i, p2i, K)
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv
    return RelativePose(R=R, t=t, E=E, F=F, pts1=p1i, pts2=p2i, n_inliers=int(mask.sum()))


def epipolar_lines(F: np.ndarray, pts: np.ndarray, which: int = 1) -> np.ndarray:
    """Lines ``(a, b, c)`` in the *other* image for each point."""
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.computeCorrespondEpilines(p, which, F).reshape(-1, 3)


def symmetric_epipolar_error(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Per-match distance to the epipolar line, in pixels - a geometry health check."""
    p1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    p2 = np.hstack([pts2, np.ones((len(pts2), 1))])
    l2 = (F @ p1.T).T           # lines in image 2
    l1 = (F.T @ p2.T).T         # lines in image 1
    d2 = np.abs(np.sum(l2 * p2, axis=1)) / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)
    d1 = np.abs(np.sum(l1 * p1, axis=1)) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)
    return 0.5 * (d1 + d2)


# --------------------------------------------------------------------------- #
# motion stereo
# --------------------------------------------------------------------------- #
def triangulate(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate matches into frame-1 coordinates, up to the scale of ``t``."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
    X = (X[:3] / np.where(np.abs(X[3]) < 1e-12, 1e-12, X[3])).T
    return X


def motion_stereo_depth(img1: np.ndarray, img2: np.ndarray, K: np.ndarray,
                        baseline_m: float = None) -> dict:
    """Depth for matched features between two frames of a moving monocular camera.

    Without ``baseline_m`` the depths are relative (scale-free), which is still
    enough to check the *ordering* and the *ratios* of the monocular estimates.
    """
    pose = estimate_relative_pose(img1, img2, K)
    X = triangulate(K, pose.R, pose.t, pose.pts1, pose.pts2)
    z = X[:, 2]
    valid = np.isfinite(z) & (z > 0)
    scale = float(baseline_m) if baseline_m else 1.0
    return {
        "points_2d": pose.pts1[valid],
        "depth": z[valid] * scale,
        "scaled": baseline_m is not None,
        "n_inliers": pose.n_inliers,
        "median_epipolar_error_px": float(np.median(
            symmetric_epipolar_error(pose.F, pose.pts1, pose.pts2))),
        "R": pose.R, "t": pose.t, "F": pose.F,
    }


def scale_from_known_height(relative_depth: float, absolute_depth_m: float) -> float:
    """The scale factor that ties a scale-free reconstruction to metres.

    On a real robot this comes from odometry.  Here it can come from one object
    whose physical height is known - which is exactly the cue
    :mod:`src.distance.estimator` already uses.
    """
    if relative_depth <= 0:
        raise ValueError("relative depth must be positive")
    return absolute_depth_m / relative_depth


# --------------------------------------------------------------------------- #
# rectified block matching, for an actual stereo rig
# --------------------------------------------------------------------------- #
def rectify_pair(K1, d1, K2, d2, R, T, image_size):
    """``cv2.stereoRectify`` wrapper returning the two remap pairs and ``Q``."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2, image_size, R, T,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(K1, d1, R1, P1, image_size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K2, d2, R2, P2, image_size, cv2.CV_32FC1)
    return m1, m2, Q


def sgbm_disparity(left: np.ndarray, right: np.ndarray, *, num_disparities: int = 128,
                   block_size: int = 5) -> np.ndarray:
    """Semi-global block matching disparity in pixels (float32)."""
    if num_disparities % 16:
        raise ValueError("num_disparities must be a multiple of 16")
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return matcher.compute(gl, gr).astype(np.float32) / 16.0


def stereo_range_table(focal_px: float, baseline_m: float,
                       ranges=(2, 5, 10, 20, 30, 50),
                       disparity_sigma_px: float = 0.25) -> list:
    """Disparity and range uncertainty at a few depths - the design trade-off table."""
    rows = []
    for z in ranges:
        d = focal_px * baseline_m / z
        s = depth_uncertainty(z, focal_px, baseline_m, disparity_sigma_px)
        rows.append({"range_m": float(z), "disparity_px": float(d),
                     "sigma_m": float(s), "relative_error": float(s / z)})
    return rows
