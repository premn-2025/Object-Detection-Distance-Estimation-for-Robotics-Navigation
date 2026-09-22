"""Image corruptions for the robustness benchmark.

Deliberately implemented here from scratch with OpenCV rather than reusing the
albumentations stack from ``src/train/augment.py``. If the corruption used to
*test* robustness is literally the same code used to *train* it, the benchmark
measures how well the model memorised one library's noise generator, not whether
it survives bad conditions. Same spirit as ImageNet-C: train and test corruptions
must be different implementations.

Each corruption takes a severity in 1..5. Severity 3 is meant to be "clearly
degraded but a human still sees the cone"; severity 5 is meant to hurt.

The fog model is the interesting one. Rather than blending a flat grey over the
frame, it uses the Koschmieder atmospheric-scattering model

    I(v) = I0(v)·t(v) + A·(1 − t(v)),      t(v) = exp(−beta·Z(v))

with the depth ``Z(v)`` of each image row taken from the *same* ground-plane
camera model the distance estimator uses. Distant objects therefore fade more
than near ones, which is what real fog does and what makes it hard for a
detector — the cones that vanish first are exactly the small distant ones.
"""

from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def _rng(seed):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# lighting
# --------------------------------------------------------------------------- #
def low_light(img, severity=3, seed=0):
    """Dusk to night: gamma compression, exposure loss, and the read noise that
    a sensor adds once it starts pushing gain."""
    gain = [0.72, 0.55, 0.40, 0.28, 0.18][severity - 1]
    gamma = [1.2, 1.4, 1.7, 2.0, 2.4][severity - 1]
    noise = [2, 4, 7, 11, 16][severity - 1]
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gamma) * gain
    x = x * 255.0 + _rng(seed).normal(0, noise, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def glare(img, severity=3, seed=0):
    """Low sun into the lens: a bright radial source near the horizon plus the
    veiling haze it throws across the whole frame."""
    h, w = img.shape[:2]
    r = _rng(seed)
    cx = int(r.uniform(0.15, 0.85) * w)
    cy = int(r.uniform(0.15, 0.45) * h)          # near/above the horizon
    radius = [0.18, 0.26, 0.34, 0.44, 0.55][severity - 1] * w
    veil = [0.05, 0.10, 0.17, 0.26, 0.36][severity - 1]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = np.clip(1.0 - d / radius, 0, 1) ** 2
    out = img.astype(np.float32)
    out += (core[..., None] * 255.0 * 0.95)
    out = out * (1 - veil) + 255.0 * veil        # contrast-killing veiling glare
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# weather
# --------------------------------------------------------------------------- #
def fog(img, severity=3, seed=0, camera=None):
    """Depth-correct fog via the Koschmieder scattering model."""
    h, w = img.shape[:2]
    beta = [0.02, 0.04, 0.07, 0.11, 0.16][severity - 1]   # extinction, 1/m
    airlight = 235.0

    depth = _row_depth_map(h, w, camera)
    t = np.exp(-beta * depth).astype(np.float32)[..., None]
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def _row_depth_map(h, w, camera=None):
    """Metres of depth for each image row, from the ground-plane model.

    Above the horizon there is no ground intersection, so those rows are given
    the far-field cap - sky should fog out completely, which it does.
    """
    from ..distance.camera import CameraModel

    cam = camera or CameraModel.from_hfov(w, h, 60.0, camera_height_m=1.3)
    cam = cam.rescaled(w, h)
    rows = np.arange(h, dtype=np.float32)
    depth = np.full(h, 150.0, np.float32)
    for i, v in enumerate(rows):
        z = cam.ground_plane_range(float(v), cam.camera_height_m)
        if np.isfinite(z) and z > 0:
            depth[i] = min(z, 150.0)
    return np.repeat(depth[:, None], w, axis=1)


def rain(img, severity=3, seed=0):
    """Streaks from falling drops, plus the contrast loss of a wet, hazy scene."""
    h, w = img.shape[:2]
    r = _rng(seed)
    n_drops = [300, 700, 1300, 2200, 3400][severity - 1]
    length = [10, 14, 18, 24, 30][severity - 1]
    slant = int(r.uniform(-8, 8))

    layer = np.zeros((h, w), np.uint8)
    xs = r.integers(0, w, n_drops)
    ys = r.integers(0, h, n_drops)
    for x, y in zip(xs, ys):
        cv2.line(layer, (int(x), int(y)), (int(x + slant), int(y + length)), 200, 1)
    layer = cv2.blur(layer, (3, 3))

    out = img.astype(np.float32)
    out = out * 0.92 + layer[..., None].astype(np.float32) * 0.6
    out = cv2.blur(out, (3, 3))
    haze = [0.02, 0.05, 0.09, 0.14, 0.20][severity - 1]
    out = out * (1 - haze) + 200.0 * haze
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# optics and sensor
# --------------------------------------------------------------------------- #
def motion_blur(img, severity=3, seed=0):
    """Directional blur from vehicle vibration / rolling shutter."""
    k = [5, 9, 13, 19, 25][severity - 1]
    angle = _rng(seed).uniform(-25, 25)          # mostly horizontal, as in driving
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(img, -1, kern)


def defocus(img, severity=3, seed=0):
    """Disk-kernel blur - focus hunting, or a dirty/wet lens."""
    radius = [2, 3, 5, 7, 10][severity - 1]
    k = 2 * radius + 1
    kern = np.zeros((k, k), np.float32)
    cv2.circle(kern, (radius, radius), radius, 1.0, -1)
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern)


def sensor_noise(img, severity=3, seed=0):
    """Shot noise (signal-dependent) plus read noise - a real sensor at high ISO."""
    r = _rng(seed)
    shot = [0.02, 0.04, 0.07, 0.11, 0.17][severity - 1]
    read = [2, 4, 7, 11, 17][severity - 1]
    x = img.astype(np.float32)
    x = x + r.normal(0, 1, img.shape) * np.sqrt(np.clip(x, 0, None)) * shot
    x = x + r.normal(0, read, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def jpeg(img, severity=3, seed=0):
    """Compression artefacts from a cheap camera or a bandwidth-limited link."""
    q = [55, 40, 28, 18, 10][severity - 1]
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def downscale(img, severity=3, seed=0):
    """Resolution loss - the cheapest way to turn every object into a small one."""
    f = [0.7, 0.55, 0.4, 0.28, 0.18][severity - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "low_light": low_light,
    "glare": glare,
    "fog": fog,
    "rain": rain,
    "motion_blur": motion_blur,
    "defocus": defocus,
    "sensor_noise": sensor_noise,
    "jpeg": jpeg,
    "downscale": downscale,
}

# What each corruption is standing in for, used in the report.
DESCRIPTIONS = {
    "low_light": "dusk / night, sensor pushing gain",
    "glare": "low sun into the lens, veiling glare",
    "fog": "depth-correct atmospheric scattering (Koschmieder)",
    "rain": "drop streaks + wet-scene contrast loss",
    "motion_blur": "vehicle vibration, rolling shutter",
    "defocus": "focus hunting, dirty or wet lens",
    "sensor_noise": "high-ISO shot + read noise",
    "jpeg": "compression artefacts",
    "downscale": "resolution loss / distant small objects",
}


def apply(img, name: str, severity: int = 3, seed: int = 0, **kw):
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; have {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 1..5, got {severity}")
    return CORRUPTIONS[name](img, severity=severity, seed=seed, **kw)


def preview_grid(img, severity: int = 3, out_path="outputs/corruption_preview.jpg"):
    """One tile per corruption, so the severities can be judged by eye."""
    from pathlib import Path

    tiles = [("clean", img)]
    for name in CORRUPTIONS:
        tiles.append((name, apply(img, name, severity)))
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    th, tw = img.shape[:2]
    grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, (name, t) in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.putText(grid, name, (c * tw + 10, r * th + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path
