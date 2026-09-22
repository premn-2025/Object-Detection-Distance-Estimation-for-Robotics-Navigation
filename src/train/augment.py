"""Hard-condition augmentation for the navigation detector.

The problem this solves is a dataset problem, not a model problem. ROADWork is
overwhelmingly clear daylight: fine weather, good exposure, sharp frames. A robot
does not get that. It gets dusk, low sun straight into the lens, rain, a cheap
sensor, and cones 40 m away that are twelve pixels tall. A detector trained only
on clean frames learns a decision boundary that sits far too close to those
conditions.

So the augmentation deliberately simulates the failure cases:

| group | what it simulates | why it matters here |
|---|---|---|
| exposure | dusk, night, back-lit scenes, auto-exposure hunting | ROADWork has almost no low light |
| glare | low sun into the lens, wet-road specular | washes out the orange/white cone bands the model keys on |
| weather | rain streaks, fog, haze | reduces contrast exactly where distant cones live |
| optics | motion blur, defocus | robot vibration and rolling shutter |
| sensor | ISO noise, JPEG artefacts, downscale-upscale | cheap cameras and compressed streams |

**Every transform here is pixel-level.** None of them move a pixel, so bounding
boxes stay exactly valid, and — more importantly for this project — none of them
change apparent object *size*. A rotation or perspective warp would silently
corrupt the very ratio the distance estimator depends on
(`Z = fy·H/h_px`), so geometric augmentation stays off. That constraint is
unusual, and it is the reason this module exists instead of just turning up
ultralytics' defaults.

Probabilities are deliberately moderate and grouped under ``OneOf``: the goal is
that a meaningful fraction of batches are hard, not that every frame is
destroyed. Over-corrupting the training set costs clean-condition accuracy for
no robustness gain.

Usage::

    from src.train.augment import install_hard_augmentation
    install_hard_augmentation("hard")      # before YOLO(...).train(...)
"""

from __future__ import annotations

PRESETS = ("off", "standard", "hard")


import albumentations as A  # noqa: E402  (module-level: see CheapHaze below)
import numpy as np  # noqa: E402


class CheapHaze(A.ImageOnlyTransform):
    """A fast stand-in for ``A.RandomFog``.

    Profiling the training pipeline found the GPU sitting at **0-1 % utilisation**
    while three dataloader workers saturated the CPU. One transform was
    responsible: ``A.RandomFog`` costs ~1800 ms per 640x640 image, against 1-50 ms
    for everything else in the stack - it composites many overlapping circles.
    At p=0.11 that is ~200 ms amortised over every image, more than the entire
    forward+backward pass. Replacing it took the whole stack from 230 ms to 23 ms
    per image.

    This uses the physical model rather than a particle simulation: a depth-proxy
    haze blend ``I = I0*t + A*(1-t)`` with transmission falling off towards the
    horizon. Visually closer to real fog than scattered circles, and ~1 ms.

    Deliberately *not* the same implementation as the depth-correct fog in
    ``src/data/degrade.py`` used for the robustness benchmark - train-time and
    test-time corruptions stay independent.

    Defined at module level on purpose: the Windows dataloader spawns worker
    processes and pickles the transform, which a class nested in a function
    cannot survive.
    """

    def __init__(self, strength=(0.15, 0.6), horizon_frac=(0.3, 0.6), p: float = 0.5):
        super().__init__(p=p)
        self.strength = strength
        self.horizon_frac = horizon_frac

    def apply(self, img, **params):
        h, w = img.shape[:2]
        s = np.random.uniform(*self.strength)
        hf = np.random.uniform(*self.horizon_frac)
        rows = np.arange(h, dtype=np.float32) / max(h - 1, 1)
        # densest at the horizon, thinning towards the near field
        dens = np.clip(1.0 - np.abs(rows - hf) / max(hf, 1e-3), 0.0, 1.0) ** 0.7
        dens = np.maximum(dens, np.clip((hf - rows) / max(hf, 1e-3), 0, 1))
        alpha = (s * dens).reshape(h, 1, 1).astype(np.float32)
        air = np.float32(np.random.uniform(200, 240))
        out = img.astype(np.float32) * (1.0 - alpha) + air * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength", "horizon_frac")


def build_transforms(preset: str = "hard"):
    """Return the albumentations transform list for a preset."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    if preset == "off":
        return []

    # scale every probability down for the milder preset
    k = 1.0 if preset == "hard" else 0.45

    return [
        # --- exposure: dusk, night, back-lit, auto-exposure hunting -----------
        A.OneOf([
            # asymmetric on purpose: under-exposure is the realistic failure
            A.RandomBrightnessContrast(brightness_limit=(-0.45, 0.20),
                                       contrast_limit=(-0.35, 0.25), p=1.0),
            A.RandomGamma(gamma_limit=(55, 165), p=1.0),
            A.RandomToneCurve(scale=0.25, p=1.0),
            # non-uniform illumination: headlights, tunnels, dappled shade
            A.Illumination(mode="gaussian", intensity_range=(0.05, 0.2), p=1.0),
        ], p=0.55 * k),

        # --- glare and shadow -------------------------------------------------
        A.OneOf([
            # low sun sits near the horizon, which is where distant cones are
            A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.55), src_radius=220,
                             method="physics_based", p=1.0),
            A.RandomShadow(shadow_roi=(0.0, 0.35, 1.0, 1.0),
                           num_shadows_limit=(1, 3), p=1.0),
        ], p=0.22 * k),

        # --- weather ----------------------------------------------------------
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.85, drop_width=1,
                         blur_value=3, p=1.0),
            CheapHaze(strength=(0.15, 0.6), p=1.0),
        ], p=0.22 * k),

        # --- optics: vibration, focus hunting ---------------------------------
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.4), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.25 * k),

        # --- sensor and pipeline ---------------------------------------------
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            A.GaussNoise(std_range=(0.03, 0.14), p=1.0),
        ], p=0.28 * k),
        A.ImageCompression(quality_range=(35, 92), p=0.25 * k),
        # resolution loss: the cheapest way to manufacture more *small* objects
        # from the frames we already have
        A.Downscale(scale_range=(0.35, 0.75), p=0.16 * k),

        # --- colour-constancy robustness --------------------------------------
        A.ToGray(p=0.02 * k),
        A.CLAHE(clip_limit=(1.0, 3.0), p=0.06 * k),
    ]


def install_hard_augmentation(preset: str = "hard", verbose: bool = True) -> int:
    """Patch ultralytics so its albumentations stage uses our transform list.

    Ultralytics constructs ``Albumentations(p=1.0)`` internally with a near-inert
    default (four transforms at p=0.01). It does accept a ``transforms``
    argument, so this wraps ``__init__`` to supply ours rather than reaching into
    the training loop. Returns the number of installed transforms.
    """
    from ultralytics.data import augment as ua

    transforms = build_transforms(preset)
    if not transforms:
        if verbose:
            print("[augment] preset=off - leaving ultralytics defaults in place")
        return 0

    original = getattr(ua.Albumentations, "_orig_init", ua.Albumentations.__init__)

    def patched_init(self, p: float = 1.0, transforms=None, flip_idx=None, **kw):
        """Supply our transform list unless a caller passed its own."""
        original(self, p=p, transforms=transforms or build_transforms(preset),
                 flip_idx=flip_idx, **kw)

    ua.Albumentations._orig_init = original
    ua.Albumentations.__init__ = patched_init

    if verbose:
        print(f"[augment] installed '{preset}' preset: {len(transforms)} transform "
              f"groups (pixel-level only, boxes and apparent size preserved)")
    return len(transforms)


def _seed_all(seed: int) -> None:
    """albumentations 2.x draws from python `random` and numpy, with no public
    seed helper, so both are seeded directly."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def preview(preset: str = "hard", image_path=None, out_path="outputs/augmentation_preview.jpg",
            n: int = 12, seed: int = 0):
    """Render a grid of augmented samples - the only honest way to tune this.

    Numbers in a config file do not tell you whether a cone is still visible
    after the transform stack. Looking at twelve of them does.
    """
    import albumentations as A
    import cv2
    import numpy as np

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read {image_path}")
    img = cv2.resize(img, (640, 360))

    tf = A.Compose(build_transforms(preset))
    tiles = [img]
    for i in range(n - 1):
        _seed_all(seed * 1000 + i)
        tiles.append(tf(image=img)["image"])

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * 360, cols * 640, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * 360:(r + 1) * 360, c * 640:(c + 1) * 640] = t
        label = "original" if i == 0 else f"aug {i}"
        cv2.putText(grid, label, (c * 640 + 10, r * 360 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path
