"""Tests for the hard-condition augmentation and the corruption benchmark.

The properties that matter here are easy to break silently:

* an augmentation that moves pixels would invalidate every bounding box *and*
  corrupt the apparent-size distance cue, without ever raising an error;
* a corruption that does nothing would make the robustness table look great;
* a corruption whose severity is not monotonic would make the table meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.degrade import CORRUPTIONS, SEVERITIES, apply  # noqa: E402
from src.distance.camera import CameraModel  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    """A structured synthetic frame - random noise is not a fair blur test."""
    rng = np.random.default_rng(0)
    img = np.zeros((240, 426, 3), np.uint8)
    img[:110] = (160, 140, 120)                 # sky
    img[110:] = (90, 90, 95)                    # road
    for x in range(0, 426, 40):                 # texture, so blur has an effect
        img[110:, x:x + 12] = (200, 200, 205)
    img[150:200, 180:210] = (40, 120, 240)      # an orange cone-ish blob
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255)
    return img.astype(np.uint8)


# --------------------------------------------------------------------------- #
# corruptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_preserves_shape_and_dtype(frame, name):
    out = apply(frame, name, 3)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corruption_actually_changes_the_image(frame, name):
    out = apply(frame, name, 3)
    assert not np.array_equal(out, frame), f"{name} at severity 3 is a no-op"


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_severity_is_monotonic(frame, name):
    """Higher severity must move the image further from the clean original.

    Without this, a 'severity 5' row in the robustness table means nothing.
    """
    dists = [float(np.abs(apply(frame, name, s).astype(np.float32)
                          - frame.astype(np.float32)).mean())
             for s in SEVERITIES]
    # allow small non-monotonicity from the stochastic corruptions, but the
    # endpoints must be clearly ordered
    assert dists[-1] > dists[0] * 1.3, f"{name}: severity had little effect {dists}"


def test_unknown_corruption_and_severity_raise(frame):
    with pytest.raises(KeyError):
        apply(frame, "snow_machine", 3)
    with pytest.raises(ValueError):
        apply(frame, "fog", 9)


def test_low_light_actually_darkens(frame):
    assert apply(frame, "low_light", 4).mean() < frame.mean() * 0.75


def test_glare_actually_brightens(frame):
    assert apply(frame, "glare", 4).mean() > frame.mean()


def test_blurs_reduce_high_frequency_energy(frame):
    """A blur must remove detail, measured as gradient energy."""
    import cv2

    def energy(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    for name in ("motion_blur", "defocus", "downscale"):
        assert energy(apply(frame, name, 4)) < energy(frame), name


def test_fog_is_depth_correct_not_uniform(frame):
    """Fog must wash out the far field more than the near field.

    This is the whole point of using the scattering model with the ground-plane
    depth map: a uniform alpha blend would affect every row equally and would be
    far too easy for a detector.
    """
    cam = CameraModel.from_hfov(frame.shape[1], frame.shape[0], 60.0,
                                camera_height_m=1.3)
    out = apply(frame, "fog", 4, camera=cam)
    d = np.abs(out.astype(np.float32) - frame.astype(np.float32)).mean(axis=(1, 2))
    horizon = int(cam.rescaled(frame.shape[1], frame.shape[0]).horizon_row)
    near = d[int(frame.shape[0] * 0.92):].mean()          # bottom rows = closest
    far = d[max(0, horizon - 10):horizon + 10].mean()     # rows at the horizon
    assert far > near * 1.5, f"fog is not depth-dependent (far={far:.1f} near={near:.1f})"


def test_corruptions_are_deterministic_given_a_seed(frame):
    for name in ("low_light", "rain", "glare", "sensor_noise"):
        a = apply(frame, name, 3, seed=7)
        b = apply(frame, name, 3, seed=7)
        assert np.array_equal(a, b), f"{name} is not reproducible"


# --------------------------------------------------------------------------- #
# training augmentation
# --------------------------------------------------------------------------- #
def test_augmentation_presets_are_ordered():
    from src.train.augment import build_transforms

    assert build_transforms("off") == []
    assert len(build_transforms("standard")) == len(build_transforms("hard"))
    with pytest.raises(ValueError):
        build_transforms("extreme")


def test_standard_preset_is_milder_than_hard():
    """Same transforms, lower probabilities."""
    from src.train.augment import build_transforms

    def total_p(ts):
        return sum(getattr(t, "p", 0.0) for t in ts)

    assert total_p(build_transforms("standard")) < total_p(build_transforms("hard"))


def test_every_training_transform_is_pixel_level():
    """The load-bearing invariant of this project.

    A spatial transform would move boxes and change apparent object height, which
    silently corrupts ``Z = fy*H/h_px``. Assert structurally, not by sampling.
    """
    import albumentations as A

    from src.train.augment import build_transforms

    def walk(ts):
        for t in ts:
            if isinstance(t, A.BaseCompose):
                yield from walk(t.transforms)
            else:
                yield t

    spatial = [type(t).__name__ for t in walk(build_transforms("hard"))
               if isinstance(t, A.DualTransform)]
    assert not spatial, f"spatial transforms would break box/size geometry: {spatial}"


def test_augmentation_leaves_boxes_untouched():
    import albumentations as A

    from src.train.augment import _seed_all, build_transforms

    tf = A.Compose(build_transforms("hard"),
                   bbox_params=A.BboxParams(format="yolo", label_fields=["cls"]))
    img = (np.random.default_rng(0).random((240, 426, 3)) * 255).astype(np.uint8)
    box = [0.5, 0.6, 0.06, 0.12]
    for i in range(25):
        _seed_all(i)
        out = tf(image=img, bboxes=[box], cls=[0])
        assert len(out["bboxes"]) == 1
        assert np.allclose(np.array(out["bboxes"][0])[:4], box, atol=1e-9)


def test_install_is_idempotent():
    """Installing twice must not wrap the patch around itself."""
    from ultralytics.data.augment import Albumentations

    from src.train.augment import install_hard_augmentation

    install_hard_augmentation("hard", verbose=False)
    first = Albumentations._orig_init
    install_hard_augmentation("hard", verbose=False)
    assert Albumentations._orig_init is first
    a = Albumentations(p=1.0)
    assert a.transform is not None


# --------------------------------------------------------------------------- #
# COCO evaluator
# --------------------------------------------------------------------------- #
def test_coco_conversion_roundtrips_boxes(tmp_path):
    import cv2

    from src.optimize.cocoeval import gt_size_histogram, yolo_labels_to_coco

    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((200, 400, 3), np.uint8))
    # one tiny box (small band) and one big box (large band)
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.05 0.05\n1 0.25 0.25 0.5 0.5")

    gt = yolo_labels_to_coco(img_dir, lbl_dir, ["cone", "barrier"])
    assert len(gt["images"]) == 1 and len(gt["annotations"]) == 2
    x, y, w, h = gt["annotations"][0]["bbox"]
    assert (x, y, w, h) == pytest.approx((190.0, 95.0, 20.0, 10.0))
    assert gt["annotations"][0]["category_id"] == 1      # COCO ids are 1-based

    hist = gt_size_histogram(gt, ["cone", "barrier"])
    assert hist["overall"]["small (<32^2 px)"] == 1
    assert hist["overall"]["large (>96^2 px)"] == 1


def test_coco_eval_is_perfect_on_perfect_detections(tmp_path):
    """Feeding the ground truth back in as detections must score AP ~1.0."""
    import cv2

    from src.optimize.cocoeval import evaluate, yolo_labels_to_coco

    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(img_dir / f"{i}.jpg"), np.zeros((200, 400, 3), np.uint8))
        (lbl_dir / f"{i}.txt").write_text("0 0.5 0.5 0.3 0.3")

    gt = yolo_labels_to_coco(img_dir, lbl_dir, ["cone"])
    dets = [{"image_id": a["image_id"], "category_id": a["category_id"],
             "bbox": list(a["bbox"]), "score": 0.9} for a in gt["annotations"]]
    res = evaluate(gt, dets, ["cone"])
    assert res["mAP50-95"] > 0.99
    assert res["per_class"]["cone"]["n_gt"] == 3


def test_coco_eval_handles_empty_detections(tmp_path):
    import cv2

    from src.optimize.cocoeval import evaluate, yolo_labels_to_coco

    img_dir, lbl_dir = tmp_path / "images", tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.zeros((200, 400, 3), np.uint8))
    (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.3 0.3")
    res = evaluate(yolo_labels_to_coco(img_dir, lbl_dir, ["cone"]), [], ["cone"])
    assert res["mAP50-95"] == 0.0 and res["n_detections"] == 0
