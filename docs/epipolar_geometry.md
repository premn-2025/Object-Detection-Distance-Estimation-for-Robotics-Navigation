# Epipolar geometry and the disparity–depth relation

Extra credit. Implemented in [`src/geometry/epipolar.py`](../src/geometry/epipolar.py),
demonstrated by [`scripts/demo_stereo.py`](../scripts/demo_stereo.py).

---

## 1. The disparity–depth relation, derived

Two identical pinhole cameras with parallel optical axes, separated by a baseline
`B` along `X`, focal length `f` in pixels. A world point `(X, Y, Z)` in the left
camera's frame projects to

```
u_L = f·X / Z + cx
u_R = f·(X − B) / Z + cx
```

Their difference — the **disparity** — is

```
d ≡ u_L − u_R = f·B / Z
```

and therefore

```
        f · B
  Z  =  ─────                                                            (1)
          d
```

Two structural consequences follow immediately.

**Depth is inversely proportional to disparity.** Disparity is bounded above by
the image width and below by the matcher's precision, so a stereo rig has a
*near* limit (disparity too large to search) and a *far* limit (disparity
indistinguishable from zero).

**Range error grows quadratically.** Differentiating (1):

```
dZ/dd = −f·B / d² = −Z² / (f·B)
```

so a matching error of `σ_d` pixels becomes

```
  σ_Z = Z² · σ_d / (f·B)                                                 (2)
```

Setting `σ_Z / Z = ε` gives the range at which relative error reaches `ε`:

```
  Z_max(ε) = ε · f · B / σ_d                                             (3)
```

### What that means in practice

`stereo_range_table()` prints this. With `f = 700 px` and `σ_d = 0.25 px`
(sub-pixel matching):

| baseline | disparity @ 10 m | σ_Z @ 10 m | σ_Z @ 30 m | 10 % range |
|---|---|---|---|---|
| 6 cm | 4.2 px | 0.60 m | 5.36 m | 16.8 m |
| 12 cm | 8.4 px | 0.30 m | 2.68 m | 33.6 m |
| 30 cm | 21.0 px | 0.12 m | 1.07 m | 84.0 m |

Doubling the baseline halves the error at every range. This is the calculation
to do *before* choosing a camera, and it is why small-baseline consumer stereo
modules are near-field devices.

---

## 2. The uncalibrated case: fundamental and essential matrices

Real cameras are never perfectly parallel, so (1) only applies after
rectification. The general two-view constraint is that corresponding points lie
on each other's **epipolar lines**:

```
x_R^T · F · x_L = 0            (pixels,     F = fundamental matrix, rank 2)
x̂_R^T · E · x̂_L = 0           (normalised, E = essential matrix)
E = K_R^T · F · K_L            and       E = [t]_× · R
```

Given intrinsics `K`, `cv2.findEssentialMat` + `cv2.recoverPose` decompose `E`
into a rotation `R` and a **unit** translation `t`. The scale of `t` is
fundamentally unobservable from images alone: doubling the scene size and the
baseline together produces identical images.

`symmetric_epipolar_error()` reports the mean distance from each matched point to
its epipolar line. It is the honest health check on an estimated `F` — if that
median is not around a pixel, nothing downstream should be believed.

---

## 3. Motion stereo: why this is relevant to a monocular project

Every dataset here is monocular, so there is no second camera. But **a moving
camera is a stereo rig spread over time.** Two frames `Δt` apart, with the robot
travelling at `v`, form a stereo pair with baseline `B = v·Δt`.

ROADWork frames carry millisecond timestamps in their file names
(`src/data/sequences.py` recovers the clips), so `Δt ≈ 30 ms`. At 10 m/s that is
a 30 cm baseline — better than most consumer stereo modules.

`motion_stereo_depth()` does the work: match features, recover `(R, t)`,
triangulate. Three caveats, all real:

1. **Scale.** Triangulation with unit `t` gives depths up to an unknown factor.
   On a robot this comes from wheel odometry or IMU. In `demo_stereo.py` it is
   anchored on the nearest detected object's monocular range instead, which makes
   the *other* objects a genuine test: the anchor is 1.00 by construction, the
   rest are not.

2. **Forward motion is the worst possible baseline.** Translation along the
   optical axis puts the epipole near the image centre, and points near the
   epipole barely move between frames, so their triangulation is ill-conditioned.
   A cone dead ahead is exactly the hardest case. Sideways motion would be ideal
   and is exactly what a road vehicle does not do.

3. **The scene must be rigid.** Other moving vehicles violate the epipolar
   constraint and get triangulated to nonsense. RANSAC inside `findEssentialMat`
   rejects them as outliers, which is why the inlier count is reported.

---

## 4. Relationship to the monocular estimator

Both the ground-plane cue (`docs/distance_estimation.md` §3) and stereo recover
depth from a small image-space quantity, so both degrade quadratically with
range:

| cue | measured quantity | error growth | needs |
|---|---|---|---|
| apparent size | box height, px | ~constant relative error, floored by `σ_H/H` | size prior |
| ground plane | box bottom row, px | `σ_Z ∝ Z²` | camera pose |
| stereo / motion stereo | disparity, px | `σ_Z ∝ Z²` | baseline (and scale) |

The size cue is the odd one out: its relative error does **not** grow with range,
it just never falls below the spread of the physical prior. That is precisely
why fusing it with a geometric cue is worth doing — they are wrong in
uncorrelated ways.

---

## 5. If a real stereo rig were available

`rectify_pair()` and `sgbm_disparity()` are implemented for that case:
`cv2.stereoRectify` to build the remaps, semi-global block matching for dense
disparity, then (1) per pixel. With a dense depth map the whole size-prior
apparatus becomes unnecessary for ranging — the prior would instead be useful for
*validating* depth, and for the far field where disparity has run out.
