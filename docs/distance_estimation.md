# Distance estimation: geometry, error model, failure modes

Everything below is implemented in [`src/distance/`](../src/distance). No neural
network estimates distance anywhere in this project — the detector produces
boxes, and geometry turns boxes into metres.

---

## 1. Camera model and conventions

Standard OpenCV pinhole. A point `p_C = (X_C, Y_C, Z_C)` in the **camera frame**
(`X` right, `Y` **down**, `Z` along the optical axis) projects to

```
u = fx · X_C / Z_C + cx
v = fy · Y_C / Z_C + cy
```

The **level frame** `L` shares the origin but has `Z_L` forward and horizontal.
It differs from `C` by the camera pitch `θ` (positive = nose down):

```
p_C = R_x(θ) · p_L,    R_x(θ) = [[1, 0, 0], [0, cosθ, −sinθ], [0, sinθ, cosθ]]
```

The road is the plane `Y_L = h`, where `h` is the camera height above it.

**Focal length in pixels.** `fx` is not a lens property in millimetres; it is the
focal length expressed in pixel units, which is what makes the projection
equations dimensionless in the numerator:

```
fx = (W / 2) / tan(HFOV / 2)         [pixels]
fx = f_mm · W / sensor_width_mm      [equivalently, from physical quantities]
```

For the 1920-px-wide ROADWork frames at an assumed 60° horizontal FOV,
`fx = 960 / tan(30°) = 1662.8 px`. Every distance below scales linearly with this
number, which is why §6 is about not trusting it.

---

## 2. Cue A — apparent size

An object of true height `H` metres, at depth `Z_C` along the optical axis, spans

```
h_px = fy · H / Z_C        ⟹        Z_C = fy · H / h_px
```

Straight similar triangles. It needs a physical size prior per class, which
`configs/camera.yaml` supplies from the US MUTCD (a traffic cone is 0.71 m or
0.91 m; a stop sign is a 0.76 m octagon).

**Error propagation.** Take logs: `ln Z = ln fy + ln H − ln h_px`. The relative
errors add in quadrature:

```
(σ_Z / Z)² = (σ_h / h_px)² + (σ_H / H)²
```

Two independent box edges, each localised to `σ_edge ≈ 2.5 px`, give
`σ_h = √2 · σ_edge`. The second term does not shrink with range: the spread of
real cone heights (σ_H/H ≈ 19 %) puts a **floor** under this cue. A perfectly
detected cone is still only known to ±19 % unless you know which cone it is.

---

## 3. Cue B — ground plane

A point on the road at horizontal range `Z` and lateral offset `X` is
`p_L = (X, h, Z)`, so

```
p_C = (X,  h·cosθ − Z·sinθ,  h·sinθ + Z·cosθ)
```

Writing `ṽ = (v − cy) / fy` for the normalised image row and substituting into
`v = fy · Y_C / Z_C + cy`:

```
ṽ · (h·sinθ + Z·cosθ) = h·cosθ − Z·sinθ
Z · (ṽ·cosθ + sinθ)   = h · (cosθ − ṽ·sinθ)
```

```
            h · (cosθ − ṽ·sinθ)
    Z  =  ─────────────────────                                          (★)
              ṽ·cosθ + sinθ
```

At `θ = 0` this collapses to the familiar `Z = h·fy / (v − cy)`.

The **horizon** is where `Z → ∞`, i.e. `ṽ = −tanθ`, so `v_horizon = cy − fy·tanθ`.
Anything whose box bottom sits above that row cannot be on the road, and the
estimator rejects it.

**Why bother, when cue A already works?** Because (★) contains no size prior at
all. It measures where the object *touches the ground*, which is a much better
conditioned quantity than how tall it looks — especially for barriers, whose true
heights range over a factor of two.

**Error propagation.** Differentiating `ln Z` from (★) analytically (implemented
in `estimator._plane_cue`, not finite-differenced):

```
∂lnZ/∂ṽ = −sinθ/(cosθ − ṽ·sinθ) − cosθ/(ṽ·cosθ + sinθ)
∂lnZ/∂h = 1/h
∂lnZ/∂θ = −(sinθ + ṽ·cosθ)/(cosθ − ṽ·sinθ) − (cosθ − ṽ·sinθ)/(ṽ·cosθ + sinθ)
```

At `θ = 0` these reduce to `−1/ṽ`, `1/h`, and `−(ṽ + 1/ṽ)`. Since `ṽ = h/Z`, that
means

```
σ_Z/Z  ≈  (σ_v/fy)·(Z/h)   ⊕   σ_h/h   ⊕   σ_θ·(Z/h)
```

Both the row-noise term and the pitch term grow **linearly in Z/h** for the
relative error, i.e. the absolute error grows quadratically with range. A 1.5°
pitch error at 30 m with a 1.3 m camera is a ~60 % range error. This cue is
excellent up close and worthless far away — the exact opposite failure profile
to the size prior's constant 19 % floor.

---

## 4. Elevated objects: one model, not two

A stop sign does not touch the road, so (★) seems inapplicable. It is not — the
sign's lower edge lies on a *different* horizontal plane, a mounting height `m`
above the road (MUTCD: 2.1 m in urban settings). Substituting the effective plane
height

```
h_eff = h_camera − m
```

into (★) handles it with no special case. For a 1.3 m camera and a 2.1 m sign,
`h_eff = −0.8 m`: negative, meaning the plane is *above* the camera, and the
box bottom correctly appears above the horizon. The same code path, the same
Jacobians. The prior is weak (`σ_m = 0.45 m`), so the fusion in §5 down-weights
it automatically rather than needing a rule.

---

## 5. Fusion

The two cues are independent — one uses box *height*, the other box *bottom row*;
one needs a size prior, the other needs camera pose. Both have multiplicative
noise, so they are combined by inverse-variance weighting **in log space**:

```
w_i = 1 / σ_{ln Z_i}²
ln Z* = Σ wᵢ · ln Zᵢ / Σ wᵢ
σ_{ln Z*} = (Σ wᵢ)^(−1/2)
```

This does the right thing automatically: near the robot the ground cue is sharp
and dominates; far away its variance explodes and the size prior takes over.

The fused depth is back-projected through the **bottom-centre** of the box to a
3-D point in the level frame, so the pipeline reports not just a range but
`forward_m` and `lateral_m` — which is what a planner actually needs.

**Disagreement is diagnostic, not noise.** When the two cues differ by more than
1.6×, the estimate is flagged `cue_disagreement`. That is the signature of a bad
box, a misclassification, a slope, or an object that is not where it appears to
be. Averaging it away silently would be worse than surfacing it.

### Gating

| condition | consequence |
|---|---|
| box height < 8 px | size cue dropped (quantisation dominates) |
| box touches top/bottom border | size cue dropped (object truncated) |
| box bottom at image bottom | plane cue dropped (base not visible) |
| box bottom above horizon | plane cue dropped (not on the plane) |
| `\|h_eff\| < 0.05 m` | plane cue dropped (degenerate) |
| neither cue valid | `valid=False`, annotation reads `Cone` with no range |

---

## 6. Calibration, and being honest about it

The datasets ship no intrinsics, no extrinsics and **no ground-truth distances
anywhere**. So there is nothing to regress against, and no honest way to report a
distance MAE in metres. What *is* available is a consistency constraint:

> a cone of known height, standing on the road, must give the same range through
> cue A and through cue B.

`src/distance/calibrate.py` minimises a Huber loss on `ln(Z_A / Z_B)` over the
labelled cones. Running it surfaced four things worth recording, because each one
would have silently poisoned every distance in this project.

### 6.1 `fy` is not identifiable this way

At small pitch, `Z_A = fy·H/h_px` and `Z_B = h·fy/(v−cy)`. Both are proportional
to `fy`, so it **cancels in the ratio**. Focal length is *not* recoverable from
cue agreement and still rests on the assumed 60° FOV. The fit returns camera
pose only, and says so.

### 6.2 There are three different cameras in the data

The assembled set contains ROADWork 16:9 dash-cam video (1280×720), ROADWork 4:3
hand-held stills (1280×960) and COCO web photos with no consistent optics at all.
One pinhole model cannot describe all of them; fitting over the mixture produces a
meaningless average. The calibration is therefore restricted to the 16:9 dash-cam
frames — the geometry a robot would actually have. The boxes also live in the
*stored* resolution, not the 1920×1080 the config describes, so the intrinsics
are rescaled before anything is compared.

### 6.3 Small boxes bias the height estimate upward

The implied height is `H·(v−cy)/h_px`, so any systematic **under**-measurement of
`h_px` inflates it — and that error is multiplicative in `1/h_px`, so it explodes
as boxes shrink. Tight annotation, blur and the downscale to 1280px each shave a
pixel or two off every box. Textbook errors-in-variables attenuation, and it is
large here because the median cone box is only 43 px tall:

| sample restriction | n | implied camera height |
|---|---|---|
| all cones | 14846 | 2.43 m |
| box ≥ 40 px | 6540 | 1.79 m |
| box ≥ 60 px, base below 0.62·H | 3474 | 1.58 m |
| box ≥ 80 px, base below 0.76·H | 1387 | 1.58 m |
| **box ≥ 120 px, base below 0.83·H** | **454** | **1.38 m** |
| box ≥ 160 px, base below 0.86·H | 161 | 1.15 m |

Fitted over everything, the answer is 2.4 m — a truck. Restricted to
well-conditioned cones it converges on 1.2–1.4 m, an actual dash-cam height. The
full curve is published in the calibration report rather than the chosen row
alone.

### 6.4 Height and pitch are degenerate

Pitching the camera further down raises the horizon, which shrinks the implied
range for a given image row, which is then compensated by a larger camera
height. The pair slides along a valley in the objective, and the joint fit simply
runs to whichever bound it is given — it returned 3.0 m / 7.7° with one bound and
2.5 m / 9.9° with another, both fitting the cones about equally well. Neither
number was separately identified.

The fix is to break the tie from outside the fit: fix the height at the
well-conditioned estimate from §6.3 and solve for **pitch alone** — one
well-posed parameter instead of two badly-posed ones.

### 6.5 Result, and the one non-circular check

Fixing `h = 1.38 m` and fitting pitch gives **−0.53°**:

| set | median \|log ratio\| before → after | median ratio after |
|---|---|---|
| cones, train (fitted) | 0.194 → 0.176 | 1.027 |
| cones, val (held out) | 0.194 → 0.187 | 1.010 |
| **barriers (never fitted)** | **0.338 → 0.244** | **1.264** |

After the fit the two cues agree on cones *by construction*, so that agreement is
not evidence. The barrier row is: it improved without ever entering the fit.

Its residual 1.264 was itself informative. A ratio above 1 means `Z_A` is too
large, so the height prior `H` is too large — by exactly that factor, giving
`1.00 / 1.264 = 0.79 m`. That is independently what the class composition
predicts: "barrier" is dominated by vertical panels (9083 boxes, 0.6–0.9 m) and
drums (4987, 0.91 m), not by the tall barricades and fencing the 1.00 m prior
assumed. Correcting the prior to 0.79 m brings the barrier agreement ratio to
**0.998** and its residual to 0.118.

**The cost of that correction, stated plainly:** having used the barrier class to
set its own prior, it is no longer an independent check. A future validation
needs a class not used here, or real ranged ground truth.

`calibrate_checkerboard()` implements the procedure a real robot should use
instead. The FOV assumption remains the largest systematic bias in every number
reported, and it scales all of them linearly.

## 7. What would actually make this better

In rough order of value per unit effort:

1. **Calibrate the camera.** Removes the linear bias that currently sits on
   everything.
2. **Estimate pitch per frame** from the road vanishing point instead of assuming
   it constant. Suspension pitch under braking is several degrees, and §3 shows
   pitch error dominates the far field.
3. **Stereo, or motion stereo.** `src/geometry/epipolar.py` derives the
   disparity–depth relation and implements pose recovery from consecutive frames;
   with wheel odometry for scale, that is a genuinely independent range source.
4. **Regress the contact point, not the box bottom.** For a partially occluded
   cone the box bottom is the occluder's edge, not the cone's base.
