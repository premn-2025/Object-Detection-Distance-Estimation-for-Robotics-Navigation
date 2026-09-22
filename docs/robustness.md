# Robustness and small-object performance

Model: `runs/stage2/nav_yolo11s-3/weights/best.pt`  ·  400 validation images  ·  inference at 640px

Corruptions are implemented in `src/data/degrade.py` with OpenCV, **independently of the albumentations stack used for training augmentation** (`src/train/augment.py`). If test-time corruption reused the training code, this table would measure memorisation of one noise generator rather than robustness.

## 1. Clean baseline, stratified by object size

| metric | value |
|---|---|
| mAP50-95 | 0.4134 |
| mAP50 | 0.6297 |
| **AP small** (<32² px) | **0.2225** |
| AP medium (32–96² px) | 0.5643 |
| AP large (>96² px) | 0.7012 |

Ground-truth boxes by size band:

| band | count |
|---|---|
| small (<32^2 px) | 1474 |
| medium (32-96^2 px) | 824 |
| large (>96^2 px) | 268 |

AP_small is the number that matters for navigation: it is the regime of cones far enough ahead to still be avoidable. It is always much lower than the headline mAP, and any report that quotes only the headline is hiding this.

## 2. Per-class (clean)

| class | mAP50-95 | mAP50 | AP small | GT boxes |
|---|---|---|---|---|
| cone | 0.3237 | 0.5501 | 0.1711 | 1211 |
| barrier | 0.2882 | 0.5467 | 0.1456 | 1235 |
| stop_sign | 0.6282 | 0.7923 | 0.3509 | 120 |

## 3. Corruption sweep

`retained` = corrupted mAP50-95 ÷ clean mAP50-95.

| corruption | simulates | severity | mAP50-95 | AP small | retained |
|---|---|---|---|---|---|
| sensor_noise | high-ISO shot + read noise | 2 | 0.4144 | 0.2230 | 100.3% |
| downscale | resolution loss / distant small objects | 2 | 0.4139 | 0.2275 | 100.1% |
| sensor_noise | high-ISO shot + read noise | 4 | 0.4125 | 0.2234 | 99.8% |
| jpeg | compression artefacts | 2 | 0.4092 | 0.2285 | 99.0% |
| rain | drop streaks + wet-scene contrast loss | 2 | 0.4090 | 0.2269 | 99.0% |
| glare | low sun into the lens, veiling glare | 4 | 0.4089 | 0.2204 | 98.9% |
| glare | low sun into the lens, veiling glare | 2 | 0.4088 | 0.2212 | 98.9% |
| defocus | focus hunting, dirty or wet lens | 2 | 0.4069 | 0.2085 | 98.4% |
| low_light | dusk / night, sensor pushing gain | 2 | 0.4068 | 0.2160 | 98.4% |
| downscale | resolution loss / distant small objects | 4 | 0.4005 | 0.1996 | 96.9% |
| jpeg | compression artefacts | 4 | 0.3954 | 0.2128 | 95.7% |
| rain | drop streaks + wet-scene contrast loss | 4 | 0.3879 | 0.2025 | 93.8% |
| motion_blur | vehicle vibration, rolling shutter | 2 | 0.3808 | 0.1815 | 92.1% |
| defocus | focus hunting, dirty or wet lens | 4 | 0.3340 | 0.1147 | 80.8% |
| low_light | dusk / night, sensor pushing gain | 4 | 0.3277 | 0.1588 | 79.3% |
| motion_blur | vehicle vibration, rolling shutter | 4 | 0.2581 | 0.0555 | 62.4% |
| fog | depth-correct atmospheric scattering (Koschmieder) | 2 | 0.2197 | 0.1364 | 53.2% |
| fog | depth-correct atmospheric scattering (Koschmieder) | 4 | 0.1960 | 0.1084 | 47.4% |

**Robustness score (mean retained mAP50-95): 88.6%**
