# Image-specific preprocessing and example presets

Python 3.10; branch `dev/version01`. This is a development-example upgrade,
not a new LOST-evals/DUST benchmark. No label coordinates, sky center or
reference quaternion are used to solve either image.

## Changes

- JSON configuration loads through `PipelineConfig.from_json` and CLI `--config`.
  Database paths are resolved relative to the JSON file, not the working directory.
  Explicit CLI flags override the preset. Unknown JSON fields fail rather than being ignored.
- `--use-tophat` / `--no-use-tophat`, kernel size and area limits are configurable.
- `component_mask=true` measures centroid, flux and peak only on the connected
  component, excluding background and other blobs inside its bounding rectangle.
  This fixes the bounding-box contamination problem for the new presets.
- `border_margin_px` rejects entire components touching the noisy edge band;
  no crop or geometric coordinate shift is introduced.
- `star_ranking=flux` orders by integrated signal, with peak as tie-breaker.
  This avoids using peak alone as brightness proxy when stars saturate.
- `detect_stars` is the shared preprocessing/detection path for diagnostics and inference.
- Green HIP labels in the example overlays come only from actual solver output.
  Yellow numbers denote selected detections, not identified stars.

The old defaults remain Top-Hat 7, peak ranking, no component mask and zero
border margin. The new behavior is opt-in to preserve the historical baseline.
The fix does not make uint8 clipping a calibrated radiometric conversion.

## Presets

| Parameter | Historical default | Stellarium | Image03 (exploratory) |
|---|---:|---:|---:|
| Horizontal FOV degrees | supplied by caller | 29.717817 | 28 ± 10, uncalibrated |
| Top-Hat | 7x7 | disabled | 31x31 |
| Area pixels | 1–50 | 3–500 | 3–300 |
| Component mask | false | true | true |
| Border margin pixels | 0 | 12 | 12 |
| Ranking | peak | integrated flux | integrated flux |
| Max stars | 20 | 20 | 20 |
| Pattern checking stars | 12 | 12 | 16 |
| Solver soft timeout ms | 600 | 5000 | 10000 |

Stellarium's `stars_labels.json` records **vertical** FOV 16.9761 degrees.
At 1920x1080 the equivalent **horizontal** FOV is 29.717817 degrees:
`HFOV = 2 atan(tan(VFOV/2) * width/height)`. USSEG always takes horizontal FOV.

## Commands

From the repository root:

```bash
.venv/bin/python examples/run_image_examples.py --output-dir out/image-examples
.venv/bin/python -m models.pipeline.cli --config configs/examples/stellarium.json \
  --image dataset/stellarium_test/stellarium_view.png --output out/stellarium.json
.venv/bin/python -m models.pipeline.cli --config configs/examples/image03.json \
  --image dataset/03.bmp --output out/image03.json
```

Changing a parameter does not require editing source:

```bash
.venv/bin/python -m models.pipeline.cli --config configs/examples/stellarium.json \
  --image dataset/stellarium_test/stellarium_view.png --max-stars 30 --solve-timeout-ms 10000
```

CLI exit 0 means processing completed, including `no_solve`. Check the JSON
`status`; it is not a guarantee that attitude was found.

## Measured example results

Stellarium blind pipeline: 16 matches; 16/16 labelled centroid/catalog pairs
agree within 3px and 0.05 degrees. Full attitude error against the label-derived
reference: 0.002738 degrees; boresight error 0.002289 degrees. Selected detections
match 19/82 labels. This low selected recall is expected with a top-20 limit;
it must not be confused with full-detector recall. Labels are pseudo-reference,
not independent physical attitude ground truth.

Image03: still `no_solve`. Tested disabled Top-Hat, kernels 15/31, sigma 3/5,
component-mask and flux ranking, border rejection, FOV starts 10.5/17/28/35/45
degrees and the existing 10–30/45-degree databases. No sky labels were fed into
these solves. These trials do not establish that every possible configuration
has been exhausted, nor prove a specific calibration defect.

Do not call image03 solved just because its manually annotated PNG names stars.
Its exact effective FOV and camera/screen projection remain unverified. Further
progress needs reliable calibration or a verified correspondence/reference for
this exact raw capture; rectifying from guessed labels would not be a blind solve.

The current artifacts are stored in the sibling evaluator under
`out/version01/configured-examples/current/`. They do not replace earlier runs.
These presets were tuned on the example images: no held-out accuracy/FPS claim
is made, and DUST must be reevaluated separately before claiming improvements.

## Image03 deeper follow-up

A systematic follow-up completed 568 solver calls: 432 coarse parameter sweeps,
40 extended calls with a 5-second deadline and 96 virtual pixel-aspect trials.
All returned `no_solve`, with no algorithm runtime errors. There are 490 distinct
grid keys ignoring timeout; this is one-image diagnosis, not 568 independent
test images. An interrupted log tail is preserved and excluded from valid counts.

Evidence and limitations are documented in the sibling evaluator:
`out/version01/image03-deep-check-20261025/summary.md`. The historical manual
Andromeda correspondences fail a standard pinhole consistency fit (roughly
46.6px RMS per coordinate), so they remain unverified. This does not identify
a unique root cause. No new image03 success is claimed.
