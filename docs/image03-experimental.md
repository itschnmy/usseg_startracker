# Image03: calibrated-FOV blind verification experiment

Python 3.10, branch `dev/version01`. This is development tuning on a known
image, **not** a held-out DUST/LOST benchmark or proof of flight readiness.

Image03 was acquired from hardware/camera according to the user, unlike the
direct Stellarium screenshot fixture. Camera model, exposure and whether it
captures the physical sky or a simulator display have not been documented.
See the [fixture README with original/result images](../dataset/README.md#image03-hardware-camera-capture).
The standard `examples/run_image_examples.py` now uses this preset for image03.

## Reproduce

```bash
.venv/bin/python -m models.pipeline.cli \
  --config configs/examples/image03-experimental.json \
  --image dataset/03.bmp --output out/image03-experimental/single.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m examples.check_image03_experimental
```

The second command writes `out/image03-experimental/summary.md`, `results.json`
and `image03-overlay.png`. The overlay uses display-only gain8 and draws only
HIP IDs returned by the actual blind solver, not manually assigned labels.

## Changes and why

| Setting | Historical image03 | Experimental image03 |
|---|---:|---:|
| Horizontal FOV | 28 ±10°, exploratory | 16.8 ±0.8°, supplied Astrometry calibration |
| Selected detections | 20 | 40 |
| Solver input cap | DB default30 | explicit40 |
| Pattern-search candidates | 16 | 16 |
| Verification catalog cap | number of input detections | up to90 visible catalog stars |
| Match radius / width | 0.015 =9.6px | 0.003 =1.92px |
| Match probability threshold | 0.001 | 0.001, unchanged |
| Soft timeout | 10s | 10s |

There is **no detector/preprocessing change** relative to the previous image03
preset: sigma3, Top-Hat31, area3–300, component mask, border12 and flux ranking
are unchanged. Extra candidates retain a weak valid star previously ranked32.
The solver previously truncated input at30 even when `max_stars` was larger;
`max_input_stars` now makes that independent cap explicit and configurable.

Verification previously assumed the detected stars were the brightest N catalog
stars. Missing bright stars, nonstellar blobs and brightness variation can break
that assumption. `verification_catalog_stars` expands the pool without changing
the search catalog/database. The mismatch calculation uses the actual enlarged
pool size, so it does **not** conceal the additional chance of random matches.
The narrower gate reduces accidental coincidences; the probability threshold
has not been relaxed. This remains Tetra's heuristic statistical model, not a
guaranteed empirical false-solve probability.

Both new controls default to `None`/JSON `null`, preserving the legacy solver
limits. CLI flags: `--max-input-stars`, `--verification-catalog-stars`.
The historical presets, database, SVD and benchmark defaults are untouched.

## Reference and limitations

User-provided screenshot of Astrometry user_image16604500/job17182488 reports
Perseus, RA48.773°, Dec41.364°, size16.8×12.6°, pixel scale94.6arcsec/px.
Radius10.508° is **not** HFOV. Only the rounded HFOV/range is provided to inference;
no RA/Dec center, roll, WCS, correspondence or ephemeris hints are passed.
This is blind sky-location solving **with supplied calibration**, not unknown-FOV
all-sky calibration. References are consulted after all inference in the runner.

Observed solution: 7 HIP stars, center approximately RA48.81634°/Dec41.36973°,
HFOV16.74719°. Approximate center discrepancy is about0.033° versus the rounded
screenshot. This is **boresight-only**, not a full quaternion/roll error measure.
The tight-gate-only control also solves6 stars with about0.031° center discrepancy;
7 matches therefore does not establish better attitude accuracy than6 matches.
The original WCS/Corr/uploaded image have not been downloaded or verified
bit-identical to the local BMP. Earlier hand-written Andromeda/Ophiuchus
annotations are not used and do not constitute valid previous solver results.

The ablation runner compares historical, calibration-only, gate-only,
catalog-only, gate+catalog, extended-input-only and full experimental configurations.
It also checks the tracked Stellarium image, up to five local synthetic LOST
fixtures and seeded random-point negatives. Synthetic smoke reports solve
counts only; it does not silently equate solve with correctness. Read the
generated report for measured results. Soft-timeout timing is not a benchmark.
Random negatives are a small sanity check, not an established false-positive rate.
Full DUST/LOST evaluation must be rerun separately before claiming improvement.
