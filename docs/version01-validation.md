# dev/version01 validation

Branch created from `abed141fead5b76e99f8bed2e31f1a2d8c653708`.
Results live in the sibling evaluator repository under `out/version01/`.

## Source fixes

- Tests and evaluators import `models.pipeline`, `models.identifier`, and
  `models.attitude` directly. Runtime no longer depends on hand-written legacy
  compatibility files inside `.venv/site-packages`.
- Integration fixtures resolve from the repository root and use
  `data/default_database.npz`. Missing required databases fail explicitly.
- The CLI defaults match `PipelineConfig`: area 1–50, Top-Hat kernel 7,
  match radius 0.015, solver timeout 600 ms.
- Public pipeline quaternions are normalized to inertial-to-camera for all four
  attitude methods. SVD returns the inverse direction internally when it has
  more than two pairs; QUEST, Davenport and TRIAD do not. Known-rotation tests
  verify the published direction through vector reprojection.
- The C++ test source/include paths follow the new directory structure.
- `pyproject.toml` supports Python 3.10 editable installation and the
  `usseg-startracker` command. Database assets remain explicit external paths.
- The DUST geometric check for LOST is diagnostic only. Baseline solve status
  is the algorithm's raw output; the check does not silently reject solves.

## Validation completed

- Python recheck (2026-10-25): 9 passed, 2 skipped because external CSV fixtures are absent.
- DUST adapter/scoring: 8 passed; one expected Astropy FITS header warning.
- C++ LIS: 5/5 catalog identities correct; rotation error about 4.94e-10 degrees.
- C++ tracking: 5 true correspondences matched, 2 simulated outliers rejected.
- C++ compiled directly with g++; CMake is not installed on the current machine.
- Editable installation and CLI invocation from `/tmp` verified.

## Image 03

`dataset/03.bmp` is 640×480 uint8 BGR, with grayscale median 3, P99 12,
and maximum 145. Original SHA-256:
`0ffe9418ff9e5d5a1beecfb9e39142dbc5f70b17783e004c1c30f2b6062d9493`.

At the unchanged default detector settings it yields 470 components before
the top-20 selection. Several retained detections lie on noisy borders. The
diagnostic image applies an 8× display gain only; solving uses the original
image. Evidence: `../lost-evals/out/version01/image03/03-audit.json` and
`03-centroids.png`.

The accompanying annotated PNG says FOV 10.508 degrees but does not specify
horizontal versus vertical. Its named-star geometry does not establish an
accurate pinhole reference. Neither the image nor that annotation provides
independently verified attitude ground truth. No attitude accuracy claim is
made for this image. HIP entries absent from the Tetra database may have been
removed deliberately by its spatial thinning; absence alone is not corruption.
In this case three displayed names also had incorrect hard-coded HIP IDs in
`data/extract_star.py`: mu And 5188→4436, pi And 3383→2912, epsilon And
3204→3031. The old IDs have catalog V magnitudes 7.80, 9.22 and 8.36,
respectively, so their absence from the magnitude-7 database is expected.
These three name mappings were corrected in the user's untracked script;
that script remains untracked. Existing annotated images were not regenerated.
`extract_star.py` captures the Windows Stellarium viewport and uses a fixed
140/255 blob threshold for that screenshot workflow. It is not the camera
image detector in `models/detector`; applying that threshold directly to image
03 (maximum grayscale 145) would discard almost all visible stars.

The user supplied experiment notes describing a camera photographing a
Stellarium screen: nominal camera DFOV 21.7 degrees, a 33.78 cm screen width,
and several different Stellarium FOV/distance settings, alongside an on-Earth
sky-imaging setup. The notes do not establish which setup produced image 03
or associate a calibrated camera model, Stellarium
projection/RA/Dec/roll, or screen bounds with the exact image 03 capture.
For a 4:3 sensor, a 21.7-degree diagonal FOV corresponds to approximately
17.4-degree horizontal FOV. Stellarium viewport FOV and the camera's optical
FOV are separate quantities in this two-stage projection.

## Remaining algorithm limitations

- Non-uint8 detector input is clipped to 0–255, not radiometrically scaled.
  DUST supplies a shared, explicitly scaled uint8 array before either algorithm.
- Detector centroid weights use the entire component bounding rectangle,
  potentially including background or other components inside the rectangle.
- Top-20 peak ranking can retain hot pixels/border noise and discard true stars.
- Tetra checks its timeout between pattern combinations; 600 ms is not a hard
  wall-clock deadline.
- Direct SVD estimator callers must account for its inverse direction; the
  public image pipeline now does so. ADCS SVD and MEKF are separate code paths
  and are not validated by the image benchmark.
- Python Tetra/SVD and C++ voting/QUEST are distinct pipelines. C++ test success
  is not evidence for Python image-to-attitude accuracy.
- Ideal pinhole projection is used; no DUST camera distortion calibration.
- WCS is a plate-solving reference, and Corr labels may be incomplete.
- The evaluator's Star-ID agreement is sky-coordinate catalog-set overlap
  within 0.05 degrees. It does not verify each centroid-to-catalog pairing and
  must not be presented as correspondence-level identification precision.
- Published outage/reacquisition metrics count available frames; gaps in session
  timestamps mean frame counts must not be interpreted as elapsed seconds.
- Peak RSS in the DUST runner is process-level, not isolated per algorithm.
- FPS includes unsuccessful frames and early exits. It is frame-processing
  throughput, not the rate of successful/correct attitude updates.
- The default NPZ is 49,414,085 bytes on disk. Its pattern table alone has
  shape (12,369,092, 4), dtype uint16, or 98,952,736 uncompressed bytes.
  The compressed database size must not be reported as resident RAM use.
- The DUST end-to-end column is decode plus algorithm wall time. LOST's
  temporary PNG encoding/write happens before its wall timer and is excluded;
  its subprocess startup/database read are included. Use compute timing for
  comparison of the algorithm core and do not call this full deployment latency.

## Benchmark outputs

Full DUST validation completed: 4,444 unique records, 1,111 frames per
algorithm/track, 17 holdout sessions, no runtime errors. Correct (<0.5 degrees)
on the 919 WCS frames: H5 LOST 1, USSEG 1; PNG LOST 0, USSEG 3.
PNG detector recall is 38.429% USSEG versus 5.078% LOST; matched-centroid
median residual is 0.346 px versus 0.814 px. These detector improvements do
not establish reliable full-pipeline attitude availability.

Synthetic one-thread correct rates (20 low/high, 45 low/high noise): LOST
98/72/100/91%, USSEG 90/11/82/42%. An earlier USSEG run produced
99/13/92/46% under different OpenCV thread/time/load conditions. Most failed
frames reach the 600 ms soft solver deadline. In a selected 10-frame paired
diagnostic, 600 ms solved 4 and 5 seconds solved 8 (all 8 correct). This
failure-selected diagnostic is not a replacement for the primary benchmark.

The renewed geometry/catalog audit is in `out/version01/audit/dust-audit.md`.
The default database path, missing sys import, and metadata-key counting in
the evaluator's audit script were corrected. Image 03 also remains no-solve
under the nominal 17.4347-degree horizontal optical FOV assumption.

`out/version01/dust/` contains development and superseded filtered/partial
runs. The authoritative fresh test outputs use `out/version01/dust-final/`.
The source fingerprint and locked configuration are recorded in each completed
`.run.json`. Existing synthetic and DUST reports outside `out/version01/`
have not been overwritten.

## Working-tree comparison with feature (2026-10-25)

This table records the snapshot BEFORE the configurable preprocessing upgrade
described in `image-example-configs.md`. Detector source is now modified to add
opt-in fixes; the historical default output remains unchanged.

Both `dev/version01` and `feature/system-design-and-evaluation` currently point
to `abed141fead5b76e99f8bed2e31f1a2d8c653708`. There are no committed differences;
the changes below are uncommitted working-tree changes, including new files.

| Area | Difference from feature | Effect |
|---|---|---|
| Detector / Tetra / attitude estimators / math utilities | No source changes | Algorithm implementation preserved; existing limitations above remain |
| Public pipeline | Removed obsolete import fallback; corrected method-dependent quaternion inversion | Modern imports fail transparently; QUEST/Davenport/TRIAD publish the proper direction; default SVD direction unchanged |
| CLI | Defaults now come from `PipelineConfig`; added max-area/kernel flags | Feature CLI defaults area=2, match radius=0.01, timeout=5000 ms become 1, 0.015, 600 ms, matching the existing Python API; these configuration changes affect results |
| Python tests | Modern imports, repository/data paths, explicit optional fixture skips, known-rotation reprojection and CLI parity tests | No silent success for missing required DB; 2 optional CSV tests still skipped |
| C++ tests/build | Fixed moved source/include paths; registered LIS/tracking with CTest | Build/test integration only, no C++ solver edits |
| Packaging | Added Python 3.10 `pyproject.toml` and console command; ignore egg-info | No dependency on manually installed legacy compatibility shims |
| Image audit | Generic filenames, same detector configuration as pipeline, explicit timeout and checked image writes | A centroid overlay is diagnostic evidence, not a star-ID or attitude reference |
| User's `data/extract_star.py` | Untracked Stellarium utility; three name/HIP mappings corrected earlier | Not called by the blind image pipeline or benchmark |
| Documentation | Historical README claims flagged; this validation record added | Fresh measured results supersede historical performance claims |

The database SHA-256 remains
`4fad84d8d7d922e240abbb557dd6329ed94468c76657eebd4a3d9d042ab0f412`.
The user's `data/extract_star.py` and `dataset/03.bmp:Zone.Identifier` remain
untracked; no datasets, old reports, or user files were deleted. No commit or
push was performed.

### Recheck evidence

- `python -m pytest -q`: 9 passed, 2 skipped under Python 3.10.21.
- Evaluator `python -m pytest -q test/test_dust.py`: 8 passed, one expected
  Astropy warning for a header-only FITS WCS.
- DUST dev smoke: 2 frames x 2 algorithms x H5/PNG = 8 records; all `no_solve`,
  zero runtime errors. Stored separately in
  `../lost-evals/out/version01/recheck-20261025/{h5,png}-dev.jsonl` with run
  metadata. This tiny smoke subset is not an accuracy estimate.
- Manual Crux catalog/attitude/MEKF verification completed successfully after
  correcting its catalog path; this does not validate missing external CSV fixtures.
- Original `scratch/03.png`, HFOV assumption 10.508 degrees: both 600 ms and
  5000 ms return `no_solve`, 470 detected components, 20 selected centroids.
  New evidence: `../lost-evals/out/version01/recheck-20261025/image03-600/`
  and `image03-5000/`. The displayed 8x gain is not used for inference.
- The historical `03_annotated_usseg.png` was drawn by an external
  `create_annotated_03_png.py` with hard-coded names, coordinates, FOV and
  RA/Dec; that script did not invoke the detector or solver. It is not evidence
  of an earlier successful attitude solution and was not regenerated.

The recheck is a regression/smoke check, not a new full-test benchmark. Earlier
synthetic/DUST measurements and their source fingerprints remain unchanged.

## Configurable preprocessing upgrade

See [image example configs](image-example-configs.md) for new opt-in component
masking, edge rejection, flux ranking, CLI/JSON presets and reproducible commands.
The new Stellarium preset solves blind with 16/16 verified labelled matches,
attitude error 0.002738 degrees. Image03 remains unresolved, including tests of
horizontal mirroring and a separate existing-solver distortion search (-0.1 to
0.1, no FOV hint, 20-second soft timeout). No fake attitude or oracle fallback
was added to the examples.

Legacy detector defaults were compared directly with the feature branch source
on image03, Stellarium and synthetic 20-low-noise frame0: component counts,
positions, peaks and integrated intensities were exactly equal. No full dataset
benchmark has been rerun with the new presets; do not replace old result tables.

## Pre-push review (2026-10-25)

The working-tree-only notes above describe earlier snapshots; the reviewed
implementation, configs, regression tests and small raw example fixtures are
being committed on `dev/version01`. Image03 remains an unresolved test only.

- Final Python suite: 39 passed, 2 optional external-CSV skips.
- Recompiled both C++ test executables directly with g++: LIS and tracking pass.
  CMake is unavailable on this machine, so no CMake execution is claimed.
- Added fail-fast checks for config counts, ranges, finite values and unsupported
  methods; invalid CLI overrides produce error JSON instead of an uncaught traceback.
- Added ndarray shape/type/finite-pixel validation before OpenCV processing.
- CLI forwarding now derives from config fields; removed redundant constructor
  mapping and unused imports. Existing supported CLI names remain intact.
- Fixed `materialize_assets.sh` after repository restructuring: restore into
  `data/`, verify temporary downloads before installing, preserve existing assets.
- Ignore generated `out/`, local `scratch/`, caches and Windows Zone.Identifier
  metadata. No raw datasets, historical results or diagnostic evidence deleted.
- Track only the intentional raw image03 and Stellarium screenshot/labels, not
  generated debug annotations or large database binaries.
- `.gitmodules` unchanged: the existing upstream LOST/LOST-evals submodule URLs
  remain as configured; the sibling evaluator worktree is not part of this push.

Remaining limits: no verified image03 reference, non-uint8 detector clipping,
ideal pinhole model and method-specific raw SVD convention. Passing tests and the
Stellarium example are not evidence of general DUST performance improvement.
