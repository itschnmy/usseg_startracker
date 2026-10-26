# OpenMV dark subtraction

Run commands from the repository root using `.venv/bin/python`.
Put lossless dark captures in `dataset/calibration/openmv/session_01/dark/`.
Record acquisition settings in the adjacent `settings.json`.

Build once (the command refuses to overwrite an existing output):

```bash
.venv/bin/python -m examples.build_master_dark \
  --input-dir dataset/calibration/openmv/session_01/dark \
  --output data/calibration/openmv/session_01/master_dark.npy
```

This median-combines all BMP/PNG/TIFF files in that directory into a float32
`.npy`. Three-channel images are converted to grayscale using OpenCV, as in
the science-image path. All frames must be 8-bit and have identical dimensions.
The builder reads the expected count and dimensions from `settings.json` when
provided; null fields are unknown. It does not configure the camera or verify
exposure/gain from image pixels. The current session contains 30 640x480 frames.

To save just a corrected image, replace `PATH_TO_STAR_IMAGE.bmp`:

```bash
.venv/bin/python -m examples.subtract_dark \
  --image PATH_TO_STAR_IMAGE.bmp \
  --dark-frame data/calibration/openmv/session_01/master_dark.npy \
  --output out/openmv-corrected.png
```

To use automatic subtraction during solving, append
`--dark-frame data/calibration/openmv/session_01/master_dark.npy` to your existing
pipeline CLI command. Alternatively add `"dark_frame": "path/to/master_dark.npy"`
to a pipeline JSON; that path resolves relative to the JSON file. CLI paths
resolve relative to the working directory. Without this option the pipeline
behaves as before. Do not pass an already corrected image with the dark option,
which would subtract the calibration twice.

The master is loaded once at pipeline initialization. Both file/array inference
and `detect_stars()` diagnostics subtract it before Top-Hat and thresholding.
Subtraction uses floating point, clips negative pixels to zero, and rounds to
8-bit for the existing detector. Dimension mismatch raises an error: darks
must never be resized. BGR inputs convert to grayscale before subtraction.

Use matching exposure, gain, sensor geometry, pixel format, camera processing
and similar temperature. Do not mix acquisition settings in one master. Dark
subtraction removes fixed sensor background; it does not replace sky-background
estimation or guarantee removal of all hot pixels. Top-Hat remains configurable.
This implementation targets the Python pipeline; the C++ preprocessor is unchanged.
