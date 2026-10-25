# Example fixtures

- `03.bmp`: original 640x480 test image. It is an **unresolved diagnostic**:
  no verified same-capture calibration/WCS or star labels are available. A
  `no_solve` is valid output; do not treat the manual Andromeda annotation as
  ground truth or a solver result.
- `stellarium_test/stellarium_view.png`: 1920x1080 perspective Stellarium
  screenshot with `stars_labels.json`. The label FOV is **vertical** 16.9761 deg;
  the matching pipeline preset uses horizontal 29.717817 deg. Labels are read
  only after blind inference and are a pseudo-reference, not physical ground truth.

The original source files are preserved. Generated annotated/debug images are
not fixtures and are ignored by Git. Run `examples/run_image_examples.py` to
create overlays from actual detections/matches. Databases are external assets;
see `examples/materialize_assets.sh` and `examples/generate_tetra_database.py`.
