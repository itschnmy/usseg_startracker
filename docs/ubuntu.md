# Ubuntu Linux setup (dev/version2)

Use the system Python (3.10–3.14) in a virtual environment. The core pipeline
works on Ubuntu desktops, servers, and Wayland; screen capture needs X11.
The previous Python 3.10-only dependency pins have been replaced by compatible
version ranges so newer Ubuntu interpreters can install binary wheels.

From the repository directory:

```bash
sudo apt update
sudo apt install python3-venv python3-dev build-essential cmake libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
bash examples/materialize_assets.sh
python -m pytest -q
python examples/run_image_examples.py --output-dir out/image-examples
```

Database restoration uses the local `origin/main` Git ref and verifies its
SHA-256. If that ref is missing, run `git fetch origin main` first. External
LOST/DUST and CSV fixtures are optional; their tests skip when absent.

## C++ build

Eigen headers are bundled. Install OpenCV development headers to build the
native image CLI and preprocessing test as well as the algorithm tests:

```bash
sudo apt install libopencv-dev
cmake -S models/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel 2
ctest --test-dir build/cpp --output-on-failure
./build/cpp/star_tracker --help
```

Without OpenCV development headers, CMake builds only `test_lis` and
`test_tracking`. Python OpenCV wheels do not supply the C++ development files.

## Optional Stellarium desktop capture

```bash
sudo apt install stellarium xdotool
python -m pip install -r tools/stellarium/requirements.txt
python tools/stellarium/extract_star.py --output-dir captures/session-001
```

Log into an **Xorg/X11** desktop session with `DISPLAY` set. Native Wayland
capture is not supported by this collector; XWayland does not bypass this
restriction. Open exactly one visible native Stellarium window. Enable its
Remote Control plugin on port 8090 (see the API, scene and labeling details in
[the Stellarium guide](stellarium_image.md)). Keep the API on loopback; set
`STELLARIUM_API_PASSWORD` if authentication is enabled.

The Linux collector verifies the Stellarium process, activates its X11 window,
and captures the client rectangle without window decorations using MSS.
Use the same local Stellarium instance for the API and screenshot. Window
scaling, overlapping windows, notifications and desktop changes during capture
can affect labels; inspect each saved screenshot and overlay. Desktop capture
requires an interactive unlocked session and is separate from blind solving.
Windows capture remains supported with its platform-specific dependency.

## Validation on this branch

Validated on Ubuntu 26.04.1, Python 3.14.4 and GCC 15.2:

- Editable package installation and `pip check` succeeded.
- Python tests: 59 passed, 4 skipped (optional external fixtures).
- Image runner: image03 `ok` with 7 matches; Stellarium `ok` with 16 matches.
- CMake 4.4.4 build and CTest: LIS and tracking both passed.
- Collector and installed pipeline CLI help commands succeeded.

The validation environment did not have OpenCV C++ development headers or a live
Stellarium X11 session. The native image CLI/preprocessing target and actual
desktop screenshot capture therefore still need validation on such a desktop.
Linux capture process checks, geometry, foreground rejection, Wayland rejection
and missing-tool errors are covered by automated tests.
