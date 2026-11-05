"""Build a float32 master dark from lossless OpenMV grayscale images."""
import argparse
import json
from pathlib import Path

import numpy as np

from utils.dark_frame import build_master_dark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .npy output file")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    master, paths = build_master_dark(args.input_dir)
    settings_path = args.input_dir.parent / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else None
    if settings:
        expected = settings.get("expected_dark_frame_count")
        if expected is not None and expected != len(paths):
            parser.error(f"Expected {expected} dark frames, found {len(paths)}")
        for key, actual in (("frame_height_px", master.shape[0]), ("frame_width_px", master.shape[1])):
            if settings.get(key) is not None and settings[key] != actual:
                parser.error(f"{key} differs from settings.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.save(output, master, allow_pickle=False)
    print(f"Saved {args.output}: {len(paths)} frames, shape={master.shape}, median stack")
    print("Use only with science images taken at matching exposure/gain, geometry and camera processing.")


if __name__ == "__main__":
    main()
