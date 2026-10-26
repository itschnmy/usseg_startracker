"""Save a dark-corrected 8-bit grayscale PNG without running the star solver."""
import argparse
from pathlib import Path

import cv2

from utils.dark_frame import load_master_dark, subtract_dark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dark-frame", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New lossless .png output")
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error("--output must be .png")
    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"Could not read image: {args.image}")
    corrected = subtract_dark(image, load_master_dark(args.dark_frame))
    ok, encoded = cv2.imencode('.png', corrected)
    if not ok:
        raise OSError("Could not encode corrected image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        output.write(encoded.tobytes())
    print(f"Saved dark-corrected image: {args.output}")


if __name__ == '__main__':
    main()
