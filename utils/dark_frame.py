"""Master-dark calibration for the pipeline's 8-bit grayscale images."""
from pathlib import Path

import cv2
import numpy as np


def validate_dark(dark):
    dark = np.asarray(dark)
    if (dark.ndim != 2 or dark.size == 0 or not np.issubdtype(dark.dtype, np.number)
            or np.iscomplexobj(dark) or not np.isfinite(dark).all()
            or dark.min() < 0 or dark.max() > 255):
        raise ValueError("Master dark must be a finite 2D grayscale array in range 0..255")
    return dark.astype(np.float32)


def load_master_dark(path):
    return validate_dark(np.load(path, allow_pickle=False))


def build_master_dark(directory):
    """Median-combine lossless 8-bit grayscale frames; never resize calibration."""
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".tif", ".tiff"})
    if len(paths) < 2:
        raise ValueError("Need at least two dark frames (BMP/PNG/TIFF)")
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read dark frame: {path}")
        if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f"Expected 8-bit grayscale dark frame: {path}")
        if frames and image.shape != frames[0].shape:
            raise ValueError(f"Dark frame dimensions do not match: {path}")
        frames.append(image)
    master = np.median(np.stack(frames), axis=0).astype(np.float32)
    return master, paths


def subtract_dark(image, master_dark):
    """Subtract before detection, avoiding unsigned wraparound; preserve input."""
    if image.dtype != np.uint8:
        raise ValueError("Dark subtraction currently requires an 8-bit image")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.shape != master_dark.shape:
        raise ValueError(f"Image shape {image.shape} differs from master dark {master_dark.shape}; do not resize dark frames")
    return np.rint(np.clip(image.astype(np.float32) - master_dark, 0, 255)).astype(np.uint8)
