"""Star detection module (Python port of StarDetector.cpp/.h).

This module corresponds to the *Star Detector* block in your pipeline diagram.

Notes:
- Uses OpenCV (cv2) + NumPy.
- The uBody computation here matches the C++ placeholder: a simple pinhole
  model with assumed intrinsics (cx, cy, f). In a real system you should
  replace this with calibrated intrinsics + distortion correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OpenCV (cv2) is required for star_detector.py. Install with: pip install opencv-python"
    ) from e


@dataclass
class DetectedStar:
    # PRIMARY IDENTIFIER
    index: int
    uBody: np.ndarray  # shape (3,)

    # Support/intermediate data
    position: np.ndarray  # shape (2,) (x, y)
    intensity: float
    peak: int
    radius: float


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def calculate_uBody(
    x: float,
    y: float,
    *,
    cx: float = 320.0,
    cy: float = 240.0,
    f: float = 500.0,
) -> np.ndarray:
    """Placeholder pinhole mapping from pixel (x,y) -> unit vector in camera/body frame.

    Mirrors the logic in the C++ demo:
        x_norm = (x - cx) / f
        y_norm = (y - cy) / f
        u = normalize([x_norm, y_norm, 1])

    Parameters are intentionally exposed so you can plug in your real intrinsics.
    """

    x_norm = (x - cx) / f
    y_norm = (y - cy) / f
    u = np.array([x_norm, y_norm, 1.0], dtype=float)
    return _normalize(u)


class StarDetector:
    """Detect stars by global thresholding + contouring + centroiding."""

    def __init__(self, sigma_threshold: float = 3.0, min_area: int = 2):
        self.sigma_threshold = float(sigma_threshold)
        self.min_area = int(min_area)

        # Default intrinsics used by calculate_uBody (same as C++ placeholder)
        self.cx = 320.0
        self.cy = 240.0
        self.f = 500.0

    def set_intrinsics(self, cx: float, cy: float, f: float) -> None:
        """Optional helper to set the intrinsics used for uBody."""
        self.cx = float(cx)
        self.cy = float(cy)
        self.f = float(f)

    def process(self, image: np.ndarray) -> List[DetectedStar]:
        """Run detection on an image.

        Args:
            image: Either grayscale (H,W) / (H,W,1) or BGR (H,W,3).

        Returns:
            List of DetectedStar objects.
        """

        # 1) Grayscale
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0].copy()
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            # Keep consistent with the C++ flow where gray is 8-bit for thresholding.
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # 2) Global threshold: mean + sigma*std
        mean, std = cv2.meanStdDev(gray)
        threshold_val = float(mean[0, 0] + self.sigma_threshold * std[0, 0])
        threshold_val = min(threshold_val, 255.0)

        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

        # 3) Find contours
        contours, _hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected: List[DetectedStar] = []
        star_id = 0

        # 4) For each contour: centroiding on ROI with local background subtraction
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            roi_float = roi.astype(np.float32)
            bg_mean_roi = float(np.mean(roi_float))
            roi_sub = roi_float - bg_mean_roi
            roi_sub = np.maximum(roi_sub, 0.0)

            m00 = float(np.sum(roi_sub))
            if m00 == 0.0:
                continue

            # centroid
            # (m10, m01) equivalent: sum(x*I), sum(y*I) within ROI coords
            ys, xs = np.indices(roi_sub.shape)
            m10 = float(np.sum(xs * roi_sub))
            m01 = float(np.sum(ys * roi_sub))

            cx_local = m10 / m00
            cy_local = m01 / m00
            global_x = float(x + cx_local)
            global_y = float(y + cy_local)

            peak = int(np.max(roi))
            radius = float(np.sqrt(area / np.pi))

            u_body = calculate_uBody(global_x, global_y, cx=self.cx, cy=self.cy, f=self.f)

            detected.append(
                DetectedStar(
                    index=star_id,
                    uBody=u_body,
                    position=np.array([global_x, global_y], dtype=float),
                    intensity=m00,
                    peak=peak,
                    radius=radius,
                )
            )
            star_id += 1

        return detected
