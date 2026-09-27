"""Star detection module (Python port of StarDetector.cpp/.h with adaptive backgrounding).

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
    """Detect stars by adaptive background subtraction + thresholding + centroiding."""

    def __init__(
        self,
        sigma_threshold: float = 3.0,
        min_area: int = 1,
        max_area: int = 50,
        use_tophat: bool = True,
        tophat_kernel_size: int = 7,
    ):
        self.sigma_threshold = float(sigma_threshold)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.use_tophat = bool(use_tophat)
        self.tophat_kernel_size = int(tophat_kernel_size)

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

        # 2) Adaptive background subtraction (Top-Hat)
        if self.use_tophat and self.tophat_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.tophat_kernel_size, self.tophat_kernel_size)
            )
            filtered = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        else:
            filtered = gray

        # 3) Threshold: mean + sigma*std on filtered image
        mean, std = cv2.meanStdDev(filtered)
        threshold_val = float(mean[0, 0] + self.sigma_threshold * std[0, 0])
        threshold_val = max(threshold_val, 2.0)
        threshold_val = min(threshold_val, 255.0)

        _, binary = cv2.threshold(filtered, threshold_val, 255, cv2.THRESH_BINARY)

        # 4) Connected components with stats to avoid polygon degenerate area issues
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        detected: List[DetectedStar] = []
        star_id = 0

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_area or area > self.max_area:
                continue

            roi = filtered[y : y + h, x : x + w].astype(np.float32)
            m00 = float(np.sum(roi))
            if m00 <= 0.0:
                continue

            # Centroid (m10, m01) within ROI coords
            ys, xs = np.indices(roi.shape)
            m10 = float(np.sum(xs * roi))
            m01 = float(np.sum(ys * roi))

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
