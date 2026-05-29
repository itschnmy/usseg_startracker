import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class DetectedStar:
    index: int
    position: np.ndarray
    intensity: float
    peak: int
    radius: float
    area: float


class StarDetector:
    def __init__(self, sigma_threshold, min_area):
        self.sigma_threshold = sigma_threshold
        self.min_area = min_area

    def process(self, image):
        if image is None:
            raise ValueError("Input image is None.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        mean = np.mean(gray)
        std = np.std(gray)

        threshold_val = mean + self.sigma_threshold * std
        threshold_val = min(threshold_val, 255)

        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected_stars = []
        star_id = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]

            local_bg = np.median(roi)
            roi_sub = roi - local_bg
            roi_sub = np.clip(roi_sub, 0, None)

            M = cv2.moments(roi_sub)

            if M["m00"] == 0:
                continue

            cx_local = M["m10"] / M["m00"]
            cy_local = M["m01"] / M["m00"]

            global_x = x + cx_local
            global_y = y + cy_local

            peak = int(np.max(roi))
            radius = np.sqrt(area / np.pi)

            detected_stars.append(
                DetectedStar(
                    index=star_id,
                    position=np.array([global_x, global_y]),
                    intensity=float(M["m00"]),
                    peak=peak,
                    radius=float(radius),
                    area=float(area)
                )
            )

            star_id += 1

        return detected_stars