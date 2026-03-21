import cv2
import numpy as np
from dataclasses import dataclass

""" GUIDANCE
Function: To give a list of detected centroids of top brightness stars from an image. The logic of filtering is based on the sigma threshold, 
which is basically a parameter that increase the average brightness of the whole image in which we used to be a limit to take the top brightness stars, and 
the min area which is the minimum area of a dot could be to reduce the possibility of detecting noises as stars
Executing: Command python3 run_det.py to run this detector """

@dataclass
class DetectedStar:
    index: int
    uBody: np.ndarray
    position: np.ndarray
    intensity: float
    peak: int
    radius: float


def calculate_uBody(x, y):
    # Assumed camera intrinsic parameters
    cx = 320.0
    cy = 240.0
    f = 500.0

    x_norm = (x - cx) / f
    y_norm = (y - cy) / f

    u = np.array([x_norm, y_norm, 1.0])
    return u / np.linalg.norm(u)

class StarDetector:

    def __init__(self, sigma_threshold=6, min_area=5): # *** ADJUST HERE ***
        self.sigma_threshold = sigma_threshold
        self.min_area = min_area

    def process(self, image):

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        # Background statistics
        mean = np.mean(gray)
        std = np.std(gray)

        threshold_val = mean + self.sigma_threshold * std
        threshold_val = min(threshold_val, 255)

        # Binary threshold
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        # Find contours
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

            # Subtract background
            bg_mean = np.mean(roi)
            roi_sub = roi - bg_mean
            roi_sub = np.clip(roi_sub, 0, None)

            # Compute centroid via image moments
            M = cv2.moments(roi_sub)

            if M["m00"] == 0:
                continue

            cx_local = M["m10"] / M["m00"]
            cy_local = M["m01"] / M["m00"]

            global_x = x + cx_local
            global_y = y + cy_local

            # Peak brightness
            peak = int(np.max(roi))

            # Estimated radius
            radius = np.sqrt(area / np.pi)

            star = DetectedStar(
                index=star_id,
                position=np.array([global_x, global_y]),
                intensity=M["m00"],
                peak=peak,
                radius=radius,
                uBody=calculate_uBody(global_x, global_y)
            )

            detected_stars.append(star)
            star_id += 1

        return detected_stars