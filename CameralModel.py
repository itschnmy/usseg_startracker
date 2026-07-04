from math import radians, tan
import numpy as np

# Convert centroids to vectors for Mortari alg
class CameraModel:
    def __init__(self, hfov_deg: float, width: int, height: int):
        self.hfov_deg = radians(hfov_deg)
        self.width = width
        self.height = height

        self.cx = width / 2.0
        self.cy = height / 2.0
        self.f = (width / 2.0) / tan(self.hfov_deg / 2.0)

    def pixel_to_vector(self, x: float, y: float) -> np.ndarray:
        X = (x - self.cx) / self.f
        Y = -(y - self.cy) / self.f
        Z = 1.0

        vector = np.array([X, Y, Z], dtype=float)
        vector /= np.linalg.norm(vector)

        return vector