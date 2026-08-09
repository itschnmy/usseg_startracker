import numpy as np
from .estimators import QUESTEstimator, DavenportQEstimator, TRIADEstimator

class AttitudeDeterminator:
    """
    Attitude Determinator module.
    Responsible for estimating the absolute spacecraft orientation (quaternion)
    from matched vectors in the camera frame and the inertial reference frame.
    Typically used during Lost-In-Space (LIS) mode.
    """
    def __init__(self, method="QUEST"):
        method_upper = method.upper()
        if method_upper == "QUEST":
            self.estimator = QUESTEstimator()
        elif method_upper == "DAVENPORT":
            self.estimator = DavenportQEstimator()
        elif method_upper == "TRIAD":
            self.estimator = TRIADEstimator()
        else:
            raise ValueError(f"Unsupported attitude estimation method: {method}")
            
    def estimate(self, body_frame, inertial_frame):
        """
        Estimate absolute attitude quaternion.
        
        Args:
            body_frame: (3, N) array of star vectors in the camera frame.
            inertial_frame: (3, N) array of corresponding vectors in the inertial reference frame.
            
        Returns:
            q: (4,) unit quaternion mapping inertial frame to body frame (passive convention).
        """
        return self.estimator.estimate(body_frame, inertial_frame)
