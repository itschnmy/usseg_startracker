import numpy as np
from .estimators import QUESTEstimator, TRIADEstimator

class RelativeAttitudeDeterminator:
    """
    Relative Attitude Determinator module.
    Responsible for estimating the relative attitude quaternion of the camera/body frame
    with respect to a previous reference frame using tracked star unit vectors.
    Used during Tracking mode.
    """
    def __init__(self, method="QUEST"):
        method_upper = method.upper()
        if method_upper == "QUEST":
            self.estimator = QUESTEstimator()
        elif method_upper == "TRIAD":
            self.estimator = TRIADEstimator()
        else:
            raise ValueError(f"Unsupported relative attitude estimation method: {method}")
            
    def estimate_relative(self, body_frame_curr, body_frame_prev):
        """
        Estimate the relative attitude quaternion that rotates the previous body frame
        to the current body frame.
        
        Args:
            body_frame_curr: (3, N) array of star unit vectors in the current camera frame.
            body_frame_prev: (3, N) array of star unit vectors in the previous camera frame.
            
        Returns:
            q_rel: (4,) relative quaternion rotating body_frame_prev to body_frame_curr.
        """
        # Solving body_frame_curr = R_rel * body_frame_prev
        # Here body_frame_curr acts as the body frame and body_frame_prev acts as the reference (inertial) frame.
        return self.estimator.estimate(body_frame_curr, body_frame_prev)
