import enum
import numpy as np
from .estimators import TRIADEstimator, QUESTEstimator, DavenportQEstimator

class ADCSMode(enum.Enum):
    DEGRADED_OR_COARSE = 1  # Only 2 vectors available (TRIAD)
    FINE_POINTING = 2       # 3 or more vectors available (QUEST/Davenport)

class AttitudeControlSystem:
    def __init__(self, use_davenport=False):
        self.triad = TRIADEstimator()
        self.quest = QUESTEstimator()
        self.davenport = DavenportQEstimator()
        self.use_davenport = use_davenport
        
    def process_sensor_data(self, body_frame, inertial_frame):
        body_frame = np.asarray(body_frame, dtype=float)
        inertial_frame = np.asarray(inertial_frame, dtype=float)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("ADCS Error: Less than 2 vectors provided or shape mismatch.")
            
        current_mode = ADCSMode.DEGRADED_OR_COARSE if N == 2 else ADCSMode.FINE_POINTING
        
        if current_mode == ADCSMode.DEGRADED_OR_COARSE:
            print("[ADCS] Operating in Coarse/Degraded Mode (TRIAD)...")
            q_estimated = self.triad.estimate(body_frame, inertial_frame)
        else:
            if self.use_davenport:
                print("[ADCS] Operating in Fine Pointing Mode (Davenport Q)...")
                q_estimated = self.davenport.estimate(body_frame, inertial_frame)
            else:
                print("[ADCS] Operating in Fine Pointing Mode (QUEST)...")
                q_estimated = self.quest.estimate(body_frame, inertial_frame)
                
        return q_estimated
