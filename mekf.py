import numpy as np
from math_utils import hat

def Xi_matrix(q):
    """Computes the 4x3 Xi matrix for scalar-first unit quaternion q = [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w]
    ], dtype=float)

def _q2rot_passive(q):
    """Converts scalar-first unit quaternion q = [w, x, y, z] to passive rotation matrix (Inertial -> Body)."""
    w, x, y, z = q
    R_active = np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z),         2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),         1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x**2 + y**2)]
    ], dtype=float)
    return R_active.T

class MEKFEstimator:
    """Multiplicative Extended Kalman Filter (MEKF) for attitude and gyro bias estimation."""
    def __init__(self, q0, bias0, P0=None, sigma_arw=0.01, sigma_rrw=0.0001, sigma_meas=0.001):
        """
        Parameters:
        -----------
        q0 : array_like (4,)
            Initial scalar-first quaternion [w, x, y, z].
        bias0 : array_like (3,)
            Initial gyroscope bias [bx, by, bz].
        P0 : array_like (6, 6), optional
            Initial error covariance matrix. Defaults to 0.1 * eye(6).
        sigma_arw : float
            Angular Random Walk standard deviation (white noise on rate, rad/s^(1/2)).
        sigma_rrw : float
            Rate Random Walk standard deviation (white noise on bias rate, rad/s^(3/2)).
        sigma_meas : float
            Star Tracker vector measurement standard deviation (rad).
        """
        self.q = np.array(q0, dtype=float)
        self.q /= np.linalg.norm(self.q)
        self.bias = np.array(bias0, dtype=float)
        
        if P0 is None:
            self.P = 0.1 * np.eye(6, dtype=float)
        else:
            self.P = np.array(P0, dtype=float)
            
        self.sigma_arw = sigma_arw
        self.sigma_rrw = sigma_rrw
        self.sigma_meas = sigma_meas

    def predict(self, w_gyro, dt):
        """
        Propagates state and covariance forward by dt.
        
        Parameters:
        -----------
        w_gyro : array_like (3,)
            Measured angular rate from gyroscope (rad/s).
        dt : float
            Time step (seconds).
        """
        w_gyro = np.asarray(w_gyro, dtype=float)
        
        # 1. State Prediction
        w_est = w_gyro - self.bias
        
        # Continuous quaternion derivative: q_dot = 0.5 * Xi(q) * w_est
        q_dot = 0.5 * Xi_matrix(self.q) @ w_est
        self.q += q_dot * dt
        self.q /= np.linalg.norm(self.q)
        if self.q[0] < 0.0:
            self.q = -self.q
            
        # Gyro bias is modeled as a random walk, so predicted bias remains the same.
        
        # 2. Covariance Prediction
        W = -hat(w_est)
        # 2nd-order Taylor approximation of state transition matrix Phi (6x6)
        Phi_11 = np.eye(3) + W * dt + 0.5 * (W @ W) * (dt**2)
        Phi_12 = -np.eye(3) * dt - 0.5 * W * (dt**2)
        
        Phi = np.zeros((6, 6), dtype=float)
        Phi[0:3, 0:3] = Phi_11
        Phi[0:3, 3:6] = Phi_12
        Phi[3:6, 3:6] = np.eye(3)
        
        # Discrete-time process noise covariance Q_d
        Q_d = np.zeros((6, 6), dtype=float)
        Q_d[0:3, 0:3] = (self.sigma_arw**2 * dt) * np.eye(3)
        Q_d[3:6, 3:6] = (self.sigma_rrw**2 * dt) * np.eye(3)
        
        self.P = Phi @ self.P @ Phi.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)  # Force symmetry

    def update(self, body_vectors, inertial_vectors):
        """
        Updates state and covariance using star vector measurements.
        Uses Murrell's sequential update scheme to process star vectors one-by-one.
        
        Parameters:
        -----------
        body_vectors : array_like (3, N)
            Unit vectors measured in the body frame.
        inertial_vectors : array_like (3, N)
            Corresponding reference unit vectors in the inertial frame.
        """
        body_vectors = np.asarray(body_vectors, dtype=float)
        inertial_vectors = np.asarray(inertial_vectors, dtype=float)
        
        N = body_vectors.shape[1]
        R_i = (self.sigma_meas**2) * np.eye(3, dtype=float)
        
        for i in range(N):
            y_i = body_vectors[:, i]
            r_i = inertial_vectors[:, i]
            
            # Predict observation: h_i = A(q) * r_i
            A = _q2rot_passive(self.q)
            h_i = A @ r_i
            
            # Measurement sensitivity matrix: H_i = [[h_i x], 0] (3x6)
            H_i = np.zeros((3, 6), dtype=float)
            H_i[:, 0:3] = hat(h_i)
            
            # Kalman gain: K_i = P * H_i^T * inv(H_i * P * H_i^T + R_i)
            S_i = H_i @ self.P @ H_i.T + R_i
            K_i = self.P @ H_i.T @ np.linalg.inv(S_i)
            
            # Measurement residual (innovation)
            residual = y_i - h_i
            
            # Update error state
            delta_x = K_i @ residual
            delta_theta = delta_x[0:3]
            delta_bias = delta_x[3:6]
            
            # Update covariance
            self.P = (np.eye(6) - K_i @ H_i) @ self.P
            self.P = 0.5 * (self.P + self.P.T)
            
            # Reset step: Inject errors back into global states
            self.q += 0.5 * Xi_matrix(self.q) @ delta_theta
            self.q /= np.linalg.norm(self.q)
            if self.q[0] < 0.0:
                self.q = -self.q
                
            self.bias += delta_bias
