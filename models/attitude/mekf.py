import numpy as np
from .math_utils import hat

def Xi_matrix(q):
    """Computes the 4x3 Xi matrix for scalar-first unit quaternion q = [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [-x, -y, -z],
        [ w, -z,  y],
        [ z,  w, -x],
        [-y,  x,  w]
    ], dtype=np.float64)

def _q2rot_passive(q):
    """Converts scalar-first unit quaternion q = [w, x, y, z] to passive rotation matrix (Inertial -> Body)."""
    w, x, y, z = q
    R_active = np.array([
        [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - w*z),         2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),         1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x**2 + y**2)]
    ], dtype=np.float64)
    return R_active.T

def q_mult(q1, q2):
    """Computes the Hamilton product of two scalar-first quaternions q1 and q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

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
        self.q = np.array(q0, dtype=np.float64)
        self.q /= np.linalg.norm(self.q)
        self.bias = np.array(bias0, dtype=np.float64)
        
        if P0 is None:
            self.P = 0.1 * np.eye(6, dtype=np.float64)
        else:
            self.P = np.array(P0, dtype=np.float64)
            
        self.sigma_arw = sigma_arw
        self.sigma_rrw = sigma_rrw
        self.sigma_meas = sigma_meas

    def predict(self, w_gyro, dt):
        """
        Propagates state and covariance forward by dt using a closed-form exponential map.
        
        Parameters:
        -----------
        w_gyro : array_like (3,)
            Measured angular rate from gyroscope (rad/s).
        dt : float
            Time step (seconds).
        """
        w_gyro = np.asarray(w_gyro, dtype=np.float64)
        
        # 1. State Prediction
        w_est = w_gyro - self.bias
        
        # Closed-form quaternion propagation using the exponential map (zeroth-order hold)
        omega_norm = np.linalg.norm(w_est)
        if omega_norm > 1e-12:
            theta = omega_norm * dt
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            dq = np.array([c, s * w_est[0] / omega_norm, s * w_est[1] / omega_norm, s * w_est[2] / omega_norm], dtype=np.float64)
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            
        self.q = q_mult(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        if self.q[0] < 0.0:
            self.q = -self.q
            
        # Gyro bias is modeled as a random walk, so predicted bias remains the same.
        
        # 2. Covariance Prediction
        W = -hat(w_est)
        # 2nd-order Taylor approximation of state transition matrix Phi (6x6)
        Phi_11 = np.eye(3, dtype=np.float64) + W * dt + 0.5 * (W @ W) * (dt**2)
        Phi_12 = -np.eye(3, dtype=np.float64) * dt - 0.5 * W * (dt**2)
        
        Phi = np.zeros((6, 6), dtype=np.float64)
        Phi[0:3, 0:3] = Phi_11
        Phi[0:3, 3:6] = Phi_12
        Phi[3:6, 3:6] = np.eye(3, dtype=np.float64)
        
        # Discrete-time process noise covariance Q_d
        Q_d = np.zeros((6, 6), dtype=np.float64)
        Q_d[0:3, 0:3] = (self.sigma_arw**2 * dt) * np.eye(3, dtype=np.float64)
        Q_d[3:6, 3:6] = (self.sigma_rrw**2 * dt) * np.eye(3, dtype=np.float64)
        
        self.P = Phi @ self.P @ Phi.T + Q_d
        self.P = 0.5 * (self.P + self.P.T)  # Force symmetry

    def update(self, body_vectors, inertial_vectors):
        """
        Updates state and covariance using star vector measurements.
        Uses Murrell's sequential update scheme with mathematically rigorous 2D projected measurements
        and the numerically stable Joseph covariance update form.
        
        Parameters:
        -----------
        body_vectors : array_like (3, N)
            Unit vectors measured in the body frame.
        inertial_vectors : array_like (3, N)
            Corresponding reference unit vectors in the inertial frame.
        """
        body_vectors = np.asarray(body_vectors, dtype=np.float64)
        inertial_vectors = np.asarray(inertial_vectors, dtype=np.float64)
        
        N = body_vectors.shape[1]
        R_i = (self.sigma_meas**2) * np.eye(2, dtype=np.float64)  # 2D measurement covariance
        
        for i in range(N):
            y_i = body_vectors[:, i]
            r_i = inertial_vectors[:, i]
            
            # Predict observation: h_i = A(q) * r_i
            A = _q2rot_passive(self.q)
            h_i = A @ r_i
            
            # Construct 2D projection matrix T_i = [t1, t2].T perpendicular to h_i
            if abs(h_i[0]) < 0.9:
                v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                v_temp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                
            t1 = np.cross(v_temp, h_i)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(h_i, t1)
            
            T_i = np.column_stack((t1, t2)).T  # Shape (2, 3)
            
            # 2D Measurement residual: res_2d = T_i @ (y_i - h_i)
            res_2d = T_i @ (y_i - h_i)
            
            # 2D Sensitivity matrix: H_i = [[-t2^T, 0], [t1^T, 0]] (shape 2x6)
            H_i = np.zeros((2, 6), dtype=np.float64)
            H_i[0, 0:3] = -t2
            H_i[1, 0:3] = t1
            
            # Kalman gain: K_i = P * H_i^T * inv(H_i * P * H_i^T + R_i) (shape 6x2)
            S_i = H_i @ self.P @ H_i.T + R_i
            K_i = self.P @ H_i.T @ np.linalg.inv(S_i)
            
            # Update error state
            delta_x = K_i @ res_2d
            delta_theta = delta_x[0:3]
            delta_bias = delta_x[3:6]
            
            # Update covariance using numerically stable Joseph form
            I_KH = np.eye(6, dtype=np.float64) - K_i @ H_i
            self.P = I_KH @ self.P @ I_KH.T + K_i @ R_i @ K_i.T
            self.P = 0.5 * (self.P + self.P.T)
            
            # Inject error state into global states
            theta_err = np.linalg.norm(delta_theta)
            if theta_err > 1e-12:
                c = np.cos(theta_err / 2.0)
                s = np.sin(theta_err / 2.0)
                dq = np.array([c, s * delta_theta[0] / theta_err, s * delta_theta[1] / theta_err, s * delta_theta[2] / theta_err], dtype=np.float64)
            else:
                dq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                
            self.q = q_mult(self.q, dq)
            self.q /= np.linalg.norm(self.q)
            if self.q[0] < 0.0:
                self.q = -self.q
                
            self.bias += delta_bias
