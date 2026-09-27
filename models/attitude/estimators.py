import numpy as np
from .math_utils import build_triad_basis, rot2q

class AttitudeEstimator:
    """Base class for attitude estimators."""
    def estimate(self, body_frame, inertial_frame, weights=None):
        raise NotImplementedError("Subclasses must implement estimate()")

class TRIADEstimator(AttitudeEstimator):
    """
    TRIAD Attitude Estimator (Deterministic 2-vector method).
    Estimates the passive rotation matrix (Inertial -> Body).
    """
    def estimate(self, body_frame, inertial_frame, weights=None):
        body_frame = np.asarray(body_frame, dtype=np.float64)
        inertial_frame = np.asarray(inertial_frame, dtype=np.float64)
        
        # TRIAD needs at least 2 vectors.
        if body_frame.shape[1] < 2 or inertial_frame.shape[1] < 2:
            raise ValueError("TRIAD estimate requires at least 2 vector pairs.")
            
        rN1 = inertial_frame[:, 0]
        rN2 = inertial_frame[:, 1]
        rB1 = body_frame[:, 0]
        rB2 = body_frame[:, 1]
        
        # Build basis matrices
        Tn = build_triad_basis(rN1, rN2)
        Tb = build_triad_basis(rB1, rB2)
        
        # Passive rotation matrix A maps Inertial -> Body.
        # A = Tb @ Tn.T (since Tn maps Triad->Inertial, Tb maps Triad->Body)
        Q = Tb @ Tn.T
        return rot2q(Q)

class QUESTEstimator(AttitudeEstimator):
    """
    QUEST Attitude Estimator (Newton-Raphson on Shuster's Quartic with Sequential Rotation).
    Estimates the passive rotation quaternion (Inertial -> Body).
    """
    def estimate(self, body_frame, inertial_frame, weights=None, eps=1e-12):
        body_frame = np.asarray(body_frame, dtype=np.float64)
        inertial_frame = np.asarray(inertial_frame, dtype=np.float64)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("QUEST estimate requires N >= 2 matching vector pairs.")
            
        if N == 2:
            # Fallback to TRIAD for exactly 2 vector pairs
            triad = TRIADEstimator()
            return triad.estimate(body_frame, inertial_frame)
            
        # Parse weights
        if weights is None:
            w = np.ones(N, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != N:
                raise ValueError("Length of weights must match the number of vectors.")
        
        # B = sum(w_i * b_i * r_i^T)
        B = body_frame @ (inertial_frame * w).T
        sigma = np.trace(B)
        
        # Davenport K-matrix diagonal elements to choose sequential rotation axis
        K_diag = np.array([
            sigma,
            2.0 * B[0, 0] - sigma,
            2.0 * B[1, 1] - sigma,
            2.0 * B[2, 2] - sigma
        ], dtype=np.float64)
        
        k = np.argmax(K_diag)
        
        # Apply sequential rotation to move away from 180 deg singularity
        if k == 0:
            B_prime = B
        elif k == 1:
            B_prime = B * np.array([1.0, -1.0, -1.0])
        elif k == 2:
            B_prime = B * np.array([-1.0, 1.0, -1.0])
        else:
            B_prime = B * np.array([-1.0, -1.0, 1.0])
            
        sigma_prime = np.trace(B_prime)
        S_prime = B_prime + B_prime.T
        
        z_prime = np.array([
            B_prime[1, 2] - B_prime[2, 1],
            B_prime[2, 0] - B_prime[0, 2],
            B_prime[0, 1] - B_prime[1, 0]
        ], dtype=np.float64)
        
        S_sq_prime = S_prime @ S_prime
        kappa_prime = 0.5 * (np.trace(S_prime)**2 - np.trace(S_sq_prime))
        delta_prime = np.linalg.det(S_prime)
        
        a_prime = sigma_prime**2 - kappa_prime
        b_prime = sigma_prime**2 + np.dot(z_prime, z_prime)
        c_prime = delta_prime + np.dot(z_prime, S_prime @ z_prime)
        d_prime = np.dot(z_prime, S_sq_prime @ z_prime)
        
        # Newton-Raphson iteration for the maximum eigenvalue
        # Initial guess is the sum of weights
        lam = float(np.sum(w))
        
        for _ in range(50):
            lam2 = lam * lam
            f = lam2 * lam2 - (a_prime + b_prime) * lam2 - c_prime * lam + (a_prime * b_prime + c_prime * sigma_prime - d_prime)
            fp = 4.0 * lam * lam2 - 2.0 * (a_prime + b_prime) * lam - c_prime
            
            if abs(fp) < eps:
                break
                
            step = f / fp
            lam -= step
            
            if abs(step) < eps:
                break
                
        # Recover rotated active quaternion using Shuster's direct formulas
        alpha = lam**2 - sigma_prime**2 + kappa_prime
        beta = lam - sigma_prime
        gamma = (lam + sigma_prime) * alpha - delta_prime
        
        X = (alpha * np.eye(3) + beta * S_prime + S_sq_prime) @ z_prime
        
        q_prime = np.array([gamma, X[0], X[1], X[2]], dtype=np.float64)
        norm_q = np.linalg.norm(q_prime)
        if norm_q < eps:
            raise ValueError("QUEST failed: quaternion norm is zero.")
        q_prime /= norm_q
        
        # Apply inverse sequential rotation to recover the active quaternion q_active = q_seq * q_prime
        if k == 0:
            q_active = q_prime
        elif k == 1:
            q_active = np.array([-q_prime[1], q_prime[0], -q_prime[3], q_prime[2]], dtype=np.float64)
        elif k == 2:
            q_active = np.array([-q_prime[2], q_prime[3], q_prime[0], -q_prime[1]], dtype=np.float64)
        else:
            q_active = np.array([-q_prime[3], -q_prime[2], q_prime[1], q_prime[0]], dtype=np.float64)
            
        if q_active[0] < 0.0:
            q_active = -q_active
            
        # Conjugate active quaternion to yield the passive quaternion (Inertial -> Body)
        q_passive = np.array([q_active[0], -q_active[1], -q_active[2], -q_active[3]], dtype=np.float64)
        if q_passive[0] < 0.0:
            q_passive = -q_passive
            
        return q_passive

class DavenportQEstimator(AttitudeEstimator):
    """
    Davenport's q-method Attitude Estimator (Eigenvalue Decomposition).
    Estimates the passive rotation quaternion (Inertial -> Body).
    """
    def estimate(self, body_frame, inertial_frame, weights=None):
        body_frame = np.asarray(body_frame, dtype=np.float64)
        inertial_frame = np.asarray(inertial_frame, dtype=np.float64)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("Davenport Q estimate requires N >= 2 matching vector pairs.")
            
        if N == 2:
            # Fallback to TRIAD for exactly 2 vector pairs
            triad = TRIADEstimator()
            return triad.estimate(body_frame, inertial_frame)
            
        # Parse weights
        if weights is None:
            w = np.ones(N, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != N:
                raise ValueError("Length of weights must match the number of vectors.")
        
        # B = sum(w_i * b_i * r_i^T)
        B = body_frame @ (inertial_frame * w).T
        
        sigma = np.trace(B)
        S = B + B.T
        
        z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ], dtype=np.float64)
        
        # Davenport K-matrix (4x4)
        K = np.zeros((4, 4), dtype=np.float64)
        K[0, 0] = sigma
        K[0, 1:4] = z
        K[1:4, 0] = z
        K[1:4, 1:4] = S - sigma * np.eye(3)
        
        # Solve symmetric eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(K)
        
        # Active quaternion corresponds to eigenvalue with maximum magnitude
        max_idx = np.argmax(eigvals)
        q_active = eigvecs[:, max_idx]
        
        norm = np.linalg.norm(q_active)
        if norm > 1e-12:
            q_active /= norm
            
        if q_active[0] < 0.0:
            q_active = -q_active
            
        # Conjugate active quaternion to yield the passive quaternion (Inertial -> Body)
        q_passive = np.array([q_active[0], -q_active[1], -q_active[2], -q_active[3]], dtype=np.float64)
        if q_passive[0] < 0.0:
            q_passive = -q_passive
            
        return q_passive

class SVDEstimator(AttitudeEstimator):
    """
    Markley's SVD Solution to Wahba's Problem.
    Estimates the passive rotation quaternion (Inertial -> Body).
    """
    def estimate(self, body_frame, inertial_frame, weights=None):
        body_frame = np.asarray(body_frame, dtype=np.float64)
        inertial_frame = np.asarray(inertial_frame, dtype=np.float64)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("SVD estimate requires N >= 2 matching vector pairs.")
            
        if N == 2:
            # Fallback to TRIAD for exactly 2 vector pairs
            triad = TRIADEstimator()
            return triad.estimate(body_frame, inertial_frame)
            
        # Parse weights
        if weights is None:
            w = np.ones(N, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != N:
                raise ValueError("Length of weights must match the number of vectors.")
        
        # B = sum(w_i * b_i * r_i^T)
        B = body_frame @ (inertial_frame * w).T
        
        # Markley's SVD algorithm takes SVD of B.T
        U, S, Vh = np.linalg.svd(B.T)
        V = Vh.T
        
        # Calculate R_opt = U * diag([1, 1, det(U)*det(V)]) * V^T
        d = np.linalg.det(U) * np.linalg.det(V)
        D = np.diag([1.0, 1.0, d])
        
        R_opt = U @ D @ Vh
        
        # Convert passive rotation matrix to quaternion
        q_passive = rot2q(R_opt)
        return q_passive
