import numpy as np
from .math_utils import build_triad_basis, rot2q

class AttitudeEstimator:
    """Base class for attitude estimators."""
    def estimate(self, body_frame, inertial_frame):
        raise NotImplementedError("Subclasses must implement estimate()")

class TRIADEstimator(AttitudeEstimator):
    """TRIAD Attitude Estimator (Deterministic 2-vector method)."""
    def estimate(self, body_frame, inertial_frame):
        body_frame = np.asarray(body_frame, dtype=float)
        inertial_frame = np.asarray(inertial_frame, dtype=float)
        
        if body_frame.shape[1] < 2 or inertial_frame.shape[1] < 2:
            raise ValueError("TRIAD estimate requires at least 2 vector pairs.")
            
        rN1 = inertial_frame[:, 0]
        rN2 = inertial_frame[:, 1]
        rB1 = body_frame[:, 0]
        rB2 = body_frame[:, 1]
        
        Tn = build_triad_basis(rN1, rN2)
        Tb = build_triad_basis(rB1, rB2)
        
        Q = Tn @ Tb.T
        return rot2q(Q)

class QUESTEstimator(AttitudeEstimator):
    """QUEST Attitude Estimator (Newton-Raphson on Shuster's Quartic)."""
    def estimate(self, body_frame, inertial_frame, eps=1e-12):
        body_frame = np.asarray(body_frame, dtype=float)
        inertial_frame = np.asarray(inertial_frame, dtype=float)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("QUEST estimate requires N >= 2 matching vector pairs.")
            
        if N == 2:
            triad = TRIADEstimator()
            return triad.estimate(body_frame, inertial_frame)
            
        # B = sum(r_inertial * r_body^T)
        B = inertial_frame @ body_frame.T
        
        sigma = np.trace(B)
        S = B + B.T
        
        z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ], dtype=float)
        
        # Characteristic quartic coefficients
        # kappa = tr(adj(S)) = 0.5 * (tr(S)^2 - tr(S^2))
        S_sq = S @ S
        kappa = 0.5 * (np.trace(S)**2 - np.trace(S_sq))
        delta = np.linalg.det(S)
        
        a = sigma**2 - kappa
        b = sigma**2 + np.dot(z, z)
        c = delta + np.dot(z, S @ z)
        d = np.dot(z, S_sq @ z)
        
        # Newton-Raphson iteration for the maximum eigenvalue (lambda_max)
        # Initial guess is the sum of weights (which is N for unweighted vectors)
        lam = float(N)
        
        for _ in range(50):
            lam2 = lam * lam
            # f(lam) and f'(lam)
            f = lam2 * lam2 - (a + b) * lam2 - c * lam + (a * b + c * sigma - d)
            fp = 4.0 * lam * lam2 - 2.0 * (a + b) * lam - c
            
            if abs(fp) < eps:
                break
                
            step = f / fp
            lam -= step
            
            if abs(step) < eps:
                break
                
        # Calculate Rodrigues parameters (Gibbs vector)
        denom = (lam + sigma) * np.eye(3) - S
        try:
            p = np.linalg.solve(denom, z)
        except np.linalg.LinAlgError:
            # Singular case (e.g. 180 deg rotation), fallback to eigenvalues or raise
            raise ValueError("QUEST denom matrix is singular (near 180 degree rotation).")
            
        factor = 1.0 / np.sqrt(1.0 + np.dot(p, p))
        q = np.array([factor, p[0]*factor, p[1]*factor, p[2]*factor], dtype=float)
        
        if q[0] < 0.0:
            q = -q
        return q

class DavenportQEstimator(AttitudeEstimator):
    """Davenport's q-method Attitude Estimator (Eigenvalue Decomposition)."""
    def estimate(self, body_frame, inertial_frame):
        body_frame = np.asarray(body_frame, dtype=float)
        inertial_frame = np.asarray(inertial_frame, dtype=float)
        
        N = body_frame.shape[1]
        if N < 2 or inertial_frame.shape[1] != N:
            raise ValueError("Davenport Q estimate requires N >= 2 matching vector pairs.")
            
        if N == 2:
            triad = TRIADEstimator()
            return triad.estimate(body_frame, inertial_frame)
            
        # B = sum(r_inertial * r_body^T)
        B = inertial_frame @ body_frame.T
        
        sigma = np.trace(B)
        S = B + B.T
        
        z = np.array([
            B[1, 2] - B[2, 1],
            B[2, 0] - B[0, 2],
            B[0, 1] - B[1, 0]
        ], dtype=float)
        
        # Davenport K-matrix (4x4)
        K = np.zeros((4, 4), dtype=float)
        K[0, 0] = sigma
        K[0, 1:4] = z
        K[1:4, 0] = z
        K[1:4, 1:4] = S - sigma * np.eye(3)
        
        # Solve symmetric eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(K)
        
        # Quaternion corresponds to eigenvalue with maximum magnitude
        max_idx = np.argmax(eigvals)
        q = eigvecs[:, max_idx]
        
        norm = np.linalg.norm(q)
        if norm > 1e-12:
            q /= norm
            
        if q[0] < 0.0:
            q = -q
        return q
