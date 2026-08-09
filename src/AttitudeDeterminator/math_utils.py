import numpy as np

def hat(v):
    """
    Skew-symmetric matrix (hat operator) for a 3D vector.
    
    Args:
        v: Array-like of shape (3,)
        
    Returns:
        3x3 skew-symmetric matrix (np.ndarray of float64)
    """
    v = np.asarray(v, dtype=np.float64).flatten()
    if len(v) != 3:
        raise ValueError("hat operator requires a 3D vector")
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)

def unhat(M):
    """
    Unhat operator to recover a 3D vector from a skew-symmetric matrix.
    
    Args:
        M: 3x3 array-like
        
    Returns:
        3D vector (np.ndarray of float64)
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError("unhat operator requires a 3x3 matrix")
    return np.array([
        -M[1, 2],
         M[0, 2],
        -M[0, 1]
    ], dtype=np.float64)

def build_triad_basis(v1, v2, eps=1e-12):
    """
    Builds TRIAD orthonormal basis matrix T = [t1, t2, t3].
    
    Args:
        v1: First 3D vector (primary direction)
        v2: Second 3D vector (secondary direction)
        eps: Small tolerance for collinearity check
        
    Returns:
        3x3 orthonormal matrix (np.ndarray of float64)
    """
    v1 = np.asarray(v1, dtype=np.float64).flatten()
    v2 = np.asarray(v2, dtype=np.float64).flatten()
    
    n1 = np.linalg.norm(v1)
    if n1 < eps:
        raise ValueError("build_triad_basis: v1 is a near-zero vector.")
    t1 = v1 / n1
    
    c12 = np.cross(t1, v2)
    n12 = np.linalg.norm(c12)
    if n12 < eps:
        raise ValueError("build_triad_basis: input vectors are nearly collinear.")
    t2 = c12 / n12
    
    t3 = np.cross(t1, t2)
    # Since t1 and t2 are orthonormal unit vectors, t3 is already normalized by construction.
    
    T = np.column_stack((t1, t2, t3))
    return T

def rot2q(Q):
    """
    Converts a 3x3 rotation matrix Q to a scalar-first unit quaternion q = [w, x, y, z].
    Uses Shepperd's method to avoid numerical instabilities and division by zero.
    
    Args:
        Q: 3x3 rotation matrix
        
    Returns:
        (4,) unit quaternion (np.ndarray of float64)
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape != (3, 3):
        raise ValueError("rot2q requires a 3x3 rotation matrix.")
        
    tr = np.trace(Q)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (Q[2, 1] - Q[1, 2]) / S
        qy = (Q[0, 2] - Q[2, 0]) / S
        qz = (Q[1, 0] - Q[0, 1]) / S
    elif (Q[0, 0] > Q[1, 1]) and (Q[0, 0] > Q[2, 2]):
        S = np.sqrt(1.0 + Q[0, 0] - Q[1, 1] - Q[2, 2]) * 2.0
        qw = (Q[2, 1] - Q[1, 2]) / S
        qx = 0.25 * S
        qy = (Q[0, 1] + Q[1, 0]) / S
        qz = (Q[0, 2] + Q[2, 0]) / S
    elif Q[1, 1] > Q[2, 2]:
        S = np.sqrt(1.0 + Q[1, 1] - Q[0, 0] - Q[2, 2]) * 2.0
        qw = (Q[0, 2] - Q[2, 0]) / S
        qx = (Q[0, 1] + Q[1, 0]) / S
        qy = 0.25 * S
        qz = (Q[1, 2] + Q[2, 1]) / S
    else:
        S = np.sqrt(1.0 + Q[2, 2] - Q[0, 0] - Q[1, 1]) * 2.0
        qw = (Q[1, 0] - Q[0, 1]) / S
        qx = (Q[0, 2] + Q[2, 0]) / S
        qy = (Q[1, 2] + Q[2, 1]) / S
        qz = 0.25 * S
        
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    
    # Enforce positive scalar part for uniqueness
    if q[0] < 0.0:
        q = -q
    return q
