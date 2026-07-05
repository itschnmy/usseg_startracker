import numpy as np

def hat(v):
    """Skew-symmetric matrix (hat operator) for a 3D vector."""
    v = np.asarray(v).flatten()
    if len(v) != 3:
        raise ValueError("hat operator requires a 3D vector")
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)

def unhat(M):
    """Unhat operator to recover 3D vector from a skew-symmetric matrix."""
    M = np.asarray(M)
    if M.shape != (3, 3):
        raise ValueError("unhat operator requires a 3x3 matrix")
    return np.array([
        -M[1, 2],
         M[0, 2],
        -M[0, 1]
    ], dtype=float)

def build_triad_basis(v1, v2, eps=1e-12):
    """Builds TRIAD orthonormal basis matrix T = [t1, t2, t3]."""
    v1 = np.asarray(v1, dtype=float).flatten()
    v2 = np.asarray(v2, dtype=float).flatten()
    
    n1 = np.linalg.norm(v1)
    if n1 < eps:
        raise ValueError("build_triad_basis: v1 is near-zero vector.")
    t1 = v1 / n1
    
    c12 = np.cross(v1, v2)
    n12 = np.linalg.norm(c12)
    if n12 < eps:
        raise ValueError("build_triad_basis: input vectors are nearly collinear.")
    t2 = c12 / n12
    
    c13 = np.cross(t1, t2)
    n13 = np.linalg.norm(c13)
    if n13 < eps:
        raise ValueError("build_triad_basis: degenerate basis while building t3.")
    t3 = c13 / n13
    
    T = np.zeros((3, 3), dtype=float)
    T[:, 0] = t1
    T[:, 1] = t2
    T[:, 2] = t3
    return T

def rot2q(Q):
    """Converts a 3x3 rotation matrix Q to a scalar-first unit quaternion q = [w, x, y, z]."""
    Q = np.asarray(Q, dtype=float)
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
        
    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    
    # Ensure scalar component is positive
    if q[0] < 0.0:
        q = -q
    return q
