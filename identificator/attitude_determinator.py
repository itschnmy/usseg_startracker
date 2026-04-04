"""Attitude Determination module (Python port of header.h/function.cpp/estimator.cpp).

This module corresponds to the *Attitude Determinator* block in your pipeline diagram.

Implemented estimators:
- TRIAD: needs at least 2 vector pairs.
- QUEST (Davenport q-method via K-matrix eigenvector): needs N>=2. Falls back to
  TRIAD when N==2 (matching your C++ implementation).

Conventions:
- Vectors are 3D numpy arrays.
- Quaternions are numpy arrays shaped (4,) in (w, x, y, z) order.
- Estimated attitude is the rotation from *body/camera frame* to *inertial frame*
  (same as your C++ comments: Q = Tn * Tb^T).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

ERROR = 1e-12


def _as_vec3(v: np.ndarray | Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    return a


def normalize(v: np.ndarray | Sequence[float], eps: float = ERROR) -> np.ndarray:
    v = _as_vec3(v)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("normalize: near-zero vector")
    return v / n


def hat(v: np.ndarray | Sequence[float]) -> np.ndarray:
    """Skew (hat) operator."""
    # Mirrors the C++ implementation exactly (no implicit normalization).
    return hat_raw(v)


def hat_raw(v: np.ndarray | Sequence[float]) -> np.ndarray:
    v = _as_vec3(v)
    x, y, z = v
    return np.array(
        [[0.0, -z, y],
         [z, 0.0, -x],
         [-y, x, 0.0]],
        dtype=float,
    )


def unhat(M: np.ndarray) -> np.ndarray:
    """Unhat operator."""
    M = np.asarray(M, dtype=float).reshape(3, 3)
    return np.array([-M[1, 2], M[0, 2], -M[0, 1]], dtype=float)


def build_triad_basis(v1: np.ndarray | Sequence[float], v2: np.ndarray | Sequence[float], eps: float = ERROR) -> np.ndarray:
    """Build TRIAD orthonormal basis T = [t1 t2 t3]."""
    v1 = _as_vec3(v1)
    v2 = _as_vec3(v2)

    t1 = normalize(v1, eps=eps)

    c12 = np.cross(v1, v2)
    n12 = float(np.linalg.norm(c12))
    if n12 < eps:
        raise ValueError("build_triad_basis: input vectors are nearly collinear.")
    t2 = c12 / n12

    c13 = np.cross(t1, t2)
    n13 = float(np.linalg.norm(c13))
    if n13 < eps:
        raise ValueError("build_triad_basis: degenerate basis while building t3.")
    t3 = c13 / n13

    T = np.column_stack([t1, t2, t3])
    return T


def rotmat_to_quat(Q: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z).

    This is a numerically stable implementation that does not require scipy.
    """
    Q = np.asarray(Q, dtype=float).reshape(3, 3)

    tr = float(np.trace(Q))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0  # S = 4*qw
        qw = 0.25 * S
        qx = (Q[2, 1] - Q[1, 2]) / S
        qy = (Q[0, 2] - Q[2, 0]) / S
        qz = (Q[1, 0] - Q[0, 1]) / S
    else:
        # Find major diagonal element
        if (Q[0, 0] > Q[1, 1]) and (Q[0, 0] > Q[2, 2]):
            S = np.sqrt(1.0 + Q[0, 0] - Q[1, 1] - Q[2, 2]) * 2.0  # S = 4*qx
            qw = (Q[2, 1] - Q[1, 2]) / S
            qx = 0.25 * S
            qy = (Q[0, 1] + Q[1, 0]) / S
            qz = (Q[0, 2] + Q[2, 0]) / S
        elif Q[1, 1] > Q[2, 2]:
            S = np.sqrt(1.0 + Q[1, 1] - Q[0, 0] - Q[2, 2]) * 2.0  # S = 4*qy
            qw = (Q[0, 2] - Q[2, 0]) / S
            qx = (Q[0, 1] + Q[1, 0]) / S
            qy = 0.25 * S
            qz = (Q[1, 2] + Q[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Q[2, 2] - Q[0, 0] - Q[1, 1]) * 2.0  # S = 4*qz
            qw = (Q[1, 0] - Q[0, 1]) / S
            qx = (Q[0, 2] + Q[2, 0]) / S
            qy = (Q[1, 2] + Q[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=float)
    n = float(np.linalg.norm(q))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


class AttitudeEstimator(ABC):
    @abstractmethod
    def estimate(self, body_frame: Sequence[np.ndarray], inertial_frame: Sequence[np.ndarray]) -> np.ndarray:
        """Estimate quaternion q (w,x,y,z) rotating body->inertial."""
        raise NotImplementedError


class TRIADEstimator(AttitudeEstimator):
    def estimate(self, body_frame: Sequence[np.ndarray], inertial_frame: Sequence[np.ndarray]) -> np.ndarray:
        if len(body_frame) < 2 or len(inertial_frame) < 2:
            raise ValueError("TRIAD estimate requires at least 2 vector pairs.")

        rN1 = _as_vec3(inertial_frame[0])
        rN2 = _as_vec3(inertial_frame[1])
        rB1 = _as_vec3(body_frame[0])
        rB2 = _as_vec3(body_frame[1])

        Tn = build_triad_basis(rN1, rN2)
        Tb = build_triad_basis(rB1, rB2)

        Q = Tn @ Tb.T
        return rotmat_to_quat(Q)


class QUESTEstimator(AttitudeEstimator):
    def estimate(self, body_frame: Sequence[np.ndarray], inertial_frame: Sequence[np.ndarray]) -> np.ndarray:
        N = len(body_frame)
        if N < 2 or len(inertial_frame) != N:
            raise ValueError("QUEST estimate requires N >= 2 matching vector pairs.")

        # TRIAD fallback when N == 2 (matches the C++ code)
        if N == 2:
            return TRIADEstimator().estimate(body_frame, inertial_frame)

        B = np.zeros((3, 3), dtype=float)
        for i in range(N):
            rB = normalize(body_frame[i])
            rN = normalize(inertial_frame[i])
            B += np.outer(rN, rB)  # rN * rB^T

        sigma = float(np.trace(B))
        S = B + B.T

        z = np.array(
            [
                B[1, 2] - B[2, 1],
                B[2, 0] - B[0, 2],
                B[0, 1] - B[1, 0],
            ],
            dtype=float,
        )

        K = np.zeros((4, 4), dtype=float)
        K[0:3, 0:3] = S - sigma * np.eye(3)
        K[0:3, 3] = z
        K[3, 0:3] = z
        K[3, 3] = sigma

        # Max eigenvector of symmetric K
        evals, evecs = np.linalg.eigh(K)
        idx = int(np.argmax(evals))
        qv = evecs[:, idx]

        # C++ converts Vector4d [qx,qy,qz,qw] -> Quaternion(w,x,y,z)
        q = np.array([qv[3], qv[0], qv[1], qv[2]], dtype=float)
        q /= float(np.linalg.norm(q))
        return q
