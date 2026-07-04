import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import Optional, Dict, List, Any


# ============================================================
# Basic helpers
# ============================================================

def normalize_rows(A):
    """
    Normalize one vector or an array of vectors.

    Input:
        A: shape (3,) or (N, 3)

    Output:
        Unit-normalized array with the same shape.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim == 1:
        norm = np.linalg.norm(A)
        if norm == 0:
            raise ValueError("Zero vector cannot be normalized.")
        return A / norm

    norms = np.linalg.norm(A, axis=1)
    if np.any(norms == 0):
        raise ValueError("Input contains zero vector.")

    return A / norms[:, None]


def arcsec_to_rad(arcsec):
    return np.deg2rad(float(arcsec) / 3600.0)


def radec_to_unit(ra_deg, dec_deg):
    """
    Convert catalog RA/Dec in degrees to inertial unit vectors.

    This is optional. If your catalog already gives unit vectors,
    you do not need this function.
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.column_stack((x, y, z))


# ============================================================
# Data structures
# ============================================================

@dataclass
class SLADatabase:
    catalog_vectors: np.ndarray
    catalog_ids: np.ndarray

    # Sorted admissible catalog star-pair index vectors.
    # These correspond to the paper's I and J vectors.
    I: np.ndarray
    J: np.ndarray

    # Paper K-vector.
    K: np.ndarray

    # Line parameters:
    # cos(theta) = a1 * k + a0
    a1: float
    a0: float

    fov_rad: float
    eps_rad: float

    @property
    def m(self):
        return int(len(self.K))


@dataclass
class SLAResult:
    status: str
    obs_to_catalog_index: Dict[int, int]
    obs_to_catalog_id: Dict[int, Any]
    reference_observed: Optional[int]
    reference_catalog_index: Optional[int]
    ambiguous: Dict[int, List[int]]
    unassigned: List[int]


# ============================================================
# Ground Program:
# Build I, J, K from catalog
# ============================================================

def build_sla_database(
    catalog_vectors,
    fov_deg,
    eps_arcsec=10.0,
    catalog_ids=None,
    keep_pair_dots=None,
):
    """
    Paper-faithful Mortari K-vector database builder.

    This corresponds to the paper's Ground Program.

    Steps:
        1. Normalize catalog vectors.
        2. Build all admissible catalog star pairs:
               v_i^T v_j >= cos(FOV)
        3. Store their dot products in P.
        4. Sort P into S, carrying I and J along with it.
        5. Build K from S.
        6. Discard S.

    Important:
        The paper says K substitutes S, so this implementation does not
        keep S for runtime edge filtering.

    Parameters
    ----------
    catalog_vectors:
        Catalog unit vectors, shape (N, 3).

    fov_deg:
        Star tracker FOV aperture used for admissible catalog pairs.
        This should be the maximum angular separation between two stars
        that can appear together in the image.

    eps_arcsec:
        Sensor angular precision epsilon.

    catalog_ids:
        Optional external catalog IDs.

    keep_pair_dots:
        Ignored. Present only so your old main.py will not crash if it
        still passes keep_pair_dots=True.
    """
    V = normalize_rows(catalog_vectors)
    n = len(V)

    if n < 2:
        raise ValueError("Need at least two catalog stars.")

    if catalog_ids is None:
        catalog_ids = np.arange(n)
    else:
        catalog_ids = np.asarray(catalog_ids)
        if len(catalog_ids) != n:
            raise ValueError("catalog_ids must have same length as catalog_vectors.")

    fov_rad = np.deg2rad(float(fov_deg))
    cos_fov = np.cos(fov_rad)
    eps_rad = arcsec_to_rad(eps_arcsec)

    I_chunks = []
    J_chunks = []
    P_chunks = []

    # Paper admissible condition:
    # v_i^T v_j >= cos(theta_FOV)
    for i in range(n - 1):
        dots = V[i + 1:] @ V[i]
        rel = np.where(dots >= cos_fov)[0]

        if rel.size == 0:
            continue

        js = rel + i + 1

        I_chunks.append(np.full(js.size, i, dtype=np.int64))
        J_chunks.append(js.astype(np.int64))
        P_chunks.append(dots[rel])

    if not P_chunks:
        raise ValueError("No admissible catalog star pairs. Increase FOV or check catalog.")

    I = np.concatenate(I_chunks)
    J = np.concatenate(J_chunks)
    P = np.concatenate(P_chunks)

    # Sort P into S while preserving matching I, J.
    order = np.argsort(P, kind="mergesort")
    S = P[order]
    I = I[order]
    J = J[order]

    m = len(S)

    if m < 2:
        raise ValueError("Need at least two admissible pairs to build K-vector.")

    # Paper:
    # D = [S(m) - S(1)] / (m - 1)
    D = (S[-1] - S[0]) / (m - 1)

    if not np.isfinite(D) or D <= 0:
        raise ValueError("Degenerate dot-product distribution; cannot build K-vector.")

    # Paper line:
    # cos(theta) = a1*k + a0
    # a1 = mD / (m - 1)
    # a0 = S(1) - a1 - D/2
    #
    # Paper indexing is 1-based, so k = 1, 2, ..., m.
    a1 = m * D / (m - 1)
    a0 = S[0] - a1 - D / 2.0

    paper_k = np.arange(1, m + 1, dtype=float)
    line_values = a1 * paper_k + a0

    # K(k) is the number of S elements satisfying:
    # S(j) <= a1*k + a0
    #
    # This gives K(1)=0 and K(m)=m, as in the paper.
    K = np.searchsorted(S, line_values, side="right").astype(np.int64)

    # Do not store S. The paper's K-vector version replaces S.
    return SLADatabase(
        catalog_vectors=V,
        catalog_ids=catalog_ids,
        I=I,
        J=J,
        K=K,
        a1=float(a1),
        a0=float(a0),
        fov_rad=fov_rad,
        eps_rad=eps_rad,
    )


# ============================================================
# K-vector SPIT:
# One observed angular separation -> candidate catalog pairs
# ============================================================

def _K_at(db, paper_k):
    """
    Access K using paper-style 1-based index.

    K stores counts, so the returned value can be used directly
    as a Python zero-based boundary.
    """
    paper_k = int(paper_k)

    if paper_k <= 1:
        return 0

    if paper_k >= db.m:
        return db.m

    return int(db.K[paper_k - 1])


def kvector_spit(db, observed_dot, eps_rad=None):
    """
    Paper-faithful K-vector Star-Pair Identification Technique.

    Input:
        observed_dot = s_i^T s_j = cos(theta)

    Output:
        Candidate catalog pairs as an array:
            [[cat_i, cat_j],
             [cat_i, cat_j],
             ...]

    Paper range:
        cos(theta + 2eps) <= v_i^T v_j <= cos(theta - 2eps)

    Important:
        This does not use S to remove edge candidates, because the paper's
        K-vector method replaces S. Therefore, a few nonmatching edge pairs
        may remain, exactly as the paper notes.
    """
    if eps_rad is None:
        eps_rad = db.eps_rad

    c = float(np.clip(observed_dot, -1.0, 1.0))
    theta = float(np.arccos(c))

    # cos decreases as angle increases.
    low = float(np.cos(min(theta + 2.0 * eps_rad, np.pi)))
    high = float(np.cos(max(theta - 2.0 * eps_rad, 0.0)))

    # Paper Eq. 13:
    # l_bot = floor((cos(theta + 2eps) - a0) / a1)
    # l_top = ceil((cos(theta - 2eps) - a0) / a1)
    l_bot = int(np.floor((low - db.a0) / db.a1))
    l_top = int(np.ceil((high - db.a0) / db.a1))

    # Paper Eq. 14:
    # k_start = K(l_bot) + 1
    # k_end   = K(l_top)
    #
    # In Python zero-based slicing:
    # start index = K(l_bot)
    # end index   = K(l_top)
    start = _K_at(db, l_bot)
    end = _K_at(db, l_top)

    if end <= start:
        return np.empty((0, 2), dtype=np.int64)

    candidate_indices = np.arange(start, end, dtype=np.int64)

    return np.column_stack(
        (
            db.I[candidate_indices],
            db.J[candidate_indices],
        )
    )


# ============================================================
# Reference-Star SMIT
# ============================================================

def _catalog_id(db, idx):
    x = db.catalog_ids[int(idx)]
    return x.item() if hasattr(x, "item") else x


def identify_sla(db, observed_vectors, eps_arcsec=None):
    """
    Paper-faithful Search-Less Algorithm:

        K-vector SPIT + Reference-Star SMIT

    Input:
        observed_vectors:
            shape (Nobs, 3)
            Unit vectors of observed stars in camera/sensor frame.

    Output:
        SLAResult:
            observed star index -> catalog star index / catalog ID

    Notes:
        - This identifies star IDs.
        - It does not compute attitude.
        - It does not use magnitude.
        - It accepts up to floor(n/4) spikes, as described in the paper.
    """
    O = normalize_rows(observed_vectors)
    n = len(O)

    if n < 3:
        return SLAResult(
            status="failed: need at least 3 observed objects",
            obs_to_catalog_index={},
            obs_to_catalog_id={},
            reference_observed=None,
            reference_catalog_index=None,
            ambiguous={},
            unassigned=list(range(n)),
        )

    eps_rad = db.eps_rad if eps_arcsec is None else arcsec_to_rad(eps_arcsec)

    # Paper spike tolerance:
    # nsmax = floor(n / 4)
    nsmax = int(np.floor(n / 4.0))

    pair_cache = {}

    def candidates_for_observed_pair(a, b):
        key = (min(a, b), max(a, b))

        if key not in pair_cache:
            observed_dot = float(O[a] @ O[b])
            pair_cache[key] = kvector_spit(
                db=db,
                observed_dot=observed_dot,
                eps_rad=eps_rad,
            )

        return pair_cache[key]

    # Try each observed star as the reference star.
    # This is the direct implementation of "choose another reference star"
    # until a valid reference is found.
    for r in range(n):
        per_k_candidates = {}
        L = []
        void_count = 0

        # Build candidate sets for pairs [s_r, s_k], k != r.
        for k in range(n):
            if k == r:
                continue

            cands = candidates_for_observed_pair(r, k)
            per_k_candidates[k] = cands

            if len(cands) == 0:
                void_count += 1
                continue

            # L_r = {I_r, J_r}^T
            L.extend(cands[:, 0].astype(int).tolist())
            L.extend(cands[:, 1].astype(int).tolist())

        # Paper rule:
        # If too many void L_{r,k}, selected reference is probably spike/noisy.
        if void_count > nsmax:
            continue

        if len(L) == 0:
            continue

        histogram = Counter(L)
        top_two = histogram.most_common(2)

        l1, f1 = top_two[0]
        f2 = top_two[1][1] if len(top_two) > 1 else 0

        cand_sizes = [len(c) for c in per_k_candidates.values()]

        print(
            f"[SMIT debug] ref_obs={r}, "
            f"void={void_count}/{n-1}, nsmax={nsmax}, "
            f"cand_min={min(cand_sizes)}, "
            f"cand_med={np.median(cand_sizes):.1f}, "
            f"cand_max={max(cand_sizes)}, "
            f"top1_cat={l1}, top1={f1}, "
            f"top2={f2}, "
            f"need_top1>{3.0*n/4.0 - 1.0:.1f}, "
            f"need_top2<{1 + int(n/5)}"
        )

        # Paper Reference-Star SMIT rule:
        # f1 > 3n/4 - 1
        # f2 < 1 + int(n/5)
        if f1 > (3.0 * n / 4.0 - 1.0) and f2 < (1 + int(n / 5)):
            reference_catalog_index = int(l1)

            obs_to_cat = {
                r: reference_catalog_index
            }

            ambiguous = {}
            unassigned = []

            # Identify every other observed star using the reference star.
            for k, cands in per_k_candidates.items():
                hits = []

                for i, j in cands:
                    i = int(i)
                    j = int(j)

                    if i == reference_catalog_index:
                        hits.append(j)
                    elif j == reference_catalog_index:
                        hits.append(i)

                # Paper rule:
                # If the reference catalog index appears once in L_{r,k},
                # then the other pair member identifies star k.
                #
                # Do not deduplicate before this test, because "appears once"
                # means exactly once in the candidate list.
                if len(hits) == 1:
                    obs_to_cat[k] = int(hits[0])
                elif len(hits) > 1:
                    ambiguous[k] = list(dict.fromkeys(int(x) for x in hits))
                else:
                    unassigned.append(k)

            obs_to_id = {
                obs_idx: _catalog_id(db, cat_idx)
                for obs_idx, cat_idx in obs_to_cat.items()
            }

            return SLAResult(
                status="ok",
                obs_to_catalog_index=obs_to_cat,
                obs_to_catalog_id=obs_to_id,
                reference_observed=r,
                reference_catalog_index=reference_catalog_index,
                ambiguous=ambiguous,
                unassigned=unassigned,
            )

    return SLAResult(
        status="failed: no reference star passed the SMIT histogram rules",
        obs_to_catalog_index={},
        obs_to_catalog_id={},
        reference_observed=None,
        reference_catalog_index=None,
        ambiguous={},
        unassigned=list(range(n)),
    )