from dataclasses import dataclass
from math import atan, tan, acos
from itertools import permutations
from typing import List, Tuple, Dict
from collections import defaultdict

from kvector1 import (
    CatalogStar,
    KVectorDatabase,
    radec_to_unit_vector,
    vector_angle
)


@dataclass
class Star:
    x: float
    y: float
    radiusX: float = 0.0
    radiusY: float = 0.0
    magnitude: int = 0


@dataclass
class StarIdentifier:
    star_index: int
    catalog_index: int
    weight: int = 1


class Camera:
    def __init__(self, x_fov: float, x_resolution: int, y_resolution: int):
        self.x_fov = x_fov
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

        self.cx = x_resolution / 2.0
        self.cy = y_resolution / 2.0

        self.y_fov = 2.0 * atan((y_resolution / x_resolution) * tan(x_fov / 2.0))

        self.fx = self.cx / tan(self.x_fov / 2.0)
        self.fy = self.cy / tan(self.y_fov / 2.0)

    def pixel_to_unit_vector(self, pixel_xy: Tuple[float, float]) -> Tuple[float, float, float]:
        x, y = pixel_xy

        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy

        # camera frame: +z optical axis
        vx = xn
        vy = yn
        vz = 1.0

        norm = (vx * vx + vy * vy + vz * vz) ** 0.5
        return (vx / norm, vy / norm, vz / norm)


def load_stars_from_txt(txt_path: str) -> List[Star]:
    stars = []
    with open(txt_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Line {line_num}: expected at least 2 values: x y")

            x = float(parts[0])
            y = float(parts[1])
            stars.append(Star(x=x, y=y))

    return stars


def compute_observed_pair_angles(
    stars: List[Star],
    camera: Camera
) -> Dict[Tuple[int, int], float]:
    obs_vectors = [camera.pixel_to_unit_vector((s.x, s.y)) for s in stars]
    pair_angles = {}

    for i in range(len(obs_vectors)):
        for j in range(i + 1, len(obs_vectors)):
            pair_angles[(i, j)] = vector_angle(obs_vectors[i], obs_vectors[j])

    return pair_angles


def geometric_voting_star_id(
    database_bytes: bytes,
    stars: List[Star],
    catalog: List[CatalogStar],
    camera: Camera,
    tolerance: float,
    top_k_per_star: int = 6,
    final_verify_tolerance: float = 0.03
) -> List[StarIdentifier]:
    db = KVectorDatabase(database_bytes)
    obs_pair_angles = compute_observed_pair_angles(stars, camera)

    # Precompute catalog vectors
    cat_vectors = [radec_to_unit_vector(c.raj2000, c.dej2000) for c in catalog]

    # Score hypotheses (observed star -> catalog star)
    support = defaultdict(int)

    for (i, j), obs_angle in obs_pair_angles.items():
        lower_bound = max(db.min_distance, obs_angle - tolerance)
        upper_bound = min(db.max_distance, obs_angle + tolerance)

        if upper_bound <= lower_bound:
            continue

        returned_pairs = db.find_possible_star_pairs_approx(lower_bound, upper_bound)

        for a, b in returned_pairs:
            # two possible assignments for pair (i, j)
            support[(i, a)] += 1
            support[(j, b)] += 1

            support[(i, b)] += 1
            support[(j, a)] += 1

    # Top candidates for each observed star
    per_star_candidates: List[List[int]] = []
    for i in range(len(stars)):
        scored = [(c_idx, support[(i, c_idx)]) for c_idx in range(len(catalog))]
        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [c_idx for c_idx, score in scored if score > 0][:top_k_per_star]

        if not filtered:
            filtered = [scored[0][0]]

        per_star_candidates.append(filtered)

    # Global consistency search
    best_assignment = None
    best_cost = float("inf")

    def pair_cost(obs_i: int, obs_j: int, cat_i: int, cat_j: int) -> float:
        obs_angle = obs_pair_angles[(min(obs_i, obs_j), max(obs_i, obs_j))]
        cat_angle = vector_angle(cat_vectors[cat_i], cat_vectors[cat_j])
        return abs(obs_angle - cat_angle)

    # brute force over small candidate sets
    from itertools import product
    for candidate_tuple in product(*per_star_candidates):
        if len(set(candidate_tuple)) < len(candidate_tuple):
            continue

        total_cost = 0.0
        valid = True
        for i in range(len(stars)):
            for j in range(i + 1, len(stars)):
                err = pair_cost(i, j, candidate_tuple[i], candidate_tuple[j])
                total_cost += err
                if err > final_verify_tolerance:
                    valid = False
                    break
            if not valid:
                break

        if valid and total_cost < best_cost:
            best_cost = total_cost
            best_assignment = candidate_tuple

    # fallback if no assignment passes hard check
    if best_assignment is None:
        for candidate_tuple in product(*per_star_candidates):
            if len(set(candidate_tuple)) < len(candidate_tuple):
                continue

            total_cost = 0.0
            for i in range(len(stars)):
                for j in range(i + 1, len(stars)):
                    total_cost += pair_cost(i, j, candidate_tuple[i], candidate_tuple[j])

            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = candidate_tuple

    identified = []
    for obs_idx, cat_idx in enumerate(best_assignment):
        identified.append(
            StarIdentifier(
                star_index=obs_idx,
                catalog_index=cat_idx,
                weight=support[(obs_idx, cat_idx)]
            )
        )

    return identified