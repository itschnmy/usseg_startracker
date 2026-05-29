from dataclasses import dataclass
from math import atan, tan
from typing import List, Tuple
from itertools import product
from collections import Counter

from kvector import (
    CatalogStar,
    KVectorDatabase,
    great_circle_distance
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


from dataclasses import dataclass
from math import tan, acos
from typing import List, Tuple
from itertools import product
from collections import Counter
import numpy as np

from kvector import (
    CatalogStar,
    KVectorDatabase,
    great_circle_distance
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

        # focal length in pixels, assuming square pixels
        self.f = (x_resolution / 2.0) / tan(x_fov / 2.0)

    def pixel_to_vector(self, star: Star) -> np.ndarray:
        x = (star.x - self.cx) / self.f
        y = -(star.y - self.cy) / self.f
        z = 1.0

        v = np.array([x, y, z], dtype=float)
        return v / np.linalg.norm(v)


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return acos(dot)


def catalog_angle(catalog: List[CatalogStar], i: int, j: int) -> float:
    return great_circle_distance(
        catalog[i].raj2000, catalog[i].dej2000,
        catalog[j].raj2000, catalog[j].dej2000
    )


def get_top_candidates(
    db: KVectorDatabase,
    stars: List[Star],
    catalog: List[CatalogStar],
    camera: Camera,
    tolerance: float,
    top_k_per_star: int = 3
):
    camera_vectors = [camera.pixel_to_vector(s) for s in stars]
    candidates = []

    for i in range(len(stars)):
        votes = Counter()

        for j in range(len(stars)):
            if i == j:
                continue

            obs_angle = angle_between_vectors(camera_vectors[i], camera_vectors[j])

            lower = max(db.min_distance, obs_angle - tolerance)
            upper = min(db.max_distance, obs_angle + tolerance)

            if upper <= lower:
                continue

            pairs = db.find_possible_star_pairs_approx(lower, upper)

            for a, b in pairs:
                votes[a] += 1
                votes[b] += 1

        top = [idx for idx, count in votes.most_common(top_k_per_star)]

        if len(top) == 0:
            top = list(range(min(top_k_per_star, len(catalog))))

        candidates.append(top)

    return candidates, camera_vectors


def global_consistency_filter(
    stars: List[Star],
    catalog: List[CatalogStar],
    candidates: List[List[int]],
    camera_vectors: List[np.ndarray],
    tolerance: float,
    min_score_ratio: float = 0.70
):
    n = len(stars)

    if n < 3:
        raise ValueError("Need at least 3 detected stars for global consistency filtering.")

    obs_angles = {}
    for i in range(n):
        for j in range(i + 1, n):
            obs_angles[(i, j)] = angle_between_vectors(camera_vectors[i], camera_vectors[j])

    best_assignment = None
    best_score = -1
    best_total = 0

    for assignment in product(*candidates):
        # one catalog star cannot represent two different observed stars
        if len(set(assignment)) < len(assignment):
            continue

        score = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                total += 1

                obs = obs_angles[(i, j)]
                cat = catalog_angle(catalog, assignment[i], assignment[j])

                if abs(obs - cat) <= tolerance:
                    score += 1

        if score > best_score:
            best_score = score
            best_total = total
            best_assignment = assignment

    if best_assignment is None:
        return []

    score_ratio = best_score / best_total

    if score_ratio < min_score_ratio:
        print(f"Rejected: global consistency score too low: {best_score}/{best_total}")
        return []

    return [
        StarIdentifier(
            star_index=i,
            catalog_index=best_assignment[i],
            weight=best_score
        )
        for i in range(n)
    ]


def geometric_voting_star_id(
    database_bytes: bytes,
    stars: List[Star],
    catalog: List[CatalogStar],
    camera: Camera,
    tolerance: float,
    top_k_per_star: int = 8,
    min_score_ratio: float = 0.70
) -> List[StarIdentifier]:

    db = KVectorDatabase(database_bytes)

    candidates, camera_vectors = get_top_candidates(
        db=db,
        stars=stars,
        catalog=catalog,
        camera=camera,
        tolerance=tolerance,
        top_k_per_star=top_k_per_star
    )

    return global_consistency_filter(
        stars=stars,
        catalog=catalog,
        candidates=candidates,
        camera_vectors=camera_vectors,
        tolerance=tolerance,
        min_score_ratio=min_score_ratio
    )