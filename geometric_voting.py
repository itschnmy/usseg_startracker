from dataclasses import dataclass
from math import atan, tan
from typing import List, Tuple
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

    def coordinate_angles(self, vector_2d: Tuple[float, float]) -> Tuple[float, float]:
        x, y = vector_2d
        ra = atan((self.x_resolution / 2.0 - x) / (self.x_resolution / 2.0 / tan(self.x_fov / 2.0)))
        y_fov = self.x_fov * (self.y_resolution / self.x_resolution)
        de = atan((self.y_resolution / 2.0 - y) / (self.y_resolution / 2.0 / tan(y_fov / 2.0))) 
        return ra, de
    
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


def geometric_voting_star_id(
    database_bytes: bytes,
    stars: List[Star],
    catalog: List[CatalogStar],
    camera: Camera,
    tolerance: float
) -> List[StarIdentifier]:
    db = KVectorDatabase(database_bytes)
    identified: List[StarIdentifier] = []

    for i in range(len(stars)):
        votes = [0] * len(catalog)

        ra1, de1 = camera.coordinate_angles((stars[i].x, stars[i].y))

        for j in range(len(stars)):
            if i == j:
                continue

            ra2, de2 = camera.coordinate_angles((stars[j].x, stars[j].y))
            gcd = great_circle_distance(ra1, de1, ra2, de2)

            lower_bound = max(db.min_distance, gcd - tolerance)
            upper_bound = min(db.max_distance, gcd + tolerance)

            if upper_bound <= lower_bound:
                continue

            returned_pairs = db.find_possible_star_pairs_approx(lower_bound, upper_bound)

            for first, second in returned_pairs:
                votes[first] += 1
                votes[second] += 1

        index_of_max = max(range(len(votes)), key=lambda idx: votes[idx])
        identified.append(StarIdentifier(star_index=i, catalog_index=index_of_max))

    return identified