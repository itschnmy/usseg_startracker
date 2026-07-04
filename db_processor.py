import numpy as np
from detector import StarDetector
from CameralModel import CameraModel
from mortari_kvector import build_sla_database,  identify_sla

""""Temporary script to process tetra database for mortari kvector"""

def load_tetra_npz_as_catalog(npz_path: str, max_mag: float = 6.0):
    """
    Use default_database.npz as star catalog input.

    star_table[:, 2:5] = catalog unit vectors
    star_table[:, 5]   = magnitude
    star_catalog_IDs   = IDs
    """
    data = np.load(npz_path)

    star_table = data["star_table"]
    star_ids = data["star_catalog_IDs"]

    mask = star_table[:, 5] <= max_mag

    catalog_vectors = star_table[mask, 2:5]
    catalog_ids = star_ids[mask]

    return catalog_vectors, catalog_ids


def detect_to_observed_vectors(
    image,
    sigma_threshold: float,
    min_area: float,
    hfov_deg: float,
    max_stars: int | None = None,
):
    height, width = image.shape[:2]

    detector = StarDetector(
        sigma_threshold=sigma_threshold,
        min_area=min_area
    )

    detected_stars = detector.process(image)

    if len(detected_stars) == 0:
        raise ValueError("No stars detected.")

    # keep brightest stars only to reduce false detections. (?)
    detected_stars = sorted(
        detected_stars,
        key=lambda s: s.intensity,
        reverse=True
    )

    if max_stars is not None:
        detected_stars = detected_stars[:max_stars]

    camera = CameraModel(
        hfov_deg=hfov_deg,
        width=width,
        height=height
    )

    observed_vectors = []
    centroid_pixels = []

    for star in detected_stars:
        x, y = star.position

        centroid_pixels.append((float(x), float(y)))
        observed_vectors.append(camera.pixel_to_vector(float(x), float(y)))

    observed_vectors = np.array(observed_vectors)

    return observed_vectors, detected_stars, centroid_pixels