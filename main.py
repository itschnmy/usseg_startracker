import cv2
import sys
from mortari_kvector import build_sla_database, identify_sla
from db_processor import load_tetra_npz_as_catalog, detect_to_observed_vectors
from visualizer import visualizer
from math import radians, tan, atan, sqrt, degrees

def main():
    """image_path: str,
    npz_catalog_path: str,
    hfov_deg: float,
    sigma_threshold: float = 6.0,
    min_area: float = 3.0,
    max_mag: float = 6.0,
    eps_arcsec: float = 60.0,
    max_stars: int = 12,"""

    image_path = str(sys.argv[1])
    npz_catalog_path = str(sys.argv[2]) # (?)
    hfov_deg = float(sys.argv[3])
    sigma_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    min_area = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    max_mag = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
    eps_arcsec = float(sys.argv[7]) if len(sys.argv) > 7 else 60.0
    max_stars = int(sys.argv[8]) if len(sys.argv) > 8 else 12

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")


    height, width = image.shape[:2]

    hfov_rad = radians(hfov_deg)

    focal_px = (width / 2.0) / tan(hfov_rad / 2.0)

    pair_fov_rad = 2.0 * atan(
        sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2) / focal_px
    )

    pair_fov_deg = degrees(pair_fov_rad)

    print("Camera HFOV:", hfov_deg)
    print("Catalog pair FOV:", pair_fov_deg)


    # 1. Load catalog from default_database.npz (consider magnitude, new db etc.)
    catalog_vectors, catalog_ids = load_tetra_npz_as_catalog(
        npz_catalog_path,
        max_mag=max_mag
    )

    print("Catalog stars used:", len(catalog_vectors))

    # 2. Build Mortari-style I, J, K database
    db = build_sla_database(
        catalog_vectors=catalog_vectors,
        fov_deg=pair_fov_deg,      # not necessarily the same as camera hfov_deg
        eps_arcsec=eps_arcsec,
        catalog_ids=catalog_ids
    )

    print("Catalog star-pairs used:", db.m)

    # 3. Detect centroids and convert to observed unit vectors
    observed_vectors, detected_stars, centroid_pixels = detect_to_observed_vectors(
        image=image,
        sigma_threshold=sigma_threshold,
        min_area=min_area,
        hfov_deg=hfov_deg,
        max_stars=max_stars
    )

    with open("centroids.txt", "w") as f:
        for x, y in centroid_pixels:
            f.write(f"{x} {y}\n")
    visualizer(image)

    print("Detected stars used:", len(observed_vectors))
    print("Centroids:")
    for i, p in enumerate(centroid_pixels):
        print(f"  obs {i}: x={p[0]:.2f}, y={p[1]:.2f}")

    # 4. Run Mortari SLA identification
    result = identify_sla(
        db=db,
        observed_vectors=observed_vectors,
        eps_arcsec=eps_arcsec
    )

    print("Status:", result.status)

    if result.status == "ok":
        print("Matched observed stars to catalog IDs:")
        for obs_idx, catalog_id in result.obs_to_catalog_id.items():
            x, y = centroid_pixels[obs_idx]
            print(
                f"  observed {obs_idx} at ({x:.2f}, {y:.2f}) "
                f"-> catalog ID {catalog_id}"
            )

        if result.ambiguous:
            print("Ambiguous matches:")
            print(result.ambiguous)

    else:
        print("No valid matches found.")

    return result, detected_stars, centroid_pixels


if __name__ == "__main__":
    main()