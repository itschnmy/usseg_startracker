import math
from det_run import detector
from det_visualize import visualizer
from kvector1 import (
    load_tetra_catalog,
    build_kvector_database,
    filter_catalog_by_magnitude,
    filter_catalog_by_ids
)
from geometric_voting1 import Camera, geometric_voting_star_id, load_stars_from_txt


# =========================
# PARAMETERS
# =========================

fov_deg = 75.9                      # horizontal FOV from astrometry
resolution = (2048, 1536)          # image width, height
pic_name = "2d6b3f0c-4c2b-4754-9b00-f2a139c10693.png"

catalog_max_mag = 3.5
tolerance = 0.02                   # pair lookup tolerance, radians
final_verify_tolerance = 0.03      # global check tolerance, radians

sigma_threshold = 7
min_area = 3

# k-vector DB range
min_distance = 0.01
max_distance = 1.40                # much larger than 0.5 rad for wide FOV
num_bins = 2000

# Case-study mode
use_crux_only_catalog = False

# Main bright Crux stars + one nearby possible neighbor
CRUX_HIP_IDS = [
    60718,  # Acrux
    62434,  # Mimosa
    61084,  # Gacrux
    59747,  # Delta Crucis
    60260,  # Epsilon Crucis
]


# =========================
# LOAD / FILTER CATALOG
# =========================

catalog = load_tetra_catalog("default_database.npz")
catalog = filter_catalog_by_magnitude(catalog, catalog_max_mag)

print(len(catalog))

if use_crux_only_catalog:
    catalog = filter_catalog_by_ids(catalog, CRUX_HIP_IDS)

if len(catalog) < 4:
    raise RuntimeError("Catalog too small after filtering. Relax filters.")

db_bytes = build_kvector_database(
    catalog=catalog,
    min_distance=min_distance,
    max_distance=max_distance,
    num_bins=num_bins
)

with open("kvector_fixed.db", "wb") as f:
    f.write(db_bytes)


# =========================
# CAMERA
# =========================

camera = Camera(
    x_fov=math.radians(fov_deg),
    x_resolution=int(resolution[0]),
    y_resolution=int(resolution[1])
)


# =========================
# DETECT / LOAD STARS
# =========================

visualizer(pic_name)
stars = load_stars_from_txt("centroids.txt")

if len(stars) < 4:
    raise RuntimeError("Need at least 4 detected stars for a stable Crux test.")


# =========================
# IDENTIFY
# =========================

results = geometric_voting_star_id(
    database_bytes=db_bytes,
    stars=stars,
    catalog=catalog,
    camera=camera,
    tolerance=tolerance,
    top_k_per_star=4,
    final_verify_tolerance=final_verify_tolerance
)


# =========================
# PRINT RESULTS
# =========================

print("KVector Fixed Results")
for r in results:
    matched_star = catalog[r.catalog_index]
    print(
        f"Observed star {r.star_index} -> "
        f"Catalog index {r.catalog_index}, "
        f"HIP/ID {matched_star.name}, "
        f"RA {matched_star.raj2000:.6f}, "
        f"Dec {matched_star.dej2000:.6f}, "
        f"Mag {matched_star.magnitude:.2f}, "
        f"Support {r.weight}"
    )