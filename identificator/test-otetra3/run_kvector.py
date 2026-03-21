import math
from kvector import load_tetra_catalog, build_kvector_database, filter_catalog_by_magnitude
from geometric_voting import Star, Camera, geometric_voting_star_id, load_stars_from_txt


# ---------------------------------
# STEP 1: Load Tetra star catalog
# ---------------------------------
catalog = load_tetra_catalog("default_database.npz")
catalog = filter_catalog_by_magnitude(catalog, max_mag=4.0)


# ---------------------------------
# STEP 2: Build k-vector database
# ---------------------------------
db_bytes = build_kvector_database(
    catalog=catalog,
    min_distance=0.01,
    max_distance=0.5,
    num_bins=1000
)

with open("kvector.db", "wb") as f:
    f.write(db_bytes)


# ---------------------------------
# STEP 3: Example detected stars
# ---------------------------------
stars = load_stars_from_txt("centroids.txt")


# ---------------------------------
# STEP 4: Camera
# ---------------------------------
camera = Camera( # ADJUST
    x_fov=math.radians(58.5),
    x_resolution=320,
    y_resolution=240
)


# ---------------------------------
# STEP 5: Run identification
# ---------------------------------
results = geometric_voting_star_id(
    database_bytes=db_bytes,
    stars=stars,
    catalog=catalog,
    camera=camera,
    tolerance=0.01
)


# ---------------------------------
# STEP 6: Print results
# ---------------------------------
print("\n=== Star Identification Results ===")
for r in results:
    matched_star = catalog[r.catalog_index]
    print(
        f"Observed star {r.star_index} -> "
        f"Catalog index {r.catalog_index}, "
        f"HIP/ID {matched_star.name}, "
        f"RA {matched_star.raj2000:.6f}, "
        f"Dec {matched_star.dej2000:.6f}, "
        f"Mag {matched_star.magnitude:.2f}"
    )