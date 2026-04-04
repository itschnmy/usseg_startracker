import math
from det_run import detector
from det_visualize import visualizer
from kvector import load_tetra_catalog, build_kvector_database, filter_catalog_by_magnitude
from geometric_voting import Star, Camera, geometric_voting_star_id, load_stars_from_txt


# parameters
fov = 50 # horizontal, unit: degree
resolution = (2048, 1536) # pixel
pic_name = "2b670e06-6068-480e-8e30-ad5a10f0e4e5.png" # need loop for automation when onboard
sigma_threshold = 3 # detector's magnitude threshold
min_area = 3 # detector's threshold on min area of star dots
catalog_max_mag = .0 # magnitude threshold to filter the catalog, regenerate a filtered catalog before onboard


# catalog
catalog = load_tetra_catalog("default_database.npz")
catalog = filter_catalog_by_magnitude(catalog, catalog_max_mag)

db_bytes = build_kvector_database(
    catalog=catalog,
    min_distance=0.01,
    max_distance=0.5,
    num_bins=1000
)

with open("kvector.db", "wb") as f:
    f.write(db_bytes)


# camera
camera = Camera( # ADJUST
    x_fov=math.radians(fov),
    x_resolution = int(resolution[0]),
    y_resolution = int(resolution[1])
)


# detecting and visualizing
detector(pic_name, sigma_threshold, min_area)
visualizer(pic_name)
stars = load_stars_from_txt("centroids.txt")

# kvector
results = geometric_voting_star_id(
    database_bytes=db_bytes,
    stars=stars,
    catalog=catalog,
    camera=camera,
    tolerance=0.01
)


# print results
print("Kvector Results")
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