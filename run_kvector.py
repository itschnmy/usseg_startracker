import math
from detector_wrapper import detector
from visualizer import visualizer
from kvector import load_tetra_catalog, build_kvector_database, filter_catalog_by_magnitude, great_circle_distance
from geometric_voting import Star, Camera, geometric_voting_star_id

# parameters
fov = 21.5 # horizontal, unit: degree
resolution = (640, 480) # pixel
pic_name = "8233ba5b-84ae-4cd7-8440-428d6e22cd64.jfif" # need loop for automation when onboard
sigma_threshold = 6 # detector's magnitude threshold
min_area = 2 # detector's threshold on min area of star dots
catalog_max_mag = 4 # magnitude threshold to filter the catalog, need to regenerate a filtered catalog before onboard


# catalog
catalog = load_tetra_catalog("default_database.npz")
catalog = filter_catalog_by_magnitude(catalog, catalog_max_mag)

db_bytes = build_kvector_database(
    catalog=catalog,
    min_distance=0.01,
    max_distance=1.4,
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
detected = detector(pic_name, sigma_threshold, min_area)
for s in detected:
    print(s.position)

# Convert to your Star class
stars = [
    Star(x=s.position[0], y=s.position[1])
    for s in detected
]

detector(pic_name, sigma_threshold, min_area)
visualizer(pic_name, detected)

# kvector
results = geometric_voting_star_id(
    database_bytes=db_bytes,
    stars=stars,
    catalog=catalog,
    camera=camera,
    tolerance=0.1,
    top_k_per_star=8,
    min_score_ratio=0.70
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

print("\nPairwise catalog angular separations:")

for i in range(len(results)):
    for j in range(i + 1, len(results)):
        s1 = catalog[results[i].catalog_index]
        s2 = catalog[results[j].catalog_index]

        ang = great_circle_distance(
            s1.raj2000, s1.dej2000,
            s2.raj2000, s2.dej2000
        )

        print(f"{i}-{j}: {math.degrees(ang):.2f} deg")