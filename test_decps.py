import numpy as np
from plateSolver import Tetra3

def load_centroids(txt_file):
    rows = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            x = float(parts[0])
            y = float(parts[1])
            rows.append([y, x])  # convert to the plate solver's format
    return np.array(rows, dtype=np.float32)

centroids = load_centroids("centroids.txt")
size = (480, 640) # *** ADJUST HERE *** (height, width)

t3 = Tetra3(load_database="default_database")
result = t3.solve_from_centroids(
    star_centroids=centroids,
    size=size,
    fov_estimate=70,
    fov_max_error=10,
    pattern_checking_stars=21,
    match_radius=0.03,
    match_threshold=1e-2
)


print(t3.database_properties)
print(t3.has_database)
print("centroid shape:", centroids.shape)
print("first 10:", centroids[:10])

for key, value in result.items():
    print(f"{key}: {value}")