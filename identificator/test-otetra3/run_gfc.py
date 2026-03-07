from plateSolver import Tetra3
import numpy as np

# Load centroid file
centroids = np.loadtxt("centroids.txt", dtype=np.float32)

# Load database
t3 = Tetra3(load_database="db_70-1deg.npz")

# Size
size = (2000, 2000)

# Solve
solution = t3.solve_from_centroids(
    star_centroids=centroids,
    size=size,
    fov_estimate=70,
    fov_max_error=5
)

print("Solution:")
print(solution)