from PIL import Image
import plateSolver

# -------- CONFIG --------
IMAGE_PATH = "pic/54_debug_visual.png"
DB_NAME = "db_70deg"   # WITHOUT .npz
FOV_ESTIMATE = 70      # degrees
# ------------------------

img = Image.open(IMAGE_PATH)

# Load database
t3 = plateSolver.Tetra3(load_database=DB_NAME)

# Centroid extraction test
centroids = plateSolver.get_centroids_from_image(img)
print("Detected centroids:", len(centroids))

# Solve
result = t3.solve_from_image(img, fov_estimate=FOV_ESTIMATE)

print("=== TETRA3 RESULT ===")
print(result)
