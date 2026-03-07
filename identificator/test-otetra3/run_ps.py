from PIL import Image
from plateSolver import Tetra3
import pprint

# Path to your star image
image_path = "23.bmp"

# Load image
image = Image.open(image_path)

# Create solver object
solver = Tetra3(load_database="db_70-1deg")

# Run plate solving
result = solver.solve_from_image(
    image,
    fov_estimate=70,        # estimated field of view
    fov_max_error=5,        # allowed error
    pattern_checking_stars=8,
    match_radius=0.01
)

# Print results nicely
pprint.pprint(result)