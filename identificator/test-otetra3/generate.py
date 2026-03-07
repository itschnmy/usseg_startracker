from pathlib import Path
from plateSolver import Tetra3

print("Imported plateSolver from:", Path(__file__).parent / "plateSolver.py")

t3 = Tetra3(load_database=None)

t3.generate_database(
    max_fov=75,
    min_fov=60,
    save_as="db_70-1deg",

    star_catalog="hip_main",
    star_max_magnitude=7,

    pattern_stars_per_fov=30,
    verification_stars_per_fov=120,

    pattern_max_error=0.005
)

print("Finished building db_70-1deg.npz")