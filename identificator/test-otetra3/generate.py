from pathlib import Path
from plateSolver import Tetra3

print("Imported plateSolver from:", Path(__file__).parent / "plateSolver.py")

t3 = Tetra3(load_database=None)

t3.generate_database(
    max_fov=70,
    min_fov=None,
    save_as=Path("db_70deg"),
    star_catalog="hip_main",
    pattern_stars_per_fov=10,
    verification_stars_per_fov=30,
    star_max_magnitude=7,
    pattern_max_error=0.005,
    simplify_pattern=False,
    presort_patterns=True,
    save_largest_edge=True,
    epoch_proper_motion="now"
)

print("Finished building db_70deg.npz")