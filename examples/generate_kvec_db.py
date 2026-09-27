import os
import sys
import numpy as np

# Add the subdirectory to python path so we can import kvector modules
sys.path.append(os.path.join(os.getcwd(), "data"))

from kvector import load_tetra_catalog, filter_catalog_by_magnitude, build_kvector_database
from export_catalog import radec_to_unit_vector

def main():
    npz_path = "data/default_database.npz"
    max_mag = 4.0
    
    print(f"Loading catalog from {npz_path}...")
    catalog = load_tetra_catalog(npz_path)
    filtered_catalog = filter_catalog_by_magnitude(catalog, max_mag)
    print(f"Found {len(filtered_catalog)} stars with mag <= {max_mag}")
    
    # 1. Export star_catalog.csv
    csv_path = "star_catalog.csv"
    with open(csv_path, "w") as f:
        f.write("id,ux,uy,uz,magnitude\n")
        for star in filtered_catalog:
            ux, uy, uz = radec_to_unit_vector(star.raj2000, star.dej2000)
            f.write(f"{star.name},{ux:.10f},{uy:.10f},{uz:.10f},{star.magnitude:.6f}\n")
    print(f"Saved consistent {csv_path}")
    
    # 2. Build kvector database
    min_distance = 0.01
    max_distance = 1.40
    num_bins = 2000
    
    print("Building k-vector database...")
    db_bytes = build_kvector_database(
        catalog=filtered_catalog,
        min_distance=min_distance,
        max_distance=max_distance,
        num_bins=num_bins
    )
    
    # Write to root
    db_paths = [
        "kvector_fixed.db"
    ]
    for path in db_paths:
        with open(path, "wb") as f:
            f.write(db_bytes)
        print(f"Saved consistent K-Vector DB to {path}")

if __name__ == "__main__":
    main()
