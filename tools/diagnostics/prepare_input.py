import math
import numpy as np
import json

def radec_to_cartesian(ra: float, dec: float) -> np.ndarray:
    """Converts RA/Dec (in radians) to 3D Cartesian Inertial Unit Vector"""
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    return np.array([x, y, z])

def pixel_to_body(x: float, y: float, cx=320.0, cy=240.0, f=500.0) -> np.ndarray:
    """Converts Pixel coordinates to 3D Camera Body Unit Vector
       Note: Adjust cx, cy, f to match your actual camera parameters.
    """
    x_norm = (x - cx) / f
    y_norm = (y - cy) / f
    z_norm = 1.0 # Assuming camera points along the +Z axis
    
    vec = np.array([x_norm, y_norm, z_norm])
    return vec / np.linalg.norm(vec)

def export_for_cpp(results, stars, catalog, export_path="attitude_data.json"):
    """Translates the K-Vector output and prepares it for C++"""
    body_vectors = []
    inertial_vectors = []
    
    for r in results:
        # 1. Translate 2D Pixel to 3D Body Frame
        measured_star = stars[r.star_index]
        u_body = pixel_to_body(measured_star.x, measured_star.y)
        body_vectors.append(u_body.tolist())
        
        # 2. Translate RA/Dec to 3D Inertial Frame
        matched_catalog_star = catalog[r.catalog_index]
        u_inertial = radec_to_cartesian(matched_catalog_star.raj2000, matched_catalog_star.dej2000)
        inertial_vectors.append(u_inertial.tolist())
    
    # Export to a lightweight format for C++ to read (JSON or binary struct)
    data = {
        "num_vectors": len(results),
        "body_frame": body_vectors,
        "inertial_frame": inertial_vectors
    }
    
    with open(export_path, "w") as f:
        json.dump(data, f)
    
    print(f"Exported {len(results)} vectors to {export_path}")
