from plateSolver import Tetra3, _compute_vectors
from detector import StarDetector
from visualize_centroids import visualizer
from attitude_control_system import AttitudeControlSystem

import cv2
import sys
import numpy as np

"""
Input Script (powershell): py main.py [img name] [sigma_threshold] [min_area] [hfov]
Eg: python3 main.py img.png 6 7 18
"""

def radec_to_unit_vector(ra_deg, dec_deg):
    """
    Convert RA/Dec in degrees into inertial-frame unit vector.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    return np.array([
        np.cos(ra) * np.cos(dec),
        np.sin(ra) * np.cos(dec),
        np.sin(dec)
    ], dtype=float)


def main():
    if len(sys.argv) != 5:
        print("At least a parameter missed!!!")
        print("Usage: python3 main.py [img] [sigma_threshold] [min_area] [hfov]")
        sys.exit()

    img = str(sys.argv[1])
    sigma = float(sys.argv[2])
    min_area = float(sys.argv[3])
    hfov = float(sys.argv[4])

    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]

    if image is None:
        print("Could not read image:", img)
        sys.exit()

    detector = StarDetector(sigma, min_area)
    stars = detector.process(image)

    # brightest stars first
    stars = sorted(stars, key=lambda s: s.intensity, reverse=True)

    # [x, y]
    centroids_xy = np.array([star.position for star in stars])

    print("Number of stars:", len(centroids_xy))

    with open("centroids.txt", "w") as f:
        for star in stars:
            x, y = star.position
            f.write(f"{x} {y}\n")

    visualizer(image)

    size = (height, width)

    # [y, x]
    centroids_yx = centroids_xy[:, ::-1]

    t3 = Tetra3(load_database="default_database")

    result = t3.solve_from_centroids(
        star_centroids=centroids_yx,
        size=size,
        fov_estimate=hfov,
        fov_max_error=10,
        pattern_checking_stars=21,
        match_radius=0.03,
        match_threshold=1e-2,
        return_matches=True
    )

    print(t3.database_properties)
    print(t3.has_database)
    print("centroid shape:", centroids_yx.shape)
    print("first 10:", centroids_yx[:10])

    print("\n--- Tetra3 result ---")
    for key, value in result.items():
        print(f"{key}: {value}")

    # stop if Tetra3 failed
    if result["RA"] is None:
        print("\nTetra3 failed, so attitude cannot be estimated.")
        return

    # get matched image centroids and matched catalog stars
    matched_centroids_yx = np.array(result["matched_centroids"], dtype=float)

    # matched_stars format: [RA_deg, Dec_deg, magnitude]
    matched_stars = np.array(result["matched_stars"], dtype=float)

    # Convert matched image centroids into body-frame unit vectors.
    fov_rad = np.deg2rad(result["FOV"])

    body_vectors = _compute_vectors(
        matched_centroids_yx,
        size,
        fov_rad
    ).T

    # Convert matched catalogue RA/Dec into inertial-frame unit vectors
    inertial_vectors = np.array([
        radec_to_unit_vector(ra, dec)
        for ra, dec, mag in matched_stars
    ], dtype=float).T

    print("\n--- Vector pair data ---")
    print("body_vectors shape:", body_vectors.shape)
    print("inertial_vectors shape:", inertial_vectors.shape)

    # attitude estimation
    adcs = AttitudeControlSystem(use_davenport=False)

    q_estimated = adcs.process_sensor_data(
        body_vectors,
        inertial_vectors
    )

    print("\n--- Attitude estimation result ---")
    print("Estimated quaternion [w, x, y, z]:")
    print(q_estimated)

   #compare with Davenport
    adcs_davenport = AttitudeControlSystem(use_davenport=True)

    q_davenport = adcs_davenport.process_sensor_data(
        body_vectors,
        inertial_vectors
    )

    print("\n--- Davenport comparison result ---")
    print("Estimated quaternion using Davenport [w, x, y, z]:")
    print(q_davenport)


if __name__ == "__main__":
    main()