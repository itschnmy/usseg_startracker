"""add --method kvector 'or' plate into the run command"""

import argparse
import serial
import time
import math
import numpy as np
from PIL import Image

from detector_wrapper import detector
from visualizer import visualizer

from kvector import (
    load_tetra_catalog,
    build_kvector_database,
    filter_catalog_by_magnitude,
    great_circle_distance
)

from geometric_voting import (
    Star,
    Camera,
    geometric_voting_star_id
)

from vector_generator import (
    CameraModel,
    generate_vectors
)

from plateSolver import Tetra3

# parameter set-ups
image = "openmv_capture.jpg"
marked_img = "marked.png"

fov = 21.5 #deg, horizontal
resolution = (640, 480)

sigma_threshold = 6
min_area = 2
max_detected_stars = 8 # need to consider

catalog_mag_max = 4

kvector_min_distance = 0.01
kvector_max_distance = 1.4
kvector_num_bins = 1000
kvector_tolerance = 0.1
kvector_top_k_per_star = 8
kvector_min_score_ratio = 0.70

plate_database = "default_database"
plate_fov_max_error = 5.0

# open mv set-up

def read_line(ser):
    return ser.readline().decode(errors="ignore").strip()


def capture_image_from_openmv(port, baudrate):
    ser = serial.Serial(port, baudrate, timeout=10)
    time.sleep(2)

    print("Sending CAPTURE command to OpenMV...")
    ser.write(b"CAPTURE\n")

    line = read_line(ser)

    if line != "START":
        ser.close()
        raise RuntimeError("Expected START from OpenMV, got: " + line)

    size_line = read_line(ser)
    image_size = int(size_line)

    print("Receiving image:", image_size, "bytes")

    image_bytes = ser.read(image_size)

    with open(image, "wb") as f:
        f.write(image_bytes)

    ser.close()

    print("Saved image:", image)


def detect_stars_from_image(image_name):
    detected = detector(
        image_name,
        sigma_threshold=sigma_threshold,
        min_area=min_area
    )

    detected = detected[:max_detected_stars]

    print("\nDetected centroids:")
    for i, s in enumerate(detected):
        print(f"{i}: x={s.position[0]:.2f}, y={s.position[1]:.2f}")

    visualizer(image_name, detected, marked_img)

    return detected


# k_vector
def run_kvector_identification(detected):
    print("\nRunning K-vector star identification...")

    catalog = load_tetra_catalog("default_database.npz")
    catalog = filter_catalog_by_magnitude(catalog, catalog_mag_max)

    db_bytes = build_kvector_database(
        catalog=catalog,
        min_distance=kvector_min_distance,
        max_distance=kvector_max_distance,
        num_bins=kvector_num_bins
    )

    camera = Camera(
        x_fov=math.radians(fov),
        x_resolution=resolution[0],
        y_resolution=resolution[1]
    )

    stars = [
        Star(x=s.position[0], y=s.position[1])
        for s in detected
    ]

    results = geometric_voting_star_id(
        database_bytes=db_bytes,
        stars=stars,
        catalog=catalog,
        camera=camera,
        tolerance=kvector_tolerance,
        top_k_per_star=kvector_top_k_per_star,
        min_score_ratio=kvector_min_score_ratio
    )

    centroid_list = []
    identified_list = []

    print("\nK-vector Results:")

    if len(results) == 0:
        print("No reliable K-vector result found.")
        return centroid_list, identified_list

    for r in results:
        matched_star = catalog[r.catalog_index]

        star_id = int(matched_star.name)

        observed = detected[r.star_index]
        x_pixel = float(observed.position[0])
        y_pixel = float(observed.position[1])

        ra_deg = math.degrees(matched_star.raj2000)
        dec_deg = math.degrees(matched_star.dej2000)

        centroid_list.append({
            "id": star_id,
            "x": x_pixel,
            "y": y_pixel
        })

        identified_list.append({
            "id": star_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg
        })

        print(
            f"Observed star {r.star_index} -> "
            f"ID {star_id}, "
            f"RA {ra_deg:.6f} deg, "
            f"Dec {dec_deg:.6f} deg, "
            f"Mag {matched_star.magnitude:.2f}"
        )

    print("\nPairwise catalog angular separations:")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            s1 = catalog[results[i].catalog_index]
            s2 = catalog[results[j].catalog_index]

            ang = great_circle_distance(
                s1.raj2000,
                s1.dej2000,
                s2.raj2000,
                s2.dej2000
            )

            print(f"{i}-{j}: {math.degrees(ang):.2f} deg")

    return centroid_list, identified_list


#plate solver

def run_plate_solver_identification(detected):
    print("\nRunning plate solver star identification...")

    image = Image.open(IMAGE_NAME)

    t3 = Tetra3(load_database=PLATE_DATABASE)

    # note: [y, x] format
    centroids_yx = np.array([
        [float(s.position[1]), float(s.position[0])]
        for s in detected
    ])

    solution = t3.solve_from_centroids(
        star_centroids=centroids_yx,
        size=(resolution[1], resolution[0]),
        fov_estimate=fov,
        fov_max_error=plate_fov_max_error,
        return_matches=True
    )

    print("\nPlate Solver Result:")
    print(solution)

    centroid_list = []
    identified_list = []

    if solution["RA"] is None:
        print("No reliable plate-solver result found.")
        return centroid_list, identified_list

    matched_centroids = solution["matched_centroids"]
    matched_stars = solution["matched_stars"]
    matched_ids = solution["matched_catID"]

    print("\nMatched stars:")

    for i in range(len(matched_stars)):
        # tetra3 matched centroid format is [y, x]
        y_pixel = float(matched_centroids[i][0])
        x_pixel = float(matched_centroids[i][1])

        ra_deg = float(matched_stars[i][0])
        dec_deg = float(matched_stars[i][1])
        mag = float(matched_stars[i][2])

        if matched_ids is None:
            star_id = i
        else:
            star_id = int(matched_ids[i])

        centroid_list.append({
            "id": star_id,
            "x": x_pixel,
            "y": y_pixel
        })

        identified_list.append({
            "id": star_id,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg
        })

        print(
            f"Matched star {i} -> "
            f"ID {star_id}, "
            f"RA {ra_deg:.6f} deg, "
            f"Dec {dec_deg:.6f} deg, "
            f"Mag {mag:.2f}"
        )

    print("\nCamera pointing from plate solver:")
    print(f"Image center RA  = {solution['RA']:.6f} deg")
    print(f"Image center Dec = {solution['Dec']:.6f} deg")
    print(f"Roll             = {solution['Roll']:.6f} deg")
    print(f"Solved FOV        = {solution['FOV']:.6f} deg")

    return centroid_list, identified_list


"""#vector generator

def run_vector_generator(centroid_list, identified_list):
    print("\nGenerating camera-frame and inertial-frame vectors...")

    camera_model = CameraModel(
        image_width=RESOLUTION[0],
        image_height=RESOLUTION[1],
        fov_x_deg=FOV_DEG,
        fov_y_deg=FOV_DEG * RESOLUTION[1] / RESOLUTION[0]
    )

    vector_pairs = generate_vectors(
        centroid_list=centroid_list,
        identified_list=identified_list,
        camera=camera_model
    )

    if len(vector_pairs) == 0:
        print("No vector pairs generated.")
        return []

    print("\nVector Generator Results:")

    for pair in vector_pairs:
        print("\nStar ID:", pair["id"])
        print("Camera/body vector:   ", pair["camera_vector"])
        print("Inertial/ECI vector:  ", pair["inertial_vector"])

    return vector_pairs
"""



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["kvector", "plate"],
        required=True,
        help="Choose method: kvector or plate"
    )

    parser.add_argument(
        "--port",
        default="COM4",
        help="OpenMV serial port, e.g. COM4 or /dev/ttyACM0"
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200
    )

    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Use existing openmv_capture.jpg instead of taking new OpenMV photo"
    )

    args = parser.parse_args()

    if not args.skip_capture:
        capture_image_from_openmv(args.port, args.baudrate)
    else:
        print("Skipping OpenMV capture. Using existing:", image)

    detected = detect_stars_from_image(image)

    if args.method == "kvector":
        centroid_list, identified_list = run_kvector_identification(detected)

    elif args.method == "plate":
        centroid_list, identified_list = run_plate_solver_identification(detected)

    else:
        raise ValueError("Unknown method.")

    """ vector_pairs = run_vector_generator(
        centroid_list=centroid_list,
        identified_list=identified_list
    )"""

    print("\nPipeline complete.")
    """print("Number of vector pairs:", len(vector_pairs))"""


if __name__ == "__main__":
    main()