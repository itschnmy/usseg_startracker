import numpy as np
import cv2
from pathlib import Path

from star_detector import StarDetector
from plateSolver import PlateSolverWrapper


def load_processed_image(img_path: str):

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    # If RGB/BGR, convert to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def main():
    image_path = r"/mnt/c/Users/TD/Dropbox/My PC (DESKTOP-V5HSTKP)/Desktop/usseg_startracker/identificator/pic/adjust/ea5e33a7-93ae-4caa-986e-395ae7d625f1/2_clean.png"

    database = r"/mnt/c/Users/TD/Dropbox/My PC (DESKTOP-V5HSTKP)/Desktop/usseg_startracker/identificator/default_database"

    fov_estimate_deg = 80.0
    fov_max_error_deg = 5.0

    img = load_processed_image(image_path)
    h, w = img.shape[:2]
    print(f"[INFO] Loaded image: {image_path}")
    print(f"[INFO] Image size: {w}x{h} (W x H)")

    detector = StarDetector(
        sigma_threshold=3.0,  # tweak if too many/few detections
        min_area=2            # tweak if blobs are tiny/noisy
    )

    detected_stars = detector.process(img)

    # sort brightest first
    detected_stars.sort(key=lambda s: s.intensity, reverse=True)

    # keep only top K (helps a lot)
    K = 50
    detected_stars = detected_stars[:50]

    centroids_xy = [(float(s.position[0]), float(s.position[1])) for s in detected_stars]
    print(f"[DETECT] Stars detected: {len(detected_stars)}")

    if len(detected_stars) == 0:
        print("[FAIL] No stars detected. Try using 2_clean.png or lower sigma_threshold.")
        return

    centroids_xy = [(float(s.position[0]), float(s.position[1])) for s in detected_stars]

    # Print a few detections
    for i, (x, y) in enumerate(centroids_xy[:10]):
        print(f"  star[{i}] centroid (x,y) = ({x:.2f}, {y:.2f})")

    if len(centroids_xy) < 4:
        print("[WARN] Plate solvers usually need >= 4 stars for a reliable lost-in-space solve.")
        print("       Try an image with more stars, or relax detection threshold.")
        # We'll continue anyway, but success is unlikely.

    solver = PlateSolverWrapper(database)
    print("DB min_fov,max_fov:", solver.engine.database_properties["min_fov"], solver.engine.database_properties["max_fov"])


    # NOTE ABOUT COORDINATE ORDER:
    # Pass centroids as (x,y). The solver's solve_from_centroids() swaps internally.
    # If your PlateSolverWrapper ALSO swaps, you will break it (double swap).
    solution = solver.solve(
        detected_list_xy=centroids_xy,     # (x,y)
        image_size_hw=(h, w),
        fov_estimate_deg=fov_estimate_deg,
        fov_max_error_deg=fov_max_error_deg
    )

    # ====== 3) REPORT RESULTS ======
    print(solver.engine.database_properties["min_fov"], solver.engine.database_properties["max_fov"])
    print("verification_stars_per_fov:", solver.engine.database_properties["verification_stars_per_fov"])
    if solution.get("RA", None) is None:
        print("[SOLVE] FAILED")
        print("  T_solve(ms):", solution.get("T_solve"))
        # Sometimes solution has extra debug fields depending on your file version
        for k in ["prob", "num_patterns_tried", "pattern_count"]:
            if k in solution:
                print(f"  {k}: {solution[k]}")
        return

    print("[SOLVE] SUCCESS ✅")
    print(f"  RA(deg):   {solution['RA']:.6f}")
    print(f"  Dec(deg):  {solution['Dec']:.6f}")
    print(f"  Roll(deg): {solution['Roll']:.6f}")
    print(f"  FOV(deg):  {solution['FOV']:.6f}")
    print(f"  Matches:   {solution.get('Matches')}")
    print(f"  RMSE(arcsec): {solution.get('RMSE')}")
    print(f"  T_solve(ms):  {solution.get('T_solve')}")

    # Optional: show how many matched stars we got
    if "matched_centroids" in solution and solution["matched_centroids"] is not None:
        print(f"  Matched centroids: {len(solution['matched_centroids'])}")
    if "matched_stars" in solution and solution["matched_stars"] is not None:
        print(f"  Matched stars: {len(solution['matched_stars'])}")

if __name__ == "__main__":
    main()
