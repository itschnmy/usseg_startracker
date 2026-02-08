from pathlib import Path
import numpy as np
import cv2

from plateSolver import Tetra3
from star_detector import StarDetector


ROOT = Path(__file__).resolve().parent


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def center_crop(img: np.ndarray, crop_factor: float) -> np.ndarray:
    if not (0.0 < crop_factor <= 1.0):
        raise ValueError("crop_factor must be in (0, 1].")
    h, w = img.shape[:2]
    new_w = int(round(w * crop_factor))
    new_h = int(round(h * crop_factor))
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w]


def main():
    # --- paths ---
    image_path = ROOT / "pic" / "adjust" / "8188a0b4-7be4-45ff-84ef-07c3f01b9a30" / "2_clean.png"      # change to your image
    db_prefix  = ROOT / "default_database"         # WITHOUT .npz

    print("Image:", image_path, "exists?", image_path.exists())
    print("DB npz:", db_prefix.with_suffix(".npz"), "exists?", db_prefix.with_suffix(".npz").exists())

    img = load_gray(image_path)
    print("Original size (h,w):", img.shape[:2])

    # Strong crop for iPhone main cam -> try 0.25 if still failing
    crop_factor = 0.32
    img = center_crop(img, crop_factor)
    h, w = img.shape[:2]
    print("Cropped size (h,w):", (h, w), "crop_factor:", crop_factor)

    # --- detect stars using YOUR detector ---
    detector = StarDetector()
    stars = detector.process(img)

    # Sort by brightness (very important)
    stars.sort(key=lambda s: float(s.intensity), reverse=True)

    # Keep only top K (avoid combinatorial explosion)
    K = 30
    stars_used = stars[:K]

    centroids = np.array(
        [[float(s.position[0]), float(s.position[1])] for s in stars_used],
        dtype=np.float64
    )

    print("\n== Star detection debug ==")
    print("Detected stars total:", len(stars))
    print("Using top K:", len(centroids))
    if len(centroids) > 0:
        print("x range:", float(centroids[:, 0].min()), float(centroids[:, 0].max()))
        print("y range:", float(centroids[:, 1].min()), float(centroids[:, 1].max()))
        print("Top 5 intensities:", [float(s.intensity) for s in stars_used[:5]])

    # --- solve using Tetra3 ---
    t3 = Tetra3(load_database=str(db_prefix))
    props = getattr(t3, "database_properties", None)
    if isinstance(props, dict):
        print("\nDB FOV range:", props.get("min_fov"), props.get("max_fov"))

    # IMPORTANT: Tetra3 expects size as (height, width)
    # Try a small sweep because effective FOV after crop is still uncertain.
    for fov in (20.0, 25.0, 30.0):
        print(f"\n== Solving with fov_estimate={fov} ==")
        result = t3.solve_from_centroids(
            centroids,
            (h, w),
            fov_estimate=fov,
            fov_max_error=10.0,
        )

        # result is usually a dict
        if isinstance(result, dict):
            print("RA:", result.get("RA"), "Dec:", result.get("Dec"),
                  "Roll:", result.get("Roll"), "FOV:", result.get("FOV"),
                  "Matches:", result.get("Matches"), "Prob:", result.get("Prob"),
                  "RMSE:", result.get("RMSE"))
        else:
            print("Returned:", result)

    print("\nDone.")


if __name__ == "__main__":
    main()