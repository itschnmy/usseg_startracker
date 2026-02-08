#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

# plateSolver.py is your local tetra3 file
from plateSolver import Tetra3


def center_crop_pil(img: Image.Image, crop_factor: float) -> Image.Image:
    """
    Center-crop an image by a linear factor (0 < crop_factor <= 1).
    crop_factor=0.32 keeps 32% of width and height (recommended for ~80° -> ~30°).
    """
    if not (0.0 < crop_factor <= 1.0):
        raise ValueError("crop_factor must be in (0, 1].")

    w, h = img.size
    new_w = int(round(w * crop_factor))
    new_h = int(round(h * crop_factor))

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    return img.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser(description="Run Tetra3 (ESA tetra3) solve_from_image() on one image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help=r"/mnt/c/Users/TD/Dropbox/My PC (DESKTOP-V5HSTKP)/Desktop/usseg_startracker/identificator/pic/8188a0b4-7be4-45ff-84ef-07c3f01b9a30.png",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="default_database",
        help=r"/mnt/c/Users/TD/Dropbox/My PC (DESKTOP-V5HSTKP)/Desktop/usseg_startracker/identificator/default_database",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=1.0,
        help="Center crop factor (e.g., 0.32 for strong crop). 1.0 = no crop.",
    )
    parser.add_argument("--fov", type=float, default=30.0, help="FOV estimate in degrees (try 25–30 for 10–30° DB).")
    parser.add_argument("--foverr", type=float, default=10.0, help="FOV max error in degrees.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Detection threshold sigma (try 2–5).")
    parser.add_argument("--filtsize", type=int, default=25, help="Background filter size (try 15–45).")
    parser.add_argument("--min_area", type=int, default=2, help="Min blob area in pixels.")
    parser.add_argument("--max_area", type=int, default=200, help="Max blob area in pixels.")
    parser.add_argument("--max_returned", type=int, default=50, help="Max stars returned by detector.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    img_path = Path(args.image)
    if not img_path.is_absolute():
        img_path = (script_dir / img_path).resolve()

    db_prefix = Path(args.db)
    if not db_prefix.is_absolute():
        db_prefix = (script_dir / db_prefix).resolve()
    db_npz = db_prefix.with_suffix(".npz")

    print("== Paths ==")
    print("Script dir:", script_dir)
    print("Image:", img_path, "exists?", img_path.exists())
    print("DB:", db_npz, "exists?", db_npz.exists())
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not db_npz.exists():
        raise FileNotFoundError(f"Database .npz not found: {db_npz} (note: pass db path WITHOUT .npz)")

    # Load image
    img = Image.open(img_path)
    print("\nOriginal size:", img.size, "mode:", img.mode)

    # Optional crop
    if args.crop < 1.0:
        img = center_crop_pil(img, args.crop)
        print("Cropped size:", img.size, f"(crop_factor={args.crop})")

    # Init solver
    t3 = Tetra3(load_database=str(db_prefix))  # prefix without .npz
    methods = [m for m in dir(t3) if any(k in m.lower() for k in ["cent", "star", "extract", "blob"])]
    print("Possible centroid/star methods:", methods)

    # Print DB range (important sanity check)
    try:
        props = t3.database_properties
        print("\n== DB properties ==")
        print("min_fov:", props.get("min_fov"), "max_fov:", props.get("max_fov"))
        print("verification_stars_per_fov:", props.get("verification_stars_per_fov"))
    except Exception as e:
        print("Could not print DB properties:", e)

    print("\n== Solve parameters ==")
    print("fov_estimate:", args.fov, "fov_max_error:", args.foverr)
    print("sigma:", args.sigma, "filtsize:", args.filtsize)
    print("min_area:", args.min_area, "max_area:", args.max_area, "max_returned:", args.max_returned)

    # Run solve_from_image (original tetra3 route)
    result = t3.solve_from_image(
        img,
        fov_estimate=args.fov,
        fov_max_error=args.foverr,
        # detection / preprocessing knobs:
        sigma=args.sigma,
        filtsize=args.filtsize,
        bg_sub_mode="local_mean",
        sigma_mode="global_root_square",
        binary_open=True,
        min_area=args.min_area,
        max_area=args.max_area,
        max_returned=args.max_returned,
        # pattern matching knobs (defaults are ok, but you can tune):
        pattern_checking_stars=12,
    )

    print("\nDetected stars (from result):", result.get("T_stars"), result.get("stars"))


    print("\n== RESULT ==")
    # Tetra3 returns a dict. Print nicely:
    for k in sorted(result.keys()):
        print(f"{k}: {result[k]}")

    # A quick success check:
    if result.get("RA") is not None and result.get("Dec") is not None:
        print("\n✅ Solve looks successful (RA/Dec present).")
    else:
        print("\n❌ Solve failed (RA/Dec missing). Try stronger crop (0.32), adjust sigma/filtsize, or regenerate DB for wider FOV.")


if __name__ == "__main__":
    main()
