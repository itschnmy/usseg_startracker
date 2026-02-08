def main():
    import numpy as np
    import logging
    from plateSolver import PlateSolverWrapper

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    DATABASE_PATH = "default_database.npz"
    IMAGE_SIZE_HW = (1080, 1920)
    FOV_ESTIMATE_DEG = 30.0
    FOV_MAX_ERROR_DEG = 5.0

    detected_list_xy = np.array([
        [960.0, 540.0],
        [1020.5, 600.2],
        [880.3, 490.1],
        [1100.8, 520.6],
        [930.4, 610.9],
        [1005.1, 455.3],
        [870.2, 575.8],
        [1120.7, 495.4],
    ], dtype=float)

    solver = PlateSolverWrapper(DATABASE_PATH)

    solution = solver.solve(
        detected_list_xy=detected_list_xy,
        image_size_hw=IMAGE_SIZE_HW,
        fov_estimate_deg=FOV_ESTIMATE_DEG,
        fov_max_error_deg=FOV_MAX_ERROR_DEG,
    )

    if solution is None:
        print("No solution returned")
        return

    # solver returns dict with keys even on failure
    if solution.get("RA") is None:
        print("Solve failed (no RA/Dec). Full result:")
        print(solution)
        return

    print(f"RA        : {solution['RA']:.6f} deg")
    print(f"Dec       : {solution['Dec']:.6f} deg")
    print(f"Roll      : {solution['Roll']:.6f} deg")
    print(f"FOV       : {solution['FOV']:.3f} deg")
    print(f"RMSE      : {solution.get('RMSE', 'N/A')} arcsec")
    print(f"Matches   : {solution.get('Matches', 'N/A')}")
    print(f"Prob      : {solution.get('Prob', 'N/A')}")

    if "matched_centroids" in solution:
        print("\nMatched stars:")
        print(f"  matched_centroids : {solution['matched_centroids'].shape}")
        print(f"  matched_stars     : {solution['matched_stars'].shape}")

if __name__ == "__main__":
    main()
