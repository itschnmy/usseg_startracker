"""Blind image runs followed by separate Stellarium-label diagnostics.

The supplied FOV is vertical, following tools/stellarium/extract_star.py. Label positions,
sky center, alignment and catalog coordinates are read only after inference.
No detector/solver settings are tuned using these labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from models.detector.star_detector import StarDetector
from models.attitude import AttitudeDeterminator
from models.identifier.plate_solver import _compute_vectors
from models.pipeline import PipelineConfig, StarTrackerPipeline


def unit_vectors(radec):
    ra, dec = np.deg2rad(np.asarray(radec)).T
    return np.column_stack((np.cos(ra) * np.cos(dec),
                            np.sin(ra) * np.cos(dec), np.sin(dec)))


def reference(labels):
    """Recreate extract_star.py's tangent-plane + alignment convention."""
    ra, dec = np.deg2rad([labels["center_ra_deg"], labels["center_dec_deg"]])
    center = unit_vectors([[labels["center_ra_deg"], labels["center_dec_deg"]]])[0]
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    matrix = np.asarray(labels["alignment_matrix"], dtype=float)
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    theta = np.arctan2(matrix[1, 0], matrix[0, 0])
    rotation = np.stack((center, np.cos(theta)*east - np.sin(theta)*north,
                         np.sin(theta)*east + np.cos(theta)*north))
    return Rotation.from_matrix(rotation), float(scale), matrix


def score(result, labels, reference_rotation):
    stars = labels["stars"]
    label_xy = np.array([[s["x"], s["y"]] for s in stars])
    selected = np.asarray(result["centroids_xy"]).reshape(-1, 2)
    distances = cdist(selected, label_xy)
    rows, cols = linear_sum_assignment(distances)
    good = distances[rows, cols] <= labels.get("match_radius_px", 3.0)
    residuals = distances[rows[good], cols[good]]
    metrics = {
        "label_count": len(stars), "selected_centroids": len(selected),
        "selected_label_matches_3px": int(good.sum()),
        "label_recall_selected_pct": float(100 * good.sum()/len(stars)),
        "selected_without_label": int(len(selected)-good.sum()),
        "centroid_residual_p50_px": float(np.median(residuals)) if len(residuals) else None,
        "centroid_residual_p95_px": float(np.percentile(residuals, 95)) if len(residuals) else None,
        "attitude_error_deg": None, "boresight_error_deg": None,
        "pairwise_star_id_correct": 0, "pairwise_star_id_labelled": 0,
    }
    if result["status"] != "ok":
        return metrics
    q = np.asarray(result["quaternion_inertial_to_body_wxyz"])
    actual = Rotation.from_quat(q[[1, 2, 3, 0]])
    metrics["attitude_error_deg"] = float(np.rad2deg((actual * reference_rotation.inv()).magnitude()))
    center_actual = actual.inv().apply([1, 0, 0])
    center_ref = reference_rotation.inv().apply([1, 0, 0])
    metrics["boresight_error_deg"] = float(np.rad2deg(np.arccos(np.clip(center_actual @ center_ref, -1, 1))))
    matched_xy = np.asarray(result["matched_centroids_yx"])[:, ::-1]
    matched_sky = unit_vectors(result["matched_stars_radec"])
    labelled_sky = unit_vectors([[s["ra_deg"], s["dec_deg"]] for s in stars])
    cost = cdist(matched_xy, label_xy)
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for row, col in zip(rows, cols):
        if cost[row, col] > 3:
            continue
        angle = float(np.rad2deg(np.arccos(np.clip(matched_sky[row] @ labelled_sky[col], -1, 1))))
        pairs.append({"hip_label": stars[col]["hip_id"], "pixel_error": float(cost[row, col]),
                      "catalog_separation_deg": angle, "correct_within_0_05deg": angle < .05})
    metrics["pairwise_star_id_labelled"] = len(pairs)
    metrics["pairwise_star_id_correct"] = sum(p["correct_within_0_05deg"] for p in pairs)
    metrics["star_id_pairs"] = pairs
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--vertical-fov", type=float, default=17)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional configured blind run, in addition to baseline diagnostics")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image_path = args.dataset / "stellarium_view.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = image.shape
    hfov = float(np.rad2deg(2*np.arctan(np.tan(np.deg2rad(args.vertical_fov)/2)*w/h)))
    config = PipelineConfig(database=args.database.resolve(), fov_deg=hfov)
    start = perf_counter()
    pipeline = StarTrackerPipeline(config)
    load_ms = (perf_counter()-start)*1000
    runs = []
    # Only pixels, dimensions and the user-supplied FOV enter these runs.
    for name, fov, timeout in (("supplied_vertical_as_horizontal", args.vertical_fov, 600),
                               ("vertical_to_horizontal", hfov, 600),
                               ("vertical_to_horizontal_5s", hfov, 5000)):
        pipeline.config = replace(config, fov_deg=fov, solve_timeout_ms=timeout)
        result = pipeline.run_array(image, source=str(image_path.resolve()))
        runs.append({"name": name, "config": {**asdict(pipeline.config), "database": str(config.database)},
                     "result": result})
        print(name, result["status"], result["num_matches"], flush=True)

    if args.config:
        pipeline.config = PipelineConfig.from_json(args.config)
        if pipeline.config.database.resolve() != config.database.resolve():
            pipeline = StarTrackerPipeline(pipeline.config)
        result = pipeline.run_array(image, source=str(image_path.resolve()))
        runs.append({"name": "configured_blind", "config": {
            **asdict(pipeline.config), "database": str(pipeline.config.database)}, "result": result})
        print("configured_blind", result["status"], result["num_matches"], flush=True)

    # Scoring starts here, after ALL blind inference calls have completed.
    labels = json.loads((args.dataset / "stars_labels.json").read_text())
    assert [labels["image_width"], labels["image_height"]] == [w, h]
    assert labels["total_stars"] == len(labels["stars"])
    ref, scale, alignment = reference(labels)
    sky = unit_vectors([[s["ra_deg"], s["dec_deg"]] for s in labels["stars"]])
    body = ref.apply(sky)
    focal = h/2/np.tan(np.deg2rad(labels["fov_deg"])/2)*scale
    projected = alignment[:, 2] - focal * body[:, 1:]/body[:, :1]
    label_xy = np.array([[s["x"], s["y"]] for s in labels["stars"]])
    reference_residual = np.linalg.norm(projected-label_xy, axis=1)
    for run in runs:
        run["metrics"] = score(run["result"], labels, ref)

    detected = sorted(StarDetector().process(image), key=lambda s: (s.peak, s.intensity), reverse=True)
    all_distance = cdist(np.asarray([s.position for s in detected]), label_xy)
    dr, dc = linear_sum_assignment(all_distance)
    count_all = int(np.sum(all_distance[dr, dc] <= 3))
    # Label-assisted solver-only diagnostic; never treated as a blind solve.
    label_hfov = float(np.rad2deg(2*np.arctan(np.tan(np.deg2rad(labels["fov_deg"])/2)*w/h)))
    brightest = sorted(labels["stars"], key=lambda star: star["vmag"])[:config.max_stars]
    oracle_start = perf_counter()
    oracle_raw = pipeline.solver.solve_from_centroids(
        np.array([[s["x"], s["y"]] for s in brightest]), (h, w),
        fov_estimate=label_hfov, fov_max_error=config.fov_max_error_deg,
        pattern_checking_stars=config.pattern_checking_stars,
        match_radius=config.match_radius, match_threshold=config.match_threshold,
        solve_timeout=5000, return_matches=True, return_visual=False,
    )
    oracle = {"track": "label_assisted_centroids_NOT_end_to_end",
              "horizontal_fov_deg": label_hfov,
              "status": "ok" if oracle_raw.get("RA") is not None else "no_solve",
              "elapsed_ms": (perf_counter()-oracle_start)*1000,
              "num_centroids": len(brightest),
              "num_matches": int(oracle_raw.get("Matches") or 0),
              "boresight_error_deg": None}
    if oracle["status"] == "ok":
        for output, key in (("ra_deg", "RA"), ("dec_deg", "Dec"),
                            ("solved_fov_deg", "FOV"), ("rmse_arcsec", "RMSE")):
            oracle[output] = float(oracle_raw[key])
        solved_center = unit_vectors([[oracle_raw["RA"], oracle_raw["Dec"]]])[0]
        true_center = unit_vectors([[labels["center_ra_deg"], labels["center_dec_deg"]]])[0]
        oracle["boresight_error_deg"] = float(np.rad2deg(np.arccos(np.clip(solved_center @ true_center,-1,1))))
        camera_vectors = _compute_vectors(oracle_raw["matched_centroids"], (h, w),
                                          np.deg2rad(oracle_raw["FOV"]))
        catalog_vectors = unit_vectors(np.asarray(oracle_raw["matched_stars"])[:, :2])
        # Same default SVD and public direction as StarTrackerPipeline.
        q = np.asarray(AttitudeDeterminator("SVD").estimate(camera_vectors.T, catalog_vectors.T))
        q = q / np.linalg.norm(q) * np.array([1, -1, -1, -1])
        oracle["quaternion_inertial_to_body_wxyz"] = q.tolist()
        oracle["metrics"] = score({
            "status": "ok", "centroids_xy": [[s["x"], s["y"]] for s in brightest],
            "quaternion_inertial_to_body_wxyz": q.tolist(),
            "matched_centroids_yx": oracle_raw["matched_centroids"],
            "matched_stars_radec": np.asarray(oracle_raw["matched_stars"])[:, :2].tolist(),
        }, labels, ref)
    filtered = cv2.morphologyEx(image, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mean, std = cv2.meanStdDev(filtered)
    threshold = float(np.clip(mean[0, 0]+config.sigma_threshold*std[0, 0], 2, 255))
    bright_diagnostics = []
    for star in brightest[:10]:
        x, y = round(star["x"]), round(star["y"])
        raw_max = int(image[max(0,y-3):y+4,max(0,x-3):x+4].max())
        filtered_max = int(filtered[max(0,y-3):y+4,max(0,x-3):x+4].max())
        bright_diagnostics.append({"hip_id": star["hip_id"], "vmag": star["vmag"],
                                   "raw_core_max": raw_max, "tophat_core_max": filtered_max,
                                   "core_crosses_threshold": filtered_max > threshold})
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for rank, star in enumerate(detected[:config.max_stars], 1):
        xy = tuple(np.rint(star.position).astype(int))
        cv2.circle(overlay, xy, 12, (0, 255, 0), 1)
        cv2.putText(overlay, str(rank), (xy[0]+13, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir / "selected-centroids.png"), overlay):
        raise OSError("Failed to save overlay")
    if args.config:
        configured = next(run for run in runs if run["name"] == "configured_blind")
        configured_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for rank, xy in enumerate(configured["result"]["centroids_xy"], 1):
            x, y = np.rint(xy).astype(int)
            cv2.circle(configured_overlay, (x, y), 12, (0, 255, 0), 1)
            cv2.putText(configured_overlay, str(rank), (x+13, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 255, 255), 1)
        if not cv2.imwrite(str(args.output_dir / "configured-selected-centroids.png"), configured_overlay):
            raise OSError("Failed to save configured overlay")
    report = {
        "python": platform.python_version(), "opencv_threads": cv2.getNumThreads(),
        "image": str(image_path.resolve()), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "database_sha256": hashlib.sha256(args.database.read_bytes()).hexdigest(),
        "database_load_ms": load_ms, "input_vertical_fov_deg": args.vertical_fov,
        "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])):
                          hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((Path(__file__).resolve().parents[1]/"models").rglob("*.py"))},
        "horizontal_fov_deg": hfov, "label_vertical_fov_deg": labels["fov_deg"],
        "label_alignment_scale": scale, "reference_reprojection_p50_px": float(np.median(reference_residual)),
        "reference_reprojection_p95_px": float(np.percentile(reference_residual,95)),
        "all_detector_components": len(detected), "all_detector_label_matches_3px": count_all,
        "detector_threshold": threshold, "bright_star_diagnostics": bright_diagnostics,
        "oracle_centroids": oracle,
        "reference_caveat": "Label-derived pseudo-reference; not independent ground truth. Translation differs slightly from the pipeline's exact image-center principal point.",
        "timing_caveat": "Single-image diagnostics, no warm-up; not a throughput benchmark.",
        "runs": runs,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# USSEG — Stellarium single-image check", "",
             f"Python {report['python']}; OpenCV one thread; unchanged detector/solver/database.", "",
             f"Image: {w}x{h}; {len(labels['stars'])} labels; perspective projection.",
             f"Label VFOV: {labels['fov_deg']} deg => HFOV {label_hfov:.6f} deg.",
             "Label data read after all blind inference calls.", "",
             "| Blind run | HFOV deg | Timeout ms | Status | Matches | Selected centroids matching labels (3px) |",
             "|---|---:|---:|---|---:|---:|"]
    for run in runs:
        lines.append(f"| {run['name']} | {run['config']['fov_deg']:.6f} | {run['config']['solve_timeout_ms']} | {run['result']['status']} | {run['result']['num_matches']} | {run['metrics']['selected_label_matches_3px']}/20 |")
    lines.extend(["", "## Detector diagnosis", "",
                  f"All components: {len(detected)}; one-to-one label matches: {count_all}/{len(labels['stars'])}.",
                  f"Top-Hat 7x7 threshold: {threshold:.3f}. See bright_star_diagnostics in JSON: many bright-star cores fall below threshold.",
                  "Unmatched detections are not automatically false stars: labels are magnitude-limited.", "",
                  "## Oracle centroids — NOT end-to-end", "",
                  f"20 brightest labelled centroids; solver status {oracle['status']}; matches {oracle['num_matches']}."])
    if oracle["status"] == "ok":
        metrics = oracle["metrics"]
        lines.extend([f"Boresight error versus metadata: {oracle['boresight_error_deg']:.6f} deg.",
                      f"Full attitude error versus label-derived reference: {metrics['attitude_error_deg']:.6f} deg.",
                      f"Correct catalog/centroid pairs (0.05 deg, 3px): {metrics['pairwise_star_id_correct']}/{metrics['pairwise_star_id_labelled']}."])
    lines.extend(["", f"Reference reprojection P50/P95: {report['reference_reprojection_p50_px']:.3f}/{report['reference_reprojection_p95_px']:.3f} px.",
                  "Labels/alignment are pseudo-reference, not independent truth. Do not subtract raw roll fields with different conventions.",
                  "This single-image diagnostic is not a dataset accuracy or FPS benchmark.", "",
                  "[Full results](results.json) | [Selected detector centroids](selected-centroids.png)"])
    if args.config:
        lines.extend(["", "The detector-diagnosis numbers and selected-centroids.png above describe the historical baseline, not the configured run.",
                      "[Configured selected centroids](configured-selected-centroids.png). See the configured_blind metrics in results.json for the preset."])
    (args.output_dir / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
