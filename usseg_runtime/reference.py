"""Reference scoring for the saved Stellarium fixture; not used during solving."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

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
