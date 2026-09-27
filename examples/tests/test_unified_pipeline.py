from pathlib import Path

import cv2

import numpy as np

from usseg_pipeline.pipeline import PipelineConfig, StarTrackerPipeline


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "identificator" / "default_database.npz"


def test_database_rejects_non_overlapping_fov():
    if not DATABASE.exists():
        return

    config = PipelineConfig(database=DATABASE, fov_deg=45, fov_max_error_deg=5)
    try:
        StarTrackerPipeline(config)
    except ValueError as exc:
        assert "does not overlap" in str(exc)
    else:
        raise AssertionError("30-degree database unexpectedly accepted 45-degree FOV")


def test_lost_quaternion_is_inverse_of_internal(tmp_path):
    """Integration fixture supplied with LOST evals, when available locally."""
    image = ROOT.parent / "lost-evals" / "scenarios-pyramid" / "20-low-noise" / "images" / "0.png"
    if not DATABASE.exists() or not image.exists():
        return

    result = StarTrackerPipeline(PipelineConfig(database=DATABASE, fov_deg=20)).run(image)
    assert result["status"] == "ok"
    q_lost = np.asarray(result["quaternion_lost_wxyz"])
    q_internal = np.asarray(result["quaternion_body_to_inertial_wxyz"])
    assert np.allclose(q_lost, q_internal * np.array([1, -1, -1, -1]))
    assert np.allclose(result["quaternion_inertial_to_body_wxyz"], q_lost)


def test_array_adapter_matches_file_pipeline():
    image_path = ROOT.parent / "lost-evals" / "scenarios-pyramid" / "20-low-noise" / "images" / "0.png"
    if not DATABASE.exists() or not image_path.exists():
        return
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    pipeline = StarTrackerPipeline(PipelineConfig(database=DATABASE, fov_deg=20))
    from_file = pipeline.run(image_path)
    from_array = pipeline.run_array(image, source="fixture")
    assert from_file["status"] == from_array["status"] == "ok"
    assert np.allclose(from_file["quaternion_inertial_to_body_wxyz"], from_array["quaternion_inertial_to_body_wxyz"])
    assert np.allclose(from_file["centroids_xy"], from_array["centroids_xy"])


def test_solver_wrapper_receives_xy_centroids(monkeypatch):
    """The local PlateSolver wrapper accepts detector coordinates as (x,y)."""
    if not DATABASE.exists():
        return

    from identificator.star_detector import DetectedStar

    image = np.zeros((32, 32), dtype=np.uint8)
    pipeline = StarTrackerPipeline(PipelineConfig(database=DATABASE, fov_deg=20))
    stars = [
        DetectedStar(
            i, np.array([1.0, 0.0, 0.0]),
            np.array([3.0 + i, 9.0 + i]), 10.0, 255, 1.0,
        )
        for i in range(4)
    ]
    monkeypatch.setattr(
        "usseg_pipeline.pipeline.StarDetector.process",
        lambda _self, _image: stars,
    )
    captured = {}

    def fake_solve(centroids, *_args, **_kwargs):
        captured["centroids"] = np.asarray(centroids)
        return {"RA": None}

    monkeypatch.setattr(pipeline.solver, "solve_from_centroids", fake_solve)
    pipeline.run_array(image)
    expected = np.asarray([[3.0 + i, 9.0 + i] for i in range(4)])
    assert np.array_equal(captured["centroids"], expected)
