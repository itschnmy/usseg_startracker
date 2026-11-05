from pathlib import Path

import cv2
import numpy as np
import pytest

from models.detector.star_detector import DetectedStar
from models.pipeline import PipelineConfig, StarTrackerPipeline


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "default_database.npz"
LOST_EVALS = ROOT.parent / "lost-evals"


@pytest.fixture
def lost_image():
    image = LOST_EVALS / "scenarios-pyramid" / "20-low-noise" / "images" / "0.png"
    if not image.exists():
        pytest.skip(f"LOST fixture is unavailable: {image}")
    return image


def test_database_rejects_non_overlapping_fov():
    assert DATABASE.exists(), f"missing test database: {DATABASE}"
    config = PipelineConfig(database=DATABASE, fov_deg=45, fov_max_error_deg=5)
    with pytest.raises(ValueError, match="does not overlap"):
        StarTrackerPipeline(config)


def test_published_quaternion_directions_are_inverse(lost_image):
    """Integration fixture supplied with LOST evals, when available locally."""
    result = StarTrackerPipeline(PipelineConfig(database=DATABASE, fov_deg=20)).run(lost_image)
    assert result["status"] == "ok"
    q_lost = np.asarray(result["quaternion_lost_wxyz"])
    q_body_to_inertial = np.asarray(result["quaternion_body_to_inertial_wxyz"])
    assert np.allclose(q_lost, q_body_to_inertial * np.array([1, -1, -1, -1]))
    assert np.allclose(result["quaternion_inertial_to_body_wxyz"], q_lost)


def test_array_adapter_matches_file_pipeline(lost_image):
    image_path = lost_image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    pipeline = StarTrackerPipeline(PipelineConfig(database=DATABASE, fov_deg=20))
    from_file = pipeline.run(image_path)
    from_array = pipeline.run_array(image, source="fixture")
    assert from_file["status"] == from_array["status"] == "ok"
    assert np.allclose(from_file["quaternion_inertial_to_body_wxyz"], from_array["quaternion_inertial_to_body_wxyz"])
    assert np.allclose(from_file["centroids_xy"], from_array["centroids_xy"])


def test_solver_wrapper_receives_xy_centroids(monkeypatch):
    """The local PlateSolver wrapper accepts detector coordinates as (x,y)."""
    assert DATABASE.exists(), f"missing test database: {DATABASE}"

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
        "models.pipeline.pipeline.StarDetector.process",
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


def test_cli_defaults_match_pipeline_config():
    from models.pipeline.cli import build_parser

    args = build_parser().parse_args([
        "--image", "example.png", "--database", str(DATABASE), "--fov", "26",
    ])
    config = PipelineConfig(database=DATABASE, fov_deg=26)
    names = {
        "fov_max_error": "fov_max_error_deg",
        **{name: name for name in (
            "sigma_threshold", "min_area", "max_area", "tophat_kernel_size",
            "use_tophat", "component_mask", "border_margin_px", "star_ranking",
            "max_stars", "pattern_checking_stars", "match_radius",
            "match_threshold", "solve_timeout_ms", "attitude_method",
        )},
    }
    for argument, field in names.items():
        assert getattr(args, argument) == getattr(config, field), argument


@pytest.mark.parametrize("image", [
    np.zeros(10), np.zeros((10, 10, 4)), np.full((10, 10), np.nan),
    np.array([["not a pixel"]]),
])
def test_array_adapter_rejects_invalid_pixels(image):
    # Input validation runs before any detector or solver accesses the image.
    pipeline = object.__new__(StarTrackerPipeline)
    pipeline.config = PipelineConfig(database=DATABASE, fov_deg=20)
    result = pipeline.run_array(image)
    assert result["status"] == "error"
