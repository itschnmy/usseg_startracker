"""Opt-in verifier controls; no Astrometry labels used as solver hints."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from models.pipeline import PipelineConfig, StarTrackerPipeline
from models.pipeline.cli import build_parser
from models.identifier.plate_solver import PlateSolver

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["max_input_stars", "verification_catalog_stars"])
@pytest.mark.parametrize("value", [0, 3, True, 4.5, "40"])
def test_invalid_limits_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        PipelineConfig(database=Path("unused.npz"), fov_deg=20, **{name: value})
    solver = object.__new__(PlateSolver)
    solver._star_table = np.ones((1, 6))
    solver._pattern_catalog = np.ones((1, 4))
    with pytest.raises(ValueError, match=name):
        solver.solve_from_centroids(np.ones((4, 2)), (480, 640), **{name: value})


def test_cli_preserves_opt_in_limits():
    config = PipelineConfig(database=Path("db.npz"), fov_deg=20)
    assert config.max_input_stars is None
    assert config.verification_catalog_stars is None
    args = build_parser(config).parse_args([
        "--image", "dataset/calibration/openmv/session_01/03_calibrated.bmp", "--max-input-stars", "40", "--verification-catalog-stars", "90"])
    assert args.max_input_stars == 40
    assert args.verification_catalog_stars == 90


def test_calibrated_image03_blind_experimental_solution():
    """Development regression, not a held-out accuracy measurement."""
    config = PipelineConfig.from_json(ROOT / "configs/examples/image03-experimental.json")
    pipeline = StarTrackerPipeline(config)
    result = pipeline.run(ROOT / "dataset/calibration/openmv/session_01/03_calibrated.bmp")
    assert result["status"] == "ok"
    assert result["num_matches"] == 7
    # Post-inference assertions against the independently checked star field.
    assert set(result["matched_catalog_ids"]) == {15863, 14576, 14632, 14354, 16335, 17529, 14668}
    assert abs(result["ra_deg"] - 48.773) < .1
    assert abs(result["dec_deg"] - 41.364) < .1
    assert 16.6 < result["solved_fov_deg"] < 16.9
    assert result["mismatch_probability"] < config.match_threshold
    # No detector change: selected top20 remain exactly the old detections.
    import cv2
    image = cv2.imread(str(ROOT / "dataset/calibration/openmv/session_01/03_calibrated.bmp"), cv2.IMREAD_GRAYSCALE)
    old = PipelineConfig.from_json(ROOT / "configs/examples/image03.json")
    pipeline.config = replace(old, fov_deg=16.8)
    previous = pipeline.detect_stars(image)
    assert np.allclose(result["centroids_xy"][:20], [s.position for s in previous])
