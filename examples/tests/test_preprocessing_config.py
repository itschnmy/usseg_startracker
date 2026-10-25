import json
from pathlib import Path

import numpy as np
import pytest

from models.detector.star_detector import StarDetector
from models.pipeline import PipelineConfig
from models.pipeline.cli import build_parser, main


def test_component_mask_excludes_other_star_in_bounding_box():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8, 8:21] = image[20, 8:21] = 40
    image[8:21, 8] = 40
    image[13, 18] = 255  # Separate component inside the C-shaped bounding box.
    stars = StarDetector(use_tophat=False, max_area=100, component_mask=True).process(image)
    component = max(stars, key=lambda s: s.intensity)
    ys, xs = np.where(image == 40)
    assert np.allclose(component.position, [xs.mean(), ys.mean()])
    assert component.peak == 40
    assert component.intensity == 40*len(xs)


def test_border_margin_rejects_entire_touching_component():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[1:4, 1:4] = image[15:18, 15:18] = 255
    stars = StarDetector(use_tophat=False, border_margin_px=4).process(image)
    assert len(stars) == 1
    assert np.allclose(stars[0].position, [16, 16])


def test_json_config_paths_and_cli_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"database": "data/db.npz", "fov_deg": 29.7,
                                       "use_tophat": False, "star_ranking": "flux"}))
    config = PipelineConfig.from_json(config_path)
    assert config.database == tmp_path / "data/db.npz"
    args = build_parser(config).parse_args(["--image", "test.png", "--min-area", "4"])
    assert args.min_area == 4 and not args.use_tophat and args.star_ranking == "flux"
    captured = {}

    class FakePipeline:
        def __init__(self, cfg):
            captured["config"] = cfg

        def run(self, image):
            return {"status": "no_solve"}

    monkeypatch.setattr("models.pipeline.cli.StarTrackerPipeline", FakePipeline)
    assert main(["--config", str(config_path), "--image", "test.png", "--min-area", "4",
                 "--output", str(tmp_path/"output.json")]) == 0
    assert captured["config"].min_area == 4
    assert not captured["config"].use_tophat


def test_legacy_defaults_remain_opt_in():
    config = PipelineConfig(database=Path("db.npz"), fov_deg=20)
    assert not config.component_mask
    assert config.border_margin_px == 0
    assert config.star_ranking == "peak"
    assert config.use_tophat and config.tophat_kernel_size == 7


def test_reject_unknown_ranking():
    with pytest.raises(ValueError, match="star_ranking"):
        PipelineConfig(database=Path("db.npz"), fov_deg=20, star_ranking="unknown")


@pytest.mark.parametrize("values", [
    {"fov_deg": float("nan")}, {"fov_max_error_deg": -1},
    {"max_stars": -1}, {"pattern_checking_stars": 3}, {"min_area": 0},
    {"min_area": 10, "max_area": 5}, {"sigma_threshold": 0},
    {"border_margin_px": -1}, {"tophat_kernel_size": 8},
    {"match_radius": -1}, {"match_threshold": 0},
    {"solve_timeout_ms": float("inf")}, {"solve_timeout_ms": -1},
    {"attitude_method": "invalid"},
])
def test_invalid_configuration_fails_early(values):
    with pytest.raises(ValueError):
        PipelineConfig(**{"database": Path("db.npz"), "fov_deg": 20, **values})


def test_json_config_requires_object(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        PipelineConfig.from_json(path)


def test_cli_reports_bad_override_as_error_json(tmp_path):
    output = tmp_path / "error.json"
    code = main(["--image", "missing.png", "--database", "missing.npz",
                 "--fov", "20", "--max-stars", "-1", "--output", str(output)])
    assert code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "error"
    assert "max_stars" in result["message"]


def test_stellarium_preset_solves_blind_then_agrees_with_labels():
    from models.pipeline import StarTrackerPipeline
    from examples.check_stellarium import reference, score

    root = Path(__file__).resolve().parents[2]
    image = root / "dataset/stellarium_test/stellarium_view.png"
    labels = image.with_name("stars_labels.json")
    if not image.exists() or not labels.exists():
        pytest.skip("Optional user-provided Stellarium example assets are unavailable")
    result = StarTrackerPipeline(PipelineConfig.from_json(
        root / "configs/examples/stellarium.json")).run(image)
    assert result["status"] == "ok"
    # No reference is loaded until inference is complete.
    label_data = json.loads(labels.read_text())
    ref, _, _ = reference(label_data)
    metrics = score(result, label_data, ref)
    assert result["num_matches"] >= 8
    assert metrics["attitude_error_deg"] < .2
    assert metrics["pairwise_star_id_correct"] == metrics["pairwise_star_id_labelled"]
    assert metrics["pairwise_star_id_labelled"] >= 8
