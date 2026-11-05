import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.dark_frame import build_master_dark, subtract_dark, load_master_dark
from models.pipeline import PipelineConfig, StarTrackerPipeline
from models.pipeline.cli import build_parser


def test_master_median_rejects_transient_and_keeps_fixed_hot_pixel(tmp_path):
    for i in range(3):
        image = np.full((20, 20), 10, np.uint8)
        image[5, 5] = 200  # Fixed sensor defect must remain in dark.
        if i == 0:
            image[8, 8] = 255  # Transient must not remain.
        cv2.imwrite(str(tmp_path / f'{i}.bmp'), image)
    master, paths = build_master_dark(tmp_path)
    assert len(paths) == 3
    assert master.dtype == np.float32
    assert master[5, 5] == 200 and master[8, 8] == 10
    science = np.full((20, 20), 10, np.uint8)
    science[5, 5] = 200
    science[12, 12] = 110
    clean = subtract_dark(science, master)
    assert clean[5, 5] == 0 and clean[12, 12] == 100
    assert science[5, 5] == 200


def test_no_unsigned_wrap_and_shape_rejected():
    assert subtract_dark(np.array([[2]], np.uint8), np.array([[10]], np.float32))[0, 0] == 0
    with pytest.raises(ValueError, match='do not resize'):
        subtract_dark(np.zeros((4, 4), np.uint8), np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match='8-bit'):
        subtract_dark(np.zeros((2, 2), np.uint16), np.zeros((2, 2)))


def test_bgr_dark_uses_same_grayscale_as_pipeline(tmp_path):
    bgr = np.full((3, 4, 3), (10, 20, 30), np.uint8)
    for i in range(2):
        cv2.imwrite(str(tmp_path / f'{i}.bmp'), bgr)
    master, _ = build_master_dark(tmp_path)
    assert np.array_equal(master, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    assert not subtract_dark(bgr, master).any()


def test_invalid_dark_and_mismatched_frames(tmp_path):
    path = tmp_path / 'bad.npy'
    np.save(path, np.array([[np.nan]], np.float32))
    with pytest.raises(ValueError, match='finite'):
        load_master_dark(path)
    cv2.imwrite(str(tmp_path / 'a.bmp'), np.zeros((2, 2), np.uint8))
    cv2.imwrite(str(tmp_path / 'b.bmp'), np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError, match='dimensions'):
        build_master_dark(tmp_path)


def test_config_cli_and_serialization(tmp_path):
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps({'database': 'db.npz', 'fov_deg': 20, 'dark_frame': 'master.npy'}))
    config = PipelineConfig.from_json(cfg)
    assert config.dark_frame == tmp_path / 'master.npy'
    args = build_parser(config).parse_args(['--image', 'image.bmp'])
    assert args.dark_frame == config.dark_frame


def test_pipeline_dark_removes_fixed_hot_pixel_and_preserves_star(tmp_path):
    dark = np.full((64, 64), 10, np.float32)
    dark[5:8, 5:8] = 200
    path = tmp_path / 'master.npy'
    np.save(path, dark)
    database = Path(__file__).resolve().parents[2] / 'data/default_database.npz'
    pipeline = StarTrackerPipeline(PipelineConfig(database=database, fov_deg=20,
        dark_frame=path, use_tophat=False, min_area=3, component_mask=True))
    image = dark.astype(np.uint8)
    image[30:33, 30:33] = 110
    stars = pipeline.detect_stars(image)
    assert len(stars) == 1
    assert np.allclose(stars[0].position, [31, 31])
    result = pipeline.run_array(image)
    assert result['status'] == 'no_solve'
    json.dumps(result, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else a)
    assert result['config']['dark_frame'] == str(path)
