import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from models.pipeline import PipelineConfig, StarTrackerPipeline
from models.identifier.plate_solver import _compute_vectors


@pytest.mark.parametrize("method", ["SVD", "QUEST", "DAVENPORT", "TRIAD"])
def test_published_quaternion_maps_catalog_to_camera(method, monkeypatch):
    """Exercise the public output with asymmetric, known matched star vectors."""
    from models.detector.star_detector import StarDetector, DetectedStar
    from models.identifier.plate_solver import PlateSolver

    xy = np.array([[20., 30.], [180., 40.], [70., 190.], [200., 200.]])
    body = _compute_vectors(xy[:, ::-1], (256, 256), np.deg2rad(26))
    expected = Rotation.from_euler("xyz", [20, 30, 40], degrees=True)
    inertial = expected.inv().apply(body)
    ra = np.degrees(np.arctan2(inertial[:, 1], inertial[:, 0])) % 360
    dec = np.degrees(np.arcsin(inertial[:, 2]))
    monkeypatch.setattr(PlateSolver, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(PlateSolver, "database_properties", property(lambda self: {"min_fov": 10, "max_fov": 30}))
    monkeypatch.setattr(StarDetector, "process", lambda self, image: [
        DetectedStar(n, body[n], point, 10., 100, 1.) for n, point in enumerate(xy)
    ])
    monkeypatch.setattr(PlateSolver, "solve_from_centroids", lambda self, *a, **k: {
        "RA": 0., "Dec": 0., "Roll": 0., "FOV": 26., "Matches": 4,
        "RMSE": 0., "Prob": 0., "matched_centroids": xy[:, ::-1].tolist(),
        "matched_stars": np.column_stack((ra, dec, np.ones(4))).tolist(),
    })
    pipeline = StarTrackerPipeline(PipelineConfig(database="unused.npz", fov_deg=26, attitude_method=method))
    result = pipeline.run_array(np.zeros((256, 256), dtype=np.uint8))
    q = np.array(result["quaternion_inertial_to_body_wxyz"])
    actual = Rotation.from_quat(q[[1, 2, 3, 0]])
    assert np.degrees((actual.inv() * expected).magnitude()) < 1e-6
    assert np.allclose(actual.apply(inertial), body, atol=1e-10)
    assert np.allclose(result["quaternion_body_to_inertial_wxyz"], q * [1, -1, -1, -1])
