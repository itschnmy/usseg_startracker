import numpy as np

from examples.check_stellarium import reference, score


def test_stellarium_reference_axes_and_roll():
    rotation, scale, _ = reference({
        "center_ra_deg": 0, "center_dec_deg": 0,
        "alignment_matrix": [[0, -1, 10], [1, 0, 20]],
    })
    assert scale == 1
    assert np.allclose(rotation.apply([1, 0, 0]), [1, 0, 0])
    assert np.allclose(rotation.apply([0, 1, 0]), [0, 0, 1])
    assert np.allclose(rotation.apply([0, 0, 1]), [0, -1, 0])


def test_stellarium_scoring_does_not_invent_attitude_for_no_solve():
    rotation, _, _ = reference({
        "center_ra_deg": 0, "center_dec_deg": 0,
        "alignment_matrix": [[1, 0, 0], [0, 1, 0]],
    })
    metrics = score({"status": "no_solve", "centroids_xy": [[10, 20]]},
                    {"stars": [{"x": 11, "y": 20}]}, rotation)
    assert metrics["selected_label_matches_3px"] == 1
    assert metrics["attitude_error_deg"] is None
    assert metrics["pairwise_star_id_labelled"] == 0
