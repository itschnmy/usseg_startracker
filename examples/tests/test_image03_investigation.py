import json

from examples.investigate_image03 import read_records, summarize, trial_key


def test_interrupted_log_is_preserved_and_reported(tmp_path):
    path = tmp_path / "trials.jsonl"
    record = {"status": "no_solve", "elapsed_ms": 1000}
    raw = json.dumps(record) + "\n\x00\x00\x00"
    path.write_text(raw)
    rows, invalid = read_records(path)
    assert rows == [record] and invalid == 1
    summary = summarize(path)
    assert summary["trials"] == 1
    assert summary["invalid_lines"] == 1
    assert summary["runtime_errors"] == 0
    assert path.read_text() == raw


def test_trial_key_distinguishes_geometry():
    base = {"profile": "test", "selected_count": 12, "mirrored": False,
            "fov_start_deg": 17, "database": "10-30"}
    assert trial_key(base) == trial_key({**base, "pixel_aspect_scale_x": 1})
    assert trial_key(base) != trial_key({**base, "pixel_aspect_scale_x": .75})
    assert trial_key(base) != trial_key({**base, "mirrored": True})
