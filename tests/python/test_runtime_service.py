from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

from usseg_runtime.config import load_config
from usseg_runtime.engine import Runner, read_job
from usseg_runtime.publish import publish_encoded
from usseg_runtime.storage import atomic_bytes, atomic_json, contained_file, file_hash, parse_json, read_json
from usseg_runtime.worker import Worker

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/"configs/runtime/development.json"


def black_job(tmp_path, name="black"):
    input_dir = tmp_path/"input"
    image = tmp_path/"black.png"
    _, buffer = cv2.imencode(".png", np.zeros((480, 640), np.uint8))
    atomic_bytes(image, buffer.tobytes())
    folder = publish_encoded(image, input_dir, name, "lab640")
    return input_dir, folder


def update_metadata(folder, values):
    metadata = read_json(folder/"frame.json")
    metadata.update(values)
    atomic_json(folder/"frame.json", metadata)
    ready = read_json(folder/"frame.ready")
    ready["metadata_sha256"] = file_hash(folder/"frame.json")
    atomic_json(folder/"frame.ready", ready)


def test_partial_files_ignored_bad_frame_does_not_block_next_and_resume(tmp_path):
    input_dir, folder = black_job(tmp_path)
    (input_dir/"partial.png").write_bytes(b"not finished")
    corrupt = publish_encoded(ROOT/"dataset/03.bmp", input_dir, "corrupt", "lab640")
    atomic_bytes(corrupt/"frame.bmp", b"wrong")
    solved = publish_encoded(ROOT/"dataset/03.bmp", input_dir, "image03", "lab640")
    runner = Runner(input_dir, tmp_path/"output", CONFIG)
    assert runner.run() == 3
    rows = {row["job_id"]: row for row in runner.current}
    assert len(rows) == 3
    assert rows["black"]["status"] == "no_solve"
    assert rows["black"]["pose"] is None
    assert rows["corrupt"]["status"] == "invalid_input"
    assert rows["image03"]["status"] == "solved"
    assert rows["image03"]["algorithm"]["num_matches"] == 7
    first_pose = rows["image03"]["pose"]
    resume = Runner(input_dir, tmp_path/"output", CONFIG)
    assert resume.run() == 0 and not resume.current
    assert len(list((tmp_path/"output/runs").glob("*/results/*.json"))) == 3
    # Loss of checkpoint after commit does not cause duplicate execution.
    atomic_json(tmp_path/"output/state/active.json", {"marker_relative": "image03/frame.ready", "attempt": 1})
    again = Runner(input_dir, tmp_path/"output", CONFIG)
    assert again.run() == 0 and not again.current
    assert again.previous["image03/frame.ready"]["pose"] == first_pose


@pytest.mark.parametrize("values", [
    {"payload": "../../outside.bmp"}, {"camera_profile_id": "unknown"},
    {"width": True}, {"reference_ra": 123}, {"orientation": "flip_y"},
    {"capture_timestamp_utc": "2026-11-04T10:00:00"},
    {"capture_timestamp_utc": 123}, {"exposure_us": -1},
])
def test_bad_protocol_rejected(tmp_path, values):
    input_dir, folder = black_job(tmp_path)
    update_metadata(folder, values)
    cfg, _, profiles, _ = load_config(CONFIG)
    with pytest.raises((ValueError, TypeError)):
        read_job(folder/"frame.ready", input_dir, profiles, cfg)


def test_duplicate_keys_nonfinite_and_symlinks_rejected(tmp_path):
    for content in [b'{"x":1,"x":2}', b'{"x":NaN}']:
        with pytest.raises(ValueError):
            parse_json(content)
    (tmp_path/"actual").write_bytes(b"x")
    (tmp_path/"link").symlink_to(tmp_path/"actual")
    with pytest.raises(ValueError, match="Symlink"):
        contained_file(tmp_path, "link")


def test_same_job_id_conflict_is_recorded(tmp_path):
    input_dir, folder = black_job(tmp_path, "duplicate")
    other = input_dir/"other"
    other.mkdir()
    for name in ("frame.png", "frame.json", "frame.ready"):
        atomic_bytes(other/name, (folder/name).read_bytes())
    update_metadata(other, {"exposure_us": 1})
    runner = Runner(input_dir, tmp_path/"output", CONFIG)
    assert runner.run() == 3
    assert [row["status"] for row in runner.current] == ["no_solve", "invalid_input"]
    assert runner.current[1]["error_code"] == "job_id_conflict"


def test_hard_deadline_kills_and_restarts_worker(tmp_path):
    cfg, db, profiles, _ = load_config(CONFIG)
    worker = Worker(db, profiles, replace(cfg, frame_timeout_s=.15))
    input_dir, folder = black_job(tmp_path)
    metadata, _ = read_job(folder/"frame.ready", input_dir, profiles, cfg)
    try:
        worker.start()
        os.kill(worker.process.pid, signal.SIGSTOP)
        started = time.monotonic()
        # Exercise blocked IPC send as well as a stopped solver. This payload
        # exceeds pipe capacity; parent still must honor the hard deadline.
        result = worker.execute({**metadata, "padding": "x"*1000000}, folder)
        assert result["status"] == "timeout"
        assert time.monotonic()-started < 2
        assert worker.process is None
        worker.start()
        assert worker.execute(metadata, folder)["status"] == "no_solve"
        assert worker.starts == 2
    finally:
        worker.close()


def wait_health(output, phase, process, timeout=20):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Runner exited prematurely: {process.returncode}")
        for path in (output/"runs").glob("*/health.json"):
            if read_json(path)["phase"] == phase:
                return path
        time.sleep(.05)
    raise AssertionError(f"No {phase} health")


def start_watch(input_dir, output):
    return subprocess.Popen([sys.executable, '-m', 'usseg_runtime.cli', 'watch', '--input', str(input_dir),
        '--output', str(output), '--config', str(CONFIG)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_watch_new_file_shutdown_and_output_lock(tmp_path):
    input_dir = tmp_path/"input"
    input_dir.mkdir()
    output = tmp_path/"output"
    process = start_watch(input_dir, output)
    try:
        health = wait_health(output, "idle", process)
        competitor = subprocess.run([sys.executable, '-m', 'usseg_runtime.cli', 'batch', '--input', str(input_dir),
            '--output', str(output), '--config', str(CONFIG)], cwd=ROOT, capture_output=True, timeout=20)
        assert competitor.returncode == 2
        black_job(tmp_path)
        deadline = time.monotonic()+10
        while not list((output/"runs").glob("*/results/*.json")) and time.monotonic() < deadline:
            time.sleep(.05)
        assert len(list((output/"runs").glob("*/results/*.json"))) == 1
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        assert read_json(health)["phase"] == "stopped"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_disk_budget_fails_before_dispatch(tmp_path):
    input_dir, _ = black_job(tmp_path)
    runner = Runner(input_dir, tmp_path/"output", CONFIG)
    runner.cfg = replace(runner.cfg, min_free_bytes=10**15)
    with pytest.raises(OSError, match="space"):
        runner.run()
    assert not list((tmp_path/"output/runs").glob("*/results/*.json"))


def test_sigkill_recovery_retries_active_job_once(tmp_path):
    input_dir = tmp_path/"input"
    input_dir.mkdir()
    output = tmp_path/"output"
    process = start_watch(input_dir, output)
    try:
        idle = wait_health(output, "idle", process)
        child_pid = read_json(idle)["worker_pid"]
        os.kill(child_pid, signal.SIGSTOP)
        black_job(tmp_path)
        wait_health(output, "processing", process)
        assert read_json(output/"state/active.json")["attempt"] == 1
        process.kill()
        process.communicate(timeout=5)
        resumed = Runner(input_dir, output, CONFIG)
        assert resumed.run() == 0
        assert len(resumed.current) == 1
        assert resumed.current[0]["status"] == "no_solve"
        assert resumed.current[0]["attempt"] == 2
        assert len(list((output/"runs").glob("*/results/*.json"))) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_shutdown_deadline_records_interrupted_then_resume(tmp_path):
    input_dir = tmp_path/"input"
    input_dir.mkdir()
    output = tmp_path/"output"
    process = start_watch(input_dir, output)
    try:
        idle = wait_health(output, "idle", process)
        os.kill(read_json(idle)["worker_pid"], signal.SIGSTOP)
        black_job(tmp_path)
        wait_health(output, "processing", process)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == 143
        results = list((output/"runs").glob("*/results/*.json"))
        assert len(results) == 1 and read_json(results[0])["status"] == "interrupted"
        resumed = Runner(input_dir, output, CONFIG)
        assert resumed.run() == 0
        assert resumed.current[0]["status"] == "no_solve"
        assert resumed.current[0]["attempt"] == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_recovery_uses_attempt_not_random_run_name(tmp_path):
    input_dir, _ = black_job(tmp_path)
    runner = Runner(input_dir, tmp_path/"output", CONFIG)
    runner.state.mkdir(parents=True)
    runner.run_dir.mkdir(parents=True)
    atomic_json(runner.state/"deployment.json", runner.deployment)
    base = {"schema_version": "1.0", "marker_relative": "black/frame.ready", "job_id": "black",
            "metadata_sha256": "1"*64}
    atomic_json(runner.output/"runs/z-earlier/results/x.json", {**base, "attempt": 1, "status": "interrupted"})
    atomic_json(runner.output/"runs/a-later/results/x.json", {**base, "attempt": 2, "status": "no_solve"})
    runner.recover()
    assert runner.previous["black/frame.ready"]["status"] == "no_solve"
    assert runner.by_job["black"]["attempt"] == 2
