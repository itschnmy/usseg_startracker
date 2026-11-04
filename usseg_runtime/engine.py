"""Sequential supervisor, durable per-frame results, crash recovery and resume."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import io
import csv
import os
from pathlib import Path
import platform
import shutil
import signal
from time import monotonic, perf_counter_ns, sleep
import uuid

from .config import DIGEST, ERRORS, dependencies, integer, load_config, require_id, source_fingerprint
from .storage import atomic_bytes, atomic_json, canonical_hash, contained_file, parse_json, read_json, sha256_bytes
from .worker import Worker

META_REQUIRED = {"schema_version", "job_id", "payload", "payload_size_bytes", "payload_sha256",
                 "camera_profile_id", "width", "height", "pixel_format", "processing_stage"}
META_ALLOWED = META_REQUIRED | {"capture_timestamp_utc", "timestamp_reference", "clock_source",
    "packing", "storage_bits", "valid_bits", "endianness", "bit_alignment", "stride_bytes",
    "header_bytes", "trailer_bytes", "roi_origin_xy", "binning_xy", "bayer_pattern_at_roi_origin",
    "orientation", "exposure_us", "gain_db", "sensor_temperature_c", "calibration_id"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_job(marker: Path, input_root: Path, profiles, cfg):
    contained_file(input_root, str(marker.relative_to(input_root)))
    ready = read_json(marker, 4096)
    if not isinstance(ready, dict) or set(ready) != {"schema_version", "job_id", "metadata_sha256"} or ready["schema_version"] != "1.0":
        raise ValueError("Invalid ready marker")
    metadata_path = contained_file(marker.parent, marker.with_suffix(".json").name)
    with metadata_path.open("rb") as stream:
        data = stream.read(65537)
    if len(data) > 65536 or sha256_bytes(data) != ready["metadata_sha256"]:
        raise ValueError("Metadata size/checksum mismatch")
    metadata = parse_json(data)
    if not isinstance(metadata, dict) or not META_REQUIRED <= set(metadata) or set(metadata)-META_ALLOWED:
        raise ValueError("Missing or unknown metadata fields")
    if metadata["schema_version"] != "1.0":
        raise ValueError("Unsupported metadata schema")
    require_id(metadata["job_id"], "job_id")
    if ready["job_id"] != metadata["job_id"]:
        raise ValueError("Ready/metadata job ID mismatch")
    if metadata["camera_profile_id"] not in profiles:
        raise ValueError("Unknown camera profile")
    integer(metadata["width"], "width", high=32768)
    integer(metadata["height"], "height", high=32768)
    integer(metadata["payload_size_bytes"], "payload_size_bytes", high=cfg.max_payload_bytes)
    if metadata["width"]*metadata["height"] > cfg.max_pixels:
        raise ValueError("Image exceeds max_pixels")
    if not isinstance(metadata["payload_sha256"], str) or not DIGEST.fullmatch(metadata["payload_sha256"]):
        raise ValueError("Invalid payload SHA256")
    contained_file(marker.parent, metadata["payload"])
    if metadata.get("orientation", "sensor_native") != "sensor_native" or metadata.get("roi_origin_xy", [0, 0]) != [0, 0] or metadata.get("binning_xy", [1, 1]) != [1, 1]:
        raise ValueError("ROI/binning/orientation requires a separately validated adapter")
    timestamp = metadata.get("capture_timestamp_utc")
    if timestamp is not None:
        if not isinstance(timestamp, str):
            raise ValueError("Capture timestamp must be a UTC string or null")
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("Capture timestamp must be timezone-aware UTC")
    import math
    for name in ("exposure_us", "gain_db", "sensor_temperature_c"):
        value = metadata.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"Invalid numeric capture field: {name}")
    if metadata.get("exposure_us") is not None and metadata["exposure_us"] < 0:
        raise ValueError("exposure_us must be nonnegative")
    # Also rejects nonfinite number overflow such as 1e999.
    return metadata, canonical_hash(metadata)


class Runner:
    def __init__(self, input_dir: Path, output_dir: Path, config_path: Path, mode="batch"):
        self.input = input_dir.resolve()
        self.output = output_dir.resolve()
        if not self.input.is_dir():
            raise ValueError("Input directory does not exist")
        if self.input == self.output or self.input in self.output.parents or self.output in self.input.parents:
            raise ValueError("Input and output trees must be disjoint")
        self.cfg, self.database, self.profiles, config_hash = load_config(config_path.resolve())
        self.deployment = {"schema_version": "1.0", "config_sha256": config_hash,
            "database_sha256": self.cfg.database_sha256, "source_sha256": source_fingerprint(),
            "dependencies": dependencies(), "python": platform.python_version(),
            "profiles": {key: value["sha256"] for key, value in self.profiles.items()}}
        self.mode = mode
        self.stop_at = None
        self.stop_signal = None
        self.worker = Worker(self.database, self.profiles, self.cfg)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        self.run_dir = self.output / "runs" / self.run_id
        self.state = self.output / "state"
        self.current = []
        self.previous = {}
        self.by_job = {}
        self.processed = set()
        self.attempts = {}
        self.initial_handlers = {}

    def request_stop(self, signum, _frame):
        if self.stop_at is None:
            self.stop_at = monotonic()
            self.stop_signal = signum

    def health(self, phase, **extra):
        atomic_json(self.run_dir / "health.json", {"schema_version": "1.0", "run_id": self.run_id,
            "updated_at_utc": utc_now(), "phase": phase, "worker_starts": self.worker.starts,
            "worker_pid": self.worker.process.pid if self.worker.process is not None else None, **extra})

    def event(self, name, **extra):
        import json
        path = self.run_dir / "events.jsonl"
        if path.exists() and path.stat().st_size >= self.cfg.max_events_bytes:
            os.replace(path, path.with_suffix(".previous.jsonl"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time_utc": utc_now(), "event": name, **extra}, allow_nan=False) + "\n")

    def disk_check(self):
        if shutil.disk_usage(self.output).free < self.cfg.min_free_bytes:
            raise OSError("Free output space below configured minimum")

    def recover(self):
        deployment_path = self.state / "deployment.json"
        if deployment_path.exists():
            if read_json(deployment_path) != self.deployment:
                raise ValueError("Output state belongs to another config/source/database; use a fresh output directory")
        else:
            if any((self.output / "runs").glob("*/results/*.json")):
                raise ValueError("Existing results without deployment state; manual recovery required")
            atomic_json(deployment_path, self.deployment)
        # Results, not summary/checkpoint, are authoritative after power loss.
        for path in sorted((self.output / "runs").glob("*/results/*.json")):
            record = read_json(path, 16*1024*1024)
            key = record["marker_relative"]
            self.attempts[key] = max(self.attempts.get(key, 0), record["attempt"])
            existing = self.previous.get(key)
            # UUID lexicographic order is not execution order, especially for
            # runs started within one second. Attempts are monotonic per marker.
            if existing is None or record["attempt"] > existing["attempt"]:
                self.previous[key] = record
        for record in self.previous.values():
            if record.get("metadata_sha256") and record.get("job_id") and record.get("error_code") != "job_id_conflict":
                self.by_job[record["job_id"]] = record
        active_path = self.state / "active.json"
        if active_path.exists():
            active = read_json(active_path)
            if active:
                key = active["marker_relative"]
                self.attempts[key] = max(self.attempts.get(key, 0), active["attempt"])
                self.event("recover_active", marker=key, attempt=active["attempt"])

    def scan(self):
        markers = []
        # Do not follow producer-controlled directory symlinks.
        for folder, dirs, files in os.walk(self.input, followlinks=False):
            dirs[:] = sorted(x for x in dirs if not (Path(folder)/x).is_symlink())
            for name in sorted(files):
                if name.endswith(".ready"):
                    markers.append(Path(folder)/name)
                    if len(markers) > self.cfg.max_ready_jobs:
                        raise OSError("Input backlog exceeds max_ready_jobs; producer backpressure required")
        return sorted(markers)

    def commit(self, record):
        self.disk_check()
        key = record["marker_relative"]
        path = self.run_dir / "results" / (sha256_bytes(key.encode()) + ".json")
        write_started = perf_counter_ns()
        record["result_path"] = str(path.relative_to(self.output))
        atomic_json(path, record)
        self.current.append(record)
        self.previous[key] = record
        self.processed.add(key)
        if record.get("job_id") and record.get("metadata_sha256") and record.get("error_code") != "job_id_conflict":
            self.by_job[record["job_id"]] = record
        # If killed here, recover() finds the already committed result first.
        atomic_json(self.state / "active.json", None)
        self.event("result_committed", job_id=record.get("job_id"), marker=key,
                   status=record["status"], output_write_ns=perf_counter_ns()-write_started)
        self.summary()

    def summary(self):
        counts = dict(Counter(row["status"] for row in self.current))
        latest = list(self.previous.values())
        summary = {"schema_version": "1.0", "run_id": self.run_id, "frames_this_run": len(self.current),
                   "statuses_this_run": counts, "unique_markers_total": len(latest),
                   "statuses_latest": dict(Counter(row["status"] for row in latest)),
                   "worker_starts": self.worker.starts, "database_load_ns": self.worker.database_load_ns,
                   "updated_at_utc": utc_now()}
        def percentiles(values):
            values = sorted(values)
            if not values:
                return None
            def quantile(q):
                index = (len(values)-1)*q
                lower = int(index)
                upper = min(lower+1, len(values)-1)
                return values[lower]+(values[upper]-values[lower])*(index-lower)
            return {"p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99)}
        summary["timing_ns"] = {name: percentiles([row[name] for row in self.current if name in row])
                                for name in ("supervisor_frame_ns", "worker_total_ns")}
        total_compute = sum(row.get("worker_total_ns", 0) for row in self.current)
        timed = [row for row in self.current if row.get("worker_total_ns")]
        summary["compute_fps"] = len(timed)*1e9/total_compute if total_compute else None
        summary["solved_updates_per_compute_second"] = (sum(row["status"] == "solved" for row in timed)*1e9/total_compute
                                                         if total_compute else None)
        summary["peak_worker_rss_kib"] = max((row.get("peak_rss_kib", 0) for row in self.current), default=0)
        atomic_json(self.run_dir / "summary.json", summary)
        buffer = io.StringIO()
        fields = ["job_id", "marker_relative", "status", "attempt", "result_path"]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.current)
        atomic_bytes(self.run_dir / "summary.csv", buffer.getvalue().encode())

    def process(self, marker):
        key = str(marker.relative_to(self.input))
        if key in self.processed:
            return
        start = perf_counter_ns()
        metadata = None
        digest = None
        record = {"schema_version": "1.0", "run_id": self.run_id, "marker_relative": key,
            "job_id": None, "metadata_sha256": None, "status": "invalid_input", "pose": None,
            "reference_status": "unavailable", "processed_at_utc": utc_now(),
            "provenance": self.deployment, "attempt": self.attempts.get(key, 0)+1}
        try:
            metadata, digest = read_job(marker, self.input, self.profiles, self.cfg)
            record.update(job_id=metadata["job_id"], metadata_sha256=digest, input=metadata)
            previous = self.previous.get(key)
            if previous is not None:
                if previous.get("metadata_sha256") != digest:
                    raise ValueError("Published job mutated; marker path must be immutable")
                if previous["status"] != "interrupted":
                    self.processed.add(key)
                    return
            existing = self.by_job.get(metadata["job_id"])
            if existing and existing["marker_relative"] != key:
                if existing["metadata_sha256"] == digest:
                    self.event("duplicate_skipped", job_id=metadata["job_id"], marker=key)
                    self.processed.add(key)
                    return
                record["error_code"] = "job_id_conflict"
                raise ValueError("Job ID already published with different metadata")
            if record["attempt"] > self.cfg.max_attempts:
                record.update(status="runtime_error", message="Retry limit exhausted after interrupted/crashed jobs")
            else:
                self.worker.start(lambda: self.stop_at is not None)
                if self.stop_at is not None:
                    raise InterruptedError("Stopped before dispatch")
                self.attempts[key] = record["attempt"]
                atomic_json(self.state / "active.json", {"marker_relative": key, "job_id": metadata["job_id"], "attempt": record["attempt"]})
                self.health("processing", active_job=metadata["job_id"])
                result = self.worker.execute(metadata, marker.parent, lambda: self.stop_at)
                record.update(result)
                if "algorithm" in result:
                    record["timings_ns"] = {**result["algorithm"]["timings"],
                        "decode_ns": result["conversion"]["decode_ns"],
                        "calibration_ns": result["conversion"]["calibration_ns"]}
                if record["status"] == "solved":
                    algorithm = record["algorithm"]
                    record["pose"] = {"quaternion_wxyz": algorithm["quaternion_inertial_to_body_wxyz"],
                        "direction": "catalog_to_camera", "camera_axes": "+X boresight,+Y image-left,+Z image-up",
                        "inertial_reference": "database catalog; epoch/frame not certified for flight",
                        "ra_deg": algorithm["ra_deg"], "dec_deg": algorithm["dec_deg"],
                        "roll_deg": algorithm["roll_deg"], "solved_fov_deg": algorithm["solved_fov_deg"]}
        except InterruptedError as exc:
            record.update(status="interrupted", message=str(exc))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # Missing/invalid ready metadata is still a terminal result with a stable marker key.
            if metadata is None:
                previous = self.previous.get(key)
                if previous and previous.get("metadata_sha256") is None:
                    self.processed.add(key)
                    return
            record.update(status="invalid_input", message=f"{type(exc).__name__}: {exc}")
        record["supervisor_frame_ns"] = perf_counter_ns()-start
        self.commit(record)

    def run(self):
        self.state.mkdir(parents=True, exist_ok=True)
        lock_path = self.state / "runner.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another runner holds this output directory") from exc
            self.run_dir.mkdir(parents=True, exist_ok=False)
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    self.initial_handlers[sig] = signal.signal(sig, self.request_stop)
                self.recover()
                atomic_json(self.run_dir / "run.json", {"schema_version": "1.0", "run_id": self.run_id,
                    "started_at_utc": utc_now(), "mode": self.mode, "provenance": self.deployment,
                    "runtime_config": asdict(self.cfg), "platform": platform.platform(),
                    "python": platform.python_version(), "input_dir": str(self.input)})
                self.disk_check()
                self.health("starting")
                self.worker.start(lambda: self.stop_at is not None)
                self.health("idle")
                while self.stop_at is None:
                    markers = self.scan()  # Batch captures one snapshot.
                    for marker in markers:
                        if self.stop_at is not None:
                            break
                        self.disk_check()
                        self.process(marker)
                    self.health("idle", ready_markers=len(markers))
                    if self.mode == "batch":
                        break
                    sleep(self.cfg.poll_interval_s)
                self.summary()
                code = (128+self.stop_signal if self.stop_signal else
                        (3 if any(row["status"] in ERRORS for row in self.current) else 0))
                self.health("stopped" if self.stop_at is not None else "completed", exit_code=code)
                return code
            except InterruptedError:
                self.summary()
                self.health("stopped", exit_code=128+(self.stop_signal or signal.SIGTERM))
                return 128+(self.stop_signal or signal.SIGTERM)
            except Exception as exc:
                try:
                    self.health("fatal", message=f"{type(exc).__name__}: {exc}")
                except OSError:
                    pass
                raise
            finally:
                self.worker.close()
                for sig, handler in self.initial_handlers.items():
                    signal.signal(sig, handler)
