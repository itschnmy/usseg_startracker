"""Ground-side producer helper for the draft ready protocol."""
from __future__ import annotations

from pathlib import Path

from .config import require_id
from .storage import atomic_bytes, atomic_json, file_hash, sha256_bytes


def publish_encoded(image: Path, input_dir: Path, job_id: str, profile_id: str):
    from .decode import _encoded_size

    require_id(job_id, "job_id")
    require_id(profile_id, "profile_id")
    data = image.read_bytes()
    width, height = _encoded_size(data)
    folder = input_dir / job_id
    folder.mkdir(parents=True, exist_ok=False)
    suffix = ".png" if data.startswith(b"\x89PNG") else ".bmp"
    payload = folder / ("frame"+suffix)
    atomic_bytes(payload, data)
    metadata_path = folder / "frame.json"
    atomic_json(metadata_path, {"schema_version": "1.0", "job_id": job_id,
        "payload": payload.name, "payload_size_bytes": len(data), "payload_sha256": sha256_bytes(data),
        "camera_profile_id": profile_id, "width": width, "height": height, "pixel_format": "encoded8",
        "processing_stage": "sensor_raw", "capture_timestamp_utc": None,
        "clock_source": "unknown", "timestamp_reference": "unknown"})
    atomic_json(folder / "frame.ready", {"schema_version": "1.0", "job_id": job_id,
                                        "metadata_sha256": file_hash(metadata_path)})
    return folder
