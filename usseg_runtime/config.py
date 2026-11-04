from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from .storage import canonical_hash, contained_file, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("solved", "no_solve", "invalid_input", "unsupported_format", "timeout", "runtime_error", "interrupted")
ERRORS = set(STATUSES) - {"solved", "no_solve"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def require_id(value, name="ID"):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def integer(value, name, low=1, high=100_000_000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: str
    database: str
    database_sha256: str
    profiles: dict
    frame_timeout_s: float = 15.0
    startup_timeout_s: float = 30.0
    shutdown_grace_s: float = 5.0
    poll_interval_s: float = .2
    max_payload_bytes: int = 64 * 1024 * 1024
    max_pixels: int = 16 * 1024 * 1024
    min_free_bytes: int = 64 * 1024 * 1024
    max_attempts: int = 2
    max_ready_jobs: int = 100000
    max_events_bytes: int = 2 * 1024 * 1024


def load_config(path: Path):
    from models.pipeline import PipelineConfig

    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Runtime config must be an object")
    cfg = RuntimeConfig(**raw)
    if cfg.schema_version != "1.0":
        raise ValueError("Unsupported runtime schema_version")
    for name in ("frame_timeout_s", "startup_timeout_s", "shutdown_grace_s", "poll_interval_s"):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
    for name in ("max_payload_bytes", "max_pixels", "max_attempts", "max_ready_jobs", "max_events_bytes"):
        integer(getattr(cfg, name), name)
    integer(cfg.min_free_bytes, "min_free_bytes", low=0, high=10**15)
    if not isinstance(cfg.database_sha256, str) or not DIGEST.fullmatch(cfg.database_sha256):
        raise ValueError("database_sha256 must be a SHA256 digest")
    # Database may be absolute inside the container; config must be trusted/read-only.
    database = (path.parent / cfg.database).resolve()
    if not database.is_file() or file_hash(database) != cfg.database_sha256:
        raise ValueError("Database missing or checksum mismatch")
    if not isinstance(cfg.profiles, dict) or not cfg.profiles:
        raise ValueError("At least one explicit camera profile is required")
    profiles = {}
    for name, relative in cfg.profiles.items():
        require_id(name, "profile ID")
        profile_path = contained_file(path.parent, relative)
        profile = read_json(profile_path)
        required = {"schema_version", "profile_id", "width", "height", "allowed_formats", "pipeline", "calibration"}
        if not isinstance(profile, dict) or set(profile) != required:
            raise ValueError(f"Invalid profile fields: {name}")
        if profile["schema_version"] != "1.0" or profile["profile_id"] != name:
            raise ValueError("Profile ID/schema mismatch")
        w, h = (integer(profile[key], key, high=32768) for key in ("width", "height"))
        if w*h > cfg.max_pixels:
            raise ValueError("Profile exceeds max_pixels")
        supported = {"encoded8", "Mono8", "Mono10", "Mono12", "BayerRG8", "BayerRG10", "BayerRG12"}
        formats = profile["allowed_formats"]
        if not isinstance(formats, list) or not formats or any(x not in supported for x in formats):
            raise ValueError("Unsupported profile pixel format")
        pipeline_values = profile["pipeline"]
        if not isinstance(pipeline_values, dict) or "database" in pipeline_values or pipeline_values.get("dark_frame") is not None:
            raise ValueError("Database/dark must be configured by runtime, not pipeline profile")
        # Reuse all algorithm validation; do not silently accept typo fields.
        PipelineConfig(database=database, **pipeline_values)
        cal = profile["calibration"]
        if not isinstance(cal, dict) or set(cal) != {"calibration_id", "black_level", "white_level", "dark", "dark_sha256"}:
            raise ValueError("Invalid calibration fields")
        require_id(cal["calibration_id"], "calibration ID")
        black, white = cal["black_level"], cal["white_level"]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (black, white)) or not 0 <= black < white <= 65535:
            raise ValueError("Invalid fixed black/white range")
        if cal["dark"] is not None:
            dark_path = contained_file(path.parent, cal["dark"])
            if file_hash(dark_path) != cal["dark_sha256"]:
                raise ValueError("Dark checksum mismatch")
            cal["dark"] = str(dark_path)
        elif cal["dark_sha256"] is not None:
            raise ValueError("dark_sha256 without dark")
        profile["sha256"] = canonical_hash(profile)
        profiles[name] = profile
    return cfg, database, profiles, canonical_hash(asdict(cfg))


def source_fingerprint():
    records = {}
    for folder in ("models", "utils", "usseg_runtime"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            records[str(path.relative_to(ROOT))] = file_hash(path)
    lock = ROOT / "requirements-runtime.lock"
    if lock.exists():
        records[lock.name] = file_hash(lock)
    return canonical_hash(records)


def dependencies():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ("numpy", "scipy", "opencv-python", "opencv-python-headless", "Pillow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            continue
    return result
