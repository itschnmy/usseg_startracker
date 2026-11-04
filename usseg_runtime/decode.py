"""Explicit RAW layout adapter. Physical-camera acceptance requires SDK samples."""
from __future__ import annotations

import struct
from time import perf_counter_ns

import cv2
import numpy as np

from .config import integer
from .storage import contained_file, sha256_bytes


class FrameError(ValueError):
    def __init__(self, message, status="invalid_input"):
        super().__init__(message)
        self.status = status


def _encoded_size(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33:
        if data[12:16] != b"IHDR" or data[24] != 8:
            raise FrameError("Only 8-bit PNG is supported")
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"BM") and len(data) >= 54:
        dib = struct.unpack_from("<I", data, 14)[0]
        if dib < 40:
            raise FrameError("Unsupported BMP header")
        width, height = struct.unpack_from("<ii", data, 18)
        bpp = struct.unpack_from("<H", data, 28)[0]
        if bpp not in (8, 24, 32) or width <= 0 or height == 0:
            raise FrameError("Unsupported BMP pixel layout")
        return width, abs(height)
    raise FrameError("Only PNG/BMP encoded input is supported", "unsupported_format")


def load_dark(profile):
    path = profile["calibration"]["dark"]
    if path is None:
        return None
    dark = np.load(path, allow_pickle=False)
    if (dark.shape != (profile["height"], profile["width"]) or
            not np.issubdtype(dark.dtype, np.number) or np.iscomplexobj(dark) or
            not np.isfinite(dark).all() or dark.min() < 0 or dark.max() > 65535):
        raise ValueError("Dark must match profile geometry and native pixel range")
    return dark.astype(np.float32)


def decode_job(metadata, directory, profile, cfg, dark=None):
    started = perf_counter_ns()
    path = contained_file(directory, metadata["payload"])
    expected = integer(metadata["payload_size_bytes"], "payload_size_bytes", high=cfg.max_payload_bytes)
    if path.stat().st_size != expected:
        raise FrameError("Payload byte count mismatch")
    with path.open("rb") as stream:
        data = stream.read(cfg.max_payload_bytes + 1)
    if len(data) != expected or sha256_bytes(data) != metadata["payload_sha256"]:
        raise FrameError("Payload checksum mismatch")
    width, height = profile["width"], profile["height"]
    if metadata["width"] != width or metadata["height"] != height:
        raise FrameError("Image dimensions disagree with camera profile")
    fmt = metadata["pixel_format"]
    if fmt not in profile["allowed_formats"]:
        raise FrameError(f"Format {fmt} is not enabled by profile", "unsupported_format")
    if fmt == "encoded8":
        if _encoded_size(data) != (width, height):
            raise FrameError("Encoded header dimensions disagree with metadata/profile")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.dtype != np.uint8 or image.shape != (height, width):
            raise FrameError("Could not decode 8-bit image")
        bits = 8
    else:
        bits = int(fmt.removeprefix("Mono").removeprefix("BayerRG"))
        if metadata.get("packing") != "unpacked":
            raise FrameError("Packed RAW is not supported until SDK layout is verified", "unsupported_format")
        storage = 8 if bits == 8 else 16
        if metadata.get("storage_bits") != storage or metadata.get("valid_bits") != bits:
            raise FrameError("RAW storage/valid bit declaration mismatch")
        endian = metadata.get("endianness")
        alignment = metadata.get("bit_alignment")
        if endian not in ("little", "big") or alignment not in ("lsb", "msb"):
            raise FrameError("RAW endian/alignment must be explicit")
        row_bytes = width * (storage // 8)
        stride = integer(metadata.get("stride_bytes"), "stride_bytes", low=row_bytes, high=cfg.max_payload_bytes)
        header = integer(metadata.get("header_bytes", 0), "header_bytes", low=0, high=cfg.max_payload_bytes)
        trailer = integer(metadata.get("trailer_bytes", 0), "trailer_bytes", low=0, high=cfg.max_payload_bytes)
        if len(data) != header + height*stride + trailer:
            raise FrameError("RAW buffer length disagrees with stride/header/trailer")
        rows = np.frombuffer(data, np.uint8, count=height*stride, offset=header).reshape(height, stride)
        dtype = np.dtype("u1" if storage == 8 else ("<u2" if endian == "little" else ">u2"))
        image = np.frombuffer(rows[:, :row_bytes].copy().tobytes(), dtype).reshape(height, width).astype(np.uint16)
        shift = storage-bits
        if alignment == "msb":
            if shift and np.any(image & ((1 << shift)-1)):
                raise FrameError("Nonzero unused low bits in MSB RAW")
            image >>= shift
        elif np.any(image >= 1 << bits):
            raise FrameError("Nonzero unused high bits in LSB RAW")
        if fmt.startswith("Bayer") and metadata.get("bayer_pattern_at_roi_origin") != "RGGB":
            raise FrameError("Only explicit RGGB at ROI origin is supported", "unsupported_format")
    decode_ns = perf_counter_ns() - started
    cal_started = perf_counter_ns()
    cal = profile["calibration"]
    if cal["white_level"] > (1 << bits)-1:
        raise FrameError("Calibration range exceeds pixel bit depth")
    stage = metadata["processing_stage"]
    if stage not in ("sensor_raw", "already_corrected"):
        raise FrameError("Unknown processing_stage")
    if stage == "already_corrected" and (dark is not None or cal["black_level"] != 0):
        raise FrameError("Already-corrected input would be calibrated twice")
    if stage == "already_corrected" and metadata.get("calibration_id") != cal["calibration_id"]:
        raise FrameError("Already-corrected input must declare its calibration ID")
    values = image.astype(np.float32)
    original_range = [float(values.min()), float(values.max())]
    if dark is not None:
        if float(dark.max()) > (1 << bits)-1:
            raise FrameError("Dark range exceeds native bit depth")
        # Master dark includes the sensor offset. Do not subtract black twice.
        values -= dark
    else:
        values -= cal["black_level"]
    scale = cal["white_level"] - cal["black_level"]
    below, above = float(np.mean(values < 0)), float(np.mean(values > scale))
    intensity = np.rint(np.clip(values / scale * 255, 0, 255)).astype(np.uint8)
    if fmt.startswith("Bayer"):
        if width < 4 or height < 4 or width % 2 or height % 2:
            raise FrameError("Bayer conversion requires even dimensions, at least 4x4")
        intensity = cv2.cvtColor(intensity, cv2.COLOR_BayerRGGB2GRAY)
    return intensity, {
        "decode_ns": decode_ns, "calibration_ns": perf_counter_ns()-cal_started,
        "native_range": original_range, "valid_bits": bits,
        "clipped_low_fraction": below, "clipped_high_fraction": above,
        "saturation_fraction": float(np.mean(image == (1 << bits)-1)),
        "conversion": "fixed_linear_uint8_then_bayer_rg_grey" if fmt.startswith("Bayer") else "fixed_linear_uint8",
        "calibration_id": cal["calibration_id"], "black_level": cal["black_level"],
        "white_level": cal["white_level"], "dark_applied": dark is not None,
    }
