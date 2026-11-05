from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from usseg_runtime.decode import FrameError, decode_job
from usseg_runtime.storage import atomic_bytes, sha256_bytes


def profile(bits, fmt, width=6, height=4):
    return {"width": width, "height": height, "allowed_formats": [fmt],
            "calibration": {"calibration_id": "synthetic", "black_level": 0,
                            "white_level": (1 << bits)-1, "dark": None}}


def request(tmp_path, data, fmt, width=6, height=4, **extra):
    atomic_bytes(tmp_path/"frame.raw", data)
    return {"payload": "frame.raw", "payload_size_bytes": len(data), "payload_sha256": sha256_bytes(data),
            "pixel_format": fmt, "width": width, "height": height, "processing_stage": "sensor_raw", **extra}


CFG = SimpleNamespace(max_payload_bytes=100000)


@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("endian", ["little", "big"])
@pytest.mark.parametrize("alignment", ["lsb", "msb"])
def test_raw_ramp_stride_endian_and_alignment(tmp_path, bits, endian, alignment):
    values = np.linspace(0, (1 << bits)-1, 24).astype(np.uint16).reshape(4, 6)
    storage = 8 if bits == 8 else 16
    packed = values << (storage-bits) if alignment == "msb" else values
    dtype = "u1" if bits == 8 else ("<u2" if endian == "little" else ">u2")
    rows = [row.astype(dtype).tobytes()+b"abc" for row in packed]
    data = b"HDR"+b"".join(rows)+b"TAIL"
    fmt = f"Mono{bits}"
    meta = request(tmp_path, data, fmt, packing="unpacked", storage_bits=storage, valid_bits=bits,
                   endianness=endian, bit_alignment=alignment, stride_bytes=6*storage//8+3,
                   header_bytes=3, trailer_bytes=4)
    image, metrics = decode_job(meta, tmp_path, profile(bits, fmt), CFG)
    expected = np.rint(values.astype(np.float32)/((1 << bits)-1)*255).astype(np.uint8)
    assert np.array_equal(image, expected)
    assert metrics["valid_bits"] == bits


def test_bayer_rg_channel_orientation(tmp_path):
    # Red at top-left; catches accidentally selecting BGR Bayer phase aliases.
    raw = np.zeros((6, 6), np.uint8)
    raw[::2, ::2] = 255
    meta = request(tmp_path, raw.tobytes(), "BayerRG8", width=6, height=6, packing="unpacked",
                   storage_bits=8, valid_bits=8, endianness="little", bit_alignment="lsb",
                   stride_bytes=6, bayer_pattern_at_roi_origin="RGGB")
    image, _ = decode_job(meta, tmp_path, profile(8, "BayerRG8", 6, 6), CFG)
    assert np.all((image > 70) & (image < 82))  # luminance of red, not blue


def test_native_dark_before_scaling_and_double_calibration_rejected(tmp_path):
    raw = np.full((4, 6), 1200, np.uint16)
    raw[1, 1] = 3900
    dark = np.full((4, 6), 200, np.float32)
    dark[1, 1] = 3900
    meta = request(tmp_path, raw.astype('<u2').tobytes(), "Mono12", packing="unpacked",
                   storage_bits=16, valid_bits=12, endianness="little", bit_alignment="lsb", stride_bytes=12)
    p = profile(12, "Mono12")
    image, metrics = decode_job(meta, tmp_path, p, CFG, dark)
    assert image[1, 1] == 0 and image[0, 0] == round(1000/4095*255)
    assert metrics["dark_applied"]
    meta["processing_stage"] = "already_corrected"
    with pytest.raises(FrameError, match="twice"):
        decode_job(meta, tmp_path, p, CFG, dark)


@pytest.mark.parametrize("change,match", [
    ({"payload_sha256": "0"*64}, "checksum"),
    ({"width": 5}, "dimensions"),
    ({"packing": "packed"}, "Packed"),
    ({"stride_bytes": 11}, "stride_bytes"),
    ({"endianness": "native"}, "endian"),
])
def test_bad_raw_rejected(tmp_path, change, match):
    meta = request(tmp_path, bytes(48), "Mono12", packing="unpacked", storage_bits=16,
                   valid_bits=12, endianness="little", bit_alignment="lsb", stride_bytes=12)
    meta.update(change)
    with pytest.raises(ValueError, match=match):
        decode_job(meta, tmp_path, profile(12, "Mono12"), CFG)


def test_encoded16bit_and_oversized_header_rejected(tmp_path):
    _, image = cv2.imencode(".png", np.zeros((4, 6), np.uint16))
    meta = request(tmp_path, image.tobytes(), "encoded8")
    with pytest.raises(FrameError, match="8-bit"):
        decode_job(meta, tmp_path, profile(8, "encoded8"), CFG)
    _, image = cv2.imencode(".png", np.zeros((4, 6), np.uint8))
    data = bytearray(image.tobytes())
    data[16:20] = (100000).to_bytes(4, 'big')
    meta = request(tmp_path, bytes(data), "encoded8")
    with pytest.raises(FrameError, match="dimensions"):
        decode_job(meta, tmp_path, profile(8, "encoded8"), CFG)


def test_no_unsigned_wraparound(tmp_path):
    _, data = cv2.imencode(".png", np.full((4, 6), 2, np.uint8))
    meta = request(tmp_path, data.tobytes(), "encoded8")
    image, _ = decode_job(meta, tmp_path, profile(8, "encoded8"), CFG, np.full((4, 6), 10, np.float32))
    assert not image.any()
