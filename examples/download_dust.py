#!/usr/bin/env python3
"""Download, verify MD5, and extract the official DUST V2 orbital flight dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

ZENODO_URL = "https://zenodo.org/records/20255672/files/DUST.zip?download=1"
EXPECTED_MD5 = "a31b62290eface15e519ed954124d59c"
EXPECTED_SIZE = 861_350_187


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    offset = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    print(f"Downloading DUST V2 from Zenodo ({EXPECTED_SIZE / (1024*1024):.1f} MB)...")
    with urllib.request.urlopen(request) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with destination.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    print("Extracting archive contents...")
    with ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        handle.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("dataset/DUST"),
        help="Target directory to store extracted DUST dataset (default: dataset/DUST)",
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "DUST.zip"

    if not archive.exists() or archive.stat().st_size < EXPECTED_SIZE:
        download(ZENODO_URL, archive)

    if archive.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"Size mismatch: expected {EXPECTED_SIZE}, got {archive.stat().st_size}")

    print("Verifying MD5 checksum...")
    actual = md5(archive)
    if actual != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: expected {EXPECTED_MD5}, got {actual}")

    if not (args.destination / "DUST").is_dir():
        safe_extract(archive, args.destination)

    provenance = {
        "source_url": ZENODO_URL,
        "zenodo_record": "20255672",
        "version": "V2",
        "archive": str(archive.resolve()),
        "md5": actual,
        "license": "CC-BY-4.0",
    }
    (args.destination / "dataset-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"DUST V2 dataset ready at: {args.destination / 'DUST'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
