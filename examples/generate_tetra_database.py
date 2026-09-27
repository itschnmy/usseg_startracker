#!/usr/bin/env python3
"""Generate a Tetra database using the complete generator preserved on origin/main."""

import argparse
import gzip
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile


GENERATOR_REF = "origin/main:identificator/test-otetra3/plateSolver.py"


def git_blob(repo: Path, ref: str) -> bytes:
    return subprocess.run(
        ["git", "show", ref], cwd=repo, check=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-fov", required=True, type=float)
    parser.add_argument("--min-fov", type=float)
    parser.add_argument("--max-magnitude", type=float, default=7.0)
    parser.add_argument("--epoch-proper-motion", default="none")
    parser.add_argument("--catalog", required=True, type=Path, help="Full CDS hip_main.dat[.gz]")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="usseg-tetra-generator-") as temp_name:
        temp = Path(temp_name)
        module_path = temp / "plateSolver.py"
        module_path.write_bytes(git_blob(repo, GENERATOR_REF))
        catalog = args.catalog.resolve()
        if catalog.suffix == ".gz":
            with gzip.open(catalog, "rb") as source, (temp / "hip_main.dat").open("wb") as target:
                shutil.copyfileobj(source, target)
        else:
            shutil.copyfile(catalog, temp / "hip_main.dat")

        catalog_rows = sum(1 for _ in (temp / "hip_main.dat").open("rb"))
        if catalog_rows != 118_218:
            raise ValueError(
                f"Expected the complete 118218-row Hipparcos catalogue, got {catalog_rows} rows"
            )

        spec = importlib.util.spec_from_file_location("usseg_tetra_generator", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        generator = module.Tetra3(load_database=None)
        generator.generate_database(
            max_fov=args.max_fov,
            min_fov=args.min_fov,
            star_max_magnitude=args.max_magnitude,
            epoch_proper_motion=args.epoch_proper_motion,
            save_as=output,
        )

    print(output)


if __name__ == "__main__":
    main()
