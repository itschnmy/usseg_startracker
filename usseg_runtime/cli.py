from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Set before importing NumPy/OpenCV or spawning workers; host limits can be stricter.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"


def main(argv=None):
    parser = argparse.ArgumentParser(description="USSEG file runtime, draft schema1.0")
    parser.add_argument("command", choices=("batch", "watch", "self-test", "prepare-input"))
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--config", type=Path, default=Path("/app/configs/runtime/development.json"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-input":
            if not args.image or not args.job_id or not args.profile:
                parser.error("prepare-input requires --image, --job-id, --profile")
            from .publish import publish_encoded
            print(publish_encoded(args.image, args.input, args.job_id, args.profile))
            return 0
        if args.command == "self-test":
            from .selftest import self_test
            return self_test(args.output, args.config)
        from .engine import Runner
        runner = Runner(args.input, args.output, args.config, args.command)
        code = runner.run()
        print(json.dumps({"run_id": runner.run_id, "output": str(runner.run_dir), "exit_code": code}))
        return code
    except (ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "preflight_error", "message": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "infrastructure_error", "message": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
