#!/usr/bin/env python3
"""Upload a local JSON profile to SwanLab for visualization.

Usage:
    python scripts/upload_profile.py profile.json
    python scripts/upload_profile.py profile.json --project mini-vllm --name run_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload profile JSON to SwanLab")
    parser.add_argument("profile", help="Path to profile JSON file")
    parser.add_argument("--project", default="mini-vllm", help="SwanLab project name")
    parser.add_argument("--name", default=None, help="SwanLab experiment name")
    args = parser.parse_args()

    path = Path(args.profile)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    with open(path) as f:
        records = json.load(f)

    try:
        import swanlab

        swanlab.init(project=args.project, experiment_name=args.name)

        for record in records:
            step = record.get("step")
            metrics = {
                k: v
                for k, v in record.items()
                if k not in ("name", "step", "timestamp") and isinstance(v, (int, float))
            }
            if metrics:
                swanlab.log(metrics, step=step)

        swanlab.finish()
        print(f"Uploaded {len(records)} records to SwanLab project '{args.project}'")
    except ImportError:
        print(
            "Error: swanlab is not installed. Install it with: pip install swanlab",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error uploading to SwanLab: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
