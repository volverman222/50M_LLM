#!/usr/bin/env python3
"""Explicit scalar-only export from a completed local training telemetry snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_mini_lab.telemetry_export import export_telemetry, load_export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument(
        "--allow-online", action="store_true", help="Explicitly permit online export"
    )
    parser.add_argument(
        "--api-key-env",
        help="Name of an existing credential variable; never the key itself",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview without importing W&B or writing files",
    )
    args = parser.parse_args()
    if args.dry_run:
        data = load_export(args.telemetry)
        print(
            json.dumps(
                {
                    "status": "preview",
                    "source_sha256": data["source_sha256"],
                    "config": data["config"],
                    "payload_count": len(data["payloads"]),
                    "first_payload": data["payloads"][0],
                },
                indent=2,
                allow_nan=False,
            )
        )
        return
    result = export_telemetry(
        args.telemetry,
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        allow_online=args.allow_online,
        api_key_env=args.api_key_env,
        output_dir=args.output_dir,
        receipt=args.receipt or args.output_dir / "export_receipt.json",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
