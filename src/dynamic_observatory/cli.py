from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .artifact import load_observation
from .reference import build_reference_frame, write_reference_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynamic-observatory")
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="build a fixed 3D frame from one observation")
    ref.add_argument("observation", type=Path)
    ref.add_argument("output", type=Path)
    ref.add_argument("--components", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reference":
        obs = load_observation(args.observation)
        states = torch.tensor(obs.states, dtype=torch.float32)
        frame = build_reference_frame(
            states,
            probe_id=obs.probe_id,
            trace_name=obs.trace_name,
            run_id=obs.run_id,
            checkpoint_tokens=obs.checkpoint_tokens,
            components=args.components,
        )
        write_reference_frame(args.output, frame)
        print(f"reference_frame={args.output}")
        print(f"run_id={frame.run_id} checkpoint_tokens={frame.checkpoint_tokens} trace={frame.trace_name}")
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
