#!/usr/bin/env python3
"""Batch entry for ContextBench — run Prometheus on one or more instances."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bench_sources import load_task_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Prometheus on ContextBench instances")
    parser.add_argument("bench_name", choices=["Multi", "Poly", "Pro", "Verified"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--prometheus-url", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    prom_dir = Path(__file__).resolve().parent
    if args.instance:
        tasks = [args.instance.strip()]
    else:
        tasks = load_task_ids(args.bench_name, args.limit)
    if not tasks:
        print("No tasks found.", file=sys.stderr)
        return 1

    failed = 0
    for lookup_id in tasks:
        cmd = [
            sys.executable,
            str(prom_dir / "run_single_instance.py"),
            args.bench_name,
            "--instance",
            lookup_id,
            "--output",
            str(args.output),
        ]
        if args.prometheus_url:
            cmd.extend(["--prometheus-url", args.prometheus_url])
        if args.timeout:
            cmd.extend(["--timeout", str(args.timeout)])
        print(f"Running Prometheus | {args.bench_name} | {lookup_id}", flush=True)
        result = subprocess.run(cmd, cwd=str(prom_dir))
        if result.returncode != 0:
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
