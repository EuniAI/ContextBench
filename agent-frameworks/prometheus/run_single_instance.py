#!/usr/bin/env python3
"""Run Prometheus on a single ContextBench / SWE-bench instance via the HTTP API."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from bench_sources import fetch_records, normalize_record, split_problem_statement
from docker_image import resolve_docker_image
from prometheus_client import (
    PrometheusClient,
    copy_log_to_output,
    find_newest_log,
    write_result_json,
)

ADAPTER_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMETHEUS_ROOT = ADAPTER_ROOT / "prometheus"


def _default_prometheus_url() -> str:
    return os.environ.get("PROMETHEUS_URL", "http://localhost:9002/v1.3")


def _prometheus_root() -> Path:
    prom_root = os.environ.get("PROMETHEUS_ROOT", "")
    return Path(prom_root) if prom_root else DEFAULT_PROMETHEUS_ROOT


def _default_log_dir() -> Path:
    wd = os.environ.get("PROMETHEUS_WORKING_DIRECTORY", "")
    if wd:
        return Path(wd) / "answer_issue_logs"
    return _prometheus_root() / "working_dir" / "answer_issue_logs"


def run_instance(
    bench_name: str,
    lookup_id: str,
    output_dir: Path,
    prometheus_url: str,
    timeout: int,
) -> None:
    records = fetch_records(bench_name, [lookup_id])
    record = records.get(lookup_id)
    if record is None:
        raise RuntimeError(f"No dataset record for {lookup_id} on bench {bench_name}")

    record = normalize_record(record, lookup_id)
    repo = str(record.get("repo") or "")
    if not repo or "/" not in repo:
        raise RuntimeError(f"Invalid repo for {lookup_id}: {repo}")

    base_commit = str(record.get("base_commit") or "")
    if len(base_commit) != 40:
        raise RuntimeError(f"Invalid base_commit for {lookup_id}: {base_commit}")

    image_name, workdir = resolve_docker_image(bench_name, record)
    title, body = split_problem_statement(str(record.get("problem_statement") or ""))

    out_bench = output_dir / bench_name
    out_bench.mkdir(parents=True, exist_ok=True)
    save_id = str(record.get("original_inst_id") or lookup_id)
    dest_log = out_bench / f"{save_id}.log"
    dest_json = out_bench / f"{save_id}.json"

    client = PrometheusClient(prometheus_url, timeout=timeout)
    log_dir = _default_log_dir()
    since = time.time()

    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("PROMETHEUS_GITHUB_TOKEN")
    repository_id = client.upload_repository(repo, commit_id=base_commit, github_token=github_token)

    candidate_patches = int(os.environ.get("PROMETHEUS_CANDIDATE_PATCHES", "3"))
    api_data = client.answer_issue(
        repository_id,
        title,
        body,
        image_name=image_name,
        workdir=workdir,
        issue_type=os.environ.get("PROMETHEUS_ISSUE_TYPE", "bug"),
        run_reproduce_test=os.environ.get("PROMETHEUS_RUN_REPRODUCE", "true").lower() in ("1", "true", "yes"),
        run_regression_test=os.environ.get("PROMETHEUS_RUN_REGRESSION", "true").lower() in ("1", "true", "yes"),
        run_existing_test=os.environ.get("PROMETHEUS_RUN_EXISTING_TEST", "false").lower() in ("1", "true", "yes"),
        number_of_candidate_patch=candidate_patches,
    )

    write_result_json(dest_json, save_id, api_data)

    log_path = find_newest_log(log_dir, since)
    if log_path:
        copy_log_to_output(log_path, dest_log)
    else:
        # Minimal log so trajectory convert still finds a file
        dest_log.write_text(
            f"Prometheus run completed for {save_id}. No answer_issue_logs found at {log_dir}.\n",
            encoding="utf-8",
        )

    print(f"OK: {save_id} -> {dest_log} (image={image_name}, workdir={workdir})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Prometheus on one SWE-bench instance")
    parser.add_argument("bench_name", choices=["Multi", "Poly", "Pro", "Verified"])
    parser.add_argument("--instance", required=True, help="instance_id or original_inst_id")
    parser.add_argument("--output", type=Path, default=Path("results/prometheus"))
    parser.add_argument("--prometheus-url", default=_default_prometheus_url())
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("PROMETHEUS_TIMEOUT", "3600")))
    args = parser.parse_args()

    try:
        run_instance(
            args.bench_name,
            args.instance.strip(),
            args.output,
            args.prometheus_url,
            args.timeout,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
