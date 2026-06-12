#!/usr/bin/env python3
"""
ContextBench Agent Runner

Unified script to:
1. Get task list from ContextBench (selected CSV, gold JSONL, or custom subset)
2. Filter by bench and/or specific instances
3. Run the specified agent from agent-frameworks, dispatching each instance to
   the bench-adapted agent runner (Verified→verified/, Pro→pro/, etc.)

Strategy: For each instance, detect its native bench (Verified/Pro/Poly/Multi),
then call the agent framework that has been adapted for that bench.

Prerequisites:
- For agentless: Ensure data/ contains dataset splits (Verified, Pro, Poly, Multi) and
  run `python agent-frameworks/agentless/run_bench.py {bench} --instance ID` for single instances.
- For prometheus: Start Prometheus (Neo4j + Postgres + API); set API URL via PROMETHEUS_URL or --prometheus-url; configure LLM keys in agent-frameworks/prometheus/prometheus/.env (not via this script). See agent-frameworks/prometheus/README.md
- Each agent framework must have its own environment/dependencies installed.

Usage:
    # Run agentless on Verified bench only (from selected_500_instances.csv)
    python -m contextbench.run --agent agentless --bench Verified

    # Run miniswe on first 5 Pro instances
    python -m contextbench.run --agent miniswe --bench Pro --limit 5

    # Run on specific instances only
    python -m contextbench.run --agent agentless --instances "scikit-learn__scikit-learn-25232,django__django-14434"

    # Use custom subset CSV
    python -m contextbench.run --agent miniswe --subset-csv my_subset.csv --output results/my_run

    # Dry run (list tasks only)
    python -m contextbench.run --agent miniswe --bench Verified --dry-run

    # Run Prometheus (LLM keys in prometheus/.env; URL via --prometheus-url or PROMETHEUS_URL)
    python -m contextbench.run --agent prometheus --bench Verified --limit 1 --prometheus-url http://localhost:9002/v1.3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root (Context-Bench)
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_FRAMEWORKS = REPO_ROOT / "agent-frameworks"
DEFAULT_TASK_CSV = REPO_ROOT / "data" / "selected_500_instances.csv"
_DEBUG = False


# ---------------------------------------------------------------------------
# Bench detection
# ---------------------------------------------------------------------------

BENCH_ALIASES = {
    "verified": "Verified",
    "pro": "Pro",
    "poly": "Poly",
    "multi": "Multi",
}


def detect_bench_from_instance_id(instance_id: str, original_inst_id: str = "") -> str:
    """Infer benchmark from instance_id format when bench column is missing."""
    if not instance_id:
        instance_id = original_inst_id or ""
    s = instance_id.strip()
    if not s:
        return "Verified"  # fallback

    if s.startswith("SWE-Bench-Pro__") or (
        s.startswith("instance_") and len(s) > 50
    ):
        return "Pro"
    if s.startswith("SWE-PolyBench__") or "polybench" in s.lower():
        return "Poly"
    if "multi" in s.lower() or "Multi" in s:
        return "Multi"
    if s.startswith("SWE-Bench-Verified__") or "__" in s and "-" in s:
        return "Verified"
    return "Verified"


# ---------------------------------------------------------------------------
# Task list loading
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Remove null bytes and other non-printable control characters (except newline/tab)."""
    return "".join(c for c in text if c.isprintable() or c in "\n\r\t")


def _run_subprocess(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    debug: bool = False,
) -> subprocess.CompletedProcess:
    """Run subprocess with optional live debug output."""
    if debug:
        print(f"[debug] cwd={cwd}")
        print(f"[debug] cmd={' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        return subprocess.CompletedProcess(cmd, result.returncode, stdout="", stderr="")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    # Clean null bytes / non-printable chars from captured output to avoid log corruption
    result.stdout = _clean_text(result.stdout)
    result.stderr = _clean_text(result.stderr)
    return result


def _openhands_has_model_config(run_dir: Path, model_config: str) -> bool:
    config_path = run_dir / "config.toml"
    if not config_path.exists() or not model_config:
        return False
    try:
        content = config_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return f"[llm.{model_config}]" in content


def _sync_dir(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    import shutil
    for root, _, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dest / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in files:
            shutil.copy2(Path(root) / fname, target_dir / fname)


def _clear_previous_outputs(agent: str, bench: str, output_dir: Path, instance_ids: List[str]) -> None:
    import shutil
    base = output_dir / agent / bench
    if not base.exists():
        return
    if agent == "prometheus":
        for inst in instance_ids:
            if not inst:
                continue
            for path in base.glob(f"{inst}.*"):
                if path.is_file():
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
    if agent == "miniswe":
        for pattern in ("preds.json", "minisweagent.log", "exit_statuses_*.yaml"):
            for path in base.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
    for inst in instance_ids:
        if not inst:
            continue
        for path in base.glob(f"{inst}*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _toml_quote(value: str) -> str:
    return "'" + value.replace("'", "\\'") + "'"


def _openhands_benchmark_dir(run_dir: Path, bench: str) -> Path:
    # Multi uses swe_bench runner in this repo layout
    return run_dir / "evaluation" / "benchmarks" / "swe_bench"


def _resolve_openhands_run_dir(base_dir: Path, bench: str) -> Path:
    if bench != "Multi":
        return base_dir
    if (base_dir / "evaluation").exists():
        return base_dir
    nested = base_dir / "MopenHands"
    if nested.exists():
        return nested
    return base_dir


def _infer_repo_from_instance_id(inst_id: str) -> Optional[str]:
    if not inst_id:
        return None
    m = re.match(r"^(?P<org>[^_]+)__(?P<repo>[^-]+)(?:-\d+)?$", inst_id)
    if not m:
        return None
    org = m.group("org")
    repo = m.group("repo")
    return f"{org}/{repo}"


def _parse_multi_instance_id(instance_id: str) -> Optional[Tuple[str, str, int]]:
    if not instance_id or "__" not in instance_id or "-" not in instance_id:
        return None
    try:
        org_repo, num_str = instance_id.rsplit("-", 1)
        org, repo = org_repo.split("__", 1)
        return org, repo, int(num_str)
    except Exception:
        return None


def _find_multi_sweagent_record(run_dir: Path, instance_id: str) -> Optional[dict]:
    parsed = _parse_multi_instance_id(instance_id)
    if not parsed:
        return None
    org, repo, number = parsed
    data_root = run_dir / "data"
    if not data_root.exists():
        return None
    target_name = f"{org}__{repo}_dataset.jsonl"
    candidates = list(data_root.rglob(target_name))
    if not candidates:
        return None
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("version https://git-lfs"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("org") == org and rec.get("repo") == repo and int(rec.get("number", -1)) == number:
                        return rec
        except Exception:
            continue
    return None


@contextmanager
def _openhands_selected_ids(run_dir: Path, bench: str, selected_ids: List[str]):
    if not selected_ids:
        yield None
        return
    bench_dir = _openhands_benchmark_dir(run_dir, bench)
    bench_dir.mkdir(parents=True, exist_ok=True)
    config_path = bench_dir / "config.toml"
    backup_content = None
    if config_path.exists():
        backup_content = config_path.read_text(encoding="utf-8")
    content = "selected_ids = [" + ", ".join(_toml_quote(s) for s in selected_ids) + "]\n"
    config_path.write_text(content, encoding="utf-8")
    try:
        yield config_path
    finally:
        if backup_content is None:
            try:
                config_path.unlink()
            except FileNotFoundError:
                pass
        else:
            config_path.write_text(backup_content, encoding="utf-8")


@contextmanager
def _openhands_temp_config(run_dir: Path):
    config_path = run_dir / "config.toml"
    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if "[llm.llm]" not in existing:
            model = os.environ.get("OPENHANDS_MODEL", "openai/gpt-5")
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
            existing += "\n" + "\n".join([
                "[llm.llm]",
                f"model = {_toml_quote(model)}",
                "api_key = \"placeholder\"",
                f"base_url = {_toml_quote(base_url)}",
                "",
            ])
            config_path.write_text(existing, encoding="utf-8")
        yield config_path
        return
    template_path = run_dir / "config.template.toml"
    backup_content = None
    if template_path.exists():
        backup_content = template_path.read_text(encoding="utf-8")
        content = backup_content
        if "[llm.llm]" not in content:
            model = os.environ.get("OPENHANDS_MODEL", "openai/gpt-5")
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
            content += "\n" + "\n".join([
                "[llm.llm]",
                f"model = {_toml_quote(model)}",
                "api_key = \"placeholder\"",
                f"base_url = {_toml_quote(base_url)}",
                "",
            ])
        config_path.write_text(content, encoding="utf-8")
    else:
        model = os.environ.get("OPENHANDS_MODEL", "openai/gpt-5")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or ""
        content = "\n".join([
            "[llm.llm]",
            f"model = {_toml_quote(model)}",
            "api_key = \"placeholder\"",
            f"base_url = {_toml_quote(base_url)}",
            "",
        ])
        config_path.write_text(content, encoding="utf-8")
    try:
        yield config_path
    finally:
        if backup_content is None and config_path.exists():
            config_path.unlink(missing_ok=True)


def load_tasks_from_csv(
    csv_path: Path,
    *,
    bench_filter: Optional[List[str]] = None,
    instance_filter: Optional[List[str]] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Load task list from data/selected_500_instances.csv or similar CSV with bench column."""
    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = (row.get("bench") or "").strip()
            instance_id = (row.get("instance_id") or "").strip()
            original_inst_id = (row.get("original_inst_id") or "").strip()

            if not bench:
                bench = detect_bench_from_instance_id(instance_id, original_inst_id)
            bench = BENCH_ALIASES.get(bench.lower(), bench) if bench else "Verified"

            if bench_filter and bench not in bench_filter:
                continue
            if instance_filter:
                if instance_id not in instance_filter and original_inst_id not in instance_filter:
                    continue

            tasks.append({
                "bench": bench,
                "instance_id": instance_id,
                "original_inst_id": original_inst_id,
                "repo": row.get("repo", "").strip(),
                "commit": row.get("commit", "").strip(),
                **{k: v for k, v in row.items() if k not in ("bench", "instance_id", "original_inst_id", "repo", "commit")},
            })
            if limit > 0 and len(tasks) >= limit:
                break
    return tasks


def load_tasks_from_gold_jsonl(
    jsonl_path: Path,
    *,
    bench_filter: Optional[List[str]] = None,
    instance_filter: Optional[List[str]] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """Load task list from gold JSONL (no bench column; infer from instance_id)."""
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = str(row.get("inst_id") or row.get("instance_id") or "")
            original_inst_id = str(row.get("original_inst_id") or "")
            bench = detect_bench_from_instance_id(instance_id, original_inst_id)

            if bench_filter and bench not in bench_filter:
                continue
            if instance_filter:
                if instance_id not in instance_filter and original_inst_id not in instance_filter:
                    continue

            tasks.append({
                "bench": bench,
                "instance_id": instance_id,
                "original_inst_id": original_inst_id,
                "repo": str(row.get("repo", "")),
                "commit": str(row.get("commit") or row.get("base_commit", "")),
                **{k: v for k, v in row.items() if k not in ("inst_id", "instance_id", "original_inst_id", "repo", "commit", "base_commit")},
            })
            if limit > 0 and len(tasks) >= limit:
                break
    return tasks


# ---------------------------------------------------------------------------
# Agent runners (bench-adapted dispatch)
# ---------------------------------------------------------------------------

def _run_agentless_unified(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    """Run Agentless via unified run_bench.py (post-refactor). Output: results/{bench}/{idx}_{instance_id}/"""
    run_script = AGENT_FRAMEWORKS / "agentless" / "run_bench.py"
    if not run_script.exists():
        return False, f"Agent script not found: {run_script}"
    bench = task.get("bench", "Verified")
    instance_id = task.get("instance_id") or task.get("original_inst_id", "")
    orig_id = task.get("original_inst_id") or instance_id
    lookup_id = orig_id if orig_id else instance_id
    cwd = AGENT_FRAMEWORKS / "agentless"
    cmd = [sys.executable, str(run_script), bench, "--instance", lookup_id, "--limit", "1"]
    try:
        result = _run_subprocess(cmd, cwd=str(cwd), timeout=timeout, debug=_DEBUG)
        if result.returncode == 0:
            results_base = cwd / "results" / bench
            out_traj = output_dir / "agentless" / bench
            out_traj.mkdir(parents=True, exist_ok=True)
            if results_base.exists():
                for folder in list(results_base.glob(f"*{lookup_id}*")) + (list(results_base.glob(f"*{instance_id}*")) if lookup_id != instance_id else []):
                    if folder.is_dir():
                        _sync_dir(folder, out_traj)
            return True, f"traj in {results_base}"
        return False, result.stderr or result.stdout or f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def run_agentless_verified(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    return _run_agentless_unified(task, output_dir, timeout)


def run_agentless_pro(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    return _run_agentless_unified(task, output_dir, timeout)


def run_agentless_poly(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    return _run_agentless_unified(task, output_dir, timeout)


def run_agentless_multi(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    return _run_agentless_unified(task, output_dir, timeout)


def run_miniswe(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    """Run MiniSWE-agent; subset determined by bench.
    Entry point: mini-swe-agent/src/minisweagent/run/extra/swebench_context_aware.py
    """
    miniswe_src = AGENT_FRAMEWORKS / "mini-swe-agent" / "multi-poly-pro-verified" / "mini-swe-agent" / "src"
    entry_module = miniswe_src / "minisweagent" / "run" / "extra" / "swebench_context_aware.py"
    if not entry_module.exists():
        return False, f"Agent entry not found: {entry_module}"
    bench = task.get("bench", "Verified")
    instance_id = task.get("instance_id") or task.get("original_inst_id", "")
    orig_id = task.get("original_inst_id", "")
    # Escape regex special chars for --filter
    filter_id = re.escape(instance_id)
    if orig_id and orig_id != instance_id:
        filter_id = f"({re.escape(instance_id)}|{re.escape(orig_id)})"

    subset_map = {
        "Verified": "verified",
        "Pro": "pro",
        "Poly": "AmazonScience/SWE-PolyBench",
        "Multi": "multi-swe-bench",
    }
    subset = subset_map.get(bench, "verified")
    out_subdir = (output_dir / "miniswe" / bench).resolve()
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Select MiniSWE config by bench:
    # - Verified/Pro/Poly → swebench_following_context.yaml
    # - Multi → swebench_multi.yaml
    miniswe_root = AGENT_FRAMEWORKS / "mini-swe-agent" / "multi-poly-pro-verified"
    if bench == "Multi":
        config_name = "swebench_multi.yaml"
    else:
        config_name = "swebench_following_context.yaml"
    config_file = miniswe_root / "configs" / config_name
    if not config_file.exists():
        return False, f"Config file not found: {config_file}"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{miniswe_src}:{env.get('PYTHONPATH', '')}"
    # Run via python -m minisweagent.run.extra.swebench_context_aware
    # Note: using split='test' as that's the correct split name for SWE-bench datasets
    cmd = [
        sys.executable, "-m", "minisweagent.run.extra.swebench_context_aware",
        "--subset", subset,
        "--split", "test",
        "--config", str(config_file),
        "--filter", f"^{filter_id}$",
        "--output", str(out_subdir),
    ]
    try:
        result = _run_subprocess(cmd, cwd=str(miniswe_src), env=env, timeout=timeout, debug=_DEBUG)
        if result.returncode == 0:
            return True, f"traj in {out_subdir}"
        return False, result.stderr or result.stdout or f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def run_sweagent(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    """Run SWE-agent; subset and config by bench.
    Entry: swe-agent/{bench}/sweagent/run/run_batch.py via sweagent run-batch
    """
    bench = task.get("bench", "Verified")
    instance_id = task.get("instance_id") or task.get("original_inst_id", "")
    orig_id = task.get("original_inst_id", "")
    filter_id = re.escape(instance_id)
    if orig_id and orig_id != instance_id:
        filter_id = f"({re.escape(instance_id)}|{re.escape(orig_id)})"

    bench_to_dir = {
        "Verified": AGENT_FRAMEWORKS / "swe-agent" / "verified",
        "Pro": AGENT_FRAMEWORKS / "swe-agent" / "pro",
        "Poly": AGENT_FRAMEWORKS / "swe-agent" / "poly",
        "Multi": AGENT_FRAMEWORKS / "swe-agent" / "multi",
    }
    run_dir = bench_to_dir.get(bench)
    if not run_dir or not run_dir.exists():
        return False, f"SWE-agent bench dir not found: {run_dir}"

    subset_map = {
        "Verified": "verified",
        "Pro": "pro",
        "Poly": "verified",  # path_override used for dataset
        "Multi": "verified",  # unused for Multi (handled separately)
    }
    path_override_map = {
        "Poly": "AmazonScience/SWE-PolyBench",
        "Multi": "ByteDance-Seed/Multi-SWE-bench",
    }
    subset = subset_map.get(bench, "verified")
    path_override = path_override_map.get(bench)
    config_path = run_dir / "config" / "azure_gpt5.yaml"
    if not config_path.exists():
        config_path = next((run_dir / "config").glob("*.yaml"), None)
    config_str = str(config_path) if config_path and config_path.exists() else ""
    if not config_str:
        config_str = os.environ.get("SWEAGENT_CONFIG", "")
    if not config_str:
        return False, "No SWE-agent config found. Set SWEAGENT_CONFIG or add config/*.yaml"

    out_subdir = output_dir / "sweagent" / bench
    out_subdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{run_dir}:{env.get('PYTHONPATH', '')}"

    if bench == "Multi":
        record = _find_multi_sweagent_record(run_dir, orig_id or instance_id)
        if not record:
            return False, f"Multi record not found for {orig_id or instance_id}"
        temp_dir = out_subdir / "data"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_jsonl = temp_dir / f"{orig_id or instance_id}.jsonl"
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        model_name = os.environ.get("SWEAGENT_MODEL") or os.environ.get("LLM_MODEL") or "openai/gpt-5"
        run_script = run_dir / "run_single_test.py"
        cmd = [
            sys.executable, str(run_script),
            "--model_name", model_name,
            "--data_file", str(temp_jsonl),
            "--config_file", config_str,
        ]
    else:
        cmd = [
            sys.executable, "-m", "sweagent.run.run_batch",
            "--instances.type", "swe_bench",
            "--instances.subset", subset,
            "--instances.split", "train" if bench == "Multi" else "test",
            "--instances.filter", f"^{filter_id}$",
            "--config", config_str,
            "--output_dir", str(out_subdir),
            "--num_workers", "1",
        ]
        if path_override:
            cmd.extend(["--instances.path_override", path_override])
    try:
        result = _run_subprocess(cmd, cwd=str(run_dir), env=env, timeout=timeout, debug=_DEBUG)
        if result.returncode == 0:
            return True, f"traj in {out_subdir}"
        return False, result.stderr or result.stdout or f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def run_prometheus(task: Dict[str, Any], output_dir: Path, timeout: int = 3600) -> Tuple[bool, str]:
    """Run Prometheus via agent-frameworks/prometheus/run_bench.py (HTTP API to Prometheus service)."""
    run_script = AGENT_FRAMEWORKS / "prometheus" / "run_bench.py"
    if not run_script.exists():
        return False, f"Agent script not found: {run_script}"
    bench = task.get("bench", "Verified")
    instance_id = task.get("instance_id") or task.get("original_inst_id", "")
    orig_id = task.get("original_inst_id") or instance_id
    lookup_id = orig_id if orig_id else instance_id
    cwd = AGENT_FRAMEWORKS / "prometheus"
    out_traj = output_dir / "prometheus" / bench
    out_traj.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(run_script),
        bench,
        "--instance",
        lookup_id,
        "--limit",
        "1",
        "--output",
        str(out_traj.parent),
    ]
    prometheus_url = os.environ.get("PROMETHEUS_URL")
    if prometheus_url:
        cmd.extend(["--prometheus-url", prometheus_url])
    prom_timeout = os.environ.get("PROMETHEUS_TIMEOUT")
    if prom_timeout:
        cmd.extend(["--timeout", prom_timeout])
    elif timeout != 3600:
        cmd.extend(["--timeout", str(max(timeout, 3600))])
    try:
        result = _run_subprocess(cmd, cwd=str(cwd), timeout=max(timeout, 3600), debug=_DEBUG)
        if result.returncode == 0:
            return True, f"traj in {out_traj}"
        return False, result.stderr or result.stdout or f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


def run_openhands(task: Dict[str, Any], output_dir: Path, timeout: int = 1800) -> Tuple[bool, str]:
    """Run OpenHands; dispatches to bench-adapted run_infer.
    Entry: openhands/{verified|poly-pro|multi}/evaluation/benchmarks/swe_bench/run_infer.py
    """
    bench = task.get("bench", "Verified")
    instance_id = task.get("instance_id") or task.get("original_inst_id", "")
    orig_id = task.get("original_inst_id", "")

    if bench == "Verified":
        run_dir = AGENT_FRAMEWORKS / "openhands" / "verified"
        dataset = "princeton-nlp/SWE-bench_Verified"
    elif bench in ("Pro", "Poly"):
        run_dir = AGENT_FRAMEWORKS / "openhands" / "poly-pro"
        dataset = "AmazonScience/SWE-PolyBench" if bench == "Poly" else "ScaleAI/SWE-bench_Pro"
    elif bench == "Multi":
        run_dir = AGENT_FRAMEWORKS / "openhands" / "multi"
        dataset = "ByteDance-Seed/Multi-SWE-bench"
    else:
        return False, f"OpenHands: unsupported bench {bench}"

    run_dir = _resolve_openhands_run_dir(run_dir, bench)

    run_script = run_dir / "evaluation" / "benchmarks" / "swe_bench" / "scripts" / "run_infer.sh"
    if bench == "Multi":
        run_script = run_dir / "evaluation" / "benchmarks" / "multi_swe_bench" / "scripts" / "run_infer.sh"
    if not run_script.exists():
        run_script = run_dir / "evaluation" / "benchmarks" / "swe_bench" / "scripts" / "run_infer.sh"
    if not run_script.exists():
        return False, f"OpenHands run script not found in {run_dir}"

    model_config = os.environ.get("OPENHANDS_MODEL_CONFIG", "llm.eval_gpt5")
    agent_cls = os.environ.get("OPENHANDS_AGENT", "CodeActAgent")
    commit_hash = os.environ.get("OPENHANDS_COMMIT", "main")
    split = "train" if bench == "Multi" else "test"

    out_subdir = output_dir / "openhands" / bench
    out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(run_script),
        model_config,
        commit_hash,
        agent_cls,
        "1",   # EVAL_LIMIT: run 1 instance
        "200", # MAX_ITER
        "1",   # NUM_WORKERS
        dataset,
        split,
        "1",   # N_RUNS
        "swe", # MODE
    ]
    selected_ids: List[str] = []
    for _id in (instance_id, orig_id):
        if _id and _id not in selected_ids:
            selected_ids.append(_id)
    env = os.environ.copy()
    if bench == "Multi":
        local_jsonl = AGENT_FRAMEWORKS / "openhands" / "multi" / "temp_multi_subset.jsonl"
        if local_jsonl.exists():
            env["OPENHANDS_DATASET_JSONL"] = str(local_jsonl)
            try:
                jsonl_records = []
                with open(local_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            jsonl_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                inst_ids = {r.get("instance_id") for r in jsonl_records if r.get("instance_id")}
                if inst_ids and not any(_id in inst_ids for _id in selected_ids):
                    repo = _infer_repo_from_instance_id(orig_id) or _infer_repo_from_instance_id(instance_id)
                    if repo:
                        candidates = [r.get("instance_id") for r in jsonl_records if r.get("repo") == repo and r.get("instance_id")]
                        if len(candidates) == 1:
                            selected_ids = [candidates[0]]
                            if _DEBUG:
                                print(f"[debug] mapped to instance_id '{candidates[0]}' via repo '{repo}'")
            except Exception:
                pass
    try:
        with _openhands_temp_config(run_dir):
            if not _openhands_has_model_config(run_dir, model_config):
                fallback = "llm"
                if _openhands_has_model_config(run_dir, fallback):
                    if _DEBUG:
                        print(f"[debug] model_config '{model_config}' not found; falling back to '{fallback}'")
                    model_config = fallback
                    cmd[2] = model_config
            with _openhands_selected_ids(run_dir, bench, selected_ids):
                result = _run_subprocess(cmd, cwd=str(run_dir), env=env, timeout=timeout, debug=_DEBUG)
        if result.returncode == 0:
            eval_out = run_dir / "evaluation" / "evaluation_outputs"
            out_traj = output_dir / "openhands" / bench
            _sync_dir(eval_out, out_traj)
            return True, f"traj in {eval_out} (instance {instance_id})"
        return False, result.stderr or result.stdout or f"exit {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)


# Agent -> bench -> runner
AGENT_RUNNERS: Dict[str, Dict[str, Any]] = {
    "agentless": {
        "Verified": run_agentless_verified,
        "Pro": run_agentless_pro,
        "Poly": run_agentless_poly,
        "Multi": run_agentless_multi,
    },
    "miniswe": {
        "Verified": run_miniswe,
        "Pro": run_miniswe,
        "Poly": run_miniswe,
        "Multi": run_miniswe,
    },
    "sweagent": {
        "Verified": run_sweagent,
        "Pro": run_sweagent,
        "Poly": run_sweagent,
        "Multi": run_sweagent,
    },
    "openhands": {
        "Verified": run_openhands,
        "Pro": run_openhands,
        "Poly": run_openhands,
        "Multi": run_openhands,
    },
    "prometheus": {
        "Verified": run_prometheus,
        "Pro": run_prometheus,
        "Poly": run_prometheus,
        "Multi": run_prometheus,
    },
}


def run_instance(
    agent: str,
    task: Dict[str, Any],
    output_dir: Path,
    timeout: int = 1800,
) -> Tuple[bool, str]:
    """Dispatch to bench-adapted agent runner."""
    bench = task.get("bench", "Verified")
    runners = AGENT_RUNNERS.get(agent)
    if not runners:
        return False, f"Unknown agent: {agent}. Available: {list(AGENT_RUNNERS.keys())}"
    runner_fn = runners.get(bench)
    if not runner_fn:
        return False, f"Agent '{agent}' has no runner for bench '{bench}'"
    return runner_fn(task, output_dir, timeout=timeout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run agents on ContextBench with bench-adapted dispatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--agent",
        required=True,
        choices=list(AGENT_RUNNERS.keys()),
        help="Agent from agent-frameworks to use",
    )
    ap.add_argument(
        "--bench",
        type=str,
        default=None,
        help="Filter by bench: Verified, Pro, Poly, Multi (comma-separated for multiple)",
    )
    ap.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Comma-separated instance_id or original_inst_id to run only those",
    )
    ap.add_argument(
        "--task-csv",
        type=Path,
        default=DEFAULT_TASK_CSV,
        help=f"Task list CSV (default: {DEFAULT_TASK_CSV.name})",
    )
    ap.add_argument(
        "--subset-csv",
        type=Path,
        default=None,
        help="Use this CSV instead of --task-csv (alias for --task-csv)",
    )
    ap.add_argument(
        "--gold-jsonl",
        type=Path,
        default=None,
        help="Use gold JSONL instead of CSV (bench inferred from instance_id)",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "results" / "agent_runs",
        help="Output directory for trajectories",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N instances (0 = no limit)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per instance in seconds",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print task list, do not run",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print live debug output from agent commands",
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerun by clearing existing outputs for matched instances",
    )
    # SWE-agent options
    ap.add_argument(
        "--sweagent-config",
        type=str,
        default=None,
        help="SWE-agent config YAML path (or set SWEAGENT_CONFIG)",
    )
    # OpenHands options
    ap.add_argument(
        "--openhands-model-config",
        type=str,
        default=None,
        help="OpenHands LLM config name (or set OPENHANDS_MODEL_CONFIG)",
    )
    ap.add_argument(
        "--openhands-agent",
        type=str,
        default=None,
        help="OpenHands agent class (or set OPENHANDS_AGENT)",
    )
    ap.add_argument(
        "--prometheus-url",
        type=str,
        default=None,
        help="Prometheus API base URL (or set PROMETHEUS_URL)",
    )
    args = ap.parse_args()
    global _DEBUG
    _DEBUG = bool(args.debug)

    # Resolve task source
    task_source = args.subset_csv or args.task_csv
    use_gold = args.gold_jsonl is not None

    bench_filter = None
    if args.bench:
        bench_filter = [BENCH_ALIASES.get(b.strip().lower(), b.strip()) for b in args.bench.split(",")]
        bench_filter = [b for b in bench_filter if b]

    instance_filter = None
    if args.instances:
        instance_filter = [s.strip() for s in args.instances.split(",") if s.strip()]

    # Load tasks
    if use_gold:
        if not args.gold_jsonl.exists():
            print(f"ERROR: Gold JSONL not found: {args.gold_jsonl}", file=sys.stderr)
            return 2
        tasks = load_tasks_from_gold_jsonl(
            args.gold_jsonl,
            bench_filter=bench_filter,
            instance_filter=instance_filter,
            limit=args.limit,
        )
    else:
        if not task_source.exists():
            print(f"ERROR: Task CSV not found: {task_source}", file=sys.stderr)
            return 2
        tasks = load_tasks_from_csv(
            task_source,
            bench_filter=bench_filter,
            instance_filter=instance_filter,
            limit=args.limit,
        )

    if not tasks:
        print("No tasks matched filters.", file=sys.stderr)
        return 0

    print(f"Loaded {len(tasks)} tasks (agent={args.agent})", file=sys.stderr)
    for i, t in enumerate(tasks[:10]):
        print(f"  [{i+1}] {t['bench']} | {t.get('instance_id') or t.get('original_inst_id')}", file=sys.stderr)
    if len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more", file=sys.stderr)

    if args.dry_run:
        return 0

    # Apply agent-specific env overrides from CLI
    if args.sweagent_config:
        os.environ["SWEAGENT_CONFIG"] = args.sweagent_config
    if args.openhands_model_config:
        os.environ["OPENHANDS_MODEL_CONFIG"] = args.openhands_model_config
    if args.openhands_agent:
        os.environ["OPENHANDS_AGENT"] = args.openhands_agent
    if args.prometheus_url:
        os.environ["PROMETHEUS_URL"] = args.prometheus_url
    # Read LLM settings from environment and propagate common aliases
    llm_url = os.environ.get("LLM_API_URL")
    llm_key = os.environ.get("LLM_API_KEY")
    if llm_url:
        if not os.environ.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = llm_url
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = llm_url
    if llm_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = llm_key

    args.output.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    for i, task in enumerate(tasks):
        inst = task.get("instance_id") or task.get("original_inst_id", "?")
        print(f"[{i+1}/{len(tasks)}] {args.agent} | {task['bench']} | {inst} ...", flush=True)
        if args.rerun:
            _clear_previous_outputs(
                args.agent,
                task.get("bench", "Verified"),
                args.output,
                [task.get("instance_id", ""), task.get("original_inst_id", "")],
            )
        ok, msg = run_instance(args.agent, task, args.output, timeout=args.timeout)
        if ok:
            success += 1
            print(f"  ✓ {msg}", flush=True)
        else:
            failed += 1
            print(f"  ✗ {msg}", flush=True)

    print(f"\nDone: {success} succeeded, {failed} failed", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
