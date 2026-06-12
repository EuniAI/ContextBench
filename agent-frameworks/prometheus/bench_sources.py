"""Load SWE-bench instance records for Prometheus runs (ContextBench data layout)."""

from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ContextBench repo root (parent of agent-frameworks/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = REPO_ROOT / "data"

csv.field_size_limit(10 * 1024 * 1024)


def _infer_repo_from_instance_id(inst_id: str) -> Optional[str]:
    if not inst_id or "__" not in inst_id:
        return None
    m = re.match(r"^(?P<org>[^_]+)__(?P<repo>[^-]+)-\d+$", inst_id.strip())
    if not m:
        return None
    return f"{m.group('org')}/{m.group('repo')}"


def _index_records(records: List[dict]) -> Dict[str, dict]:
    """Index by instance_id and original_inst_id."""
    out: Dict[str, dict] = {}
    for rec in records:
        for key in ("instance_id", "original_inst_id", "inst_id"):
            val = rec.get(key)
            if val and str(val).strip():
                out[str(val).strip()] = rec
    return out


def _load_parquet_records(path: Path, target_set: Set[str]) -> Dict[str, dict]:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required to read parquet. Install: pip install pandas pyarrow")

    df = pd.read_parquet(path)
    if "original_inst_id" in df.columns:
        mask = df["instance_id"].isin(target_set) | df["original_inst_id"].isin(target_set)
    else:
        mask = df["instance_id"].isin(target_set)
    filtered = df[mask]
    records = []
    for _, row in filtered.iterrows():
        records.append(row.to_dict())
    return _index_records(records)


def _load_hf_dataset(dataset_name: str, split: str, target_set: Set[str]) -> Dict[str, dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets is required for this bench. Install: pip install datasets")

    ds = load_dataset(dataset_name, split=split)
    records: List[dict] = []
    for row in ds:
        rec = dict(row)
        inst = str(rec.get("instance_id") or "")
        orig = str(rec.get("original_inst_id") or "")
        # Multi-SWE-bench style
        if not inst and rec.get("org") and rec.get("repo") and rec.get("number") is not None:
            inst = f"{rec['org']}__{rec['repo']}-{rec['number']}"
            rec["instance_id"] = inst
        if inst in target_set or orig in target_set:
            records.append(rec)
        if len(records) >= len(target_set):
            break
    return _index_records(records)


def _fetch_github_pr_text(repo: str, pr_number: int) -> str:
    """Fetch PR title + body from GitHub (public repos)."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ""
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def _load_multi_from_csv(target_set: Set[str]) -> Dict[str, dict]:
    csv_path = DATA_DIR / "Multi.csv"
    if not csv_path.exists():
        csv_path = REPO_ROOT / "data" / "Multi.csv"
    records: List[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = (row.get("original_inst_id") or "").strip()
            inst = (row.get("instance_id") or "").strip()
            if orig not in target_set and inst not in target_set:
                continue
            lookup = orig if orig in target_set else inst
            m = re.match(r"^(?P<org>[^_]+)__(?P<repo>[^-]+)-(?P<num>\d+)$", lookup)
            if not m:
                continue
            org, repo_name, num = m.group("org"), m.group("repo"), int(m.group("num"))
            full_repo = f"{org}/{repo_name}"
            base_commit = (row.get("commit") or "").strip()
            problem = _fetch_github_pr_text(full_repo, num)
            if not problem:
                problem = f"Fix issue for {lookup} (Multi-SWE-bench PR #{num})"
            rec = {
                "instance_id": lookup,
                "original_inst_id": lookup,
                "org": org,
                "repo": full_repo,
                "number": num,
                "base_commit": base_commit,
                "base": {"sha": base_commit},
                "problem_statement": problem,
                "language": (row.get("language") or "").strip(),
            }
            records.append(rec)
    return _index_records(records)


def load_task_ids(bench_name: str, limit: Optional[int] = None) -> List[str]:
    csv_path = DATA_DIR / f"{bench_name}.csv"
    if not csv_path.exists():
        csv_path = REPO_ROOT / "data" / f"{bench_name}.csv"
    tasks: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_inst_id = (row.get("original_inst_id") or "").strip()
            instance_id = (row.get("instance_id") or "").strip()
            tid = original_inst_id or instance_id
            if not tid:
                continue
            tasks.append(tid)
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def fetch_records(bench_name: str, target_ids: List[str]) -> Dict[str, dict]:
    target_set = set(target_ids)
    indexed: Dict[str, dict] = {}

    if bench_name == "Verified":
        candidates = [
            DATA_DIR / "contextbench_verified.parquet",
            REPO_ROOT / "data" / "contextbench_verified.parquet",
            DATA_DIR / "train.parquet",
            REPO_ROOT / "data" / "train.parquet",
        ]
        for path in candidates:
            if path.exists():
                indexed = _load_parquet_records(path, target_set)
                break
        if not indexed:
            indexed = _load_hf_dataset("princeton-nlp/SWE-Bench_Verified", "test", target_set)

    elif bench_name == "Pro":
        indexed = _load_hf_dataset("ScaleAI/SWE-bench_Pro", "test", target_set)

    elif bench_name == "Poly":
        indexed = _load_hf_dataset("AmazonScience/SWE-PolyBench", "test", target_set)

    elif bench_name == "Multi":
        indexed = _load_multi_from_csv(target_set)
        if not indexed:
            try:
                indexed = _load_hf_dataset("bytedance-research/Multi-SWE-Bench", "train", target_set)
            except Exception:
                indexed = {}

    else:
        raise ValueError(f"Unknown bench_name: {bench_name}")

    # Return keyed by requested lookup ids
    out: Dict[str, dict] = {}
    for tid in target_ids:
        if tid in indexed:
            out[tid] = indexed[tid]
            continue
        for rec in indexed.values():
            if str(rec.get("instance_id") or "") == tid or str(rec.get("original_inst_id") or "") == tid:
                out[tid] = rec
                break
    return out


def split_problem_statement(problem_statement: str) -> tuple[str, str]:
    if not problem_statement:
        return "SWE-bench issue", ""
    parts = problem_statement.splitlines()
    title = (parts[0].strip() if parts else "") or "SWE-bench issue"
    body = "\n".join(parts[1:]).strip() if len(parts) > 1 else problem_statement
    return title, body


def normalize_record(record: dict, lookup_id: str) -> dict:
    """Ensure repo, base_commit, problem_statement, and canonical instance ids."""
    rec = dict(record)
    inst_id = str(rec.get("instance_id") or lookup_id)
    rec["instance_id"] = inst_id
    if not rec.get("original_inst_id"):
        if lookup_id != inst_id and "__" in lookup_id and "-" in lookup_id:
            rec["original_inst_id"] = lookup_id
        else:
            rec["original_inst_id"] = lookup_id

    repo = rec.get("repo") or ""
    if not repo:
        inferred = _infer_repo_from_instance_id(str(rec.get("original_inst_id") or lookup_id))
        if inferred:
            repo = inferred
            rec["repo"] = repo

    base_commit = rec.get("base_commit") or rec.get("commit") or ""
    if not base_commit and isinstance(rec.get("base"), dict):
        base_commit = rec["base"].get("sha") or ""
    rec["base_commit"] = base_commit

    if not rec.get("problem_statement"):
        rec["problem_statement"] = rec.get("issue_body") or rec.get("text") or ""

    if not rec.get("number") and rec.get("original_inst_id"):
        m = re.search(r"-(\d+)$", str(rec["original_inst_id"]))
        if m:
            rec["number"] = int(m.group(1))

    return rec
