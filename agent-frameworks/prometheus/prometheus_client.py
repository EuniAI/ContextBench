"""HTTP client for Prometheus API (repository upload + issue answer)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPOSITORY_UPLOAD = "/repository/upload/"
ISSUE_ANSWER = "/issue/answer/"


class PrometheusClient:
    def __init__(self, base_url: str, timeout: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        token = os.environ.get("PROMETHEUS_JWT_TOKEN") or os.environ.get("PROMETHEUS_API_TOKEN")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def health_check(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url.rsplit('/', 2)[0]}/docs", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def upload_repository(
        self,
        repo: str,
        commit_id: Optional[str] = None,
        github_token: Optional[str] = None,
    ) -> int:
        https_url = f"https://github.com/{repo}.git"
        payload: Dict[str, Any] = {"https_url": https_url}
        if commit_id:
            payload["commit_id"] = commit_id
        if github_token:
            payload["github_token"] = github_token
        url = self.base_url + REPOSITORY_UPLOAD
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Repository upload failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        repository_id = data.get("data", {}).get("repository_id")
        if repository_id is None:
            raise RuntimeError(f"Unexpected upload response: {resp.text}")
        return int(repository_id)

    def answer_issue(
        self,
        repository_id: int,
        issue_title: str,
        issue_body: str,
        *,
        image_name: str,
        workdir: str,
        issue_type: str = "bug",
        run_reproduce_test: bool = True,
        run_regression_test: bool = True,
        run_existing_test: bool = False,
        run_build: bool = False,
        number_of_candidate_patch: int = 3,
        issue_comments: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "repository_id": repository_id,
            "issue_title": issue_title,
            "issue_body": issue_body,
            "issue_comments": issue_comments or [],
            "issue_type": issue_type,
            "run_build": run_build,
            "run_existing_test": run_existing_test,
            "run_regression_test": run_regression_test,
            "run_reproduce_test": run_reproduce_test,
            "number_of_candidate_patch": number_of_candidate_patch,
            "image_name": image_name,
            "workdir": workdir,
        }
        url = self.base_url + ISSUE_ANSWER
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Issue answer failed ({resp.status_code}): {resp.text}")
        body = resp.json()
        if body.get("data") is None:
            raise RuntimeError(f"Empty issue answer data: {resp.text}")
        return body["data"]


def find_newest_log(log_dir: Path, since_ts: float) -> Optional[Path]:
    if not log_dir.is_dir():
        return None
    candidates = [
        p for p in log_dir.glob("*.log")
        if p.is_file() and p.stat().st_mtime >= since_ts - 1.0
    ]
    if not candidates:
        candidates = list(log_dir.glob("*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_log_to_output(log_path: Path, dest_log: Path) -> None:
    dest_log.parent.mkdir(parents=True, exist_ok=True)
    dest_log.write_bytes(log_path.read_bytes())


def write_result_json(dest: Path, instance_id: str, api_data: Dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance_id,
        "edit_patch": api_data.get("patch") or "",
        "passed_reproducing_test": api_data.get("passed_reproducing_test"),
        "passed_regression_test": api_data.get("passed_regression_test"),
        "passed_existing_test": api_data.get("passed_existing_test"),
        "issue_response": api_data.get("issue_response"),
        "issue_type": str(api_data.get("issue_type") or ""),
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
