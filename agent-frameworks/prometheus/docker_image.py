"""Docker image URI resolution for SWE-bench family datasets (Prometheus UserDefinedContainer)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def get_swebench_verified_image(instance_id: str) -> str:
    """SWE-bench Verified / Lite eval image on Docker Hub."""
    iid = instance_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()


def get_swebench_pro_image(instance_id: str, repo_name: str = "") -> str:
    if not repo_name or "/" not in repo_name:
        return ""
    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    hsh = instance_id.replace("instance_", "")
    if instance_id == (
        "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
    ):
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    return f"jefzda/sweap-images:{tag}"


def get_multiswe_image(instance_id: str, repo_name: str = "", pr_number: str = "") -> str:
    if not repo_name or "/" not in repo_name:
        return ""
    org, repo = repo_name.lower().split("/", 1)
    org_clean = org.replace(".", "_")
    repo_clean = repo.replace(".", "_")
    if not pr_number:
        if "__" in instance_id and "-" in instance_id.split("__")[-1]:
            pr_number = instance_id.split("__")[-1].split("-")[-1]
        elif "_pr" in instance_id:
            pr_number = instance_id.split("_pr")[-1]
        elif "-" in instance_id:
            pr_number = instance_id.split("-")[-1]
    if not pr_number or not str(pr_number).isdigit():
        return ""
    return f"mswebench/{org_clean}_m_{repo_clean}:pr-{pr_number}"


def get_polybench_image(instance_id: str) -> str:
    return f"ghcr.io/timesler/swe-polybench.eval.x86_64.{instance_id}:latest"


def resolve_docker_image(bench_name: str, record: dict) -> Tuple[str, str]:
    """
    Return (image_name, workdir) for Prometheus IssueRequest.
    """
    inst_id = str(record.get("instance_id") or "")
    repo = str(record.get("repo") or "")
    number = record.get("number") or record.get("pr_number") or ""

    if bench_name == "Verified":
        lookup = str(record.get("original_inst_id") or inst_id)
        return get_swebench_verified_image(lookup), "/testbed"

    if bench_name == "Pro":
        pro_id = inst_id if inst_id.startswith("instance_") else str(record.get("instance_id") or "")
        if not pro_id.startswith("instance_"):
            pro_id = f"instance_{pro_id}"
        image = get_swebench_pro_image(pro_id, repo)
        if not image:
            raise ValueError(f"Cannot resolve Pro docker image for {inst_id}")
        return image, "/app"

    if bench_name == "Poly":
        poly_id = inst_id
        if not poly_id and record.get("original_inst_id"):
            poly_id = str(record["original_inst_id"])
        return get_polybench_image(poly_id), "/testbed"

    if bench_name == "Multi":
        image = get_multiswe_image(inst_id, repo, str(number))
        if not image:
            raise ValueError(f"Cannot resolve Multi docker image for {inst_id} repo={repo}")
        repo_part = repo.split("/", 1)[1] if "/" in repo else repo
        workdir = f"/home/{repo_part.replace('-', '_')}"
        return image, workdir

    raise ValueError(f"Unknown bench: {bench_name}")
