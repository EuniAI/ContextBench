#!/usr/bin/env python3
"""Run SWE-bench with context-aware agent that captures patch generation context."""

import concurrent.futures
import json
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from enum import Enum
from abc import ABC, abstractmethod

from minisweagent import Environment, global_config_dir
from minisweagent.agents.context_aware import ContextAwareAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import DATASET_MAPPING, get_sb_environment
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances with context-aware agent.

Supports multiple benchmarks:
- SWE-bench (lite, verified)
- SWE-bench Pro 
- Multi-SWE-bench (multi-language)
- PolyBench

[not dim]
More information: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
DEFAULT_OUTPUT = global_config_dir / "last_swebench_context_aware_run.traj.json"


class BenchmarkType(str, Enum):
    """Benchmark type enumeration."""
    SWEBENCH_PRO = "swebench_pro"
    MULTI_SWEBENCH = "multi_swe_bench"
    POLYBENCH = "polybench"
    SWEBENCH_DEFAULT = "swebench_default"
    UNKNOWN = "unknown"


def detect_benchmark_type(instance: dict) -> BenchmarkType:
    """Detect benchmark type from instance metadata (fallback when not explicitly provided).
    
    Note: This is only used as fallback. The benchmark type should be determined
    from the --subset parameter in the main function and passed explicitly.
    
    Args:
        instance: Instance dictionary containing instance_id, dataset_name, repo, etc.
        
    Returns:
        BenchmarkType enum value
    """
    instance_id = instance.get("instance_id", "")
    dataset_name = instance.get("dataset_name", "").lower()
    
    # Check dataset_name first (most reliable)
    if dataset_name:
        if "multi-swe-bench" in dataset_name or "multi_swe_bench" in dataset_name or "bytedance" in dataset_name:
            return BenchmarkType.MULTI_SWEBENCH
        if "polybench" in dataset_name or "swe-polybench" in dataset_name or "amazonscience" in dataset_name:
            return BenchmarkType.POLYBENCH
        if "pro" in dataset_name and "swe" in dataset_name:
            return BenchmarkType.SWEBENCH_PRO
        if "swebench" in dataset_name or "swe-bench" in dataset_name:
            return BenchmarkType.SWEBENCH_DEFAULT
    
    # Fallback: Check instance_id format patterns (simple heuristics)
    # SWE-bench Pro: instance_* with long identifier
    if instance_id.startswith("instance_") and len(instance_id) > 50:
        return BenchmarkType.SWEBENCH_PRO
    
    # Standard SWE-bench format: {org}__{repo}-{number}
    if "__" in instance_id and "-" in instance_id:
        return BenchmarkType.SWEBENCH_DEFAULT
    
    # Unknown format
    logger.warning(f"Could not determine benchmark type for instance {instance_id}, defaulting to SWEBENCH_DEFAULT")
    return BenchmarkType.SWEBENCH_DEFAULT


class DockerStrategy(ABC):
    """Abstract base class for Docker configuration strategies."""
    
    @abstractmethod
    def get_docker_config(self, instance: dict, auto_pull: bool = True) -> dict:
        """Get Docker configuration for an instance.
        
        Args:
            instance: Instance dictionary
            auto_pull: Whether to automatically pull images if needed
            
        Returns:
            Dictionary with keys: base_image, dockerfile_content, source, cwd, timeout, pull_timeout, run_args
        """
        pass
    
    @abstractmethod
    def get_environment_config(self, instance: dict, docker_config: dict) -> dict:
        """Get environment configuration for an instance.
        
        Args:
            instance: Instance dictionary
            docker_config: Docker configuration from get_docker_config
            
        Returns:
            Environment configuration dictionary
        """
        pass


class SWEBenchProStrategy(DockerStrategy):
    """Strategy for SWE-bench Pro instances."""
    
    def get_docker_config(self, instance: dict, auto_pull: bool = True) -> dict:
        instance_id = instance.get("instance_id", "")
        repo_name = instance.get("repo", "")
        
        image_uri = DockerConfigExtractor.get_dockerhub_image_uri(instance_id, repo_name)
        
        if not image_uri:
            raise RuntimeError(f"Failed to generate SWE-bench Pro image URI for {instance_id}")
        
        if auto_pull and not DockerConfigExtractor.pull_image_if_needed(image_uri, pull_timeout=600):
            raise RuntimeError(f"Failed to pull SWE-bench Pro image: {image_uri}")
        
        # Detect working directory
        cwd = _detect_docker_image_workdir(image_uri) or "/app"
        
        return {
            "base_image": image_uri,
            "dockerfile_content": None,
            "source": "swebench_pro",
            "cwd": cwd,
            "timeout": 120,  # 2 minutes for commands
            "pull_timeout": 600,  # 10 minutes for large images
            "run_args": ["--rm", "--platform", "linux/amd64"],
        }
    
    def get_environment_config(self, instance: dict, docker_config: dict) -> dict:
        return {
            "image": docker_config["base_image"],
            "cwd": docker_config.get("cwd", "/app"),
            "timeout": docker_config.get("timeout", 120),
            "pull_timeout": docker_config.get("pull_timeout", 600),
            "run_args": docker_config.get("run_args", ["--rm", "--platform", "linux/amd64"]),
        }


class MultiSWEBenchStrategy(DockerStrategy):
    """Strategy for Multi-SWE-bench instances."""
    
    def get_docker_config(self, instance: dict, auto_pull: bool = True) -> dict:
        instance_id = instance.get("instance_id", "")
        repo_name = instance.get("repo", "")
        pr_number = instance.get("number", "") or instance.get("pr_number", "")
        
        image_uri = DockerConfigExtractor.get_multiswe_image_uri(instance_id, repo_name, pr_number)
        
        if not image_uri:
            raise RuntimeError(f"Failed to generate Multi-SWE-bench image URI for {instance_id}")
        
        # Multi-SWE-bench images can be large, use longer timeout
        if auto_pull and not DockerConfigExtractor.pull_image_if_needed(image_uri, pull_timeout=600):
            raise RuntimeError(f"Failed to pull Multi-SWE-bench image: {image_uri}")
        
        # Detect working directory
        cwd = _detect_docker_image_workdir(image_uri)
        if not cwd and repo_name and "/" in repo_name:
            # Fallback: extract repo name for /home/{repo} pattern
            repo_clean = repo_name.split("/", 1)[1].replace("-", "_")
            cwd = f"/home/{repo_clean}"
        else:
            cwd = cwd or "/home"
        
        return {
            "base_image": image_uri,
            "dockerfile_content": None,
            "source": "multi_swe_bench",
            "cwd": cwd,
            "timeout": 600,  # 10 minutes for complex builds (increased for large repos like MUI)
            "pull_timeout": 600,  # 10 minutes
            "run_args": ["--rm", "--platform", "linux/amd64"],
        }
    
    def get_environment_config(self, instance: dict, docker_config: dict) -> dict:
        return {
            "image": docker_config["base_image"],
            "cwd": docker_config.get("cwd", "/home"),
            "timeout": docker_config.get("timeout", 600),  # Default 10 minutes for large repos
            "pull_timeout": docker_config.get("pull_timeout", 600),
            "run_args": docker_config.get("run_args", ["--rm", "--platform", "linux/amd64"]),
        }


class PolyBenchStrategy(DockerStrategy):
    """Strategy for PolyBench instances."""
    
    def __init__(self, poly_data_dir: Optional[Path] = None):
        self.poly_data_dir = poly_data_dir
    
    def get_docker_config(self, instance: dict, auto_pull: bool = True) -> dict:
        instance_id = instance.get("instance_id", "")
        
        # Try Priority 1: PolyBench pre-built image
        polybench_image = f"ghcr.io/timesler/swe-polybench.eval.x86_64.{instance_id}:latest"
        
        if auto_pull and DockerConfigExtractor.pull_image_if_needed(polybench_image, pull_timeout=600):
            cwd = _detect_docker_image_workdir(polybench_image) or "/testbed"
            return {
                "base_image": polybench_image,
                "dockerfile_content": None,
                "source": "polybench",
                "cwd": cwd,
                "timeout": 120,
                "pull_timeout": 600,
                "run_args": ["--rm", "--platform", "linux/amd64"],
            }
        
        # Try Priority 2: Extract Dockerfile from poly data
        if self.poly_data_dir and self.poly_data_dir.exists():
            dockerfile_content = DockerConfigExtractor.extract_dockerfile_from_poly_data(
                self.poly_data_dir, instance_id
            )
            if dockerfile_content:
                return {
                    "base_image": None,
                    "dockerfile_content": dockerfile_content,
                    "source": "poly_dataset",
                    "cwd": "/testbed",
                    "timeout": 120,
                    "pull_timeout": 600,
                    "run_args": ["--rm", "--platform", "linux/amd64"],
                }
        
        raise RuntimeError(
            f"No Docker configuration found for PolyBench instance {instance_id}. "
            f"Neither pre-built image nor Dockerfile from poly data available."
        )
    
    def get_environment_config(self, instance: dict, docker_config: dict) -> dict:
        if docker_config.get("base_image"):
            return {
                "image": docker_config["base_image"],
                "cwd": docker_config.get("cwd", "/testbed"),
                "timeout": docker_config.get("timeout", 120),
                "pull_timeout": docker_config.get("pull_timeout", 600),
                "run_args": docker_config.get("run_args", ["--rm", "--platform", "linux/amd64"]),
            }
        elif docker_config.get("dockerfile_content"):
            # Will be handled by build logic in get_sb_environment_with_docker_config
            return {
                "cwd": docker_config.get("cwd", "/testbed"),
                "timeout": docker_config.get("timeout", 120),
                "pull_timeout": docker_config.get("pull_timeout", 600),
                "run_args": docker_config.get("run_args", ["--rm", "--platform", "linux/amd64"]),
            }
        else:
            raise RuntimeError("PolyBench strategy: no base_image or dockerfile_content available")


class SWEBenchDefaultStrategy(DockerStrategy):
    """Strategy for standard SWE-bench instances (lite, verified)."""
    
    def get_docker_config(self, instance: dict, auto_pull: bool = True) -> dict:
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name
        
        image_name = get_swebench_docker_image_name(instance)
        
        if auto_pull and not DockerConfigExtractor.pull_image_if_needed(image_name, pull_timeout=600):
            raise RuntimeError(f"Failed to pull SWE-bench default image: {image_name}")
        
        cwd = _detect_docker_image_workdir(image_name) or "/testbed"
        
        return {
            "base_image": image_name,
            "dockerfile_content": None,
            "source": "swebench_default",
            "cwd": cwd,
            "timeout": 120,
            "pull_timeout": 600,
            "run_args": ["--rm", "--platform", "linux/amd64"],
        }
    
    def get_environment_config(self, instance: dict, docker_config: dict) -> dict:
        return {
            "image": docker_config["base_image"],
            "cwd": docker_config.get("cwd", "/testbed"),
            "timeout": docker_config.get("timeout", 120),
            "pull_timeout": docker_config.get("pull_timeout", 600),
            "run_args": docker_config.get("run_args", ["--rm", "--platform", "linux/amd64"]),
        }


def get_docker_strategy(
    benchmark_type: BenchmarkType, 
    poly_data_dir: Optional[Path] = None
) -> DockerStrategy:
    """Get Docker strategy for a benchmark type.
    
    Args:
        benchmark_type: The detected benchmark type
        poly_data_dir: Optional path to poly dataset directory for PolyBench
        
    Returns:
        DockerStrategy instance
    """
    strategies = {
        BenchmarkType.SWEBENCH_PRO: SWEBenchProStrategy(),
        BenchmarkType.MULTI_SWEBENCH: MultiSWEBenchStrategy(),
        BenchmarkType.POLYBENCH: PolyBenchStrategy(poly_data_dir),
        BenchmarkType.SWEBENCH_DEFAULT: SWEBenchDefaultStrategy(),
    }
    
    strategy = strategies.get(benchmark_type)
    if not strategy:
        raise ValueError(f"No strategy available for benchmark type: {benchmark_type}")
    
    return strategy


class DockerConfigExtractor:
    """Extract and generate Docker configurations from various sources."""
    
    @staticmethod
    def get_dockerhub_image_uri(instance_id: str, repo_name: str = "") -> str:
        """Generate Docker Hub image URI for SWE-bench Pro instances."""
        if not repo_name or "/" not in repo_name:
            return ""
            
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = instance_id.replace("instance_", "")
        
        # Handle special cases
        if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
            repo_name_only = 'element-web'
        elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
            repo_name_only = 'element'
            if hsh.endswith('-vnan'):
                hsh = hsh[:-5]
        elif hsh.endswith('-vnan'):
            hsh = hsh[:-5]
        
        tag = f"{repo_base}.{repo_name_only}-{hsh}"
        if len(tag) > 128:
            tag = tag[:128]
            
        return f"jefzda/sweap-images:{tag}"

    @staticmethod
    def get_multiswe_image_uri(instance_id: str, repo_name: str = "", pr_number: str = "") -> str:
        """Generate Multi-SWE-bench image URI following mswebench/* pattern."""
        if not repo_name or "/" not in repo_name:
            return ""
        
        # Multi-SWE-bench uses pattern: mswebench/{org}_m_{repo}:pr-{number}
        # Note: Hyphens in org/repo names are preserved (e.g., clap-rs, go-zero)
        org, repo = repo_name.lower().split("/", 1)
        
        # Only replace dots with underscores; preserve hyphens
        org_clean = org.replace(".", "_")
        repo_clean = repo.replace(".", "_")
        
        # Extract PR number from various formats
        if not pr_number:
            # Try to extract from instance_id patterns
            if "__" in instance_id:
                # Format: org__repo-pr_number or similar
                parts = instance_id.split("__")
                if len(parts) >= 2 and "-" in parts[-1]:
                    pr_number = parts[-1].split("-")[-1]
            elif "_pr" in instance_id:
                pr_number = instance_id.split("_pr")[-1]
            elif "-" in instance_id:
                pr_number = instance_id.split("-")[-1]
        
        if not pr_number or not pr_number.isdigit():
            return ""
            
        return f"mswebench/{org_clean}_m_{repo_clean}:pr-{pr_number}"

    @staticmethod
    def pull_image_if_needed(image_uri: str, pull_timeout: int = 300) -> bool:
        """Pull Docker image if not available locally.
        
        Args:
            image_uri: Docker image URI to pull
            pull_timeout: Timeout in seconds for pulling (default 300 = 5 minutes)
            
        Returns:
            True if image is available (either locally or after successful pull), False otherwise
        """
        if not image_uri:
            return False
            
        try:
            import docker
            client = docker.from_env(timeout=pull_timeout)
            
            # PRIORITY 1: Check if image exists locally (fastest, no network needed)
            try:
                image = client.images.get(image_uri)
                logger.info(f"✓ Using locally cached image: {image_uri}")
                # Image exists locally - return immediately without health check to save time
                return True
                    
            except docker.errors.ImageNotFound:
                logger.info(f"Image not found locally, will attempt to pull: {image_uri}")
            except Exception as local_check_err:
                logger.warning(f"Error checking local image {image_uri}: {local_check_err}")
                # Continue to try pulling
            
            # PRIORITY 2: Try to pull the image from remote registry
            try:
                logger.info(f"⬇️  Pulling Docker image: {image_uri} (timeout: {pull_timeout}s)")
                # Use longer timeout for large images
                client.images.pull(image_uri, platform="linux/amd64")
                logger.info(f"✓ Successfully pulled: {image_uri}")
                return True
            except Exception as pull_err:
                logger.error(f"✗ Failed to pull image: {image_uri}, {pull_err}")
                return False
                    
        except Exception as e:
            logger.error(f"Docker client error for {image_uri}: {e}")
            return False

    @staticmethod
    def extract_dockerfile_from_patch(model_patch: str) -> Optional[str]:
        """Extract Dockerfile content from a model patch string.
        
        Args:
            model_patch: The patch string that may contain Dockerfile changes
            
        Returns:
            Dockerfile content if found, None otherwise
        """
        if not model_patch or "Dockerfile" not in model_patch:
            return None
            
        # Find Dockerfile section in the patch
        dockerfile_match = re.search(
            r'diff --git a/Dockerfile b/Dockerfile.*?\n(.*?)(?=diff --git|\Z)', 
            model_patch, 
            re.DOTALL | re.MULTILINE
        )
        
        if not dockerfile_match:
            return None
        
        dockerfile_section = dockerfile_match.group(1)
        dockerfile_lines = []
        
        # Extract added lines (lines starting with +)
        for line in dockerfile_section.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                # Remove the + prefix and add to dockerfile content
                dockerfile_lines.append(line[1:])
        
        return '\n'.join(dockerfile_lines) if dockerfile_lines else None

    @staticmethod 
    def extract_dockerfile_from_poly_data(poly_data_dir: Path, instance_id: str) -> Optional[str]:
        """Extract Dockerfile content from poly dataset trajectory files.
        
        Args:
            poly_data_dir: Path to the poly dataset directory
            instance_id: The instance ID to look up
            
        Returns:
            Dockerfile content if found, None otherwise
        """
        try:
            # First try to read from preds.json
            preds_file = poly_data_dir / "preds.json"
            if preds_file.exists():
                preds_data = json.loads(preds_file.read_text())
                if instance_id in preds_data:
                    model_patch = preds_data[instance_id].get("model_patch", "")
                    dockerfile_content = DockerConfigExtractor.extract_dockerfile_from_patch(model_patch)
                    if dockerfile_content:
                        return dockerfile_content
            
            # Fallback to instance-specific trajectory file
            instance_dir = poly_data_dir / instance_id
            traj_file = instance_dir / f"{instance_id}.traj.json"
            if traj_file.exists():
                traj_data = json.loads(traj_file.read_text())
                # Check if there's a result with Dockerfile content
                for message in traj_data.get("messages", []):
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        if "Dockerfile" in content:
                            dockerfile_content = DockerConfigExtractor.extract_dockerfile_from_patch(content)
                            if dockerfile_content:
                                return dockerfile_content
                                
        except Exception as e:
            logger.debug(f"Error extracting Dockerfile from poly data for {instance_id}: {e}")
            
        return None

    @staticmethod
    def generate_multiswe_dockerfile(
        language: str,
        base_image: str = "",
        project_dir: str = "/home",
        custom_commands: Optional[list[str]] = None
    ) -> str:
        """Generate Multi-SWE-bench compatible Dockerfile for different languages.
        
        Args:
            language: Programming language (python, javascript, golang, rust, java, etc.)
            base_image: Base Docker image to use (auto-detected if empty)
            project_dir: Project directory path in container
            custom_commands: Additional RUN commands to include
            
        Returns:
            Generated Dockerfile content
        """
        language = language.lower()
        
        # Auto-select base image if not provided
        if not base_image:
            base_images = {
                "python": "python:3.9-slim",
                "javascript": "node:18-slim", 
                "typescript": "node:18-slim",
                "golang": "golang:1.21-alpine",
                "go": "golang:1.21-alpine",
                "rust": "rust:1.70-slim",
                "java": "openjdk:17-slim",
                "c": "gcc:latest",
                "cpp": "gcc:latest",
                "c++": "gcc:latest"
            }
            base_image = base_images.get(language, "ubuntu:20.04")
        
        # Language-specific setup commands
        setup_commands = {
            "python": [
                "RUN apt-get update && apt-get install -y git",
                "RUN pip install --upgrade pip"
            ],
            "javascript": [
                "RUN apt-get update && apt-get install -y git",
                "RUN npm install -g npm@latest"
            ],
            "typescript": [
                "RUN apt-get update && apt-get install -y git",
                "RUN npm install -g npm@latest typescript"
            ],
            "golang": [
                "RUN apk add --no-cache git"
            ],
            "go": [
                "RUN apk add --no-cache git"
            ],
            "rust": [
                "RUN apt-get update && apt-get install -y git",
                "RUN rustup component add rustfmt clippy"
            ],
            "java": [
                "RUN apt-get update && apt-get install -y git maven"
            ],
            "c": [
                "RUN apt-get update && apt-get install -y git make"
            ],
            "cpp": [
                "RUN apt-get update && apt-get install -y git make cmake"
            ]
        }
        
        dockerfile_lines = [
            f"FROM {base_image}",
            f"WORKDIR {project_dir}",
        ]
        
        # Add language-specific setup
        if language in setup_commands:
            dockerfile_lines.extend(setup_commands[language])
        
        # Add custom commands
        if custom_commands:
            dockerfile_lines.extend(custom_commands)
            
        return '\n'.join(dockerfile_lines)

    @staticmethod
    def get_docker_config_for_instance(
        instance: dict,
        poly_data_dir: Optional[Path] = None,
        auto_pull: bool = True,
        benchmark_type: Optional[BenchmarkType] = None
    ) -> dict:
        """Get Docker configuration for a SWE-bench instance using strategy pattern.
        
        Args:
            instance: Instance dictionary
            poly_data_dir: Optional path to poly dataset directory
            auto_pull: Whether to automatically pull images if needed
            benchmark_type: Optional pre-determined benchmark type (if None, will auto-detect)
            
        Returns:
            Dictionary with Docker configuration (base_image, dockerfile_content, source, etc.)
        """
        instance_id = instance.get("instance_id", "")
        
        # Use provided benchmark_type if available, otherwise detect from instance metadata
        if benchmark_type is None:
            benchmark_type = detect_benchmark_type(instance)
            logger.info(f"Detected benchmark type '{benchmark_type.value}' for instance {instance_id}")
        else:
            logger.info(f"Using provided benchmark type '{benchmark_type.value}' for instance {instance_id}")
        
        # Get appropriate strategy for this benchmark type
        try:
            strategy = get_docker_strategy(benchmark_type, poly_data_dir)
            config = strategy.get_docker_config(instance, auto_pull)
            logger.info(f"Successfully configured Docker for {instance_id} using {benchmark_type.value} strategy")
            return config
        except Exception as e:
            logger.error(f"Failed to configure Docker for {instance_id} using {benchmark_type.value} strategy: {e}")
            raise RuntimeError(
                f"Cannot configure Docker for instance {instance_id} (benchmark: {benchmark_type.value}): {e}"
            ) from e

    # Multi-SWE-bench language to repository mapping
    MULTISWE_LANGUAGE_REPOS = {
        "c": ["facebook/zstd", "jqlang/jq", "ponylang/ponyc"],
        "cpp": ["catchorg/Catch2", "fmtlib/fmt", "nlohmann/json", "simdjson/simdjson", "yhirose/cpp-httplib"],
        "go": ["cli/cli", "grpc/grpc-go", "zeromicro/go-zero"],
        "java": ["alibaba/fastjson2", "elastic/logstash", "mockito/mockito"],
        "javascript": ["anuraghazra/github-readme-stats", "axios/axios", "expressjs/express", 
                      "iamkun/dayjs", "Kong/insomnia", "sveltejs/svelte"],
        "rust": ["BurntSushi/ripgrep", "clap-rs/clap", "nushell/nushell", "serde-rs/serde",
                "sharkdp/bat", "sharkdp/fd", "rayon-rs/rayon", "tokio-rs/bytes", "tokio-rs/tokio", "tokio-rs/tracing"],
        "typescript": ["darkreader/darkreader", "mui/material-ui", "vuejs/core"]
    }

    @staticmethod
    def _detect_language_from_repo(repo_name: str, instance: dict) -> str:
        """Detect programming language from repository name using Multi-SWE-bench mapping."""
        # First check Multi-SWE-bench known repositories
        for language, repos in DockerConfigExtractor.MULTISWE_LANGUAGE_REPOS.items():
            if repo_name in repos:
                return language
        
        # Fallback to heuristic detection for unknown repositories
        repo_lower = repo_name.lower()
        
        # JavaScript/TypeScript detection
        if any(kw in repo_lower for kw in ["vue", "react", "svelte", "express", "axios", "dayjs", "insomnia", "material-ui", "darkreader"]):
            if any(kw in repo_lower for kw in ["material-ui", "darkreader", "vuejs"]):
                return "typescript"
            return "javascript"
        
        # Rust detection
        if any(kw in repo_lower for kw in ["rust", "ripgrep", "clap", "nushell", "serde", "bat", "fd", "rayon", "tokio", "tracing"]):
            return "rust"
        
        # Go detection
        if any(kw in repo_lower for kw in ["go", "grpc", "cli/cli", "zero"]):
            return "go"
        
        # Java detection
        if any(kw in repo_lower for kw in ["java", "fastjson", "logstash", "mockito"]):
            return "java"
        
        # C++ detection
        if any(kw in repo_lower for kw in ["catch2", "fmt", "json", "simdjson", "httplib"]):
            return "cpp"
        
        # C detection
        if any(kw in repo_lower for kw in ["zstd", "jq", "pony"]):
            return "c"
        
        # Default to Python for unknown cases
        return "python"


def _detect_docker_image_workdir(image_name: str) -> Optional[str]:
    """
    Detect the working directory from a Docker image's metadata.
    
    Args:
        image_name: Full Docker image name (e.g., 'mswebench/org_m_repo:pr-123')
        
    Returns:
        Working directory path if found, None otherwise
    """
    if not image_name:
        return None
        
    try:
        # Use docker inspect to get the image's working directory
        result = subprocess.run(
            ["docker", "inspect", image_name, "--format={{.Config.WorkingDir}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True
        )
        workdir = result.stdout.strip()
        
        # Docker returns empty string if WORKDIR wasn't set, defaulting to "/"
        if not workdir or workdir == "/":
            # Multi-SWE-bench pattern: check /home/{repo} first
            if image_name.startswith("mswebench/"):
                # Extract repo name from mswebench/{org}_m_{repo}:tag pattern
                repo_part = image_name.split("/")[1].split(":")[0]  # get org_m_repo part
                if "_m_" in repo_part:
                    repo_name = repo_part.split("_m_", 1)[1].replace("_", "-")
                    potential_dirs = [f"/home/{repo_name}", "/home", "/testbed", "/app", "/workspace"]
                else:
                    potential_dirs = ["/home", "/testbed", "/app", "/workspace"]
            else:
                # For other images, try common locations with memory-based priority [[memory:12334727]]
                potential_dirs = ["/app", "/testbed", "/workspace", "/home"]
            
            for test_dir in potential_dirs:
                if _check_directory_exists_in_image(image_name, test_dir):
                    return test_dir
            return "/"
            
        return workdir
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"Failed to detect working directory for {image_name}: {e}")
        return None


def _check_directory_exists_in_image(image_name: str, directory: str) -> bool:
    """
    Check if a directory exists in a Docker image by running a quick test.
    
    Args:
        image_name: Docker image name
        directory: Directory path to check
        
    Returns:
        True if directory exists, False otherwise
    """
    try:
        result = subprocess.run([
            "docker", "run", "--rm", "--entrypoint", "/bin/sh",
            "--platform", "linux/amd64", 
            image_name, "-c", f"test -d {directory} && echo exists"
        ], capture_output=True, text=True, timeout=60)
        
        return "exists" in result.stdout.strip()
    except Exception:
        return False


def get_sb_environment_with_docker_config(config: dict, instance: dict, docker_config: dict) -> Environment:
    """Get SWE-bench environment with Docker configuration using strategy pattern.
    
    Args:
        config: Agent configuration dictionary
        instance: Instance dictionary
        docker_config: Docker configuration from get_docker_config_for_instance
        
    Returns:
        Environment instance
    """
    from jinja2 import StrictUndefined, Template
    from minisweagent.environments import get_environment
    
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    instance_id = instance["instance_id"]
    
    # Apply configuration from docker_config (already set by strategy)
    if docker_config.get("cwd"):
        env_config["cwd"] = docker_config["cwd"]
    if docker_config.get("timeout"):
        env_config["timeout"] = docker_config["timeout"]
    if docker_config.get("pull_timeout"):
        env_config["pull_timeout"] = docker_config["pull_timeout"]
    if docker_config.get("run_args"):
        env_config["run_args"] = docker_config["run_args"]
    
    # Handle Dockerfile building (for PolyBench poly_dataset source)
    if docker_config.get("dockerfile_content") and not docker_config.get("base_image"):
        # Build temporary Docker image from Dockerfile
        temp_dockerfile_path = f"/tmp/Dockerfile.{instance_id}"
        temp_image_name = f"polybench-temp-{instance_id.replace('__', '-').replace('_', '-').lower()}"
        
        try:
            Path(temp_dockerfile_path).write_text(docker_config["dockerfile_content"])
            
            logger.info(f"Building temporary Docker image: {temp_image_name}")
            build_result = subprocess.run([
                "docker", "build", "-f", temp_dockerfile_path, 
                "-t", temp_image_name, ".", "--platform", "linux/amd64"
            ], cwd="/tmp", capture_output=True, text=True, timeout=1800)
            
            if build_result.returncode == 0:
                if env_config["environment_class"] == "docker":
                    env_config["image"] = temp_image_name
                elif env_config["environment_class"] == "singularity":
                    env_config["image"] = f"docker://{temp_image_name}"
                logger.info(f"Successfully built temporary image: {temp_image_name}")
            else:
                logger.error(f"Docker build failed: {build_result.stderr}")
                raise RuntimeError(f"Docker build failed for {instance_id}")
                
        except Exception as e:
            logger.error(f"Failed to build Docker image for {instance_id}: {e}")
            raise RuntimeError(f"Cannot build image for {instance_id}: {e}")
        finally:
            # Clean up temporary Dockerfile
            Path(temp_dockerfile_path).unlink(missing_ok=True)
    elif docker_config.get("base_image"):
        # Use base image directly (most common case)
        base_image = docker_config["base_image"]
        if env_config["environment_class"] == "docker":
            env_config["image"] = base_image
        elif env_config["environment_class"] == "singularity":
            env_config["image"] = f"docker://{base_image}"
        logger.info(f"Using base Docker image: {base_image} for {instance_id}")
    else:
        # Should not reach here if strategy is working correctly
        raise RuntimeError(
            f"No Docker image or Dockerfile available for {instance_id}. "
            f"Docker config: {docker_config}"
        )
    
    # Create environment
    env = get_environment(env_config)
    logger.info(f"Created environment for {instance_id} using image: {env_config.get('image', 'unknown')}")
    
    # Execute startup command if specified
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        logger.info(f"Executing startup command: {startup_command}")
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            logger.error(f"Startup command failed: {out}")
            raise RuntimeError(f"Startup command failed for {instance_id}: {out}")
    
    return env


class ProgressTrackingContextAwareAgent(ContextAwareAgent):
    """Context-aware agent with progress tracking for batch processing."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    poly_data_dir: Optional[Path] = None,
    auto_pull: bool = True,
    benchmark_type: Optional[BenchmarkType] = None,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Configuring Docker")

    # Get Docker configuration - use provided benchmark_type if available
    docker_config = DockerConfigExtractor.get_docker_config_for_instance(
        instance, poly_data_dir, auto_pull, benchmark_type=benchmark_type
    )
    logger.info(f"Docker config for {instance_id}: {docker_config['source']}")
    
    # Save Dockerfile for debugging if needed
    if docker_config.get("dockerfile_content"):
        instance_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = instance_dir / "Dockerfile.extracted"
        dockerfile_path.write_text(docker_config["dockerfile_content"])

    progress_manager.update_instance_status(instance_id, "Starting environment")

    agent = None
    extra_info = None
    env = None

    try:
        env = get_sb_environment_with_docker_config(config, instance, docker_config)
        agent = ProgressTrackingContextAwareAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config.get("agent", {}),
        )
        exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Include docker configuration in extra_info for debugging
        if extra_info is None:
            extra_info = {}
        extra_info["docker_config"] = docker_config
        
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        if agent:
            logger.info(f"Context data for {instance_id}: {agent.get_context_data()}")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)
        # Cleanup Docker environment
        if env is not None:
            try:
                env.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up environment for {instance_id}: {cleanup_error}")


def _load_multiswe_dataset_safely(dataset_path: str, split: str) -> list[dict]:
    """Load Multi-SWE-bench by directly downloading and parsing JSONL shard files.

    ByteDance-Seed/Multi-SWE-bench consists of 40+ per-language JSONL files whose
    schemas are incompatible (e.g. ``skipped_tests`` may be ``List(null)`` in one
    shard but contain actual values in another).  The Hugging Face ``datasets``
    library forces a single PyArrow schema across all shards, causing cast failures.
    This function bypasses that layer entirely.
    """
    import json
    from pathlib import Path

    # ── Step 1: discover JSONL shard filenames from the builder ──────────────
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(dataset_path)
        data_files: list[str] = list(builder.config.data_files.get(split, []))
    except Exception as e:
        logger.warning(f"Could not probe builder config: {e}. Falling back to listing repo files.")
        data_files = []

    # Fallback: use hf_hub API to list files (if builder probing failed)
    if not data_files:
        try:
            from huggingface_hub import list_repo_files
            raw_files = list_repo_files(dataset_path, repo_type="dataset")
            data_files = [
                f for f in raw_files
                if f.endswith(".jsonl") and "/" in f and f.split("/")[0] != ".gitattributes"
            ]
        except Exception as e:
            raise RuntimeError(f"Cannot enumerate {dataset_path} data files: {e}")

    if not data_files:
        raise RuntimeError(f"No JSONL data files found for split '{split}' in {dataset_path}")

    # ── Step 2: download & parse every shard ────────────────────────────────
    #   hf:// URL format:  hf://datasets/ByteDance-Seed/Multi-SWE-bench@<rev>/<path>
    from huggingface_hub import hf_hub_download

    total_loaded = 0
    total_files = len(data_files)
    instances: list[dict] = []

    logger.info(f"Loading Multi-SWE-bench dataset ({total_files} JSONL shards, split={split})...")

    for df_idx, data_file in enumerate(data_files, 1):
        # extract the relative path inside the repo
        if data_file.startswith("hf://") and "@" in data_file:
            # hf://datasets/.../Multi-SWE-bench@<rev>/path/to/file.jsonl → path/to/file.jsonl
            relative_path = data_file.split("@", 1)[1].split("/", 1)[1]
        else:
            relative_path = data_file  # already a plain relative path

        try:
            local_path = hf_hub_download(
                repo_id=dataset_path,
                filename=relative_path,
                repo_type="dataset",
            )
        except Exception as e:
            logger.warning(f"[{df_idx}/{total_files}] Failed to download {relative_path}: {e}")
            continue

        shard_count = 0
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    simplified = _simplify_multiswe_instance(raw)
                    instances.append(simplified)
                    shard_count += 1
                except Exception as inst_error:
                    logger.debug(f"Skipping instance in {relative_path}: {inst_error}")
                    continue

        total_loaded += shard_count
        logger.info(f"[{df_idx}/{total_files}] {shard_count:4d} instances from {Path(relative_path).name}")

    if not instances:
        raise RuntimeError(f"Failed to load any instances from {dataset_path} (tried {total_files} shards)")

    logger.info(f"Successfully loaded {len(instances)} Multi-SWE-bench instances via direct JSONL parsing")
    return instances


def _simplify_multiswe_instance(instance: dict) -> dict:
    """Simplify complex nested structures in Multi-SWE-bench instances.
    
    This also converts to SWE-bench compatible format in one pass to avoid redundancy.
    """
    simplified = {}
    
    # Set dataset_name to identify this as Multi-SWE-bench (CRITICAL for correct detection)
    simplified["dataset_name"] = "ByteDance-Seed/Multi-SWE-bench"
    
    # Preserve original instance_id if present (Multi-SWE-bench has its own format)
    if "instance_id" in instance and instance["instance_id"]:
        simplified["instance_id"] = instance["instance_id"]
    elif "org" in instance and "repo" in instance and "number" in instance:
        # Fallback: create instance_id from org/repo/number
        org = instance["org"]
        repo = instance["repo"]
        number = instance["number"]
        simplified["instance_id"] = f"{org}__{repo}-{number}"
    
    # Basic fields - create repo in org/repo format
    if "org" in instance and "repo" in instance:
        org = instance["org"]
        repo = instance["repo"]
        simplified["repo"] = f"{org}/{repo}"
        simplified["org"] = org
    elif "repo" in instance:
        simplified["repo"] = instance["repo"]
    
    # Handle number/PR fields
    if "number" in instance:
        number = instance["number"]
        simplified["number"] = str(number)
        simplified["pr_number"] = str(number)
    
    # Handle problem_statement field - Multi-SWE-bench uses 'body' or 'title'
    if "body" in instance and instance["body"]:
        simplified["problem_statement"] = instance["body"]
    elif "title" in instance and instance["title"]:
        simplified["problem_statement"] = instance["title"]
    elif "problem_statement" in instance:
        simplified["problem_statement"] = instance["problem_statement"]
    
    # Copy simple string fields
    simple_string_fields = ["state", "title", "hints"]
    for field in simple_string_fields:
        if field in instance and instance[field]:
            simplified[field] = instance[field]
    
    # Handle patch fields
    if "fix_patch" in instance and instance["fix_patch"]:
        simplified["patch"] = instance["fix_patch"]
        simplified["model_patch"] = instance["fix_patch"]
        simplified["fix_patch"] = instance["fix_patch"]
    
    if "test_patch" in instance and instance["test_patch"]:
        simplified["test_patch"] = instance["test_patch"]
    
    # Handle base commit info (flatten structure)
    if "base" in instance and isinstance(instance["base"], dict):
        base_info = instance["base"]
        if "sha" in base_info:
            simplified["base_commit"] = base_info["sha"]
        if "ref" in base_info:
            simplified["version"] = base_info["ref"]
            simplified["base_ref"] = base_info["ref"]
        if "label" in base_info:
            simplified["base_label"] = base_info["label"]
    
    # Handle resolved_issues (extract simple info only)
    if "resolved_issues" in instance and instance["resolved_issues"]:
        try:
            issues = instance["resolved_issues"]
            if issues and isinstance(issues, (list, tuple)) and len(issues) > 0:
                first_issue = issues[0]
                if isinstance(first_issue, dict):
                    if "number" in first_issue:
                        simplified["resolved_issue_number"] = first_issue["number"]
                    if "title" in first_issue:
                        simplified["resolved_issue_title"] = first_issue["title"]
        except Exception as e:
            logger.debug(f"Error processing resolved_issues: {e}")
    
    # Skip complex nested test structures that cause PyArrow issues:
    # fixed_tests, p2p_tests, f2p_tests, s2p_tests, n2p_tests, run_result, test_patch_result, fix_patch_result
    skip_fields = {
        "fixed_tests", "p2p_tests", "f2p_tests", "s2p_tests", "n2p_tests",
        "run_result", "test_patch_result", "fix_patch_result", "resolved_issues", "base"
    }
    
    # Add any other simple fields that might be useful
    for key, value in instance.items():
        if key not in simplified and key not in skip_fields:
            if isinstance(value, (str, int, float, bool)) and value is not None:
                simplified[key] = value
    
    return simplified


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset. Supports: lite, verified, pro, multi-swe-bench, or custom path", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    poly_data_dir: str = typer.Option("", "--poly-data-dir", help="Path to poly dataset directory for Dockerfile extraction", rich_help_panel="Advanced"),
    auto_pull: bool = typer.Option(True, "--auto-pull/--no-auto-pull", help="Automatically pull missing Docker images", rich_help_panel="Advanced"),
    multiswe_format: bool = typer.Option(False, "--multiswe-format", help="Use Multi-SWE-bench data format with org/repo/number fields", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    # Set up poly data directory if provided
    poly_path = Path(poly_data_dir) if poly_data_dir else None
    if poly_path and not poly_path.exists():
        logger.warning(f"Poly data directory not found: {poly_path}")
        poly_path = None
    elif poly_path:
        logger.info(f"Using poly data directory: {poly_path}")

    # Determine benchmark type from subset parameter (simple and explicit mapping)
    # Based on actual usage in run scripts:
    # - multi-swe-bench → MULTI_SWEBENCH
    # - AmazonScience/SWE-PolyBench or polybench → POLYBENCH
    # - pro → SWEBENCH_PRO
    # - verified, lite → SWEBENCH_DEFAULT
    
    subset_lower = subset.lower()
    
    # Multi-SWE-bench detection (from run_multi_*.sh scripts)
    if "multi" in subset_lower and "swe" in subset_lower:
        determined_benchmark_type = BenchmarkType.MULTI_SWEBENCH
        if "mini" in subset_lower:
            dataset_path = "ByteDance-Seed/Multi-SWE-bench_mini"
            logger.info(f"Loading Multi-SWE-bench mini dataset...")
        elif "flash" in subset_lower:
            dataset_path = "ByteDance-Seed/Multi-SWE-bench-flash"
            logger.info(f"Loading Multi-SWE-bench flash dataset...")
        else:
            dataset_path = "ByteDance-Seed/Multi-SWE-bench"
            logger.info(f"Loading Multi-SWE-bench dataset...")
        
        # Multi-SWE-bench only has 'train' split
        if split in ["test", "dev"]:
            logger.info(f"Multi-SWE-bench only has 'train' split, using 'train' instead of '{split}'")
            split = "train"
    
    # PolyBench detection (from run_poly_*.sh scripts: --subset AmazonScience/SWE-PolyBench)
    elif "polybench" in subset_lower or "amazonscience/swe-polybench" in subset_lower:
        determined_benchmark_type = BenchmarkType.POLYBENCH
        dataset_path = DATASET_MAPPING.get(subset, subset)
    
    # SWE-bench Pro detection (from run_pro_*.sh scripts: --subset pro)
    elif subset_lower == "pro" or "pro" in subset_lower and "swe" in subset_lower:
        determined_benchmark_type = BenchmarkType.SWEBENCH_PRO
        dataset_path = DATASET_MAPPING.get(subset, subset)
    
    # Default SWE-bench (from run_verified_*.sh scripts: --subset verified)
    else:
        determined_benchmark_type = BenchmarkType.SWEBENCH_DEFAULT
        dataset_path = DATASET_MAPPING.get(subset, subset)
        
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    
    # Load Multi-SWE-bench datasets with special handling for complex nested structures
    if determined_benchmark_type == BenchmarkType.MULTI_SWEBENCH:
        instances = _load_multiswe_dataset_safely(dataset_path, split)
        # Conversion already done in _simplify_multiswe_instance, no need for additional processing
    else:
        instances = list(load_dataset(dataset_path, split=split))
    
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    # Log the determined benchmark type
    logger.info(f"Using benchmark type: {determined_benchmark_type.value}")
    
    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance, 
                    instance, 
                    output_path, 
                    config, 
                    progress_manager, 
                    poly_path, 
                    auto_pull,
                    determined_benchmark_type
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()


