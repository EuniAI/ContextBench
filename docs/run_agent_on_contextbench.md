# ContextBench Agent Runner Usage Guide

`contextbench.run` is the unified Agent runner for ContextBench. It supports a three-step workflow: **load task list**, **run Agent**, and **output trajectories**.

## Overview

1. **Load task list**: Load instances from CSV or gold JSONL
2. **Filter subset**: Filter by bench type, instance ID, or custom CSV
3. **Run Agent and output trajectories**: Dispatch each instance to the appropriate agent-framework implementation based on its bench

## Run Strategy

For each instance, the script:

1. Parses the **Agent** to use (e.g. agentless, miniswe)
2. Determines the instance's **native bench** (Verified / Pro / Poly / Multi)
3. Invokes the agent framework adapted for that bench:

| Agent   | Verified       | Pro               | Poly                | Multi              |
|---------|----------------|-------------------|---------------------|--------------------|
| agentless | run_bench.py   | run_bench.py      | run_bench.py        | run_bench.py       |
| miniswe | swebench_context_aware | swebench_context_aware | swebench_context_aware | swebench_context_aware |
| prometheus | run_bench.py (HTTP API) | run_bench.py | run_bench.py | run_bench.py |

## Command-Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent` | Agent to use: `agentless`, `miniswe`, `sweagent`, `openhands`, or `prometheus` |

### Task Source

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-csv` | `data/selected_500_instances.csv` | Path to task list CSV |
| `--subset-csv` | - | Custom subset CSV (overrides `--task-csv`) |
| `--gold-jsonl` | - | Use gold JSONL instead of CSV (bench inferred from instance_id) |

### Task Filtering

| Argument | Description |
|----------|-------------|
| `--bench` | Filter by bench: Verified, Pro, Poly, Multi (comma-separated for multiple) |
| `--instances` | Specify instance_id or original_inst_id (comma-separated) |
| `--limit` | Process at most N instances (0 = no limit) |

### Output & Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` / `-o` | `results/agent_runs` | Trajectory output directory |
| `--timeout` | 1800 | Timeout per instance (seconds) |
| `--dry-run` | false | Only list tasks, do not run Agent |
| `--debug` | false | Enable debug mode with verbose logs |
| `--rerun` | false | Rerun instances that already have trajectories (default: skip) |

### SWE-agent Options

| Argument | Description |
|----------|-------------|
| `--sweagent-config` | Path to SWE-agent config YAML (or set `SWEAGENT_CONFIG` env var) |

### OpenHands Options

| Argument | Description |
|----------|-------------|
| `--openhands-model-config` | OpenHands LLM config name (or set `OPENHANDS_MODEL_CONFIG`) |
| `--openhands-agent` | OpenHands Agent class name (or set `OPENHANDS_AGENT`) |

### Prometheus Options

| Argument | Description |
|----------|-------------|
| `--prometheus-url` | Prometheus API base URL (or set `PROMETHEUS_URL`; default `http://localhost:9002/v1.3`) |

Prometheus LLM credentials are **not** passed through `contextbench.run`. Configure them in
`agent-frameworks/prometheus/prometheus/.env` and restart the Prometheus container after changes.
See [agent-frameworks/prometheus/README.md](../agent-frameworks/prometheus/README.md).

## Usage Examples

### Basic Usage

```bash
# Run agentless on Verified
python -m contextbench.run --agent agentless --bench Verified

# Run miniswe on Pro, first 5 instances only
python -m contextbench.run --agent miniswe --bench Pro --limit 5

# Run agentless on Poly
python -m contextbench.run --agent agentless --bench Poly
```

### Specific Instances

```bash
# Run only specified instances (instance_id or original_inst_id)
python -m contextbench.run --agent agentless \
    --instances "scikit-learn__scikit-learn-25232,django__django-14434"

# Specify via original_inst_id
python -m contextbench.run --agent miniswe \
    --instances "keras-team__keras-18553"
```

### Custom Task List

```bash
# Use custom subset CSV
python -m contextbench.run --agent miniswe \
    --subset-csv my_subset.csv \
    --output results/my_run

# Use gold JSONL (bench inferred automatically)
python -m contextbench.run --agent agentless \
    --gold-jsonl results/gold/contextbench_verified.gold.jsonl \
    --limit 10
```

### Debug & Preview

```bash
# List tasks only, do not run
python -m contextbench.run --agent miniswe --bench Verified --dry-run

# Combined filters
python -m contextbench.run --agent agentless \
    --bench Verified,Pro \
    --limit 3 \
    --dry-run
```

### Prometheus

```bash
# Default local API (http://localhost:9002/v1.3)
python -m contextbench.run --agent prometheus --bench Verified --limit 1

# Custom API URL via CLI
python -m contextbench.run --agent prometheus --bench Verified --limit 1 \
    --prometheus-url http://localhost:9002/v1.3

# Or via environment variable
export PROMETHEUS_URL=http://localhost:9002/v1.3
python -m contextbench.run --agent prometheus --bench Pro --limit 1
```

## Prerequisites

### Agentless

- Uses unified entry point `agent-frameworks/agentless/run_bench.py` with `--instance` for single-instance runs
- Ensure `data/` contains datasets for each bench (Verified, Pro, Poly, Multi); see agentless README for details
- Configure OpenAI API and related services per Agentless requirements (`script/api_key.sh`)

### MiniSWE-agent

- Entry point: `mini-swe-agent/multi-poly-pro-verified/mini-swe-agent/src/minisweagent/run/extra/swebench_context_aware.py`
- Install mini-swe-agent and its dependencies
- When using Docker, ensure the environment can pull the relevant bench images

### SWE-agent

- Entry point: `swe-agent/{bench}/sweagent/run/run_batch.py` (via `sweagent run-batch`)
- Configure `--sweagent-config` or `SWEAGENT_CONFIG` to point to a valid config YAML
- Supports Verified, Pro, Poly, Multi

### OpenHands

- Entry point: `openhands/{verified|poly-pro|multi}/evaluation/benchmarks/swe_bench/scripts/run_infer.sh`
- Model and Agent can be configured via `OPENHANDS_MODEL_CONFIG`, `OPENHANDS_AGENT`
- Single-instance runs use `EVAL_LIMIT=1`; exact filtering requires a pre-configured `config.toml`

### Prometheus

- Entry point: `agent-frameworks/prometheus/run_bench.py` (calls Prometheus HTTP API)
- Vendored service: `agent-frameworks/prometheus/prometheus/`
- Start stack: `cd agent-frameworks/prometheus/prometheus && cp example.env .env && docker-compose up -d`
- **LLM keys** (required for `/issue/answer/`): edit `prometheus/.env` (`PROMETHEUS_OPENAI_FORMAT_API_KEY`, `PROMETHEUS_OPENAI_FORMAT_BASE_URL`, model names). Restart after changes: `docker compose restart prometheus`
- **API URL** (adapter → service): pass `--prometheus-url` to `contextbench.run`, or set `PROMETHEUS_URL`. LLM keys cannot be passed via the run script.
- Optional adapter env vars: `PROMETHEUS_TIMEOUT`, `GITHUB_TOKEN`, `PROMETHEUS_JWT_TOKEN` (if authentication enabled)
- Uses SWE-bench Docker images via `image_name` + `workdir` on `/issue/answer/`
- Output: `prometheus/{bench}/{instance_id}.log` + `.json` (patch)

## Output Structure

Trajectories are organized by agent and bench:

```
<output_dir>/
├── agentless/
│   ├── Verified/
│   │   └── *_traj.json
│   ├── Pro/
│   ├── Poly/
│   └── Multi/
└── miniswe/
    ├── Verified/
    ├── Pro/
    ├── Poly/
    └── Multi/
└── prometheus/
    ├── Verified/
    │   ├── {instance_id}.log
    │   └── {instance_id}.json
    ├── Pro/
    ├── Poly/
    └── Multi/
```

## Bench Inference Rules

When the CSV has no `bench` column, bench is inferred from `instance_id`:

| instance_id pattern | Inferred bench |
|---------------------|----------------|
| `SWE-Bench-Pro__*` or `instance_*` (length > 50) | Pro |
| `SWE-PolyBench__*` | Poly |
| Contains `multi` | Multi |
| `SWE-Bench-Verified__*` or `org__repo-number` | Verified |

## Troubleshooting

- **Agent script not found**: Check that the run script exists under `agent-frameworks/` for the chosen agent.
- **No tasks matched filters**: Verify that `--bench` and `--instances` match the task list.
- **Timeout**: Increase `--timeout` or test with `--limit 1` first.
