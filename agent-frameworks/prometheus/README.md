# Prometheus (ContextBench adapter)

ContextBench wrapper around [Prometheus](https://github.com/EuniAI/Prometheus). The vendored service lives under `prometheus/` next to this README, so you do **not** need to clone the upstream repository separately.

## What We Changed / Optimized

- **Unified multi-bench runner**: `run_bench.py` provides a single CLI entry for Verified, Pro, Poly, and Multi.
- **Data adapter layer**: `bench_sources.py` loads CSV/Parquet/HuggingFace sources and resolves SWE-bench Docker images per bench.
- **HTTP API integration**: each instance is uploaded and answered via Prometheus REST endpoints (`/repository/upload/`, `/issue/answer/`).
- **Self-contained layout**: vendored Prometheus stack (`docker-compose.yml`, Python package, tests) ships inside `prometheus/`.

## Data & Setup

### Folder layout (adapter and data are siblings)

```
ContextBench/
├── agent-frameworks/
│   └── prometheus/
│       ├── run_bench.py
│       ├── run_single_instance.py
│       ├── bench_sources.py
│       ├── prometheus_client.py
│       ├── prometheus/              # vendored Prometheus service (no separate git clone)
│       │   ├── docker-compose.yml
│       │   ├── example.env
│       │   ├── prometheus/          # Python package
│       │   └── working_dir/         # runtime logs (created on first run)
│       └── results/
└── data/
    ├── Multi.csv
    ├── contextbench_verified.parquet
    └── ...
```

### 1) Install adapter dependencies

```bash
cd agent-frameworks/prometheus
pip install -r requirements.txt
```

### 2) Start Prometheus services

```bash
cd prometheus
cp example.env .env
# Edit .env: set PROMETHEUS_ENABLE_AUTHENTICATION=false and LLM API keys
docker-compose up -d
```

Default API URL: `http://localhost:9002/v1.3`

### 3) Configure LLM keys (in `prometheus/.env`)

Set at least one provider block, for example:

- `PROMETHEUS_OPENAI_FORMAT_BASE_URL`
- `PROMETHEUS_OPENAI_FORMAT_API_KEY`
- `PROMETHEUS_ADVANCED_MODEL` / `PROMETHEUS_BASE_MODEL`

Optional:

- `GITHUB_TOKEN` — private repository access during upload
- `PROMETHEUS_TAVILY_API_KEY` — web search tool

After editing `.env`, restart the service so the container picks up new values:

```bash
docker compose restart prometheus
```

## Configuration: URL vs LLM keys

Prometheus uses two separate configuration layers:

| Layer | What it configures | How to set |
|-------|-------------------|------------|
| **Prometheus service** | LLM provider URL/key, models, Neo4j/Postgres | `prometheus/.env` (read by `docker-compose`) |
| **ContextBench adapter** | HTTP API endpoint, timeouts, log paths | env vars or CLI flags below |

**LLM API keys cannot be passed through `contextbench.run` or `run_bench.py`.** The adapter only calls the Prometheus HTTP API; LLM calls happen inside the Prometheus container.

### Service-side (LLM) — `prometheus/.env`

Required for `/issue/answer/` to succeed:

- `PROMETHEUS_OPENAI_FORMAT_BASE_URL`
- `PROMETHEUS_OPENAI_FORMAT_API_KEY`
- `PROMETHEUS_ADVANCED_MODEL` / `PROMETHEUS_BASE_MODEL`

### Adapter-side (HTTP API)

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMETHEUS_URL` | `http://localhost:9002/v1.3` | API base URL |
| `PROMETHEUS_ROOT` | `./prometheus` | Vendored service root (for log capture) |
| `PROMETHEUS_WORKING_DIRECTORY` | — | Override server working dir parent |
| `PROMETHEUS_TIMEOUT` | `3600` | Per-instance HTTP timeout (seconds) |
| `PROMETHEUS_CANDIDATE_PATCHES` | `3` | Candidate patches per issue |
| `PROMETHEUS_RUN_REPRODUCE` | `true` | Run reproduction tests |
| `PROMETHEUS_RUN_REGRESSION` | `true` | Run regression tests |
| `PROMETHEUS_JWT_TOKEN` | — | Bearer token if `PROMETHEUS_ENABLE_AUTHENTICATION=true` |
| `GITHUB_TOKEN` | — | Private repo access during upload (adapter) |

Pass API URL via CLI:

```bash
python run_bench.py Verified --limit 1 --prometheus-url http://localhost:9002/v1.3
python run_single_instance.py Verified --instance ID --prometheus-url http://localhost:9002/v1.3
```

## Usage

Unified interface:

```bash
python run_bench.py {bench_name} --limit N
```

- `bench_name`: `Multi | Poly | Pro | Verified`
- `--limit N`: optional; run only the first N instances

### Example (smoke test)

```bash
python run_bench.py Multi --limit 1
python run_bench.py Poly --limit 1
python run_bench.py Pro --limit 1
python run_bench.py Verified --limit 1
```

### Single instance

```bash
python run_single_instance.py Verified --instance scikit-learn__scikit-learn-25232 \
  --output results/prometheus
```

### Via ContextBench runner

```bash
python -m contextbench.run --agent prometheus --bench Verified --limit 1

# Custom API URL (LLM keys still go in prometheus/.env)
python -m contextbench.run --agent prometheus --bench Verified --limit 1 \
  --prometheus-url http://localhost:9002/v1.3
```

## Outputs

Results are stored under:

```
<output>/prometheus/{bench}/{original_inst_id}.log   # trajectory (context blocks)
<output>/prometheus/{bench}/{original_inst_id}.json  # patch + test flags
```

Convert for evaluation:

```bash
python -m contextbench.process_trajectories convert \
  -i results/agent_runs/prometheus -o pred.jsonl --agent prometheus
```

## Benches

| Bench | Dataset source | Docker image pattern |
|-------|----------------|----------------------|
| Verified | `data/contextbench_verified.parquet` or HF | `swebench/sweb.eval.x86_64.*` |
| Pro | HuggingFace `ScaleAI/SWE-bench_Pro` | `jefzda/sweap-images:*` |
| Poly | HuggingFace `AmazonScience/SWE-PolyBench` | `ghcr.io/timesler/swe-polybench.*` |
| Multi | `data/Multi.csv` + GitHub PR text | `mswebench/*` |

## Notes / Troubleshooting

- Start `docker-compose` in `prometheus/` before running benchmarks; the adapter only talks to the HTTP API.
- LLM credentials belong in `prometheus/.env`, not ContextBench `LLM_API_URL`.
- If `.log` files are empty, check `prometheus/working_dir/answer_issue_logs/` or set `PROMETHEUS_WORKING_DIRECTORY`.
- Network access is required for repository upload unless repos are already cached by Prometheus.
