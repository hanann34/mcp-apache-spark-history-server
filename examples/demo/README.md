# Presentation demo

A page you can put on a screen. A model running on your computer asks this repo's MCP server about the sample Spark jobs, and the page shows each lookup as it happens.

Three programs are involved:

1. **Spark History Server**, from Apache Spark, reading the sample jobs in `examples/basic`.
2. **spark-mcp**, from this repo. It is the connector the model calls.
3. **A local model**, through [Ollama](https://ollama.com/). This page talks to that model. The model is not part of the repo.

## Start it

From the repository root, with [Docker](https://docs.docker.com/get-docker/) running:

```bash
task install
task start-spark-bg
task start-mcp-bg
```

Install Ollama from https://ollama.com/download, then pull the model this page expects:

```bash
ollama pull qwen2.5:1.5b
```

Start the page:

```bash
uv run python -m examples.demo.server
```

Open http://127.0.0.1:8787

The three labels at the top should say the model, spark-mcp, and Spark History Server are connected.

## What to click

1. **What Spark jobs are stored?** The model calls `list_applications` and names the sample jobs.
2. **Why was the later taxi job slower?** The model calls `list_applications`, `compare_job_performance`, `get_job_bottlenecks`, and `compare_job_environments`. On a CPU this takes about a minute. The page updates as each lookup comes back, so leave it on the screen. Read the cards first. Those are the facts from Spark. The green card is the model's answer, and it should use those facts.

A larger local model gives a cleaner answer. Point the page at one you have already pulled:

```bash
SHS_DEMO_MODEL=qwen2.5:7b uv run python -m examples.demo.server
```

## If a label stays red

| Label | Check |
|---|---|
| Spark History Server | `curl http://127.0.0.1:18080/api/v1/applications` should return JSON. Start it with `task start-spark-bg`. |
| spark-mcp | `curl -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18888/` returning 404 means the server is up. Start it with `task start-mcp-bg` from the repo root, so it finds `config.yaml`. |
| Local model | `ollama list` should include `qwen2.5:1.5b`. Start the Ollama app if the command cannot connect. |
