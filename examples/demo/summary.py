"""Turn Spark MCP tool payloads into short text a presenter can read."""

from __future__ import annotations

import json
from typing import Any

# These differ between two runs of the same job, but they are not Spark settings.
_IDENTITY_KEYS = {
    "spark.app.id",
    "spark.app.name",
    "spark.app.startTime",
    "spark.app.submitTime",
    "spark.driver.bindAddress",
    "spark.driver.host",
    "spark.executorEnv.SPARK_DRIVER_POD_IP",
    "spark.kubernetes.driver.label.sparkoperator.k8s.io/submission-id",
    "spark.kubernetes.executor.label.sparkoperator.k8s.io/submission-id",
}


def parse_tool_text(text: str) -> Any:
    """Parse a tool result that may be JSON, or several JSON values in a row."""
    raw = text.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_concatenated(raw)
    if isinstance(value, str):
        inner = value.strip()
        if not inner:
            return value
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            try:
                return _parse_concatenated(inner)
            except json.JSONDecodeError:
                return value
    return value


def _parse_concatenated(text: str) -> Any:
    decoder = json.JSONDecoder()
    index = 0
    items: list[Any] = []
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            item, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            if items:
                break
            return text
        items.append(item)
    if len(items) == 1:
        return items[0]
    return items


def summarize_tool(name: str, payload: Any) -> str:
    """Plain-language facts for one tool result."""
    if isinstance(payload, dict) and payload.get("error"):
        return f"The tool returned an error: {payload['error']}"
    if name == "list_applications":
        return _summarize_applications(payload)
    if name == "compare_job_performance":
        return _summarize_performance(payload)
    if name == "compare_job_environments":
        return _summarize_environments(payload)
    if name == "get_job_bottlenecks":
        return _summarize_bottlenecks(payload)
    if name == "list_sql_executions":
        return _summarize_sql(payload)
    return _fallback(payload)


def _summarize_applications(payload: Any) -> str:
    apps = payload if isinstance(payload, list) else [payload]
    lines = []
    for app in apps:
        if not isinstance(app, dict):
            continue
        attempt = _first_attempt(app)
        duration = _milliseconds(attempt.get("duration"))
        state = "completed" if attempt.get("completed") else "not completed"
        version = attempt.get("appSparkVersion") or "unknown Spark version"
        lines.append(
            f"{app.get('name') or 'unnamed'}  {app.get('id') or ''}  "
            f"{state}  {duration}  Spark {version}"
        )
    return "\n".join(lines) if lines else "No applications came back."


def _summarize_performance(payload: Any) -> str:
    if not isinstance(payload, dict):
        return _fallback(payload)
    apps = payload.get("applications") or {}
    jobs = payload.get("job_performance") or {}
    executors = payload.get("executor_metrics") or {}
    lines = []
    for key, label in (("app1", "First job"), ("app2", "Second job")):
        app = apps.get(key) or {}
        job = jobs.get(key) or {}
        exe = executors.get(key) or {}
        lines.append(
            f"{label}: {app.get('name')} ({app.get('id')})\n"
            f"  jobs: {job.get('count')}  "
            f"total job time: {_seconds(job.get('total_duration'))}  "
            f"slowest job: {_seconds(job.get('max_duration'))}\n"
            f"  executors: {exe.get('total_executors')}  "
            f"completed tasks: {exe.get('completed_tasks')}  "
            f"failed tasks: {exe.get('failed_tasks')}\n"
            f"  input: {_gigabytes(exe.get('total_input_bytes'))}  "
            f"shuffle read: {_gigabytes(exe.get('total_shuffle_read'))}  "
            f"gc: {_milliseconds(exe.get('total_gc_time'))}"
        )
    return "\n".join(lines)


def _summarize_environments(payload: Any) -> str:
    if not isinstance(payload, dict):
        return _fallback(payload)
    props = payload.get("spark_properties") or {}
    different = props.get("different") or {}
    common_count = len(props.get("common") or {})
    meaningful = sorted(key for key in different if key not in _IDENTITY_KEYS)
    if meaningful:
        shown = ", ".join(meaningful[:8])
        return f"{common_count} Spark settings match. Settings that differ: {shown}."
    return (
        f"{common_count} Spark settings match. "
        "The only differences are the application id, name, start time, and host."
    )


def _summarize_bottlenecks(payload: Any) -> str:
    if not isinstance(payload, dict):
        return _fallback(payload)
    stages = (payload.get("performance_bottlenecks") or {}).get("slowest_stages") or []
    lines = [f"Application {payload.get('application_id')}"]
    for stage in stages[:3]:
        lines.append(
            f"  stage {stage.get('stage_id')}: {stage.get('name')}  "
            f"{_seconds(stage.get('duration_seconds'))}  "
            f"{stage.get('task_count')} tasks  "
            f"{stage.get('failed_tasks')} failed"
        )
    spill = (payload.get("resource_bottlenecks") or {}).get("memory_spill_stages")
    if spill:
        lines.append(f"  stages with memory spill: {len(spill)}")
    else:
        lines.append("  no stage spilled more than 100 MB")
    return "\n".join(lines)


def _summarize_sql(payload: Any) -> str:
    rows = payload if isinstance(payload, list) else [payload]
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"SQL {row.get('id')}: {row.get('description')}  "
            f"{row.get('status')}  {_milliseconds(row.get('duration'))}"
        )
    return "\n".join(lines) if lines else "No SQL executions came back."


def _first_attempt(app: dict[str, Any]) -> dict[str, Any]:
    attempts = app.get("attempts") or []
    if attempts and isinstance(attempts[0], dict):
        return attempts[0]
    return {}


def _milliseconds(value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown duration"
    return _seconds(float(value) / 1000.0)


def _seconds(value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    return f"{value:.1f} s"


def _gigabytes(value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    return f"{value / 1_000_000_000:.2f} GB"


def _fallback(payload: Any) -> str:
    text = json.dumps(payload, default=str)
    if len(text) > 1200:
        return text[:1200] + "…"
    return text
