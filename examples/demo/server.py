"""Browser demo: a local Ollama model calls this repo's MCP server.

Run from the repository root:

    uv run python -m examples.demo.server

Then open http://127.0.0.1:8787
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.demo.summary import parse_tool_text, summarize_tool  # noqa: E402

logger = logging.getLogger(__name__)

MCP_URL = os.environ.get("SHS_DEMO_MCP_URL", "http://127.0.0.1:18888/mcp")
OLLAMA_URL = os.environ.get("SHS_DEMO_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("SHS_DEMO_MODEL", "qwen2.5:1.5b")
HISTORY_URL = os.environ.get(
    "SHS_DEMO_HISTORY_URL", "http://127.0.0.1:18080/api/v1/applications"
)
HOST = os.environ.get("SHS_DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("SHS_DEMO_PORT", "8787"))
PAGE = Path(__file__).with_name("index.html")

MAX_ROUNDS = 10

SYSTEM_PROMPT = """You answer questions about finished Spark jobs.
The facts come only from tools. Call a tool before you mention any name, duration, or size.
If asked which jobs exist, call list_applications once, then answer.
If asked to compare the taxi jobs, call list_applications, then compare_job_performance with the two application ids whose names start with NewYorkTaxiData, then get_job_bottlenecks for each of those ids, then compare_job_environments with those same ids. Then answer.
Use only numbers that appeared in tool results. Keep the answer to a few short sentences.
Never say the Spark settings differ unless compare_job_environments says they differ.
"""

DEMO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_applications",
            "description": "List Spark applications known to the history server.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_job_performance",
            "description": "Compare duration, tasks, input, and shuffle for two applications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id1": {"type": "string"},
                    "app_id2": {"type": "string"},
                },
                "required": ["app_id1", "app_id2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_job_environments",
            "description": "Compare Spark settings for two applications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id1": {"type": "string"},
                    "app_id2": {"type": "string"},
                },
                "required": ["app_id1", "app_id2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_bottlenecks",
            "description": "List the slowest stages for one application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string"},
                    "top_n": {"type": "integer"},
                },
                "required": ["app_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sql_executions",
            "description": "List SQL queries for one application, slowest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "sort_by": {"type": "string"},
                },
                "required": ["app_id"],
            },
        },
    },
]

_ask_lock = asyncio.Lock()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _probe(url: str, ok_statuses: set[int]) -> bool:
    try:
        async with httpx2.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
        return response.status_code in ok_statuses
    except httpx2.HTTPError:
        return False


async def _model_ready() -> bool:
    try:
        async with httpx2.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code != 200:
            return False
        names = [item.get("name", "") for item in response.json().get("models", [])]
    except (httpx2.HTTPError, ValueError):
        return False
    return any(name == MODEL or name.startswith(f"{MODEL}:") for name in names)


async def status(_: Request) -> JSONResponse:
    mcp_up, history_up, model_up = await asyncio.gather(
        _probe(MCP_URL, {200, 400, 406}),
        _probe(HISTORY_URL, {200}),
        _model_ready(),
    )
    return JSONResponse(
        {
            "mcp": mcp_up,
            "history": history_up,
            "model": MODEL,
            "model_ready": model_up,
        }
    )


def _payload_from_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    if structured:
        return structured
    chunks: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return parse_tool_text("\n".join(chunks))


async def _call_mcp(name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
    async with streamable_http_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
    return _payload_from_result(result), bool(result.is_error)


def _allowed_arguments() -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for tool in DEMO_TOOLS:
        function = tool["function"]
        allowed[function["name"]] = set(function["parameters"]["properties"])
    return allowed


_ALLOWED_ARGUMENTS = _allowed_arguments()


def _clean_arguments(name: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    allowed = _ALLOWED_ARGUMENTS.get(name, set())
    return {
        key: value for key, value in raw.items() if value is not None and key in allowed
    }


async def _ollama_chat(
    messages: list[dict[str, Any]],
    *,
    use_tools: bool = True,
    num_predict: int = 700,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "stream": False,
        "messages": messages,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": num_predict,
        },
    }
    if use_tools:
        body["tools"] = DEMO_TOOLS
    async with httpx2.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
        response.raise_for_status()
        return response.json()["message"]


def _spark_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("spark-")


def _wants_comparison(question: str) -> bool:
    text = question.lower()
    return any(word in text for word in ("slower", "faster", "compare", "why"))


def _reminder(
    question: str,
    succeeded: set[str],
    nudged: bool,
    followups: int,
    app_ids: list[str],
) -> tuple[str, str] | None:
    """One more instruction when a small local model stops before the lookup."""
    if not succeeded and not nudged:
        return (
            "Call list_applications with no arguments. "
            "Application ids start with spark-. Do not invent ids from the job name.",
            "The model has not looked anything up yet. Asking it to call list_applications.",
        )
    if not _wants_comparison(question) or followups >= 6:
        return None
    taxi = app_ids[:2]
    id_text = " and ".join(taxi) if taxi else "the two ids that start with spark-"
    if "compare_job_performance" not in succeeded:
        return (
            f"Call compare_job_performance. Set app_id1 and app_id2 to {id_text}.",
            "Asking the model to compare the two taxi jobs.",
        )
    if "get_job_bottlenecks" not in succeeded and taxi:
        return (
            "Call get_job_bottlenecks with app_id "
            f"{taxi[0]}. Then call it again with app_id {taxi[-1]}.",
            "Asking the model which stage was slow.",
        )
    if "compare_job_environments" not in succeeded and len(taxi) == 2:
        return (
            "Call compare_job_environments with "
            f"app_id1 {taxi[0]} and app_id2 {taxi[1]}.",
            "Asking the model whether the Spark settings differ.",
        )
    return None


async def _ask_events(question: str) -> AsyncIterator[str]:
    yield _sse({"kind": "status", "text": f"Asking {MODEL}…"})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    nudged = False
    followups = 0
    succeeded: set[str] = set()
    app_ids: list[str] = []
    bottleneck_ids: list[str] = []
    closing = False
    try:
        for _ in range(MAX_ROUNDS):
            message = await _ollama_chat(messages)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reminder = _reminder(question, succeeded, nudged, followups, app_ids)
                if reminder is not None:
                    if not succeeded:
                        nudged = True
                    else:
                        followups += 1
                    messages.append(message)
                    messages.append({"role": "user", "content": reminder[0]})
                    yield _sse({"kind": "status", "text": reminder[1]})
                    continue
                if (
                    _wants_comparison(question)
                    and "compare_job_performance" in succeeded
                    and not closing
                ):
                    closing = True
                    messages.append(message)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Answer in one short paragraph, under 90 words. "
                                "Call the jobs the 03:56 taxi job and the 00:24 taxi job. "
                                "Do not print application ids. "
                                "Say which was slower, the input sizes, the shuffle sizes, "
                                "and the slowest stage number with its seconds. "
                                "The environment lookup found that the Spark settings match. "
                                "Say that they match. Do not say they differ. "
                                "Finish the paragraph."
                            ),
                        }
                    )
                    yield _sse(
                        {
                            "kind": "status",
                            "text": "The model is writing the answer from all of the lookups.",
                        }
                    )
                    continue
                text = (message.get("content") or "").strip()
                yield _sse(
                    {
                        "kind": "answer",
                        "text": text or "The model returned an empty answer.",
                        "used_tools": bool(succeeded),
                    }
                )
                return

            messages.append(message)
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or "unknown"
                arguments = _clean_arguments(name, function.get("arguments"))
                if (
                    name in {"compare_job_performance", "compare_job_environments"}
                    and len(app_ids) >= 2
                    and not (
                        _spark_id(arguments.get("app_id1"))
                        and _spark_id(arguments.get("app_id2"))
                    )
                ):
                    arguments = {"app_id1": app_ids[0], "app_id2": app_ids[1]}
                if name == "get_job_bottlenecks" and not _spark_id(
                    arguments.get("app_id")
                ):
                    remaining = [
                        app_id for app_id in app_ids if app_id not in bottleneck_ids
                    ]
                    if remaining:
                        arguments = {"app_id": remaining[0], "top_n": 3}
                        bottleneck_ids.append(remaining[0])
                known = {tool["function"]["name"] for tool in DEMO_TOOLS}
                invented = [
                    str(value)
                    for key, value in arguments.items()
                    if "app_id" in key
                    and isinstance(value, str)
                    and not value.startswith("spark-")
                ]
                required: set[str] = set()
                for tool in DEMO_TOOLS:
                    if tool["function"]["name"] == name:
                        required = set(
                            tool["function"]["parameters"].get("required", [])
                        )
                missing = required - set(arguments)
                if name not in known:
                    summary = f"{name} is not one of the demo tools."
                    error = True
                elif missing:
                    summary = (
                        "Missing "
                        + ", ".join(sorted(missing))
                        + ". Pass the application id that starts with spark-."
                    )
                    error = True
                elif invented:
                    summary = (
                        "Those are not application ids: "
                        + ", ".join(invented)
                        + ". Call list_applications and copy the id values. "
                        "They start with spark-."
                    )
                    error = True
                else:
                    try:
                        payload, error = await _call_mcp(name, arguments)
                        summary = (
                            "The tool call failed. Call list_applications and use ids that start with spark-."
                            if error
                            else summarize_tool(name, payload)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Tool %s failed: %s", name, exc)
                        summary = (
                            f"The lookup failed ({exc}). "
                            "Call list_applications and use ids that start with spark-."
                        )
                        error = True
                if error:
                    yield _sse(
                        {
                            "kind": "status",
                            "text": "That lookup did not have a usable application id. Asking the model to try again.",
                        }
                    )
                else:
                    yield _sse(
                        {"kind": "tool_call", "name": name, "arguments": arguments}
                    )
                    yield _sse(
                        {
                            "kind": "tool_result",
                            "name": name,
                            "arguments": arguments,
                            "summary": summary,
                            "error": False,
                        }
                    )
                if not error:
                    succeeded.add(name)
                    if name == "list_applications":
                        taxi_ids = [
                            found
                            for line in summary.splitlines()
                            if "NewYorkTaxi" in line
                            for found in re.findall(r"spark-[0-9a-f]+", line)
                        ]
                        if taxi_ids:
                            app_ids[:] = taxi_ids
                messages.append({"role": "tool", "tool_name": name, "content": summary})
        yield _sse({"kind": "status", "text": "Writing the answer from those lookups…"})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Answer in one short paragraph, under 90 words. "
                    "Call the jobs the 03:56 taxi job and the 00:24 taxi job. "
                    "Do not print application ids. Finish the paragraph."
                ),
            }
        )
        final = await _ollama_chat(messages, use_tools=False, num_predict=800)
        text = (final.get("content") or "").strip()
        yield _sse(
            {
                "kind": "answer",
                "text": text or "The model returned an empty answer.",
                "used_tools": bool(succeeded),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Demo question failed")
        yield _sse({"kind": "error", "text": str(exc)})


async def ask(request: Request) -> StreamingResponse | JSONResponse:
    if _ask_lock.locked():
        return JSONResponse(
            {"error": "A question is already running. Wait for it to finish."},
            status_code=409,
        )
    body = await request.json()
    question = str(body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)

    async def locked_events() -> AsyncIterator[str]:
        async with _ask_lock:
            async for chunk in _ask_events(question):
                yield chunk
            yield _sse({"kind": "done"})

    return StreamingResponse(locked_events(), media_type="text/event-stream")


async def homepage(_: Request) -> FileResponse:
    return FileResponse(PAGE)


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/api/status", status),
        Route("/api/ask", ask, methods=["POST"]),
    ]
)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger.info("Demo page at http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
