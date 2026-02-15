from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.graph.build_graph import run_graph

app = FastAPI(title="Agent Runtime", version="v1")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "agent-runtime", "version": "v1"}


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    """
    AgentCore container contract endpoint.
    Whatever AgentCore sends as payload, we interpret a prompt and run the graph.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    prompt = ""
    if isinstance(payload, dict):
        prompt = payload.get("prompt") or payload.get("text") or ""

    # Default prompt fallback (prevents empty input crashes)
    if not prompt:
        prompt = "hello"

    # Run your graph ONLY per-request (not at container start)
    try:
        result = run_graph(prompt)
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": {"code": "RUNTIME_ERROR", "message": str(e)}},
        )

    # Keep response JSON-safe
    if isinstance(result, (dict, list)):
        out = result
    else:
        out = {"result": str(result)}

    return JSONResponse(status_code=200, content={"ok": True, "output": out})
