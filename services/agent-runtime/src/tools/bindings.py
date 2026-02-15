import os
import json
import uuid
from typing import List, Dict, Any

import requests

# Contract version (frozen handshake between agent and gateway)
CONTRACT_VERSION = "v1"

# Mode switch:
# - "http" (default): call Tool Gateway over HTTP (local dev / docker-compose)
# - "agentcore": call Tool Gateway hosted runtime via bedrock-agentcore InvokeAgentRuntime
TOOL_GATEWAY_MODE = os.getenv("TOOL_GATEWAY_MODE", "http").lower()

# HTTP mode settings
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL", "http://localhost:8080")

# AgentCore mode settings
TOOL_GATEWAY_RUNTIME_ARN = os.getenv("TOOL_GATEWAY_RUNTIME_ARN")  # required for agentcore mode
TOOL_GATEWAY_QUALIFIER = os.getenv("TOOL_GATEWAY_QUALIFIER", "DEFAULT")
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


def _new_runtime_session_id() -> str:
    # AgentCore requires 33+ chars; uuid4 hex is 32, so prefix to be safe.
    return "session-" + uuid.uuid4().hex  # 40 chars


def _call_tool_gateway_http(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{TOOL_GATEWAY_URL}/tools/invoke",
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def _call_tool_gateway_agentcore(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TOOL_GATEWAY_RUNTIME_ARN:
        raise RuntimeError("TOOL_GATEWAY_RUNTIME_ARN is required when TOOL_GATEWAY_MODE=agentcore")

    import boto3  # imported here so local HTTP users don't need boto3 installed

    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    # Tool-gateway container should accept this JSON payload (via /invocations mapping)
    body = json.dumps(payload)

    response = client.invoke_agent_runtime(
        agentRuntimeArn=TOOL_GATEWAY_RUNTIME_ARN,
        runtimeSessionId=_new_runtime_session_id(),
        payload=body,
        qualifier=TOOL_GATEWAY_QUALIFIER,  # DEFAULT
    )

    # AgentCore returns a streaming body in response["response"]
    raw = response["response"].read()
    data = json.loads(raw)

    return data


def _call_tool_gateway(payload: Dict[str, Any]) -> Dict[str, Any]:
    if TOOL_GATEWAY_MODE == "agentcore":
        return _call_tool_gateway_agentcore(payload)
    return _call_tool_gateway_http(payload)


def search_kb(query: str) -> List[Dict[str, Any]]:
    """
    Calls the Tool Gateway search_kb tool.
    Works in two modes:
      - HTTP mode (local): POST http://TOOL_GATEWAY_URL/tools/invoke
      - AgentCore mode: InvokeAgentRuntime -> Tool Gateway hosted runtime
    """
    payload = {
        "contract_version": CONTRACT_VERSION,
        "tool_name": "search_kb",
        "input": {"query": query},
        "tenant_id": os.getenv("TENANT_ID"),
        "user_id": os.getenv("USER_ID"),
        "correlation_id": os.getenv("CORRELATION_ID"),
    }

    body = _call_tool_gateway(payload)

    # Defensive checks
    if body.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("Tool Gateway contract version mismatch")

    if not body.get("ok"):
        error = body.get("error", {}) or {}
        raise RuntimeError(error.get("message", "Tool call failed"))

    output = body.get("output", {}) or {}
    results = output.get("results")

    if results is None:
        raise RuntimeError("Malformed tool response: missing 'results'")

    return results
