import os
import json
import uuid
from typing import List, Dict, Any

import requests

# Optional (only needed in AgentCore mode)
try:
    import boto3
except Exception:
    boto3 = None

CONTRACT_VERSION = "v1"

# Local/dev HTTP gateway
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL", "http://localhost:8080")

# AgentCore hosted Tool Gateway runtime ARN (Pattern A)
TOOL_GATEWAY_RUNTIME_ARN = os.getenv("TOOL_GATEWAY_RUNTIME_ARN")  # e.g. arn:aws:bedrock-agentcore:...:runtime/hosted_agent_...

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TOOL_GATEWAY_QUALIFIER = os.getenv("TOOL_GATEWAY_QUALIFIER", "DEFAULT")

# AgentCore requires runtimeSessionId length >= 33
def _new_session_id() -> str:
    return f"session-{uuid.uuid4()}-{uuid.uuid4()}"  # safely > 33 chars


def _invoke_gateway_agentcore(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TOOL_GATEWAY_RUNTIME_ARN:
        raise RuntimeError("TOOL_GATEWAY_RUNTIME_ARN is not set")
    if boto3 is None:
        raise RuntimeError("boto3 not available in this environment")

    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    resp = client.invoke_agent_runtime(
        agentRuntimeArn=TOOL_GATEWAY_RUNTIME_ARN,
        runtimeSessionId=_new_session_id(),
        payload=json.dumps(payload),
        qualifier=TOOL_GATEWAY_QUALIFIER,  # DEFAULT is fine
    )

    raw = resp["response"].read()
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise RuntimeError("Empty response from Tool Gateway runtime")

    return json.loads(text)


def _invoke_gateway_http(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        f"{TOOL_GATEWAY_URL}/tools/invoke",
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _invoke_gateway(payload: Dict[str, Any]) -> Dict[str, Any]:
    # If TOOL_GATEWAY_RUNTIME_ARN is set, assume we are in AgentCore-to-AgentCore mode.
    if TOOL_GATEWAY_RUNTIME_ARN:
        return _invoke_gateway_agentcore(payload)
    return _invoke_gateway_http(payload)


def search_kb(query: str) -> List[Dict[str, Any]]:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "tool_name": "search_kb",
        "input": {"query": query},
        "tenant_id": os.getenv("TENANT_ID"),
        "user_id": os.getenv("USER_ID"),
        "correlation_id": os.getenv("CORRELATION_ID"),
    }

    body = _invoke_gateway(payload)

    if body.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("Tool Gateway contract version mismatch")

    if not body.get("ok"):
        error = body.get("error", {})
        raise RuntimeError(error.get("message", "Tool call failed"))

    output = body.get("output", {})
    results = output.get("results")
    if results is None:
        raise RuntimeError("Malformed tool response: missing 'results'")

    return results
