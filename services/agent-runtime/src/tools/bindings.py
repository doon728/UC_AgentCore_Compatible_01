import os
import requests
from typing import List, Dict, Any

# Base URL of the Tool Gateway service
TOOL_GATEWAY_URL = os.getenv("TOOL_GATEWAY_URL", "http://localhost:8080")

# Contract version (frozen handshake between agent and gateway)
CONTRACT_VERSION = "v1"


def search_kb(query: str) -> List[Dict[str, Any]]:
    """
    Calls the Tool Gateway search_kb tool.

    This function is part of the Agent ↔ Tool Gateway contract.
    It sends a standardized request envelope and expects
    a standardized response envelope.
    """

    payload = {
        "contract_version": CONTRACT_VERSION,
        "tool_name": "search_kb",
        "input": {"query": query},
        "tenant_id": os.getenv("TENANT_ID"),
        "user_id": os.getenv("USER_ID"),
        "correlation_id": os.getenv("CORRELATION_ID"),
    }

    response = requests.post(
        f"{TOOL_GATEWAY_URL}/tools/invoke",
        json=payload,
        timeout=10,
    )

    response.raise_for_status()
    body = response.json()

    # Validate contract version (defensive check)
    if body.get("contract_version") != CONTRACT_VERSION:
        raise RuntimeError("Tool Gateway contract version mismatch")

    # Validate success flag
    if not body.get("ok"):
        error = body.get("error", {})
        raise RuntimeError(error.get("message", "Tool call failed"))

    # Defensive output validation
    output = body.get("output", {})
    results = output.get("results")

    if results is None:
        raise RuntimeError("Malformed tool response: missing 'results'")

    return results
