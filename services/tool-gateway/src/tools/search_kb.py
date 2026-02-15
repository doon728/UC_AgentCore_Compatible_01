from typing import Dict, Any


def search_kb(query: str) -> Dict[str, Any]:
    # Stubbed response for now. Later you’ll call Bedrock KB / OpenSearch / vector DB here.
    results = []
    if query:
        results = [
            {"id": "doc-001", "title": "Sample KB Doc", "score": 0.87, "snippet": f"Matched on: {query}"}
        ]
    return {"results": results}
