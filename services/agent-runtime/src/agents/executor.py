from src.tools.bindings import search_kb

def execute(steps: list[str]) -> str:
    # Minimal: treat first step as a KB query
    query = steps[0] if steps else ""
    results = search_kb(query) if query else []
    return f"RESULTS: {results}"
