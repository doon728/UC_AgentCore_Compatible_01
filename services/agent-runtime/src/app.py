from __future__ import annotations

from src.graph.build_graph import run_graph


def main() -> None:
    # Minimal entrypoint for local smoke usage.
    # Example: poetry run python -m src.app
    text = "hello"
    result = run_graph(text)
    print(result)


if __name__ == "__main__":
    main()
