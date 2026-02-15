from src.agents.planner import plan
from src.agents.executor import execute

def run_graph(user_input: str) -> str:
    steps = plan(user_input)
    return execute(steps)
