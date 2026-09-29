import random
from pathlib import Path

from algorithms.result import SearchResult
from core.models import load_problem
from core.neighbors import random_state

TINY = Path(__file__).parent.parent / "data" / "tiny.json"


def test_load_problem():
    problem = load_problem(TINY)
    assert problem.name == "tiny"
    assert problem.truck.max_capacity == 100
    assert len(problem.packages) == 8
    assert problem.packages["P07"].is_fragile


def test_search_result_save_load(tmp_path):
    problem = load_problem(TINY)
    state = random_state(problem, random.Random(0))
    result = SearchResult(
        algorithm="random",
        params={},
        seed=0,
        problem=problem.name,
        initial_state=state,
        final_state=state,
        final_objective=1.0,
        feasible=True,
        history=[1.0],
        iterations=0,
        duration_s=0.0,
    )

    path = tmp_path / "result.json"
    result.save(path)
    assert SearchResult.load(path) == result
