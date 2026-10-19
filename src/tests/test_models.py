import json
import random
from pathlib import Path

from algorithms.result import SearchResult
from core.models import load_problem
from core.neighbors import random_state

TINY = Path(__file__).parent.parent / "data" / "tiny.json"
LARGE = Path(__file__).parent.parent / "data" / "large.json"


def test_load_problem():
    problem = load_problem(TINY)
    assert problem.name == "tiny"
    assert problem.truck.max_capacity == 100
    assert len(problem.packages) == 8
    assert problem.packages["P07"].is_fragile


# Memastikan dataset besar valid dan cukup variatif untuk eksperimen tambahan.
def test_load_large_problem():
    problem = load_problem(LARGE)
    packages = list(problem.packages.values())

    assert problem.name == "large"
    assert len(packages) == 45
    assert sum(package.weight for package in packages) > problem.truck.max_capacity
    assert sum(package.w * package.l * package.h for package in packages) > (
        problem.truck.w * problem.truck.l * problem.truck.h
    )
    assert any(package.is_fragile for package in packages)
    assert any(not package.is_fragile for package in packages)
    assert len({package.dims for package in packages}) >= 40


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


# Memastikan file hasil lama tanpa objective_name tetap dapat dibuka.
def test_search_result_loads_legacy_total_value_file(tmp_path):
    problem = load_problem(TINY)
    state = random_state(problem, random.Random(1))
    result = SearchResult(
        algorithm="legacy",
        params={},
        seed=1,
        problem=problem.name,
        initial_state=state,
        final_state=state,
        final_objective=0.0,
        feasible=True,
        history=[0.0],
        iterations=0,
        duration_s=0.0,
    )
    path = tmp_path / "legacy.json"
    result.save(path)
    raw = json.loads(path.read_text())
    raw.pop("objective_name")
    path.write_text(json.dumps(raw))

    loaded = SearchResult.load(path)

    assert loaded.objective_name == "total_value"
