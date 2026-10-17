from pathlib import Path

import pytest

from algorithms import hill_climbing
from core.models import load_problem
from core.objective import is_feasible, objective

PROBLEM = load_problem(Path(__file__).parent.parent / "data" / "tiny.json")


def test_hasil_feasible_dan_konsisten():
    result = hill_climbing.run(PROBLEM, seed=0)
    assert result.algorithm == "hill_climbing"
    assert result.feasible == is_feasible(result.final_state, PROBLEM)
    assert result.feasible
    assert result.extra["violations"] == 0
    assert result.final_objective == objective(result.final_state, PROBLEM)


def test_panjang_history_sama():
    result = hill_climbing.run(PROBLEM, seed=1, max_iter=2)
    assert result.iterations <= 2
    assert len(result.history) == result.iterations + 1


def test_score_tidak_menurun():
    result = hill_climbing.run(PROBLEM, seed=2, max_iter=50)
    assert result.extra["final_score"] >= result.extra["initial_score"]


def test_seed_sama_hasil_sama():
    a = hill_climbing.run(PROBLEM, seed=3, max_iter=20)
    b = hill_climbing.run(PROBLEM, seed=3, max_iter=20)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_berhenti_di_local_optimum():
    result = hill_climbing.run(PROBLEM, seed=4, max_iter=10_000)
    assert result.iterations < 10_000
    assert len(result.history) == result.iterations + 1


def test_max_iter_nol():
    result = hill_climbing.run(PROBLEM, seed=5, max_iter=0)
    assert result.iterations == 0
    assert len(result.history) == 1


def test_parameter_tidak_valid():
    with pytest.raises(ValueError):
        hill_climbing.run(PROBLEM, seed=0, max_iter=-1)
