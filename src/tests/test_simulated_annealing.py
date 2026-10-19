from pathlib import Path

import pytest

from algorithms import simulated_annealing
from core.models import Problem, Truck, load_problem
from core.objective import is_feasible, objective

PROBLEM = load_problem(Path(__file__).parent.parent / "data" / "tiny.json")


def test_hasil_feasible_dan_konsisten():
    result = simulated_annealing.run(PROBLEM, seed=0)
    assert result.feasible
    assert is_feasible(result.final_state, PROBLEM)
    assert result.final_objective == objective(result.final_state, PROBLEM)
    assert result.final_objective == result.extra["best_feasible_history"][-1]


def test_panjang_history_sama():
    result = simulated_annealing.run(PROBLEM, seed=1, max_iter=300)
    n = result.iterations + 1
    assert result.iterations == 300
    assert len(result.history) == n
    assert len(result.extra["temperature"]) == n
    assert len(result.extra["acceptance_prob"]) == n
    assert len(result.extra["search_score_history"]) == n
    assert result.history[0] == objective(result.initial_state, PROBLEM)


def test_acceptance_prob_dan_suhu():
    result = simulated_annealing.run(PROBLEM, seed=2, max_iter=2000)
    probs = [p for p in result.extra["acceptance_prob"] if p is not None]
    assert probs
    assert all(0 < p <= 1 for p in probs)
    temps = result.extra["temperature"]
    assert all(a > b for a, b in zip(temps, temps[1:]))


def test_best_tidak_pernah_turun():
    best = simulated_annealing.run(PROBLEM, seed=3).extra["best_feasible_history"]
    assert all(a <= b for a, b in zip(best, best[1:]))


def test_seed_sama_hasil_sama():
    a = simulated_annealing.run(PROBLEM, seed=4, max_iter=500)
    b = simulated_annealing.run(PROBLEM, seed=4, max_iter=500)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_parameter_tidak_valid():
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, alpha=1.0)
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, t0=0.001)
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, t_min=-1)
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, max_iter=-1)
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, stuck_threshold=0)
    with pytest.raises(ValueError):
        simulated_annealing.run(PROBLEM, seed=0, penalty_weight=0)


# Memastikan problem tanpa paket berhenti dengan solusi kosong yang feasible.
def test_problem_tanpa_paket_berhenti_dengan_aman():
    problem = Problem("empty", Truck(1, 1, 1, 0), {})

    result = simulated_annealing.run(problem, seed=0, max_iter=10)

    assert result.iterations == 0
    assert result.final_state == {}
    assert result.final_objective == 0.0
    assert result.feasible
