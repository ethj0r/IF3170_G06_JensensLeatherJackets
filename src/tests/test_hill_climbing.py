from pathlib import Path

import pytest

from algorithms import hill_climbing
from core.models import load_problem
from core.objective import is_feasible, objective

PROBLEM = load_problem(Path(__file__).parent.parent / "data" / "tiny.json")


def test_hasil_feasible_dan_konsisten():
    result = hill_climbing.run(PROBLEM, seed=0)
    assert result.algorithm == "hill_climbing_steepest"
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


def test_stochastic_history_dan_iterasi():
    result = hill_climbing.run_stochastic(PROBLEM, seed=10, max_iter=50)
    assert result.algorithm == "hill_climbing_stochastic"
    assert result.params["max_stuck"] == 100
    assert result.iterations <= 50
    assert len(result.history) == result.iterations + 1
    assert result.extra["proposals"] >= result.iterations
    assert result.feasible == is_feasible(result.final_state, PROBLEM)


def test_stochastic_menerima_langkah():
    accepted = [
        hill_climbing.run_stochastic(PROBLEM, seed=seed, max_iter=500).iterations
        for seed in range(5)
    ]
    assert all(count >= 1 for count in accepted)


def test_stochastic_rejection_budget():
    result = hill_climbing.run_stochastic(
        PROBLEM, seed=20, max_iter=10_000, max_stuck=3
    )
    assert result.iterations < 10_000
    assert result.extra["proposals"] >= result.iterations + 3
    assert len(result.history) == result.iterations + 1


def test_stochastic_klasik_satu_penolakan():
    result = hill_climbing.run_stochastic(PROBLEM, seed=21, max_iter=10_000, max_stuck=1)
    assert result.iterations < 10_000
    assert result.extra["proposals"] == result.iterations + 1


def test_stochastic_seed_sama_hasil_sama():
    a = hill_climbing.run_stochastic(PROBLEM, seed=11, max_iter=30)
    b = hill_climbing.run_stochastic(PROBLEM, seed=11, max_iter=30)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_stochastic_max_iter_nol():
    result = hill_climbing.run_stochastic(PROBLEM, seed=13, max_iter=0)
    assert result.iterations == 0
    assert result.extra["proposals"] == 0
    assert len(result.history) == 1


def test_stochastic_parameter_tidak_valid():
    with pytest.raises(ValueError):
        hill_climbing.run_stochastic(PROBLEM, seed=0, max_iter=-1)
    with pytest.raises(ValueError):
        hill_climbing.run_stochastic(PROBLEM, seed=0, max_stuck=-1)


def test_sideways_nol_sama_dengan_steepest():
    a = hill_climbing.run(PROBLEM, seed=14, max_iter=100)
    b = hill_climbing.run_sideways(PROBLEM, seed=14, max_iter=100, max_sideways=0)
    assert b.final_state == a.final_state
    assert b.history == a.history
    assert b.iterations == a.iterations


def test_sideways_history_dan_hitungan():
    result = hill_climbing.run_sideways(PROBLEM, seed=15, max_iter=100, max_sideways=10)
    assert result.algorithm == "hill_climbing_sideways"
    assert result.iterations <= 100
    assert len(result.history) == result.iterations + 1
    assert 0 <= result.extra["sideways_moves"] <= result.iterations
    assert result.extra["max_sideways"] == 10
    assert result.feasible == is_feasible(result.final_state, PROBLEM)


def test_sideways_seed_sama_hasil_sama():
    a = hill_climbing.run_sideways(PROBLEM, seed=16, max_iter=40, max_sideways=3)
    b = hill_climbing.run_sideways(PROBLEM, seed=16, max_iter=40, max_sideways=3)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_sideways_parameter_tidak_valid():
    with pytest.raises(ValueError):
        hill_climbing.run_sideways(PROBLEM, seed=0, max_sideways=-1)
    with pytest.raises(ValueError):
        hill_climbing.run_sideways(PROBLEM, seed=0, max_iter=-1)


def test_restart_nol_sama_dengan_steepest():
    a = hill_climbing.run(PROBLEM, seed=17, max_iter=100)
    b = hill_climbing.run_restart(PROBLEM, seed=17, max_iter=100, max_restarts=0)
    assert b.final_state == a.final_state
    assert b.history == a.history
    assert b.iterations == a.iterations
    assert b.extra["restarts"] == 0


def test_restart_struktur():
    result = hill_climbing.run_restart(PROBLEM, seed=18, max_iter=50, max_restarts=3)
    extra = result.extra
    assert result.algorithm == "hill_climbing_restart"
    assert extra["restarts"] <= 3
    per_restart = extra["iterations_per_restart"]
    assert len(per_restart) == extra["restarts"] + 1
    assert sum(per_restart) == result.iterations
    assert len(result.history) == result.iterations + extra["restarts"] + 1
    boundaries = extra["restart_boundaries"]
    assert len(boundaries) == extra["restarts"] + 1
    assert boundaries[0] == 0
    assert boundaries == sorted(boundaries)
    assert result.feasible == is_feasible(result.final_state, PROBLEM)


def test_restart_seed_sama_hasil_sama():
    a = hill_climbing.run_restart(PROBLEM, seed=19, max_iter=30, max_restarts=2)
    b = hill_climbing.run_restart(PROBLEM, seed=19, max_iter=30, max_restarts=2)
    assert a.final_state == b.final_state
    assert a.history == b.history


def test_restart_parameter_tidak_valid():
    with pytest.raises(ValueError):
        hill_climbing.run_restart(PROBLEM, seed=0, max_restarts=-1)
    with pytest.raises(ValueError):
        hill_climbing.run_restart(PROBLEM, seed=0, max_iter=-1)
