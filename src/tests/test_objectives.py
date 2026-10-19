import random

import pytest

from algorithms import genetic, hill_climbing, simulated_annealing
from core.models import Package, Problem, State, Truck
from core.objective import is_feasible, objective, penalized_objective
from core.objectives import (
    DEFAULT_OBJECTIVE,
    PRIORITY_OBJECTIVE,
    resolve_objective,
    state_metrics,
)
from core.packing import repair_state

IDENTITY = (0, 1, 2)


# Membuat problem kecil dengan konflik antara raw value dan urgensi ETA.
def make_priority_problem() -> Problem:
    return Problem(
        name="priority",
        truck=Truck(w=2, l=1, h=1, max_capacity=10),
        packages={
            "A": Package("A", 1, 1, 1, value=120, weight=10, eta=5),
            "B": Package("B", 1, 1, 1, value=100, weight=10, eta=0),
        },
    )


# Memastikan total_value tetap identik dengan objective utama sebelumnya.
def test_total_value_remains_default_objective():
    problem = make_priority_problem()
    state: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": None,
    }

    assert objective(state, problem) == 120.0
    assert objective(state, problem, DEFAULT_OBJECTIVE) == 120.0


# Memastikan priority_value memberi utility lebih tinggi kepada ETA lebih kecil.
def test_priority_value_uses_value_and_eta():
    problem = make_priority_problem()
    evaluator = resolve_objective(problem, PRIORITY_OBJECTIVE)

    assert evaluator.package_utilities["A"] == 120.0
    assert evaluator.package_utilities["B"] == 200.0
    assert evaluator.package_utilities["B"] > evaluator.package_utilities["A"]


# Memastikan posisi dan orientasi tidak mengubah nilai objective paket.
def test_priority_value_depends_only_on_loaded_packages():
    problem = make_priority_problem()
    first: State = {"A": None, "B": (0, 0, 0, IDENTITY)}
    second: State = {"A": None, "B": (1, 0, 0, (1, 0, 2))}

    assert objective(first, problem, PRIORITY_OBJECTIVE) == 200.0
    assert objective(second, problem, PRIORITY_OBJECTIVE) == 200.0


# Memastikan ETA seragam nol membuat priority_value sama dengan total_value.
def test_priority_value_handles_zero_eta_scale():
    problem = Problem(
        name="zero_eta",
        truck=Truck(2, 1, 1, 10),
        packages={
            "A": Package("A", 1, 1, 1, value=30, weight=1, eta=0),
            "B": Package("B", 1, 1, 1, value=40, weight=1, eta=0),
        },
    )
    state: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": (1, 0, 0, IDENTITY),
    }

    assert objective(state, problem, PRIORITY_OBJECTIVE) == 70.0


# Memastikan nama objective tidak dikenal dan ETA negatif ditolak dengan jelas.
def test_invalid_objective_configuration_is_rejected():
    problem = make_priority_problem()
    with pytest.raises(ValueError, match="Unknown objective"):
        resolve_objective(problem, "unknown")

    invalid_problem = Problem(
        name="invalid_eta",
        truck=problem.truck,
        packages={
            "A": Package("A", 1, 1, 1, value=10, weight=1, eta=-1),
        },
    )
    with pytest.raises(ValueError, match="non-negative ETA"):
        resolve_objective(invalid_problem, PRIORITY_OBJECTIVE)


# Memastikan penalized score memakai objective yang dipilih.
def test_penalized_objective_supports_priority_value():
    problem = make_priority_problem()
    feasible: State = {"A": None, "B": (0, 0, 0, IDENTITY)}

    assert penalized_objective(
        feasible,
        problem,
        objective_name=PRIORITY_OBJECTIVE,
    ) == 200.0


# Memastikan repair menggunakan utility objective, bukan selalu raw value.
def test_repair_priority_matches_selected_objective():
    problem = make_priority_problem()
    conflicting: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": (0, 0, 0, IDENTITY),
    }

    value_repair = repair_state(
        conflicting,
        problem,
        random.Random(0),
        objective_name=DEFAULT_OBJECTIVE,
    )
    priority_repair = repair_state(
        conflicting,
        problem,
        random.Random(0),
        objective_name=PRIORITY_OBJECTIVE,
    )

    assert value_repair["A"] is not None
    assert value_repair["B"] is None
    assert priority_repair["A"] is None
    assert priority_repair["B"] is not None
    assert is_feasible(priority_repair, problem)


# Memastikan metrik pembanding tetap melaporkan raw value, berat, jumlah, dan ETA.
def test_state_metrics_are_independent_from_selected_objective():
    problem = make_priority_problem()
    state: State = {"A": None, "B": (0, 0, 0, IDENTITY)}

    assert state_metrics(state, problem) == {
        "total_value": 100.0,
        "total_weight": 10,
        "loaded_packages": 1,
        "average_eta": 0.0,
    }


# Memastikan ketiga algoritma mengoptimalkan dan mencatat objective bonus.
@pytest.mark.parametrize(
    "runner, params",
    [
        (hill_climbing.run, {"max_iter": 5}),
        (simulated_annealing.run, {"max_iter": 50}),
        (genetic.run, {"population_size": 6, "iterations": 3}),
    ],
)
def test_algorithms_support_priority_objective(runner, params):
    problem = make_priority_problem()
    result = runner(
        problem,
        seed=7,
        objective_name=PRIORITY_OBJECTIVE,
        **params,
    )

    assert result.objective_name == PRIORITY_OBJECTIVE
    assert result.final_objective == objective(
        result.final_state,
        problem,
        PRIORITY_OBJECTIVE,
    )
    assert result.extra["final_metrics"] == state_metrics(
        result.final_state,
        problem,
    )
