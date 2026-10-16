import random
from pathlib import Path

from algorithms.genetic import (
    move_mutation,
    mutate,
    rotate_mutation,
    run,
    swap_mutation,
    tournament_selection,
    uniform_crossover,
)
from core.models import State, load_problem
from core.objective import is_feasible, objective
from core.packing import random_feasible_state

TINY = Path(__file__).parent.parent / "data" / "tiny.json"
IDENTITY = (0, 1, 2)


def full_tiny_state() -> State:
    coordinates = [
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 0),
        (2, 2, 0),
        (0, 0, 1),
        (2, 0, 1),
        (0, 2, 1),
        (2, 2, 1),
    ]
    return {
        package_id: (*coordinate, IDENTITY)
        for package_id, coordinate in zip(
            (f"P{index:02d}" for index in range(1, 9)),
            coordinates,
        )
    }


def assert_value_error(callable_object) -> None:
    try:
        callable_object()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_tournament_selection_returns_best_when_every_individual_competes():
    population = [
        {"A": None},
        {"A": (0, 0, 0, IDENTITY)},
        {"A": (1, 0, 0, IDENTITY)},
    ]
    fitnesses = [0.0, 10.0, 5.0]

    selected = tournament_selection(
        population,
        fitnesses,
        random.Random(0),
        tournament_size=3,
    )

    assert selected == population[1]
    assert selected is not population[1]


def test_uniform_crossover_uses_only_parent_genes():
    problem = load_problem(TINY)
    parent_a = full_tiny_state()
    parent_b: State = {package_id: None for package_id in problem.packages}

    child = uniform_crossover(parent_a, parent_b, random.Random(5))

    assert set(child) == set(problem.packages)
    assert all(
        child[package_id] in (parent_a[package_id], parent_b[package_id])
        for package_id in child
    )
    assert child != parent_a
    assert child != parent_b


def test_mutation_operators_do_not_mutate_input_and_return_feasible_states():
    problem = load_problem(TINY)
    state = full_tiny_state()
    original = dict(state)

    results = [
        swap_mutation(state, problem, random.Random(1)),
        move_mutation(state, problem, random.Random(2)),
        rotate_mutation(state, problem, random.Random(3)),
        mutate(state, problem, random.Random(4)),
    ]

    assert state == original
    assert all(is_feasible(result, problem) for result in results)


def test_mutate_keeps_random_feasible_states_feasible():
    problem = load_problem(TINY)

    for seed in range(20):
        state = random_feasible_state(problem, random.Random(seed))
        mutated = mutate(state, problem, random.Random(seed + 100))
        assert is_feasible(mutated, problem)


def test_run_returns_complete_result_required_by_specification():
    problem = load_problem(TINY)
    result = run(
        problem,
        seed=42,
        population_size=12,
        iterations=10,
        elite_count=2,
    )

    assert result.algorithm == "genetic"
    assert result.problem == problem.name
    assert result.seed == 42
    assert result.params["population_size"] == 12
    assert result.params["iterations"] == 10
    assert result.iterations == 10
    assert result.feasible
    assert is_feasible(result.initial_state, problem)
    assert is_feasible(result.final_state, problem)
    assert result.final_objective == objective(result.final_state, problem)
    assert len(result.history) == 11
    assert len(result.extra["average_history"]) == 11
    assert result.history[0] == result.extra["initial_best_objective"]
    assert result.final_objective == max(result.history)
    assert result.duration_s >= 0.0


def test_elitism_makes_maximum_history_non_decreasing():
    problem = load_problem(TINY)
    result = run(
        problem,
        seed=7,
        population_size=10,
        iterations=12,
        elite_count=1,
    )

    assert all(
        current <= following
        for current, following in zip(result.history, result.history[1:])
    )
    assert all(
        average <= maximum
        for average, maximum in zip(
            result.extra["average_history"],
            result.history,
        )
    )


def test_run_is_reproducible_except_for_duration():
    problem = load_problem(TINY)
    parameters = {
        "population_size": 10,
        "iterations": 8,
        "elite_count": 1,
    }

    first = run(problem, seed=99, **parameters)
    second = run(problem, seed=99, **parameters)

    assert first.initial_state == second.initial_state
    assert first.final_state == second.final_state
    assert first.final_objective == second.final_objective
    assert first.history == second.history
    assert first.extra == second.extra


def test_zero_iterations_returns_best_initial_individual():
    problem = load_problem(TINY)
    result = run(
        problem,
        seed=12,
        population_size=6,
        iterations=0,
        elite_count=1,
    )

    assert result.initial_state == result.final_state
    assert result.final_objective == result.history[0]
    assert result.extra["best_iteration"] == 0


def test_invalid_parameters_are_rejected():
    problem = load_problem(TINY)

    assert_value_error(lambda: run(problem, 0, population_size=1))
    assert_value_error(lambda: run(problem, 0, iterations=-1))
    assert_value_error(lambda: run(problem, 0, crossover_rate=1.1))
    assert_value_error(lambda: run(problem, 0, mutation_rate=-0.1))
    assert_value_error(lambda: run(problem, 0, tournament_size=101))
    assert_value_error(lambda: run(problem, 0, elite_count=100))
    assert_value_error(lambda: run(problem, 0, inclusion_probability=-0.1))
