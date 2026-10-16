import random

from core.models import Package, Problem, State, Truck
from core.neighbors import random_state
from core.objective import is_feasible, objective
from core.packing import (
    can_place,
    find_valid_placement,
    random_feasible_state,
    repair_state,
)

IDENTITY = (0, 1, 2)


def make_problem(max_capacity: int = 30) -> Problem:
    packages = {
        "A": Package("A", 2, 2, 1, value=50, weight=10),
        "B": Package("B", 2, 2, 1, value=40, weight=10),
        "F": Package("F", 2, 2, 1, value=30, weight=5, is_fragile=True),
    }
    return Problem(
        name="packing",
        truck=Truck(w=6, l=4, h=3, max_capacity=max_capacity),
        packages=packages,
    )


def empty_state(problem: Problem) -> State:
    return {package_id: None for package_id in problem.packages}


def test_can_place_accepts_valid_and_rejects_invalid_placement():
    problem = make_problem()
    state = empty_state(problem)

    assert can_place(state, "A", (0, 0, 0, IDENTITY), problem)
    assert not can_place(state, "A", (0, 0, 1, IDENTITY), problem)
    assert not can_place(state, "A", (5, 0, 0, IDENTITY), problem)


def test_can_place_rejects_overlap_and_fragile_support():
    problem = make_problem()
    normal_state = empty_state(problem)
    normal_state["A"] = (0, 0, 0, IDENTITY)
    fragile_state = empty_state(problem)
    fragile_state["F"] = (0, 0, 0, IDENTITY)

    assert not can_place(normal_state, "B", (1, 1, 0, IDENTITY), problem)
    assert can_place(normal_state, "B", (0, 0, 1, IDENTITY), problem)
    assert not can_place(fragile_state, "B", (0, 0, 1, IDENTITY), problem)


def test_find_valid_placement_uses_floor_for_empty_truck():
    problem = make_problem()
    state = empty_state(problem)

    placement = find_valid_placement(state, "A", problem, random.Random(7))

    assert placement is not None
    assert placement[2] == 0
    assert can_place(state, "A", placement, problem)


def test_find_valid_placement_can_stack_on_non_fragile_package():
    packages = {
        "A": Package("A", 2, 2, 1, value=10, weight=1),
        "B": Package("B", 2, 2, 1, value=10, weight=1),
    }
    problem = Problem(
        name="stacking",
        truck=Truck(w=2, l=2, h=2, max_capacity=2),
        packages=packages,
    )
    state: State = {"A": (0, 0, 0, IDENTITY), "B": None}

    placement = find_valid_placement(state, "B", problem, random.Random(3))

    assert placement is not None
    assert placement[2] == 1
    assert can_place(state, "B", placement, problem)


def test_find_valid_placement_returns_none_when_capacity_is_full():
    problem = make_problem(max_capacity=20)
    state = empty_state(problem)
    state["A"] = (0, 0, 0, IDENTITY)
    state["B"] = (2, 0, 0, IDENTITY)

    assert find_valid_placement(state, "F", problem, random.Random(0)) is None


def test_random_feasible_state_is_reproducible_and_always_feasible():
    problem = make_problem()
    first = random_feasible_state(problem, random.Random(42))
    second = random_feasible_state(problem, random.Random(42))

    assert first == second
    assert is_feasible(first, problem)
    for seed in range(20):
        assert is_feasible(random_feasible_state(problem, random.Random(seed)), problem)


def test_zero_inclusion_probability_keeps_every_package_outside():
    problem = make_problem()
    state = random_feasible_state(
        problem,
        random.Random(0),
        inclusion_probability=0.0,
    )

    assert state == empty_state(problem)


def test_invalid_inclusion_probability_is_rejected():
    problem = make_problem()

    try:
        random_feasible_state(problem, random.Random(0), inclusion_probability=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("inclusion_probability above one must be rejected")


def test_repair_resolves_overlap_and_keeps_all_fitting_packages():
    problem = make_problem()
    invalid: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": (0, 0, 0, IDENTITY),
        "F": (0, 0, 0, IDENTITY),
    }

    repaired = repair_state(invalid, problem, random.Random(4))

    assert is_feasible(repaired, problem)
    assert objective(repaired, problem) == 120.0


def test_repair_enforces_capacity():
    problem = make_problem(max_capacity=20)
    overweight: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": (2, 0, 0, IDENTITY),
        "F": (4, 0, 0, IDENTITY),
    }

    repaired = repair_state(overweight, problem, random.Random(8))

    assert is_feasible(repaired, problem)
    assert sum(placement is not None for placement in repaired.values()) == 2


def test_repair_does_not_mutate_input():
    problem = make_problem()
    invalid: State = {
        "A": (0, 0, 1, IDENTITY),
        "B": (0, 0, 1, IDENTITY),
        "F": None,
    }
    original = dict(invalid)

    repair_state(invalid, problem, random.Random(9))

    assert invalid == original


def test_repair_preserves_valid_state_when_outside_fill_is_disabled():
    problem = make_problem()
    state: State = {
        "A": (0, 0, 0, IDENTITY),
        "B": None,
        "F": None,
    }

    repaired = repair_state(
        state,
        problem,
        random.Random(2),
        fill_outside=False,
    )

    assert repaired == state


def test_repair_normalizes_missing_and_unknown_package_ids():
    problem = make_problem()
    malformed = {
        "A": (0, 0, 0, IDENTITY),
        "UNKNOWN": (2, 0, 0, IDENTITY),
    }

    repaired = repair_state(
        malformed,
        problem,
        random.Random(1),
        fill_outside=False,
    )

    assert set(repaired) == set(problem.packages)
    assert repaired["A"] == malformed["A"]
    assert repaired["B"] is None
    assert repaired["F"] is None
    assert is_feasible(repaired, problem)


def test_repair_handles_random_invalid_states():
    problem = make_problem()

    for seed in range(20):
        invalid = random_state(problem, random.Random(seed))
        repaired = repair_state(invalid, problem, random.Random(seed))
        assert is_feasible(repaired, problem)
