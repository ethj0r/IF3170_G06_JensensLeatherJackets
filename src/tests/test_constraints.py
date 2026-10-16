from core.models import Package, Problem, State, Truck
from core.objective import (
    is_feasible,
    objective,
    placements_overlap,
    supports,
    total_weight,
)

IDENTITY = (0, 1, 2)


def make_problem(max_capacity: int = 30) -> Problem:
    packages = {
        "A": Package("A", 2, 2, 1, value=50, weight=10),
        "B": Package("B", 2, 2, 1, value=40, weight=10),
        "F": Package("F", 2, 2, 1, value=30, weight=5, is_fragile=True),
    }
    return Problem(
        name="constraints",
        truck=Truck(w=6, l=4, h=3, max_capacity=max_capacity),
        packages=packages,
    )


def make_state(**placements) -> State:
    state: State = {"A": None, "B": None, "F": None}
    state.update(placements)
    return state


def test_objective_and_weight_only_count_packages_inside():
    problem = make_problem()
    state = make_state(A=(0, 0, 0, IDENTITY), F=(2, 0, 0, IDENTITY))

    assert objective(state, problem) == 80.0
    assert total_weight(state, problem) == 15


def test_valid_packages_on_floor_are_feasible():
    problem = make_problem()
    state = make_state(
        A=(0, 0, 0, IDENTITY),
        B=(2, 0, 0, IDENTITY),
        F=(4, 0, 0, IDENTITY),
    )

    assert is_feasible(state, problem)


def test_packages_may_touch_without_overlapping():
    problem = make_problem()
    package_a = problem.packages["A"]
    package_b = problem.packages["B"]
    placement_a = (0, 0, 0, IDENTITY)
    placement_b = (2, 0, 0, IDENTITY)

    assert not placements_overlap(package_a, placement_a, package_b, placement_b)
    assert is_feasible(make_state(A=placement_a, B=placement_b), problem)


def test_overlapping_packages_are_infeasible():
    problem = make_problem()
    state = make_state(
        A=(0, 0, 0, IDENTITY),
        B=(1, 1, 0, IDENTITY),
    )

    assert not is_feasible(state, problem)


def test_package_outside_truck_is_infeasible():
    problem = make_problem()
    state = make_state(A=(5, 0, 0, IDENTITY))

    assert not is_feasible(state, problem)


def test_capacity_may_equal_but_not_exceed_maximum():
    problem = make_problem(max_capacity=20)
    exactly_at_capacity = make_state(
        A=(0, 0, 0, IDENTITY),
        B=(2, 0, 0, IDENTITY),
    )
    over_capacity = make_state(
        A=(0, 0, 0, IDENTITY),
        B=(2, 0, 0, IDENTITY),
        F=(4, 0, 0, IDENTITY),
    )

    assert is_feasible(exactly_at_capacity, problem)
    assert not is_feasible(over_capacity, problem)


def test_floating_package_is_infeasible():
    problem = make_problem()
    state = make_state(B=(0, 0, 1, IDENTITY))

    assert not is_feasible(state, problem)


def test_non_fragile_package_can_support_an_upper_package():
    problem = make_problem()
    lower = (0, 0, 0, IDENTITY)
    upper = (0, 0, 1, IDENTITY)

    assert supports(problem.packages["A"], lower, problem.packages["B"], upper)
    assert is_feasible(make_state(A=lower, B=upper), problem)


def test_partial_base_support_matches_specification():
    problem = make_problem()
    lower = (0, 0, 0, IDENTITY)
    upper = (1, 1, 1, IDENTITY)

    assert supports(problem.packages["A"], lower, problem.packages["B"], upper)
    assert is_feasible(make_state(A=lower, B=upper), problem)


def test_touching_only_at_base_edge_is_not_support():
    problem = make_problem()
    lower = (0, 0, 0, IDENTITY)
    upper = (2, 0, 1, IDENTITY)

    assert not supports(problem.packages["A"], lower, problem.packages["B"], upper)
    assert not is_feasible(make_state(A=lower, B=upper), problem)


def test_fragile_package_may_be_upper_but_may_not_support_another_package():
    problem = make_problem()
    normal_below_fragile = make_state(
        A=(0, 0, 0, IDENTITY),
        F=(0, 0, 1, IDENTITY),
    )
    fragile_below_normal = make_state(
        F=(0, 0, 0, IDENTITY),
        B=(0, 0, 1, IDENTITY),
    )

    assert is_feasible(normal_below_fragile, problem)
    assert not is_feasible(fragile_below_normal, problem)


def test_any_contact_with_fragile_support_is_infeasible():
    normal = Package("N", 2, 2, 1, value=10, weight=1)
    fragile = Package("F", 2, 2, 1, value=10, weight=1, is_fragile=True)
    wide = Package("W", 4, 2, 1, value=10, weight=1)
    problem = Problem(
        name="mixed-support",
        truck=Truck(w=4, l=2, h=2, max_capacity=3),
        packages={package.id: package for package in (normal, fragile, wide)},
    )
    state: State = {
        "N": (0, 0, 0, IDENTITY),
        "F": (2, 0, 0, IDENTITY),
        "W": (0, 0, 1, IDENTITY),
    }

    assert not is_feasible(state, problem)


def test_state_must_map_every_package_exactly_once():
    problem = make_problem()

    assert not is_feasible({"A": None, "B": None}, problem)
    assert not is_feasible({**make_state(), "UNKNOWN": None}, problem)


def test_coordinates_must_be_integers_and_orientation_must_be_valid():
    problem = make_problem()

    assert not is_feasible(make_state(A=(0.0, 0, 0, IDENTITY)), problem)
    assert not is_feasible(make_state(A=(0, 0, 0, (0, 0, 1))), problem)
