import random
from dataclasses import replace
from pathlib import Path

from core.models import State, load_problem
from core.neighbors import random_neighbor, random_state
from core.objective import (
    default_penalty_weight,
    is_feasible,
    objective,
    penalized_objective,
    violations,
)

PROBLEM = load_problem(Path(__file__).parent.parent / "data" / "tiny.json")
O = (0, 1, 2)


def with_placements(**placements) -> State:
    state: State = {pid: None for pid in PROBLEM.packages}
    state.update(placements)
    return state


FULL = with_placements(
    P01=(0, 0, 0, O), P02=(2, 0, 0, O), P03=(0, 2, 0, O), P04=(2, 2, 0, O),
    P05=(0, 0, 1, O), P06=(2, 0, 1, O), P07=(0, 2, 1, O), P08=(2, 2, 1, O),
)


def test_state_kosong_tanpa_pelanggaran():
    assert violations(with_placements(), PROBLEM) == 0


def test_optimum_tiny_feasible():
    assert violations(FULL, PROBLEM) == 0
    assert is_feasible(FULL, PROBLEM)
    assert penalized_objective(FULL, PROBLEM) == 350


def test_overlap_dihitung_volume():
    state = with_placements(P01=(0, 0, 0, O), P02=(0, 0, 0, O))
    assert violations(state, PROBLEM) == 4


def test_melayang():
    assert violations(with_placements(P01=(0, 0, 1, O)), PROBLEM) == 1


def test_cukup_satu_titik_tersangga():
    state = with_placements(P01=(0, 0, 0, O), P02=(1, 1, 1, O))
    assert violations(state, PROBLEM) == 0


def test_di_atas_fragile():
    state = with_placements(P07=(0, 0, 0, O), P01=(0, 0, 1, O))
    assert violations(state, PROBLEM) == 1


def test_kelebihan_berat():
    problem = replace(PROBLEM, truck=replace(PROBLEM.truck, max_capacity=15))
    state = with_placements(P01=(0, 0, 0, O), P02=(2, 0, 0, O))
    assert violations(state, problem) == 5


def test_penalized_objective():
    state = with_placements(P01=(0, 0, 1, O))
    assert penalized_objective(state, PROBLEM, weight=10) == 50 - 10
    assert penalized_objective(state, PROBLEM) == 50 - default_penalty_weight(PROBLEM)


def test_violations_nol_sama_dengan_feasible():
    rng = random.Random(0)
    state = random_state(PROBLEM, rng)
    for _ in range(1000):
        assert (violations(state, PROBLEM) == 0) == is_feasible(state, PROBLEM)
        assert penalized_objective(state, PROBLEM) <= objective(state, PROBLEM)
        state = random_neighbor(state, PROBLEM, rng)
