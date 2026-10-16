import random
from pathlib import Path

from core.models import State, load_problem
from core.neighbors import (
    OUTSIDE,
    all_neighbors,
    extreme_points,
    random_neighbor,
    random_state,
    relocate,
    rotate,
    swap,
)
from core.objective import in_bounds

PROBLEM = load_problem(Path(__file__).parent.parent / "data" / "tiny.json")  # truk 4x4x2, paket 2x2x1
O = (0, 1, 2)


def empty_state() -> State:
    return {pid: None for pid in PROBLEM.packages}


def with_placements(**placements) -> State:
    state = empty_state()
    state.update(placements)
    return state


def assert_valid(state: State) -> None:
    assert set(state) == set(PROBLEM.packages)
    for pid, placement in state.items():
        if placement is not None:
            assert in_bounds(PROBLEM.packages[pid], placement, PROBLEM.truck)


# --- extreme points ---

def test_extreme_points_truk_kosong():
    assert extreme_points(empty_state(), PROBLEM) == [(0, 0, 0)]


def test_extreme_points_satu_paket():
    state = with_placements(P01=(0, 0, 0, O))
    assert extreme_points(state, PROBLEM) == [(0, 0, 0), (0, 0, 1), (0, 2, 0), (2, 0, 0)]


def test_extreme_points_di_luar_truk_dibuang():
    state = with_placements(P01=(2, 2, 1, O))  # semua sudutnya menyentuh dinding
    assert extreme_points(state, PROBLEM) == [(0, 0, 0)]


def test_extreme_points_exclude():
    state = with_placements(P01=(0, 0, 0, O))
    assert extreme_points(state, PROBLEM, exclude="P01") == [(0, 0, 0)]


# --- swap ---

def test_swap_dalam_dalam_orientasi_ikut_paket():
    state = with_placements(P01=(0, 0, 0, O), P02=(2, 0, 0, (1, 0, 2)))
    neighbor = swap(state, PROBLEM, "P01", "P02")
    assert neighbor["P01"] == (2, 0, 0, O)
    assert neighbor["P02"] == (0, 0, 0, (1, 0, 2))


def test_swap_dalam_luar():
    state = with_placements(P01=(0, 0, 0, O))
    neighbor = swap(state, PROBLEM, "P01", "P02")
    assert neighbor["P01"] is None
    assert neighbor["P02"] == (0, 0, 0, O)


def test_swap_luar_luar_tidak_valid():
    assert swap(empty_state(), PROBLEM, "P01", "P02") is None


# --- relocate ---

def test_relocate_ke_titik_dan_keluar():
    state = with_placements(P01=(0, 0, 0, O))
    assert relocate(state, PROBLEM, "P01", (2, 0, 0))["P01"] == (2, 0, 0, O)
    assert relocate(state, PROBLEM, "P01", OUTSIDE)["P01"] is None


def test_relocate_tidak_valid():
    state = with_placements(P01=(0, 0, 0, O))
    assert relocate(state, PROBLEM, "P01", (3, 0, 0)) is None   # 3 + 2 > 4
    assert relocate(state, PROBLEM, "P01", (0, 0, 0)) is None   # tidak berubah
    assert relocate(state, PROBLEM, "P02", OUTSIDE) is None     # sudah di luar


# --- rotate ---

def test_rotate():
    state = with_placements(P01=(0, 0, 0, O))
    assert rotate(state, PROBLEM, "P01", "x")["P01"] == (0, 0, 0, (0, 2, 1))


def test_rotate_tidak_valid():
    assert rotate(with_placements(P01=(0, 0, 0, O)), PROBLEM, "P01", "z") is None  # 2x2 diputar = sama
    assert rotate(with_placements(P01=(0, 0, 1, O)), PROBLEM, "P01", "x") is None  # 1 + 2 > 2
    assert rotate(empty_state(), PROBLEM, "P01", "x") is None                     # di luar truk


# --- random & all ---

def test_random_walk_selalu_valid_dan_berubah():
    rng = random.Random(0)
    state = random_state(PROBLEM, rng)
    assert_valid(state)
    for _ in range(500):
        before = dict(state)
        neighbor = random_neighbor(state, PROBLEM, rng)
        assert state == before   # state asal tidak ikut berubah
        assert neighbor != state
        assert_valid(neighbor)
        state = neighbor


def test_all_neighbors_valid_dan_berubah():
    state = random_state(PROBLEM, random.Random(1))
    neighbors = list(all_neighbors(state, PROBLEM))
    assert neighbors
    for neighbor in neighbors:
        assert neighbor != state
        assert_valid(neighbor)


def test_seed_sama_hasil_sama():
    a = random_state(PROBLEM, random.Random(42))
    b = random_state(PROBLEM, random.Random(42))
    assert a == b
    assert random_neighbor(a, PROBLEM, random.Random(7)) == random_neighbor(b, PROBLEM, random.Random(7))
