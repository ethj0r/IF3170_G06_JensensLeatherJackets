import random
from collections.abc import Iterator

from core.models import Placement, Problem, State
from core.objective import ORIENTATIONS, in_bounds, oriented_dims
from core.objective import rotate as rotate_orientation

Point = tuple[int, int, int]

OUTSIDE = None
AXES = ("x", "y", "z")
DEFAULT_ORIENTATION = (0, 1, 2)
MAX_TRIES = 100


def _fits(problem: Problem, pid: str, placement: Placement | None) -> bool:
    return placement is None or in_bounds(problem.packages[pid], placement, problem.truck)


def _apply(state: State, changes: dict[str, Placement | None]) -> State:
    neighbor = dict(state)
    neighbor.update(changes)
    return neighbor


def extreme_points(state: State, problem: Problem, exclude: str | None = None) -> list[Point]:
    # (0, 0, 0) + sudut kanan, depan, dan atas setiap paket di dalam truk
    truck = problem.truck
    points = {(0, 0, 0)}
    for pid, placement in state.items():
        if placement is None or pid == exclude:
            continue
        x, y, z, orientation = placement
        dx, dy, dz = oriented_dims(problem.packages[pid].dims, orientation)
        points.update([(x + dx, y, z), (x, y + dy, z), (x, y, z + dz)])
    return sorted(p for p in points if p[0] < truck.w and p[1] < truck.l and p[2] < truck.h)


def swap(state: State, problem: Problem, a: str, b: str) -> State | None:
    pa, pb = state[a], state[b]
    if a == b or (pa is None and pb is None):
        return None

    # Posisi ditukar, orientasi tetap ikut paket masing-masing.
    new_a = None if pb is None else (pb[0], pb[1], pb[2], pa[3] if pa else DEFAULT_ORIENTATION)
    new_b = None if pa is None else (pa[0], pa[1], pa[2], pb[3] if pb else DEFAULT_ORIENTATION)

    if (new_a, new_b) == (pa, pb):
        return None
    if not (_fits(problem, a, new_a) and _fits(problem, b, new_b)):
        return None
    return _apply(state, {a: new_a, b: new_b})


def relocate(state: State, problem: Problem, pid: str, target: Point | None) -> State | None:
    current = state[pid]
    if target is OUTSIDE:
        new = None
    else:
        orientation = current[3] if current else DEFAULT_ORIENTATION
        new = (target[0], target[1], target[2], orientation)

    if new == current or not _fits(problem, pid, new):
        return None
    return _apply(state, {pid: new})


def rotate(state: State, problem: Problem, pid: str, axis: str) -> State | None:
    current = state[pid]
    if current is None:
        return None

    x, y, z, orientation = current
    new_orientation = rotate_orientation(orientation, axis)
    dims = problem.packages[pid].dims
    if oriented_dims(dims, new_orientation) == oriented_dims(dims, orientation):
        return None  # bentuk tidak berubah (mis. sisi yang diputar sama panjang)

    new = (x, y, z, new_orientation)
    if not _fits(problem, pid, new):
        return None
    return _apply(state, {pid: new})


def random_state(problem: Problem, rng: random.Random, p_insert: float = 0.5) -> State:
    # Tiap paket (urutan acak) berpeluang p_insert masuk di extreme point acak.
    # Tidak mengecek overlap: state awal boleh tidak valid, penalti yang membereskan.
    state: State = {pid: None for pid in problem.packages}
    order = list(problem.packages)
    rng.shuffle(order)

    for pid in order:
        if rng.random() >= p_insert:
            continue
        point = rng.choice(extreme_points(state, problem))
        placement = (point[0], point[1], point[2], rng.choice(ORIENTATIONS))
        if _fits(problem, pid, placement):
            state[pid] = placement
    return state


def random_neighbor(state: State, problem: Problem, rng: random.Random) -> State:
    ids = list(state)
    for _ in range(MAX_TRIES):
        move = rng.choice(("swap", "relocate", "rotate"))
        if move == "swap":
            a, b = rng.sample(ids, 2)
            neighbor = swap(state, problem, a, b)
        elif move == "relocate":
            pid = rng.choice(ids)
            targets = [*extreme_points(state, problem, exclude=pid), OUTSIDE]
            neighbor = relocate(state, problem, pid, rng.choice(targets))
        else:
            neighbor = rotate(state, problem, rng.choice(ids), rng.choice(AXES))

        if neighbor is not None:
            return neighbor
    raise RuntimeError(f"Tidak menemukan neighbor valid setelah {MAX_TRIES} percobaan")


def all_neighbors(state: State, problem: Problem) -> Iterator[State]:
    ids = list(state)

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            neighbor = swap(state, problem, a, b)
            if neighbor is not None:
                yield neighbor

    for pid in ids:
        for target in [*extreme_points(state, problem, exclude=pid), OUTSIDE]:
            neighbor = relocate(state, problem, pid, target)
            if neighbor is not None:
                yield neighbor

    for pid in ids:
        for axis in AXES:
            neighbor = rotate(state, problem, pid, axis)
            if neighbor is not None:
                yield neighbor
