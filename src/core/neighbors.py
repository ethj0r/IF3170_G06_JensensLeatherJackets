import random
from collections.abc import Iterator

from core.fleet import join_placement, project_truck_state, split_placement
from core.models import Placement, Problem, State, StatePlacement
from core.objective import ORIENTATIONS, in_bounds, oriented_dims
from core.objective import rotate as rotate_orientation

Point = tuple[int, int, int]
FleetPoint = tuple[str, int, int, int]
Target = Point | FleetPoint | None

OUTSIDE = None
AXES = ("x", "y", "z")
DEFAULT_ORIENTATION = (0, 1, 2)
MAX_TRIES = 100


def _fits(problem: Problem, pid: str, placement: StatePlacement | None) -> bool:
    if placement is None:
        return True
    try:
        truck_id, local_placement = split_placement(placement, problem)
    except (TypeError, ValueError):
        return False
    return in_bounds(
        problem.packages[pid],
        local_placement,
        problem.truck_for(truck_id),
    )


def _apply(state: State, changes: dict[str, StatePlacement | None]) -> State:
    neighbor = dict(state)
    neighbor.update(changes)
    return neighbor


def _local_extreme_points(
    state: State,
    problem: Problem,
    exclude: str | None = None,
) -> list[Point]:
    truck = problem.truck
    points = {(0, 0, 0)}
    for pid, placement in state.items():
        if placement is None or pid == exclude:
            continue
        x, y, z, orientation = placement
        dx, dy, dz = oriented_dims(problem.packages[pid].dims, orientation)
        points.update([(x + dx, y, z), (x, y + dy, z), (x, y, z + dz)])
    return sorted(p for p in points if p[0] < truck.w and p[1] < truck.l and p[2] < truck.h)


# Menghasilkan extreme points seluruh armada atau satu truk tertentu.
def extreme_points(
    state: State,
    problem: Problem,
    exclude: str | None = None,
    truck_id: str | None = None,
) -> list[Point] | list[FleetPoint]:
    if not problem.is_multi_truck:
        return _local_extreme_points(state, problem, exclude)

    selected_ids = problem.truck_ids if truck_id is None else (truck_id,)
    points: list[FleetPoint] = []
    for selected_id in selected_ids:
        local_problem = Problem(
            problem.name,
            problem.truck_for(selected_id),
            problem.packages,
        )
        local_state = project_truck_state(state, problem, selected_id)
        points.extend(
            (selected_id, x, y, z)
            for x, y, z in _local_extreme_points(
                local_state,
                local_problem,
                exclude,
            )
        )
    return points


def swap(state: State, problem: Problem, a: str, b: str) -> State | None:
    pa, pb = state[a], state[b]
    if a == b or (pa is None and pb is None):
        return None

    # Posisi dan truk tujuan ditukar, sedangkan orientasi tetap ikut paket.
    if pb is None:
        new_a = None
    else:
        destination_truck, destination = split_placement(pb, problem)
        orientation = (
            split_placement(pa, problem)[1][3]
            if pa is not None
            else DEFAULT_ORIENTATION
        )
        new_a = join_placement(
            destination_truck,
            (destination[0], destination[1], destination[2], orientation),
            problem,
        )

    if pa is None:
        new_b = None
    else:
        destination_truck, destination = split_placement(pa, problem)
        orientation = (
            split_placement(pb, problem)[1][3]
            if pb is not None
            else DEFAULT_ORIENTATION
        )
        new_b = join_placement(
            destination_truck,
            (destination[0], destination[1], destination[2], orientation),
            problem,
        )

    if (new_a, new_b) == (pa, pb):
        return None
    if not (_fits(problem, a, new_a) and _fits(problem, b, new_b)):
        return None
    return _apply(state, {a: new_a, b: new_b})


def relocate(state: State, problem: Problem, pid: str, target: Target) -> State | None:
    current = state[pid]
    if target is OUTSIDE:
        new = None
    else:
        orientation = (
            split_placement(current, problem)[1][3]
            if current is not None
            else DEFAULT_ORIENTATION
        )
        if problem.is_multi_truck:
            truck_id, x, y, z = target
            new = join_placement(
                truck_id,
                (x, y, z, orientation),
                problem,
            )
        else:
            x, y, z = target
            new = (x, y, z, orientation)

    if new == current or not _fits(problem, pid, new):
        return None
    return _apply(state, {pid: new})


def rotate(state: State, problem: Problem, pid: str, axis: str) -> State | None:
    current = state[pid]
    if current is None:
        return None

    truck_id, local_placement = split_placement(current, problem)
    x, y, z, orientation = local_placement
    new_orientation = rotate_orientation(orientation, axis)
    dims = problem.packages[pid].dims
    if oriented_dims(dims, new_orientation) == oriented_dims(dims, orientation):
        return None  # bentuk tidak berubah (mis. sisi yang diputar sama panjang)

    new = join_placement(
        truck_id,
        (x, y, z, new_orientation),
        problem,
    )
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
        if problem.is_multi_truck:
            truck_id, x, y, z = point
            placement = join_placement(
                truck_id,
                (x, y, z, rng.choice(ORIENTATIONS)),
                problem,
            )
        else:
            x, y, z = point
            placement = (x, y, z, rng.choice(ORIENTATIONS))
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
