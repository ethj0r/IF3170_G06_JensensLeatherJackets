import random
from collections.abc import Iterator

from core.models import Package, Placement, Problem, State, Truck
from core.objective import ORIENTATIONS, oriented_dims, in_bounds
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
    # (0, 0, 0) + sudut kanan, depan dan atas setiap packacge di dalam truckn
    truck = problem.truck
    points = {(0, 0, 0)}
    for pid, placement in state.items():
        if placement is None or pid == exclude: 
            continue
        x, y, z, orientation = placement
        dx, dy, dz = oriented_dims(problem.packages[pid].dims, orientation)
        points.update([(x+dx, y, z), (x, y+dy, z), (x, y, z+dz)])
    return sorted(p for p in points if p[0] < truck.w and p[1] < truck.l and p[2])

def random_placement(pkg: Package, truck: Truck, rng: random.Random) -> Placement | None:
    orientation = rng.choice(ORIENTATIONS)
    dx, dy, dz = oriented_dims(pkg.dims, orientation)
    if dx > truck.w or dy > truck.l or dz > truck.h:
        return None
    return (
        rng.randint(0, truck.w - dx),
        rng.randint(0, truck.l - dy),
        rng.randint(0, truck.h - dz),
        orientation,
    )


def random_state(problem: Problem, rng: random.Random) -> State:
    return {
        pid: random_placement(pkg, problem.truck, rng) if rng.random() < 0.5 else None
        for pid, pkg in problem.packages.items()
    }


def random_neighbor(state: State, problem: Problem, rng: random.Random) -> State:
    pid = rng.choice(list(state))
    neighbor = dict(state)
    neighbor[pid] = None if state[pid] is not None else random_placement(
        problem.packages[pid], problem.truck, rng
    )
    return neighbor


def all_neighbors(state: State, problem: Problem) -> Iterator[State]:
    raise NotImplementedError
