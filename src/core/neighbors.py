import random
from collections.abc import Iterator

from core.models import Package, Placement, Problem, State, Truck
from core.objective import ORIENTATIONS, oriented_dims


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
