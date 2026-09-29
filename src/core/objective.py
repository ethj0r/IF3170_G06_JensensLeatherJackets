from itertools import permutations
from core.models import Orientation, Package, Placement, Problem, State, Truck

ORIENTATIONS: list[Orientation] = list(permutations(range(3)))

def oriented_dims(dims: tuple[int, int, int], orientation: Orientation) -> tuple[int, int, int]:
    return (dims[orientation[0]], dims[orientation[1]], dims[orientation[2]])

def rotate(orientation: Orientation, axis: str) -> Orientation:
    p0, p1, p2 = orientation
    if axis == "x":
        return (p0, p2, p1)
    if axis == "y":
        return (p2, p1, p0)
    if axis == "z":
        return (p1, p0, p2)
    raise ValueError(f"Axis unknown: {axis}")

def box(pkg: Package, placement: Placement) -> tuple[int, int, int, int, int, int]:
    x, y, z, orientation = placement
    dx, dy, dz = oriented_dims(pkg.dims, orientation)
    return x, y, z, dx, dy, dz

def in_bounds(pkg: Package, placement: Placement, truck: Truck) -> bool:
    x, y, z, dx, dy, dz = box(pkg, placement)
    return (
        x >= 0 and y >= 0 and z >= 0
        and x+dx <= truck.w
        and y+dy <= truck.l
        and z+dz <= truck.h
    )

def objective(state: State, problem: Problem) -> float:
    return float(sum(
        problem.packages[pid].value
        for pid, placement in state.items()
        if placement is not None
    ))

def is_feasible(state: State, problem: Problem) -> bool:
    return all(
        in_bounds(problem.packages[pid], placement, problem.truck)
        for pid, placement in state.items()
        if placement is not None
    )