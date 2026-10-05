from itertools import count
import random
import time

from algorithms.result import SearchResult
from core.models import Problem, State
from core.neighbors import random_state
from core.objective import ORIENTATIONS, box, objective, oriented_dims, rotate

EPSILON = 1e-9

def get_violations(state: State, problem: Problem) -> int:
    truck = problem.truck
    boxes = []
    weight = 0
    for pid, placement in state.items():
        if placement is None:
            continue
        pkg = problem.packages[pid]
        x, y, z, dx, dy, dz = box(pkg, placement)
        if (x < 0) or (y < 0) or (z < 0) or (x + dx > truck.w) or (y + dy > truck.l) or (z + dz > truck.h):
            return 1_000_000
        boxes.append((x, y, z, dx, dy, dz, pkg.is_fragile))
        weight += pkg.weight
    count = 0
    for i in range(len(boxes)):
        xi, yi, zi, dxi, dyi, dzi, _ = boxes[i]
        for j in range(i + 1, len(boxes)):
            xj, yj, zj, dxj, dyj, dzj, _ = boxes[j]
            if (
                xi < xj + dxj and xj < xi + dxi
                and yi < yj + dyj and yj < yi + dyi
                and zi < zj + dzj and zj < zi + dzi
            ):
                count += 1
    
    if weight > truck.max_capacity:
        count += 1

    for x, y, z, dx, dy, dz, _ in boxes:
        if z == 0:
            continue
        for px in range(x, x + dx):
            for py in range(y, y+ dy):
                supported = False
                for bx, by, bz, bdx, bdy, bdz, fragile in boxes:
                    if fragile:
                        continue
                    if (bz + bdz == z) and (bx <= px < bx + bdx) and (by <= py < by + bdy):
                        supported = True
                        break
                if not supported:
                    count += 1
    return count


def get_penalty_weight(problem: Problem) -> float:
    return float(sum(pkg.value for pkg in problem.packages.values()) + 1)

def get_score(state: State, problem: Problem, penalty: float) -> float:
    return objective(state, problem) - penalty*get_violations(state, problem)

def get_neighbors(state: State, problem: Problem):
    pids = list(state)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if state[pids[i]] == state[pids[j]]:
                continue
            neighbor = dict(state)
            neighbor[pids[i]], neighbor[pids[j]] = state[pids[j]], state[pids[i]]
            yield neighbor
    truck = problem.truck

    for pid, current in state.items():
        pkg = problem.packages[pid]
        for orientation in ORIENTATIONS:
            dx, dy, dz = oriented_dims(pkg.dims, orientation)
            if dx > truck.w or dy > truck.l or dz > truck.h:
                continue
            for x in range(truck.w - dx + 1):
                for y in range(truck.l - dy + 1):
                    for z in range(truck.h - dz + 1):
                        placement = (x, y, z, orientation)
                        if placement == current:
                            continue
                        neighbor = dict(state)
                        neighbor[pid] = placement
                        yield neighbor
            if current is not None:
                neighbor = dict(state)
                neighbor[pid] = None
                yield neighbor
            else:
                continue

            x, y, z, orientation = current
            for axis in ("x", "y", "z"):
                new_orientation = rotate(orientation, axis)
                if new_orientation == orientation:
                    continue
                ndx, ndy, ndz = oriented_dims(pkg.dims, new_orientation)
                if x + ndx > truck.w or y + ndy > truck.l or z + ndz > truck.h:
                    continue
                neighbor = dict(state)
                neighbor[pid] = (x, y, z, new_orientation)
                yield neighbor

def run(problem: Problem, seed: int, max_iter: int, **params) -> SearchResult:
    # raise NotImplementedError
    
    start = time.perf_counter()
    init_state = random_state(problem, random.Random(seed))
    penalty = get_penalty_weight(problem)

    state = dict(init_state)
    best_score = get_score(state, problem, penalty)
    history = [objective(state, problem)]
    iterations = 0

    while iterations < max_iter:
        best_neighbor = None
        for candidate in get_neighbors(state, problem):
            candidate_score = get_score(candidate, problem, penalty)
            if candidate_score > best_score + EPSILON:
                best_score = candidate_score
                best_neighbor = candidate
        if best_neighbor is None:
            break
        state = best_neighbor
        iterations += 1
        history.append(objective(state, problem))
    duration_s = time.perf_counter() - start
    violations = get_violations(state, problem)
    return SearchResult(
        algorithm = "hc_steepest",
        params    = {"variant": "steepest", "max_iter": max_iter, **params},
        seed      = seed,
        problem   = problem.name,
        initial_state = init_state,
        final_state = state,
        final_objective = objective(state, problem),
        feasible  = (violations==0),
        history   = history,
        iterations = iterations,
        duration_s = duration_s,
        extra     = {
            "violations": violations,
            "initial_objective": history[0],
            "initial_score": get_score(init_state, problem, penalty),
            "final_score": best_score,
        },
    )
