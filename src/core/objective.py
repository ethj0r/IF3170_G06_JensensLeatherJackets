from itertools import permutations

from core.fleet import fleet_weights, project_truck_state, split_placement
from core.models import Orientation, Package, Placement, Problem, State, Truck
from core.objectives import DEFAULT_OBJECTIVE, resolve_objective

ORIENTATIONS: list[Orientation] = list(permutations(range(3)))


# Menghasilkan dimensi paket setelah orientasi diterapkan sebagai permutasi sumbu.
def oriented_dims(dims: tuple[int, int, int], orientation: Orientation) -> tuple[int, int, int]:
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Invalid orientation: {orientation}")
    return (dims[orientation[0]], dims[orientation[1]], dims[orientation[2]])


# Merotasi orientasi paket 90 derajat terhadap sumbu x, y, atau z.
def rotate(orientation: Orientation, axis: str) -> Orientation:
    p0, p1, p2 = orientation
    if axis == "x":
        return (p0, p2, p1)
    if axis == "y":
        return (p2, p1, p0)
    if axis == "z":
        return (p1, p0, p2)
    raise ValueError(f"Axis unknown: {axis}")


# Menghasilkan posisi dan dimensi aktual paket dalam bentuk (x, y, z, dx, dy, dz).
def box(pkg: Package, placement: Placement) -> tuple[int, int, int, int, int, int]:
    x, y, z, orientation = placement
    dx, dy, dz = oriented_dims(pkg.dims, orientation)
    return x, y, z, dx, dy, dz


# Memeriksa apakah seluruh bagian paket berada di dalam batas peti kemas.
def in_bounds(pkg: Package, placement: Placement, truck: Truck) -> bool:
    x, y, z, dx, dy, dz = box(pkg, placement)
    return (
        x >= 0 and y >= 0 and z >= 0
        and x + dx <= truck.w
        and y + dy <= truck.l
        and z + dz <= truck.h
    )


# Memeriksa apakah volume bagian dalam dua paket saling beririsan.
# Paket yang hanya bersentuhan pada sisi, rusuk, atau titik sudut tidak dianggap overlap.
def placements_overlap(
    pkg_a: Package,
    placement_a: Placement,
    pkg_b: Package,
    placement_b: Placement,
) -> bool:
    ax, ay, az, adx, ady, adz = box(pkg_a, placement_a)
    bx, by, bz, bdx, bdy, bdz = box(pkg_b, placement_b)
    return (
        ax < bx + bdx and bx < ax + adx
        and ay < by + bdy and by < ay + ady
        and az < bz + bdz and bz < az + adz
    )


# Memeriksa apakah bagian atas paket bawah menopang minimal satu bagian alas paket atas.
# Sesuai spesifikasi, seluruh alas paket atas tidak harus tertopang.
def supports(
    lower_pkg: Package,
    lower_placement: Placement,
    upper_pkg: Package,
    upper_placement: Placement,
) -> bool:
    lx, ly, lz, ldx, ldy, ldz = box(lower_pkg, lower_placement)
    ux, uy, uz, udx, udy, _ = box(upper_pkg, upper_placement)
    return (
        lz + ldz == uz
        and lx < ux + udx and ux < lx + ldx
        and ly < uy + udy and uy < ly + ldy
    )


# Menghitung total berat semua paket yang berada di dalam peti kemas.
def total_weight(state: State, problem: Problem) -> int:
    return sum(
        problem.packages[pid].weight
        for pid, placement in state.items()
        if placement is not None
    )


# Menghitung total berat paket pada satu truk tertentu.
def truck_weight(state: State, problem: Problem, truck_id: str) -> int:
    return fleet_weights(state, problem)[truck_id]


# Menghitung objective terpilih; total value tetap menjadi perilaku default.
def objective(
    state: State,
    problem: Problem,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> float:
    return resolve_objective(problem, objective_name).evaluate(state)


# Memeriksa seluruh hard constraint dan kelengkapan representasi state.
# Paket dengan placement None dianggap berada di luar peti kemas.
def is_feasible(state: State, problem: Problem) -> bool:
    if set(state) != set(problem.packages):
        return False

    if problem.is_multi_truck:
        for placement in state.values():
            if placement is None:
                continue
            try:
                _, local_placement = split_placement(placement, problem)
            except (TypeError, ValueError):
                return False
            if not is_well_formed_placement(local_placement):
                return False

        for truck_id, truck in problem.fleet.items():
            local_problem = Problem(problem.name, truck, problem.packages)
            local_state = project_truck_state(state, problem, truck_id)
            if not _is_single_truck_feasible(local_state, local_problem):
                return False
        return True

    return _is_single_truck_feasible(state, problem)


# Memeriksa constraint geometri dan kapasitas pada tepat satu truk.
def _is_single_truck_feasible(state: State, problem: Problem) -> bool:
    if set(state) != set(problem.packages):
        return False

    placed: list[tuple[str, Package, Placement]] = []
    for pid, placement in state.items():
        if placement is None:
            continue

        if not is_well_formed_placement(placement):
            return False

        package = problem.packages[pid]
        if not in_bounds(package, placement, problem.truck):
            return False
        placed.append((pid, package, placement))

    if total_weight(state, problem) > problem.truck.max_capacity:
        return False

    # Memastikan tidak ada dua paket yang menempati volume peti kemas yang sama.
    for index, (_, package_a, placement_a) in enumerate(placed):
        for _, package_b, placement_b in placed[index + 1:]:
            if placements_overlap(package_a, placement_a, package_b, placement_b):
                return False

    # Memastikan setiap paket ditopang lantai atau paket non-fragile di bawahnya.
    for upper_pid, upper_package, upper_placement in placed:
        if upper_placement[2] == 0:
            continue

        direct_supporters = [
            lower_package
            for lower_pid, lower_package, lower_placement in placed
            if lower_pid != upper_pid
            and supports(
                lower_package,
                lower_placement,
                upper_package,
                upper_placement,
            )
        ]
        if not direct_supporters:
            return False
        if any(package.is_fragile for package in direct_supporters):
            return False

    return True


# Memeriksa bahwa placement memakai koordinat integer dan salah satu orientasi valid.
def is_well_formed_placement(placement: object) -> bool:
    if not isinstance(placement, tuple) or len(placement) != 4:
        return False

    x, y, z, orientation = placement
    if any(type(coordinate) is not int for coordinate in (x, y, z)):
        return False
    return isinstance(orientation, tuple) and orientation in ORIENTATIONS


# Menghitung volume irisan dua paket untuk penalti overlap.
def overlap_volume(
    pkg_a: Package,
    placement_a: Placement,
    pkg_b: Package,
    placement_b: Placement,
) -> int:
    ax, ay, az, adx, ady, adz = box(pkg_a, placement_a)
    bx, by, bz, bdx, bdy, bdz = box(pkg_b, placement_b)
    cx = min(ax + adx, bx + bdx) - max(ax, bx)
    cy = min(ay + ady, by + bdy) - max(ay, by)
    cz = min(az + adz, bz + bdz) - max(az, bz)
    return max(0, cx) * max(0, cy) * max(0, cz)


# Menghitung total pelanggaran hard constraint pada state.
def violations(state: State, problem: Problem) -> int:
    if set(state) != set(problem.packages):
        return len(set(state).symmetric_difference(problem.packages)) or 1

    if problem.is_multi_truck:
        malformed = 0
        for placement in state.values():
            if placement is None:
                continue
            try:
                _, local_placement = split_placement(placement, problem)
                if not is_well_formed_placement(local_placement):
                    malformed += 1
            except (TypeError, ValueError):
                malformed += 1
        if malformed:
            return malformed

        return sum(
            _single_truck_violations(
                project_truck_state(state, problem, truck_id),
                Problem(problem.name, truck, problem.packages),
            )
            for truck_id, truck in problem.fleet.items()
        )

    return _single_truck_violations(state, problem)


# Menghitung besar pelanggaran constraint pada tepat satu truk.
def _single_truck_violations(state: State, problem: Problem) -> int:
    placed = [
        (problem.packages[pid], placement)
        for pid, placement in state.items()
        if placement is not None
    ]
    total = sum(1 for pkg, placement in placed if not in_bounds(pkg, placement, problem.truck))

    for i, (pkg_a, placement_a) in enumerate(placed):
        for pkg_b, placement_b in placed[i + 1:]:
            total += overlap_volume(pkg_a, placement_a, pkg_b, placement_b)

    for i, (upper, upper_placement) in enumerate(placed):
        if upper_placement[2] == 0:
            continue
        supporters = [
            lower
            for j, (lower, lower_placement) in enumerate(placed)
            if j != i and supports(lower, lower_placement, upper, upper_placement)
        ]
        if not supporters or any(lower.is_fragile for lower in supporters):
            total += 1

    total += max(0, total_weight(state, problem) - problem.truck.max_capacity)
    return total


# Menghasilkan penalti di atas batas objective agar state infeasible selalu
# lebih buruk daripada state kosong yang feasible.
def default_penalty_weight(
    problem: Problem,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> float:
    evaluator = resolve_objective(problem, objective_name)
    return max(1.0, evaluator.maximum_score + 1.0)


# Menggabungkan objective dengan penalti hard constraint untuk pencarian.
def penalized_objective(
    state: State,
    problem: Problem,
    weight: float | None = None,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> float:
    if weight is None:
        weight = default_penalty_weight(problem, objective_name)
    return (
        objective(state, problem, objective_name)
        - weight * violations(state, problem)
    )
