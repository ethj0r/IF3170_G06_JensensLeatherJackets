from itertools import permutations

from core.models import Orientation, Package, Placement, Problem, State, Truck

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


# Menghitung objective function berupa total value paket di dalam peti kemas.
def objective(state: State, problem: Problem) -> float:
    return float(sum(
        problem.packages[pid].value
        for pid, placement in state.items()
        if placement is not None
    ))


# Memeriksa seluruh hard constraint dan kelengkapan representasi state.
# Paket dengan placement None dianggap berada di luar peti kemas.
def is_feasible(state: State, problem: Problem) -> bool:
    if set(state) != set(problem.packages):
        return False

    placed: list[tuple[str, Package, Placement]] = []
    for pid, placement in state.items():
        if placement is None:
            continue

        if not _is_well_formed_placement(placement):
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
def _is_well_formed_placement(placement: object) -> bool:
    if not isinstance(placement, tuple) or len(placement) != 4:
        return False

    x, y, z, orientation = placement
    if any(type(coordinate) is not int for coordinate in (x, y, z)):
        return False
    return isinstance(orientation, tuple) and orientation in ORIENTATIONS
