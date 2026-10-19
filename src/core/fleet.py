from core.models import (
    DEFAULT_TRUCK_ID,
    FleetPlacement,
    Placement,
    Problem,
    State,
    StatePlacement,
)


# Memisahkan assignment armada menjadi identifier truk dan placement lokal.
def split_placement(
    placement: StatePlacement,
    problem: Problem,
) -> tuple[str, Placement]:
    if problem.is_multi_truck:
        if not isinstance(placement, tuple) or len(placement) != 5:
            raise ValueError(f"Invalid fleet placement: {placement}")
        truck_id, x, y, z, orientation = placement
        if truck_id not in problem.fleet:
            raise ValueError(f"Unknown truck ID: {truck_id}")
        return truck_id, (x, y, z, orientation)

    if not isinstance(placement, tuple) or len(placement) != 4:
        raise ValueError(f"Invalid placement: {placement}")
    return DEFAULT_TRUCK_ID, placement


# Menggabungkan placement lokal dengan identifier truk bila problem multi-truk.
def join_placement(
    truck_id: str,
    placement: Placement,
    problem: Problem,
) -> StatePlacement:
    if truck_id not in problem.fleet:
        raise ValueError(f"Unknown truck ID: {truck_id}")
    if not problem.is_multi_truck:
        return placement
    x, y, z, orientation = placement
    return (truck_id, x, y, z, orientation)


# Mengambil identifier truk dari assignment paket, atau None bila di luar.
def assigned_truck_id(
    placement: StatePlacement | None,
    problem: Problem,
) -> str | None:
    if placement is None:
        return None
    truck_id, _ = split_placement(placement, problem)
    return truck_id


# Menghasilkan state lokal lengkap untuk satu truk agar validator lama dapat dipakai.
def project_truck_state(
    state: State,
    problem: Problem,
    truck_id: str,
) -> dict[str, Placement | None]:
    if truck_id not in problem.fleet:
        raise KeyError(f"Unknown truck ID: {truck_id}")

    projected: dict[str, Placement | None] = {}
    for package_id in problem.packages:
        assignment = state.get(package_id)
        if assignment is None:
            projected[package_id] = None
            continue
        assigned_id, placement = split_placement(assignment, problem)
        projected[package_id] = placement if assigned_id == truck_id else None
    return projected


# Mengubah placement lokal dalam state satu truk menjadi state armada.
def replace_truck_state(
    state: State,
    local_state: dict[str, Placement | None],
    problem: Problem,
    truck_id: str,
) -> State:
    updated = dict(state)
    for package_id, placement in local_state.items():
        current = state.get(package_id)
        current_truck = assigned_truck_id(current, problem)
        if placement is not None:
            updated[package_id] = join_placement(truck_id, placement, problem)
        elif current_truck == truck_id:
            updated[package_id] = None
    return updated


# Menghasilkan posisi z dari placement lokal maupun placement armada.
def placement_z(placement: StatePlacement, problem: Problem) -> int:
    _, local = split_placement(placement, problem)
    return local[2]


# Menghasilkan total berat paket pada setiap truk.
def fleet_weights(state: State, problem: Problem) -> dict[str, int]:
    weights = {truck_id: 0 for truck_id in problem.truck_ids}
    for package_id, placement in state.items():
        if placement is None or package_id not in problem.packages:
            continue
        try:
            truck_id = assigned_truck_id(placement, problem)
        except ValueError:
            continue
        weights[truck_id] += problem.packages[package_id].weight
    return weights


# Menghasilkan statistik value, berat, dan paket untuk setiap truk.
def fleet_metrics(state: State, problem: Problem) -> dict[str, dict[str, float | int]]:
    metrics = {
        truck_id: {
            "total_value": 0.0,
            "total_weight": 0,
            "loaded_packages": 0,
            "max_capacity": truck.max_capacity,
        }
        for truck_id, truck in problem.fleet.items()
    }
    for package_id, placement in state.items():
        if placement is None or package_id not in problem.packages:
            continue
        truck_id = assigned_truck_id(placement, problem)
        package = problem.packages[package_id]
        metrics[truck_id]["total_value"] += float(package.value)
        metrics[truck_id]["total_weight"] += package.weight
        metrics[truck_id]["loaded_packages"] += 1
    return metrics
