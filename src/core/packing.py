import random

from core.fleet import join_placement, placement_z, project_truck_state, split_placement
from core.models import Orientation, Package, Placement, Problem, State, StatePlacement
from core.objectives import (
    DEFAULT_OBJECTIVE,
    ObjectiveEvaluator,
    resolve_objective,
)
from core.objective import (
    ORIENTATIONS,
    in_bounds,
    is_feasible,
    is_well_formed_placement,
    oriented_dims,
    placements_overlap,
    supports,
    total_weight,
)


# Memeriksa apakah paket dapat ditambahkan ke state feasible tanpa melanggar constraint.
# Fungsi ini mengasumsikan package_id sedang berada di luar state sebelum ditambahkan.
def _can_add_to_feasible_state(
    state: State,
    package_id: str,
    placement: StatePlacement,
    problem: Problem,
) -> bool:
    if problem.is_multi_truck:
        try:
            truck_id, local_placement = split_placement(placement, problem)
        except (TypeError, ValueError):
            return False
        local_problem = Problem(
            problem.name,
            problem.truck_for(truck_id),
            problem.packages,
        )
        local_state = project_truck_state(state, problem, truck_id)
        local_state[package_id] = None
        return _can_add_to_single_truck(
            local_state,
            package_id,
            local_placement,
            local_problem,
        )
    return _can_add_to_single_truck(state, package_id, placement, problem)


# Memeriksa penambahan paket pada satu truk tanpa melanggar hard constraint.
def _can_add_to_single_truck(
    state: State,
    package_id: str,
    placement: Placement,
    problem: Problem,
) -> bool:
    package = problem.packages[package_id]

    if not is_well_formed_placement(placement):
        return False
    if not in_bounds(package, placement, problem.truck):
        return False
    if total_weight(state, problem) + package.weight > problem.truck.max_capacity:
        return False

    direct_supporters: list[Package] = []
    for other_id, other_placement in state.items():
        if other_id == package_id or other_placement is None:
            continue

        other_package = problem.packages[other_id]
        if placements_overlap(package, placement, other_package, other_placement):
            return False
        if supports(other_package, other_placement, package, placement):
            direct_supporters.append(other_package)

        # Paket fragile tidak boleh menjadi support bagi paket yang sudah ada.
        if package.is_fragile and supports(
            package,
            placement,
            other_package,
            other_placement,
        ):
            return False

    if placement[2] == 0:
        return True
    if not direct_supporters:
        return False
    return not any(supporter.is_fragile for supporter in direct_supporters)


# Memeriksa apakah placement baru menghasilkan state lengkap yang feasible.
# Placement lama dari package_id diganti, sedangkan state input tidak dimutasi.
def can_place(
    state: State,
    package_id: str,
    placement: StatePlacement,
    problem: Problem,
) -> bool:
    if package_id not in problem.packages:
        raise KeyError(f"Unknown package ID: {package_id}")
    if set(state) != set(problem.packages):
        return False

    candidate = dict(state)
    candidate[package_id] = placement
    return is_feasible(candidate, problem)


# Menghasilkan orientasi fisik yang unik dalam urutan acak.
# Dimensi yang sama akibat sisi paket identik tidak dicoba berulang kali.
def _unique_random_orientations(
    package: Package,
    rng: random.Random,
) -> list[Orientation]:
    orientations = list(ORIENTATIONS)
    rng.shuffle(orientations)

    unique_orientations: list[Orientation] = []
    seen_dimensions: set[tuple[int, int, int]] = set()
    for orientation in orientations:
        dimensions = oriented_dims(package.dims, orientation)
        if dimensions in seen_dimensions:
            continue
        seen_dimensions.add(dimensions)
        unique_orientations.append(orientation)
    return unique_orientations


# Mencari placement valid secara bottom-up agar paket tidak melayang.
# Orientasi serta koordinat horizontal diacak untuk menjaga keberagaman state.
def find_valid_placement(
    state: State,
    package_id: str,
    problem: Problem,
    rng: random.Random,
    excluded_placements: set[StatePlacement] | None = None,
) -> StatePlacement | None:
    if package_id not in problem.packages:
        raise KeyError(f"Unknown package ID: {package_id}")
    if set(state) != set(problem.packages):
        return None

    if problem.is_multi_truck:
        truck_ids = list(problem.truck_ids)
        rng.shuffle(truck_ids)
        for truck_id in truck_ids:
            local_problem = Problem(
                problem.name,
                problem.truck_for(truck_id),
                problem.packages,
            )
            local_state = project_truck_state(state, problem, truck_id)
            local_excluded: set[Placement] = set()
            for excluded in excluded_placements or set():
                try:
                    excluded_truck, local = split_placement(excluded, problem)
                except (TypeError, ValueError):
                    continue
                if excluded_truck == truck_id:
                    local_excluded.add(local)

            placement = _find_valid_single_truck_placement(
                local_state,
                package_id,
                local_problem,
                rng,
                local_excluded or None,
            )
            if placement is not None:
                return join_placement(truck_id, placement, problem)
        return None

    return _find_valid_single_truck_placement(
        state,
        package_id,
        problem,
        rng,
        excluded_placements,
    )


# Mencari placement valid pada tepat satu truk secara bottom-up.
def _find_valid_single_truck_placement(
    state: State,
    package_id: str,
    problem: Problem,
    rng: random.Random,
    excluded_placements: set[Placement] | None = None,
) -> Placement | None:

    base_state = dict(state)
    base_state[package_id] = None
    if not is_feasible(base_state, problem):
        return None

    package = problem.packages[package_id]
    for orientation in _unique_random_orientations(package, rng):
        dx, dy, dz = oriented_dims(package.dims, orientation)
        if dx > problem.truck.w or dy > problem.truck.l or dz > problem.truck.h:
            continue

        x_coordinates = list(range(problem.truck.w - dx + 1))
        y_coordinates = list(range(problem.truck.l - dy + 1))
        rng.shuffle(x_coordinates)
        rng.shuffle(y_coordinates)

        for z in range(problem.truck.h - dz + 1):
            for y in y_coordinates:
                for x in x_coordinates:
                    placement: Placement = (x, y, z, orientation)
                    if excluded_placements and placement in excluded_placements:
                        continue
                    if _can_add_to_feasible_state(
                        base_state,
                        package_id,
                        placement,
                        problem,
                    ):
                        return placement
    return None


# Membuat state acak yang selalu memenuhi seluruh hard constraint.
# Setiap paket mempunyai peluang untuk dicoba masukkan ke dalam peti kemas.
def random_feasible_state(
    problem: Problem,
    rng: random.Random,
    inclusion_probability: float = 0.8,
) -> State:
    if not 0.0 <= inclusion_probability <= 1.0:
        raise ValueError("inclusion_probability must be between 0 and 1")

    state: State = {package_id: None for package_id in problem.packages}
    package_ids = list(problem.packages)
    rng.shuffle(package_ids)

    for package_id in package_ids:
        if rng.random() > inclusion_probability:
            continue
        placement = find_valid_placement(state, package_id, problem, rng)
        if placement is not None:
            state[package_id] = placement

    if not is_feasible(state, problem):
        raise RuntimeError("random_feasible_state produced an infeasible state")
    return state


# Menghitung prioritas paket berdasarkan utility objective per unit volume.
def _utility_density(
    package_id: str,
    problem: Problem,
    evaluator: ObjectiveEvaluator,
) -> float:
    package = problem.packages[package_id]
    volume = package.w * package.l * package.h
    if volume <= 0:
        return float("-inf")
    return evaluator.package_utilities[package_id] / volume


# Memperbaiki state tanpa memutasi input dan selalu menghasilkan state feasible.
# Placement valid dipertahankan dari bawah ke atas, lalu paket bermasalah dicoba
# ditempatkan kembali berdasarkan value density. Paket None juga dicoba jika
# fill_outside bernilai True.
def repair_state(
    state: State,
    problem: Problem,
    rng: random.Random,
    fill_outside: bool = True,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> State:
    evaluator = resolve_objective(problem, objective_name)
    repaired: State = {package_id: None for package_id in problem.packages}
    valid_candidates: list[tuple[str, Placement]] = []
    rejected_ids: list[str] = []
    outside_ids: list[str] = []

    for package_id in problem.packages:
        placement = state.get(package_id)
        if placement is None:
            outside_ids.append(package_id)
        else:
            try:
                _, local_placement = split_placement(placement, problem)
            except (TypeError, ValueError):
                rejected_ids.append(package_id)
                continue
            if is_well_formed_placement(local_placement):
                valid_candidates.append((package_id, placement))
            else:
                rejected_ids.append(package_id)

    # Urutan z menjaga dependency support; value density memutus konflik setingkat.
    valid_candidates.sort(
        key=lambda item: (
            placement_z(item[1], problem),
            -_utility_density(item[0], problem, evaluator),
            -evaluator.package_utilities[item[0]],
            item[0],
        )
    )

    for package_id, placement in valid_candidates:
        if _can_add_to_feasible_state(
            repaired,
            package_id,
            placement,
            problem,
        ):
            repaired[package_id] = placement
        else:
            rejected_ids.append(package_id)

    pending_ids = list(rejected_ids)
    if fill_outside:
        pending_ids.extend(outside_ids)

    # Shuffle sebelum sort membuat paket dengan prioritas sama tetap bervariasi.
    rng.shuffle(pending_ids)
    pending_ids.sort(
        key=lambda package_id: (
            problem.packages[package_id].is_fragile,
            -_utility_density(package_id, problem, evaluator),
            -evaluator.package_utilities[package_id],
        )
    )

    for package_id in pending_ids:
        placement = find_valid_placement(repaired, package_id, problem, rng)
        if placement is not None:
            repaired[package_id] = placement

    if not is_feasible(repaired, problem):
        raise RuntimeError("repair_state produced an infeasible state")
    return repaired
