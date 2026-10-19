import random
import time
from statistics import fmean

from algorithms.result import SearchResult
from core.models import Problem, State
from core.objective import is_feasible, rotate
from core.objectives import (
    DEFAULT_OBJECTIVE,
    ObjectiveEvaluator,
    resolve_objective,
    state_metrics,
)
from core.packing import find_valid_placement, random_feasible_state, repair_state


# Memilih satu individu melalui tournament selection dan mengembalikan salinannya.
def tournament_selection(
    population: list[State],
    fitnesses: list[float],
    rng: random.Random,
    tournament_size: int,
) -> State:
    if len(population) != len(fitnesses):
        raise ValueError("population and fitnesses must have the same length")
    if not 1 <= tournament_size <= len(population):
        raise ValueError("tournament_size must be between 1 and population size")

    candidate_indices = rng.sample(range(len(population)), tournament_size)
    winner_index = max(candidate_indices, key=fitnesses.__getitem__)
    return dict(population[winner_index])


# Menggabungkan dua parent dengan memilih placement setiap paket secara independen.
# Hasil crossover belum tentu feasible sehingga harus diperbaiki sebelum digunakan.
def uniform_crossover(
    parent_a: State,
    parent_b: State,
    rng: random.Random,
) -> State:
    if set(parent_a) != set(parent_b):
        raise ValueError("parents must contain the same package IDs")
    return {
        package_id: parent_a[package_id]
        if rng.random() < 0.5
        else parent_b[package_id]
        for package_id in parent_a
    }


# Menukar placement dua paket, termasuk pertukaran paket dalam dan luar truk.
def swap_mutation(
    state: State,
    problem: Problem,
    rng: random.Random,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> State:
    if len(state) < 2:
        return dict(state)

    first_id, second_id = rng.sample(list(state), 2)
    mutated = dict(state)
    mutated[first_id], mutated[second_id] = mutated[second_id], mutated[first_id]
    return repair_state(
        mutated,
        problem,
        rng,
        fill_outside=False,
        objective_name=objective_name,
    )


# Memindahkan satu paket ke placement feasible yang berbeda dari placement awal.
# Paket yang sebelumnya di luar truk juga dapat dipilih untuk dimasukkan.
def move_mutation(
    state: State,
    problem: Problem,
    rng: random.Random,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> State:
    package_ids = list(state)
    rng.shuffle(package_ids)

    for package_id in package_ids:
        original_placement = state[package_id]
        base_state = dict(state)
        base_state[package_id] = None
        excluded = (
            {original_placement}
            if original_placement is not None
            else None
        )
        new_placement = find_valid_placement(
            base_state,
            package_id,
            problem,
            rng,
            excluded_placements=excluded,
        )
        if new_placement is None:
            continue

        mutated = dict(base_state)
        mutated[package_id] = new_placement
        return mutated

    return dict(state)


# Merotasi satu paket di dalam truk 90 derajat pada sumbu x, y, atau z.
# Repair menangani kemungkinan overlap atau pelanggaran batas setelah rotasi.
def rotate_mutation(
    state: State,
    problem: Problem,
    rng: random.Random,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> State:
    placed_ids = [
        package_id
        for package_id, placement in state.items()
        if placement is not None
    ]
    if not placed_ids:
        return dict(state)

    package_id = rng.choice(placed_ids)
    x, y, z, orientation = state[package_id]
    axis = rng.choice(("x", "y", "z"))

    mutated = dict(state)
    mutated[package_id] = (x, y, z, rotate(orientation, axis))
    return repair_state(
        mutated,
        problem,
        rng,
        fill_outside=False,
        objective_name=objective_name,
    )


# Memilih secara acak satu operator mutation yang dapat diterapkan pada state.
def mutate(
    state: State,
    problem: Problem,
    rng: random.Random,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> State:
    if not state:
        return dict(state)

    operators = [move_mutation]
    if len(state) >= 2:
        operators.append(swap_mutation)
    if any(placement is not None for placement in state.values()):
        operators.append(rotate_mutation)

    operator = rng.choice(operators)
    return operator(state, problem, rng, objective_name)


# Memvalidasi seluruh hyperparameter sebelum proses Genetic Algorithm dimulai.
def _validate_parameters(
    population_size: int,
    iterations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_size: int,
    elite_count: int,
    inclusion_probability: float,
) -> None:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not 1 <= tournament_size <= population_size:
        raise ValueError("tournament_size must be between 1 and population size")
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be between 1 and population_size - 1")
    if not 0.0 <= inclusion_probability <= 1.0:
        raise ValueError("inclusion_probability must be between 0 and 1")


# Menghitung fitness seluruh populasi menggunakan objective yang dipilih.
def _evaluate_population(
    population: list[State],
    evaluator: ObjectiveEvaluator,
) -> list[float]:
    return [evaluator.evaluate(individual) for individual in population]


# Menjalankan Genetic Algorithm dan mencatat fitness maksimum serta rata-rata.
# History indeks 0 merepresentasikan populasi awal sebelum iterasi pertama.
def run(
    problem: Problem,
    seed: int,
    population_size: int = 100,
    iterations: int = 500,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.15,
    tournament_size: int = 3,
    elite_count: int = 2,
    inclusion_probability: float = 0.8,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> SearchResult:
    _validate_parameters(
        population_size,
        iterations,
        crossover_rate,
        mutation_rate,
        tournament_size,
        elite_count,
        inclusion_probability,
    )

    start = time.perf_counter()
    rng = random.Random(seed)
    evaluator = resolve_objective(problem, objective_name)
    population = [
        random_feasible_state(problem, rng, inclusion_probability)
        for _ in range(population_size)
    ]
    fitnesses = _evaluate_population(population, evaluator)

    initial_best_index = max(range(population_size), key=fitnesses.__getitem__)
    initial_state = dict(population[initial_best_index])
    best_state = dict(initial_state)
    best_fitness = fitnesses[initial_best_index]
    best_iteration = 0

    max_history = [max(fitnesses)]
    average_history = [fmean(fitnesses)]

    for iteration in range(1, iterations + 1):
        ranked_indices = sorted(
            range(population_size),
            key=fitnesses.__getitem__,
            reverse=True,
        )
        next_population = [
            dict(population[index])
            for index in ranked_indices[:elite_count]
        ]

        while len(next_population) < population_size:
            parent_a = tournament_selection(
                population,
                fitnesses,
                rng,
                tournament_size,
            )
            parent_b = tournament_selection(
                population,
                fitnesses,
                rng,
                tournament_size,
            )

            if rng.random() < crossover_rate:
                child = uniform_crossover(parent_a, parent_b, rng)
            else:
                child = dict(parent_a)

            child = repair_state(
                child,
                problem,
                rng,
                fill_outside=False,
                objective_name=objective_name,
            )
            if rng.random() < mutation_rate:
                child = mutate(child, problem, rng, objective_name)
            next_population.append(child)

        population = next_population
        fitnesses = _evaluate_population(population, evaluator)
        generation_best_index = max(
            range(population_size),
            key=fitnesses.__getitem__,
        )
        generation_best = fitnesses[generation_best_index]

        max_history.append(generation_best)
        average_history.append(fmean(fitnesses))

        if generation_best > best_fitness:
            best_fitness = generation_best
            best_state = dict(population[generation_best_index])
            best_iteration = iteration

    if not is_feasible(best_state, problem):
        raise RuntimeError("Genetic Algorithm produced an infeasible best state")

    params = {
        "population_size": population_size,
        "iterations": iterations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "tournament_size": tournament_size,
        "elite_count": elite_count,
        "inclusion_probability": inclusion_probability,
    }
    return SearchResult(
        algorithm="genetic",
        params=params,
        seed=seed,
        problem=problem.name,
        initial_state=initial_state,
        final_state=best_state,
        final_objective=best_fitness,
        feasible=True,
        history=max_history,
        iterations=iterations,
        duration_s=time.perf_counter() - start,
        objective_name=objective_name,
        extra={
            "average_history": average_history,
            "initial_best_objective": max_history[0],
            "final_population_average": average_history[-1],
            "best_iteration": best_iteration,
            "initial_metrics": state_metrics(initial_state, problem),
            "final_metrics": state_metrics(best_state, problem),
        },
    )
