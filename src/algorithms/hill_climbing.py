import random
import time

from algorithms.result import SearchResult
from core.models import Problem, State
from core.neighbors import all_neighbors, random_state
from core.objective import (
    default_penalty_weight,
    is_feasible,
    objective,
    penalized_objective,
    violations,
)

EPSILON = 1e-9


# Menjalankan Steepest Ascent Hill-Climbing.
# Tiap iterasi pindah ke neighbor dengan skor (penalized objective) tertinggi
# selama masih ada perbaikan, sampai local optimum atau max_iter tercapai.
# History indeks 0 merepresentasikan objective state awal sebelum iterasi pertama.
def run(problem: Problem, seed: int, max_iter: int = 500) -> SearchResult:
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")

    start = time.perf_counter()
    init_state = random_state(problem, random.Random(seed))
    penalty = default_penalty_weight(problem)

    state = dict(init_state)
    best_score = penalized_objective(state, problem, penalty)
    history = [objective(state, problem)]
    iterations = 0

    while iterations < max_iter:
        best_neighbor = None
        for candidate in all_neighbors(state, problem):
            candidate_score = penalized_objective(candidate, problem, penalty)
            if candidate_score > best_score + EPSILON:
                best_score = candidate_score
                best_neighbor = candidate
        if best_neighbor is None:
            break
        state = best_neighbor
        iterations += 1
        history.append(objective(state, problem))

    return SearchResult(
        algorithm="hill_climbing",
        params={"variant": "steepest", "max_iter": max_iter},
        seed=seed,
        problem=problem.name,
        initial_state=init_state,
        final_state=state,
        final_objective=objective(state, problem),
        feasible=is_feasible(state, problem),
        history=history,
        iterations=iterations,
        duration_s=time.perf_counter() - start,
        extra={
            "violations": violations(state, problem),
            "initial_objective": history[0],
            "initial_score": penalized_objective(init_state, problem, penalty),
            "final_score": best_score,
            "penalty_weight": penalty,
        },
    )
