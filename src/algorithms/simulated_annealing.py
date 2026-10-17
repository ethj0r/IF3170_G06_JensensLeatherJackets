import math
import random
import time

from algorithms.result import SearchResult
from core.models import Problem, State
from core.neighbors import random_neighbor, random_state
from core.objective import is_feasible, penalized_objective


def run(
    problem: Problem,
    seed: int,
    t0: float = 100.0,
    alpha: float = 0.999,
    t_min: float = 0.01,
    max_iter: int = 50_000,
    penalty_weight: float | None = None,
    stuck_threshold: int = 500,
) -> SearchResult:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if t0 <= t_min:
        raise ValueError("t0 must be greater than t_min")
    if penalty_weight is None:
        penalty_weight = float(max(pkg.value for pkg in problem.packages.values()))

    start = time.perf_counter()
    rng = random.Random(seed)

    initial_state = random_state(problem, rng)
    current = initial_state
    current_score = penalized_objective(current, problem, penalty_weight)

    best_state: State = {pid: None for pid in problem.packages}
    best_value = 0.0
    if is_feasible(current, problem):
        best_state, best_value = dict(current), current_score

    temperature = t0
    history = [current_score]
    best_history = [best_value]
    temperatures = [temperature]
    acceptance_prob: list[float | None] = [None]
    accepted_worse = 0
    stuck_count = 0
    no_improve = 0
    iterations = 0

    while temperature > t_min and iterations < max_iter:
        candidate = random_neighbor(current, problem, rng)
        candidate_score = penalized_objective(candidate, problem, penalty_weight)
        delta = candidate_score - current_score

        prob = None
        if delta >= 0:
            accept = True
        else:
            prob = math.exp(delta / temperature)
            accept = rng.random() < prob
            accepted_worse += accept

        if accept:
            current, current_score = candidate, candidate_score

        if current_score > best_value and is_feasible(current, problem):
            best_state, best_value = dict(current), current_score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= stuck_threshold:
                stuck_count += 1
                no_improve = 0

        iterations += 1
        temperature *= alpha
        history.append(current_score)
        best_history.append(best_value)
        temperatures.append(temperature)
        acceptance_prob.append(prob)

    return SearchResult(
        algorithm="simulated_annealing",
        params={
            "t0": t0,
            "alpha": alpha,
            "t_min": t_min,
            "max_iter": max_iter,
            "penalty_weight": penalty_weight,
            "stuck_threshold": stuck_threshold,
        },
        seed=seed,
        problem=problem.name,
        initial_state=initial_state,
        final_state=best_state,
        final_objective=best_value,
        feasible=is_feasible(best_state, problem),
        history=history,
        iterations=iterations,
        duration_s=time.perf_counter() - start,
        extra={
            "best_feasible_history": best_history,
            "temperature": temperatures,
            "acceptance_prob": acceptance_prob,
            "accepted_worse": accepted_worse,
            "stuck_count": stuck_count,
            "last_state": current,
            "last_score": current_score,
        },
    )
