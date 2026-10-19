import math
import random
import time

from algorithms.result import SearchResult
from core.models import Problem, State
from core.neighbors import random_neighbor, random_state
from core.objective import is_feasible, violations
from core.objectives import (
    DEFAULT_OBJECTIVE,
    resolve_objective,
    state_metrics,
)


def run(
    problem: Problem,
    seed: int,
    t0: float = 100.0,
    alpha: float = 0.99982,
    t_min: float = 0.01,
    max_iter: int = 50_000,
    penalty_weight: float | None = None,
    stuck_threshold: int = 500,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> SearchResult:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if t0 <= t_min:
        raise ValueError("t0 must be greater than t_min")
    evaluator = resolve_objective(problem, objective_name)
    if penalty_weight is None:
        penalty_weight = max(
            evaluator.package_utilities.values(),
            default=1.0,
        )

    # Menghitung skor pencarian dengan objective terpilih dan penalti constraint.
    def search_score(state: State) -> float:
        return (
            evaluator.evaluate(state)
            - penalty_weight * violations(state, problem)
        )

    start = time.perf_counter()
    rng = random.Random(seed)

    initial_state = random_state(problem, rng)
    current = initial_state
    current_score = search_score(current)

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
        candidate_score = search_score(candidate)
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
        objective_name=objective_name,
        extra={
            "best_feasible_history": best_history,
            "temperature": temperatures,
            "acceptance_prob": acceptance_prob,
            "accepted_worse": accepted_worse,
            "stuck_count": stuck_count,
            "last_state": current,
            "last_score": current_score,
            "initial_metrics": state_metrics(initial_state, problem),
            "final_metrics": state_metrics(best_state, problem),
        },
    )
