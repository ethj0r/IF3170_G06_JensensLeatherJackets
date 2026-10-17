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


# Memastikan parameter tidak negatif sebelum pencarian dimulai.
def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


# Membuat kunci hashable dari state untuk melacak state yang sudah
# dikunjungi saat menelusuri plateau pada Hill-Climbing with Sideways Move.
def _freeze(state: State) -> frozenset:
    return frozenset(state.items())


# Steepest ascent penuh dari satu state awal: tiap iterasi pindah ke neighbor
# dengan skor (penalized objective) tertinggi selama masih ada perbaikan,
# sampai local optimum atau max_iter tercapai.
# Mengembalikan (state akhir, history objective, iterasi yang diterima, skor akhir).
# History indeks 0 merepresentasikan objective state awal sebelum iterasi pertama.
def _steepest_climb(
    state: State,
    problem: Problem,
    penalty: float,
    max_iter: int,
) -> tuple[State, list[float], int, float]:
    score = penalized_objective(state, problem, penalty)
    history = [objective(state, problem)]
    iterations = 0

    while iterations < max_iter:
        best_neighbor = None
        best_candidate_score = score
        for candidate in all_neighbors(state, problem):
            candidate_score = penalized_objective(candidate, problem, penalty)
            if candidate_score > best_candidate_score + EPSILON:
                best_candidate_score = candidate_score
                best_neighbor = candidate
        if best_neighbor is None:
            break
        state = best_neighbor
        score = best_candidate_score
        iterations += 1
        history.append(objective(state, problem))

    return state, history, iterations, score


# Menjalankan Steepest Ascent Hill-Climbing.
def run(problem: Problem, seed: int, max_iter: int = 500) -> SearchResult:
    _require_non_negative("max_iter", max_iter)

    start = time.perf_counter()
    init_state = random_state(problem, random.Random(seed))
    penalty = default_penalty_weight(problem)

    state, history, iterations, score = _steepest_climb(
        dict(init_state), problem, penalty, max_iter
    )

    return SearchResult(
        algorithm="hill_climbing_steepest",
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
            "final_score": score,
            "penalty_weight": penalty,
        },
    )


# Menjalankan Stochastic Hill-Climbing.
# Tiap state, neighbor dipilih secara acak berulang kali; langkah diterima bila
# skor membaik, dan pencarian berhenti setelah max_stuck neighbor berturut-turut
# ditolak. Iterasi hanya dihitung untuk langkah yang diterima, sedangkan
# "proposals" mencatat seluruh evaluasi neighbor sampai pencarian berhenti.
# max_stuck=1 menghasilkan perilaku klasik: berhenti pada penolakan pertama.
def run_stochastic(
    problem: Problem,
    seed: int,
    max_iter: int = 500,
    max_stuck: int = 100,
) -> SearchResult:
    _require_non_negative("max_iter", max_iter)
    _require_non_negative("max_stuck", max_stuck)

    start = time.perf_counter()
    rng = random.Random(seed)
    init_state = random_state(problem, rng)
    penalty = default_penalty_weight(problem)

    state = dict(init_state)
    score = penalized_objective(state, problem, penalty)
    history = [objective(state, problem)]
    iterations = 0
    proposals = 0
    stuck = 0

    while iterations < max_iter:
        candidates = [
            (candidate, penalized_objective(candidate, problem, penalty))
            for candidate in all_neighbors(state, problem)
        ]
        if not candidates:
            break

        moved = False
        while not moved:
            proposals += 1
            candidate, candidate_score = rng.choice(candidates)
            if candidate_score > score + EPSILON:
                state = candidate
                score = candidate_score
                stuck = 0
                iterations += 1
                history.append(objective(state, problem))
                moved = True
                continue
            stuck += 1
            if stuck >= max_stuck:
                break
        if not moved:
            break

    return SearchResult(
        algorithm="hill_climbing_stochastic",
        params={"variant": "stochastic", "max_iter": max_iter, "max_stuck": max_stuck},
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
            "final_score": score,
            "penalty_weight": penalty,
            "proposals": proposals,
            "max_stuck": max_stuck,
        },
    )


# Menjalankan Hill-Climbing with Sideways Move.
# Jika tidak ada perbaikan, pindah ke neighbor dengan skor sama (sideways)
# yang belum dikunjungi pada plateau ini; penghitung sideways berurutan
# direset setiap kali skor membaik dan pencarian berhenti ketika mencapai
# max_sideways. max_sideways=0 membuat algoritma identik dengan steepest ascent.
def run_sideways(
    problem: Problem,
    seed: int,
    max_iter: int = 500,
    max_sideways: int = 5,
) -> SearchResult:
    _require_non_negative("max_iter", max_iter)
    _require_non_negative("max_sideways", max_sideways)

    start = time.perf_counter()
    rng = random.Random(seed)
    init_state = random_state(problem, rng)
    penalty = default_penalty_weight(problem)

    state = dict(init_state)
    score = penalized_objective(state, problem, penalty)
    history = [objective(state, problem)]
    iterations = 0
    sideways_moves = 0
    consecutive = 0
    visited = {_freeze(state)}

    while iterations < max_iter:
        best_neighbor = None
        best_candidate_score = score
        equal_neighbors = []
        for candidate in all_neighbors(state, problem):
            candidate_score = penalized_objective(candidate, problem, penalty)
            if candidate_score > best_candidate_score + EPSILON:
                best_candidate_score = candidate_score
                best_neighbor = candidate
            elif (
                abs(candidate_score - score) <= EPSILON
                and _freeze(candidate) not in visited
            ):
                equal_neighbors.append(candidate)

        if best_neighbor is not None:
            state = best_neighbor
            score = best_candidate_score
            consecutive = 0
            visited = {_freeze(state)}
            iterations += 1
            history.append(objective(state, problem))
            continue

        if consecutive >= max_sideways:
            break
        if not equal_neighbors:
            break

        state = rng.choice(equal_neighbors)
        consecutive += 1
        sideways_moves += 1
        visited.add(_freeze(state))
        iterations += 1
        history.append(objective(state, problem))

    return SearchResult(
        algorithm="hill_climbing_sideways",
        params={
            "variant": "sideways",
            "max_iter": max_iter,
            "max_sideways": max_sideways,
        },
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
            "final_score": score,
            "penalty_weight": penalty,
            "sideways_moves": sideways_moves,
            "max_sideways": max_sideways,
        },
    )


# Menjalankan Random Restart Hill-Climbing.
# Steepest ascent diulang dari state awal acak baru setiap kali local optimum
# tercapai, sebanyak maksimal max_restarts restart (total max_restarts + 1 climb).
# Hasil akhir adalah climb dengan skor tertinggi. Pencarian berhenti lebih awal
# bila skor menyentuh total value seluruh paket (optimum global mustahil dilampaui).
def run_restart(
    problem: Problem,
    seed: int,
    max_iter: int = 500,
    max_restarts: int = 10,
) -> SearchResult:
    _require_non_negative("max_iter", max_iter)
    _require_non_negative("max_restarts", max_restarts)

    start = time.perf_counter()
    rng = random.Random(seed)
    penalty = default_penalty_weight(problem)
    total_value = float(sum(pkg.value for pkg in problem.packages.values()))

    init_state = random_state(problem, rng)
    best_state = dict(init_state)
    best_score = penalized_objective(best_state, problem, penalty)
    history: list[float] = []
    iterations = 0
    iterations_per_restart: list[int] = []
    restart_boundaries = [0]

    for climb in range(max_restarts + 1):
        if climb == 0:
            climb_init = init_state
        else:
            climb_init = random_state(problem, rng)
            restart_boundaries.append(len(history))

        state, climb_history, climb_iterations, score = _steepest_climb(
            dict(climb_init), problem, penalty, max_iter
        )
        history.extend(climb_history)
        iterations += climb_iterations
        iterations_per_restart.append(climb_iterations)

        if score > best_score:
            best_score = score
            best_state = dict(state)
        if best_score >= total_value - EPSILON:
            break

    restarts = len(iterations_per_restart) - 1
    return SearchResult(
        algorithm="hill_climbing_restart",
        params={
            "variant": "restart",
            "max_iter": max_iter,
            "max_restarts": max_restarts,
        },
        seed=seed,
        problem=problem.name,
        initial_state=init_state,
        final_state=best_state,
        final_objective=objective(best_state, problem),
        feasible=is_feasible(best_state, problem),
        history=history,
        iterations=iterations,
        duration_s=time.perf_counter() - start,
        extra={
            "violations": violations(best_state, problem),
            "initial_objective": objective(init_state, problem),
            "initial_score": penalized_objective(init_state, problem, penalty),
            "final_score": best_score,
            "penalty_weight": penalty,
            "restarts": restarts,
            "iterations_per_restart": iterations_per_restart,
            "restart_boundaries": restart_boundaries,
        },
    )
