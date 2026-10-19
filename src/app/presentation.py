from pathlib import Path

from algorithms.result import SearchResult
from core.models import Problem, load_problem, problem_from_dict


# Memuat problem embedded pada hasil baru atau file data untuk hasil legacy.
def load_result_problem(result: SearchResult, data_dir: Path) -> Problem:
    if result.problem_definition is not None:
        return problem_from_dict(
            result.problem_definition,
            default_name=result.problem,
        )

    problem_path = data_dir / f"{result.problem}.json"
    if not problem_path.exists():
        raise FileNotFoundError(
            f"Problem definition not found for result '{result.problem}'. "
            f"Expected embedded data or {problem_path}."
        )
    return load_problem(problem_path)


# Menyusun metadata algoritma yang diwajibkan spesifikasi untuk visualizer.
def experiment_details(result: SearchResult) -> list[tuple[str, object]]:
    params = result.params
    extra = result.extra

    if result.algorithm == "genetic":
        return [
            ("Jumlah populasi", params.get("population_size", "-")),
            ("Iterasi GA", params.get("iterations", result.iterations)),
            ("Iterasi terbaik", extra.get("best_iteration", "-")),
        ]
    if result.algorithm == "hill_climbing_stochastic":
        return [
            ("Iterasi diterima", result.iterations),
            ("Kandidat dievaluasi", extra.get("proposals", "-")),
            ("Maksimum stuck", params.get("max_stuck", "-")),
        ]
    if result.algorithm == "hill_climbing_sideways":
        return [
            ("Iterasi diterima", result.iterations),
            ("Sideways move", extra.get("sideways_moves", "-")),
            ("Maksimum sideways", params.get("max_sideways", "-")),
        ]
    if result.algorithm == "hill_climbing_restart":
        return [
            ("Total iterasi", result.iterations),
            ("Jumlah restart", extra.get("restarts", "-")),
            ("Maksimum restart", params.get("max_restarts", "-")),
        ]
    if result.algorithm == "hill_climbing_steepest":
        return [("Iterasi hingga berhenti", result.iterations)]
    if result.algorithm == "simulated_annealing":
        return [
            ("Iterasi", result.iterations),
            ("Langkah buruk diterima", extra.get("accepted_worse", "-")),
            ("Frekuensi stuck", extra.get("stuck_count", "-")),
        ]
    return [("Iterasi", result.iterations)]
