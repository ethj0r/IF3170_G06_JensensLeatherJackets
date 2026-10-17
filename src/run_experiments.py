import argparse
import random
import time
from pathlib import Path

from algorithms import genetic, hill_climbing, simulated_annealing
from algorithms.result import SearchResult
from core.models import Problem, load_problem
from core.neighbors import random_state
from core.objective import is_feasible, objective

RESULTS_DIR = Path(__file__).parent / "results"


def run_random(problem: Problem, seed: int, **params) -> SearchResult:
    start = time.perf_counter()
    state = random_state(problem, random.Random(seed))
    value = objective(state, problem)
    return SearchResult(
        algorithm="random",
        params=params,
        seed=seed,
        problem=problem.name,
        initial_state=state,
        final_state=state,
        final_objective=value,
        feasible=is_feasible(state, problem),
        history=[value],
        iterations=0,
        duration_s=time.perf_counter() - start,
    )


ALGORITHMS = {
    "random": run_random,
    "hc": hill_climbing.run,
    "sa": simulated_annealing.run,
    "ga": genetic.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan eksperimen local search")
    parser.add_argument("--problem", required=True, help="path ke problem JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--algo", choices=ALGORITHMS)
    mode.add_argument("--all", action="store_true", help="jalankan semua 24 run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="jumlah individu populasi Genetic Algorithm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="banyak iterasi Genetic Algorithm",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="banyak iterasi maksimum Hill Climbing",
    )
    args = parser.parse_args()

    problem = load_problem(args.problem)

    if args.all:
        raise NotImplementedError("Diisi di Fase 3")

    params = {}
    output_suffix = f"seed{args.seed}"
    if args.algo == "hc":
        params = {"max_iter": args.max_iter}
        output_suffix = f"iter{args.max_iter}_seed{args.seed}"
    elif args.algo == "ga":
        params = {
            "population_size": args.population_size,
            "iterations": args.iterations,
        }
        output_suffix = (
            f"pop{args.population_size}_iter{args.iterations}_seed{args.seed}"
        )

    result = ALGORITHMS[args.algo](problem, args.seed, **params)
    out = RESULTS_DIR / f"{args.algo}_{output_suffix}.json"
    result.save(out)
    print(f"{args.algo}: objective={result.final_objective} feasible={result.feasible} -> {out}")


if __name__ == "__main__":
    main()
