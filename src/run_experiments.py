import argparse
import random
import time
from pathlib import Path

import pandas as pd

from algorithms import genetic, hill_climbing, simulated_annealing
from algorithms.result import SearchResult
from core.models import Problem, load_problem
from core.neighbors import random_state
from core.objective import is_feasible, objective

RESULTS_DIR = Path(__file__).parent / "results"
SEEDS = (1, 2, 3)
GA_POPULATION_CONTROL = 50
GA_ITERATION_CONTROL = 100
GA_ITERATIONS = (50, 100, 200)
GA_POPULATIONS = (20, 50, 100)


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


def experiment_plan() -> list[tuple[str, str, dict]]:
    plan = [("random", "random", {}), ("hc", "hc", {}), ("sa", "sa", {})]
    configs = {(GA_POPULATION_CONTROL, i) for i in GA_ITERATIONS}
    configs |= {(p, GA_ITERATION_CONTROL) for p in GA_POPULATIONS}
    for pop, iters in sorted(configs):
        plan.append(("ga", f"ga_pop{pop}_iter{iters}", {"population_size": pop, "iterations": iters}))
    return plan


def run_all(problem: Problem, skip: list[str]) -> None:
    rows = []
    for algo, name, params in experiment_plan():
        if algo in skip:
            continue
        for run, seed in enumerate(SEEDS, start=1):
            result = ALGORITHMS[algo](problem, seed, **params)
            out = RESULTS_DIR / f"{name}_run{run}.json"
            result.save(out)
            print(f"{out.name}: objective={result.final_objective} feasible={result.feasible} "
                  f"{result.duration_s:.1f}s")
            rows.append({
                "config": name,
                "objective": result.final_objective,
                "feasible": result.feasible,
                "duration_s": result.duration_s,
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("config", sort=False).agg(
        mean=("objective", "mean"),
        min=("objective", "min"),
        max=("objective", "max"),
        std=("objective", "std"),
        feasible=("feasible", "sum"),
        duration_s=("duration_s", "mean"),
    )
    print()
    print(summary.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan eksperimen local search")
    parser.add_argument("--problem", required=True, help="path ke problem JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--algo", choices=ALGORITHMS)
    mode.add_argument("--all", action="store_true", help="jalankan semua run eksperimen")
    parser.add_argument("--skip", nargs="*", choices=ALGORITHMS, default=[], help="algoritma yang dilewati pada --all")
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
    args = parser.parse_args()

    problem = load_problem(args.problem)

    if args.all:
        run_all(problem, args.skip)
        return

    params = {}
    output_suffix = f"seed{args.seed}"
    if args.algo == "ga":
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
