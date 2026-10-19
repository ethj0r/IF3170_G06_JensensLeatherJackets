import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from algorithms import genetic, hill_climbing, simulated_annealing
from algorithms.result import SearchResult
from core.models import Problem, load_problem, problem_to_dict
from core.neighbors import random_state
from core.objective import is_feasible
from core.objectives import (
    DEFAULT_OBJECTIVE,
    OBJECTIVE_NAMES,
    resolve_objective,
    state_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_GA_POPULATION_VARIATIONS = (25, 50, 100)
DEFAULT_GA_ITERATION_VARIATIONS = (100, 500, 1000)


@dataclass(frozen=True)
class Experiment:
    algorithm: str
    seed: int
    params: dict
    output_label: str

    # Menghasilkan nama berkas unik dari label konfigurasi dan seed.
    @property
    def filename(self) -> str:
        return f"{self.output_label}_seed{self.seed}.json"


# Menjalankan baseline random untuk membantu pemeriksaan representasi state.
def run_random(problem: Problem, seed: int, **params) -> SearchResult:
    start = time.perf_counter()
    state = random_state(problem, random.Random(seed))
    objective_name = params.get("objective_name", DEFAULT_OBJECTIVE)
    evaluator = resolve_objective(problem, objective_name)
    value = evaluator.evaluate(state)
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
        objective_name=objective_name,
        truck_count=problem.truck_count,
        extra={
            "initial_metrics": state_metrics(state, problem),
            "final_metrics": state_metrics(state, problem),
        },
    )


ALGORITHMS = {
    "random": run_random,
    "hc": hill_climbing.run,
    "hc_stochastic": hill_climbing.run_stochastic,
    "hc_sideways": hill_climbing.run_sideways,
    "hc_restart": hill_climbing.run_restart,
    "sa": simulated_annealing.run,
    "ga": genetic.run,
}


# Memastikan tiga nilai variasi eksperimen berbeda dan dapat dijalankan GA.
def _validate_ga_variations(
    population_variations: tuple[int, ...],
    iteration_variations: tuple[int, ...],
) -> None:
    if len(population_variations) != 3 or len(set(population_variations)) != 3:
        raise ValueError("GA requires exactly three distinct population variations")
    if len(iteration_variations) != 3 or len(set(iteration_variations)) != 3:
        raise ValueError("GA requires exactly three distinct iteration variations")
    if any(population < 3 for population in population_variations):
        raise ValueError("GA population variations must be at least 3")
    if any(iterations < 0 for iterations in iteration_variations):
        raise ValueError("GA iteration variations must be non-negative")


# Menambahkan objective non-default ke parameter dan label hasil eksperimen.
def _objective_configuration(
    objective_name: str,
) -> tuple[dict[str, str], str]:
    if objective_name not in OBJECTIVE_NAMES:
        choices = ", ".join(OBJECTIVE_NAMES)
        raise ValueError(
            f"Unknown objective '{objective_name}'. Choose one of: {choices}"
        )
    if objective_name == DEFAULT_OBJECTIVE:
        return {}, ""
    return {"objective_name": objective_name}, f"_{objective_name}"


# Menyusun 24 eksperimen wajib: masing-masing tiga run HC dan SA, lalu
# dua studi GA yang masing-masing memiliki tiga konfigurasi dan tiga seed.
def build_all_experiments(
    *,
    seeds: tuple[int, int, int] = DEFAULT_SEEDS,
    max_iter: int = 500,
    fixed_population: int = 100,
    fixed_iterations: int = 500,
    population_variations: tuple[int, ...] = DEFAULT_GA_POPULATION_VARIATIONS,
    iteration_variations: tuple[int, ...] = DEFAULT_GA_ITERATION_VARIATIONS,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> list[Experiment]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("all experiments require exactly three distinct seeds")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if fixed_population < 3:
        raise ValueError("fixed GA population must be at least 3")
    if fixed_iterations < 0:
        raise ValueError("fixed GA iterations must be non-negative")

    _validate_ga_variations(population_variations, iteration_variations)
    objective_params, objective_suffix = _objective_configuration(objective_name)

    experiments = [
        Experiment(
            "hc",
            seed,
            {"max_iter": max_iter, **objective_params},
            f"hc{objective_suffix}_iter{max_iter}",
        )
        for seed in seeds
    ]
    experiments.extend(
        Experiment("sa", seed, dict(objective_params), f"sa{objective_suffix}")
        for seed in seeds
    )

    for iterations in iteration_variations:
        experiments.extend(
            Experiment(
                "ga",
                seed,
                {
                    "population_size": fixed_population,
                    "iterations": iterations,
                    **objective_params,
                },
                f"ga{objective_suffix}_iteration_study_"
                f"pop{fixed_population}_iter{iterations}",
            )
            for seed in seeds
        )

    for population_size in population_variations:
        experiments.extend(
            Experiment(
                "ga",
                seed,
                {
                    "population_size": population_size,
                    "iterations": fixed_iterations,
                    **objective_params,
                },
                f"ga{objective_suffix}_population_study_"
                f"pop{population_size}_iter{fixed_iterations}",
            )
            for seed in seeds
        )

    return experiments


# Mengubah pilihan single-run CLI menjadi parameter dan nama berkas yang jelas.
def build_single_experiment(args: argparse.Namespace) -> Experiment:
    objective_name = getattr(args, "objective", DEFAULT_OBJECTIVE)
    objective_params, objective_suffix = _objective_configuration(objective_name)
    params = dict(objective_params)
    output_label = f"{args.algo}{objective_suffix}"

    if args.algo == "hc":
        params = {"max_iter": args.max_iter, **objective_params}
        output_label = f"hc{objective_suffix}_iter{args.max_iter}"
    elif args.algo == "hc_stochastic":
        params = {
            "max_iter": args.max_iter,
            "max_stuck": args.max_stuck,
            **objective_params,
        }
        output_label = (
            f"hc_stochastic{objective_suffix}_"
            f"iter{args.max_iter}_stuck{args.max_stuck}"
        )
    elif args.algo == "hc_sideways":
        params = {
            "max_iter": args.max_iter,
            "max_sideways": args.max_sideways,
            **objective_params,
        }
        output_label = (
            f"hc_sideways{objective_suffix}_"
            f"iter{args.max_iter}_sw{args.max_sideways}"
        )
    elif args.algo == "hc_restart":
        params = {
            "max_iter": args.max_iter,
            "max_restarts": args.max_restarts,
            **objective_params,
        }
        output_label = (
            f"hc_restart{objective_suffix}_"
            f"iter{args.max_iter}_rst{args.max_restarts}"
        )
    elif args.algo == "ga":
        params = {
            "population_size": args.population_size,
            "iterations": args.iterations,
            **objective_params,
        }
        output_label = (
            f"ga{objective_suffix}_"
            f"pop{args.population_size}_iter{args.iterations}"
        )

    return Experiment(args.algo, args.seed, params, output_label)


# Menjalankan satu konfigurasi dan menyimpan seluruh data yang dibutuhkan laporan.
def execute_experiment(
    experiment: Experiment,
    problem: Problem,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    result = ALGORITHMS[experiment.algorithm](
        problem,
        experiment.seed,
        **experiment.params,
    )
    result.problem_definition = problem_to_dict(problem)
    problem_prefix = f"{problem.name}_" if problem.is_multi_truck else ""
    output_path = output_dir / f"{problem_prefix}{experiment.filename}"
    result.save(output_path)
    print(
        f"{experiment.algorithm}: seed={experiment.seed} "
        f"objective={result.final_objective:g} feasible={result.feasible} "
        f"duration={result.duration_s:.3f}s -> {output_path}"
    )
    return output_path


# Membuat parser CLI untuk single run dan rangkaian eksperimen spesifikasi.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jalankan eksperimen local search")
    parser.add_argument("--problem", required=True, help="path ke problem JSON")
    parser.add_argument(
        "--objective",
        choices=OBJECTIVE_NAMES,
        default=DEFAULT_OBJECTIVE,
        help="objective function yang dioptimalkan",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="folder hasil eksperimen; default: output/ pada root repository",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--algo", choices=ALGORITHMS)
    mode.add_argument("--all", action="store_true", help="jalankan semua 24 eksperimen wajib")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=3,
        default=DEFAULT_SEEDS,
        metavar=("SEED_1", "SEED_2", "SEED_3"),
        help="tiga seed untuk mode --all",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="populasi single-run GA atau kontrol studi variasi iterasi",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="iterasi single-run GA atau kontrol studi variasi populasi",
    )
    parser.add_argument(
        "--ga-population-variations",
        type=int,
        nargs=3,
        default=DEFAULT_GA_POPULATION_VARIATIONS,
        metavar=("POP_1", "POP_2", "POP_3"),
        help="tiga variasi populasi GA untuk mode --all",
    )
    parser.add_argument(
        "--ga-iteration-variations",
        type=int,
        nargs=3,
        default=DEFAULT_GA_ITERATION_VARIATIONS,
        metavar=("ITER_1", "ITER_2", "ITER_3"),
        help="tiga variasi iterasi GA untuk mode --all",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="banyak iterasi maksimum Hill Climbing",
    )
    parser.add_argument(
        "--max-sideways",
        type=int,
        default=5,
        help="maksimum sideways move berurutan Hill-Climbing with Sideways Move",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="maksimum restart Random Restart Hill-Climbing",
    )
    parser.add_argument(
        "--max-stuck",
        type=int,
        default=100,
        help="maksimum penolakan neighbor berturut-turut Stochastic Hill-Climbing",
    )
    return parser


# Menjalankan mode CLI dan menampilkan ringkasan jumlah hasil yang dibuat.
def main() -> None:
    args = build_parser().parse_args()
    problem = load_problem(args.problem)

    if args.all:
        experiments = build_all_experiments(
            seeds=tuple(args.seeds),
            max_iter=args.max_iter,
            fixed_population=args.population_size,
            fixed_iterations=args.iterations,
            population_variations=tuple(args.ga_population_variations),
            iteration_variations=tuple(args.ga_iteration_variations),
            objective_name=args.objective,
        )
    else:
        experiments = [build_single_experiment(args)]

    print(
        f"Menjalankan {len(experiments)} eksperimen untuk problem {problem.name} "
        f"dengan objective {args.objective}"
    )
    for index, experiment in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}]", end=" ", flush=True)
        execute_experiment(experiment, problem, args.output_dir)
    print(f"Selesai. Hasil tersimpan di {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
