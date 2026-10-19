import argparse
from pathlib import Path

import pytest

from run_experiments import (
    OUTPUT_DIR,
    Experiment,
    build_all_experiments,
    build_single_experiment,
    execute_experiment,
)
from core.models import load_problem

MULTI = Path(__file__).parent.parent / "data" / "multi_truck.json"


# Memastikan lokasi output sesuai struktur pengumpulan pada root repository.
def test_output_directory_is_at_repository_root():
    assert OUTPUT_DIR == Path(__file__).resolve().parents[2] / "output"


# Memastikan mode --all membentuk tepat 24 run dan tidak menimpa nama berkas.
def test_build_all_experiments_matches_required_matrix():
    experiments = build_all_experiments()

    assert len(experiments) == 24
    assert len({experiment.filename for experiment in experiments}) == 24
    assert sum(experiment.algorithm == "hc" for experiment in experiments) == 3
    assert sum(experiment.algorithm == "sa" for experiment in experiments) == 3
    assert sum(experiment.algorithm == "ga" for experiment in experiments) == 18


# Memastikan setiap konfigurasi GA dijalankan dengan tiga seed yang ditentukan.
def test_each_ga_configuration_has_three_runs():
    experiments = build_all_experiments(seeds=(7, 11, 19))
    ga_experiments = [experiment for experiment in experiments if experiment.algorithm == "ga"]
    configurations = {}

    for experiment in ga_experiments:
        configuration = (
            experiment.output_label.rsplit("_pop", 1)[0],
            experiment.params["population_size"],
            experiment.params["iterations"],
        )
        configurations.setdefault(configuration, set()).add(experiment.seed)

    assert len(configurations) == 6
    assert all(seeds == {7, 11, 19} for seeds in configurations.values())


# Memastikan variasi eksperimen yang ambigu atau tidak valid ditolak lebih awal.
@pytest.mark.parametrize(
    ("population_variations", "iteration_variations"),
    [
        ((25, 25, 100), (100, 500, 1000)),
        ((1, 50, 100), (100, 500, 1000)),
        ((25, 50, 100), (500, 500, 1000)),
        ((25, 50, 100), (-1, 500, 1000)),
    ],
)
def test_invalid_ga_variations_are_rejected(
    population_variations,
    iteration_variations,
):
    with pytest.raises(ValueError):
        build_all_experiments(
            population_variations=population_variations,
            iteration_variations=iteration_variations,
        )


# Memastikan parameter single-run GA diteruskan tanpa perubahan.
def test_build_single_ga_experiment():
    args = argparse.Namespace(
        algo="ga",
        seed=42,
        population_size=30,
        iterations=75,
        max_iter=500,
        max_stuck=100,
        max_sideways=5,
        max_restarts=10,
    )

    experiment = build_single_experiment(args)

    assert experiment.params == {"population_size": 30, "iterations": 75}
    assert experiment.filename == "ga_pop30_iter75_seed42.json"


# Memastikan objective bonus diteruskan dan dibedakan pada nama output.
def test_build_single_priority_objective_experiment():
    args = argparse.Namespace(
        algo="ga",
        seed=42,
        objective="priority_value",
        population_size=30,
        iterations=75,
        max_iter=500,
        max_stuck=100,
        max_sideways=5,
        max_restarts=10,
    )

    experiment = build_single_experiment(args)

    assert experiment.params == {
        "population_size": 30,
        "iterations": 75,
        "objective_name": "priority_value",
    }
    assert experiment.filename == (
        "ga_priority_value_pop30_iter75_seed42.json"
    )


# Memastikan mode --all juga membentuk 24 output bonus yang unik.
def test_build_all_priority_objective_experiments():
    experiments = build_all_experiments(objective_name="priority_value")

    assert len(experiments) == 24
    assert len({experiment.filename for experiment in experiments}) == 24
    assert all("priority_value" in experiment.filename for experiment in experiments)
    assert all(
        experiment.params["objective_name"] == "priority_value"
        for experiment in experiments
    )


# Memastikan output multi-truk tidak menimpa eksperimen problem satu-truk.
def test_multi_truck_output_uses_problem_prefix(tmp_path):
    problem = load_problem(MULTI)
    experiment = Experiment("random", 0, {}, "random")

    output_path = execute_experiment(experiment, problem, tmp_path)

    assert output_path.name == "multi_truck_random_seed0.json"
