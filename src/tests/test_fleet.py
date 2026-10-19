import json
import random
from pathlib import Path

import pytest

from algorithms import genetic, hill_climbing, simulated_annealing
from algorithms.result import SearchResult
from app.plots import package_table, truck_figure
from core.fleet import assigned_truck_id, fleet_metrics, fleet_weights
from core.models import Package, Problem, State, Truck, load_problem
from core.neighbors import all_neighbors, random_neighbor, relocate
from core.objective import is_feasible, objective, violations
from core.packing import random_feasible_state, repair_state

IDENTITY = (0, 1, 2)
MULTI = Path(__file__).parent.parent / "data" / "multi_truck.json"


# Membuat problem kecil agar perilaku constraint per truk mudah diverifikasi.
def make_fleet_problem(max_capacity: int = 20) -> Problem:
    trucks = {
        "T01": Truck(2, 2, 2, max_capacity),
        "T02": Truck(2, 2, 2, max_capacity),
    }
    packages = {
        "A": Package("A", 2, 2, 1, value=100, weight=12),
        "B": Package("B", 2, 2, 1, value=80, weight=12),
        "F": Package("F", 2, 2, 1, value=30, weight=3, is_fragile=True),
    }
    return Problem("fleet", trucks["T01"], packages, trucks)


# Membuat state armada lengkap dengan paket yang tidak disebut berada di luar.
def make_state(**placements) -> State:
    state: State = {"A": None, "B": None, "F": None}
    state.update(placements)
    return state


# Memastikan dataset armada heterogen dapat dibaca dengan identifier stabil.
def test_load_multi_truck_problem():
    problem = load_problem(MULTI)

    assert problem.is_multi_truck
    assert problem.truck_count == 3
    assert problem.truck_ids == ("T01", "T02", "T03")
    assert problem.truck_for("T02").max_capacity == 140
    assert len(problem.packages) == 18


# Memastikan truck_count membentuk armada identik tanpa merusak format lama.
def test_load_identical_fleet_with_truck_count(tmp_path):
    path = tmp_path / "identical.json"
    path.write_text(json.dumps({
        "name": "identical",
        "truck": {"w": 2, "l": 2, "h": 1, "max_capacity": 10},
        "truck_count": 3,
        "packages": [],
    }))

    problem = load_problem(path)

    assert problem.truck_ids == ("T01", "T02", "T03")
    assert len(set(problem.fleet.values())) == 1


# Memastikan konfigurasi armada ambigu dan identifier duplikat ditolak.
def test_invalid_fleet_configuration_is_rejected(tmp_path):
    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(json.dumps({
        "truck": {"w": 1, "l": 1, "h": 1, "max_capacity": 1},
        "trucks": [{"id": "T01", "w": 1, "l": 1, "h": 1, "max_capacity": 1}],
        "packages": [],
    }))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps({
        "trucks": [
            {"id": "T01", "w": 1, "l": 1, "h": 1, "max_capacity": 1},
            {"id": "T01", "w": 1, "l": 1, "h": 1, "max_capacity": 1},
        ],
        "packages": [],
    }))

    with pytest.raises(ValueError, match="exactly one"):
        load_problem(ambiguous)
    with pytest.raises(ValueError, match="Duplicated truck ID"):
        load_problem(duplicate)


# Memastikan koordinat sama pada truk berbeda tidak dianggap overlap.
def test_same_coordinates_on_different_trucks_are_feasible():
    problem = make_fleet_problem()
    state = make_state(
        A=("T01", 0, 0, 0, IDENTITY),
        B=("T02", 0, 0, 0, IDENTITY),
    )

    assert is_feasible(state, problem)
    assert objective(state, problem) == 180.0
    assert fleet_weights(state, problem) == {"T01": 12, "T02": 12}


# Memastikan kapasitas diperiksa per truk, bukan hanya sebagai total armada.
def test_capacity_is_enforced_for_each_truck():
    problem = make_fleet_problem(max_capacity=20)
    invalid = make_state(
        A=("T01", 0, 0, 0, IDENTITY),
        B=("T01", 0, 0, 1, IDENTITY),
    )

    assert sum(fleet_weights(invalid, problem).values()) <= 40
    assert not is_feasible(invalid, problem)
    assert violations(invalid, problem) > 0


# Memastikan support dan aturan fragile tidak dapat melintasi truk.
def test_support_is_local_to_each_truck():
    problem = make_fleet_problem()
    cross_truck = make_state(
        A=("T01", 0, 0, 0, IDENTITY),
        B=("T02", 0, 0, 1, IDENTITY),
    )
    fragile_support = make_state(
        F=("T01", 0, 0, 0, IDENTITY),
        B=("T01", 0, 0, 1, IDENTITY),
    )

    assert not is_feasible(cross_truck, problem)
    assert not is_feasible(fragile_support, problem)


# Memastikan random initialization dan repair selalu menghasilkan armada feasible.
def test_fleet_initialization_and_repair_are_feasible_and_reproducible():
    problem = make_fleet_problem()
    first = random_feasible_state(problem, random.Random(12))
    second = random_feasible_state(problem, random.Random(12))
    overlapping = make_state(
        A=("T01", 0, 0, 0, IDENTITY),
        B=("T01", 0, 0, 0, IDENTITY),
        F=("T02", 0, 0, 1, IDENTITY),
    )
    repaired = repair_state(overlapping, problem, random.Random(8))

    assert first == second
    assert is_feasible(first, problem)
    assert is_feasible(repaired, problem)
    assert set(repaired) == set(problem.packages)


# Memastikan neighborhood mencakup perpindahan paket antar-truk.
def test_neighbors_support_inter_truck_relocation():
    problem = make_fleet_problem()
    state = make_state(A=("T01", 0, 0, 0, IDENTITY))
    relocated = relocate(state, problem, "A", ("T02", 0, 0, 0))

    assert relocated is not None
    assert assigned_truck_id(relocated["A"], problem) == "T02"
    assert any(
        assigned_truck_id(neighbor["A"], problem) == "T02"
        for neighbor in all_neighbors(state, problem)
        if neighbor["A"] is not None
    )
    assert random_neighbor(state, problem, random.Random(3)) != state


# Memastikan state multi-truk tersimpan dan dimuat kembali tanpa kehilangan tipe.
def test_multi_truck_search_result_round_trip(tmp_path):
    problem = make_fleet_problem()
    state = make_state(A=("T01", 0, 0, 0, IDENTITY))
    result = SearchResult(
        algorithm="test",
        params={},
        seed=0,
        problem=problem.name,
        initial_state=state,
        final_state=state,
        final_objective=100.0,
        feasible=True,
        history=[100.0],
        iterations=0,
        duration_s=0.0,
        truck_count=2,
    )
    path = tmp_path / "fleet-result.json"

    result.save(path)
    loaded = SearchResult.load(path)

    assert loaded == result
    assert loaded.final_state["A"] == ("T01", 0, 0, 0, IDENTITY)


# Memastikan setiap truk dapat dirender terpisah dan tabel menampilkan assignment.
def test_multi_truck_visualization_projects_each_truck():
    problem = make_fleet_problem()
    state = make_state(
        A=("T01", 0, 0, 0, IDENTITY),
        B=("T02", 0, 0, 0, IDENTITY),
    )

    first_figure = truck_figure(state, problem, "Final ·", truck_id="T01")
    second_figure = truck_figure(state, problem, "Final ·", truck_id="T02")
    table = package_table(state, problem)

    assert "T01" in first_figure.layout.title.text
    assert "T02" in second_figure.layout.title.text
    assert len(first_figure.data) == 3
    assert len(second_figure.data) == 3
    assignments = dict(zip(table["id"], table["truck"]))
    assert assignments == {"A": "T01", "B": "T02", "F": "-"}


# Memastikan semua local-search dapat memproses armada dan melaporkan metriknya.
@pytest.mark.parametrize(
    ("runner", "params"),
    [
        (hill_climbing.run, {"max_iter": 40}),
        (
            hill_climbing.run_stochastic,
            {"max_iter": 40, "max_stuck": 30},
        ),
        (
            hill_climbing.run_sideways,
            {"max_iter": 40, "max_sideways": 5},
        ),
        (
            hill_climbing.run_restart,
            {"max_iter": 20, "max_restarts": 2},
        ),
        (simulated_annealing.run, {"max_iter": 100, "alpha": 0.95}),
        (genetic.run, {"population_size": 8, "iterations": 6, "elite_count": 1}),
    ],
)
def test_algorithms_support_multi_truck_problem(runner, params):
    problem = make_fleet_problem()
    result = runner(problem, seed=7, **params)

    assert result.truck_count == 2
    assert result.feasible
    assert is_feasible(result.final_state, problem)
    assert result.final_objective == objective(result.final_state, problem)
    metrics = result.extra["final_metrics"]
    assert set(metrics["truck_metrics"]) == {"T01", "T02"}
    assert sum(
        truck["total_value"]
        for truck in metrics["truck_metrics"].values()
    ) == metrics["total_value"]


# Memastikan regresi seed yang dahulu menghasilkan Stochastic HC infeasible teratasi.
def test_multi_truck_stochastic_default_seed_zero_is_feasible():
    problem = load_problem(MULTI)
    result = hill_climbing.run_stochastic(problem, seed=0)

    assert result.feasible
    assert is_feasible(result.final_state, problem)
    assert result.extra["violations"] == 0


# Memastikan armada dua truk dapat mengangkut value lebih besar dari satu truk.
def test_second_truck_increases_value_on_controlled_problem():
    fleet = make_fleet_problem(max_capacity=12)
    single = Problem("single", fleet.truck, fleet.packages)
    fleet_result = genetic.run(
        fleet,
        seed=4,
        population_size=10,
        iterations=8,
        elite_count=1,
    )
    single_result = genetic.run(
        single,
        seed=4,
        population_size=10,
        iterations=8,
        elite_count=1,
    )

    assert fleet_result.final_objective > single_result.final_objective
    assert fleet_metrics(fleet_result.final_state, fleet)["T01"]["total_weight"] <= 12
    assert fleet_metrics(fleet_result.final_state, fleet)["T02"]["total_weight"] <= 12
