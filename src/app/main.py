import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from algorithms.result import SearchResult
from app.plots import (
    acceptance_figure,
    objective_figure,
    package_table,
    temperature_figure,
    truck_figure,
)
from app.presentation import experiment_details, load_result_problem
from core.models import load_problem, problem_to_dict
from core.objectives import DEFAULT_OBJECTIVE, OBJECTIVE_NAMES, resolve_objective, state_metrics
from run_experiments import ALGORITHMS, OUTPUT_DIR

SRC = Path(__file__).resolve().parent.parent
RESULTS_DIR = OUTPUT_DIR
DATA_DIR = SRC / "data"

st.set_page_config(page_title="Truck Packing", layout="wide")
st.title("Truck Packing")
st.caption("Optimasi penempatan paket tiga dimensi dengan local search")

st.session_state.setdefault("latest_experiment", None)
algorithm_labels = {
    "hc": "Hill Climbing",
    "hc_stochastic": "Stochastic Hill Climbing",
    "hc_sideways": "Hill Climbing dengan sideways move",
    "hc_restart": "Random Restart Hill Climbing",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
    "random": "Random (baseline)",
}
mode = st.segmented_control(
    "Mode", ["Eksperimen baru", "Hasil tersimpan"], default="Eksperimen baru", key="mode"
)

if mode == "Hasil tersimpan":
    files = sorted(
        path for path in RESULTS_DIR.rglob("*.json")
        if not path.name.endswith("_manifest.json")
    )
    if not files:
        st.info("Belum ada hasil. Pilih Eksperimen baru untuk menjalankan algoritma.")
        st.stop()
    path = st.selectbox(
        "Hasil", files, format_func=lambda p: str(p.relative_to(RESULTS_DIR))
    )
    try:
        result = SearchResult.load(path)
        problem = load_result_problem(result, DATA_DIR)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Hasil tidak dapat dibuka: {exc}")
        st.stop()
else:
    dataset_paths = sorted(DATA_DIR.glob("*.json"), key=lambda p: (p.stem != "tiny", p.stem))
    if not dataset_paths:
        st.info("Belum ada dataset di src/data/.")
        st.stop()
    left, right = st.columns(2)
    dataset = left.selectbox(
        "Dataset", dataset_paths, format_func=lambda p: p.stem, key="dataset"
    )
    algorithm = right.selectbox(
        "Algoritma", list(algorithm_labels),
        format_func=algorithm_labels.get, key="algorithm",
    )
    selected_problem = load_problem(dataset)
    objective_name = st.selectbox("Objective", OBJECTIVE_NAMES, key="objective_name")
    truck = selected_problem.truck
    st.caption(
        f"{len(selected_problem.packages)} paket · "
        f"{selected_problem.truck_count} truk · "
        f"Truk pertama: {truck.w} × {truck.l} × {truck.h} · "
        f"Kapasitas berat truk pertama {truck.max_capacity}"
    )

    with st.form(f"experiment_{algorithm}", enter_to_submit=False):
        seed = st.number_input("Seed", min_value=0, value=0, step=1, key=f"{algorithm}_seed")
        params = {"objective_name": objective_name}
        if algorithm.startswith("hc"):
            params["max_iter"] = st.number_input(
                "Maksimum iterasi", min_value=0, value=500, step=100, key=f"{algorithm}_max_iter"
            )
            if algorithm == "hc_stochastic":
                params["max_stuck"] = st.number_input(
                    "Maksimum penolakan berturut-turut", min_value=1, value=100, step=1
                )
            elif algorithm == "hc_sideways":
                params["max_sideways"] = st.number_input(
                    "Maksimum sideways move", min_value=0, value=5, step=1
                )
            elif algorithm == "hc_restart":
                params["max_restarts"] = st.number_input(
                    "Maksimum restart", min_value=0, value=10, step=1
                )
        elif algorithm == "sa":
            left, right = st.columns(2)
            params["t0"] = left.number_input(
                "Temperatur awal", min_value=0.000001, value=100.0, format="%.6f"
            )
            params["alpha"] = right.number_input(
                "Faktor pendinginan (alpha)", min_value=0.000001,
                max_value=0.999999, value=0.99982, format="%.6f"
            )
            params["t_min"] = left.number_input(
                "Temperatur minimum", min_value=0.000001, value=0.01, format="%.6f"
            )
            params["max_iter"] = right.number_input(
                "Maksimum iterasi", min_value=0, value=100, step=1,
                key="sa_max_iter", help="Untuk laporan: 100, 500, atau 1000."
            )
            params["penalty_weight"] = left.number_input(
                "Bobot penalti", min_value=0.000001,
                value=max(resolve_objective(selected_problem, objective_name).package_utilities.values(), default=1.0),
                format="%.2f", key=f"sa_penalty_{dataset.stem}_{objective_name}",
            )
            params["stuck_threshold"] = right.number_input(
                "Stuck threshold", min_value=1, value=500, step=1, key="sa_stuck_threshold"
            )
        elif algorithm == "ga":
            left, right = st.columns(2)
            params["population_size"] = left.number_input(
                "Ukuran populasi", min_value=3, value=100, step=1
            )
            params["iterations"] = right.number_input(
                "Jumlah generasi", min_value=0, value=500, step=50
            )
            params["crossover_rate"] = left.number_input(
                "Peluang crossover", min_value=0.0, max_value=1.0, value=0.9, step=0.05
            )
            params["mutation_rate"] = right.number_input(
                "Peluang mutasi", min_value=0.0, max_value=1.0, value=0.15, step=0.05
            )
        submitted = st.form_submit_button("Jalankan eksperimen", type="primary", key="run_experiment")

    if submitted:
        try:
            with st.spinner("Menjalankan eksperimen…", show_time=True):
                new_result = ALGORITHMS[algorithm](selected_problem, seed, **params)
                new_result.problem_definition = problem_to_dict(selected_problem)
            st.session_state.latest_experiment = (new_result, selected_problem, None)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Eksperimen gagal: {exc}")
            st.stop()
        else:
            filename = f"{dataset.stem}_{algorithm}_seed{seed}_{uuid4().hex}.json"
            path = RESULTS_DIR / filename
            try:
                new_result.save(path)
            except OSError as exc:
                st.error(f"Eksperimen selesai, tetapi hasil gagal disimpan: {exc}")
            else:
                st.session_state.latest_experiment = (new_result, selected_problem, path)

    if st.session_state.latest_experiment is None:
        st.info("Pilih dataset dan algoritma, atur parameter, lalu klik Jalankan eksperimen.")
        st.stop()
    result, problem, path = st.session_state.latest_experiment
    if path is not None:
        st.success(f"Hasil tersimpan di output/{path.name}")

st.subheader(f"Hasil: {result.problem} · {result.algorithm} · seed {result.seed}")
with st.expander("Parameter eksperimen"):
    st.json(result.params)

evaluator = resolve_objective(problem, result.objective_name)
final_metrics = result.extra.get("final_metrics", state_metrics(result.final_state, problem))

metrics = [
    (evaluator.label, f"{result.final_objective:g}"),
]
if result.objective_name != DEFAULT_OBJECTIVE:
    metrics.append(("Total value", f"{final_metrics['total_value']:g}"))
metrics.extend([
    ("Jumlah truk", problem.truck_count),
    ("Feasible", "ya" if result.feasible else "tidak"),
    ("Iterasi", result.iterations),
    ("Durasi", f"{result.duration_s:.2f} s"),
])

cols = st.columns(len(metrics))
for column, (label, value) in zip(cols, metrics):
    column.metric(label, value)

st.subheader("Detail eksperimen")
details = experiment_details(result)
detail_columns = st.columns(len(details))
for column, (label, value) in zip(detail_columns, details):
    column.metric(label, value)
if result.algorithm == "hill_climbing_restart":
    per_restart = result.extra.get("iterations_per_restart", [])
    st.caption(
        "Iterasi per climb: "
        + ", ".join(str(value) for value in per_restart)
    )

st.subheader("Visualisasi muatan")
truck_tabs = st.tabs(list(problem.truck_ids))
for tab, truck_id in zip(truck_tabs, problem.truck_ids):
    with tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                truck_figure(
                    result.initial_state,
                    problem,
                    "Initial ·",
                    result.objective_name,
                    truck_id,
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                truck_figure(
                    result.final_state,
                    problem,
                    "Final ·",
                    result.objective_name,
                    truck_id,
                ),
                width="stretch",
            )

st.subheader("Pemetaan seluruh paket")
initial_table_tab, final_table_tab = st.tabs(["Initial state", "Final state"])
with initial_table_tab:
    st.dataframe(
        package_table(result.initial_state, problem, result.objective_name),
        hide_index=True,
        width="stretch",
    )
with final_table_tab:
    st.dataframe(
        package_table(result.final_state, problem, result.objective_name),
        hide_index=True,
        width="stretch",
    )

st.plotly_chart(objective_figure(result), width="stretch")

if "acceptance_prob" in result.extra:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(acceptance_figure(result), width="stretch")
    with right:
        st.plotly_chart(temperature_figure(result), width="stretch")
