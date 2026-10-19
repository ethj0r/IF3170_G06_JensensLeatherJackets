import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from algorithms.result import SearchResult
from app.plots import acceptance_figure, objective_figure, package_table, temperature_figure, truck_figure
from core.models import load_problem
from core.objectives import DEFAULT_OBJECTIVE, resolve_objective, state_metrics

SRC = Path(__file__).resolve().parent.parent
RESULTS_DIR = SRC.parent / "output"
DATA_DIR = SRC / "data"

st.set_page_config(page_title="Truck Packing", layout="wide")
st.title("Truck Packing")

files = sorted(RESULTS_DIR.glob("*.json"))
if not files:
    st.info("Belum ada hasil. Jalankan run_experiments.py dulu.")
    st.stop()

path = st.selectbox("Hasil", files, format_func=lambda p: p.name)
result = SearchResult.load(path)
problem = load_problem(DATA_DIR / f"{result.problem}.json")
evaluator = resolve_objective(problem, result.objective_name)
final_metrics = result.extra.get(
    "final_metrics",
    state_metrics(result.final_state, problem),
)

metrics = [
    (evaluator.label, f"{result.final_objective:g}"),
]
if result.objective_name != DEFAULT_OBJECTIVE:
    metrics.append(("Total value", f"{final_metrics['total_value']:g}"))
metrics.extend([
    ("Feasible", "ya" if result.feasible else "tidak"),
    ("Iterasi", result.iterations),
    ("Durasi", f"{result.duration_s:.2f} s"),
])

cols = st.columns(len(metrics))
for column, (label, value) in zip(cols, metrics):
    column.metric(label, value)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        truck_figure(
            result.initial_state,
            problem,
            "Initial ·",
            result.objective_name,
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
        ),
        width="stretch",
    )

with st.expander("Daftar paket (final)"):
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
    st.write(
        f"Langkah memburuk yang diterima: {result.extra['accepted_worse']} · "
        f"stuck: {result.extra['stuck_count']}"
    )
