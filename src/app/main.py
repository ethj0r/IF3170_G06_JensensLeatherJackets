import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from algorithms.result import SearchResult
from app.plots import acceptance_figure, objective_figure, package_table, temperature_figure, truck_figure
from core.models import load_problem

SRC = Path(__file__).resolve().parent.parent
RESULTS_DIR = SRC / "results"
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

cols = st.columns(4)
cols[0].metric("Objective", f"{result.final_objective:g}")
cols[1].metric("Feasible", "ya" if result.feasible else "tidak")
cols[2].metric("Iterasi", result.iterations)
cols[3].metric("Durasi", f"{result.duration_s:.2f} s")

left, right = st.columns(2)
with left:
    st.plotly_chart(truck_figure(result.initial_state, problem, "Initial ·"), width="stretch")
with right:
    st.plotly_chart(truck_figure(result.final_state, problem, "Final ·"), width="stretch")

with st.expander("Daftar paket (final)"):
    st.dataframe(package_table(result.final_state, problem), hide_index=True, width="stretch")

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
