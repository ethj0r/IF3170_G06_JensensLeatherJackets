from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import run_experiments
from algorithms.result import SearchResult


APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.mark.parametrize("algorithm", run_experiments.ALGORITHMS)
def test_run_algorithm_from_interface(monkeypatch, tmp_path, algorithm):
    monkeypatch.setattr(run_experiments, "OUTPUT_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    assert not app.exception
    assert not list(tmp_path.glob("*.json"))

    app.selectbox(key="algorithm").select(algorithm).run()
    assert not app.exception
    for widget in app.number_input:
        if widget.label == "Maksimum iterasi":
            widget.set_value(2)
        elif widget.label == "Maksimum restart":
            widget.set_value(1)
        elif widget.label == "Ukuran populasi":
            widget.set_value(3)
        elif widget.label == "Jumlah generasi":
            widget.set_value(1)
    app.button(key="run_experiment").click().run()

    assert not app.exception
    assert not app.error
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    result = SearchResult.load(paths[0])
    assert result.problem == "tiny"
    assert result.seed == 0
    assert app.metric[0].value == f"{result.final_objective:g}"

    # Mengubah pilihan tidak boleh menjalankan ulang algoritma atau menghilangkan hasil.
    app.selectbox(key="dataset").select(APP.parents[1] / "data" / "large.json").run()
    assert not app.exception
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert app.metric[0].value == f"{result.final_objective:g}"

    app.segmented_control(key="mode").set_value("Hasil tersimpan").run()
    assert not app.exception
    assert app.metric[0].value == f"{result.final_objective:g}"


def test_sa_invalid_temperatures_do_not_save_result(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "OUTPUT_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.selectbox(key="algorithm").select("sa").run()
    next(widget for widget in app.number_input if widget.label == "Temperatur minimum").set_value(200.0)
    app.button(key="run_experiment").click().run()
    assert not app.exception
    assert app.error
    assert not list(tmp_path.glob("*.json"))


def test_repeated_runs_save_distinct_results(monkeypatch, tmp_path):
    monkeypatch.setattr(run_experiments, "OUTPUT_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.selectbox(key="algorithm").select("random").run()
    app.button(key="run_experiment").click().run()
    app.button(key="run_experiment").click().run()
    assert not app.exception
    assert len(list(tmp_path.glob("*.json"))) == 2


@pytest.mark.parametrize("max_iter", [100, 500, 1000])
def test_manual_sa_report_inputs(monkeypatch, tmp_path, max_iter):
    monkeypatch.setattr(run_experiments, "OUTPUT_DIR", tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.selectbox(key="dataset").select(APP.parents[1] / "data" / "large.json").run()
    app.selectbox(key="algorithm").select("sa").run()
    app.number_input(key="sa_max_iter").set_value(max_iter)
    app.number_input(key="sa_seed").set_value(2)
    app.button(key="run_experiment").click().run()

    assert not app.exception
    assert not app.error
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    result = SearchResult.load(paths[0])
    assert result.problem == "large"
    assert result.problem_definition is not None
    assert result.objective_name == "total_value"
    assert result.seed == 2
    assert result.iterations == max_iter
    assert result.params["max_iter"] == max_iter
    assert result.params["penalty_weight"] == 760.0
    assert result.params["stuck_threshold"] == 500
    assert result.feasible

    # Mengganti seed untuk percobaan berikutnya tidak mengeksekusi run tambahan.
    app.number_input(key="sa_seed").set_value(1).run()
    assert not app.exception
    assert len(list(tmp_path.glob("*.json"))) == 1
