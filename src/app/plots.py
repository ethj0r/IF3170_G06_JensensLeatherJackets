import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from algorithms.result import SearchResult
from core.fleet import assigned_truck_id, project_truck_state, split_placement
from core.models import Problem, State
from core.objective import box
from core.objectives import (
    DEFAULT_OBJECTIVE,
    resolve_objective,
)

FACE_I = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
FACE_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
FACE_K = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
FRAGILE_COLOR = "#e4572e"


def corners(x, y, z, dx, dy, dz):
    return [
        (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
        (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz),
    ]


def edge_coords(points):
    xs, ys, zs = [], [], []
    for a, b in EDGES:
        xs += [points[a][0], points[b][0], None]
        ys += [points[a][1], points[b][1], None]
        zs += [points[a][2], points[b][2], None]
    return xs, ys, zs


def truck_figure(
    state: State,
    problem: Problem,
    title: str = "",
    objective_name: str = DEFAULT_OBJECTIVE,
    truck_id: str | None = None,
) -> go.Figure:
    selected_truck_id = truck_id or problem.truck_ids[0]
    truck = problem.truck_for(selected_truck_id)
    local_state = project_truck_state(state, problem, selected_truck_id)
    fig = go.Figure()
    edges_x, edges_y, edges_z = [], [], []
    palette = qualitative.Pastel + qualitative.Set3

    for n, (pid, placement) in enumerate(sorted(local_state.items())):
        if placement is None:
            continue
        pkg = problem.packages[pid]
        x, y, z, dx, dy, dz = box(pkg, placement)
        points = corners(x, y, z, dx, dy, dz)
        color = FRAGILE_COLOR if pkg.is_fragile else palette[n % len(palette)]
        hover = (
            f"<b>{pid}</b>{' (fragile)' if pkg.is_fragile else ''}<br>"
            f"posisi ({x}, {y}, {z})<br>ukuran {dx}×{dy}×{dz}<br>"
            f"value {pkg.value} · berat {pkg.weight}"
        )
        fig.add_trace(go.Mesh3d(
            x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
            i=FACE_I, j=FACE_J, k=FACE_K,
            color=color, opacity=0.9, flatshading=True,
            name=pid, hovertext=hover, hoverinfo="text",
        ))
        ex, ey, ez = edge_coords(points)
        edges_x += ex
        edges_y += ey
        edges_z += ez

    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z, mode="lines",
        line=dict(color="#333", width=2), hoverinfo="skip", showlegend=False,
    ))
    tx, ty, tz = edge_coords(corners(0, 0, 0, truck.w, truck.l, truck.h))
    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="lines",
        line=dict(color="#888", width=4, dash="dash"), hoverinfo="skip", showlegend=False,
    ))

    evaluator = resolve_objective(problem, objective_name)
    score = evaluator.evaluate(local_state)
    raw_value = resolve_objective(problem, DEFAULT_OBJECTIVE).evaluate(local_state)
    weight = sum(
        problem.packages[package_id].weight
        for package_id, placement in local_state.items()
        if placement is not None
    )
    objective_text = f"{evaluator.label} {score:g}"
    if objective_name != DEFAULT_OBJECTIVE:
        objective_text += f" · total value {raw_value:g}"
    truck_prefix = f"{selected_truck_id} · " if problem.is_multi_truck else ""
    fig.update_layout(
        title=(
            f"{title} {truck_prefix}{objective_text} · "
            f"berat {weight}/{truck.max_capacity}"
        ).strip(),
        scene=dict(
            xaxis=dict(title="x (w)", range=[0, truck.w]),
            yaxis=dict(title="y (l)", range=[0, truck.l]),
            zaxis=dict(title="z (h)", range=[0, truck.h]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


def package_table(
    state: State,
    problem: Problem,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> pd.DataFrame:
    evaluator = resolve_objective(problem, objective_name)
    rows = []
    for pid, pkg in sorted(problem.packages.items()):
        placement = state.get(pid)
        if placement is None:
            truck_id = "-"
            status, position, size = "luar", "-", f"{pkg.w}×{pkg.l}×{pkg.h}"
        else:
            truck_id = assigned_truck_id(placement, problem)
            _, local_placement = split_placement(placement, problem)
            x, y, z, dx, dy, dz = box(pkg, local_placement)
            status, position, size = "dalam", f"({x}, {y}, {z})", f"{dx}×{dy}×{dz}"
        rows.append({
            "id": pid,
            "status": status,
            "truck": truck_id,
            "posisi": position,
            "ukuran": size,
            "value": pkg.value,
            "eta": pkg.eta,
            "objective utility": evaluator.package_utilities[pid],
            "berat": pkg.weight,
            "fragile": pkg.is_fragile,
        })
    return pd.DataFrame(rows)


def objective_figure(result: SearchResult) -> go.Figure:
    extra = result.extra
    fig = go.Figure()
    if "average_history" in extra:
        fig.add_trace(go.Scatter(y=result.history, mode="lines", name="maksimum populasi"))
        fig.add_trace(go.Scatter(y=extra["average_history"], mode="lines", name="rata-rata populasi"))
    else:
        fig.add_trace(go.Scatter(y=result.history, mode="lines", name="current", line=dict(width=1)))
    if "best_feasible_history" in extra:
        fig.add_trace(go.Scatter(y=extra["best_feasible_history"], mode="lines", name="best feasible"))
    fig.update_layout(
        title=(
            f"{result.algorithm} · {result.objective_name} · seed {result.seed} "
            f"· final {result.final_objective:g}"
        ),
        xaxis_title="iterasi",
        yaxis_title=result.objective_name,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def acceptance_figure(result: SearchResult) -> go.Figure:
    probs = result.extra["acceptance_prob"]
    points = [(i, p) for i, p in enumerate(probs) if p is not None]
    fig = go.Figure(go.Scatter(
        x=[i for i, _ in points], y=[p for _, p in points],
        mode="markers", marker=dict(size=3, opacity=0.4),
    ))
    fig.update_layout(
        title="exp(Δ/T) untuk langkah yang memburuk",
        xaxis_title="iterasi",
        yaxis_title="probabilitas",
        yaxis_range=[0, 1.05],
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def temperature_figure(result: SearchResult) -> go.Figure:
    fig = go.Figure(go.Scatter(y=result.extra["temperature"], mode="lines"))
    fig.update_layout(
        title="suhu",
        xaxis_title="iterasi",
        yaxis_title="T",
        yaxis_type="log",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
