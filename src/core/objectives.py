from dataclasses import dataclass

from core.fleet import fleet_metrics
from core.models import Problem, State

DEFAULT_OBJECTIVE = "total_value"
PRIORITY_OBJECTIVE = "priority_value"


@dataclass(frozen=True)
class ObjectiveEvaluator:
    name: str
    label: str
    package_utilities: dict[str, float]

    # Menghitung objective state dari utility paket yang berada di dalam truk.
    def evaluate(self, state: State) -> float:
        return float(sum(
            self.package_utilities[package_id]
            for package_id, placement in state.items()
            if placement is not None
        ))

    # Menghasilkan batas atas objective ketika seluruh paket dapat dimasukkan.
    @property
    def maximum_score(self) -> float:
        return float(sum(self.package_utilities.values()))


# Membentuk evaluator objective total value sesuai spesifikasi utama.
def _build_total_value(problem: Problem) -> ObjectiveEvaluator:
    return ObjectiveEvaluator(
        name=DEFAULT_OBJECTIVE,
        label="Total value",
        package_utilities={
            package_id: float(package.value)
            for package_id, package in problem.packages.items()
        },
    )


# Membentuk objective bonus yang menyeimbangkan value dan urgensi ETA.
# ETA terkecil mendapat pengali terbesar, sedangkan ETA terbesar tetap
# memperoleh utility sebesar value aslinya.
def _build_priority_value(problem: Problem) -> ObjectiveEvaluator:
    if any(package.eta < 0 for package in problem.packages.values()):
        raise ValueError("priority_value requires non-negative ETA values")

    maximum_eta = max(
        (package.eta for package in problem.packages.values()),
        default=0,
    )
    eta_scale = max(1, maximum_eta)
    utilities = {}

    for package_id, package in problem.packages.items():
        urgency = (maximum_eta - package.eta) / eta_scale
        utilities[package_id] = float(package.value * (1.0 + urgency))

    return ObjectiveEvaluator(
        name=PRIORITY_OBJECTIVE,
        label="Priority-adjusted value",
        package_utilities=utilities,
    )


OBJECTIVE_BUILDERS = {
    DEFAULT_OBJECTIVE: _build_total_value,
    PRIORITY_OBJECTIVE: _build_priority_value,
}
OBJECTIVE_NAMES = tuple(OBJECTIVE_BUILDERS)


# Memilih objective berdasarkan nama dan mengikat utility ke problem instance.
def resolve_objective(
    problem: Problem,
    objective_name: str = DEFAULT_OBJECTIVE,
) -> ObjectiveEvaluator:
    try:
        builder = OBJECTIVE_BUILDERS[objective_name]
    except KeyError as error:
        choices = ", ".join(OBJECTIVE_NAMES)
        raise ValueError(
            f"Unknown objective '{objective_name}'. Choose one of: {choices}"
        ) from error
    return builder(problem)


# Menghasilkan metrik state yang tetap dapat dibandingkan lintas objective.
def state_metrics(state: State, problem: Problem) -> dict[str, object]:
    loaded_packages = [
        problem.packages[package_id]
        for package_id, placement in state.items()
        if placement is not None
    ]
    loaded_count = len(loaded_packages)
    metrics: dict[str, object] = {
        "total_value": float(sum(package.value for package in loaded_packages)),
        "total_weight": sum(package.weight for package in loaded_packages),
        "loaded_packages": loaded_count,
        "average_eta": (
            float(sum(package.eta for package in loaded_packages) / loaded_count)
            if loaded_count
            else 0.0
        ),
    }
    if problem.is_multi_truck:
        metrics["truck_metrics"] = fleet_metrics(state, problem)
    return metrics
