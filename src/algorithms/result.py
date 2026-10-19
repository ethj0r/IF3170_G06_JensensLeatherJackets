import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.models import State, state_from_json


@dataclass
class SearchResult:
    algorithm: str
    params: dict
    seed: int
    problem: str
    initial_state: State
    final_state: State
    final_objective: float
    feasible: bool
    history: list[float]  # objektif per iterasi (GA: maksimum populasi)
    iterations: int
    duration_s: float
    objective_name: str = "total_value"
    truck_count: int = 1
    extra: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SearchResult":
        with open(path) as f:
            raw = json.load(f)
        raw["initial_state"] = state_from_json(raw["initial_state"])
        raw["final_state"] = state_from_json(raw["final_state"])
        return cls(**raw)
