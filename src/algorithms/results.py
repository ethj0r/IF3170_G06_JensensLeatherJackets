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
    initialState: State
    finalState: State
    finalObjective: float
    feasible: bool
    history: list[float]
    iterations: int
    durationS: float
    extra: dict=field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "W") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SearchResult":
        with open(path) as f:
            raw = json.load(f)
        raw["initial_state"] = state_from_json(raw["initial_state"])
        raw["final_state"] = state_from_json(raw["final_state"])
        return cls(**raw)