import json
from dataclasses import dataclass
from pathlib import Path

Orientation = tuple[int, int, int]  # permutasi (w, l, h)
Placement = tuple[int, int, int, Orientation]  # (x, y, z, orientation)
State = dict[str, Placement | None]  # None = paket di luar truk


@dataclass(frozen=True)
class Package:
    id: str
    w: int
    l: int
    h: int
    value: int
    weight: int
    is_fragile: bool = False
    eta: int = 0

    @property
    def dims(self) -> tuple[int, int, int]:
        return (self.w, self.l, self.h)


@dataclass(frozen=True)
class Truck:
    w: int
    l: int
    h: int
    max_capacity: int


@dataclass(frozen=True)
class Problem:
    name: str
    truck: Truck
    packages: dict[str, Package]


def load_problem(path: str | Path) -> Problem:
    with open(path) as f:
        raw = json.load(f)

    packages = {p["id"]: Package(**p) for p in raw["packages"]}
    if len(packages) != len(raw["packages"]):
        raise ValueError("There are duplicated package IDs")

    return Problem(
        name=raw.get("name", Path(path).stem),
        truck=Truck(**raw["truck"]),
        packages=packages,
    )


def state_from_json(raw: dict) -> State:
    return {
        pid: None if p is None else (p[0], p[1], p[2], tuple(p[3]))
        for pid, p in raw.items()
    }
