import json
from dataclasses import dataclass
from pathlib import Path

Orientation = tuple[int, int, int]  # permutasi (w, l, h)
Placement = tuple[int, int, int, Orientation]  # (x, y, z, orientation)
FleetPlacement = tuple[str, int, int, int, Orientation]
StatePlacement = Placement | FleetPlacement
State = dict[str, StatePlacement | None]  # None = paket di luar armada

DEFAULT_TRUCK_ID = "T01"


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
    trucks: dict[str, Truck] | None = None

    # Menghasilkan seluruh truk dalam bentuk mapping yang seragam.
    @property
    def fleet(self) -> dict[str, Truck]:
        if self.trucks is not None:
            return self.trucks
        return {DEFAULT_TRUCK_ID: self.truck}

    # Menghasilkan urutan identifier truk yang stabil sesuai input.
    @property
    def truck_ids(self) -> tuple[str, ...]:
        return tuple(self.fleet)

    # Menghasilkan jumlah truk dalam armada.
    @property
    def truck_count(self) -> int:
        return len(self.fleet)

    # Menandai problem yang membutuhkan representasi placement multi-truk.
    @property
    def is_multi_truck(self) -> bool:
        return self.truck_count > 1

    # Mengambil konfigurasi truk berdasarkan identifier.
    def truck_for(self, truck_id: str) -> Truck:
        try:
            return self.fleet[truck_id]
        except KeyError as error:
            raise KeyError(f"Unknown truck ID: {truck_id}") from error


# Membaca satu objek truk dan memisahkan identifier dari atribut dimensinya.
def _parse_truck(raw: dict, default_id: str) -> tuple[str, Truck]:
    truck_data = dict(raw)
    truck_id = truck_data.pop("id", default_id)
    if not isinstance(truck_id, str) or not truck_id:
        raise ValueError("Truck ID must be a non-empty string")
    return truck_id, Truck(**truck_data)


# Membaca konfigurasi satu truk, armada identik, atau daftar truk heterogen.
def _load_trucks(raw: dict) -> dict[str, Truck]:
    has_single = "truck" in raw
    has_multiple = "trucks" in raw
    if has_single == has_multiple:
        raise ValueError("Problem must define exactly one of 'truck' or 'trucks'")

    if has_multiple:
        if "truck_count" in raw:
            raise ValueError("truck_count cannot be combined with trucks")
        raw_trucks = raw["trucks"]
        if not isinstance(raw_trucks, list) or not raw_trucks:
            raise ValueError("trucks must be a non-empty list")
        trucks: dict[str, Truck] = {}
        for index, raw_truck in enumerate(raw_trucks, start=1):
            truck_id, truck = _parse_truck(raw_truck, f"T{index:02d}")
            if truck_id in trucks:
                raise ValueError(f"Duplicated truck ID: {truck_id}")
            trucks[truck_id] = truck
        return trucks

    truck_count = raw.get("truck_count", 1)
    if type(truck_count) is not int or truck_count < 1:
        raise ValueError("truck_count must be a positive integer")
    _, truck = _parse_truck(raw["truck"], DEFAULT_TRUCK_ID)
    return {
        f"T{index:02d}": truck
        for index in range(1, truck_count + 1)
    }


def load_problem(path: str | Path) -> Problem:
    with open(path) as f:
        raw = json.load(f)

    packages = {p["id"]: Package(**p) for p in raw["packages"]}
    if len(packages) != len(raw["packages"]):
        raise ValueError("There are duplicated package IDs")

    trucks = _load_trucks(raw)

    return Problem(
        name=raw.get("name", Path(path).stem),
        truck=next(iter(trucks.values())),
        packages=packages,
        trucks=trucks if len(trucks) > 1 else None,
    )


def state_from_json(raw: dict) -> State:
    state: State = {}
    for package_id, placement in raw.items():
        if placement is None:
            state[package_id] = None
        elif len(placement) == 4:
            state[package_id] = (
                placement[0],
                placement[1],
                placement[2],
                tuple(placement[3]),
            )
        elif len(placement) == 5:
            state[package_id] = (
                placement[0],
                placement[1],
                placement[2],
                placement[3],
                tuple(placement[4]),
            )
        else:
            raise ValueError(
                f"Invalid placement for package {package_id}: {placement}"
            )
    return state
