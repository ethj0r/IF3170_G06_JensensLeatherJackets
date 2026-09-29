from core.models import Package, Truck
from core.objective import ORIENTATIONS, in_bounds, oriented_dims, rotate

PKG = Package(id="A", w=3, l=4, h=1, value=10, weight=1)
TRUCK = Truck(w=10, l=20, h=5, max_capacity=100)


def test_ada_enam_orientasi():
    assert len(set(ORIENTATIONS)) == 6


def test_oriented_dims():
    assert oriented_dims(PKG.dims, (0, 1, 2)) == (3, 4, 1)
    assert oriented_dims(PKG.dims, (2, 0, 1)) == (1, 3, 4)


def test_rotate_dua_kali_kembali_semula():
    for axis in "xyz":
        for o in ORIENTATIONS:
            assert rotate(rotate(o, axis), axis) == o


def test_rotate_z_menukar_lebar_dan_panjang():
    o = rotate((0, 1, 2), "z")
    assert oriented_dims(PKG.dims, o) == (4, 3, 1)


def test_in_bounds():
    assert in_bounds(PKG, (7, 16, 4, (0, 1, 2)), TRUCK)
    assert not in_bounds(PKG, (8, 0, 0, (0, 1, 2)), TRUCK)
    assert not in_bounds(PKG, (0, 0, -1, (0, 1, 2)), TRUCK)
