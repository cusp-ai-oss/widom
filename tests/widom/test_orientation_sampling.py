"""Distribution and rigid-body checks independent of stored sample values."""

import numpy as np
from ase import Atoms
from ase.build import molecule

from widom.utils import sample_gas_positions


def frame(positions):
    first = positions[..., 1, :] - positions[..., 0, :]
    second = positions[..., 2, :] - positions[..., 0, :]
    first -= 100 * np.round(first / 100)
    second -= 100 * np.round(second / 100)
    x = first / np.linalg.norm(first, axis=-1, keepdims=True)
    z = np.cross(x, second)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def test_nonlinear_molecular_frames_are_uniform_rotations():
    n, seed = 8192, 5791
    gas = molecule("H2O")
    original = gas.positions.copy()
    positions = sample_gas_positions(
        Atoms(cell=[100] * 3, pbc=True), gas, n, np.random.default_rng(seed)
    )
    axes = frame(positions)
    rotations = axes @ frame(original).T
    np.testing.assert_allclose(axes.mean(axis=0), 0, atol=6 * np.sqrt(1 / (3 * n)), rtol=0)
    np.testing.assert_allclose(
        (axes**2).mean(axis=0), 1 / 3, atol=6 * np.sqrt(4 / (45 * n)), rtol=0
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), (n, 3, 3)),
        atol=1e-12,
        rtol=0,
    )
    traces = np.trace(rotations, axis1=1, axis2=2)
    assert abs(traces.mean()) < 0.08
    assert abs((traces**2).mean() - 1) < 0.12
    centers = 100 * np.random.default_rng(seed).random((n, 3))
    relative = positions - centers[:, None, :]
    relative -= 100 * np.round(relative / 100)
    np.testing.assert_allclose(
        np.einsum("nij,i->nj", relative, gas.get_masses()) / gas.get_masses().sum(),
        0,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_array_equal(gas.positions, original)


def test_linear_molecule_orientations_are_isotropic():
    n = 8192
    host = Atoms(cell=[100, 100, 100], pbc=True)
    positions = sample_gas_positions(host, molecule("N2"), n, np.random.default_rng(2718))
    vectors = positions[:, 1] - positions[:, 0]
    vectors -= 100 * np.round(vectors / 100)
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    np.testing.assert_allclose(unit.mean(axis=0), 0, atol=6 * np.sqrt(1 / (3 * n)), rtol=0)
    np.testing.assert_allclose(
        (unit**2).mean(axis=0), 1 / 3, atol=6 * np.sqrt(4 / (45 * n)), rtol=0
    )
    p2 = (3 * np.mean(unit[:, 2] ** 2) - 1) / 2
    assert abs(p2) <= 6 * np.sqrt(0.2 / n)


def test_asymmetric_molecule_geometry_and_fixed_seed_are_preserved():
    host = Atoms(cell=[100, 100, 100], pbc=True)
    gas = molecule("H2O")
    before = gas.positions.copy()
    first = sample_gas_positions(host, gas, 32, np.random.default_rng(31415))
    second = sample_gas_positions(host, gas, 32, np.random.default_rng(31415))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(gas.positions, before)
    expected = gas.get_all_distances()
    deltas = first[:, :, None, :] - first[:, None, :, :]
    deltas -= 100 * np.round(deltas / 100)
    actual = np.linalg.norm(deltas, axis=-1)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-10, rtol=0)


def test_zero_insertions_preserve_position_array_shape():
    host = Atoms(cell=[100, 100, 100], pbc=True)
    gas = Atoms("Ar", positions=[[0, 0, 0]])
    empty = sample_gas_positions(host, gas, 0, np.random.default_rng(2718))
    assert empty.shape == (0, 1, 3)


def test_monatomic_gas_positions_match_uniform_fractional_samples():
    host = Atoms(cell=[100, 100, 100], pbc=True)
    gas = Atoms("Ar", positions=[[0, 0, 0]])
    seed = 31415
    expected = 100 * np.random.default_rng(seed).random((8, 3))
    actual = sample_gas_positions(host, gas, 8, np.random.default_rng(seed))
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-12, rtol=0)
