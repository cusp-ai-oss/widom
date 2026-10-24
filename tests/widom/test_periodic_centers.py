"""Explicit molecular-center controls independent of atom wrapping."""

import numpy as np
import pytest
from ase import Atoms


from widom.structure_preparation import prepare_structures_for_insertion
from widom.utils import check_accessibility, sample_gas_positions


def test_wrapped_equal_mass_molecule_is_rejected_at_its_periodic_center():
    poses = np.array([[[9.8, 5, 5], [0.2, 5, 5]]])
    accessible = check_accessibility(
        poses,
        np.array([[0.0, 5, 5]]),
        np.eye(3) * 10,
        0.5,
        True,
        gas_centers=np.array([[0.0, 5, 5]]),
    )
    assert not accessible[0]


def test_center_check_uses_supplied_center_in_a_skew_cell():
    cell = np.array([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 2.0, 8.0]])
    center = np.array([[0.1, 0.4, 0.6]]) @ cell
    # Deliberately misleading wrapped centroid; the sampled center is authoritative.
    poses = np.array([[[9.0, 5.0, 5.0], [1.0, 5.0, 5.0]]])
    periodic_host = center + cell[0]
    accessible = check_accessibility(
        poses,
        periodic_host,
        cell,
        0.2,
        True,
        gas_centers=center,
    )
    assert not accessible[0]


@pytest.mark.parametrize("centers", [None, np.zeros((2, 3)), [[np.nan, 0, 0]]])
def test_center_mode_requires_finite_explicit_center_data(centers):
    poses = np.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="gas_centers"):
        check_accessibility(
            poses,
            np.zeros((1, 3)),
            np.eye(3) * 10,
            0.5,
            True,
            gas_centers=centers,
        )


def test_default_atom_mode_needs_no_center_metadata():
    poses = np.array([[[9.8, 5, 5], [0.2, 5, 5]]])
    accessible = check_accessibility(
        poses,
        np.array([[0.0, 5, 5]]),
        np.eye(3) * 10,
        0.5,
        False,
    )
    assert not accessible[0]


@pytest.mark.parametrize(
    "cell",
    [
        np.eye(3) * 10,
        np.array([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 2.0, 8.0]]),
    ],
)
def test_preparation_passes_the_sampled_mass_center_before_wrapping(cell):
    seed = 1701
    center = np.random.default_rng(seed).random((1, 3)) @ cell
    host = Atoms("H", positions=center, cell=cell, pbc=True)
    # Unequal masses and a span greater than half the box prevent safe centroid
    # or minimum-image reconstruction from the wrapped coordinates alone.
    gas = Atoms("HHe", positions=[[0.0, 0, 0], [8.0, 0, 0]])
    before = gas.positions.copy()
    _, positions, accessible = prepare_structures_for_insertion(
        host,
        gas,
        num_insertions=1,
        cutoff_distance=0.1,
        cutoff_to_com=True,
        min_interplanar_distance=6.0,
        random_seed=seed,
    )
    assert not accessible[0]
    np.testing.assert_array_equal(gas.positions, before)
    expected = sample_gas_positions(host, gas, 1, np.random.default_rng(seed))
    np.testing.assert_array_equal(positions, expected)
