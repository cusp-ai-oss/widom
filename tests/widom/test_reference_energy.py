"""Independent energy-reference controls for replicated Widom hosts."""

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from widom import run_widom_insertion
from widom.utils import calculate_atomic_density


class KnownInsertion(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, host_offset, direct=False):
        super().__init__()
        self.host_offset = host_offset
        self.direct = direct

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        numbers = self.atoms.numbers
        host_count = int((numbers == 1).sum())
        guest_present = bool((numbers == 7).any())
        interaction = -0.05 if host_count and guest_present else 0.0
        self.results = {
            "energy": interaction if self.direct else host_count * self.host_offset + interaction
        }


@pytest.mark.parametrize("minimum", [6.0, 12.0])
@pytest.mark.parametrize("offset", [-0.01, 0.01])
def test_total_energy_uses_the_simulated_host_reference(minimum, offset):
    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    original_positions = host.positions.copy()
    result = run_widom_insertion(
        KnownInsertion(offset),
        host,
        "N2",
        300.0,
        False,
        num_insertions=4,
        cutoff_distance=0.0,
        min_interplanar_distance=minimum,
        random_seed=1729,
    )
    expected_host_atoms = 2 if minimum == 6.0 else 16
    assert result.energy_structure == pytest.approx(expected_host_atoms * offset)
    np.testing.assert_allclose(result.interaction_energies, -0.05, atol=1e-12, rtol=0)
    kT = units._k * 300.0 / units._e
    expected = np.exp(0.05 / kT) / (units._k * units._Nav * 300.0 * calculate_atomic_density(host))
    assert result.henry_coefficient == pytest.approx(expected, rel=1e-10)
    # Returned energies and structure metadata must describe the same host.
    from ase.io import read
    from io import StringIO

    returned = read(StringIO(result.optimized_structure_cif), format="cif")
    assert len(returned) == expected_host_atoms
    assert len(host) == 2
    np.testing.assert_array_equal(host.positions, original_positions)


@pytest.mark.parametrize("minimum", [6.0, 12.0])
def test_direct_interaction_mode_remains_mass_normalized(minimum):
    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    result = run_widom_insertion(
        KnownInsertion(0.01, direct=True),
        host,
        "N2",
        300.0,
        True,
        num_insertions=4,
        cutoff_distance=0.0,
        min_interplanar_distance=minimum,
        random_seed=1729,
    )
    np.testing.assert_allclose(result.interaction_energies, -0.05, atol=1e-12, rtol=0)
    kT = units._k * 300.0 / units._e
    expected = np.exp(0.05 / kT) / (units._k * units._Nav * 300.0 * calculate_atomic_density(host))
    assert result.henry_coefficient == pytest.approx(expected, rel=1e-10)
