"""Exercise calculator/reference contracts with distinct ASE energy functions."""

import io

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.io import read


from widom import run_widom_insertion


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LennardJones(epsilon=0.01, sigma=2.0, rc=4.0),
        lambda: MorsePotential(epsilon=0.01, r0=2.0, rho0=6.0),
        EMT,
    ],
    ids=["lennard-jones", "morse", "emt"],
)
def test_replicated_reference_matches_explicit_calculator_queries(factory):
    host = Atoms("Cu2", positions=[[1, 1, 1], [3, 3, 3]], cell=[8] * 3, pbc=True)
    original_positions = host.positions.copy()
    original_cell = host.cell.array.copy()
    result = run_widom_insertion(
        factory(),
        host,
        "N2",
        300.0,
        False,
        num_insertions=16,
        cutoff_distance=1.0,
        min_interplanar_distance=12.0,
        random_seed=8101,
    )
    prepared = read(io.StringIO(result.optimized_structure_cif), format="cif")
    gas = molecule("N2")
    host_energy = factory().get_potential_energy(prepared)
    gas_energy = factory().get_potential_energy(gas)
    assert len(prepared) == 16
    assert result.energy_structure == pytest.approx(host_energy, abs=1e-8)
    assert result.energy_gas == pytest.approx(gas_energy, abs=1e-8)
    for pose, accessible, interaction in zip(
        result.gas_positions,
        result.is_accessible,
        result.interaction_energies,
    ):
        if accessible:
            placed = gas.copy()
            placed.positions = np.asarray(pose)
            combined = prepared + placed
            combined.wrap()
            expected = factory().get_potential_energy(combined) - host_energy - gas_energy
            assert interaction == pytest.approx(expected, abs=1e-8, rel=1e-10)
    np.testing.assert_array_equal(host.positions, original_positions)
    np.testing.assert_array_equal(host.cell, original_cell)
    assert host.calc is None
