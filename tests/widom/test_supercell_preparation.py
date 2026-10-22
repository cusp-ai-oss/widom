"""Repeated cell preparation and public energy-reference regressions."""

import numpy as np
import pytest
from ase import Atoms

from widom.utils import create_supercell_if_needed


@pytest.mark.parametrize("minimum", [4.0, 6.0, 12.0])
def test_repeated_cell_preparation_is_idempotent(minimum):
    rng = np.random.default_rng(881)
    cells = [np.eye(3) * minimum]
    for index in range(20):
        cell = np.diag(rng.uniform(0.5, 1.0, 3))
        cell[1, 0] = rng.uniform(-0.3, 0.3)
        cell[2, :2] = rng.uniform(-0.3, 0.3, 2)
        planes = abs(np.linalg.det(cell)) / np.linalg.norm(
            np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
        )
        cells.append(cell * minimum / (planes.min() * (3 if index % 2 else 1)))
    for cell in cells:
        host = Atoms("H", cell=cell, pbc=True)
        first = create_supercell_if_needed(host, minimum)
        second = create_supercell_if_needed(first, minimum)
        assert len(first) == len(second)
        np.testing.assert_allclose(first.cell, second.cell, atol=1e-12, rtol=0)


@pytest.mark.parametrize("relative_deficit,expected_count", [(1e-14, 1), (1e-8, 8), (-1e-8, 1)])
def test_cell_boundary_tolerance_is_narrow(relative_deficit, expected_count):
    host = Atoms("H", cell=[6 * (1 - relative_deficit)] * 3, pbc=True)
    result = create_supercell_if_needed(host, 6)
    assert len(result) == expected_count


def test_boundary_replication_keeps_public_runner_energy_reference():
    from ase.calculators.calculator import Calculator, all_changes
    from widom import run_widom_insertion

    class KnownInsertion(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            host_count = int((self.atoms.numbers == 1).sum())
            interaction = -0.05 if host_count and (self.atoms.numbers == 7).any() else 0.0
            self.results = {"energy": 0.01 * host_count + interaction}

    # Seed-881 triclinic fixture: the old helper expands 27 -> 54 host atoms
    # on consecutive calls because a plane spacing rounds just below 6 A.
    cell = [
        [3.038064365395629, 0.0, 0.0],
        [-0.7086229039387212, 2.0146732703427355, 0.0],
        [0.8885393229175839, 0.2902009853915734, 2.3913344236216996],
    ]
    host = Atoms("H", cell=cell, pbc=True)
    result = run_widom_insertion(
        KnownInsertion(),
        host,
        "N2",
        300,
        False,
        num_insertions=2,
        cutoff_distance=0,
        min_interplanar_distance=6,
        random_seed=1701,
    )
    np.testing.assert_allclose(result.interaction_energies, -0.05, atol=1e-12, rtol=0)
