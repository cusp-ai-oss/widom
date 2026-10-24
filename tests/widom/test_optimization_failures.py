"""Fixed-cell convergence and failure paths with conservative analytic forces."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE


from widom.utils import optimize_atoms


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        self.results = {"energy": float(0.5 * (positions**2).sum()), "forces": -positions}


class LargeForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(-2000 * self.atoms.positions[:, 0].sum()),
            "forces": np.tile([2000.0, 0.0, 0.0], (len(self.atoms), 1)),
        }


@pytest.mark.parametrize(
    "position,cycles,fmax,converged",
    [
        (1.0, 1, 1e-8, False),
        (1.0, 2, 0.985, True),
        (0.0, 1, 1e-8, True),
    ],
)
def test_fixed_cell_convergence_paths(position, cycles, fmax, converged):
    atoms = Atoms("H", positions=[[position, 0, 0]], cell=[8, 8, 8], pbc=True)
    original = atoms.positions.copy()
    result = optimize_atoms(
        HarmonicCalculator(),
        atoms,
        cell_relax=False,
        num_total_optimization=cycles,
        num_internal_steps=1,
        fmax=fmax,
    )
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms.calc is None
    if converged:
        assert result is not None
        assert np.linalg.norm(result.get_forces(), axis=1).max() <= fmax
        np.testing.assert_array_equal(result.cell, atoms.cell)
    else:
        assert result is None


def test_fixed_cell_excessive_forces_stop_before_retry(monkeypatch):
    optimizers = []

    def record_optimizer(atoms):
        optimizer = FIRE(atoms)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr("widom.utils.FIRE", record_optimizer)
    atoms = Atoms("H", positions=[[1, 0, 0]], cell=[8, 8, 8], pbc=True)
    result = optimize_atoms(
        LargeForceCalculator(),
        atoms,
        cell_relax=False,
        num_total_optimization=2,
        num_internal_steps=1,
        fmax=1e-8,
    )
    assert result is None
    # The force guard must stop after the first cycle, before the retry budget is exhausted.
    assert len(optimizers) == 1
    assert optimizers[0].nsteps == 1
    np.testing.assert_array_equal(atoms.positions, [[1, 0, 0]])
