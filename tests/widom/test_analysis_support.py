"""Analytic controls for unsupported or rejected insertion samples."""

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes


from widom import run_widom_insertion
from widom.analyze import analyze_widom_insertions
from widom.utils import calculate_atomic_density


def analyze(energies, accessible):
    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    return analyze_widom_insertions(
        energies=np.asarray(energies, dtype=float),
        is_accessible=np.asarray(accessible, dtype=bool),
        gas_positions=np.zeros((len(energies), 2, 3)),
        energy_structure=0.0,
        energy_gas=0.0,
        temperature=300.0,
        structure=host,
        energies_are_interaction=True,
        min_interaction_energy=-1.25,
        random_seed=1701,
    )


def test_empty_sample_set_has_a_meaningful_error():
    with pytest.raises(ValueError, match="At least one insertion"):
        analyze([], [])


@pytest.mark.parametrize(
    "energies, accessible",
    [
        ([0.0] * 4, [False] * 4),
        ([-2.0] * 4, [True] * 4),
        ([np.nan, np.inf, -np.inf], [True] * 3),
    ],
)
def test_no_supported_energy_samples_raise(energies, accessible):
    with pytest.raises(ValueError, match="No accessible insertions"):
        analyze(energies, accessible)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -2.0])
def test_invalid_energy_has_zero_weight_but_remains_in_normalization(invalid):
    result = analyze([-0.05] * 7 + [invalid], [True] * 8)
    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    kT = 300.0 * units._k / units._e
    expected = (
        (7 / 8)
        * np.exp(0.05 / kT)
        / (units._k * units._Nav * 300.0 * calculate_atomic_density(host))
    )
    assert result.henry_coefficient == pytest.approx(expected, rel=1e-12)
    assert result.averaged_interaction_energy == pytest.approx(-0.05, abs=1e-12)
    assert np.isfinite(result.heat_of_adsorption_std)
    assert result.is_valid == [True] * 7 + [False]


def test_inaccessible_energy_is_not_used_even_if_numerically_favorable():
    result = analyze([-0.05] * 7 + [-0.5], [True] * 7 + [False])
    reference = analyze([-0.05] * 8, [True] * 8)
    assert result.henry_coefficient == pytest.approx(reference.henry_coefficient * 7 / 8, rel=1e-12)
    assert result.averaged_interaction_energy == pytest.approx(-0.05, abs=1e-12)
    assert result.is_valid[-1] is False


def test_public_api_rejects_all_inaccessible_insertions():
    class ZeroEnergy(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0.0}

    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    with pytest.raises(ValueError, match="No accessible insertions"):
        run_widom_insertion(
            ZeroEnergy(),
            host,
            "N2",
            300.0,
            False,
            num_insertions=4,
            cutoff_distance=8.0,
            random_seed=1701,
        )
