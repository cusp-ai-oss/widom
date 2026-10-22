# Copyright (c) 2025 CuspAI
# All rights reserved.
#
# Modified in 2026: replace the seeded snapshot with an analytic control.

"""Integration tests for the widom package.

This module contains tests that verify the full functionality of the widom package,
including the example from the README.
"""

import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones
from widom import run_widom_insertion
from widom.analyze import WidomInsertionResults


class TestWidomIntegration:
    """Integration tests for Widom insertion simulation."""

    def test_readme_example(self):
        """Test the example from the README to ensure it works correctly.

        This test replicates the exact example shown in the README documentation
        using a Si crystal and Lennard-Jones calculator.
        """
        # Set up the same configuration as in the README
        calculator = LennardJones(epsilon=1, sigma=0.6)  # Example calculator
        structure = bulk("Si")  # Example structure

        # Run Widom insertion simulation with the same parameters as README
        results = run_widom_insertion(
            calculator=calculator,
            structure=structure,
            gas="CO2",
            temperature=298.15,  # K
            model_outputs_interaction_energy=False,
            num_insertions=1000,
            random_seed=42,
        )

        # Verify that we get a valid WidomInsertionResults object
        assert isinstance(results, WidomInsertionResults)

        # Check that all required properties are present and have reasonable values
        assert hasattr(results, "henry_coefficient")
        assert hasattr(results, "heat_of_adsorption")
        assert hasattr(results, "henry_coefficient_std")
        assert hasattr(results, "heat_of_adsorption_std")
        assert hasattr(results, "averaged_interaction_energy")
        assert hasattr(results, "total_energies")
        assert hasattr(results, "interaction_energies")
        assert hasattr(results, "is_accessible")
        assert hasattr(results, "gas_positions")

        # Verify that henry_coefficient is a positive finite number
        assert isinstance(results.henry_coefficient, (int, float))
        assert results.henry_coefficient > 0
        assert not (
            results.henry_coefficient == float("inf")
            or results.henry_coefficient != results.henry_coefficient
        )

        # Verify that heat_of_adsorption is a finite number
        assert isinstance(results.heat_of_adsorption, (int, float))
        assert not (
            results.heat_of_adsorption == float("inf")
            or results.heat_of_adsorption != results.heat_of_adsorption
        )

        # Verify that lists have the expected lengths
        assert len(results.total_energies) <= 1000  # May be fewer due to inaccessible positions
        assert len(results.interaction_energies) == len(results.total_energies)
        assert len(results.is_accessible) == 1000
        assert len(results.gas_positions) == 1000

        # Check that some positions are accessible (would be very unlikely if none are)
        assert sum(results.is_accessible) > 0

        # Verify that gas_positions has the right shape (1000 insertions, CO2 has 3 atoms, 3D positions)
        assert np.array(results.gas_positions).shape == (1000, 3, 3)

    def test_reproducibility(self):
        """Test that results are reproducible when using the same random seed."""
        calculator = LennardJones(epsilon=1, sigma=0.6)
        structure = bulk("Si")

        # Run the same simulation twice with the same random seed
        results1 = run_widom_insertion(
            calculator=calculator,
            structure=structure,
            gas="CO2",
            temperature=298.15,
            model_outputs_interaction_energy=False,
            num_insertions=100,  # Use fewer insertions for faster testing
            random_seed=42,
        )

        results2 = run_widom_insertion(
            calculator=calculator,
            structure=structure,
            gas="CO2",
            temperature=298.15,
            model_outputs_interaction_energy=False,
            num_insertions=100,
            random_seed=42,
        )

        # Results should be identical when using the same seed
        assert results1.henry_coefficient == results2.henry_coefficient
        assert results1.heat_of_adsorption == results2.heat_of_adsorption
        assert len(results1.total_energies) == len(results2.total_energies)

        # Check that energies are the same
        for e1, e2 in zip(results1.total_energies, results2.total_energies):
            assert (
                abs(e1 - e2) < 1e-10
            )  # Should be exactly equal, but allow for floating point precision

    def test_different_gas_molecules(self):
        """Test that the simulation works with different gas molecules."""
        calculator = LennardJones(epsilon=1, sigma=0.6)
        structure = bulk("Si")

        # Test with different gas molecules
        for gas in ["H2", "CH4", "N2"]:
            results = run_widom_insertion(
                calculator=calculator,
                structure=structure,
                gas=gas,
                temperature=298.15,
                model_outputs_interaction_energy=False,
                num_insertions=50,  # Small number for fast testing
                random_seed=42,
            )

            # Verify basic properties
            assert isinstance(results, WidomInsertionResults)
            assert results.henry_coefficient > 0
            assert not (
                results.henry_coefficient == float("inf")
                or results.henry_coefficient != results.henry_coefficient
            )

    def test_different_temperatures(self):
        """Test that the simulation works at different temperatures."""
        calculator = LennardJones(epsilon=1, sigma=0.6)
        structure = bulk("Si")

        # Test at different temperatures
        for temperature in [200.0, 298.15, 400.0]:
            results = run_widom_insertion(
                calculator=calculator,
                structure=structure,
                gas="CO2",
                temperature=temperature,
                model_outputs_interaction_energy=False,
                num_insertions=50,  # Small number for fast testing
                random_seed=42,
            )

            # Verify basic properties
            assert isinstance(results, WidomInsertionResults)
            assert results.henry_coefficient > 0
            assert not (
                results.henry_coefficient == float("inf")
                or results.henry_coefficient != results.henry_coefficient
            )

    def test_constant_interaction_matches_analytic_thermodynamics(self):
        """Check a known ensemble rather than freezing a biased pose stream."""
        from ase import Atoms, units
        from ase.calculators.calculator import Calculator, all_changes
        from widom.utils import calculate_atomic_density

        class ConstantInteraction(Calculator):
            implemented_properties = ["energy"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                numbers = self.atoms.numbers
                host_count = int((numbers == 14).sum())
                has_guest = bool((numbers == 6).any())
                self.results = {
                    "energy": 0.01 * host_count + (-0.05 if host_count and has_guest else 0.0)
                }

        structure = Atoms("Si2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
        results = run_widom_insertion(
            ConstantInteraction(),
            structure,
            "CO2",
            300.0,
            False,
            num_insertions=8,
            cutoff_distance=0.0,
            random_seed=42,
        )
        kT = units._k * 300.0 / units._e
        expected_henry = np.exp(0.05 / kT) / (
            units._k * units._Nav * 300.0 * calculate_atomic_density(structure)
        )
        expected_heat = (-0.05 * units._e - units._k * 300.0) * units._Nav * 1e-3
        np.testing.assert_allclose(results.interaction_energies, -0.05, atol=1e-12, rtol=0)
        np.testing.assert_allclose(results.henry_coefficient, expected_henry, rtol=1e-10)
        np.testing.assert_allclose(results.heat_of_adsorption, expected_heat, rtol=1e-10)
        np.testing.assert_allclose(results.henry_coefficient_std, 0.0, atol=1e-12)
        np.testing.assert_allclose(results.heat_of_adsorption_std, 0.0, atol=1e-12)
