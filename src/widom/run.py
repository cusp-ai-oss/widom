# Copyright (c) 2025 CuspAI
# All rights reserved.
#
# Modified in 2026: align host references and clarify calculator/result contracts.

import logging

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator

from .analyze import (
    WidomInsertionResults,
    analyze_widom_insertions,
)
from .sample_compute_energies import sample_compute_energies
from .utils import create_supercell_if_needed, optimize_atoms

logger = logging.getLogger(__name__)


def run_widom_insertion(
    calculator: Calculator,
    structure: Atoms,
    gas: str,
    temperature: float,
    model_outputs_interaction_energy: bool,
    num_insertions: int = 10000,
    optimize_structures: bool = False,
    cutoff_distance: float = 1.00,
    cutoff_to_com: bool = False,  # Whether to use center of mass for cutoff
    min_interplanar_distance: float = 6.0,
    random_seed: int = 0,
    min_interaction_energy: float = -1.25,
) -> WidomInsertionResults:
    """Run Widom insertion simulation to calculate Henry coefficient and heat of adsorption.

    The code in this function is derived from
    https://github.com/hspark1212/DAC-SIM
    MIT-licensed

    Args:
        calculator: ASE calculator responsible for host and gas parameterization,
            including any required charges. It must support combined structures,
            bare hosts and a nonperiodic ASE gas molecule. Host/gas reference
            queries are performed even in direct-interaction-energy mode.
        structure: ASE Atoms object representing the framework structure. The input
            object is not modified.
        gas: Name of the gas molecule to insert (e.g., 'H2', 'CO2').
        temperature: Temperature in Kelvin for the simulation.
        model_outputs_interaction_energy: Whether the calculator outputs interaction
            energies directly, using consistent host/gas reference and periodic
            boundary conventions.
        num_insertions: Number of random insertion attempts.
        optimize_structures: Whether to optimize the structure and gas molecule before insertion.
        cutoff_distance: Minimum allowed distance between framework and gas atoms in angstroms.
        cutoff_to_com: Whether to use center of mass for distance calculations.
        min_interplanar_distance: Minimum interplanar distance before constructing a supercell, in angstroms.
        random_seed: Seed for repeatable sampling within the same environment.
        min_interaction_energy: Minimum valid interaction energy for the gas molecule, in eV.

    Example:
        >>> from widom import run_widom_insertion
        >>> from ase.build import bulk
        >>> from ase.calculators.lj import LennardJones
        >>>
        >>> calculator = LennardJones(epsilon=1, sigma=0.6)  # Example calculator
        >>> structure = bulk("Si")  # Example structure
        >>>
        >>> # Run Widom insertion simulation
        >>> results = run_widom_insertion(
        ...     calculator=calculator,
        ...     structure=structure,
        ...     gas="CO2",
        ...     temperature=298.15,  # K
        ...     model_outputs_interaction_energy=False,
        ...     num_insertions=1000,
        ...     random_seed=42,
        ... )
        >>>
        >>> # Access results
        >>> print(f"Henry coefficient: {results.henry_coefficient:.2e} mol/kg/Pa")
        >>> print(f"Heat of adsorption: {results.heat_of_adsorption:.2f} kJ/mol")

    Returns:
        Results containing Henry coefficient, heat of adsorption, and other computed
        properties. In total-energy mode, energy_structure is the reference energy
        of the actual host supercell used for insertion. optimized_structure_cif
        describes that same host, including any cell replication.
    """
    gas_atoms = molecule(gas)

    # Optionally optimize structures
    if optimize_structures:
        logger.info("Optimizing structure...")
        optimized_structure = optimize_atoms(
            calculator=calculator,
            atoms=structure,
        )
        if optimized_structure is None:
            raise ValueError("Structure optimization failed.")

        logger.info("Optimizing gas molecule...")
        optimized_gas = optimize_atoms(
            calculator=calculator,
            atoms=gas_atoms,
            cell_relax=False,
        )
        if optimized_gas is None:
            raise ValueError("Gas molecule optimization failed.")
    else:
        optimized_structure = structure
        optimized_gas = gas_atoms

    # Use the same host cell for combined energies and the bare-host reference.
    # The sampler also checks the cell size, but must not expand a different
    # structure after the reference has been chosen.
    optimized_structure = create_supercell_if_needed(optimized_structure, min_interplanar_distance)

    # Run Widom insertion
    logger.info(f"Running Widom insertion with {num_insertions} insertions...")
    energies, is_accessible, gas_positions = sample_compute_energies(
        calculator=calculator,
        structure=optimized_structure,
        gas=optimized_gas,
        num_insertions=num_insertions,
        cutoff_distance=cutoff_distance,
        cutoff_to_com=cutoff_to_com,
        min_interplanar_distance=min_interplanar_distance,
        random_seed=random_seed,
    )

    energy_structure = calculator.get_potential_energy(optimized_structure)
    energy_gas = calculator.get_potential_energy(optimized_gas)
    logger.info(f"Energy of structure: {energy_structure} eV")
    logger.info(f"Energy of gas: {energy_gas} eV")

    assert energy_structure is not None, "Energy of the structure must be computed."
    assert energy_gas is not None, "Energy of the gas molecule must be computed."

    # Analyze results
    logger.info("Analyzing results...")
    results = analyze_widom_insertions(
        energies=energies,
        is_accessible=is_accessible,
        gas_positions=gas_positions,
        temperature=temperature,
        structure=optimized_structure,
        energy_structure=energy_structure,
        energy_gas=energy_gas,
        energies_are_interaction=model_outputs_interaction_energy,
        random_seed=random_seed,
        min_interaction_energy=min_interaction_energy,
    )

    logger.info("Results:")
    logger.info(results)
    return results
