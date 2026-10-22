"""Independent sample-variance and unavailable-uncertainty contracts."""

import itertools
from statistics import stdev

import numpy as np
import pytest
from ase import Atoms, units


from widom.analyze import WidomInsertionResults, analyze_widom_insertions
from widom.utils import bootstrap_ratio_std, calculate_atomic_density


def analyze(energies, accessible=None):
    energies = np.asarray(energies, dtype=float)
    host = Atoms("H2", positions=[[1, 1, 1], [2, 1, 1]], cell=[8, 8, 8], pbc=True)
    return analyze_widom_insertions(
        energies,
        np.ones(len(energies), dtype=bool) if accessible is None else np.asarray(accessible),
        np.zeros((len(energies), 2, 3)),
        0.0,
        0.0,
        300.0,
        host,
        True,
        -1.25,
        1701,
    )


def test_henry_sample_variance_is_unbiased_over_exact_bernoulli_ensemble():
    thermal = units._k * 300 / units._e
    variances = []
    for bits in itertools.product([0, 1], repeat=4):
        result = analyze(-thermal * np.log(2) * np.asarray(bits))
        variances.append(result.henry_coefficient_std**2)
    host = Atoms("H2", cell=[8, 8, 8], pbc=True)
    scale = 1 / (units._k * units._Nav * 300 * calculate_atomic_density(host))
    expected = (0.25 / 4) * scale**2
    np.testing.assert_allclose(np.mean(variances), expected, rtol=1e-12, atol=0)


def test_one_attempt_preserves_point_estimates_but_reports_unavailable_errors():
    with pytest.warns(RuntimeWarning, match="at least two"):
        result = analyze([-0.05])
    assert result.henry_coefficient > 0
    assert result.averaged_interaction_energy == pytest.approx(-0.05)
    assert result.henry_coefficient_std is None
    assert result.averaged_interaction_energy_std is None
    assert result.heat_of_adsorption_std is None


def test_zero_weight_bootstrap_resample_does_not_report_a_finite_ratio_error():
    with pytest.warns(RuntimeWarning, match="zero total weight"):
        result = analyze([-0.05, 0, 0, 0], [True, False, False, False])
    assert np.isfinite(result.henry_coefficient_std)
    assert result.averaged_interaction_energy == pytest.approx(-0.05)
    assert result.averaged_interaction_energy_std is None
    assert result.heat_of_adsorption_std is None


@pytest.mark.parametrize(
    "energies, accessible",
    [
        ([-0.05], [True]),
        ([-0.05, 0, 0, 0], [True, False, False, False]),
    ],
)
def test_unavailable_errors_have_a_valid_json_round_trip(energies, accessible):
    with pytest.warns(RuntimeWarning):
        result = analyze(energies, accessible)
    restored = WidomInsertionResults.model_validate_json(result.model_dump_json())
    assert restored.heat_of_adsorption_std is None
    assert restored.henry_coefficient == result.henry_coefficient


def test_supported_constant_energies_keep_zero_uncertainty():
    result = analyze([-0.05] * 8)
    assert result.henry_coefficient_std == pytest.approx(0, abs=1e-12)
    assert result.heat_of_adsorption_std == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "numerator,denominator",
    [
        ([], []),
        ([1, 2], [1]),
        ([np.nan], [1]),
        ([1], [-1]),
        ([0, 0], [0, 0]),
    ],
)
def test_bootstrap_rejects_unsupported_input(numerator, denominator):
    with pytest.raises(ValueError):
        bootstrap_ratio_std(np.asarray(numerator), np.asarray(denominator), 100, 1701)


def test_bootstrap_requires_multiple_replicates():
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_ratio_std(np.array([1.0, 2.0]), np.array([1.0, 1.0]), 1, 1701)


def test_bootstrap_error_matches_sample_std_of_nonconstant_replicate_ratios(monkeypatch):
    resamples = iter([[0, 0], [0, 1], [1, 0], [1, 1]])

    class KnownResamples:
        def choice(self, n, size, replace):
            assert n == size == 2
            assert replace
            return np.asarray(next(resamples))

    monkeypatch.setattr(np.random, "default_rng", lambda seed: KnownResamples())
    actual = bootstrap_ratio_std(np.array([0.0, 2.0]), np.array([1.0, 2.0]), 4, 1701)
    # The four ordered resamples give these ratios; stdev uses the sample denominator.
    expected = stdev([0.0, 2 / 3, 2 / 3, 1.0])
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)
    assert next(resamples, None) is None
