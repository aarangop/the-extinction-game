import pytest
import numpy as np
from the_extinction_game.models.disjoint_events_constant_risk import DisjointEventsConstantRiskModel


@pytest.fixture
def model():
    return DisjointEventsConstantRiskModel(n_centuries=5, initial_risk=0.1)


def test_compute_century_risk(model):
    centuries = np.arange(1, 6)
    expected_risks = (1 - 0.1) ** centuries
    computed_risks = model.compute_survival_probability(np.arange(1, 6))
    assert np.allclose(computed_risks, expected_risks)


def test_compute_survival_probability_zero_risk(model):
    expected_risks = np.ones(5)
    model.initial_risk = 0
    computed_risks = model.compute_survival_probability(np.arange(1, 6))
    assert np.allclose(computed_risks, expected_risks)


def test_compute_survival_probability_full_risk(model):
    expected_risks = np.zeros(5)
    model.initial_risk = 1
    computed_risks = model.compute_survival_probability(np.arange(1, 6))
    assert np.allclose(computed_risks, expected_risks)


def test_run_simulation_no_extinction(model):
    model.initial_risk = 0
    survival, extinction_century = model.run_simulation()
    assert np.all(survival)
    assert extinction_century is -1


def test_run_simulation_immediate_extinction(model):
    model.initial_risk = 1
    survival, extinction_century = model.run_simulation()
    assert not np.any(survival)
    assert extinction_century == 0


def test_survival_after_first_extinction_event_is_zero(model):
    model.survival_probability_per_century = np.array([1, 1, 0, 1, 0])
    survival, extinction_century = model.run_simulation()
    expected_extinction_century = 2
    expected_survival = [1, 1, 0, 0, 0]
    assert expected_extinction_century == extinction_century
    assert np.all(expected_survival == survival)


def test_survival_probability_per_century_is_correct(model):
    model.initial_risk = 0.5
    model.survival_probability_per_century = model.compute_survival_probability(
        model.centuries)
    expected_survival_probability = np.array(
        [0.5, 0.25, 0.125, 0.0625, 0.03125])
    assert np.allclose(model.survival_probability_per_century,
                       expected_survival_probability)


if __name__ == '__main__':
    pytest.main([__file__])
