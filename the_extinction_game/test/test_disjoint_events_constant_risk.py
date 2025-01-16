import pytest
import numpy as np
from the_extinction_game.models.disjoint_events_constant_risk import DisjoinEventsConstantRiskModel


@pytest.fixture
def model():
    return DisjoinEventsConstantRiskModel(n_centuries=5, initial_risk=0.1)


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


if __name__ == '__main__':
    pytest.main([__file__])
