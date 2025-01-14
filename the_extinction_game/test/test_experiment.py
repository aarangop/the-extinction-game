import pytest
import numpy as np
from the_extinction_game.experiment import Experiment


class MockModel:
    def __init__(self, n_centuries, n_branches):
        self.n_centuries = n_centuries
        self.n_branches = n_branches
        self.extinction_century = np.zeros(n_branches)

    def run_simulation(self):
        # Mock simulation results
        return np.ones((self.n_centuries, self.n_branches)), np.zeros(self.n_branches)


@pytest.fixture
def mock_model():
    return MockModel(n_centuries=5, n_branches=3)


@pytest.fixture
def experiment(mock_model):
    return Experiment(model=mock_model, n_simulations=10)


def test_run_results_shape(experiment):
    results, extinction_centuries = experiment.run()
    assert results.shape == (
        experiment.model.n_centuries, experiment.model.n_branches)
    assert extinction_centuries.shape == (
        experiment.n_simulations, experiment.model.n_branches)


def test_run_results_values(experiment):
    results, extinction_centuries = experiment.run()
    # Since the mock model returns ones, the results should be ones divided by n_simulations
    expected_results = np.ones(
        (experiment.model.n_centuries, experiment.model.n_branches))
    assert np.allclose(results, expected_results)


def test_run_extinction_centuries(experiment):
    results, extinction_centuries = experiment.run()
    # Since the mock model sets extinction_century to zeros, the extinction_centuries should be zeros
    expected_extinction_centuries = np.zeros(
        (experiment.n_simulations, experiment.model.n_branches))
    assert np.allclose(extinction_centuries, expected_extinction_centuries)


if __name__ == '__main__':
    pytest.main([__file__])
