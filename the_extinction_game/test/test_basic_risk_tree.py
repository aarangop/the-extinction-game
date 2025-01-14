import pytest
import numpy as np
from the_extinction_game.basic_risk_tree import (
    get_xrisk_tree_shape,
    build_xrisk_tree,
    run_survival_simulation,
    estimate_simulation_runtime,
    set_extinction_event
)


class TestXRiskTreeShape:
    def test_basic_shapes(self):
        """Test basic tree shape calculations."""
        assert get_xrisk_tree_shape(1) == (1, 1)
        assert get_xrisk_tree_shape(2) == (2, 2)
        assert get_xrisk_tree_shape(3) == (3, 4)
        assert get_xrisk_tree_shape(4) == (4, 8)

    def test_larger_shapes(self):
        """Test shape calculations for larger trees."""
        rows, cols = get_xrisk_tree_shape(10)
        assert rows == 10
        assert cols == 512  # 2^9

    def test_shape_relationships(self):
        """Test that shapes follow expected relationships."""
        for n in range(1, 6):
            rows, cols = get_xrisk_tree_shape(n)
            assert rows == n
            assert cols == 2**(n-1)


class TestBuildXRiskTree:
    @pytest.mark.parametrize("n,initial_risk,alpha", [
        (3, 0.1, 0.01),
        (4, 0.5, 0.1),
        (2, 0.0, 0.0),
        (3, 1.0, 0.1),
    ])
    def test_tree_dimensions(self, n, initial_risk, alpha):
        """Test that built trees have correct dimensions."""
        tree = build_xrisk_tree(n, initial_risk, alpha)
        expected_rows, expected_cols = get_xrisk_tree_shape(n)
        assert tree.shape == (expected_rows, expected_cols)

    def test_initial_risk_propagation(self):
        """Test that initial risk is correctly set in first row."""
        initial_risk = 0.3
        tree = build_xrisk_tree(3, initial_risk, 0.1)
        assert np.all(tree[0] == initial_risk)

    def test_risk_bounds(self):
        """Test that risk values stay within [0,1] bounds."""
        tree = build_xrisk_tree(4, 0.5, 0.3)
        assert np.all(tree >= 0)
        assert np.all(tree <= 1)

    def test_risk_changes(self):
        """Test that risk changes follow alpha parameter."""
        n, initial_risk, alpha = 3, 0.5, 0.1
        tree = build_xrisk_tree(n, initial_risk, alpha)

        # Check second row (first split)
        assert np.isclose(tree[1, 0], min(initial_risk + alpha, 1.0))
        assert np.isclose(tree[1, -1], max(initial_risk - alpha, 0.0))

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(AssertionError):
            build_xrisk_tree(0, 0.1, 0.1)  # Invalid n
        with pytest.raises(AssertionError):
            build_xrisk_tree(3, -0.1, 0.1)  # Invalid initial risk
        with pytest.raises(AssertionError):
            build_xrisk_tree(3, 0.1, 1.5)   # Invalid alpha


class TestEstimateSimulationRuntime:
    def test_runtime_scaling(self):
        """Test that runtime estimates scale appropriately."""
        base_estimate = estimate_simulation_runtime(5, 1000)
        double_sims_estimate = estimate_simulation_runtime(5, 2000)

        # Runtime should roughly double with double the simulations
        assert (double_sims_estimate['estimated_runtime_seconds'] /
                base_estimate['estimated_runtime_seconds']) > 1.5

    def test_memory_scaling(self):
        """Test that memory estimates scale appropriately."""
        base_estimate = estimate_simulation_runtime(5, 1000)
        double_sims_estimate = estimate_simulation_runtime(5, 2000)

        # Memory should double with double the simulations
        assert np.isclose(double_sims_estimate['memory_gb'] /
                          base_estimate['memory_gb'], 2.0)

    def test_warning_generation(self):
        """Test that warnings are generated appropriately."""
        # Test high memory warning
        high_mem_estimate = estimate_simulation_runtime(20, 10000)
        assert 'warnings' in high_mem_estimate
        assert any('memory' in w.lower()
                   for w in high_mem_estimate['warnings'])

        # Test long runtime warning
        long_runtime_estimate = estimate_simulation_runtime(25, 100000)
        assert 'warnings' in long_runtime_estimate
        assert any('runtime' in w.lower()
                   for w in long_runtime_estimate['warnings'])

    def test_benchmark_scaling(self):
        """Test that benchmark size affects runtime estimates."""
        standard_estimate = estimate_simulation_runtime(
            5, 1000, benchmark_size=0.01)
        larger_benchmark = estimate_simulation_runtime(
            5, 1000, benchmark_size=0.1)

        assert larger_benchmark['simulations_tested'] > standard_estimate['simulations_tested']


class TestRunSurvivalSimulation:
    @pytest.fixture
    def sample_tree(self):
        """Fixture providing a sample risk tree."""
        return build_xrisk_tree(3, 0.3, 0.1)

    def test_output_shape(self, sample_tree):
        """Test that simulation output has same shape as input."""
        result = run_survival_simulation(sample_tree)
        assert result.shape == sample_tree.shape

    def test_binary_output(self, sample_tree):
        """Test that simulation produces only 0s and 1s."""
        result = run_survival_simulation(sample_tree)
        assert np.all(np.logical_or(result == 0, result == 1))

    def test_extinction_permanence(self, sample_tree):
        """Test that extinction (0) is permanent within each branch."""
        result, extinction_century = run_survival_simulation(sample_tree)

        for branch in result.T:
            extinction_points = np.where(branch == 0)[0]
            if len(extinction_points) > 0:
                first_extinction = extinction_points[0]
                assert np.all(branch[first_extinction:] == 0)

    def test_certain_survival(self):
        """Test simulation with zero risk."""
        tree = build_xrisk_tree(3, 0.0, 0.0)
        result = run_survival_simulation(tree)
        assert np.all(result == 1)

    def test_certain_extinction(self):
        """Test simulation with 100% risk."""
        tree = build_xrisk_tree(3, 1.0, 0.0)
        result = run_survival_simulation(tree)
        # All should be extinct after first generation
        assert np.all(result[1:] == 0)


class TestSetExtinctionEvent:
    def test_no_extinction(self):
        """Test that no extinction occurs when all values are 1."""
        survival_matrix = np.ones((5, 4))
        updated_matrix, extinction_century = set_extinction_event(
            survival_matrix)
        assert np.all(updated_matrix == 1)
        assert np.all(extinction_century == -1)

    def test_immediate_extinction(self):
        """Test that extinction occurs immediately when all values are 0."""
        survival_matrix = np.zeros((5, 4))
        updated_matrix, extinction_century = set_extinction_event(
            survival_matrix)
        assert np.all(updated_matrix == 0)
        assert np.all(extinction_century == 0)

    def test_partial_extinction(self):
        """Test that extinction occurs at the correct time step."""
        survival_matrix = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        updated_matrix, extinction_century = set_extinction_event(
            survival_matrix)
        expected_matrix = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        expected_extinction_century = np.array([-1, 1, 2, 3])
        assert np.all(updated_matrix == expected_matrix)
        assert np.all(extinction_century ==
                      expected_extinction_century)

    def test_extinction_with_survivors(self):
        """Test that some branches go extinct while others survive."""
        survival_matrix = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])
        updated_matrix, extinction_century = set_extinction_event(
            survival_matrix)
        expected_matrix = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])
        expected_extinction_century = np.array([-1, 1, -1, 2])
        assert np.all(updated_matrix == expected_matrix)
        assert np.all(extinction_century ==
                      expected_extinction_century)


if __name__ == '__main__':
    pytest.main([__file__])
