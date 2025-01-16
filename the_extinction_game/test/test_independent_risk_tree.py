import pytest
from the_extinction_game.models.multi_risk_binary_tree import MultiRiskBinaryTreeModel


@pytest.fixture
def multi_risk_model():
    model = MultiRiskBinaryTreeModel({
        "R1": (0.1, 0.01),
        "R2": (0.2, 0.02),
    }, 3)
    return model


def test_build_all_risk_trees_empty():
    model = MultiRiskBinaryTreeModel({}, n_centuries=3)
    trees = model.build_all_risk_trees()

    # Should return a list of length 0
    assert len(trees) == 0
    # The shape should be (0, 3, 4) (categories, centuries, branches)
    assert trees.shape == (0, 3, 4)


def test_build_all_risk_trees_single_risk():
    model = MultiRiskBinaryTreeModel({"R1": (0.1, 0.01)}, 3)
    trees = model.build_all_risk_trees()
    assert len(trees) == 1
    assert trees.shape == (1, 3, 4)


def test_build_all_risk_trees(multi_risk_model):
    trees = multi_risk_model.build_all_risk_trees()
    assert len(trees) == 2
    assert trees.shape == (2, 3, 4)
    assert trees[0, 0, 0] == 0.1
    assert trees[1, 0, 0] == 0.2


def test_run_multi_risk_simulation(multi_risk_model):
    results, extinction_century, partial_results, partial_extinction_centuries = multi_risk_model.run_multi_risk_simulation()

    n_categories, n_centuries, n_branches = multi_risk_model.risk_trees.shape

    # The resulting array should have the shape (3, 4) (number of centuries, and number of branches)
    assert results.shape == (n_centuries, n_branches)
    # The extinction century should be an array of length 4
    assert len(extinction_century) == n_branches
    # The partial results should have the shape (2, 3, 4) (categories, centuries, branches)
    assert partial_results.shape == (n_categories, n_centuries, n_branches)
    # The partial extinction centuries should have the shape (2, 4) (categories, branches)
    assert partial_extinction_centuries.shape == (n_categories, n_branches)


def test_run_experiment(multi_risk_model):
    n_simulations = 5
    results, extinction_centuries = multi_risk_model.run_experiment(
        n_simulations)

    _, n_centuries, n_branches = multi_risk_model.risk_trees.shape

    # The resulting array should have the shape (3, 4) (number of centuries, and number of branches)
    assert results.shape == (n_centuries, n_branches)
    # The extinction centuries should have the shape (5, 4) (number of simulations, and number of branches)
    assert extinction_centuries.shape == (n_simulations, n_branches)
